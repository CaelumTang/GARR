import math

import torch
import torch.nn as nn


class StarAttentionWeights(nn.Module):
    def __init__(self, *, dim: int, heads: int = 4, prior_beta_init: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        if self.dim <= 0:
            raise RuntimeError("dim must be > 0")
        if self.heads <= 0:
            raise RuntimeError("heads must be > 0")

        pair_out = 2 * self.dim
        if pair_out % self.heads != 0:
            raise RuntimeError(f"2*dim must be divisible by heads, got dim={self.dim} heads={self.heads}")
        self.head_dim = pair_out // self.heads
        self.linear = nn.Linear(4 * self.dim, self.heads * self.head_dim, bias=True)
        self.activation = nn.LeakyReLU(0.2)
        self.attn_vec = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.prior_beta = nn.Parameter(torch.tensor(float(prior_beta_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(
        self,
        *,
        q_v: torch.Tensor,
        q_t: torch.Tensor,
        nb_v: torch.Tensor,
        nb_t: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        batch_size, num_neighbors, dim = nb_v.shape
        if q_v.shape != (batch_size, dim) or q_t.shape != (batch_size, dim):
            raise RuntimeError("q_v/q_t must align with neighbors batch and dim")
        if nb_t.shape != (batch_size, num_neighbors, dim):
            raise RuntimeError("nb_t must align with nb_v")
        if w.shape != (batch_size, num_neighbors):
            raise RuntimeError("w must be [B,K]")
        if dim != self.dim:
            raise RuntimeError(f"dim mismatch: expected {self.dim}, got {dim}")
        if not torch.isfinite(w).all().item() or (w <= 0).any().item():
            raise RuntimeError("w must be finite and strictly positive")

        query = torch.cat([q_v, q_t], dim=1).unsqueeze(1).expand(-1, num_neighbors, -1)
        neighbors = torch.cat([nb_v, nb_t], dim=2)
        pair = torch.cat([query, neighbors], dim=2)

        hidden = self.activation(self.linear(pair))
        hidden = hidden.view(batch_size, num_neighbors, self.heads, self.head_dim)
        logits = (hidden * self.attn_vec.view(1, 1, self.heads, self.head_dim)).sum(dim=-1) / math.sqrt(float(self.head_dim))
        logits = logits.mean(dim=2)
        logits = logits + self.prior_beta.to(dtype=logits.dtype) * torch.log(torch.clamp(w.to(dtype=logits.dtype), min=eps))
        return torch.softmax(logits, dim=1)


class Stage3Predictor(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden: int = 512,
        heads: int = 4,
        prior_beta_init: float = 1.0,
        pre_gate_init: float = -3.0,
    ):
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise RuntimeError("dim must be > 0")

        self.attn = StarAttentionWeights(dim=self.dim, heads=int(heads), prior_beta_init=float(prior_beta_init))
        self.score_embed = nn.Linear(1, self.dim)
        self.pre_gate = nn.Parameter(torch.tensor(float(pre_gate_init)))
        self.head = nn.Sequential(
            nn.Linear(5 * self.dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    @staticmethod
    def weighted_pool(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (weights.unsqueeze(-1) * values).sum(dim=1)

    def forward(
        self,
        *,
        q_v: torch.Tensor,
        q_t: torch.Tensor,
        nb_v: torch.Tensor,
        nb_t: torch.Tensor,
        w: torch.Tensor,
        q_pre: torch.Tensor,
        nb_y: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_neighbors, dim = nb_v.shape
        if nb_t.shape != (batch_size, num_neighbors, dim) or w.shape != (batch_size, num_neighbors) or nb_y.shape != (batch_size, num_neighbors, 1):
            raise RuntimeError("nb_t/w/nb_y must align with nb_v")
        if q_v.shape != (batch_size, dim) or q_t.shape != (batch_size, dim):
            raise RuntimeError("q_v/q_t must align with neighbor batch and dim")
        if q_pre.shape != (batch_size, 1):
            raise RuntimeError("q_pre must be [B,1]")

        attn = self.attn(q_v=q_v, q_t=q_t, nb_v=nb_v, nb_t=nb_t, w=w)
        pooled_v = self.weighted_pool(attn, nb_v)
        pooled_t = self.weighted_pool(attn, nb_t)
        knn_score = self.weighted_pool(attn, nb_y).to(q_v.dtype)
        gate = torch.sigmoid(self.pre_gate).to(q_v.dtype)
        mixed_score = gate * q_pre.to(q_v.dtype) + (1.0 - gate) * knn_score
        features = [q_v, q_t, pooled_v, pooled_t, self.score_embed(mixed_score)]
        return self.head(torch.cat(features, dim=1)).squeeze(1)
