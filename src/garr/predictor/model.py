import math

import torch
import torch.nn as nn


class _ContextAwareNeighborAttention(nn.Module):
    """Target-conditioned neighbor attention."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int = 16,
        prior_beta_init: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        if self.dim <= 0:
            raise RuntimeError("dim must be > 0")
        if self.heads <= 0:
            raise RuntimeError("heads must be > 0")

        pair_in = 4 * self.dim
        pair_out = 2 * self.dim
        if pair_out % self.heads != 0:
            raise RuntimeError(
                f"2*dim must be divisible by heads, got dim={self.dim} heads={self.heads}"
            )
        self.dh = pair_out // self.heads

        self.lin = nn.Linear(pair_in, self.heads * self.dh, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.attn_vec = nn.Parameter(torch.empty(self.heads, self.dh))

        self.prior_beta = nn.Parameter(torch.tensor(float(prior_beta_init)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
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
        B, K, C = nb_v.shape
        if q_v.shape != (B, C) or q_t.shape != (B, C):
            raise RuntimeError("q_v/q_t must align with neighbors batch and dim")
        if nb_t.shape != (B, K, C):
            raise RuntimeError("nb_t must align with nb_v")
        if w.shape != (B, K):
            raise RuntimeError("w must be [B,K] aligned with neighbors")
        if C != self.dim:
            raise RuntimeError(f"dim mismatch: expected {self.dim}, got {C}")
        if not torch.isfinite(w).all().item():
            raise RuntimeError("w contains NaN/Inf")
        if (w <= 0).any().item():
            raise RuntimeError("w must be strictly positive for log-prior")

        q = torch.cat([q_v, q_t], dim=1)
        nb = torch.cat([nb_v, nb_t], dim=2)

        q_exp = q.unsqueeze(1).expand(-1, K, -1)
        pair = torch.cat([q_exp, nb], dim=2)

        h = self.act(self.lin(pair))
        h = h.view(B, K, self.heads, self.dh)

        logits_h = (h * self.attn_vec.view(1, 1, self.heads, self.dh)).sum(dim=-1) / math.sqrt(
            float(self.dh)
        )
        logits = logits_h.mean(dim=2)

        beta = self.prior_beta.to(dtype=logits.dtype)
        logits = logits + beta * torch.log(torch.clamp(w.to(dtype=logits.dtype), min=eps))

        return torch.softmax(logits, dim=1)


class GARRPredictor(nn.Module):
    """GARR Retrieval Refinement predictor."""

    def __init__(
        self,
        *,
        dim: int,
        hidden: int = 512,
        heads: int = 16,
        prior_beta_init: float = 1.0,
        pre_gate_init: float = -3.0,
    ):
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise RuntimeError("dim must be > 0")
        self.attn = _ContextAwareNeighborAttention(
            dim=self.dim,
            heads=int(heads),
            prior_beta_init=float(prior_beta_init),
        )

        self.score_embed = nn.Linear(1, self.dim)
        self.pre_gate = nn.Parameter(torch.tensor(float(pre_gate_init)))

        self.head = nn.Sequential(
            nn.Linear(5 * self.dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    @staticmethod
    def _weighted_pool(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (w.unsqueeze(-1) * x).sum(dim=1)

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
        B, K, C = nb_v.shape
        if nb_t.shape != (B, K, C) or w.shape != (B, K) or nb_y.shape != (B, K, 1):
            raise RuntimeError("nb_t, w, and nb_y must align with nb_v")
        if q_v.shape != (B, C) or q_t.shape != (B, C):
            raise RuntimeError("Query representations must align with neighbor representations")
        if q_pre.shape != (B, 1):
            raise RuntimeError("q_pre must have shape [B, 1]")

        attention = self.attn(q_v=q_v, q_t=q_t, nb_v=nb_v, nb_t=nb_t, w=w)
        pool_v = self._weighted_pool(attention, nb_v)
        pool_t = self._weighted_pool(attention, nb_t)
        neighbor_score = self._weighted_pool(attention, nb_y).to(q_v.dtype)
        gate = torch.sigmoid(self.pre_gate).to(q_v.dtype)
        mixed_score = gate * q_pre.to(q_v.dtype) + (1.0 - gate) * neighbor_score
        score_embedding = self.score_embed(mixed_score)

        # Eq. (14): [Q_i; F_i^knn; E_i^s], where Q_i=[E_v^i; E_t^i].
        feat = torch.cat([q_v, q_t, pool_v, pool_t, score_embedding], dim=1)
        return self.head(feat).squeeze(1)
