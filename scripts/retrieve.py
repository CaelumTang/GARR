#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def load_npz(path: str) -> dict[str, np.ndarray]:
    path = os.path.abspath(path)
    require(os.path.isfile(path), f"Missing npz: {path}")
    data = np.load(path)
    required_keys = {"video_id", "vision_emb", "text_emb", "ground_truth"}
    missing_keys = sorted(required_keys - set(data.keys()))
    require(len(missing_keys) == 0, f"{path}: missing keys={missing_keys}")
    return {key: np.asarray(data[key]) for key in required_keys}


@dataclass(frozen=True)
class SplitTensors:
    ids: torch.Tensor
    v: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor


def to_split_tensors(npz: dict[str, np.ndarray], device: torch.device) -> SplitTensors:
    ids = torch.from_numpy(np.asarray(npz["video_id"], dtype=np.int64))
    vision = torch.from_numpy(np.asarray(npz["vision_emb"], dtype=np.float32))
    text = torch.from_numpy(np.asarray(npz["text_emb"], dtype=np.float32))
    target = torch.from_numpy(np.asarray(npz["ground_truth"], dtype=np.float32))

    require(ids.ndim == 1, f"video_id must be 1-D, got {tuple(ids.shape)}")
    require(vision.ndim == 2 and text.ndim == 2, "vision_emb/text_emb must be 2-D")
    require(target.ndim == 1, "ground_truth must be 1-D")
    require(vision.shape[0] == text.shape[0] == target.shape[0] == ids.shape[0], "Split arrays must align on N")
    require(vision.shape[1] == text.shape[1], "vision/text dim mismatch")

    return SplitTensors(
        ids=ids.to(device=device),
        v=vision.to(device=device),
        t=text.to(device=device),
        y=target.to(device=device),
    )


def infer_default_out_dir(checkpoint_dir: str) -> str:
    path = os.path.abspath(checkpoint_dir)
    parts = path.split(os.sep)
    require("artifacts" in parts, f"checkpoint_dir must be under artifacts/: {path}")
    index = parts.index("artifacts")
    require(index + 4 < len(parts), f"checkpoint_dir path too short: {path}")
    dataset = parts[index + 2]
    split_name = parts[index + 3]
    checkpoint_name = parts[index + 4]
    require(split_name.startswith("split_"), f"checkpoint_dir missing split_*/ segment: {path}")
    require(checkpoint_name.startswith("checkpoint-"), f"checkpoint_dir missing checkpoint-*/ segment: {path}")
    artifacts_root = os.sep.join(parts[: index + 1])
    return os.path.join(artifacts_root, "garr_stage2", dataset, split_name, checkpoint_name)


def build_fused_embeddings(vision: torch.Tensor, text: torch.Tensor, alpha: float) -> torch.Tensor:
    vision = l2norm(vision)
    text = l2norm(text)
    return l2norm(float(alpha) * vision + (1.0 - float(alpha)) * text)


def safe_sqrt_norm(norm2: torch.Tensor) -> torch.Tensor:
    norm2 = torch.clamp(norm2, min=0.0)
    norm = torch.sqrt(norm2)
    return torch.where(norm == 0, torch.ones_like(norm), norm)


@torch.no_grad()
def search_best_alpha(
    *,
    train_v: np.ndarray,
    train_t: np.ndarray,
    train_y: np.ndarray,
    val_v: np.ndarray,
    val_t: np.ndarray,
    val_y: np.ndarray,
    alphas: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    require(alphas.ndim == 1 and alphas.size > 0, "alphas must be non-empty")
    require(k > 0, "k must be > 0")

    train_v = l2norm(torch.from_numpy(np.asarray(train_v, dtype=np.float32)).to(device))
    train_t = l2norm(torch.from_numpy(np.asarray(train_t, dtype=np.float32)).to(device))
    val_v = l2norm(torch.from_numpy(np.asarray(val_v, dtype=np.float32)).to(device))
    val_t = l2norm(torch.from_numpy(np.asarray(val_t, dtype=np.float32)).to(device))

    train_cross = torch.sum(train_v * train_t, dim=1)
    val_cross = torch.sum(val_v * val_t, dim=1)
    train_y_tensor = torch.from_numpy(np.asarray(train_y, dtype=np.float32)).to(device)
    val_y_array = np.asarray(val_y, dtype=np.float64)

    k = min(int(k), int(train_v.shape[0]))
    require(k > 0, "effective k must be > 0")

    alpha_tensor = torch.from_numpy(np.asarray(alphas, dtype=np.float32)).to(device)
    beta_tensor = 1.0 - alpha_tensor
    alpha_sq = alpha_tensor * alpha_tensor
    beta_sq = beta_tensor * beta_tensor
    alpha_beta = alpha_tensor * beta_tensor
    train_norm = safe_sqrt_norm(
        alpha_sq[:, None] + beta_sq[:, None] + (2.0 * alpha_beta)[:, None] * train_cross[None, :]
    )

    predictions = np.empty((int(alphas.size), int(val_v.shape[0])), dtype=np.float32)

    for start in range(0, int(val_v.shape[0]), int(batch_size)):
        end = min(int(val_v.shape[0]), start + int(batch_size))
        query_v = val_v[start:end]
        query_t = val_t[start:end]
        query_cross = val_cross[start:end]

        vv = query_v @ train_v.t()
        tt = query_t @ train_t.t()
        vt = query_v @ train_t.t()
        tv = query_t @ train_v.t()
        cross = vt + tv

        val_norm = safe_sqrt_norm(
            alpha_sq[:, None] + beta_sq[:, None] + (2.0 * alpha_beta)[:, None] * query_cross[None, :]
        )

        for alpha_index in range(int(alphas.size)):
            scores = alpha_sq[alpha_index] * vv + beta_sq[alpha_index] * tt + alpha_beta[alpha_index] * cross
            scores = scores / (val_norm[alpha_index][:, None] * train_norm[alpha_index][None, :])
            top_scores, top_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            neighbor_y = train_y_tensor[top_indices]
            score_sum = torch.sum(top_scores, dim=1, keepdim=True)
            weighted_pred = torch.sum((top_scores / score_sum) * neighbor_y, dim=1)
            mean_pred = torch.mean(neighbor_y, dim=1)
            pred = torch.where(score_sum.squeeze(1) > 0.0, weighted_pred, mean_pred)
            predictions[alpha_index, start:end] = pred.detach().cpu().numpy().astype(np.float32)

    srcc_values = np.empty((int(alphas.size),), dtype=np.float64)
    for alpha_index in range(int(alphas.size)):
        srcc_values[alpha_index] = float(spearmanr(val_y_array, predictions[alpha_index].astype(np.float64)).correlation)

    best_index = int(np.nanargmax(srcc_values))
    return float(alphas[best_index]), float(srcc_values[best_index]), srcc_values


def run_alpha_search(
    *,
    checkpoint_dir: str,
    out_dir: str,
    k: int,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    batch_size: int,
    device: torch.device,
) -> float:
    train_npz = load_npz(os.path.join(checkpoint_dir, "train.npz"))
    val_npz = load_npz(os.path.join(checkpoint_dir, "val.npz"))

    alphas = np.arange(float(alpha_min), float(alpha_max) + 1e-9, float(alpha_step), dtype=np.float64)
    require(alphas.size > 0, "Empty alpha grid")

    best_alpha, best_srcc, srcc_values = search_best_alpha(
        train_v=train_npz["vision_emb"],
        train_t=train_npz["text_emb"],
        train_y=train_npz["ground_truth"],
        val_v=val_npz["vision_emb"],
        val_t=val_npz["text_emb"],
        val_y=val_npz["ground_truth"],
        alphas=alphas,
        k=int(k),
        batch_size=int(batch_size),
        device=device,
    )

    alpha_dir = os.path.join(out_dir, "alpha_search")
    os.makedirs(alpha_dir, exist_ok=True)
    pd.DataFrame({"alpha": alphas, "k": int(k), "srcc": srcc_values}).sort_values(
        ["srcc"], ascending=False
    ).to_csv(os.path.join(alpha_dir, f"alpha_sweep_k{int(k)}.csv"), index=False)
    with open(os.path.join(out_dir, "best_alpha.txt"), "w", encoding="utf-8") as file:
        file.write(f"{best_alpha:.6f}\n")
    with open(os.path.join(out_dir, "best_srcc.txt"), "w", encoding="utf-8") as file:
        file.write(f"{best_srcc:.6f}\n")
    return best_alpha


@torch.no_grad()
def export_neighbors(
    *,
    out_csv: str,
    query_ids: torch.Tensor,
    query_embeddings: torch.Tensor,
    gallery_ids: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    k: int,
    batch_size: int,
    exclude_self: bool,
    self_pos_map: dict[int, int] | None,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as file:
        file.write("video_id,topk_id,topk_sim\n")
        for start in range(0, int(query_ids.shape[0]), int(batch_size)):
            end = min(int(query_ids.shape[0]), start + int(batch_size))
            batch_ids = query_ids[start:end]
            batch_embeddings = query_embeddings[start:end]
            scores = batch_embeddings @ gallery_embeddings.t()
            if exclude_self:
                require(self_pos_map is not None, "self_pos_map is required when exclude_self=True")
                positions = [self_pos_map[int(video_id)] for video_id in batch_ids.detach().cpu().tolist()]
                row_index = torch.arange(scores.shape[0], device=scores.device)
                col_index = torch.as_tensor(positions, dtype=torch.long, device=scores.device)
                scores[row_index, col_index] = -1e9
            top_scores, top_indices = torch.topk(scores, k=int(k), dim=1, largest=True, sorted=True)
            top_ids = gallery_ids[top_indices]

            query_ids_cpu = batch_ids.detach().cpu().numpy().astype(np.int64)
            top_ids_cpu = top_ids.detach().cpu().numpy().astype(np.int64)
            top_scores_cpu = top_scores.detach().cpu().numpy().astype(np.float32)
            for video_id, neighbor_ids, neighbor_scores in zip(
                query_ids_cpu.tolist(), top_ids_cpu.tolist(), top_scores_cpu.tolist()
            ):
                file.write(
                    f"{int(video_id)},{' '.join(str(int(item)) for item in neighbor_ids)},{' '.join(str(float(item)) for item in neighbor_scores)}\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GARR Stage-2 retrieval")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--k_alpha", type=int, default=20)
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_step", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    require(os.path.isdir(checkpoint_dir), f"checkpoint_dir not found: {checkpoint_dir}")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else infer_default_out_dir(checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)

    if str(args.device).lower().startswith("cuda") and str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
    device = torch.device(args.device)

    best_alpha = run_alpha_search(
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        k=int(args.k_alpha),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_step=float(args.alpha_step),
        batch_size=int(args.batch_size),
        device=device,
    )

    train = to_split_tensors(load_npz(os.path.join(checkpoint_dir, "train.npz")), device)
    val = to_split_tensors(load_npz(os.path.join(checkpoint_dir, "val.npz")), device)
    test = to_split_tensors(load_npz(os.path.join(checkpoint_dir, "test.npz")), device)

    k_max = int(args.k_max)
    k_alpha = int(args.k_alpha)
    require(k_alpha > 0, "--k_alpha must be > 0")
    require(k_max > 0, "--k_max must be > 0")
    require(k_alpha <= k_max, f"Require k_alpha <= k_max, got {k_alpha} > {k_max}")
    require(k_max < int(train.ids.shape[0]), "--k_max must be smaller than train size")

    fused_train = build_fused_embeddings(train.v, train.t, best_alpha)
    fused_val = build_fused_embeddings(val.v, val.t, best_alpha)
    fused_test = build_fused_embeddings(test.v, test.t, best_alpha)

    train_pos = {int(video_id): int(position) for position, video_id in enumerate(train.ids.detach().cpu().tolist())}
    train_val_ids = torch.cat([train.ids, val.ids], dim=0)
    train_val_embeddings = torch.cat([fused_train, fused_val], dim=0)

    export_neighbors(
        out_csv=os.path.join(out_dir, "neighbors_train.csv"),
        query_ids=train.ids,
        query_embeddings=fused_train,
        gallery_ids=train.ids,
        gallery_embeddings=fused_train,
        k=k_max,
        batch_size=int(args.batch_size),
        exclude_self=True,
        self_pos_map=train_pos,
    )
    export_neighbors(
        out_csv=os.path.join(out_dir, "neighbors_val.csv"),
        query_ids=val.ids,
        query_embeddings=fused_val,
        gallery_ids=train.ids,
        gallery_embeddings=fused_train,
        k=k_max,
        batch_size=int(args.batch_size),
        exclude_self=False,
        self_pos_map=None,
    )
    export_neighbors(
        out_csv=os.path.join(out_dir, "neighbors_test.csv"),
        query_ids=test.ids,
        query_embeddings=fused_test,
        gallery_ids=train_val_ids,
        gallery_embeddings=train_val_embeddings,
        k=k_max,
        batch_size=int(args.batch_size),
        exclude_self=False,
        self_pos_map=None,
    )


if __name__ == "__main__":
    main()
