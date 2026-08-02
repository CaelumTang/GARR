"""Run GARR retrieval with train-to-train, val-to-train, and test-to-train+val galleries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    p = os.path.abspath(path)
    _require(os.path.isfile(p), f"Missing npz: {p}")
    with np.load(p) as data:
        keys = set(data.keys())
        required = {"video_id", "vision_emb", "text_emb", "ground_truth"}
        missing = sorted(required - keys)
        _require(not missing, f"{p}: missing keys={missing}, got keys={sorted(keys)}")
        return {key: np.asarray(data[key]) for key in required}


@dataclass(frozen=True)
class SplitTensors:
    ids: torch.Tensor
    v: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor


def _to_split_tensors(npz: dict[str, np.ndarray], *, device: torch.device) -> SplitTensors:
    ids = torch.from_numpy(np.asarray(npz["video_id"], dtype=np.int64))
    v = torch.from_numpy(np.asarray(npz["vision_emb"], dtype=np.float32))
    t = torch.from_numpy(np.asarray(npz["text_emb"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(npz["ground_truth"], dtype=np.float32))

    _require(ids.ndim == 1, f"video_id must be 1-D, got {tuple(ids.shape)}")
    _require(v.ndim == 2 and t.ndim == 2, "vision_emb/text_emb must be 2-D")
    _require(y.ndim == 1, "ground_truth must be 1-D")
    _require(v.shape[0] == ids.shape[0] == t.shape[0] == y.shape[0], "Split arrays must align on N")
    _require(v.shape[1] == t.shape[1], f"vision/text dim mismatch: v={v.shape[1]} t={t.shape[1]}")
    _require(len(set(ids.tolist())) == ids.numel(), "video_id values must be unique")
    _require(torch.isfinite(v).all().item(), "vision_emb contains NaN/Inf")
    _require(torch.isfinite(t).all().item(), "text_emb contains NaN/Inf")
    _require(torch.isfinite(y).all().item(), "ground_truth contains NaN/Inf")

    return SplitTensors(
        ids=ids.to(device=device),
        v=v.to(device=device),
        t=t.to(device=device),
        y=y.to(device=device),
    )


def _build_fused_key(v: torch.Tensor, t: torch.Tensor, rho: float) -> torch.Tensor:
    v_n = _l2norm(v)
    t_n = _l2norm(t)
    rho_tensor = torch.full((v_n.shape[0], 1), float(rho), device=v_n.device, dtype=v_n.dtype)
    return _l2norm(rho_tensor * v_n + (1.0 - rho_tensor) * t_n)


def _safe_sqrt_norm(norm2: torch.Tensor) -> torch.Tensor:
    norm2 = torch.clamp(norm2, min=0.0)
    n = torch.sqrt(norm2)
    return torch.where(n == 0, torch.ones_like(n), n)


@torch.no_grad()
def _srcc_grid_val_to_train_fast(
    *,
    tr_v: np.ndarray,
    tr_t: np.ndarray,
    tr_y: np.ndarray,
    va_v: np.ndarray,
    va_t: np.ndarray,
    va_y: np.ndarray,
    rhos: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute validation SRCC over the rho grid."""
    _require(rhos.ndim == 1 and rhos.size > 0, "rhos must be 1-D non-empty")
    _require(int(k) > 0, "k must be > 0")

    tr_v_t = torch.from_numpy(np.asarray(tr_v, dtype=np.float32)).to(device)
    tr_t_t = torch.from_numpy(np.asarray(tr_t, dtype=np.float32)).to(device)
    va_v_t = torch.from_numpy(np.asarray(va_v, dtype=np.float32)).to(device)
    va_t_t = torch.from_numpy(np.asarray(va_t, dtype=np.float32)).to(device)

    tr_v_n = _l2norm(tr_v_t)
    tr_t_n = _l2norm(tr_t_t)
    va_v_n = _l2norm(va_v_t)
    va_t_n = _l2norm(va_t_t)

    c_tr = torch.sum(tr_v_n * tr_t_n, dim=1)
    c_va_all = torch.sum(va_v_n * va_t_n, dim=1)

    tr_y_t = torch.from_numpy(np.asarray(tr_y, dtype=np.float32)).to(device)
    va_y_np = np.asarray(va_y, dtype=np.float64)

    n_va = int(va_v_n.shape[0])
    n_tr = int(tr_v_n.shape[0])
    k_eff = int(min(int(k), n_tr))
    _require(k_eff > 0, "k_eff must be > 0")

    rho_tensor = torch.from_numpy(np.asarray(rhos, dtype=np.float32)).to(device)
    one_minus_rho = 1.0 - rho_tensor
    rho_squared = rho_tensor * rho_tensor
    one_minus_rho_squared = one_minus_rho * one_minus_rho
    cross_weight = rho_tensor * one_minus_rho
    norm_tr = _safe_sqrt_norm(
        rho_squared[:, None]
        + one_minus_rho_squared[:, None]
        + (2.0 * cross_weight)[:, None] * c_tr[None, :]
    )  # [R,Ntr], where R is the number of rho candidates

    preds = np.empty((int(rhos.size), n_va), dtype=np.float32)

    for start in tqdm(
        range(0, n_va, int(batch_size)),
        desc="[Retrieval][rho search] validation -> train",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        end = min(n_va, start + int(batch_size))
        vq = va_v_n[start:end]
        tq = va_t_n[start:end]
        c_va = c_va_all[start:end]

        VV = vq @ tr_v_n.t()
        TT = tq @ tr_t_n.t()
        VT = vq @ tr_t_n.t()
        TV = tq @ tr_v_n.t()
        X = VT + TV

        norm_va = _safe_sqrt_norm(
            rho_squared[:, None]
            + one_minus_rho_squared[:, None]
            + (2.0 * cross_weight)[:, None] * c_va[None, :]
        )

        for rho_index in range(int(rhos.size)):
            numer = (
                rho_squared[rho_index] * VV
                + one_minus_rho_squared[rho_index] * TT
                + cross_weight[rho_index] * X
            )
            denom = norm_va[rho_index][:, None] * norm_tr[rho_index][None, :]
            sims = numer / denom

            top_vals, top_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
            nb_y = tr_y_t[top_idx]
            ssum = torch.sum(top_vals, dim=1, keepdim=True)
            pred_w = torch.sum((top_vals / ssum) * nb_y, dim=1)
            preds[rho_index, start:end] = pred_w.detach().cpu().numpy().astype(np.float32)

    srccs = np.empty((int(rhos.size),), dtype=np.float64)
    for rho_index in range(int(rhos.size)):
        srccs[rho_index] = float(
            spearmanr(va_y_np, preds[rho_index].astype(np.float64)).correlation
        )
    return srccs


def _rho_sweep_val_to_train(
    *,
    train_npz: dict[str, np.ndarray],
    val_npz: dict[str, np.ndarray],
    out_dir: str,
    k: int,
    rho_min: float,
    rho_max: float,
    rho_step: float,
    batch_size: int,
    device: torch.device,
) -> float:
    os.makedirs(out_dir, exist_ok=True)
    tr_v = np.asarray(train_npz["vision_emb"], dtype=np.float32)
    tr_t = np.asarray(train_npz["text_emb"], dtype=np.float32)
    tr_y = np.asarray(train_npz["ground_truth"], dtype=np.float32)
    va_v = np.asarray(val_npz["vision_emb"], dtype=np.float32)
    va_t = np.asarray(val_npz["text_emb"], dtype=np.float32)
    va_y = np.asarray(val_npz["ground_truth"], dtype=np.float32)

    step0 = float(rho_step)
    rhos0 = np.arange(float(rho_min), float(rho_max) + 1e-9, step0, dtype=np.float64)
    _require(rhos0.size > 0, "Empty rho grid. Check rho_min/rho_max/rho_step.")
    print(
        f"[Retrieval][rho search] grid: n={int(rhos0.size)} "
        f"range=[{float(rhos0.min()):.4f},{float(rhos0.max()):.4f}] step={step0:.6f} "
        f"k={int(k)} batch={int(batch_size)} device={device.type}"
    )
    srcc0 = _srcc_grid_val_to_train_fast(
        tr_v=tr_v,
        tr_t=tr_t,
        tr_y=tr_y,
        va_v=va_v,
        va_t=va_t,
        va_y=va_y,
        rhos=rhos0,
        k=int(k),
        batch_size=int(batch_size),
        device=device,
    )
    best0_idx = int(np.nanargmax(srcc0))
    best_rho = float(rhos0[best0_idx])
    best_srcc_val = float(srcc0[best0_idx])
    print(f"[Retrieval][rho search] best: rho={best_rho:.4f} srcc@{int(k)}={best_srcc_val:.6f}")

    import pandas as pd

    pd.DataFrame(
        {"rho": rhos0.astype(np.float64), "k": int(k), "srcc": srcc0.astype(np.float64)}
    ).sort_values(["srcc"], ascending=False).to_csv(
        os.path.join(out_dir, f"rho_sweep_val_to_train_k{int(k)}.csv"), index=False
    )
    return float(best_rho)


@torch.no_grad()
def _export_neighbors(
    *,
    out_csv: str,
    query_ids: torch.Tensor,
    query_z: torch.Tensor,
    gallery_ids: torch.Tensor,
    gallery_z: torch.Tensor,
    k: int,
    batch: int,
    exclude_self: bool,
    self_pos_map: dict[int, int] | None,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("video_id,topk_id,topk_sim\n")
        for start in range(0, int(query_ids.shape[0]), int(batch)):
            end = min(int(query_ids.shape[0]), start + int(batch))
            qid = query_ids[start:end]
            qz = query_z[start:end]
            sims = qz @ gallery_z.t()
            if exclude_self:
                _require(self_pos_map is not None, "self_pos_map required for exclude_self")
                pos = [self_pos_map[int(i)] for i in qid.detach().cpu().tolist()]
                row = torch.arange(sims.shape[0], device=sims.device)
                col = torch.as_tensor(pos, dtype=torch.long, device=sims.device)
                sims[row, col] = -1e9
            top_vals, top_idx = torch.topk(sims, k=int(k), dim=1, largest=True, sorted=True)
            top_ids = gallery_ids[top_idx]
            qid_cpu = qid.detach().cpu().numpy().astype(np.int64)
            top_ids_cpu = top_ids.detach().cpu().numpy().astype(np.int64)
            top_vals_cpu = top_vals.detach().cpu().numpy().astype(np.float32)
            for vid, nbs, vals in zip(
                qid_cpu.tolist(), top_ids_cpu.tolist(), top_vals_cpu.tolist()
            ):
                f.write(
                    f"{int(vid)},{' '.join(str(int(x)) for x in nbs)},{' '.join(str(float(v)) for v in vals)}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing train.npz, val.npz, and test.npz.",
    )
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--k-rho", type=int, default=20)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--rho-step", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _require(args.k_rho > 0, "--k-rho must be > 0")
    _require(args.k_max > 0, "--k-max must be > 0")
    _require(args.k_rho <= args.k_max, "--k-rho must be <= --k-max")
    _require(args.rho_step > 0.0, "--rho-step must be > 0")
    _require(args.rho_min <= args.rho_max, "--rho-min must be <= --rho-max")
    _require(args.batch_size > 0, "--batch-size must be > 0")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    features_dir = os.path.abspath(args.features_dir)
    _require(os.path.isdir(features_dir), f"features directory not found: {features_dir}")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)

    train_npz = _load_npz(os.path.join(features_dir, "train.npz"))
    val_npz = _load_npz(os.path.join(features_dir, "val.npz"))
    test_npz = _load_npz(os.path.join(features_dir, "test.npz"))
    _require(args.k_max < len(train_npz["video_id"]), "--k-max must be < train size")

    train_ids = set(np.asarray(train_npz["video_id"], dtype=np.int64).tolist())
    val_ids = set(np.asarray(val_npz["video_id"], dtype=np.int64).tolist())
    test_ids = set(np.asarray(test_npz["video_id"], dtype=np.int64).tolist())
    _require(not (train_ids & val_ids), "train and val video IDs overlap")
    _require(not (train_ids & test_ids), "train and test video IDs overlap")
    _require(not (val_ids & test_ids), "val and test video IDs overlap")

    rho_out = os.path.join(output_dir, "rho_search_results")
    os.makedirs(rho_out, exist_ok=True)
    best_rho = _rho_sweep_val_to_train(
        train_npz=train_npz,
        val_npz=val_npz,
        out_dir=rho_out,
        k=int(args.k_rho),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        rho_step=float(args.rho_step),
        batch_size=int(args.batch_size),
        device=device,
    )
    with open(os.path.join(output_dir, "selected_rho.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_rho}\n")

    train = _to_split_tensors(train_npz, device=device)
    val = _to_split_tensors(val_npz, device=device)
    test = _to_split_tensors(test_npz, device=device)

    k_max = int(args.k_max)
    k_rho = int(args.k_rho)
    with torch.no_grad():
        z_train = _build_fused_key(train.v, train.t, best_rho)
        z_val = _build_fused_key(val.v, val.t, best_rho)
        z_test = _build_fused_key(test.v, test.t, best_rho)

    train_pos = {int(i): int(p) for p, i in enumerate(train.ids.detach().cpu().tolist())}
    tv_ids = torch.cat([train.ids, val.ids], dim=0)
    tv_z = torch.cat([z_train, z_val], dim=0)

    _export_neighbors(
        out_csv=os.path.join(output_dir, "neighbors_train.csv"),
        query_ids=train.ids,
        query_z=z_train,
        gallery_ids=train.ids,
        gallery_z=z_train,
        k=k_max,
        batch=int(args.batch_size),
        exclude_self=True,
        self_pos_map=train_pos,
    )
    _export_neighbors(
        out_csv=os.path.join(output_dir, "neighbors_val.csv"),
        query_ids=val.ids,
        query_z=z_val,
        gallery_ids=train.ids,
        gallery_z=z_train,
        k=k_max,
        batch=int(args.batch_size),
        exclude_self=False,
        self_pos_map=None,
    )
    _export_neighbors(
        out_csv=os.path.join(output_dir, "neighbors_test.csv"),
        query_ids=test.ids,
        query_z=z_test,
        gallery_ids=tv_ids,
        gallery_z=tv_z,
        k=k_max,
        batch=int(args.batch_size),
        exclude_self=False,
        self_pos_map=None,
    )
    print(
        f"[Retrieval] exported neighbors under {output_dir} "
        f"(k_max={k_max}, k_rho={k_rho}, selected_rho={best_rho})"
    )
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
