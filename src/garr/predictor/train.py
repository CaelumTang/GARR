import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from garr.predictor.dataset import PredictorDataset, _require
from garr.predictor.model import GARRPredictor


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float, float]:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    _require(
        y.ndim == p.ndim == 1 and y.size == p.size,
        "Targets and predictions must be aligned vectors",
    )
    _require(y.size >= 2, "At least two samples are required to compute correlations")
    _require(
        np.all(np.isfinite(y)) and np.all(np.isfinite(p)), "Targets or predictions contain NaN/Inf"
    )
    mse = float(np.mean((p - y) ** 2))
    mae = float(np.mean(np.abs(p - y)))
    srcc = float(spearmanr(y, p).correlation)
    plcc = float(np.corrcoef(y, p)[0, 1])
    var = float(np.var(y))
    nmse = float(mse / (var + 1e-12))
    return mse, mae, srcc, plcc, nmse


@torch.no_grad()
def _predict(
    model: GARRPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []
    qids = []
    for batch in loader:
        q_v = batch["q_v"].to(device)
        q_t = batch["q_t"].to(device)
        nb_v = batch["nb_v"].to(device)
        nb_t = batch["nb_t"].to(device)
        w = batch["w"].to(device)
        y = batch["y"].to(device).squeeze(1)
        q_pre = batch["q_pre"].to(device)
        nb_y = batch["nb_y"].to(device)
        out = model(q_v=q_v, q_t=q_t, nb_v=nb_v, nb_t=nb_t, w=w, q_pre=q_pre, nb_y=nb_y)
        ys.append(y.detach().cpu().numpy())
        ps.append(out.detach().cpu().numpy())
        qids.extend([int(x) for x in batch["qid"]])
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return qids, y, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GARR Retrieval Refinement module.")
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Generative Alignment feature directory containing train.npz, val.npz, and test.npz.",
    )
    parser.add_argument(
        "--retrieval-dir",
        type=str,
        required=True,
        help="Directory containing neighbors_train.csv, neighbors_val.csv, and neighbors_test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints, predictions, metrics, and configuration.",
    )

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument(
        "--pre-gate-init",
        type=float,
        default=-3.0,
        help="Init of sigmoid gate for pre_score branch.",
    )

    parser.add_argument("--heads", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    _require(args.epochs > 0, "--epochs must be > 0")
    _require(args.patience > 0, "--patience must be > 0")
    _require(args.batch_size > 0, "--batch-size must be > 0")
    _require(args.num_workers >= 0, "--num-workers must be >= 0")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    features_dir = os.path.abspath(args.features_dir)
    retrieval_dir = os.path.abspath(args.retrieval_dir)

    train_npz = os.path.join(features_dir, "train.npz")
    val_npz = os.path.join(features_dir, "val.npz")
    test_npz = os.path.join(features_dir, "test.npz")

    k = int(args.k)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_tr = os.path.join(retrieval_dir, "neighbors_train.csv")
    n_va = os.path.join(retrieval_dir, "neighbors_val.csv")
    n_te = os.path.join(retrieval_dir, "neighbors_test.csv")

    ds_tr = PredictorDataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=n_tr,
        split="train",
        k=k,
    )
    ds_va = PredictorDataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=n_va,
        split="val",
        k=k,
    )
    ds_te = PredictorDataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=n_te,
        split="test",
        k=k,
    )

    pin_memory = str(args.device).startswith("cuda")
    loader_tr = DataLoader(
        ds_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )
    loader_te = DataLoader(
        ds_te,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    dim = int(ds_tr.tr.v.shape[1])
    device = torch.device(args.device)
    model = GARRPredictor(
        dim=dim,
        hidden=int(args.hidden),
        heads=int(args.heads),
        pre_gate_init=float(args.pre_gate_init),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    loss_fn = torch.nn.MSELoss()

    best_val_mse = float("inf")
    bad = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        cnt = 0
        for batch in tqdm(
            loader_tr,
            desc=f"Train ep{epoch}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            q_v = batch["q_v"].to(device)
            q_t = batch["q_t"].to(device)
            nb_v = batch["nb_v"].to(device)
            nb_t = batch["nb_t"].to(device)
            w = batch["w"].to(device)
            y = batch["y"].to(device).squeeze(1)
            q_pre = batch["q_pre"].to(device)
            nb_y = batch["nb_y"].to(device)

            pred = model(q_v=q_v, q_t=q_t, nb_v=nb_v, nb_t=nb_t, w=w, q_pre=q_pre, nb_y=nb_y)
            loss = loss_fn(pred, y)
            _require(torch.isfinite(loss).item(), f"Loss became NaN/Inf at epoch {epoch}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item()) * int(y.shape[0])
            cnt += int(y.shape[0])

        train_mse = total / cnt
        _, y_va, p_va = _predict(model, loader_va, device)
        val_mse, val_mae, val_srcc, val_plcc, val_nmse = _metrics(y_va, p_va)
        print(
            f"[GARR][ep {epoch}] train_mse={train_mse:.6f} | "
            f"val_mse={val_mse:.6f} val_nmse={val_nmse:.6f} val_mae={val_mae:.6f} "
            f"val_srcc={val_srcc:.6f} val_plcc={val_plcc:.6f}",
            flush=True,
        )

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            bad = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
        else:
            bad += 1
            if bad >= int(args.patience):
                break

    best_path = os.path.join(output_dir, "best.pth")
    _require(os.path.isfile(best_path), f"Missing best.pth: {best_path}")
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    def _write_pred(name: str, loader: DataLoader) -> dict[str, float]:
        qids, y, p = _predict(model, loader, device)
        mse, mae, srcc, plcc, nmse = _metrics(y, p)
        out_csv = os.path.join(output_dir, f"{name}_pred.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["video_id", "pred"])
            for vid, pred in zip(qids, p.tolist()):
                wcsv.writerow([int(vid), float(pred)])
        return {"mse": mse, "nmse": nmse, "mae": mae, "srcc": srcc, "plcc": plcc}

    metrics = {
        "val": _write_pred("val", loader_va),
        "test": _write_pred("test", loader_te),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    run_config = vars(args).copy()
    run_config.update(
        {
            "features_dir": features_dir,
            "retrieval_dir": retrieval_dir,
            "output_dir": output_dir,
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    print(f"[GARR] done: output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
