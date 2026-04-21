#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from predictor.dataset import Stage3Dataset, require
from predictor.model import Stage3Predictor


def infer_out_dir(*, checkpoint_dir: str, k: int) -> str:
    path = os.path.abspath(checkpoint_dir)
    parts = path.split(os.sep)
    require("artifacts" in parts, f"checkpoint_dir must be under artifacts/: {path}")
    index = parts.index("artifacts")
    require(index + 4 < len(parts), f"checkpoint_dir path too short: {path}")
    dataset = parts[index + 2]
    split_name = parts[index + 3]
    checkpoint_name = parts[index + 4]
    require(split_name.startswith("split_"), f"checkpoint_dir missing split_*/ segment: {path}")
    require(
        checkpoint_name.startswith("checkpoint-") or checkpoint_name == "base_model",
        f"checkpoint_dir missing checkpoint-*/ or base_model segment: {path}",
    )
    root = os.sep.join(parts[: index + 1])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root, "stage3", dataset, split_name, checkpoint_name, f"k{k}", f"run_{timestamp}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    srcc = float(spearmanr(y_true, y_pred).correlation)
    plcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    var = float(np.var(y_true))
    nmse = float(mse / (var + 1e-12))
    return mse, mae, srcc, plcc, nmse


@torch.no_grad()
def predict(model: Stage3Predictor, loader: DataLoader, device: torch.device):
    model.eval()
    all_targets = []
    all_predictions = []
    all_query_ids = []

    for batch in tqdm(loader, desc="Predict", leave=False):
        pred = model(
            q_v=batch["q_v"].to(device),
            q_t=batch["q_t"].to(device),
            nb_v=batch["nb_v"].to(device),
            nb_t=batch["nb_t"].to(device),
            w=batch["w"].to(device),
            q_pre=batch["q_pre"].to(device),
            nb_y=batch["nb_y"].to(device),
        )
        all_targets.append(batch["y"].to(device).squeeze(1).detach().cpu().numpy())
        all_predictions.append(pred.detach().cpu().numpy())
        all_query_ids.extend(int(item) for item in batch["qid"])

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    return all_query_ids, y_true, y_pred


@torch.no_grad()
def gate_stats(model: Stage3Predictor) -> dict[str, float]:
    pre_gate = getattr(model, "pre_gate", None)
    require(pre_gate is not None, "Expected model.pre_gate")
    prior_beta = getattr(model.attn, "prior_beta", None)
    return {
        "gate_global": float(torch.sigmoid(pre_gate.detach().cpu()).item()),
        "prior_beta": float(prior_beta.detach().cpu().item()) if prior_beta is not None else 0.0,
    }


def write_predictions(name: str, loader: DataLoader, model: Stage3Predictor, device: torch.device, out_dir: str):
    query_ids, y_true, y_pred = predict(model, loader, device)
    mse, mae, srcc, plcc, nmse = compute_metrics(y_true, y_pred)
    out_csv = os.path.join(out_dir, f"{name}_pred.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "pred"])
        for video_id, pred in zip(query_ids, y_pred.tolist()):
            writer.writerow([int(video_id), float(pred)])
    return out_csv, mse, mae, srcc, plcc, nmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GARR Stage-3 prediction")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--stage2_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--pre_gate_init", type=float, default=-3.0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    stage2_dir = os.path.abspath(args.stage2_dir)
    require(os.path.isdir(checkpoint_dir), f"checkpoint_dir not found: {checkpoint_dir}")
    require(os.path.isdir(stage2_dir), f"stage2_dir not found: {stage2_dir}")

    train_npz = os.path.join(checkpoint_dir, "train.npz")
    val_npz = os.path.join(checkpoint_dir, "val.npz")
    test_npz = os.path.join(checkpoint_dir, "test.npz")
    require(os.path.isfile(train_npz) and os.path.isfile(val_npz) and os.path.isfile(test_npz), "Missing train/val/test.npz")

    neighbors_train = os.path.join(stage2_dir, "neighbors_train.csv")
    neighbors_val = os.path.join(stage2_dir, "neighbors_val.csv")
    neighbors_test = os.path.join(stage2_dir, "neighbors_test.csv")
    require(
        os.path.isfile(neighbors_train) and os.path.isfile(neighbors_val) and os.path.isfile(neighbors_test),
        "Missing neighbors_{train,val,test}.csv in stage2_dir",
    )

    k = int(args.k)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else infer_out_dir(checkpoint_dir=checkpoint_dir, k=k)
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = Stage3Dataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=neighbors_train,
        split="train",
        k=k,
    )
    val_dataset = Stage3Dataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=neighbors_val,
        split="val",
        k=k,
    )
    test_dataset = Stage3Dataset(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        neighbors_csv=neighbors_test,
        split="test",
        k=k,
    )

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    device = torch.device(args.device)
    dim = int(train_dataset.tr.v.shape[1])
    model = Stage3Predictor(
        dim=dim,
        hidden=int(args.hidden),
        heads=int(args.heads),
        prior_beta_init=1.0,
        pre_gate_init=float(args.pre_gate_init),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.MSELoss()
    best_val_mse = float("inf")
    bad_epochs = 0

    for _epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in tqdm(train_loader, desc="Train", leave=False):
            pred = model(
                q_v=batch["q_v"].to(device),
                q_t=batch["q_t"].to(device),
                nb_v=batch["nb_v"].to(device),
                nb_t=batch["nb_t"].to(device),
                w=batch["w"].to(device),
                q_pre=batch["q_pre"].to(device),
                nb_y=batch["nb_y"].to(device),
            )
            target = batch["y"].to(device).squeeze(1)
            loss = loss_fn(pred, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss became NaN/Inf.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, val_y, val_pred = predict(model, val_loader, device)
        val_mse, _, _, _, _ = compute_metrics(val_y, val_pred)
        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pth"))
    with open(os.path.join(out_dir, "best_val_mse.txt"), "w", encoding="utf-8") as file:
        file.write(f"{best_val_mse}\n")

    best_path = os.path.join(out_dir, "best.pth")
    require(os.path.isfile(best_path), f"Missing best.pth: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device), strict=True)

    val_csv, val_mse, val_mae, val_srcc, val_plcc, val_nmse = write_predictions("val", val_loader, model, device, out_dir)
    test_csv, test_mse, test_mae, test_srcc, test_plcc, test_nmse = write_predictions("test", test_loader, model, device, out_dir)
    stats = gate_stats(model)

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as file:
        file.write(f"val_mse={val_mse}\nval_nmse={val_nmse}\nval_mae={val_mae}\nval_srcc={val_srcc}\nval_plcc={val_plcc}\n")
        file.write(f"test_mse={test_mse}\ntest_nmse={test_nmse}\ntest_mae={test_mae}\ntest_srcc={test_srcc}\ntest_plcc={test_plcc}\n")
        file.write(f"val_pred_csv={val_csv}\ntest_pred_csv={test_csv}\n")
        file.write(f"gate_global={stats['gate_global']}\n")
        file.write(f"prior_beta={stats['prior_beta']}\n")

    with open(os.path.join(out_dir, "config.txt"), "w", encoding="utf-8") as file:
        for key, value in sorted(vars(args).items()):
            file.write(f"{key}={value}\n")
        file.write(f"out_dir={out_dir}\n")


if __name__ == "__main__":
    main()
