#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def infer_split_name(final_dir: str) -> str:
    final_dir = os.path.abspath(final_dir)
    for split in ("train", "val", "test"):
        if split in final_dir.split(os.sep):
            return split
    raise RuntimeError(f"Cannot infer split from final_dir={final_dir!r}")


def load_ground_truth_map(all_csv: str) -> dict[int, float]:
    ground_truth: dict[int, float] = {}
    with open(os.path.abspath(all_csv), "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames is not None, f"Empty CSV: {all_csv}")
        require("video_id" in reader.fieldnames and "score" in reader.fieldnames, f"{all_csv}: expected columns video_id,score")
        for row in reader:
            ground_truth[int(row["video_id"])] = float(round(float(row["score"]), 2))
    return ground_truth


def read_predictions(pred_csv: str) -> tuple[list[int], list[str], list[str]]:
    with open(os.path.abspath(pred_csv), "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames is not None, f"Empty CSV: {pred_csv}")
        require("video_id" in reader.fieldnames, f"{pred_csv}: missing video_id")
        require("pre_score" in reader.fieldnames, f"{pred_csv}: missing pre_score")

        video_ids: list[int] = []
        ground_truth: list[str] = []
        pre_scores: list[str] = []
        for row in reader:
            video_ids.append(int(row["video_id"]))
            ground_truth.append(str(row.get("ground_truth", "") or ""))
            pre_scores.append(str(row.get("pre_score", "") or ""))
    return video_ids, ground_truth, pre_scores


def parse_float_or_nan(value: str) -> float:
    if value is None or str(value) == "":
        return float("nan")
    parsed = float(value)
    if not math.isfinite(parsed):
        return float("nan")
    return float(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack final embeddings into split.npz")
    parser.add_argument("--final_dir", type=str, required=True)
    parser.add_argument("--all_csv", type=str, default="datasets/MicroLens/processed/all.csv")
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = os.path.abspath(args.final_dir)
    split = infer_split_name(final_dir)

    pred_csv = os.path.join(final_dir, "predictions", "gen_text.csv")
    vision_dir = os.path.join(final_dir, "embeddings", "vision")
    text_dir = os.path.join(final_dir, "embeddings", "text")
    out_npz = os.path.join(final_dir, "embeddings", f"{split}.npz")

    require(os.path.isfile(pred_csv), f"Missing predictions CSV: {pred_csv}")
    require(os.path.isdir(vision_dir) and os.path.isdir(text_dir), "Missing embeddings directories")

    video_ids, ground_truth_strs, pre_score_strs = read_predictions(pred_csv)
    pre_score = np.asarray([parse_float_or_nan(item) for item in pre_score_strs], dtype=np.float32)

    ground_truth_map = load_ground_truth_map(args.all_csv) if split == "test" else {}
    ground_truth = []
    for video_id, value in zip(video_ids, ground_truth_strs):
        if value != "":
            ground_truth.append(float(value))
        else:
            require(split == "test", f"Missing ground_truth for non-test split: video_id={video_id}")
            require(video_id in ground_truth_map, f"video_id not found in all.csv: {video_id}")
            ground_truth.append(float(ground_truth_map[video_id]))

    vision_emb = []
    text_emb = []
    for video_id in video_ids:
        vision_path = os.path.join(vision_dir, f"{video_id}.npy")
        text_path = os.path.join(text_dir, f"{video_id}.npy")
        require(os.path.isfile(vision_path) and os.path.isfile(text_path), f"Missing embedding npy for video_id={video_id}")
        vision_emb.append(np.asarray(np.load(vision_path), dtype=np.float32).reshape(-1))
        text_emb.append(np.asarray(np.load(text_path), dtype=np.float32).reshape(-1))

    np.savez_compressed(
        out_npz,
        video_id=np.asarray(video_ids, dtype=np.int64),
        vision_emb=np.stack(vision_emb, axis=0),
        text_emb=np.stack(text_emb, axis=0),
        pre_score=pre_score,
        ground_truth=np.asarray(ground_truth, dtype=np.float32),
    )

    if args.cleanup:
        for directory in (vision_dir, text_dir):
            for name in os.listdir(directory):
                if name.endswith(".npy"):
                    os.remove(os.path.join(directory, name))
            os.rmdir(directory)


if __name__ == "__main__":
    main()
