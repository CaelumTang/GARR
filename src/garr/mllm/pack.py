"""Pack MLLM scores and representations into one NPZ file."""

import argparse
import csv
import math
import os

import numpy as np


def _load_ground_truth(ground_truth_csv: str) -> dict[int, float]:
    path = os.path.abspath(ground_truth_csv)
    ground_truth_by_id: dict[int, float] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {path}")
        if "video_id" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise RuntimeError(f"{path}: expected columns video_id,score, got {reader.fieldnames}")
        for row in reader:
            video_id = int(row["video_id"])
            if video_id in ground_truth_by_id:
                raise RuntimeError(f"{path}: duplicate video_id={video_id}")
            score = round(float(row["score"]), 2)
            if not math.isfinite(score):
                raise RuntimeError(f"{path}: non-finite score for video_id={video_id}")
            ground_truth_by_id[video_id] = score
    return ground_truth_by_id


def _read_predictions(pred_csv: str) -> tuple[list[int], list[str], list[str]]:
    path = os.path.abspath(pred_csv)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {path}")
        if "video_id" not in reader.fieldnames:
            raise RuntimeError(f"{path}: missing required column video_id, got {reader.fieldnames}")
        if "pre_score" not in reader.fieldnames:
            raise RuntimeError(
                f"{path}: missing required column pre_score (run postprocess first), got {reader.fieldnames}"
            )
        video_ids: list[int] = []
        ground_truth_values: list[str] = []
        pre_score_values: list[str] = []
        for row in reader:
            video_ids.append(int(row["video_id"]))
            ground_truth_values.append(str(row.get("ground_truth", "") or ""))
            pre_score_values.append(str(row.get("pre_score", "") or ""))
    if not video_ids:
        raise RuntimeError(f"No prediction rows in {path}")
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError(f"{path}: video_id values must be unique")
    return video_ids, ground_truth_values, pre_score_values


def _parse_float_or_nan(value: str) -> float:
    return float(value) if value else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing predictions/ and embeddings/.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=str, required=True, help="Output NPZ path.")
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default="",
        help="CSV containing video_id and score; required when labels are absent.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the split work directory after packing.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    split = args.split

    pred_csv = os.path.join(input_dir, "predictions", "gen_text.csv")
    emb_v_dir = os.path.join(input_dir, "embeddings", "vision")
    emb_t_dir = os.path.join(input_dir, "embeddings", "text")
    out_npz = os.path.abspath(args.output)

    video_ids, ground_truth_strings, pre_score_strings = _read_predictions(pred_csv)

    ground_truth_by_id: dict[int, float] = {}
    if any(value == "" for value in ground_truth_strings):
        if not args.ground_truth_csv:
            raise RuntimeError("Missing ground_truth values; provide --ground-truth-csv.")
        ground_truth_by_id = _load_ground_truth(args.ground_truth_csv)

    video_id_array = np.asarray(video_ids, dtype=np.int64)
    pre_score_array = np.asarray(
        [_parse_float_or_nan(value) for value in pre_score_strings],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(pre_score_array)):
        bad_ids = video_id_array[~np.isfinite(pre_score_array)].tolist()
        raise RuntimeError(
            f"Invalid pre_score for video IDs {bad_ids[:20]}; inspect generated text before packing."
        )

    ground_truth_values: list[float] = []
    for video_id, value in zip(video_ids, ground_truth_strings):
        if value != "":
            ground_truth_values.append(float(value))
        else:
            if video_id not in ground_truth_by_id:
                raise RuntimeError(f"video_id not found in ground-truth CSV: {video_id}")
            ground_truth_values.append(ground_truth_by_id[video_id])
    ground_truth_array = np.asarray(ground_truth_values, dtype=np.float32)
    if not np.all(np.isfinite(ground_truth_array)):
        raise RuntimeError("ground_truth contains NaN/Inf")

    vision_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    for video_id in video_ids:
        vision_path = os.path.join(emb_v_dir, f"{video_id}.npy")
        text_path = os.path.join(emb_t_dir, f"{video_id}.npy")
        vision = np.asarray(np.load(vision_path), dtype=np.float32).reshape(-1)
        text = np.asarray(np.load(text_path), dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(vision)) or not np.all(np.isfinite(text)):
            raise RuntimeError(f"Embedding contains NaN/Inf for video_id={video_id}")
        vision_embeddings.append(vision)
        text_embeddings.append(text)

    vision_array = np.stack(vision_embeddings, axis=0)
    text_array = np.stack(text_embeddings, axis=0)
    if vision_array.shape != text_array.shape:
        raise RuntimeError(
            f"Vision/text embedding shape mismatch: {vision_array.shape} != {text_array.shape}"
        )

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        video_id=video_id_array,
        vision_emb=vision_array,
        text_emb=text_array,
        pre_score=pre_score_array,
        ground_truth=ground_truth_array,
    )
    if args.cleanup:
        for directory in (emb_v_dir, emb_t_dir):
            for filename in os.listdir(directory):
                os.remove(os.path.join(directory, filename))
            os.rmdir(directory)
        os.rmdir(os.path.join(input_dir, "embeddings"))

        os.remove(pred_csv)
        os.rmdir(os.path.join(input_dir, "predictions"))

        run_config = os.path.join(input_dir, "metadata", "run_config.json")
        if os.path.isfile(run_config):
            os.remove(run_config)
            os.rmdir(os.path.join(input_dir, "metadata"))

        os.rmdir(input_dir)

    print(f"[GARR][pack_npz] split={split} rows={len(video_ids)} output={out_npz}")


if __name__ == "__main__":
    main()
