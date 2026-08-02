"""Build ms-swift JSONL files backed by sharded HDF5-encoded images."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    metadata_fields: tuple[str, ...]
    system_prompt: str
    prediction_prompt: str


DATASETS = {
    "MicroLens": DatasetSpec(
        metadata_fields=("title", "category"),
        system_prompt=(
            "You are a helpful language-and-vision assistant for Micro-Video Popularity Prediction (MVPP). "
            "You can understand a cover image, a few video frames, text metadata (title and category), and an optional popularity-related video description provided by the user.\n"
            "The current MVPP target is to estimate the number of comments. Predict a single comment-popularity score in the range 0.00-9.99 with TWO decimals, derived from the comment count. "
            "Base your estimate only on the given visuals, metadata, and (if present) the user description; do not provide explanations. "
            "Output only one numeric value with two decimals (0.00-9.99); no extra text. If any field is missing, use the remaining inputs."
        ),
        prediction_prompt=(
            "Please predict the video's comment popularity by outputting ONE number "
            "between 0.00 and 9.99 with TWO decimals. Return only the number."
        ),
    ),
    "TopicVid_douyin": DatasetSpec(
        metadata_fields=("topic", "title", "desc"),
        system_prompt=(
            "You are a helpful language-and-vision assistant for Micro-Video Popularity Prediction (MVPP). "
            "You can understand a cover image, a few video frames, and text metadata (topic, title, description).\n"
            "The current MVPP target is to estimate the 14-day like popularity. "
            "Predict a single like-popularity score in the range 0.00-9.99 with TWO decimals, derived from the 14-day like count. "
            "Base your estimate only on the given visuals and metadata; do not provide explanations. "
            "Output only one numeric value with two decimals (0.00-9.99); no extra text. If any field is missing, use the remaining inputs."
        ),
        prediction_prompt=(
            "Please predict the video's like popularity by outputting ONE number "
            "between 0.00 and 9.99 with TWO decimals. Return only the number."
        ),
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_records(
    processed_csv: Path,
    split_csv: Path,
    spec: DatasetSpec,
) -> tuple[dict[int, dict[str, str]], dict[str, list[int]]]:
    processed_rows = _read_csv(processed_csv)
    required = {"video_id", "score", *spec.metadata_fields}
    missing = required - set(processed_rows[0].keys()) if processed_rows else required
    if missing:
        raise RuntimeError(f"{processed_csv}: missing columns {sorted(missing)}")

    records: dict[int, dict[str, str]] = {}
    for row in processed_rows:
        video_id = int(row["video_id"])
        if video_id in records:
            raise RuntimeError(f"{processed_csv}: duplicate video_id={video_id}")
        score = float(row["score"])
        if not 0.0 <= score <= 9.99:
            raise RuntimeError(f"{processed_csv}: score out of range for video_id={video_id}")
        records[video_id] = row

    split_rows = _read_csv(split_csv)
    if split_rows and set(split_rows[0]) != {"video_id", "split"}:
        raise RuntimeError(f"{split_csv}: expected columns video_id,split")

    split_ids: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    seen: set[int] = set()
    for row in split_rows:
        video_id = int(row["video_id"])
        split = row["split"].strip().lower()
        if split not in split_ids:
            raise RuntimeError(f"{split_csv}: invalid split={split!r}")
        if video_id in seen:
            raise RuntimeError(f"{split_csv}: duplicate video_id={video_id}")
        seen.add(video_id)
        split_ids[split].append(video_id)

    if seen != set(records):
        missing_in_split = sorted(set(records) - seen)[:10]
        missing_in_data = sorted(seen - set(records))[:10]
        raise RuntimeError(
            "Processed and split IDs differ: "
            f"missing_in_split={missing_in_split}, missing_in_data={missing_in_data}"
        )
    for values in split_ids.values():
        values.sort()
    return records, split_ids


def _image_urls(
    video_id: int,
    h5_root: Path,
    shard_count: int,
    num_frames: int,
) -> list[str]:
    shard_index = video_id % shard_count
    shard_path = (h5_root / f"vision_{shard_index:02d}.h5").resolve()
    urls = [f"h5://{shard_path}::/covers/{video_id}"]
    urls.extend(
        f"h5://{shard_path}::/frames/{video_id}/{frame_index}" for frame_index in range(num_frames)
    )
    return urls


def _user_prompt(
    row: dict[str, str],
    spec: DatasetSpec,
    num_frames: int,
) -> str:
    lines = ["Video-Cover: <image>"]
    lines.append(f"Video-Frames ({num_frames}): {' '.join(['<image>'] * num_frames)}")
    labels = {
        "title": "Video-Title",
        "category": "Video-Category",
        "topic": "Video-Topic",
        "desc": "Video-Description",
    }
    lines.extend(f"{labels[field]}: {row[field]}" for field in spec.metadata_fields)
    lines.append(spec.prediction_prompt)
    return "\n".join(lines)


def _write_split(
    output_path: Path,
    video_ids: Iterable[int],
    records: dict[int, dict[str, str]],
    spec: DatasetSpec,
    h5_root: Path,
    shard_count: int,
    num_frames: int,
    include_label: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temporary_path.open("w", encoding="utf-8") as handle:
        for video_id in video_ids:
            row = records[video_id]
            messages = [
                {"role": "system", "content": spec.system_prompt},
                {
                    "role": "user",
                    "content": _user_prompt(row, spec=spec, num_frames=num_frames),
                },
            ]
            if include_label:
                messages.append({"role": "assistant", "content": f"{float(row['score']):.2f}"})
            request = {
                "messages": messages,
                "images": _image_urls(
                    video_id,
                    h5_root=h5_root,
                    shard_count=shard_count,
                    num_frames=num_frames,
                ),
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    temporary_path.replace(output_path)
    return count


def build_dataset(
    dataset: str,
    processed_csv: Path,
    split_csv: Path,
    h5_root: Path,
    output_dir: Path,
    h5_shards: int = 16,
    num_frames: int = 16,
) -> dict[str, int]:
    if h5_shards <= 0:
        raise ValueError("h5_shards must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    missing_shards = [
        str(h5_root / f"vision_{index:02d}.h5")
        for index in range(h5_shards)
        if not (h5_root / f"vision_{index:02d}.h5").is_file()
    ]
    if missing_shards:
        raise FileNotFoundError(f"Missing HDF5 shards: {missing_shards}")

    spec = DATASETS[dataset]
    records, split_ids = _load_records(processed_csv, split_csv, spec)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        counts[split] = _write_split(
            output_dir / f"{split}.jsonl",
            split_ids[split],
            records=records,
            spec=spec,
            h5_root=h5_root,
            shard_count=h5_shards,
            num_frames=num_frames,
            include_label=split != "test",
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h5-shards", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()

    counts = build_dataset(
        dataset=args.dataset,
        processed_csv=args.processed_csv,
        split_csv=args.split_csv,
        h5_root=args.h5_root,
        output_dir=args.output_dir,
        h5_shards=args.h5_shards,
        num_frames=args.num_frames,
    )
    print(
        f"Prepared {args.dataset}: "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
    )
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
