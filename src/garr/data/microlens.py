"""Build GARR MicroLens metadata from the official MicroLens-100k files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

EXPECTED_VIDEO_COUNT = 19_738
MINIMUM_LIKES = 10_000


def _read_interaction_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "item" not in reader.fieldnames:
            raise RuntimeError(f"{path}: expected a CSV containing an 'item' column")
        for line_number, row in enumerate(reader, start=2):
            try:
                video_id = int(row["item"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"{path}:{line_number}: invalid item ID") from error
            counts[video_id] += 1
    if not counts:
        raise RuntimeError(f"{path}: no interactions found")
    return counts


def _read_two_column_csv(path: Path, value_name: str) -> dict[int, str]:
    values: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 2:
                raise RuntimeError(f"{path}:{line_number}: expected at least two columns")
            try:
                video_id = int(row[0].strip())
            except ValueError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid video ID") from error
            if video_id in values:
                raise RuntimeError(f"{path}:{line_number}: duplicate video_id={video_id}")
            values[video_id] = row[1].strip()
    if not values:
        raise RuntimeError(f"{path}: no {value_name} values found")
    return values


def _read_likes(path: Path) -> dict[int, int]:
    likes: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = re.split(r"[,\s]+", line.strip())
            if len(parts) < 3:
                raise RuntimeError(f"{path}:{line_number}: expected video_id, likes, views")
            try:
                video_id, like_count = int(parts[0]), int(parts[1])
            except ValueError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid integer value") from error
            if video_id in likes:
                raise RuntimeError(f"{path}:{line_number}: duplicate video_id={video_id}")
            likes[video_id] = like_count
    if not likes:
        raise RuntimeError(f"{path}: no like counts found")
    return likes


def _scale_log_counts(counts: dict[int, int]) -> dict[int, float]:
    log_counts = {video_id: math.log2(count) for video_id, count in counts.items()}
    lower, upper = min(log_counts.values()), max(log_counts.values())
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise RuntimeError("MicroLens comment counts do not have a valid finite range")
    return {
        video_id: min(9.99, max(0.0, (value - lower) / (upper - lower) * 10.0))
        for video_id, value in log_counts.items()
    }


def build_metadata(
    raw_dir: Path,
    output_csv: Path,
    minimum_likes: int = MINIMUM_LIKES,
    expected_count: int = EXPECTED_VIDEO_COUNT,
) -> int:
    interactions = _read_interaction_counts(raw_dir / "MicroLens-100k_pairs.csv")
    titles = _read_two_column_csv(raw_dir / "MicroLens-100k_title_en.csv", "titles")
    categories = _read_two_column_csv(raw_dir / "tags_to_summary.csv", "categories")
    likes = _read_likes(raw_dir / "MicroLens-100k_likes_and_views.txt")

    video_ids = sorted(video_id for video_id, count in likes.items() if count >= minimum_likes)
    if expected_count and len(video_ids) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} videos with at least {minimum_likes} likes, "
            f"but found {len(video_ids)}"
        )
    if video_ids != list(range(video_ids[0], video_ids[-1] + 1)):
        raise RuntimeError("Eligible MicroLens video IDs are not contiguous")

    required_sources = {
        "interaction counts": set(interactions),
        "titles": set(titles),
        "categories": set(categories),
    }
    selected = set(video_ids)
    for source_name, available in required_sources.items():
        missing = sorted(selected - available)
        if missing:
            raise RuntimeError(f"Missing {source_name} for video IDs: {missing[:10]}")

    selected_counts = {video_id: interactions[video_id] for video_id in video_ids}
    scores = _scale_log_counts(selected_counts)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "score", "category", "title"])
        for video_id in video_ids:
            writer.writerow(
                [video_id, f"{scores[video_id]:.8f}", categories[video_id], titles[video_id]]
            )
    temporary.replace(output_csv)
    return len(video_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--minimum-likes", type=int, default=MINIMUM_LIKES)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_VIDEO_COUNT)
    args = parser.parse_args()

    count = build_metadata(
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        minimum_likes=args.minimum_likes,
        expected_count=args.expected_count,
    )
    print(f"Prepared MicroLens metadata: videos={count}")
    print(f"Output: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
