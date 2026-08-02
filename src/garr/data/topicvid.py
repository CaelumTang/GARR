"""Build GARR TopicVid_douyin metadata from the official TopicVid JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_VIDEO_COUNT = 35_314
URL_ID_PATTERN = re.compile(r"/(?:share/)?video/(\d+)")


@dataclass(frozen=True)
class Day14Like:
    value: float
    method: str


def extract_url_id(url: object) -> int | None:
    if not isinstance(url, str):
        return None
    match = URL_ID_PATTERN.search(url)
    return int(match.group(1)) if match else None


def compute_day14_like(record: dict[str, Any]) -> Day14Like:
    post_time = record.get("post_create_time")
    time_frames = record.get("time_frames")
    if not isinstance(post_time, str) or not isinstance(time_frames, dict):
        raise ValueError("missing post_create_time or time_frames")

    try:
        post_date = datetime.strptime(post_time, "%Y-%m-%d %H:%M:%S").date()
    except ValueError as error:
        raise ValueError(f"invalid post_create_time={post_time!r}") from error

    observations: dict[int, float] = {}
    for date_text, metrics in time_frames.items():
        if not isinstance(date_text, str) or not isinstance(metrics, dict):
            continue
        try:
            frame_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            like_count = float(metrics["like_count"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(like_count):
            observations[(frame_date - post_date).days] = like_count

    if 14 in observations:
        return Day14Like(observations[14], "exact")
    before = [day for day in observations if day < 14]
    after = [day for day in observations if day > 14]
    if not before or not after:
        raise ValueError("day 14 is outside the observed range")
    lower_day, upper_day = max(before), min(after)
    lower_value, upper_value = observations[lower_day], observations[upper_day]
    value = lower_value + (14 - lower_day) * (upper_value - lower_value) / (upper_day - lower_day)
    return Day14Like(value, "interpolate")


def _read_selection(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "url_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{path}: expected columns video_id,url_id")
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append((int(row["video_id"]), int(row["url_id"])))
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"{path}:{line_number}: invalid ID") from error

    video_ids = [video_id for video_id, _ in rows]
    url_ids = [url_id for _, url_id in rows]
    if video_ids != list(range(len(rows))):
        raise RuntimeError(f"{path}: video_id must be contiguous and ordered from 0")
    if len(set(url_ids)) != len(url_ids):
        raise RuntimeError(f"{path}: duplicate url_id")
    return rows


def _is_douyin(platform: object) -> bool:
    return str(platform or "").strip().lower() in {"抖音", "douyin"}


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _scale_log_likes(log_likes: dict[int, float]) -> dict[int, float]:
    lower, upper = min(log_likes.values()), max(log_likes.values())
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise RuntimeError("TopicVid_douyin 14-day likes do not have a valid finite range")
    return {
        video_id: min(9.99, max(0.0, (value - lower) / (upper - lower) * 10.0))
        for video_id, value in log_likes.items()
    }


def build_metadata(
    raw_json: Path,
    selection_csv: Path,
    output_csv: Path,
    expected_count: int = EXPECTED_VIDEO_COUNT,
) -> dict[str, int]:
    selection = _read_selection(selection_csv)
    if expected_count and len(selection) != expected_count:
        raise RuntimeError(f"Expected {expected_count} selected videos, but found {len(selection)}")

    with raw_json.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    if not isinstance(raw_data, dict):
        raise RuntimeError(f"{raw_json}: JSON root must be an object")

    requested_url_ids = {url_id for _, url_id in selection}
    records_by_url: dict[int, dict[str, Any]] = {}
    douyin_url_ids: set[int] = set()
    for record in raw_data.values():
        if not isinstance(record, dict):
            continue
        url_id = extract_url_id(record.get("url"))
        if url_id in requested_url_ids:
            if url_id not in records_by_url:
                records_by_url[url_id] = record
            if _is_douyin(record.get("platform")):
                douyin_url_ids.add(url_id)

    missing = sorted(requested_url_ids - set(records_by_url))
    if missing:
        raise RuntimeError(f"Selected URL IDs missing from TopicVid JSON: {missing[:10]}")
    missing_douyin = sorted(requested_url_ids - douyin_url_ids)
    if missing_douyin:
        raise RuntimeError(
            f"Selected URL IDs have no Douyin record in TopicVid JSON: {missing_douyin[:10]}"
        )

    prepared: list[tuple[int, dict[str, Any], Day14Like]] = []
    method_counts = {"exact": 0, "interpolate": 0}
    log_likes: dict[int, float] = {}
    for video_id, url_id in selection:
        record = records_by_url[url_id]
        try:
            day14 = compute_day14_like(record)
        except ValueError as error:
            raise RuntimeError(f"url_id={url_id}: {error}") from error
        if not math.isfinite(day14.value) or day14.value <= 0.0:
            raise RuntimeError(f"url_id={url_id}: non-positive 14-day like count")
        log_likes[video_id] = math.log10(day14.value)
        method_counts[day14.method] += 1
        prepared.append((video_id, record, day14))

    scores = _scale_log_likes(log_likes)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "score", "topic", "title", "desc"])
        for video_id, record, _ in prepared:
            writer.writerow(
                [
                    video_id,
                    f"{scores[video_id]:.8f}",
                    _as_text(record.get("topic")),
                    _as_text(record.get("title")),
                    _as_text(record.get("desc")),
                ]
            )
    temporary.replace(output_csv)
    return {"videos": len(prepared), **method_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_VIDEO_COUNT)
    args = parser.parse_args()

    stats = build_metadata(
        raw_json=args.raw_json,
        selection_csv=args.selection_csv,
        output_csv=args.output_csv,
        expected_count=args.expected_count,
    )
    print(
        "Prepared TopicVid_douyin metadata: "
        f"videos={stats['videos']}, exact={stats['exact']}, "
        f"interpolated={stats['interpolate']}"
    )
    print(f"Output: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
