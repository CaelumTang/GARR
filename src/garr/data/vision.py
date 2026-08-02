"""Create the sharded HDF5 visual inputs used by GARR."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


def _read_video_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise RuntimeError(f"{path}: expected a video_id column")
        video_ids = [int(row["video_id"]) for row in reader]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError(f"{path}: duplicate video_id")
    if not video_ids:
        raise RuntimeError(f"{path}: no videos found")
    return sorted(video_ids)


def _read_asset_ids(
    video_ids: list[int], asset_map: Path | None, asset_id_column: str
) -> dict[int, str]:
    if asset_map is None:
        return {video_id: str(video_id) for video_id in video_ids}

    mapping: dict[int, str] = {}
    with asset_map.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", asset_id_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{asset_map}: expected columns video_id,{asset_id_column}")
        for line_number, row in enumerate(reader, start=2):
            video_id = int(row["video_id"])
            asset_id = row[asset_id_column].strip()
            if not asset_id:
                raise RuntimeError(f"{asset_map}:{line_number}: empty asset ID")
            if video_id in mapping:
                raise RuntimeError(f"{asset_map}:{line_number}: duplicate video_id={video_id}")
            mapping[video_id] = asset_id

    missing = sorted(set(video_ids) - set(mapping))
    extra = sorted(set(mapping) - set(video_ids))
    if missing or extra:
        raise RuntimeError(
            f"Processed and asset-map IDs differ: missing={missing[:10]}, extra={extra[:10]}"
        )
    return mapping


def _add_unique(index: dict[object, Path], key: object, path: Path) -> None:
    previous = index.get(key)
    if previous is not None and previous != path:
        raise RuntimeError(f"Ambiguous asset key {key!r}: {previous} and {path}")
    index[key] = path


def _index_assets(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {directory}")
    index: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        _add_unique(index, path.stem, path)
    return index


def _resolve_file(index: dict[str, Path], asset_id: str, asset_type: str) -> Path:
    try:
        return index[asset_id]
    except KeyError as error:
        raise FileNotFoundError(f"Missing {asset_type} for asset_id={asset_id}") from error


def _resolve_topicvid_file(directory: Path, asset_id: str, suffix: str) -> Path:
    path = directory / f"{asset_id}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TopicVid_douyin asset: {path}")
    return path


def _read_image_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        image.verify()
    return path.read_bytes()


def _encode_jpeg(frame: np.ndarray) -> bytes:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _extract_uniform_frames(video_path: Path, frame_count: int) -> list[bytes]:
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise RuntimeError("Video decoding requires decord; run bash setup.sh") from error

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frames = len(reader)
    if total_frames <= 0:
        raise RuntimeError(f"No decodable frames in {video_path}")

    indices = np.linspace(0, total_frames - 1, num=frame_count, dtype=np.int64)
    return [_encode_jpeg(frame) for frame in reader.get_batch(indices).asnumpy()]


def _write_bytes(group: h5py.Group, key: str, payload: bytes) -> None:
    group.create_dataset(
        key,
        data=np.frombuffer(payload, dtype=np.uint8),
        compression="lzf",
    )


def build_vision_shards(
    processed_csv: Path,
    covers_dir: Path,
    videos_dir: Path,
    output_dir: Path,
    asset_map: Path | None = None,
    asset_id_column: str = "url_id",
    shard_count: int = 16,
    frame_count: int = 16,
) -> dict[str, int]:
    if shard_count <= 0 or frame_count <= 0:
        raise ValueError("shard_count and frame_count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    video_ids = _read_video_ids(processed_csv)
    asset_ids = _read_asset_ids(video_ids, asset_map, asset_id_column)
    topicvid_layout = asset_map is not None
    if topicvid_layout:
        if not covers_dir.is_dir() or not videos_dir.is_dir():
            raise FileNotFoundError("TopicVid_douyin covers/ and videos/ directories are required")
        cover_index = video_index = None
    else:
        cover_index = _index_assets(covers_dir, IMAGE_EXTENSIONS)
        video_index = _index_assets(videos_dir, VIDEO_EXTENSIONS)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    shards = [h5py.File(temporary / f"vision_{index:02d}.h5", "w") for index in range(shard_count)]

    try:
        for video_id in tqdm(
            video_ids,
            desc="visual inputs",
            unit="video",
            disable=not sys.stderr.isatty(),
        ):
            asset_id = asset_ids[video_id]
            if topicvid_layout:
                cover_path = _resolve_topicvid_file(covers_dir, asset_id, ".jpg")
                video_path = _resolve_topicvid_file(videos_dir, asset_id, ".mp4")
            else:
                cover_path = _resolve_file(cover_index, asset_id, "cover")
                video_path = _resolve_file(video_index, asset_id, "video")
            frames = _extract_uniform_frames(video_path, frame_count)

            shard = shards[video_id % shard_count]
            _write_bytes(
                shard.require_group("covers"), str(video_id), _read_image_bytes(cover_path)
            )
            frame_group = shard.require_group(f"frames/{video_id}")
            for frame_number, payload in enumerate(frames):
                _write_bytes(frame_group, str(frame_number), payload)
    except Exception:
        for shard in shards:
            shard.close()
        shutil.rmtree(temporary)
        raise
    else:
        for shard in shards:
            shard.flush()
            shard.close()

    temporary.replace(output_dir)
    return {
        "videos": len(video_ids),
        "covers": len(video_ids),
        "frames": len(video_ids) * frame_count,
        "shards": shard_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--covers-dir", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--asset-map", type=Path)
    parser.add_argument("--asset-id-column", default="url_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    stats = build_vision_shards(
        processed_csv=args.processed_csv,
        covers_dir=args.covers_dir,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        asset_map=args.asset_map,
        asset_id_column=args.asset_id_column,
        shard_count=args.shards,
        frame_count=args.frames,
    )
    print(
        "Prepared visual inputs: " + ", ".join(f"{name}={value}" for name, value in stats.items())
    )
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
