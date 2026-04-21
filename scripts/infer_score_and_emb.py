#!/usr/bin/env python3
"""Run Stage-1 inference and export generated scores with optional embeddings."""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Any

import numpy as np
from tqdm import tqdm

from swift.llm import get_model_tokenizer, get_template
from swift.llm.infer.infer_engine.pt_engine import PtEngine
from swift.llm.infer.protocol import RequestConfig
from swift.utils import read_from_jsonl


def load_requests(jsonl_path: str) -> list[dict[str, Any]]:
    rows = read_from_jsonl(jsonl_path)
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"Empty or invalid jsonl: {jsonl_path}")
    return rows


def extract_video_id_from_images(images: list[Any]) -> int:
    if not isinstance(images, list) or len(images) == 0:
        raise RuntimeError("Missing images list; cannot infer video_id.")

    image_ref = str(images[0])
    match = re.search(r"/covers/(\d+)", image_ref)
    if match is None:
        match = re.search(r"/frames/(\d+)/", image_ref)
    if match is None:
        raise RuntimeError(f"Failed to parse video_id from images[0]={image_ref!r}")
    return int(match.group(1))


def normalize_rows_for_infer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"Invalid row type: {type(row)}")

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            raise RuntimeError("Row missing messages list.")

        objects = row.get("objects")
        if objects is None:
            objects = {}
            row["objects"] = objects
        if not isinstance(objects, dict):
            raise RuntimeError("Row.objects must be a dict.")

        if isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
            ground_truth = messages[-1].get("content", "")
            objects.setdefault("ground_truth", "" if ground_truth is None else str(ground_truth))
            row["messages"] = messages[:-1]

        if "video_id" not in objects and "video_id" not in row:
            objects["video_id"] = extract_video_id_from_images(row.get("images", []))
        elif "video_id" in row and "video_id" not in objects:
            objects["video_id"] = int(row["video_id"])

        normalized_rows.append(row)
    return normalized_rows


def shard_rows(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise RuntimeError(f"num_shards must be > 0, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise RuntimeError(f"shard_index must be in [0, num_shards), got {shard_index}")
    if num_shards == 1:
        return rows

    shard = rows[shard_index : len(rows) : num_shards]
    if len(shard) == 0:
        raise RuntimeError(
            f"Empty shard after slicing: shard_index={shard_index}, "
            f"num_shards={num_shards}, total={len(rows)}"
        )
    return shard


def prepare_output_dirs(out_dir: str, mode: str) -> dict[str, str]:
    out_dir = os.path.abspath(out_dir)
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    dirs = {
        "out_dir": out_dir,
        "pred_dir": pred_dir,
        "emb_v_dir": "",
        "emb_t_dir": "",
    }
    if mode == "score_emb":
        dirs["emb_v_dir"] = os.path.join(out_dir, "embeddings", "vision")
        dirs["emb_t_dir"] = os.path.join(out_dir, "embeddings", "text")
        os.makedirs(dirs["emb_v_dir"], exist_ok=True)
        os.makedirs(dirs["emb_t_dir"], exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GARR Stage-1 inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["score", "score_emb"], default="score_emb")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dirs = prepare_output_dirs(args.out_dir, args.mode)

    model, tokenizer = get_model_tokenizer(
        args.model,
        model_type=args.model_type,
        trust_remote_code=True,
    )
    template = get_template(model.model_meta.template, tokenizer, default_system=None)
    engine = PtEngine.from_model_template(model, template, max_batch_size=args.batch_size)

    adapter_request = None
    if args.adapter:
        from swift.llm.infer.infer_engine.utils import AdapterRequest

        adapter_request = AdapterRequest("_lora", args.adapter)

    rows = normalize_rows_for_infer(load_requests(args.dataset_jsonl))
    rows = shard_rows(rows, args.num_shards, args.shard_index)
    request_config = RequestConfig(
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        stream=False,
        return_details=False,
    )

    gen_text_csv = os.path.join(out_dirs["pred_dir"], "gen_text.csv")
    batch_size = max(1, int(getattr(engine, "max_batch_size", args.batch_size)))

    with open(gen_text_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_id", "gen_text", "ground_truth"])

        for start in tqdm(range(0, len(rows), batch_size), dynamic_ncols=True):
            batch = rows[start : start + batch_size]
            batch_results = engine._infer_batch(
                batch,
                request_config=request_config,
                template=template,
                adapter_request=adapter_request,
                mode=args.mode,
                id_key="video_id",
            )

            for row in batch_results:
                video_id = str(row["video_id"])
                writer.writerow([
                    video_id,
                    str(row["gen_text"]),
                    str(row.get("ground_truth", "") or ""),
                ])

                if args.mode == "score_emb":
                    vision_emb = row.get("vision_emb")
                    text_emb = row.get("text_emb")
                    if vision_emb is None or text_emb is None:
                        raise RuntimeError("mode=score_emb requires vision_emb/text_emb in results.")
                    np.save(
                        os.path.join(out_dirs["emb_v_dir"], f"{video_id}.npy"),
                        np.asarray(vision_emb, dtype=np.float32),
                    )
                    np.save(
                        os.path.join(out_dirs["emb_t_dir"], f"{video_id}.npy"),
                        np.asarray(text_emb, dtype=np.float32),
                    )

            csv_file.flush()


if __name__ == "__main__":
    main()
