#!/usr/bin/env python3
"""Export MLLM-generated scores and popularity-adapted representations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any

import numpy as np
from swift.llm import get_model_tokenizer, get_template
from swift.llm.infer.infer_engine.pt_engine import PtEngine
from swift.llm.infer.protocol import RequestConfig
from swift.utils import read_from_jsonl
from tqdm import tqdm


def _load_requests(jsonl_path: str) -> list[dict[str, Any]]:
    rows = read_from_jsonl(jsonl_path)
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"Empty or invalid jsonl: {jsonl_path}")
    return rows


def _extract_video_id_from_images(images: list[Any]) -> int:
    if not isinstance(images, list) or len(images) == 0:
        raise RuntimeError("Missing images list; cannot infer video_id.")

    image_url = str(images[0])
    match = re.search(r"/covers/(\d+)", image_url)
    if match is None:
        raise RuntimeError(f"Failed to parse video_id from images[0]={image_url!r}")
    return int(match.group(1))


def _normalize_rows_for_infer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
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
            if "ground_truth" not in objects:
                objects["ground_truth"] = "" if ground_truth is None else str(ground_truth)
            row["messages"] = messages[:-1]

        if "video_id" not in objects:
            objects["video_id"] = _extract_video_id_from_images(row.get("images", []))
        requests.append(row)
    return requests


def _ensure_output_dirs(output_dir: str, mode: str) -> dict[str, str]:
    output_dir = os.path.abspath(output_dir)
    predictions_dir = os.path.join(output_dir, "predictions")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    vision_dir = ""
    text_dir = ""
    if mode == "score_emb":
        embeddings_dir = os.path.join(output_dir, "embeddings")
        vision_dir = os.path.join(embeddings_dir, "vision")
        text_dir = os.path.join(embeddings_dir, "text")
        os.makedirs(vision_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
    return {
        "output_dir": output_dir,
        "predictions_dir": predictions_dir,
        "metadata_dir": metadata_dir,
        "vision_dir": vision_dir,
        "text_dir": text_dir,
    }


def _write_run_config(
    metadata_dir: str,
    args,
    request_config: RequestConfig,
    engine: PtEngine,
) -> None:
    path = os.path.join(metadata_dir, "run_config.json")
    prompt_markers = {
        "title": PtEngine._GARR_TITLE_MARKER,
        "predict": PtEngine._GARR_PRED_MARKER,
        "img_context_token": "<IMG_CONTEXT>",
    }
    if args.dataset_name == "TopicVid_douyin":
        prompt_markers["topic"] = PtEngine._GARR_TOPIC_MARKER
        prompt_markers["description"] = PtEngine._GARR_DESC_MARKER
    else:
        prompt_markers["category"] = PtEngine._GARR_CAT_MARKER

    config = {
        "engine": "PtEngine",
        "mode": args.mode,
        "video_id_key": "video_id",
        "model": args.model,
        "adapter": args.adapter,
        "model_type": args.model_type,
        "dataset_name": args.dataset_name,
        "batch_size": int(getattr(engine, "max_batch_size", args.batch_size)),
        "prompt_markers": prompt_markers,
        "request_config": {
            "max_tokens": request_config.max_tokens,
            "temperature": request_config.temperature,
            "num_beams": request_config.num_beams,
            "stop": request_config.stop,
            "seed": request_config.seed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, required=True, help="Base model path (e.g., InternVL3-2B)."
    )
    parser.add_argument("--model-type", type=str, default="internvl3")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path.")
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        required=True,
        help="JSONL dataset in InferRequest format.",
    )
    parser.add_argument(
        "--dataset-name",
        choices=("MicroLens", "TopicVid_douyin"),
        required=True,
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["score", "score_emb"], default="score")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    output_dirs = _ensure_output_dirs(args.output_dir, args.mode)
    gen_text_csv = os.path.join(output_dirs["predictions_dir"], "gen_text.csv")

    model, tokenizer = get_model_tokenizer(
        args.model,
        model_type=args.model_type,
        trust_remote_code=True,
    )
    template = get_template(model.model_meta.template, tokenizer, default_system=None)

    engine = PtEngine.from_model_template(model, template, max_batch_size=args.batch_size)
    from swift.llm.infer.infer_engine.utils import AdapterRequest

    adapter_request = AdapterRequest("_lora", args.adapter)

    requests = _normalize_rows_for_infer(_load_requests(args.dataset_jsonl))
    request_config = RequestConfig(
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        stream=False,
        return_details=False,
    )

    _write_run_config(output_dirs["metadata_dir"], args, request_config, engine)

    with open(gen_text_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "gen_text", "ground_truth"])
        f.flush()

        bs = int(engine.max_batch_size)
        progress = tqdm(
            total=len(requests),
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        start = 0
        while start < len(requests):
            batch = requests[start : start + bs]
            batch_results = engine.infer_garr_batch(
                batch,
                request_config=request_config,
                template=template,
                adapter_request=adapter_request,
                mode=args.mode,
                id_key="video_id",
            )
            expected_ids = [str(request["objects"]["video_id"]) for request in batch]
            returned_ids = [str(row["video_id"]) for row in batch_results]
            if returned_ids != expected_ids:
                raise RuntimeError(
                    f"Inference result IDs do not match requests: "
                    f"expected={expected_ids}, returned={returned_ids}"
                )
            for row in batch_results:
                video_id = str(row["video_id"])
                gen_text = str(row["gen_text"])
                ground_truth = str(row.get("ground_truth", "") or "")
                writer.writerow([video_id, gen_text, ground_truth])

                if args.mode == "score_emb":
                    vision_embedding = row.get("vision_emb")
                    text_embedding = row.get("text_emb")
                    if vision_embedding is None or text_embedding is None:
                        raise RuntimeError(
                            "mode=score_emb requires vision_emb/text_emb in results."
                        )
                    np.save(
                        os.path.join(output_dirs["vision_dir"], f"{video_id}.npy"),
                        np.asarray(vision_embedding, dtype=np.float32),
                    )
                    np.save(
                        os.path.join(output_dirs["text_dir"], f"{video_id}.npy"),
                        np.asarray(text_embedding, dtype=np.float32),
                    )

            f.flush()
            start += len(batch)
            progress.update(len(batch))
        progress.close()


if __name__ == "__main__":
    main()
