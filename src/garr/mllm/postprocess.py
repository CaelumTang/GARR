#!/usr/bin/env python3
"""Postprocess MLLM-generated token sequences into popularity scores."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re

_NUM_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)$")


def _parse_pre_score(gen_text: str) -> str:
    match = _NUM_RE.match(gen_text)
    if match is None:
        return ""
    value = float(match.group(1))
    return f"{value:.2f}" if math.isfinite(value) and 0.0 <= value <= 9.99 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to merged predictions/gen_text.csv"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.csv)

    with open(input_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) == 0:
        raise RuntimeError(f"Empty CSV: {input_path}")
    header = rows[0]
    if len(header) < 2:
        raise RuntimeError(f"Invalid CSV header: {header!r} in {input_path}")

    accepted_headers = (
        ["video_id", "gen_text", "ground_truth"],
        ["video_id", "gen_text", "ground_truth", "pre_score"],
    )
    if header not in accepted_headers:
        raise RuntimeError(f"Unexpected header in {input_path}: {header!r}")

    invalid = 0
    out_rows = [["video_id", "gen_text", "ground_truth", "pre_score"]]

    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3:
            raise RuntimeError(f"Invalid row at line {line_number} in {input_path}: {row!r}")
        video_id = str(row[0])
        gen_text = str(row[1])
        ground_truth = str(row[2])
        pre_score = _parse_pre_score(gen_text)
        invalid += int(not pre_score)

        out_rows.append([video_id, gen_text, ground_truth, pre_score])

    temporary_path = input_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(out_rows)
    os.replace(temporary_path, input_path)

    print(f"[GARR][postprocess] csv={input_path} rows={len(out_rows) - 1} invalid={invalid}")


if __name__ == "__main__":
    main()
