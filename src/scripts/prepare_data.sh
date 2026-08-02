#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/src/configs/defaults.env"

DATASET="${DATASET:-MicroLens}"

if [[ "${DATASET}" != "MicroLens" && "${DATASET}" != "TopicVid_douyin" ]]; then
  echo "Set DATASET=MicroLens or DATASET=TopicVid_douyin." >&2
  exit 1
fi

RAW_DIR="${RAW_DIR:-${REPO_ROOT}/data/raw/${DATASET}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/${DATASET}}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/src/data/${DATASET}}"
PROCESSED_CSV="${DATA_ROOT}/processed/all.csv"
VISION_ROOT="${DATA_ROOT}/vision_h5_shards"
SPLIT_INDICES=(0 1 2 3 4)

for split in "${SPLIT_INDICES[@]}"; do
  if [[ ! -f "${SPLIT_ROOT}/split_${split}.csv" ]]; then
    echo "Split file not found: ${SPLIT_ROOT}/split_${split}.csv" >&2
    exit 1
  fi
done

if [[ "${DATASET}" == "MicroLens" ]]; then
  garr-preprocess-microlens \
    --raw-dir "${RAW_DIR}" \
    --output-csv "${PROCESSED_CSV}"
  ASSET_MAP_ARGS=()
else
  SELECTION_CSV="${SPLIT_ROOT}/source_ids.csv"
  garr-preprocess-topicvid \
    --raw-json "${RAW_DIR}/available_dataset_with_subtopic.json" \
    --selection-csv "${SELECTION_CSV}" \
    --output-csv "${PROCESSED_CSV}"
  ASSET_MAP_ARGS=(--asset-map "${SELECTION_CSV}" --asset-id-column url_id)
fi

if [[ -d "${VISION_ROOT}" ]]; then
  echo "Reusing visual inputs: ${VISION_ROOT}"
else
  garr-build-vision \
    --processed-csv "${PROCESSED_CSV}" \
    "${ASSET_MAP_ARGS[@]}" \
    --covers-dir "${RAW_DIR}/covers" \
    --videos-dir "${RAW_DIR}/videos" \
    --output-dir "${VISION_ROOT}" \
    --shards "${H5_SHARDS}" \
    --frames "${NUM_FRAMES}"
fi

for split in "${SPLIT_INDICES[@]}"; do
  garr-prepare \
    --dataset "${DATASET}" \
    --processed-csv "${PROCESSED_CSV}" \
    --split-csv "${SPLIT_ROOT}/split_${split}.csv" \
    --h5-root "${VISION_ROOT}" \
    --output-dir "${DATA_ROOT}/processed/mllm/split_${split}" \
    --h5-shards "${H5_SHARDS}" \
    --num-frames "${NUM_FRAMES}"
done
