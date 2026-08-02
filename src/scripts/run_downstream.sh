#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/src/configs/defaults.env"

DATASET="${DATASET:-MicroLens}"
SPLIT="${SPLIT:-0}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/${DATASET}/split_${SPLIT}}"

if [[ "${DATASET}" != "MicroLens" && "${DATASET}" != "TopicVid_douyin" ]]; then
  echo "Set DATASET=MicroLens or DATASET=TopicVid_douyin." >&2
  exit 1
fi

if [[ "${DATASET}" == "MicroLens" ]]; then
  DEFAULT_CHECKPOINT="${MICROLENS_MLLM_CHECKPOINT}"
else
  DEFAULT_CHECKPOINT="${TOPICVID_MLLM_CHECKPOINT}"
fi
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/outputs/features/${DATASET}/split_${SPLIT}/${DEFAULT_CHECKPOINT}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

garr-retrieve \
  --features-dir "${FEATURES_DIR}" \
  --output-dir "${OUTPUT_ROOT}/retrieval" \
  --k-max "${K}" \
  --k-rho "${K}" \
  --rho-step "${RHO_STEP:-0.001}" \
  --batch-size "${RETRIEVAL_BATCH_SIZE:-2048}" \
  --device cuda

garr-train-predictor \
  --features-dir "${FEATURES_DIR}" \
  --retrieval-dir "${OUTPUT_ROOT}/retrieval" \
  --output-dir "${OUTPUT_ROOT}/predictor" \
  --k "${K}" \
  --heads "${HEADS}" \
  --hidden "${HIDDEN}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --device cuda

echo "Retrieval and Retrieval Refinement outputs: ${OUTPUT_ROOT}"
