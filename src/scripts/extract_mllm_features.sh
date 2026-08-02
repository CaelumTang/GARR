#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/src/configs/defaults.env"

DATASET="${DATASET:-MicroLens}"
SPLIT="${SPLIT:-0}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/${DATASET}}"
MODEL_TYPE="${MODEL_TYPE:-internvl3}"

if [[ "${DATASET}" != "MicroLens" && "${DATASET}" != "TopicVid_douyin" ]]; then
  echo "Set DATASET=MicroLens or DATASET=TopicVid_douyin." >&2
  exit 1
fi

if [[ "${DATASET}" == "MicroLens" ]]; then
  DEFAULT_CHECKPOINT="${MICROLENS_MLLM_CHECKPOINT}"
else
  DEFAULT_CHECKPOINT="${TOPICVID_MLLM_CHECKPOINT}"
fi
ADAPTER="${ADAPTER:-${REPO_ROOT}/outputs/mllm/${DATASET}/split_${SPLIT}/${DEFAULT_CHECKPOINT}}"

adapter_name="$(basename "${ADAPTER}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/features/${DATASET}/split_${SPLIT}/${adapter_name}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_HF="${USE_HF:-0}"
export INPUT_SIZE="${INPUT_SIZE:-448}"
export MAX_NUM="${MAX_NUM:-1}"
export GARR_DATASET_NAME="${DATASET}"

for split_name in train val test; do
  jsonl_path="${DATA_ROOT}/processed/mllm/split_${SPLIT}/${split_name}.jsonl"
  split_output="${OUTPUT_ROOT}/work/${split_name}"
  garr-infer-mllm \
    --model "${BASE_MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --adapter "${ADAPTER}" \
    --dataset-jsonl "${jsonl_path}" \
    --dataset-name "${DATASET}" \
    --output-dir "${split_output}" \
    --mode score_emb \
    --batch-size "${INFER_BATCH_SIZE:-4}" \
    --max-tokens 4 \
    --temperature 0

  garr-postprocess-mllm --csv "${split_output}/predictions/gen_text.csv"
  garr-pack-mllm \
    --input-dir "${split_output}" \
    --split "${split_name}" \
    --ground-truth-csv "${DATA_ROOT}/processed/all.csv" \
    --output "${OUTPUT_ROOT}/${split_name}.npz" \
    --cleanup
done

rmdir "${OUTPUT_ROOT}/work"
echo "Generative Alignment features: ${OUTPUT_ROOT}"
