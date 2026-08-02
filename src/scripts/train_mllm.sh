#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/src/configs/defaults.env"

DATASET="${DATASET:-MicroLens}"
SPLIT="${SPLIT:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mllm/${DATASET}/split_${SPLIT}}"

if [[ "${DATASET}" != "MicroLens" && "${DATASET}" != "TopicVid_douyin" ]]; then
  echo "Set DATASET=MicroLens or DATASET=TopicVid_douyin." >&2
  exit 1
fi

TRAIN_JSONL="${DATA_ROOT}/processed/mllm/split_${SPLIT}/train.jsonl"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export USE_HF="${USE_HF:-0}"
export NPROC_PER_NODE="${#GPU_ARRAY[@]}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29503}"
export INPUT_SIZE="${INPUT_SIZE:-448}"
export MAX_NUM="${MAX_NUM:-1}"

export GARR_DATASET_NAME="${DATASET}"
export GARR_USE_CE="${GARR_USE_CE:-1}"
export GARR_USE_CON="${GARR_USE_CON:-1}"
export GARR_CON_WEIGHT="${GARR_CON_WEIGHT:-0.002}"
export GARR_CON_LEARNABLE="${GARR_CON_LEARNABLE:-1}"
export GARR_CON_T_INIT="${GARR_CON_T_INIT:-0.15}"
export GARR_CON_INV_T_MAX="${GARR_CON_INV_T_MAX:-100}"
export GARR_CON_GATHER="${GARR_CON_GATHER:-True}"
export GARR_QUEUE_SIZE="${GARR_QUEUE_SIZE:-4096}"
export GARR_CON_EPOCH_STOP="${GARR_CON_EPOCH_STOP:-1}"

swift sft \
  --model "${BASE_MODEL}" \
  --train_type lora \
  --loss_type garr_loss \
  --dataset "${TRAIN_JSONL}" \
  --split_dataset_ratio 0 \
  --output_dir "${OUTPUT_DIR}" \
  --add_version false \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --freeze_vit false \
  --freeze_aligner false \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy epoch \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --attn_impl flash_attn \
  --deepspeed zero2
