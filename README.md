# GARR: Micro-Video Popularity Prediction with MLLMs via Generative Alignment and Retrieval Refinement

## Pipeline Overview

GARR mainly consists of three stages:

1. **Generative Alignment**: reformulate popularity regression as next-token prediction with an MLLM backbone.
2. **Task-aware Retrieval**: retrieve historical samples with consistent popularity trends based on aligned multimodal representations.
3. **Retrieval Refinement**: combine the generated prior and retrieved contextual information for final prediction.

This repository provides the inference pipeline for reproducing the results reported in our paper.

## Dependencies and Installation

```bash
pip install -r requirements.txt
```

## Quick Inference

### 1. Stage-1 Inference

```bash
python scripts/infer_score_and_emb.py \
  --model models/OpenGVLab/InternVL3-2B \
  --adapter artifacts/stage1/MicroLens/split_0/ \
  --dataset_jsonl datasets/MicroLens/processed/split_0/test.jsonl \
  --out_dir artifacts/stage1/MicroLens/split_0/output/
```

### 2. Pack Stage-1 Outputs

```bash
python scripts/pack_embeddings.py \
  --final_dir artifacts/stage1/MicroLens/split_0/output/ \
  --all_csv datasets/MicroLens/processed/all.csv
```

### 3. Stage-2 Retrieval

```bash
python scripts/retrieve.py \
  --checkpoint_dir artifacts/stage1/MicroLens/split_0/
```

### 4. Stage-3 Prediction

```bash
python scripts/predict.py \
  --checkpoint_dir artifacts/stage1/MicroLens/split_0/ \
  --stage2_dir artifacts/stage2/MicroLens/split_0/ \
  --k 20
```

## To Do List

- [ ] Release dataset preparation code
- [x] Release inference code
- [ ] Release weights
- [ ] Release training code

## Acknowledgement

We appreciate the open-source contribution of [ms-swift](https://github.com/modelscope/ms-swift).
