<div align="center">
<h2>

GARR
</h2>

  <a href="https://github.com/CaelumTang/GARR">
    <img
      src="https://img.shields.io/badge/GARR-Code-181717?logo=github&logoColor=white"
      alt="GARR source code"
    />
  </a>
  <a href="LICENSE">
    <img
      src="https://img.shields.io/badge/License-Apache%202.0-blue"
      alt="Apache License 2.0"
    />
  </a>
  <a href="requirements.txt">
    <img
      src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"
      alt="Python 3.10"
    />
  </a>

</div>

A reference implementation for *GARR: Micro-Video Popularity Prediction with
MLLMs via Generative Alignment and Retrieval Refinement*.

## 🔧 Dependencies and Installation

```bash
git clone https://github.com/CaelumTang/GARR.git
cd GARR
conda create -n GARR python=3.10 -y
conda activate GARR
bash setup.sh
```

## 📖 Dataset Preparation

### MicroLens

Download the MicroLens-100k metadata, covers, and videos from the
[MicroLens website](https://recsys.westlake.edu.cn/), then arrange them as:

```text
data/raw/MicroLens/
├── MicroLens-100k_pairs.csv
├── MicroLens-100k_title_en.csv
├── tags_to_summary.csv
├── MicroLens-100k_likes_and_views.txt
├── covers/**/<video_id>.jpg
└── videos/**/<video_id>.mp4
```

Prepare all five fixed MicroLens splits:

```bash
DATASET=MicroLens bash src/scripts/prepare_data.sh
```

### TopicVid_douyin

Download `available_dataset_with_subtopic.json` from
[TopicVid](https://huggingface.co/datasets/chensh911/TopicVid). Obtain the
selected source videos and covers using the URL IDs listed in
`src/data/TopicVid_douyin/source_ids.csv`, then arrange them as:

```text
data/raw/TopicVid_douyin/
├── available_dataset_with_subtopic.json
├── covers/<url_id>.jpg
└── videos/<url_id>.mp4
```

Prepare all five fixed TopicVid_douyin splits:

```bash
DATASET=TopicVid_douyin bash src/scripts/prepare_data.sh
```

## 🚀 Training and Evaluation

### MLLM: Generative Alignment

```bash
DATASET=MicroLens bash src/scripts/train_mllm.sh
```

### Feature Extraction

```bash
DATASET=MicroLens bash src/scripts/extract_mllm_features.sh
```

### Retrieval and Predictor

```bash
DATASET=MicroLens bash src/scripts/run_downstream.sh
```

For TopicVid_douyin, replace `DATASET=MicroLens` with
`DATASET=TopicVid_douyin` in the commands above.

## Acknowledgement

We appreciate the open-source contributions of
[ms-swift](https://github.com/modelscope/ms-swift),
[InternVL3](https://github.com/OpenGVLab/InternVL), and
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).
