# GARR: Micro-Video Popularity Prediction with MLLMs via Generative Alignment and Retrieval Refinement

## Pipeline Overview

GARR mainly consists of three stages:

1. **Generative Alignment**: reformulate popularity regression as next-token prediction with an MLLM backbone.
2. **Task-aware Retrieval**: retrieve historical samples with consistent popularity trends based on aligned multimodal representations.
3. **Retrieval Refinement**: combine the generative prior and retrieved contextual information for final prediction.

This repository will provide the code and resources needed to reproduce the experiments and results reported in our paper.

## To Do List

- [ ] Release training code
- [ ] Release inference code and weights
- [ ] Release dataset preparation code

## Acknowledgement

We appreciate the releasing codes of [ms-swift](https://github.com/modelscope/ms-swift).
