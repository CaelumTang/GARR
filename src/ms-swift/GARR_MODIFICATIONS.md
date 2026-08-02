# GARR modifications to ms-swift

This directory contains the installable ms-swift runtime source derived from
upstream commit `6d0bcfba8ce7a1dedfd20e1ac8fc887bb16619a0`.

GARR changes the following upstream files:

| File | Modification |
| --- | --- |
| `swift/llm/template/vision_utils.py` | Decode images addressed by HDF5 URLs. |
| `swift/llm/template/base.py` | Preserve data required by GARR templates. |
| `swift/plugin/loss.py` | Register the Generative Alignment objective, FIFO contrastive queue. |
| `swift/trainers/mixin.py` | Expose hidden states and GARR training metrics. |
| `swift/trainers/trainers.py` | Connect the custom objective to trainer outputs. |
| `swift/llm/infer/infer_engine/pt_engine.py` | Export batched generated scores and popularity-adapted visual/text representations. |

All other files in this directory are unmodified runtime or packaging files
from that pinned upstream revision. Upstream copyright and Apache-2.0 license
terms remain in effect; see `LICENSE` and the repository-level
`THIRD_PARTY_NOTICE.txt`.
