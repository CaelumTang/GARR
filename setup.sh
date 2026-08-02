#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e "${REPO_ROOT}/src" --no-deps
python -m pip install -e "${REPO_ROOT}/src/ms-swift"
