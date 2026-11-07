#!/bin/bash
# One-time setup (run in Terminal from this folder):  bash setup.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch transformers peft accelerate datasets huggingface_hub sacrebleu pandas pyarrow
python -c "import torch; print('torch', torch.__version__, 'MPS available:', torch.backends.mps.is_available())"
echo "SETUP OK. Next: hf auth login (if not already), accept the BPCC and Gemma-2 terms on huggingface.co, then: bash smoke_test.sh"
