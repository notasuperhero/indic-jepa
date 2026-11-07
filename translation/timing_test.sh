#!/bin/bash
# ~10 minutes: how fast is a JEPA step on this Mac, in float32 vs bfloat16?
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL=${MODEL:-google/gemma-2-2b-it}
mkdir -p logs
for DT in float32 bfloat16; do
  echo "=== jepa, $DT ==="
  python train.py --model $MODEL --mode jepa --train data/asm_Beng_train_1000.jsonl \
    --out runs/timing_$DT --max_steps 10 --dtype $DT 2>&1 | grep -E "step|done|Error" | tee logs/timing_$DT.log
done
