#!/bin/bash
# 10-minute check that everything works on your Mac before the real pilot.
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL=${MODEL:-google/gemma-2-2b-it}
mkdir -p logs
python prepare_data.py --lang asm_Beng --sizes 1000 5000 20000 2>&1 | tee logs/prepare.log
[ -f data/asm_Beng_train_1000.jsonl ] || { echo "STOP: no training data was written - send Claude the lines above"; exit 1; }
for MODE in sft jepa; do
  python train.py --model $MODEL --mode $MODE --train data/asm_Beng_train_1000.jsonl \
    --out runs/smoke_$MODE --max_steps 10 2>&1 | tee logs/smoke_$MODE.log
done
python evaluate.py --model $MODEL --adapter runs/smoke_jepa/adapter --name smoke --test_n 16 2>&1 | tee logs/smoke_eval.log
sed -i '' '/^smoke,/d' results.csv 2>/dev/null || true
echo "SMOKE TEST OK - now run: caffeinate -i bash run_pilot.sh"
