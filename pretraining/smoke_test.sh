#!/bin/bash
# ~10-15 min: download data, then 200 steps of each method + probe, to check speed and that it runs.
set -eo pipefail
cd "$(dirname "$0")"
source ../translation/.venv/bin/activate          # reuses the environment from the translation experiment
mkdir -p logs
[ -f data/sentiment.jsonl ] || python prepare.py 2>&1 | tee logs/prepare.log
# 600 steps of JEPA with the anti-collapse penalty (plain JEPA collapsed in the first smoke test)
python pretrain.py --method jepa --reg 1 --out runs/smoke_jepareg --steps 600 --log_every 50 --warmup 60 2>&1 | tee logs/smoke_jepareg.log
python probe.py --ckpt runs/smoke_jepareg/encoder.pt --name smoke_jepareg --draws 3 2>&1 | tee logs/smoke_probe.log
python probe.py --random --name smoke_random --draws 3 2>&1 | tee -a logs/smoke_probe.log
sed -i '' '/^smoke_/d' results.csv 2>/dev/null || true
echo "SMOKE TEST OK"
