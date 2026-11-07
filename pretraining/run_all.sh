#!/bin/bash
# The experiment: BERT-style vs JEPA pre-training from scratch on Assamese, same model / data / steps.
#   caffeinate -i bash run_all.sh                 # seed 1
#   SEEDS="2 3" caffeinate -i bash run_all.sh     # more seeds later
set -eo pipefail
cd "$(dirname "$0")"
source ../translation/.venv/bin/activate
STEPS=${STEPS:-20000}
SEEDS=${SEEDS:-1}
mkdir -p logs runs
[ -f data/sentiment.jsonl ] || python prepare.py 2>&1 | tee logs/prepare.log
grep -q "^random_init," results.csv 2>/dev/null || python probe.py --random --name random_init 2>&1 | tee logs/random_init.log
for SEED in $SEEDS; do
  for M in mlm jepareg; do
    NAME=${M}_s${SEED}
    if grep -q "^${NAME}," results.csv 2>/dev/null; then echo "skip $NAME (done)"; continue; fi
    if [ $M = mlm ]; then ARGS="--method mlm"; else ARGS="--method jepa --reg ${REG:-1}"; fi
    python pretrain.py $ARGS --seed $SEED --steps $STEPS --out runs/$NAME 2>&1 | tee logs/$NAME.log
    python probe.py --ckpt runs/$NAME/encoder.pt --name $NAME 2>&1 | tee logs/$NAME.probe.log
  done
done
column -s, -t results.csv | tee logs/summary.txt
echo "ALL DONE"
