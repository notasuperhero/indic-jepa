#!/bin/bash
# Follow-up to the pilot: try the settings the LLM-JEPA paper tunes (predictor tokens, lambda)
# plus two fixes for the "all embeddings already ~0.99 similar" problem (centered, contrastive).
# Assamese, 1,000 pairs, seed 1. Each run ~1 h (training + scoring 300 sentences); 4 runs ~4.5 h.
#   caffeinate -i bash run_sweep.sh
#   SEEDS="2" CONFIGS="centered:0:1" caffeinate -i bash run_sweep.sh     # re-run the winner with seed 2
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL=${MODEL:-google/gemma-2-2b-it}
LANG_CODE=${LANG_CODE:-asm_Beng}
SHORT=${LANG_CODE%%_*}
TEST_N=${TEST_N:-300}
N=${N:-1000}
SEEDS=${SEEDS:-1}
# variant:pred_k:lambda
CONFIGS=${CONFIGS:-"plain:1:1 plain:0:4 centered:0:1 contrastive:0:0.5"}
mkdir -p logs runs
for SEED in $SEEDS; do
  for C in $CONFIGS; do
    IFS=: read V K L <<< "$C"
    NAME=${SHORT}_${N}_jepa-${V}-k${K}-l${L}_s${SEED}
    if grep -q "^${NAME}," results.csv 2>/dev/null; then echo "skip $NAME (done)"; continue; fi
    echo "=== $NAME ==="
    python train.py --model $MODEL --lang $LANG_CODE --mode jepa --variant $V --pred_k $K --lbd $L \
      --seed $SEED --train data/${LANG_CODE}_train_${N}.jsonl --out runs/$NAME 2>&1 | tee logs/$NAME.train.log
    python evaluate.py --model $MODEL --lang $LANG_CODE --adapter runs/$NAME/adapter \
      --name $NAME --test_n $TEST_N 2>&1 | tee logs/$NAME.eval.log
  done
done
python summarize.py | tee logs/summary.txt
echo "SWEEP DONE"
