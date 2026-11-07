#!/bin/bash
# Pilot: Assamese, SFT vs LLM-JEPA, 2 seeds (+ zero-shot). Default 1k pairs (~3 h); SIZES="1000 5000" adds 5k (~11 h more).
# Evaluates on the first 300 IN22-Gen sentences to keep it fast.
#   caffeinate -i bash run_pilot.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash run_pilot.sh     # if you don't have Gemma access
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL=${MODEL:-google/gemma-2-2b-it}
LANG_CODE=${LANG_CODE:-asm_Beng}
SHORT=${LANG_CODE%%_*}
TEST_N=${TEST_N:-300}
SIZES=${SIZES:-1000}
mkdir -p logs runs

[ -f data/${LANG_CODE}_test.jsonl ] || python prepare_data.py --lang $LANG_CODE --sizes 1000 5000 20000 2>&1 | tee logs/prepare.log

grep -q "^${SHORT}_zeroshot," results.csv 2>/dev/null || \
  python evaluate.py --model $MODEL --lang $LANG_CODE --name ${SHORT}_zeroshot --test_n $TEST_N 2>&1 | tee logs/${SHORT}_zeroshot.log

for N in $SIZES; do
  for SEED in 1 2; do
    for MODE in sft jepa; do
      NAME=${SHORT}_${N}_${MODE}_s${SEED}
      if grep -q "^${NAME}," results.csv 2>/dev/null; then echo "skip $NAME (done)"; continue; fi
      python train.py --model $MODEL --lang $LANG_CODE --mode $MODE --seed $SEED \
        --train data/${LANG_CODE}_train_${N}.jsonl --out runs/$NAME 2>&1 | tee logs/$NAME.train.log
      python evaluate.py --model $MODEL --lang $LANG_CODE --adapter runs/$NAME/adapter \
        --name $NAME --test_n $TEST_N 2>&1 | tee logs/$NAME.eval.log
    done
  done
done
python summarize.py | tee logs/summary.txt
echo "PILOT DONE"
