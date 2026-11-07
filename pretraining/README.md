# Experiment 2: JEPA vs BERT-style pre-training from scratch (Assamese)

**Question:** when you train a small encoder **from scratch** on a low-resource language, is it better
to predict the *representations* of hidden text (I-JEPA style) or the hidden *words* (BERT-style)?

Both methods use the same encoder (6 layers, width 256, ~7M parameters), the same 40M tokens of Assamese
(AI4Bharat **Sangraha**, own 8k BPE tokenizer), the same hidden spans (4 spans of 10–20% each),
20,000 steps × 64 × 128 tokens. Only the prediction target differs.

| | BERT-style (MLM) | JEPA |
|---|---|---|
| Input | hidden spans → `[MASK]` | hidden spans invisible to the context encoder |
| Predicts | the hidden tokens | the target encoder's vectors at the hidden positions |
| Target | real tokens | EMA copy of the encoder (momentum 0.996 → 1) |
| Extra | output head tied to embeddings | predictor (3-layer transformer, width 128); VICReg-style variance/covariance penalty |

**Evaluation:** frozen encoder → mean-pooled token vectors → logistic regression on Assamese
**IndicSentiment** (1,154 reviews). k labelled reviews (balanced, 10 draws, 400 held-out test reviews), plus 5-fold CV on all.

## Results (accuracy %, mean of draws)

| Labelled reviews | Untrained encoder | BERT-style | JEPA + penalty |
|---|---|---|---|
| 16 | 53.8 | **59.2** | 55.1 |
| 64 | 55.4 | **65.2** | 61.1 |
| 256 | 58.1 | **70.7** | 61.4 |
| all (5-fold CV) | 64.3 | **76.0** | 63.0 |

Plain JEPA (no penalty, 200-step test): 53.9 / 53.7 / 58.5 / 65.0, i.e. collapsed.

## What happened

| Run | Step | Loss | Target std | Prediction std |
|---|---|---|---|---|
| Plain JEPA | 1 | 0.53 | 0.66 | 0.71 |
| | 200 | 0.006 | 0.11 | **0.009**, collapsed |
| JEPA + penalty | 1,000 | 0.361 | 0.997 | 0.98 |
| | 20,000 | 0.425 | 1.000 | 0.98 |

- **Plain JEPA collapsed** within ~50 steps despite the EMA target and predictor: every token got the same vector, so the loss went to ~0.
- **With the penalty there is no collapse, but no learning either:** the prediction loss stopped improving after ~1k steps, and the features are no better than an untrained encoder.
- **BERT-style learned useful features** (+5 to +12 points over untrained), even though its word accuracy on long hidden spans stayed low (1.9%).

## Run
```
# uses the .venv created by ../translation/setup.sh
bash smoke_test.sh                 # downloads Sangraha (one ~330 MB file) + IndicSentiment, quick checks
caffeinate -i bash run_all.sh      # untrained probe, BERT-style (~45 min), JEPA (~55 min), probes
python pretrain.py --method jepa --out runs/plain_jepa --steps 200 --log_every 20 --warmup 20   # reproduce the collapse
```
