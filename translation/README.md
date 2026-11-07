# Experiment 1: LLM-JEPA for low-resource translation (English → Assamese)

**Question:** does adding the LLM-JEPA loss (Huang, LeCun & Balestriero, ICLR 2026) help a small LLM
learn to translate from very little data?

**Setup**
- Model: `google/gemma-2-2b-it`, LoRA r = 16 on all attention/MLP projections (20.8M trainable, 0.79%), bfloat16 on Apple MPS
- Training: 1,000 English–Assamese pairs from AI4Bharat **BPCC** (`daily`, `wiki`, `ilci`), 4 epochs, batch 8, lr 2e-4
- Test: first 300 sentences of **IN22-Gen** (expert-translated), greedy decoding, **chrF++**
- JEPA term: view A = the English prompt, view B = the Assamese sentence; embedding = last-layer hidden state of the last token; `total = LM loss + λ · JEPA loss`

## Results (chrF++, higher is better)

| Method | Seed 1 | Seed 2 | Mean |
|---|---|---|---|
| Zero-shot (no fine-tuning) | 13.50 | – | 13.50 |
| Fine-tuning (SFT) | 22.58 | 22.67 | **22.62** |
| SFT + LLM-JEPA (plain, λ = 1) | 23.30 | 22.78 | 23.04 |
| + 1 predictor token (λ = 1) | 22.50 | – | 22.50 |
| plain, λ = 4 | 22.57 | – | 22.57 |
| centered cosine (λ = 1) | 21.39 | – | 21.39 |
| contrastive InfoNCE (λ = 0.5, τ = 0.1) | 21.81 | – | 21.81 |

**Paired bootstrap** (1,000 resamples of the 300 test sentences):

| Comparison | Δ chrF++ | 95% interval | Better in |
|---|---|---|---|
| JEPA s1 vs SFT s1 | +0.71 | −0.02 to +1.47 | 97% |
| JEPA s1 vs SFT s2 | +0.62 | −0.14 to +1.36 | 94% |
| SFT s2 vs SFT s1 (control) | +0.09 | −0.79 to +1.01 | 57% |

JEPA seed 2 (+0.1–0.2 over SFT) erased most of seed 1's edge, so the mean +0.4 is not a reliable effect.

## Why (from the training logs)

| Variant | cos (right pair) | cos (wrong pair) | In-batch retrieval | chrF++ |
|---|---|---|---|---|
| plain, end of training | 0.995 | 0.990 | 0.88–1.0 | 23.04 |
| + predictor token | 0.999 | 0.999 | ~0.5 | 22.50 |
| centered | 0.99 | 0.36 | 1.0 | 21.39 |
| contrastive | 0.82 | 0.11 | 1.0 | 21.81 |

- **Plain JEPA barely trains:** Gemma's last-token embeddings start ~0.79 similar for *any* pair and reach 0.99 within ~100 steps, so the JEPA loss is ~0.005 and gives almost no gradient.
- **When the alignment works, translation gets worse.** The aligned vector is the model's state at the end of "…Assamese:", whose job is to plan the next token; forcing it to equal the target sentence's end state competes with generation (centered run: final LM loss 0.24 vs 0.13 for SFT).

## Engineering notes
- float32 on a 48 GB Mac swapped (memory 36 → 58 GB in 10 steps, 5 s → 218 s per step); bfloat16 ran at 5.3 s/step with identical losses (1.471 vs 1.470 at step 10).
- View A's embedding is read from the LM forward pass (identical to a separate pass, max diff 4e-7), saving one forward per step.

## Run
```
bash setup.sh                         # .venv with torch, transformers, peft, sacrebleu
hf auth login                         # accept terms for ai4bharat/BPCC, ai4bharat/IN22-Gen, google/gemma-2-2b-it
bash smoke_test.sh                    # ~10 min check
caffeinate -i bash run_pilot.sh       # zero-shot, SFT and JEPA, 2 seeds (~3.5 h)
caffeinate -i bash run_sweep.sh       # predictor tokens, λ, centered, contrastive (~4.5 h)
python summarize.py
```
`outputs/` holds every model translation (with IN22-Gen references, CC BY 4.0); `logs/` holds the training logs.
