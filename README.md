# Does JEPA help low-resource Indian-language NLP?

Two controlled experiments on **Assamese**, run end to end on a single 48 GB MacBook Pro.
**Both results are negative**, and the logs show *why*.

| # | Question | Result |
|---|---|---|
| 1 | [`translation/`](translation): does adding the **LLM-JEPA** loss (Huang, LeCun & Balestriero, ICLR 2026) help a small LLM learn English→Assamese translation from only 1,000 sentence pairs? | **No.** Best case +0.4 chrF++ (noise). The variants that really aligned the embeddings **hurt** (−0.8 to −1.2 chrF++). |
| 2 | [`pretraining/`](pretraining): when training a small encoder **from scratch** on Assamese, is predicting hidden *representations* (I-JEPA style) better than predicting hidden *words* (BERT-style)? | **No.** BERT-style: 76.0% sentiment accuracy; JEPA: 63.0%, no better than an untrained encoder (64.3%). |

## Key findings
1. **The plain LLM-JEPA loss gives almost no signal on a decoder LLM.** Gemma-2's last-token embeddings are already ~0.99 cosine-similar for *any* two sentences, so `1 − cos` drops to ~0.005 within 100 steps without learning to match translations (a translation is barely closer than a random sentence: 0.995 vs 0.990).
2. **Fixing that makes translation worse.** Centering the embeddings or using a contrastive (InfoNCE) loss does make each English sentence clearly closest to its own translation (cos 0.82 vs 0.11 for a wrong pair; in-batch retrieval 100%), but chrF++ falls below plain fine-tuning: the alignment objective competes with generation.
3. **Plain I-JEPA on text collapses at small scale.** Despite the EMA target encoder and predictor, every token mapped to the same vector within ~50 steps (prediction std 0.97 → 0.009).
4. **An anti-collapse penalty prevents collapse but not uselessness.** With a VICReg-style variance/covariance term the vectors stay spread out (std ≈ 1.0), but the prediction loss stops improving after ~1k steps and the features are no better than random for sentiment.

## Hardware and data
- One 48 GB MacBook Pro (Apple GPU via PyTorch MPS); no cloud GPUs.
- Public data only: AI4Bharat **BPCC** (training pairs), **IN22-Gen** (MT test, CC BY 4.0), **Sangraha** (Assamese pre-training text), **IndicSentiment** (classification test). Data is downloaded by the scripts, not stored here.

## Limitations
- One language (Assamese), small models (Gemma-2-2B with LoRA; a 7M-parameter encoder), 1–2 seeds per setting.
- chrF++ only for MT (no COMET / human evaluation); sentiment probe only for the encoders.
- JEPA methods are known to need large scale and careful tuning; these results say "not at laptop scale", not "never".
