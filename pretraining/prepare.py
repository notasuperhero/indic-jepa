"""Step 1: get the data.

  - Pre-training text: AI4Bharat Sangraha, verified Assamese, one parquet file (~330 MB).
  - Tokenizer: a BPE tokenizer (8,000 tokens) trained on that text.
  - Token stream: up to --max_tokens tokens packed into data/train.bin and data/val.bin (uint16).
  - Downstream test: IndicSentiment, Assamese (1,156 product reviews, Positive / Negative).

  python prepare.py                      # default: 40M training tokens
  python prepare.py --max_tokens 2000000 # quick test
"""
import argparse
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="", help="use a local parquet file instead of downloading")
ap.add_argument("--sentiment_dir", default="", help="local folder with validation/as.json and test/as.json")
ap.add_argument("--vocab", type=int, default=8000)
ap.add_argument("--max_tokens", type=int, default=40_000_000)
ap.add_argument("--tok_docs", type=int, default=200_000, help="documents used to train the tokenizer")
a = ap.parse_args()
os.makedirs("data", exist_ok=True)
SPECIALS = ["[PAD]", "[MASK]", "[SEP]", "[UNK]"]   # ids 0, 1, 2, 3


def bengali_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum("ঀ" <= c <= "৿" for c in letters) / max(1, len(letters))


# ---- 1. Sangraha text ----------------------------------------------------------------
path = a.parquet or hf_hub_download("ai4bharat/sangraha", "verified/asm/data-0.parquet", repo_type="dataset")
pf = pq.ParquetFile(path)
cols = [f.name for f in pf.schema_arrow if str(f.type) in ("string", "large_string")]
first = pf.read_row_group(0, columns=cols).slice(0, 50).to_pydict()
text_col = max(cols, key=lambda c: sum(len(x or "") for x in first[c]))
print(f"parquet: {pf.metadata.num_rows:,} documents; string columns {cols}; using '{text_col}'")
print("  example:", (first[text_col][0] or "")[:150].replace("\n", " "))


def docs():
    for batch in pf.iter_batches(columns=[text_col], batch_size=2000):
        for t in batch.column(0).to_pylist():
            if t and len(t) > 20 and bengali_ratio(t) > 0.5:
                yield t


# ---- 2. tokenizer ----------------------------------------------------------------------
if not os.path.exists("data/tokenizer.json"):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIALS, min_frequency=2)

    def sample():
        for i, d in enumerate(docs()):
            if i >= a.tok_docs:
                break
            yield d[:5000]
    tok.train_from_iterator(sample(), trainer)
    tok.save("data/tokenizer.json")
tok = Tokenizer.from_file("data/tokenizer.json")
print(f"tokenizer: {tok.get_vocab_size()} tokens")
ex = "অসমীয়া ভাষা অসমৰ ৰাজ্যিক ভাষা।"
print(f"  '{ex}' -> {tok.encode(ex).tokens}")

# ---- 3. packed token stream ------------------------------------------------------------
if not os.path.exists("data/train.bin"):
    SEP = tok.token_to_id("[SEP]")
    buf, n = [], 0
    batch = []
    for d in docs():
        batch.append(d)
        if len(batch) == 1000:
            for e in tok.encode_batch(batch):
                buf.append(np.array(e.ids + [SEP], dtype=np.uint16))
                n += len(e.ids) + 1
            batch = []
            print(f"\r  {n / 1e6:.1f}M tokens", end="", flush=True)
            if n >= a.max_tokens:
                break
    if batch and n < a.max_tokens:
        for e in tok.encode_batch(batch):
            buf.append(np.array(e.ids + [SEP], dtype=np.uint16))
    arr = np.concatenate(buf)[: a.max_tokens]
    cut = int(len(arr) * 0.99)
    arr[:cut].tofile("data/train.bin")
    arr[cut:].tofile("data/val.bin")
    print(f"\n  train {cut:,} tokens, val {len(arr) - cut:,} tokens")

# ---- 4. IndicSentiment (Assamese) ------------------------------------------------------
rows = []
for split in ("validation", "test"):
    p = (os.path.join(a.sentiment_dir, split, "as.json") if a.sentiment_dir else
         hf_hub_download("ai4bharat/IndicSentiment", f"data/{split}/as.json", repo_type="dataset"))
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text, label = (r.get("INDIC REVIEW") or "").strip(), (r.get("LABEL") or "").strip()
            if text and label in ("Positive", "Negative"):
                rows.append({"text": text, "label": int(label == "Positive")})
random.Random(0).shuffle(rows)
with open("data/sentiment.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pos = sum(r["label"] for r in rows)
print(f"IndicSentiment (as): {len(rows)} reviews, {pos} positive / {len(rows) - pos} negative")
print("PREPARE OK")
