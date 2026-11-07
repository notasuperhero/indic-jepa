"""Step 1: build training and test sets.

Training pool: human-translated English->target pairs from AI4Bharat BPCC
(subsets: daily, wiki, ilci). Gated: log in (`hf auth login`) and accept the
terms on https://huggingface.co/datasets/ai4bharat/BPCC first.
Test set: IN22-Gen (1024 expert-translated sentences, CC BY 4.0).

Writes data/<lang>_train_<N>.jsonl for each size (nested: 1k is inside 5k is
inside 20k) and data/<lang>_test.jsonl. Any training pair whose English or
target sentence appears in the test set is removed.

  python prepare_data.py --lang asm_Beng --sizes 1000 5000 20000
"""
import argparse
import json
import os
import random

import pandas as pd
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--lang", default="asm_Beng")
ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
ap.add_argument("--subsets", nargs="+", default=["daily", "wiki", "ilci"])
ap.add_argument("--min_words", type=int, default=3)
ap.add_argument("--max_words", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs("data", exist_ok=True)


def latin_letter_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


def indic_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(not c.isascii() for c in letters) / max(1, len(letters))


def read_bpcc(path):
    """BPCC .tsv: header row src_lang, tgt_lang, src, tgt. Uses the named columns and makes
    sure English ends up first (swaps if src_lang is not eng_Latn). Falls back to guessing
    columns (English = most Latin letters among multi-word columns) if there is no header."""
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return []
    print(f"    first raw row ({len(rows[0])} columns): {rows[0]}")
    head = [h.strip().lower() for h in rows[0]]
    if "src" in head and "tgt" in head:
        si, ti = head.index("src"), head.index("tgt")
        li = head.index("src_lang") if "src_lang" in head else None
        body = rows[1:]
        if body:
            print(f"    first data row: {body[0]}")
        out = []
        for r in body:
            if len(r) <= max(si, ti):
                continue
            s, t = r[si].strip(), r[ti].strip()
            if li is not None and len(r) > li and not r[li].strip().lower().startswith("eng"):
                s, t = t, s          # this row is target -> English; flip it
            out.append((s, t))
        print(f"    using header columns: English = 'src' ({si}), target = 'tgt' ({ti})")
        return out
    sample = rows[: min(300, len(rows))]
    ncol = len(rows[0])
    def score(c, fn):
        vals = [r[c] for r in sample if len(r) > c and any(ch.isalpha() for ch in r[c])]
        multi = sum(len(v.split()) >= 2 for v in vals) / max(1, len(vals))
        if multi < 0.5:              # language codes / IDs, not sentences
            return 0.0
        return sum(fn(v) for v in vals) / max(1, len(sample))
    eng_col = max(range(ncol), key=lambda c: score(c, latin_letter_ratio))
    tgt_col = max((c for c in range(ncol) if c != eng_col), key=lambda c: score(c, indic_ratio))
    print(f"    English column = {eng_col}, target column = {tgt_col}")
    return [(r[eng_col].strip(), r[tgt_col].strip()) for r in rows if len(r) > max(eng_col, tgt_col)]


# ---- test set: IN22-Gen
pq = hf_hub_download("ai4bharat/IN22-Gen", "data/train-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(pq)
eng_col = next(c for c in df.columns if "eng_Latn" in c)
tgt_col = next(c for c in df.columns if a.lang in c)
test = [{"id": int(i), "src": s, "tgt": t} for i, (s, t) in enumerate(zip(df[eng_col], df[tgt_col]))]
with open(f"data/{a.lang}_test.jsonl", "w") as f:
    for r in test:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
test_src = {r["src"].strip() for r in test}
test_tgt = {r["tgt"].strip() for r in test}
print(f"IN22-Gen test: {len(test)} sentences (columns {eng_col} -> {tgt_col})")

# ---- training pool: BPCC human-translated subsets
pool, seen = [], set()
for sub in a.subsets:
    try:
        p = hf_hub_download("ai4bharat/BPCC", f"{sub}/{a.lang}.tsv", repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print(f"  skip {sub}: {e.__class__.__name__}")
        continue
    got = read_bpcc(p)
    kept, why = 0, {"length": 0, "empty": 0, "in_test": 0, "duplicate": 0}
    for s, t in got:
        n = len(s.split())
        if not s or not t:
            why["empty"] += 1
        elif not (a.min_words <= n <= a.max_words):
            why["length"] += 1
        elif s in test_src or t in test_tgt:
            why["in_test"] += 1
        elif s in seen:
            why["duplicate"] += 1
        else:
            seen.add(s)
            pool.append({"src": s, "tgt": t, "subset": sub})
            kept += 1
    print(f"  {sub}: {len(got):,} pairs read, {kept:,} kept, dropped: {why}")
print(f"training pool: {len(pool):,} pairs")
if pool:
    print("example:", pool[0])

random.Random(a.seed).shuffle(pool)
for n in sorted(a.sizes):
    if n > len(pool):
        print(f"  size {n} > pool {len(pool)}; skipped")
        continue
    with open(f"data/{a.lang}_train_{n}.jsonl", "w") as f:
        for r in pool[:n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote data/{a.lang}_train_{n}.jsonl")
