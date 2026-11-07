"""Step 3: translate the IN22-Gen test sentences and score them with chrF++.

  python evaluate.py --adapter runs/asm_1000_jepa_s1/adapter --name asm_1000_jepa_s1
  python evaluate.py --name asm_zeroshot              # base model, no fine-tuning

Appends one line to results.csv: name, lang, n_test, chrF++.
"""
import argparse
import csv
import json
import os
import time

import sacrebleu
import torch

from common import LANG_NAMES, load_model, pick_device, prompt_text, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-2-2b-it")
ap.add_argument("--lang", default="asm_Beng")
ap.add_argument("--adapter", default="")
ap.add_argument("--name", required=True)
ap.add_argument("--test_n", type=int, default=1024, help="first N test sentences (300 for quick pilots)")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_new", type=int, default=160)
ap.add_argument("--device", default="")
ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
a = ap.parse_args()

dev = pick_device(a.device)
lang_name = LANG_NAMES.get(a.lang, a.lang)
tok, model = load_model(a.model, dev, a.dtype)
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).to(dev)
model.eval()
tok.padding_side = "left"   # left padding for batched generation

test = read_jsonl(f"data/{a.lang}_test.jsonl")[: a.test_n]
hyps, t0 = [], time.time()
for i in range(0, len(test), a.bs):
    rows = test[i:i + a.bs]
    enc = tok([prompt_text(r["src"], lang_name) for r in rows], return_tensors="pt", padding=True)
    enc = {k: enc[k].to(dev) for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    for seq in out[:, enc["input_ids"].shape[1]:]:
        text = tok.decode(seq, skip_special_tokens=True)
        hyps.append(text.strip().split("\n")[0].strip())   # keep the first line only
    print(f"  {len(hyps)}/{len(test)}  ({time.time() - t0:.0f}s)", flush=True)

refs = [r["tgt"] for r in test]
chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{a.name}.jsonl", "w") as f:
    for r, h in zip(test, hyps):
        f.write(json.dumps({"src": r["src"], "ref": r["tgt"], "hyp": h}, ensure_ascii=False) + "\n")
new = not os.path.exists("results.csv")
with open("results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["name", "lang", "n_test", "chrf++"])
    w.writerow([a.name, a.lang, len(test), round(chrf, 2)])
print(f"{a.name}: chrF++ = {chrf:.2f} on {len(test)} sentences")
for r, h in list(zip(test, hyps))[:3]:
    print("  EN :", r["src"][:100], "\n  REF:", r["tgt"][:100], "\n  HYP:", h[:100])
