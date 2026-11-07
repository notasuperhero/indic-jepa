"""Step 2: pre-train a small encoder from scratch on Assamese text, with BERT-style or JEPA training.

  python pretrain.py --method mlm  --out runs/mlm_s1
  python pretrain.py --method jepa --out runs/jepa_s1
  python pretrain.py --method jepa --out runs/smoke --steps 50     # quick check

Both use the same encoder (6 layers, width 256, ~7M parameters incl. embeddings), the same data,
the same hidden spans and the same number of steps.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import JEPA, MLM, span_masks

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["mlm", "jepa"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--ema", type=float, default=0.996, help="JEPA: start EMA momentum (rises to 1.0)")
ap.add_argument("--reg", type=float, default=0.0, help="JEPA: weight of the anti-collapse penalty")
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="")
ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--save_every", type=int, default=5000)
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)

V = Tokenizer.from_file("data/tokenizer.json").get_vocab_size()
train = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
print(f"device {dev}  method {a.method}  vocab {V}  train tokens {len(train):,}  "
      f"epochs over data ~{a.steps * a.bs * a.seq / len(train):.1f}")

kw = dict(d=a.d, layers=a.layers, heads=4, ff=4 * a.d, max_len=a.seq)
model = (MLM(V, **kw) if a.method == "mlm" else JEPA(V, reg=a.reg, **kw)).to(dev)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters {n_train / 1e6:.2f}M")

decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                         {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                          0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
g = torch.Generator().manual_seed(a.seed)
rng = np.random.default_rng(a.seed)


def batch(src, bs):
    ix = rng.integers(0, len(src) - a.seq, bs)
    return torch.from_numpy(np.stack([src[i:i + a.seq].astype(np.int64) for i in ix])).to(dev)


@torch.no_grad()
def evaluate(n=8):
    model.eval()
    vg = torch.Generator().manual_seed(123)
    vr = np.random.default_rng(123)
    out = []
    for _ in range(n):
        ix = vr.integers(0, len(val) - a.seq, a.bs)
        ids = torch.from_numpy(np.stack([val[i:i + a.seq].astype(np.int64) for i in ix])).to(dev)
        _, m = model.loss(ids, span_masks(a.bs, a.seq, generator=vg, device=dev))
        out.append(m)
    model.train()
    return {f"val_{k}": round(sum(o[k] for o in out) / n, 4) for k in out[0]}


log, t0 = [], time.time()
model.train()
for step in range(1, a.steps + 1):
    ids = batch(train, a.bs)
    hidden = span_masks(a.bs, a.seq, generator=g, device=dev)
    loss, m = model.loss(ids, hidden)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    sched.step()
    if a.method == "jepa":
        model.ema_update(a.ema + (1.0 - a.ema) * step / a.steps)
    if step % a.log_every == 0 or step == 1:
        row = {"step": step, **{k: round(v, 4) for k, v in m.items()}, "lr": round(sched.get_last_lr()[0], 6),
               "sec": round(time.time() - t0), "ms_per_step": round(1000 * (time.time() - t0) / step)}
        if step % (a.log_every * 10) == 0 or step == a.steps:
            row.update(evaluate())
        log.append(row)
        print(row, flush=True)
    if step % a.save_every == 0 or step == a.steps:
        torch.save({"args": vars(a), "vocab": V, "encoder": model.features_encoder().state_dict()},
                   os.path.join(a.out, "encoder.pt"))
json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s -> {a.out}/encoder.pt")
