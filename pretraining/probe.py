"""Step 3: how good are the learned representations? Assamese sentiment (IndicSentiment).

The encoder is FROZEN. Each review -> mean of the encoder's token vectors -> a logistic-regression
classifier on top. We vary how many labelled reviews the classifier gets (scarce labels!):
  k = 16, 64, 256 labelled reviews, tested on 400 held-out reviews, 10 random draws each
  plus "all": 5-fold cross-validation over all reviews.

  python probe.py --ckpt runs/jepa_s1/encoder.pt --name jepa_s1
  python probe.py --random --name random_init          # no pre-training (floor)
"""
import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import PAD, Encoder

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--random", action="store_true")
ap.add_argument("--name", required=True)
ap.add_argument("--ks", default="16,64,256")
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--n_test", type=int, default=400)
ap.add_argument("--l2", type=float, default=1e-3)
ap.add_argument("--device", default="")
a = ap.parse_args()
dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

tok = Tokenizer.from_file("data/tokenizer.json")
if a.random:
    torch.manual_seed(0)
    args = {"d": 256, "layers": 6, "seq": 128}
    enc = Encoder(tok.get_vocab_size(), d=256, layers=6, heads=4, ff=1024, max_len=128)
else:
    ck = torch.load(a.ckpt, map_location="cpu")
    args = ck["args"]
    enc = Encoder(ck["vocab"], d=args["d"], layers=args["layers"], heads=4, ff=4 * args["d"],
                  max_len=args["seq"])
    enc.load_state_dict(ck["encoder"])
enc.to(dev).eval()
T = args["seq"]

rows = [json.loads(l) for l in open("data/sentiment.jsonl", encoding="utf-8")]
y = torch.tensor([r["label"] for r in rows])


@torch.no_grad()
def featurise(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc_ = [e.ids[:T] for e in tok.encode_batch(texts[i:i + bs])]
        L = max(map(len, enc_))
        ids = torch.tensor([x + [PAD] * (L - len(x)) for x in enc_], device=dev)
        vis = ids != PAD
        h = enc(ids, visible=vis)
        m = vis.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out)


X = featurise([r["text"] for r in rows])
print(f"{a.name}: {len(rows)} reviews -> features {tuple(X.shape)}")


def fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.size(1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr.float()) + a.l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return ((Xte @ w + b) > 0).long()


res = {}
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(rows), generator=g)
test_idx, pool = perm[: a.n_test], perm[a.n_test:]
for k in [int(x) for x in a.ks.split(",")]:
    accs = []
    for d in range(a.draws):
        gi = torch.Generator().manual_seed(1000 + d)
        # balanced draw: k/2 positive, k/2 negative
        pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
        tr = torch.cat([pos[torch.randperm(len(pos), generator=gi)[: k // 2]],
                        neg[torch.randperm(len(neg), generator=gi)[: k // 2]]])
        pred = fit_predict(X[tr], y[tr], X[test_idx])
        accs.append((pred == y[test_idx]).float().mean().item())
    t = torch.tensor(accs)
    res[f"k{k}"] = (t.mean().item() * 100, t.std().item() * 100)
folds = perm.chunk(5)
accs = []
for i in range(5):
    te = folds[i]
    tr = torch.cat([folds[j] for j in range(5) if j != i])
    accs.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).float().mean().item())
t = torch.tensor(accs)
res["all"] = (t.mean().item() * 100, t.std().item() * 100)

for k, (m, s) in res.items():
    print(f"  {k:>5}: accuracy {m:.1f}% ± {s:.1f}")
new = not os.path.exists("results.csv")
with open("results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["name"] + [f"{k}_acc" for k in res] + [f"{k}_sd" for k in res])
    w.writerow([a.name] + [round(v[0], 2) for v in res.values()] + [round(v[1], 2) for v in res.values()])
