"""Step 2: fine-tune a small LLM to translate English -> target, with or without the LLM-JEPA loss.

  --mode sft   normal fine-tuning: next-token loss on the target sentence only
  --mode jepa  the same + LLM-JEPA term (Huang, LeCun & Balestriero, ICLR 2026):
               view A = the prompt (instruction + English sentence),
               view B = the target-language sentence on its own.
               Each view is run through the model; its embedding is the last layer's
               hidden state at its last token. JEPA loss = 1 - cos(emb_A, emb_B),
               i.e. the model learns to place a sentence and its translation at the
               same point in its representation space.
               total = LM loss + lbd * JEPA loss   (as in the official finetune.py)
               Logged "cos_wrong_pair" = similarity to a mismatched translation. If it climbs
               as high as "cos", the embeddings have collapsed and the JEPA term is not
               learning meaning - an important thing to check and report.

LoRA is used so a 1-2B model fits and trains on a 48 GB Mac (MPS).

  python train.py --mode jepa --train data/asm_Beng_train_1000.jsonl --out runs/asm_1000_jepa_s1 --seed 1
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import LANG_NAMES, load_model, pick_device, prompt_text, read_jsonl, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-2-2b-it")
ap.add_argument("--lang", default="asm_Beng")
ap.add_argument("--train", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["sft", "jepa"], required=True)
ap.add_argument("--lbd", type=float, default=1.0, help="weight of the JEPA loss")
ap.add_argument("--variant", default="plain", choices=["plain", "centered", "contrastive"],
                help="plain = 1-cos (LLM-JEPA paper); centered = 1-cos after removing the batch-mean "
                     "embedding; contrastive = InfoNCE over the batch (right translation vs the others)")
ap.add_argument("--pred_k", type=int, default=0,
                help="number of [PRED] predictor tokens appended to view A (LLM-JEPA paper tunes 0-4)")
ap.add_argument("--tau", type=float, default=0.1, help="temperature for --variant contrastive")
ap.add_argument("--epochs", type=int, default=4, help="LLM-JEPA paper fixes 4 epochs")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--max_len", type=int, default=192)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="")
ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
ap.add_argument("--max_steps", type=int, default=0, help="stop early (0 = full run); for smoke tests")
a = ap.parse_args()

set_seed(a.seed)
dev = pick_device(a.device)
lang_name = LANG_NAMES.get(a.lang, a.lang)
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)

tok, model = load_model(a.model, dev, a.dtype)
tok.padding_side = "right"
from peft import LoraConfig, get_peft_model  # noqa: E402

targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
           if any(n in name for name, _ in model.named_modules())]
model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                         target_modules=targets, task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# predictor token: a reserved, otherwise unused token (Gemma has <unused0>..); the LoRA layers learn
# what to do with it. Falls back to EOS for tokenizers without reserved tokens.
vocab = tok.get_vocab()
PRED_ID = vocab.get("<unused0>", tok.eos_token_id)
if a.mode == "jepa" and a.pred_k:
    print(f"predictor tokens: {a.pred_k} x id {PRED_ID} ({tok.convert_ids_to_tokens(PRED_ID)})")

data = read_jsonl(a.train)
print(f"device {dev}  train pairs {len(data)}  mode {a.mode}  lang {lang_name}"
      + (f"  variant {a.variant}  lbd {a.lbd}  pred_k {a.pred_k}" if a.mode == "jepa" else ""))


def encode_batch(rows):
    """Tokenise one batch: the LM sequence (prompt + target, loss on target only) and, for JEPA, the two views."""
    eos = tok.eos_token or ""
    full_ids, labels, plen = [], [], []
    for r in rows:
        p = tok(prompt_text(r["src"], lang_name), add_special_tokens=True)["input_ids"]
        t = tok(" " + r["tgt"] + eos, add_special_tokens=False)["input_ids"]
        ids = (p + t)[: a.max_len]
        lab = ([-100] * len(p) + t)[: a.max_len]
        full_ids.append(ids)
        labels.append(lab)
        plen.append(min(len(p), a.max_len))
    L = max(map(len, full_ids))
    pad = tok.pad_token_id
    batch = {
        "input_ids": torch.tensor([x + [pad] * (L - len(x)) for x in full_ids]),
        "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in full_ids]),
        "labels": torch.tensor([x + [-100] * (L - len(x)) for x in labels]),
    }
    batch["prompt_last"] = torch.tensor(plen) - 1
    if a.mode == "jepa":
        vb = tok([r["tgt"] for r in rows], padding=True, truncation=True, max_length=a.max_len,
                 return_tensors="pt")
        batch["view_b"] = vb
        if a.pred_k:   # view A needs its own pass: prompt + k predictor tokens
            ids = [tok(prompt_text(r["src"], lang_name), add_special_tokens=True)["input_ids"][: a.max_len]
                   + [PRED_ID] * a.pred_k for r in rows]
            La = max(map(len, ids))
            batch["view_a"] = {
                "input_ids": torch.tensor([x + [pad] * (La - len(x)) for x in ids]),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (La - len(x)) for x in ids])}
    return batch


def last_token_embedding(view):
    out = model(input_ids=view["input_ids"].to(dev), attention_mask=view["attention_mask"].to(dev),
                output_hidden_states=True)
    h = out.hidden_states[-1]                                   # [B, T, d]
    last = view["attention_mask"].sum(1).to(dev) - 1            # right padding: last real token
    return h[torch.arange(h.size(0), device=dev), last]


steps_per_epoch = math.ceil(len(data) / a.bs)
total = steps_per_epoch * a.epochs // a.grad_accum
if a.max_steps:
    total = min(total, a.max_steps)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(           # linear warm-up, then linear decay
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

g = torch.Generator().manual_seed(a.seed)
model.train()
step, t0, log = 0, time.time(), []
for ep in range(a.epochs):
    order = torch.randperm(len(data), generator=g).tolist()
    for bi in range(steps_per_epoch):
        rows = [data[i] for i in order[bi * a.bs:(bi + 1) * a.bs]]
        b = encode_batch(rows)
        out = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev),
                    labels=b["labels"].to(dev), output_hidden_states=(a.mode == "jepa"))
        lm = out.loss
        jepa = torch.zeros((), device=dev)
        cos = torch.zeros((), device=dev)
        cos_neg = torch.zeros((), device=dev)
        match_acc = torch.zeros((), device=dev)
        if a.mode == "jepa":
            # view A = the prompt. In a causal model the hidden state at the prompt's last token
            # inside the full sequence is exactly the prompt's own embedding, so we read it from
            # the LM pass instead of running the model a third time.
            if a.pred_k:
                ea = last_token_embedding(b["view_a"])      # embedding at the last [PRED] token
            else:
                h = out.hidden_states[-1]
                ea = h[torch.arange(h.size(0), device=dev), b["prompt_last"].to(dev)]
            eb = last_token_embedding(b["view_b"])
            cos = F.cosine_similarity(ea.float(), eb.float(), dim=-1).mean()
            if a.variant == "plain":
                jepa = 1.0 - cos
            elif a.variant == "centered":
                # LLM embeddings all share one big common direction (cos ~0.99 for any pair), which
                # makes plain 1-cos nearly zero. Remove the batch mean so only differences count.
                mu = torch.cat([ea, eb]).float().mean(0, keepdim=True)
                jepa = 1.0 - F.cosine_similarity(ea.float() - mu, eb.float() - mu, dim=-1).mean()
            else:
                za, zb = F.normalize(ea.float(), dim=-1), F.normalize(eb.float(), dim=-1)
                logits = za @ zb.T / a.tau
                tgt = torch.arange(logits.size(0), device=dev)
                jepa = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            with torch.no_grad():   # collapse check: similarity to a NON-matching translation
                cos_neg = F.cosine_similarity(ea.float(), eb.float().roll(1, 0), dim=-1).mean()
                sim = F.normalize(ea.float(), dim=-1) @ F.normalize(eb.float(), dim=-1).T   # [B, B]
                match_acc = (sim.argmax(1) == torch.arange(sim.size(0), device=dev)).float().mean()
        loss = (lm + a.lbd * jepa) / a.grad_accum
        loss.backward()
        if (bi + 1) % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if dev == "mps":
                torch.mps.empty_cache()
            if step % 20 == 0 or step <= 10:
                mem = round(torch.mps.driver_allocated_memory() / 2**30, 1) if dev == "mps" else None
                row = {"step": step, "epoch": ep, "lm": round(lm.item(), 4), "jepa": round(jepa.item(), 4),
                       "cos": round(cos.item(), 4), "cos_wrong_pair": round(cos_neg.item(), 4),
                       "match_acc": round(match_acc.item(), 3),
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0),
                       "sec_per_step": round((time.time() - t0) / step, 1), "mem_GB": mem}
                log.append(row)
                print(row, flush=True)
            if a.max_steps and step >= a.max_steps:
                break
    if a.max_steps and step >= a.max_steps:
        break

model.save_pretrained(os.path.join(a.out, "adapter"))
json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s -> {a.out}/adapter")
