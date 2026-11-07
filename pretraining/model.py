"""The two pre-training methods, built on the SAME small transformer encoder.

  BERT-style (MLM): hide some spans, replace them by [MASK], predict the hidden WORDS.
  JEPA (I-JEPA for text): hide the same kind of spans, predict their REPRESENTATIONS:
      context encoder  : sees only the visible tokens (cannot attend to hidden positions)
      target encoder   : a slow copy of the context encoder (EMA, no gradients) sees everything;
                         its outputs at the hidden positions are the answers
      predictor        : small transformer; gets the context outputs + "a token is missing at
                         position p" and predicts the target encoder's vector there
      loss             : smooth-L1 between predicted and target vectors
      --reg > 0        : plus a VICReg-style variance/covariance penalty on the context encoder,
                         because plain I-JEPA collapsed here (all vectors the same) within ~50 steps
  Both methods hide exactly the same spans, so the only difference is WHAT is predicted.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, MASK, SEP = 0, 1, 2


class Encoder(nn.Module):
    def __init__(self, vocab, d=256, layers=6, heads=4, ff=1024, max_len=128, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # small init (BERT/GPT style). PyTorch's default N(0, 1) embeddings make tied-output
        # logits huge (MLM loss 62 at step 1 instead of ln(8000) = 9) and destabilise both methods.
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids, visible=None):
        """visible: bool [B, T], True = other tokens may attend to this position."""
        T = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        x = self.blocks(x, src_key_padding_mask=None if visible is None else ~visible)
        return self.norm(x)


def span_masks(B, T, n_spans=4, lo=0.10, hi=0.20, generator=None, device="cpu"):
    """Hide n_spans contiguous spans, each 10-20% of the sequence (~40-50% hidden in total)."""
    m = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        for _ in range(n_spans):
            L = max(1, int(T * (lo + (hi - lo) * torch.rand(1, generator=generator).item())))
            s = int(torch.randint(0, T - L + 1, (1,), generator=generator).item())
            m[b, s:s + L] = True
    return m.to(device)


class MLM(nn.Module):
    def __init__(self, vocab, **kw):
        super().__init__()
        self.enc = Encoder(vocab, **kw)
        d = self.enc.tok.embedding_dim
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def loss(self, ids, hidden):
        inp = ids.masked_fill(hidden, MASK)
        h = self.head(self.enc(inp))[hidden]
        logits = h @ self.enc.tok.weight.T + self.bias          # tied output embedding
        loss = F.cross_entropy(logits, ids[hidden])
        acc = (logits.argmax(-1) == ids[hidden]).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item()}

    def features_encoder(self):
        return self.enc


class JEPA(nn.Module):
    def __init__(self, vocab, d_pred=128, pred_layers=3, reg=0.0, **kw):
        super().__init__()
        self.reg = reg
        self.ctx = Encoder(vocab, **kw)
        self.tgt = copy.deepcopy(self.ctx)
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        d, T = self.ctx.tok.embedding_dim, self.ctx.pos.num_embeddings
        self.inp = nn.Linear(d, d_pred)
        self.mask_tok = nn.Parameter(torch.zeros(d_pred))
        nn.init.normal_(self.mask_tok, std=0.02)
        self.ppos = nn.Embedding(T, d_pred)
        nn.init.normal_(self.ppos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d_pred, 4, 4 * d_pred, 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.pred = nn.TransformerEncoder(layer, pred_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_pred), nn.Linear(d_pred, d))

    @torch.no_grad()
    def ema_update(self, m):
        for pt, pc in zip(self.tgt.parameters(), self.ctx.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)

    def loss(self, ids, hidden):
        with torch.no_grad():
            target = F.layer_norm(self.tgt(ids), (self.tgt.tok.embedding_dim,))
        c = self.ctx(ids, visible=~hidden)                      # hidden tokens are invisible
        T = ids.size(1)
        z = torch.where(hidden.unsqueeze(-1), self.mask_tok.expand_as(self.inp(c)), self.inp(c))
        z = z + self.ppos(torch.arange(T, device=ids.device))
        p = self.out(self.pred(z))
        loss = F.smooth_l1_loss(p[hidden], target[hidden])
        info = {}
        if self.reg:
            # anti-collapse (VICReg-style, as in C-JEPA): the context encoder's token vectors must
            # keep spread (std >= 1 per dimension) and use different dimensions (low covariance).
            zc = c[~hidden]
            zc = zc[torch.randperm(zc.size(0), device=zc.device)[:2048]].float()
            zc = zc - zc.mean(0)
            std = torch.sqrt(zc.var(0) + 1e-4)
            var_l = F.relu(1.0 - std).mean()
            cov = (zc.T @ zc) / (zc.size(0) - 1)
            cov_l = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / zc.size(1)
            loss = loss + self.reg * (var_l + 0.04 * cov_l)
            info = {"var_l": var_l.item(), "cov_l": cov_l.item()}
        # collapse checks: if everything maps to one vector, these fall towards 0
        tstd = target[hidden].std(0).mean()
        pstd = F.normalize(p[hidden], dim=-1).std(0).mean() * math.sqrt(p.size(-1))
        return loss, {"loss": loss.item(), "target_std": tstd.item(), "pred_std": pstd.item(), **info}

    def features_encoder(self):
        return self.tgt                                         # I-JEPA evaluates the target encoder
