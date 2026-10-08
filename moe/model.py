"""A ~20M-parameter decoder-only LM whose feed-forward block is either one dense SwiGLU MLP or a
mixture of SwiGLU experts behind a top-k router.

Everything except the feed-forward block (embedding, RoPE attention, RMSNorm, tied output head) is
identical in both, so a dense checkpoint loads into the MoE model unchanged and only the
feed-forward weights have to be turned into experts (see upcycle.py).

The MoE layer:
  logits = x W_r                      (fp32, W_r is d_model x n_experts)
  scores = softmax(logits)
  chosen = top_k(scores + b)          b is a per-expert selection bias, not trained by gradient
  y      = route_scale * sum_{i in chosen} (s_i / sum_chosen s) * E_i(x)
Balancing is either the auxiliary-loss-free bias of Wang et al. (2024) (b_i moves by
rate * sign(mean load - load_i) after every optimizer step), the Switch Transformer auxiliary
loss (Fedus et al., 2021), or nothing. No token is ever dropped: every expert processes every
token routed to it.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    seq_len: int = 512
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    d_ff: int = 1024              # dense MLP inner width
    ffn: str = "dense"            # dense | moe
    n_experts: int = 8
    top_k: int = 2
    d_expert: int = 1024          # inner width of one expert
    route_scale: float = 1.0      # multiplies the renormalised top-k weights
    balance: str = "bias"         # bias | aux | none
    bias_rate: float = 1e-3       # update speed of the selection bias
    aux_alpha: float = 0.01       # weight of the Switch auxiliary loss (balance="aux")
    router_std: float = 0.002     # router init, one tenth of the usual 0.02 (Switch Transformer)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(seq_len, head_dim, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(seq_len).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.hd = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    """Dense SwiGLU: down(silu(gate x) * up x). gate_up.weight holds the gate rows, then the up rows."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class MoE(nn.Module):
    """Top-k routed SwiGLU experts. Expert e computes (silu(x G_e) * x U_e) D_e with
    w_gate_up[e] = [G_e | U_e] (d_model x 2*d_expert) and w_down[e] = D_e (d_expert x d_model)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        E, d, de = cfg.n_experts, cfg.d_model, cfg.d_expert
        self.router = nn.Parameter(torch.randn(E, d) * cfg.router_std)
        self.w_gate_up = nn.Parameter(torch.randn(E, d, 2 * de) * 0.02)
        self.w_down = nn.Parameter(torch.randn(E, de, d) * 0.02 / math.sqrt(2 * cfg.n_layers))
        self.register_buffer("sel_bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))   # tokens per expert since the last reset
        self.aux_loss = None

    def route(self, x):
        """x: (N, d). Returns chosen expert ids (N, k), their weights (N, k) and the full scores (N, E)."""
        with torch.autocast("cuda", enabled=False):
            scores = F.softmax(x.float() @ self.router.float().t(), dim=-1)
        chosen = torch.topk(scores + self.sel_bias, self.cfg.top_k, dim=-1).indices
        w = scores.gather(1, chosen)
        w = w / w.sum(-1, keepdim=True) * self.cfg.route_scale
        return chosen, w, scores

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        N, E, k = x.shape[0], self.cfg.n_experts, self.cfg.top_k
        chosen, w, scores = self.route(x)
        flat = chosen.reshape(-1)
        counts = torch.bincount(flat, minlength=E)
        self.load += counts.float().detach()
        if self.training and self.cfg.balance == "aux":
            f = counts.float() / (N * k)                       # share of routed copies, no gradient
            self.aux_loss = E * (f * scores.mean(0)).sum()      # Switch Transformer, Eq. 4
        # Sort the N*k (token, expert) pairs by expert, run each expert on its contiguous slice.
        order = torch.argsort(flat, stable=True)
        tok = order // k
        wts = w.reshape(-1)[order]
        xs = x[tok]
        out = torch.zeros(N, D, device=x.device, dtype=torch.float32)
        start = 0
        for e, n_e in enumerate(counts.tolist()):
            if n_e == 0:
                continue
            sl = slice(start, start + n_e)
            g, u = (xs[sl] @ self.w_gate_up[e]).chunk(2, dim=-1)
            ye = (F.silu(g) * u) @ self.w_down[e]
            out.index_add_(0, tok[sl], (ye.float() * wts[sl, None]))
            start += n_e
        return out.view(B, T, D).to(x.dtype)

    @torch.no_grad()
    def update_bias(self, load):
        """Auxiliary-loss-free balancing: push quiet experts up and busy ones down by a fixed step."""
        self.sel_bias += self.cfg.bias_rate * torch.sign(load.mean() - load)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MoE(cfg) if cfg.ffn == "moe" else MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model)
        cos, sin = rope_tables(cfg.seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for n, prm in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(prm, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def moe_layers(self):
        return [b.mlp for b in self.blocks if isinstance(b.mlp, MoE)]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_active_params(self):
        """Parameters one token uses: everything except the experts it was not routed to."""
        total = self.n_params()
        for m in self.moe_layers():
            per_expert = m.w_gate_up[0].numel() + m.w_down[0].numel()
            total -= (m.cfg.n_experts - m.cfg.top_k) * per_expert
        return total

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = F.linear(self.ln_f(x), self.emb.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return loss

    def aux_loss(self):
        terms = [m.aux_loss for m in self.moe_layers() if m.aux_loss is not None]
        return sum(terms) / len(terms) if terms else None

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=40):
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.seq_len:])[:, -1].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx
