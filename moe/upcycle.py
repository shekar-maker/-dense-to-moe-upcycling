"""Turning a trained dense model into a mixture-of-experts model.

Attention, embeddings and norms are copied unchanged. Each dense SwiGLU MLP becomes n_experts
experts in one of three ways, and every MoE layer gets a fresh router:

  copy       every expert is an exact copy of the dense MLP (sparse upcycling, Komatsuzaki et al., 2022).
             Because the top-k weights add up to one, the MoE model computes exactly the dense function
             at the moment of conversion.
  drop       every expert is a copy with a fraction r of its inner neurons re-initialised from the
             statistics of the original matrices (Drop-Upcycling, Nakamura et al., 2025).
  partition  every expert takes a random subset of d_expert of the d_ff inner neurons (subsets overlap
             between experts), and the output is scaled by d_ff / d_expert so that k experts add up to
             about one dense MLP (neuron splitting as in LLaMA-MoE, Zhu et al., 2024). With
             d_expert * top_k = d_ff the MoE model uses the same compute per token as the dense one.
"""
from dataclasses import replace

import torch

from .model import LM, Config


def _dense_mats(mlp):
    """Dense MLP -> (gate, up, down) as (d_model x d_ff, d_model x d_ff, d_ff x d_model)."""
    gu = mlp.gate_up.weight.detach().t()                 # d_model x 2*d_ff
    gate, up = gu.chunk(2, dim=1)
    return gate, up, mlp.down.weight.detach().t()


def _redraw_like(t):
    return torch.randn_like(t) * t.std() + t.mean()


@torch.no_grad()
def upcycle(dense: LM, method="copy", n_experts=8, top_k=2, d_expert=None, drop_ratio=0.5, balance="bias",
            seed=0, **moe_kw) -> LM:
    dcfg = dense.cfg
    d_expert = d_expert or dcfg.d_ff
    route_scale = 1.0
    if method == "partition":
        assert d_expert < dcfg.d_ff
        route_scale = dcfg.d_ff / d_expert
    else:
        assert d_expert == dcfg.d_ff, "copy and drop keep the full width"
    cfg = replace(dcfg, ffn="moe", n_experts=n_experts, top_k=top_k, d_expert=d_expert,
                  route_scale=route_scale, balance=balance, **moe_kw)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                                # router init
    moe = LM(cfg).to(next(dense.parameters()).device)

    # everything that is not a feed-forward weight loads unchanged
    sd = {k: v for k, v in dense.state_dict().items() if ".mlp." not in k}
    missing, unexpected = moe.load_state_dict(sd, strict=False)
    assert not unexpected and all(".mlp." in k for k in missing), (missing, unexpected)

    for blk_d, blk_m in zip(dense.blocks, moe.blocks):
        gate, up, down = _dense_mats(blk_d.mlp)
        m = blk_m.mlp
        for e in range(n_experts):
            if method in ("copy", "drop"):
                g, u, dn = gate.clone(), up.clone(), down.clone()
                if method == "drop":
                    idx = torch.randperm(dcfg.d_ff, generator=gen)[: int(drop_ratio * dcfg.d_ff)].to(g.device)
                    g[:, idx] = _redraw_like(gate)[:, idx]
                    u[:, idx] = _redraw_like(up)[:, idx]
                    dn[idx] = _redraw_like(down)[idx]
            elif method == "partition":
                idx = torch.randperm(dcfg.d_ff, generator=gen)[:d_expert].sort().values.to(gate.device)
                g, u, dn = gate[:, idx], up[:, idx], down[idx]
            else:
                raise ValueError(method)
            m.w_gate_up[e].copy_(torch.cat([g, u], dim=1))
            m.w_down[e].copy_(dn)
    return moe


@torch.no_grad()
def expert_similarity(model: LM):
    """Mean pairwise cosine similarity between the flattened experts of each MoE layer.
    1.0 means the experts are identical (a fresh copy-upcycle)."""
    out = []
    for m in model.moe_layers():
        flat = torch.cat([m.w_gate_up.flatten(1), m.w_down.flatten(1)], dim=1).float()
        flat = torch.nn.functional.normalize(flat, dim=1)
        sim = flat @ flat.t()
        E = sim.shape[0]
        out.append(((sim.sum() - E) / (E * (E - 1))).item())
    return out
