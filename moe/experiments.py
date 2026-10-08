"""The runs behind the README, in the order they are trained.

1. dense            the 20.8M dense model, 30M tokens from scratch
2. dense_continue   the dense model trained on for 20M more tokens (the baseline every MoE run is compared with)
3. moe_copy         dense -> 8 experts, top-2, every expert a copy           (+20M tokens)
4. moe_drop         dense -> 8 experts, top-2, copies with 50% neurons redrawn (+20M tokens)
5. moe_partition    dense -> 8 experts of half width, top-2: same compute per token as the dense model (+20M tokens)
6. moe_copy_nobal   as moe_copy but with no load balancing at all          (+20M tokens)

All continuation runs see the same 20M tokens, which come straight after the 30M used for pre-training,
with the same learning-rate schedule (3% warm-up to 3e-4, cosine to 3e-5) and a fresh AdamW state.
Every run skips itself if its results/<name>.json and checkpoint already exist.
"""
import torch

from .model import LM, Config
from .train import train, load_model
from .upcycle import upcycle

BATCH = 32                      # sequences of 512 tokens per optimizer step = 16,384 tokens
PRETRAIN_TOKENS = 30_000_000
CONTINUE_TOKENS = 20_000_000
CONTINUE = dict(lr=3e-4, warmup_frac=0.03, min_ratio=0.1)

MOE_RUNS = {
    "moe_copy": dict(method="copy", balance="bias", accum=2),
    "moe_drop": dict(method="drop", balance="bias", accum=2),
    "moe_partition": dict(method="partition", d_expert=512, balance="bias", accum=1),
    "moe_copy_nobal": dict(method="copy", balance="none", accum=2),
}


def pretrain_windows():
    """Number of 512-token windows the pre-training run consumed (the continuation starts after them)."""
    return (PRETRAIN_TOKENS // (BATCH * 512)) * BATCH


def run_dense(**kw):
    return train("dense", LM(Config()).cuda(), batch=BATCH, total_tokens=PRETRAIN_TOKENS, lr=1e-3, **kw)


def run_continue(name, **kw):
    dense = load_model("checkpoints/dense.pt")
    common = dict(batch=BATCH, total_tokens=CONTINUE_TOKENS, skip_windows=pretrain_windows(), **CONTINUE, **kw)
    if name == "dense_continue":
        return train(name, dense, **common)
    spec = dict(MOE_RUNS[name])
    accum = spec.pop("accum")
    method = spec.pop("method")
    model = upcycle(dense, method, **spec)
    del dense
    return train(name, model, accum=accum, **common)


ALL = ["dense", "dense_continue", *MOE_RUNS]


def run_all(names=ALL, **kw):
    logs = {}
    for n in names:
        logs[n], model = run_dense(**kw) if n == "dense" else run_continue(n, **kw)
        del model
        torch.cuda.empty_cache()
    return logs
