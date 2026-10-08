"""Training loop for the dense and the MoE models. Every number in the README comes from here.

Each run writes three things:
  logs/<name>.log          the plain-text training log, one line every `log_every` steps plus every evaluation
  results/<name>.json      the same numbers in machine-readable form (loss curve, val curve, expert loads, ...)
  checkpoints/<name>.pt    the final weights (git-ignored)
"""
import json
import math
import os
import time

import torch

from .data import TokenLoader, load_tokenizer
from .model import LM, Config
from .upcycle import expert_similarity

# On Windows the driver can silently spill CUDA allocations into system RAM instead of failing.
MEM_FRACTION = 0.92


def setup_cuda():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    free, total = torch.cuda.mem_get_info()
    return {"gpu": torch.cuda.get_device_name(0), "total_gib": total / 2**30, "torch": torch.__version__}


def make_optimizer(model, lr, wd=0.1):
    # the router is a matrix but is not decayed: decay would pull every expert's score towards uniform
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.endswith("router")]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or n.endswith("router")]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_at(step, n_steps, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, n_steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x)
        tot += torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                                                 reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n


def load_stats(load):
    """load: (layers, experts) token counts. Returns per-layer MaxVio and the number of dead experts
    (no token at all) and starved experts (under 10% of the even share)."""
    mean = load.mean(1, keepdim=True)
    maxvio = ((load.max(1).values - mean[:, 0]) / mean[:, 0]).tolist()
    return maxvio, int((load == 0).sum()), int((load < 0.1 * mean).sum())


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    model = LM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    return model


def train(name, model: LM, batch=32, total_tokens=20_000_000, lr=1e-3, warmup_frac=0.03, min_ratio=0.1,
          data_dir="data", out_dir="results", log_dir="logs", ckpt_dir="checkpoints",
          eval_every_tokens=1_000_000, log_every=10, seed=1337, n_eval_batches=16, data_seed=None, skip_windows=0, accum=1, quiet=False):
    """Trains `model` for total_tokens. If the run already finished (JSON and checkpoint exist), the
    saved log and weights are returned instead of training again."""
    done, ckpt = os.path.join(out_dir, f"{name}.json"), os.path.join(ckpt_dir, f"{name}.pt")
    if os.path.exists(done) and os.path.exists(ckpt):
        with open(done) as f:
            log = json.load(f)
        if not quiet:
            print(f"[{name}] loaded finished run: final val {log['final_val_loss']:.4f}")
        return log, load_model(ckpt)

    info = setup_cuda()
    torch.manual_seed(seed)
    dev = "cuda"
    cfg = model.cfg
    moes = model.moe_layers()
    opt = make_optimizer(model, lr)
    tokens_per_step = batch * cfg.seq_len
    n_steps = total_tokens // tokens_per_step
    warmup = max(10, int(warmup_frac * n_steps))
    loader = TokenLoader(os.path.join(data_dir, "train.bin"), batch // accum, cfg.seq_len, dev,
                         seed=seed if data_seed is None else data_seed)
    loader._pos = skip_windows  # continue the pre-training data stream instead of replaying it
    val = TokenLoader(os.path.join(data_dir, "val.bin"), 16, cfg.seq_len, dev).fixed_batches(n_eval_batches, 16)
    eval_every = max(1, eval_every_tokens // tokens_per_step)

    os.makedirs(log_dir, exist_ok=True)
    txt = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")

    def emit(line):
        txt.write(line + "\n"); txt.flush()
        if not quiet:
            print(line, flush=True)

    log = {"name": name, "config": cfg.__dict__, "batch": batch, "accum": accum, "tokens_per_step": tokens_per_step,
           "n_steps": n_steps, "lr": lr, "warmup": warmup, "min_ratio": min_ratio,
           "n_params": model.n_params(), "n_active_params": model.n_active_params(), "env": info,
           "steps": [], "loss": [], "aux": [], "tok_per_s": [], "lr_trace": [], "gnorm": [],
           "maxvio": [], "dead": [], "starved": [],
           "val_steps": [], "val_loss": [], "val_expert_cos": [], "val_load": []}
    emit(f"run {name}: {cfg.ffn}, {model.n_params():,} parameters ({model.n_active_params():,} active per token), "
         f"{n_steps} steps x {tokens_per_step:,} tokens, lr {lr:g} (warm-up {warmup}, cosine to {min_ratio:g}x)")
    if moes:
        emit(f"  experts {cfg.n_experts} x {cfg.d_expert} wide, top-{cfg.top_k}, route scale {cfg.route_scale:g}, "
             f"balance {cfg.balance}")

    window_load = torch.zeros(len(moes), cfg.n_experts, device=dev) if moes else None

    def do_eval(step):
        vl = evaluate(model, val)
        log["val_steps"].append(step); log["val_loss"].append(vl)
        sim = expert_similarity(model) if moes else []
        log["val_expert_cos"].append(sim)
        extra = f" | expert cos-sim {sum(sim) / len(sim):.3f}" if sim else ""
        emit(f"eval step {step:5d} | tokens {step * tokens_per_step / 1e6:6.2f}M | val loss {vl:.4f}{extra}")
        return vl

    torch.cuda.reset_peak_memory_stats()
    model.train()
    do_eval(0)
    t_start = time.perf_counter()
    t_window, tok_window = time.perf_counter(), 0
    timed_tokens, timed_seconds = 0, 0.0
    for step in range(n_steps):
        cur_lr = lr_at(step, n_steps, lr, warmup, min_ratio)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for m in moes:
            m.load.zero_()
        loss_sum, aux_sum = 0.0, 0.0
        for _ in range(accum):  # gradient accumulation: same optimizer step, smaller activations
            x, y = loader.next()
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
                aux = model.aux_loss() if cfg.balance == "aux" else None
                total = loss + cfg.aux_alpha * aux if aux is not None else loss
            (total / accum).backward()
            loss_sum = loss_sum + loss.detach() / accum
            aux_sum = aux_sum + aux.detach() / accum if aux is not None else None
        loss, aux = loss_sum, aux_sum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if moes:
            step_load = torch.stack([m.load for m in moes])
            window_load += step_load
            if cfg.balance == "bias":
                for m, ld in zip(moes, step_load):
                    m.update_bias(ld)
        tok_window += tokens_per_step
        if (step + 1) % log_every == 0 or step == n_steps - 1:
            lv = loss.item()
            dt = time.perf_counter() - t_window
            tps = tok_window / dt
            if step >= 20:
                timed_tokens += tok_window; timed_seconds += dt
            log["steps"].append(step + 1); log["loss"].append(lv); log["tok_per_s"].append(tps)
            log["lr_trace"].append(cur_lr); log["gnorm"].append(gnorm.item())
            log["aux"].append(aux.item() if aux is not None else None)
            line = (f"step {step + 1:5d}/{n_steps} | tokens {(step + 1) * tokens_per_step / 1e6:6.2f}M | "
                    f"loss {lv:.4f} | lr {cur_lr:.2e} | gnorm {gnorm.item():.2f} | {tps:7,.0f} tok/s")
            if moes:
                mv, dead, starved = load_stats(window_load)
                log["maxvio"].append(mv); log["dead"].append(dead); log["starved"].append(starved)
                line += (f" | maxvio mean {sum(mv) / len(mv):.2f} max {max(mv):.2f} | "
                         f"dead {dead}/{window_load.numel()} | starved {starved}")
                if aux is not None:
                    line += f" | aux {aux.item():.4f}"
                if (step + 1) % eval_every == 0 or step == n_steps - 1:
                    log["val_load"].append((window_load / window_load.sum(1, keepdim=True)).tolist())
                window_load.zero_()
            emit(line)
            t_window, tok_window = time.perf_counter(), 0
            if not math.isfinite(lv):
                log["diverged_at_step"] = step + 1
                break
        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            t0 = time.perf_counter()
            do_eval(step + 1)
            t_window += time.perf_counter() - t0
    wall = time.perf_counter() - t_start
    tail = log["loss"][-max(1, len(log["loss"]) // 20):]
    log.update({
        "final_train_loss": sum(tail) / len(tail),
        "final_val_loss": log["val_loss"][-1],
        "tokens_seen": (step + 1) * tokens_per_step,
        "tok_per_s_mean": timed_tokens / timed_seconds if timed_seconds else float("nan"),
        "wall_s": wall,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    if moes:
        log["final_sel_bias"] = [m.sel_bias.tolist() for m in moes]
    emit(f"done {name}: val loss {log['val_loss'][0]:.4f} -> {log['final_val_loss']:.4f}, "
         f"{log['tok_per_s_mean']:,.0f} tok/s, peak {log['peak_alloc_gib']:.2f} GiB, {wall / 60:.1f} min")
    txt.close()
    os.makedirs(out_dir, exist_ok=True)
    with open(done, "w") as f:
        json.dump(log, f)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, ckpt)
    return log, model


def sample(model, prompts, data_dir="data", n_new=120, seed=0):
    model.eval()
    tok = load_tokenizer(data_dir)
    torch.manual_seed(seed)
    outs = []
    for p in prompts:
        idx = torch.tensor([tok.encode(p).ids], device="cuda")
        with torch.autocast("cuda", torch.bfloat16):
            out = model.generate(idx, n_new)
        outs.append(tok.decode(out[0].tolist(), skip_special_tokens=False).split("<|eos|>")[0])
    model.train()
    return outs
