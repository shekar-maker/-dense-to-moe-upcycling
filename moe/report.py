"""Figures and tables built from results/*.json. Nothing here trains; it only reads the run logs."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "dense_continue": "dense, continued",
    "moe_copy": "MoE copy (8x, top-2)",
    "moe_drop": "MoE drop r=0.5 (8x, top-2)",
    "moe_partition": "MoE partition (8 x half, top-2)",
    "moe_copy_nobal": "MoE copy, no balancing",
}
COLORS = {"dense": "#555555", "dense_continue": "#555555", "moe_copy": "#1f77b4", "moe_drop": "#2ca02c",
          "moe_partition": "#ff7f0e", "moe_copy_nobal": "#d62728"}


def load(name, out_dir="results"):
    with open(os.path.join(out_dir, f"{name}.json")) as f:
        return json.load(f)


def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def summary(names, out_dir="results"):
    rows = []
    for n in names:
        r = load(n, out_dir)
        row = {"run": n, "params": r["n_params"], "active": r["n_active_params"], "tokens": r["tokens_seen"],
               "val_start": r["val_loss"][0], "val_end": r["final_val_loss"],
               "train_end": r["final_train_loss"],
               "peak_gib": r["peak_alloc_gib"]}
        # median over 10-step windows: the laptop slept during some runs, which wrecks a whole-run average
        row["tok_s"] = float(np.median(r["tok_per_s"][2:]))
        row["minutes"] = r["tokens_seen"] / row["tok_s"] / 60
        if r["maxvio"]:
            k = max(1, len(r["maxvio"]) // 10)       # last 10% of the run
            row["maxvio_end"] = float(np.mean([np.mean(m) for m in r["maxvio"][-k:]]))
            row["dead_end"] = float(np.mean(r["dead"][-k:]))
            row["starved_end"] = float(np.mean(r["starved"][-k:]))
            row["cos_start"] = float(np.mean(r["val_expert_cos"][0]))
            row["cos_end"] = float(np.mean(r["val_expert_cos"][-1]))
        rows.append(row)
    return rows


def plot_curves(cont_names, path="results/loss_curves.png", out_dir="results"):
    """Left: the whole history (dense pre-training, then every continuation). Right: val loss of the
    continuation only."""
    dense = load("dense", out_dir)
    off = dense["tokens_seen"] / 1e6
    tps = dense["tokens_per_step"] / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    s = smooth(dense["loss"])
    ax[0].plot(np.array(dense["steps"][len(dense["steps"]) - len(s):]) * tps, s, color=COLORS["dense"], label="dense pre-training")
    for n in cont_names:
        r = load(n, out_dir)
        tp = r["tokens_per_step"] / 1e6
        s = smooth(r["loss"])
        ax[0].plot(off + np.array(r["steps"][len(r["steps"]) - len(s):]) * tp, s, color=COLORS[n], label=LABELS[n], lw=1)
        ax[1].plot(off + np.array(r["val_steps"]) * tp, r["val_loss"], "o-", ms=3, color=COLORS[n], label=LABELS[n])
    ax[0].axvline(off, color="k", ls=":", lw=1)
    ax[0].text(off, 3.5, " conversion", fontsize=9)
    ax[0].set_ylim(1.4, 4.0)
    ax[0].set_xlabel("tokens (M)"); ax[0].set_ylabel("train loss (10-point moving average)")
    ax[0].set_title("Training loss: 30M dense tokens, then 20M more per run")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("tokens (M)"); ax[1].set_ylabel("val loss (131k held-out tokens)")
    ax[1].set_title("Validation loss after the conversion")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_conversion_zoom(cont_names, n_points=30, path="results/conversion_zoom.png", out_dir="results"):
    """The first steps after the conversion, unsmoothed: does the loss jump, and how fast does it come back?"""
    dense = load("dense", out_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in cont_names:
        r = load(n, out_dir)
        ax.plot(r["steps"][:n_points], r["loss"][:n_points], "o-", ms=3, color=COLORS[n], label=LABELS[n])
    ax.axhline(dense["final_train_loss"], color="k", ls=":", lw=1, label="dense, end of pre-training")
    ax.set_xlabel("optimizer steps after the conversion"); ax.set_ylabel("train loss")
    ax.set_title("Right after the conversion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_routing(moe_names, path="results/routing.png", out_dir="results"):
    """Load balance and expert diversity over the continuation."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for n in moe_names:
        r = load(n, out_dir)
        tp = r["tokens_per_step"] / 1e6
        ax[0].plot(np.array(r["steps"]) * tp, smooth([np.mean(m) for m in r["maxvio"]], 1), color=COLORS[n], label=LABELS[n], lw=1)
        ax[1].plot(np.array(r["steps"]) * tp, r["starved"], color=COLORS[n], label=LABELS[n], lw=1)
        ax[2].plot(np.array(r["val_steps"]) * tp, [np.mean(c) for c in r["val_expert_cos"]], "o-", ms=3, color=COLORS[n], label=LABELS[n])
    ax[0].set_yscale("log")
    ax[0].set_title("MaxVio per 10-step window, mean over 10 layers")
    ax[1].set_title("starved experts (<10% of an even share), of 80")
    ax[2].set_title("mean pairwise cosine similarity between experts")
    for a in ax:
        a.set_xlabel("tokens after the conversion (M)"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_load_heatmaps(moe_names, path="results/expert_load.png", out_dir="results"):
    """Share of routed tokens per (layer, expert) over the last 10 optimizer steps (164k tokens) of each run. Even = 12.5%."""
    fig, ax = plt.subplots(1, len(moe_names), figsize=(4.2 * len(moe_names), 4))
    for a, n in zip(np.atleast_1d(ax), moe_names):
        r = load(n, out_dir)
        L = np.array(r["val_load"][-1]) * 100
        im = a.imshow(L, cmap="viridis", vmin=0, vmax=50, aspect="auto")
        a.set_title(LABELS[n], fontsize=9)
        a.set_xlabel("expert"); a.set_ylabel("layer")
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                a.text(j, i, f"{L[i, j]:.0f}", ha="center", va="center", fontsize=6,
                       color="white" if L[i, j] < 25 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of routed tokens")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path
