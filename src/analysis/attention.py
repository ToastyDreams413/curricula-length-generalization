"""Attention maps and the layer-2 "successor attention" score.

CLI:
    python -m src.analysis.attention --run results/<run_id> --phase 1 --length 20 [--example 0]
        [--out FILE.png|.pdf] [--device cpu] [--succ-n 500] [--succ-out FILE.json]
--succ-n N > 0 also computes successor_attention_score on uniform_set(L, N, TEST_SEED) and writes it
to --succ-out (default <run>/succ_attn_phase{p}_len{L}.json).

Conventions (spec section 1). The model sees the teacher-forced input encode_full(x)[:, :-1], of
length T = 2n+1: BOS at 0, x_1..x_n at 1..n, SEP at n+1, and y_1..y_{n-1} at n+2..2n. The query at
index n+1+k (k = 0..n-1; k = 0 is SEP) predicts y_{k+1}.
Example k of plot_attention is row k of uniform_set(L, ., TEST_SEED). numpy fills rows in order, so
it is the same list as row k of the probe and test sets.

successor_attention_score answers "does a layer-2 head attend to the input copy of the next token?":
  * queries: indices n+1..2n, so SEP is INCLUDED by default (include_sep). Their targets are
    y_1..y_n. At SEP the target is y_1 = min(x).
  * keys: input indices 1..n holding the target value. Attention is SUMMED over all copies when the
    value repeats. Copies already written on the output side (indices > n+1) are NOT counted.
  * per head: the mean attention mass on those keys ("per_head") and the fraction of queries whose
    argmax key is such a copy ("per_head_argmax"). Also broken down by query type: SEP, "change"
    (y_{k+1} != y_k, a true successor) and "dup" (y_{k+1} == y_k, the next token repeats the current
    one). "uniform_baseline" is what a head attending uniformly over the k+1 visible keys would score.

Tuning hygiene: for a run whose config has eval.id_only: true, plot_attention, the CLI and
successor_attention_score (on a model from probes.load_run_model, or with max_length=) raise
probes.IdOnlyError for lengths above the max training length (max upper bound over data.groups).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from src.analysis.probes import (DEFAULT_BATCH, DEFAULT_MAX_ATTN_MB, IdOnlyError, _forward,
                                 _model_device, check_id_only, chunk_rows, id_only_max_len,
                                 load_run_checkpoint, load_run_model, probe_inputs, run_id_of)
from src.data import BOS, PAD, SEP, TEST_SEED, sort_targets, uniform_set

# Single-hue sequential ramp (light -> dark blue). White at 0, so near-zero attention recedes;
# causally masked cells get a neutral gray, so "not allowed" reads differently from "zero".
_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_MASKED = "#f0efec"
_INK, _MUTED = "#2b2b2b", "#8a8a85"


def token_labels(seq) -> list:
    names = {BOS: "BOS", SEP: "SEP", PAD: "PAD"}
    return [names.get(int(t), str(int(t))) for t in seq]


@torch.no_grad()
def attention_maps(model, x: np.ndarray, device=None):
    """One unsorted list x (n,) -> (tokens (T,) int64, [ (H, T, T) numpy per layer ]), teacher-forced."""
    dev = _model_device(model, device)
    tokens = probe_inputs(np.asarray(x, dtype=np.int64)[None, :])
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        _, internals = _forward(model, torch.as_tensor(tokens, device=dev))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return tokens[0], [a[0].float().cpu().numpy() for a in internals["attn"]]


def _attention_figure(tokens, maps, title: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    cmap = LinearSegmentedColormap.from_list("attn_blue", _RAMP).with_extremes(bad=_MASKED)
    labels = token_labels(tokens)
    T, n_layers, H = len(tokens), len(maps), maps[0].shape[0]
    sep = labels.index("SEP")
    panel = float(np.clip(0.065 * T, 2.6, 5.5))                 # inches per heatmap
    fs = float(np.clip(0.8 * panel * 72 / T, 2.5, 7.0))          # tick label size, pt
    fig = Figure(figsize=(panel * H + 1.2, panel * n_layers + 0.9), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_layers, H, squeeze=False)
    future = np.triu(np.ones((T, T), dtype=bool), 1)
    im = None
    for l in range(n_layers):
        for h in range(H):
            ax = axes[l][h]
            im = ax.imshow(np.ma.masked_array(maps[l][h], mask=future), cmap=cmap, vmin=0.0,
                           vmax=1.0, interpolation="nearest", aspect="equal")
            ax.set_title(f"layer {l + 1}, head {h}", fontsize=8, color=_INK)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(labels, rotation=90, fontsize=fs, color=_INK)
            ax.set_yticklabels(labels, fontsize=fs, color=_INK)
            ax.tick_params(length=0, pad=1)
            for s in ax.spines.values():
                s.set_color(_MUTED)
                s.set_linewidth(0.5)
            for p in (0.5, sep + 0.5):                           # BOS | input | SEP + output
                ax.axhline(p, color=_MUTED, lw=0.5)
                ax.axvline(p, color=_MUTED, lw=0.5)
            if h == 0:
                ax.set_ylabel("query (row)", fontsize=7, color=_INK)
            if l == n_layers - 1:
                ax.set_xlabel("key (column)", fontsize=7, color=_INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, fraction=0.02, pad=0.01)
    cb.set_label("attention probability", fontsize=8, color=_INK)
    cb.ax.tick_params(labelsize=7, colors=_INK)
    cb.outline.set_edgecolor(_MUTED)
    fig.suptitle(title, fontsize=10, color=_INK)
    return fig


def plot_attention(run_dir: str, phase: int, length: int, example_idx: int = 0, out=None,
                   device="cpu") -> list:
    """Heatmaps (one per layer x head) for test example `example_idx` of length `length`.
    out=None writes <run_dir>/attention_phase{p}_len{L}_ex{i}.png and .pdf; otherwise writes
    `out` (format from its extension). Returns the written paths.
    IdOnlyError if the run has eval.id_only: true and `length` exceeds its max training length."""
    length, example_idx = int(length), int(example_idx)
    ckpt = load_run_checkpoint(run_dir, phase)
    check_id_only(length, id_only_max_len(run_dir, ckpt), f"run {run_id_of(run_dir, ckpt)}")
    model, ckpt, dev = load_run_model(run_dir, phase, device, min_len=2 * length + 2, ckpt=ckpt)
    x = uniform_set(length, example_idx + 1, TEST_SEED)[example_idx]
    tokens, maps = attention_maps(model, x, dev)
    title = (f"{run_id_of(run_dir, ckpt)}, phase {int(phase)}: length {length}, test example "
             f"{example_idx} (teacher-forced; rows = queries, columns = keys)")
    fig = _attention_figure(tokens, maps, title)
    if out is None:
        stem = os.path.join(run_dir, f"attention_phase{int(phase)}_len{length}_ex{example_idx}")
        paths = [stem + ".png", stem + ".pdf"]
    else:
        paths = [out]
    for p in paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=200)
    return paths


@torch.no_grad()
def successor_attention_score(model, inputs: np.ndarray, device=None, *, layer: int = -1,
                              include_sep: bool = True, batch_size: int = DEFAULT_BATCH,
                              max_attn_mb: float = DEFAULT_MAX_ATTN_MB, max_length=None) -> dict:
    """Per-head attention of layer `layer` (default: last = layer 2) from each prediction position
    to the INPUT copies of the token it must predict next (module docstring). inputs (N, n), with all
    rows the same length. Returns numpy (H,) arrays for "per_head", "per_head_argmax",
    "per_head_sep", "per_head_change" and "per_head_dup" (NaN where a group is empty), plus
    "uniform_baseline", "layer", "n_rows" and "n_queries".
    Raises IdOnlyError if n exceeds max_length or model.id_only_max_len (set by
    probes.load_run_model for eval.id_only runs)."""
    xs = np.asarray(inputs, dtype=np.int64)
    if xs.ndim != 2 or xs.shape[0] == 0 or xs.shape[1] == 0:
        raise ValueError(f"inputs must be a non-empty (N, n) array, got shape {xs.shape}")
    N, n = xs.shape
    check_id_only(n, max_length)
    check_id_only(n, getattr(model, "id_only_max_len", None))
    dev = _model_device(model, device)
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    qpos = torch.arange(n + 1, 2 * n + 1, device=dev)                  # (Q,) Q = n
    groups = ("sep", "change", "dup", "all")
    mass, hit, cnt, base = {}, {}, {g: 0 for g in groups}, {g: 0.0 for g in groups}
    layer_idx = None
    try:
        rows = chunk_rows(n, model, batch_size, max_attn_mb)
        for start in range(0, N, rows):
            xb = xs[start:start + rows]
            idx = torch.as_tensor(probe_inputs(xb), device=dev)
            _, internals = _forward(model, idx)
            attn_all = internals["attn"]
            layer_idx = layer % len(attn_all)
            att = attn_all[layer_idx][:, :, qpos, :]                       # (B, H, Q, T)
            B, H = att.shape[:2]
            x_t = torch.as_tensor(xb, device=dev)
            y = torch.as_tensor(sort_targets(xb), device=dev)              # target of query k is y[:, k]
            match = x_t[:, None, :] == y[:, :, None]                       # (B, Q, n): key j+1 holds it
            m = (att[..., 1:n + 1] * match[:, None].to(att.dtype)).sum(-1)  # (B, H, Q)
            am = att.argmax(-1)                                            # (B, H, Q) key index
            key_ok = torch.zeros(B, n, att.shape[-1], dtype=torch.bool, device=dev)
            key_ok[..., 1:n + 1] = match
            h_ok = key_ok[:, None].expand(B, H, n, -1).gather(-1, am[..., None])[..., 0]
            # Query-type masks over (B, Q)
            is_sep = torch.zeros(B, n, dtype=torch.bool, device=dev)
            is_sep[:, 0] = True
            chg = torch.zeros_like(is_sep)
            chg[:, 1:] = y[:, 1:] != y[:, :-1]
            masks = {"sep": is_sep, "change": chg, "dup": ~is_sep & ~chg,
                     "all": torch.ones_like(is_sep) if include_sep else ~is_sep}
            # Uniform-attention baseline: copies / visible keys (query k sees n+2+k keys)
            uni = match.sum(-1).to(torch.float64) / (qpos + 1).to(torch.float64)[None, :]
            for g, mk in masks.items():
                w = mk[:, None, :].to(att.dtype)                           # (B, 1, Q)
                mass[g] = mass.get(g, 0) + (m * w).sum((0, 2)).double().cpu()
                hit[g] = hit.get(g, 0) + (h_ok & mk[:, None, :]).sum((0, 2)).double().cpu()
                cnt[g] += int(mk.sum())
                base[g] += float(uni[mk].sum())
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    def mean(d, g):
        if cnt[g] == 0:
            return np.full(len(next(iter(mass.values()))), np.nan)
        return (d[g] / cnt[g]).numpy()

    return {"layer": int(layer_idx), "n_rows": int(N), "n_queries": int(cnt["all"]),
            "per_head": mean(mass, "all"), "per_head_argmax": mean(hit, "all"),
            "per_head_sep": mean(mass, "sep"), "per_head_change": mean(mass, "change"),
            "per_head_dup": mean(mass, "dup"),
            "uniform_baseline": base["all"] / cnt["all"] if cnt["all"] else float("nan")}


def to_jsonable(obj):
    """numpy arrays/scalars -> lists/floats, NaN -> None (strict JSON)."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description="Attention heatmaps and successor-attention scores.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--example", type=int, default=0)
    ap.add_argument("--out", default=None, help="default <run>/attention_phase{p}_len{L}_ex{i}.png/.pdf")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--succ-n", type=int, default=0, help="> 0: also score successor attention")
    ap.add_argument("--succ-out", default=None)
    args = ap.parse_args(argv)
    try:
        paths = plot_attention(args.run, args.phase, args.length, args.example, args.out,
                               args.device)
    except IdOnlyError as e:
        ap.error(str(e))
    for p in paths:
        print(f"[attention] wrote {p}")
    if args.succ_n > 0:
        model, _, dev = load_run_model(args.run, args.phase, args.device,
                                       min_len=2 * args.length + 2)
        res = successor_attention_score(model, uniform_set(args.length, args.succ_n, TEST_SEED), dev,
                                        max_length=model.id_only_max_len)
        res = {"length": args.length, "phase": args.phase, **to_jsonable(res)}
        out = args.succ_out or os.path.join(args.run, f"succ_attn_phase{args.phase}_len{args.length}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, allow_nan=False)
        print(f"[attention] successor attention (layer {res['layer'] + 1}) per head: "
              + " ".join("n/a" if v is None else f"{v:.3f}" for v in res["per_head"])
              + f"  | uniform baseline {res['uniform_baseline']:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
