"""Paper figures. Every function takes DataFrames from
src.analysis.collect plus an output path, saves BOTH <out>.pdf and <out>.png, closes the
figure and returns the list of written paths. It raises NoDataError (a ValueError) if the data it
needs is absent, so callers can skip that figure cleanly.

CLI (makes every figure whose data exists and prints a skip message for the rest):
    python -m src.analysis.plots --results results --out figures [--tag ""] [--dpi 300]

Figures (file stems written by the CLI; _p1/_p2 = Phase 1/2):
    fig1_length_curves_p{p}   plot_length_curves     exact match vs test length, one panel per PE
    fig2_order_vs_content_p{p} plot_twin_comparison  schedule vs shuffled twin, per-seed pairs
    fig3_learning_curves      plot_learning_curves   val_group A/B/C vs step + stage boundaries
    fig4_phase1_vs_phase2     plot_phase_comparison  bucket exact match before/after Phase 2
    fig5_error_types_p{p}     plot_error_types       stacked err_* fractions at OOD lengths
    fig6_probes_p{p}          plot_probe_accuracy    min after L1 / L2 attn, id+succ vs length,
                                                     min_chance (best constant) dashed grey
    fig7_rep_p{p}             plot_rep               rep(L, r) exact match vs r
    fig8_per_position_p{p}    plot_per_position      accuracy by output position, OOD lengths

Visual system (dataviz skill, adapted to static matplotlib figures on a white page):
  * One schedule -> colour mapping for every figure (SCHEDULE_COLORS), drawn from the skill's
    validated reference palette, slots 1-6 (validate_palette.js on #ffffff: worst adjacent
    CVD dE 9.1, normal-vision dE 19.6). Slots are assigned so that the comparisons the paper
    cares about get well-separated hues: R (blue) is far from every other hue, SL (aqua) vs
    LS (orange) is dE 9.2 CVD / 27.6 normal, the three SLr-* hues are >= 16 apart pairwise. The
    weak all-pairs combinations (LS vs SLr-*, SL vs SLr-mid) are covered by the secondary
    encoding: every schedule also has its own marker shape, and a legend is always shown.
  * A shuffled twin uses its BASE schedule's hue with a dashed line and hollow markers
    (SL-shuf == LS-shuf is drawn in SL's hue).
  * Error types (fig5) use the two unused palette slots (violet, red) plus two neutrals, so they
    never impersonate a schedule. Thin 1.4 pt lines, white marker rings, hairline solid grid,
    no top/right spines, TrueType fonts in the PDF (pdf.fonttype 42).
  * Light-hue contrast: aqua, yellow and magenta are below 3:1 on white. The relief channels
    are the legend + marker shapes, and the LaTeX tables from src.analysis.stats.

Data conventions: plotting functions filter `tag` (default "", the main grid; pass tag=None to
skip the filter), average to one value per run before computing seed statistics, show the mean
with a +-1 std seed band (band="ci" for a bootstrap CI, "minmax", or None) and faint per-seed
lines. With one seed there is no band. Missing PEs/phases/schedules are left out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from src.analysis.stats import (  # noqa: E402
    BUCKET_TITLES, MAX_TRAIN_LEN, aggregate_buckets, bootstrap_ci, order_schedules, twin_of)
from src.data import DEFAULT_GROUPS, SCHEDULES  # noqa: E402

DPI = 300  # PNG resolution (PDF is vector)
FULL_W = 5.5  # NeurIPS text width (inches)

# --- palette (dataviz reference palette, light mode) -----------------------------------------
SCHEDULE_COLORS = {
    "R": "#2a78d6",          # slot 1 blue
    "LS": "#eb6834",         # slot 2 orange
    "SL": "#1baf7a",         # slot 3 aqua
    "SLr-early": "#eda100",  # slot 4 yellow
    "SLr-mid": "#e87ba4",    # slot 5 magenta
    "SLr-late": "#008300",   # slot 6 green
}
_SPARE_COLORS = ["#6250d6", "#e34948"]  # slots 7-8: unknown schedules, then gray
SCHEDULE_MARKERS = {"R": "o", "SL": "s", "SLr-early": "^", "SLr-mid": "D", "SLr-late": "v",
                    "LS": "X"}
ERROR_COLORS = {"correct": "#52514e", "order_only": "#6250d6", "multiset_only": "#e34948",
                "both": "#c3c2b7"}
ERROR_LABELS = {"correct": "correct", "order_only": "order only (right values)",
                "multiset_only": "values only (sorted, wrong multiset)", "both": "both wrong"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SHADE, SURFACE = "#e1e0d9", "#c3c2b7", "#f0efec", "#ffffff"
PE_ORDER = ["sinusoidal", "none"]
PE_TITLES = {"sinusoidal": "Sinusoidal PE", "none": "NoPE"}
YLABELS = {"exact_match": "Exact match", "token_acc": "Token accuracy",
           "prefix_frac": "Prefix fraction", "edit_distance_norm": "Norm. edit distance",
           "min_acc": "Min after L1 attn", "min_acc_after_attn2": "Min after L2 attn",
           "id_succ_acc": "Id+succ after L2 attn", "min_chance": "Chance (best constant)",
           "acc": "Accuracy at position", "value": "Exact match (val)"}
BOUNDED = {"exact_match", "token_acc", "prefix_frac", "multiset_ok", "min_acc", "id_succ_acc",
           "min_acc_after_attn2", "min_chance", "acc", "value", "err_correct", "err_order_only",
           "err_multiset_only", "err_both"}
CHANCE_LS = (0, (3, 2))  # min_chance reference in fig6: neutral grey, dashed, no markers
NEAR_OOD = (21, 24)

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titlelocation": "left",
    "axes.titlepad": 4.0, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}


class NoDataError(ValueError):
    """The data a figure needs is not present (the CLI prints a skip message)."""


# ----------------------------------------------------------------------------------------------
# style helpers
# ----------------------------------------------------------------------------------------------
def schedule_color(schedule: str) -> str:
    base = str(schedule)[:-5] if str(schedule).endswith("-shuf") else str(schedule)
    if base in SCHEDULE_COLORS:
        return SCHEDULE_COLORS[base]
    # Unknown schedule: spare slots, by a stable hash of the name (never cycles into slots 1-6)
    return _SPARE_COLORS[sum(map(ord, base)) % len(_SPARE_COLORS)]


def schedule_style(schedule: str) -> dict:
    """Line/marker style for a schedule: base hue; twins dashed with hollow markers."""
    s = str(schedule)
    twin = s.endswith("-shuf")
    base = s[:-5] if twin else s
    color = schedule_color(s)
    return {"color": color, "ls": (0, (4, 2)) if twin else "-",
            "marker": SCHEDULE_MARKERS.get(base, "o"),
            "mfc": SURFACE if twin else color, "mec": color if twin else SURFACE,
            "mew": 0.9 if twin else 0.5, "label": s}


def _handle(schedule: str, ms: float = 4.0) -> Line2D:
    st = schedule_style(schedule)
    return Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"], ms=ms,
                  mfc=st["mfc"], mec=st["mec"], mew=st["mew"], lw=1.4, label=st["label"])


def _save(fig, out, dpi=None) -> list[Path]:
    out = Path(out)
    stem = out.with_suffix("") if out.suffix.lower() in (".pdf", ".png") else out
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.parent / (stem.name + ".pdf"), stem.parent / (stem.name + ".png")]
    try:
        fig.savefig(paths[0])
        fig.savefig(paths[1], dpi=dpi or DPI)
    finally:
        plt.close(fig)
    return paths


def _top_legend(fig, handles, ncol=None, y=1.0):
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                   ncol=ncol or min(len(handles), 6), frameon=False)


def _pct_axis(ax, metric, full=True):
    """Percent ticks for [0, 1] metrics; full=True also fixes the limits to 0-100%."""
    if metric in BOUNDED:
        if full:
            ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def _auto_ylim(ax, values, bounded=True, min_span=0.10, pad=0.10):
    """Zoom the y axis on `values`, but never to less than `min_span` (10 percentage points by
    default, so seed noise is not blown up), staying inside [0, 1] for bounded metrics."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, min_span)
    mid = (lo + hi) / 2
    lo, hi = mid - span / 2 - pad * span, mid + span / 2 + pad * span
    if bounded:  # keep inside [-1%, 101%] (no tick label above 100%)
        if lo < -0.01:
            hi, lo = hi + (-0.01 - lo), -0.01
        if hi > 1.01:
            lo, hi = max(-0.01, lo - (hi - 1.01)), 1.01
    ax.set_ylim(lo, hi)


def _label_grid(axes, pes, col_titles, ylabel):
    """Column titles on the top row only; the PE as a row label in each row's y label."""
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t)
    for i, pe in enumerate(pes):
        axes[i][0].set_ylabel(f"{PE_TITLES.get(pe, pe)}\n{ylabel}")


def _length_axis(ax, lengths, max_len=MAX_TRAIN_LEN, label=True, shade=True):
    """Near-OOD shading, train-max marker and readable ticks for a test-length x axis."""
    lengths = sorted({int(L) for L in lengths})
    if not lengths:
        return
    lo, hi = min(lengths), max(lengths)
    if shade and hi > max_len:
        ax.axvspan(NEAR_OOD[0] - 0.5, NEAR_OOD[1] + 0.5, color=SHADE, lw=0, zorder=0)
    if lo <= max_len <= hi or hi > max_len:
        ax.axvline(max_len, color=MUTED, lw=0.8, zorder=1)
    ticks = [t for t in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100) if lo <= t <= hi]
    ax.set_xticks(ticks or lengths)
    ax.set_xlim(lo - 1, hi + 1)
    if label:
        ax.set_xlabel("Test length")


def _annotate_length_axis(ax, max_len=MAX_TRAIN_LEN, hi=None):
    """Small labels for the train-max marker and the near-OOD band (top panel only). Nothing is
    drawn outside the axis range (e.g. an id_only run whose lengths all lie below max_len)."""
    lo_x, hi_x = ax.get_xlim()
    if not (lo_x < max_len - 1 and max_len < hi_x):
        return
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(max_len - 0.4, 0.03, "max train", transform=tr, ha="right", va="bottom",
            fontsize=5.5, color=MUTED, rotation=90)
    if hi is not None and hi > max_len and NEAR_OOD[1] < hi_x:
        ax.text((NEAR_OOD[0] + NEAR_OOD[1]) / 2, 0.03, "near-OOD", transform=tr, ha="center",
                va="bottom", fontsize=5.5, color=MUTED, rotation=90)


def _pes(df) -> list[str]:
    present = [str(p) for p in df["pe"].dropna().unique()]
    return [p for p in PE_ORDER if p in present] + sorted(p for p in present if p not in PE_ORDER)


def _filter(df, tag="", phase=None, kind=None, include_twins=True, schedules=None):
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.iloc[:0]
    d = df
    if tag is not None and "tag" in d.columns:
        d = d[d["tag"].fillna("") == tag]
    if phase is not None and "phase" in d.columns:
        d = d[d["phase"] == phase]
    if kind is not None and "kind" in d.columns:
        d = d[d["kind"] == kind]
    if not include_twins and "is_twin" in d.columns:
        d = d[~d["is_twin"].astype(bool)]
    if schedules is not None:
        d = d[d["schedule"].isin(list(schedules))]
    return d


def _grid(nrows, ncols, panel_w, panel_h, sharey=True, sharex=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False, sharey=sharey, sharex=sharex)
    return fig, axes


def _band(per_run: pd.DataFrame, x: str, y: str, band: str):
    """(xs, mean, lo, hi, count) per x from one-row-per-(run, x) data."""
    g = per_run.groupby(x)[y]
    s = g.agg(["mean", "std", "count", "min", "max"]).sort_index()
    xs = s.index.to_numpy(dtype=float)
    m = s["mean"].to_numpy(dtype=float)
    cnt = s["count"].to_numpy()
    if band == "std":
        sd = s["std"].to_numpy(dtype=float)
        lo, hi = m - sd, m + sd
    elif band == "minmax":
        lo, hi = s["min"].to_numpy(dtype=float), s["max"].to_numpy(dtype=float)
    elif band == "ci":
        cis = [bootstrap_ci(v.to_numpy(dtype=float), n_boot=2000) for _, v in g]
        lo, hi = np.array([c[0] for c in cis]), np.array([c[1] for c in cis])
    else:
        lo = hi = np.full_like(m, np.nan)
    lo = np.where(cnt >= 2, lo, np.nan)
    hi = np.where(cnt >= 2, hi, np.nan)
    return xs, m, lo, hi, cnt


def _plot_schedule(ax, sub: pd.DataFrame, x: str, y: str, schedule: str, band="std",
                   show_seeds=True, bounded=True, ms=3.2, markevery=None):
    """Mean line (+ seed band, + faint per-seed lines) of one schedule. One value per (run, x)."""
    if sub.empty:
        return
    st = schedule_style(schedule)
    per_run = sub.groupby(["run_id", x], dropna=False)[y].mean().reset_index()
    if show_seeds and per_run["run_id"].nunique() > 1:
        for _, g in per_run.groupby("run_id"):
            g = g.sort_values(x)
            ax.plot(g[x].to_numpy(float), g[y].to_numpy(float), color=st["color"], ls=st["ls"],
                    lw=0.5, alpha=0.3, zorder=2)
    xs, m, lo, hi, _ = _band(per_run, x, y, band)
    if band and np.isfinite(lo).any():
        if bounded:
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.12, lw=0, zorder=1)
    ax.plot(xs, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=ms, mfc=st["mfc"],
            mec=st["mec"], mew=st["mew"], zorder=3, markevery=markevery, label=st["label"])


def _schedules_in(df) -> list[str]:
    return order_schedules(df["schedule"].dropna().unique())


def _seed_note(df) -> str:
    n = df.groupby("schedule")["seed"].nunique() if len(df) else pd.Series(dtype=int)
    if n.empty:
        return ""
    lo, hi = int(n.min()), int(n.max())
    return f"{lo} seed{'s' if lo != 1 else ''}" if lo == hi else f"{lo}–{hi} seeds"


# ----------------------------------------------------------------------------------------------
# stage boundaries (from data.SCHEDULES; LS is SL's blocks reversed)
# ----------------------------------------------------------------------------------------------
def stage_blocks(schedule: str, steps: int) -> list[tuple[int, int, str]]:
    """[(start_step, end_step, groups)] per stage, using the same floor rule as data.py:
    stage i covers [floor(cum_{i-1}*steps), floor(cum_i*steps)), the last ends at `steps`.
    LS = SL's blocks in reverse order (e.g. at 20000 steps: 6667/6667/6666, groups C, B, A).
    R, twins and unknown names have a single block (no boundaries)."""
    from fractions import Fraction
    import math

    s, steps = str(schedule), int(steps)
    if s == "LS":
        sl = stage_blocks("SL", steps)
        out, start = [], 0
        for a, b, grp in reversed(sl):
            out.append((start, start + (b - a), grp))
            start += b - a
        return out
    if s.endswith("-shuf") or s not in SCHEDULES:
        return [(0, steps, "")]
    out, cum, start = [], Fraction(0), 0
    stages = SCHEDULES[s]
    for i, (frac, groups) in enumerate(stages):
        cum += Fraction(frac)
        end = steps if i == len(stages) - 1 else math.floor(cum * steps)
        out.append((start, end, groups))
        start = end
    return out


def stage_boundaries(schedule: str, steps: int) -> list[int]:
    """Steps at which a new stage starts (empty for R and twins)."""
    return [b for (_a, b, _g) in stage_blocks(schedule, steps)[:-1]]


# ----------------------------------------------------------------------------------------------
# 1. exact match vs test length
# ----------------------------------------------------------------------------------------------
def plot_length_curves(metrics_df, out, metric="exact_match", phase=1, tag="", schedules=None,
                       include_twins=False, band="std", show_seeds=True, dpi=None):
    """Main result: `metric` vs test length (uniform sets), one line per schedule (mean, seed
    band, faint per-seed lines), one panel per PE, near-OOD (21-24) shaded, marker at 20."""
    d = _filter(metrics_df, tag, phase, "uniform", include_twins, schedules)
    if d.empty:
        raise NoDataError(f"no uniform test sets for phase {phase} (tag={tag!r})")
    pes = _pes(d)
    with plt.rc_context(STYLE):
        fig, axes = _grid(1, len(pes), FULL_W / 2 if len(pes) > 1 else 3.6, 2.3)
        scheds = _schedules_in(d)
        for ax, pe in zip(axes[0], pes):
            dp = d[d["pe"] == pe]
            _length_axis(ax, dp["length"])
            for s in scheds:
                _plot_schedule(ax, dp[dp["schedule"] == s], "length", metric, s, band,
                               show_seeds, metric in BOUNDED)
            _pct_axis(ax, metric)
            ax.set_title(f"{PE_TITLES.get(pe, pe)}  ·  {_seed_note(dp)}")
        _annotate_length_axis(axes[0][0], hi=int(d["length"].max()))
        axes[0][0].set_ylabel(YLABELS.get(metric, metric) + (f" (Phase {phase})"
                                                             if phase != 1 else ""))
        fig.tight_layout(w_pad=1.0)
        _top_legend(fig, [_handle(s) for s in scheds])
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 2. order vs content: schedule vs shuffled twin
# ----------------------------------------------------------------------------------------------
def plot_twin_comparison(metrics_df, out, metric="exact_match", phase=1, tag="",
                         buckets=("near_ood", "far_ood"), show_reference=True, ylim="auto",
                         dpi=None):
    """Order vs content: each schedule vs its shuffled twin (LS vs SL-shuf) on per-run bucket
    means. Per-seed paired points joined by lines (filled = ordered, hollow = shuffled), short
    bars at the means, the mean paired difference (percentage points, with n) above each pair,
    and R's mean as a dotted line. Rows = PE, columns = buckets. ylim="auto" zooms each panel
    (min. 10-point span; panels then have different scales), "full" uses 0-100%."""
    d = _filter(metrics_df, tag, phase, "uniform")
    bdf = aggregate_buckets(d, metrics=[metric]) if len(d) else d
    if len(bdf):
        bdf = bdf[bdf["bucket"].isin(list(buckets))]
    if bdf is None or len(bdf) == 0:
        raise NoDataError(f"no {'/'.join(buckets)} results for phase {phase} (tag={tag!r})")

    def pairs_in(df):
        present = set(df["schedule"].unique())
        return [(b, twin_of(b)) for b in order_schedules(present)
                if twin_of(b) is not None and twin_of(b) in present]

    if not pairs_in(bdf):
        raise NoDataError("no schedule/shuffled-twin pairs present")
    pes = [pe for pe in _pes(bdf) if pairs_in(bdf[bdf["pe"] == pe])]
    if not pes:  # e.g. a schedule under one PE and its twin only under the other
        raise NoDataError("no PE has both a schedule and its shuffled twin")
    cols = [b for b in buckets if b in set(bdf["bucket"])]
    pct = metric in BOUNDED
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(cols), FULL_W / max(len(cols), 2), 2.1, sharey=False)
        for i, pe in enumerate(pes):
            pairs = pairs_in(bdf[bdf["pe"] == pe])
            for j, b in enumerate(cols):
                ax = axes[i][j]
                db = bdf[(bdf["pe"] == pe) & (bdf["bucket"] == b)]
                tr = blended_transform_factory(ax.transData, ax.transAxes)
                shown = []
                if show_reference and "R" in set(db["schedule"]):
                    ref = db.loc[db["schedule"] == "R", metric].mean()
                    ax.axhline(ref, color=SCHEDULE_COLORS["R"], lw=0.9, ls=(0, (1, 1.5)),
                               zorder=1)
                    shown.append(ref)
                for k, (base, twin) in enumerate(pairs):
                    sa = db[db["schedule"] == base].groupby("seed")[metric].mean()
                    sb = db[db["schedule"] == twin].groupby("seed")[metric].mean()
                    shown += list(sa.values) + list(sb.values)
                    seeds = sorted(set(sa.index) | set(sb.index))
                    jit = {sd: (r - (len(seeds) - 1) / 2) * 0.03 for r, sd in enumerate(seeds)}
                    xa, xb = k - 0.2, k + 0.2
                    ca, cb = schedule_style(base), schedule_style(twin)
                    both = sa.index.intersection(sb.index)
                    for sd in both:
                        ax.plot([xa + jit[sd], xb + jit[sd]], [sa[sd], sb[sd]],
                                color=ca["color"], lw=0.7, alpha=0.55, zorder=2)
                    for x0, v, c in ((xa, sa, ca), (xb, sb, cb)):
                        if not len(v):
                            continue
                        ax.hlines(v.mean(), x0 - 0.14, x0 + 0.14, color=c["color"], lw=1.6,
                                  alpha=0.9, zorder=2.5)
                        ax.plot([x0 + jit[sd] for sd in v.index], v.values, ls="none",
                                marker=c["marker"], ms=3.8, mfc=c["mfc"], mec=c["mec"],
                                mew=c["mew"], color=c["color"], zorder=3)
                    if len(both):
                        diff = float((sa[both] - sb[both]).mean())
                        val = round(diff * 100, 1) if pct else diff
                        txt = (f"{val:+.1f} pp" if val != 0 else "0.0 pp") if pct \
                            else f"{diff:+.3g}"
                        ax.text(k, 1.01, f"{txt}\n(n={len(both)})", transform=tr, ha="center",
                                va="bottom", fontsize=5.5, color=INK2)
                ax.set_xticks(range(len(pairs)))
                ax.set_xticklabels([f"{a}\nvs {'SL-shuf' if a == 'LS' else 'shuf'}"
                                    for a, _ in pairs], fontsize=6)
                ax.set_xlim(-0.6, len(pairs) - 0.4)
                ax.grid(axis="x", visible=False)
                _pct_axis(ax, metric, full=(ylim == "full"))
                if ylim != "full":
                    _auto_ylim(ax, shown, bounded=pct)
            if i == 0:
                for j, b in enumerate(cols):
                    axes[0][j].set_title(BUCKET_TITLES.get(b, b), pad=18)
            axes[i][0].set_ylabel(f"{PE_TITLES.get(pe, pe)}\n{YLABELS.get(metric, metric)}")
        fig.tight_layout(h_pad=1.4, w_pad=1.2)
        handles = [Line2D([], [], color=INK2, ls="none", marker="o", ms=4, mfc=INK2,
                          mec=SURFACE, label="ordered schedule (filled)"),
                   Line2D([], [], color=INK2, ls="none", marker="o", ms=4, mfc=SURFACE,
                          mec=INK2, label="shuffled twin (hollow)"),
                   Line2D([], [], color=INK2, lw=0.7, alpha=0.6, label="same seed"),
                   Line2D([], [], color=INK2, lw=1.6, label="mean")]
        if show_reference and "R" in set(bdf["schedule"]):
            handles.append(Line2D([], [], color=SCHEDULE_COLORS["R"], lw=0.9, ls=(0, (1, 1.5)),
                                  label="R mean"))
        _top_legend(fig, handles)
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 3. learning curves / forgetting
# ----------------------------------------------------------------------------------------------
def plot_learning_curves(curves_df, out, schedules=("R", "SL", "SLr-mid", "LS"),
                         groups=("A", "B", "C"), tag="", include_phase2=True, band="std",
                         show_seeds=False, which="val_group", dpi=None):
    """Training dynamics / forgetting: val_group exact match vs training step, one panel per
    group (columns) and PE (rows), one line per schedule (schedules=None: every base schedule
    present). Stage boundaries (data.SCHEDULES with the same floor rule; LS mirrored from SL)
    are dotted verticals in each schedule's colour; Phase 2 (if logged) is shaded."""
    d = _filter(curves_df, tag, None, None, schedules is not None, schedules)
    if len(d):
        d = d[(d["which"] == which) & (d["key"].isin([str(g) for g in groups]))]
        if not include_phase2:
            d = d[d["phase"] == 1]
    if d is None or len(d) == 0:
        raise NoDataError(f"no {which} learning curves for schedules {schedules} (tag={tag!r})")
    pes = _pes(d)
    keys = [str(g) for g in groups if str(g) in set(d["key"])]
    scheds = _schedules_in(d)
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(keys), FULL_W / max(len(keys), 2), 1.9, sharey=True,
                          sharex=True)
        for i, pe in enumerate(pes):
            dp = d[d["pe"] == pe]
            p1_end = dp.loc[dp["phase"] == 1, "step"].max()
            has_p2 = include_phase2 and (dp["phase"] == 2).any()
            for j, key in enumerate(keys):
                ax = axes[i][j]
                dk = dp[dp["key"] == key]
                if has_p2 and pd.notna(p1_end):
                    ax.axvspan(p1_end, dp["global_step"].max(), color=SHADE, lw=0, zorder=0)
                    if i == 0 and j == 0:
                        ax.text(p1_end, 0.02, " Phase 2", transform=blended_transform_factory(
                            ax.transData, ax.transAxes), fontsize=5.5, color=MUTED, va="bottom")
                bnds = {}  # 0.5%-of-phase bin -> [step, schedules with a boundary there]
                for s in scheds:
                    ds = dk[dk["schedule"] == s]
                    if ds.empty:
                        continue
                    steps1 = ds.loc[ds["phase"] == 1, "step"].max()
                    if pd.notna(steps1) and steps1 > 0:
                        for bnd in stage_boundaries(s, int(steps1)):
                            bnds.setdefault(round(200 * bnd / steps1), [bnd, set()])[1].add(s)
                    _plot_schedule(ax, ds, "global_step", "value", s, band, show_seeds, True,
                                   ms=2.2, markevery=max(1, ds["global_step"].nunique() // 12))
                # Boundaries shared by several schedules (e.g. SL / SLr-mid / LS at 1/3, 2/3) are
                # drawn once in neutral grey; a boundary unique to one schedule takes its hue.
                for bnd, ss in bnds.values():
                    col = schedule_color(next(iter(ss))) if len(ss) == 1 else MUTED
                    ax.axvline(bnd, color=col, lw=0.7, ls=(0, (1, 1.5)), alpha=0.9, zorder=1)
                _pct_axis(ax, "value")
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda v, _p: f"{v / 1000:g}k" if v else "0"))
                if i == len(pes) - 1:
                    ax.set_xlabel("Training step")
        titles = []
        for key in keys:
            lo, hi = DEFAULT_GROUPS.get(key, (None, None))
            titles.append(f"Group {key}" + (f" (lengths {lo}–{hi})" if lo is not None else ""))
        _label_grid(axes, pes, titles, "Val exact match")
        fig.tight_layout(h_pad=1.0, w_pad=0.8)
        handles = [_handle(s, 3) for s in scheds]
        handles.append(Line2D([], [], color=MUTED, lw=0.7, ls=(0, (1, 1.5)),
                              label="stage boundary"))
        _top_legend(fig, handles, ncol=min(len(handles), 6))
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 4. Phase 1 vs Phase 2
# ----------------------------------------------------------------------------------------------
def plot_phase_comparison(metrics_df, out, metric="exact_match",
                          buckets=("in_dist", "near_ood", "far_ood"), tag="",
                          include_twins=False, show_seeds=True, ylim="auto", dpi=None):
    """Lasting effects: slope chart per bucket (columns) and PE (rows) of each schedule's
    bucket mean after Phase 1 and after Phase 2 (mean line + faint per-seed lines). Do the
    curriculum advantages persist? ylim="auto" zooms each panel (min. 10-point span)."""
    d = _filter(metrics_df, tag, None, "uniform", include_twins)
    if d.empty or not (d["phase"] == 2).any():
        raise NoDataError("no Phase 2 results")
    bdf = aggregate_buckets(d, metrics=[metric])
    bdf = bdf[bdf["bucket"].isin(list(buckets))]
    if bdf.empty:
        raise NoDataError(f"no {'/'.join(buckets)} results")
    pes = _pes(bdf)
    cols = [b for b in buckets if b in set(bdf["bucket"])]
    scheds = _schedules_in(bdf)
    offs = {s: (k - (len(scheds) - 1) / 2) * 0.035 for k, s in enumerate(scheds)}
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(cols), FULL_W / max(len(cols), 2), 2.0, sharey=False)
        for i, pe in enumerate(pes):
            for j, b in enumerate(cols):
                ax = axes[i][j]
                db = bdf[(bdf["pe"] == pe) & (bdf["bucket"] == b)]
                for s in scheds:
                    ds = db[db["schedule"] == s].copy()
                    if ds.empty:
                        continue
                    ds["x"] = ds["phase"].astype(float) - 1 + offs[s]
                    _plot_schedule(ax, ds, "x", metric, s, band=None, show_seeds=show_seeds,
                                   bounded=metric in BOUNDED, ms=3.5)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["after Phase 1", "after Phase 2"])
                ax.set_xlim(-0.35, 1.35)
                ax.grid(axis="x", visible=False)
                _pct_axis(ax, metric, full=(ylim == "full"))
                if ylim != "full":
                    _auto_ylim(ax, db[metric], bounded=metric in BOUNDED)
        _label_grid(axes, pes, [BUCKET_TITLES.get(b, b) for b in cols],
                    YLABELS.get(metric, metric))
        fig.tight_layout(h_pad=1.0, w_pad=1.0)
        _top_legend(fig, [_handle(s) for s in scheds])
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 5. error types
# ----------------------------------------------------------------------------------------------
def plot_error_types(metrics_df, out, lengths=(22, 25, 30, 40, 50), phase=1, tag="",
                     include_twins=False, dpi=None):
    """Error-type partition (correct / order only / values only / both; seed means) as stacked
    bars per schedule, one panel per OOD length (columns) and PE (rows). Each bar has its
    schedule's marker underneath and its name as the tick label."""
    d = _filter(metrics_df, tag, phase, "uniform", include_twins)
    if len(d):
        d = d[d["length"].isin([int(L) for L in lengths])]
    if d.empty:
        raise NoDataError(f"no uniform sets at lengths {tuple(lengths)} for phase {phase}")
    pes = _pes(d)
    cols = sorted(int(L) for L in d["length"].unique())
    keys = ["correct", "order_only", "multiset_only", "both"]
    ecols = [f"err_{k}" for k in keys]
    n_max = max(len(_schedules_in(d[d["pe"] == pe])) for pe in pes)
    with plt.rc_context(STYLE):
        pw = max(0.95, 0.17 * n_max + 0.25)
        fig, axes = _grid(len(pes), len(cols), min(pw, FULL_W / len(cols)) if len(cols) > 3
                          else pw, 2.0, sharey=True)
        for i, pe in enumerate(pes):
            dp = d[d["pe"] == pe]
            scheds = _schedules_in(dp)  # per row: a PE may have fewer schedules
            for j, L in enumerate(cols):
                ax = axes[i][j]
                dl = dp[dp["length"] == L]
                per_run = dl.groupby(["schedule", "run_id"])[ecols].mean()
                means = per_run.groupby(level="schedule").mean()
                xs = np.arange(len(scheds))
                bottom = np.zeros(len(scheds))
                for k, col in zip(keys, ecols):
                    v = np.array([means.loc[s, col] if s in means.index else 0.0 for s in scheds])
                    v = np.nan_to_num(v)
                    ax.bar(xs, v, bottom=bottom, width=0.72, color=ERROR_COLORS[k],
                           edgecolor=SURFACE, linewidth=0.6, zorder=2)
                    bottom += v
                for x0, s in zip(xs, scheds):  # schedule identity marker under each bar
                    if s not in means.index:
                        continue
                    st = schedule_style(s)
                    ax.plot([x0], [-0.07], marker=st["marker"], ms=3.2, mfc=st["mfc"],
                            mec=st["mec"] if st["mfc"] == SURFACE else st["color"],
                            mew=0.8, ls="none", clip_on=False, zorder=3)
                ax.set_xticks(xs)
                ax.set_xticklabels(scheds, rotation=90, fontsize=5.8)
                ax.tick_params(axis="x", pad=8, length=0)
                ax.set_xlim(-0.6, max(len(scheds), 1) - 0.4)
                ax.grid(axis="x", visible=False)
                ax.set_ylim(0, 1.0)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        _label_grid(axes, pes, [f"L = {L}" for L in cols], "Fraction of lists")
        fig.tight_layout(h_pad=1.0, w_pad=0.4)
        handles = [Patch(color=ERROR_COLORS[k], label=ERROR_LABELS[k]) for k in keys]
        _top_legend(fig, handles, ncol=2)
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 6. probe accuracy vs length
# ----------------------------------------------------------------------------------------------
def plot_probe_accuracy(probes_df, out, metrics=("min_acc", "min_acc_after_attn2", "id_succ_acc"),
                        phase=1, tag="", readout=None, include_twins=False, band="std",
                        show_seeds=True, chance="min_chance", dpi=None):
    """A&G mechanism probes vs test length: min-finding at SEP after layer-1 attention (min_acc)
    and after layer-2 attention (min_acc_after_attn2), and identity+successor after layer-2
    attention (id_succ_acc). One line per schedule, columns = probe, rows = PE. On the min panels
    `chance` (min_chance: accuracy of the best constant guess of min(x) on the same rows; mean over
    runs per length) is a dashed grey reference: the most a readout collapsed to one constant
    token can score (min_acc at or below it is no evidence of a min readout).
    Columns whose data is absent (older probes files) are left out, as is the chance line.
    If several readouts exist, `readout` picks one (default: the first)."""
    d = _filter(probes_df, tag, phase, None, include_twins)
    metrics = [m for m in metrics if len(d) and m in d.columns and d[m].notna().any()]
    if d.empty or not metrics:
        raise NoDataError(f"no probe results for phase {phase} (tag={tag!r})")
    if "readout" in d.columns and d["readout"].nunique(dropna=False) > 1:
        readout = readout if readout is not None else sorted(map(str, d["readout"].unique()))[0]
        print(f"[plots] probes: several readouts; plotting readout={readout!r}")
    if readout is not None and "readout" in d.columns:
        d = d[d["readout"].astype(str) == str(readout)]
    pes = _pes(d)
    scheds = _schedules_in(d)
    has_chance = bool(chance) and chance in d.columns and d[chance].notna().any()
    drew_chance = False
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(metrics), FULL_W / max(len(metrics), 2), 2.0)
        for i, pe in enumerate(pes):
            dp = d[d["pe"] == pe]
            ref = None
            if has_chance:  # one value per (run, length), then the mean over runs
                ref = (dp.groupby(["run_id", "length"])[chance].mean().groupby(level="length")
                       .mean().dropna().sort_index())
            for j, m in enumerate(metrics):
                ax = axes[i][j]
                _length_axis(ax, dp["length"], label=(i == len(pes) - 1))
                # chance applies to plain argmax min readouts, not to the centred one
                if (ref is not None and len(ref) and str(m).startswith("min_acc")
                        and "centered" not in str(m)):
                    ax.plot(ref.index.to_numpy(float), ref.to_numpy(float), color=MUTED,
                            ls=CHANCE_LS, lw=0.9, zorder=1.5, gid=chance)
                    drew_chance = True
                for s in scheds:
                    _plot_schedule(ax, dp[dp["schedule"] == s], "length", m, s, band,
                                   show_seeds, True)
                _pct_axis(ax, m)
        _label_grid(axes, pes, [YLABELS.get(m, m) for m in metrics], "Probe accuracy")
        _annotate_length_axis(axes[0][0], hi=int(d["length"].max()))
        fig.tight_layout(h_pad=1.0, w_pad=1.0)
        handles = [_handle(s) for s in scheds]
        if drew_chance:
            handles.append(Line2D([], [], color=MUTED, ls=CHANCE_LS, lw=0.9,
                                  label="chance (best constant)"))
        _top_legend(fig, handles)
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 7. rep(i, r)
# ----------------------------------------------------------------------------------------------
def plot_rep(metrics_df, out, metric="exact_match", phase=1, tag="", include_twins=False,
             band="std", show_seeds=True, dpi=None):
    """A&G repetition test: `metric` vs repeat count r on rep(L, r) sets, one panel per rep
    length L (columns) and PE (rows), one line per schedule."""
    d = _filter(metrics_df, tag, phase, "rep", include_twins)
    if len(d):
        d = d[d["r"].notna()].copy()
        d["r_f"] = d["r"].astype(float)
    if d.empty:
        raise NoDataError(f"no rep(L, r) sets for phase {phase} (tag={tag!r})")
    pes = _pes(d)
    cols = sorted(int(L) for L in d["length"].unique())
    scheds = _schedules_in(d)
    rs = sorted(int(r) for r in d["r"].unique())
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(cols), FULL_W / max(len(cols), 2), 1.9, sharey=True)
        for i, pe in enumerate(pes):
            for j, L in enumerate(cols):
                ax = axes[i][j]
                dl = d[(d["pe"] == pe) & (d["length"] == L)]
                for s in scheds:
                    _plot_schedule(ax, dl[dl["schedule"] == s], "r_f", metric, s, band,
                                   show_seeds, metric in BOUNDED)
                ax.set_xticks(rs)
                ax.set_xlim(min(rs) - 0.3, max(rs) + 0.3)
                _pct_axis(ax, metric)
                if i == len(pes) - 1:
                    ax.set_xlabel("Repeats per value r")
        _label_grid(axes, pes, [f"rep(L={L})" + (", OOD" if L > MAX_TRAIN_LEN else "")
                                for L in cols], YLABELS.get(metric, metric))
        fig.tight_layout(h_pad=1.0, w_pad=0.8)
        _top_legend(fig, [_handle(s) for s in scheds])
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# 8. per-position accuracy
# ----------------------------------------------------------------------------------------------
def plot_per_position(pp_df, out, lengths=(25, 30, 40, 50), phase=1, tag="", kind="uniform",
                      include_twins=False, band="std", show_seeds=False, dpi=None):
    """Where does it break? Accuracy at each output position (1-based) for a few OOD lengths
    (columns) and PEs (rows), one line per schedule. The vertical line marks position 20, the
    longest output seen in training."""
    d = _filter(pp_df, tag, phase, kind, include_twins)
    if len(d):
        d = d[d["length"].isin([int(L) for L in lengths])]
    if d.empty:
        raise NoDataError(f"no per-position data at lengths {tuple(lengths)} for phase {phase}")
    pes = _pes(d)
    cols = sorted(int(L) for L in d["length"].unique())
    scheds = _schedules_in(d)
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(pes), len(cols), FULL_W / max(len(cols), 2), 1.9, sharey=True)
        for i, pe in enumerate(pes):
            for j, L in enumerate(cols):
                ax = axes[i][j]
                dl = d[(d["pe"] == pe) & (d["length"] == L)]
                if L > MAX_TRAIN_LEN:
                    ax.axvline(MAX_TRAIN_LEN, color=MUTED, lw=0.8, zorder=1)
                for s in scheds:
                    _plot_schedule(ax, dl[dl["schedule"] == s], "position", "acc", s, band,
                                   show_seeds, True, ms=2.0,
                                   markevery=max(1, L // 10))
                ax.set_xlim(0.5, L + 0.5)
                ax.set_xticks([t for t in (1, 10, 20, 30, 40, 50) if t <= L])
                _pct_axis(ax, "acc")
                if i == len(pes) - 1:
                    ax.set_xlabel("Output position")
        _label_grid(axes, pes, [f"L = {L}" for L in cols], "Token accuracy")
        fig.tight_layout(h_pad=1.0, w_pad=0.8)
        _top_legend(fig, [_handle(s, 3) for s in scheds])
        return _save(fig, out, dpi)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def make_all(results_dir, out_dir="figures", tag="", dpi=None) -> dict:
    """Make every figure whose data exists. Returns {figure stem: [paths] or 'skipped: why'}."""
    from src.analysis.collect import load_per_position, load_probes, load_runs

    out_dir = Path(out_dir)
    metrics_df, curves_df = load_runs(results_dir)
    probes_df = load_probes(results_dir)
    ood_sets = [f"len_{L}" for L in (*range(21, 25), 25, 30, 40, 50)]
    pp_df = load_per_position(results_dir, set_names=ood_sets)
    phases = sorted(int(p) for p in metrics_df["phase"].unique()) if len(metrics_df) else [1]
    jobs = []
    for p in phases:
        jobs += [
            (f"fig1_length_curves_p{p}", lambda p=p, o=None: plot_length_curves(
                metrics_df, o, phase=p, tag=tag, dpi=dpi)),
            (f"fig2_order_vs_content_p{p}", lambda p=p, o=None: plot_twin_comparison(
                metrics_df, o, phase=p, tag=tag, dpi=dpi)),
        ]
    jobs += [("fig3_learning_curves", lambda o=None: plot_learning_curves(
                curves_df, o, tag=tag, dpi=dpi)),
             ("fig4_phase1_vs_phase2", lambda o=None: plot_phase_comparison(
                 metrics_df, o, tag=tag, dpi=dpi))]
    for p in phases:
        jobs += [
            (f"fig5_error_types_p{p}", lambda p=p, o=None: plot_error_types(
                metrics_df, o, phase=p, tag=tag, dpi=dpi)),
            (f"fig6_probes_p{p}", lambda p=p, o=None: plot_probe_accuracy(
                probes_df, o, phase=p, tag=tag, dpi=dpi)),
            (f"fig7_rep_p{p}", lambda p=p, o=None: plot_rep(
                metrics_df, o, phase=p, tag=tag, dpi=dpi)),
            (f"fig8_per_position_p{p}", lambda p=p, o=None: plot_per_position(
                pp_df, o, phase=p, tag=tag, dpi=dpi)),
        ]
    done = {}
    for name, fn in jobs:
        try:
            paths = fn(o=out_dir / name)
            done[name] = paths
            print(f"[plots] wrote {paths[0]} (+ .png)")
        except NoDataError as e:
            done[name] = f"skipped: {e}"
            print(f"[plots] skip {name}: {e}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make the paper figures from finished runs.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="", help='only runs with this tag (default "": main grid)')
    ap.add_argument("--dpi", type=int, default=None, help=f"PNG dpi (default {DPI})")
    args = ap.parse_args(argv)
    return make_all(args.results, args.out, tag=args.tag, dpi=args.dpi)


if __name__ == "__main__":
    main(sys.argv[1:])
