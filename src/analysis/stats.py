"""Seed statistics, paired comparisons, length buckets and LaTeX tables. See docs/CODE_SPEC.md s.6.

The unit of replication is the RUN (one seed of one condition). Every function first reduces the
data to one value per run (or per seed within a condition) before computing anything over seeds,
so several rows per run (e.g. several lengths) are never treated as independent samples.

Main functions:
    summarize(df, metric, by)                      mean, std, n, bootstrap CI of the mean over runs
    paired_compare(df, cond_a, cond_b, metric)     paired-by-seed difference a - b, CI, Wilcoxon p
    compare_to_reference(df, metric, "R")          every condition vs R
    compare_twins(df, metric)                      every schedule vs its shuffled twin
    aggregate_buckets(metrics_df)                  per-run in-dist / near-OOD / far-OOD means
    latex_table(bucket_df, metric)                 booktabs schedule x bucket table of mean +- std

CLI (minimal; writes LaTeX tables and paired-comparison CSVs):
    python -m src.analysis.stats --results results --out tables [--tag ""] [--metric exact_match]

Resolved details:
  * Bootstrap: percentile bootstrap of the mean over runs, n_boot resamples, with a FRESH
    np.random.default_rng(seed) for every group, so a group's CI does not depend on which other
    groups are present or on call order. With n < 2 the std and CI are NaN (one seed has no
    spread; a bootstrap CI from it would be a misleading zero-width interval).
  * Wilcoxon: zero differences (|d| <= 1e-12) are dropped first (zero_method "wilcox"). The
    remaining n_nonzero differences are rounded to 12 decimals, so |differences| that are equal
    up to float noise (e.g. 0.021 - 0.018 vs 0.563 - 0.560) count as ties, and the p-value comes
    from scipy's permutation null (method=scipy.stats.PermutationMethod) over ALL 2**n_nonzero
    sign flips of the (mid-)ranks: exact even with tied |differences|. scipy's "exact" table
    assumes no ties and can be off either way: for d = [.003, -.001, .001, .003, .001] it gives
    two-sided p = 0.1875, while the true sign-flip p is 0.25.
    Above EXACT_MAX_N = 12 non-zero differences (4096 sign flips; the enumeration cost grows
    steeply beyond that, ~0.2 s at 12 vs ~2 s at 13+): scipy's "exact" table when there are no
    ties and at most 50 differences (exact then), else scipy's tie-corrected normal
    approximation ("approx"), which is adequate at that size. One non-zero difference:
    scipy's "exact" method (no ties possible; two-sided p = 1). All differences zero ->
    statistic 0, p = 1.0. n_nonzero is reported because dropping zeros raises the p floor
    2 / 2**n_nonzero (ties do not change that floor).
  * Buckets use only kind == "uniform" sets (rep sets have lengths 10-40 too and must not leak
    into far-OOD). A bucket value is the unweighted mean over the lengths present (every uniform
    set has the same number of rows). n_lengths is stored, and a warning is issued if runs being
    compared have different numbers of lengths in the same bucket.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon

from src.data import GRID_CONDITIONS

META_COLS = ["run_id", "pe", "schedule", "base_schedule", "is_twin", "seed", "tag"]
ERR_COLS = ["err_correct", "err_order_only", "err_multiset_only", "err_both"]

# Length buckets (inclusive lists of test lengths). Max train length is 20.
MAX_TRAIN_LEN = 20
BUCKETS = {
    "in_dist": tuple(range(5, 21)),
    "near_ood": (21, 22, 23, 24),
    "far_ood": (25, 30, 40, 50),
}
BUCKET_ORDER = list(BUCKETS)
BUCKET_TITLES = {"in_dist": "In-dist (5–20)", "near_ood": "Near-OOD (21–24)",
                 "far_ood": "Far-OOD (25–50)"}
BUCKET_LATEX = {"in_dist": "In-dist (5--20)", "near_ood": "Near-OOD (21--24)",
                "far_ood": "Far-OOD (25--50)"}
DEFAULT_BUCKET_METRICS = ("exact_match", "token_acc", "prefix_frac", "edit_distance_norm",
                          "inversions", "multiset_diff", *ERR_COLS)
METRIC_LATEX = {"exact_match": "Exact match", "token_acc": "Token accuracy",
                "prefix_frac": "Prefix fraction", "edit_distance_norm": "Normalised edit distance",
                "min_acc": "Min-finding probe accuracy",
                "id_succ_acc": "Identity+successor probe accuracy"}
EXACT_MAX_N = 12  # Wilcoxon: enumerate all 2**n sign flips up to this many non-zero differences
LOWER_IS_BETTER = {"edit_distance", "edit_distance_norm", "inversions", "adjacent_inversions",
                   "multiset_diff", "err_order_only", "err_multiset_only", "err_both"}


# ----------------------------------------------------------------------------------------------
# schedule helpers
# ----------------------------------------------------------------------------------------------
def schedule_sort_key(schedule: str):
    """Grid order (R, SL, SLr-early, SLr-mid, SLr-late, LS, then twins); unknown names last."""
    s = str(schedule)
    return (GRID_CONDITIONS.index(s), s) if s in GRID_CONDITIONS else (len(GRID_CONDITIONS), s)


def order_schedules(schedules) -> list[str]:
    return sorted({str(s) for s in schedules}, key=schedule_sort_key)


def twin_of(schedule: str) -> str | None:
    """Shuffled twin of a base schedule. R is its own twin (None); LS's twin is SL-shuf, because
    SL and LS consume the same multiset of examples (SL-shuf == LS-shuf)."""
    s = str(schedule)
    if s == "R" or s.endswith("-shuf"):
        return None
    if s == "LS":
        return "SL-shuf"
    return s + "-shuf"


# ----------------------------------------------------------------------------------------------
# basic statistics
# ----------------------------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0,
                 stat=np.mean) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` (default: the mean). NaNs are dropped. Returns
    (nan, nan) when fewer than 2 values remain. Uses a fresh default_rng(seed) on every call."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    boots = stat(x[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def _per_unit(df: pd.DataFrame, metric: str, keys: list[str], unit: str | None) -> pd.DataFrame:
    """Reduce to one value per (keys, unit) by averaging (a no-op if already one row each)."""
    if unit is None or unit not in df.columns or unit in keys:
        return df[keys + [metric]]
    return df.groupby(keys + [unit], dropna=False, sort=False)[metric].mean().reset_index()


def summarize(df: pd.DataFrame, metric: str, by, unit: str | None = "run_id",
              n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Mean, std (ddof=1), n and a `ci` bootstrap CI of the mean over runs, per group of `by`.

    Rows are first averaged within each (by, unit) cell, so n counts RUNS (seeds), not rows.
    Pass unit=None if df already has exactly one row per seed and no unit column.
    Returns columns by + [mean, std, n, ci_lo, ci_hi]; std/CI are NaN when n < 2."""
    by = [by] if isinstance(by, str) else list(by)
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} not in DataFrame")
    cols = by + ["mean", "std", "n", "ci_lo", "ci_hi"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    per = _per_unit(df, metric, by, unit)
    out = []
    for keys, g in per.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        v = g[metric].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        n = len(v)
        lo, hi = bootstrap_ci(v, n_boot=n_boot, ci=ci, seed=seed)
        out.append(dict(zip(by, keys), mean=float(v.mean()) if n else np.nan,
                        std=float(v.std(ddof=1)) if n >= 2 else np.nan, n=n, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out, columns=cols)


def _wilcoxon_method(nz: np.ndarray):
    n = len(nz)
    if n < 2:  # permutation_test needs >= 2 values; one value cannot tie anyway
        return "exact"
    if n <= EXACT_MAX_N:  # n_resamples >= 2**n: scipy enumerates every sign flip
        return PermutationMethod(n_resamples=2 ** n)
    if n <= 50 and len(np.unique(np.abs(nz))) == n:  # no ties: scipy's exact table is exact
        return "exact"
    return "approx"


def wilcoxon_signed_rank(diffs, alternative: str = "two-sided") -> tuple[float, float, int]:
    """(statistic, p, n_nonzero) for paired differences. Zeros (|d| <= 1e-12) are dropped, the
    rest rounded to 12 decimals, and p comes from the exact permutation (sign-flip) null, which
    stays exact under tied |differences| (above EXACT_MAX_N: scipy's exact table if untied,
    else the normal approximation; see the module docstring). The statistic is scipy's:
    min(W+, W-) for two-sided, W+ otherwise. All zero -> (0.0, 1.0, 0); no data -> (nan, nan, 0)."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan, 0)
    nz = np.round(d[np.abs(d) > 1e-12], 12)
    if len(nz) == 0:
        return (0.0, 1.0, 0)
    method = _wilcoxon_method(nz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = wilcoxon(nz, zero_method="wilcox", alternative=alternative, method=method)
            return (float(res.statistic), float(res.pvalue), len(nz))
        except ValueError:
            return (np.nan, np.nan, len(nz))


def _default_within(df: pd.DataFrame) -> list[str]:
    cand = ["pe", "tag", "phase", "bucket", "kind", "set_name", "length", "r"]
    return [c for c in cand if c in df.columns]


PAIRED_COLS = ["cond_a", "cond_b", "metric", "n", "n_nonzero", "seeds", "mean_a", "mean_b",
               "mean_diff", "std_diff", "ci_lo", "ci_hi", "dz", "wilcoxon_stat", "p_value",
               "p_min_possible"]


def paired_compare(df: pd.DataFrame, cond_a: str, cond_b: str, metric: str,
                   cond_col: str = "schedule", pair_on: str = "seed", within=None,
                   n_boot: int = 10_000, ci: float = 0.95, seed: int = 0,
                   alternative: str = "two-sided") -> pd.DataFrame:
    """Paired comparison of condition a vs condition b over seeds: difference = a - b.

    Runs are paired by `pair_on` (the seed) inside each group of `within` (default: whichever of
    pe, tag, phase, bucket, kind, set_name, length, r are columns of df), so it works on
    metrics_df (per eval set) and on aggregate_buckets() output (per bucket) alike. Only seeds
    present for BOTH conditions in a group are used; groups with no such seed are omitted.

    Returns one row per group: within cols + cond_a, cond_b, metric, n (paired seeds),
    n_nonzero, seeds, mean_a, mean_b, mean_diff, std_diff, ci_lo/ci_hi (percentile bootstrap CI
    of the mean difference), dz (Cohen's d_z = mean_diff / std_diff), wilcoxon_stat, p_value
    (Wilcoxon signed-rank, exact permutation p also under ties, zeros dropped; p = 1 if every
    difference is 0) and
    p_min_possible (= 2 / 2**n_nonzero, the smallest two-sided p the test can produce).

    IMPORTANT, small n: with n = 5 seeds the smallest possible two-sided Wilcoxon p-value is
    2/2**5 = 0.0625, so p < 0.05 is IMPOSSIBLE at n = 5 (and any zero difference raises the
    floor further: 0.125 at n_nonzero = 4). Report the mean paired difference, its bootstrap CI
    and the effect size (and plot the per-seed pairs); treat the p-value as secondary evidence.
    """
    within = _default_within(df) if within is None else list(within)
    for c in [cond_col, pair_on, metric] + within:
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in DataFrame")
    sub = df[df[cond_col].isin([cond_a, cond_b])]
    rows = []
    groups = sub.groupby(within, dropna=False, sort=True) if within else [((), sub)]
    for keys, g in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        per = g.groupby([cond_col, pair_on], dropna=False)[metric]
        if (per.size() > 1).any():
            warnings.warn(f"paired_compare: several rows per ({cond_col}, {pair_on}) in group "
                          f"{dict(zip(within, keys))}; averaging them (filter tag/phase first?)")
        means = per.mean()
        try:
            a = means.xs(cond_a, level=0)
            b = means.xs(cond_b, level=0)
        except KeyError:
            continue
        pair = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
        if pair.empty:
            continue
        pair = pair.sort_index()
        d = (pair["a"] - pair["b"]).to_numpy(dtype=float)
        n = len(d)
        stat, p, n_nz = wilcoxon_signed_rank(d, alternative=alternative)
        std = float(d.std(ddof=1)) if n >= 2 else np.nan
        lo, hi = bootstrap_ci(d, n_boot=n_boot, ci=ci, seed=seed)
        if n >= 2 and np.all(np.abs(d - d[0]) <= 1e-12):  # constant differences: no spread
            lo = hi = float(d[0])
        rows.append(dict(zip(within, keys), cond_a=cond_a, cond_b=cond_b, metric=metric, n=n,
                         n_nonzero=n_nz, seeds=" ".join(str(s) for s in pair.index),
                         mean_a=float(pair["a"].mean()), mean_b=float(pair["b"].mean()),
                         mean_diff=float(d.mean()), std_diff=std, ci_lo=lo, ci_hi=hi,
                         dz=float(d.mean() / std) if n >= 2 and std > 1e-12 else np.nan,
                         wilcoxon_stat=stat, p_value=p,
                         p_min_possible=min(1.0, 2.0 / 2 ** n_nz) if n_nz else np.nan))
    return pd.DataFrame(rows, columns=within + PAIRED_COLS)


def compare_to_reference(df: pd.DataFrame, metric: str, reference: str = "R", conditions=None,
                         cond_col: str = "schedule", **kw) -> pd.DataFrame:
    """paired_compare(cond, reference) for every condition present (difference = cond - ref)."""
    present = order_schedules(df[cond_col].dropna().unique())
    conds = present if conditions is None else [c for c in conditions if c in present]
    parts = [paired_compare(df, c, reference, metric, cond_col=cond_col, **kw)
             for c in conds if c != reference]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else paired_compare(
        df.iloc[:0], reference, reference, metric, cond_col=cond_col, **kw)


def compare_twins(df: pd.DataFrame, metric: str, cond_col: str = "schedule",
                  **kw) -> pd.DataFrame:
    """paired_compare(base, twin_of(base)) for every base schedule whose twin is present
    (difference = ordered - shuffled, so > 0 means the ORDER helps). LS is paired with SL-shuf."""
    present = set(df[cond_col].dropna().unique())
    parts = []
    for base in order_schedules(present):
        twin = twin_of(base)
        if twin is not None and twin in present:
            parts.append(paired_compare(df, base, twin, metric, cond_col=cond_col, **kw))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else paired_compare(
        df.iloc[:0], "SL", "SL-shuf", metric, cond_col=cond_col, **kw)


# ----------------------------------------------------------------------------------------------
# length buckets
# ----------------------------------------------------------------------------------------------
def bucket_of(length, buckets: dict = BUCKETS) -> str | None:
    """Bucket name for a test length (None if it is in no bucket)."""
    try:
        L = int(length)
    except (TypeError, ValueError):
        return None
    for name, lengths in buckets.items():
        if L in lengths:
            return name
    return None


def aggregate_buckets(metrics_df: pd.DataFrame, metrics=DEFAULT_BUCKET_METRICS,
                      buckets: dict = BUCKETS) -> pd.DataFrame:
    """Per-run bucket means over the uniform test sets.

    Returns one row per (run, phase, bucket): META_COLS + phase, bucket, n_lengths + metrics.
    Only kind == "uniform" rows are used. Missing buckets (e.g. id_only runs have no OOD
    lengths) are simply absent."""
    metrics = [m for m in metrics if m in metrics_df.columns]
    keys = [c for c in META_COLS if c in metrics_df.columns] + ["phase"]
    cols = keys + ["bucket", "n_lengths"] + metrics
    if metrics_df.empty:
        return pd.DataFrame(columns=cols)
    df = metrics_df[metrics_df["kind"] == "uniform"].copy()
    df["bucket"] = df["length"].map(lambda L: bucket_of(L, buckets))
    df = df[df["bucket"].notna()]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby(keys + ["bucket"], dropna=False, sort=False)
    out = g[metrics].mean().reset_index()
    out.insert(len(keys) + 1, "n_lengths", g["length"].nunique().to_numpy())
    # Comparability check: same number of lengths per bucket across runs of a phase?
    for (ph, b), gg in out.groupby(["phase", "bucket"]):
        if gg["n_lengths"].nunique() > 1:
            warnings.warn(f"aggregate_buckets: bucket {b!r} (phase {ph}) has a different number "
                          f"of lengths across runs ({sorted(gg['n_lengths'].unique())}); "
                          "bucket means are not directly comparable")
    out["_b"] = out["bucket"].map(lambda b: list(buckets).index(b))
    out["_s"] = out["schedule"].map(schedule_sort_key) if "schedule" in out else 0
    out = out.sort_values(["pe", "_s", "tag", "seed", "phase", "_b"] if "pe" in out else ["_b"],
                          kind="stable").drop(columns=["_b", "_s"]).reset_index(drop=True)
    return out[cols]


# ----------------------------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------------------------
def _tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def latex_table(df: pd.DataFrame, metric: str = "exact_match", row: str = "schedule",
                col: str = "bucket", row_order=None, col_order=None, col_labels=None,
                scale: float = 100.0, decimals: int = 1, bold_best: str | None = "auto",
                caption: str | None = None, label: str | None = None,
                unit: str | None = "run_id", table_env: bool = True, context: str = "") -> str:
    """NeurIPS-ready booktabs table: rows = `row` values (schedules), columns = `col` values
    (buckets), cells = mean +- std over runs (x `scale`, e.g. percent). Cells with one run show the
    mean alone; missing cells show "--". The best mean per column is bold (bold_best "max",
    "min", None, or "auto" = "min" for lower-is-better metrics else "max"). A \\midrule separates
    base schedules from shuffled twins. df must hold ONE pe, phase and tag (filter first).
    Requires \\usepackage{booktabs} in the paper."""
    for c in ("pe", "phase", "tag"):
        if c in df.columns and df[c].nunique(dropna=False) > 1:
            raise ValueError(f"latex_table: df mixes several values of {c!r} "
                             f"({sorted(map(str, df[c].unique()))}); filter it first")
    s = summarize(df, metric, by=[row, col], unit=unit)
    if row_order is None:
        row_order = order_schedules(s[row].unique()) if row == "schedule" else sorted(
            s[row].unique(), key=str)
    if col_order is None:
        present = list(s[col].unique())
        col_order = ([b for b in BUCKET_ORDER if b in present] if col == "bucket" else [])
        col_order += sorted((c for c in present if c not in col_order), key=str)
    if col_labels is None:
        col_labels = {c: BUCKET_LATEX.get(c, _tex_escape(c)) for c in col_order}
    if bold_best == "auto":
        bold_best = "min" if metric in LOWER_IS_BETTER else "max"
    cell = {(r[row], r[col]): r for _, r in s.iterrows()}

    best = {}
    for c in col_order:
        vals = [round(cell[(r, c)]["mean"] * scale, decimals) for r in row_order
                if (r, c) in cell and not np.isnan(cell[(r, c)]["mean"])]
        if vals and bold_best in ("max", "min"):
            best[c] = max(vals) if bold_best == "max" else min(vals)

    def fmt(r, c):
        if (r, c) not in cell or np.isnan(cell[(r, c)]["mean"]):
            return "--"
        st = cell[(r, c)]
        m = round(st["mean"] * scale, decimals)
        ms = f"{m:.{decimals}f}"
        if c in best and m == best[c]:
            ms = rf"\mathbf{{{ms}}}"
        if st["n"] >= 2 and not np.isnan(st["std"]):
            return rf"${ms} \pm {st['std'] * scale:.{decimals}f}$"
        return f"${ms}$"

    head = "Schedule" if row == "schedule" else _tex_escape(row)
    lines = [r"\begin{tabular}{l" + "c" * len(col_order) + "}", r"\toprule",
             " & ".join([head] + [col_labels[c] for c in col_order]) + r" \\", r"\midrule"]
    prev_twin = None
    for r in row_order:
        twin = str(r).endswith("-shuf")
        if row == "schedule" and prev_twin is False and twin:
            lines.append(r"\midrule")
        prev_twin = twin
        lines.append(" & ".join([_tex_escape(r)] + [fmt(r, c) for c in col_order]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if not table_env:
        return body + "\n"
    ns = sorted(set(int(n) for n in s["n"]))
    seeds = (f"{ns[0]} seed" + ("s" if ns[0] != 1 else "")) if len(ns) == 1 else \
        f"{ns[0]}--{ns[-1]} seeds"
    if caption is None:
        what = METRIC_LATEX.get(metric, _tex_escape(metric))
        pct = r" (\%)" if scale == 100 else ""
        caption = (f"{what}{pct} by schedule and test-length bucket: mean $\\pm$ std over "
                   f"{seeds}{context} (bold: best mean per column).")
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}"]
    if label:
        out.append(rf"\label{{{label}}}")
    out += [body, r"\end{table}"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main(argv=None):
    from src.analysis.collect import load_runs  # lazy: keeps this module importable alone

    ap = argparse.ArgumentParser(description="Bucket tables and paired comparisons.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--tag", default="", help='only runs with this tag (default "": main grid)')
    ap.add_argument("--metric", default="exact_match")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_df, _ = load_runs(args.results)
    metrics_df = metrics_df[metrics_df["tag"] == args.tag]
    if metrics_df.empty:
        print(f"[stats] no finished runs with tag {args.tag!r} in {args.results}; nothing to do")
        return
    bdf = aggregate_buckets(metrics_df)
    bdf.to_csv(out / "buckets_per_run.csv", index=False)
    for (pe, phase), g in bdf.groupby(["pe", "phase"]):
        path = out / f"table_{args.metric}_{pe}_phase{phase}.tex"
        ctx = f", {'sinusoidal PE' if pe == 'sinusoidal' else 'NoPE'}, end of Phase {phase}"
        path.write_text(latex_table(g, args.metric, label=f"tab:{args.metric}_{pe}_p{phase}",
                                    context=ctx),
                        encoding="utf-8")
        print(f"[stats] wrote {path}")
    vs_r = compare_to_reference(bdf, args.metric, "R")
    tw = compare_twins(bdf, args.metric)
    vs_r.to_csv(out / f"paired_vs_R_{args.metric}.csv", index=False)
    tw.to_csv(out / f"paired_twins_{args.metric}.csv", index=False)
    print(f"[stats] wrote {out / f'paired_vs_R_{args.metric}.csv'} ({len(vs_r)} rows), "
          f"{out / f'paired_twins_{args.metric}.csv'} ({len(tw)} rows)")
    print("[stats] note: with n=5 seeds the smallest two-sided Wilcoxon p is 0.0625; "
          "read the CIs and effect sizes first.")


if __name__ == "__main__":
    main(sys.argv[1:])
