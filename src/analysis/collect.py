"""Collect per-run outputs into tidy DataFrames. See docs/CODE_SPEC.md sections 4-6.

CLI:
    python -m src.analysis.collect --results results [--out DIR]
writes <out>/metrics.csv, <out>/curves.csv and (if any probe files exist) <out>/probes.csv.
<out> defaults to --results.

What counts as a run: an IMMEDIATE subdirectory of the results dir that contains `done.json`
(the sweep's completion marker). Unfinished runs (no done.json) are skipped silently; files a
finished run is missing (e.g. eval_phase2.json when phase2.steps == 0) are simply absent.

Run metadata comes from the run's `config.yaml`, never from parsing the run_id string:
    run_id  <- config["run_id"] (fallback: the directory name)
    pe      <- config["model"]["pe"]
    schedule<- config["data"]["schedule"]
    seed    <- config["seed"]
    tag     <- config["tag"] ("" when absent or null)
    base_schedule = schedule without a trailing "-shuf"; is_twin = schedule ends with "-shuf".
A run's identity is (pe, schedule, seed, tag). Several tags (e.g. the lr tuning grid) can share
a results dir, so analyses should filter on `tag` (the plots/stats CLIs default to tag == "").

Resolved details (not fixed by the spec):
  * Only the exact file names `eval_phase1.json`, `eval_phase2.json`, `probes_phase1.json`,
    `probes_phase2.json` are read. In particular `eval_phase{p}_cli.json` (written by
    `python -m src.evaluate`) is ignored, so rows are never double counted.
  * metrics_df column `r` is a nullable Int64 (<NA> for uniform sets). In addition to the spec
    columns it carries the per-(run, phase) eval-file fields `steps`, `tokens_nonpad`,
    `tokens_padded`, `wall_time_s` (handy for the "tokens processed" appendix table).
  * curves_df adds two convenience columns: `length` (float; the numeric key for which in
    {"val", "curve_ood"}, NaN otherwise) and `global_step` (= step in Phase 1, phase1_steps + step
    in Phase 2, where phase1_steps is config train.steps, else done.json phase1_steps, else the
    largest Phase-1 step seen in the log). A scalar `val_token_acc` gets key "all"; if it is a
    dict it is expanded per key like `val`. Duplicate records for the same (run, phase, step)
    (e.g. a periodic eval that coincides with the end-of-phase eval) are de-duplicated,
    keeping the LAST one in the file.
  * Per-position rows use a 1-based `position` (position j = the j-th sorted output y_j) and
    `rel_position` = position / length.
  * Unparsable JSONL lines (e.g. a truncated last line) are skipped with a warning.
  * After a CSV round trip, `tag` == "" comes back as NaN and `r` as float; use read_csv() below,
    which restores both.
  * Every table (metrics, curves, probes, losses, per-position) carries an `experiment_hash`
    column right after META_COLS, copied from done.json (None when done.json has none, i.e. runs
    made by older code). Runs meant to be compared share it (src.train.experiment_hash:
    the config without seed, tag, schedule and pe). load_runs warns, listing the run ids, when
    finished runs with the same (pe, tag) have more than one distinct value; a missing hash
    counts as a value of its own, so old runs mixed with new ones are flagged too.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

META_COLS = ["run_id", "pe", "schedule", "base_schedule", "is_twin", "seed", "tag"]
HASH_COL = "experiment_hash"
RUN_COLS = META_COLS + [HASH_COL]  # per-run columns at the front of every table
SET_COLS = ["phase", "set_name", "kind", "length", "r"]
# Scalar metrics in schema order (any other scalar found in a metric dict is appended after them).
SCALAR_METRICS = ["n", "exact_match", "token_acc", "prefix_len", "prefix_frac", "inversions",
                  "adjacent_inversions", "multiset_diff", "multiset_ok", "edit_distance",
                  "edit_distance_norm"]
ERROR_TYPES = ["correct", "order_only", "multiset_only", "both"]
ERR_COLS = [f"err_{k}" for k in ERROR_TYPES]
EVAL_FILE_FIELDS = ["steps", "tokens_nonpad", "tokens_padded", "wall_time_s"]
DROPPED_KEYS = {"per_position_acc", "samples", "error_types", "kind", "length", "r"}
CURVE_WHICH = ["val", "val_group", "curve_ood", "val_token_acc"]
CURVE_COLS = RUN_COLS + ["phase", "step", "global_step", "which", "key", "length", "value"]
LOSS_FIELDS = ["loss", "lr", "tokens_nonpad", "tokens_padded", "examples"]
PROBE_BASE_COLS = RUN_COLS + ["phase", "readout", "length"]
PHASES = (1, 2)


# ----------------------------------------------------------------------------------------------
# run discovery and metadata
# ----------------------------------------------------------------------------------------------
def base_schedule_of(schedule: str) -> str:
    """'SLr-mid-shuf' -> 'SLr-mid'; base names are returned unchanged."""
    s = str(schedule)
    return s[: -len("-shuf")] if s.endswith("-shuf") else s


def is_twin_schedule(schedule: str) -> bool:
    return str(schedule).endswith("-shuf")


def find_run_dirs(results_dir) -> list[Path]:
    """Immediate subdirectories of results_dir that contain done.json, sorted by name."""
    root = Path(results_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "done.json").is_file())


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"could not read {path}: {e}")
        return None


def read_config(run_dir) -> dict:
    with open(Path(run_dir) / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_meta(run_dir, cfg: dict | None = None) -> dict:
    """The metadata columns for one run, read from its config.yaml (see module docstring)."""
    run_dir = Path(run_dir)
    cfg = read_config(run_dir) if cfg is None else cfg
    schedule = str((cfg.get("data") or {}).get("schedule"))
    tag = cfg.get("tag")
    return {
        "run_id": str(cfg.get("run_id") or run_dir.name),
        "pe": str((cfg.get("model") or {}).get("pe")),
        "schedule": schedule,
        "base_schedule": base_schedule_of(schedule),
        "is_twin": is_twin_schedule(schedule),
        "seed": int(cfg.get("seed", 0)),
        "tag": "" if tag is None else str(tag),
    }


def _phase1_steps(cfg: dict, done: dict | None):
    steps = (cfg.get("train") or {}).get("steps")
    if steps is None and done:
        steps = done.get("phase1_steps")
    return None if steps is None else int(steps)


def complete_seed_filter(metas) -> set:
    """run_ids to DROP so that every (pe, tag) group keeps only its complete seeds.

    A seed is complete when it has a finished run for every schedule seen in that (pe, tag) group
    (the union over all seeds). Without this, a half-finished extra seed would enter some
    conditions but not others, giving unequal n and unpaired seeds. Set CS652_ALL_SEEDS=1 to
    disable the filter."""
    import os
    if os.environ.get("CS652_ALL_SEEDS") == "1":
        return set()
    groups = {}
    for m in metas:
        groups.setdefault((m["pe"], m["tag"]), []).append(m)
    drop = set()
    for (pe, tag), ms in groups.items():
        schedules = {m["schedule"] for m in ms}
        by_seed = {}
        for m in ms:
            by_seed.setdefault(m["seed"], set()).add(m["schedule"])
        partial = sorted(s for s, have in by_seed.items() if have != schedules)
        if partial:
            ids = sorted(m["run_id"] for m in ms if m["seed"] in partial)
            warnings.warn(f"collect: pe={pe!r}, tag={tag!r}: leaving out incomplete seed(s) "
                          f"{partial} ({len(ids)} runs) so every condition has the same seeds; "
                          f"set CS652_ALL_SEEDS=1 to keep them")
            drop.update(ids)
    return drop


def _finished_runs(results_dir):
    """Yield (run_dir, cfg, meta, done) for each finished run with a readable config.yaml.
    meta = run_meta(...) plus HASH_COL (done.json's experiment_hash, or None).
    Runs of incomplete seeds are left out (complete_seed_filter)."""
    found = []
    for run_dir in find_run_dirs(results_dir):
        try:
            cfg = read_config(run_dir)
        except (OSError, yaml.YAMLError) as e:
            warnings.warn(f"skipping {run_dir}: unreadable config.yaml ({e})")
            continue
        done = _read_json(run_dir / "done.json") or {}
        h = done.get("experiment_hash") if isinstance(done, dict) else None
        meta = {**run_meta(run_dir, cfg), HASH_COL: str(h) if h else None}
        found.append((run_dir, cfg, meta, done))
    drop = complete_seed_filter([f[2] for f in found])
    for item in found:
        if item[2]["run_id"] not in drop:
            yield item


def warn_mixed_experiments(metas) -> list:
    """warnings.warn once per (pe, tag) whose finished runs have more than one distinct
    experiment_hash (None counts as one value). Returns the offending (pe, tag) pairs."""
    by = {}
    for m in metas:
        by.setdefault((m["pe"], m["tag"]), {}).setdefault(m.get(HASH_COL), []).append(m["run_id"])
    bad = []
    for (pe, tag), hashes in sorted(by.items()):
        if len(hashes) < 2:
            continue
        bad.append((pe, tag))
        detail = "; ".join(f"{(h or 'none')[:10]}: {', '.join(sorted(ids))}"
                           for h, ids in sorted(hashes.items(), key=lambda kv: kv[0] or ""))
        warnings.warn(f"collect: finished runs with pe={pe!r}, tag={tag!r} come from "
                      f"{len(hashes)} different experiment configs (experiment_hash), so they "
                      f"are not comparable: {detail}. Rerun the stale ones (sweep --overwrite) "
                      f"or move them to their own --tag.")
    return bad


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str, np.integer, np.floating))


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def metric_rows(meta: dict, eval_json: dict, phase: int) -> list[dict]:
    """One row per eval set of one eval_phase{p}.json (per_position_acc and samples dropped)."""
    rows = []
    file_fields = {k: eval_json.get(k) for k in EVAL_FILE_FIELDS}
    for set_name, m in (eval_json.get("test_sets") or {}).items():
        row = dict(meta)
        row.update({"phase": int(phase), "set_name": str(set_name), "kind": m.get("kind"),
                    "length": m.get("length"), "r": m.get("r")})
        for k in SCALAR_METRICS:
            row[k] = m.get(k, np.nan)
        for k, v in m.items():  # any extra scalar metric a later version adds
            if k not in DROPPED_KEYS and k not in row and _is_scalar(v):
                row[k] = v
        et = m.get("error_types") or {}
        for k in ERROR_TYPES:
            row[f"err_{k}"] = et.get(k, np.nan)
        row.update(file_fields)
        rows.append(row)
    return rows


def _finalize_metrics(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        cols = RUN_COLS + SET_COLS + SCALAR_METRICS + ERR_COLS + EVAL_FILE_FIELDS
        return _set_dtypes(pd.DataFrame(columns=cols))
    df = pd.DataFrame(rows)
    front = RUN_COLS + SET_COLS + SCALAR_METRICS
    extra = [c for c in df.columns if c not in front + ERR_COLS + EVAL_FILE_FIELDS]
    df = df[front + extra + ERR_COLS + EVAL_FILE_FIELDS]
    df = df.sort_values(["pe", "schedule", "tag", "seed", "phase", "kind", "length", "r"],
                        na_position="first", kind="stable").reset_index(drop=True)
    return _set_dtypes(df)


def _set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical dtypes (also used after a CSV round trip)."""
    df = df.copy()
    for c in ("run_id", "pe", "schedule", "base_schedule", "set_name", "kind", "which", "key",
              "readout", HASH_COL):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
            df[c] = df[c].map(lambda v: v if v is None else str(v))
    if "tag" in df.columns:
        df["tag"] = df["tag"].fillna("").astype(str)
    if "is_twin" in df.columns:
        df["is_twin"] = df["is_twin"].map(
            lambda v: str(v).lower() in ("true", "1") if isinstance(v, str) else bool(v)
        ).astype(bool)
    for c in ("seed", "phase"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c]).astype("int64") if len(df) else df[c].astype("int64")
    for c in ("r",):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "length" in df.columns and "which" not in df.columns:  # metrics/probes: int lengths
        length = pd.to_numeric(df["length"], errors="coerce")
        df["length"] = length.astype("int64") if length.notna().all() else length
    return df


# ----------------------------------------------------------------------------------------------
# learning curves and loss
# ----------------------------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    recs = []
    if not path.is_file():
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.warn(f"{path}:{i}: skipping unparsable line")
    return recs


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _key_length(which: str, key: str) -> float:
    if which in ("val", "curve_ood", "val_token_acc"):
        try:
            return float(int(key))
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def _phase_offsets(recs: list[dict], cfg: dict, done: dict) -> dict:
    """Phase -> global-step offset."""
    p1 = _phase1_steps(cfg, done)
    if p1 is None:
        p1_steps = [int(r.get("step", 0)) for r in recs if int(r.get("phase", 1)) == 1]
        p1 = max(p1_steps) if p1_steps else 0
    return {1: 0, 2: int(p1)}


def curve_rows(meta: dict, recs: list[dict], offsets: dict) -> list[dict]:
    """Long-format rows from the type=="eval" records of one train_log.jsonl."""
    latest = {}  # (phase, step) -> record; the last record for a step wins
    for rec in recs:
        if rec.get("type") != "eval":
            continue
        latest[(int(rec.get("phase", 1)), int(rec.get("step", 0)))] = rec
    rows = []
    for (phase, step), rec in sorted(latest.items()):
        base = dict(meta)
        base.update({"phase": phase, "step": step,
                     "global_step": offsets.get(phase, 0) + step})
        for which in CURVE_WHICH:
            val = rec.get(which)
            if val is None:
                continue
            items = val.items() if isinstance(val, dict) else [("all", val)]
            for key, v in items:
                row = dict(base)
                row.update({"which": which, "key": str(key),
                            "length": _key_length(which, str(key)), "value": _to_float(v)})
                rows.append(row)
    return rows


def loss_rows(meta: dict, recs: list[dict], offsets: dict) -> list[dict]:
    rows = []
    for rec in recs:
        if rec.get("type") != "loss":
            continue
        phase, step = int(rec.get("phase", 1)), int(rec.get("step", 0))
        row = dict(meta)
        row.update({"phase": phase, "step": step, "global_step": offsets.get(phase, 0) + step})
        for k in LOSS_FIELDS:
            row[k] = _to_float(rec.get(k))
        rows.append(row)
    return rows


def load_losses(results_dir) -> pd.DataFrame:
    """Loss records of every finished run: RUN_COLS + phase, step, global_step, loss, lr,
    tokens_nonpad, tokens_padded, examples (cumulative counters as logged)."""
    rows = []
    for run_dir, cfg, meta, done in _finished_runs(results_dir):
        recs = _read_jsonl(run_dir / "train_log.jsonl")
        rows += loss_rows(meta, recs, _phase_offsets(recs, cfg, done))
    cols = RUN_COLS + ["phase", "step", "global_step"] + LOSS_FIELDS
    df = pd.DataFrame(rows, columns=cols)
    return _set_dtypes(df) if len(df) else df


# ----------------------------------------------------------------------------------------------
# main entry points
# ----------------------------------------------------------------------------------------------
def load_runs(results_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics_df, curves_df) for every finished run (with done.json) in results_dir.

    metrics_df: one row per (run, phase, eval set); columns META_COLS + experiment_hash + phase,
        set_name, kind, length, r + scalar metrics + err_correct/err_order_only/
        err_multiset_only/err_both + steps/tokens_nonpad/tokens_padded/wall_time_s.
    curves_df: one row per (run, phase, step, which, key) from the type=="eval" log records;
        columns CURVE_COLS (which in "val"|"val_group"|"curve_ood"|"val_token_acc").
    Warns (warn_mixed_experiments) if runs sharing (pe, tag) have different experiment hashes."""
    m_rows, c_rows, metas = [], [], []
    for run_dir, cfg, meta, done in _finished_runs(results_dir):
        metas.append(meta)
        for p in PHASES:
            path = run_dir / f"eval_phase{p}.json"
            if path.is_file():
                ej = _read_json(path)
                if ej is not None:
                    m_rows += metric_rows(meta, ej, int(ej.get("phase", p)))
        recs = _read_jsonl(run_dir / "train_log.jsonl")
        c_rows += curve_rows(meta, recs, _phase_offsets(recs, cfg, done))
    warn_mixed_experiments(metas)
    metrics_df = _finalize_metrics(m_rows)
    curves_df = pd.DataFrame(c_rows, columns=CURVE_COLS)
    curves_df = _set_dtypes(curves_df) if len(curves_df) else curves_df
    return metrics_df, curves_df


def load_probes(results_dir) -> pd.DataFrame:
    """Rows from probes_phase{p}.json of finished runs, one per (run, phase, readout, length):
    META_COLS + experiment_hash + phase, readout, length + every scalar field of by_length[L]
    (n, min_acc, id_succ_acc, ...). Nested (non-scalar) fields are dropped. Empty DataFrame if
    none exist."""
    rows = []
    for run_dir, _cfg, meta, _done in _finished_runs(results_dir):
        for p in PHASES:
            path = run_dir / f"probes_phase{p}.json"
            if not path.is_file():
                continue
            pj = _read_json(path)
            if not pj:
                continue
            readout = pj.get("readout")
            if not _is_scalar(readout):
                readout = json.dumps(readout, sort_keys=True)
            for L, entry in (pj.get("by_length") or {}).items():
                row = dict(meta)
                row.update({"phase": int(pj.get("phase", p)), "readout": readout,
                            "length": int(L)})
                for k, v in (entry or {}).items():
                    if _is_scalar(v) and k not in row:
                        row[k] = v
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=PROBE_BASE_COLS)
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in PROBE_BASE_COLS]
    df = df[PROBE_BASE_COLS + rest].sort_values(
        ["pe", "schedule", "tag", "seed", "phase", "length"], kind="stable").reset_index(drop=True)
    return _set_dtypes(df)


def load_per_position(results_dir, set_names=None, phases=PHASES) -> pd.DataFrame:
    """Per-position accuracy rows: RUN_COLS + phase, set_name, kind, length, r, position
    (1-based), rel_position (= position / length), acc. set_names=None means every set."""
    wanted = None if set_names is None else {str(s) for s in set_names}
    rows = []
    for run_dir, _cfg, meta, _done in _finished_runs(results_dir):
        for p in phases:
            path = run_dir / f"eval_phase{p}.json"
            if not path.is_file():
                continue
            ej = _read_json(path) or {}
            for set_name, m in (ej.get("test_sets") or {}).items():
                if wanted is not None and set_name not in wanted:
                    continue
                ppa = m.get("per_position_acc") or []
                L = int(m.get("length") or len(ppa))
                for j, acc in enumerate(ppa, 1):
                    row = dict(meta)
                    row.update({"phase": int(ej.get("phase", p)), "set_name": set_name,
                                "kind": m.get("kind"), "length": L, "r": m.get("r"),
                                "position": j, "rel_position": j / L, "acc": float(acc)})
                    rows.append(row)
    cols = RUN_COLS + SET_COLS + ["position", "rel_position", "acc"]
    df = pd.DataFrame(rows, columns=cols)
    return _set_dtypes(df) if len(df) else df


def read_csv(path) -> pd.DataFrame:
    """Read a CSV written by this module and restore dtypes (tag "" not NaN, r as Int64)."""
    df = pd.read_csv(path, keep_default_na=True)
    return _set_dtypes(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect finished runs into CSV tables.")
    ap.add_argument("--results", default="results", help="results directory (one subdir per run)")
    ap.add_argument("--out", default=None, help="output directory (default: --results)")
    args = ap.parse_args(argv)
    out = Path(args.out or args.results)
    out.mkdir(parents=True, exist_ok=True)

    n_runs = len(find_run_dirs(args.results))
    metrics_df, curves_df = load_runs(args.results)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    curves_df.to_csv(out / "curves.csv", index=False)
    print(f"[collect] {n_runs} finished runs -> {out / 'metrics.csv'} ({len(metrics_df)} rows), "
          f"{out / 'curves.csv'} ({len(curves_df)} rows)")
    probes_df = load_probes(args.results)
    if len(probes_df):
        probes_df.to_csv(out / "probes.csv", index=False)
        print(f"[collect] -> {out / 'probes.csv'} ({len(probes_df)} rows)")
    else:
        print("[collect] no probes_phase*.json files found; probes.csv not written")
    return metrics_df, curves_df, probes_df


if __name__ == "__main__":
    main(sys.argv[1:])
