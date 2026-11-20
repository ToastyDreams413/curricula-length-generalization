"""Sweep launcher (docs/CODE_SPEC.md section 5).

    python scripts/sweep.py --grid {pilot,tune,main,nope,nope_full} [--seeds 0 1 2 3 4]
                            [--dry-run] [--overwrite] [--out-dir DIR] [--set key=value ...]

Grids (seeds default to 0 1 2 3 4; pilot defaults to seed 0, tune to seeds 0 1):
  pilot      R, SLr-mid  x {sinusoidal, none}
  tune       R x lr {0.0003, 0.0005, 0.001} (tag lr<value>), base.yaml + tune.yaml (id_only,
             no Phase 2, no curve_ood), pe from the config (sinusoidal)
  main       GRID_CONDITIONS x sinusoidal
  nope       [R, SL, SLr-mid, LS, SL-shuf, SLr-mid-shuf] x none
  nope_full  GRID_CONDITIONS x none
Runs go seed by seed (all conditions of seed 0 first), so a partial sweep has complete seeds
for paired comparisons.

Each run is `python -m src.train ...` in a subprocess, from the project root, one at a time.
Its stdout/stderr go to <out_dir>/sweep_logs/<run_id>.log. Failures don't stop the sweep.
Extra --set overrides are passed to every run, after the grid's own overrides.
The run id, output dir and full config of each run are computed with src.train's own config
resolution, so they always match what src.train will do.

Finished runs (done.json exists) are compared with the config they WOULD run now:
  * done.json "config_hash" == src.train.config_hash(planned config)  -> SKIP (already done)
  * different hash, or done.json without one (made by older code)    -> STALE: reported as
        STALE <run_id>: finished with a different config (stored <h>, now <h>); rerun with
        --overwrite or use a --tag
    and NOT run, so a changed base.yaml never silently mixes old and new results.
--overwrite re-runs every planned run, finished or stale (it is passed on to src.train).
--dry-run prints the same SKIP / STALE decisions and the commands it would run.
A summary of succeeded / skipped / stale / failed runs is printed at the end. Exit code: 1 if
any run failed or was stale (also for --dry-run when anything is stale), 2 if the plan cannot be
built, else 0. Don't run two sweeps on the same out_dir at once: src.train clears run dirs that
have no done.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import train  # noqa: E402
from src.data import GRID_CONDITIONS  # noqa: E402

BASE = "configs/base.yaml"
TUNE = "configs/tune.yaml"
TUNE_LRS = ["0.0003", "0.0005", "0.001"]  # written with a dot so YAML parses them as floats
PILOT_SCHEDULES = ["R", "SLr-mid"]
NOPE_SCHEDULES = ["R", "SL", "SLr-mid", "LS", "SL-shuf", "SLr-mid-shuf"]
GRIDS = ("pilot", "tune", "main", "nope", "nope_full")
DEFAULT_SEEDS = {"pilot": [0], "tune": [0, 1]}
ALL_SEEDS = [0, 1, 2, 3, 4]


def grid_runs(grid: str, seeds) -> list:
    """List of run specs: dict(configs, schedule, pe, seed, tag, sets)."""
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}")

    def spec(schedule, pe, seed, configs=(BASE,), tag=None, sets=()):
        return {"configs": list(configs), "schedule": schedule, "pe": pe, "seed": int(seed),
                "tag": tag, "sets": list(sets)}

    runs = []
    for seed in seeds:
        if grid == "pilot":
            runs += [spec(s, pe, seed) for pe in ("sinusoidal", "none") for s in PILOT_SCHEDULES]
        elif grid == "tune":
            runs += [spec("R", None, seed, configs=(BASE, TUNE), tag=f"lr{lr}",
                          sets=[f"train.lr={lr}"]) for lr in TUNE_LRS]
        elif grid == "main":
            runs += [spec(s, "sinusoidal", seed) for s in GRID_CONDITIONS]
        elif grid == "nope":
            runs += [spec(s, "none", seed) for s in NOPE_SCHEDULES]
        elif grid == "nope_full":
            runs += [spec(s, "none", seed) for s in GRID_CONDITIONS]
    return runs


def train_argv(run: dict, out_dir=None, overwrite=False, extra_sets=()) -> list:
    """Arguments for `python -m src.train` (config paths relative to the project root)."""
    a = []
    for c in run["configs"]:
        a += ["--config", c]
    a += ["--schedule", run["schedule"]]
    if run["pe"]:
        a += ["--pe", run["pe"]]
    a += ["--seed", str(run["seed"])]
    if run["tag"]:
        a += ["--tag", run["tag"]]
    for s in list(run["sets"]) + list(extra_sets):
        a += ["--set", s]
    if out_dir:
        a += ["--out-dir", out_dir]
    if overwrite:
        a.append("--overwrite")
    return a


def resolve(argv: list) -> tuple:
    """(run_id, absolute run dir, absolute out_dir, resolved config) exactly as src.train will
    compute them."""
    cfg = train.config_from_args(train.parse_args(argv), base_dir=ROOT)
    out_dir = cfg["out_dir"]
    if not os.path.isabs(out_dir):  # src.train runs with cwd = ROOT
        out_dir = os.path.join(ROOT, out_dir)
    run_id = train.run_id_of(cfg)
    return run_id, os.path.join(out_dir, run_id), out_dir, cfg


def stored_hash(done_path: str):
    """config_hash recorded in a done.json, or None (no hash, or an unreadable file)."""
    try:
        with open(done_path, "r", encoding="utf-8") as f:
            done = json.load(f)
    except (OSError, ValueError):
        return None
    h = done.get("config_hash") if isinstance(done, dict) else None
    return h if isinstance(h, str) and h else None


def build_plan(grid, seeds=None, out_dir=None, overwrite=False, extra_sets=()) -> list:
    """One dict per run: run_id, run_dir, log, argv, done (done.json exists), config_hash (of the
    config it would run now), stored_hash (from done.json, or None) and status: "run",
    "skip" (finished with the same config) or "stale" (finished with a different config, or
    without a stored hash). With overwrite every status is "run"."""
    seeds = list(seeds) if seeds else DEFAULT_SEEDS.get(grid, ALL_SEEDS)
    plan = []
    for r in grid_runs(grid, seeds):
        argv = train_argv(r, out_dir=out_dir, overwrite=overwrite, extra_sets=extra_sets)
        run_id, run_dir, od, cfg = resolve(argv)
        done_path = os.path.join(run_dir, "done.json")
        done = os.path.exists(done_path)
        want = train.config_hash(cfg)
        have = stored_hash(done_path) if done else None
        if overwrite or not done:
            status = "run"
        else:
            status = "skip" if have == want else "stale"
        plan.append({"run_id": run_id, "run_dir": run_dir, "argv": argv,
                     "log": os.path.join(od, "sweep_logs", f"{run_id}.log"), "done": done,
                     "config_hash": want, "stored_hash": have, "status": status})
    ids = [p["run_id"] for p in plan]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids in the plan")
    return plan


def stale_message(p: dict) -> str:
    return (f"STALE {p['run_id']}: finished with a different config (stored "
            f"{(p['stored_hash'] or 'none')[:10]}, now {p['config_hash'][:10]}); rerun with "
            f"--overwrite or use a --tag")


def show(argv: list) -> str:
    return subprocess.list2cmdline(["python", "-m", "src.train"] + argv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a grid of src.train runs sequentially.")
    ap.add_argument("--grid", required=True, choices=GRIDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default 0 1 2 3 4 (pilot: 0; tune: 0 1)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and the count")
    ap.add_argument("--overwrite", action="store_true", help="re-run finished runs too")
    ap.add_argument("--out-dir", default=None, help="default: out_dir from the config (results)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="passed to every run as --set KEY=VALUE")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    try:
        plan = build_plan(args.grid, args.seeds, out_dir, args.overwrite, args.sets)
    except (train.ConfigError, ValueError, OSError) as e:
        print(f"[sweep] cannot build the plan: {e}", file=sys.stderr)
        return 2
    n = len(plan)
    count = {s: sum(p["status"] == s for p in plan) for s in ("skip", "stale", "run")}
    counts = (f"{n} runs ({count['skip']} already done, {count['stale']} stale, "
              f"{count['run']} to run)")

    if args.dry_run:
        for i, p in enumerate(plan, 1):
            if p["status"] == "stale":
                print(f"[{i}/{n}] {stale_message(p)}")
            else:
                print(f"[{i}/{n}] {'SKIP (done) ' if p['status'] == 'skip' else ''}"
                      f"{show(p['argv'])}")
        print(f"[sweep] dry run, grid {args.grid}: {counts}")
        return 1 if count["stale"] else 0

    print(f"[sweep] grid {args.grid}: {counts}", flush=True)
    ok, skipped, stale, failed = [], [], [], []
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t_sweep = time.perf_counter()
    try:
        for i, p in enumerate(plan, 1):
            if p["status"] == "skip":
                skipped.append(p["run_id"])
                print(f"[{i}/{n}] skip {p['run_id']} (done.json exists, same config)", flush=True)
                continue
            if p["status"] == "stale":
                stale.append(p["run_id"])
                print(f"[{i}/{n}] {stale_message(p)}", flush=True)
                continue
            os.makedirs(os.path.dirname(p["log"]), exist_ok=True)
            print(f"[{i}/{n}] run  {p['run_id']}  (log: {p['log']})", flush=True)
            t0 = time.perf_counter()
            with open(p["log"], "w", encoding="utf-8") as log:
                log.write(show(p["argv"]) + "\n")
                log.flush()
                rc = subprocess.run([sys.executable, "-m", "src.train"] + p["argv"], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            dt = time.perf_counter() - t0
            if rc == 0 and os.path.exists(os.path.join(p["run_dir"], "done.json")):
                ok.append(p["run_id"])
                print(f"[{i}/{n}] ok   {p['run_id']}  {dt:.0f}s", flush=True)
            else:
                failed.append(p["run_id"])
                print(f"[{i}/{n}] FAIL {p['run_id']}  exit code {rc}, {dt:.0f}s; see {p['log']}",
                      flush=True)
    except KeyboardInterrupt:
        print("[sweep] interrupted", flush=True)
        failed.append("<interrupted>")
    finally:
        mins = (time.perf_counter() - t_sweep) / 60
        print(f"[sweep] done in {mins:.1f} min: {len(ok)} succeeded, {len(skipped)} skipped, "
              f"{len(stale)} stale, {len(failed)} failed", flush=True)
        for rid in stale:
            print(f"  STALE   {rid}  (not run: rerun with --overwrite or use a --tag)", flush=True)
        for rid in failed:
            print(f"  FAILED  {rid}", flush=True)
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
