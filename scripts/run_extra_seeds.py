"""Run additional seeds of a grid (default: the sinusoidal main grid, seeds 5-9), then refresh the
probes, tables and figures.

    python scripts/run_extra_seeds.py                              # main grid, seeds 5-9
    python scripts/run_extra_seeds.py --grid nope                  # NoPE grid, seeds 5-9
    python scripts/run_extra_seeds.py --seeds 5 6 7
    python scripts/run_extra_seeds.py --until "YYYY-MM-DD HH:MM"   # start no run that would end later

Runs go seed by seed, using sweep.py's plan, with the same config-hash check. Finished runs are
skipped, so the script can be stopped between runs and restarted with the same command. To stop
after the current run, create results/STOP_EXTRA_SEEDS (see STOP_FILE).

With --until, a run starts only if it should finish (RUN_BUDGET_S, a little over the slowest run)
before the given time; the analysis is then refreshed and the script exits. Failed runs get one
retry. On Windows the system is kept awake while it runs.
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402  (puts the project root on sys.path)
from run_all import ROOT, analysis, keep_awake, log  # noqa: E402

RUN_BUDGET_S = 15 * 60  # slowest full run so far: about 800 s
# Create this file (any content) to pause cleanly: the run in progress finishes, no new run starts,
# the analysis is refreshed and the script exits. Delete it before resuming.
STOP_FILE = os.path.join(ROOT, "results", "STOP_EXTRA_SEEDS")


def run_one(p) -> bool:
    """One src.train run, exactly as sweep.py runs it (same argv, same log file)."""
    os.makedirs(os.path.dirname(p["log"]), exist_ok=True)
    with open(p["log"], "w", encoding="utf-8") as f:
        f.write(sweep.show(p["argv"]) + "\n")
        f.flush()
        rc = subprocess.run([sys.executable, "-m", "src.train"] + p["argv"], cwd=ROOT, stdout=f,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1")).returncode
    return rc == 0 and os.path.exists(os.path.join(p["run_dir"], "done.json"))


def run_list(plan, deadline, failed) -> bool:
    """Run the plan items in order. Appends failures to `failed`. True if stopped by the deadline."""
    for p in plan:
        if os.path.exists(STOP_FILE):
            log(f"stop file found ({STOP_FILE}): not starting {p['run_id']}. Delete it and rerun "
                "this script to continue.")
            return True
        if deadline is not None and time.time() + RUN_BUDGET_S > deadline:
            log(f"deadline reached: not starting {p['run_id']}. Rerun this script to continue.")
            return True
        log(f"run  {p['run_id']}  (log: {p['log']})")
        t = time.time()
        ok = run_one(p)
        log(f"{'ok  ' if ok else 'FAIL'} {p['run_id']}  {time.time() - t:.0f}s")
        if not ok:
            failed.append(p)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", default="main", choices=("main", "nope"),
                    help="main = sinusoidal grid (10 schedules), nope = NoPE grid (6 schedules)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--until", default=None,
                    help='local time "YYYY-MM-DD HH:MM": no run starts unless it should end by then')
    args = ap.parse_args(argv)
    # fromisoformat accepts "YYYY-MM-DD HH:MM" and "YYYY-MM-DDTHH:MM" (the latter has no space,
    # so it survives being passed through a shell without quotes).
    deadline = dt.datetime.fromisoformat(args.until).timestamp() if args.until else None

    keep_awake()
    t0 = time.time()
    plan = sweep.build_plan(args.grid, args.seeds)
    for p in plan:
        if p["status"] == "stale":
            log(sweep.stale_message(p))
    todo = [p for p in plan if p["status"] == "run"]
    log(f"extra seeds {args.grid} {args.seeds}: {len(plan)} runs, {len(todo)} to run"
        + (f", deadline {args.until}" if args.until else ""))

    failed = []
    stopped = run_list(todo, deadline, failed)
    if failed and not stopped:
        retry, failed = failed, []
        log(f"retrying {len(retry)} failed run(s)")
        stopped = run_list(retry, deadline, failed)

    analysis()
    left = sum(p["status"] != "skip" for p in sweep.build_plan(args.grid, args.seeds))
    log(f"extra seeds: exiting after {(time.time() - t0) / 3600:.1f} h; "
        f"{left} run(s) still to do{' (paused by deadline)' if stopped else ''}")


if __name__ == "__main__":
    main()
