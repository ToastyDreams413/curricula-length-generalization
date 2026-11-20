"""Run seeds 0-4 of every condition (both positional encodings), then the probes, tables and
figures. Seeds 5-9 are run with scripts/run_extra_seeds.py.

    python scripts/run_all.py              # all runs for seeds 0-4, then the analysis
    python scripts/run_all.py --analysis   # only probes + collect + stats + plots on what exists

Safe to re-run: finished runs are skipped by the sweep (config-hash checked), and probes are
skipped for run/phase pairs that already have a probes file. Seeds 0-2 of both positional
encodings run first, then seeds 3-4. On Windows the system is kept awake while it runs
(SetThreadExecutionState).
"""

import argparse
import ctypes
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SWEEPS = [
    ["--grid", "main", "--seeds", "0", "1", "2"],
    ["--grid", "nope", "--seeds", "0", "1", "2"],
    ["--grid", "main", "--seeds", "3", "4"],
    ["--grid", "nope", "--seeds", "3", "4"],
]
PROBE_LENGTHS = ["5", "10", "15", "20", "22", "25", "30", "40", "50"]
PROBE_N = "500"


def log(msg):
    print(f"[run_all {time.strftime('%a %H:%M:%S')}] {msg}", flush=True)


def run(args):
    log("$ " + " ".join(args))
    return subprocess.run([PY, *args], cwd=ROOT).returncode


def keep_awake():
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while we run
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def analysis():
    done = sorted(glob.glob(os.path.join(ROOT, "results", "*", "done.json")))
    log(f"probes for {len(done)} finished runs")
    for d in (os.path.dirname(p) for p in done):
        for phase in ("1", "2"):
            if (os.path.exists(os.path.join(d, f"model_phase{phase}.pt"))
                    and not os.path.exists(os.path.join(d, f"probes_phase{phase}.json"))):
                run(["-m", "src.analysis.probes", "--run", d, "--phase", phase,
                     "--lengths", *PROBE_LENGTHS, "--n", PROBE_N])
    run(["-m", "src.analysis.collect", "--results", "results"])
    run(["-m", "src.analysis.stats", "--results", "results", "--out", "paper/tables"])
    run(["-m", "src.analysis.plots", "--results", "results", "--out", "paper/figures"])
    run(["-m", "src.analysis.paper_numbers", "--results", "results", "--out", "paper/tables"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis", action="store_true", help="skip training; analysis only")
    args = ap.parse_args()
    keep_awake()
    t0 = time.time()
    if not args.analysis:
        for s in SWEEPS:
            rc = run(["scripts/sweep.py", *s])
            log(f"sweep {' '.join(s)} exited {rc} ({(time.time() - t0) / 3600:.1f} h so far)")
            analysis()  # refresh figures after every sweep so partial results are usable
    else:
        analysis()
    log(f"all done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
