# Does Training Order Matter for Length Generalization?

Code and results for the paper *Does Training Order Matter for Length Generalization? A Controlled
Study of Curricula in Transformer Sorting*.

We train 2-layer decoder-only transformers to sort lists of integers 1–100 of length 5–20 and test
them on lengths up to 50. Six training schedules present the training lengths in different orders.
Each ordered schedule is paired with a **shuffled twin** that consumes exactly the same examples,
with the same number of updates and the same initial weights, in random order, so any difference
between a schedule and its twin is due to order alone. Every condition is run with sinusoidal and
with no positional encoding (NoPE), followed by a shared phase of random-mix training.

## Setup

- Python 3.12 and PyTorch 2.5 with CUDA. Install PyTorch from [pytorch.org](https://pytorch.org)
  (CUDA 12.1 wheels), then `pip install -r requirements.txt`.
- Run every command from the repository root.
- Tests run on the CPU: `python -m pytest -q`.

## Reproducing the paper

```
python scripts/run_all.py                          # seeds 0-4, sinusoidal and NoPE grids (80 runs)
python scripts/run_extra_seeds.py --grid main      # sinusoidal seeds 5-9 (50 runs)
python scripts/run_extra_seeds.py --grid nope      # NoPE seeds 5-9 (30 runs)
python scripts/run_all.py --analysis               # probes, tables, figures and paper numbers
```

- Each run takes about 15 minutes on an 8 GB laptop GPU (RTX 3070 Ti), about 40 GPU-hours for
  all 160 runs.
- Finished runs are skipped, so every script can be stopped and restarted.
- The analysis writes tables and the numbers quoted in the paper to `paper/tables/`, and figures
  to `paper/figures/`.

`results/` contains the outputs of all 160 runs except the model checkpoints (`*.pt`, about
2 GB), so the analysis can be rerun without training:

- `<run_id>/`: `config.yaml`, `train_log.jsonl`, `eval_phase{1,2}.json`, `probes_phase{1,2}.json`
  and `done.json` for each run (see [Outputs](#outputs)).
- `metrics.csv`, `curves.csv`, `probes.csv`: the same results collected into tables.
- `prestated_predictions.md`: the predictions written after seeds 0–4 that the paper checks on
  seeds 5–9.

Re-running the probes needs the checkpoints, so `run_all.py --analysis` keeps the existing
`probes_phase*.json` files.

## Schedules

Training lengths are split into three groups: A (5–10), B (11–15) and C (16–20). Fractions are
of the 20,000 Phase-1 updates.

| Schedule | Stages |
|---|---|
| `R` | random mix of A, B and C throughout |
| `SL` | A (1/3) → B (1/3) → C (1/3) |
| `LS` | SL's stages in reverse order: C → B → A |
| `SLr-early` | A (15%) → A+B (15%) → A+B+C (70%) |
| `SLr-mid` | A (1/3) → A+B (1/3) → A+B+C (1/3) |
| `SLr-late` | A (50%) → A+B (30%) → A+B+C (20%) |
| `<schedule>-shuf` | the same examples as `<schedule>`, in random order |

`LS` consumes exactly the same examples as `SL`, so its twin is `SL-shuf`. After Phase 1, every
run receives 5,000 further updates of random-mix data (Phase 2).

## Training a single run

```
python -m src.train --config configs/base.yaml --schedule SLr-mid --pe sinusoidal --seed 0
```

- `--schedule`: one of `R SL SLr-early SLr-mid SLr-late LS`, or a twin `<schedule>-shuf`.
- `--pe`: `sinusoidal` or `none`.
- `--seed`, `--tag`, `--out-dir` (default `results/`).
- `--set dotted.key=value`: override any config key (value parsed as YAML), e.g.
  `--set train.lr=0.001`. Repeatable.
- `--config` can be repeated; later files override earlier ones. `configs/base.yaml` must come
  first. `configs/smoke.yaml` is a tiny CPU configuration for quick checks (about 15 s), and
  `configs/tune.yaml` evaluates only training lengths.
- `--overwrite`: re-run a finished run.

The run id is `{pe}_{schedule}_s{seed}[_{tag}]`. A run with `done.json` is never re-run without
`--overwrite`; an unfinished run directory is cleared and restarted.

## Sweeps

```
python scripts/sweep.py --grid main        # 10 conditions x seeds 0-4, sinusoidal (50 runs)
python scripts/sweep.py --grid nope        # R, SL, LS, SLr-mid and twins x seeds 0-4, NoPE (30 runs)
python scripts/sweep.py --grid nope_full   # 10 conditions x seeds 0-4, NoPE (50 runs)
python scripts/sweep.py --grid pilot       # R and SLr-mid, both encodings, seed 0
python scripts/sweep.py --grid tune        # R, learning rates {3e-4, 5e-4, 1e-3}, seeds 0-1
```

- Options: `--seeds 0 1 2`, `--dry-run`, `--overwrite`, `--out-dir DIR`, `--set key=value`.
- Runs execute one at a time; a failed run does not stop the sweep.
- Each finished run stores a hash of its resolved configuration. The sweep skips a run only if
  that hash matches the configuration it would run now; otherwise it reports the run as STALE,
  does not run it, and exits with an error, so results from different configurations are never
  mixed.

## Outputs

Each run writes to `<out_dir>/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | Fully resolved configuration (can be passed back as `--config`) |
| `train_log.jsonl` | Loss records every 100 updates and learning-curve evaluations every 1,000 updates |
| `eval_phase1.json`, `eval_phase2.json` | Test-set metrics at the end of each phase |
| `probes_phase1.json`, `probes_phase2.json` | Mechanism probe accuracies per test length |
| `model_phase1.pt`, `model_phase2.pt` | Model checkpoints (not included in this repository) |
| `done.json` | Completion marker with per-phase step, token and example counts, configuration hashes, and train/test overlap counts |

The learning rate warms up linearly over the first 1,000 updates and then stays constant. Phase 2
continues with the same optimizer state and learning rate.

## Analysis

```
python -m src.analysis.collect       --results results                 # metrics.csv, curves.csv
python -m src.analysis.stats         --results results --out tables    # seed statistics and LaTeX tables
python -m src.analysis.plots         --results results --out figures   # figures
python -m src.analysis.paper_numbers --results results --out tables    # numbers quoted in the paper
python -m src.analysis.probes        --run results/<run_id> --phase 1  # needs the checkpoint
python -m src.analysis.attention     --run results/<run_id> --phase 1 --length 20
python -m src.evaluate               --run results/<run_id> --phase 1  # re-evaluate a checkpoint
```

Each module documents its options in its docstring and `--help`.

## Repository layout

```
src/               data generation, model, training, evaluation and metrics
src/analysis/      result collection, statistics, figures, probes and paper numbers
scripts/           sweep launcher and the scripts that run the paper's experiments
configs/           base, smoke-test and tuning configurations
tests/             unit tests (CPU)
docs/CODE_SPEC.md  module interfaces, data formats, output schemas and implementation notes
results/           outputs of all runs (without checkpoints) and the pre-stated predictions
```
