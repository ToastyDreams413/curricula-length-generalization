"""Tests for src/analysis/{collect,stats,plots}.py on a SYNTHETIC results directory that follows
docs/CODE_SPEC.md section 5 (config.yaml, train_log.jsonl, eval_phase{1,2}.json, done.json,
probes_phase{p}.json). CPU only, no torch.

Synthetic exact match (chosen so that nothing is clipped and bucket means are hand-checkable):
    em = BASE_EM[L] + SCHED_OFF[schedule] + 0.01 * seed + 0.04 * (phase - 1)
    BASE_EM: 0.9 for L <= 20; 21..24 -> .8 .7 .6 .5; 25 .4, 30 .3, 40 .25, 50 .21
    rep sets: uniform em(L) - 0.1 * r   (so a rep set leaking into a bucket changes its mean)
"""

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import rankdata, wilcoxon

from src.analysis import collect, plots, stats
from src.analysis.plots import NoDataError
from src.data import length_sequence

TEST_LENGTHS = [5, 10, 15, 20, 21, 22, 23, 24, 25, 30, 40, 50]
REP_SETS = [(10, 2), (20, 2), (20, 3), (30, 2), (30, 4)]
VAL_LENGTHS = list(range(5, 21))
CURVE_OOD = [22, 25, 30]
P1_STEPS, P2_STEPS, EVAL_EVERY = 200, 100, 50
BASE_EM = {**{L: 0.9 for L in range(5, 21)}, 21: 0.8, 22: 0.7, 23: 0.6, 24: 0.5, 25: 0.4,
           30: 0.3, 40: 0.25, 50: 0.21}
SCHED_OFF = {"R": 0.0, "SL": -0.10, "SL-shuf": -0.05, "LS": -0.20, "SLr-mid": 0.03,
             "SLr-mid-shuf": 0.02, "SLr-late": 0.0}
MAIN_RUNS = ([("sinusoidal", s, seed) for s in ["R", "SL", "SL-shuf", "LS", "SLr-mid",
                                                "SLr-mid-shuf"] for seed in (0, 1, 2)]
             + [("none", s, seed) for s in ["R", "SL", "SL-shuf"] for seed in (0, 1)])
N_MAIN = len(MAIN_RUNS)  # 24 finished main-grid runs, each with Phase 1 and Phase 2
TUNE_UNIFORM = [L for L in TEST_LENGTHS if L <= 20]
TUNE_REP = [(L, r) for (L, r) in REP_SETS if L <= 20]
PLOT_DPI = 40


def em_of(L, schedule, seed, phase, r=None):
    v = BASE_EM[L] + SCHED_OFF[schedule] + 0.01 * seed + 0.04 * (phase - 1)
    return v - 0.1 * r if r is not None else v


def metric_dict(kind, L, r, em, n=100):
    """A full spec-section-4 metric dict whose scalar values are simple functions of em."""
    wrong = 1.0 - em
    return {"kind": kind, "length": L, "r": r, "n": n, "exact_match": em,
            "token_acc": em + 0.5 * wrong,
            "per_position_acc": [float(min(1.0, em + 0.1)) if j < 20 else float(em * 0.5)
                                 for j in range(L)],
            "prefix_len": L * (em + 0.2 * wrong), "prefix_frac": em + 0.2 * wrong,
            "inversions": 3.0 * wrong, "adjacent_inversions": 1.0 * wrong,
            "multiset_diff": 2.0 * wrong, "multiset_ok": em + 0.5 * wrong,
            "edit_distance": L * 0.3 * wrong, "edit_distance_norm": 0.3 * wrong,
            "error_types": {"correct": em, "order_only": 0.5 * wrong,
                            "multiset_only": 0.3 * wrong, "both": 0.2 * wrong},
            "samples": [{"input": [3, 1, 2][:min(3, L)], "target": [1, 2, 3][:min(3, L)],
                         "output": [1, 2, 3][:min(3, L)]}]}


def config_for(pe, schedule, seed, run_id, tag="", id_only=False, p2_steps=P2_STEPS):
    return {"out_dir": "results", "seed": seed, "tag": tag, "run_id": run_id,
            "model": {"d_model": 256, "n_layers": 2, "n_heads": 8, "d_mlp": 1024, "pe": pe,
                      "max_len": 512},
            "data": {"schedule": schedule, "groups": {"A": [5, 10], "B": [11, 15],
                                                      "C": [16, 20]}, "mix": "lengths"},
            "train": {"steps": P1_STEPS, "batch_size": 256, "lr": 5.0e-4, "warmup_steps": 10,
                      "weight_decay": 0.0, "betas": [0.9, 0.98], "grad_clip": 1.0,
                      "device": "auto", "log_every": 25},
            "phase2": {"steps": p2_steps, "reset_optimizer": False},
            "eval": {"every": EVAL_EVERY, "val_per_length": 100,
                     "curve_ood_lengths": [] if id_only else CURVE_OOD,
                     "test_lengths": TEST_LENGTHS, "test_per_length": 100,
                     "rep_lengths": [10, 20, 30], "rep_rs": [2, 3, 4], "rep_per_set": 50,
                     "batch_size": 250, "decode_vocab": "values", "n_samples": 1,
                     "id_only": id_only},
            "save_checkpoints": False}


def curve_value(schedule, seed, phase, step, group):
    """Learning-curve value: rises with step; SL 'forgets' A in its later stages."""
    frac = step / (P1_STEPS if phase == 1 else P2_STEPS)
    v = 0.3 + 0.6 * frac if phase == 1 else 0.9
    if schedule == "SL" and group == "A" and phase == 1 and step > 66:
        v -= 0.3
    return round(v + 0.01 * seed, 6)


def write_run(root, pe, schedule, seed, *, run_id=None, tag="", finished=True,
              id_only=False, phases=(1, 2), probe_phases=(1,), decoy=False,
              experiment_hash=None):
    run_id = run_id or f"{pe}_{schedule}_s{seed}" + (f"_{tag}" if tag else "")
    d = root / run_id
    d.mkdir(parents=True)
    p2_steps = P2_STEPS if 2 in phases else 0
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_for(pe, schedule, seed, run_id, tag, id_only, p2_steps), f)
    log = []
    for phase in phases:
        steps = P1_STEPS if phase == 1 else P2_STEPS
        for s in range(25, steps + 1, 25):
            log.append({"type": "loss", "phase": phase, "step": s, "loss": 1.0 / s, "lr": 5e-4,
                        "tokens_nonpad": 100 * s, "tokens_padded": 120 * s, "examples": 256 * s})
        for s in range(0, steps + 1, EVAL_EVERY):
            rec = {"type": "eval", "phase": phase, "step": s,
                   "val": {str(L): curve_value(schedule, seed, phase, s, "A") for L in
                           VAL_LENGTHS},
                   "val_group": {g: curve_value(schedule, seed, phase, s, g) for g in "ABC"},
                   "val_token_acc": curve_value(schedule, seed, phase, s, "B"),
                   "curve_ood": {} if id_only else {str(L): 0.1 for L in CURVE_OOD}}
            log.append(rec)
        # end-of-phase eval coincides with the last periodic one: a duplicate record (last wins)
        last = json.loads(json.dumps(log[-1]))
        last["val_group"]["A"] = round(last["val_group"]["A"] + 0.001, 6)
        log.append(last)
    with open(d / "train_log.jsonl", "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")
    for phase in phases:
        sets = {}
        for L in TEST_LENGTHS:
            if id_only and L > 20:
                continue
            sets[f"len_{L}"] = metric_dict("uniform", L, None, em_of(L, schedule, seed, phase))
        for L, r in REP_SETS:
            if id_only and L > 20:
                continue
            sets[f"rep_{L}_r{r}"] = metric_dict("rep", L, r, em_of(L, schedule, seed, phase, r))
        ej = {"run_id": run_id, "phase": phase, "steps": P1_STEPS if phase == 1 else P2_STEPS,
              "tokens_nonpad": 12345 * phase, "tokens_padded": 23456 * phase,
              "wall_time_s": 10.0 * phase, "test_sets": sets}
        with open(d / f"eval_phase{phase}.json", "w", encoding="utf-8") as f:
            json.dump(ej, f)
    if decoy:  # written by `python -m src.evaluate` by default; must be ignored
        with open(d / "eval_phase1_cli.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "phase": 1, "test_sets": {
                "len_5": metric_dict("uniform", 5, None, 0.0)}}, f)
    for phase in probe_phases:
        pj = {"run_id": run_id, "phase": phase, "readout": "decoder_basis",
              "by_length": {str(L): {"n": 50, "min_acc": round(1.0 - 0.01 * L, 4),
                                     "id_succ_acc": round(0.9 - 0.005 * L + 0.01 * seed, 4),
                                     "nested": {"ignored": 1}}
                            for L in (10, 20, 30, 50)}}
        with open(d / f"probes_phase{phase}.json", "w", encoding="utf-8") as f:
            json.dump(pj, f)
    if finished:
        done = {"run_id": run_id, "finished": "2000-01-01T00:00:00", "phase1_steps": P1_STEPS,
                "phase2_steps": p2_steps, "tokens_nonpad_phase1": 12345}
        if experiment_hash is not None:  # runs made by older code have no hashes
            done.update(config_hash="f" * 64, experiment_hash=experiment_hash)
        with open(d / "done.json", "w", encoding="utf-8") as f:
            json.dump(done, f)
    return d


def build_results(root, runs=MAIN_RUNS, extras=True, phases=(1, 2), probes=True):
    root.mkdir(parents=True, exist_ok=True)
    for k, (pe, s, seed) in enumerate(runs):
        pph = ((1, 2) if pe == "none" else (1,)) if probes else ()
        write_run(root, pe, s, seed, phases=phases, probe_phases=pph, decoy=(k == 0))
    if not extras:
        return root
    # Unfinished run (no done.json): must be skipped everywhere.
    write_run(root, "sinusoidal", "SLr-late", 0, finished=False, probe_phases=(1,))
    # Tuning-style run: tag, id_only, no Phase 2, run_id that does NOT follow the pattern.
    write_run(root, "sinusoidal", "R", 0, run_id="tune_lr1e-3_seed0", tag="lr1e-3",
              id_only=True, phases=(1,), probe_phases=())
    (root / "stray_file.txt").write_text("not a run")
    return root


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    return build_results(tmp_path_factory.mktemp("results"))


@pytest.fixture(scope="module")
def degraded_results_dir(tmp_path_factory):
    """Degraded results dir for the end-to-end CLI test: NoPE runs only (missing PE), Phase 1
    only (missing Phase 2), no probe files. Small, so the CLI test stays fast."""
    runs = [r for r in MAIN_RUNS if r[0] == "none"]
    return build_results(tmp_path_factory.mktemp("results_degraded"), runs=runs, extras=False,
                         phases=(1,), probes=False)


@pytest.fixture(scope="module")
def loaded(results_dir):
    metrics_df, curves_df = collect.load_runs(results_dir)
    return metrics_df, curves_df


@pytest.fixture(scope="module")
def probes_df(results_dir):
    return collect.load_probes(results_dir)


@pytest.fixture(scope="module")
def pp_df(results_dir):
    return collect.load_per_position(results_dir)


@pytest.fixture(scope="module")
def bucket_df(loaded):
    m = loaded[0]
    return stats.aggregate_buckets(m[m["tag"] == ""])


# ==============================================================================================
# collect
# ==============================================================================================
def test_find_run_dirs_skips_unfinished(results_dir):
    names = {p.name for p in collect.find_run_dirs(results_dir)}
    assert len(names) == N_MAIN + 1
    assert "sinusoidal_SLr-late_s0" not in names
    assert "tune_lr1e-3_seed0" in names


def test_metrics_df_rows_columns_metadata(loaded):
    m, _ = loaded
    n_sets = len(TEST_LENGTHS) + len(REP_SETS)
    n_tune = len(TUNE_UNIFORM) + len(TUNE_REP)
    assert len(m) == N_MAIN * 2 * n_sets + n_tune
    for c in ["run_id", "pe", "schedule", "base_schedule", "is_twin", "seed", "tag", "phase",
              "set_name", "kind", "length", "r", "n", "exact_match", "token_acc", "prefix_len",
              "prefix_frac", "inversions", "adjacent_inversions", "multiset_diff",
              "multiset_ok", "edit_distance", "edit_distance_norm", "err_correct",
              "err_order_only", "err_multiset_only", "err_both", "tokens_nonpad"]:
        assert c in m.columns, c
    assert "per_position_acc" not in m.columns and "samples" not in m.columns
    assert "error_types" not in m.columns
    assert not m["schedule"].eq("SLr-late").any()  # unfinished run skipped
    # one row per (run, phase, set): no double counting from eval_phase1_cli.json
    assert not m.duplicated(["run_id", "phase", "set_name"]).any()

    meta = m.drop_duplicates("run_id").set_index("run_id")
    assert meta.loc["sinusoidal_SL-shuf_s1", "base_schedule"] == "SL"
    assert bool(meta.loc["sinusoidal_SL-shuf_s1", "is_twin"]) is True
    assert meta.loc["sinusoidal_SLr-mid-shuf_s0", "base_schedule"] == "SLr-mid"
    assert meta.loc["sinusoidal_LS_s2", "base_schedule"] == "LS"
    assert bool(meta.loc["sinusoidal_LS_s2", "is_twin"]) is False
    assert meta.loc["none_R_s1", "pe"] == "none" and meta.loc["none_R_s1", "seed"] == 1
    assert meta.loc["sinusoidal_R_s0", "tag"] == ""
    # metadata comes from config.yaml, not from the run_id string
    t = meta.loc["tune_lr1e-3_seed0"]
    assert (t["pe"], t["schedule"], t["seed"], t["tag"]) == ("sinusoidal", "R", 0, "lr1e-3")
    assert m["is_twin"].dtype == bool and m["seed"].dtype == np.int64

    row = m[(m["run_id"] == "sinusoidal_SL_s2") & (m["phase"] == 2) &
            (m["set_name"] == "len_25")].iloc[0]
    em = em_of(25, "SL", 2, 2)
    assert row["kind"] == "uniform" and row["length"] == 25 and pd.isna(row["r"])
    assert row["exact_match"] == pytest.approx(em)
    assert row["err_order_only"] == pytest.approx(0.5 * (1 - em))
    assert row["tokens_nonpad"] == 12345 * 2
    rep = m[(m["run_id"] == "sinusoidal_SL_s2") & (m["phase"] == 1) &
            (m["set_name"] == "rep_30_r4")].iloc[0]
    assert rep["kind"] == "rep" and rep["r"] == 4 and rep["length"] == 30
    assert rep["exact_match"] == pytest.approx(em_of(30, "SL", 2, 1, 4))


def test_curves_df(loaded):
    _, c = loaded
    n_main_recs = (P1_STEPS // EVAL_EVERY + 1) + (P2_STEPS // EVAL_EVERY + 1)  # duplicates gone
    per_rec = len(VAL_LENGTHS) + 3 + len(CURVE_OOD) + 1
    tune = (P1_STEPS // EVAL_EVERY + 1) * (len(VAL_LENGTHS) + 3 + 1)
    assert len(c) == N_MAIN * n_main_recs * per_rec + tune
    assert list(c.columns) == collect.CURVE_COLS
    assert set(c["which"]) == {"val", "val_group", "curve_ood", "val_token_acc"}
    assert not c.duplicated(["run_id", "phase", "step", "which", "key"]).any()
    assert set(c.loc[c["which"] == "val_token_acc", "key"]) == {"all"}
    assert set(c.loc[c["which"] == "val_group", "key"]) == {"A", "B", "C"}
    v = c[c["which"] == "val"]
    assert (v["length"] == v["key"].astype(int)).all()
    assert c.loc[c["which"] == "val_group", "length"].isna().all()
    # Phase 2 steps are offset by train.steps on the global axis
    p2 = c[c["phase"] == 2]
    assert ((p2["global_step"] - p2["step"]) == P1_STEPS).all()
    assert (c.loc[c["phase"] == 1, "global_step"] == c.loc[c["phase"] == 1, "step"]).all()
    # the duplicate end-of-phase record wins (last in file)
    a = c[(c["run_id"] == "sinusoidal_R_s0") & (c["phase"] == 1) & (c["step"] == P1_STEPS) &
          (c["which"] == "val_group") & (c["key"] == "A")]
    assert len(a) == 1
    assert a["value"].iloc[0] == pytest.approx(curve_value("R", 0, 1, P1_STEPS, "A") + 0.001)
    assert not c["run_id"].eq("sinusoidal_SLr-late_s0").any()


def test_load_losses(results_dir):
    lo = collect.load_losses(results_dir)
    per_run = P1_STEPS // 25 + P2_STEPS // 25
    assert len(lo) == N_MAIN * per_run + P1_STEPS // 25
    assert {"loss", "lr", "tokens_nonpad", "tokens_padded", "examples",
            "global_step"} <= set(lo.columns)
    p2 = lo[lo["phase"] == 2]
    assert ((p2["global_step"] - p2["step"]) == P1_STEPS).all()


def test_load_probes(probes_df):
    p = probes_df
    # phase 1 for all 24 main runs, phase 2 for the 6 NoPE runs; 4 lengths each
    assert len(p) == (N_MAIN + 6) * 4
    for c in ["run_id", "pe", "schedule", "base_schedule", "is_twin", "seed", "tag", "phase",
              "readout", "length", "n", "min_acc", "id_succ_acc"]:
        assert c in p.columns, c
    assert "nested" not in p.columns
    assert not p["schedule"].eq("SLr-late").any()
    r = p[(p["run_id"] == "sinusoidal_SL_s1") & (p["length"] == 30)].iloc[0]
    assert r["min_acc"] == pytest.approx(0.7) and r["id_succ_acc"] == pytest.approx(0.76)
    assert r["readout"] == "decoder_basis"


def test_load_per_position(results_dir, pp_df):
    sub = collect.load_per_position(results_dir, set_names=["len_25", "len_30"])
    assert set(sub["set_name"]) == {"len_25", "len_30"}
    assert len(sub) == N_MAIN * 2 * (25 + 30)
    g = sub[(sub["run_id"] == "sinusoidal_R_s0") & (sub["phase"] == 1) &
            (sub["set_name"] == "len_25")]
    assert list(g["position"]) == list(range(1, 26))
    assert g["rel_position"].iloc[-1] == pytest.approx(1.0)
    assert {"acc", "position", "rel_position", "length", "kind", "r"} <= set(pp_df.columns)


def test_collect_cli_and_csv_roundtrip(results_dir, loaded, tmp_path):
    m, c = loaded
    collect.main(["--results", str(results_dir), "--out", str(tmp_path)])
    for name in ("metrics.csv", "curves.csv", "probes.csv"):
        assert (tmp_path / name).is_file(), name
    m2 = collect.read_csv(tmp_path / "metrics.csv")
    assert len(m2) == len(m)
    assert (m2["tag"] == "").sum() == (m["tag"] == "").sum()  # "" survives the round trip
    assert str(m2["r"].dtype) == "Int64" and m2["r"].isna().sum() == m["r"].isna().sum()
    assert m2["is_twin"].dtype == bool
    # groupby on the round-tripped table gives the same bucket means
    b1 = stats.aggregate_buckets(m[m["tag"] == ""])
    b2 = stats.aggregate_buckets(m2[m2["tag"] == ""])
    pd.testing.assert_series_equal(b1["exact_match"].reset_index(drop=True),
                                   b2["exact_match"].reset_index(drop=True))
    c2 = collect.read_csv(tmp_path / "curves.csv")
    assert len(c2) == len(c)


def test_collect_on_missing_or_empty_dir(tmp_path):
    m, c = collect.load_runs(tmp_path / "does_not_exist")
    assert m.empty and c.empty and "exact_match" in m.columns
    assert "experiment_hash" in m.columns
    assert collect.load_probes(tmp_path).empty


def _hash_warnings(results_dir):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = collect.load_runs(results_dir)
    return out, [str(x.message) for x in w if "experiment_hash" in str(x.message)]


def test_experiment_hash_column_and_no_warning_without_hashes(results_dir, loaded):
    # the synthetic dir predates the hashes: the column exists (all None) and nothing warns
    _, msgs = _hash_warnings(results_dir)
    assert msgs == []
    m, c = loaded
    for df in (m, c):
        assert list(df.columns[:len(collect.RUN_COLS)]) == collect.RUN_COLS
        assert df["experiment_hash"].isna().all()


def test_mixed_experiment_hashes_warn(tmp_path):
    root = tmp_path / "res"
    h1, h2, h3 = "a" * 64, "b" * 64, "c" * 64
    write_run(root, "sinusoidal", "R", 0, experiment_hash=h1)
    write_run(root, "sinusoidal", "SL", 0, experiment_hash=h1)
    write_run(root, "sinusoidal", "R", 1, experiment_hash=h2)  # e.g. after base.yaml changed
    write_run(root, "sinusoidal", "SL", 1)  # made by older code: no hash
    write_run(root, "none", "R", 0, experiment_hash=h2)  # alone in its (pe, tag): fine
    for seed in (0, 1):  # another tag with another config (the lr grid): its own group, fine
        write_run(root, "sinusoidal", "R", seed, tag="lr1", experiment_hash=h3, phases=(1,),
                  probe_phases=())
    (m, c), msgs = _hash_warnings(root)
    assert len(msgs) == 1, msgs
    msg = msgs[0]
    assert "pe='sinusoidal', tag=''" in msg and "3 different experiment configs" in msg
    assert "aaaaaaaaaa: sinusoidal_R_s0, sinusoidal_SL_s0" in msg
    assert "bbbbbbbbbb: sinusoidal_R_s1" in msg and "none: sinusoidal_SL_s1" in msg
    assert "lr1" not in msg and "none_R_s0" not in msg

    want = {"sinusoidal_R_s0": h1, "sinusoidal_SL_s0": h1, "sinusoidal_R_s1": h2,
            "sinusoidal_SL_s1": None, "none_R_s0": h2, "sinusoidal_R_s0_lr1": h3}
    p = collect.load_probes(root)
    for name, df in (("metrics", m), ("curves", c), ("probes", p)):
        got = df.drop_duplicates("run_id").set_index("run_id")["experiment_hash"].to_dict()
        for rid, h in want.items():
            if name == "probes" and rid.endswith("lr1"):
                continue  # no probe files for the tag runs
            assert got[rid] == h, (name, rid)
    # a consistent dir: no warning
    root2 = tmp_path / "res2"
    for s in ("R", "SL"):
        write_run(root2, "sinusoidal", s, 0, experiment_hash=h1)
    assert _hash_warnings(root2)[1] == []


# ==============================================================================================
# stats
# ==============================================================================================
def test_summarize_hand_values():
    df = pd.DataFrame({"run_id": ["a", "b", "c", "d"], "schedule": ["X", "X", "X", "Y"],
                       "v": [1.0, 2.0, 3.0, 5.0]})
    s = stats.summarize(df, "v", by="schedule").set_index("schedule")
    assert s.loc["X", "mean"] == pytest.approx(2.0)
    assert s.loc["X", "std"] == pytest.approx(1.0)
    assert s.loc["X", "n"] == 3
    assert 1.0 <= s.loc["X", "ci_lo"] <= 2.0 <= s.loc["X", "ci_hi"] <= 3.0
    assert s.loc["Y", "n"] == 1 and math.isnan(s.loc["Y", "std"]) and math.isnan(
        s.loc["Y", "ci_lo"])
    # fixed RNG: identical CIs on repeat, and independent of the other groups present
    s2 = stats.summarize(df[df["schedule"] == "X"], "v", by="schedule").set_index("schedule")
    assert s2.loc["X", "ci_lo"] == s.loc["X", "ci_lo"]
    assert s2.loc["X", "ci_hi"] == s.loc["X", "ci_hi"]


def test_summarize_averages_within_run_first():
    # run a has two rows (two lengths): it must count once, as its mean 2.0
    df = pd.DataFrame({"run_id": ["a", "a", "b"], "g": ["X"] * 3, "v": [1.0, 3.0, 4.0]})
    s = stats.summarize(df, "v", by=["g"]).iloc[0]
    assert s["n"] == 2 and s["mean"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(np.std([2.0, 4.0], ddof=1))


def _paired_df(a, b, seeds=None, pe="sinusoidal"):
    seeds = list(range(len(a))) if seeds is None else seeds
    rows = [{"pe": pe, "phase": 1, "schedule": "A", "seed": s, "run_id": f"A{s}{pe}", "v": x}
            for s, x in zip(seeds, a)]
    rows += [{"pe": pe, "phase": 1, "schedule": "B", "seed": s, "run_id": f"B{s}{pe}", "v": x}
             for s, x in zip(seeds, b)]
    return pd.DataFrame(rows)


def test_paired_compare_known_differences():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    diffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    df = _paired_df(a, a - diffs)
    extra = pd.DataFrame([{"pe": "sinusoidal", "phase": 1, "schedule": "B", "seed": 9,
                           "run_id": "B9", "v": 0.0}])  # unpaired seed: ignored
    r = stats.paired_compare(pd.concat([df, extra]), "A", "B", "v").iloc[0]
    assert r["n"] == 5 and r["n_nonzero"] == 5 and r["seeds"] == "0 1 2 3 4"
    assert r["mean_diff"] == pytest.approx(0.3)
    assert r["mean_a"] == pytest.approx(0.7) and r["mean_b"] == pytest.approx(0.4)
    assert r["std_diff"] == pytest.approx(np.std(diffs, ddof=1))
    assert r["dz"] == pytest.approx(0.3 / np.std(diffs, ddof=1))
    assert 0.1 <= r["ci_lo"] < 0.3 < r["ci_hi"] <= 0.5
    assert r["p_value"] == pytest.approx(0.0625)  # the n=5 floor
    assert r["p_min_possible"] == pytest.approx(0.0625)
    assert "0.0625" in stats.paired_compare.__doc__
    # reversed direction flips the sign
    r2 = stats.paired_compare(df, "B", "A", "v").iloc[0]
    assert r2["mean_diff"] == pytest.approx(-0.3)


def test_paired_compare_edge_cases():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no scipy warnings may leak
        z = stats.paired_compare(_paired_df(a, a), "A", "B", "v").iloc[0]
    assert z["mean_diff"] == 0 and z["p_value"] == 1.0 and z["n_nonzero"] == 0
    assert z["ci_lo"] == 0 and z["ci_hi"] == 0
    # one zero difference: exact test on the 4 non-zero ones
    one0 = stats.paired_compare(_paired_df(a, a - np.array([0, .1, .2, .3, .4])), "A", "B",
                                "v").iloc[0]
    assert one0["n"] == 5 and one0["n_nonzero"] == 4
    assert one0["p_value"] == pytest.approx(0.125) and one0["p_min_possible"] == 0.125
    # n = 1
    n1 = stats.paired_compare(_paired_df([0.5], [0.4]), "A", "B", "v").iloc[0]
    assert n1["n"] == 1 and n1["mean_diff"] == pytest.approx(0.1)
    assert math.isnan(n1["std_diff"]) and math.isnan(n1["ci_lo"])
    assert n1["n_nonzero"] == 1 and n1["p_value"] == 1.0 and n1["p_min_possible"] == 1.0
    # groups are kept separate (within pe), and a condition missing from a group is skipped
    df = pd.concat([_paired_df(a, a - 0.1), _paired_df(a, a - 0.2, pe="none")])
    out = stats.paired_compare(df, "A", "B", "v").set_index("pe")
    assert out.loc["sinusoidal", "mean_diff"] == pytest.approx(0.1)
    assert out.loc["none", "mean_diff"] == pytest.approx(0.2)
    assert stats.paired_compare(df, "A", "MISSING", "v").empty


def brute_force_signflip_p(d, alternative="two-sided"):
    """Wilcoxon signed-rank p by enumerating all 2**n sign flips of the mid-ranks of |d|."""
    d = np.asarray(d, dtype=float)
    ranks = rankdata(np.abs(d))
    obs = ranks[d > 0].sum()
    null = np.array([ranks[np.array(s, dtype=bool)].sum()
                     for s in itertools.product((0, 1), repeat=len(d))])
    ge, le = np.mean(null >= obs - 1e-9), np.mean(null <= obs + 1e-9)
    return {"greater": ge, "less": le, "two-sided": min(1.0, 2 * min(ge, le))}[alternative]


def test_wilcoxon_exact_under_ties():
    d = [.003, -.001, .001, .003, .001]  # |d| ties: three at .001, two at .003
    assert brute_force_signflip_p(d) == pytest.approx(0.25)  # 8 of 32 sign flips
    for alt in ("two-sided", "greater", "less"):
        stat, p, n_nz = stats.wilcoxon_signed_rank(d, alternative=alt)
        assert n_nz == 5 and p == pytest.approx(brute_force_signflip_p(d, alt)), alt
    assert stats.wilcoxon_signed_rank(d)[0] == 2.0  # min(W+, W-) = min(13, 2)
    # through paired_compare, where a - (a - d) adds float noise to d
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    r = stats.paired_compare(_paired_df(a, a - np.array(d)), "A", "B", "v").iloc[0]
    assert r["p_value"] == pytest.approx(0.25) and r["n_nonzero"] == 5
    assert r["p_min_possible"] == pytest.approx(0.0625)

    # Differences of k/1000 accuracies that are equal up to the last bits must still tie.
    # Without the rounding these ranks would be 3/4/5 instead of 4,4,4, and p would be 0.3125.
    noisy = [0.021 - 0.018, 0.141 - 0.138, -(0.563 - 0.560), 0.005 - 0.004, 0.009 - 0.008]
    clean = [.003, .003, -.003, .001, .001]
    assert len({abs(x) for x in noisy[:3]}) == 3  # really three different floats
    assert brute_force_signflip_p(noisy) == pytest.approx(0.3125)
    assert brute_force_signflip_p(clean) == pytest.approx(0.4375)
    assert stats.wilcoxon_signed_rank(noisy)[1] == pytest.approx(0.4375)

    # no ties: the classic exact values, also above EXACT_MAX_N (scipy's exact table, as before);
    # many tied differences: normal approximation, still fast
    assert stats.wilcoxon_signed_rank([.1, .2, .3, .4, .5])[1] == pytest.approx(0.0625)
    assert stats.wilcoxon_signed_rank([.1, .2, -.3, .4, .5])[1] == pytest.approx(
        brute_force_signflip_p([.1, .2, -.3, .4, .5]))
    d14 = np.arange(1, 15) * 0.01 * np.resize([1, 1, -1], 14)
    assert stats.wilcoxon_signed_rank(d14)[1] == pytest.approx(
        wilcoxon(d14, method="exact").pvalue)
    big = stats.wilcoxon_signed_rank(np.repeat([0.01, 0.02, 0.03], 10))
    assert big[2] == 30 and 0 < big[1] < 1e-4


def test_paired_compare_on_raw_metrics_df(loaded, tmp_path, results_dir):
    """Per eval set on the real metrics table, where r is <NA> for every uniform set (NaN group
    keys must not drop rows), also after a CSV round trip."""
    m, _ = loaded
    collect.main(["--results", str(results_dir), "--out", str(tmp_path)])
    m2 = collect.read_csv(tmp_path / "metrics.csv")
    n_sets = len(TEST_LENGTHS) + len(REP_SETS)
    for df in (m, m2):
        out = stats.paired_compare(df[df["tag"] == ""], "SL", "SL-shuf", "exact_match")
        assert len(out) == 2 * 2 * n_sets  # 2 PEs x 2 phases x 17 sets
        assert out["mean_diff"].to_numpy() == pytest.approx(-0.05)
        u = out[(out["set_name"] == "len_25") & (out["pe"] == "sinusoidal")]
        r = out[(out["set_name"] == "rep_30_r4") & (out["pe"] == "sinusoidal")]
        assert len(u) == 2 and len(r) == 2 and (u["n"] == 3).all()
        assert u["r"].isna().all() and (r["r"] == 4).all()


def test_twin_comparison_needs_pairs_within_a_pe(loaded, tmp_path):
    m, _ = loaded  # SL only under sinusoidal, SL-shuf only under NoPE: no pair inside a PE
    odd = m[((m["schedule"] == "SL") & (m["pe"] == "sinusoidal")) |
            ((m["schedule"] == "SL-shuf") & (m["pe"] == "none"))]
    with pytest.raises(NoDataError):
        plots.plot_twin_comparison(odd, tmp_path / "x", dpi=PLOT_DPI)


def test_twin_of():
    assert stats.twin_of("R") is None
    assert stats.twin_of("SL") == "SL-shuf"
    assert stats.twin_of("LS") == "SL-shuf"
    assert stats.twin_of("SLr-mid") == "SLr-mid-shuf"
    assert stats.twin_of("SLr-mid-shuf") is None


def test_aggregate_buckets(bucket_df, loaded):
    b = bucket_df
    assert set(b["bucket"]) == {"in_dist", "near_ood", "far_ood"}
    assert len(b) == N_MAIN * 2 * 3
    r = b[(b["run_id"] == "sinusoidal_R_s1") & (b["phase"] == 1)].set_index("bucket")
    assert r.loc["in_dist", "exact_match"] == pytest.approx(0.91)
    assert r.loc["near_ood", "exact_match"] == pytest.approx(np.mean([.8, .7, .6, .5]) + .01)
    # far-OOD = mean over uniform 25/30/40/50 only (rep_30_* must not leak in)
    assert r.loc["far_ood", "exact_match"] == pytest.approx(np.mean([.4, .3, .25, .21]) + .01)
    assert r.loc["in_dist", "n_lengths"] == 4 and r.loc["far_ood", "n_lengths"] == 4
    # id_only tuning run: only the in-dist bucket exists
    m = loaded[0]
    t = stats.aggregate_buckets(m[m["tag"] == "lr1e-3"])
    assert set(t["bucket"]) == {"in_dist"}


def test_compare_twins_and_reference(bucket_df):
    b = bucket_df[bucket_df["bucket"] == "near_ood"]
    tw = stats.compare_twins(b, "exact_match")
    got = {(r.pe, r.phase, r.cond_a, r.cond_b): r for r in tw.itertuples()}
    assert set(k[2:] for k in got) == {("SL", "SL-shuf"), ("LS", "SL-shuf"),
                                       ("SLr-mid", "SLr-mid-shuf")}
    assert got[("sinusoidal", 1, "SL", "SL-shuf")].mean_diff == pytest.approx(-0.05)
    assert got[("sinusoidal", 1, "LS", "SL-shuf")].mean_diff == pytest.approx(-0.15)
    assert got[("sinusoidal", 2, "SLr-mid", "SLr-mid-shuf")].mean_diff == pytest.approx(0.01)
    assert got[("sinusoidal", 1, "SL", "SL-shuf")].n == 3
    assert got[("none", 1, "SL", "SL-shuf")].n == 2
    assert ("none", 1, "LS", "SL-shuf") not in got
    ref = stats.compare_to_reference(b, "exact_match", "R")
    x = ref[(ref["pe"] == "sinusoidal") & (ref["phase"] == 1) & (ref["cond_a"] == "SLr-mid")]
    assert x["mean_diff"].iloc[0] == pytest.approx(0.03)
    assert (ref["cond_b"] == "R").all() and not (ref["cond_a"] == "R").any()


def test_latex_table(bucket_df):
    g = bucket_df[(bucket_df["pe"] == "sinusoidal") & (bucket_df["phase"] == 1)]
    tex = stats.latex_table(g, "exact_match", label="tab:main")
    for tok in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{tabular}{lccc}",
                r"\label{tab:main}", "In-dist (5--20)", "Near-OOD (21--24)", "Far-OOD (25--50)",
                "3 seeds"):
        assert tok in tex, tok
    assert r"R & $91.0 \pm 1.0$" in tex  # (0.90, 0.91, 0.92) -> 91.0 +- 1.0
    assert r"SLr-mid & $\mathbf{94.0} \pm 1.0$" in tex  # best in-dist mean is bold
    lines = tex.splitlines()
    order = [ln.split(" & ")[0] for ln in lines if " & " in ln][1:]
    assert order == ["R", "SL", "SLr-mid", "LS", "SL-shuf", "SLr-mid-shuf"]
    # a \midrule separates base schedules from twins
    i_ls = next(i for i, ln in enumerate(lines) if ln.startswith("LS &"))
    assert lines[i_ls + 1] == r"\midrule"
    with pytest.raises(ValueError):
        stats.latex_table(bucket_df, "exact_match")  # mixes pe and phase
    one = g[g["seed"] == 0].assign(schedule=lambda x: x["schedule"].str.replace("-", "_"))
    tex1 = stats.latex_table(one, "exact_match", table_env=False)
    assert r"SL\_shuf & $" in tex1 and r"\pm" not in tex1  # n=1: mean only; "_" escaped
    assert "edit" not in tex1


def test_stats_cli(results_dir, tmp_path):
    stats.main(["--results", str(results_dir), "--out", str(tmp_path)])
    assert (tmp_path / "table_exact_match_sinusoidal_phase1.tex").is_file()
    assert (tmp_path / "table_exact_match_none_phase2.tex").is_file()
    tw = pd.read_csv(tmp_path / "paired_twins_exact_match.csv")
    assert {"mean_diff", "ci_lo", "ci_hi", "p_value", "n"} <= set(tw.columns)


# ==============================================================================================
# plots
# ==============================================================================================
def test_stage_boundaries():
    assert plots.stage_blocks("SL", 20000) == [(0, 6666, "A"), (6666, 13333, "B"),
                                               (13333, 20000, "C")]
    assert plots.stage_blocks("LS", 20000) == [(0, 6667, "C"), (6667, 13334, "B"),
                                               (13334, 20000, "A")]
    assert plots.stage_boundaries("SLr-early", 20000) == [3000, 6000]
    assert plots.stage_boundaries("SLr-late", 20000) == [10000, 16000]
    assert plots.stage_boundaries("R", 20000) == []
    assert plots.stage_boundaries("SL-shuf", 20000) == []
    # agrees with the actual data order from src.data (batch_size 1 -> one example per step)
    lo = {"A": 5, "AB": 5, "ABC": 5, "B": 11, "C": 16}
    hi = {"A": 10, "AB": 15, "ABC": 20, "B": 15, "C": 20}
    for sched in ("SL", "LS", "SLr-mid", "SLr-early"):
        seq = length_sequence(sched, 20, 1, seed=0)
        for a, b, g in plots.stage_blocks(sched, 20):
            assert lo[g] <= seq[a:b].min() and seq[a:b].max() <= hi[g], (sched, a, b, g)
    seq = length_sequence("LS", 10, 1, seed=0)
    assert seq[:4].min() >= 16 and 11 <= seq[4:7].min() and seq[4:7].max() <= 15
    assert seq[7:].max() <= 10


def test_schedule_styles_consistent():
    assert plots.schedule_color("SL-shuf") == plots.schedule_color("SL")
    assert plots.schedule_style("SL-shuf")["ls"] != "-"
    assert plots.schedule_style("SL")["ls"] == "-"
    assert plots.schedule_style("SLr-mid-shuf")["mfc"] == plots.SURFACE  # hollow
    assert len(set(plots.SCHEDULE_COLORS.values())) == 6
    assert len(set(plots.SCHEDULE_MARKERS.values())) == 6


def _check_files(paths):
    assert len(paths) == 2
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    for p in paths:
        assert p.is_file() and p.stat().st_size > 1000, p


def _all_plots(m, c, p, pp, out, tag=""):
    """Run every plot function; return {name: paths or the NoDataError message}."""
    calls = {
        "fig1": lambda: plots.plot_length_curves(m, out / "fig1", tag=tag, dpi=PLOT_DPI),
        "fig2": lambda: plots.plot_twin_comparison(m, out / "fig2", tag=tag, dpi=PLOT_DPI),
        "fig3": lambda: plots.plot_learning_curves(c, out / "fig3", tag=tag, dpi=PLOT_DPI),
        "fig4": lambda: plots.plot_phase_comparison(m, out / "fig4", tag=tag, dpi=PLOT_DPI),
        # figs 5/8 get two lengths each to keep the file fast (defaults run in the CLI test)
        "fig5": lambda: plots.plot_error_types(m, out / "fig5", lengths=(25, 50), tag=tag,
                                               dpi=PLOT_DPI),
        "fig6": lambda: plots.plot_probe_accuracy(p, out / "fig6", tag=tag, dpi=PLOT_DPI),
        "fig7": lambda: plots.plot_rep(m, out / "fig7", tag=tag, dpi=PLOT_DPI),
        "fig8": lambda: plots.plot_per_position(pp, out / "fig8", lengths=(25, 40), tag=tag,
                                                dpi=PLOT_DPI),
    }
    res = {}
    for k, fn in calls.items():
        try:
            res[k] = fn()
        except NoDataError as e:
            res[k] = str(e)
    return res


def test_plots_full_data(loaded, probes_df, pp_df, tmp_path):
    m, c = loaded
    res = _all_plots(m, c, probes_df, pp_df, tmp_path)
    for k, v in res.items():
        assert not isinstance(v, str), f"{k} skipped: {v}"
        _check_files(v)
    # optional variants (twins drawn dashed/hollow, bootstrap band); an explicit .pdf suffix
    # still writes both formats
    _check_files(plots.plot_length_curves(m, tmp_path / "f1t.pdf", include_twins=True,
                                          band="ci", phase=2, dpi=PLOT_DPI))


# Degradations: a single seed + missing PE here; missing Phase 2 + no probes + missing PE in
# test_plots_cli_degraded (end to end). Scenarios keep one PE (one panel row) to stay fast.
@pytest.mark.parametrize("scenario", ["one_seed_one_pe", "id_only_tag"])
def test_plots_degraded_data(loaded, probes_df, pp_df, tmp_path, scenario):
    m, c = loaded
    p, pp, tag = probes_df, pp_df, ""
    expect_skip = set()
    if scenario == "one_seed_one_pe":  # missing PE (no NoPE runs) and a single seed
        f = lambda d: d[(d["seed"] == 0) & (d["pe"] == "sinusoidal")]  # noqa: E731
    else:  # only the id_only tuning run: no OOD lengths, no twins, no Phase 2, no probes
        f, tag = (lambda d: d), "lr1e-3"
        expect_skip = {"fig2", "fig4", "fig5", "fig6", "fig8"}
    res = _all_plots(f(m), f(c), f(p), f(pp), tmp_path, tag=tag)
    for k, v in res.items():
        if k in expect_skip:
            assert isinstance(v, str), f"{k} should have raised NoDataError"
        else:
            assert not isinstance(v, str), f"{k} skipped: {v}"
            _check_files(v)


def test_plots_no_probes_and_empty(tmp_path):
    with pytest.raises(NoDataError):
        plots.plot_probe_accuracy(collect.load_probes(tmp_path), tmp_path / "x", dpi=PLOT_DPI)
    with pytest.raises(NoDataError):
        plots.plot_length_curves(pd.DataFrame(columns=collect.META_COLS), tmp_path / "x")


def test_plots_cli_degraded(degraded_results_dir, tmp_path):
    out = tmp_path / "figs"
    res = plots.main(["--results", str(degraded_results_dir), "--out", str(out), "--dpi",
                      str(PLOT_DPI)])
    skipped = {k for k, v in res.items() if isinstance(v, str)}
    assert skipped == {"fig4_phase1_vs_phase2", "fig6_probes_p1"}, skipped
    made = {k for k in res if k not in skipped}
    assert made == {"fig1_length_curves_p1", "fig2_order_vs_content_p1", "fig3_learning_curves",
                    "fig5_error_types_p1", "fig7_rep_p1", "fig8_per_position_p1"}
    for k in made:
        assert (out / f"{k}.pdf").is_file() and (out / f"{k}.png").is_file()
    # a results dir with nothing finished: every figure is skipped, nothing crashes
    res2 = plots.main(["--results", str(tmp_path / "nothing"), "--out", str(out)])
    assert res2 and all(isinstance(v, str) for v in res2.values())


def test_complete_seed_filter_drops_partial_seeds(monkeypatch):
    from src.analysis.collect import complete_seed_filter
    monkeypatch.delenv("CS652_ALL_SEEDS", raising=False)
    metas = []
    for seed, scheds in [(0, ["R", "SL"]), (1, ["R", "SL"]), (2, ["R"])]:
        for s in scheds:
            metas.append({"run_id": f"sinusoidal_{s}_s{seed}", "pe": "sinusoidal", "tag": "",
                          "schedule": s, "seed": seed})
    metas.append({"run_id": "none_R_s0", "pe": "none", "tag": "", "schedule": "R", "seed": 0})
    import pytest as _pt
    with _pt.warns(UserWarning, match="incomplete seed"):
        drop = complete_seed_filter(metas)
    assert drop == {"sinusoidal_R_s2"}
    monkeypatch.setenv("CS652_ALL_SEEDS", "1")
    assert complete_seed_filter(metas) == set()
