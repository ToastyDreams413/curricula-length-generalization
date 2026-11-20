"""Tests for src/train.py, configs/ and scripts/sweep.py. CPU only; tiny smoke config."""

import dataclasses
import glob
import importlib.util
import json
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from src import train
from src.data import (GRID_CONDITIONS, PAD, TEST_SEED, VAL_SEED, VOCAB_SIZE, encode_batch,
                      pool_examples, uniform_set)
from src.model import count_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "configs", "base.yaml")
SMOKE = os.path.join(ROOT, "configs", "smoke.yaml")
TUNE = os.path.join(ROOT, "configs", "tune.yaml")

LOSS_KEYS = {"type", "phase", "step", "loss", "lr", "tokens_nonpad", "tokens_padded", "examples"}
EVAL_KEYS = {"type", "phase", "step", "val", "val_group", "val_token_acc", "curve_ood"}
EVAL_JSON_KEYS = {"run_id", "phase", "steps", "tokens_nonpad", "tokens_padded", "wall_time_s",
                  "test_sets"}
METRIC_KEYS = {"kind", "length", "r", "n", "exact_match", "token_acc", "per_position_acc",
               "prefix_len", "prefix_frac", "inversions", "adjacent_inversions", "multiset_diff",
               "multiset_ok", "edit_distance", "edit_distance_norm", "error_types", "samples"}
DONE_KEYS = {"run_id", "finished", "phase1_steps", "phase2_steps", "tokens_nonpad_phase1",
             "tokens_padded_phase1", "examples_phase1", "tokens_nonpad_phase2",
             "tokens_padded_phase2", "examples_phase2", "n_params", "wall_time_s",
             "config_hash", "experiment_hash", "overlap"}

# Even smaller than smoke: for tests that only need the plumbing. Narrow groups (lengths 5..10)
# keep the learning-curve decodes cheap.
FAST = ["--set", "data.groups={A: [5, 6], B: [7, 8], C: [9, 10]}",
        "--set", "train.steps=6", "--set", "train.log_every=3", "--set", "phase2.steps=3",
        "--set", "eval.every=100", "--set", "eval.val_per_length=2", "--set", "eval.test_lengths=[5]",
        "--set", "eval.rep_per_set=0", "--set", "eval.curve_ood_lengths=[]"]


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    """Tiny models run faster (and with less timing noise) on one CPU thread."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def smoke_argv(out_dir, *extra):
    return ["--config", BASE, "--config", SMOKE, "--out-dir", str(out_dir), *extra]


def smoke_cfg(*sets, **kw):
    return train.resolve_config([BASE, SMOKE], sets=list(sets), **kw)


def read_log(run_dir):
    with open(os.path.join(run_dir, "train_log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def records(run_dir, kind, phase=None):
    return [r for r in read_log(run_dir)
            if r["type"] == kind and (phase is None or r["phase"] == phase)]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One full smoke run (R, seed 0). make_batches is wrapped so the tests can see exactly the
    batches the training loop consumed."""
    out = tmp_path_factory.mktemp("smoke")
    consumed = {1: [], 2: []}
    orig = train.make_batches

    def recording(cfg, phase):
        for b in orig(cfg, phase):
            consumed[phase].append(b)
            yield b

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "make_batches", recording)
        run_dir = train.main(smoke_argv(out))
    return {"dir": run_dir, "out": str(out), "consumed": consumed}


# ---------------------------------------------------------------------------------------- 1
def test_smoke_run_writes_every_output(smoke_run):
    d = smoke_run["dir"]
    assert os.path.basename(d) == "sinusoidal_R_s0"
    for f in ("config.yaml", "train_log.jsonl", "eval_phase1.json", "eval_phase2.json",
              "model_phase1.pt", "model_phase2.pt", "done.json"):
        assert os.path.isfile(os.path.join(d, f)), f
    assert not glob.glob(os.path.join(d, "*.tmp"))

    # config.yaml: the fully resolved config plus run_id
    with open(os.path.join(d, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["run_id"] == "sinusoidal_R_s0"
    expected = smoke_cfg(out_dir=smoke_run["out"])
    assert {k: v for k, v in cfg.items() if k != "run_id"} == expected
    assert train.resolve_config([os.path.join(d, "config.yaml")]) == expected  # reusable as input

    # train_log.jsonl
    recs = read_log(d)
    assert {r["type"] for r in recs} == {"loss", "eval"}
    for r in recs:
        assert set(r) == (LOSS_KEYS if r["type"] == "loss" else EVAL_KEYS)
    for phase, steps, eval_steps, loss_steps in [(1, 40, [0, 20, 40], [10, 20, 30, 40]),
                                                 (2, 10, [0, 10], [10])]:
        assert [r["step"] for r in records(d, "eval", phase)] == eval_steps
        assert [r["step"] for r in records(d, "loss", phase)] == loss_steps
    for r in records(d, "eval"):
        assert list(r["val"]) == [str(L) for L in range(5, 21)]
        assert list(r["val_group"]) == ["A", "B", "C"]
        assert list(r["curve_ood"]) == ["22", "25"]
        for g, (lo, hi) in {"A": (5, 10), "B": (11, 15), "C": (16, 20)}.items():
            assert r["val_group"][g] == pytest.approx(
                np.mean([r["val"][str(L)] for L in range(lo, hi + 1)]))
        assert 0.0 <= r["val_token_acc"] <= 1.0
    losses = [r["loss"] for r in records(d, "loss", 1)]
    assert all(math.isfinite(x) for x in losses) and losses[-1] < losses[0]  # it learns a bit

    # eval_phase{1,2}.json
    names = {f"len_{L}" for L in (5, 12, 20, 22, 25)} | {
        f"rep_{L}_r{r}" for L in (10, 20, 30) for r in (2, 5)}
    for phase, steps in [(1, 40), (2, 10)]:
        ev = read_json(os.path.join(d, f"eval_phase{phase}.json"))
        assert set(ev) == EVAL_JSON_KEYS
        assert ev["run_id"] == "sinusoidal_R_s0" and ev["phase"] == phase
        assert ev["steps"] == steps and ev["tokens_nonpad"] > 0 and ev["wall_time_s"] > 0
        assert ev["tokens_padded"] >= ev["tokens_nonpad"]
        assert set(ev["test_sets"]) == names
        for name, m in ev["test_sets"].items():
            assert set(m) == METRIC_KEYS, name
            assert m["n"] == (16 if m["kind"] == "uniform" else 6)
            assert len(m["per_position_acc"]) == m["length"] and len(m["samples"]) == 3

    # model_phase{1,2}.pt
    sds = []
    for phase in (1, 2):
        ck = torch.load(os.path.join(d, f"model_phase{phase}.pt"), weights_only=True)
        assert set(ck) == {"state_dict", "model_config", "run_config"}
        assert ck["model_config"] == dataclasses.asdict(train.model_config(expected))
        assert ck["run_config"] == cfg
        m = train.make_model(expected)
        m.load_state_dict(ck["state_dict"])
        sds.append(ck["state_dict"])
    assert any(not torch.equal(sds[0][k], sds[1][k]) for k in sds[0])  # phase 2 trained

    # done.json
    done = read_json(os.path.join(d, "done.json"))
    assert DONE_KEYS <= set(done)
    assert done["run_id"] == "sinusoidal_R_s0"
    assert (done["phase1_steps"], done["phase2_steps"]) == (40, 10)
    assert done["examples_phase1"] == 40 * 12 and done["examples_phase2"] == 10 * 12
    assert done["n_params"] == count_params(train.make_model(expected))
    for p in (1, 2):
        ev = read_json(os.path.join(d, f"eval_phase{p}.json"))
        assert done[f"tokens_nonpad_phase{p}"] == ev["tokens_nonpad"]
        assert done[f"tokens_padded_phase{p}"] == ev["tokens_padded"]
    # done.json is the last file written
    assert os.path.getmtime(os.path.join(d, "done.json")) >= max(
        os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))


# ---------------------------------------------------------------------------------------- 2
def test_rerun_refused_and_crashed_dir_restarted(tmp_path, capsys):
    argv = smoke_argv(tmp_path, *FAST)
    run_dir = train.main(argv)
    done = os.path.join(run_dir, "done.json")
    with open(done, encoding="utf-8") as f:
        before = f.read()

    with pytest.raises(SystemExit) as ei:
        train.main(argv)
    assert "--overwrite" in str(ei.value.code)
    with open(done, encoding="utf-8") as f:
        assert f.read() == before  # untouched

    capsys.readouterr()
    train.main(argv + ["--overwrite"])
    assert "--overwrite: clearing" in capsys.readouterr().out
    assert os.path.isfile(done)

    # Simulate a crash: no done.json, a stray file, a truncated log.
    os.remove(done)
    with open(os.path.join(run_dir, "stray.txt"), "w") as f:
        f.write("left over")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as f:
        f.write('{"type": "loss"')
    assert train.main(argv) == run_dir  # no --overwrite needed
    assert "without done.json" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(run_dir, "stray.txt"))
    assert os.path.isfile(done)
    assert all(r["type"] in ("loss", "eval") for r in read_log(run_dir))


# ---------------------------------------------------------------------------------------- 3
def test_id_only_never_evaluates_beyond_train_range(tmp_path):
    # Training range narrowed to 5..18, so the filter must also drop 19 and 20 (a hard-coded 20
    # would let them through). Test, rep and curve lengths all include lengths above 18.
    run_dir = train.main(smoke_argv(
        tmp_path, "--set", "eval.id_only=true", "--set", "data.groups.C=[16,18]",
        "--set", "eval.test_lengths=[5,12,18,19,20,22,25,40]",
        "--set", "eval.rep_lengths=[10,18,20,30]", "--set", "eval.curve_ood_lengths=[19,22,25]"))
    max_len = 18
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["eval"]["id_only"] and cfg["eval"]["curve_ood_lengths"] == [19, 22, 25]

    expected = {"len_5", "len_12", "len_18"} | {f"rep_{L}_r{r}" for L in (10, 18) for r in (2, 5)}
    for p in (1, 2):
        ev = read_json(os.path.join(run_dir, f"eval_phase{p}.json"))
        assert set(ev["test_sets"]) == expected
        for m in ev["test_sets"].values():
            assert m["length"] <= max_len and len(m["per_position_acc"]) <= max_len
            assert all(len(s["input"]) <= max_len for s in m["samples"])
    evals = records(run_dir, "eval")
    assert evals
    for r in evals:
        assert r["curve_ood"] == {}
        assert list(r["val"]) == [str(L) for L in range(5, max_len + 1)]
    assert all(int(L) <= max_len for L in read_json(
        os.path.join(run_dir, "done.json"))["examples_per_length_phase1"])

    # The evaluate CLI on the same run keeps to the training range as well.
    from src import evaluate
    out = tmp_path / "cli.json"
    evaluate.main(["--run", run_dir, "--phase", "1", "--out", str(out)])
    assert set(read_json(out)["test_sets"]) == expected


# ---------------------------------------------------------------------------------------- 4
def _equal_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_initial_weights_depend_only_on_model_config_and_seed():
    sd = {s: train.make_model(smoke_cfg(schedule=s)).state_dict()
          for s in ("R", "SL", "LS", "SL-shuf", "SLr-early")}
    for s in sd:
        assert _equal_state(sd["SL"], sd[s]), s
    torch.rand(1000)  # prior use of the global RNG does not matter
    assert _equal_state(sd["SL"], train.make_model(smoke_cfg(schedule="LS")).state_dict())
    other_seed = train.make_model(smoke_cfg(schedule="SL", seed=1)).state_dict()
    assert not _equal_state(sd["SL"], other_seed)


def test_runs_of_different_schedules_start_from_identical_weights(tmp_path, monkeypatch):
    """End to end: run() builds its model with make_model (captured here) before training."""
    init = {}
    orig = train.make_model

    def capture(cfg):
        m = orig(cfg)
        init[(cfg["data"]["schedule"], cfg["seed"])] = {k: v.clone()
                                                       for k, v in m.state_dict().items()}
        return m

    monkeypatch.setattr(train, "make_model", capture)
    dirs = {}
    for sched, seed in [("SL", 0), ("LS", 0), ("SL", 1)]:
        dirs[(sched, seed)] = train.main(smoke_argv(
            tmp_path, "--schedule", sched, "--seed", str(seed), *FAST, "--set", "phase2.steps=0"))
    assert _equal_state(init[("SL", 0)], init[("LS", 0)])
    assert not _equal_state(init[("SL", 0)], init[("SL", 1)])
    step0 = [records(dirs[k], "eval", 1)[0] for k in [("SL", 0), ("LS", 0)]]
    assert step0[0] == step0[1]
    assert not os.path.exists(os.path.join(dirs[("SL", 0)], "eval_phase2.json"))  # steps 0


# ---------------------------------------------------------------------------------------- 5
def _batches_equal(a, b):
    keys = ("input", "target", "loss_mask", "lengths")
    return len(a) == len(b) and all(np.array_equal(x[k], y[k]) for x, y in zip(a, b) for k in keys)


def test_phase2_batches_identical_across_schedules():
    ref = list(train.make_batches(smoke_cfg(schedule="SL"), 2))
    assert len(ref) == 10
    for s in ("LS", "R", "SLr-mid-shuf"):
        assert _batches_equal(ref, list(train.make_batches(smoke_cfg(schedule=s), 2))), s
    # Phase 2 depends on the seed, and is not Phase 1's R data.
    assert not _batches_equal(ref, list(train.make_batches(smoke_cfg(schedule="SL", seed=1), 2)))
    r1 = list(train.make_batches(smoke_cfg(schedule="R"), 1))[:10]
    assert not _batches_equal(ref, r1)
    # Phase 1 of SL and LS: different order, same multiset of examples.
    sl, ls = (list(train.make_batches(smoke_cfg(schedule=s), 1)) for s in ("SL", "LS"))
    assert not _batches_equal(sl, ls)
    rows = [sorted(map(tuple, (x for b in bs for x in b["inputs_raw"]))) for bs in (sl, ls)]
    assert rows[0] == rows[1]
    # A schedule and its shuffled twin: different order, the same examples.
    sm, tw = (list(train.make_batches(smoke_cfg(schedule=s), 1)) for s in ("SLr-mid", "SLr-mid-shuf"))
    assert not _batches_equal(sm, tw)
    assert (sorted(map(tuple, (x for b in sm for x in b["inputs_raw"])))
            == sorted(map(tuple, (x for b in tw for x in b["inputs_raw"]))))


# ---------------------------------------------------------------------------------------- 6
def test_lr_schedule_values():
    assert train.lr_at(0, 1e-3, 100) == 0.0
    assert train.lr_at(50, 1e-3, 100) == pytest.approx(5e-4)
    assert train.lr_at(99, 1e-3, 100) == pytest.approx(9.9e-4)
    assert train.lr_at(100, 1e-3, 100) == 1e-3
    assert train.lr_at(12345, 1e-3, 100) == 1e-3
    assert train.lr_at(0, 1e-3, 0) == 1e-3  # no warmup


def test_logged_lr_in_smoke_run(smoke_run):
    cfg = smoke_cfg()
    lr, warm = cfg["train"]["lr"], cfg["train"]["warmup_steps"]
    for r in records(smoke_run["dir"], "loss", 1):
        assert r["lr"] == pytest.approx(train.lr_at(r["step"] - 1, lr, warm))
    assert all(r["lr"] == lr for r in records(smoke_run["dir"], "loss", 2))


@pytest.mark.parametrize("reset", [False, True])
def test_phase2_constant_lr_without_new_warmup(tmp_path, monkeypatch, reset):
    made = []
    orig = train.make_optimizer
    monkeypatch.setattr(train, "make_optimizer", lambda m, c: made.append(1) or orig(m, c))
    # Phase 1 (6 steps) ends long before warmup (100) does; Phase 2 is still at the full lr.
    run_dir = train.main(smoke_argv(tmp_path, *FAST, "--set", "train.warmup_steps=100",
                                    "--set", "train.log_every=1",
                                    "--set", f"phase2.reset_optimizer={str(reset).lower()}"))
    lr = smoke_cfg()["train"]["lr"]
    p1 = records(run_dir, "loss", 1)
    assert [r["lr"] for r in p1] == pytest.approx([lr * s / 100 for s in range(6)])
    assert [r["lr"] for r in records(run_dir, "loss", 2)] == [lr] * 3
    assert len(made) == (2 if reset else 1)


# ---------------------------------------------------------------------------------------- 7
@pytest.mark.parametrize("norm", ["example", "token"])
def test_masked_loss_ignores_unmasked_positions(norm):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=g)
    target = torch.randint(0, 7, (2, 5), generator=g)
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)  # 2 and 4 tokens
    base = train.masked_loss(logits, target, mask, norm=norm)

    changed = logits.clone()
    changed[~mask] = 100 * torch.randn(int((~mask).sum()), 7, generator=g)
    tgt2 = target.masked_fill(~mask, PAD)
    assert torch.equal(train.masked_loss(changed, tgt2, mask, norm=norm), base)

    # The mode's reference value, and no gradient reaches unmasked logits.
    lg = logits.clone().requires_grad_(True)
    loss = train.masked_loss(lg, target, mask, norm=norm)
    if norm == "token":  # mean over all 6 masked tokens
        ref = F.cross_entropy(logits[mask], target[mask])
    else:  # mean over the 2 rows of each row's own mean
        ref = torch.stack([F.cross_entropy(logits[i][mask[i]], target[i][mask[i]])
                           for i in range(2)]).mean()
    assert torch.allclose(loss, ref)
    assert not torch.allclose(train.masked_loss(logits, target, mask, norm="example"),
                              train.masked_loss(logits, target, mask, norm="token"))
    loss.backward()
    assert torch.count_nonzero(lg.grad[~mask]) == 0 and torch.count_nonzero(lg.grad[mask]) > 0


def test_masked_loss_hand_computed():
    # Row 0: one target token (position 1). Row 1: two target tokens (positions 0 and 1).
    # Position 2 of both rows is unmasked and carries extreme logits that must not matter.
    logits = torch.tensor([[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [50.0, -50.0, 0.0]],
                           [[0.0, 0.0, math.log(2.0)], [0.0, 0.0, 0.0], [-50.0, 50.0, 9.0]]])
    target = torch.tensor([[1, 2, 0], [0, 1, 2]])
    mask = torch.tensor([[False, True, False], [True, True, False]])
    a = -(3.0 - math.log(math.e + math.e ** 2 + math.e ** 3))  # -log softmax([1, 2, 3])[2]
    b, c = math.log(4.0), math.log(3.0)  # -log(1 / (1 + 1 + 2)), -log(1 / 3)
    example, token = (a + (b + c) / 2) / 2, (a + b + c) / 3  # 1/2 per example vs 1/3 per token
    assert train.masked_loss(logits, target, mask, norm="example").item() == pytest.approx(
        example, rel=1e-6)
    assert train.masked_loss(logits, target, mask, norm="token").item() == pytest.approx(
        token, rel=1e-6)
    assert train.masked_loss(logits, target, mask).item() == pytest.approx(example, rel=1e-6)
    for norm in ("example", "token"):  # a single row: both modes agree
        one = train.masked_loss(logits[1:], target[1:], mask[1:], norm=norm).item()
        assert one == pytest.approx((b + c) / 2, rel=1e-6)
    with pytest.raises(ValueError):
        train.masked_loss(logits, target, mask, norm="tokens")


def _lists(*lens, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(1, 101, size=n) for n in lens]


def test_masked_loss_example_mode_weights_every_example_equally():
    g = torch.Generator().manual_seed(1)
    # "example" mode = the plain mean of the rows' losses, each computed alone in its own batch
    # (a duplicated row simply counts twice), whatever the lengths and the padding.
    x3, x1, x7 = _lists(3, 1, 7, seed=2)
    b = encode_batch([x3, x1, x7, x3])
    logits = torch.randn(*b["input"].shape, VOCAB_SIZE, generator=g)
    logits[3] = logits[0]
    tgt, mask = torch.from_numpy(b["target"]), torch.from_numpy(b["loss_mask"])
    rows = torch.stack([train.masked_loss(logits[i:i + 1], tgt[i:i + 1], mask[i:i + 1])
                        for i in range(4)])
    assert torch.allclose(train.masked_loss(logits, tgt, mask, norm="example"), rows.mean())
    n = mask.sum(1).float()  # token mode instead weighs rows by their token counts 3, 1, 7, 3
    assert n.tolist() == [3, 1, 7, 3]
    assert torch.allclose(train.masked_loss(logits, tgt, mask, norm="token"),
                          (rows * n).sum() / n.sum())

    # One fixed example x in batches with different partners (a short / a long one): its
    # gradient, i.e. its weight in the update, is identical in "example" mode, not in "token".
    x, short, long_ = _lists(4, 1, 9, seed=3)
    T = 2 * 4 + 1
    x_logits = torch.randn(T, VOCAB_SIZE, generator=g)

    def grad_of_x(other, norm):
        bb = encode_batch([x, other])
        lg = torch.randn(*bb["input"].shape, VOCAB_SIZE, generator=g)
        lg[0, :T] = x_logits
        lg.requires_grad_(True)
        train.masked_loss(lg, torch.from_numpy(bb["target"]), torch.from_numpy(bb["loss_mask"]),
                          norm=norm).backward()
        assert torch.count_nonzero(lg.grad[0, T:]) == 0  # x's padding gets no gradient
        return lg.grad[0, :T]

    assert torch.allclose(grad_of_x(short, "example"), grad_of_x(long_, "example"))
    assert not torch.allclose(grad_of_x(short, "token"), grad_of_x(long_, "token"))


def test_training_uses_configured_loss_norm(tmp_path, monkeypatch):
    seen = []
    orig = train.masked_loss
    monkeypatch.setattr(train, "masked_loss",
                        lambda *a, **k: seen.append(k.get("norm")) or orig(*a, **k))
    fast = [*FAST, "--set", "phase2.steps=0", "--set", "save_checkpoints=false"]
    train.main(smoke_argv(tmp_path, *fast, "--set", "train.loss_norm=token"))
    assert seen == ["token"] * 6
    seen.clear()
    train.main(smoke_argv(tmp_path, *fast, "--seed", "1"))
    assert seen == ["example"] * 6  # the default


# ---------------------------------------------------------------------------------------- 8
def test_token_accounting_matches_consumed_batches(smoke_run):
    d, cfg = smoke_run["dir"], smoke_cfg()
    done = read_json(os.path.join(d, "done.json"))
    for phase, steps in [(1, 40), (2, 10)]:
        batches = smoke_run["consumed"][phase]
        assert len(batches) == steps
        assert _batches_equal(batches, list(train.make_batches(cfg, phase)))  # the spec's data
        nonpad = np.cumsum([np.count_nonzero(b["input"] != PAD) for b in batches])
        padded = np.cumsum([b["input"].size for b in batches])
        examples = np.cumsum([len(b["lengths"]) for b in batches])
        # every input position of an example of length n is a real token: 2n+1 of them
        assert nonpad[-1] == sum(int((2 * b["lengths"] + 1).sum()) for b in batches)
        for r in records(d, "loss", phase):
            i = r["step"] - 1
            assert (r["tokens_nonpad"], r["tokens_padded"], r["examples"]) == (
                nonpad[i], padded[i], examples[i])
        ev = read_json(os.path.join(d, f"eval_phase{phase}.json"))
        assert (ev["tokens_nonpad"], ev["tokens_padded"]) == (nonpad[-1], padded[-1])
        assert (done[f"tokens_nonpad_phase{phase}"], done[f"tokens_padded_phase{phase}"],
                done[f"examples_phase{phase}"]) == (nonpad[-1], padded[-1], examples[-1])
        counts = np.bincount(np.concatenate([b["lengths"] for b in batches]))
        assert done[f"examples_per_length_phase{phase}"] == {
            str(L): int(c) for L, c in enumerate(counts) if c}
    assert done["tokens_nonpad_total"] == done["tokens_nonpad_phase1"] + done["tokens_nonpad_phase2"]


# ---------------------------------------------------------------------------------------- 8b
def test_config_hash_in_done_json_and_stable(smoke_run, tmp_path):
    d = smoke_run["dir"]
    done = read_json(os.path.join(d, "done.json"))
    cfg = smoke_cfg(out_dir=smoke_run["out"])
    h, e = train.config_hash(cfg), train.experiment_hash(cfg)
    assert re.fullmatch(r"[0-9a-f]{64}", h) and re.fullmatch(r"[0-9a-f]{64}", e) and h != e
    assert (done["config_hash"], done["experiment_hash"]) == (h, e)

    # stable across reloading config.yaml: resolved again, or the raw YAML (with run_id)
    path = os.path.join(d, "config.yaml")
    assert train.config_hash(train.resolve_config([path])) == h
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert "run_id" in raw and train.config_hash(raw) == h
    assert train.experiment_hash(raw) == e
    # YAML spellings of the same numbers hash alike (string "2e-3", int 0 / 1 for floats)
    assert train.config_hash(smoke_cfg("train.lr=2e-3", "train.weight_decay=0",
                                       "train.grad_clip=1", "phase2.steps=10.0")) == h
    # out_dir and run_id are ignored
    assert train.config_hash(smoke_cfg(out_dir=str(tmp_path / "elsewhere"))) == h
    assert train.config_hash({**raw, "run_id": "other", "out_dir": "x"}) == h
    # everything else counts
    for sets in (["train.lr=0.001"], ["train.loss_norm=token"], ["eval.test_per_length=17"],
                 ["data.groups.C=[16, 19]"]):
        assert train.config_hash(smoke_cfg(*sets)) != h, sets
        assert train.experiment_hash(smoke_cfg(*sets)) != e, sets
    # the experiment hash ignores seed, tag, schedule and pe; the config hash does not
    for kw in (dict(seed=3), dict(tag="x"), dict(schedule="SL-shuf"), dict(pe="none")):
        c = smoke_cfg(**kw)
        assert train.experiment_hash(c) == e and train.config_hash(c) != h, kw
    # keys that cannot change results are ignored, so editing them never marks runs STALE
    for sets in (["train.log_every=7"], ["eval.batch_size=33"], ["eval.n_samples=3"],
                 ["save_checkpoints=false"], ["train.device=cpu"]):
        assert train.config_hash(smoke_cfg(*sets)) == h, sets
    canon = train.canonical_config(cfg)
    assert "out_dir" not in canon and "save_checkpoints" not in canon
    assert "batch_size" not in canon["eval"] and "n_samples" not in canon["eval"]
    assert "device" not in canon["train"] and "log_every" not in canon["train"]
    assert canon["train"]["lr"] == cfg["train"]["lr"] and canon["seed"] == cfg["seed"]
    assert "seed" not in train.canonical_config(cfg, train.EXPERIMENT_HASH_EXCLUDE)


def test_overlap_in_done_json(smoke_run, capsys):
    done = read_json(os.path.join(smoke_run["dir"], "done.json"))
    ov = done["overlap"]
    assert set(ov) == {"phase1", "phase2"}
    for p in (1, 2):
        trained = {int(L) for L in done[f"examples_per_length_phase{p}"]}
        per = ov[f"phase{p}"]
        assert set(per) == {str(L) for L in trained if L <= train.OVERLAP_MAX_LEN} == {
            "5", "6", "7", "8"}
        for L, counts in per.items():
            # smoke: a test set only at length 5 (test_lengths 5 12 20 22 25), no OOD curve <= 8
            assert set(counts) == ({"val", "test"} if L == "5" else {"val"}), L
            for v in counts.values():
                assert isinstance(v, int) and v >= 0


def test_overlap_check_counts_planted_rows():
    """overlap_check on hand-made eval sets: rows planted from the consumed training rows are
    counted (duplicates too), a row just past the consumed ones and rep sets are not, lengths
    above 8 and unconsumed lengths are skipped, and each phase uses its own pool."""
    seed = 4
    cfg = smoke_cfg(seed=seed)
    counts = {1: {5: 50, 6: 0, 9: 10}, 2: {5: 20}}
    p1, p2 = pool_examples(seed, 1, 5, 0, 50), pool_examples(seed, 2, 5, 0, 20)
    past_end = pool_examples(seed, 1, 5, 50, 1)[0]  # the 51st row: not consumed
    val = uniform_set(5, 6, VAL_SEED)
    val[0], val[1], val[2], val[3] = p1[3], p1[49], p1[3], past_end
    test = uniform_set(5, 16, TEST_SEED)
    test[5], test[6] = p2[0], past_end
    ood = np.stack([p1[10], p2[19]])
    curve = SimpleNamespace(val={5: val, 6: uniform_set(6, 6, VAL_SEED),
                                 9: np.stack([pool_examples(seed, 1, 9, 0, 1)[0]])},
                            ood={5: ood})
    test_sets = {"len_5": {"kind": "uniform", "length": 5, "r": None, "inputs": test},
                 "rep_5_r2": {"kind": "rep", "length": 5, "r": 2, "inputs": p1[:4]}}
    ov = train.overlap_check(cfg, counts, curve, test_sets)
    assert ov == {"phase1": {"5": {"val": 3, "test": 0, "curve_ood": 1}},
                  "phase2": {"5": {"val": 0, "test": 1, "curve_ood": 1}}}
    sizes = {"val": 6, "test": 16, "curve_ood": 2}
    assert train._fmt_overlap(ov, sizes).startswith("phase1 val 3/6 test 0/16 curve_ood 1/2 (L 5-5)")


# ---------------------------------------------------------------------------------------- 9
def test_evaluate_cli_reproduces_eval_json(smoke_run, tmp_path):
    from src import evaluate
    for phase in (1, 2):
        out = tmp_path / f"cli_phase{phase}.json"
        evaluate.main(["--run", smoke_run["dir"], "--phase", str(phase), "--out", str(out)])
        cli = read_json(out)
        ref = read_json(os.path.join(smoke_run["dir"], f"eval_phase{phase}.json"))
        assert cli["run_id"] == ref["run_id"] and cli["phase"] == phase
        assert set(cli["test_sets"]) == set(ref["test_sets"])
        for name, m in ref["test_sets"].items():
            c = cli["test_sets"][name]
            assert c["exact_match"] == m["exact_match"], name
            assert c["token_acc"] == m["token_acc"], name
            assert c["per_position_acc"] == m["per_position_acc"], name
            assert c["samples"] == m["samples"], name


# ---------------------------------------------------------------------------------------- 10
def _load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_under_test",
                                                  os.path.join(ROOT, "scripts", "sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dry_run(sweep, capsys, *argv, rc=0):
    assert sweep.main(["--dry-run", *argv]) == rc
    out = capsys.readouterr().out
    return out, [line for line in out.splitlines() if "src.train" in line]


def smoke_sets():
    """configs/smoke.yaml flattened into --set items, so a sweep grid can use the smoke config."""
    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}={json.dumps(v)}"
    with open(SMOKE, encoding="utf-8") as f:
        return list(flat(yaml.safe_load(f)))


def _write_done(run_dir, obj):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f)


@pytest.mark.parametrize("grid,seeds,n", [("main", ["0", "1", "2", "3", "4"], 50),
                                          ("main", [], 50), ("nope", [], 30),
                                          ("nope_full", [], 50), ("pilot", [], 4),
                                          ("tune", [], 6)])
def test_sweep_dry_run_counts(tmp_path, capsys, grid, seeds, n):
    sweep = _load_sweep()
    argv = ["--grid", grid, "--out-dir", str(tmp_path)] + (["--seeds", *seeds] if seeds else [])
    out, cmds = _dry_run(sweep, capsys, *argv)
    assert len(cmds) == n and f"{n} runs" in out
    if grid == "tune":
        for c in cmds:
            assert "configs/tune.yaml" in c and "configs/base.yaml" in c
            assert "--set train.lr=0.0" in c and "--tag lr0.0" in c


def test_sweep_plan_run_ids_and_skip(tmp_path, capsys):
    sweep = _load_sweep()
    main_ids = {p["run_id"] for p in sweep.build_plan("main", out_dir=str(tmp_path))}
    assert main_ids == {f"sinusoidal_{c}_s{s}" for c in GRID_CONDITIONS for s in range(5)}
    nope_ids = {p["run_id"] for p in sweep.build_plan("nope", out_dir=str(tmp_path))}
    assert nope_ids == {f"none_{c}_s{s}" for c in ["R", "SL", "SLr-mid", "LS", "SL-shuf",
                                                    "SLr-mid-shuf"] for s in range(5)}
    pilot_ids = {p["run_id"] for p in sweep.build_plan("pilot", out_dir=str(tmp_path))}
    assert pilot_ids == {"sinusoidal_R_s0", "sinusoidal_SLr-mid_s0", "none_R_s0", "none_SLr-mid_s0"}
    tune = sweep.build_plan("tune", out_dir=str(tmp_path))
    assert {p["run_id"] for p in tune} == {f"sinusoidal_R_s{s}_lr{lr}" for s in (0, 1)
                                           for lr in ("0.0003", "0.0005", "0.001")}
    cfg = train.config_from_args(train.parse_args(tune[0]["argv"]), base_dir=ROOT)
    assert cfg["eval"]["id_only"] and cfg["phase2"]["steps"] == 0
    assert cfg["eval"]["curve_ood_lengths"] == [] and cfg["train"]["lr"] == 0.0003

    # A finished run with the same config is skipped; its log goes to
    # <out_dir>/sweep_logs/<run_id>.log. A done.json without config_hash (older code) is STALE.
    p = sweep.build_plan("pilot", out_dir=str(tmp_path))[0]
    assert p["log"] == os.path.join(str(tmp_path), "sweep_logs", f"{p['run_id']}.log")
    assert p["status"] == "run" and not p["done"] and p["stored_hash"] is None
    _write_done(p["run_dir"], {})
    out, cmds = _dry_run(sweep, capsys, "--grid", "pilot", "--out-dir", str(tmp_path), rc=1)
    assert "0 already done, 1 stale, 3 to run" in out and len(cmds) == 3
    assert (f"STALE {p['run_id']}: finished with a different config (stored none, now "
            f"{p['config_hash'][:10]}); rerun with --overwrite or use a --tag") in out
    _write_done(p["run_dir"], {"config_hash": p["config_hash"]})
    out, cmds = _dry_run(sweep, capsys, "--grid", "pilot", "--out-dir", str(tmp_path))
    assert "1 already done, 0 stale, 3 to run" in out and sum("SKIP" in c for c in cmds) == 1
    out, cmds = _dry_run(sweep, capsys, "--grid", "pilot", "--out-dir", str(tmp_path),
                         "--overwrite")
    assert "0 already done, 0 stale, 4 to run" in out and all("--overwrite" in c for c in cmds)


def test_sweep_skips_same_config_and_flags_stale_runs(smoke_run, capsys):
    """smoke_run is a real finished run, sinusoidal_R_s0 = the pilot grid's first run when the
    sweep is given the smoke config as --set overrides."""
    sweep = _load_sweep()
    grid = ["--grid", "pilot", "--out-dir", smoke_run["out"]]
    for s in smoke_sets():
        grid += ["--set", s]
    out, cmds = _dry_run(sweep, capsys, *grid)
    assert "4 runs (1 already done, 0 stale, 3 to run)" in out and "STALE" not in out
    skip = [c for c in cmds if "SKIP (done)" in c]
    assert len(skip) == 1 and "--schedule R --pe sinusoidal --seed 0" in skip[0]

    # the same grid with another lr: the finished run is STALE, and is not run
    stored = read_json(os.path.join(smoke_run["dir"], "done.json"))["config_hash"]
    now = train.config_hash(smoke_cfg("train.lr=0.003"))
    out, cmds = _dry_run(sweep, capsys, *grid, "--set", "train.lr=0.003", rc=1)
    assert (f"STALE sinusoidal_R_s0: finished with a different config (stored {stored[:10]}, "
            f"now {now[:10]}); rerun with --overwrite or use a --tag") in out
    assert "4 runs (0 already done, 1 stale, 3 to run)" in out and "SKIP" not in out
    assert len(cmds) == 3 and not any("--schedule R --pe sinusoidal" in c for c in cmds)
    # --overwrite still re-runs it
    out, cmds = _dry_run(sweep, capsys, *grid, "--set", "train.lr=0.003", "--overwrite")
    assert "4 runs (0 already done, 0 stale, 4 to run)" in out and "STALE" not in out
    assert len(cmds) == 4


def test_sweep_real_run_reports_stale_and_exits_nonzero(tmp_path, capsys, monkeypatch):
    """Not a dry run: every planned run is finished (2 same config, 1 other config, 1 without a
    hash), so nothing is launched; the stale ones are counted and the exit code is 1."""
    sweep = _load_sweep()
    plan = sweep.build_plan("pilot", out_dir=str(tmp_path))
    _write_done(plan[0]["run_dir"], {"config_hash": "0" * 64})
    _write_done(plan[1]["run_dir"], {"run_id": plan[1]["run_id"]})
    for p in plan[2:]:
        _write_done(p["run_dir"], {"config_hash": p["config_hash"]})

    def no_launch(*a, **k):
        raise AssertionError("the sweep must not launch a run here")

    monkeypatch.setattr(sweep.subprocess, "run", no_launch)
    argv = ["--grid", "pilot", "--out-dir", str(tmp_path)]
    assert sweep.main(argv) == 1
    out = capsys.readouterr().out
    assert "4 runs (2 already done, 2 stale, 0 to run)" in out
    assert "0 succeeded, 2 skipped, 2 stale, 0 failed" in out
    assert (f"STALE {plan[0]['run_id']}: finished with a different config (stored 0000000000, "
            f"now {plan[0]['config_hash'][:10]})") in out
    assert f"STALE {plan[1]['run_id']}: finished with a different config (stored none," in out
    assert not os.path.exists(os.path.join(str(tmp_path), "sweep_logs"))

    for p in plan[:2]:  # all finished with the current config: plain skips, exit code 0
        _write_done(p["run_dir"], {"config_hash": p["config_hash"]})
    assert sweep.main(argv) == 0
    assert "0 succeeded, 4 skipped, 0 stale, 0 failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------- 11
def test_identical_runs_are_deterministic(smoke_run, tmp_path):
    run_dir = train.main(smoke_argv(tmp_path))
    a, b = read_log(smoke_run["dir"]), read_log(run_dir)
    assert [r for r in a if r["type"] == "loss"] == [r for r in b if r["type"] == "loss"]
    assert [r for r in a if r["type"] == "eval"] == [r for r in b if r["type"] == "eval"]
    for p in (1, 2):
        ea = read_json(os.path.join(smoke_run["dir"], f"eval_phase{p}.json"))
        eb = read_json(os.path.join(run_dir, f"eval_phase{p}.json"))
        assert ea["test_sets"] == eb["test_sets"]


# ---------------------------------------------------------------------------------------- config
def test_config_resolution_and_validation():
    # later files override earlier ones; lists are replaced wholesale (tune empties curve_ood)
    cfg = train.resolve_config([BASE, TUNE])
    assert cfg["eval"]["curve_ood_lengths"] == [] and cfg["eval"]["id_only"] is True
    assert cfg["phase2"]["steps"] == 0 and cfg["model"]["d_model"] == 256
    base = train.resolve_config([BASE])
    assert base["eval"]["batch_size"] == 250 and base["train"]["lr"] == 5e-4
    assert base["train"]["loss_norm"] == "example"
    assert train.resolve_config([BASE], sets=["train.loss_norm=token"])["train"]["loss_norm"] == \
        "token"
    assert base["eval"]["test_lengths"] == list(range(5, 25)) + [25, 30, 40, 50]
    # --set values are YAML; "5e-4" (a string to PyYAML) is coerced to a float
    cfg = train.resolve_config([BASE], sets=["train.lr=5e-4", "eval.rep_rs=[2]", "tag=x"],
                               seed=3, schedule="SL-shuf", pe="none")
    assert cfg["train"]["lr"] == 5e-4 and cfg["eval"]["rep_rs"] == [2]
    assert train.run_id_of(cfg) == "none_SL-shuf_s3_x"
    for bad in (dict(sets=["train.lrr=1"]), dict(schedule="SL-shuf-shuf"), dict(schedule="XYZ"),
                dict(tag="a/b"), dict(sets=["train.steps=abc"]),
                dict(sets=["train.loss_norm=tokens"])):
        with pytest.raises(train.ConfigError):
            train.resolve_config([BASE], **bad)
    with pytest.raises(train.ConfigError):  # an override file alone is incomplete
        train.resolve_config([TUNE])
    with pytest.raises(SystemExit):
        train.main(["--config", BASE, "--schedule", "nope"])
