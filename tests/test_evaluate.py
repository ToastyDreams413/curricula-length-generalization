"""Tests for src/evaluate.py, using hand-built stand-in models (no dependency on src/model.py)."""

import dataclasses
import json
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.data
from src.data import BOS, PAD, SEP, VALUE_MAX, VALUE_MIN, VOCAB_SIZE
from src.evaluate import (build_prompt, eval_lengths_from_config, evaluate_sets, greedy_decode,
                          main, max_train_length)

CPU = torch.device("cpu")


def assert_plain_json_types(obj, path="root"):
    t = type(obj)
    if t is dict:
        for k, v in obj.items():
            assert type(k) is str, f"{path}: key {k!r} is {type(k)}"
            assert_plain_json_types(v, f"{path}.{k}")
    elif t is list:
        for i, v in enumerate(obj):
            assert_plain_json_types(v, f"{path}[{i}]")
    else:
        assert t in (int, float, str, bool, type(None)), f"{path} has type {t}"


# ----------------------------------------------------------------------------- stand-in models
class Oracle(nn.Module):
    """Reads x from the prompt inside idx. At every position t >= SEP it puts a large logit on the
    correct next token of sorted(x) (the prediction at SEP is y1, at y_j it is y_{j+1}). Everywhere
    else it emits garbage: random noise, with non-value ids (SEP, PAD, BOS) as the runners-up. So
    reading the wrong position, or a misaligned prompt, produces wrong values."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.modes_seen = []
        self.grad_seen = []
        self.calls = 0

    def forward(self, idx):
        self.calls += 1
        self.modes_seen.append(self.training)
        self.grad_seen.append(torch.is_grad_enabled())
        B, T = idx.shape
        g = torch.Generator().manual_seed(1000 + T)
        logits = torch.randn(B, T, VOCAB_SIZE, generator=g)
        logits[:, :, SEP] = 50.0
        logits[:, :, PAD] = 40.0
        logits[:, :, BOS] = 30.0
        is_sep = idx == SEP
        assert bool(is_sep.any(dim=1).all()), "every row must contain the SEP delimiter"
        sep_pos = is_sep.int().argmax(dim=1)          # first SEP per row
        n = int(sep_pos[0]) - 1
        assert bool((sep_pos == n + 1).all()), "all rows in a batch have the same length"
        assert bool((idx[:, 0] == BOS).all())
        y = torch.sort(idx[:, 1:n + 1], dim=1).values  # target, read from the prompt
        ts = torch.arange(n + 1, min(T, 2 * n + 1))    # positions SEP .. y_{n-1}
        if len(ts):
            rows = torch.arange(B)[:, None]
            logits[rows, ts[None, :], y[:, ts - (n + 1)]] = 100.0
        return logits + 0.0 * self.dummy


class Successor(nn.Module):
    """next token = f(token at the current position), so the output is correct only if the decode
    loop feeds its own generated tokens back in."""

    @staticmethod
    def f(tok):
        return (tok * 7 + 3) % 100 + 1

    def forward(self, idx):
        B, T = idx.shape
        logits = torch.zeros(B, T, VOCAB_SIZE)
        logits[:, :, SEP] = 5.0                      # non-value runner-up
        logits.scatter_(2, self.f(idx).unsqueeze(-1), 10.0)
        return logits


class PrefersSep(nn.Module):
    def forward(self, idx):
        B, T = idx.shape
        g = torch.Generator().manual_seed(T)
        logits = torch.randn(B, T, VOCAB_SIZE, generator=g)
        logits[:, :, SEP] = 10.0
        return logits


class IntHashLM(nn.Module):
    """Deterministic, history-dependent toy LM with exactly representable (integer) logits, so
    results cannot depend on batch composition via floating-point rounding."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("a", torch.randint(1, 1000, (VOCAB_SIZE,), generator=g))
        self.register_buffer("b", torch.randint(0, 1000, (VOCAB_SIZE,), generator=g))
        self.lin = nn.Linear(1, 1)  # so it has parameters / a training flag that matters

    def forward(self, idx):
        T = idx.shape[1]
        h = torch.cumsum(idx * torch.arange(1, T + 1), dim=1)       # (B, T) int64, causal
        return ((h.unsqueeze(-1) * self.a + self.b) % 997).float()  # (B, T, V)


class Spy(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, idx):
        self.seen.append(idx.clone())
        return torch.zeros(idx.shape[0], idx.shape[1], VOCAB_SIZE)


def random_inputs(rng, N, n, hi=100):
    x = rng.integers(VALUE_MIN, hi + 1, size=(N, n))
    # force the extreme values in, so an off-by-one in the 1..100 slice would show up
    x[0, 0] = VALUE_MIN
    x[1 % N, -1] = VALUE_MAX
    x[2 % N, :] = VALUE_MAX
    return x.astype(np.int64)


# ----------------------------------------------------------------------------- greedy_decode
def test_build_prompt_layout():
    x = np.array([[5, 3, 9], [1, 1, 100]])
    np.testing.assert_array_equal(build_prompt(x), [[BOS, 5, 3, 9, SEP], [BOS, 1, 1, 100, SEP]])


def test_prompt_and_number_of_forward_passes():
    rng = np.random.default_rng(0)
    x = random_inputs(rng, 10, 4)
    spy = Spy()
    out = greedy_decode(spy, x, CPU, batch_size=4)
    assert out.shape == (10, 4) and out.dtype == np.int64
    assert len(spy.seen) == 3 * 4                  # ceil(10/4) chunks x n steps
    np.testing.assert_array_equal(spy.seen[0].numpy(), build_prompt(x[:4]))
    assert [s.shape[1] for s in spy.seen[:4]] == [6, 7, 8, 9]  # n+2 .. 2n+1
    # all-zero logits: values argmax ties -> lowest value id (1); never PAD
    assert (out == VALUE_MIN).all()


@pytest.mark.parametrize("n", [1, 5, 13, 20, 50])
def test_oracle_exact_match(n):
    rng = np.random.default_rng(n)
    x = random_inputs(rng, 64, n)
    x[3:20] = random_inputs(rng, 17, n, hi=4)       # heavy duplicates
    model = Oracle()
    for vocab in ("values", "full"):
        out = greedy_decode(model, x, CPU, batch_size=1000, decode_vocab=vocab)
        np.testing.assert_array_equal(out, np.sort(x, axis=1))


def test_successor_feeds_back_generated_tokens():
    x = np.array([[4, 2, 9, 9, 1, 7], [100, 1, 50, 3, 3, 3]])
    out = greedy_decode(Successor(), x, CPU)
    for row in out:
        expected, tok = [], SEP
        for _ in range(x.shape[1]):
            tok = Successor.f(tok)
            expected.append(tok)
        assert row.tolist() == expected


def test_decode_vocab_restriction():
    rng = np.random.default_rng(1)
    x = random_inputs(rng, 30, 8)
    vals = greedy_decode(PrefersSep(), x, CPU, decode_vocab="values")
    assert not (vals == SEP).any()
    assert ((vals >= VALUE_MIN) & (vals <= VALUE_MAX)).all()
    full = greedy_decode(PrefersSep(), x, CPU, decode_vocab="full")
    assert (full == SEP).all()
    with pytest.raises(ValueError):
        greedy_decode(PrefersSep(), x, CPU, decode_vocab="everything")


def test_values_slice_reaches_100():
    # A model that always prefers id 100 must produce 100 (catches a 1..99 slice).
    class Prefers100(nn.Module):
        def forward(self, idx):
            logits = torch.zeros(*idx.shape, VOCAB_SIZE)
            logits[:, :, VALUE_MAX] = 3.0
            logits[:, :, SEP] = 9.0
            return logits
    out = greedy_decode(Prefers100(), np.ones((3, 5), dtype=np.int64), CPU, decode_vocab="values")
    assert (out == VALUE_MAX).all()


def test_batch_size_chunking_invariance():
    rng = np.random.default_rng(2)
    x = random_inputs(rng, 37, 9)
    model = IntHashLM(seed=3)
    ref = greedy_decode(model, x, CPU, batch_size=1000, decode_vocab="full")
    assert len(np.unique(ref)) > 5                 # nontrivial outputs
    for bs in (37, 36, 10, 7, 1):
        np.testing.assert_array_equal(greedy_decode(model, x, CPU, batch_size=bs,
                                                    decode_vocab="full"), ref)
    ref_v = greedy_decode(model, x, CPU, batch_size=1000)
    np.testing.assert_array_equal(greedy_decode(model, x, CPU, batch_size=4), ref_v)


def test_restores_training_mode_and_uses_eval_no_grad():
    rng = np.random.default_rng(3)
    x = random_inputs(rng, 5, 6)
    model = Oracle()
    model.train()
    greedy_decode(model, x, CPU, batch_size=2)
    assert model.training is True
    assert model.calls == 3 * 6
    assert not any(model.modes_seen) and not any(model.grad_seen)
    model.eval()
    greedy_decode(model, x, CPU)
    assert model.training is False

    class Boom(nn.Module):
        def forward(self, idx):
            raise RuntimeError("boom")
    b = Boom()
    b.train()
    with pytest.raises(RuntimeError):
        greedy_decode(b, x, CPU)
    assert b.training is True                       # restored even on error


def test_greedy_decode_edge_cases():
    out = greedy_decode(Oracle(), np.zeros((0, 7), dtype=np.int64), CPU)
    assert out.shape == (0, 7)
    with pytest.raises(ValueError):
        greedy_decode(Oracle(), np.array([1, 2, 3]), CPU)
    with pytest.raises(ValueError):
        greedy_decode(Oracle(), np.ones((2, 3), dtype=np.int64), CPU, batch_size=0)


def test_device_none_uses_model_device():
    x = np.array([[3, 2, 1]])
    np.testing.assert_array_equal(greedy_decode(Oracle(), x, None), [[1, 2, 3]])


# ----------------------------------------------------------------------------- evaluate_sets
def _hand_sets(rng):
    rep = np.repeat(rng.permuted(np.tile(np.arange(1, 101), (40, 1)), axis=1)[:, :5], 2, axis=1)
    rep = rng.permuted(rep, axis=1)
    return {
        "len_5": {"kind": "uniform", "length": 5, "r": None, "inputs": random_inputs(rng, 200, 5)},
        "len_17": {"kind": "uniform", "length": np.int64(17), "r": None,
                   "inputs": random_inputs(rng, 120, 17, hi=6)},
        "rep_10_r2": {"kind": "rep", "length": np.int64(10), "r": np.int64(2),
                      "inputs": rep.astype(np.int64)},
    }


def test_evaluate_sets_with_oracle():
    rng = np.random.default_rng(4)
    sets = _hand_sets(rng)
    res = evaluate_sets(Oracle(), sets, CPU, batch_size=64, n_samples=3)
    assert list(res) == list(sets)
    for name, m in res.items():
        s = sets[name]
        assert m["kind"] == s["kind"]
        assert m["length"] == int(s["length"]) and type(m["length"]) is int
        assert m["r"] == (None if s["r"] is None else int(s["r"]))
        assert m["n"] == s["inputs"].shape[0]
        assert m["exact_match"] == 1.0
        assert m["token_acc"] == 1.0
        assert m["edit_distance"] == 0.0 and m["edit_distance_norm"] == 0.0
        assert m["error_types"] == {"correct": 1.0, "order_only": 0.0, "multiset_only": 0.0,
                                    "both": 0.0}
        assert m["prefix_frac"] == 1.0 and m["multiset_ok"] == 1.0
        assert len(m["per_position_acc"]) == int(s["length"])
        assert len(m["samples"]) == 3
        assert m["samples"][0]["output"] == m["samples"][0]["target"]
    assert type(res["rep_10_r2"]["r"]) is int
    assert_plain_json_types(res)
    json.dumps(res, allow_nan=False)


def test_evaluate_sets_imperfect_model_metrics():
    rng = np.random.default_rng(5)
    sets = _hand_sets(rng)
    res = evaluate_sets(PrefersSep(), sets, CPU, decode_vocab="full")
    for m in res.values():
        assert m["exact_match"] == 0.0 and m["multiset_ok"] == 0.0
        assert m["error_types"]["multiset_only"] == 1.0   # all-SEP output is "non-decreasing"


def test_evaluate_sets_length_mismatch_raises():
    sets = {"len_5": {"kind": "uniform", "length": 6, "r": None,
                      "inputs": np.ones((4, 5), dtype=np.int64)}}
    with pytest.raises(ValueError):
        evaluate_sets(Oracle(), sets, CPU)


# ----------------------------------------------------------------------------- config helpers
GROUPS = {"A": [5, 10], "B": [11, 15], "C": [16, 20]}
TEST_LENGTHS = list(range(5, 26)) + [30, 40, 50]


def _cfg(id_only, groups=GROUPS):
    return {"data": {"groups": groups},
            "eval": {"test_lengths": TEST_LENGTHS, "test_per_length": 1000,
                     "rep_lengths": [10, 20, 30, 40], "rep_rs": [2, 3, 4, 5], "rep_per_set": 500,
                     "id_only": id_only}}


def test_eval_lengths_id_only_filters_test_and_rep():
    spec = eval_lengths_from_config(_cfg(True))
    assert spec["max_len"] == 20
    assert spec["test_lengths"] == list(range(5, 21))
    assert spec["rep_lengths"] == [10, 20]
    assert spec["rep_rs"] == [2, 3, 4, 5] and spec["rep_per_set"] == 500
    spec = eval_lengths_from_config(_cfg(False))
    assert spec["test_lengths"] == TEST_LENGTHS and spec["rep_lengths"] == [10, 20, 30, 40]
    assert max_train_length(_cfg(True, groups={"A": [3, 7], "B": [8, 12]})) == 12


# ----------------------------------------------------------------------------- CLI plumbing
@dataclasses.dataclass
class _FakeModelConfig:
    vocab_size: int = 103
    d_model: int = 4


def test_main_cli_plumbing(tmp_path, monkeypatch):
    """main() with stand-ins for src.model and data.build_eval_sets (both written by other tasks):
    checks config -> eval-set spec (id_only), checkpoint loading, and the output JSON format."""
    import yaml

    calls = {}

    def fake_build_eval_sets(lengths, n, seed, rep_lengths=(), rep_rs=(), rep_n=0):
        calls.update(lengths=list(lengths), n=n, seed=seed, rep_lengths=list(rep_lengths),
                     rep_rs=list(rep_rs), rep_n=rep_n)
        rng = np.random.default_rng(0)
        sets = {f"len_{L}": {"kind": "uniform", "length": L, "r": None,
                             "inputs": rng.integers(1, 101, size=(n, L))} for L in lengths}
        for L in rep_lengths:
            for r in rep_rs:
                sets[f"rep_{L}_r{r}"] = {"kind": "rep", "length": L, "r": r,
                                         "inputs": rng.integers(1, 101, size=(rep_n, L))}
        return sets

    def fake_build_model(cfg, seed):
        assert isinstance(cfg, _FakeModelConfig)
        return Oracle()

    fake_model_mod = types.ModuleType("src.model")
    fake_model_mod.ModelConfig = _FakeModelConfig
    fake_model_mod.build_model = fake_build_model
    monkeypatch.setitem(sys.modules, "src.model", fake_model_mod)
    monkeypatch.setattr(src.data, "build_eval_sets", fake_build_eval_sets, raising=False)
    monkeypatch.setattr(src.data, "TEST_SEED", 2_000_002, raising=False)

    cfg = {"run_id": "sinusoidal_R_s0", "seed": 0, "data": {"groups": GROUPS},
           "train": {"device": "cpu"},
           "eval": {"test_lengths": [5, 20, 21, 30], "test_per_length": 7,
                    "rep_lengths": [10, 30], "rep_rs": [2, 3], "rep_per_set": 4,
                    "batch_size": 5, "decode_vocab": "values", "n_samples": 2, "id_only": True}}
    run = tmp_path / "sinusoidal_R_s0"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save({"state_dict": Oracle().state_dict(),
                "model_config": dataclasses.asdict(_FakeModelConfig()), "run_config": cfg},
               run / "model_phase2.pt")

    main(["--run", str(run), "--phase", "2"])
    assert calls == {"lengths": [5, 20], "n": 7, "seed": 2_000_002, "rep_lengths": [10],
                     "rep_rs": [2, 3], "rep_n": 4}
    out = json.loads((run / "eval_phase2_cli.json").read_text(encoding="utf-8"))
    assert set(out) == {"run_id", "phase", "test_sets"}
    assert out["run_id"] == "sinusoidal_R_s0" and out["phase"] == 2
    assert set(out["test_sets"]) == {"len_5", "len_20", "rep_10_r2", "rep_10_r3"}
    assert all(m["exact_match"] == 1.0 for m in out["test_sets"].values())
    assert out["test_sets"]["rep_10_r3"]["r"] == 3 and out["test_sets"]["len_20"]["n"] == 7

    explicit = tmp_path / "sub" / "x.json"
    main(["--run", str(run), "--phase", "2", "--out", str(explicit), "--decode-vocab", "full"])
    assert json.loads(explicit.read_text(encoding="utf-8"))["test_sets"]["len_5"]["exact_match"] == 1.0
