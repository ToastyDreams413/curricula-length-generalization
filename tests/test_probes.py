"""Tests for src/analysis/probes.py and src/analysis/attention.py (CPU, fast).

Synthetic internals: a fake model with a one-hot decoder basis (unembed.weight = I), so the score
of token v is the residual's v-th coordinate. The tests put known directions at the exact indices
the probes must read (SEP = n+1; y_1..y_{n-1} at n+2..2n) and wrong directions next to them, so an
off-by-one fails.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml

from src.analysis import attention as att_mod
from src.analysis import probes
from src.data import BOS, SEP, TEST_SEED, VOCAB_SIZE, encode_full, rep_set, uniform_set
from src.model import ModelConfig, build_model

D = VOCAB_SIZE
DECOY, DECOY2 = 1, 100          # test rows never contain these values


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------
class FakeModel:
    """Exposes forward(idx, return_internals=True) and unembed / tok_emb / ln_f, nothing else.
    mid0_fn / mid1_fn(x, y, n, t) -> [(token, weight), ...] give resid_mid[0] / resid_mid[1] at
    position t, as a sum of one-hot directions."""

    def __init__(self, mid0_fn, mid1_fn, ln_gain=None):
        self.unembed = torch.nn.Linear(D, D, bias=False)
        self.tok_emb = torch.nn.Embedding(D, D)
        self.ln_f = torch.nn.LayerNorm(D)
        with torch.no_grad():
            self.unembed.weight.copy_(torch.eye(D))
            self.tok_emb.weight.copy_(torch.eye(D))
            for tok, g in (ln_gain or {}).items():
                self.ln_f.weight[tok] = g
        self.mid0_fn, self.mid1_fn = mid0_fn, mid1_fn
        self.batch_sizes = []
        self.training = False

    def eval(self):
        self.training = False
        return self

    def forward(self, idx, return_internals=False):
        ids = idx.cpu().numpy()
        B, T = ids.shape
        n = (T - 1) // 2
        self.batch_sizes.append(B)
        mid0 = np.zeros((B, T, D), np.float32)
        mid1 = np.zeros((B, T, D), np.float32)
        logits = np.zeros((B, T, D), np.float32)
        for b in range(B):
            x = ids[b, 1:n + 1]
            y = np.sort(x)
            for t in range(T):
                for tok, w in self.mid0_fn(x, y, n, t):
                    mid0[b, t, tok] += w
                for tok, w in self.mid1_fn(x, y, n, t):
                    mid1[b, t, tok] += w
            logits[b, np.arange(n + 1, 2 * n + 1), y] = 1.0     # perfect teacher-forced outputs
        m0, m1 = torch.from_numpy(mid0), torch.from_numpy(mid1)
        internals = {"resid_pre": [m0, m0], "resid_mid": [m0, m1], "resid_post": [m0, m1],
                     "attn": [torch.zeros(B, 1, T, T)] * 2, "final": m1}
        return torch.from_numpy(logits), internals


def mid0_fn(kind="correct"):
    def f(x, y, n, t):
        if t == n + 1:                                     # SEP
            if kind == "correct":
                return [(int(x.min()), 1.0)]
            if kind == "wrong":
                return [(int(np.unique(x)[1]), 1.0)]        # second-smallest distinct value
            if kind == "sep_token_wins":                    # min wins among values, SEP among all
                return [(int(x.min()), 1.0), (SEP, 2.0)]
            if kind == "ln_flips":                          # raw: min wins; ln_f gain flips it
                return [(int(x.min()), 1.0), (int(x.max()), 0.5)]
            if kind == "shifted":
                return [(int(x.max()), 1.0)]
        if kind == "shifted" and t in (n, n + 2):           # the right answer, one index off
            return [(int(x.min()), 1.0)]
        if 1 <= t <= n:
            return [(int(x[t - 1]), 1.0)]                   # encoder-basis copy (one-hot E = I)
        return [(int(x.max()), 1.0)]                        # BOS and output positions: wrong value
    return f


def mid1_fn(mode="theory", shift=0):
    """y_j sits at index n+1+j. Top-2 = {y_j, second}, with second chosen by mode:
    theory: strict successor (min input value > y_j); next: y_{j+1} (decoy if it repeats);
    wrong: decoy. Other indices get a background {DECOY, DECOY2}."""
    def f(x, y, n, t):
        j = t - (n + 1) - shift
        if not 1 <= j <= n - 1:
            return [(DECOY, 1.0), (DECOY2, 0.5)]
        a, nxt = int(y[j - 1]), int(y[j])
        greater = y[y > a]
        strict = int(greater.min()) if len(greater) else None
        if mode == "theory":
            second = strict if strict is not None else DECOY
        elif mode == "next":
            second = nxt if nxt != a else DECOY
        else:
            second = DECOY
        return [(a, 1.0), (second, 0.7)]
    return f


DISTINCT = np.array([[7, 3, 9, 2, 5], [40, 12, 88, 7, 65], [50, 60, 30, 20, 10]])
# y = [2,5,5,9,9]: j=1 change (2->5); j=2 dup (strict succ 9); j=3 change (5->9); j=4 dup, 9 = max
DUPS = np.array([[9, 5, 2, 5, 9], [9, 5, 2, 5, 9]])


# ---------------------------------------------------------------------------
# Positions and targets
# ---------------------------------------------------------------------------
def test_positions_match_layout():
    x = np.array([[7, 3, 9]])
    inp = probes.probe_inputs(x)
    assert inp.tolist() == [[BOS, 7, 3, 9, SEP, 3, 7]] == encode_full(x)[:, :-1].tolist()
    n = 3
    assert inp[0, probes.sep_index(n)] == SEP
    assert probes.output_positions(n).tolist() == [5, 6]          # y_1, y_2 (n+2..2n)
    assert inp[0, probes.output_positions(n)].tolist() == [3, 7]


def test_succ_targets_against_bruteforce():
    xs = np.concatenate([rep_set(12, 3, 30), rep_set(12, 2, 20), uniform_set(12, 30, TEST_SEED)])
    tg = probes.succ_targets(xs)
    for i, x in enumerate(xs):
        y = sorted(x)
        for j in range(len(x) - 1):
            a, nxt = y[j], y[j + 1]
            greater = [v for v in x if v > a]
            assert tg["cur"][i, j] == a and tg["nxt"][i, j] == nxt
            assert tg["change"][i, j] == (nxt != a)
            assert tg["has_strict"][i, j] == bool(greater)
            if greater:
                assert tg["strict"][i, j] == min(greater)
    # On change positions the strict successor is the next output token.
    assert np.all(tg["strict"][tg["change"]] == tg["nxt"][tg["change"]])


# ---------------------------------------------------------------------------
# Min-finding probe
# ---------------------------------------------------------------------------
def test_min_probe_correct_wrong_and_shifted():
    good = probes.run_probes(FakeModel(mid0_fn("correct"), mid1_fn()), DISTINCT, "cpu")
    for k in ("min_acc", "min_acc_raw_full", "min_acc_raw_values", "min_acc_lnf_full",
              "min_acc_lnf_values", "enc_copy_acc", "tf_sep_acc", "tf_next_acc"):
        assert good[k] == 1.0, k
    assert good["min_rank_mean"] == 0.0
    assert good["n"] == len(DISTINCT)
    # resid_mid[1] at SEP holds only the background {DECOY, DECOY2}: no min after layer 2
    assert good["min_acc_after_attn1"] == 1.0 and good["min_acc_after_attn2"] == 0.0
    bad = probes.run_probes(FakeModel(mid0_fn("wrong"), mid1_fn()), DISTINCT, "cpu")
    assert bad["min_acc"] == 0.0 and bad["min_rank_mean"] == 1.0
    # The right direction one index before/after SEP must not count.
    shifted = probes.run_probes(FakeModel(mid0_fn("shifted"), mid1_fn()), DISTINCT, "cpu")
    assert shifted["min_acc"] == 0.0


def test_min_probe_vocab_restriction():
    m = FakeModel(mid0_fn("sep_token_wins"), mid1_fn())
    r = probes.run_probes(m, DISTINCT, "cpu")                        # default vocab="full"
    assert r["min_acc"] == 0.0 and r["min_acc_raw_full"] == 0.0
    assert r["min_acc_raw_values"] == 1.0
    assert probes.run_probes(m, DISTINCT, "cpu", vocab="values")["min_acc"] == 1.0


def test_min_probe_ln_f_option():
    # SEP holds e_min + 0.5 e_max. An ln_f gain of 10 on every row-max token flips the argmax to the
    # max under ln_f, while the raw readout still picks the min.
    gains = {int(v): 10.0 for v in DISTINCT.max(axis=1)}
    m = FakeModel(mid0_fn("ln_flips"), mid1_fn(), ln_gain=gains)
    r = probes.run_probes(m, DISTINCT, "cpu")
    assert r["min_acc_raw_full"] == 1.0 and r["min_acc_lnf_full"] == 0.0
    assert r["min_acc"] == 1.0                                       # default: raw residual
    assert probes.run_probes(m, DISTINCT, "cpu", apply_ln_f=True)["min_acc"] == 0.0


# ---------------------------------------------------------------------------
# Identity+Successor probe
# ---------------------------------------------------------------------------
def test_id_succ_distinct_values():
    r = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("theory")), DISTINCT, "cpu")
    assert r["id_succ_acc"] == 1.0 and r["id_succ_acc_strict"] == 1.0
    assert r["id_succ_n_change"] == r["id_succ_n_strict"] == r["id_succ_n_positions"] == 3 * 4
    assert r["id_succ_acc_dup"] is None                              # no duplicate positions
    w = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("wrong")), DISTINCT, "cpu")
    assert w["id_succ_acc"] == 0.0 and w["id_in_top2"] == 1.0 and w["succ_in_top2"] == 0.0
    # Content written one index late (y_j's two peaks at y_{j+1}'s position) must fail.
    s = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("theory", shift=1)), DISTINCT, "cpu")
    assert s["id_succ_acc"] == 0.0
    s = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("theory", shift=-1)), DISTINCT, "cpu")
    assert s["id_succ_acc"] == 0.0


def test_id_succ_duplicates_policies_differ():
    # Rows y = [2,5,5,9,9]: change positions j=1,3; strict positions j=1,2,3; j=4 (9 = max) dropped.
    theory = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("theory")), DUPS, "cpu")
    assert theory["id_succ_n_change"] == 2 * 2 and theory["id_succ_n_strict"] == 2 * 3
    assert theory["id_succ_n_positions"] == 2 * 4
    assert theory["id_succ_acc"] == theory["id_succ_acc_strict"] == theory["id_succ_acc_dup"] == 1.0
    # "next" puts the second peak on y_{j+1}; at the dup position j=2 that is 5 itself, so the
    # fake falls back to the decoy: change-only scores 1, strict scores 2/3, dup scores 0.
    nxt = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("next")), DUPS, "cpu")
    assert nxt["id_succ_acc"] == nxt["id_succ_acc_change"] == 1.0
    assert nxt["id_succ_acc_strict"] == pytest.approx(2 / 3)
    assert nxt["id_succ_acc_dup"] == 0.0
    strict = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("next")), DUPS, "cpu",
                               dup_policy="strict")
    assert strict["id_succ_acc"] == pytest.approx(2 / 3)
    assert strict["id_succ_acc_raw_full"] == pytest.approx(2 / 3)


def test_id_succ_duplicate_heavy_random():
    xs = rep_set(10, 4, 12)                                          # many repeats
    xs = np.where(np.isin(xs, (DECOY, DECOY2)), 50, xs)
    r = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("theory")), xs, "cpu")
    assert r["id_succ_acc"] == r["id_succ_acc_strict"] == 1.0
    assert r["id_succ_n_change"] < r["id_succ_n_strict"] < r["id_succ_n_positions"]
    nxt = probes.run_probes(FakeModel(mid0_fn(), mid1_fn("next")), xs, "cpu")
    assert nxt["id_succ_acc"] == 1.0 and nxt["id_succ_acc_dup"] == 0.0


def test_chunking_gives_same_result():
    xs = np.concatenate([DISTINCT, DUPS, DISTINCT])
    m = FakeModel(mid0_fn(), mid1_fn("next"))
    full = probes.run_probes(m, xs, "cpu")
    m.batch_sizes.clear()
    chunked = probes.run_probes(m, xs, "cpu", batch_size=3)
    assert max(m.batch_sizes) <= 3 and sum(m.batch_sizes) == len(xs)
    assert chunked == full


# ---------------------------------------------------------------------------
# Chance references for the min probe
# ---------------------------------------------------------------------------
CONST = 42
# Row mins 2, 2, 42, 10, 2, 42: the best constant guess (2) is right on 3 of 6 rows; the
# constant 42 on 2 of 6. No row holds DECOY / DECOY2.
REPEAT_MINS = np.array([[2, 50, 60, 70], [9, 2, 33, 99], [42, 50, 51, 52], [10, 20, 30, 40],
                        [5, 4, 3, 2], [45, 44, 43, 42]])


def const_sep_fn(extra=None):
    """resid_mid[0] holds e_CONST at every position (a readout collapsed to one token), plus
    `extra(x)` -> [(token, weight)] at SEP."""
    def f(x, y, n, t):
        out = [(CONST, 3.0)]
        if extra is not None and t == n + 1:
            out += extra(x)
        return out
    return f


def test_min_chance_and_mode_collapsed_readout():
    m = FakeModel(const_sep_fn(), mid1_fn())
    r = probes.run_probes(m, REPEAT_MINS, "cpu")
    assert r["min_mode_frac"] == 1.0 and r["min_mode_token"] == CONST
    assert r["min_chance"] == pytest.approx(3 / 6)                  # min 2 on 3 of the 6 rows
    assert r["min_acc"] == pytest.approx(2 / 6)                     # rows whose min IS 42
    assert r["min_acc_centered"] == 0.0                             # all ties after centring
    # resid_mid[1] at SEP is the constant background {DECOY, DECOY2}: also collapsed
    assert r["min_mode_frac_after_attn2"] == 1.0 and r["min_acc_after_attn2"] == 0.0
    assert r["min_mode_frac_after_attn1"] == r["min_mode_frac"]
    assert r["min_acc_centered_after_attn1"] == r["min_acc_centered"]
    # The same with ln_f and the value-only vocabulary
    for kw in (dict(apply_ln_f=True), dict(vocab="values")):
        rk = probes.run_probes(m, REPEAT_MINS, "cpu", **kw)
        assert rk["min_mode_frac"] == 1.0 and rk["min_chance"] == pytest.approx(0.5), kw
    # Stats are global over the rows, not per chunk
    assert probes.run_probes(m, REPEAT_MINS, "cpu", batch_size=4) == r


def test_min_chance_and_mode_perfect_readout():
    r = probes.run_probes(FakeModel(mid0_fn("correct"), mid1_fn()), REPEAT_MINS, "cpu")
    assert r["min_acc"] == 1.0 and r["min_acc_centered"] == 1.0
    assert r["min_chance"] == pytest.approx(0.5)
    # A perfect readout's modal argmax is the modal min: mode == chance < 1
    assert r["min_mode_frac"] == r["min_chance"] < 1.0 and r["min_mode_token"] == 2
    # Varied mins (2, 7, 10, ...) with ties in the mode: lowest token id wins
    xs = np.concatenate([DISTINCT, [[2, 8, 9, 11, 4], [33, 7, 90, 41, 8]]])   # mins 2,7,10,2,7
    r = probes.run_probes(FakeModel(mid0_fn("correct"), mid1_fn()), xs, "cpu")
    assert r["min_acc"] == 1.0 and r["min_acc_centered"] == 1.0
    assert r["min_mode_frac"] == r["min_chance"] == pytest.approx(2 / 5)
    assert r["min_mode_token"] == 2


def test_min_acc_centered_removes_constant_bias():
    # SEP = 3 e_CONST + e_min: the raw argmax is always CONST (collapsed; it scores only on rows
    # whose min is 42), but CONST's score is the same on every row, so centring removes it.
    m = FakeModel(const_sep_fn(lambda x: [(int(x.min()), 1.0)]), mid1_fn())
    r = probes.run_probes(m, REPEAT_MINS, "cpu")
    assert r["min_mode_frac"] == 1.0 and r["min_acc"] == pytest.approx(2 / 6)
    assert r["min_acc_centered"] == 1.0
    # A real model: the centred readout agrees with a manual computation
    model = build_model(ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mlp=64, max_len=24), 0)
    xs = uniform_set(6, 40, TEST_SEED)
    res = probes.run_probes(model, xs, "cpu", batch_size=7)
    with torch.no_grad():
        _, internals = model(torch.as_tensor(probes.probe_inputs(xs)), return_internals=True)
        s = (internals["resid_mid"][0][:, 7] @ model.unembed.weight.T).double()
    s = s - s.mean(0, keepdim=True)
    assert res["min_acc_centered"] == pytest.approx(np.mean(s.argmax(-1).numpy() == xs.min(1)))
    pred = (internals["resid_mid"][0][:, 7] @ model.unembed.weight.T).argmax(-1).numpy()
    assert res["min_mode_frac"] == pytest.approx(np.bincount(pred).max() / len(xs))
    assert res["min_chance"] == pytest.approx(np.bincount(xs.min(1)).max() / len(xs))
    json.dumps(res, allow_nan=False)


def test_length_one_and_bad_inputs():
    r = probes.run_probes(FakeModel(mid0_fn(), mid1_fn()), np.array([[5], [9]]), "cpu")
    assert r["min_acc"] == 1.0 and r["id_succ_acc"] is None and r["id_succ_n_positions"] == 0
    one = probes.run_probes(FakeModel(mid0_fn(), mid1_fn()), np.array([[5, 9]]), "cpu")
    assert one["min_acc_centered"] is None and one["min_chance"] == one["min_mode_frac"] == 1.0
    with pytest.raises(ValueError):
        probes.run_probes(FakeModel(mid0_fn(), mid1_fn()), np.array([[0, 5]]), "cpu")
    with pytest.raises(ValueError):
        probes.run_probes(FakeModel(mid0_fn(), mid1_fn()), DISTINCT, "cpu", vocab="nope")


# ---------------------------------------------------------------------------
# End to end on a tiny real model
# ---------------------------------------------------------------------------
def _save_run(tmp_path, pe="sinusoidal"):
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mlp=64, pe=pe, max_len=24)
    model = build_model(cfg, seed=0)
    run_id = f"{pe}_R_s0"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    torch.save({"state_dict": model.state_dict(), "model_config": asdict(cfg),
                "run_config": {"seed": 0}}, run_dir / "model_phase1.pt")
    (run_dir / "config.yaml").write_text(f"run_id: {run_id}\nseed: 0\n", encoding="utf-8")
    return run_dir, model, cfg


@pytest.fixture
def tiny_run(tmp_path):
    return _save_run(tmp_path)


@pytest.mark.parametrize("pe", ["sinusoidal", "none"])
def test_probe_run_end_to_end(tmp_path, pe):
    run_dir, model, cfg = _save_run(tmp_path, pe)
    if pe == "sinusoidal":
        # Length 15 needs 31 positions > max_len 24: the saved config alone cannot run it.
        with pytest.raises(ValueError):
            model(torch.as_tensor(probes.probe_inputs(uniform_set(15, 2, TEST_SEED))))
    res = probes.probe_run(str(run_dir), 1, [5, 15], n=40, device="cpu", batch_size=16)
    with open(run_dir / "probes_phase1.json", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == res
    assert set(res) == {"run_id", "phase", "readout", "by_length"}
    # run_id comes from config.yaml here (the checkpoint's run_config has none)
    assert res["run_id"] == f"{pe}_R_s0" and res["phase"] == 1 and isinstance(res["readout"], str)
    assert set(res["by_length"]) == {"5", "15"}
    for r in res["by_length"].values():
        assert r["n"] == 40
        for k in ("min_acc", "id_succ_acc"):
            assert isinstance(r[k], float) and 0.0 <= r[k] <= 1.0
        for k, v in r.items():
            assert v is None or isinstance(v, (int, float)), k


def test_probe_run_matches_manual_readout(tiny_run):
    run_dir, model, _ = tiny_run
    res = probes.probe_run(str(run_dir), 1, [6], n=30, device="cpu",
                           out=str(run_dir / "p.json"))["by_length"]["6"]
    xs = uniform_set(6, 30, TEST_SEED)
    n = 6
    with torch.no_grad():
        _, internals = model(torch.as_tensor(probes.probe_inputs(xs)), return_internals=True)
        s_sep = internals["resid_mid"][0][:, n + 1] @ model.unembed.weight.T
        s_out = internals["resid_mid"][1][:, n + 2:2 * n + 1] @ model.unembed.weight.T
        s_att = (internals["resid_mid"][0] - internals["resid_pre"][0])[:, n + 1] @ model.unembed.weight.T
    assert res["min_acc"] == pytest.approx(np.mean(s_sep.argmax(-1).numpy() == xs.min(1)))
    assert res["min_acc_attn_only"] == pytest.approx(np.mean(s_att.argmax(-1).numpy() == xs.min(1)))
    s_l2 = internals["resid_mid"][1][:, n + 1] @ model.unembed.weight.T
    assert res["min_acc_after_attn2"] == pytest.approx(np.mean(s_l2.argmax(-1).numpy() == xs.min(1)))
    assert res["min_acc_after_attn1"] == res["min_acc"]
    y = np.sort(xs, axis=1)
    top2 = np.sort(s_out.topk(2, dim=-1).indices.numpy(), axis=-1)
    hits = tot = 0
    for i in range(len(xs)):
        for j in range(n - 1):
            if y[i, j + 1] != y[i, j]:
                tot += 1
                hits += top2[i, j].tolist() == sorted([y[i, j], y[i, j + 1]])
    assert res["id_succ_acc"] == pytest.approx(hits / tot)
    assert res["id_succ_n_change"] == tot


def test_probe_run_cli(tiny_run):
    run_dir, _, _ = tiny_run
    assert probes.main(["--run", str(run_dir), "--phase", "1", "--lengths", "5", "12",
                        "--n", "10", "--device", "cpu"]) == 0
    with open(run_dir / "probes_phase1.json", encoding="utf-8") as f:
        assert set(json.load(f)["by_length"]) == {"5", "12"}


# ---------------------------------------------------------------------------
# Tuning hygiene: eval.id_only runs never evaluate beyond the training range
# ---------------------------------------------------------------------------
TUNE_MAX = 18                       # not 20, so a hard-coded 20 would fail


def _make_id_only(run_dir, model, cfg, where):
    """Mark a saved tiny run as a tuning run (training lengths up to TUNE_MAX), with the flag in
    config.yaml or only in the checkpoint's run_config."""
    run_cfg = {"run_id": run_dir.name, "seed": 0,
               "data": {"groups": {"A": [5, 10], "B": [11, 15], "C": [16, TUNE_MAX]}},
               "eval": {"id_only": True}}
    if where == "config":
        (run_dir / "config.yaml").write_text(yaml.safe_dump(run_cfg), encoding="utf-8")
    else:
        torch.save({"state_dict": model.state_dict(), "model_config": asdict(cfg),
                    "run_config": run_cfg}, run_dir / "model_phase1.pt")
    return run_dir


@pytest.fixture(params=["config", "ckpt"])
def tune_run(request, tmp_path):
    (tmp_path / "tune").mkdir()                    # tiny_run may use tmp_path itself
    run_dir, model, cfg = _save_run(tmp_path / "tune")
    return _make_id_only(run_dir, model, cfg, request.param)


def test_id_only_max_len(tiny_run, tune_run):
    assert probes.id_only_max_len(str(tiny_run[0])) is None        # no eval section: not id_only
    assert probes.id_only_max_len(str(tune_run),
                                  probes.load_run_checkpoint(str(tune_run), 1)) == TUNE_MAX
    model, _, _ = probes.load_run_model(str(tune_run), 1, "cpu")
    assert model.id_only_max_len == TUNE_MAX
    assert probes.load_run_model(str(tiny_run[0]), 1, "cpu")[0].id_only_max_len is None


def test_probe_run_drops_lengths_above_train_range(tune_run, capsys):
    out = tune_run / "p.json"
    res = probes.probe_run(str(tune_run), 1, [5, TUNE_MAX, TUNE_MAX + 1, 25], n=8, device="cpu",
                           out=str(out))
    assert set(res["by_length"]) == {"5", str(TUNE_MAX)}
    with open(out, encoding="utf-8") as f:
        assert set(json.load(f)["by_length"]) == {"5", str(TUNE_MAX)}
    note = capsys.readouterr().out
    assert "id_only" in note and str([TUNE_MAX + 1, 25]) in note
    with pytest.raises(probes.IdOnlyError):                         # nothing left
        probes.probe_run(str(tune_run), 1, [TUNE_MAX + 1, 30], n=8, device="cpu", out=str(out))
    with pytest.raises(SystemExit):                                 # the CLI: a usage error
        probes.main(["--run", str(tune_run), "--lengths", "25", "--n", "4", "--device", "cpu",
                     "--out", str(out)])
    # A model loaded from the run refuses longer inputs even when called directly
    model, _, _ = probes.load_run_model(str(tune_run), 1, "cpu", min_len=60)
    with pytest.raises(probes.IdOnlyError):
        probes.run_probes(model, uniform_set(TUNE_MAX + 1, 4, TEST_SEED), "cpu")
    assert probes.run_probes(model, uniform_set(TUNE_MAX, 4, TEST_SEED), "cpu")["n"] == 4


def test_attention_refuses_lengths_above_train_range(tune_run, tmp_path):
    out = tmp_path / "att" / "a.png"
    with pytest.raises(probes.IdOnlyError):
        att_mod.plot_attention(str(tune_run), 1, TUNE_MAX + 1, out=str(out))
    assert not out.exists()
    with pytest.raises(SystemExit):
        att_mod.main(["--run", str(tune_run), "--length", "25", "--out", str(out),
                      "--succ-n", "4"])
    assert not out.exists()
    assert att_mod.plot_attention(str(tune_run), 1, TUNE_MAX, out=str(out)) == [str(out)]
    model, _, _ = probes.load_run_model(str(tune_run), 1, "cpu", min_len=60)
    with pytest.raises(probes.IdOnlyError):
        att_mod.successor_attention_score(model, uniform_set(TUNE_MAX + 1, 4, TEST_SEED), "cpu")
    # explicit limit on a model that carries none
    with pytest.raises(probes.IdOnlyError):
        att_mod.successor_attention_score(FakeAttnModel("copy"), DISTINCT, "cpu", max_length=4)
    r = att_mod.successor_attention_score(model, uniform_set(TUNE_MAX, 4, TEST_SEED), "cpu")
    assert r["n_rows"] == 4
    # the CLI on an allowed length still writes the successor score
    succ = tmp_path / "succ.json"
    assert att_mod.main(["--run", str(tune_run), "--length", str(TUNE_MAX), "--out", str(out),
                         "--succ-n", "4", "--succ-out", str(succ)]) == 0
    assert succ.is_file()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def test_plot_attention_writes_files(tiny_run, tmp_path):
    run_dir, _, _ = tiny_run
    paths = att_mod.plot_attention(str(run_dir), 1, 5, example_idx=1)
    assert [p[-4:] for p in paths] == [".png", ".pdf"]
    out = tmp_path / "figs" / "a.png"
    assert att_mod.plot_attention(str(run_dir), 1, 12, out=str(out)) == [str(out)]  # needs a longer table
    for p in paths + [str(out)]:
        with open(p, "rb") as f:
            assert len(f.read()) > 1000


def test_successor_attention_score_real_model(tiny_run):
    _, model, cfg = tiny_run
    xs = uniform_set(8, 20, TEST_SEED)
    r = att_mod.successor_attention_score(model, xs, "cpu", batch_size=7)
    assert r["layer"] == 1 and r["n_rows"] == 20 and r["n_queries"] == 20 * 8
    for k in ("per_head", "per_head_argmax", "per_head_sep", "per_head_change"):
        assert r[k].shape == (cfg.n_heads,)
        assert np.all((r[k] >= 0) & (r[k] <= 1 + 1e-6)), k
    assert 0 < r["uniform_baseline"] < 1
    json.dumps(att_mod.to_jsonable(r), allow_nan=False)


class FakeAttnModel:
    """Layer-2 attention: 'copy' puts all mass on the first input copy of each query's next token;
    'uniform' spreads it over the visible keys; 'output_copy' puts it on the query itself."""

    def __init__(self, kind):
        self.kind = kind

    def forward(self, idx, return_internals=False):
        ids = idx.numpy()
        B, T = ids.shape
        n = (T - 1) // 2
        a = np.zeros((B, 1, T, T), np.float32)
        for b in range(B):
            x, y = ids[b, 1:n + 1], np.sort(ids[b, 1:n + 1])
            for t in range(T):
                if self.kind == "uniform" or t <= n:
                    a[b, 0, t, :t + 1] = 1.0 / (t + 1)
                elif self.kind == "copy":
                    a[b, 0, t, 1 + int(np.flatnonzero(x == y[t - n - 1])[0])] = 1.0
                else:
                    a[b, 0, t, t] = 1.0
        z = torch.zeros(B, T, 4)
        return torch.zeros(B, T, D), {"attn": [torch.from_numpy(a)] * 2, "resid_mid": [z, z]}


def test_successor_attention_score_synthetic():
    xs = np.concatenate([DISTINCT, DUPS])
    copy = att_mod.successor_attention_score(FakeAttnModel("copy"), xs, "cpu")
    assert copy["per_head"].tolist() == [1.0] and copy["per_head_argmax"].tolist() == [1.0]
    assert copy["per_head_dup"].tolist() == [1.0]
    uni = att_mod.successor_attention_score(FakeAttnModel("uniform"), xs, "cpu")
    assert uni["per_head"][0] == pytest.approx(uni["uniform_baseline"])
    # Mass on output-side copies (the query's own token) never counts as successor attention.
    own = att_mod.successor_attention_score(FakeAttnModel("output_copy"), xs, "cpu",
                                            include_sep=False)
    assert own["per_head"].tolist() == [0.0] and own["n_queries"] == len(xs) * 4
