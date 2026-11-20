"""Tests for src/data.py (docs/CODE_SPEC.md sections 1-2).

Expected values are recomputed here independently (hand-written layouts, floor rule, simple
balanced-count formula) rather than through data.py's private helpers.
"""

import math
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from src import data as D
from src.data import (BOS, PAD, SEP, ExampleStream, build_eval_sets, count_overlap, encode_batch,
                      encode_full, encode_prompt, iterate_batches, length_sequence, pool_examples,
                      rep_set, sort_targets, uniform_set)

BASES = ["R", "SL", "SLr-early", "SLr-mid", "SLr-late"]  # schedules with their own stage list


def group_lengths(groups_str, groups=D.DEFAULT_GROUPS):
    return sorted({L for g in groups_str for L in range(groups[g][0], groups[g][1] + 1)})


def stage_bounds(name, steps):
    """Batch boundaries [b0=0, b1, ..., steps] by the spec's floor(cum_frac * steps) rule."""
    bounds, cum = [0], Fraction(0)
    for frac, _ in D.SCHEDULES[name]:
        cum += Fraction(frac)
        bounds.append(math.floor(cum * steps))
    bounds[-1] = steps
    return bounds


def expected_counts(n_examples, lengths):
    """Balanced counts under mix="lengths": n//m each, +1 for the n%m smallest lengths."""
    q, r = divmod(n_examples, len(lengths))
    return {L: q + (i < r) for i, L in enumerate(sorted(lengths))}


def expected_layout(x, Tm):
    """Hand-built (input, target, loss_mask) for one list x, padded to width Tm."""
    n = len(x)
    full = [BOS] + list(x) + [SEP] + sorted(x)
    inp = full[:-1] + [PAD] * (Tm - (2 * n + 1))
    tgt = full[1:] + [PAD] * (Tm - (2 * n + 1))
    mask = [n + 1 <= t <= 2 * n for t in range(Tm)]
    return inp, tgt, mask


# ---------------------------------------------------------------------------
# 1. Layout
# ---------------------------------------------------------------------------
def test_layout_hand_example():
    x = [3, 1, 2]
    n = 3
    assert encode_full(np.array([x])).tolist() == [[BOS, 3, 1, 2, SEP, 1, 2, 3]]
    assert encode_prompt(np.array([x])).tolist() == [[BOS, 3, 1, 2, SEP]]
    assert sort_targets(np.array([x])).tolist() == [[1, 2, 3]]

    b = encode_batch([np.array(x)])
    assert b["input"].tolist() == [[BOS, 3, 1, 2, SEP, 1, 2]]     # seq[:-1]
    assert b["target"].tolist() == [[3, 1, 2, SEP, 1, 2, 3]]      # seq[1:]
    assert b["lengths"].tolist() == [3]
    # loss mask True exactly at t in [n+1, 2n]
    assert np.flatnonzero(b["loss_mask"][0]).tolist() == list(range(n + 1, 2 * n + 1)) == [4, 5, 6]
    # the prediction made at the SEP position is y1
    assert b["input"][0, n + 1] == SEP and b["target"][0, n + 1] == 1
    # no input-side position (BOS, x1..xn) is in the loss
    assert not b["loss_mask"][0, : n + 1].any()
    # the masked targets are exactly sorted(x)
    assert b["target"][0][b["loss_mask"][0]].tolist() == [1, 2, 3]


def test_layout_duplicates_kept():
    b = encode_batch([np.array([5, 5, 1])])
    assert b["target"][0][b["loss_mask"][0]].tolist() == [1, 5, 5]
    assert encode_full(np.array([[5, 5, 1]])).tolist() == [[BOS, 5, 5, 1, SEP, 1, 5, 5]]


def test_loss_mask_never_on_input_side_random_batch():
    rng = np.random.default_rng(0)
    xs = [rng.integers(1, 101, size=rng.integers(1, 21)) for _ in range(64)]
    b = encode_batch(xs)
    for i, x in enumerate(xs):
        n = len(x)
        masked = np.flatnonzero(b["loss_mask"][i])
        assert masked.tolist() == list(range(n + 1, 2 * n + 1))
        assert not b["loss_mask"][i, : n + 1].any()            # BOS and x1..xn never in the loss
        assert b["input"][i, n + 1] == SEP                     # first loss position reads SEP
        assert b["target"][i, masked].tolist() == sorted(x.tolist())


# ---------------------------------------------------------------------------
# 2. encode_batch with mixed lengths
# ---------------------------------------------------------------------------
def test_encode_batch_mixed_lengths():
    xs = [np.array([3, 1, 2]), np.array([9, 8, 7, 6, 5]), np.array([4]), np.array([2, 2])]
    b = encode_batch(xs)
    Tm = 2 * 5 + 1
    assert b["input"].shape == b["target"].shape == b["loss_mask"].shape == (4, Tm)
    assert b["input"].dtype == np.int64 and b["target"].dtype == np.int64
    assert b["loss_mask"].dtype == np.bool_ and b["lengths"].dtype == np.int64
    assert b["lengths"].tolist() == [3, 5, 1, 2]
    for i, x in enumerate(xs):
        n, k = len(x), 2 * len(x) + 1
        inp, tgt, mask = expected_layout(x.tolist(), Tm)
        assert b["input"][i].tolist() == inp
        assert b["target"][i].tolist() == tgt
        assert b["loss_mask"][i].tolist() == mask
        # pads are PAD on both sides and never in the loss
        assert (b["input"][i, k:] == PAD).all() and (b["target"][i, k:] == PAD).all()
        assert not b["loss_mask"][i, k:].any()
        # the unpadded part equals the row's own (unpadded) encoding
        single = encode_batch([x])
        assert np.array_equal(b["input"][i, :k], single["input"][0])
        assert np.array_equal(b["target"][i, :k], single["target"][0])
        assert np.array_equal(b["loss_mask"][i, :k], single["loss_mask"][0])
    # the loss never touches a PAD target
    assert not (b["loss_mask"] & (b["target"] == PAD)).any()


# ---------------------------------------------------------------------------
# 3. length_sequence
# ---------------------------------------------------------------------------
def test_schedule_table_matches_spec():
    spec = {"R": [(1, "ABC")],
            "SL": [(1 / 3, "A"), (1 / 3, "B"), (1 / 3, "C")],
            "SLr-early": [(0.15, "A"), (0.15, "AB"), (0.70, "ABC")],
            "SLr-mid": [(1 / 3, "A"), (1 / 3, "AB"), (1 / 3, "ABC")],
            "SLr-late": [(0.50, "A"), (0.30, "AB"), (0.20, "ABC")]}
    assert set(D.SCHEDULES) == set(spec)
    for name, stages in spec.items():
        got = D.SCHEDULES[name]
        assert [g for _, g in got] == [g for _, g in stages]
        assert [float(f) for f, _ in got] == pytest.approx([f for f, _ in stages])
        assert sum(Fraction(f) for f, _ in got) == 1          # exact, so the last stage ends at steps
    assert D.GRID_CONDITIONS == ["R", "SL", "SLr-early", "SLr-mid", "SLr-late", "LS",
                                 "SL-shuf", "SLr-early-shuf", "SLr-mid-shuf", "SLr-late-shuf"]


def test_stage_bounds_hand_values():
    # hand-computed floor(cum_frac * steps)
    assert stage_bounds("SL", 10) == [0, 3, 6, 10]
    assert stage_bounds("SL", 20000) == [0, 6666, 13333, 20000]
    assert stage_bounds("SLr-early", 20) == [0, 3, 6, 20]
    assert stage_bounds("SLr-late", 10) == [0, 5, 8, 10]
    assert stage_bounds("SLr-early", 20000) == [0, 3000, 6000, 20000]


@pytest.mark.parametrize("name", D.GRID_CONDITIONS + ["LS-shuf", "R-shuf"])
def test_shape_and_dtype(name):
    seq = length_sequence(name, steps=10, batch_size=7, seed=0)
    assert seq.shape == (70,) and seq.dtype == np.int64
    assert set(seq.tolist()) <= set(range(5, 21))


@pytest.mark.parametrize("name", BASES)
@pytest.mark.parametrize("steps,batch_size", [(10, 7), (9, 8), (7, 5), (20, 3), (20000, 2)])
def test_stages_bounds_and_balanced_counts(name, steps, batch_size):
    seq = length_sequence(name, steps, batch_size, seed=1)
    assert seq.shape == (steps * batch_size,)
    bounds = stage_bounds(name, steps)
    batches = seq.reshape(steps, batch_size)
    for i, (_, avail) in enumerate(D.SCHEDULES[name]):
        lo, hi = bounds[i], bounds[i + 1]
        allowed = group_lengths(avail)
        # every batch of this stage only holds lengths available in this stage
        assert set(batches[lo:hi].ravel().tolist()) <= set(allowed)
        # exact balanced counts on the floor-rule segment (this also pins the boundaries)
        counts = Counter(batches[lo:hi].ravel().tolist())
        exp = expected_counts((hi - lo) * batch_size, allowed)
        assert {L: counts.get(L, 0) for L in allowed} == exp
        if hi > lo:
            assert max(exp.values()) - min(exp.values()) <= 1


@pytest.mark.parametrize("name", ["SL", "SLr-early"])
def test_stage_permutation_follows_recipe(name):
    # stage i = rng([seed, phase, 100, i]).permutation(np.repeat(ascending lengths, counts))
    steps, bs, seed, phase = 20, 7, 4, 2
    seq = length_sequence(name, steps, bs, seed, phase=phase)
    b = stage_bounds(name, steps)
    for i, (_, avail) in enumerate(D.SCHEDULES[name]):
        lengths = group_lengths(avail)
        exp = expected_counts((b[i + 1] - b[i]) * bs, lengths)
        block = np.repeat(np.array(lengths, dtype=np.int64), [exp[L] for L in lengths])
        expected = np.random.default_rng([seed, phase, 100, i]).permutation(block)
        assert np.array_equal(seq[b[i] * bs: b[i + 1] * bs], expected)


def test_sl_batches_inside_one_group_and_boundary_batches():
    steps, bs = 20000, 2
    seq = length_sequence("SL", steps, bs, seed=0).reshape(steps, bs)
    A, B, C = (set(group_lengths(g)) for g in "ABC")
    assert set(seq[6665]) <= A and set(seq[6666]) <= B
    assert set(seq[13332]) <= B and set(seq[13333]) <= C
    assert set(seq[-1]) <= C


@pytest.mark.parametrize("steps", [9, 10, 11, 20])
def test_sl_ls_same_multiset_and_reversed_blocks(steps):
    bs = 6
    sl = length_sequence("SL", steps, bs, seed=3)
    ls = length_sequence("LS", steps, bs, seed=3)
    assert np.array_equal(np.sort(sl), np.sort(ls))
    b = stage_bounds("SL", steps)
    blocks = [sl[b[i] * bs: b[i + 1] * bs] for i in range(3)]
    assert np.array_equal(ls, np.concatenate(blocks[::-1]))
    # LS batches each sit inside one group too (C first)
    batches = ls.reshape(steps, bs)
    groups = [set(group_lengths(g)) for g in "ABC"]
    assert all(any(set(row) <= g for g in groups) for row in batches)
    assert set(batches[0]) <= groups[2] and set(batches[-1]) <= groups[0]


@pytest.mark.parametrize("steps", [9, 10])
def test_sl_shuf_equals_ls_shuf(steps):
    a = length_sequence("SL-shuf", steps, 8, seed=5)
    b = length_sequence("LS-shuf", steps, 8, seed=5)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("base", BASES + ["LS"])
def test_twin_is_permutation_of_base(base):
    steps, bs = 10, 8
    seq = length_sequence(base, steps, bs, seed=2)
    twin = length_sequence(base + "-shuf", steps, bs, seed=2)
    assert np.array_equal(np.sort(seq), np.sort(twin))
    assert not np.array_equal(seq, twin)
    # the recipe: rng([seed, phase, 101]).permutation(sort(base))
    expected = np.random.default_rng([2, 1, 101]).permutation(np.sort(seq))
    assert np.array_equal(twin, expected)


def test_twin_mixes_stages():
    # a shuffled twin of SL is not staged: long lists already appear in its first batches
    twin = length_sequence("SL-shuf", 30, 16, seed=0).reshape(30, 16)
    assert twin[:10].max() >= 16 and twin[-10:].min() <= 10


def test_mix_groups_equal_mass_per_group():
    steps, bs = 3, 180  # 180 examples per stage: every quota is an integer
    seq = length_sequence("SLr-mid", steps, bs, seed=0, mix="groups").reshape(steps, bs)
    exp = [{"A": 180},
           {"A": 90, "B": 90},
           {"A": 60, "B": 60, "C": 60}]
    for stage, (_, avail) in enumerate(D.SCHEDULES["SLr-mid"]):
        counts = Counter(seq[stage].tolist())
        by_group = {g: sum(counts[L] for L in group_lengths(g)) for g in avail}
        assert by_group == exp[stage]
        assert sum(by_group.values()) == bs
        for g in avail:  # uniform over lengths inside a group
            per_len = {counts[L] for L in group_lengths(g)}
            assert len(per_len) == 1
    # R with mix="groups": equal mass per group too
    r = Counter(length_sequence("R", 2, 90, seed=0, mix="groups").tolist())
    assert [sum(r[L] for L in group_lengths(g)) for g in "ABC"] == [60, 60, 60]
    # ... whereas mix="lengths" gives group A (6 of 16 lengths) more mass than 1/3
    r2 = Counter(length_sequence("R", 2, 90, seed=0, mix="lengths").tolist())
    assert [sum(r2[L] for L in group_lengths(g)) for g in "ABC"] != [60, 60, 60]
    assert sum(r2[L] for L in group_lengths("A")) > 60


def test_groups_as_lists_like_yaml():
    yaml_groups = {"A": [5, 10], "B": [11, 15], "C": [16, 20]}
    a = length_sequence("SLr-mid", 10, 8, seed=0, groups=yaml_groups)
    b = length_sequence("SLr-mid", 10, 8, seed=0)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad", ["XYZ", "ls", "SL-shuf-shuf", "-shuf", "shuf", "SLr", ""])
def test_unknown_schedule_raises(bad):
    with pytest.raises(ValueError):
        length_sequence(bad, 10, 4, seed=0)


def test_unknown_mix_raises():
    with pytest.raises(ValueError):
        length_sequence("R", 10, 4, seed=0, mix="bogus")


@pytest.mark.parametrize("name", ["R", "SL", "SLr-mid", "LS", "SL-shuf", "SLr-late-shuf"])
def test_deterministic_and_seed_dependent(name):
    a = length_sequence(name, 10, 8, seed=7)
    assert np.array_equal(a, length_sequence(name, 10, 8, seed=7))
    assert not np.array_equal(a, length_sequence(name, 10, 8, seed=8))
    assert not np.array_equal(a, length_sequence(name, 10, 8, seed=7, phase=2))


# ---------------------------------------------------------------------------
# 4. ExampleStream / pool_examples
# ---------------------------------------------------------------------------
def test_stream_follows_recipe():
    s = ExampleStream(3, phase=1)
    got = s.take(6, 4096 + 5)
    rng = np.random.default_rng([3, 1, 6])
    chunks = [rng.integers(1, 101, size=(4096, 6), dtype=np.int64) for _ in range(2)]
    assert got.dtype == np.int64 and got.shape == (4101, 6)
    assert np.array_equal(got, np.concatenate(chunks)[:4101])
    assert got.min() >= 1 and got.max() <= 100


def test_stream_one_at_a_time_equals_bulk_across_chunk_boundary():
    n = ExampleStream.CHUNK + 20
    bulk = ExampleStream(0).take(5, n)
    s = ExampleStream(0)
    single = []
    for k in range(n):
        single.append(s.take(5, 1)[0])
        if k % 1000 == 0:
            s.take(7, 3)  # other lengths in between must not matter
    assert np.array_equal(np.stack(single), bulk)
    # irregular pieces straddling the boundary
    s2 = ExampleStream(0)
    pieces = [s2.take(5, c) for c in (4000, 90, 1, 5, 20)]
    assert np.array_equal(np.concatenate(pieces), bulk[:4116])
    assert s.consumed() == {5: n, 7: 15}
    assert s2.consumed() == {5: 4116}


def test_stream_take_zero_and_consumed():
    s = ExampleStream(1)
    assert s.consumed() == {}
    assert s.take(9, 0).shape == (0, 9)
    s.take(9, 4)
    s.take(12, 2)
    s.take(9, 1)
    assert s.consumed() == {9: 5, 12: 2}
    # the next row continues where the stream stopped
    assert np.array_equal(s.take(9, 1)[0], ExampleStream(1).take(9, 6)[5])


def test_pool_examples_matches_stream():
    bulk = ExampleStream(4, phase=1).take(8, 4200)
    assert np.array_equal(pool_examples(4, 1, 8, 0, 10), bulk[:10])
    assert np.array_equal(pool_examples(4, 1, 8, 4090, 12), bulk[4090:4102])  # straddles chunk
    assert np.array_equal(pool_examples(4, 1, 8, 4096, 50), bulk[4096:4146])
    assert np.array_equal(pool_examples(4, 1, 8, 9000, 3), ExampleStream(4).take(8, 9003)[9000:])


def test_phase2_stream_differs_from_phase1():
    p1 = ExampleStream(0, phase=1).take(10, 50)
    p2 = ExampleStream(0, phase=2).take(10, 50)
    assert not np.array_equal(p1, p2)
    assert count_overlap(p2, p1) == 0
    assert not np.array_equal(p1, ExampleStream(1, phase=1).take(10, 50))
    assert not np.array_equal(p1, pool_examples(0, 2, 10, 0, 50))


# ---------------------------------------------------------------------------
# iterate_batches
# ---------------------------------------------------------------------------
def test_iterate_batches_order_and_slots():
    steps, bs = 6, 16
    seq = length_sequence("SLr-mid-shuf", steps, bs, seed=0)
    batches = list(iterate_batches(seq, bs, ExampleStream(0)))
    assert len(batches) == steps
    ref = ExampleStream(0)  # literal reference: one take per slot, in slot order
    for b, batch in enumerate(batches):
        lens = seq[b * bs:(b + 1) * bs]
        assert batch["lengths"].tolist() == lens.tolist()
        assert len(batch["inputs_raw"]) == bs
        for i, L in enumerate(lens):
            x = batch["inputs_raw"][i]
            assert len(x) == L
            assert np.array_equal(x, ref.take(int(L), 1)[0])
            assert np.array_equal(batch["input"][i, 1:L + 1], x)
        # same arrays as encode_batch on the raw lists
        enc = encode_batch(batch["inputs_raw"])
        for key in ("input", "target", "loss_mask", "lengths"):
            assert np.array_equal(batch[key], enc[key])


def test_iterate_batches_rejects_ragged_tail():
    with pytest.raises(ValueError):
        list(iterate_batches(np.array([5, 6, 7]), 2, ExampleStream(0)))


# ---------------------------------------------------------------------------
# 5. Shuffled-twin exactness on the actual token rows consumed
# ---------------------------------------------------------------------------
def consumed_rows(schedule, steps, bs=8, seed=0):
    seq = length_sequence(schedule, steps, bs, seed)
    stream = ExampleStream(seed)
    rows = [tuple(x.tolist()) for batch in iterate_batches(seq, bs, stream)
            for x in batch["inputs_raw"]]
    return rows, stream.consumed()


@pytest.mark.parametrize("steps", [9, 10])
def test_twins_consume_identical_rows(steps):
    sl, sl_used = consumed_rows("SL", steps)
    sl_shuf, sl_shuf_used = consumed_rows("SL-shuf", steps)
    ls, _ = consumed_rows("LS", steps)
    mid, _ = consumed_rows("SLr-mid", steps)
    mid_shuf, _ = consumed_rows("SLr-mid-shuf", steps)
    assert len(sl) == steps * 8
    assert Counter(sl) == Counter(sl_shuf)
    assert Counter(sl) == Counter(ls)
    assert Counter(mid) == Counter(mid_shuf)
    assert sl_used == sl_shuf_used
    # negative controls: the comparison is not vacuous
    assert sl != sl_shuf                    # same rows, different order
    assert Counter(sl) != Counter(mid)      # a different schedule consumes different rows


# ---------------------------------------------------------------------------
# 6. rep_set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [7, 10, 20, 30, 40])
@pytest.mark.parametrize("r", [2, 3, 4, 5])
def test_rep_set_structure(L, r):
    n = 200
    x = rep_set(L, r, n)
    assert x.shape == (n, L) and x.dtype == np.int64
    assert x.min() >= 1 and x.max() <= 100
    k = math.ceil(L / r)
    for row in x:
        counts = Counter(row.tolist())
        assert len(counts) <= k and max(counts.values()) <= r
        # truncation leaves k-1 values with r copies and one with the remainder (>= 1)
        assert sorted(counts.values()) == sorted([r] * (k - 1) + [L - (k - 1) * r])
        if L % r == 0:
            assert len(counts) == L // r and set(counts.values()) == {r}


def test_rep_set_deterministic_shuffled_and_guarded():
    a = rep_set(20, 2, 300)
    assert np.array_equal(a, rep_set(20, 2, 300))
    assert np.array_equal(a, rep_set(20, 2, 300, seed=D.REP_SEED))
    assert not np.array_equal(a, rep_set(20, 2, 300, seed=1))
    # rows are shuffled: copies of a value are usually not adjacent
    grouped = sum(all(row[2 * j] == row[2 * j + 1] for j in range(10)) for row in a)
    assert grouped < 10
    with pytest.raises(ValueError):
        rep_set(300, 2, 5)  # needs 150 distinct values


# ---------------------------------------------------------------------------
# 7. uniform_set / build_eval_sets
# ---------------------------------------------------------------------------
def test_uniform_set():
    a = uniform_set(12, 500, D.VAL_SEED)
    assert a.shape == (500, 12) and a.dtype == np.int64
    assert a.min() >= 1 and a.max() <= 100
    assert np.array_equal(a, uniform_set(12, 500, D.VAL_SEED))
    expected = np.random.default_rng([D.VAL_SEED, 12]).integers(1, 101, size=(500, 12))
    assert np.array_equal(a, expected)
    assert not np.array_equal(a, uniform_set(12, 500, D.TEST_SEED))
    assert (D.VAL_SEED, D.TEST_SEED, D.REP_SEED) == (1_000_001, 2_000_002, 3_000_003)


def test_build_eval_sets():
    sets = build_eval_sets([5, 22, 30], 40, D.TEST_SEED, rep_lengths=[10, 20], rep_rs=[2, 3],
                           rep_n=15)
    assert set(sets) == {"len_5", "len_22", "len_30",
                         "rep_10_r2", "rep_10_r3", "rep_20_r2", "rep_20_r3"}
    for L in (5, 22, 30):
        s = sets[f"len_{L}"]
        assert (s["kind"], s["length"], s["r"]) == ("uniform", L, None)
        assert np.array_equal(s["inputs"], uniform_set(L, 40, D.TEST_SEED))
    for L in (10, 20):
        for r in (2, 3):
            s = sets[f"rep_{L}_r{r}"]
            assert (s["kind"], s["length"], s["r"]) == ("rep", L, r)
            assert s["inputs"].shape == (15, L)
            assert np.array_equal(s["inputs"], rep_set(L, r, 15))
    assert set(build_eval_sets([5], 10, D.VAL_SEED)) == {"len_5"}
    assert set(build_eval_sets([5], 10, D.VAL_SEED, [10], [2], rep_n=0)) == {"len_5"}


# ---------------------------------------------------------------------------
# 8. count_overlap
# ---------------------------------------------------------------------------
def test_count_overlap_hand_examples():
    a = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
    b = np.array([[1, 2], [7, 8], [3, 4], [3, 4]])
    assert count_overlap(a, b) == 3          # [1,2] twice and [3,4] once
    assert count_overlap(b, a) == 3          # [1,2] once and [3,4] twice
    assert count_overlap(np.array([[2, 1]]), np.array([[1, 2]])) == 0   # order inside a row matters
    assert count_overlap(a, a) == 4
    assert count_overlap(np.empty((0, 2), dtype=np.int64), b) == 0
    assert count_overlap(a, np.empty((0, 2), dtype=np.int64)) == 0
    assert count_overlap(a.astype(np.int32), b) == 3                    # dtype does not matter
    with pytest.raises(ValueError):
        count_overlap(a, np.array([[1, 2, 3]]))
