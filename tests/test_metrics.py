"""Tests for src/metrics.py: hand-computed cases plus a pure-Python Levenshtein reference."""

import json

import numpy as np
import pytest

from src.data import BOS, PAD, SEP
from src.metrics import (adjacent_inversions, compute_metrics, edit_distance, error_types,
                         exact_match, inversions, multiset_diff, per_position_acc, prefix_len)

A = lambda rows: np.array(rows, dtype=np.int64)  # noqa: E731

SCHEMA_KEYS = {"kind", "length", "r", "n", "exact_match", "token_acc", "per_position_acc",
               "prefix_len", "prefix_frac", "inversions", "adjacent_inversions", "multiset_diff",
               "multiset_ok", "edit_distance", "edit_distance_norm", "error_types", "samples"}
ERROR_KEYS = {"correct", "order_only", "multiset_only", "both"}


def ref_levenshtein(a, b):
    """Textbook O(len(a)*len(b)) Levenshtein with unit costs."""
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D[m][n]


def assert_plain_json_types(obj, path="root"):
    """Only exact Python builtins (np.float64 subclasses float, so isinstance is not enough)."""
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


# ---------------------------------------------------------------------------- exact / per-pos
def test_exact_match_and_per_position_acc():
    target = A([[1, 2, 3], [1, 2, 3], [4, 4, 5]])
    pred = A([[1, 2, 3], [1, 3, 3], [5, 4, 4]])
    assert exact_match(pred, target).tolist() == [True, False, False]
    assert exact_match(pred, target).dtype == bool
    np.testing.assert_allclose(per_position_acc(pred, target), [2 / 3, 2 / 3, 2 / 3])
    pred2 = A([[1, 2, 3], [9, 2, 3], [9, 9, 5]])
    np.testing.assert_allclose(per_position_acc(pred2, target), [1 / 3, 2 / 3, 1.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        exact_match(A([[1, 2]]), A([[1, 2, 3]]))
    with pytest.raises(ValueError):
        exact_match(A([1, 2, 3]), A([1, 2, 3]))


# ---------------------------------------------------------------------------- prefix_len
def test_prefix_len_hand():
    target = A([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    pred = A([[1, 2, 3, 4],    # perfect -> n
              [9, 2, 3, 4],    # error at position 0 -> 0
              [1, 2, 9, 4],    # first error at 2 -> 2
              [1, 2, 3, 9]])   # error at last position -> 3
    assert prefix_len(pred, target).tolist() == [4, 0, 2, 3]


def test_prefix_len_perfect_equals_n():
    rng = np.random.default_rng(0)
    for n in (1, 5, 50):
        t = np.sort(rng.integers(1, 101, size=(7, n)), axis=1)
        assert prefix_len(t, t).tolist() == [n] * 7


# ---------------------------------------------------------------------------- inversions
def test_inversions_hand():
    assert inversions(A([[1, 2, 3, 4]])).tolist() == [0]
    assert inversions(A([[2, 1, 3, 4]])).tolist() == [1]
    assert inversions(A([[3, 1, 2]])).tolist() == [2]
    assert inversions(A([[2, 2, 1]])).tolist() == [2]      # equal values are not inversions
    assert inversions(A([[5, 5, 5]])).tolist() == [0]
    assert adjacent_inversions(A([[3, 1, 2]])).tolist() == [1]
    assert adjacent_inversions(A([[2, 2, 1]])).tolist() == [1]
    assert adjacent_inversions(A([[1, 3, 2, 4, 1]])).tolist() == [2]
    assert inversions(A([[1, 3, 2, 4, 1]])).tolist() == [4]  # (3,2) (3,1) (2,1) (4,1)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 50])
def test_inversions_reversed_list(n):
    rev = A([list(range(n, 0, -1))])
    assert inversions(rev).tolist() == [n * (n - 1) // 2]
    assert adjacent_inversions(rev).tolist() == [max(n - 1, 0)]


def test_inversions_matches_bruteforce():
    rng = np.random.default_rng(1)
    pred = rng.integers(1, 8, size=(100, 9))
    ref = [sum(int(r[i] > r[j]) for i in range(9) for j in range(i + 1, 9)) for r in pred]
    ref_adj = [sum(int(r[i] > r[i + 1]) for i in range(8)) for r in pred]
    assert inversions(pred).tolist() == ref
    assert adjacent_inversions(pred).tolist() == ref_adj


# ---------------------------------------------------------------------------- multiset_diff
def test_multiset_diff_hand():
    inputs = A([[1, 2, 3]] * 6)
    pred = A([[3, 1, 2],     # same multiset, other order -> 0
              [1, 2, 2],     # dropped 3, invented an extra 2 -> 2
              [1, 2, SEP],   # SEP counts as a wrong value: dropped 3 + invented SEP -> 2
              [PAD, 1, 2],   # PAD likewise -> 2
              [BOS, BOS, BOS],  # everything dropped (3) + 3 invented -> 6
              [7, 8, 9]])    # completely different -> 6
    assert multiset_diff(pred, inputs).tolist() == [0, 2, 2, 2, 6, 6]


def test_multiset_diff_duplicates():
    inputs = A([[4, 4, 4, 1]])
    assert multiset_diff(A([[1, 4, 4, 4]]), inputs).tolist() == [0]
    assert multiset_diff(A([[1, 4, 4, 5]]), inputs).tolist() == [2]
    assert multiset_diff(A([[1, 1, 4, 4]]), inputs).tolist() == [2]
    assert multiset_diff(A([[4, 4, 4, 4]]), inputs).tolist() == [2]


def test_multiset_diff_matches_counter():
    from collections import Counter
    rng = np.random.default_rng(2)
    x = rng.integers(1, 101, size=(200, 12))
    p = rng.integers(0, 103, size=(200, 12))   # includes PAD/SEP/BOS
    p[:50] = rng.permuted(x[:50], axis=1)      # some exact multisets
    ref = []
    for pr, xr in zip(p, x):
        cp, cx = Counter(pr.tolist()), Counter(xr.tolist())
        ref.append(sum(abs(cp[v] - cx[v]) for v in set(cp) | set(cx)))
    assert multiset_diff(p, x).tolist() == ref
    assert (multiset_diff(p[:50], x[:50]) == 0).all()


# ---------------------------------------------------------------------------- edit_distance
def test_edit_distance_hand():
    t = A([[1, 2, 3, 4, 5]] * 5)
    p = A([[1, 2, 3, 4, 5],   # identical -> 0
           [5, 1, 2, 3, 4],   # rotated by one: Hamming 5 but Levenshtein 2 (insert 5, delete 5)
           [1, 2, 9, 4, 5],   # one substitution -> 1
           [6, 7, 8, 9, 10],  # fully different, same length -> 5
           [1, 3, 4, 5, 5]])  # delete 2, insert 5 at the end -> 2
    assert edit_distance(p, t).tolist() == [0, 2, 1, 5, 2]
    assert edit_distance(t, p).tolist() == [0, 2, 1, 5, 2]  # symmetric


def test_edit_distance_matches_reference_random():
    rng = np.random.default_rng(3)
    checked = 0
    for n in range(1, 9):
        # tiny alphabets, so insertions/deletions actually beat substitutions
        alphabet = 2 if n % 2 else 4
        N = 25
        a = rng.integers(1, alphabet + 1, size=(N, n))
        b = rng.integers(1, alphabet + 1, size=(N, n))
        b[0] = a[0]                         # identical row -> 0
        b[1] = a[1] + 50                    # fully different row -> n
        b[2] = np.roll(a[2], 1)             # rotation
        got = edit_distance(a, b).tolist()
        ref = [ref_levenshtein(a[i].tolist(), b[i].tolist()) for i in range(N)]
        assert got == ref, f"n={n}"
        assert got[0] == 0 and got[1] == n
        checked += N
    assert checked >= 200


def test_edit_distance_unequal_widths():
    rng = np.random.default_rng(4)
    for m, n in [(0, 3), (3, 0), (2, 5), (6, 3), (1, 1)]:
        a = rng.integers(1, 4, size=(30, m))
        b = rng.integers(1, 4, size=(30, n))
        got = edit_distance(a, b).tolist()
        assert got == [ref_levenshtein(a[i].tolist(), b[i].tolist()) for i in range(30)]


def test_edit_distance_rejects_row_mismatch():
    with pytest.raises(ValueError):
        edit_distance(A([[1, 2]]), A([[1, 2], [1, 2]]))


# ---------------------------------------------------------------------------- error_types
def test_error_types_hand_cases():
    inputs = A([[3, 1, 2]])
    cases = {
        "correct": [1, 2, 3],
        "order_only": [2, 1, 3],      # right multiset, not sorted
        "multiset_only": [1, 2, 2],   # sorted, wrong multiset
        "both": [2, 2, 1],            # neither
    }
    for bucket, row in cases.items():
        et = error_types(A([row]), inputs)
        assert set(et) == ERROR_KEYS
        assert et[bucket] == 1.0, (bucket, et)
        assert sum(et.values()) == 1.0
    # a SEP at the end keeps the row non-decreasing by token id, but the multiset is wrong
    assert error_types(A([[1, 2, SEP]]), inputs)["multiset_only"] == 1.0
    # mixed batch -> fractions
    et = error_types(A([cases[k] for k in ("correct", "correct", "order_only", "both")]),
                     A([[3, 1, 2]] * 4))
    assert et == {"correct": 0.5, "order_only": 0.25, "multiset_only": 0.0, "both": 0.25}


def test_error_types_partition_random():
    rng = np.random.default_rng(5)
    x = rng.integers(1, 6, size=(500, 6))
    p = rng.integers(1, 6, size=(500, 6))
    p[:100] = np.sort(x[:100], axis=1)                  # correct
    p[100:200] = rng.permuted(x[100:200], axis=1)       # mostly order_only
    p[200:300] = np.sort(p[200:300], axis=1)            # mostly multiset_only
    et = error_types(p, x)
    assert set(et) == ERROR_KEYS
    assert abs(sum(et.values()) - 1.0) < 1e-12
    assert all(type(v) is float and 0.0 <= v <= 1.0 for v in et.values())
    # correct == exact match with sorted(inputs)
    assert et["correct"] == exact_match(p, np.sort(x, axis=1)).mean()
    assert et["correct"] >= 0.2


# ---------------------------------------------------------------------------- compute_metrics
def test_compute_metrics_hand_values():
    inputs = A([[3, 1, 2], [3, 1, 2]])
    pred = A([[1, 2, 3], [2, 1, 3]])
    m = compute_metrics(pred, inputs, n_samples=5)
    assert set(m) == SCHEMA_KEYS
    assert m["n"] == 2 and m["length"] == 3 and m["kind"] is None and m["r"] is None
    assert m["exact_match"] == 0.5
    assert m["token_acc"] == pytest.approx(4 / 6)
    assert m["per_position_acc"] == [0.5, 0.5, 1.0]
    assert m["prefix_len"] == 1.5 and m["prefix_frac"] == 0.5
    assert m["inversions"] == 0.5 and m["adjacent_inversions"] == 0.5
    assert m["multiset_diff"] == 0.0 and m["multiset_ok"] == 1.0
    assert m["edit_distance"] == 1.0 and m["edit_distance_norm"] == pytest.approx(1 / 3)
    assert m["error_types"] == {"correct": 0.5, "order_only": 0.5, "multiset_only": 0.0,
                                "both": 0.0}
    assert m["samples"] == [{"input": [3, 1, 2], "target": [1, 2, 3], "output": [1, 2, 3]},
                            {"input": [3, 1, 2], "target": [1, 2, 3], "output": [2, 1, 3]}]


def test_compute_metrics_schema_and_json():
    rng = np.random.default_rng(6)
    N, n = 300, 17
    x = rng.integers(1, 101, size=(N, n))
    p = np.sort(x, axis=1)
    p[::3] = rng.integers(0, 103, size=p[::3].shape)   # some garbage rows, incl. special tokens
    p[1::3, 5] = 7                                     # some single-token errors
    m = compute_metrics(p, x, n_samples=20)
    assert set(m) == SCHEMA_KEYS
    assert set(m["error_types"]) == ERROR_KEYS
    assert len(m["per_position_acc"]) == n
    assert m["n"] == N and m["length"] == n
    assert len(m["samples"]) == 20
    assert set(m["samples"][0]) == {"input", "target", "output"}
    assert m["samples"][3]["input"] == x[3].tolist()
    assert m["samples"][3]["output"] == p[3].tolist()
    assert m["samples"][3]["target"] == sorted(x[3].tolist())
    assert_plain_json_types(m)
    s = json.dumps(m, allow_nan=False)
    assert json.loads(s) == m
    # internal consistency
    assert m["token_acc"] == pytest.approx(np.mean(m["per_position_acc"]))
    assert m["error_types"]["correct"] == m["exact_match"]
    assert abs(sum(m["error_types"].values()) - 1.0) < 1e-12
    assert m["prefix_frac"] == pytest.approx(m["prefix_len"] / n)
    assert m["edit_distance_norm"] == pytest.approx(m["edit_distance"] / n)
    assert m["multiset_ok"] == pytest.approx(
        np.mean(multiset_diff(p, x) == 0))


def test_compute_metrics_n_samples_bounds():
    x = A([[2, 1]] * 3)
    assert len(compute_metrics(np.sort(x, 1), x, n_samples=0)["samples"]) == 0
    assert len(compute_metrics(np.sort(x, 1), x, n_samples=50)["samples"]) == 3


def test_compute_metrics_perfect():
    rng = np.random.default_rng(7)
    x = rng.integers(1, 101, size=(50, 10))
    m = compute_metrics(np.sort(x, axis=1), x)
    assert m["exact_match"] == 1.0 and m["token_acc"] == 1.0
    assert m["prefix_frac"] == 1.0 and m["prefix_len"] == 10.0
    assert m["edit_distance"] == 0.0 and m["inversions"] == 0.0 and m["multiset_diff"] == 0.0
    assert m["error_types"]["correct"] == 1.0
