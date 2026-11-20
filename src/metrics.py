"""Evaluation metrics for the sorting task (pure numpy). See docs/CODE_SPEC.md section 4.

All per-row functions take 2-D integer arrays of shape (N, n) (N rows, n tokens per row) and are
vectorised over rows. ``target`` is always ``np.sort(inputs, axis=1)``. ``pred`` may contain any
token id (a decode_vocab="full" decode can emit PAD/SEP/BOS), and every metric treats such a token
as an ordinary (wrong) symbol compared by its integer id.

compute_metrics returns only plain Python ints/floats/lists/dicts/None, so the result can go
straight into json.dump.

Implementation notes and resolved ambiguities:
  * edit_distance is the standard Levenshtein DP (unit insertion/deletion/substitution costs). It is
    batched over rows and also vectorised over the target index j. The insertion recurrence
    D[i][j] = min(tmp[j], D[i][j-1] + 1) unrolls to D[i][j] = j + min_{k<=j}(tmp[k] - k), which is
    a running minimum (np.minimum.accumulate). So each DP row costs O(1) numpy calls instead of n,
    and the result is identical to the textbook double loop (checked against a pure-Python
    reference in tests/test_metrics.py).
  * error_types defines "correct" as sorted_ok & ms_ok. This guarantees a partition, and it is
    equivalent to an exact match with sorted(inputs): a non-decreasing sequence with the same
    multiset as the inputs IS sorted(inputs). sorted_ok compares raw token ids, so in a "full"
    decode a trailing SEP/BOS or a leading PAD does not by itself break sortedness. Such a row
    still always has ms_ok False.
  * compute_metrics fills "kind"/"length"/"r" with placeholders (None, n, None) so that its
    output always has the complete schema. evaluate.evaluate_sets overwrites them with the eval
    set's real values.
  * The schema's "n" is the number of ROWS N. "length" is the list length n, which is also the
    denominator of prefix_frac and edit_distance_norm.
"""

from __future__ import annotations

import numpy as np

from src.data import VOCAB_SIZE

__all__ = [
    "exact_match",
    "per_position_acc",
    "prefix_len",
    "inversions",
    "adjacent_inversions",
    "multiset_diff",
    "edit_distance",
    "error_types",
    "compute_metrics",
]


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _as_2d_int(a, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (N, n) array, got shape {a.shape}")
    if a.size and not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {a.dtype}")
    return a.astype(np.int64, copy=False)


def _pair(pred, target, name_b: str = "target") -> tuple[np.ndarray, np.ndarray]:
    p = _as_2d_int(pred, "pred")
    t = _as_2d_int(target, name_b)
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} != {name_b} shape {t.shape}")
    return p, t


def _row_counts(a: np.ndarray, num_ids: int) -> np.ndarray:
    """(N, num_ids) int64 histogram of the token ids in each row (one bincount, no Python loop)."""
    N = a.shape[0]
    offsets = (np.arange(N, dtype=np.int64) * num_ids)[:, None]
    flat = (a + offsets).ravel()
    return np.bincount(flat, minlength=N * num_ids).reshape(N, num_ids)


# ----------------------------------------------------------------------------------------------
# per-row / per-position metrics
# ----------------------------------------------------------------------------------------------
def exact_match(pred, target) -> np.ndarray:
    """(N,) bool: the whole row is correct."""
    p, t = _pair(pred, target)
    return np.all(p == t, axis=1)


def per_position_acc(pred, target) -> np.ndarray:
    """(n,) float64: accuracy at each output position, averaged over rows."""
    p, t = _pair(pred, target)
    if p.shape[0] == 0:
        raise ValueError("per_position_acc needs at least one row")
    return (p == t).mean(axis=0, dtype=np.float64)


def prefix_len(pred, target) -> np.ndarray:
    """(N,) int64: number of correct tokens before the first error (n for a perfect row)."""
    p, t = _pair(pred, target)
    wrong = p != t
    first_wrong = np.argmax(wrong, axis=1)  # 0 if the row has no error; fixed by the where below
    return np.where(wrong.any(axis=1), first_wrong, p.shape[1]).astype(np.int64)


def inversions(pred) -> np.ndarray:
    """(N,) int64: number of pairs i < j with pred[i] > pred[j]."""
    p = _as_2d_int(pred, "pred")
    N, n = p.shape
    out = np.zeros(N, dtype=np.int64)
    # Loop over the offset k = j - i (n-1 vectorised comparisons). Memory stays O(N*n) instead of
    # the O(N*n^2) of a full pairwise broadcast.
    for k in range(1, n):
        out += np.count_nonzero(p[:, :-k] > p[:, k:], axis=1)
    return out


def adjacent_inversions(pred) -> np.ndarray:
    """(N,) int64: number of positions i with pred[i] > pred[i+1]."""
    p = _as_2d_int(pred, "pred")
    if p.shape[1] < 2:
        return np.zeros(p.shape[0], dtype=np.int64)
    return np.count_nonzero(p[:, :-1] > p[:, 1:], axis=1).astype(np.int64)


def multiset_diff(pred, inputs) -> np.ndarray:
    """(N,) int64: sum over ALL token ids v of |count_pred(v) - count_inputs(v)|.

    A dropped value and an invented value each count 1, so one substitution costs 2. Non-value
    tokens in pred (PAD/SEP/BOS) count as invented values. The histogram covers ids
    0..max(VOCAB_SIZE-1, largest id present), so nothing is silently clipped.
    """
    p = _as_2d_int(pred, "pred")
    x = _as_2d_int(inputs, "inputs")
    if p.shape[0] != x.shape[0]:
        raise ValueError(f"pred has {p.shape[0]} rows but inputs has {x.shape[0]}")
    if p.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    lo = min(p.min(initial=0), x.min(initial=0))
    if lo < 0:
        raise ValueError("token ids must be non-negative")
    num_ids = int(max(VOCAB_SIZE, p.max(initial=0) + 1, x.max(initial=0) + 1))
    diff = _row_counts(p, num_ids) - _row_counts(x, num_ids)
    return np.abs(diff).sum(axis=1).astype(np.int64)


def edit_distance(pred, target) -> np.ndarray:
    """(N,) int64: Levenshtein distance between pred[r] and target[r] (unit costs).

    Batched DP vectorised over rows. pred and target need the same number of rows, but their
    widths may differ. See the module docstring for the running-minimum form of the insertion step.
    """
    a = _as_2d_int(pred, "pred")
    b = _as_2d_int(target, "target")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"pred has {a.shape[0]} rows but target has {b.shape[0]}")
    N, m = a.shape
    n = b.shape[1]
    j = np.arange(n + 1, dtype=np.int64)  # (n+1,)
    prev = np.broadcast_to(j, (N, n + 1)).copy()  # D[0][j] = j
    tmp = np.empty((N, n + 1), dtype=np.int64)
    for i in range(1, m + 1):
        sub = prev[:, :-1] + (a[:, i - 1 : i] != b)  # D[i-1][j-1] + cost(i, j)
        tmp[:, 0] = i  # D[i][0] = i
        np.minimum(prev[:, 1:] + 1, sub, out=tmp[:, 1:])  # deletion vs substitution
        # insertion chain: D[i][j] = min_{k<=j} (tmp[k] + j - k)
        prev = np.minimum.accumulate(tmp - j, axis=1) + j
    return prev[:, n].copy()


def _error_masks(pred: np.ndarray, inputs: np.ndarray, msd: np.ndarray | None = None):
    sorted_ok = adjacent_inversions(pred) == 0
    if msd is None:
        msd = multiset_diff(pred, inputs)
    ms_ok = msd == 0
    return sorted_ok, ms_ok


def _error_fractions(sorted_ok: np.ndarray, ms_ok: np.ndarray) -> dict[str, float]:
    N = sorted_ok.shape[0]
    if N == 0:
        raise ValueError("error_types needs at least one row")
    return {
        "correct": float(np.count_nonzero(sorted_ok & ms_ok) / N),
        "order_only": float(np.count_nonzero(ms_ok & ~sorted_ok) / N),
        "multiset_only": float(np.count_nonzero(sorted_ok & ~ms_ok) / N),
        "both": float(np.count_nonzero(~sorted_ok & ~ms_ok) / N),
    }


def error_types(pred, inputs) -> dict[str, float]:
    """Fractions of rows in each cell of the (sorted_ok, multiset_ok) partition. They sum to 1.

    correct       = sorted_ok &  ms_ok   (== exact match with sorted(inputs))
    order_only    = ~sorted_ok &  ms_ok  (right values, wrong order)
    multiset_only = sorted_ok & ~ms_ok   (non-decreasing, but values dropped/duplicated/invented)
    both          = ~sorted_ok & ~ms_ok
    """
    p, x = _pair(pred, inputs, "inputs")
    return _error_fractions(*_error_masks(p, x))


# ----------------------------------------------------------------------------------------------
# full metric dict
# ----------------------------------------------------------------------------------------------
def compute_metrics(pred, inputs, n_samples: int = 20) -> dict:
    """Metric dict for one eval set (schema in docs/CODE_SPEC.md section 4).

    pred, inputs: (N, n) integer arrays. The target is np.sort(inputs, axis=1). Scalar metrics are
    means over the N rows. "kind"/"r" are None and "length" is n here; evaluate_sets overwrites
    them with the eval set's metadata.
    """
    p, x = _pair(pred, inputs, "inputs")
    N, n = p.shape
    if N == 0:
        raise ValueError("compute_metrics needs at least one row")
    if n == 0:
        raise ValueError("compute_metrics needs lists of length >= 1")
    t = np.sort(x, axis=1)

    correct_tok = p == t
    em = np.all(correct_tok, axis=1)
    pl = prefix_len(p, t)
    inv = inversions(p)
    adj = adjacent_inversions(p)
    msd = multiset_diff(p, x)
    ed = edit_distance(p, t)
    sorted_ok = adj == 0
    ms_ok = msd == 0
    etypes = _error_fractions(sorted_ok, ms_ok)

    k = max(0, min(int(n_samples), N))
    samples = [
        {"input": x[i].tolist(), "target": t[i].tolist(), "output": p[i].tolist()}
        for i in range(k)
    ]

    return {
        "kind": None,
        "length": int(n),
        "r": None,
        "n": int(N),
        "exact_match": float(em.mean()),
        "token_acc": float(correct_tok.mean()),
        "per_position_acc": [float(v) for v in correct_tok.mean(axis=0, dtype=np.float64)],
        "prefix_len": float(pl.mean()),
        "prefix_frac": float(pl.mean() / n),
        "inversions": float(inv.mean()),
        "adjacent_inversions": float(adj.mean()),
        "multiset_diff": float(msd.mean()),
        "multiset_ok": float(ms_ok.mean()),
        "edit_distance": float(ed.mean()),
        "edit_distance_norm": float(ed.mean() / n),
        "error_types": etypes,
        "samples": samples,
    }
