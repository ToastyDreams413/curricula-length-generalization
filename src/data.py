"""Data for the sorting task. See docs/CODE_SPEC.md sections 1-2.

An example is an unsorted list x of n ints in [1, 100]; its target is sorted(x).
Full sequence: [BOS, x1..xn, SEP, y1..yn]; the loss covers only the predictions of y1..yn.

Training data per (seed, phase) is a deterministic virtual pool (ExampleStream): the k-th example
of length L is fixed. A schedule only decides the ORDER of lengths (length_sequence), so a
schedule and its shuffled twin consume exactly the same examples, just in a different order.
"""

import math
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------
# 1. Tokens (fixed by the spec; other modules import these)
# ---------------------------------------------------------------------------
PAD = 0
VALUE_MIN = 1
VALUE_MAX = 100
SEP = 101
BOS = 102
VOCAB_SIZE = 103

# ---------------------------------------------------------------------------
# 2. Schedules
# ---------------------------------------------------------------------------
DEFAULT_GROUPS = {"A": (5, 10), "B": (11, 15), "C": (16, 20)}  # inclusive length ranges

# Each schedule is a list of stages: (fraction of steps, groups available in that stage).
# Fractions are exact, as the spec asks, so the floor(cum_frac * steps) stage boundaries never
# depend on float rounding. (The spec's table writes them as floats such as 1/3 and 0.15.)
SCHEDULES = {
    "R":         [(Fraction(1), "ABC")],
    "SL":        [(Fraction(1, 3), "A"), (Fraction(1, 3), "B"), (Fraction(1, 3), "C")],
    "SLr-early": [(Fraction(15, 100), "A"), (Fraction(15, 100), "AB"), (Fraction(70, 100), "ABC")],
    "SLr-mid":   [(Fraction(1, 3), "A"), (Fraction(1, 3), "AB"), (Fraction(1, 3), "ABC")],
    "SLr-late":  [(Fraction(50, 100), "A"), (Fraction(30, 100), "AB"), (Fraction(20, 100), "ABC")],
}
# "LS" is NOT its own stage list: it is SL's stage blocks (built exactly as for SL) concatenated
# in reverse order, so SL and LS consume the identical multiset of examples.
BASE_SCHEDULES = list(SCHEDULES) + ["LS"]
GRID_CONDITIONS = ["R", "SL", "SLr-early", "SLr-mid", "SLr-late", "LS",
                   "SL-shuf", "SLr-early-shuf", "SLr-mid-shuf", "SLr-late-shuf"]


def _stage_weights(avail, groups, mix):
    """{length: Fraction weight} for one stage; weights sum to 1."""
    ranges = [range(groups[g][0], groups[g][1] + 1) for g in avail]  # groups may be lists (YAML)
    if mix == "lengths":  # uniform over every available length
        lengths = sorted({L for r in ranges for L in r})
        return {L: Fraction(1, len(lengths)) for L in lengths}
    if mix == "groups":   # equal mass per group, uniform over lengths inside a group
        w = {}
        for r in ranges:
            for L in r:
                w[L] = w.get(L, Fraction(0)) + Fraction(1, len(ranges) * len(r))
        return w
    raise ValueError(f"unknown mix {mix!r} (use 'lengths' or 'groups')")


def _balanced_counts(n_examples, weights):
    """Exact integer counts per length summing to n_examples, by largest remainder.
    Leftover examples go to the largest fractional parts; ties go to the smaller length."""
    quota = {L: n_examples * w for L, w in weights.items()}
    counts = {L: math.floor(q) for L, q in quota.items()}
    leftover = n_examples - sum(counts.values())
    for L in sorted(weights, key=lambda L: (-(quota[L] - counts[L]), L))[:leftover]:
        counts[L] += 1
    return counts


def _stage_blocks(base, steps, batch_size, seed, phase, groups, mix):
    """One int64 array of lengths per stage of SCHEDULES[base], in stage order.
    Stage i covers batches [floor(cum_{i-1} * steps), floor(cum_i * steps)); the last ends at steps."""
    stages = SCHEDULES[base]
    blocks, cum, start = [], Fraction(0), 0
    for i, (frac, avail) in enumerate(stages):
        cum += frac
        end = steps if i == len(stages) - 1 else math.floor(cum * steps)
        counts = _balanced_counts((end - start) * batch_size, _stage_weights(avail, groups, mix))
        lengths = sorted(counts)  # ascending, so the permutation input is canonical
        block = np.repeat(np.array(lengths, dtype=np.int64), [counts[L] for L in lengths])
        rng = np.random.default_rng([seed, phase, 100, i])
        blocks.append(rng.permutation(block))
        start = end
    return blocks


def length_sequence(schedule: str, steps: int, batch_size: int, seed: int, phase: int = 1,
                    groups: dict = DEFAULT_GROUPS, mix: str = "lengths") -> np.ndarray:
    """int64 array of shape (steps*batch_size,): the length of every training example, in order.
    Names: a base schedule (SCHEDULES keys or "LS"), or a base + "-shuf" (its shuffled twin)."""
    if schedule.endswith("-shuf"):
        base = schedule[: -len("-shuf")]
        if base not in BASE_SCHEDULES:  # rejects e.g. "SL-shuf-shuf" and unknown bases
            raise ValueError(f"unknown schedule {schedule!r}")
        base_seq = length_sequence(base, steps, batch_size, seed, phase, groups, mix)
        # Sorting first makes the twin depend only on the multiset, so SL-shuf == LS-shuf exactly.
        rng = np.random.default_rng([seed, phase, 101])
        return rng.permutation(np.sort(base_seq))
    if schedule == "LS":
        return np.concatenate(_stage_blocks("SL", steps, batch_size, seed, phase, groups, mix)[::-1])
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}")
    return np.concatenate(_stage_blocks(schedule, steps, batch_size, seed, phase, groups, mix))


# ---------------------------------------------------------------------------
# Training examples: one deterministic virtual pool per (seed, phase)
# ---------------------------------------------------------------------------
class ExampleStream:
    """Deterministic virtual pool for one (seed, phase). The k-th example of length L is fixed
    and does not depend on how it was requested (one at a time or in bulk), or on which schedule
    requested it. Per length L, rows come from rng = default_rng([seed, phase, L]) in chunks of
    CHUNK rows and are consumed in order. Only the current chunk per length is kept in memory."""
    CHUNK = 4096

    def __init__(self, seed: int, phase: int = 1):
        self.seed, self.phase = seed, phase
        self._rng = {}    # length -> np.random.Generator
        self._chunk = {}  # length -> current chunk, shape (CHUNK, L)
        self._pos = {}    # length -> index of the next unread row in the current chunk
        self._used = {}   # length -> number of examples consumed so far

    def take(self, length: int, count: int) -> np.ndarray:
        """The next `count` unconsumed examples of this length, shape (count, length) int64."""
        L = int(length)
        if L not in self._rng:
            self._rng[L] = np.random.default_rng([self.seed, self.phase, L])
            self._chunk[L] = np.empty((0, L), dtype=np.int64)
            self._pos[L] = 0
            self._used[L] = 0
        parts, need = [], int(count)
        while need > 0:
            if self._pos[L] == len(self._chunk[L]):  # current chunk used up: draw the next one
                self._chunk[L] = self._rng[L].integers(VALUE_MIN, VALUE_MAX + 1,
                                                       size=(self.CHUNK, L), dtype=np.int64)
                self._pos[L] = 0
            k = min(need, len(self._chunk[L]) - self._pos[L])
            parts.append(self._chunk[L][self._pos[L]: self._pos[L] + k])
            self._pos[L] += k
            need -= k
        self._used[L] += int(count)
        return np.concatenate(parts) if parts else np.empty((0, L), dtype=np.int64)

    def consumed(self) -> dict:
        """length -> number of examples consumed so far."""
        return dict(self._used)


def pool_examples(seed: int, phase: int, length: int, start: int, count: int) -> np.ndarray:
    """Random access into the same virtual pool: examples [start, start+count) of that length."""
    stream = ExampleStream(seed, phase)
    skipped = 0
    while skipped < start:  # discard the first `start` rows, at most one chunk at a time
        k = min(ExampleStream.CHUNK, start - skipped)
        stream.take(length, k)
        skipped += k
    return stream.take(length, count)


# ---------------------------------------------------------------------------
# Encoding (layout in spec section 1)
# ---------------------------------------------------------------------------
def sort_targets(xs: np.ndarray) -> np.ndarray:
    """(N, n) -> (N, n): each row sorted (non-decreasing, duplicates kept)."""
    return np.sort(np.asarray(xs, dtype=np.int64), axis=1)


def encode_prompt(xs: np.ndarray) -> np.ndarray:
    """(N, n) -> (N, n+2) = [BOS, x, SEP]."""
    xs = np.asarray(xs, dtype=np.int64)
    col = np.ones((len(xs), 1), dtype=np.int64)
    return np.concatenate([BOS * col, xs, SEP * col], axis=1)


def encode_full(xs: np.ndarray) -> np.ndarray:
    """(N, n) -> (N, 2n+2) = [BOS, x, SEP, sorted(x)]."""
    return np.concatenate([encode_prompt(xs), sort_targets(xs)], axis=1)


def _encode_groups(lengths, groups) -> dict:
    """Shared by encode_batch and iterate_batches. lengths: (B,) int64.
    groups: list of (row indices, (k, n) int64 array holding the lists of those rows)."""
    B, Tm = len(lengths), 2 * int(lengths.max()) + 1
    inp = np.full((B, Tm), PAD, dtype=np.int64)
    tgt = np.full((B, Tm), PAD, dtype=np.int64)
    for rows, xs in groups:  # encode all rows of one length at once
        n = xs.shape[1]
        seq = encode_full(xs)  # (k, 2n+2)
        # Write input and target separately: slicing one padded full sequence would leave y_n
        # (instead of PAD) at input index 2n+1 for rows shorter than the batch max.
        inp[rows, : 2 * n + 1] = seq[:, :-1]
        tgt[rows, : 2 * n + 1] = seq[:, 1:]
    # Target index t predicts seq[t+1]; y1..yn sit at seq[n+2..2n+1], so t in [n+1, 2n].
    t = np.arange(Tm)
    loss_mask = (t >= lengths[:, None] + 1) & (t <= 2 * lengths[:, None])
    return {"input": inp, "target": tgt, "loss_mask": loss_mask, "lengths": lengths}


def encode_batch(xs: list) -> dict:
    """xs: list of 1-D int arrays (possibly different lengths), in batch order.
    Returns "input" = padded seq[:-1], "target" = padded seq[1:], "loss_mask", "lengths".
    Rows are right-padded with PAD to Tm = 2*max_len + 1; the loss mask is False on pads."""
    xs = [np.asarray(x, dtype=np.int64) for x in xs]
    if not xs:
        raise ValueError("encode_batch needs at least one example")
    lengths = np.array([len(x) for x in xs], dtype=np.int64)
    groups = []
    for n in np.unique(lengths):
        rows = np.flatnonzero(lengths == n)
        groups.append((rows, np.stack([xs[i] for i in rows])))
    return _encode_groups(lengths, groups)


def iterate_batches(length_seq: np.ndarray, batch_size: int, stream: ExampleStream):
    """Yield encode_batch(...) (plus "inputs_raw": the unsorted lists, in order) for consecutive
    batches of length_seq. Each slot gets the next unconsumed example of its length."""
    length_seq = np.asarray(length_seq, dtype=np.int64)
    if len(length_seq) % batch_size != 0:  # the spec always gives steps*batch_size lengths
        raise ValueError("len(length_seq) must be a multiple of batch_size")
    for start in range(0, len(length_seq), batch_size):
        lens = length_seq[start: start + batch_size].copy()
        groups, xs = [], [None] * batch_size
        for L in np.unique(lens):
            # One bulk take per length, handed out in slot order: identical to one take per slot.
            slots = np.flatnonzero(lens == L)
            rows = stream.take(int(L), len(slots))
            groups.append((slots, rows))
            for slot, row in zip(slots, rows):
                xs[slot] = row
        # Same result as encode_batch(xs), without re-grouping the rows by length.
        batch = _encode_groups(lens, groups)
        batch["inputs_raw"] = xs
        yield batch


# ---------------------------------------------------------------------------
# Fixed evaluation sets: identical for every run, never drawn from the training streams
# ---------------------------------------------------------------------------
VAL_SEED, TEST_SEED, REP_SEED = 1_000_001, 2_000_002, 3_000_003


def uniform_set(length: int, n: int, seed: int) -> np.ndarray:
    """(n, length) iid uniform over [1, 100] with replacement."""
    rng = np.random.default_rng([seed, length])
    return rng.integers(VALUE_MIN, VALUE_MAX + 1, size=(n, length), dtype=np.int64)


def rep_set(length: int, r: int, n: int, seed: int = REP_SEED) -> np.ndarray:
    """A&G rep(i, r), shape (n, length): per row, draw k = ceil(length/r) DISTINCT values,
    repeat each r times, truncate to `length` (so the last value may appear < r times), shuffle."""
    k = math.ceil(length / r)
    if k > VALUE_MAX - VALUE_MIN + 1:
        raise ValueError(f"rep_set needs {k} distinct values but only 100 exist")
    rng = np.random.default_rng([seed, length, r])
    out = np.empty((n, length), dtype=np.int64)
    for i in range(n):
        values = rng.choice(np.arange(VALUE_MIN, VALUE_MAX + 1), size=k, replace=False)
        out[i] = rng.permutation(np.repeat(values, r)[:length])
    return out


def build_eval_sets(lengths: list, n: int, seed: int,
                    rep_lengths: list = (), rep_rs: list = (), rep_n: int = 0) -> dict:
    """name -> {"kind": "uniform"|"rep", "length": int, "r": int|None, "inputs": (N, L) int64}.
    Names are "len_{L}" and "rep_{L}_r{r}". Uniform sets use `seed` (VAL_SEED or TEST_SEED);
    rep sets always use rep_set's default REP_SEED, so they are the same whatever `seed` is.
    rep sets are only built when rep_n > 0 (an empty eval set would have no metrics)."""
    sets = {}
    for L in lengths:
        sets[f"len_{L}"] = {"kind": "uniform", "length": int(L), "r": None,
                            "inputs": uniform_set(int(L), n, seed)}
    if rep_n > 0:
        for L in rep_lengths:
            for r in rep_rs:
                sets[f"rep_{L}_r{r}"] = {"kind": "rep", "length": int(L), "r": int(r),
                                         "inputs": rep_set(int(L), int(r), rep_n)}
    return sets


def count_overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Number of rows of a that also appear as rows of b (same width). Duplicate rows of a
    are each counted."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"widths differ: {a.shape[1]} vs {b.shape[1]}")
    rows_b = {row.tobytes() for row in b}
    return sum(row.tobytes() in rows_b for row in a)
