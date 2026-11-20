# Code specification

Module interfaces, data formats and output schemas for the code in this repository.

**Environment:**
- Windows 11; Python 3.12.1; torch 2.5.1+cu121 (CUDA available, RTX 3070 Ti 8 GB); numpy 2.1; pandas 2.2; matplotlib 3.9; scipy 1.14; PyYAML 6; pytest 9; tqdm.
- Run tests from the project root with `python -m pytest tests/<file> -q`. Unit tests must run on **CPU** and finish quickly (< ~30 s per file).
- No `torch.compile`. No DataLoader worker processes (Windows); generate batches in-process with numpy.

---

## 1. Tokens and sequence layout (`src/data.py` owns these constants)

| Name | Value |
|---|---|
| `PAD` | 0 |
| values v ∈ [1, 100] | token id = v (identity mapping) |
| `SEP` | 101 (end-of-input delimiter) |
| `BOS` | 102 |
| `VOCAB_SIZE` | 103 |
| `VALUE_MIN`, `VALUE_MAX` | 1, 100 |

**Example layout:** an example of length n is a list x of n ints. Its target is `y = sorted(x)` (non-decreasing; duplicates kept).
```
full sequence (len 2n+2):  [BOS, x1..xn, SEP, y1..yn]
seq index:                   0   1..n    n+1  n+2..2n+1
model input  = seq[:-1]   (len 2n+1)
model target = seq[1:]    (len 2n+1); target index t is the prediction of seq[t+1]
loss mask    = True exactly for t in [n+1, 2n]  (n positions)
```
So the prediction made **at the SEP position (t = n+1)** is y1, and the prediction at the position holding y_j is y_{j+1}. Nothing on the input side contributes to the loss. There is **no EOS**.

**Padded batches:** sequences are right-padded with PAD to the longest in the batch. Causal attention plus right padding means real positions never see pads, so no key-padding mask is needed. The loss mask is False on pads.

---

## 2. `src/data.py`

```python
PAD, SEP, BOS, VOCAB_SIZE, VALUE_MIN, VALUE_MAX          # constants above
DEFAULT_GROUPS = {"A": (5, 10), "B": (11, 15), "C": (16, 20)}   # inclusive ranges

# Base schedules: a list of stages (fraction_of_steps, groups available in that stage)
SCHEDULES = {
    "R":         [(1,   "ABC")],
    "SL":        [(1/3, "A"), (1/3, "B"), (1/3, "C")],
    "SLr-early": [(0.15, "A"), (0.15, "AB"), (0.70, "ABC")],
    "SLr-mid":   [(1/3, "A"), (1/3, "AB"), (1/3, "ABC")],
    "SLr-late":  [(0.50, "A"), (0.30, "AB"), (0.20, "ABC")],
}
# "LS" is NOT an independent stage list. It is SL's stage blocks built exactly as for SL, then
# concatenated in reverse block order, so SL and LS consume the IDENTICAL multiset of examples.
# Any base name + "-shuf" is its shuffled twin (see below).
GRID_CONDITIONS = ["R", "SL", "SLr-early", "SLr-mid", "SLr-late", "LS",
                   "SL-shuf", "SLr-early-shuf", "SLr-mid-shuf", "SLr-late-shuf"]

def length_sequence(schedule: str, steps: int, batch_size: int, seed: int, phase: int = 1,
                    groups: dict = DEFAULT_GROUPS, mix: str = "lengths") -> np.ndarray:
    """int64 array of shape (steps*batch_size,): the length of every training example, in order.
    Stage boundaries fall on batch boundaries: stage i covers batches
    [floor(cum_frac_{i-1}*steps), floor(cum_frac_i*steps)), with exact fractions
    (fractions.Fraction(1, 3) for the thirds). The last stage ends at `steps`.
    Within a stage, lengths are BALANCED, not iid. Exact per-length counts come from the
    stage's weights by largest remainder (ties broken by smaller length first), then
    np.repeat, then a permutation with rng = np.random.default_rng([seed, phase, 100, stage_idx]).
    mix="lengths": uniform over all lengths in the available groups (the default; the final
                   stage of SLr-* then has the same length distribution as R).
    mix="groups":  equal mass per available group, uniform over lengths within a group.
    Twins ("<base>-shuf"): rng = np.random.default_rng([seed, phase, 101]);
        twin = rng.permutation(np.sort(base_sequence))
      Sorting first makes "SL-shuf" and "LS-shuf" produce the IDENTICAL array.
    Must raise ValueError for unknown names."""

class ExampleStream:
    """Deterministic virtual pool for one (seed, phase). The k-th example of length L is fixed
    and does not depend on how it was requested (one at a time or in bulk), or on which
    schedule requested it. Per length L: rng = np.random.default_rng([seed, phase, L]); rows are
    produced in chunks of CHUNK = 4096 via rng.integers(VALUE_MIN, VALUE_MAX+1, size=(CHUNK, L),
    dtype=np.int64) and consumed in order."""
    CHUNK = 4096
    def __init__(self, seed: int, phase: int = 1): ...
    def take(self, length: int, count: int) -> np.ndarray:   # (count, length) int64, consumes them
        ...
    def consumed(self) -> dict[int, int]:                    # length -> number consumed so far
        ...

def pool_examples(seed: int, phase: int, length: int, start: int, count: int) -> np.ndarray:
    """Random access into the same virtual pool: examples [start, start+count) of that length."""

def encode_batch(xs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """xs: list of 1-D int arrays (unsorted lists, possibly different lengths), in batch order.
    Returns int64/bool numpy arrays:
      "input":     (B, Tm) int64   = padded seq[:-1]
      "target":    (B, Tm) int64   = padded seq[1:]   (PAD where padded)
      "loss_mask": (B, Tm) bool    (section 1)
      "lengths":   (B,)    int64
    with Tm = 2*max_len + 1."""

def iterate_batches(length_seq: np.ndarray, batch_size: int, stream: ExampleStream):
    """Yield encode_batch(...) for consecutive batches of length_seq. The example for each slot
    is the next unconsumed example of that slot's length (stream.take). Batch order is
    preserved. Also add key "inputs_raw": list[np.ndarray] (the unsorted lists, in order)."""

def encode_prompt(xs: np.ndarray) -> np.ndarray:   # (N, n) -> (N, n+2) = [BOS, x, SEP]
def encode_full(xs: np.ndarray) -> np.ndarray:     # (N, n) -> (N, 2n+2)
def sort_targets(xs: np.ndarray) -> np.ndarray:    # np.sort(xs, axis=1)

# Fixed evaluation sets: identical for every run, never drawn from the training streams
VAL_SEED, TEST_SEED, REP_SEED = 1_000_001, 2_000_002, 3_000_003
def uniform_set(length: int, n: int, seed: int) -> np.ndarray:
    """(n, length) iid uniform over [1,100] with replacement; rng default_rng([seed, length])."""
def rep_set(length: int, r: int, n: int, seed: int = REP_SEED) -> np.ndarray:
    """A&G rep(i, r): per row, draw k = ceil(length/r) DISTINCT values from [1,100], repeat each
    r times, truncate to `length`, shuffle. rng default_rng([seed, length, r])."""
def build_eval_sets(lengths: list[int], n: int, seed: int,
                    rep_lengths: list[int] = (), rep_rs: list[int] = (), rep_n: int = 0) -> dict:
    """name -> {"kind": "uniform"|"rep", "length": int, "r": int|None, "inputs": (N, L) int64}.
    Names are "len_{L}" and "rep_{L}_r{r}"."""
def count_overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Number of rows of a that also appear as rows of b (same width)."""
```

**Phase 2 data** (identical for every model of a seed, whatever its Phase-1 schedule):
`length_sequence("R", phase2_steps, batch_size, seed, phase=2)` with `ExampleStream(seed, phase=2)`.

---

## 3. `src/model.py`

```python
@dataclass
class ModelConfig:
    vocab_size: int = 103
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    d_mlp: int = 1024
    pe: str = "sinusoidal"      # "sinusoidal" | "none" (NoPE)
    max_len: int = 512          # sinusoidal table length (>= 2*50+2)

class SortTransformer(nn.Module):
    tok_emb: nn.Embedding        # (vocab, d), the "encoder basis"
    blocks: nn.ModuleList        # pre-LN: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
    ln_f: nn.LayerNorm
    unembed: nn.Linear           # d -> vocab, bias=False, UNTIED from tok_emb ("decoder basis")
    def forward(self, idx: torch.LongTensor, return_internals: bool = False):
        """idx (B, T) -> logits (B, T, vocab). With return_internals=True, returns
        (logits, internals), where internals is a dict of lists over layers l:
          "resid_pre"[l]  (B,T,d)   input to block l
          "resid_mid"[l]  (B,T,d)   after the attention residual add
          "resid_post"[l] (B,T,d)   after the MLP residual add
          "attn"[l]       (B,H,T,T) attention probabilities (rows = queries)
          "final"         (B,T,d)   ln_f output (what unembed sees)"""

def build_model(cfg: ModelConfig, seed: int) -> SortTransformer:
    """torch.manual_seed(seed), then construct. Init: normal(0, 0.02) for Linear/Embedding
    weights, zeros for biases, LayerNorm weight 1 and bias 0. Init depends only on (cfg, seed)."""
def count_params(model) -> int
```
- Embedding: `x = tok_emb(idx) (+ pe[:T] if sinusoidal)`. Vaswani sin/cos table, position 0 = the BOS position, **no sqrt(d) scaling**. NoPE adds nothing.
- Attention: causal, multi-head, scale 1/sqrt(head_dim), q/k/v/out Linear layers with bias. GELU MLP. No dropout.
- The fast path (`return_internals=False`) may use `F.scaled_dot_product_attention(is_causal=True)`. The internals path computes attention explicitly. **Both must give the same logits** (tested).

---

## 4. `src/metrics.py` (pure numpy) and `src/evaluate.py` (torch)

```python
# metrics.py: pred, target, inputs are (N, n) int arrays; target = sorted(inputs)
def exact_match(pred, target) -> np.ndarray          # (N,) bool
def per_position_acc(pred, target) -> np.ndarray     # (n,) float
def prefix_len(pred, target) -> np.ndarray           # (N,) int: correct tokens before the first error
def inversions(pred) -> np.ndarray                   # (N,) int: pairs i<j with pred[i] > pred[j]
def adjacent_inversions(pred) -> np.ndarray          # (N,) int: i with pred[i] > pred[i+1]
def multiset_diff(pred, inputs) -> np.ndarray        # (N,) int: sum_v |count_pred(v) - count_in(v)|, over all token ids
def edit_distance(pred, target) -> np.ndarray        # (N,) int: Levenshtein, batched numpy DP (vectorized over N)
def error_types(pred, inputs) -> dict[str, float]    # fractions: correct / order_only / multiset_only / both
def compute_metrics(pred, inputs, n_samples: int = 20) -> dict   # schema below
```
**Error-type partition:**
- `sorted_ok` = pred is non-decreasing; `ms_ok` = (multiset_diff == 0).
- correct = exact match (equivalent to sorted_ok & ms_ok).
- order_only = ms_ok & ~sorted_ok.
- multiset_only = sorted_ok & ~ms_ok.
- both = ~sorted_ok & ~ms_ok.

**Metric dict schema** (one per eval set; JSON-serialisable Python floats/ints/lists only):
```json
{"kind": "uniform", "length": 30, "r": null, "n": 1000,
 "exact_match": 0.0, "token_acc": 0.0, "per_position_acc": [/* n floats */],
 "prefix_len": 0.0, "prefix_frac": 0.0,
 "inversions": 0.0, "adjacent_inversions": 0.0,
 "multiset_diff": 0.0, "multiset_ok": 0.0,
 "edit_distance": 0.0, "edit_distance_norm": 0.0,
 "error_types": {"correct": 0.0, "order_only": 0.0, "multiset_only": 0.0, "both": 0.0},
 "samples": [{"input": [], "target": [], "output": []}]}
```
- `kind`/`length`/`r` are filled in by `evaluate_sets` (`compute_metrics` may leave them out).
- `samples` holds the first `n_samples` rows, for inspecting raw outputs.
- Scalar metrics are means over the N rows.
- `prefix_frac = prefix_len / n`; `edit_distance_norm = edit_distance / n`.

```python
# evaluate.py
@torch.no_grad()
def greedy_decode(model, inputs: np.ndarray, device, batch_size: int = 1000,
                  decode_vocab: str = "values") -> np.ndarray:
    """inputs (N, n), all the same length. Prompt [BOS, x, SEP]; generate exactly n tokens greedily.
    decode_vocab="values": argmax restricted to ids 1..100 (default). "full": argmax over the whole vocab.
    Puts model in eval() and restores its previous training mode. Returns (N, n) int64."""
def evaluate_sets(model, sets: dict, device, batch_size=1000, decode_vocab="values",
                  n_samples=20) -> dict:   # name -> metric dict (with kind/length/r filled in)
def main():  # CLI: python -m src.evaluate --run results/<run_id> --phase {1,2} [--out FILE]
             # rebuild the eval sets from the run's config.yaml, load model_phase{p}.pt, write JSON
```

---

## 5. Training and run outputs (`src/train.py`, `configs/`, `scripts/sweep.py`)

**CLI:**
```
python -m src.train --config configs/base.yaml [--schedule X] [--pe sinusoidal|none] [--seed N]
                    [--tag T] [--set dotted.key=value ...]
```
- Run id: `f"{pe}_{schedule}_s{seed}"`, plus `f"_{tag}"` if a tag is given.
- Output dir: `<out_dir>/<run_id>/`. Re-running an existing finished run (one that has `done.json`) is refused unless `--overwrite` is passed.

**Config keys** (`configs/base.yaml`; all overridable):
```yaml
out_dir: results
seed: 0
tag: ""
model: {d_model: 256, n_layers: 2, n_heads: 8, d_mlp: 1024, pe: sinusoidal, max_len: 512}
data:  {schedule: R, groups: {A: [5, 10], B: [11, 15], C: [16, 20]}, mix: lengths}
train: {steps: 20000, batch_size: 256, lr: 5.0e-4, warmup_steps: 1000, weight_decay: 0.0,
        betas: [0.9, 0.98], grad_clip: 1.0, device: auto, log_every: 100,
        loss_norm: example}     # example | token (see implementation notes)
phase2: {steps: 5000, reset_optimizer: false}       # steps: 0 disables Phase 2
eval:
  every: 500                    # learning-curve eval interval (steps)
  val_per_length: 100           # val set = uniform_set(L, n, VAL_SEED) for train lengths
  curve_ood_lengths: [22, 25, 30]   # also tracked during training (ignored when id_only)
  test_lengths: [5, 6, ..., 24, 25, 30, 40, 50]   # written out explicitly
  test_per_length: 1000
  rep_lengths: [10, 20, 30, 40]
  rep_rs: [2, 3, 4, 5]
  rep_per_set: 500
  batch_size: 250               # not 1000: see implementation notes
  decode_vocab: values
  n_samples: 20
  id_only: false                # TUNING MODE: if true, no length > max train length is ever evaluated
save_checkpoints: true
```
- **LR schedule:** linear warmup from 0 to `lr` over `warmup_steps`, then constant. Optimizer: AdamW.
- **Phase 2** continues from the Phase-1 weights. The optimizer state carries over unless `reset_optimizer: true`. There is no new warmup, and the LR stays constant.
- **Loss:** with `loss_norm: example` (default), each example's mean cross-entropy over its `loss_mask` positions, then the mean over the batch. With `token`, the mean over all masked positions in the batch.
- **`id_only` enforcement:** when `eval.id_only` is true, `train.py` must hard-filter every eval set (test, rep, curve) to lengths ≤ max train length. A test checks this.

**Per-run output files** in `<out_dir>/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | Fully resolved config, plus `run_id` |
| `train_log.jsonl` | Two record types. Loss records: `{"type":"loss","phase":1,"step":s,"loss":mean_since_last,"lr":..,"tokens_nonpad":cum,"tokens_padded":cum,"examples":cum}`. Eval records: `{"type":"eval","phase":1,"step":s,"val":{"5":em,...},"val_group":{"A":em,"B":em,"C":em},"val_token_acc":..,"curve_ood":{"22":em,...}}` (em = greedy exact match). Steps count from 0 within each phase. Also log an eval record at step 0 and at the end of each phase. |
| `eval_phase1.json`, `eval_phase2.json` | `{"run_id","phase","steps","tokens_nonpad","tokens_padded","wall_time_s","test_sets":{name: metric dict}}` |
| `model_phase1.pt`, `model_phase2.pt` | `{"state_dict":..., "model_config": asdict(ModelConfig), "run_config": dict}` |
| `done.json` | Written LAST: `{"run_id", "finished": iso_time, "phase1_steps", "phase2_steps", "tokens_nonpad_phase1", ..., "config_hash", "experiment_hash", "overlap": {"phase1": {"5": {"val": k, "test": k}, ...}, "phase2": {...}}}`. It is the sweep's completion marker. |

**`scripts/sweep.py`:**
- `--grid {pilot,tune,main,nope,nope_full} [--seeds 0 1 2 3 4] [--dry-run] [--overwrite]`. Runs each config sequentially in a subprocess.
  - It skips a run only if its `done.json` has the same `config_hash` as the config it would run now.
  - On a mismatch it reports the run as STALE and does not run it; the exit code is nonzero.
- **pilot:** R and SLr-mid × {sinusoidal, none} × seed 0.
- **tune:** R × an lr grid {3e-4, 5e-4, 1e-3} (tag `lr{value}`) × seeds {0, 1}, using `configs/tune.yaml` (`id_only: true`, `phase2.steps: 0`, `curve_ood_lengths: []`, `save_checkpoints: false`).
- **main:** GRID_CONDITIONS × seeds × sinusoidal.
- **nope:** [R, SL, SLr-mid, LS, SL-shuf, SLr-mid-shuf] × seeds × none.
- **nope_full:** GRID_CONDITIONS × seeds × none.

---

## 6. Analysis (`src/analysis/`)

- `collect.py`:
  - `load_runs(results_dir) -> (metrics_df, curves_df)`, using only runs that have `done.json`.
  - metrics_df has one row per (run, phase, eval set), with columns `run_id, pe, schedule, base_schedule, is_twin, seed, tag, experiment_hash, phase, set_name, kind, length, r` plus every scalar metric.
  - It warns when finished runs sharing (pe, tag) have different experiment hashes.
  - curves_df comes from the eval records in `train_log.jsonl`.
  - CLI writes `results/metrics.csv` and `results/curves.csv`.
- `stats.py`: seed mean/std; bootstrap CIs; paired comparisons over seeds (schedule vs R, schedule vs its twin): mean paired difference, bootstrap CI and Wilcoxon p-value, with an explicit n.
- `plots.py`: the paper figures (length curves per PE, schedule vs twin, learning/forgetting curves, Phase 1 vs 2, error types, probe accuracy vs length). They go to `figures/` as PDF and PNG.
- `probes.py`: A&G Section 4 mechanism probes (min-finding at SEP after layer-1 attention; identity+successor after layer-2 attention) per run and per length. CLI writes `<run_dir>/probes_phase{p}.json`. The readout details must follow the paper (arXiv:2310.00726, Section 4). Anything the paper leaves unspecified is documented in the module docstring.
- `attention.py`: attention heatmaps for one example (per layer and head) for a given run, phase and length.

---

## 7. Design defaults

1. **Replay mixing** (`data.mix`): `lengths`, i.e. uniform over all lengths currently available. The alternative is `groups`.
2. **Decoding vocabulary** (`eval.decode_vocab`): `values`, i.e. argmax only over 1–100. The alternative is `full`.
3. **Phase 2 optimizer** (`phase2.reset_optimizer`): `false`, so the AdamW state carries over.
4. Pre-LN blocks; no embedding scaling with sinusoidal PE.
5. **Loss normalization** (`train.loss_norm`): `example`, i.e. every example has equal weight. This keeps a schedule and its shuffled twin identical in everything except order. The alternative is `token`.
6. **Warmup** (1000 steps) falls inside the first stage, so LS warms up on lengths 16–20 and SL on 5–10. This is inherent to the design.

## Implementation notes

- **Evaluation batch size** (`eval.batch_size: 250`): greedy decoding creates a slightly larger activation buffer at every step. At batch 1000 this fragments the CUDA caching allocator on an 8 GB GPU, and decoding becomes 10–25× slower or runs out of memory. At batch 250, 1000 length-50 lists decode in about 4 s. Any GPU code that decodes or probes long sequences uses chunks of at most 250 rows.
- **Loss normalization** (`train.loss_norm: example` | `token`):
  - `example` (default) takes each example's mean cross-entropy over its n masked tokens, then the mean over the batch, so every example has weight 1/B regardless of batch composition.
  - With `token`, a schedule's batches (a single length group) and its shuffled twin's batches (mixed lengths) weight the same examples differently, which confounds order with loss weighting: SL gives about 33% of the weight to each group, SL-shuf 19.5/33.8/46.7%.
- **Stale-run protection:** `done.json` stores `config_hash`, the sha256 of the canonical JSON of the resolved config, excluding keys that can't change results (`out_dir`, `run_id`, `train.device`, `train.log_every`, `eval.batch_size`, `eval.n_samples`, `save_checkpoints`).
  - `scripts/sweep.py` skips a finished run only if its stored hash equals the hash of the config it would run now. Otherwise it reports STALE and does not run it; pass `--overwrite` or use a `--tag`.
  - `collect.py` adds an `experiment_hash` column (the hash also excluding `seed`, `tag`, `data.schedule` and `model.pe`) and warns when finished runs sharing (pe, tag) have different experiment hashes.
- **Overlap check:** `done.json` contains `overlap`, nested as {phase: {length: {set: count}}}. Each count is the number of eval rows (val, test and uniform curve sets) that also appear among the training rows consumed at that length, for lengths ≤ 8.
- **Probe chance references:** each `by_length` entry has `min_mode_frac` (the share of rows whose min-probe argmax equals the most common argmax) and `min_chance` (max over tokens c of the share of rows whose min is c, i.e. the best constant guess). The same pair is given for the after-layer-2 readout.
  - fig6 draws `min_chance` as a dashed reference and also plots the after-layer-2 min readout.
  - Further keys: `min_mode_token`, `min_mode_frac_after_attn1`, and `min_acc_centered` with its `_after_attn1` and `_after_attn2` variants (argmax after subtracting each token's mean score over the rows of that length; use it only as a per-length check for row-specific signal, not across lengths).
  - probes/attention refuse lengths above the max train length when the run's `eval.id_only` is true.
- **Statistics:** the Wilcoxon test uses `scipy.stats.PermutationMethod()`, which is exact under ties.
- **Tuning runs:** `configs/tune.yaml` sets `save_checkpoints: false`, so a tuning model can never be evaluated at lengths > 20 later. Add `--set save_checkpoints=true` to keep a checkpoint.
