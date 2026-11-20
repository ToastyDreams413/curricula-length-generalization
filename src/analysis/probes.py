"""Mechanism probes of Awasthi & Gupta (2023), arXiv:2310.00726v1, Section 4, run per training run.

CLI:
    python -m src.analysis.probes --run results/<run_id> --phase 1 \
        --lengths 5 10 15 20 25 30 40 50 --n 500 [--device auto] [--max-len N]
        [--apply-ln-f] [--vocab full|values] [--dup-policy exclude|strict] [--out FILE]
writes <run_dir>/probes_phase{p}.json:
    {"run_id", "phase", "readout": str, "by_length": {"<L>": {"n", "min_acc", "id_succ_acc",
     "min_chance", "min_mode_frac", "min_acc_centered", ...}}}
min_acc must be read against min_chance / min_mode_frac (see CHANCE REFERENCES below).

WHAT THE PAPER SAYS (Section 4, "Interpreting Hints")
  Notation: E (q x d) is the input embedding table; its rows are the "encoder basis". (W, b) is the
  final classifier; "the columns of W ... we call these vectors the decoder basis". X^pre_{i,j} is
  "the embedding of token sigma_i obtained at depth j-th before applying the MLP at that depth",
  X^post_{i,j} the embedding after the MLP. Depth is 0-indexed in the text and 1-indexed in the
  captions of Figs. 4.15/4.16 ("after layer-1 attention" = depth 0 pre-MLP). Their sequence is
  sigma_0..sigma_{n-1}, then the delimiter at index n, then the output. There is NO BOS, so every one
  of their indices is ours minus 1.
  Measured probes (Figs. 4.15, 4.16; Theorem 5.1 items 2-3 state the same properties):
    (i)  min finding: "the embedding for the [delimiter] token after the depth-0 self-attention layer
         computes the minimum input element (this is measured by computing the dot-product of the
         embedding with the decoding basis)". Fig. 4.14 text: "the embedding has largest inner product
         with the vector in the decoding basis that corresponds to the minimum element in the input".
    (ii) Identity+Successor: "the embedding for tokens in the output sequence (those after [delimiter])
         correctly implement the Identity+Successor mechanism after the layer-2 attention operation".
         Fig. 4.13 (depth-1 pre-MLP, decoder basis) is "highly concentrated on both token 43 and on its
         successor in the sorted sequence, i.e., on token 78" ("this two-peak phenomenon").
  Also observed, but not measured as an accuracy: input tokens copy themselves in the encoder basis
  (Fig. 4.12, Theorem 5.1 item 1), and the layer-2 MLP is a "denoiser" that removes the identity
  peak, leaving only the successor (Fig. 4.13, post-MLP).
  Evaluated lengths: the vector figures (figs/sorting_acc_101_min_finding.pdf, figs/sorting_acc_succ.pdf
  in the arXiv source) have 11 points per curve, at L = 20, 40, 60, 80, 90, 100, 150, 200, 300, 400, 500.
  Training lengths were 2-20. Uniform test data, values 1..100 with replacement.

OUR MAPPING (pre-LN blocks: x = x + attn(ln1(x)); x = x + mlp(ln2(x)); then ln_f; then unembed)
  * "after layer-1 attention / depth-0 pre-MLP" = internals["resid_mid"][0], the residual stream after
    the attention residual add and before ln2/MLP. "after layer-2 attention" = internals["resid_mid"][1].
    (In A&G's theory section the block is post-LN, normalize(.) after the MLP. Under either placement,
    the pre-MLP embedding has not yet been normalized in that block, so resid_mid is the match.)
  * Decoder basis = rows of unembed.weight (vocab x d; our W^T). Score = <h, unembed.weight[v]>.
    The paper mentions no bias, and our unembed has none.
  * Positions in the model input encode_full(x)[:, :-1] (length 2n+1): BOS at 0, x at 1..n, SEP at
    n+1, y_1..y_{n-1} at n+2..2n. y_n is never an input token (no EOS, and nothing predicts after it).
    min probe: index n+1. Identity+Successor: indices n+2..2n (the token there is y_j, j = 1..n-1).
  * Architecture differences that affect these readouts:
    - A&G's model is NoPE (App. B, Table 4). With pe="sinusoidal", every residual also carries pe[t].
      The SEP readout then includes a row-constant, length-dependent term (tok_emb[SEP] + pe[n+1]) . W,
      at positions (> 2*20+1) that training never reached. min_acc can therefore drop with length
      for reasons unrelated to the mechanism. Diagnostic: min_acc_attn_only scores the layer-1
      attention write alone (resid_mid[0] - resid_pre[0]), which removes that term.
    - Table 4 does not state where A&G's LayerNorms sit (their theory section uses post-LN). Ours is
      pre-LN with a final ln_f.

CHOICES THE PAPER LEAVES UNSPECIFIED (each one is an argument of run_probes)
  1. LayerNorm before projecting (apply_ln_f, default False). The paper only says "dot-product of the
     embedding with the decoding basis", so the default is the literal raw residual. ln_f changes the
     argmax only through mean-centering and its elementwise gain/bias (the 1/sigma scale cannot).
     Both variants are always reported (keys *_raw_* and *_lnf_*).
  2. Vocabulary scored (vocab, default "full"). Figs. 4.12-4.14 plot projections on all ~103 basis
     vectors (x-axis 0..102), and "largest inner product with the vector in the decoding basis" is an
     argmax over all of them. "values" restricts to ids 1..100. Both are always reported.
  3. Success criterion for Identity+Successor. The paper gives no formula ("correctly implement").
     We read the "two-peak phenomenon" as: the top-2 set equals {current, successor}.
  4. Duplicates (dup_policy). The paper's example has distinct values, and the text pulls two ways:
     - Theorem 5.1(3): "For any position i>n, ... concentrated ... on the token at position i and the
       next largest element in the sorted sequence". This supports "strict": every output position
       whose token a has a strictly larger input value, with target {a, min{b in x : b > a}}.
       Positions whose token is the input max are dropped.
     - App. A.8(2) claims the two peaks only for "the last occurrence of any symbol a in the output".
       This supports "exclude" (the DEFAULT): score only positions with y_{j+1} != y_j. There, y_{j+1} is
       both the next output and the strict successor, so every reading agrees.
     Duplicates do NOT seem to count as failures in Fig. 4.16: their no-hint curve reads ~96% at
     L = 40, while about 18% of length-40 positions are duplicates, which would cap the score near 82%.
     Both policies are always reported (id_succ_acc_change, id_succ_acc_strict), along with the
     number of scored positions, since the scored share shrinks as lists get longer and repeat more.
  5. Teacher forcing. The paper does not say whether the output tokens are ground truth or
     model-generated. We always teacher-force the true sorted prefix. This separates the mechanism from
     upstream decoding errors, and keeps "current token" and "successor" well defined. (Under free
     generation, the successor of a wrong prefix is undefined.)
  6. Ties in the argmax / top-2 are left to torch.topk (measure zero for float scores).

EXTRA DIAGNOSTIC KEYS (per length; plain JSON; a ratio with an empty denominator is null)
  min_acc_{raw,lnf}_{full,values}, id_succ_acc_{raw,lnf}_{full,values}  all four readouts
  id_succ_acc_change / _strict / _dup     value-change positions / strict policy / duplicate positions
                                          only (strict target); primary ln_f + vocab setting
  id_succ_n_change, id_succ_n_strict, id_succ_n_positions   scored / total output positions
  min_rank_mean      mean rank of the true min among the scored tokens (0 = top)
  min_acc_attn_only  min readout on the layer-1 attention write only, resid_mid[0] - resid_pre[0]
                     (raw, same vocab). It drops the row-constant tok_emb[SEP] + pe term; see above.
  min_acc_after_attn1, min_acc_after_attn2   the same min readout at SEP from resid_mid[0] / [1]
                     (primary ln_f + vocab). The first equals min_acc under the defaults. A sanity run
                     on quickly trained scratch models (d=256, 6k steps, R schedule; not project runs)
                     found the min readable only AFTER LAYER-2 attention (about 0.7-0.96 at L <= 20, vs
                     0.06-0.18 after layer 1), unlike A&G (about 60% after layer 1 at L = 20). So track both.

CHANCE REFERENCES FOR THE MIN PROBE (per length, on the same rows)
  Why: on a scratch model the min readout after layer 1 collapsed to ONE token per length (every
  row's argmax = 31 at L=10, 99 at L=20, ...). min_acc then only measures how often min(x) happens to
  equal that constant, and that rate moves with L by itself (P(min(x) = 1) = 1 - 0.99^L: 5% at L=5,
  40% at L=50), which can fake a trend or hide a real one. Read min_acc against these keys:
  min_chance         max over tokens c of the share of rows with min(x) == c: the accuracy of the
                     best CONSTANT guess, i.e. the most a collapsed readout can score (one collapsed
                     onto another token scores P(min(x) = that token), less). min_acc at or below
                     min_chance is no evidence that the min is read out.
  min_mode_frac      share of rows whose min argmax (the readout behind min_acc) equals the most
                     common argmax at this length. 1.0 = collapsed to a constant token. A perfect
                     readout gives min_mode_frac == min_chance.
  min_mode_token     that most common argmax token id (ties: lowest id)
  min_mode_frac_after_attn1, min_mode_frac_after_attn2   the same for min_acc_after_attn1 / _attn2
  min_acc_centered   the min readout after subtracting, from each token's score, that token's MEAN
                     score over the rows of this length (column-centring the N x vocab score matrix),
                     then taking the argmax. This removes any row-constant bias exactly, e.g. the
                     tok_emb[SEP] + pe[n+1] term (and any constant part of the attention write), so it
                     asks "is there row-specific min information?". A row counts only if the true min's
                     centred score is STRICTLY above every other token's (an exact tie, as for a fully
                     collapsed readout, is a miss). Caveats: the mean is estimated on the scored rows
                     themselves (transductive; the leak shrinks as 1/n). Centring also subtracts
                     MORE from tokens that are often the min, so rows with a frequent min are
                     handicapped; a token that is the min of EVERY row is centred to 0 and can never
                     be credited. Because P(min(x) = 1) grows with L, this handicap grows with L:
                     min_acc_centered is NOT trend-safe across lengths (on the accept6k scratch
                     model, after layer 2, rows with min <= 2 fall from 0.91 to 0.71 centred
                     accuracy between L = 5 and 20, other rows stay at 0.91-0.96). Use it to ask
                     "is there any row-specific signal at this length?", not to compare lengths.
                     null when n < 2.
  min_acc_centered_after_attn1, min_acc_centered_after_attn2   the same per layer (primary ln_f +
                     vocab); the first equals min_acc_centered under the defaults.

TUNING HYGIENE (eval.id_only)
  If the run's config (checkpoint run_config or <run_dir>/config.yaml) has eval.id_only: true, no
  length above the max training length (max upper bound over data.groups) may be evaluated.
  probe_run drops such lengths with a printed note (IdOnlyError if none are left). load_run_model
  stamps the limit on the model as model.id_only_max_len, and run_probes refuses longer inputs on
  such a model. A config without an eval section counts as id_only: false.
  id_in_top2, succ_in_top2   on value-change positions: is the identity / successor peak in the top 2?
  enc_copy_acc       input positions 1..n: argmax of resid_mid[0] in the ENCODER basis (tok_emb rows,
                     raw, same vocab) == the token itself (Fig. 4.12 / Theorem 5.1(1))
  tf_sep_acc, tf_next_acc   the model's own teacher-forced logits (same vocab restriction): argmax at
                     SEP == min, and over all n prediction positions == next target. Shows whether the
                     output is right even when a probe fails (the "denoiser" view).

GPU memory (docs/CODE_SPEC.md, implementation notes): rows are processed in chunks of at most
batch_size (default 250). Chunks shrink further for long lists, so the explicit (B, H, T, T)
attention tensors that return_internals builds for every layer stay under max_attn_mb.
Everything runs under torch.no_grad(), because the internals are not detached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from src.data import VALUE_MAX, VALUE_MIN, encode_full, sort_targets

VOCABS = ("full", "values")
DUP_POLICIES = ("exclude", "strict")
DEFAULT_BATCH = 250
DEFAULT_MAX_ATTN_MB = 512
DEFAULT_LENGTHS = (5, 10, 15, 20, 25, 30, 40, 50)


# ---------------------------------------------------------------------------
# Positions (spec section 1)
# ---------------------------------------------------------------------------
def probe_inputs(xs: np.ndarray) -> np.ndarray:
    """(N, n) unsorted lists -> (N, 2n+1) teacher-forced model input [BOS, x, SEP, y_1..y_{n-1}]."""
    return encode_full(np.asarray(xs, dtype=np.int64))[:, :-1]


def sep_index(n: int) -> int:
    """Index of SEP in the model input (BOS at 0, x at 1..n)."""
    return n + 1


def output_positions(n: int) -> np.ndarray:
    """Input indices holding y_1..y_{n-1} (the Identity+Successor probe positions): n+2..2n."""
    return np.arange(n + 2, 2 * n + 1)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def succ_targets(xs: np.ndarray) -> dict:
    """Targets at the probe positions j = 1..n-1 (the token there is y_j). All arrays are (N, n-1):
      cur        y_j
      nxt        y_{j+1}  (the correct next output)
      change     y_{j+1} != y_j  (last occurrence of y_j in the output)
      strict     smallest input value > y_j (only meaningful where has_strict)
      has_strict y_j is not the input max
    On change positions strict == nxt."""
    xs = np.asarray(xs, dtype=np.int64)
    N = xs.shape[0]
    y = sort_targets(xs)
    cur, nxt = y[:, :-1], y[:, 1:]
    V = VALUE_MAX + 2                                  # columns 0..VALUE_MAX+1; V is the "none" sentinel
    present = np.zeros((N, V), dtype=bool)
    present[np.arange(N)[:, None], xs] = True
    cand = np.where(present, np.arange(V), V)
    next_ge = np.minimum.accumulate(cand[:, ::-1], axis=1)[:, ::-1]  # smallest present value >= v
    strict = np.take_along_axis(next_ge, cur + 1, axis=1)
    has_strict = strict < V
    return {"cur": cur, "nxt": nxt, "change": nxt != cur,
            "strict": np.where(has_strict, strict, 0), "has_strict": has_strict}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ratio(num, den):
    return None if den == 0 else float(num) / float(den)


class IdOnlyError(ValueError):
    """A length above the max training length was requested for a run with eval.id_only: true."""


def check_id_only(n: int, limit, what: str = "this model") -> None:
    """Raise IdOnlyError if list length n exceeds the id_only limit (None = no limit)."""
    if limit is not None and int(n) > int(limit):
        raise IdOnlyError(f"{what} comes from a run with eval.id_only: true (tuning mode): lengths "
                          f"above the max training length {int(limit)} must not be evaluated, "
                          f"got length {int(n)}")


def _modal(hist: np.ndarray, N: int):
    """(share of the N rows at the most common token, that token id) from a token histogram."""
    tok = int(np.argmax(hist))
    return _ratio(int(hist[tok]), N), tok


def _centred_acc(scores: torch.Tensor, mins: np.ndarray, a: int):
    """Min accuracy after column-centring the (N, V) score matrix (token id = column + a): a row
    is a hit only if the true min's centred score is strictly above every other column's."""
    N = scores.shape[0]
    if N < 2:
        return None
    s = scores.double()
    s = s - s.mean(dim=0, keepdim=True)
    true = s.gather(1, torch.as_tensor(mins - a, dtype=torch.long)[:, None])
    return _ratio(int(((s >= true).sum(dim=1) == 1).sum()), N)


def resolve_device(name) -> torch.device:
    if name is None or str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_device(model, device):
    if device is not None:
        return resolve_device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _forward(model, idx):
    fwd = model if callable(model) else model.forward
    return fwd(idx, return_internals=True)


def chunk_rows(n: int, model, batch_size: int = DEFAULT_BATCH,
               max_attn_mb: float = DEFAULT_MAX_ATTN_MB) -> int:
    """Rows per forward pass: <= batch_size, and small enough that the explicit fp32 attention
    probabilities of all layers (n_layers x B x H x T x T, T = 2n+1), counted twice for the
    transient scores, fit in max_attn_mb."""
    cfg = getattr(model, "cfg", None)
    H = int(getattr(cfg, "n_heads", 8))
    n_layers = int(getattr(cfg, "n_layers", 2))
    T = 2 * n + 1
    per_row = 2 * n_layers * H * T * T * 4
    return max(1, min(int(batch_size), int(max_attn_mb * 2**20 // per_row)))


def _vocab_slice(vocab: str):
    """(start, stop) column range of the scored tokens; the token id is column index + start."""
    if vocab == "full":
        return 0, None
    return VALUE_MIN, VALUE_MAX + 1


def describe_readout(apply_ln_f=False, vocab="full", dup_policy="exclude",
                     min_layer=0, succ_layer=1) -> str:
    ln = "ln_f applied" if apply_ln_f else "raw residual (no LayerNorm)"
    voc = "full vocab" if vocab == "full" else "value ids 1..100"
    if dup_policy == "exclude":
        pos = "scored where y_{j+1} != y_j"
    else:
        pos = ("target {y_j, min input value > y_j}, scored where y_j < max(x)")
    return (f"teacher-forced; {ln}; dot product with unembed rows (decoder basis) over {voc}. "
            f"min: resid_mid[{min_layer}] (after layer-{min_layer + 1} attention) at SEP (t=n+1), "
            f"argmax == min(x). id+succ: resid_mid[{succ_layer}] (after layer-{succ_layer + 1} "
            f"attention) at y_1..y_(n-1) (t=n+2..2n), top-2 set == {{y_j, y_(j+1)}}, {pos}. "
            f"Chance refs: min_chance = best constant guess of min(x) on the same rows; "
            f"min_mode_frac = share of rows at the modal min argmax (1 = collapsed); "
            f"min_acc_centered = min argmax after centring each token's score over the rows.")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_probes(model, inputs: np.ndarray, device=None, *, apply_ln_f: bool = False,
               vocab: str = "full", dup_policy: str = "exclude", min_layer: int = 0,
               succ_layer: int = 1, batch_size: int = DEFAULT_BATCH,
               max_attn_mb: float = DEFAULT_MAX_ATTN_MB) -> dict:
    """Probe one length. inputs: (N, n) unsorted lists in [1, 100], all rows the same length.

    model must provide forward(idx, return_internals=True) -> (logits, internals) with
    internals["resid_mid"][l] of shape (B, T, d), plus model.unembed.weight (vocab, d). model.ln_f
    is needed for the *_lnf_* keys (and for apply_ln_f=True), and model.tok_emb.weight for
    enc_copy_acc; each key is skipped if its module is missing. The model should already be on
    `device` (None = the device of its parameters). Returns the per-length dict of the JSON schema.
    A model loaded by load_run_model from an eval.id_only run refuses n > its training range
    (IdOnlyError).
    """
    if vocab not in VOCABS:
        raise ValueError(f"vocab must be one of {VOCABS}, got {vocab!r}")
    if dup_policy not in DUP_POLICIES:
        raise ValueError(f"dup_policy must be one of {DUP_POLICIES}, got {dup_policy!r}")
    xs = np.asarray(inputs, dtype=np.int64)
    if xs.ndim != 2 or xs.shape[0] == 0 or xs.shape[1] == 0:
        raise ValueError(f"inputs must be a non-empty (N, n) array, got shape {xs.shape}")
    if xs.min() < VALUE_MIN or xs.max() > VALUE_MAX:
        raise ValueError("input values must lie in [1, 100]")
    N, n = xs.shape
    check_id_only(n, getattr(model, "id_only_max_len", None))
    ln_f = getattr(model, "ln_f", None)
    if apply_ln_f and ln_f is None:
        raise ValueError("apply_ln_f=True but the model has no ln_f")
    dev = _model_device(model, device)
    W = model.unembed.weight                                   # (q, d) decoder basis (rows)
    tok_emb = getattr(model, "tok_emb", None)
    E = tok_emb.weight if tok_emb is not None else None        # (q, d) encoder basis
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    lns = ["raw"] + (["lnf"] if ln_f is not None else [])
    prim = ("lnf" if apply_ln_f else "raw", vocab)
    c = {}                                                     # counters (python ints / floats)

    def add(key, value):
        c[key] = c.get(key, 0) + value

    hists, sep_scores = {}, {}             # per layer: argmax histogram, SEP score rows (CPU)
    rows = chunk_rows(n, model, batch_size, max_attn_mb)
    s_idx, out_idx = sep_index(n), output_positions(n)
    try:
        for start in range(0, N, rows):
            xb = xs[start:start + rows]
            B = len(xb)
            idx = torch.as_tensor(probe_inputs(xb), device=dev)
            logits, internals = _forward(model, idx)
            mids = internals["resid_mid"]
            if len(mids) <= max(min_layer, succ_layer):
                raise ValueError(f"model has {len(mids)} layers; probes need layers "
                                 f"{min_layer} and {succ_layer}")
            h_sep = mids[min_layer][:, s_idx, :]                # (B, d)
            h_out = mids[succ_layer][:, out_idx, :]             # (B, n-1, d)

            mins = torch.as_tensor(xb.min(axis=1), device=dev)            # (B,)
            tg = {k: torch.as_tensor(v, device=dev) for k, v in succ_targets(xb).items()}
            change, has_strict = tg["change"], tg["has_strict"]
            add("n_change", int(change.sum()))
            add("n_strict", int(has_strict.sum()))
            add("n_positions", B * (n - 1))

            for ln in lns:
                hs = ln_f(h_sep) if ln == "lnf" else h_sep
                ho = ln_f(h_out) if ln == "lnf" else h_out
                s_full, o_full = hs @ W.T, ho @ W.T            # (B, q), (B, n-1, q)
                for voc in VOCABS:
                    a, b = _vocab_slice(voc)
                    s_sep, s_out = s_full[:, a:b], o_full[..., a:b]
                    # min finding
                    pred = s_sep.argmax(dim=-1) + a
                    add(f"min_{ln}_{voc}", int((pred == mins).sum()))
                    # Identity+Successor: top-2 set
                    if n >= 2:
                        top2 = s_out.topk(2, dim=-1).indices + a          # (B, n-1, 2)
                        t0, t1 = top2[..., 0], top2[..., 1]
                        cur = tg["cur"]
                        ok_next = ((t0 == cur) & (t1 == tg["nxt"])) | ((t1 == cur) & (t0 == tg["nxt"]))
                        ok_strict = ((t0 == cur) & (t1 == tg["strict"])) | \
                                    ((t1 == cur) & (t0 == tg["strict"]))
                        add(f"chg_{ln}_{voc}", int((ok_next & change).sum()))
                        add(f"str_{ln}_{voc}", int((ok_strict & has_strict).sum()))
                    if (ln, voc) == prim:
                        true_score = s_sep.gather(1, (mins - a)[:, None])
                        add("min_rank", int((s_sep > true_score).sum()))
                        if n >= 2:
                            in2 = lambda t: (t0 == t) | (t1 == t)   # noqa: E731
                            add("id_in2", int((in2(cur) & change).sum()))
                            add("succ_in2", int((in2(tg["nxt"]) & change).sum()))

            a, b = _vocab_slice(vocab)
            # Min readout at SEP after EVERY layer's attention (primary ln_f / vocab setting)
            for l, mid in enumerate(mids):
                h_l = mid[:, s_idx, :]
                h_l = ln_f(h_l) if apply_ln_f else h_l
                s_l = (h_l @ W.T)[:, a:b]
                pred = s_l.argmax(dim=-1) + a
                add(f"min_after_attn{l + 1}", int((pred == mins).sum()))
                # chance references: modal argmax (global over all chunks) and centred scores
                hist = torch.bincount(pred.cpu(), minlength=W.shape[0]).numpy()
                hists[l] = hists[l] + hist if l in hists else hist
                sep_scores.setdefault(l, []).append(s_l.detach().float().cpu())
            # Min readout on the layer-1 attention write alone (raw, primary vocab)
            if "resid_pre" in internals:
                h_att = h_sep - internals["resid_pre"][min_layer][:, s_idx, :]
                pred = (h_att @ W.T)[:, a:b].argmax(dim=-1) + a
                add("min_attn", int((pred == mins).sum()))
            # Encoder-basis copy at input positions 1..n (raw residual, primary vocab)
            if E is not None:
                h_in = mids[min_layer][:, 1:n + 1, :]
                pred_in = (h_in @ E.T)[..., a:b].argmax(dim=-1) + a
                add("enc_copy", int((pred_in == torch.as_tensor(xb, device=dev)).sum()))
            # The model's own teacher-forced predictions at the n prediction positions n+1..2n
            pred_tf = logits[:, n + 1:2 * n + 1, a:b].argmax(dim=-1) + a          # (B, n)
            y = torch.as_tensor(sort_targets(xb), device=dev)
            add("tf_sep", int((pred_tf[:, 0] == y[:, 0]).sum()))
            add("tf_next", int((pred_tf == y).sum()))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    key = "chg" if dup_policy == "exclude" else "str"
    den = {"chg": c["n_change"], "str": c["n_strict"]}
    res = {"n": int(N)}
    res["min_acc"] = _ratio(c[f"min_{prim[0]}_{prim[1]}"], N)
    res["id_succ_acc"] = _ratio(c.get(f"{key}_{prim[0]}_{prim[1]}", 0), den[key])
    for ln in lns:
        for voc in VOCABS:
            res[f"min_acc_{ln}_{voc}"] = _ratio(c[f"min_{ln}_{voc}"], N)
    for ln in lns:
        for voc in VOCABS:
            res[f"id_succ_acc_{ln}_{voc}"] = _ratio(c.get(f"{key}_{ln}_{voc}", 0), den[key])
    chg = c.get(f"chg_{prim[0]}_{prim[1]}", 0)
    strict = c.get(f"str_{prim[0]}_{prim[1]}", 0)
    res["id_succ_acc_change"] = _ratio(chg, c["n_change"])
    res["id_succ_acc_strict"] = _ratio(strict, c["n_strict"])
    # Change positions are a subset of strict positions, where both criteria coincide.
    res["id_succ_acc_dup"] = _ratio(strict - chg, c["n_strict"] - c["n_change"])
    res["id_succ_n_change"] = int(c["n_change"])
    res["id_succ_n_strict"] = int(c["n_strict"])
    res["id_succ_n_positions"] = int(c["n_positions"])
    res["min_rank_mean"] = _ratio(c["min_rank"], N)
    if "min_attn" in c:
        res["min_acc_attn_only"] = _ratio(c["min_attn"], N)
    for k in sorted(k for k in c if k.startswith("min_after_attn")):
        res["min_acc_" + k[len("min_"):]] = _ratio(c[k], N)
    # Chance references (module docstring): best constant guess, modal argmax, centred readout
    true_mins = xs.min(axis=1)
    a, _b = _vocab_slice(vocab)
    res["min_chance"] = _ratio(int(np.bincount(true_mins).max()), N)
    res["min_mode_frac"], res["min_mode_token"] = _modal(hists[min_layer], N)
    res["min_acc_centered"] = _centred_acc(torch.cat(sep_scores[min_layer]), true_mins, a)
    for l in sorted(hists):
        res[f"min_mode_frac_after_attn{l + 1}"] = _modal(hists[l], N)[0]
    for l in sorted(sep_scores):
        res[f"min_acc_centered_after_attn{l + 1}"] = _centred_acc(torch.cat(sep_scores[l]),
                                                                  true_mins, a)
    res["id_in_top2"] = _ratio(c.get("id_in2", 0), c["n_change"])
    res["succ_in_top2"] = _ratio(c.get("succ_in2", 0), c["n_change"])
    if E is not None:
        res["enc_copy_acc"] = _ratio(c["enc_copy"], N * n)
    res["tf_sep_acc"] = _ratio(c["tf_sep"], N)
    res["tf_next_acc"] = _ratio(c["tf_next"], N * n)
    return res


# ---------------------------------------------------------------------------
# Runs on disk
# ---------------------------------------------------------------------------
def _load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older torch / non-tensor objects in run_config
        return torch.load(path, map_location="cpu", weights_only=False)


def run_id_of(run_dir: str, ckpt: dict | None = None) -> str:
    """run_id from the checkpoint's run_config, else config.yaml, else the directory name."""
    rc = (ckpt or {}).get("run_config") or {}
    if rc.get("run_id"):
        return str(rc["run_id"])
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("run_id"):
            return str(cfg["run_id"])
    return os.path.basename(os.path.normpath(run_dir))


def load_run_checkpoint(run_dir: str, phase: int) -> dict:
    """The raw checkpoint dict of <run_dir>/model_phase{phase}.pt (on CPU)."""
    return _load_checkpoint(os.path.join(run_dir, f"model_phase{int(phase)}.pt"))


def run_configs(run_dir: str, ckpt: dict | None = None) -> list:
    """The run's resolved config as stored in the checkpoint (run_config) and in
    <run_dir>/config.yaml; missing or empty sources are left out."""
    out = []
    rc = (ckpt or {}).get("run_config")
    if isinstance(rc, dict) and rc:
        out.append(rc)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and cfg:
            out.append(cfg)
    return out


def id_only_max_len(run_dir: str, ckpt: dict | None = None):
    """Tuning guard. If the run's config has eval.id_only: true, return its max training length
    (the max upper bound over data.groups, as in src.evaluate); otherwise None. Both the
    checkpoint's run_config and config.yaml are checked; if either says id_only, the guard
    applies (the smaller limit wins). A config without an eval section is not id_only."""
    from src.evaluate import max_train_length
    limits = [max_train_length(cfg) for cfg in run_configs(run_dir, ckpt)
              if bool((cfg.get("eval") or {}).get("id_only", False))]
    return min(limits) if limits else None


def load_run_model(run_dir: str, phase: int, device="auto", min_len: int = 0, max_len=None,
                   ckpt: dict | None = None):
    """Load <run_dir>/model_phase{phase}.pt (or the already loaded `ckpt`) into a fresh
    SortTransformer on `device`, in eval mode. The sinusoidal table is a non-persistent buffer, so
    max_len is raised to at least max(min_len, max_len) without touching the learned weights.
    model.id_only_max_len is set to id_only_max_len(run_dir, ckpt) (None unless the run is a
    tuning run), which run_probes / attention.successor_attention_score enforce.
    Returns (model, ckpt, device)."""
    from src.model import ModelConfig, build_model
    ckpt = ckpt if ckpt is not None else load_run_checkpoint(run_dir, phase)
    cfg = dict(ckpt["model_config"])
    cfg["max_len"] = max(int(cfg.get("max_len", 0)), int(min_len), int(max_len or 0))
    model = build_model(ModelConfig(**cfg), seed=0)        # weights are overwritten just below
    model.load_state_dict(ckpt["state_dict"])
    model.id_only_max_len = id_only_max_len(run_dir, ckpt)
    dev = resolve_device(device)
    model.to(dev).eval()
    return model, ckpt, dev


def probe_run(run_dir: str, phase: int, lengths, n: int, device="auto", max_len=None, *,
              batch_size: int = DEFAULT_BATCH, max_attn_mb: float = DEFAULT_MAX_ATTN_MB,
              out: str | None = None, **options) -> dict:
    """Probe <run_dir>/model_phase{phase}.pt on data.uniform_set(L, n, data.TEST_SEED) for each L.
    Writes the JSON (default <run_dir>/probes_phase{phase}.json) and returns it. `options` go to
    run_probes (apply_ln_f, vocab, dup_policy, min_layer, succ_layer).
    Tuning runs (eval.id_only: true): lengths above the max training length are dropped with a
    printed note; IdOnlyError if none are left."""
    from src.data import TEST_SEED, uniform_set
    lengths = [int(L) for L in lengths]
    if not lengths:
        raise ValueError("no lengths given")
    ckpt = load_run_checkpoint(run_dir, phase)
    limit = id_only_max_len(run_dir, ckpt)
    if limit is not None:
        dropped = [L for L in lengths if L > limit]
        lengths = [L for L in lengths if L <= limit]
        if dropped:
            print(f"[probes] {run_id_of(run_dir, ckpt)}: eval.id_only is true (tuning run); "
                  f"dropping length(s) {dropped} above the max training length {limit}",
                  flush=True)
        if not lengths:
            raise IdOnlyError(f"{run_dir}: eval.id_only is true and every requested length "
                              f"{dropped} is above the max training length {limit}")
    model, ckpt, dev = load_run_model(run_dir, phase, device, min_len=2 * max(lengths) + 2,
                                      max_len=max_len, ckpt=ckpt)
    by_length = {}
    for L in lengths:
        by_length[str(L)] = run_probes(model, uniform_set(L, int(n), TEST_SEED), dev,
                                       batch_size=batch_size, max_attn_mb=max_attn_mb, **options)
    keep = ("apply_ln_f", "vocab", "dup_policy", "min_layer", "succ_layer")
    result = {"run_id": run_id_of(run_dir, ckpt), "phase": int(phase),
              "readout": describe_readout(**{k: v for k, v in options.items() if k in keep}),
              "by_length": by_length}
    out = out or os.path.join(run_dir, f"probes_phase{int(phase)}.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    os.replace(tmp, out)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="A&G Section 4 mechanism probes for one run.")
    ap.add_argument("--run", required=True, help="run directory, e.g. results/<run_id>")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    ap.add_argument("--n", type=int, default=500, help="test lists per length")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=None, help="minimum sinusoidal table length")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-attn-mb", type=float, default=DEFAULT_MAX_ATTN_MB)
    ap.add_argument("--apply-ln-f", action="store_true", help="apply ln_f before projecting")
    ap.add_argument("--vocab", choices=VOCABS, default="full")
    ap.add_argument("--dup-policy", choices=DUP_POLICIES, default="exclude")
    ap.add_argument("--out", default=None, help="default <run>/probes_phase{p}.json")
    args = ap.parse_args(argv)
    try:
        res = probe_run(args.run, args.phase, args.lengths, args.n, device=args.device,
                        max_len=args.max_len, batch_size=args.batch_size,
                        max_attn_mb=args.max_attn_mb, out=args.out, apply_ln_f=args.apply_ln_f,
                        vocab=args.vocab, dup_policy=args.dup_policy)
    except IdOnlyError as e:
        ap.error(str(e))
    fmt = lambda v: "  n/a" if v is None else f"{v:5.3f}"   # noqa: E731
    print(f"[probes] {res['run_id']} phase {res['phase']}: {res['readout']}")
    print("   L  min_acc  chance  mode  centred  min@L2  id_succ_acc  tf_next_acc")
    for L, r in res["by_length"].items():
        print(f"{int(L):4d}    {fmt(r['min_acc'])}   {fmt(r['min_chance'])}  "
              f"{fmt(r['min_mode_frac'])}  {fmt(r['min_acc_centered'])}    "
              f"{fmt(r.get('min_acc_after_attn2'))}        {fmt(r['id_succ_acc'])}        "
              f"{fmt(r['tf_next_acc'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
