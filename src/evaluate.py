"""Greedy decoding and test-set evaluation. See docs/CODE_SPEC.md sections 1, 4 and 5.

CLI (re-evaluates a saved checkpoint):
    python -m src.evaluate --run results/<run_id> --phase {1,2} [--out FILE]
                           [--device DEV] [--decode-vocab values|full]

Notes and resolved ambiguities:
  * Only the token constants are imported at module level. Everything else from src.data
    (build_eval_sets, TEST_SEED, DEFAULT_GROUPS) and everything from src.model is imported lazily
    inside main(), so this module and its tests do not depend on those files.
  * greedy_decode builds the prompt [BOS, x, SEP] itself (identical to data.encode_prompt). It
    does not move the model: the caller puts the model on `device`. If device is None, the device
    of the model's first parameter is used.
  * Default --out is `<run>/eval_phase{p}_cli.json`, NOT `eval_phase{p}.json`. train.py writes
    the latter with extra fields (steps, tokens, wall time) that this CLI cannot reconstruct, so
    the default never overwrites it. Pass --out explicitly to write anywhere else.
  * --device and --decode-vocab are optional extensions to the spec's CLI. They default to the
    run config's train.device ("auto" -> cuda if available) and eval.decode_vocab.
  * id_only: the max training length is the max upper bound over config data.groups. Test AND rep
    lengths above it are dropped, and every built set is filtered by length again afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from src.data import BOS, SEP, VALUE_MAX, VALUE_MIN, VOCAB_SIZE
from src.metrics import compute_metrics

DECODE_VOCABS = ("values", "full")


def build_prompt(inputs: np.ndarray) -> np.ndarray:
    """(N, n) -> (N, n+2) int64 = [BOS, x, SEP] (same layout as data.encode_prompt)."""
    x = np.asarray(inputs, dtype=np.int64)
    N = x.shape[0]
    bos = np.full((N, 1), BOS, dtype=np.int64)
    sep = np.full((N, 1), SEP, dtype=np.int64)
    return np.concatenate([bos, x, sep], axis=1)


def _resolve_model_device(model, device):
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def greedy_decode(model, inputs: np.ndarray, device, batch_size: int = 1000,
                  decode_vocab: str = "values") -> np.ndarray:
    """inputs (N, n), all the same length. Prompt [BOS, x, SEP]; generate exactly n tokens greedily.

    decode_vocab="values": argmax restricted to ids VALUE_MIN..VALUE_MAX (1..100).
    decode_vocab="full":   argmax over the whole vocabulary (may emit PAD/SEP/BOS).
    Each step is a full forward pass over the current sequence (no KV cache). The prediction for
    the next token is read from the LAST position's logits. Ties go to the lowest id (torch.argmax).
    Puts the model in eval() and restores its previous training mode. Returns (N, n) int64.
    """
    if decode_vocab not in DECODE_VOCABS:
        raise ValueError(f"decode_vocab must be one of {DECODE_VOCABS}, got {decode_vocab!r}")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    x = np.asarray(inputs)
    if x.ndim != 2:
        raise ValueError(f"inputs must be a 2-D (N, n) array, got shape {x.shape}")
    x = x.astype(np.int64, copy=False)
    N, n = x.shape
    out = np.empty((N, n), dtype=np.int64)
    if N == 0 or n == 0:
        return out

    dev = _resolve_model_device(model, device)
    prompt = build_prompt(x)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, N, batch_size):
            seq = torch.from_numpy(prompt[start:start + batch_size]).to(dev)
            for _ in range(n):
                logits = model(seq)
                if logits.ndim != 3 or logits.shape[:2] != seq.shape or logits.shape[-1] != VOCAB_SIZE:
                    raise ValueError(f"model returned logits of shape {tuple(logits.shape)}; "
                                     f"expected {(*seq.shape, VOCAB_SIZE)}")
                last = logits[:, -1, :]
                if decode_vocab == "values":
                    nxt = last[:, VALUE_MIN:VALUE_MAX + 1].argmax(dim=-1) + VALUE_MIN
                else:
                    nxt = last.argmax(dim=-1)
                seq = torch.cat([seq, nxt.to(seq.dtype).unsqueeze(1)], dim=1)
            out[start:start + seq.shape[0]] = seq[:, n + 2:].cpu().numpy()
    finally:
        model.train(was_training)
        # Decoding creates many distinct tensor shapes, which fragments the CUDA caching allocator
        # (several GB reserved after a full test evaluation); on Windows the reservation also
        # counts against system commit memory. Releasing the cache here costs little and never
        # changes results.
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def evaluate_sets(model, sets: dict, device, batch_size: int = 1000, decode_vocab: str = "values",
                  n_samples: int = 20) -> dict:
    """sets: name -> {"kind", "length", "r", "inputs" (N, L)} (as from data.build_eval_sets).
    Returns name -> metric dict (metrics.compute_metrics) with kind/length/r filled in."""
    results = {}
    for name, s in sets.items():
        inputs = np.asarray(s["inputs"])
        length = int(s["length"])
        if inputs.ndim != 2 or inputs.shape[1] != length:
            raise ValueError(f"eval set {name!r}: inputs shape {inputs.shape} does not match "
                             f"length {length}")
        pred = greedy_decode(model, inputs, device, batch_size=batch_size, decode_vocab=decode_vocab)
        m = compute_metrics(pred, inputs, n_samples=n_samples)
        m["kind"] = str(s["kind"])
        m["length"] = length
        m["r"] = None if s.get("r") is None else int(s["r"])
        results[name] = m
    return results


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def max_train_length(cfg: dict) -> int:
    """Largest training length: the max upper bound over config data.groups."""
    groups = (cfg.get("data") or {}).get("groups")
    if not groups:
        from src.data import DEFAULT_GROUPS  # lazy: data.py is extended by another task
        groups = DEFAULT_GROUPS
    return int(max(int(hi) for (_lo, hi) in groups.values()))


def eval_lengths_from_config(cfg: dict) -> dict:
    """The test/rep set specification from a resolved run config (with id_only applied).

    Returns {"test_lengths", "test_per_length", "rep_lengths", "rep_rs", "rep_per_set",
             "id_only", "max_len"}."""
    ev = cfg["eval"]
    test_lengths = [int(L) for L in ev["test_lengths"]]
    rep_lengths = [int(L) for L in (ev.get("rep_lengths") or [])]
    rep_rs = [int(r) for r in (ev.get("rep_rs") or [])]
    rep_per_set = int(ev.get("rep_per_set") or 0)
    id_only = bool(ev.get("id_only", False))
    max_len = max_train_length(cfg)
    if id_only:
        test_lengths = [L for L in test_lengths if L <= max_len]
        rep_lengths = [L for L in rep_lengths if L <= max_len]
    if rep_per_set <= 0 or not rep_rs:
        rep_lengths, rep_rs, rep_per_set = [], [], 0
    return {"test_lengths": test_lengths, "test_per_length": int(ev["test_per_length"]),
            "rep_lengths": rep_lengths, "rep_rs": rep_rs, "rep_per_set": rep_per_set,
            "id_only": id_only, "max_len": max_len}


def _resolve_device(name) -> torch.device:
    if name is None or str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - own checkpoints; fall back if they hold non-allowlisted types
        return torch.load(path, map_location=device, weights_only=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-evaluate a saved run checkpoint on the test sets.")
    ap.add_argument("--run", required=True, help="run directory, e.g. results/<run_id>")
    ap.add_argument("--phase", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <run>/eval_phase{p}_cli.json)")
    ap.add_argument("--device", default=None, help="default: config train.device ('auto')")
    ap.add_argument("--decode-vocab", choices=DECODE_VOCABS, default=None,
                    help="default: config eval.decode_vocab")
    args = ap.parse_args(argv)

    import yaml

    from src.data import TEST_SEED, build_eval_sets
    from src.model import ModelConfig, build_model

    run_dir = args.run
    with open(os.path.join(run_dir, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_id = cfg.get("run_id") or os.path.basename(os.path.normpath(run_dir))
    ev = cfg["eval"]

    spec = eval_lengths_from_config(cfg)
    sets = build_eval_sets(spec["test_lengths"], spec["test_per_length"], TEST_SEED,
                           rep_lengths=spec["rep_lengths"], rep_rs=spec["rep_rs"],
                           rep_n=spec["rep_per_set"])
    if spec["id_only"]:  # belt and braces: no length above the training range, ever
        sets = {k: v for k, v in sets.items() if int(v["length"]) <= spec["max_len"]}

    device = _resolve_device(args.device or (cfg.get("train") or {}).get("device", "auto"))
    ckpt = _load_checkpoint(os.path.join(run_dir, f"model_phase{args.phase}.pt"), device)
    model = build_model(ModelConfig(**ckpt["model_config"]), seed=int(cfg.get("seed", 0)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    decode_vocab = args.decode_vocab or ev.get("decode_vocab", "values")
    test_sets = evaluate_sets(model, sets, device, batch_size=int(ev.get("batch_size", 250)),
                              decode_vocab=decode_vocab, n_samples=int(ev.get("n_samples", 20)))
    result = {"run_id": run_id, "phase": int(args.phase), "test_sets": test_sets}

    out = args.out or os.path.join(run_dir, f"eval_phase{args.phase}_cli.json")
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, out)

    print(f"[evaluate] {run_id} phase {args.phase} ({decode_vocab}, {device}) -> {out}")
    for name, m in test_sets.items():
        print(f"  {name:>14s}  exact={m['exact_match']:.3f}  tok={m['token_acc']:.3f}  "
              f"ed={m['edit_distance']:.2f}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
