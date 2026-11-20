"""Training loop for the sorting / curriculum experiments (docs/CODE_SPEC.md section 5).

CLI:
    python -m src.train --config configs/base.yaml [--config configs/tune.yaml ...]
                        [--schedule X] [--pe sinusoidal|none] [--seed N] [--tag T]
                        [--set dotted.key=value ...] [--out-dir DIR] [--overwrite]

Config resolution (in this order, later wins):
  1. every --config file, deep-merged left to right (dicts merge key by key; lists and scalars
     are replaced wholesale, so `curve_ood_lengths: []` really empties the list). The first file
     must be a complete config (configs/base.yaml); tune.yaml / smoke.yaml are small override
     files layered on top of it. Without --config, configs/base.yaml is used.
  2. every --set dotted.key=value, in order. The value is parsed with yaml.safe_load. The key
     must already exist (typo guard).
  3. the explicit flags --schedule, --pe, --seed, --tag, --out-dir.
  The result is checked against a fixed schema (unknown or missing keys are errors) and the types
  are coerced (so `--set train.lr=5e-4`, which PyYAML reads as a string, still becomes a float).
  A `run_id` key in a config file (e.g. a saved config.yaml) is ignored and recomputed.

Run id: f"{pe}_{schedule}_s{seed}" (+ f"_{tag}" if tag). Output dir: <out_dir>/<run_id>/.
  * <run>/done.json exists      -> refused unless --overwrite (then the dir is cleared).
  * <run> exists, no done.json  -> a crashed run: the dir is cleared and the run restarts.

Experimental controls (why the code is shaped this way):
  * The model is built FIRST, by make_model(cfg) = build_model(ModelConfig(**cfg.model), seed),
    so the initial weights depend only on (model config, seed): identical across schedules.
  * Phase 1 data: length_sequence(schedule, steps, B, seed, phase=1, groups, mix) with
    ExampleStream(seed, 1). A schedule and its shuffled twin get the same examples.
  * Phase 2 data: length_sequence("R", phase2.steps, B, seed, phase=2, groups, mix) with
    ExampleStream(seed, 2): identical for every schedule of a seed (make_batches(cfg, 2)).
  * Every schedule makes exactly train.steps updates with the same batch size.

Resolved details (choices the spec leaves open):
  * LR: lr_at(s) = lr * s / warmup_steps for update s < warmup_steps (0-based), else lr. So the
    very first update has lr 0 (same convention as HF get_constant_schedule_with_warmup). The
    "lr" in a loss record is the lr of the last update in that logging window.
  * Phase 2 continues with the same AdamW (moments carry over) at constant lr, no new warmup.
    If phase2.reset_optimizer is true, a fresh AdamW (zero moments, same hyperparameters) is
    created at the start of Phase 2, also at constant lr with no warmup. phase2.steps: 0 skips
    Phase 2 entirely (no eval_phase2.json / model_phase2.pt).
  * AdamW weight decay (default 0.0) applies to every parameter. Gradients are clipped to
    train.grad_clip (global L2 norm) when grad_clip > 0.
  * steps, tokens_nonpad, tokens_padded, examples and wall_time_s count from 0 WITHIN each phase
    (in loss records and in eval_phase{p}.json). done.json has both phases and the totals.
    Tokens are counted on the CPU from the numpy batch, so they never force a GPU sync; the loss
    is accumulated on the device and read with .item() only at log points.
  * Eval records: "val" is greedy exact match on uniform_set(L, val_per_length, VAL_SEED) for
    every training length L; "val_group" is the mean of "val" over each group's lengths;
    "val_token_acc" is the mean over training lengths of per-length token accuracy;
    "curve_ood" is exact match on uniform_set(L, val_per_length, VAL_SEED) for
    eval.curve_ood_lengths, and is EMPTY when eval.id_only is true. Phase 2 also gets a step-0
    eval record (identical weights to the end of Phase 1, so it repeats those numbers).
  * id_only: final test/rep sets come from evaluate.eval_lengths_from_config and are then
    hard-filtered again to lengths <= max train length (max upper bound of data.groups).
  * A non-finite mean loss at a log point aborts the run (no done.json, so a sweep reports it).
  * eval_phase{p}.json wall_time_s = the phase's wall time, including its learning-curve evals
    and its final test eval.
  * Loss (train.loss_norm): "example" (default) = each example's mean
    CE over its n masked tokens, then the mean over the examples of the batch, so every example
    weighs exactly 1/B whatever else is in its batch (a schedule and its shuffled twin then weight
    the same examples identically). "token" = the mean over all masked tokens of the batch (long
    examples weigh more, and how much more depends on the batch mix).
  * done.json "config_hash" = sha256 hex of json.dumps(config, sort_keys=True,
    separators=(",", ":")), where the config is the resolved one passed through the schema
    normalization again (so 5e-4 / "5e-4" / 0.0005 and 0 / 0.0 hash alike) minus out_dir and
    run_id. "experiment_hash" also leaves out seed, tag, data.schedule and model.pe: runs that
    should be compared with each other share it. scripts/sweep.py compares config_hash with the
    config it would run now; src.analysis.collect warns about mixed experiment hashes.
  * done.json "overlap": {"phase{p}": {"L": {"val": k, "test": k[, "curve_ood": k]}}}
    for every length L <= OVERLAP_MAX_LEN (8) trained on in phase p; k = number of rows of that
    eval set (val = uniform_set(L, val_per_length, VAL_SEED), test = the uniform TEST_SEED set,
    curve_ood = a learning-curve OOD set of length L, keys present only when that set exists)
    that also appear among the training rows consumed at length L in that phase, regenerated
    with data.pool_examples(seed, phase, L, 0, count) and compared with data.count_overlap
    (about 0.2 s per comparison at 640k rows). Longer lists essentially never collide.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data import (PAD, TEST_SEED, VAL_SEED, ExampleStream, build_eval_sets, count_overlap,
                      iterate_batches, length_sequence, pool_examples, sort_targets, uniform_set)
from src.evaluate import (eval_lengths_from_config, evaluate_sets, greedy_decode,
                          max_train_length)
from src.metrics import exact_match
from src.model import ModelConfig, build_model, count_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "base.yaml")
IGNORE_INDEX = -100
TAG_RE = re.compile(r"^[A-Za-z0-9._+-]*$")
LOSS_NORMS = ("example", "token")
# Keys that cannot change a run's results (where it is written, logging cadence, eval chunking,
# how many raw samples are saved, whether a checkpoint is kept, which device string is used) are
# left out, so changing them does not mark finished runs STALE. Decoding is identical across eval
# batch sizes (fp32, checked on the RTX 3070 Ti).
HASH_EXCLUDE = ("out_dir", "run_id", "train.device", "train.log_every", "eval.batch_size",
                "eval.n_samples", "save_checkpoints")
EXPERIMENT_HASH_EXCLUDE = HASH_EXCLUDE + ("seed", "tag", "data.schedule", "model.pe")
OVERLAP_MAX_LEN = 8


class ConfigError(ValueError):
    pass


class RunExistsError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# Config: schema, merging, overrides
# ----------------------------------------------------------------------------------------------
INT, FLOAT, BOOL, STR, INTS, FLOATS, GROUPS = "int", "float", "bool", "str", "ints", "floats", "groups"
SCHEMA = {
    "out_dir": STR,
    "seed": INT,
    "tag": STR,
    "model": {"d_model": INT, "n_layers": INT, "n_heads": INT, "d_mlp": INT, "pe": STR,
              "max_len": INT},
    "data": {"schedule": STR, "groups": GROUPS, "mix": STR},
    "train": {"steps": INT, "batch_size": INT, "lr": FLOAT, "warmup_steps": INT,
              "weight_decay": FLOAT, "betas": FLOATS, "grad_clip": FLOAT, "device": STR,
              "log_every": INT, "loss_norm": STR},
    "phase2": {"steps": INT, "reset_optimizer": BOOL},
    "eval": {"every": INT, "val_per_length": INT, "curve_ood_lengths": INTS, "test_lengths": INTS,
             "test_per_length": INT, "rep_lengths": INTS, "rep_rs": INTS, "rep_per_set": INT,
             "batch_size": INT, "decode_vocab": STR, "n_samples": INT, "id_only": BOOL},
    "save_checkpoints": BOOL,
}


def _to_int(v, key):
    if isinstance(v, bool):
        raise ConfigError(f"{key}: expected an int, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            f = None
        if f is not None and f.is_integer():
            return int(f)
    raise ConfigError(f"{key}: expected an int, got {v!r}")


def _to_float(v, key):
    if isinstance(v, bool):
        raise ConfigError(f"{key}: expected a number, got {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):  # PyYAML 1.1 reads "5e-4" (no dot) as a string
        try:
            return float(v)
        except ValueError:
            pass
    raise ConfigError(f"{key}: expected a number, got {v!r}")


def _coerce(kind, v, key):
    if kind == INT:
        return _to_int(v, key)
    if kind == FLOAT:
        return _to_float(v, key)
    if kind == BOOL:
        if not isinstance(v, bool):
            raise ConfigError(f"{key}: expected true/false, got {v!r}")
        return v
    if kind == STR:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            raise ConfigError(f"{key}: expected a string, got {v!r}")
        return str(v)
    if kind in (INTS, FLOATS):
        if v is None:
            return []
        if not isinstance(v, (list, tuple)):
            raise ConfigError(f"{key}: expected a list, got {v!r}")
        f = _to_int if kind == INTS else _to_float
        return [f(x, key) for x in v]
    if kind == GROUPS:
        if not isinstance(v, dict) or not v:
            raise ConfigError(f"{key}: expected a mapping name -> [lo, hi], got {v!r}")
        out = {}
        for name, rng in v.items():
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise ConfigError(f"{key}.{name}: expected [lo, hi], got {rng!r}")
            lo, hi = _to_int(rng[0], key), _to_int(rng[1], key)
            if not 1 <= lo <= hi:
                raise ConfigError(f"{key}.{name}: need 1 <= lo <= hi, got {rng!r}")
            out[str(name)] = [lo, hi]
        return out
    raise AssertionError(kind)


def _normalize(cfg, schema=SCHEMA, prefix=""):
    """Check keys against the schema (none missing, none unknown) and coerce types."""
    if not isinstance(cfg, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {cfg!r}")
    unknown = sorted(set(cfg) - set(schema))
    if unknown:
        raise ConfigError(f"unknown config key(s): {[prefix + k for k in unknown]}")
    missing = sorted(set(schema) - set(cfg))
    if missing:
        raise ConfigError(f"missing config key(s): {[prefix + k for k in missing]} "
                          f"(is configs/base.yaml the first --config?)")
    out = {}
    for k, kind in schema.items():
        key = prefix + k
        out[k] = (_normalize(cfg[k], kind, key + ".") if isinstance(kind, dict)
                  else _coerce(kind, cfg[k], key))
    return out


def _validate(cfg):
    t, ev = cfg["train"], cfg["eval"]
    if cfg["model"]["pe"] not in ("sinusoidal", "none"):
        raise ConfigError(f"model.pe must be 'sinusoidal' or 'none', got {cfg['model']['pe']!r}")
    if cfg["data"]["mix"] not in ("lengths", "groups"):
        raise ConfigError(f"data.mix must be 'lengths' or 'groups', got {cfg['data']['mix']!r}")
    if ev["decode_vocab"] not in ("values", "full"):
        raise ConfigError(f"eval.decode_vocab must be 'values' or 'full', got {ev['decode_vocab']!r}")
    if t["loss_norm"] not in LOSS_NORMS:
        raise ConfigError(f"train.loss_norm must be one of {LOSS_NORMS}, got {t['loss_norm']!r}")
    if not TAG_RE.match(cfg["tag"]):
        raise ConfigError(f"tag {cfg['tag']!r} may only contain letters, digits and . _ + -")
    for key, v, lo in [("train.steps", t["steps"], 1), ("train.batch_size", t["batch_size"], 1),
                       ("train.warmup_steps", t["warmup_steps"], 0),
                       ("train.log_every", t["log_every"], 1),
                       ("phase2.steps", cfg["phase2"]["steps"], 0), ("eval.every", ev["every"], 1),
                       ("eval.val_per_length", ev["val_per_length"], 1),
                       ("eval.test_per_length", ev["test_per_length"], 1),
                       ("eval.batch_size", ev["batch_size"], 1)]:
        if v < lo:
            raise ConfigError(f"{key} must be >= {lo}, got {v}")
    if len(t["betas"]) != 2:
        raise ConfigError(f"train.betas must have 2 entries, got {t['betas']}")
    if t["lr"] <= 0:
        raise ConfigError(f"train.lr must be > 0, got {t['lr']}")
    try:  # schedule name (and groups/mix) via data.length_sequence; cheap with steps=1, B=1
        length_sequence(cfg["data"]["schedule"], 1, 1, cfg["seed"], 1, **data_kwargs(cfg))
    except (ValueError, KeyError) as e:
        raise ConfigError(f"bad data config (schedule {cfg['data']['schedule']!r}): {e}") from e


def deep_merge(base: dict, override: dict) -> dict:
    """Dicts merge recursively; everything else (lists included) is replaced wholesale."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return d


def apply_set(cfg: dict, item: str) -> None:
    """In-place `dotted.key=value` override; the value is parsed with yaml.safe_load."""
    if "=" not in item:
        raise ConfigError(f"--set expects dotted.key=value, got {item!r}")
    key, raw = item.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for i, p in enumerate(parts):
        if not isinstance(node, dict) or p not in node:
            where = ".".join(parts[:i]) or "<top level>"
            have = sorted(node) if isinstance(node, dict) else []
            raise ConfigError(f"--set {key}: no key {p!r} under {where} (have {have})")
        if i == len(parts) - 1:
            node[p] = yaml.safe_load(raw)
        else:
            node = node[p]


def resolve_config(config_paths, sets=(), schedule=None, pe=None, seed=None, tag=None,
                   out_dir=None, base_dir=None) -> dict:
    """Merge config files, apply --set overrides and explicit flags, then normalize/validate.
    Relative config paths are resolved against base_dir (default: the current directory)."""
    cfg = {}
    for p in config_paths:
        if base_dir and not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        cfg = deep_merge(cfg, load_yaml(p))
    cfg.pop("run_id", None)  # a saved config.yaml may be reused as input; run_id is recomputed
    for item in sets:
        apply_set(cfg, item)
    if schedule is not None:
        cfg.setdefault("data", {})["schedule"] = schedule
    if pe is not None:
        cfg.setdefault("model", {})["pe"] = pe
    if seed is not None:
        cfg["seed"] = seed
    if tag is not None:
        cfg["tag"] = tag
    if out_dir is not None:
        cfg["out_dir"] = out_dir
    cfg = _normalize(cfg)
    _validate(cfg)
    return cfg


def run_id_of(cfg: dict) -> str:
    rid = f"{cfg['model']['pe']}_{cfg['data']['schedule']}_s{cfg['seed']}"
    return f"{rid}_{cfg['tag']}" if cfg["tag"] else rid


def run_dir_of(cfg: dict) -> str:
    return os.path.join(cfg["out_dir"], run_id_of(cfg))


def canonical_config(cfg: dict, exclude=HASH_EXCLUDE) -> dict:
    """The config as hashed: run_id dropped, passed through the schema normalization again (so
    YAML ints/floats/strings such as 5e-4, "5e-4", 0.0005 or 0 / 0.0 come out identical), then
    every key in `exclude` (dotted paths, e.g. "data.schedule") removed. Returns a new dict."""
    c = _normalize({k: v for k, v in cfg.items() if k != "run_id"})
    for key in exclude:
        *path, last = key.split(".")
        node = c
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(last, None)
    return c


def config_hash(cfg: dict, exclude=HASH_EXCLUDE) -> str:
    """sha256 hex of the canonical JSON of the config (default: everything but out_dir/run_id)."""
    blob = json.dumps(canonical_config(cfg, exclude), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def experiment_hash(cfg: dict) -> str:
    """config_hash without seed, tag, data.schedule and model.pe: equal for runs meant to be
    compared with each other (same experiment, different condition or seed)."""
    return config_hash(cfg, EXPERIMENT_HASH_EXCLUDE)


# ----------------------------------------------------------------------------------------------
# Model, data, optimizer, loss
# ----------------------------------------------------------------------------------------------
def model_config(cfg: dict) -> ModelConfig:
    return ModelConfig(**cfg["model"])


def make_model(cfg: dict):
    """Initial weights depend only on (model config, seed), never on the schedule."""
    return build_model(model_config(cfg), cfg["seed"])


def data_kwargs(cfg: dict) -> dict:
    groups = {k: (int(v[0]), int(v[1])) for k, v in cfg["data"]["groups"].items()}
    return {"groups": groups, "mix": cfg["data"]["mix"]}


def phase_steps(cfg: dict, phase: int) -> int:
    return cfg["train"]["steps"] if phase == 1 else cfg["phase2"]["steps"]


def make_length_sequence(cfg: dict, phase: int) -> np.ndarray:
    """Phase 1: the run's schedule. Phase 2: "R" for everyone (same for every schedule of a seed)."""
    schedule = cfg["data"]["schedule"] if phase == 1 else "R"
    return length_sequence(schedule, phase_steps(cfg, phase), cfg["train"]["batch_size"],
                           cfg["seed"], phase=phase, **data_kwargs(cfg))


def make_batches(cfg: dict, phase: int):
    """Iterator over the phase's training batches (dicts from data.iterate_batches)."""
    return iterate_batches(make_length_sequence(cfg, phase), cfg["train"]["batch_size"],
                           ExampleStream(cfg["seed"], phase))


def lr_at(step: int, lr: float, warmup_steps: int) -> float:
    """LR of (0-based) update `step`: linear from 0 to lr over warmup_steps, then constant."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * step / warmup_steps
    return lr


def make_optimizer(model, cfg: dict) -> torch.optim.AdamW:
    t = cfg["train"]
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]),
                             weight_decay=t["weight_decay"])


def masked_loss(logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
                norm: str = "example") -> torch.Tensor:
    """Cross-entropy over the loss_mask positions only. logits (B,T,V), target/mask (B,T).

    norm="example": per-example mean CE over that example's masked tokens, then the mean over the
        examples, so each example weighs exactly 1/B whatever else is in its batch. (A row with no
        masked token, which the data never produces, is left out of the mean.)
    norm="token":   mean CE over all masked tokens of the batch (rows weigh by token count).
    Unmasked positions are ignored via ignore_index (no boolean indexing, so no GPU sync)."""
    tgt = target.masked_fill(~loss_mask, IGNORE_INDEX)
    flat = logits.reshape(-1, logits.shape[-1]).float()
    if norm == "token":
        return F.cross_entropy(flat, tgt.reshape(-1), ignore_index=IGNORE_INDEX)
    if norm != "example":
        raise ValueError(f"norm must be one of {LOSS_NORMS}, got {norm!r}")
    ce = F.cross_entropy(flat, tgt.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none")
    ce = ce.reshape(target.shape)  # 0 exactly at ignored positions (and no gradient there)
    n_tok = loss_mask.sum(dim=1)
    per_example = ce.sum(dim=1) / n_tok.clamp(min=1)
    n_examples = (n_tok > 0).sum().clamp(min=1)
    return per_example.sum() / n_examples


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def train_lengths(cfg: dict) -> list:
    return sorted({L for lo, hi in cfg["data"]["groups"].values() for L in range(lo, hi + 1)})


class CurveEvaluator:
    """Learning-curve eval: val sets for every training length (+ curve_ood unless id_only)."""

    def __init__(self, cfg: dict, device):
        ev = cfg["eval"]
        self.device, self.batch_size, self.decode_vocab = device, ev["batch_size"], ev["decode_vocab"]
        self.groups = cfg["data"]["groups"]
        n = ev["val_per_length"]
        self.val = {L: uniform_set(L, n, VAL_SEED) for L in train_lengths(cfg)}
        max_len = max_train_length(cfg)
        ood = [] if ev["id_only"] else list(ev["curve_ood_lengths"])
        if ev["id_only"]:  # hard filter (already empty; kept so the invariant is local)
            ood = [L for L in ood if L <= max_len]
        self.ood = {L: uniform_set(L, n, VAL_SEED) for L in ood}

    def _score(self, model, X):
        pred = greedy_decode(model, X, self.device, batch_size=self.batch_size,
                             decode_vocab=self.decode_vocab)
        tgt = sort_targets(X)
        return float(exact_match(pred, tgt).mean()), float((pred == tgt).mean())

    def __call__(self, model) -> dict:
        val, tok = {}, []
        for L, X in self.val.items():
            em, ta = self._score(model, X)
            val[str(L)] = em
            tok.append(ta)
        val_group = {g: float(np.mean([val[str(L)] for L in range(lo, hi + 1)]))
                     for g, (lo, hi) in self.groups.items()}
        ood = {str(L): self._score(model, X)[0] for L, X in self.ood.items()}
        return {"val": val, "val_group": val_group, "val_token_acc": float(np.mean(tok)),
                "curve_ood": ood}


def final_eval_sets(cfg: dict) -> dict:
    """Test + rep sets (TEST_SEED / REP_SEED). With id_only, nothing above the max train length."""
    spec = eval_lengths_from_config(cfg)
    sets = build_eval_sets(spec["test_lengths"], spec["test_per_length"], TEST_SEED,
                           rep_lengths=spec["rep_lengths"], rep_rs=spec["rep_rs"],
                           rep_n=spec["rep_per_set"])
    if cfg["eval"]["id_only"]:
        max_len = max_train_length(cfg)
        sets = {k: v for k, v in sets.items() if int(v["length"]) <= max_len}
    return sets


def overlap_check(cfg: dict, length_counts: dict, curve: CurveEvaluator, test_sets: dict,
                  max_len: int = OVERLAP_MAX_LEN) -> dict:
    """Eval rows that also occur among the training rows.

    length_counts: {phase: {L: number of training examples of length L consumed in that phase}}.
    For every phase and every trained length L <= max_len, the consumed rows are regenerated with
    pool_examples(seed, phase, L, 0, count) (the stream hands them out in pool order) and
    count_overlap(eval rows, training rows) is taken for the val set, the uniform test set and a
    curve_ood set of that length (a key only for sets that exist).
    Returns {"phase{p}": {str(L): {"val": k, "test": k, ...}}}."""
    test_by_len = {int(s["length"]): s["inputs"] for s in test_sets.values()
                   if s["kind"] == "uniform"}
    out = {}
    for phase, counts in sorted(length_counts.items()):
        per = {}
        for L, count in sorted((int(L), int(c)) for L, c in counts.items()):
            if L > max_len or count <= 0:
                continue
            train_rows = pool_examples(cfg["seed"], phase, L, 0, count)
            evals = {"val": curve.val[L]}
            if L in test_by_len:
                evals["test"] = test_by_len[L]
            if L in curve.ood:
                evals["curve_ood"] = curve.ood[L]
            per[str(L)] = {name: int(count_overlap(X, train_rows)) for name, X in evals.items()}
        out[f"phase{phase}"] = per
    return out


def _fmt_overlap(overlap: dict, sizes: dict) -> str:
    """'phase1 val 0/400 test 0/1000 (L 5-8); phase2 ...' (sizes: set name -> rows per length)."""
    parts = []
    for ph, per in overlap.items():
        if not per:
            parts.append(f"{ph} -")
            continue
        hit, rows = {}, {}
        for d in per.values():
            for k, v in d.items():
                hit[k] = hit.get(k, 0) + v
                rows[k] = rows.get(k, 0) + sizes.get(k, 0)
        lens = [int(L) for L in per]
        parts.append(f"{ph} " + " ".join(f"{k} {hit[k]}/{rows[k]}" for k in hit)
                     + f" (L {min(lens)}-{max(lens)})")
    return "; ".join(parts)


# ----------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------
def write_json_atomic(path: str, obj, indent=None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def write_yaml_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=None)
    os.replace(tmp, path)


def save_checkpoint(path: str, model, run_config: dict) -> None:
    obj = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "model_config": asdict(model.cfg), "run_config": run_config}
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prepare_run_dir(run_dir: str, overwrite: bool) -> None:
    done = os.path.join(run_dir, "done.json")
    if os.path.exists(run_dir) and not os.path.isdir(run_dir):
        raise RunExistsError(f"{run_dir} exists and is not a directory; not touching it")
    if os.path.exists(done):
        if not overwrite:
            raise RunExistsError(f"{run_dir} is a finished run (done.json exists); "
                                 f"pass --overwrite to re-run it")
        print(f"[train] --overwrite: clearing finished run {run_dir}", flush=True)
        os.remove(done)  # first, so a half-deleted dir never looks finished
        shutil.rmtree(run_dir)
    elif os.path.exists(run_dir):
        print(f"[train] {run_dir} exists without done.json (crashed or interrupted run): "
              f"clearing it and starting over", flush=True)
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)


class JsonlLog:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _fmt_eval(rec: dict) -> str:
    g = " ".join(f"{k} {v:.3f}" for k, v in rec["val_group"].items())
    s = f"val {g} tok {rec['val_token_acc']:.3f}"
    if rec["curve_ood"]:
        s += "  ood " + " ".join(f"{k}:{v:.2f}" for k, v in rec["curve_ood"].items())
    return s


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def _to_device(a: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(a)
    if device.type == "cuda":  # pinned + non_blocking: the copy does not stall the host
        return t.pin_memory().to(device, non_blocking=True)
    return t


def train_phase(model, opt, cfg: dict, phase: int, device, log: JsonlLog, curve: CurveEvaluator,
                run_id: str) -> dict:
    t, ev = cfg["train"], cfg["eval"]
    steps = phase_steps(cfg, phase)
    base_lr, warmup = t["lr"], t["warmup_steps"]
    lr_fn = (lambda s: lr_at(s, base_lr, warmup)) if phase == 1 else (lambda s: base_lr)
    log_every, eval_every, clip = t["log_every"], ev["every"], t["grad_clip"]
    loss_norm = t["loss_norm"]
    t0 = time.perf_counter()

    tokens_nonpad = tokens_padded = examples = 0
    length_counts: dict = {}
    last_loss = float("nan")

    def do_eval(step):
        rec = {"type": "eval", "phase": phase, "step": step, **curve(model)}
        log.write(rec)
        loss_s = f"{last_loss:.4f}" if math.isfinite(last_loss) else "  -   "
        print(f"[{run_id}] p{phase} step {step:>6d}/{steps}  loss {loss_s}  "
              f"{_fmt_eval(rec)}  {time.perf_counter() - t0:.1f}s", flush=True)
        return rec

    last_eval = do_eval(0)
    model.train()
    loss_sum = torch.zeros((), device=device)
    n_since, lr, step = 0, 0.0, 0
    for i, batch in enumerate(make_batches(cfg, phase)):
        lr = lr_fn(i)
        for g in opt.param_groups:
            g["lr"] = lr
        inp = _to_device(batch["input"], device)
        tgt = _to_device(batch["target"], device)
        mask = _to_device(batch["loss_mask"], device)

        loss = masked_loss(model(inp), tgt, mask, norm=loss_norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        loss_sum += loss.detach()
        n_since += 1
        B, Tm = batch["input"].shape
        tokens_nonpad += int(np.count_nonzero(batch["input"] != PAD))
        tokens_padded += B * Tm
        examples += B
        for L, c in zip(*np.unique(batch["lengths"], return_counts=True)):
            length_counts[int(L)] = length_counts.get(int(L), 0) + int(c)

        step = i + 1
        if step % log_every == 0 or step == steps:
            last_loss = loss_sum.item() / n_since  # the only host sync in the loop
            if not math.isfinite(last_loss):
                raise RuntimeError(f"non-finite loss {last_loss} at phase {phase} step {step}")
            log.write({"type": "loss", "phase": phase, "step": step, "loss": last_loss, "lr": lr,
                       "tokens_nonpad": tokens_nonpad, "tokens_padded": tokens_padded,
                       "examples": examples})
            loss_sum.zero_()
            n_since = 0
        if step % eval_every == 0 or step == steps:
            last_eval = do_eval(step)
            model.train()
    if step != steps:
        raise RuntimeError(f"phase {phase}: consumed {step} batches, expected {steps}")
    return {"steps": steps, "tokens_nonpad": tokens_nonpad, "tokens_padded": tokens_padded,
            "examples": examples, "length_counts": dict(sorted(length_counts.items())),
            "last_eval": last_eval, "t0": t0}


def finish_phase(model, cfg: dict, phase: int, stats: dict, test_sets: dict, device, run_dir: str,
                 run_id: str) -> float:
    """Final test eval -> eval_phase{p}.json, then model_phase{p}.pt. Returns the phase wall time."""
    ev = cfg["eval"]
    res = evaluate_sets(model, test_sets, device, batch_size=ev["batch_size"],
                        decode_vocab=ev["decode_vocab"], n_samples=ev["n_samples"])
    wall = time.perf_counter() - stats["t0"]
    write_json_atomic(os.path.join(run_dir, f"eval_phase{phase}.json"),
                      {"run_id": run_id, "phase": phase, "steps": stats["steps"],
                       "tokens_nonpad": stats["tokens_nonpad"],
                       "tokens_padded": stats["tokens_padded"], "wall_time_s": wall,
                       "test_sets": res})
    if cfg["save_checkpoints"]:
        save_checkpoint(os.path.join(run_dir, f"model_phase{phase}.pt"), model,
                        {"run_id": run_id, **cfg})
    uni = " ".join(f"{m['length']}:{m['exact_match']:.2f}" for m in res.values()
                   if m["kind"] == "uniform")
    print(f"[{run_id}] phase {phase} done in {wall:.1f}s  test exact match {uni}", flush=True)
    return wall


def run(cfg: dict, overwrite: bool = False) -> str:
    """Train one fully resolved config (from resolve_config). Returns the run directory."""
    t_start = time.perf_counter()
    run_id = run_id_of(cfg)
    model = make_model(cfg)  # FIRST: init depends only on (model config, seed)
    n_params = count_params(model)
    run_dir = run_dir_of(cfg)
    prepare_run_dir(run_dir, overwrite)
    run_cfg = {"run_id": run_id, **cfg}
    write_yaml_atomic(os.path.join(run_dir, "config.yaml"), run_cfg)

    device = resolve_device(cfg["train"]["device"])
    model.to(device)
    opt = make_optimizer(model, cfg)
    curve = CurveEvaluator(cfg, device)
    test_sets = final_eval_sets(cfg)
    p2 = cfg["phase2"]["steps"]
    print(f"[{run_id}] {n_params:,} params on {device}; phase 1: {cfg['data']['schedule']} x "
          f"{cfg['train']['steps']} steps (batch {cfg['train']['batch_size']}, loss per "
          f"{cfg['train']['loss_norm']}); phase 2: "
          f"{p2} steps{' (fresh optimizer)' if p2 and cfg['phase2']['reset_optimizer'] else ''}; "
          f"{len(test_sets)} test sets -> {run_dir}", flush=True)
    if p2 and cfg["train"]["steps"] < cfg["train"]["warmup_steps"]:
        print(f"[{run_id}] warning: phase 1 ends before warmup does; phase 2 jumps to lr "
              f"{cfg['train']['lr']}", flush=True)

    log = JsonlLog(os.path.join(run_dir, "train_log.jsonl"))
    try:
        phases = {}
        for phase in (1, 2):
            if phase == 2:
                if p2 == 0:
                    break
                if cfg["phase2"]["reset_optimizer"]:
                    opt = make_optimizer(model, cfg)
            stats = train_phase(model, opt, cfg, phase, device, log, curve, run_id)
            stats["wall_time_s"] = finish_phase(model, cfg, phase, stats, test_sets, device,
                                                run_dir, run_id)
            phases[phase] = stats
    finally:
        log.close()

    t_ov = time.perf_counter()
    try:
        overlap = overlap_check(cfg, {p: s["length_counts"] for p, s in phases.items()}, curve,
                                test_sets)
        ev = cfg["eval"]
        sizes = {"val": ev["val_per_length"], "test": ev["test_per_length"],
                 "curve_ood": ev["val_per_length"]}
        print(f"[{run_id}] overlap check, eval rows also among the training rows (lengths <= "
              f"{OVERLAP_MAX_LEN}): {_fmt_overlap(overlap, sizes)}  "
              f"{time.perf_counter() - t_ov:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - never lose a finished run to the bookkeeping
        overlap = {"error": f"{type(e).__name__}: {e}"}
        print(f"[{run_id}] WARNING: overlap check failed ({overlap['error']})", flush=True)

    def get(p, k, default=0):
        return phases[p][k] if p in phases else default

    done = {"run_id": run_id,
            "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "schedule": cfg["data"]["schedule"], "pe": cfg["model"]["pe"], "seed": cfg["seed"],
            "tag": cfg["tag"], "n_params": n_params, "device": str(device),
            "config_hash": config_hash(cfg), "experiment_hash": experiment_hash(cfg)}
    for p in (1, 2):
        done[f"phase{p}_steps"] = get(p, "steps")
        for k in ("tokens_nonpad", "tokens_padded", "examples"):
            done[f"{k}_phase{p}"] = get(p, k)
    done["tokens_nonpad_total"] = done["tokens_nonpad_phase1"] + done["tokens_nonpad_phase2"]
    done["tokens_padded_total"] = done["tokens_padded_phase1"] + done["tokens_padded_phase2"]
    for p in (1, 2):
        done[f"phase{p}_wall_time_s"] = get(p, "wall_time_s", 0.0)
        done[f"examples_per_length_phase{p}"] = {str(L): c for L, c in
                                                  get(p, "length_counts", {}).items()}
        le = get(p, "last_eval", None)
        done[f"final_val_phase{p}"] = (None if le is None else
                                       {"val_group": le["val_group"],
                                        "val_token_acc": le["val_token_acc"]})
    done["overlap"] = overlap
    done["wall_time_s"] = time.perf_counter() - t_start
    done["torch_version"] = torch.__version__
    done["python_version"] = platform.python_version()
    write_json_atomic(os.path.join(run_dir, "done.json"), done, indent=2)  # LAST
    print(f"[{run_id}] finished in {done['wall_time_s']:.1f}s", flush=True)
    return run_dir


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train one sorting run (docs/CODE_SPEC.md section 5).")
    ap.add_argument("--config", action="append", default=None,
                    help="YAML config; repeat to layer overrides (later wins). "
                         "Default: configs/base.yaml")
    ap.add_argument("--schedule", default=None, help="data.schedule, e.g. R, SL, LS, SL-shuf")
    ap.add_argument("--pe", choices=("sinusoidal", "none"), default=None, help="model.pe")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None, help="appended to the run id")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="override, e.g. --set train.lr=0.001 (value parsed as YAML); repeatable")
    ap.add_argument("--out-dir", default=None, help="overrides out_dir")
    ap.add_argument("--overwrite", action="store_true", help="re-run a finished run")
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace, base_dir=None) -> dict:
    return resolve_config(args.config or [DEFAULT_CONFIG], sets=args.sets,
                          schedule=args.schedule, pe=args.pe, seed=args.seed, tag=args.tag,
                          out_dir=args.out_dir, base_dir=base_dir)


def main(argv=None) -> str:
    args = parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as e:
        raise SystemExit(f"[train] config error: {e}")
    try:
        return run(cfg, overwrite=args.overwrite)
    except RunExistsError as e:
        raise SystemExit(f"[train] {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
