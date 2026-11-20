"""2-layer decoder-only transformer for the sorting task (docs/CODE_SPEC.md section 3).

Architecture: token embedding (+ optional sinusoidal PE) -> n_layers pre-LN blocks
(causal multi-head attention, GELU MLP, no dropout) -> final LayerNorm -> untied unembed.

Design choices:
- pe="none" (NoPE) is the faithful Awasthi & Gupta baseline; pe="sinusoidal" is our deliberate change.
- The unembedding is UNTIED from tok_emb, because the A&G probes project onto separate
  "encoder" (tok_emb) and "decoder" (unembed) bases.
- forward(idx, return_internals=True) exposes the residual stream at each block boundary plus
  the attention probabilities that were actually used to compute the output (the probes read these).

Resolutions of points the spec leaves open (also reported to the orchestrator):
- The sinusoidal table is a NON-persistent buffer. It is a deterministic function of the config,
  so the state_dict holds only learned parameters, and a checkpoint can be loaded into a model
  rebuilt with a larger max_len (e.g. for probes on length-500 lists = 1002 positions > 512).
- Interleaved layout as in Vaswani et al.: pe[p, 2i] = sin(p / 10000^(2i/d)), pe[p, 2i+1] = cos(...).
- max_len only limits the sinusoidal table. NoPE has no positional table, so it accepts any length.
- GELU is the exact (erf) form (nn.GELU default); LayerNorm eps is the PyTorch default 1e-5.
- build_model calls torch.manual_seed(seed) as the spec says, which also reseeds the GLOBAL torch
  RNG (CPU and CUDA). Harmless here: data uses numpy Generators and the model has no dropout.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 103       # = data.VOCAB_SIZE (PAD, values 1..100, SEP, BOS)
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    d_mlp: int = 1024
    pe: str = "sinusoidal"      # "sinusoidal" | "none" (NoPE)
    max_len: int = 512          # sinusoidal table length (>= 2*50+2)


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) Vaswani table. Row p is added to the token at position p (p=0 is BOS)."""
    pos = torch.arange(max_len, dtype=torch.float64)[:, None]
    inv_freq = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    table = torch.zeros(max_len, d_model, dtype=torch.float64)
    # Built in float64 and cast once, so large positions are not hurt by float32 phase error.
    table[:, 0::2] = torch.sin(pos * inv_freq)
    table[:, 1::2] = torch.cos(pos * inv_freq)
    return table.float()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads, self.head_dim = cfg.n_heads, d // cfg.n_heads
        # Separate q/k/v/out projections (all with bias) so each can be inspected on its own.
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, D = x.shape
        # (B, T, D) -> (B, H, T, head_dim)
        q, k, v = (p(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        if return_attn:
            # Explicit path: the returned probabilities are exactly the ones used to mix v.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            att = scores.masked_fill(future, float("-inf")).softmax(dim=-1)  # exact 0 above diagonal
            y = att @ v
        else:
            # Fast path; same math (default scale is 1/sqrt(head_dim)), fused kernel.
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(y), att


class Block(nn.Module):
    """Pre-LN block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
                                 nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        a, att = self.attn(self.ln1(x), return_attn)
        mid = x + a
        post = mid + self.mlp(self.ln2(mid))
        return mid, post, att


class SortTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.pe not in ("sinusoidal", "none"):
            raise ValueError(f"unknown pe {cfg.pe!r}; expected 'sinusoidal' or 'none'")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} is not divisible by n_heads={cfg.n_heads}")
        if cfg.pe == "sinusoidal" and cfg.d_model % 2 != 0:
            raise ValueError(f"sinusoidal PE needs an even d_model, got {cfg.d_model}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # "encoder basis"
        # Non-persistent (see module docstring); None for NoPE, so nothing is added.
        self.register_buffer("pe", sinusoidal_table(cfg.max_len, cfg.d_model)
                             if cfg.pe == "sinusoidal" else None, persistent=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # untied "decoder basis"

    def forward(self, idx: torch.Tensor, return_internals: bool = False):
        """idx (B, T) -> logits (B, T, vocab), or (logits, internals) if return_internals.

        internals (tensors are NOT detached; wrap probes in torch.no_grad()):
          "resid_pre"[l]  (B,T,d)   input to block l (resid_pre[0] = embedding output)
          "resid_mid"[l]  (B,T,d)   after the attention residual add
          "resid_post"[l] (B,T,d)   after the MLP residual add (same tensor as resid_pre[l+1])
          "attn"[l]       (B,H,T,T) attention probabilities (rows = queries, cols = keys)
          "final"         (B,T,d)   ln_f output (what unembed sees)
        Right padding needs no mask: under causal attention real positions never attend to pads.
        """
        T = idx.shape[1]
        x = self.tok_emb(idx)
        if self.pe is not None:
            if T > self.pe.shape[0]:
                raise ValueError(f"sequence length {T} exceeds the sinusoidal table "
                                 f"(max_len={self.pe.shape[0]}); build the model with a larger max_len")
            x = x + self.pe[:T]   # no sqrt(d) scaling of the token embedding

        if not return_internals:
            for block in self.blocks:
                _, x, _ = block(x)
            return self.unembed(self.ln_f(x))

        internals = {"resid_pre": [], "resid_mid": [], "resid_post": [], "attn": []}
        for block in self.blocks:
            internals["resid_pre"].append(x)
            mid, x, att = block(x, return_attn=True)
            internals["resid_mid"].append(mid)
            internals["resid_post"].append(x)
            internals["attn"].append(att)
        internals["final"] = self.ln_f(x)
        return self.unembed(internals["final"]), internals


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(cfg: ModelConfig, seed: int) -> SortTransformer:
    """Seed, construct on CPU, init. The result depends only on (cfg, seed), not on prior RNG use.
    Note: this reseeds the global torch RNG (spec section 3); move to a device afterwards."""
    torch.manual_seed(seed)
    model = SortTransformer(cfg)
    model.apply(_init_weights)   # deterministic module order -> deterministic draws
    return model


def count_params(model: nn.Module) -> int:
    """Number of learned parameters (the sinusoidal table is a buffer, so it is not counted)."""
    return sum(p.numel() for p in model.parameters())
