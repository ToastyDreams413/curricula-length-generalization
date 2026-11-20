"""Tests for src/model.py (docs/CODE_SPEC.md section 3). CPU only, small configs, fast."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ModelConfig, SortTransformer, build_model, count_params

# Token ids from spec section 1, hardcoded on purpose: src/data.py is being rewritten concurrently.
PAD, SEP, BOS, VOCAB = 0, 101, 102, 103
DEVICE = torch.device("cpu")   # build_model constructs on CPU; inputs are created on DEVICE
PES = ["sinusoidal", "none"]
INTERNAL_KEYS = {"resid_pre", "resid_mid", "resid_post", "attn", "final"}


def small_cfg(pe="sinusoidal", **kw):
    return ModelConfig(d_model=64, n_layers=2, n_heads=4, d_mlp=256, pe=pe, **kw)


def scrambled_model(pe, seed=0, **kw):
    """A small model with large random weights. At the 0.02 init attention is nearly uniform, so a
    wrong scale or mask could hide below tolerance; here attention is peaked and logits are O(1)."""
    model = build_model(small_cfg(pe, **kw), seed).to(DEVICE).eval()
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


def rand_idx(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (B, T), generator=g).to(DEVICE)   # values, SEP and BOS


def run(model, idx, internals):
    with torch.no_grad():
        out = model(idx, return_internals=internals)
    return out[0] if internals else out


def close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-5)


# 1. output shapes
@pytest.mark.parametrize("pe", PES)
def test_output_shapes(pe):
    model = build_model(small_cfg(pe), 0).eval()
    idx = rand_idx(3, 11)
    assert run(model, idx, False).shape == (3, 11, VOCAB)
    with torch.no_grad():
        logits, internals = model(idx, return_internals=True)
    assert logits.shape == (3, 11, VOCAB) and set(internals) == INTERNAL_KEYS


# 2. causality, both paths
@pytest.mark.parametrize("pe", PES)
@pytest.mark.parametrize("internals", [False, True])
def test_causality(pe, internals):
    model = scrambled_model(pe)
    idx = rand_idx(2, 12)
    base = run(model, idx, internals)
    for k in [0, 1, 5, 11]:
        alt = idx.clone()
        alt[:, k] = alt[:, k] % 100 + 1          # always a different token id
        out = run(model, alt, internals)
        assert close(out[:, :k], base[:, :k], atol=1e-6), f"positions < {k} saw the change at {k}"
        assert not close(out[:, k], base[:, k], atol=1e-3), "the change should be visible at k"


# 3. fast path == internals path (and no dropout: train/eval agree)
@pytest.mark.parametrize("pe", PES)
@pytest.mark.parametrize("scrambled", [False, True])
def test_fast_path_matches_internals_path(pe, scrambled):
    model = scrambled_model(pe) if scrambled else build_model(small_cfg(pe), 0).eval()
    idx = rand_idx(4, 21, seed=3)                 # a length-10 example has 2n+1 = 21 inputs
    fast, slow = run(model, idx, False), run(model, idx, True)
    assert close(fast, slow, atol=1e-5), (fast - slow).abs().max()
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())
    model.train()
    assert close(run(model, idx, False), fast, atol=1e-6)


# 4. right padding needs no mask
@pytest.mark.parametrize("pe", PES)
@pytest.mark.parametrize("internals", [False, True])
def test_right_padding_does_not_change_real_positions(pe, internals):
    model = scrambled_model(pe)
    a, b = rand_idx(1, 9, seed=1), rand_idx(1, 15, seed=2)
    alone_a, alone_b = run(model, a, internals), run(model, b, internals)
    padded_a = torch.cat([a, torch.full((1, 6), PAD)], dim=1)
    assert close(run(model, padded_a, internals)[:, :9], alone_a)
    # mixed-length batch, as encode_batch produces it
    out = run(model, torch.cat([padded_a, b], dim=0), internals)
    assert close(out[0, :9], alone_a[0]) and close(out[1], alone_b[0])


# 5. internals consistency
@pytest.mark.parametrize("pe", PES)
def test_internals_consistency(pe):
    cfg = small_cfg(pe)
    model = scrambled_model(pe)
    B, T, L, H, d = 3, 13, cfg.n_layers, cfg.n_heads, cfg.d_model
    hd = d // H
    idx = rand_idx(B, T, seed=4)
    with torch.no_grad():
        logits, it = model(idx, return_internals=True)

        assert set(it) == INTERNAL_KEYS
        for key in ("resid_pre", "resid_mid", "resid_post"):
            assert len(it[key]) == L and all(t.shape == (B, T, d) for t in it[key])
        assert len(it["attn"]) == L and all(a.shape == (B, H, T, T) for a in it["attn"])
        assert it["final"].shape == (B, T, d)

        upper = torch.ones(T, T, dtype=torch.bool).triu(1)
        for att in it["attn"]:
            assert close(att.sum(-1), torch.ones(B, H, T), atol=1e-5)
            assert (att >= 0).all() and (att[..., upper] == 0).all()   # exactly zero on the future
        for l in range(L - 1):
            assert torch.equal(it["resid_post"][l], it["resid_pre"][l + 1])
        assert close(it["final"], model.ln_f(it["resid_post"][-1]), atol=1e-6)
        assert close(logits, model.unembed(it["final"]), atol=1e-6)

        emb = model.tok_emb(idx)
        if pe == "none":
            assert torch.equal(it["resid_pre"][0], emb)
        else:
            assert close(it["resid_pre"][0], emb + model.pe[:T], atol=1e-6)

        # Rebuild every block by hand from the returned probabilities: this checks that "attn" is
        # what the model actually used, the 1/sqrt(head_dim) scale, the mask, and the residual adds.
        for l, blk in enumerate(model.blocks):
            h = blk.ln1(it["resid_pre"][l])
            q, k, v = (p(h).view(B, T, H, hd).transpose(1, 2)
                       for p in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            expected_att = scores.masked_fill(upper, float("-inf")).softmax(-1)
            assert close(it["attn"][l], expected_att, atol=1e-6)
            y = (it["attn"][l] @ v).transpose(1, 2).reshape(B, T, d)
            assert close(it["resid_mid"][l], it["resid_pre"][l] + blk.attn.out_proj(y))
            mid = it["resid_mid"][l]
            assert close(it["resid_post"][l], mid + blk.mlp(blk.ln2(mid)))
        # sanity: the scrambled model has peaked attention, so the checks above have teeth
        assert max(a.max().item() for a in it["attn"]) > 0.9


def test_sinusoidal_table_values():
    model = build_model(small_cfg("sinusoidal"), 0)
    d, table = 64, model.pe
    assert table.shape == (512, d) and table.dtype == torch.float32
    for p in [0, 1, 7, 41, 101, 511]:
        for i in [0, 1, 5, 31]:                  # dims 2i (sin) and 2i+1 (cos)
            angle = p / 10000 ** (2 * i / d)
            assert abs(table[p, 2 * i].item() - math.sin(angle)) < 1e-6
            assert abs(table[p, 2 * i + 1].item() - math.cos(angle)) < 1e-6
    assert build_model(small_cfg("none"), 0).pe is None


def test_pe_table_is_a_non_persistent_buffer():
    """The table is rebuilt from the config, so a checkpoint loads into a longer-max_len model."""
    model = scrambled_model("sinusoidal")
    sd = model.state_dict()
    assert "pe" not in sd and all("pe" != n for n, _ in model.named_parameters())
    longer = build_model(small_cfg("sinusoidal", max_len=1100), 7).eval()
    longer.load_state_dict(sd)                    # strict load works despite the different max_len
    idx = rand_idx(2, 21)
    assert close(run(longer, idx, False), run(model, idx, False), atol=1e-6)
    assert run(longer, rand_idx(1, 1002), False).shape == (1, 1002, VOCAB)  # length-500 probes


# 6. table length and the max_len error
@pytest.mark.parametrize("internals", [False, True])
def test_max_len(internals):
    assert ModelConfig().max_len >= 2 * 50 + 2
    assert run(build_model(small_cfg("sinusoidal"), 0), rand_idx(1, 102), internals).shape == (1, 102, VOCAB)
    model = build_model(small_cfg("sinusoidal", max_len=102), 0)
    assert run(model, rand_idx(2, 102), internals).shape == (2, 102, VOCAB)
    with pytest.raises(ValueError, match="max_len"):
        run(model, rand_idx(1, 103), internals)
    # NoPE has no table, so max_len does not limit it (documented in model.py)
    nope = build_model(small_cfg("none", max_len=102), 0)
    assert run(nope, rand_idx(1, 130), internals).shape == (1, 130, VOCAB)


def test_invalid_configs_raise():
    for bad in [small_cfg("learned"), ModelConfig(d_model=64, n_heads=5), ModelConfig(d_model=63, n_heads=7)]:
        with pytest.raises(ValueError):
            SortTransformer(bad)


# 7. build_model determinism and init
def test_build_model_determinism():
    cfg = small_cfg()
    a = build_model(cfg, 123).state_dict()
    torch.randn(1000), np.random.rand(1000), torch.manual_seed(999), torch.rand(17)  # consume RNG
    b = build_model(cfg, 123).state_dict()
    c = build_model(cfg, 124).state_dict()
    assert a.keys() == b.keys() == c.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    matrices = [k for k in a if a[k].ndim == 2]   # LN params are 1/0 under every seed
    assert matrices and all(not torch.equal(a[k], c[k]) for k in matrices)


def test_init_distribution():
    model = build_model(ModelConfig(), 0)        # default size: enough entries for tight estimates
    n_checked = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            w = m.weight
            assert 0.0185 < w.std().item() < 0.0215 and abs(w.mean().item()) < 2e-3, name
            if getattr(m, "bias", None) is not None:
                assert (m.bias == 0).all(), name
            n_checked += 1
        elif isinstance(m, nn.LayerNorm):
            assert (m.weight == 1).all() and (m.bias == 0).all(), name
    assert n_checked == 1 + 2 * (4 + 2) + 1        # tok_emb, per block q/k/v/out + 2 MLP, unembed
    assert model.tok_emb.padding_idx is None
    for blk in model.blocks:                     # exact (erf) GELU, default LayerNorm eps
        assert isinstance(blk.mlp[1], nn.GELU) and blk.mlp[1].approximate == "none"
        assert blk.ln1.eps == blk.ln2.eps == model.ln_f.eps == 1e-5


# 8. untied unembedding
def test_unembed_is_untied_and_bias_free():
    model = build_model(small_cfg(), 0)
    assert model.unembed.weight is not model.tok_emb.weight
    assert model.unembed.weight.data_ptr() != model.tok_emb.weight.data_ptr()
    assert model.unembed.bias is None
    assert model.unembed.weight.shape == model.tok_emb.weight.shape == (VOCAB, 64)
    assert not torch.equal(model.unembed.weight, model.tok_emb.weight)


# 9. parameter count
def expected_params(cfg):
    d, V, m = cfg.d_model, cfg.vocab_size, cfg.d_mlp
    attn = 4 * (d * d + d)                 # q, k, v, out with bias
    mlp = (d * m + m) + (m * d + d)
    block = attn + mlp + 2 * (2 * d)       # + ln1, ln2
    return V * d + cfg.n_layers * block + 2 * d + d * V   # tok_emb, blocks, ln_f, unembed (no bias)


def test_count_params_default_config():
    cfg = ModelConfig()
    assert (cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_mlp) == (103, 256, 2, 8, 1024)
    n = count_params(build_model(cfg, 0))
    print(f"\ndefault-config params: {n:,}")
    assert n == expected_params(cfg) == 1_632_768
    assert 1.5e6 < n < 1.8e6
    assert count_params(build_model(ModelConfig(pe="none"), 0)) == n   # PE adds no parameters
    assert count_params(build_model(small_cfg(), 0)) == expected_params(small_cfg())


# 10. tiny overfit sanity check
def sorting_batch(n=5, B=8, seed=0):
    """[BOS, x, SEP, sorted(x)] -> (input, target, loss_mask) exactly as in spec section 1."""
    xs = np.random.default_rng(seed).integers(1, 101, size=(B, n))
    seq = np.concatenate([np.full((B, 1), BOS), xs, np.full((B, 1), SEP), np.sort(xs, axis=1)], axis=1)
    inp, tgt = torch.from_numpy(seq[:, :-1]), torch.from_numpy(seq[:, 1:])
    mask = torch.zeros(B, 2 * n + 1, dtype=torch.bool)
    mask[:, n + 1: 2 * n + 1] = True             # t in [n+1, 2n]
    return inp, tgt, mask, xs


def masked_ce(logits, tgt, mask):
    return F.cross_entropy(logits[mask], tgt[mask])


@pytest.fixture
def one_thread():
    """A tiny model is dispatch-bound; extra CPU threads only add overhead (about 3x here)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("pe", PES)
def test_overfit_tiny_batch(pe, one_thread):
    n, B, seed = 5, 8, 0                         # margin was checked offline on seeds 0-2
    inp, tgt, mask, xs = sorting_batch(n, B, seed)
    assert inp.shape == tgt.shape == (B, 2 * n + 1)
    assert (inp[:, 0] == BOS).all() and (inp[:, n + 1] == SEP).all()
    assert torch.equal(tgt[mask].view(B, n), torch.from_numpy(np.sort(xs, axis=1)))

    model = build_model(small_cfg(pe), seed).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = None
    for _ in range(200):                         # loss is ~0.005 by step 200 (seeds 0-2)
        loss = masked_ce(model(inp), tgt, mask)
        initial = loss.item() if initial is None else initial
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(inp)
    final = masked_ce(logits, tgt, mask).item()
    assert initial > 3.0                           # about ln(103) = 4.63 at init
    assert final < 0.05 and final < 0.05 * initial, (initial, final)
    assert torch.equal(logits.argmax(-1)[mask], tgt[mask])
