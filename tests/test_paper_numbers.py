"""Tests for the frozen-prediction replication check in src/analysis/paper_numbers.py."""

import itertools

import numpy as np
import pandas as pd

from src.analysis import paper_numbers as pn

SIN_SCHEDS = ["R", "SL", "LS", "SL-shuf", "SLr-early", "SLr-early-shuf", "SLr-mid", "SLr-mid-shuf",
              "SLr-late", "SLr-late-shuf"]
NOPE_SCHEDS = ["R", "SL", "LS", "SL-shuf", "SLr-mid", "SLr-mid-shuf"]


def _frames(sin_seeds, nope_seeds, late_gap=-0.3):
    """Synthetic bucket and per-length tables in which H1-H3, H5, H7 hold for sinusoidal."""
    bk, uni = [], []
    for pe, scheds, seeds in (("sinusoidal", SIN_SCHEDS, sin_seeds), ("none", NOPE_SCHEDS, nope_seeds)):
        for s, seed, ph in itertools.product(scheds, seeds, (1, 2)):
            ind = 0.99
            near = 0.8
            far = 0.05 if pe == "sinusoidal" else 0.5
            if pe == "sinusoidal" and ph == 1 and s in ("SL", "LS"):
                ind = 0.5
            if ph == 1 and s == "LS":
                near = 0.0
            if pe == "sinusoidal" and s == "SLr-late":
                near += late_gap
            if pe == "sinusoidal" and s == "SLr-early" and seed % 2:  # mixed signs, small mean
                near += 0.05
            if pe == "none" and s in ("SLr-mid", "LS"):
                near, far = near + 0.05, far + 0.05
            for bucket, v in (("in_dist", ind), ("near_ood", near), ("far_ood", far)):
                bk.append(dict(pe=pe, schedule=s, seed=seed, phase=ph, bucket=bucket, exact_match=v))
            for L in range(5, 21):
                v = ind
                if pe == "sinusoidal" and ph == 1 and s == "SL" and L <= 10:
                    v = 0.0
                if pe == "sinusoidal" and ph == 1 and s == "LS" and L >= 16:
                    v = 0.0
                uni.append(dict(pe=pe, schedule=s, seed=seed, phase=ph, length=L, exact_match=v))
    return pd.DataFrame(bk), pd.DataFrame(uni)


def test_check_hypothesis_verdicts():
    bk, uni = _frames(range(8), range(5))
    new = [5, 6, 7]
    assert pn.check_hypothesis("H1", bk, uni, new)[1] is True
    assert pn.check_hypothesis("H2", bk, uni, new)[1] is True
    assert pn.check_hypothesis("H3", bk, uni, new)[1] is True
    assert pn.check_hypothesis("H5", bk, uni, new)[1] is True
    assert pn.check_hypothesis("H7", bk, uni, new)[1] is True
    assert pn.check_hypothesis("H8", bk, uni, new)[1] is True
    ev, held = pn.check_hypothesis("H5", bk, uni, new)
    assert "[3/3]" in ev and "-30.0" in ev
    # SLr-late above its twin: H5 fails
    bk2, uni2 = _frames(range(8), range(5), late_gap=+0.1)
    assert pn.check_hypothesis("H5", bk2, uni2, new)[1] is False


def test_replication_rows_and_numbers():
    bk, uni = _frames(range(8), range(5))  # no NoPE seed >= 5
    N = pn.Numbers()
    rows = pn.replication(bk, uni, N)
    assert [r["id"] for r in rows] == [h[0] for h in pn.HYPOTHESES]
    assert N.d["rep/seeds/sinusoidal"] == "5--7"
    assert N.d["rep/tested/sinusoidal"] == "8"
    assert N.d["rep/tested/none"] == "0"
    assert N.d["rep/H9/held"] == "--"
    assert N.d["rep/H1/ls1820"] == "0.0"
    tex = pn.table_replication(rows)
    assert "\\label{tab:replication}" in tex and "not run" in tex
    # with NoPE seeds 5-9 the NoPE predictions are tested
    bk, uni = _frames(range(8), range(10))
    N = pn.Numbers()
    pn.replication(bk, uni, N)
    assert N.d["rep/tested/none"] == "3"
    assert N.d["rep/H9/held"] == "yes" and N.d["rep/H10/held"] == "yes"
    assert N.d["rep/H11/held"] == "yes"
    assert np.isfinite(float(N.d["rep/held/none"]))
