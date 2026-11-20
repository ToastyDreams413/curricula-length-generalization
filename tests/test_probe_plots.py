"""Tests for fig6 (src/analysis/plots.py plot_probe_accuracy) with new-style probes data (min
readout after layer 1 and 2, min_chance) and old-style data (min_acc / id_succ_acc only).
CPU only, no torch. The probes DataFrame is built here with the columns that
collect.load_probes produces, so the test does not depend on collect.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import plots
from src.analysis.plots import NoDataError

LENGTHS = (5, 10, 20, 30)
PLOT_DPI = 60


def chance_of(L):
    return round(1 - 0.99 ** L, 4)          # stands in for the best-constant rate at length L


def probes_frame(new_style=True, seeds=(0, 1), pes=("sinusoidal", "none"),
                 schedules=("R", "SL")):
    rows = []
    for pe in pes:
        for s in schedules:
            for seed in seeds:
                for L in LENGTHS:
                    row = {"run_id": f"{pe}_{s}_s{seed}", "pe": pe, "schedule": s,
                           "base_schedule": s, "is_twin": False, "seed": seed, "tag": "",
                           "phase": 1, "readout": "decoder_basis", "length": L, "n": 500,
                           "min_acc": 0.05 + 0.002 * L + 0.01 * seed,
                           "id_succ_acc": 0.9 - 0.004 * L}
                    if new_style:
                        row.update({"min_acc_after_attn2": 0.95 - 0.003 * L,
                                    "min_chance": chance_of(L), "min_mode_frac": 1.0,
                                    "min_acc_centered": 0.2})
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def captured(monkeypatch):
    """Record, for every figure saved by plots._save, each axis's title and its chance lines."""
    figs = []
    real_save = plots._save

    def save(fig, out, dpi=None):
        figs.append([{"title": ax.get_title(loc="left"),
                      "chance": [ln for ln in ax.get_lines() if ln.get_gid() == "min_chance"],
                      "n_lines": len(ax.get_lines())} for ax in fig.axes])
        figs[-1].append([t.get_text() for lg in fig.legends for t in lg.get_texts()])
        return real_save(fig, out, dpi)

    monkeypatch.setattr(plots, "_save", save)
    return figs


def _check_files(paths):
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    for p in paths:
        assert p.is_file() and p.stat().st_size > 1000, p


def test_fig6_new_style(captured, tmp_path):
    _check_files(plots.plot_probe_accuracy(probes_frame(), tmp_path / "fig6", dpi=PLOT_DPI))
    *axes, legend = captured[-1]
    assert len(axes) == 2 * 3                                      # 2 PEs x 3 probes
    assert [a["title"] for a in axes[:3]] == ["Min after L1 attn", "Min after L2 attn",
                                              "Id+succ after L2 attn"]
    assert "chance (best constant)" in legend and {"R", "SL"} <= set(legend)
    for k, a in enumerate(axes):
        if k % 3 == 2:                                             # id+succ: no chance line
            assert a["chance"] == []
            continue
        assert len(a["chance"]) == 1
        ln = a["chance"][0]
        assert ln.get_color() == plots.MUTED and ln.get_linestyle() == "--"
        assert ln.get_marker() in ("None", None, "")
        assert list(ln.get_xdata()) == list(LENGTHS)
        assert np.allclose(ln.get_ydata(), [chance_of(L) for L in LENGTHS])


def test_fig6_chance_is_the_mean_over_runs(captured, tmp_path):
    d = probes_frame(pes=("sinusoidal",), schedules=("R",))
    d.loc[d["seed"] == 1, "min_chance"] += 0.1                     # runs disagree: plot the mean
    plots.plot_probe_accuracy(d, tmp_path / "fig6m", dpi=PLOT_DPI)
    ln = captured[-1][0]["chance"][0]
    assert np.allclose(ln.get_ydata(), [chance_of(L) + 0.05 for L in LENGTHS])


def test_fig6_old_style_degrades(captured, tmp_path):
    _check_files(plots.plot_probe_accuracy(probes_frame(new_style=False), tmp_path / "fig6o",
                                           dpi=PLOT_DPI))
    *axes, legend = captured[-1]
    assert len(axes) == 2 * 2                                      # no after-L2 min column
    assert [a["title"] for a in axes[:2]] == ["Min after L1 attn", "Id+succ after L2 attn"]
    assert all(a["chance"] == [] for a in axes)
    assert "chance (best constant)" not in legend


def test_fig6_mixed_old_and_new_runs(captured, tmp_path):
    # Old probes files (no new keys) next to new ones: columns exist but are partly NaN.
    new = probes_frame(pes=("sinusoidal",), schedules=("R",), seeds=(0,))
    old = probes_frame(new_style=False, pes=("sinusoidal",), schedules=("SL",), seeds=(0,))
    d = pd.concat([new, old], ignore_index=True)
    _check_files(plots.plot_probe_accuracy(d, tmp_path / "fig6x", dpi=PLOT_DPI))
    *axes, _legend = captured[-1]
    assert len(axes) == 3 and len(axes[0]["chance"]) == 1 and axes[2]["chance"] == []
    # Chance present but all NaN: no line, no legend entry
    d2 = probes_frame(pes=("sinusoidal",), seeds=(0,))
    d2["min_chance"] = np.nan
    plots.plot_probe_accuracy(d2, tmp_path / "fig6n", dpi=PLOT_DPI)
    *axes, legend = captured[-1]
    assert all(a["chance"] == [] for a in axes) and "chance (best constant)" not in legend
    # chance=None switches the reference off
    plots.plot_probe_accuracy(probes_frame(pes=("none",)), tmp_path / "fig6c", chance=None,
                              dpi=PLOT_DPI)
    assert all(a["chance"] == [] for a in captured[-1][:-1])


def test_fig6_no_data(tmp_path):
    with pytest.raises(NoDataError):
        plots.plot_probe_accuracy(probes_frame().iloc[:0], tmp_path / "x", dpi=PLOT_DPI)
    with pytest.raises(NoDataError):                               # none of the probe columns
        plots.plot_probe_accuracy(probes_frame().drop(
            columns=["min_acc", "id_succ_acc", "min_acc_after_attn2"]), tmp_path / "x",
            dpi=PLOT_DPI)
