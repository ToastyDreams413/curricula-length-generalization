"""Numbers quoted in the paper's prose, written as LaTeX macros, plus small generated tables.

    python -m src.analysis.paper_numbers --results results --out paper/tables

Writes <out>/numbers.tex, <out>/table_twins_primary.tex, <out>/table_twins_timeavg.tex and
<out>/table_tokens.tex. Every number in the prose of paper/main.tex comes from numbers.tex, so
re-running this after adding seeds updates the text. Only complete seeds are used
(collect.complete_seed_filter) and only the main tag ("").

In LaTeX, \\nm{key} prints a number. An unknown key prints a red "??", so a typo can't go unseen.
main.tex defines \\nm; numbers.tex only fills the table. Keys use the pe name (sinusoidal|none),
the schedule name (e.g. SLr-mid-shuf), the bucket (in_dist|near_ood|far_ood) or length, and the
phase (1|2):

    nseeds/<pe>                           number of complete seeds
    seeds/<pe>                            e.g. "0--4"
    em/<pe>/<sched>/<bucket>/<phase>      seed-mean exact match, %, 1 decimal
    sd/<pe>/<sched>/<bucket>/<phase>      seed std of the same, %
    emg/<pe>/<sched>/<A|B|C>/<phase>      mean over a training group's test lengths
                                          (A 5-10, B 11-15, C 16-20)
    eml/<pe>/<sched>/<L>/<phase>          a single test length
    emlmax/<pe>/<L>/<phase>               the best schedule's mean at that length
    emrangerep/<pe>/<bucket>/<phase>      emmaxrep - emminrep, points (R, replay schedules, twins)
    dtw/<pe>/<sched>/<bucket>/<phase>     schedule minus its shuffled twin, pp, signed (math mode)
                                          (LS is paired with SL-shuf)
    agtw/..., ptw/...                     seeds agreeing with the mean's sign; exact Wilcoxon p
    dR/..., agR/..., pR/...               schedule minus R, same seed
    ta/<pe>/<sched>/<L>/<phase>           time-averaged exact match on the learning-curve set of
                                          length L (100 lists): Phase 1 steps 16k-20k, Phase 2
                                          steps 1k-5k (5 checkpoints), %
    dta/..., agta/...                     the twin difference of that, pp, and sign agreement
    dta4/<pe>/<a>/<L>/1, agta4/...        same over Phase-1 steps 17k-20k only (sensitivity)
    sdsnap/<pe>/R/22/1, sdta/<pe>/R/22/1  seed std of R at length 22: final snapshot vs
                                          time-averaged, %
    curve/<pe>/<sched>/<L>/<phase>/<step> seed-mean learning-curve exact match (length 22/25/30)
    tok/<sched>/<phase>                   non-pad training tokens, millions (sinusoidal runs)
    overlap/distinct, overlap/runs        distinct training lists also in a val/test set; runs hit
    overlap/checks, overlap/maxlen        number of (run, phase, length, set) checks; max length
    nruns/<pe>, nruns/total               runs used (complete seeds)
    compute/runmin, compute/gpuhours      mean wall time per run (min); total GPU hours
    pmin/<pe>                             smallest attainable two-sided exact Wilcoxon p at that n
    pcons/<pe>                            null probability that a paired cell is "consistent"
                                          (at most one seed disagrees in sign): 2(n+1)/2^n
    err/<pe>/<type>/<bucket>/<phase>      share of incorrect outputs that are order_only,
                                          multiset_only or both (pooled over schedules), %
    pr/<pe>/<sched|all>/<field>/<L|id>/<phase>  probe accuracy, %; field: min1, min2, idsucc,
                                          chance, idtop, succtop; id = mean over lengths 5-20

When seeds >= 5 exist for a pe, the dtw/agtw/dR/agR keys also come in "s04" (seeds 0-4) and
"s5p" (seeds 5+) variants, e.g. dtw.s5p/..., for the replication report.

Replication of the pre-stated predictions (results/prestated_predictions.md, written after
seeds 0-4 and before any seed >= 5 was looked at) is written to <out>/table_replication.tex:

    rep/<H1..H11>/held                    yes | no | -- (not tested: no seeds >= 5 for that pe)
    rep/held/<pe>, rep/tested/<pe>        predictions that held / were tested, per pe
    rep/seeds/<pe>                        the seeds >= 5 used, e.g. "5--7"
    rep/H1/ls1820                         LS exact match (%) on lengths 18-20, end of P1, seeds >= 5
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src.analysis import collect, stats

PES = ("sinusoidal", "none")
BUCKETS = ("in_dist", "near_ood", "far_ood")
GROUPS = {"A": range(5, 11), "B": range(11, 16), "C": range(16, 21)}
TWIN_PAIRS = [("SL", "SL-shuf"), ("SLr-early", "SLr-early-shuf"), ("SLr-mid", "SLr-mid-shuf"),
              ("SLr-late", "SLr-late-shuf"), ("LS", "SL-shuf")]
TA_STEPS = {1: range(16000, 20001, 1000), 2: range(1000, 5001, 1000)}
TA_LENGTHS = (22, 25, 30)


def pct(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def signed_pp(x) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    v = 100 * x
    return f"$+{v:.1f}$" if v >= 0.05 else (f"$-{abs(v):.1f}$" if v <= -0.05 else "$0.0$")


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def agree(d: np.ndarray) -> str:
    """'k/n': seeds whose difference has the same sign as the mean difference (zeros never agree)."""
    d = np.asarray(d, float)
    m = d.mean()
    k = int((d > 0).sum()) if m > 0 else int((d < 0).sum()) if m < 0 else 0
    return f"{k}/{len(d)}"


def wilcoxon_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or not np.any(d != 0):
        return float("nan")
    return float(stats.wilcoxon_signed_rank(d)[1])


def pmin_exact(n: int) -> float:
    return 2.0 / 2 ** n if n > 0 else float("nan")


def pcons_null(n: int) -> float:
    """Null probability that at most one of n seeds disagrees in sign (either direction)."""
    return min(1.0, 2.0 * (n + 1) / 2 ** n) if n > 0 else float("nan")


class Numbers:
    def __init__(self):
        self.d = {}

    def __setitem__(self, k, v):
        self.d[k] = str(v)

    def tex(self) -> str:
        lines = ["% Generated by src/analysis/paper_numbers.py -- do not edit by hand.",
                 "% Use \\nm{key} in the text (\\nm is defined in main.tex)."]
        for k in sorted(self.d):
            lines.append(f"\\expandafter\\def\\csname nm@{k}\\endcsname{{{self.d[k]}}}")
        return "\n".join(lines) + "\n"


def paired(table: pd.DataFrame, a: str, b: str):
    """table: index seed, columns schedules. Returns aligned diffs a-b over seeds having both."""
    if a not in table or b not in table:
        return None
    t = table[[a, b]].dropna()
    return (t[a] - t[b]).to_numpy() if len(t) else None


def build(results: str):
    m, curves = collect.load_runs(results)
    m = m[m["tag"].fillna("") == ""]
    curves = curves[curves["tag"].fillna("") == ""]
    N = Numbers()
    uni = m[m["kind"] == "uniform"].copy()
    uni["length"] = uni["length"].astype(int)
    bk = stats.aggregate_buckets(m, metrics=("exact_match",))
    rows_primary, rows_ta = [], []

    for pe in PES:
        b_pe = bk[bk["pe"] == pe]
        if b_pe.empty:
            continue
        seeds = sorted(b_pe["seed"].unique())
        n = len(seeds)
        N[f"nseeds/{pe}"] = n
        N[f"seeds/{pe}"] = f"{seeds[0]}--{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) \
            else ", ".join(map(str, seeds))
        N[f"pmin/{pe}"] = f"{pmin_exact(n):.3g}"
        N[f"pcons/{pe}"] = f"{pcons_null(n):.2g}"
        subsets = {"": seeds}
        if any(s >= 5 for s in seeds):
            subsets[".s04"] = [s for s in seeds if s < 5]
            subsets[".s5p"] = [s for s in seeds if s >= 5]
            N[f"nseeds.s5p/{pe}"] = len(subsets[".s5p"])

        for ph in (1, 2):
            for bucket in BUCKETS:
                sub = b_pe[(b_pe["phase"] == ph) & (b_pe["bucket"] == bucket)]
                tab = sub.pivot_table(index="seed", columns="schedule", values="exact_match")
                for s in tab.columns:
                    N[f"em/{pe}/{s}/{bucket}/{ph}"] = pct(tab[s].mean())
                    N[f"sd/{pe}/{s}/{bucket}/{ph}"] = pct(tab[s].std(ddof=1))
                means = tab.mean()
                if len(means):
                    N[f"emmax/{pe}/{bucket}/{ph}"] = pct(means.max())
                    N[f"emmin/{pe}/{bucket}/{ph}"] = pct(means.min())
                    rep = means.drop([c for c in ("SL", "LS") if c in means.index])
                    if len(rep):  # R, the replay schedules and all twins (no SL, LS)
                        N[f"emminrep/{pe}/{bucket}/{ph}"] = pct(rep.min())
                        N[f"emmaxrep/{pe}/{bucket}/{ph}"] = pct(rep.max())
                        N[f"emrangerep/{pe}/{bucket}/{ph}"] = f"{100 * (rep.max() - rep.min()):.0f}"
                sds = []
                for a, b in TWIN_PAIRS:
                    d = paired(tab, a, b)
                    if d is not None and len(d) > 1:
                        N[f"sdtw/{pe}/{a}/{bucket}/{ph}"] = pct(d.std(ddof=1))
                        if a != "LS":
                            sds.append(100 * d.std(ddof=1))
                if sds:  # paired-difference sd range over the schedule-twin pairs except LS
                    N[f"sdtwrange/{pe}/{bucket}/{ph}"] = f"{min(sds):.0f}--{max(sds):.0f}"
                for suffix, sd in subsets.items():
                    t = tab.loc[tab.index.isin(sd)]
                    for a, b in TWIN_PAIRS:
                        d = paired(t, a, b)
                        if d is None:
                            continue
                        key = f"{pe}/{a}/{bucket}/{ph}"
                        N[f"dtw{suffix}/{key}"] = signed_pp(d.mean())
                        N[f"agtw{suffix}/{key}"] = agree(d)
                        N[f"ptw{suffix}/{key}"] = fmt_p(wilcoxon_p(d))
                        if suffix == "" and bucket in ("near_ood", "far_ood"):
                            rows_primary.append({"pe": pe, "a": a, "b": b, "phase": ph,
                                                 "bucket": bucket, "mean": d.mean(),
                                                 "sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                                                 "agree": agree(d), "p": wilcoxon_p(d),
                                                 "n": len(d)})
                    for s in t.columns:
                        if s == "R":
                            continue
                        d = paired(t, s, "R")
                        if d is None:
                            continue
                        key = f"{pe}/{s}/{bucket}/{ph}"
                        N[f"dR{suffix}/{key}"] = signed_pp(d.mean())
                        N[f"agR{suffix}/{key}"] = agree(d)
                        N[f"pR{suffix}/{key}"] = fmt_p(wilcoxon_p(d))

            u = uni[(uni["pe"] == pe) & (uni["phase"] == ph)]
            per = u.groupby(["schedule", "seed", "length"])["exact_match"].mean()
            for g, lengths in GROUPS.items():
                gm = per[per.index.get_level_values("length").isin(list(lengths))] \
                    .groupby(["schedule", "seed"]).mean().groupby("schedule").mean()
                for s, v in gm.items():
                    N[f"emg/{pe}/{s}/{g}/{ph}"] = pct(v)
            lm = per.groupby(["schedule", "length"]).mean()
            for (s, L), v in lm.items():
                N[f"eml/{pe}/{s}/{L}/{ph}"] = pct(v)
            for L, v in lm.groupby("length").max().items():  # best schedule at each length
                N[f"emlmax/{pe}/{L}/{ph}"] = pct(v)

        # time-averaged learning-curve exact match (post hoc secondary measure)
        c = curves[(curves["pe"] == pe) & (curves["which"] == "curve_ood")].copy()
        if not c.empty:
            c["length"] = c["length"].astype(int)
            # seed-mean learning curve at every checkpoint: curve/<pe>/<sched>/<L>/<phase>/<step>
            cm = c.groupby(["schedule", "length", "phase", "step"])["value"].mean()
            for (s, L, ph, st), v in cm.items():
                N[f"curve/{pe}/{s}/{L}/{ph}/{int(st)}"] = pct(v)
            for ph, steps in TA_STEPS.items():
                cc = c[(c["phase"] == ph) & (c["step"].isin(list(steps)))]
                ta = cc.groupby(["schedule", "seed", "length"])["value"].mean()
                for L in TA_LENGTHS:
                    tl = ta[ta.index.get_level_values("length") == L].droplevel("length") \
                        .unstack("schedule")
                    if tl.empty:
                        continue
                    for s in tl.columns:
                        N[f"ta/{pe}/{s}/{L}/{ph}"] = pct(tl[s].mean())
                    for a, b in TWIN_PAIRS:
                        d = paired(tl, a, b)
                        if d is None:
                            continue
                        N[f"dta/{pe}/{a}/{L}/{ph}"] = signed_pp(d.mean())
                        N[f"agta/{pe}/{a}/{L}/{ph}"] = agree(d)
                        rows_ta.append({"pe": pe, "a": a, "b": b, "phase": ph, "L": L,
                                        "mean": d.mean(), "agree": agree(d)})
                    if ph == 1 and L == 22 and "R" in tl:
                        N[f"sdta/{pe}/R/22/1"] = pct(tl["R"].std(ddof=1))
            # sensitivity: Phase 1 without the 16k checkpoint, which precedes SLr-late's ABC stage
            cc = c[(c["phase"] == 1) & (c["step"].isin(list(TA_STEPS[1])[1:]))]
            ta4 = cc.groupby(["schedule", "seed", "length"])["value"].mean()
            for L in TA_LENGTHS:
                tl = ta4[ta4.index.get_level_values("length") == L].droplevel("length") \
                    .unstack("schedule")
                for a, b in TWIN_PAIRS:
                    d = paired(tl, a, b) if not tl.empty else None
                    if d is not None:
                        N[f"dta4/{pe}/{a}/{L}/1"] = signed_pp(d.mean())
                        N[f"agta4/{pe}/{a}/{L}/1"] = agree(d)
            snap = uni[(uni["pe"] == pe) & (uni["phase"] == 1) & (uni["schedule"] == "R")
                       & (uni["length"] == 22)]["exact_match"]
            N[f"sdsnap/{pe}/R/22/1"] = pct(snap.std(ddof=1))

        # error types: share of INCORRECT outputs of each type, pooled over schedules, seeds and
        # the lengths of a bucket (each run-length weighted equally)
        for ph in (1, 2):
            u = uni[(uni["pe"] == pe) & (uni["phase"] == ph)].copy()
            u["bucket"] = u["length"].map(stats.bucket_of)
            for bucket in BUCKETS:
                ub = u[u["bucket"] == bucket]
                wrong = (1 - ub["err_correct"]).mean() if len(ub) else np.nan
                for et in ("order_only", "multiset_only", "both"):
                    share = ub[f"err_{et}"].mean() / wrong if wrong and wrong > 0 else np.nan
                    N[f"err/{pe}/{et}/{bucket}/{ph}"] = pct(share)
                    if et == "order_only" and bucket != "in_dist" and np.isfinite(share):
                        k = f"errmax/order_only/{ph}"
                        N[k] = pct(max(share, float(N.d.get(k, "0")) / 100))

    # probes (mean over seeds): min readout after layer-1 / layer-2 attention, identity+successor,
    # chance of the best constant guess; per length, and pooled over in-distribution lengths
    probe_fields = {"min1": "min_acc_after_attn1", "min2": "min_acc_after_attn2",
                    "idsucc": "id_succ_acc", "chance": "min_chance", "idtop": "id_in_top2",
                    "succtop": "succ_in_top2"}
    try:
        pr = collect.load_probes(results)
    except Exception as e:  # noqa: BLE001 - probes are optional
        warnings.warn(f"paper_numbers: no probes ({e})")
        pr = pd.DataFrame()
    if not pr.empty:
        pr = pr[pr["tag"].fillna("") == ""]
        pr["length"] = pr["length"].astype(int)
        for short, col in probe_fields.items():
            if col not in pr:
                continue
            g = pr.groupby(["pe", "schedule", "phase", "length"])[col].mean()
            for (pe, s, ph, L), v in g.items():
                N[f"pr/{pe}/{s}/{short}/{L}/{ph}"] = pct(v)
            idl = pr[pr["length"].isin([5, 10, 15, 20])]
            for (pe, ph), v in idl.groupby(["pe", "phase"])[col].mean().items():
                N[f"pr/{pe}/all/{short}/id/{ph}"] = pct(v)
            for (pe, s, ph), v in idl.groupby(["pe", "schedule", "phase"])[col].mean().items():
                N[f"pr/{pe}/{s}/{short}/id/{ph}"] = pct(v)

    # tokens processed (identical data for both pe, so report the sinusoidal runs)
    tok = m[(m["pe"] == "sinusoidal")].drop_duplicates(["run_id", "phase"])
    tok_rows = tok.groupby(["schedule", "phase"])["tokens_nonpad"].mean() / 1e6
    for (s, ph), v in tok_rows.items():
        N[f"tok/{s}/{ph}"] = f"{v:.1f}"

    # train/test overlap recorded in done.json. All runs of a seed draw from the same example
    # pool, so one colliding list shows up in every run of that seed: count distinct lists as the
    # max over a seed's runs, summed over seeds.
    checks, runs_hit, by_seed = 0, 0, {}
    lengths_checked = set()
    walls, nruns = [], {}
    for run_dir, _cfg, meta, done in collect._finished_runs(results):
        if meta["tag"] != "":
            continue
        nruns[meta["pe"]] = nruns.get(meta["pe"], 0) + 1
        if done.get("wall_time_s"):
            walls.append(float(done["wall_time_s"]))
        hit = False
        for ph, by_len in (done.get("overlap") or {}).items():
            for L, d in by_len.items():
                lengths_checked.add(int(L))
                for kind, k in d.items():
                    checks += 1
                    key = (meta["seed"], ph, L, kind)
                    by_seed[key] = max(by_seed.get(key, 0), int(k))
                    hit = hit or int(k) > 0
        runs_hit += hit
    N["overlap/distinct"] = sum(by_seed.values())
    N["overlap/runs"] = runs_hit
    N["overlap/checks"] = checks
    N["overlap/maxlen"] = max(lengths_checked) if lengths_checked else "--"
    # run counts (complete seeds only) and compute: wall time per run incl. both phases and evals
    for pe, k in nruns.items():
        N[f"nruns/{pe}"] = k
    N["nruns/total"] = sum(nruns.values())
    if walls:
        N["compute/runmin"] = f"{np.mean(walls) / 60:.0f}"
        N["compute/gpuhours"] = f"{np.sum(walls) / 3600:.0f}"
    rep_rows = replication(bk, uni, N)
    return N, pd.DataFrame(rows_primary), pd.DataFrame(rows_ta), tok_rows, rep_rows


# ---------------------------------------------------------------------------------------------
# Pre-stated predictions (results/prestated_predictions.md), checked on the seeds added after
# they were written (seed >= 5). Seeds 0-4 are shown next to them for reference. Directional
# predictions are judged on the seed mean, as written in that file; "in every seed" ones on all
# seeds. Each check returns (evidence string, held) where held is True/False.
REPLAY = ("SLr-early", "SLr-mid", "SLr-late", "SLr-early-shuf", "SLr-mid-shuf", "SLr-late-shuf")
HYPOTHESES = [  # id, pe, short text for the table (LaTeX)
    ("H1", "sinusoidal", r"End of P1: SL and LS below R in-dist.\ in every seed; SL on lengths "
                         r"5--10 and LS on 18--20 $\le$5\%"),
    ("H2", "sinusoidal", r"End of P1: LS below SL-shuf near-OOD in every seed; LS near-OOD "
                         r"$\le$5\%"),
    ("H3", "sinusoidal", r"End of P1: each replay schedule and twin within 1\,pp of R in-dist."),
    ("H4", "sinusoidal", r"End of P1, near-OOD: SLr-early and SLr-mid vs.\ twin "
                         r"$|\Delta|<10$\,pp and seeds not unanimous"),
    ("H5", "sinusoidal", r"End of P1, near-OOD: SLr-late below its twin"),
    ("H6", "sinusoidal", r"End of P2, far-OOD: SL above SL-shuf"),
    ("H7", "sinusoidal", r"End of P2: every schedule within 1\,pp of R in-dist."),
    ("H8", "sinusoidal", r"Far-OOD $\le$15\% for every schedule and phase"),
    ("H9", "none", r"NoPE, end of P1: SLr-mid above its twin, near- and far-OOD"),
    ("H10", "none", r"NoPE, end of P2: LS above SL-shuf, near- and far-OOD"),
    ("H11", "none", r"NoPE, end of P1: SL and LS in-dist.\ $\ge$85\%"),
]


def _btab(bk, pe, ph, bucket, seeds):
    sub = bk[(bk["pe"] == pe) & (bk["phase"] == ph) & (bk["bucket"] == bucket)
             & (bk["seed"].isin(seeds))]
    return sub.pivot_table(index="seed", columns="schedule", values="exact_match")


def _lenmean(uni, pe, ph, sched, lengths, seeds):
    u = uni[(uni["pe"] == pe) & (uni["phase"] == ph) & (uni["schedule"] == sched)
            & (uni["length"].isin(list(lengths))) & (uni["seed"].isin(seeds))]
    if u.empty:
        return np.nan
    return float(u.groupby(["seed", "length"])["exact_match"].mean().groupby("seed").mean().mean())


def _d(d):
    """Mean difference [sign agreement]."""
    return f"{signed_pp(d.mean())} [{agree(d)}]"


def check_hypothesis(hid, bk, uni, seeds):
    """Evidence string and verdict for one frozen prediction on the given seeds."""
    if hid == "H1":
        t = _btab(bk, "sinusoidal", 1, "in_dist", seeds)
        dsl, dls = paired(t, "SL", "R"), paired(t, "LS", "R")
        a = _lenmean(uni, "sinusoidal", 1, "SL", range(5, 11), seeds)
        b = _lenmean(uni, "sinusoidal", 1, "LS", range(18, 21), seeds)
        held = bool((dsl < 0).all() and (dls < 0).all() and a <= 0.05 and b <= 0.05)
        return f"{_d(dsl)}, {_d(dls)}; {pct(a)}\\%, {pct(b)}\\%", held
    if hid == "H2":
        t = _btab(bk, "sinusoidal", 1, "near_ood", seeds)
        d = paired(t, "LS", "SL-shuf")
        return f"{_d(d)}; {pct(t['LS'].mean())}\\%", bool((d < 0).all() and t["LS"].mean() <= 0.05)
    if hid in ("H3", "H7"):
        ph = 1 if hid == "H3" else 2
        t = _btab(bk, "sinusoidal", ph, "in_dist", seeds)
        scheds = REPLAY if hid == "H3" else [c for c in t.columns if c != "R"]
        devs = [abs(paired(t, s, "R").mean()) for s in scheds if s in t]
        worst = max(devs)
        return f"max $|\\Delta|$ = {100 * worst:.1f}\\,pp", bool(worst <= 0.01)
    if hid == "H4":
        t = _btab(bk, "sinusoidal", 1, "near_ood", seeds)
        parts, held = [], True
        for a in ("SLr-early", "SLr-mid"):
            d = paired(t, a, a + "-shuf")
            v = 100 * d.mean()
            # near the 10-point bound, one decimal can round across it: show two
            parts.append(f"${v:+.2f}$ [{agree(d)}]" if 9.95 <= abs(v) < 10.05 else _d(d))
            unanimous = bool((d > 0).all() or (d < 0).all())
            held = held and abs(d.mean()) < 0.10 and not unanimous
        return ", ".join(parts), bool(held)
    if hid == "H5":
        d = paired(_btab(bk, "sinusoidal", 1, "near_ood", seeds), "SLr-late", "SLr-late-shuf")
        return _d(d), bool(d.mean() < 0)
    if hid == "H6":
        d = paired(_btab(bk, "sinusoidal", 2, "far_ood", seeds), "SL", "SL-shuf")
        return _d(d), bool(d.mean() > 0)
    if hid == "H8":
        means = {(s, ph): v for ph in (1, 2)
                 for s, v in _btab(bk, "sinusoidal", ph, "far_ood", seeds).mean().items()}
        (s, ph), worst = max(means.items(), key=lambda kv: kv[1])
        return f"max {pct(worst)}\\% ({s}, P{ph})", bool(worst <= 0.15)
    if hid in ("H9", "H10"):
        a, b, ph = ("SLr-mid", "SLr-mid-shuf", 1) if hid == "H9" else ("LS", "SL-shuf", 2)
        ds = [paired(_btab(bk, "none", ph, bucket, seeds), a, b)
              for bucket in ("near_ood", "far_ood")]
        return ", ".join(_d(d) for d in ds), bool(all(d.mean() > 0 for d in ds))
    if hid == "H11":
        t = _btab(bk, "none", 1, "in_dist", seeds)
        v = [t[s].mean() for s in ("SL", "LS")]
        return f"{pct(v[0])}\\%, {pct(v[1])}\\%", bool(min(v) >= 0.85)
    raise KeyError(hid)


def _seed_range(seeds) -> str:
    seeds = list(seeds)
    if not seeds:
        return "--"
    return f"{seeds[0]}--{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) \
        else ", ".join(map(str, seeds))


def replication(bk, uni, N) -> list:
    """Rows for the replication table; also fills the rep/* numbers."""
    rows, held_n, tested_n = [], {}, {}
    for hid, pe, text in HYPOTHESES:
        seeds = sorted(int(s) for s in bk.loc[bk["pe"] == pe, "seed"].unique())
        old, new = [s for s in seeds if s < 5], [s for s in seeds if s >= 5]
        ev_old = check_hypothesis(hid, bk, uni, old)[0] if old else "--"
        if new:
            ev_new, held = check_hypothesis(hid, bk, uni, new)
            tested_n[pe] = tested_n.get(pe, 0) + 1
            held_n[pe] = held_n.get(pe, 0) + int(held)
            verdict = "yes" if held else "no"
        else:
            ev_new, verdict = "not run", "--"
        N[f"rep/{hid}/held"] = verdict
        if hid == "H1" and new:  # the H1 threshold quantity the text quotes (LS on 18-20)
            N["rep/H1/ls1820"] = pct(_lenmean(uni, "sinusoidal", 1, "LS", range(18, 21), new))
        rows.append({"id": hid, "pe": pe, "text": text, "old": ev_old, "new": ev_new,
                     "held": verdict, "new_seeds": new})
    for pe in PES:
        new = sorted(int(s) for s in bk.loc[bk["pe"] == pe, "seed"].unique() if s >= 5)
        N[f"rep/held/{pe}"] = held_n.get(pe, 0)
        N[f"rep/tested/{pe}"] = tested_n.get(pe, 0)
        if new:
            N[f"rep/seeds/{pe}"] = _seed_range(new)
    return rows


def table_replication(rows: list) -> str:
    if not rows:
        return "% no data\n"
    out = [r"\begin{table}[h]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Predictions written down after seeds 0--4 and before any result from a "
           r"later seed existed, checked on the later seeds only. Differences are in percentage "
           r"points of exact match, paired within a seed, with [seeds agreeing in sign]. "
           r"Directional and threshold predictions are judged on the seed mean (H8: the largest "
           r"schedule mean); ``in every seed'' predictions on every later seed. Exact-match values "
           r"are seed means.}",
           r"\label{tab:replication}",
           r"\begin{tabular}{@{}lp{4.0cm}p{3.5cm}p{3.5cm}c@{}}", r"\toprule",
           r"& Prediction & Seeds 0--4 & Later seeds & Held \\", r"\midrule"]
    last_pe = None
    for r in rows:
        if last_pe is not None and r["pe"] != last_pe:
            out.append(r"\midrule")
        last_pe = r["pe"]
        new = r["new"] if not r["new_seeds"] else \
            f"{r['new']} \\newline (seeds {_seed_range(r['new_seeds'])})"
        out.append(f"{r['id']} & {r['text']} & {r['old']} & {new} & {r['held']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


PE_TEX = {"sinusoidal": "Sinusoidal", "none": "NoPE"}
BUCKET_TEX = {"near_ood": "Near-OOD (21--24)", "far_ood": "Far-OOD (25--50)"}


def table_primary(df: pd.DataFrame, nseeds: dict) -> str:
    """Main-text table: ordered schedule minus shuffled twin, near/far OOD, both phases."""
    if df.empty:
        return "% no data\n"
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Order versus content. Each entry is the ordered schedule minus its shuffled "
           r"twin, in percentage points of exact match, averaged over seeds (paired within a seed). "
           r"In brackets: the number of seeds whose difference has the same sign as the mean. "
           r"LS is paired with SL-shuf, which holds the same examples. "
           + ", ".join(f"{PE_TEX[p]}: {k} seeds" for p, k in nseeds.items()) + r".}",
           r"\label{tab:twins}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llcccc@{}}",
           r"\toprule",
           r" & & \multicolumn{2}{c}{End of Phase 1} & \multicolumn{2}{c}{End of Phase 2} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"PE & Schedule vs.\ twin & Near-OOD & Far-OOD & Near-OOD & Far-OOD \\", r"\midrule"]
    for pe in PES:
        d = df[df["pe"] == pe]
        if d.empty:
            continue
        first = True
        for a, b in TWIN_PAIRS:
            dd = d[d["a"] == a]
            if dd.empty:
                continue
            cells = []
            for ph in (1, 2):
                for bucket in ("near_ood", "far_ood"):
                    r = dd[(dd["phase"] == ph) & (dd["bucket"] == bucket)]
                    cells.append("--" if r.empty else
                                 f"{signed_pp(r['mean'].iloc[0])} [{r['agree'].iloc[0]}]")
            label = PE_TEX[pe] if first else ""
            first = False
            out.append(f"{label} & {a} vs.\\ {b} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def table_timeavg(df: pd.DataFrame) -> str:
    if df.empty:
        return "% no data\n"
    out = [r"\begin{table}[h]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Ordered schedule minus shuffled twin on the time-averaged learning-curve "
           r"measure (post hoc): exact match on the 100-list curve sets at lengths 22, 25 and 30, "
           r"averaged over the last five checkpoints of each phase (Phase 1: steps 16k--20k; "
           r"Phase 2: steps 1k--5k). Percentage points; [seeds agreeing in sign]. Each schedule "
           r"is compared with its own shuffled twin; LS with SL-shuf.}",
           r"\label{tab:twins_timeavg}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{Phase 1} & \multicolumn{3}{c}{Phase 2} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Schedule & 22 & 25 & 30 & 22 & 25 & 30 \\", r"\midrule"]
    for pe in PES:
        d = df[df["pe"] == pe]
        if d.empty:
            continue
        out.append(r"\multicolumn{7}{@{}l}{\emph{" + PE_TEX[pe] + r"}} \\")
        for a, b in TWIN_PAIRS:
            dd = d[d["a"] == a]
            if dd.empty:
                continue
            cells = []
            for ph in (1, 2):
                for L in TA_LENGTHS:
                    r = dd[(dd["phase"] == ph) & (dd["L"] == L)]
                    cells.append("--" if r.empty else
                                 f"{signed_pp(r['mean'].iloc[0])} [{r['agree'].iloc[0]}]")
            out.append(f"{a} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def table_tokens(tok_rows: pd.Series) -> str:
    if tok_rows.empty:
        return "% no data\n"
    order = stats.order_schedules(sorted(set(tok_rows.index.get_level_values(0))))
    out = [r"\begin{table}[h]", r"\centering", r"\small",
           r"\caption{Training tokens processed (non-padding, millions; mean over seeds). Every "
           r"schedule takes the same number of updates (20k in Phase 1, 5k in Phase 2), so "
           r"schedules that favour short lists see fewer tokens. Phase 2 is identical for all.}",
           r"\label{tab:tokens}", r"\begin{tabular}{lcc}", r"\toprule",
           r"Schedule & Phase 1 & Phase 2 \\", r"\midrule"]
    for s in order:
        v1 = tok_rows.get((s, 1), np.nan)
        v2 = tok_rows.get((s, 2), np.nan)
        out.append(f"{s} & {v1:.1f} & {v2:.1f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the paper's numbers as LaTeX macros.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        N, prim, ta, tok, rep = build(args.results)
    os.makedirs(args.out, exist_ok=True)
    nseeds = {pe: int(N.d[f"nseeds/{pe}"]) for pe in PES if f"nseeds/{pe}" in N.d}
    files = {"numbers.tex": N.tex(), "table_twins_primary.tex": table_primary(prim, nseeds),
             "table_twins_timeavg.tex": table_timeavg(ta), "table_tokens.tex": table_tokens(tok),
             "table_replication.tex": table_replication(rep)}
    for name, text in files.items():
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[paper_numbers] wrote {os.path.join(args.out, name)}")
    print(f"[paper_numbers] {len(N.d)} numbers; seeds: {nseeds}")


if __name__ == "__main__":
    main()
