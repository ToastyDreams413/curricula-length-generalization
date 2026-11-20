# Pre-stated predictions for seeds 5–9

These predictions were written after examining seeds 0–4 of every condition and before any result from seeds 5–9 existed. The paper checks each one on seeds 5–9 alone (replication table, Appendix C) and reports pooled results over all seeds separately.

- **Metric:** exact match on the test sets at the end of each phase.
- **Buckets:**
  - in-dist: lengths 5–20
  - near-OOD: 21–24
  - far-OOD: 25, 30, 40, 50
- **Diff:** a paired difference within a seed.

## Sinusoidal PE (main grid)

**Seed 0–4 status.** H1 and H2 were seen in 5/5 seeds. H5 and H6 are weaker: 4/5 seeds, or 5/5 only on the time-averaged curve measure.

| # | Prediction | Seeds 0–4 evidence |
|---|---|---|
| H1 | **Forgetting without replay.** At the end of Phase 1, SL and LS in-dist < R in every seed. SL exact match at lengths 5–10 ≤ 0.05, and LS at lengths ≥ 18 ≤ 0.05. | diffs −0.49 and −0.40, 5/5 |
| H2 | At the end of Phase 1, LS near-OOD < SL-shuf in every seed, with LS near-OOD ≤ 0.05. | 0.00 vs 0.59, 5/5 |
| H3 | Replay schedules (SLr-early/mid/late and their twins), Phase 1: in-dist within 0.01 of R on average. | holds |
| H4 | **Order vs content, null.** SLr-early vs twin and SLr-mid vs twin, near-OOD, Phase 1: \|mean diff\| < 0.10, and not all seeds agree in sign. | +0.10, 4 of 5 seeds; −0.01 |
| H5 | SLr-late < SLr-late-shuf at near-OOD, Phase 1: the mean diff is negative. | −0.19; time-averaged L22, 4/5 |
| H6 | SL > SL-shuf at far-OOD after Phase 2: the mean diff is positive. | +0.046; time-averaged L25, 5/5 |
| H7 | After Phase 2, every schedule's in-dist is within 0.01 of R. | holds |
| H8 | Far-OOD ≤ 0.15 for every sinusoidal schedule and phase (seed mean). | holds |

## NoPE

These are exploratory, flagged by 5/5 seed sign agreement among 16 twin cells.

| # | Prediction | Seeds 0–4 evidence |
|---|---|---|
| H9 | NoPE SLr-mid > SLr-mid-shuf at near-OOD and far-OOD, Phase 1. | +0.05 each, 5/5 |
| H10 | NoPE LS > SL-shuf at near-OOD and far-OOD after Phase 2. | +0.06 and +0.18, 5/5 |
| H11 | NoPE SL and LS in-dist at the end of Phase 1 ≥ 0.85 seed mean. | 0.98 and 0.92 |
