# Retrain comparison — does more training data = better predictions?

_Experiment run 2026-06-30. Question: the Redshift `soh_etl.jun_26_featEngg_results_*` tables hold many more
vehicles than our local cohorts. Does retraining on them improve forecast accuracy?_

## Setup
- **Source:** `data/redshift/{bajaj,euler}_featengg.parquet` (pulled from Redshift `turno-master.soh_etl`
  over VPN). Same engineered features as our local tables (only `ymd`↔`month` differs).
- **Cohort sizes:** Bajaj **58 → 1,025** vins (our 58 are a clean subset). Euler **124 → 729** (+623 new,
  18 local-only kept aside). Mahindra: ~~only 3 in Redshift → not tested~~ **CORRECTION 2026-06-30: the
  store actually has 233 vins** (our local copy was stale at 3); re-pulled, now testable — re-run when convenient.
- **Fair before/after:** held-out test = NEW vehicles (in the expanded set, NOT in the old cohort, so the
  small model never trained on them). Model **A** trained on the OLD/small cohort, model **B** on the
  EXPANDED cohort; **both** scored on the **same** held-out vehicles. Metric = free-run forecast MAE (pp):
  anchor at the first half of each vehicle's history, forecast the rest, compare P50 to actual. 3 random
  test splits.

## Result

| OEM | model | trained on | held-out forecast MAE (3 seeds, pp) | mean |
|---|---|---|---|---|
| **Bajaj** | A (old) | 58 | 1.098 / 1.105 / 1.071 | **1.092** |
| **Bajaj** | B (expanded) | ~973 | 1.311 / 1.276 / 1.239 | **1.275** (**+0.18 worse**) |
| **Euler** | A (old) | 106 | 1.190 / 1.218 / 1.614 | **1.341** |
| **Euler** | B (expanded) | ~729 | 1.182 / 1.186 / 1.564 | **1.311** (−0.03, ≈ noise) |

## Verdict: **more data did NOT improve predictions here**
- **Bajaj — consistently WORSE** with 17× more vehicles (+0.18 pp, all 3 seeds).
- **Euler — no meaningful change** (−0.03 pp, within seed noise).

**Why:** the new vehicles are **younger** (Bajaj median 13 vs 25 mo) and **less degraded** (4 vs 5 pp drop),
with the same degrader fraction — so they add *coverage* but little new *degradation signal*. Bajaj's
decline is slow and calendar-driven; the 58 well-aged vehicles already captured the pattern, and the flood
of young/flat vehicles slightly dilutes it (the model learns lower monthly loss → under-predicts on
degraders). Only ~5% are data-thin, so quality-gating won't fix it — it's a maturity/age issue, not noise.

## Implication
- **For forecasting accuracy:** keep the current (small-cohort) models. More data is not better *yet*.
- **For coverage / population stats:** the expanded tables are still valuable — they let us *score* ~1,025
  Bajaj / ~729 Euler vehicles and make the warranty/cycle analysis far more robust. As the young vehicles
  age (and accumulate real degradation), retraining should start to help — re-run this experiment then.

## Business metric: % at-risk (won't reach warranty at ≥EoL) vs the ~2% (98%-survive) prior

MAE is the wrong yardstick for warranty decisions — what matters is the **at-risk rate**. Scored the full
fleets to warranty with the realistic near-term-rate projection (Step-11 style), old vs expanded model:

| | 3-yr horizon | 5-yr horizon |
|---|---|---|
| **Bajaj** (EoL 70%) | OLD 68.5% → EXP **60.7%** | OLD 85.6% → EXP 98.0% |
| **Euler** (EoL 80%) | OLD 15.2% → EXP 15.1% | OLD 36.2% → EXP 35.7% |

- **More data is mixed, not a win:** Bajaj at its *binding* horizon (~3.6 yr, km-bound) is slightly *better*
  with more data (60.7% vs 68.5%, closer to 2%); at 5 yr it flips; Euler is a tie.
- **Neither model supports the 98%-survive prior** — both flag 15–98% at-risk, ≫ 2%. **But** this is a
  3–5-yr extrapolation and **no Bajaj has actually crossed 70% EoL yet** (observed min 71%, fleet ≤29 mo).
  The models project the steep *early-life* decline forward; real degradation **decelerates** (√t), so they
  almost certainly **over-predict** long-horizon failure. The at-risk % isn't trustworthy to act on yet.
- **Real levers (not more rows):** (a) a deceleration-aware long-horizon model, (b) vehicles old enough to
  *observe* warranty outcomes. (at-risk scripts: `scratchpad/atrisk_compare.py`, `atrisk_realistic.py`.)

## Artifacts (the "previous model" kept for comparison)
- `data/bajaj/features/feature_table.parquet.pre_redshift_bak` (58-vin table)
- `data/euler/features/feature_table.parquet.pre_redshift_bak` (124-vin table)
- `models/euler/latest.pkl.pre_redshift_bak` (previous persisted Euler model)
- Experiment script: `scratchpad/retrain_experiment.py`
