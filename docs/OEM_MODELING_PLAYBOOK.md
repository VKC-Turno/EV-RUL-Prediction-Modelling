# OEM Battery SoH & RUL Modeling Playbook

**Audience:** the modeling team onboarding new OEMs (Piaggio, Montra, JBM, … next).
**Purpose:** a single, reproducible reference for *how* we turn raw EV telemetry into a per-vehicle
**State-of-Health (SoH)** time series and a **degradation-forecasting / Remaining-Useful-Life (RUL)**
model — including every assumption, the data-selection rules, the SoH-method decision tree, the
feature engineering, the model architecture, and the **train / validation / test** protocol.

It is written so that adding a new OEM is a *configuration + audit* exercise, not a from-scratch
rebuild. The worked examples are **Mahindra**, **Euler** and **Bajaj**, which between them cover all
four SoH methods we support.

> Companion docs (read alongside this one):
> - [`PROCEDURE.md`](PROCEDURE.md) — the operational runbook (extract → SoH → features).
> - [`SOH_COLUMN_USABILITY.md`](SOH_COLUMN_USABILITY.md) — the per-column audit method & results.
> - [`OEM_DATA_REQUIREMENTS.md`](OEM_DATA_REQUIREMENTS.md) — what to ask a new OEM to stream.
> - [`../README.md`](../README.md) — repo layout & "add a new OEM" quick start.

---

## 0. TL;DR — the one-page mental model

1. **There is no single SoH method.** The method is *dictated by what each feed actually carries*.
   Run the [SoH decision tree](#3-soh-method-decision-tree) first; everything else follows from it.
2. **OEM ≠ table.** A vehicle can appear in *two* feeds: the OEM's **native feed** and the shared
   **intellicar** aggregator. Only intellicar carries pack **current** → only there is coulomb
   counting even attemptable. Native feeds fall back to BMS-capacity, reported SoH, or a distance proxy.
3. **SoH is always forced monotone non-increasing** — but *smooth first, envelope second*, and prefer
   **isotonic** over a hard cumulative-min so behaviour-driven dips don't become permanent fake fade.
4. **The target is noisy and the cohort is small.** We validate with **leave-one-vehicle-out (LOVO)**
   backtests, split into *degrading* vs *flat* vehicles, against **persistence** and **√t-trend** baselines.
5. **Usage does not cleanly predict degradation in this fleet.** Ah-throughput / DoD / C-rate *mislead*;
   the only robust (still weak) driver we found is **cumulative heat exposure**. Don't oversell usage features.
6. **Onboarding a new OEM = `src/config.py` entry + column audit + cohort manifest + rerun importers.**
   No per-OEM code forks.

---

## 1. Goal & scope

Build, per OEM:

- **A SoH target time series** — one SoH value per *(vehicle, month)*, physically plausible and
  monotone non-increasing, anchored to 100% at registration where possible.
- **A degradation-forecasting model** — given a vehicle's history + operating conditions, project SoH
  forward to the **80% end-of-first-life (EoFL)** threshold and to the **warranty horizon**, with
  **P10/P50/P90** uncertainty bands.
- **Decision products** — warranty-risk flags, RUL, and (for resale) a forecast to the in-house
  warranty window. Surfaced in the Streamlit dashboard (`dashboard/app.py`).

Vehicles are commercial electric **3-wheelers** (Mahindra Treo / Zor Grand, Euler HiLoad/HiRange,
Bajaj, …). They are **low-mileage, high-calendar-age** assets — which is *why* calendar/thermal aging
dominates mileage aging in every result below.

---

## 2. Data landscape (read before selecting anything)

### 2.1 Sources

All telemetry lives in `s3://oem-data-iot/battery-oem-data/parquet/`, credentials in `.env`.

```
parquet/<oem>/vehicle-data/        # per-OEM NATIVE feed: bajaj, euler, mahindra, montra, piaggio, jbm
parquet/intellicar/battery-data/   # shared telematics AGGREGATOR feed (multi-OEM) — has `current`!
parquet/intellicar/location-data/  # GPS for the intellicar feed
```

| Feed | Span | File density | `current`? | `voltage`? | reported SoH? | Best SoH method |
|---|---|---|---|---|---|---|
| **intellicar** (shared) | 2022→2026 | dense (~1–3k rows/file) | ✅ signed | ✅ | ❌ | **Coulomb counting** |
| **mahindra** (native) | 2024-11→2026-06 | tiny (~5 rows/file, ~70M files) | ❌ | ❌ | ❌ | Distance-per-SoC proxy + thermal features |
| **euler** (native) | 2022→2026 | medium (~1440 rows/day, 60 s) | ⚠️ *2023+ batch only* | ⚠️ *2023+ only* | ✅ (coarse) | **BMS remaining-capacity** |
| **bajaj** (native) | 2025-09→2026-06 | dense (~400 rows/file) | ❌ | ❌ | ✅ (clean) | **Reported SoH** |

### 2.2 Gotchas that bit us on real data (every one is load-bearing)

- **Partitioning is by *ingest* date, not *event* date.** A `day=01` partition can hold events from
  weeks later. **Never prune partitions by event time.**
- **Schemas differ per OEM *and per batch within an OEM*.** Euler's 2022 vehicles carry only a coarse
  reported `batterySoh` (voltage 100% NULL); the **2023+ batch** adds `batteryCurrent`,
  `batteryVoltage`, `batteryRemainingCapacity`, `batteryTemperature`, `cellImbalance`. **Re-audit per
  batch, not just per OEM.** `src/config.py`'s euler entry reflects the *2022* batch only.
- **intellicar is multi-OEM and sparsely populated per OEM.** Of its 84 columns, **only 9 are populated
  for Mahindra rows** (`vin, eventAt, make, model, soc, current, batteryVoltage, odometer, dte`); the
  temp/cycle/fault/motor columns are filled only for *other* OEMs. Audit `cols_by_oem` per OEM.
- **`current` is a reserved word in S3 Select** → must be quoted `s."current"`. S3 Select omits null
  fields → always `df.reindex(columns=ALL_COLS)` after parsing.
- **Sentinels masquerade as data.** Raw rows carry `SoC=79903`, `current=−22220 A`, `temp=1412 °C`,
  `remainingCapacity=101140`. These **pass naive `notnull()` filters** and corrupt the math — clip with
  **physical bounds** before any computation (see `BOUNDS` dicts in the feature builders).
- **Scale (Mahindra).** ~70M tiny Parquet files, no VIN index → a full per-VIN extract is infeasible.
  Use the **monthly-sample** strategy (1 representative day/month, per-day file cap). Dense feeds
  (intellicar, bajaj) tolerate more days/month — see `DAYS_PER_MONTH` / `FILES_PER_DAY_CAP` in config.
- **`kwh` (Mahindra) is NOT integrable** — it is signed *instantaneous* power, stale when parked, not
  cumulative. The energy-per-SoC method is therefore not viable on that feed.

---

## 3. SoH method decision tree

**Run this first for every new OEM.** Pick the *highest* tier the data supports; keep a lower tier as a
cross-validator where possible.

```
Does a feed for this vehicle carry signed pack CURRENT + SoC + sub-minute timestamps?
│
├─ YES (e.g. intellicar)  ──────────────►  TIER A: COULOMB COUNTING
│                                          C = Σ|∫I·dt| / Σ(|ΔSoC|/100), SoH = C / C_ref
│                                          ⚠ field-data caveat — see §3.A
│
├─ NO, but BMS reports REMAINING / FULL-CHARGE CAPACITY (Ah) + SoC?
│        (e.g. euler 2023+)  ───────────►  TIER B: BMS-CAPACITY SoH
│                                          full_cap = remCap / (SoC/100) at HIGH SoC, isotonic ↓
│
├─ NO, but BMS reports a CLEAN reported SoH (%)?
│        (e.g. bajaj)  ──────────────────►  TIER C: REPORTED SoH (cleaned)
│                                          median/month → smooth → monotone envelope
│
└─ NONE of the above, only ODOMETER + SoC?
         (e.g. mahindra native)  ────────►  TIER D: DISTANCE-PER-SoC PROXY (last resort)
                                            capacity ∝ Σ Δodo / Σ Δsoc on discharge; indicative only
```

After **any** tier, apply the **universal post-processing**:
1. **Clip sentinels** with physical bounds (per-signal).
2. **Aggregate to monthly** grain (require ≥15 obs/month, ≥4–6 months/vehicle).
3. **Smooth** (rolling median, 3–5 months) **before** enforcing monotonicity.
4. **Force monotone non-increasing.** Use **isotonic** decreasing fit when the raw signal has
   behaviour-driven dips (Tier A/B); a hard `cummin` is acceptable only for an already-clean signal
   (Tier C). *Never* let SoH rise.
5. **Anchor 100% at registration** where a reg date exists and predates telemetry (back-extrapolate the
   reference along the fade trend). For an absolute BMS value (Tier C) **do not** renormalize to 100% —
   aged vehicles legitimately enter the feed below 100%.

### 3.A Tier A — Coulomb counting (`src/soh.py`)

The gold standard *in principle*, and the method originally requested. **It is fragile on field data**
and required correction:

- **Sessionize** by gaps > `GAP_S = 300 s`. Within a session integrate trapezoidally:
  `dQ = Σ (Iₖ+Iₖ₋₁)/2 · Δt`.
- **Per-session capacity** `= |ΔQ| / (|ΔSoC|/100)`, kept only if `n ≥ 5`, `|ΔSoC| ≥ 2%`, capacity ∈
  `(40, 400) Ah`.
- **Monthly capacity = ΔSoC-weighted *pooled*** `= Σ|∫I·dt| / Σ(|ΔSoC|/100)` (NOT a per-session median).
  Large reliable swings dominate; the many tiny-ΔSoC noisy sessions barely move the ratio.
- **Robust baseline** = median capacity over **age 1–12 months** (skip the noisy settling month);
  `SMOOTH_WIN = 5`-month rolling median; cap loss at `MAX_DROP_PER_MONTH = 6 pp/mo`.
- **Why the correction mattered:** the earlier per-session-percentile approach produced an artifactual
  fast early drop ("80% in 1 year"). Root cause — ~98% of continuous sessions span only 2–10% SoC
  (~30% capacity noise) and the baseline anchored to a noisy first month. The pooled, age-1–12-mo-baseline
  version gives a fleet fade **median ≈ 2.0 %/yr** (was 3.8), 80/91 vehicles in a plausible 0–8 %/yr band.

> **Field-data warning:** on the **Euler 2023+** feed, gap-based sessions span whole charge+discharge
> cycles, so `|∫I·dt|` measures *net* charge against *net* ΔSoC → impossible capacities (44–345 Ah for a
> 133 Ah pack). Coulomb counting there needs sign/rest-based session splitting + CV/top-balancing
> exclusion before it is even usable as corroboration. **We do NOT use coulomb for Euler** — see Tier B.

### 3.B Tier B — BMS remaining-capacity SoH (`src/euler_features.py::bms_soh_monthly`)

Used for **Euler 2023+**. Validated as the method to trust via a 4-agent adversarial cross-validation
(`dashboard/crossval_workflow.js`).

- `full_cap = batteryRemainingCapacity / (SoC/100)` **only at high SoC (95–100%)**. `remCap` has a
  negative-intercept SoC bias (`remCap ≈ 1.551·SoC − 28.7`), so low-SoC rows read artificially low.
- **Adaptive plausibility window** around *this vehicle's* pack: keep `full_cap ∈ [0.6, 1.4] × median`
  (a HiLoad pack ≠ the ~133 Ah cohort, so don't hard-code bounds).
- Monthly median (≥15 obs/month, ≥6 months). `nominal` = 90th-pct of the first 6 months' full_cap.
- **Isotonic decreasing** fit over month index, `SoH = 100·fit/nominal`, clipped ≤ 100.
- **Isotonic, NOT `cummin`:** a hard cumulative-min ratchets behaviour-driven low-SoC dips into permanent
  fake degradation; isotonic finds the best monotone *fit*.
- Returns `None` (skip vehicle) if the pack's median full_cap is broken/zero — this is why only **80 of
  181** imported Euler vehicles yield a usable target.

### 3.C Tier C — Reported SoH, cleaned (`src/bajaj_features.py::reported_soh_monthly`)

Used for **Bajaj** (`essBmsSohcEstPercValue`). The reported value is **clean and well-behaved** (one
quantized value/month, declines smoothly with cycling/odometer) — so unlike Euler's coarse `batterySoh`
it needs only light cleaning:

- Clip to `[40, 100]` (anything <40 is garbage). Per-month median, require ≥15 obs/month, ≥4 months.
- **3-month rolling median → `np.minimum.accumulate`** (monotone). Smooth *before* the envelope so one
  noisy month can't carve a permanent cliff.
- **Do NOT renormalize to 100% at t0** — reported SoH is a BMS absolute; aged vehicles correctly start
  below 100%.

> **Euler's reported `batterySoh` is the counter-example:** coarse (~8 quantized levels), quasi-stuck,
> and off the 100 scale (~83% when near-new) → unusable as a fade signal. *Always validate that a
> reported SoH actually moves before trusting it.* "Reported" is clean for Bajaj, useless for Euler.

### 3.D Tier D — Distance-per-SoC proxy (`notebooks/01_soh_target/mahindra_soh_distance_proxy.ipynb`)

Last resort for a native feed with neither current nor a usable BMS value (Mahindra native). Capacity ∝
`Σ Δodometer / Σ Δsoc` over discharge segments; same smoothing + monotone envelope. **Treat as
indicative, not calibrated** — confounded by season/driving and tends to under-read degradation. Where a
vehicle also appears in intellicar, **cross-validate against coulomb SoH** (`notebooks/01_soh_target/mahindra_soh_method_compare.ipynb`;
agreed within ~3 pp on the shared vehicle).

---

## 4. Data selection (cohort, training, validation, testing)

### 4.1 Cohort selection — *which vehicles to model*

We do **not** model the whole fleet; we select an aged, well-observed cohort.

1. **Most-aged & recurring.** Rank vehicles by **odometer** (a robust proxy for age/usage) among those
   that **recur across partitions**. The earliest partition is unreliable (single-vehicle backfill dumps)
   — do not select "oldest" by first-seen partition alone.
2. **Dual-feed overlap (Mahindra).** Intersect intellicar-OEM VINs with native-feed VINs (~96 overlap)
   → richer features (intellicar current + native thermal/usage) **and** lets coulomb SoH cross-validate
   the proxy. Take top ~15 by odometer.
3. **Resale cohort (Euler).** Join the pricing/resold CSV (`data/Repo - Pricing - Sold.csv`, full-VIN
   match) to telemetry; keep resale-date groups with **≥3 well-observed vehicles** (≥12 months SoH before
   resale, data ending ≤6 months before resale).
4. **Clean-start filter for any km/age analysis.** A vehicle qualifies for distance-vs-degradation work
   only if its **first valid odometer < 3000 km AND SoH ≥ 99** at that point. *This is strict on purpose:*
   it yields **0 Mahindra vehicles** (odometer logged too late — see §7) and 48 Euler vehicles.

### 4.2 Aggregation grain

One row per **(vin, month)**. Monthly is the right grain: it averages out sub-daily noise, matches the
forecasting horizon (months-to-EoFL), and survives sparse feeds. Require **≥15 raw obs/month** for a
month to count and **≥4–6 months/vehicle** for the vehicle to enter the model.

### 4.3 Train / validation / test protocol — **Leave-One-Vehicle-Out (LOVO)**

This is the heart of "validation & testing" and it is **identical across OEMs** (`src/euler_backtest.py`;
mirror it for new OEMs).

The cohort is small (tens of vehicles) and **the unit of generalization is the *vehicle*, not the
month** — months within a vehicle are highly autocorrelated. A random row split would leak. So:

| Element | Rule | Constant |
|---|---|---|
| **Split unit** | Hold out **one whole vehicle at a time**; train on all *other* vehicles. | true LOVO, no leakage |
| **Within held-out vehicle** | Give the model the **first 60%** of months as history; **forecast the last 40%**. | `HOLDOUT = 0.40` |
| **Min history** | Need ≥4 months of history to forecast from; ≥`MIN_HIST+2` months total to be testable. | `MIN_HIST = 4` |
| **Cohort split for reporting** | Label a held-out tail **"degrading"** if it loses ≥2 pp, else **"flat"**; report RMSE/MAE **separately**. | `DEGRADE_TAIL_PP = 2.0` |
| **Baselines to beat** | **Persistence** (last SoH flat) and **√t-trend** (fit `a + b·√age`, slope clamped ≤0). | — |
| **Band check** | Fraction of actuals inside **P10–P90**; target ≈ **0.80** coverage. | `calibrate_band(...)` |

**Why split degrading vs flat?** A model that just predicts "flat" scores well on the (majority) flat
vehicles and hides its failure on the few that actually decline — which are the ones that *matter* for
warranty risk. We require beating persistence **on the degrading cohort specifically**.

**Reported result (Euler, LOVO K=6 degraders):** trajectory model RMSE **≈4.23** vs trend **4.92** vs
persistence **5.74** (≈26% better than persistence). Flat vehicles are routed to persistence-like
behaviour by the gate (below). Re-run `python src/euler_backtest.py` after any model or data change.

### 4.4 What "test" means in production

There is no held-out future yet (the fleet is still aging), so the **LOVO backtest *is* the test set**.
When forecasting a *live* vehicle we use **all** its history and project forward; the LOVO RMSE is the
honest error bar we attach to that projection. Re-backtest whenever the cohort grows.

---

## 5. Feature engineering

Built per *(vin, month)* by the per-OEM feature builders (`src/{features,euler_features,bajaj_features}.py`).
Features fall into three groups; the model consumes all three.

| Group | Features | Notes |
|---|---|---|
| **STATE** (where the vehicle is now) | `soh`, `age_months`, `cum_ah`, `cum_km`, `odo_max` | autoregressive anchors |
| **STRESS** (operating conditions) | `ah_throughput`, `cur_abs_mean/p95`, `cur_chg/dis_mean`, `soc_mean`, `frac_soc_high/low`, `volt_mean/min/max`, `temp_mean/max`, `dod_mean`, `crate_p95`, `km_month`, `imbalance_mean`, `dte_mean` | availability is OEM-dependent — see per-OEM tables |
| **CURVATURE** (fade shape) | `inv_sqrt_age` = 1/√(age+1), `soh_deficit` = 100−SoH | encode the √t-fade shape & the low-SoH acceleration knee; **the most important engineered features** |

**Design rules:**
- **Everything is NaN-tolerant** — gradient-boosted trees (XGBoost/LightGBM) handle NaN natively, so a
  feature absent on an older sub-cohort (e.g. `imbalance_mean`) doesn't force a row drop.
- **DoD** = mean per-discharge SoC drop over sessions with ≥3% drop (`MIN_DOD`).
- **C-rate** = `cur_abs_p95 / NOMINAL` (use the *per-vehicle* nominal pack Ah).
- **Use current *magnitude*** — `cur_dis_mean` is signed negative; the model wants `|·|`.
- **Climate features** (`lat_mean`, `lon_mean`, `wh_per_km`) are computed but **left out of the Mahindra
  model** — they overfit on small regional data. Re-add only when more diverse data justifies it.

---

## 6. Modeling details

Two complementary forecasters, both gradient-boosted, both kept API-compatible so the dashboard and
notebooks can call either. **Trajectory is the production forecaster; Rate is the legacy/diagnostic one.**

### 6.1 RATE model — predict monthly ΔSoH, roll forward (`*_model.py::build_transitions/train/free_run`)

- **Target:** forward monthly SoH loss (clipped ≥0), one row per vin-month transition, `gap ≤ 3 months`.
- **Sample weighting** `w = 1 + loss.clip(0,5)` so the zero-inflated plateau months don't drown out the
  rare real-decline months.
- **Glitch winsorization (Euler):** a single-step |ΔSoH| > `GLITCH_PP = 10 pp` is a BMS capacity-recal
  artifact (isotonic full_cap steps), **not** a month of wear. Winsorize its loss to `GLITCH_CAP = 4 pp/mo`
  (don't drop — keep a bounded large-loss signal).
- **Estimator (XGBoost):** `n_estimators=400, lr=0.04, max_depth=4, subsample=0.85, colsample=0.85,
  min_child_weight=4, reg_lambda=1.5, gamma=0.05`. A global bias (`_cal_bias`) shifts mean predicted loss
  to the observed fleet mean (the regularized target otherwise regresses toward 0).
- **Flat-vs-degrading gate** (`free_run`, keyed on the vehicle's recent observed loss rate `obs_rate`):
  - **flat** (`obs_rate < 0.04`): `step = 0.35·pred` — don't manufacture loss on flat vehicles.
  - **degrading** (`obs_rate ≥ 0.10`): `step = max(pred, obs_rate·1.25)` — never pull a real decliner down
    to a lower historical rate; allow late-life acceleration.
  - **between:** linear blend.
  This gate is the fix for both failure modes the user flagged ("too optimistic" on decliners, "manufactures
  loss" on flat vehicles).

### 6.2 TRAJECTORY model — predict cumulative loss vs horizon (production)

The better-validated forecaster. Instead of stepping monthly it learns **cumulative SoH lost between an
anchor month and a horizon `dage` months later**, conditioned on anchor state + recent stress.

- **Sample expansion** (`build_traj_samples`): every (anchor with ≥4 mo history, later month) pair →
  one row; target = SoH lost; key features `age0, soh0, deficit0, obs_rate, dage, sqrt_dage` + 10 pruned
  stress medians (`TRAJ_STRESS`).
- **Estimator:** **LightGBM quantile** `alpha=0.5` (`n_estimators=400, lr=0.03, num_leaves=15,
  min_child_samples=20, subsample=0.8, colsample=0.8, reg_lambda=2.0`); sklearn `GradientBoostingRegressor`
  fallback if LightGBM is absent.
- **Own-slope blend:** the pooled P50 is blended with a continuation of the vehicle's *own* recent slope,
  weighted up the more it's already degrading (`w_own = clip(obs_rate/0.4, 0, 0.7)`). Fixes the pooled
  model's regress-to-the-fleet-mean under-prediction on steep decliners.
- **Flat pin:** vehicles at ≈100% with no observed slope get the central path pinned to their (≈0) slope
  and a **half-width band** — no manufactured degradation.
- **P10/P90 bands:** an **empirical √-horizon envelope** around P50: `P10 = P50 − lo·√h`,
  `P90 = min(P50 + hi·√h, soh0)`, defaults `{lo: 1.56, hi: 1.18}` calibrated to ≈80% LOVO coverage.
  `calibrate_band()` recomputes `lo/hi` from fresh backtest residuals as the fleet grows. (lo>hi because
  actuals sit below P50 more often than above.) P10 = pessimistic (low SoH), P90 = optimistic.

### 6.3 From SoH forecast to decisions

- **EoFL / RUL:** project P50 to **80% SoH**; months-to-80% is the RUL. Bands give P10/P90 RUL.
- **Warranty risk:** free-run to each vehicle's warranty boundary (`config.warranty_for(oem, model)` →
  `(years, km)`, first limit hit). Status tiers (`RISK_MARGIN = 2.0`): proj SoH ≥80 **OK**, 78–80 **WATCH**,
  <78 **AT-RISK**. Always **split warranty risk by model/term** and flag low-confidence vehicles
  (<8 mo history, horizon > 2× current age) — see §7.
- **Second life (BESS):** fixed user-supplied cubic of cycles (in `build_dashboard.py`), EoSL at 20% / ~6,314
  cycles. Keep coefficients verbatim.

---

## 7. Insights from the data so far (what to expect, what to *not* oversell)

These are the substantive, sometimes counter-intuitive findings. They are the difference between a model
that looks plausible and one that is honest.

1. **Same kilometres ≠ same degradation.** With a trustworthy km axis and an exact 100%-SoH start, Euler
   vehicles driven the *same* long distance still differ **~9–15 pp** SoH (e.g. at 38,592 km one is 100%,
   another 90%). **Distance does not set degradation; operating conditions do.** Do not market "km driven"
   as a SoH predictor.

2. **Usage proxies mislead in this fleet.** `ah_throughput`, `crate_p95`, `cur_abs_p95` *top the XGBoost
   gain but point the wrong way* on heavy-vs-flat pairs, because the "flat" reference vehicles are **young,
   high-utilization** units that simply haven't aged into loss yet. Across 6 clean same-distance vehicles,
   **deeper DoD tracks *less* loss** (ρ=−0.66), Ah-throughput ρ≈0. The cherry-picked "deeper DoD → more
   loss" pair does **not** generalize.

3. **The one robust (still weak) driver is cumulative heat.** `hot_months_gt40` (count of months with
   `temp_max > 40 °C`) orders degradation with **ρ ≈ +0.88, leave-one-out stable** — but **per-month *mean*
   temperature does not** (only the integrated count does). n=6 → indicative, not proof. The defensible
   top-5 stress features are `temp_mean, dod_mean, |cur_dis_mean|, frac_soc_high, imbalance_mean` — *best-
   justified drivers, not proven causal levers.* No single stressor is a strong cross-vehicle discriminator
   (best |ρ|≈0.28).

4. **Mahindra odometer is logged too late for km-analysis.** Not one Mahindra vehicle has its odometer
   recorded from a fresh (~0 km, 100% SoH) state — the first non-null reading is always already tens of
   thousands of km at a degraded SoH (e.g. one VIN's first reading is 23,352 km at 78% SoH). So km-vs-
   degradation work is **Euler-only**.

5. **Warranty risk is term-dependent, not a single fleet number.** Free-running the model to each
   vehicle's warranty boundary: **Treo (3-yr) ~89% survive** above 80% SoH (the working "90% survive"
   assumption holds), but **Zor Grand (5-yr) only ~21% survive** — the 5-year term is where the liability
   sits. **km is never the binding limit (0/95)** — low utilization means calendar+condition aging drives
   risk. Report warranty risk **split by model/term**, and exclude/flag low-confidence vehicles first
   (only 14 of 21 at-risk hits were reliable).

6. **Reported SoH quality is OEM-specific.** Clean and usable for Bajaj; coarse/stuck and *useless* for
   Euler. Validate that the reported series actually moves before adopting Tier C.

7. **Coulomb counting is treacherous on field data** (gap-sessions span whole cycles). It works for
   intellicar *with* the ΔSoC-weighted-pooled correction, and is **broken** as-is for Euler 2023+. When in
   doubt, prefer a BMS-capacity or reported target and use coulomb only as a corroborator.

---

## 8. Per-OEM playbooks

### 8.1 Mahindra (Treo / Zor Grand) — **dual-feed, coulomb + proxy**

| | |
|---|---|
| **Vehicles modeled** | 95 (feature table: 1,309 vin-months, median 14 mo/veh, 66 degraders >2 pp) |
| **SoH method** | **Coulomb** (Tier A) on the intellicar overlap (has current); **distance-per-SoC** (Tier D) on the native feed as cross-check |
| **Data selection** | dual-feed overlap (~96 VINs), top ~15 by odometer per cohort; native feed adds thermal/usage/location |
| **Target builder** | `src/soh.py` (coulomb, ΔSoC-weighted pooled, reg-anchored) |
| **Features** | `src/features.py` (electrical) + native-feed thermal/usage; model uses `src/model.py` STATE+STRESS+CURV. Climate features computed but excluded (overfit) |
| **Model** | LightGBM quantile rate model (q10/q50/q90), `n_estimators=500, lr=0.03, num_leaves=15` |
| **Warranty** | Treo/Treo+ 3 yr·120k; Zor Grand 5 yr·120k; Treo Zor 3 yr·80k — `config.WARRANTY["mahindra"]` |
| **Key caveats** | odometer logged too late (no clean km-start); `kwh` not integrable; native feed is ~70M tiny files → monthly sample only |

### 8.2 Euler (HiLoad / HiRange) — **batch-dependent, BMS-capacity**

| | |
|---|---|
| **Vehicles modeled** | 80 with usable SoH (181 dense imported; 101 skipped for broken/zero remaining-capacity) — 1,349 vin-months, median 15 mo/veh, 54 degraders |
| **SoH method** | **BMS remaining-capacity** (Tier B), high-SoC band, isotonic decreasing — *validated as the method to trust over coulomb & reported* |
| **Schema note** | **2023+ batch** has current/voltage/remaining-capacity/temperature/cellImbalance; **2022 batch** has only coarse reported `batterySoh` → reported path + univariate forecast |
| **Data selection** | dense cohort imported per resale group (≥3 well-observed vehicles); reg dates in `data/euler/Euler_Regd_Details.csv` |
| **Target builder** | `src/euler_features.py::bms_soh_monthly` |
| **Model** | `src/euler_model.py` — Rate (XGBoost) + **Trajectory (LightGBM quantile, P10/P50/P90)**; validated by `src/euler_backtest.py` |
| **Warranty** | HiRange 6 yr·150k; else 5 yr·125k — `config.WARRANTY["euler"]` |
| **Key caveats** | coulomb is **broken** on this feed; reported SoH unusable as a fade signal; `cellImbalance` only ~66% non-null (NaN on 2022 cohort) |

### 8.3 Bajaj — **reported-SoH path (new-OEM worked example)**

| | |
|---|---|
| **Status** | Scaffolded: 16 dense vehicles imported; feature builder written (`src/bajaj_features.py`); **feature table not yet built** — the canonical "new OEM mid-onboarding" state |
| **SoH method** | **Reported SoH** (Tier C) — `essBmsSohcEstPercValue`, clean & monotone; coulomb & BMS-capacity both impossible (no current/voltage/remaining-capacity) |
| **Schema note** | Bajaj-native verbose names (`essBms*/etsVcu*/hmiIcl*`); odometer in **metres** (`hmiIclOdoActMValue`/1000); span ~10 months (2025-09→2026-06) |
| **Target builder** | `src/bajaj_features.py::reported_soh_monthly` (median/month → 3-mo rolling median → `cummin`; **not** renormalized to 100%) |
| **Features** | charge-cycle count, SoC dwell, pack & ambient temp, drive-efficiency (Wh/km), odometer-km |
| **Model** | reuse the shared rate/trajectory pattern once the feature table exists |
| **Warranty** | default 3 yr·120k (no spec sheet; inferred from pricing CSV reg→warranty_end ~3.2 yr median) — refine if a spec is found |
| **Next step** | run `python src/bajaj_features.py`, confirm reported SoH actually declines, then LOVO-backtest as in §4.3 |

---

## 9. Onboarding a new OEM — step-by-step checklist

1. **Audit columns** in *both* the OEM's native feed and (if present) its intellicar rows. Fill the
   `SOH_COLUMN_USABILITY.md` template. **Check per batch**, not just per OEM.
2. **Run the [SoH decision tree](#3-soh-method-decision-tree)** → pick the target method. Keep a lower
   tier as a cross-validator if available.
3. **Add a `src/config.py` entry:** `OEM_FEEDS["<oem>"]` (prefix, useful `cols`, `has_current`,
   `has_reported_soh`, dense-or-not, reserved cols, notes); `INTELLICAR["cols_by_oem"]["<oem>"]` if it
   appears there; `WARRANTY["<oem>"]`; `SOH_METHOD["<oem>"]`; sampling cadence if non-default.
4. **Build the cohort manifest** (`data/manifests/<oem>_cohort.csv`): most-aged & recurring; add the
   overlap set if dual-feed.
5. **Extract:** `python src/import_cohort.py intellicar <oem>` and `python src/import_cohort.py <oem> <oem>`
   (or a dense importer like `import_euler_dense.py` / `import_bajaj_dense.py`). Resumable; clip sentinels.
6. **Build the SoH target + feature table:** copy the closest `*_features.py` (Euler for Tier B, Bajaj for
   Tier C, Mahindra/`soh.py` for Tier A/D). **Define physical `BOUNDS` for that OEM's signals.**
7. **Confirm the SoH series actually declines** and is monotone. If a "reported" SoH is stuck/coarse →
   fall to a lower tier.
8. **Train + LOVO-backtest** (`§4.3`, mirror `euler_backtest.py`). Require beating persistence **on the
   degrading cohort**. Calibrate the P10/P90 band.
9. **Wire into the dashboard** (`dashboard/build_dashboard.py` data layer → `app.py`). ⚠ restart Streamlit
   after regenerating data — `@st.cache_data` does not invalidate on code change.
10. **Write down new assumptions & caveats** in the register below.

---

## 10. Assumptions register

| # | Assumption | Status / evidence |
|---|---|---|
| A1 | SoH is monotone non-increasing | **Enforced** (isotonic / cummin). A design choice, not measured — real SoH can wiggle; we deliberately suppress upward moves. |
| A2 | 100% SoH at registration | Anchored where reg date predates telemetry & method allows; **not** applied to Tier C (absolute reported SoH). |
| A3 | Monthly grain, recent-stress persists in forecast | Recent-6-month median stress is held constant in roll-forward. Reasonable for stable duty cycles; revisit for seasonal fleets. |
| A4 | Coulomb counting is feasible where current exists | **True only with the ΔSoC-weighted-pooled correction**; broken on Euler 2023+ gap-sessions. |
| A5 | Reported SoH is trustworthy | **OEM-specific** — true for Bajaj, false for Euler. Verify it moves. |
| A6 | Usage (Ah/DoD/C-rate) predicts degradation | **Largely false in this fleet** — proxies mislead; only cumulative heat is a (weak) robust driver. |
| A7 | ~90% of the fleet survives warranty above 80% SoH | **Term-dependent**: holds for Treo 3-yr (~89%), fails for Zor Grand 5-yr (~21%). |
| A8 | km warranty limit can bind | **Never observed** (0/95) — low-mileage fleet, calendar/thermal aging dominates. |
| A9 | LOVO RMSE is the production error bar | Accepted — no held-out future exists yet; re-backtest as the cohort grows. |
| A10 | Sentinels are absent after `notnull()` | **False** — must clip with physical bounds (`SoC=79903`, `current=−22220`, etc.). |

---

## 11. Pitfalls checklist (paste into every new-OEM PR)

- [ ] Clipped sentinels with **physical bounds** before any math?
- [ ] Quoted `s."current"` and `reindex`-ed columns after S3 Select?
- [ ] Did **not** prune partitions by event time?
- [ ] Smoothed **before** the monotone envelope (no single-month fake cliff)?
- [ ] Used **isotonic** (not hard `cummin`) where the raw signal has behaviour-driven dips?
- [ ] Verified the chosen "reported" SoH actually **declines** (not stuck/coarse)?
- [ ] Split LOVO results **degrading vs flat**; beats **persistence on the degrading cohort**?
- [ ] Forecast **does not manufacture loss** on flat vehicles (gate / flat-pin active)?
- [ ] Warranty risk reported **per model/term**, low-confidence vehicles flagged?
- [ ] Re-ran the backtest and **restarted Streamlit** after regenerating data?

---

*Maintained by the battery analytics team. When a finding here changes, update the assumptions register
(§10) and the relevant per-OEM playbook (§8), and re-run the LOVO backtest.*
