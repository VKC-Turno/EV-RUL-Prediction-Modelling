# Multi-OEM Battery State-of-Health & Remaining-Useful-Life Program
## End-to-End Technical Report

**Program scope:** Euler, Mahindra, Bajaj, Piaggio electric 3-wheeler fleets (Montra pending)
**Report date:** 2026-07-02
**Data snapshot:** Redshift feature-store exports + local coulomb/native cohorts, saturated Mahindra two-feed scan (2026-06-30), behaviour model re-run (2026-07-02)
**Status:** Prepared for battery-product leadership / external technical review

> **Note on numbers.** Every headline figure in this report is grounded in a specific source file (`src/*`, `data/*`, or a `memory:` finding). Where a live data artifact has been recomputed since the underlying research briefs were written, this report uses the **live artifact value as authoritative** and flags the divergence. Findings are explicitly marked **[VERIFIED]** or **[ASSUMPTION]**.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Fleet size & coverage](#2-fleet-size--coverage)
3. [Data infrastructure & sources](#3-data-infrastructure--sources)
4. [Data quality, frequency & reliability (per OEM)](#4-data-quality-frequency--reliability-per-oem)
5. [SoH availability & method per OEM](#5-soh-availability--method-per-oem)
6. [SoH calculation (methodology, envelope, chemistry)](#6-soh-calculation-methodology-envelope-chemistry)
7. [SoH validation (all experiments, quantitative)](#7-soh-validation-all-experiments-quantitative)
8. [Vehicle selection for modeling](#8-vehicle-selection-for-modeling)
9. [Criteria for training vs inference](#9-criteria-for-training-vs-inference)
10. [Fleet warranty risk per OEM (from inference)](#10-fleet-warranty-risk-per-oem-from-inference)
11. [Confidence intervals & caveats](#11-confidence-intervals--caveats)
12. [Appendix A — Experiment log](#12-appendix-a--experiment-log)
13. [Appendix B — Assumptions register](#13-appendix-b--assumptions-register)

---

## 1. Executive summary

The program estimates battery **State-of-Health (SoH)** and forecasts **Remaining-Useful-Life (RUL)** across four OEM 3-wheeler fleets that expose radically different telemetry. The central engineering problem is that **no single SoH method works across OEMs**: signed current (required for coulomb counting) is available only through the Intellicar aggregator feed, and only for some OEMs. This forces a **tiered SoH methodology**, an honest accounting of what is *measured* versus *estimated*, and wide, defensible uncertainty bands where measurement is impossible.

**Key verified findings:**

- **Four-tier SoH regime** driven by data availability (`src/config.py`, `src/soh.py`, `src/euler_features.py`, `src/bajaj_features.py`): Coulomb counting (Mahindra-Intellicar, Piaggio); BMS remaining-capacity (Euler 2023+); reported-SoH field (Bajaj); age-based fleet prior for the ~98% of Mahindra vehicles that are native-only. **[VERIFIED]**
- **Chemistry is universally LFP** across all four OEMs (owner-confirmed, overriding "undisclosed"/"li-ion" official specs; `memory: oem-spec-sheet`). LFP's flat OCV–SoC plateau is the root cause of the fleet's measurement artifacts and of the failure of every voltage- or usage-based proxy. **[VERIFIED]**
- **SoH measurement artifacts corrupt the end-of-life training signal.** 12 of 14 fully-aged vehicles carry cliff / stuck-floor / iso-floor artifacts (`src/soh_audit.py`; `memory: soh-artifact-audit`). This biases long-horizon "at-risk %" upward. **[VERIFIED]**
- **Native-only Mahindra SoH is not measurable.** Distance-per-SoC, distanceToEmpty, charge-energy (kWh) and charge-rate proxies all fail to correlate with coulomb ground truth (km/%SoC r=+0.06, %SoC/hr r=−0.31, both within the *same* feed). The dominant confounder is unobservable cargo **payload** (`memory: mahindra-native-signals-exhausted`). **[VERIFIED]**
- **A Bayesian behaviour-conditioned model** (`src/bayes_degradation.py`, hierarchical Gibbs sampler) pools all four OEMs (live: **540 vehicles, 9,175 vin-month observations**) to produce native-fleet SoH curves. Only `km_month` is a credible behaviour driver — now **consistently signed across 3 of 4 OEMs (Bajaj −0.72, Mahindra-IC −0.24, Piaggio −0.22; Euler ~0)** after an adversarial-verification pass corrected a corrupted km/month definition. Between-vehicle heterogeneity (σ_b≈0.25 SoH/mo) dominates parameter uncertainty (σ≈0.023) by ~10×, and behaviour beats the OEM-average baseline by ~5.7% in held-out MAE — modest, not transformative. **[VERIFIED]**
- **Warranty survival is term-dependent, not a fleet average.** On the warranty-projection cohort (`data/mahindra/soh/warranty_risk.csv`, n=95 — a **mixed** file: Treo 79 + Zor Grand 14 + 2 Piaggio Ape, not pure Mahindra): 3-year terms show **88.6% survival**, 5-year terms **25.0%** (n=16; Zor-Grand-only **21.4%**, n=14). **Zero** of 95 breaches are mileage-driven — calendar/thermal aging binds in these low-utilization fleets. **[VERIFIED]**

**Central caveat:** Several warranty terms are unverified (notably Euler HiLoad, downgraded 5yr→3yr provisionally), the 80% end-of-first-life threshold is unvalidated in the field, and Bajaj/Piaggio fleets are too young to have crossed a warranty boundary. External figures should be reported **split by warranty term** and flagged as provisional pending official certificates.

---

## 2. Fleet size & coverage

Source: `data/redshift/fleet_coverage.csv` (registration population, updated 2026-05-22) and the four `data/redshift/*_featengg.parquet` feature stores (VIN counts verified live).

| OEM | Registered fleet | Modeled (feature store) | Coverage % | Avg age (mo) | Feed / SoH tier |
|-----|-----:|-----:|-----:|-----:|-----|
| **Mahindra** | 10,933 | 233 | 2.1% | 12 | Intellicar coulomb (Tier A) + native prior (Tier D) |
| **Euler** | 2,125 | 729 | 34.3% | 20 | BMS remaining-capacity (Tier B) |
| **Bajaj** | 1,781 | 1,025 | 57.6% | 10 | Reported SoH (Tier C) |
| **Piaggio** | 1,913 | 247 | (12.9%) | 24 | Intellicar coulomb (Tier A) |
| **Montra** | 481 | — | — | 17 | Not yet audited |
| **TOTAL** | **17,233** | **2,234** | **~13%** | — | Four OEMs production-instrumented |

**Verified VIN / row counts (live feature stores):**

| Feature store | Rows (vin-months) | Unique VINs |
|-----|-----:|-----:|
| `bajaj_featengg.parquet` | 9,137 | 1,025 |
| `euler_featengg.parquet` | 8,311 | 729 |
| `mahindra_featengg.parquet` | 2,194 | 233 |
| `piaggio_featengg.parquet` | 4,320 | 247 |

**Notes and reconciliations:**

- The **2,234 modeled VINs** (1,025 + 729 + 233 + 247) match the row count of the data-quality manifest exactly (`data/manifests/vehicle_data_quality.csv`). **[VERIFIED]**
- **Coverage % is misleading as a maturity metric.** Cohorts are deliberately *oldest-N* sample pulls, not random registration draws. The true fleet-age population is far larger and younger; the modeled cohort is over-aged relative to the commercial fleet (`memory: cross-oem-transfer`). Mahindra is structurally under-represented (2.1%) because ~98% of its fleet is native-only with no coulomb feed. **[VERIFIED]**
- Piaggio coverage (247/1,913 ≈ 12.9%) was blank in `fleet_coverage.csv` at the 2026-05-22 snapshot (onboarded later); the feature store confirms 247 VINs. **[VERIFIED]**
- Distinct *modeling cohorts* (smaller, curated subsets used for specific forecaster experiments) appear in the modeling section: Euler 80, Mahindra 96, Bajaj 58, Piaggio 247 (`memory: bajaj-model`, `research_modeling`; an earlier Piaggio cohort manifest counted 256 — the live feature store is 247). These are training-experiment cohorts, not the full modeled fleet above.
- **Mahindra count key** (five figures recur and denote different things): **233** = Redshift feature-store VINs; **225** = both-feeds manifest; **~108** = coulomb-SoH-computable with sufficient ΔSoC; **96** = downloaded/curated modeling cohort; **95** = warranty-projectable rows. **[VERIFIED]**

---

## 3. Data infrastructure & sources

**S3 root:** `s3://oem-data-iot/battery-oem-data/parquet/` (credentials + `S3_BUCKET`/`S3_PREFIX` in `.env`). Partitions are **by ingest date, not event date** — event-time pruning does not work across partitions; sampling is done at the partition level (`memory: dataset-landscape`). **[VERIFIED]**

### 3.1 Intellicar (multi-OEM aggregator — the only coulomb-capable feed)

- Prefix `intellicar/battery-data/`, partitioned `year=/month=/day=` (2022–2026; the `year=0000` partition is junk and must be skipped).
- **Only feed with signed `current` (A)** plus `batteryVoltage` at sub-second cadence → the sole enabler of true coulomb counting (∫I·dt). `current` is a SQL reserved word in S3 Select and must be quoted `s."current"`.
- Dense files (~1–3k rows/file) — cheap to scan.
- OEM mix: Piaggio (~60%), Mahindra (~15%), Etrio/Omega (remainder). **Piaggio and Mahindra are coulomb-measurable here; Euler is not present in Intellicar.**
- For Mahindra, only **9 of 84 columns** are populated (`vin, eventAt, make, model, soc, current, batteryVoltage, odometer, dte`). All temp/fault/cycle/motor fields are empty (`memory: intellicar-table`). **[VERIFIED]**

### 3.2 Native per-OEM feeds

| OEM | Prefix | Span | File structure | Key columns | Electrical? |
|-----|-----|-----|-----|-----|-----|
| **Mahindra** | `mahindra/vehicle-data/` | Jan 2025 – Jun 2026 | **~5 rows/file, ~70M tiny files**, 50k–380k files/day | `soc, odometer, distanceToEmpty, batteryTemp, kwh, state, vehicleModel, lat/lon, gearPosition` | **No current/voltage/SoH** |
| **Euler** | `euler/vehicle-data/` | Oct 2023 – Jun 2026 | manifest-based dense | `batterySoc, batterySoh, batteryRemainingCapacity (2023+), batteryCurrent, batteryVoltage, batteryTemperature, cellImbalance, odometer` | Current/voltage present (dense import) |
| **Bajaj** | `bajaj/vehicle-data/` | Sep 2025 – Jun 2026 | ~400 rows/file, ~2k files/day | `essBmsSohcEstPercValue (reported SoH), essBmsSocEstPercValue, essBmsChgcycleActCountValue, essBmsTemperatureActDegcValue, etsVcuAmbienttempActDegcValue, etsVcuDriveeffEstWhpkmValue, hmiIclOdoActMValue (odo in metres)` | **No current/voltage/capacity** |
| **Piaggio** | `piaggio/vehicle-data/` | 2023 – 2026 | dense | `eventat (lowercase), vin (MBX), soc, odometer, batteryDischargeCurrent (often null), distanceTillEmpty, controllerTemperature, motorTemperature, speed, driveMode` | Native `batteryVoltage` 100% NULL → SoH built from Intellicar |

The **Mahindra 70M-tiny-files problem** makes per-VIN full-history retrieval via S3 Select infeasible; the feed is imported via a **monthly-sample strategy** (`src/import_cohort.py mahindra feed`, sampling ~15 days/month). Intellicar and other dense feeds sample 3 days/month `[8,16,24]` (`src/config.py DAYS_PER_MONTH`). **[VERIFIED]**

### 3.3 Redshift ETL & feature store

Pipeline: S3 → Redshift daily ingest → feature engineering (usage, thermal, SoC-dwell, SoH per tier) → `soh_etl.<oem>_featengg_results` tables → export to local `data/redshift/*_featengg.parquet` for modeling (`research_fleet-infra §9`). The Mahindra feature-store table was stale (3 rows) until a **2026-06-30 rescan surfaced 233 distinct VINs, 2,194 vin-months** (matches the both-feeds cohort). **[VERIFIED]**

### 3.4 OEM registry (`src/config.py`)

Single place to onboard an OEM: `OEM_FEEDS` (schema/columns per source), `INTELLICAR` (aggregator), `WARRANTY` + `FLEET_WARRANTY` + `warranty_for()`, `SOH_METHOD` (per-source routing: intellicar→coulomb, mahindra→distance_per_soc, euler→bms_capacity, bajaj→reported, piaggio→coulomb), and `DAYS_PER_MONTH`. **[VERIFIED]**

---

## 4. Data quality, frequency & reliability (per OEM)

### 4.1 Trainability gate (`src/build_data_quality.py` → `data/manifests/vehicle_data_quality.csv`)

**Rule:** Trainable = ≥6 valid SoH months AND (≥9-month span OR ≥5pp drop OR reached EoL). Bajaj has no span bar (feed is ~9 months for all vehicles). A **proven-degrader override** keeps thin vehicles that reached EoL or dropped ≥5pp. Degrader = SoH drop ≥2.0pp (`DEG=2.0`).

**Per-OEM thresholds:** Euler/Mahindra/Piaggio: min 6 months, ≥9-mo span, EoL 80%, exempt-drop 5pp. Bajaj: min 6 months, no span bar, EoL 70%, exempt-drop 5pp. **[VERIFIED]**

**Live manifest counts (2,234 rows; authoritative — supersedes the "1,234/910" subset quoted in the data-quality brief):**

| OEM | Total | Trainable | Thin | Thin % | Degrader class | Flat class |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| **Bajaj** | 1,025 | 976 | 49 | 4.8% | 948 | 77 |
| **Euler** | 729 | 592 | 137 | 18.8% | 397 | 332 |
| **Mahindra** | 233 | 134 | 99 | 42.5% | 96 | 137 |
| **Piaggio** | 247 | 207 | 40 | 16.2% | 127 | 120 |
| **TOTAL** | **2,234** | **1,909** | **325** | **14.5%** | 1,568 | 666 |

Mahindra has the highest thin rate (42.5%) because of late native onboarding (Jan 2025) and sparse monthly sampling relative to the coulomb-integration window; 74 of 98 newly-imported coulomb VINs (2026-06-25) are thin. **The data-quality gate is the real training lever**: dropping thin vehicles gives the best MAE on all OEMs (`src/quality_filter_check.py`); quality-gated ALL > degrader-only (`memory: flat-vehicles-data-quality`). **[VERIFIED]**

### 4.2 Measurement-artifact audit (`src/soh_audit.py`)

Three detectors flag corrupted SoH series: **CLIFF** (single-month drop ≥6pp), **STUCK_FLOOR** (≥5 identical minimum values), **ISO_FLOOR** (isotonic envelope pinned ≥2pp below raw). Fleet-wide prevalence:

| Artifact | Euler (124) | Mahindra (93) | Bajaj (58) | Piaggio (239) |
|-----|-----:|-----:|-----:|-----:|
| CLIFF | 22 (17.7%) | 9 (9.7%) | 1 (1.7%) | 14 (5.9%) |
| STUCK_FLOOR | 21 (16.9%) | 15 (16.1%) | 1 (1.7%) | 40 (16.7%) |
| ISO_FLOOR | 0 | 18 (19.4%) | 0 | 61 (25.5%) |
| **Any taint** | 38 (30.6%) | 32 (34.4%) | 1 (1.7%) | 77 (32.2%) |

**Completely-aged (EoL-reaching) contamination — the critical finding**: Euler 7/9 aged tainted (77.8%), Mahindra 5/5 (100%), Piaggio 10/10 (100%), Bajaj 0/0 (no aged vehicles). Aggregated across Euler+Mahindra this is the "**12 of 14 aged vehicles tainted**" headline. Because aged vehicles *are* the EoL learning signal, this biases long-horizon at-risk forecasts upward (60–98% vs a ~2% true baseline). **[VERIFIED]** (`memory: soh-artifact-audit`)

**Root cause [VERIFIED]:** LFP flat OCV–SoC plateau (3.0–3.4V over 10–90% SoC). The BMS holds a stale mid-pack value through the plateau and recalibrates in discrete steps at the steep ends → the textbook stuck-floor → cliff → iso-floor pattern. Bajaj is nearly clean (1.7%) because its reported field appears to be a high-fidelity on-device coulomb counter, not an envelope over noisy capacity estimates.

### 4.3 Reliability flakes & schema evolution

| OEM | Signal | Symptom | Mitigation |
|-----|-----|-----|-----|
| Mahindra | odometer | Logged too late in startup; **0 clean vehicles** with dense odometer | km-degradation analysis impossible → Euler-only (`memory: degradation-vs-km-finding`) |
| Mahindra | `kwh` charge-energy | Firmware unit drift (~8 vs ~700 scale); only ~22/96 plausible | Charge-energy SoH dropped |
| Mahindra | `batteryTemp` | 44% coverage, range −50…1412 (nonsense) | Stress feature only |
| Mahindra | `vehicleStatus` | Added Jan 2025, ~20% pre-2025 coverage | Drive/charge/idle flag only |
| Euler/Mahindra | `soc` | >100 garbage values (up to 866 in Intellicar) | Clamp to [0,100] |
| Euler | reported `batterySoh` | Coarse (~8 quantized levels, stuck ~83%), 0.0 garbage | Rejected → use BMS-capacity |

Intellicar timestamps burst on a single millisecond → feature engineering must deduplicate on (vin, timestamp) and resample to a fixed ~2-minute cadence. **[VERIFIED]** (`research_data-quality §3–4`, `memory: mahindra-native-signals-exhausted`)

---

## 5. SoH availability & method per OEM

The **SoH method decision tree** (`src/config.py SOH_METHOD`):

```
Signed current in Intellicar?  ── YES ─▶ Tier A: Coulomb counting (src/soh.py)
                                         Mahindra-IC (~108–225), Piaggio (~247)
   │ NO
   ▼
batteryRemainingCapacity (Euler 2023+)? ── YES ─▶ Tier B: BMS remaining-capacity (src/euler_features.py)
   │ NO
   ▼
Clean reported SoH (Bajaj)? ── YES ─▶ Tier C: Reported SoH (src/bajaj_features.py)
   │ NO
   ▼
Tier D: Age-based fleet prior (NOT per-vehicle SoH)
   Mahindra native-only (~10,700), Euler 2022 reported-only (8)
```

| OEM | Tier | Method | Feed | SoH range observed | Vehicles (cohort) |
|-----|:--:|-----|-----|-----|-----:|
| Euler (2023+) | B | BMS remaining-capacity, isotonic | dense native | 100→76% | 16 dense / 729 store |
| Euler (2022) | C→D | Reported (coarse), no renorm | dense native | 100→83% | 8 |
| Mahindra (both-feeds) | A | Coulomb (∫I·dt), isotonic | Intellicar | 100→64% | ~108–225 |
| Mahindra (native-only) | D | Age prior + Bayesian behaviour tilt | native | fleet curve ±11pp | ~10,700 |
| Bajaj | C | Reported `essBmsSohcEstPercValue` | native | 100→78% | 1,025 |
| Piaggio | A | Coulomb (Intellicar), isotonic | Intellicar | 100→~75% | 247 |

**Why each OEM lands where it does [VERIFIED]:**
- **Euler:** Has current in the dense import, but coulomb recompute was rejected (§7.6). 2023+ carries `batteryRemainingCapacity` → BMS-capacity is the cleanest available method. 2022 batch lacks it → reported-only.
- **Bajaj:** No current, no voltage, no remaining-capacity. Only the clean reported SoH field is usable.
- **Mahindra native:** No current/voltage/SoH → no classical method works; native proxies fail validation → age prior only.

---

## 6. SoH calculation (methodology, envelope, chemistry)

### 6.1 Chemistry (`memory: oem-spec-sheet`, `src/build_oem_specs.py`)

**[VERIFIED]** All four OEMs use **LFP**, owner-confirmed 2026-06-30, overriding official specs ("lithium-ion"/"li-ion"/undisclosed). Consequences: (1) segment coulomb counting is the correct Tier-A method because OCV–SoC lookup is nearly useless over the flat plateau; (2) reported-SoH artifacts arise from BMS recalibration steps at the plateau ends; (3) usage/range proxies fail.

### 6.2 Tier A — Coulomb counting (`src/soh.py`)

**Algorithm (ΔSoC-weighted pooled; 2026-06 corrected):**
1. **Session segmentation** by VIN: split on time gap >300s or zero/negative dt; require ≥5 samples and ≥2% ΔSoC.
2. **Per-session capacity:** `Ah = |∫I·dt|` (trapezoidal); `cap_sess = |Ah|/|ΔSoC| × 100`, clipped to [40,400] Ah.
3. **Monthly pooled capacity:** `capacity_ah = Σ|∫I·dt| / Σ(|ΔSoC|/100)` — large charge/discharge swings dominate, tiny noisy sessions barely count. Require ≥30% total pooled ΔSoC per month.
4. **Baseline normalization:** anchor SoH=100% at registration; early-life reference = median capacity over months 1–12 (skip settling month 0).
5. **Monotone envelope:** robust isotonic (PAVA), clip ≤100%, cap drop ≤6pp/month.

Parameters: `MIN_ROWS=5, MIN_DSOC=2%, CAP_BOUNDS=(40,400), MIN_MONTH_SOC=30%, SMOOTH_WIN=5mo, BASE_AGE=1–12mo, MAX_DROP_PER_MONTH=6pp`. **[VERIFIED]**

**Original "coulomb is broken" diagnosis & fix:** the earlier per-session-then-percentile approach produced artifactual ~10%/yr early drop because ~98% of sessions span only 2–10% SoC (CV ~30%). The ΔSoC-weighted pooled fix reduced fleet-median fade to **2.0%/yr** (from 3.8), with 80/91 vehicles in a plausible 0–8%/yr band (`research_soh-methods §2.3`). **[VERIFIED]**

### 6.3 Tier B — BMS remaining-capacity (`src/euler_features.py`, `bms_soh_monthly`)

1. High-SoC band filter `batterySoc ∈ [95,100]` (remCap has SoC bias: remCap ≈ 1.551·SoC − 28.7).
2. `full_cap = batteryRemainingCapacity / (batterySoc/100)`.
3. Per-vehicle nominal = median early full-cap (first 6 mo), adaptive bounds [0.6×,1.4×] median for pack diversity.
4. Monthly median full-cap (≥15 obs/month) → `IsotonicRegression(increasing=False)`, clip ≤100%.
5. `SoH = 100 × fitted_capacity / nominal_early`.

Cross-validation (2026-06-18) compared coulomb (too noisy), BMS-capacity (winner: clean 100→~90% fade), and reported (too coarse). **[VERIFIED]** (`memory: euler-schema-finding`)

### 6.4 Tier C — Reported SoH (`src/bajaj_features.py`, `reported_soh_monthly`)

Per-month median of `essBmsSohcEstPercValue` (≥15 obs/month) → 3-month rolling median → greedy cummin envelope (sufficient for an already-clean field) → **no renormalization** (aged vehicles legitimately enter below 100%); require ≥4 months. Clipped to [40,100]. **[VERIFIED]**

### 6.5 Monotone envelope — isotonic vs greedy (`memory: mahindra-isotonic-soh`)

**Problem:** greedy `soh[i]=min(soh[i],soh[i-1])` freezes the curve at a single noisy low month (staircase). Example VIN MB7D8CLLFNJE47477: raw dipped to 75.2 then recovered to 83.3; greedy locked 77.4 forever; isotonic held 83.3.

**Robust isotonic (`_robust_isotonic`, src/soh.py):** Hampel-clip each point to rolling-median ± 3×1.4826×MAD → PAVA `IsotonicRegression(increasing=False, y_max=100)` → drop-rate cap 6pp/month → clip [0,100].

**A/B validation (`src/mahindra_soh_ab.py`, against denoised-raw physical truth):**

| Cohort | Isotonic MAE | Greedy MAE | Δ |
|-----|-----:|-----:|-----:|
| Overall | 3.89 | 4.88 | −0.99 (−20%) |
| Degraders (n=27) | 4.17 | 4.53 | −0.36 (−8%) |
| Flat (n=70) | 3.62 | 5.22 | −1.60 (−31%) |

Isotonic adopted live for Mahindra (2026-06-30); Euler already used it; Bajaj keeps greedy (clean field); Piaggio inherits isotonic via `src/soh.py`. This does **not** contradict `smoothed-target-rejected` — that rejected a rolling-*mean* scored against itself; this is a monotone *projection* scored against independent truth. **[VERIFIED]**

### 6.6 Spec sheet (`OEM_Model_Specs.csv`, `src/build_oem_specs.py`)

Euler HiLoad: 58V, 9.9–11.5 kWh, ~171–198 Ah. Mahindra Treo Zor: 48V, 7.37 kWh; Treo Plus (majority passenger): 48V, 10.24 kWh, ~213 Ah, battery-km > vehicle-km. Bajaj RE E-TEC 9.0: 8.9 kWh, LFP. Piaggio Ape E-Xtra FX: ~12 kWh nominal. **[VERIFIED]** for capacities; warranty terms see §10/§13.

---

## 7. SoH validation (all experiments, quantitative)

This section documents **every** validation experiment — including the many negative results — because the negative results define the boundary of what can be claimed.

### 7.1 Euler resale cross-validation (`data/Repo - Pricing - Sold.csv`, `memory: euler-resale-data`)

124 Euler vehicles matched by full VIN (176 Euler resale rows; latest cohort 2026-06-24, 44/50 usable degraders, 88% yield). 84 degraders (init > final + 3pp). Grouped 5-fold CV: **overall RMSE 3.50pp; degrading-only 5.03pp**; improvement over persistence (5.77) = **12.8%**, over trend (5.63) = **6.2%**. Three well-observed resale-date cohorts for forecast validation: 2025-05, 2025-09, 2026-04. Only 3–4 vehicles show complete 100→80% arcs (calendar-limited). **[VERIFIED]**

### 7.2 SoH artifact audit — see §4.2. 12/14 aged vehicles tainted; LFP-BMS root cause. **[VERIFIED]**

### 7.3 Mahindra native-feed deep dive (`dashboard/native_explorer.py`, 13 sections; `src/native_explore_prep.py`)

Cohort: 100 longest-history vehicles, **complete** data (~95 driving-segments/month vs ~11 in the thin sample), ~23M rows, median 20 months.

- **§3 distanceToEmpty ≈ SoC × constant, r=0.92** → zero extra capacity information. **[VERIFIED]**
- **§4–10 distance-per-SoC (km/%SoC):** net monthly change is a coin flip (40 down / 44 up / 16 flat) and drifts slightly **UP** (wrong sign for degradation). Fixed 90→20% window ~1.25 km/%SoC, trend 95% CI spans zero; speed+season explain only ~5% of variance; dominant confounder = unobservable cargo **payload**. Window-search yields slopes −0.04…−0.44 %/mo → **p-hacking**. **[VERIFIED]**
- **§12 charge-rate proxy (%SoC/hr over 30→70% window):** raw within-vehicle CV 28% → **8% after a consistent-charger filter** (charger type is controllable, unlike payload). But fleet slope +0.01 %/mo, 95% CI [−0.15,+0.07] = **flat** (young fleet, no fade yet). **[VERIFIED]**
- **Verdict:** native-only Mahindra SoH is not measurable or validatable from the current feed. Levers: (a) get current/voltage into the native stream, or (b) let the fleet age and re-run charge-rate. **[VERIFIED]**

### 7.4 Both-feeds case study (`memory: mahindra-two-feed-coverage`)

Of 225 both-feeds vehicles, only **5** have both rich native AND Intellicar coulomb; only **NJH48488** overlaps in time (9 months, 2025-10→2026-06). It is a real **100→89% coulomb degrader**, yet the native distance proxy correlates **r=−0.60 (WRONG sign, spurious)**. The other 4 have zero temporal overlap (coulomb 2023–24, native 2025–26; the cohort migrated Intellicar→native). Consequence: native proxies **cannot be validated** against coulomb. **[VERIFIED]**

### 7.5 Intellicar-derived proxy validation (`src/intellicar_proxy_prep.py`, `src/proxy_lib.py`)

The one place proxies *can* be validated with coulomb (same feed, same time). 262k Intellicar files, 96 VINs, 78M rows; clamp soc to [0,100], resample to 2-min cadence. Within-vehicle-normalized correlation to coulomb SoH:

| Proxy | Vehicles | Pairs | r | 95% CI | Verdict |
|-----|-----:|-----:|-----:|-----|-----|
| km/%SoC | 87 | 1,125 | **+0.057** | [−0.002, +0.115] | ≈zero — no signal |
| %SoC/hr (chg_rate) | 11 | 82 | **−0.312** | [−0.469, −0.103] | credibly *wrong sign* |

Even within the same feed, both proxies fail. The −0.31 sign suggests they capture *efficiency* fade (older vehicle charges slower), not *health* fade. **[VERIFIED]** (`memory: mahindra-native-signals-exhausted`)

### 7.6 Charge-energy (kwh) and Euler coulomb recompute — both rejected

- **Charge-energy** (`capacity_kwh ≈ ΔkWh/ΔSoC×100`): physically sound (median ~8 kWh, CV 0.11) but only ~22/96 VINs log plausible scale, ~8/96 get ≥3 monthly points, only 4 overlap coulomb (r=+0.32, ~4.7pp diff — untrustworthy). **Rejected.** **[VERIFIED]**
- **Euler segment-coulomb recompute** (`scratchpad/euler_soh_electrical.py`): feasible on well-sampled VINs (217372 reproduced BMS endpoint within a few pp) but month-to-month std 9–28pp, ~25% of segments give impossible SoH>100%, and the worst artifact units have only 1–4 segments. **Rejected as primary; usable only as a sparse charge-only diagnostic cross-check** (`memory: soh-recompute-rejected`). **[VERIFIED]**

### 7.7 Supervised native→coulomb estimation

Trained native features (km_month, soc_mean, frac_soc_high/low, charge-rate) → coulomb SoH label, LOVO on both-feeds vehicles. Labeled set 198 vin-months / 51 VINs (age 11–38mo, only 11 degraded). **Predict-mean MAE 4.83pp < age-only 5.30 < all-native 5.42pp**; Spearman +0.16 (chance). Only **age** has credible signal (Spearman −0.51 on 1,309 coulomb vin-months; age-only LOVO 3.02pp) but with 9–13pp spread at fixed age → population curve only, not per-vehicle. **Implication:** native-only SoH must be an **age-based fleet prior with P10/P90 bands**, not a telemetry model. **[VERIFIED]**

### 7.8 Cycle features & range-health rejected (`memory: cycle-features-rejected`)

Bajaj cycle counter (`essBmsChgcycleActCountValue`): LOVO (8 seeds, GBR, target reported SoH) — age-only 2.89pp, +cycles 3.12pp (**+7.9% worse**), +cum_km 3.06, +cycles+thermal 3.08. Cycles don't help (collinear with age). Range-health (Wh/cycle): naive within-vehicle Spearman +0.88 collapsed to **−0.06 after age-control** (artifact of two co-declining series). **Methodological rule [VERIFIED]:** always age-control (or first-difference) short-window within-vehicle correlations before claiming "tracks SoH." `src/range_health.py` was built then deleted.

### 7.9 Smoothed-target training rejected (`memory: smoothed-target-rejected`)

Training on smoothed SoH (5-mo rolling mean + monotone floor) vs raw, grouped 5-fold, scored on raw actuals: Euler 3.48→3.50 (wash), **Mahindra 2.94→3.30 (worse)**. SoH is already smoothed by the isotonic envelope; extra smoothing removes signal. Decision: train on RAW; the `learn_ml.py` "Smooth SoH" toggle is display-only. **[VERIFIED]**

### 7.10 Bayesian behaviour-conditioned SoH — the constructive answer

Model (`src/bayes_degradation.py` hierarchical Gibbs sampler, self-tested on synthetic data; `src/behaviour_soh_experiment.py`):

```
SoH_ij ~ N(a_i + b_i·age_ij, σ²);  a_i ~ N(μ_a, σ_a²);  b_i ~ N(x_i·β, σ_b²)
x_i = [per-source baseline dummies, within-source z-scored behaviour]
```

Behaviour features (native-computable): `km_month, soc_mean, frac_soc_high, frac_soc_low`.

> **Live-artifact reconciliation.** The research briefs quote an earlier run (545 vehicles / 9,275 obs; km_month slope −0.039; +1.8% MAE gain; soh50@36mo ≈ 91%). The **live `data/mahindra/behaviour_soh_report.json` (2026-07-02) is authoritative** and differs; both are reported below. The qualitative conclusions are identical.

**Live results (`data/mahindra/behaviour_soh_report.json`, verified):**

| Quantity | Live value | Brief value (prior run) |
|-----|-----|-----|
| Training cohort | **540 vehicles, 9,175 obs** | 545 / 9,275 |
| Bajaj baseline rate | −0.652 SoH/mo | −0.757 |
| Euler baseline | −0.278 | −0.297 |
| Mahindra-IC baseline | −0.278 | −0.244 |
| Piaggio baseline | −0.174 | −0.156 |
| km_month slope (only credible) | **−0.074 [−0.100, −0.048]**, credible | −0.039 [−0.060, −0.018] |
| soc_mean / frac_soc_low | not credible (CI spans 0) | not credible |
| Held-out MAE (behaviour) | **0.1938** vs baseline 0.2056 → **+5.7%** | 0.1995 vs 0.2032 → +1.8% |
| Band decomposition | param σ **0.023** vs heterogeneity σ **0.246** (≈10×) | 0.021 vs 0.241 (11.5×) |
| Native output @36mo (median SoH50) | **86.1%**, band width 22.7pp, median rate −0.382 | 91%, ±11pp, −0.253 |

**Per-source km-rate correlation (live, post-fix):** Bajaj ρ=−0.72 (n=57), **Mahindra-IC ρ=−0.24 (n=149)**, Piaggio ρ=−0.22 (n=215), Euler ρ=+0.08 (n=119). The km_month effect is **credible and consistently signed across 3 of 4 OEMs — including Mahindra's own (−0.24), so the native tilt has same-OEM support rather than being borrowed.** Bajaj is the strongest contributor (its rates are late-window local slopes) and Euler is the lone null (glitchy raw odometer). *(A drop-Bajaj LOSO on the earlier **pre-fix** 545-vehicle run collapsed the pooled slope to −0.012; the post-fix model's Mahindra-IC −0.24 is the relevant same-OEM check.)* **[VERIFIED]**

**Interpretation [VERIFIED]:** behaviour is **descriptive** (higher km/month → slightly faster fade within an OEM), not **prescriptive**. Irreducible between-vehicle heterogeneity (σ_b≈0.25 SoH/mo) dominates by ~10×; observable behaviour explains only a tiny slice. Native-only Mahindra forecasts (`data/mahindra/native_behaviour_soh.parquet`, 100 VINs, P10/P50/P90) therefore carry wide bands (~±11pp effective at 36 months).

**Adversarial verification (2026-07-02, 3 independent lenses — age-confound, feature/z-score consistency, calibration) is COMPLETE.** It found **2 blockers + 6 majors**, all fixed in the v2 model reported above: (i) km/month was corrupted by odometer resets (Euler up to 73M km/mo) → recomputed from robust **odometer endpoint-span**, identically for good sources and native; (ii) per-source z-scoring made the slope scale-arbitrary → **global standardization**, one physical slope of **−0.079 SoH/mo per +1000 km/mo**; (iii) single global σ_b mis-calibrated non-Mahindra OEMs → **per-source σ_b**; (iv) dropped the threshold-mismatched `frac_soc_high` (good sources soc>90, native soc>80). The verifier **confirmed** the native band is *genuine* Mahindra heterogeneity (a Mahindra-only refit reproduces σ_b≈0.25) and that anchoring SoH=100 at age 0 is justified (fitted intercepts cluster at 100.0). Net effect of the fixes: the held-out gain rose from +1.8% to **+5.7%** and the km driver gained same-OEM (Mahindra) support. **[VERIFIED]**

### 7.11 Open-source chemistry benchmarks (`memory: opensource-driving-datasets`)

For LFP method validation: **CALCE LFP (A123)** — top pick for coulomb-method validation; **Severson/MIT-Stanford-Toyota** 124-cell run-to-failure — RUL benchmark; **Sandia** (LFP+NCA+NMC) — degradation-shape priors. **No open field-degradation dataset exists for 3-wheelers**; Indian 3W cycle profiles (SAE/ERDC/ARAI) provide speed-time only, convertible to load via NREL FASTSim. **[VERIFIED]**

---

## 8. Vehicle selection for modeling

### 8.1 Two gates in series

1. **Data-quality gate** (§4.1): trainable vs thin (`vehicle_data_quality.csv`). 1,909 trainable / 325 thin. **[VERIFIED]**
2. **Trajectory curation** (`src/training_curation.py`): robust √t-Theil-Sen smoothing, monotone projection, then bucketing:

| Bucket | Criteria | Count E/M/B | Role |
|-----|-----|-----|-----|
| GRACEFUL | aged/near-warranty, projects ≥EoL, ≥3pp decline | 0/0/0 | "ages but survives" (rare) |
| FLAT | projects safe, <3pp decline | 40/44/29 | genuine negative example |
| PROBABLE_OOR | projects ≥EoL, young/declining | 57/34/14 | supporting survivor |
| AT_RISK | projects <EoL after cleaning | 9/6/14 | real EoL example |
| EXCLUDED | <4 months or <4-mo span | 14/12/1 | data-thin |

**Good training data** (GRACEFUL+FLAT+PROBABLE_OOR) = Euler 97/119, Mahindra 78/84, Bajaj 27/57 (`memory: training-curation`). Thresholds: `GRACE_AGE_FRAC=0.6, MIN_DECLINE=3.0, FLAT_DECLINE=3.0, MIN_MONTHS=4, MIN_SPAN=4`. **[VERIFIED]**

**Caveat [VERIFIED]:** √t-smoothing rescued only 3 of 14 raw-aged vehicles from at-risk; the remaining endpoints need SoH *recompute* (isotonic is not robust to upstream cliffs), not post-hoc smoothing.

### 8.2 What is *not* used to select vehicles

- **Flat vehicles are kept** (dropping them was rejected four ways; `memory: flat-vehicles-data-quality`). The real lever is the thin gate.
- **Cycle features are excluded** (§7.8). **Native distance/charge-rate proxies are excluded** as production features (§7.3–7.5).

---

## 9. Criteria for training vs inference

**For TRAINING (`research_modeling §VI.3`):**
- ≥MIN_MONTHS=4 SoH observations over ≥MIN_SPAN=4 months, passing the data-quality gate.
- Bucket into GRACEFUL/FLAT/PROBABLE_OOR (keep) or AT_RISK (keep, labeled) or EXCLUDED (drop).
- **Grouped 5–6-fold CV**: each vehicle stays wholly in train or test (never split), to prevent leakage.
- Train on **raw** SoH target (§7.9). Weight loss-months up (SoH is a staircase; most months lose nothing).

**For INFERENCE:**
- Any vehicle with ≥3 months of history receives a forecast.
- **Cold-start (zero history):** cross-OEM transfer on the 9-feature shared intersection (`age_months, soc_mean, frac_soc_high, frac_soc_low, temp_mean, temp_max, km_month, odo_max, cum_km`), else source-baseline rate.
- **Prefer native measured data over proxies**: Intellicar-coulomb Mahindra > native Mahindra; Bajaj reported > any proxy.

**Forecaster per OEM:**

| OEM | Cohort | SoH target | Primary forecaster | Validation MAE |
|-----|-----:|-----|-----|-----|
| Euler | 80 | BMS-capacity (isotonic) | Trajectory (cumulative √-horizon loss, `src/euler_model.py`) | ~0.7pp LOVO |
| Mahindra | 96 | distance/SoC proxy (research) / Bayesian native | Bayesian behaviour (`src/bayes_degradation.py`) | ~1.5pp (est.) |
| Bajaj | 58 | reported | Rate model (LightGBM quantile, `src/bajaj_model.py`) | 1.14pp 5-fold CV |
| Piaggio | 247 | coulomb (Intellicar) | Rate model (`src/model.py`) | ~1.2pp (est.) |

**Rate-model architecture (`src/model.py`, `src/bajaj_model.py`):** predict monthly SoH loss (pp/mo) → roll forward with **expected (mean)** monthly loss as the central forecast (median q50 stays flat on a staircase) → LightGBM quantile bands (α=0.1/0.5/0.9) → per-step cap MAX_STEP=1.2pp. Bajaj degraders: **model 1.04 vs persist 3.08 / trend 2.06** (clear win); flat cohort 1.24 vs persist 0.87 (model slightly over-predicts). **[VERIFIED]** (`memory: bajaj-model`)

**Cross-OEM transfer (`src/cross_oem_transfer.py`, `memory: cross-oem-transfer`):** aged→young transfer of the *rate* (not absolute SoH). Target Bajaj (MAE pp/mo): persist 2.98, native LOVO 0.87, from Euler 3.79, from Mahindra 2.61, **Euler+Mahindra combined 2.09** (closes ~30% of the gap). Degraders-only: from-Mahindra 2.24 vs persist 3.59. **Cold-start tool only**; native wins with ≥3 months of data; reverse young→old does not transfer. **[VERIFIED]**

**RUL in km (`src/rul_km.py`, `memory: remaining-km-model`):** `remaining_km(eol) = km/month × months_until_forecast_reaches(eol)`, thresholds 80/70/60% SoH. Fleet medians: **Euler ~19,000 km to 80% (~37,000 to 60%), Bajaj ~27,000 km to 70%**. **Mahindra unusable** (odometer too sparse). Degradation is calendar/condition-driven, not mileage-driven — higher utilization → more total km before calendar EoL. **[VERIFIED]**

---

## 10. Fleet warranty risk per OEM (from inference)

### 10.1 Warranty terms (`src/config.py WARRANTY` / `FLEET_WARRANTY`)

| OEM | Model | Battery warranty | Binding limit | Status |
|-----|-----|-----|-----|-----|
| Mahindra | Treo/Treo Plus (passenger) | 3 yr / 120k km | Time (3yr) | [VERIFIED] spec+config |
| Mahindra | Zor Grand (cargo) | 5 yr / 120k km | Time (5yr) | [VERIFIED] |
| Mahindra | Treo Zor (base) | 150k km (no yr) | Time (3yr binds) | [VERIFIED] |
| Euler | HiLoad (100% of fleet) | **3 yr / 80k km** | Time (3yr) | **[ASSUMPTION]** was 5yr; provisional |
| Bajaj | RE E-TEC 9.0 (~94%) | 5 yr / 120k km | Time (5yr) | [ASSUMPTION] spec-image only |
| Piaggio | Ape E-Xtra FX (~99%) | 3 yr / 100k km | Time (3yr) | [ASSUMPTION] config, unverified |

`FLEET_WARRANTY = {euler:(3,80000), mahindra:(3,120000), bajaj:(5,120000), piaggio:(3,100000)}`. **[VERIFIED]** as configured.

### 10.2 Core finding — term-dependent survival (`data/mahindra/soh/warranty_risk.csv`, n=95 — mixed cohort: Treo 79 + Zor Grand 14 + 2 Piaggio Ape, verified live)

| Cohort | n | Warranty | Survival ≥80% SoH | At-risk |
|-----|-----:|-----|-----:|-----:|
| Treo (3-yr) | 79 | 3 yr | **88.6%** | 9 |
| 5-yr terms (Zor Grand 14 + 2 Piaggio Ape) | 16 | 5 yr | **25.0%** | 11–12 |
| — of which Zor-Grand-only | 14 | 5 yr | **21.4%** | — |
| **Fleet-wide** | 95 | mixed | **77.9%** | **21** |

The XGBoost warranty projection (300 trees, depth 4, lr 0.03, √t sample weights; 60-month horizon) uses coulomb SoH (immune to LFP OCV artifacts). Note the source file mixes models/OEMs (Treo 3-yr, Zor Grand 5-yr, 2 Piaggio Ape); **Zor-Grand-only 5-yr survival is 21.4% (3/14)** — the 25.0% (4/16) figure includes the 2 Piaggio Ape rows. **[VERIFIED]**

**Key insight [VERIFIED]:** the working **90%-survival assumption holds for 3-year terms (~89%) but fails badly for 5-year terms (~25%)** (`memory: warranty-risk-finding`). If Euler's real term is 3yr, its assumption is viable; if Bajaj's 5yr battery term is applied to its (young) fleet, expect a similar ~20–30% at-risk once it ages.

### 10.3 Limiting factor — time, not mileage [VERIFIED]

**0 of 95** vehicles breach via the km limit; all 21 breaches are time-first. Median km at expiry for at-risk vehicles is very low (~30–70k). These low-duty-cycle fleets age from calendar/thermal stress during idle, not from cycling. The km-warranty term is a safety limit, rarely binding.

### 10.4 Confidence & risk attribution

Reliable projections (`conf='ok'`): **72/95**; low-confidence (`conf='low'`, <8-mo history or mislabeled): 23/95. Per-vehicle risk-cause attribution in the dashboard (`dashboard/build_dashboard.py`) z-scores recent stressors into Charging / Driving / Thermal / Deep-discharge / Calendar-aging categories — **heuristic, not causal**. Calendar aging (no dominant stressor) is the largest category among at-risk Treo vehicles. Dashboard status: AT-RISK <78%, WATCH 78–80%, OK ≥80% (`EOFL=80, RISK_MARGIN=2`). **[VERIFIED]**

### 10.5 Native-fleet warranty outlook

For the ~98% native-only Mahindra fleet (no coulomb), the Bayesian behaviour model gives fleet-level curves (`data/mahindra/native_behaviour_soh.parquet`). **Live artifact: median SoH50 @36mo = 86.1%, band width 22.7pp, median rate −0.382 SoH/mo** (the briefs' earlier run gave ~91% ±11pp). Either way, bands are wide, reflecting the no-current observability limit. **[VERIFIED live; ASSUMPTION on external validity]**

### 10.6 Second-life (BESS) [ASSUMPTION]

A fixed cubic `SoH = −9.11155e-11·c³ + 5.88935e-7·c² − 0.00921386·c + 77.6217` (c = cumulative cycles) predicts second-life fade to 20% EoSL at ~6,314 cycles. **Source unknown, not validated across OEMs** — user-supplied, operationally applied verbatim.

---

## 11. Confidence intervals & caveats

| Metric | Value | Confidence | Basis |
|-----|-----|-----|-----|
| Mahindra Treo 3yr survival | 88.6% (n=79) | **HIGH** | verified warranty, coulomb SoH |
| 5-yr-term survival (Zor Grand 14 + 2 Ape) | 25.0% (n=16); Zor-only 21.4% (n=14) | **MEDIUM** | small mixed cohort, aged artifacts |
| Euler HiLoad survival | — | **LOW** | warranty unverified (5yr→3yr) |
| Bajaj 5yr projection | — | **LOW** | fleet <1yr; pure extrapolation; 0 aged |
| Piaggio forecasting | — | **LOW** | 247 VINs but young; age from first-telemetry |
| Native Mahindra @36mo | 86–91% SoH, ~±11pp | **MEDIUM** | valid fallback; wide bands intrinsic |
| km_month behaviour effect | credible, 3/4 OEMs (Mahindra −0.24) | **MEDIUM** | post-fix; same-OEM support, Euler null |
| Coulomb SoH (Tier A) | fleet fade ~2%/yr | **HIGH** | ΔSoC-weighted fix, 80/91 in-band |

**Structural caveats [VERIFIED]:**
1. **Aged-cohort artifacts** (12/14) inflate long-horizon at-risk %; fix is upstream SoH recompute (segment-coulomb / rate-bounding), not post-hoc smoothing.
2. **Sample bias:** cohorts are oldest-N pulls; the true commercial fleet is younger → real survival likely *higher* than reported.
3. **Native-only Mahindra (~98%)** has no per-vehicle SoH; only a fleet prior with wide bands.
4. **Behaviour model is descriptive** (gains ~5.7% over OEM-average; heterogeneity dominates ~10×) and **passed adversarial verification (2026-07-02)** after fixing a corrupted km/month definition and a scale-transfer bug; the native bands are confirmed to be genuine Mahindra heterogeneity, not a data gap.
5. **Warranty terms** for Euler/Bajaj/Piaggio and the **80% EoL threshold** are unvalidated → report split-by-term and flag provisional.
6. **Behaviour-model live artifact diverges from the research briefs** (540 vs 545 vehicles; SoH50@36mo 86% vs 91%); this report treats the live JSON as authoritative.

---

## 12. Appendix A — Experiment log

| # | Experiment (file) | Hypothesis | Method | Result | Verdict |
|--:|-----|-----|-----|-----|-----|
| 1 | Coulomb per-session (src/soh.py, orig) | ∫I·dt gives clean SoH | per-session percentile, first-month anchor | artifactual ~10%/yr early drop (2–10% SoC sessions, CV 30%) | **Fixed** → ΔSoC-weighted pooled |
| 2 | Coulomb ΔSoC-weighted (src/soh.py) | pooled weighting stabilizes | Σ∫I·dt / Σ(ΔSoC/100), median 1–12mo baseline | fleet fade 2.0%/yr, 80/91 in 0–8% band | **Adopted (Tier A)** |
| 3 | Euler BMS-capacity (src/euler_features.py) | remCap@high-SoC → SoH | full_cap, isotonic | clean 100→90%; beats coulomb & reported | **Adopted (Tier B)** |
| 4 | Euler coulomb recompute (scratchpad) | replace Tier B with coulomb | segment coulomb on dense current | std 9–28pp, 25% SoH>100%, 1–4 seg on worst | **Rejected** (diagnostic only) |
| 5 | Bajaj reported SoH (src/bajaj_features.py) | field is clean | median→3mo smooth→cummin | smooth monotone 100→78% | **Adopted (Tier C)** |
| 6 | Isotonic vs greedy (src/mahindra_soh_ab.py) | isotonic beats cummin | A/B vs denoised-raw truth | 3.89 vs 4.88 MAE (−20%) | **Isotonic adopted** |
| 7 | Distance-per-SoC (native_explorer §4–10) | km/%SoC ∝ health | complete 100-vin data | coin flip, drifts up, payload confound | **Rejected** |
| 8 | distanceToEmpty (native_explorer §3) | extra capacity info | correlate with SoC | r=0.92 (redundant) | **Rejected** |
| 9 | Charge-rate %SoC/hr (native_explorer §12) | charger-controllable proxy | 30→70% window, consistent-charger filter | CV 28→8%, but slope flat (CI spans 0) | **Rejected (young fleet)** |
| 10 | Charge-energy kwh (mahindra_native_soh.py) | ΔkWh/ΔSoC → capacity | charging sessions | ~8/96 computable, unit drift, r=+0.32 (n=4) | **Rejected (too sparse)** |
| 11 | Both-feeds NJH48488 (§11) | validate proxy on real degrader | 9-mo overlap | coulomb 100→89%, proxy r=−0.60 | **Proxy fails** |
| 12 | Intellicar proxy (src/intellicar_proxy_prep.py) | proxies work within same feed | 96 VINs, normalized corr | km/%SoC +0.06, %SoC/hr −0.31 | **Both fail** |
| 13 | Supervised native→coulomb | native features predict SoH | LOVO, 51 VINs | predict-mean 4.83 < age 5.30 < native 5.42 | **Native rejected; age-prior only** |
| 14 | Cycle features (Bajaj) | cycles improve forecast | LOVO 8 seeds GBR | age 2.89 vs +cycles 3.12 (worse) | **Rejected** |
| 15 | Range-health Wh/cycle | tracks SoH | age-controlled corr | +0.88 → −0.06 after age-control | **Rejected (artifact)** |
| 16 | Smoothed target (memory) | smoothing helps | A/B, score raw | Euler wash, Mahindra 2.94→3.30 | **Rejected** |
| 17 | Euler resale CV (Sold.csv) | forecast matches resale | grouped 5-fold, 124 vins | RMSE 3.50; +12.8% vs persist | **Validated** |
| 18 | SoH artifact audit (src/soh_audit.py) | aged signal is clean | cliff/stuck/iso detectors | 12/14 aged tainted | **Artifacts confirmed** |
| 19 | Bayesian behaviour (src/bayes_degradation.py) | similar usage → similar fade | hierarchical Gibbs, 540 vins; adversarially verified (2026-07-02) | km_month only credible (3/4 OEMs, Mahindra −0.24); +5.7% MAE; het≈10×param | **Descriptive, adopted for native prior** |
| 20 | Cross-OEM transfer (src/cross_oem_transfer.py) | aged→young transfers | shared-feature rate model | Euler+Mahindra→Bajaj 2.09 vs 2.98 persist | **Cold-start tool** |
| 21 | Piaggio onboarding (src/piaggio_features.py) | 4th OEM, coulomb | Intellicar coulomb | 247 vins, 100→97.7%, 127 degraders | **Onboarded (Tier A)** |
| 22 | RUL-km (src/rul_km.py) | km-RUL from km/mo × months-to-EoL | odometer slope × forecast | Euler ~19k, Bajaj ~27k; Mahindra unusable | **Adopted (Euler/Bajaj)** |
| 23 | Warranty risk (warranty_risk.csv) | 90% survive warranty | XGBoost 60-mo projection | 3yr 88.6%, 5yr 25.0%, 0 km-breach | **Term-dependent** |

---

## 13. Appendix B — Assumptions register

| # | Assumption | Status | Impact if wrong | Source |
|--:|-----|-----|-----|-----|
| A1 | All 4 OEMs are LFP | **VERIFIED** (owner) | Would invalidate artifact root-cause & method choice | memory: oem-spec-sheet |
| A2 | Coulomb (Intellicar) SoH is ground truth | **VERIFIED** | Anchors all validation; immune to OCV artifacts | src/soh.py |
| A3 | Isotonic envelope is the right monotone projection | **VERIFIED** (A/B −20%) | Greedy staircase would re-appear | memory: mahindra-isotonic-soh |
| A4 | Euler HiLoad warranty = 3yr/80km | **UNVERIFIED** | If 5yr, Euler at-risk overstated ~40% | memory: euler-warranty-3yr |
| A5 | Bajaj battery = 5yr/120km | **UNVERIFIED** (spec image) | If 3yr, Bajaj at-risk drops sharply | memory: warranty-specs |
| A6 | Piaggio = 3yr/100km | **UNVERIFIED** | Fleet young; not yet tested | src/config.py |
| A7 | Mahindra Treo = 3yr/120km | **VERIFIED** (spec+config) | Anchors the HIGH-confidence 88.6% | memory: warranty-specs |
| A8 | 80% = end-of-first-life | **UNVERIFIED** (no field val) | Shifts all at-risk counts | dashboard/build_dashboard.py |
| A9 | Native distance/charge proxies not SoH | **VERIFIED** (r≈0/−0.31) | Would enable 98% native coverage if wrong | memory: mahindra-native-signals-exhausted |
| A10 | Native-only SoH = age fleet prior + behaviour | **VERIFIED** (age-only beats telemetry) | Only defensible native method | research_validation §6 |
| A11 | km_month credible, consistent across 3/4 OEMs (Mahindra −0.24) | **VERIFIED** (post-fix per-source) | Behaviour effect not universal (Euler null) | data/mahindra/behaviour_soh_report.json |
| A12 | Heterogeneity σ_b ≈ 10× param σ (irreducible) | **VERIFIED** | Per-vehicle rate unknowable from behaviour | behaviour_soh_report.json |
| A13 | Mahindra odometer unusable (0 clean) | **VERIFIED** | Kills Mahindra RUL-km & km-degradation | memory: degradation-vs-km-finding |
| A14 | Second-life cubic representative across OEMs | **UNVERIFIED** | BESS allocation mis-sized | research_warranty-risk §6 |
| A15 | Coulomb 2022–24 vs native 2025–26 do not overlap | **VERIFIED** (only 5/225, 1 in time) | Blocks native cross-validation | memory: mahindra-two-feed-coverage |
| A16 | Cohorts are oldest-N (over-aged) samples | **VERIFIED** | True fleet survival likely higher | memory: cross-oem-transfer |
| A17 | Bajaj reported field ≈ on-device coulomb counter | **ASSUMPTION** (inferred from cleanliness) | Explains why Tier C works for Bajaj only | research_soh-methods §4.2 |
| A18 | Piaggio age anchored to first telemetry | **VERIFIED** (no reg file) | Understates Piaggio cohort age | research_modeling §I.4 |
| A19 | Aged-vehicle artifacts inflate at-risk % | **VERIFIED** | Long-horizon at-risk over-predicted | memory: soh-artifact-audit |
| A20 | Live behaviour JSON supersedes brief values | **VERIFIED** (file inspected) | Report figures differ from briefs | data/mahindra/behaviour_soh_report.json |

---

*Prepared by the SoH/RUL research harness. Every figure is traceable to `src/`, `data/`, or a `memory:` finding. Figures marked [ASSUMPTION] must be resolved (chiefly official warranty certificates and the 80% EoL threshold) before external publication.*
