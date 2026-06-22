# EULER_RUL — Data & SoH Procedure (Runbook)

End-to-end procedure for building the battery SoH / degradation-forecasting dataset from the
S3 telemetry lake. Written for reproducibility. Last updated 2026-06-17.

---

## 0. Goal

Build a per-vehicle time series of **State of Health (SoH)** plus **degradation-driver features**
for Mahindra electric 3-wheelers (Treo / Zor Grand), to train an **SoH forecasting / RUL model**.

---

## 1. Data sources (S3)

Bucket `s3://oem-data-iot/`, credentials in `.env` (`AWS_*`, `S3_BUCKET`). Layout:

```
battery-oem-data/parquet/<oem>/...        # per-OEM feeds: bajaj, euler, mahindra, montra, piaggio, jbm
battery-oem-data/parquet/intellicar/      # telematics-aggregator feed (multi-OEM), battery-data + location-data
```

All feeds partition by `year=/month=/day=`. **Partition date = ingest date, NOT event date** — do
not prune partitions by event time.

| Feed | Path | Span | File density | Has current? |
|---|---|---|---|---|
| mahindra (OEM) | `mahindra/vehicle-data/` | 2024-11 → 2026-06 | tiny (~5 rows/file, ~70M files) | ❌ |
| intellicar | `intellicar/battery-data/` | 2022 → 2026 (+junk `year=0000`) | dense (~1–3k rows/file) | ✅ `current` |
| euler (OEM) | `euler/vehicle-data/` | 2022 → 2026 | medium | ❌ (voltage null too) |

See `SOH_COLUMN_USABILITY.md` for the full per-column audit.

---

## 2. Key lessons learned (read before re-running)

1. **OEM ≠ table.** The `mahindra/` OEM feed has Mahindra vehicles but **no current** → no coulomb
   counting. The `intellicar/` table also contains Mahindra vehicles (≈15% of rows) **and has
   current**. For physics-based SoH, use intellicar; the OEM feed adds temperature/location/usage.
2. **Scale.** mahindra OEM feed is ~70M tiny files; a full per-VIN scan is infeasible. Use a
   **monthly sample** (1 representative day/month) with a per-day file cap. intellicar is dense, so
   it tolerates more days/month.
3. **Selecting "oldest".** Earliest partition is unreliable (mahindra's first partition is a
   single-vehicle backfill dump). Select **most-aged by odometer** among vehicles that **recur
   across partitions** — robust and RUL-relevant.
4. **S3 Select gotchas.** `current` is a reserved word → quote it: `s."current"`. S3 Select omits
   null fields → always `df.reindex(columns=ALL_COLS)` after parsing.
5. **Column population is record-type-dependent.** In the mahindra feed, `state`/`batteryTemp`/`kwh`
   appear only on ~30% "status" rows; `soc`/`odometer`/`lat`/`long`/`gearPosition` are on ~100%.
   In intellicar, only **9 columns are populated for Mahindra** (the rest are other-OEM only).
6. **`kwh` (mahindra) is not integrable** (signed instantaneous, stale when parked) — do not use for
   energy counting.

---

## 3. Pipeline steps

### 3a. Profile & pick the cohort
- Sample a few dense partitions of each feed; collect VIN → max(odometer), recurrence, rows.
- Intersect intellicar-Mahindra VINs with mahindra-feed VINs → **overlap set** (`data/overlap_vins.csv`, ~96 VINs).
- Rank overlap by odometer (most-aged) requiring data in both feeds → **cohort**
  (`data/forecast_cohort.csv`, top 15).

### 3b. Extract (S3 Select, monthly sample, projected columns)
- `python src/import_cohort.py intellicar <oem>` → `data/<oem>/intellicar/` (coulomb-counting cols:
  vin, eventAt, soc, current, batteryVoltage, odometer, dte, make, model; 3 days/month).
- `python src/import_cohort.py <oem> <oem>`      → `data/<oem>/feed/` (stress/usage cols: vin,
  eventAt, soc, odometer, distanceToEmpty, latitude, longitude, gearPosition, batteryTemp, state,
  kwh, vehicleModel; 1 day/month).
- Per-OEM params live in `src/config.py`; layout under `data/<oem>/` (see README).
- Each S3-Selects `WHERE vin IN (cohort)`, writes one Parquet per source file with matches, and
  skips already-done files (resumable).

### 3c. Compute SoH (target)
- **Coulomb counting (intellicar)** — `notebooks/01_soh_target/mahindra_soh_coulomb.ipynb`:
  split into continuous sessions (gap > 300 s), integrate `Q = ∫I·dt` (Ah), capacity per session =
  `|ΔQ| / (|ΔSoC|/100)`; per month take the **75th percentile** (upper envelope, since gaps make
  coulomb counting under-estimate); smooth (rolling median); normalise early-life = 100%; enforce
  monotonic non-increasing (`cummin`).
- **Distance-per-SoC proxy (mahindra feed)** — `notebooks/01_soh_target/mahindra_soh_distance_proxy.ipynb`: capacity ∝
  `Σ Δodometer / Σ Δsoc` during discharge; same smoothing + monotonic envelope.
- **Comparison** — `notebooks/01_soh_target/mahindra_soh_method_compare.ipynb`: both methods on a shared vehicle; agreement
  validates the proxy. (On `MB7F8CLLFNJH48488` they agreed within ~3 pp where they overlapped.)

### 3d. Feature engineering for forecasting (see §4)

---

## 4. Forecasting features (degradation drivers)

Target = SoH(t) from §3c. Features are engineered per vehicle, per month (or per cycle):

| Driver | Feature(s) | Source signal |
|---|---|---|
| **Cycling / throughput** | cumulative Ah, equivalent full cycles, cycles/month | intellicar `current` (∫\|I\|dt) |
| **C-rate stress** | mean/95th-pct charge & discharge C-rate | intellicar `current` |
| **Depth of discharge** | per-discharge ΔSoC distribution, mean DoD | `soc` |
| **SoC dwell** | % time at high (>90) / low (<20) SoC | `soc` |
| **Voltage stress** | min/max pack voltage, time near cutoffs | intellicar `batteryVoltage` |
| **Thermal** | mean/max battery temp, time > 40 °C (sparse) | mahindra `batteryTemp` (~30%) |
| **Ambient / climate** | location → seasonal/region temperature proxy | mahindra `latitude`/`longitude` |
| **Usage intensity** | km/day, trips/day, drive vs idle time | `odometer`, `gearPosition`, `state` |
| **Calendar age** | months since first observation | `eventAt` |
| **Range fade** | distance-to-empty at given SoC over time | mahindra `distanceToEmpty`, intellicar `dte` |

Cross-feed join key: **VIN + month** (cohort vehicles exist in both feeds).

---

## 4b. Modeling options (open-source)

No off-the-shelf pretrained model transfers to Treo **field** telemetry — academic models/datasets
are lab full-cycle data. Reuse the *architectures and toolkits*, train on our cohort. Two sub-tasks:

**(a) SoH estimation** (features → SoH at month *t*):
- **Gaussian Process Regression** (scikit-learn) — preferred for field SoH: gives calibrated
  uncertainty (needed for RUL bounds). Reference: **BattGP** (field-data GPR).
- **Gradient boosting** (LightGBM/XGBoost) — robust on noisy tabular features; good feature importance.
- **BatteryML** (github.com/microsoft/BatteryML) — model zoo + within/between-cycle feature recipes
  (lab-oriented loaders; swap in our data).

**(b) SoH forecasting / RUL** (project trajectory → months until SoH<80%):
- Per-vehicle **empirical fade fit** (capacity ∝ √time + cycle terms) extrapolated — strong baseline.
- **LSTM / TCN** sequence models; or open-weight **time-series foundation models** (Chronos-2,
  TimesFM, Moirai 2.0 on HuggingFace) for zero/few-shot baselines.
- Toolkits: Nixtla (`mlforecast`/`neuralforecast`), Darts, sktime, GluonTS.

**Plan:** GPR/LightGBM for estimation (uncertainty + importance), empirical-fade + Chronos baseline
for forecasting, validate against the coulomb-vs-proxy agreement. Baseline notebook:
`notebooks/02_features_model/mahindra_features_baseline.ipynb`.

## 5. Artifacts index

| Path | Purpose |
|---|---|
| `src/config.py` | **OEM registry** — add a new OEM here |
| `src/paths.py` | repo-root + per-OEM path helpers |
| `src/import_cohort.py` | **current** cohort extractor: `python src/import_cohort.py <feed> <oem>` |
| `src/import_{intellicar,mahindra,dense,compare_vin}.py` | earlier per-feed / dense / single-VIN extractors |
| `notebooks/` | analysis organized as the pipeline flow — see `notebooks/README.md` |
| `notebooks/01_soh_target/mahindra_soh_coulomb.ipynb` | coulomb-counting SoH (intellicar) |
| `notebooks/01_soh_target/mahindra_soh_distance_proxy.ipynb` | distance-per-SoC SoH (OEM feed) |
| `notebooks/01_soh_target/mahindra_soh_method_compare.ipynb` | same-vehicle method comparison |
| `notebooks/02_features_model/mahindra_features_baseline.ipynb` | features + GPR/LightGBM + RUL |
| `docs/OEM_MODELING_PLAYBOOK.md` | end-to-end modeling playbook (methods, assumptions, train/val/test) |
| `docs/SOH_COLUMN_USABILITY.md` | per-column usability audit |
| `data/manifests/<oem>_cohort.csv` | most-aged overlapping VINs per OEM |
| `data/manifests/<oem>_overlap.csv` | all VINs present in both feeds per OEM |
| `data/<oem>/{intellicar,feed,features,soh}/` | per-OEM extracts, features, SoH outputs |

## 6. Reproduce from scratch
```bash
pip install -r requirements.txt          # boto3, pandas, pyarrow, matplotlib, sklearn, lightgbm, ...
cp .env.example .env                      # fill AWS creds + S3_BUCKET
python src/import_cohort.py intellicar mahindra   # -> data/mahindra/intellicar/
python src/import_cohort.py mahindra   mahindra   # -> data/mahindra/feed/
jupyter nbconvert --to notebook --execute notebooks/01_soh_target/mahindra_soh_coulomb.ipynb
```
