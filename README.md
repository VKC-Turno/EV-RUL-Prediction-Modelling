# EULER_RUL_MODEL — Battery SoH & Degradation Forecasting

Extract EV battery telemetry from S3, compute **State of Health (SoH)**, engineer monthly features,
and train a per-OEM **SoH-forecasting / warranty-risk model**. Runs across five OEMs today —
**Euler · Mahindra · Bajaj · Piaggio · Montra** (JBM next) — each with a deployed model + registry.

**If you're building the preprocessing / feature-generation job on AWS, read
["Preprocessing & feature-generation pipeline"](#preprocessing--feature-generation-pipeline-aws) below —
it is the runbook.** The `src/*_features.py` + `src/soh.py` + `src/features.py` scripts are the reference
implementation of exactly what that job must produce.

---

## Preprocessing & feature-generation pipeline (AWS)

The job turns **raw S3 telemetry → one row per (vehicle, month)** carrying a SoH target + engineered
features: the `<oem>_featengg` table that *every* model and dashboard consumes unchanged. Reference code:

| Stage | Reference implementation |
|---|---|
| End-to-end per-OEM assembly | `src/piaggio_features.py` (coulomb), `src/montra_features.py` (BMS-capacity) |
| Coulomb SoH | `src/soh.py` — `coulomb_capacity_monthly()` → `capacity_to_soh()` |
| BMS remaining-capacity SoH | `src/euler_bms_soh.py` |
| Electrical features | `src/features.py` — `electrical_features()` |
| Extraction | `src/import_cohort.py` (full cohort), `src/montra_sample.py` (day-sampled POC) |

### 1 · Source (S3 bucket `oem-data-iot`, ap-south-1)

```
battery-oem-data/parquet/<oem>/battery-data/year=YYYY/month=MM/day=DD/*.parquet   # per-OEM raw telemetry
battery-oem-data/parquet/<oem>/gps-data/...                                        # GPS supplement
battery-oem-data/parquet/intellicar/...                                            # shared intellicar feed (Mahindra/Piaggio) — SIGNED current
battery-oem-data/parquet/{euler,bajaj,mahindra,montra,piaggio,jbm}/...             # OEMs present
battery-bms-data/...                                                               # BMS-level data
```
Raw columns vary by OEM: `vin, eventAt(ms epoch), soc, current, batteryVoltage|batteryPackVoltage,
resCapacity|batteryRemainingCapacity, batterySoh, temperature, odometer, chargeCycle`.
(Redshift schema `soh_etl.jun_26_featengg_results_{euler,mahindra,bajaj}` holds already-computed tables.)

> ⚠️ **Three layout gotchas that dominate the job design:**
> 1. **Millions of tiny files** (~2 VINs each): Piaggio ~308k, Montra ~200k+. Read in **threaded batches**; cast `vin` to a category to fit RAM.
> 2. **Day-partitioned, but a vehicle's events scatter across *every* daily file** — do **not** aggregate per ingest-partition. Pool the whole `(vin, month)` frame first, then aggregate once.
> 3. **Sentinels to clip**: SoC seen up to 837 (valid 0–100), current up to 65,279 A (valid |I| ≤ 400), voltage 0 / >1000 V (valid ~20–120).

### 2 · Clean
- `t = to_datetime(eventAt, unit="ms")`; drop NaN `t`/`soc`/`current`.
- Bound each channel out-of-range → NaN: **SoC 0–100 · |current| ≤ 400 A · voltage 20–120 V · resCapacity 1–500 Ah**.
- **Sessions**: a gap > 5 min (`GAP_S = 300 s`) starts a new continuous-logging session.

### 3 · SoH target — branch on what the feed carries

| OEM(s) | Feed carries | Method | Code |
|---|---|---|---|
| **Mahindra, Piaggio** | signed current + SoC | **Coulomb**: ΔSoC-weighted pooled `Σ\|∫I·dt\| / Σ(\|ΔSoC\|/100)` per month → robust-isotonic envelope | `soh.coulomb_capacity_monthly` → `capacity_to_soh` |
| **Euler, Montra** | remaining-capacity (Montra current is *unsigned* → no coulomb) | **BMS remaining-capacity**: `remCap ÷ (SoC/100)` at near-full SoC (95–100), monthly median → normalize to a beginning-of-life `cap0` → isotonic ≤ 100 | `euler_bms_soh.py`, `montra_features.py` |
| **Bajaj** | reported SoH only (no I/V) | **BMS-reported**: monthly median of `batterySoh`, kept non-increasing | `bajaj_model` path |

Coulomb knobs (`src/soh.py`): `MIN_ROWS=5`, `MIN_DSOC=2%`, `CAP_BOUNDS=(40,400) Ah`, `MIN_MONTH_SOC=30%`
pooled coverage, `SMOOTH_WIN=5` months, `MAX_DROP=6 pp/month` (bigger monthly jumps are artifacts, capped).
SoH is **anchored 100 % at registration** (`capacity_to_soh(reg=…)`); with no reg date it falls back to
**first-telemetry** anchoring (`used_reg=False`) — the first observed point then sits at ~100 %.

### 4 · Features — `features.electrical_features()` + assembly
Per `(vin, month)`: `ah_throughput` (∫|I|·dt), `cur_abs_mean`, `cur_abs_p95`, `cur_chg_mean`,
`cur_dis_mean`, `soc_mean`, `frac_soc_high/low`, `volt_mean/min/max`, `dod_mean`; then joined with
`odo_max`, `temp_mean/max` (raw aggregates), `km_month` (odo diff, clipped ≥ 0), `cum_ah`/`cum_km`
(cumsum), `inv_sqrt_age = 1/√(age+1)`, `soh_deficit = 100 − soh`, `age_months`.

### 5 · Output — the `featengg` schema (**one row per vin-month; keep identical across OEMs**)
```
vin, ymd, capacity_ah, n_sessions, tot_dsoc, age_months, used_reg, soh_raw, soh,
ah_throughput, cur_abs_mean, cur_dis_mean, cur_chg_mean, soc_mean, frac_soc_high, frac_soc_low,
volt_mean, volt_min, volt_max, n_rows_ic, cur_abs_p95, dod_mean, temp_mean, temp_max, dte_mean,
odo_max, km_month, cum_ah, cum_km, inv_sqrt_age, soh_deficit
```
Write two views: **`data/redshift/<oem>_featengg.parquet`** (store — keys on `ymd`) and
**`data/<oem>/features/feature_table.parquet`** (local — keys on `month`). Everything downstream
(`src/model.py`, `euler_model`, `bajaj_model`, `src/oem_train.py`, the dashboards) reads this schema
directly — **do not rename or drop columns.** Missing-per-OEM columns (e.g. `dte_mean`, cell imbalance)
stay as `NaN`; the LightGBM/XGBoost models are NaN-tolerant.

---

## Folder structure
```
EULER_RUL_MODEL/
├── .env / .env.example       # AWS creds + S3_BUCKET=oem-data-iot (gitignored .env); scratchpad redshift.env for the store
├── src/                      # pipeline code (run from repo root: `python src/<script>.py`)
│   ├── config.py             # ★ OEM registry — FLEET_WARRANTY, per-model warranty, quirks
│   ├── soh.py                # coulomb-counting SoH (cuDF/GPU or CPU)
│   ├── features.py           # electrical_features() — the monthly STRESS features
│   ├── <oem>_features.py     # end-to-end feature assembly (piaggio_features, montra_features, …)
│   ├── euler_bms_soh.py      # BMS remaining-capacity SoH (Euler) + recovery-aware clean label
│   ├── euler_train.py        # Euler deployed model (rate + trajectory) + registry
│   ├── oem_train.py          # ★ deployed model + registry for Mahindra/Bajaj/Piaggio/Montra
│   ├── euler_accept_gate.py  # coulomb-yardstick acceptance gate for target changes
│   ├── import_cohort.py      # cohort extractor:  python src/import_cohort.py <feed> <oem>
│   └── montra_sample.py      # day-sampled N-vehicle S3 extract (new-OEM POC)
├── models/<oem>/             # latest.pkl (gitignored) + registry.json / diagnostics.json (committed)
├── dashboard/                # Streamlit apps (learn_ml teaching · battery_status customer · …)
├── docs/                     # PROCEDURE.md · OEM_MODELING_PLAYBOOK.md · SOH_RUL_TECHNICAL_REPORT.md
└── data/                     # gitignored working data
    ├── redshift/<oem>_featengg.parquet   # ← the feature-generation OUTPUT the whole pipeline runs on
    ├── manifests/                        # per-vehicle data-quality + cohort manifests
    └── <oem>/{intellicar,feed,features,…}
```

## Add a new OEM
1. **Inspect the S3 feed** (sample a few files): signed current? remaining-capacity? reported SoH only? → picks the SoH method in §3.
2. **Extract**: `src/import_cohort.py <feed> <oem>` (full) or adapt `src/montra_sample.py` (day-sampled POC).
3. **Feature-engineer** with the matching template (`piaggio_features.py` coulomb · `montra_features.py` BMS-capacity) → `<oem>_featengg.parquet` in the schema above.
4. **Register + train**: add to `config.FLEET_WARRANTY` and `oem_train.CFG` (module/eol/warranty), then
   `python src/oem_train.py <oem>` → `models/<oem>/latest.pkl` + registry. Wire the dashboards' OEM configs.

Watch for: **unsigned current** (→ can't coulomb-count; use remaining-capacity, like Montra) and
**very short history** (a brand-new fleet like Montra sits ~flat with no decliners yet — the model is a
placeholder until it ages).

## Quick start (local)
```bash
pip install -r requirements.txt
cp .env.example .env               # fill AWS creds + S3_BUCKET
python src/montra_sample.py        # sample a new OEM from S3
python src/montra_features.py      # -> data/redshift/montra_featengg.parquet
python src/oem_train.py montra     # -> models/montra/latest.pkl + registry.json
streamlit run dashboard/learn_ml.py --server.port 8501   # teaching dashboard (all OEMs, side by side)
```

> Deeper references: [`docs/PROCEDURE.md`](docs/PROCEDURE.md) (runbook + lessons learned),
> [`docs/OEM_MODELING_PLAYBOOK.md`](docs/OEM_MODELING_PLAYBOOK.md) (methods & assumptions),
> [`docs/SOH_RUL_TECHNICAL_REPORT.md`](docs/SOH_RUL_TECHNICAL_REPORT.md) (full technical report).
> Notebooks (`notebooks/`) are exploratory analysis; the `src/` scripts above are the canonical pipeline.
