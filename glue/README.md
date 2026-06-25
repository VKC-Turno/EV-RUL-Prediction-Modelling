# AWS Glue pipeline — feature extraction + prediction (full fleet, 3 OEMs)

Productionizes the local pipeline on AWS Glue. **Two job types**, parameterized per OEM, reusing the
repo's `src/` code unchanged.

```
raw S3 telemetry ──▶ [Glue Spark]  feature_extraction_job  ──▶  features S3
(per OEM feed)         (1 run per OEM, applyInPandas)            (oem=euler|mahindra|bajaj)
                                                                      │
persist_models  ──▶  models S3 ───────────────────────────────────┐  │
(periodic train)     (<oem>/latest.pkl)                            ▼  ▼
                                            [Glue Python Shell]  prediction_job  ──▶  predictions S3
                                            (1 run per OEM)        (SoH forecast + RUL + at-risk)
```

## Why this shape is the quickest at fleet scale
- **Feature extraction = Spark.** The bottleneck for the full fleet is *reading raw S3* (esp. Mahindra's
  ~70M tiny files). Spark distributes the read; the per-VIN math stays your **exact pandas code** via
  `groupBy("vin").applyInPandas(...)` — no rewrite, no divergence from local.
- **Prediction = Python Shell.** Feature tables are tiny (one row per vehicle-month), so a single node
  forecasts the whole fleet in seconds. Spark would be pure overhead here.
- **Models persisted to S3**, trained on a schedule — prediction never retrains, so it's fast and
  reproducible.

## Files
| File | Job type | Purpose |
|---|---|---|
| `feature_logic.py` | (library) | Per-VIN feature pipeline for all 3 OEMs; reuses `src/<oem>_features.py`, `src/soh.py`, `src/features.py`. |
| `feature_extraction_job.py` | Glue **Spark** | raw S3 → `applyInPandas(feature_logic)` → feature_table S3. One run per OEM. |
| `persist_models.py` | Glue **Python Shell** | Train + version the 3 models to S3 (quality-gated). Run periodically. |
| `prediction_job.py` | Glue **Python Shell** | feature_table + model → SoH forecast, RUL, at-risk → S3. One run per OEM. |

## Source feed per OEM (from `src/config.py`)
| OEM | Raw source (`--RAW_S3`) | SoH method | Note |
|---|---|---|---|
| bajaj | `…/parquet/bajaj/vehicle-data/` | reported `essBmsSohcEstPercValue` | single source — fully implemented |
| euler | `…/parquet/euler/vehicle-data/` | BMS remaining-capacity (isotonic) | single source — fully implemented |
| mahindra | `…/parquet/intellicar/battery-data/` (filtered to Mahindra VINs) | coulomb counting (needs `current`) | **dual-source**: thermal/GPS/dte come from Mahindra's *native* feed — see the `TODO` in `feature_logic.mahindra()` before relying on those features. |

## One-time setup
1. **Package the code.** Zip `src/` **and** `glue/feature_logic.py` into `src.zip`, upload to S3:
   ```bash
   cd EULER_RUL_MODEL && zip -r src.zip src glue/feature_logic.py && aws s3 cp src.zip s3://<code-bucket>/glue/src.zip
   ```
2. **Mahindra compaction (do this first, once).** ~70M tiny files = slow Spark listing. Run a one-time
   Glue/Spark compaction that reads the raw intellicar Mahindra rows and rewrites larger parquet
   partitioned by `year=/month=` (or by VIN). Point `--RAW_S3` for Mahindra at the **compacted** path.
   (Bajaj ~400 rows/file and Euler are fine read directly.)
3. **Registration CSVs** on S3, one per OEM (`vin` + `regd_date`/`vehicle_registration_date`).
4. **Glue IAM role** with read on the data + code buckets and write on the features/models/predictions
   buckets.

## Create the jobs (CLI sketch)
```bash
# Spark feature job (Glue 4.0)
aws glue create-job --name feat-bajaj --role <GlueRole> \
  --command '{"Name":"glueetl","ScriptLocation":"s3://<code>/glue/feature_extraction_job.py","PythonVersion":"3"}' \
  --default-arguments '{
    "--extra-py-files":"s3://<code>/glue/src.zip",
    "--additional-python-modules":"xgboost,lightgbm,scikit-learn,pyarrow",
    "--OEM":"bajaj",
    "--RAW_S3":"s3://oem-data-iot/battery-oem-data/parquet/bajaj/vehicle-data/",
    "--REG_CSV_S3":"s3://<data>/reg/bajaj_reg.csv",
    "--OUT_S3":"s3://<data>/features/oem=bajaj/"}' --glue-version 4.0 --worker-type G.1X --number-of-workers 10

# Python-Shell prediction job
aws glue create-job --name predict-bajaj --role <GlueRole> \
  --command '{"Name":"pythonshell","ScriptLocation":"s3://<code>/glue/prediction_job.py","PythonVersion":"3.9"}' \
  --default-arguments '{
    "--extra-py-files":"s3://<code>/glue/src.zip",
    "--additional-python-modules":"pandas,pyarrow,xgboost,lightgbm,scikit-learn,s3fs",
    "--OEM":"bajaj",
    "--FEATURE_S3":"s3://<data>/features/oem=bajaj/",
    "--MODEL_S3":"s3://<data>/models/bajaj/latest.pkl",
    "--OUT_S3":"s3://<data>/predictions/oem=bajaj/"}' --max-capacity 1.0
```
(Repeat with `--OEM euler` / `mahindra`.)

## Orchestrate + schedule
Use a **Glue Workflow** with triggers: `persist_models` → (fan-out) 3× `feature_extraction` → 3×
`prediction`. Put it on a **scheduled trigger** (e.g. weekly `cron(0 3 ? * MON *)`) or fire on new-data
via EventBridge. Run order: **compact (Mahindra) → persist_models → feature jobs → prediction jobs.**

## Outputs (all Parquet on S3, Athena-queryable — crawl into the Glue Catalog)
- `features/oem=<oem>/` — feature_table, **one row per vehicle-month**.
- `predictions/oem=<oem>/predictions_<oem>.parquet` — **one row per vehicle** (the decision view):
  `current_soh`, `rul_months`, `at_risk_by_warranty` (P50<EoL by warranty), `at_risk_worstcase` (P10),
  `eol_pct`, `warranty_age_months`.
- `predictions/oem=<oem>/predictions_monthly_<oem>.parquet` — **vehicle × forecast-month** (the curves):
  `horizon_month`, `forecast_age_months`, `p10/p50/p90` SoH — the full trajectory in long, numeric form
  (no JSON), so `WHERE horizon_month=24 AND p50<80` works directly.

## Training options
`persist_models` trains on all quality-gated vehicles. Add `--DEGRADERS_ONLY true` to train on degraders
only — **default off and not recommended**: our A/B showed it gives no degrader-accuracy gain and hurts
the healthy fleet (+18–43% RMSE). The quality gate (`data_quality.apply_quality`) is always applied.

## Caveats / TODO
- **Mahindra dual-source** features (temp/GPS/dte) are stubbed — add the native-feed merge per
  `feature_logic.mahindra()` if your model's `FEATS` need them, else Mahindra trains on the coulomb+
  electrical subset.
- **Quality gate**: prediction and training both call `data_quality.apply_quality`, which reads
  `data/manifests/vehicle_data_quality.csv`. Ship that manifest in `src.zip` (or regenerate it in-job)
  so thin vehicles are excluded in AWS too.
- The `src/*_features.py` modules `os.chdir()` on import (harmless in Glue — they only affect the local
  file-based `main()`); guard it if your environment is strict.
- Validate on **one OEM (Bajaj)** end-to-end before fanning out to all three.
```
