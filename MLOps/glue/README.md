# Glue preprocessing — incremental & cost-optimised

Two jobs, both on the third party's incremental Glue/Iceberg **architecture** (parameterised read, `MERGE`
upserts, cost-bounded, SageMaker-ready outputs). They differ in whose **logic** runs:

| Job | Logic | Output | Status |
|---|---|---|---|
| **`euler_featengg_incremental.py`** | **Ours** — calls `src/euler_features.py` (bms-capacity SoH + our features) | our deployed **25-col monthly featengg** | **canonical — deploy this** |
| `euler_preprocessing_incremental.py` | Third party's (as received) | their 21-col daily table | reference only |

> Decision (recorded): base SoH in Glue (bms_capacity → isotonic); the recovery-aware clean label, hybrid
> target, and coulomb acceptance gate stay in the SageMaker/model-build layer. Grain = monthly featengg.

---

## `euler_featengg_incremental.py` — our methodology (canonical)

**All logic is ours, in one place.** The per-vehicle work calls `src/euler_features.py`
(`load_clean` / `bms_soh_monthly` / `monthly_features`) via `applyInPandas` — the *same* functions our models
and dashboards already consume, so there is no second implementation to drift. Output is our deployed
**25-column monthly featengg** (`vin, ymd, soh, ah_throughput, cur_abs_mean, …, cum_ah, cum_km`), with
`soh` = BMS remaining-capacity → isotonic.

### Separation of concerns (why there's no train/inference output)
Feature engineering runs for **all vehicles** — features are always emitted; `soh` (our label) is simply
**null where a vehicle has no usable high-SoC signal**. A missing label never drops a vehicle's features.
The job's **only output is the `euler_featengg` feature store** (one row per vin-month). It does **not**
materialise a "training set" or an "inference set": *which* vehicles/rows are used for training vs serving is
a **point-in-time selection made downstream** (SageMaker training / serving), so a retrain a year later picks
the then-current, in-service, sufficiently-labelled cohort **without any change here**. The train/val/test
split + cohort gate live in the training step (`src/oem_train.py::_split` + `data_quality.apply_quality`).

### What each run does
1. Read **only** the new day's raw partition (`--process_date`); skip if absent. Rename the raw lowercase
   columns to the camelCase our module expects.
2. `MERGE` the day's events into Iceberg **`euler_clean_events`** (idempotent on `(vin, eventAt)`).
3. For **touched** vehicles (those reporting that day), read their full clean-event history back from
   `euler_clean_events`, run our `vin_featengg` per vehicle (`groupBy("vin").applyInPandas`), and `MERGE`
   into **`euler_featengg`** on `(vin, ymd)`. Touched-only = which vehicles get *recomputed* this run; every
   reporting vehicle is covered over time, and none is dropped for lacking a label.

> **Cumulative features fixed:** because FE now runs over all months, `cum_ah` / `cum_km` / `km_month`
> cumulate across *every* month. The old labeled-only assembly (inner-join in `euler_features.main()`) skipped
> months without a SoH reading and thus **under-counted** throughput/distance — the feature store corrects
> this. (Per-month features + `soh` are byte-identical to `euler_features`; see the offline check.)

### Cost model (honest)
Incremental at ingest — only the new day's raw is parsed and appended. featengg is recomputed for **touched
vehicles only**, over their full clean-event history read from a durable **columnar Iceberg** store (not the
tiny S3 raw files). Our SoH uses a per-vehicle **adaptive capacity window + isotonic fit + first-6-months
baseline**, which are *not* additively-incremental (a new reading can shift the vehicle's global median and
the isotonic fit), so recomputing a touched vehicle over its full history is the **exact-correct floor** — far
below full-fleet/full-history, but not O(1). *Optional further optimisation:* keep a per-`(vin, month)`
`full_cap`/aggregate table and refresh the global median periodically, so only the affected month re-reads
events; adds a second table and a small approximation window. Ship the exact version first.

### Packaging (job parameters)
- `--datalake-formats iceberg`
- `--additional-python-modules scikit-learn` (`bms_soh_monthly` uses `IsotonicRegression`)
- `--extra-py-files s3://…/euler_features.py` — our module, importable on the executors
- `--process_date`, the Iceberg catalog `--conf`s, optional `--reg_table` (for exact `age_months`),
  `--raw_bucket`, `--warehouse`
- Enable **Glue Flex** (spare-capacity pricing) — non-SLA daily batch.

> `src/euler_features.py` does an `os.chdir` at import (harmless here — we call only the pure functions,
> which don't use the working directory). Verify `--raw_bucket` for account `894429711714`
> (`oem-iot-data` vs `oem-data-iot`).

---

## Offline equivalence checks (no Spark/Glue/S3)

Both are validated against **local `data/euler/`** only — never a Glue run or an S3 scan.

- **`local_featengg_equivalence.py`** (our-methodology port) — proves, using our *actual* `euler_features`
  functions, that **incremental == batch** (0.0), and that per-month features + `soh` **== `feature_table.parquet`**
  (0.0 on labelled rows). Cumulative cols (`cum_ah`/`cum_km`/`km_month`) differ *by design* — the all-months
  fix above. Latest run: 3 vehicles, 122 rows, **PASS**.
- **`local_equivalence_check.py`** (as-received job) — proves incremental == batch for the third party's logic.

```bash
.venv/bin/python MLOps/glue/local_featengg_equivalence.py
.venv/bin/python MLOps/glue/local_equivalence_check.py
```

## Deploy

Job definition: **`euler_featengg_incremental.job.json`** — the Glue Studio config (like the third party's
`DataPreproccessing job.json`), with Flex on, the Iceberg `--conf`s, `--extra-py-files`, and
`--additional-python-modules scikit-learn`. Unlike theirs it does **not** embed the script inline — keep the
script in git and reference it from S3.

```bash
# 1. upload the script + our module to the Glue assets bucket
aws s3 cp MLOps/glue/euler_featengg_incremental.py s3://aws-glue-assets-894429711714-ap-south-1/scripts/
aws s3 cp src/euler_features.py                    s3://aws-glue-assets-894429711714-ap-south-1/scripts/

# 2. create the job — either import euler_featengg_incremental.job.json in Glue Studio, or via CLI:
aws glue create-job \
  --name euler-featengg-incremental \
  --role arn:aws:iam::894429711714:role/AWSGlueServiceRole-TurnoML \
  --command '{"Name":"glueetl","ScriptLocation":"s3://aws-glue-assets-894429711714-ap-south-1/scripts/euler_featengg_incremental.py","PythonVersion":"3"}' \
  --glue-version "5.1" --worker-type G.1X --number-of-workers 2 \
  --execution-class FLEX --execution-property '{"MaxConcurrentRuns":1}' \
  --default-arguments '{
    "--datalake-formats":"iceberg",
    "--additional-python-modules":"scikit-learn",
    "--extra-py-files":"s3://aws-glue-assets-894429711714-ap-south-1/scripts/euler_features.py",
    "--raw_bucket":"oem-iot-data",
    "--warehouse":"s3://rcs-mlops-data/iceberg/",
    "--enable-continuous-cloudwatch-log":"true","--enable-metrics":"true",
    "--enable-observability-metrics":"true","--enable-spark-ui":"true",
    "--spark-event-logs-path":"s3://aws-glue-assets-894429711714-ap-south-1/sparkHistoryLogs/",
    "--conf":"spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/ --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO --conf spark.sql.catalog.glue_catalog.glue.skip-name-validation=true"
  }'
```

## Schedule

The script defaults `--process_date` to **yesterday (UTC)**, so a plain daily trigger needs no dynamic
argument. Simplest is a **native Glue scheduled trigger**:

```bash
aws glue create-trigger --name euler-featengg-daily --type SCHEDULED \
  --schedule "cron(30 1 * * ? *)" --start-on-creation \
  --actions '[{"JobName":"euler-featengg-incremental"}]'          # 01:30 UTC daily -> processes yesterday
```

Prefer **EventBridge Scheduler** if you want retries/DLQ or to fan out across OEMs (target = Glue
`StartJobRun`; pass `--process_date` via the input if you need a specific day). Use **Glue Workflows / Step
Functions** once this chains into compaction → featengg → training.

**Backfill** (oldest → newest, so the isotonic/baseline history is built in order; MERGE-idempotent so re-runs
are safe):

```bash
for d in 2025-01-01 2025-01-02 2025-01-03 2025-01-04 2025-01-05; do
  aws glue start-job-run --job-name euler-featengg-incremental --arguments '{"--process_date":"'"$d"'"}'
done
```

## Logs & run history

The job hasn't been executed yet (validated offline only). When it runs, logs/metrics land in:

- **CloudWatch Logs** — `/aws-glue/jobs/output` (driver+executor stdout, incl. our `logger.info(...)`) and
  `/aws-glue/jobs/error`; continuous logging (`--enable-continuous-cloudwatch-log`) streams live under
  `/aws-glue/jobs/logs-v2/<job-run-id>`.
- **Run history / status / DPU-seconds** — `aws glue get-job-runs --job-name euler-featengg-incremental`
  (or the Runs tab in Glue Studio): state, duration, error string, worker count.
- **Spark UI / History Server** — event logs at the `--spark-event-logs-path` above; open via the Glue Spark
  History Server.
- **Observability metrics** — `--enable-observability-metrics` publishes to CloudWatch
  (`Glue/Observability`) for skew/OOM/throughput dashboards.

Generalises to the other OEMs by swapping the per-vehicle module (each OEM's `*_features` / SoH method) —
same registry idea as `../model-build/pipelines/common/config.py`.
