# Glue preprocessing — incremental

`euler_preprocessing_incremental.py` is the production-incremental version of the third party's static
prototype (the `DataPreproccessing job` built against the small static sample we shared —
`year=2025/month=01/day=01..05`). Same cleaning + feature logic (**21 output columns unchanged**), made
incremental in read, state, and write.

## What changed vs the prototype

| Prototype (static) | This job (incremental) |
|---|---|
| Hardcoded `RAW_YEAR/MONTH/DAYS` window | `--process_date` job param (scheduler passes the new day) |
| `spark.read.parquet(fixed paths)` | reads only the new day's partition (or catalog + bookmark option) |
| Trend features over the 5-day read | trend features over each touched VIN's **full history** (lifetime-correct) |
| Training write = overwrite of the window | daily rows **MERGE-upserted** into Iceberg `euler_daily` (idempotent) |
| — | features **MERGE-upserted** into Iceberg `euler_features` |
| Training snapshot = current window only | training snapshot = **full durable table** (overwrite, complete + idempotent) |
| Inference = latest within the window | inference = latest per VIN over the **full table** (silent vehicles keep their real snapshot) |

The `append` dedupe problem they hit is gone: `MERGE ON (vin, event_date)` makes re-running a day a no-op.

## Job configuration diffs (from their `DataPreproccessing job.json`)

- `bookmark`: `job-bookmark-disable` → **`job-bookmark-enable`** (only needed if you switch `_read_new_days`
  to the catalogued bookmarked read; the parameterised-partition read is already incremental without it).
- Add job parameters:
  - `--datalake-formats = iceberg`
  - `--process_date = 2025-01-05` (scheduler overrides per run)
  - `--raw_bucket`, `--warehouse`, `--training_output`, `--inference_output` (defaults baked in; override as needed)
  - `--conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/ --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO`
  - (their existing `spark.sql.catalog.glue_catalog.glue.skip-name-validation=true` confirms `glue_catalog`
    was already intended as an Iceberg catalog — keep it.)

> **Verify the bucket:** the prototype reads `s3://oem-iot-data/...`; your pipeline elsewhere uses
> `oem-data-iot`. Confirm which is correct for account `894429711714` and set `--raw_bucket` accordingly.

## Durable tables (created automatically on first run)

- `glue_catalog.turno_ml.euler_daily` — one row per (vin, event_date), Sections 1–4 daily aggregates.
- `glue_catalog.turno_ml.euler_features` — one row per (vin, event_date), Section 5–6 trend features.

Both partitioned by `months(event_date)`. Training reads a Parquet snapshot of `euler_features`; models can
also read the Iceberg table directly.

## Bootstrap / backfill

The lifetime features are only correct once history exists in `euler_daily`. To seed it, run the job once
per historical day in order (a simple loop over dates, or a Glue workflow) before switching to daily
incremental. Because writes are MERGE-idempotent, re-running any day is safe.

```bash
# backfill example
for d in 2025-01-01 2025-01-02 2025-01-03 2025-01-04 2025-01-05; do
  aws glue start-job-run --job-name euler-preprocessing-incremental \
    --arguments '{"--process_date":"'"$d"'"}'
done
```

## Schedule (production)

EventBridge Scheduler → `start-job-run` daily with `--process_date` = yesterday. Each run parses one new
day, upserts it, recomputes trend features for the vehicles that moved, and refreshes the training +
inference snapshots.

## Notes / limitations

- `first_vehicle_date`/`vehicle_age_days` use the earliest **observed** day in `euler_daily`. If a true
  registration date is available, join it in for exact age (same caveat as the prototype, now over full
  history instead of a 5-day window).
- Cross-day boundary: a SoC change spanning midnight isn't counted in `estimated_cycle_count` (daily grain).
  Negligible and identical to the prototype's behaviour.
- Generalises to the other OEMs by parameterising `OEM`, the sensor range table, and the SoH column — the
  same per-OEM registry idea as `../model-build/pipelines/common/config.py`.
