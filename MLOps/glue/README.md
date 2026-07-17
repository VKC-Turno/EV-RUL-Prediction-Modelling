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

### Two-tier design (no full-history scan)

Three Iceberg tables, all partitioned by **month**:

| Table | Grain | What it holds |
|---|---|---|
| `euler_clean_events` | event | cleaned raw events (ingest MERGEs the new day) |
| `euler_monthly` | (vin, month) | month-**local** features (`monthly_features`) + that month's high-SoC `full_cap` samples (`hi_full_cap`) |
| `euler_featengg` | (vin, month) | the 25-col feature store; cross-month SoH (`soh_from_hi_full_cap`) + cumulative/age assembly |

**What each run does:**
1. **Ingest** — read only the planned day partition(s) (`--process_date` + catch-up); `MERGE` into
   `euler_clean_events`.
2. **Stage B — `euler_monthly`** — recompute the **affected month(s)** for touched vehicles, reading
   `euler_clean_events` **month-pruned** (`WHERE month IN affected`). `monthly_features` and `hi_full_cap` are
   month-local (verified 0.0), so a month's row depends only on that month's events. `MERGE` on `(vin, month)`.
3. **Stage C — `euler_featengg`** — recompute touched vehicles' SoH from the **small** `euler_monthly` store
   (the persisted `full_cap` samples give the exact global-median adaptive window + isotonic fit) and assemble
   `age_months`/`km_month`/`cum_*`. `MERGE` on `(vin, ymd, month)`. All vehicles; `soh` null where no signal.

`euler_features.bms_soh_monthly` was refactored into `hi_full_cap` (month-local) ∘ `soh_from_hi_full_cap`
(cross-month) so the split is our *exact* code — the offline check confirms **two-tier == single-pass (0.0)**.

### Cost model

- **No full raw scan.** Stage B reads only the **affected month** partition(s) of `euler_clean_events`
  (month-partitioned → real pruning), not all history. The old version scanned the whole events table each
  run to reconstruct each vehicle's history; that is gone.
- Stage C reads the **small** `euler_monthly` store (one row per vin-month + a modest `full_cap` array),
  not raw events. It's currently a scan of that small table filtered to touched vehicles — partition
  `euler_monthly` by `bucket(N, vin)` if you later want that pruned too.
- Every `MERGE` carries the partition column (`month`) in its `ON` clause, so Iceberg prunes the target
  instead of scanning it.

> **Cumulative features:** `cum_ah`/`cum_km`/`km_month` cumulate across *every* month. The old labeled-only
> inner-join skipped months without a SoH reading and **under-counted**; the feature store fixes this.
> Per-month features + `soh` are byte-identical to `euler_features` (offline check).

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
  functions: **(A) incremental == batch**, **(C) two-tier (`euler_monthly` store) == single-pass**, and
  **(B) per-month features + `soh` == `feature_table.parquet`** — all **0.0**. Cumulative cols
  (`cum_ah`/`cum_km`/`km_month`) differ *by design* (all-months fix). Latest run: 3 vehicles, 122 rows, **PASS**.
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
# 1. upload the script + our modules to the Glue assets bucket
aws s3 cp MLOps/glue/euler_featengg_incremental.py s3://aws-glue-assets-894429711714-ap-south-1/scripts/
aws s3 cp MLOps/glue/catchup.py                    s3://aws-glue-assets-894429711714-ap-south-1/scripts/
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
    "--extra-py-files":"s3://aws-glue-assets-894429711714-ap-south-1/scripts/euler_features.py,s3://aws-glue-assets-894429711714-ap-south-1/scripts/catchup.py",
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

## Failure handling & self-healing catch-up

Every write is an **idempotent, atomic Iceberg MERGE** and featengg is a pure function of the clean-event
history, so **re-running any `--process_date` is always safe** (no dup, converges to the same state). Glue
also auto-retries once (`maxRetries: 1`).

On top of that the job is **self-healing**: a 1-row watermark table **`euler_ingest_state`** records the last
successfully-processed day. Each run plans its ingest with `catchup.plan_days(watermark, process_date,
lookback_days, max_catchup_days)`:

- **normal day** → just `process_date`;
- **after a failed/skipped day (or a multi-day outage)** → every day from `watermark+1` up to `process_date`,
  so the gap is **recovered automatically on the next run** — no day silently becomes a permanent hole;
- capped at **`max_catchup_days`** (default 14) so a long outage can't process months in one job; dropped older
  days are **logged**, never silently skipped (backfill them explicitly);
- **`lookback_days`** (default 1) optionally reprocesses the last N days each run to absorb late-arriving data.

The watermark advances **only after a successful featengg MERGE** and only when the target partition was
actually present, so a still-missing day is retried rather than skipped. The date logic is pure Python and
unit-tested offline:

```bash
.venv/bin/python MLOps/glue/local_catchup_check.py
```

**Per-vehicle isolation:** both `applyInPandas` UDFs wrap their body in try/except — a data error in one
`(vin, month)` / vehicle is logged to the executor stderr (`[SKIP monthly] …` / `[SKIP featengg] …`) and
skipped, so **one bad vehicle can't fail the whole run**. Systemic errors (e.g. a failed `import
euler_features`) are left to propagate — they *should* fail the run, not silently skip every vehicle. A
driver-side safety net **fails the run loudly if 0 of N touched vehicles produced any rows** (so a systemic
failure can't masquerade as a silent empty run), and logs a warning for a partial shortfall.

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

### Alerting on failure

So a run that exhausts retries (or the driver-side safety net) isn't silent, wire an **EventBridge rule** on
Glue job failure → **SNS**. The event pattern is in `euler_featengg_failure_pattern.json`:

```bash
# SNS topic + subscription (email / Slack / PagerDuty via a Lambda)
TOPIC=$(aws sns create-topic --name euler-featengg-alerts --query TopicArn --output text)
aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint battery.product@turno.club

# rule: FAILED/TIMEOUT for this job -> SNS
aws events put-rule --name euler-featengg-failed \
  --event-pattern file://MLOps/glue/euler_featengg_failure_pattern.json
aws events put-targets --rule euler-featengg-failed --targets "Id=1,Arn=$TOPIC"

# let EventBridge publish to the topic
aws sns set-topic-attributes --topic-arn "$TOPIC" --attribute-name Policy --attribute-value '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},
  "Action":"sns:Publish","Resource":"'"$TOPIC"'"}]}'
```

The job also `raise`s (→ run `FAILED`) if a systemic failure produced zero rows, so that case triggers this
alarm too rather than committing an empty feature store.

Generalises to the other OEMs by swapping the per-vehicle module (each OEM's `*_features` / SoH method) —
same registry idea as `../model-build/pipelines/common/config.py`.
