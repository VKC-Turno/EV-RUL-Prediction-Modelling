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

### What each run does
1. Read **only** the new day's raw partition (`--process_date`); skip if absent. Rename the raw lowercase
   columns to the camelCase our module expects.
2. `MERGE` the day's events into Iceberg **`euler_clean_events`** (idempotent on `(vin, eventAt)`).
3. For **touched** vehicles: read their full clean-event history back from `euler_clean_events`, run our
   `vin_featengg` per vehicle (`groupBy("vin").applyInPandas`), and `MERGE` into **`euler_featengg`** on
   `(vin, ymd)`.
4. `euler_latest`: latest month per touched vehicle → `MERGE` (+ JSON, dynamic partition overwrite) for
   inference.
5. Training snapshot only with `--emit_training_snapshot=true`; otherwise consumers read `euler_featengg`.

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
  `--emit_training_snapshot`, `--raw_bucket`, `--warehouse`, `--training_output`, `--inference_output`
- Enable **Glue Flex** (spare-capacity pricing) — non-SLA daily batch.

> `src/euler_features.py` does an `os.chdir` at import (harmless here — we call only the pure functions,
> which don't use the working directory). Verify `--raw_bucket` for account `894429711714`
> (`oem-iot-data` vs `oem-data-iot`).

---

## Offline equivalence checks (no Spark/Glue/S3)

Both are validated against **local `data/euler/`** only — never a Glue run or an S3 scan.

- **`local_featengg_equivalence.py`** (our-methodology port) — proves, using our *actual* `euler_features`
  functions, that **incremental == batch** *and* the output **== `data/euler/features/feature_table.parquet`**.
  Latest run: 3 vehicles, 84 rows, **both 0.0 diff, PASS**.
- **`local_equivalence_check.py`** (as-received job) — proves incremental == batch for the third party's logic.

```bash
.venv/bin/python MLOps/glue/local_featengg_equivalence.py
.venv/bin/python MLOps/glue/local_equivalence_check.py
```

## Bootstrap / backfill & schedule

featengg needs history for the isotonic/baseline to be meaningful — backfill **oldest → newest**, then
schedule daily with `--process_date=yesterday` (EventBridge → `start-job-run`). Because writes are
`MERGE`-idempotent, re-running a day is safe.

```bash
for d in 2025-01-01 2025-01-02 2025-01-03 2025-01-04 2025-01-05; do   # oldest -> newest
  aws glue start-job-run --job-name euler-featengg-incremental --arguments '{"--process_date":"'"$d"'"}'
done
```

Generalises to the other OEMs by swapping the per-vehicle module (each OEM's `*_features` / SoH method) —
same registry idea as `../model-build/pipelines/common/config.py`.
