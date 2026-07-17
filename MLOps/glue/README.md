# Glue preprocessing — incremental & cost-optimised

`euler_preprocessing_incremental.py` is the production version of the third party's static prototype (the
`DataPreproccessing job`, built against the small static sample we shared: `year=2025/month=01/day=01..05`).
Same cleaning + feature logic, **21-column output reproduced exactly** — but every daily run costs
**O(new data + touched vehicles)**, never O(full history).

## The cost design in one idea

The prototype (and my first incremental cut) reintroduced full-history work in three places: it rewrote the
whole training snapshot each run, full-scanned for latest-per-VIN, and — because the two VIN-level
degradation rates were stored on *every* per-date row — a new day rewrote historical rows across every month
partition (write amplification). The fix is to **normalise**:

| Table | Grain | Written each run |
|---|---|---|
| `euler_daily` | (vin, event_date) | today's rows only (current-month partition) |
| `euler_vin_stats` | **vin** (1 row) | touched vins only — holds `soh_degradation_per_day/_per_km`, first-date, counters |
| `euler_features_daily` | (vin, event_date) | today's rows only; historical rows are **immutable** |
| `euler_latest` | **vin** (1 row) | touched vins only — the inference snapshot |

Because the degradation rates live in `euler_vin_stats` (not on every row), a new day **never rewrites
historical feature rows** → no cross-month write amplification. The original 21-column shape is rebuilt by a
read-time join `euler_features_daily ⋈ euler_vin_stats`.

## What each run does (all O(new + touched))

1. Read **only** the new day's partition (`--process_date`); skip the run entirely if it's empty/absent.
2. Clean + daily-aggregate → `MERGE` into `euler_daily` (current-month partition, merge-on-read).
3. `euler_vin_stats`: aggregate the **touched** vins' history → 1 row/VIN, `MERGE` upsert.
4. `euler_features_daily`: compute **today's** row from `vin_stats` (expanding features) + a bounded N-day
   window (rolling features); `MERGE` today's rows only.
5. `euler_latest`: today's row + rates → `MERGE` per touched VIN; JSON written with **dynamic partition
   overwrite** so silent vehicles' snapshots are untouched.
6. Training snapshot: **not** written on the daily path. Consumers read `euler_features_daily ⋈
   euler_vin_stats`; a Parquet export runs only with `--emit_training_snapshot=true` (i.e. when a training
   run needs fresh data).

The only remaining history-scaled operation is the *read* in step 3 (touched vins' history for exact
soh_max/min) — a cheap aggregation with a tiny write. All full-table **scans and rewrites are gone.**

## Job configuration

- `bookmark`: leave disabled (the parameterised read is already incremental); enable only if you switch
  `read_new_day` to the catalogued bookmarked read.
- **Enable Glue FLEX execution** — this is a non-SLA daily batch, so spare-capacity pricing applies cleanly.
- Keep the cluster small (2× G.1X); auto-scaling optional.
- Job parameters: `--datalake-formats iceberg`, `--process_date`, and the Iceberg catalog `--conf`s (your
  existing `glue_catalog.glue.skip-name-validation=true` confirms `glue_catalog` was already meant to be
  Iceberg). Optional: `--rolling_window_days` (default 7), `--emit_training_snapshot` (default false),
  `--raw_bucket`, `--warehouse`, `--training_output`, `--inference_output`.
- Add a **weekly Iceberg compaction** job (`CALL glue_catalog.system.rewrite_data_files` +
  `expire_snapshots`) — merge-on-read writes small delete/data files that compaction folds back for cheap reads.

> **Verify the bucket:** prototype reads `s3://oem-iot-data/...`; the rest of the stack uses `oem-data-iot`.
> Set `--raw_bucket` for account `894429711714`.

## Idempotency & ordering (important)

- `euler_daily` and `euler_vin_stats` are **order-independent** (MERGE / pure aggregation) — re-running any day
  is safe and self-healing.
- The per-date **expanding** features (`cumulative_distance`, `avg_daily_distance`) are computed as *as-of the
  latest day in history*, which is correct **only when days are processed in chronological order**. Forward
  daily runs satisfy this automatically; **backfill must run oldest→newest** (the loop below). Re-running the
  *latest* day is idempotent. To correct a *mid-history* day, reprocess that day **and all following days in
  order** (or rebuild the affected vins).

## Bootstrap / backfill

```bash
for d in 2025-01-01 2025-01-02 2025-01-03 2025-01-04 2025-01-05; do   # oldest -> newest
  aws glue start-job-run --job-name euler-preprocessing-incremental \
    --arguments '{"--process_date":"'"$d"'"}'
done
```

## Schedule (production)

EventBridge Scheduler → `start-job-run` daily with `--process_date=yesterday`. Generalises to the other OEMs
by parameterising `OEM`, the sensor-range table, and the SoH column — same registry idea as
`../model-build/pipelines/common/config.py`.
