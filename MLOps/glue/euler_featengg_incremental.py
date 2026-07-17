"""Euler feature-engineering — OUR methodology, incremental Glue/Iceberg, TWO-TIER (no full-history scan).

Ports preprocessing to Turno's SoH/feature logic on the third party's MLOps architecture. "All logic is
ours": the per-vehicle work calls src/euler_features.py via applyInPandas — the same functions our models +
dashboards consume. Output = our deployed 25-col monthly featengg (base SoH = bms_capacity -> isotonic).

TWO-TIER design (so each run's I/O is bounded, not O(full history)):
  euler_clean_events  raw cleaned events, partitioned by month. Ingest MERGEs the new day(s).
  euler_monthly       per (vin, month): month-LOCAL features (euler_features.monthly_features) + the month's
                      high-SoC full_cap samples (euler_features.hi_full_cap). Recomputed only for the
                      AFFECTED month(s), read month-pruned from euler_clean_events -> no full raw scan.
  euler_featengg      the 25-col feature store. STAGE C recomputes the CROSS-month SoH
                      (euler_features.soh_from_hi_full_cap) from the small persisted monthly samples + the
                      cumulative/age assembly -> one row per vin-month. All vehicles; soh null where no signal.
Every MERGE carries the partition column (month) in its ON clause so Iceberg prunes the target.

Validated offline (no Glue/S3) by MLOps/glue/local_featengg_equivalence.py: incremental == batch,
two-tier == single-pass, and == data/euler/features/feature_table.parquet (all 0.0 on labelled per-month cols).

Self-healing: a 1-row Iceberg watermark (euler_ingest_state) + catchup.plan_days catch up from the last
processed day, so a failed/skipped day is auto-recovered (MERGE is idempotent). See README.

REQUIRED JOB PARAMETERS (besides --JOB_NAME):
    --datalake-formats            iceberg
    --additional-python-modules   scikit-learn
    --extra-py-files              s3://.../euler_features.py,s3://.../catchup.py
    --process_date                2025-01-05   (OPTIONAL — defaults to yesterday UTC; scheduled runs omit it)
    --conf  spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog
            --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog
            --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/
            --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO
OPTIONAL: --raw_bucket --warehouse --reg_table --lookback_days (1) --max_catchup_days (14)
"""
import sys
from datetime import datetime, timedelta, timezone

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, LongType, DateType,
                               TimestampType, ArrayType)

from catchup import plan_days

ARG_KEYS = ["JOB_NAME", "process_date", "raw_bucket", "warehouse", "reg_table",
            "lookback_days", "max_catchup_days"]
_defaults = dict(raw_bucket="oem-iot-data", warehouse="s3://rcs-mlops-data/iceberg/", reg_table="",
                 lookback_days="1", max_catchup_days="14")
_present = [k for k in ARG_KEYS if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, _present)
for k, v in _defaults.items():
    args.setdefault(k, v)
if "process_date" not in args or not args["process_date"]:
    args["process_date"] = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

OEM, CATALOG, DB = "euler", "glue_catalog", "turno_ml"
EVENTS_TABLE = f"{CATALOG}.{DB}.{OEM}_clean_events"
MONTHLY_TABLE = f"{CATALOG}.{DB}.{OEM}_monthly"
FEATENGG_TABLE = f"{CATALOG}.{DB}.{OEM}_featengg"           # the feature store (all vehicles, soh nullable)
STATE_TABLE = f"{CATALOG}.{DB}.{OEM}_ingest_state"          # 1-row watermark
PROCESS_DATE = datetime.strptime(args["process_date"], "%Y-%m-%d").date()
LOOKBACK_DAYS = int(args["lookback_days"])
MAX_CATCHUP_DAYS = int(args["max_catchup_days"])

for k, v in {f"spark.sql.catalog.{CATALOG}": "org.apache.iceberg.spark.SparkCatalog",
             f"spark.sql.catalog.{CATALOG}.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
             f"spark.sql.catalog.{CATALOG}.warehouse": args["warehouse"],
             f"spark.sql.catalog.{CATALOG}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
             "spark.sql.shuffle.partitions": "64", "spark.sql.adaptive.enabled": "true",
             "spark.sql.adaptive.coalescePartitions.enabled": "true"}.items():
    spark.conf.set(k, v)

RENAME = {"batterysoc": "batterySoc", "batterysoh": "batterySoh",
          "batteryremainingcapacity": "batteryRemainingCapacity", "batterycurrent": "batteryCurrent",
          "batteryvoltage": "batteryVoltage", "batterytemperature": "batteryTemperature",
          "cellimbalance": "cellImbalance", "eventat": "eventAt", "odometer": "odometer"}
NEEDED = ["vin", "eventAt", "batterySoc", "batterySoh", "batteryRemainingCapacity", "batteryCurrent",
          "batteryVoltage", "batteryTemperature", "cellImbalance", "odometer"]
REQ = ("batterySoc", "batterySoh", "batteryRemainingCapacity", "batteryCurrent",
       "batteryVoltage", "batteryTemperature", "cellImbalance", "odometer")

# month-local aggregates (euler_features.monthly_features) — the euler_monthly store
FEAT_COLS = ["ah_throughput", "cur_abs_mean", "soc_mean", "volt_mean", "volt_min", "volt_max",
             "temp_mean", "temp_max", "odo_max", "n_rows", "cur_abs_p95", "cur_chg_mean", "cur_dis_mean",
             "frac_soc_high", "frac_soc_low", "imbalance_mean", "dod_mean", "crate_p95"]
_dbl = lambda n: StructField(n, DoubleType())
MONTHLY_SCHEMA = StructType(
    [StructField("vin", StringType()), StructField("month", TimestampType())]
    + [(_dbl(c) if c != "n_rows" else StructField("n_rows", LongType())) for c in FEAT_COLS]
    + [StructField("fullcap_hi", ArrayType(DoubleType()))])
MONTHLY_COLS = [f.name for f in MONTHLY_SCHEMA.fields]

# the feature store: our deployed 25 columns + a `month` partition column (consumers use the 25)
FEATENGG_SCHEMA = StructType(
    [StructField("vin", StringType()), StructField("ymd", StringType()),
     StructField("month", TimestampType()), _dbl("soh")]
    + [(_dbl(c) if c != "n_rows" else StructField("n_rows", LongType())) for c in FEAT_COLS]
    + [_dbl("age_months"), _dbl("km_month"), _dbl("cum_ah"), _dbl("cum_km")])
FEATENGG_COLS = [f.name for f in FEATENGG_SCHEMA.fields]


def monthly_udf(pdf):
    """STAGE B — one (vin, month)'s events -> a euler_monthly row: month-local features + this month's
    high-SoC full_cap samples. Calls our euler_features.monthly_features / hi_full_cap verbatim."""
    import numpy as np
    import pandas as pd
    import euler_features as ef
    vin = str(pdf["vin"].iloc[0])
    for c in REQ:
        if c not in pdf.columns:
            pdf[c] = np.nan
    df = ef.load_clean(pdf)
    feat = ef.monthly_features(df)
    if feat is None or not len(feat):
        return pd.DataFrame(columns=MONTHLY_COLS)
    row = feat.iloc[[0]].copy()
    row["vin"] = vin
    row["fullcap_hi"] = [ef.hi_full_cap(df)["full_cap"].astype(float).tolist()]
    for c in MONTHLY_COLS:
        if c not in row.columns:
            row[c] = np.nan
    row["n_rows"] = pd.to_numeric(row["n_rows"], errors="coerce").fillna(0).astype("int64")
    return row[MONTHLY_COLS]


def featengg_udf(pdf):
    """STAGE C — one vehicle's euler_monthly rows -> its featengg. Recomputes the CROSS-month SoH from the
    persisted per-month samples (euler_features.soh_from_hi_full_cap) + the cumulative/age assembly. Features
    are emitted for all vehicles; soh is null where there is no usable high-SoC signal."""
    import numpy as np
    import pandas as pd
    import euler_features as ef
    vin = str(pdf["vin"].iloc[0])
    reg = pd.to_datetime(pdf["reg_date"].iloc[0]) if "reg_date" in pdf and pdf["reg_date"].notna().any() else None
    mon = pdf.sort_values("month").reset_index(drop=True)
    parts = [pd.DataFrame({"month": r["month"], "full_cap": r["fullcap_hi"]})
             for _, r in mon.iterrows() if r["fullcap_hi"] is not None and len(r["fullcap_hi"])]
    hi_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["month", "full_cap"])
    soh = ef.soh_from_hi_full_cap(hi_all)
    m = mon.drop(columns=["fullcap_hi"])
    m = m.merge(soh, on="month", how="left") if soh is not None else m.assign(soh=np.nan)
    m = m.sort_values("month")
    if not len(m):
        return pd.DataFrame(columns=FEATENGG_COLS)
    base = reg if (reg is not None and pd.notna(reg) and reg <= m["month"].iloc[0]) else m["month"].iloc[0]
    m["age_months"] = ((m["month"] - base).dt.days / 30.4).round(1)
    m["cur_chg_mean"] = m["cur_chg_mean"].fillna(0.0)
    m["cur_dis_mean"] = m["cur_dis_mean"].fillna(0.0)
    m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
    m["cum_ah"] = m["ah_throughput"].cumsum()
    m["cum_km"] = m["km_month"].cumsum()
    m["vin"] = vin
    m["ymd"] = pd.to_datetime(m["month"]).dt.strftime("%Y-%m-%d")
    for c in FEATENGG_COLS:
        if c not in m.columns:
            m[c] = np.nan
    m["n_rows"] = pd.to_numeric(m["n_rows"], errors="coerce").fillna(0).astype("int64")
    return m[FEATENGG_COLS]


def ensure_table(df, table, partition=None):
    if not spark.catalog.tableExists(table):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DB}")
        w = (df.limit(0).writeTo(table).using("iceberg").tableProperty("write.merge.mode", "merge-on-read"))
        if partition is not None:
            w = w.partitionedBy(F.months(partition))
        w.create()
        logger.info(f"created {table}")


def merge_upsert(df, table, keys):
    v = "_stg_" + table.split(".")[-1]
    df.createOrReplaceTempView(v)
    on = " AND ".join(f"t.{k}=s.{k}" for k in keys)          # keys include the partition col -> Iceberg prunes
    spark.sql(f"MERGE INTO {table} t USING {v} s ON {on} "
              f"WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")


def read_watermark():
    if spark.catalog.tableExists(STATE_TABLE):
        rows = spark.table(STATE_TABLE).collect()
        if rows and rows[0]["last_processed_date"] is not None:
            return rows[0]["last_processed_date"]
    return None


def write_watermark(d):
    schema = StructType([StructField("last_processed_date", DateType())])
    spark.createDataFrame([(d,)], schema=schema).writeTo(STATE_TABLE).using("iceberg").createOrReplace()


def day_path(d):
    return (f"s3://{args['raw_bucket']}/battery-oem-data/parquet/{OEM}/vehicle-data/"
            f"year={d.year}/month={d.month:02d}/day={d.day:02d}/")


def day_exists(d):
    p = spark._jvm.org.apache.hadoop.fs.Path(day_path(d))
    return p.getFileSystem(sc._jsc.hadoopConfiguration()).exists(p)


# ── STAGE A: self-healing catch-up ingest -> euler_clean_events (partitioned by month) ─────
wm = read_watermark()
plan = plan_days(wm, PROCESS_DATE, LOOKBACK_DAYS, MAX_CATCHUP_DAYS)
if plan["clamped"]:
    logger.warn(f"catch-up clamped to {MAX_CATCHUP_DAYS}d; days before {plan['days'][0]} need an explicit "
                f"--process_date backfill (watermark={wm})")
present = [d for d in plan["days"] if day_exists(d)]
logger.info(f"watermark={wm} process_date={PROCESS_DATE} -> plan {plan['days'][0]}..{plan['days'][-1]} "
            f"({len(plan['days'])} day(s)); {len(present)} partition(s) present")
if not present:
    logger.warn("no raw partitions present in the planned window — nothing to ingest")
    job.commit(); sys.exit(0)

raw = spark.read.parquet(*[day_path(d) for d in present])
raw = raw.filter(F.col("vin").isNotNull() & (F.trim(F.col("vin")) != ""))
for lo, cc in RENAME.items():
    if lo in raw.columns and cc not in raw.columns:
        raw = raw.withColumnRenamed(lo, cc)
events = (raw.select([F.col(c) for c in NEEDED if c in raw.columns])
             .withColumn("eventAt", F.col("eventAt").cast("long"))
             .withColumn("month", F.date_trunc("month", F.to_timestamp(F.col("eventAt") / 1000))))
if len(events.take(1)) == 0:
    logger.warn("no valid rows in the planned window"); job.commit(); sys.exit(0)

ensure_table(events, EVENTS_TABLE, partition="month")
merge_upsert(events, EVENTS_TABLE, keys=["vin", "eventAt", "month"])   # +month -> prune the MERGE target
logger.info(f"MERGED events for {len(present)} day(s) into {EVENTS_TABLE}")

touched = events.select("vin").distinct()
affected_months = [r["month"] for r in events.select("month").distinct().collect()]

# ── STAGE B: recompute euler_monthly for AFFECTED (vin, month) — read month-PRUNED (no full raw scan) ──
month_events = (spark.table(EVENTS_TABLE).filter(F.col("month").isin(affected_months))
                .join(F.broadcast(touched), "vin"))
monthly = month_events.groupBy("vin", "month").applyInPandas(monthly_udf, schema=MONTHLY_SCHEMA)
ensure_table(monthly, MONTHLY_TABLE, partition="month")
merge_upsert(monthly, MONTHLY_TABLE, keys=["vin", "month"])
logger.info(f"MERGED euler_monthly for {len(affected_months)} month(s) into {MONTHLY_TABLE}")

# ── STAGE C: recompute featengg for touched vehicles from the small euler_monthly store ────
hist = spark.table(MONTHLY_TABLE).join(F.broadcast(touched), "vin")
if args["reg_table"]:
    reg = spark.table(args["reg_table"]).select("vin", F.col("reg_date"))
    hist = hist.join(F.broadcast(reg), "vin", "left")
else:
    hist = hist.withColumn("reg_date", F.lit(None).cast("timestamp"))
featengg = hist.groupBy("vin").applyInPandas(featengg_udf, schema=FEATENGG_SCHEMA)
featengg = featengg.filter(F.col("ymd").isNotNull())
ensure_table(featengg, FEATENGG_TABLE, partition="month")
merge_upsert(featengg, FEATENGG_TABLE, keys=["vin", "ymd", "month"])
logger.info(f"MERGED featengg into {FEATENGG_TABLE} (feature store — all vehicles, soh nullable)")

# ── advance the watermark only after a successful featengg MERGE and only if the target day was present
if plan["advance_to"] is not None and PROCESS_DATE in present:
    write_watermark(PROCESS_DATE)
    logger.info(f"watermark advanced to {PROCESS_DATE}")

# Downstream (NOT here): SageMaker training selects the cohort from euler_featengg at run time and does the
# train/val/test split; serving reads latest-per-vin. The 25 deployed columns are FEATENGG_COLS minus `month`.
job.commit()
