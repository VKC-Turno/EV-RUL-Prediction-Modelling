"""Euler feature-engineering — OUR methodology on the incremental Glue/Iceberg architecture.

Ports the preprocessing job to Turno's own SoH/feature logic while keeping the third party's MLOps
architecture (incremental read, Iceberg MERGE, cost-bounded). "All logic is ours": the per-vehicle work
calls src/euler_features.py directly (load_clean / bms_soh_monthly / monthly_features) via applyInPandas —
the same functions our models + dashboards consume, so there is no second implementation to drift.

SEPARATION OF CONCERNS (deliberate):
  * Feature engineering runs for ALL vehicles. Features are always emitted; `soh` (our BMS-capacity label)
    is null where a vehicle lacks a usable high-SoC signal — a missing LABEL never drops a vehicle's FEATURES.
  * The job's ONLY output is the `euler_featengg` feature store (one row per vin-month). It does NOT emit a
    "training set" or an "inference set" — which vehicles/rows are used for training vs inference is a
    point-in-time SELECTION made downstream (SageMaker training / serving), so a retrain a year later picks
    the then-current, in-service cohort without any change here. The train/val/test split + cohort gate live
    in the training step (src/oem_train.py::_split + data_quality.apply_quality), not in feature generation.

Output = our deployed 25-col monthly featengg (base SoH = bms_capacity -> isotonic; the recovery-aware clean
label / hybrid target / coulomb gate stay in the SageMaker layer).

Validated offline (no Glue/S3) by MLOps/glue/local_featengg_equivalence.py:
  incremental == batch, and == data/euler/features/feature_table.parquet on labelled rows (both 0.0 diff).

Cost model: incremental at ingest (only the new day's raw is parsed + appended). featengg is recomputed for
TOUCHED vehicles only, over their full clean-event history read from a durable Iceberg store (cheap columnar,
not tiny S3 files). Our SoH uses a per-vehicle adaptive window + isotonic + first-6-months baseline, which are
not additively-incremental, so the touched-vehicle recompute is the exact-correct floor (see README).

REQUIRED JOB PARAMETERS (besides --JOB_NAME):
    --datalake-formats            iceberg
    --additional-python-modules   scikit-learn        (bms_soh_monthly uses IsotonicRegression)
    --extra-py-files              s3://.../euler_features.py   (our module, importable on the executors)
    --process_date                2025-01-05
    --conf  spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog
            --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog
            --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/
            --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO
OPTIONAL: --raw_bucket --warehouse --reg_table
"""
import sys
from datetime import datetime

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.utils import AnalysisException
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

ARG_KEYS = ["JOB_NAME", "process_date", "raw_bucket", "warehouse", "reg_table"]
_defaults = dict(raw_bucket="oem-iot-data", warehouse="s3://rcs-mlops-data/iceberg/", reg_table="")
_present = [k for k in ARG_KEYS if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, _present)
for k, v in _defaults.items():
    args.setdefault(k, v)
if "process_date" not in args:
    raise ValueError("--process_date=YYYY-MM-DD is required.")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

OEM, CATALOG, DB = "euler", "glue_catalog", "turno_ml"
EVENTS_TABLE = f"{CATALOG}.{DB}.{OEM}_clean_events"
FEATENGG_TABLE = f"{CATALOG}.{DB}.{OEM}_featengg"           # the ONLY output: the feature store
PROCESS_DATE = datetime.strptime(args["process_date"], "%Y-%m-%d").date()

for k, v in {f"spark.sql.catalog.{CATALOG}": "org.apache.iceberg.spark.SparkCatalog",
             f"spark.sql.catalog.{CATALOG}.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
             f"spark.sql.catalog.{CATALOG}.warehouse": args["warehouse"],
             f"spark.sql.catalog.{CATALOG}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
             "spark.sql.shuffle.partitions": "64", "spark.sql.adaptive.enabled": "true",
             "spark.sql.adaptive.coalescePartitions.enabled": "true"}.items():
    spark.conf.set(k, v)

# our raw feed uses lowercase names; euler_features.py expects camelCase
RENAME = {"batterysoc": "batterySoc", "batterysoh": "batterySoh",
          "batteryremainingcapacity": "batteryRemainingCapacity", "batterycurrent": "batteryCurrent",
          "batteryvoltage": "batteryVoltage", "batterytemperature": "batteryTemperature",
          "cellimbalance": "cellImbalance", "eventat": "eventAt", "odometer": "odometer"}
NEEDED = ["vin", "eventAt", "batterySoc", "batterySoh", "batteryRemainingCapacity", "batteryCurrent",
          "batteryVoltage", "batteryTemperature", "cellImbalance", "odometer"]

FEATENGG_SCHEMA = StructType([
    StructField("vin", StringType()), StructField("ymd", StringType()), StructField("soh", DoubleType()),
    StructField("ah_throughput", DoubleType()), StructField("cur_abs_mean", DoubleType()),
    StructField("soc_mean", DoubleType()), StructField("volt_mean", DoubleType()),
    StructField("volt_min", DoubleType()), StructField("volt_max", DoubleType()),
    StructField("temp_mean", DoubleType()), StructField("temp_max", DoubleType()),
    StructField("odo_max", DoubleType()), StructField("n_rows", LongType()),
    StructField("cur_abs_p95", DoubleType()), StructField("cur_chg_mean", DoubleType()),
    StructField("cur_dis_mean", DoubleType()), StructField("frac_soc_high", DoubleType()),
    StructField("frac_soc_low", DoubleType()), StructField("imbalance_mean", DoubleType()),
    StructField("dod_mean", DoubleType()), StructField("crate_p95", DoubleType()),
    StructField("age_months", DoubleType()), StructField("km_month", DoubleType()),
    StructField("cum_ah", DoubleType()), StructField("cum_km", DoubleType()),
])
OUT_COLS = [f.name for f in FEATENGG_SCHEMA.fields]
REQ = ("batterySoc", "batterySoh", "batteryRemainingCapacity", "batteryCurrent",
       "batteryVoltage", "batteryTemperature", "cellImbalance", "odometer")


def vin_featengg(pdf):
    """applyInPandas UDF — one vehicle's full clean-event history -> its monthly featengg rows, for ALL
    vehicles. Calls OUR functions (euler_features). FEATURES are always emitted; `soh` is null when the
    BMS-capacity method has no usable high-SoC signal for this vehicle (a missing label never drops it)."""
    import numpy as np
    import pandas as pd
    import euler_features as ef            # our module (shipped via --extra-py-files)

    vin = str(pdf["vin"].iloc[0])
    reg = pd.to_datetime(pdf["reg_date"].iloc[0]) if "reg_date" in pdf and pdf["reg_date"].notna().any() else None
    for c in REQ:                          # some vehicles/feeds lack a channel -> keep the column, values null
        if c not in pdf.columns:
            pdf[c] = np.nan
    df = ef.load_clean(pdf)
    feat = ef.monthly_features(df)         # the base: features for every vehicle-month
    if feat is None or not len(feat):
        return pd.DataFrame(columns=OUT_COLS)
    soh = ef.bms_soh_monthly(df)           # our base SoH label; None when no usable high-SoC capacity signal
    m = feat.merge(soh, on="month", how="left") if soh is not None else feat.assign(soh=np.nan)
    m = m.sort_values("month")
    base = reg if (reg is not None and pd.notna(reg) and reg <= m["month"].iloc[0]) else m["month"].iloc[0]
    m["age_months"] = ((m["month"] - base).dt.days / 30.4).round(1)
    m["cur_chg_mean"] = m["cur_chg_mean"].fillna(0.0)
    m["cur_dis_mean"] = m["cur_dis_mean"].fillna(0.0)
    m["km_month"] = m["odo_max"].diff().clip(lower=0).fillna(0.0)
    m["cum_ah"] = m["ah_throughput"].cumsum()
    m["cum_km"] = m["km_month"].cumsum()
    m["vin"] = vin
    m["ymd"] = m["month"].dt.strftime("%Y-%m-%d")
    for c in OUT_COLS:
        if c not in m.columns:
            m[c] = np.nan
    m["n_rows"] = pd.to_numeric(m["n_rows"], errors="coerce").fillna(0).astype("int64")
    return m[OUT_COLS]


def ensure_table(df, table, partition=None):
    if not spark.catalog.tableExists(table):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DB}")
        w = (df.limit(0).writeTo(table).using("iceberg")
             .tableProperty("write.merge.mode", "merge-on-read"))
        if partition is not None:
            w = w.partitionedBy(F.months(partition))
        w.create()
        logger.info(f"created {table}")


def merge_upsert(df, table, keys):
    v = "_stg_" + table.split(".")[-1]
    df.createOrReplaceTempView(v)
    on = " AND ".join(f"t.{k}=s.{k}" for k in keys)
    spark.sql(f"MERGE INTO {table} t USING {v} s ON {on} "
              f"WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")


# ── 1) ingest the new day's raw -> euler_clean_events ─────────────────────────────────────
path = (f"s3://{args['raw_bucket']}/battery-oem-data/parquet/{OEM}/vehicle-data/"
        f"year={PROCESS_DATE.year}/month={PROCESS_DATE.month:02d}/day={PROCESS_DATE.day:02d}/")
try:
    raw = spark.read.parquet(path)
except AnalysisException:
    logger.warn(f"partition absent: {path} — nothing to do")
    job.commit(); sys.exit(0)

raw = raw.filter(F.col("vin").isNotNull() & (F.trim(F.col("vin")) != ""))
for lo, cc in RENAME.items():
    if lo in raw.columns and cc not in raw.columns:
        raw = raw.withColumnRenamed(lo, cc)
events = (raw.select([F.col(c) for c in NEEDED if c in raw.columns])
             .withColumn("eventAt", F.col("eventAt").cast("long"))
             .withColumn("month", F.date_trunc("month", F.to_timestamp(F.col("eventAt") / 1000))))
if len(events.take(1)) == 0:
    logger.warn("no valid rows for the day"); job.commit(); sys.exit(0)

ensure_table(events, EVENTS_TABLE, partition="month")
merge_upsert(events, EVENTS_TABLE, keys=["vin", "eventAt"])
logger.info(f"MERGED new-day events into {EVENTS_TABLE}")

# ── 2) recompute featengg for TOUCHED vehicles over their full clean-event history ────────
# (touched-only = which vehicles get RECOMPUTED this run; every reporting vehicle is covered over time.
#  No vehicle is dropped for lacking SoH — features are emitted for all; soh is null where unavailable.)
touched = events.select("vin").distinct()
hist = spark.table(EVENTS_TABLE).join(F.broadcast(touched), "vin")
if args["reg_table"]:
    reg = spark.table(args["reg_table"]).select("vin", F.col("reg_date"))
    hist = hist.join(F.broadcast(reg), "vin", "left")
else:
    hist = hist.withColumn("reg_date", F.lit(None).cast("timestamp"))

featengg = hist.groupBy("vin").applyInPandas(vin_featengg, schema=FEATENGG_SCHEMA)
featengg = featengg.filter(F.col("ymd").isNotNull())
ensure_table(featengg, FEATENGG_TABLE)
merge_upsert(featengg, FEATENGG_TABLE, keys=["vin", "ymd"])
logger.info(f"MERGED featengg into {FEATENGG_TABLE} (feature store — all vehicles, soh nullable)")

# Downstream (NOT here): the SageMaker training step selects the cohort from euler_featengg at run time
# (in-service + labelled + data-quality gate) and does the train/val/test split; serving reads latest-per-vin.
job.commit()
