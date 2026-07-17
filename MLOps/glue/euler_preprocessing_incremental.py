"""Euler battery preprocessing — INCREMENTAL, cost-optimised Glue ETL.

Every daily run costs O(new data + touched vehicles), never O(full history). Same cleaning + feature logic
as the third party's static prototype; the 21-column output is reproduced exactly (per-date features live in
`euler_features_daily`, the two VIN-level degradation rates in `euler_vin_stats`; a read-time join rebuilds
the original schema).

Cost design (what makes it cheap):
  * READ  — only the new day's raw partition (--process_date), guarded against an empty/absent partition.
  * DAILY — MERGE today's rows into Iceberg `euler_daily` (current-month partition only), merge-on-read.
  * STATE — the VIN-level degradation rates + first-date/counters are a 1-row-per-VIN table upserted for
            TOUCHED vins only. Because these are NOT stored on every per-date row, a new day never rewrites
            historical feature rows -> no cross-month write amplification (the big win).
  * FEATURES — today's per-date derived row is computed from `vin_stats` + a bounded N-day window; historical
            rows are immutable, written once. MERGE touches only the current-month partition.
  * INFERENCE — `euler_latest` upserted for touched vins only (JSON written with dynamic partition overwrite,
            so silent vehicles keep their existing snapshot untouched). No full-table latest-per-VIN scan.
  * TRAINING — no per-run snapshot. Consumers read `euler_features_daily` JOIN `euler_vin_stats` directly; a
            Parquet export is produced ONLY when --emit_training_snapshot=true (i.e. when a training run needs it).

REQUIRED JOB PARAMETERS (in addition to --JOB_NAME):
    --datalake-formats            iceberg
    --process_date                2025-01-05         (scheduler passes yesterday; run backfill days in order)
    --conf  spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog
            --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog
            --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/
            --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO
OPTIONAL: --raw_bucket --warehouse --training_output --inference_output --rolling_window_days --emit_training_snapshot
ALSO (job config, not code): enable Glue FLEX execution (spare-capacity pricing) — this is a non-SLA daily
batch; and keep the cluster at 2x G.1X (auto-scaling optional). See MLOps/glue/README.md.
"""
import sys
from datetime import datetime, timedelta

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql.utils import AnalysisException

from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lit, when, trim, from_unixtime, to_timestamp, to_date, year, month, dayofmonth,
    hex, lag, lead, avg, min, max, stddev, sum, count, abs, datediff, unix_timestamp,
    broadcast, months, struct,
)

############################################################
# Parameters
############################################################

ARG_KEYS = ["JOB_NAME", "process_date", "raw_bucket", "warehouse", "training_output",
            "inference_output", "rolling_window_days", "emit_training_snapshot"]
_defaults = dict(
    raw_bucket="oem-iot-data",
    warehouse="s3://rcs-mlops-data/iceberg/",
    training_output="s3://rcs-mlops-data/training/",
    inference_output="s3://rcs-mlops-data/inference_input/latest/",
    rolling_window_days="7",
    emit_training_snapshot="false",
)
_present = [k for k in ARG_KEYS if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, _present)
for k, v in _defaults.items():
    args.setdefault(k, v)
if "process_date" not in args:
    raise ValueError("--process_date=YYYY-MM-DD is required (the scheduler passes it).")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

OEM, CATALOG, DB = "euler", "glue_catalog", "turno_ml"
DAILY_TABLE = f"{CATALOG}.{DB}.{OEM}_daily"
STATS_TABLE = f"{CATALOG}.{DB}.{OEM}_vin_stats"
FEATURES_TABLE = f"{CATALOG}.{DB}.{OEM}_features_daily"
LATEST_TABLE = f"{CATALOG}.{DB}.{OEM}_latest"

PROCESS_DATE = datetime.strptime(args["process_date"], "%Y-%m-%d").date()
ROLL_DAYS = int(args["rolling_window_days"])
ROLL_START = str(PROCESS_DATE - timedelta(days=ROLL_DAYS - 1))
EMIT_SNAPSHOT = args["emit_training_snapshot"].lower() == "true"

spark.conf.set(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.warehouse", args["warehouse"])
spark.conf.set(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.shuffle.partitions", "64")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")   # only touched vins' JSON rewritten

############################################################
# Helpers
############################################################

MOR_PROPS = {"write.merge.mode": "merge-on-read",   # avoid rewriting whole partitions on daily upsert
             "write.update.mode": "merge-on-read",
             "write.delete.mode": "merge-on-read"}


def ensure_table(df, table, partition_col=None):
    if not spark.catalog.tableExists(table):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DB}")
        w = df.limit(0).writeTo(table).using("iceberg")
        for k, v in MOR_PROPS.items():
            w = w.tableProperty(k, v)
        if partition_col is not None:
            w = w.partitionedBy(months(partition_col))
        w.create()
        logger.info(f"created {table}")


def merge_upsert(df, table, keys):
    view = "_stg_" + table.split(".")[-1]
    df.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k}=s.{k}" for k in keys)
    spark.sql(f"MERGE INTO {table} t USING {view} s ON {on} "
              f"WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")


def read_new_day():
    """Read only the target day's partition; return None if the partition is absent/empty (skip the run)."""
    path = (f"s3://{args['raw_bucket']}/battery-oem-data/parquet/{OEM}/vehicle-data/"
            f"year={PROCESS_DATE.year}/month={PROCESS_DATE.month:02d}/day={PROCESS_DATE.day:02d}/")
    logger.info(f"reading {path}")
    try:
        d = spark.read.parquet(path)
    except AnalysisException:
        logger.warn(f"partition absent: {path} — nothing to do")
        return None
    return d if len(d.take(1)) else None

############################################################
# SECTION 1-2 : clean + feature-prep (new day)
############################################################

df = read_new_day()
if df is None:
    job.commit(); sys.exit(0)                    # no new data -> cheapest possible run

df = df.filter(col("vin").isNotNull() & (trim(col("vin")) != ""))
df = (df.withColumn("event_timestamp", to_timestamp(from_unixtime(col("eventat") / 1000)))
        .withColumn("event_date", to_date(col("event_timestamp"))))
df = df.filter(col("event_date") == lit(str(PROCESS_DATE)))   # drop timezone spillover to neighbour day

RANGE_COLUMNS = [
    ("batterysoh", "battery_soh", 70, 100), ("batterysoc", "battery_soc", 0, 100),
    ("odometer", "odometer_clean", 0, 125000), ("speed", "speed_clean", 0, 60),
    ("batterycurrent", "battery_current", -150, 150), ("batterytemperature", "battery_temperature", -20, 60),
    ("batteryremainingcapacity", "battery_remaining_capacity", 0, 210),
    ("controllertemperature", "controller_temperature", -20, 120),
    ("motortemperature", "motor_temperature", -20, 150), ("cellimbalance", "cell_imbalance", 0, 200),
    ("batteryvoltage", "battery_voltage", 0, 80),
]
for raw_col, alias, low, high in RANGE_COLUMNS:
    df = df.withColumn(alias, when(col(raw_col).between(low, high), col(raw_col)))

VEHICLE_MODE_MAP = {"30": 0, "31": 1, "32": 2, "33": 3, "34": 4, "35": 5, "45636F6E6F6D79": 6}
df = df.withColumn("vehicle_mode_hex", hex(col("vehiclemode")))
mode_expr = None
for hv, mv in VEHICLE_MODE_MAP.items():
    c = col("vehicle_mode_hex") == hv
    mode_expr = when(c, mv) if mode_expr is None else mode_expr.when(c, mv)
df = df.withColumn("vehicle_mode", mode_expr)

required = ["vin", "event_timestamp", "event_date", "battery_soh", "battery_soc", "odometer_clean",
            "speed_clean", "battery_current", "battery_temperature", "battery_remaining_capacity",
            "controller_temperature", "motor_temperature", "cell_imbalance", "battery_voltage", "vehicle_mode"]
df = df.na.drop(subset=required).select(
    "vin", "event_timestamp", "event_date", "battery_soh", "battery_soc", "odometer_clean", "speed_clean",
    "battery_current", "battery_temperature", "battery_remaining_capacity", "controller_temperature",
    "motor_temperature", "cell_imbalance", "battery_voltage", "vehicle_mode",
)
df.persist(StorageLevel.MEMORY_AND_DISK)

############################################################
# SECTION 3 : intra-day window features
############################################################

vin_window = Window.partitionBy("vin").orderBy("event_timestamp")
df = (df.withColumn("previous_soc", lag("battery_soc").over(vin_window))
        .withColumn("next_timestamp", lead("event_timestamp").over(vin_window)))
df = df.withColumn("duration_minutes",
                   (unix_timestamp("next_timestamp") - unix_timestamp("event_timestamp")) / 60
                   ).fillna({"duration_minutes": 0})
df = df.withColumn("cycle_difference",
                   when(col("previous_soc").isNotNull() & (abs(col("battery_soc") - col("previous_soc")) >= 1),
                        abs(col("battery_soc") - col("previous_soc"))).otherwise(0))
df = df.withColumn("running_minutes", when(col("vehicle_mode") == 3, col("duration_minutes")).otherwise(0))
df = df.withColumn("high_temp_minutes", when(col("battery_temperature") > 45, col("duration_minutes")).otherwise(0))

############################################################
# SECTION 4 : daily aggregation (new day) -> euler_daily
############################################################

daily_new = df.groupBy("vin", "event_date").agg(
    max("battery_soh").alias("current_soh"),
    (max("odometer_clean") - min("odometer_clean")).alias("daily_distance"),
    avg(when(col("vehicle_mode") == 3, col("speed_clean"))).alias("avg_speed"),
    max(when(col("vehicle_mode") == 3, col("speed_clean"))).alias("max_speed"),
    avg(when(col("vehicle_mode") == 3, col("battery_current"))).alias("avg_current"),
    max(when(col("vehicle_mode") == 3, col("battery_current"))).alias("max_current"),
    avg("battery_temperature").alias("avg_battery_temperature"),
    max("battery_temperature").alias("max_battery_temperature"),
    max("cell_imbalance").alias("max_cell_imbalance"),
    (sum("cycle_difference") / 100).alias("estimated_cycle_count"),
    (max("battery_soc") - min("battery_soc")).alias("soc_variation"),
    sum("high_temp_minutes").alias("high_temp_exposure_minutes"),
    (sum("running_minutes") / 60).alias("running_hours"),
)
daily_new = daily_new.withColumn(
    "driving_intensity", when(col("running_hours") > 0, col("daily_distance") / col("running_hours")))
daily_new.persist(StorageLevel.MEMORY_AND_DISK)

ensure_table(daily_new, DAILY_TABLE, "event_date")
merge_upsert(daily_new, DAILY_TABLE, keys=["vin", "event_date"])
logger.info(f"MERGED daily into {DAILY_TABLE}")

touched = daily_new.select("vin").distinct().persist()

############################################################
# STATE : euler_vin_stats — VIN-level rates, upserted for touched vins only (1 row/VIN)
# Read touched vins' full daily history (cheap read, tiny write). No per-date rewrite anywhere.
############################################################

hist = spark.table(DAILY_TABLE).join(broadcast(touched), "vin")
stats = (hist.groupBy("vin").agg(
            min("event_date").alias("first_vehicle_date"),
            max("event_date").alias("last_event_date"),
            count("event_date").alias("day_count"),
            sum("daily_distance").alias("total_distance"),          # = cumulative_distance as of last day
            min(struct("event_date", "daily_distance")).alias("_first"),
            max("current_soh").alias("soh_max"),
            min("current_soh").alias("soh_min"))
         .withColumn("first_daily_distance", col("_first.daily_distance"))
         .withColumn("total_km", col("total_distance") - col("first_daily_distance"))
         .withColumn("soh_degradation_per_day",
                     when(col("day_count") > 1, (col("soh_max") - col("soh_min")) / (col("day_count") - 1))
                     .otherwise(lit(0.0)))
         .withColumn("soh_degradation_per_km",
                     when(col("total_km") > 0, (col("soh_max") - col("soh_min")) / col("total_km"))
                     .otherwise(lit(0.0)))
         .drop("_first"))
stats.persist(StorageLevel.MEMORY_AND_DISK)
ensure_table(stats, STATS_TABLE)
merge_upsert(stats, STATS_TABLE, keys=["vin"])
logger.info(f"MERGED vin stats into {STATS_TABLE}")

############################################################
# FEATURES : today's per-date derived row (touched vins) -> euler_features_daily
# Expanding features come from `stats` (as-of-today); rolling features from a bounded N-day window.
# Historical rows are immutable and untouched -> the MERGE only writes the current-month partition.
############################################################

rolling = (spark.table(DAILY_TABLE).join(broadcast(touched), "vin")
           .filter(col("event_date").between(lit(ROLL_START), lit(str(PROCESS_DATE))))
           .groupBy("vin").agg(
               sum("high_temp_exposure_minutes").alias("rolling_temperature_exposure"),
               avg("estimated_cycle_count").alias("rolling_cycle_count")))

features_today = (
    daily_new.alias("d")
    .join(stats.select("vin", "first_vehicle_date", "total_distance", "day_count").alias("s"), "vin")
    .join(rolling, "vin")
    .withColumn("cumulative_distance", col("total_distance"))
    .withColumn("avg_daily_distance", col("total_distance") / col("day_count"))
    .withColumn("vehicle_age_days", datediff(col("event_date"), col("first_vehicle_date")))
    .select(
        "vin", "event_date", "current_soh", "vehicle_age_days", "estimated_cycle_count",
        "cumulative_distance", "avg_daily_distance", "avg_battery_temperature", "max_battery_temperature",
        "avg_current", "max_current", "rolling_temperature_exposure", "high_temp_exposure_minutes",
        "soc_variation", "driving_intensity", "max_cell_imbalance", "avg_speed", "max_speed",
        "rolling_cycle_count"))
features_today.persist(StorageLevel.MEMORY_AND_DISK)
ensure_table(features_today, FEATURES_TABLE, "event_date")
merge_upsert(features_today, FEATURES_TABLE, keys=["vin", "event_date"])
logger.info(f"MERGED per-date features into {FEATURES_TABLE}")

############################################################
# INFERENCE : euler_latest (1 row/VIN, touched only) = today's row + VIN rates -> 21-col snapshot
############################################################

latest_new = (features_today.join(
    stats.select("vin", "soh_degradation_per_day", "soh_degradation_per_km"), "vin"))
ensure_table(latest_new, LATEST_TABLE)
merge_upsert(latest_new, LATEST_TABLE, keys=["vin"])
# emit JSON for touched vins only (dynamic overwrite keeps silent vehicles' snapshots intact)
(latest_new.repartition("vin").write.mode("overwrite").partitionBy("vin")
           .option("compression", "gzip").json(args["inference_output"]))
logger.info(f"inference latest upserted + written to {args['inference_output']}")

############################################################
# TRAINING : consumers read euler_features_daily JOIN euler_vin_stats directly.
# Full Parquet snapshot only ON DEMAND (a training run), never on the daily incremental path.
############################################################

if EMIT_SNAPSHOT:
    rates = spark.table(STATS_TABLE).select("vin", "soh_degradation_per_day", "soh_degradation_per_km")
    train = (spark.table(FEATURES_TABLE).join(rates, "vin").select(
        "vin", "event_date", "current_soh", "vehicle_age_days", "estimated_cycle_count",
        "cumulative_distance", "avg_daily_distance", "avg_battery_temperature", "max_battery_temperature",
        "avg_current", "max_current", "rolling_temperature_exposure", "high_temp_exposure_minutes",
        "soc_variation", "driving_intensity", "max_cell_imbalance", "soh_degradation_per_day",
        "avg_speed", "max_speed", "soh_degradation_per_km", "rolling_cycle_count"))
    train.write.mode("overwrite").option("maxRecordsPerFile", 500000).parquet(args["training_output"])
    logger.info(f"training snapshot exported to {args['training_output']}")

df.unpersist(); daily_new.unpersist(); touched.unpersist(); stats.unpersist(); features_today.unpersist()
job.commit()
