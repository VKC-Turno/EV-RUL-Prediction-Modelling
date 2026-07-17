"""Euler battery preprocessing — INCREMENTAL Glue ETL.

Incremental rewrite of the third party's static prototype ("DataPreproccessing job"). Their version read a
hardcoded 5-day window (year=2025/month=01/day=01..05 — the static sample we shared) and full-overwrote the
training set every run, and computed the "lifetime" trend features over just that 5-day read. This version:

  1. READS only the new day(s), passed as a job parameter (--process_date) — scheduler-driven, not hardcoded.
  2. Keeps their Section 1-4 daily aggregation logic unchanged, on the new day(s) only.
  3. UPSERTS the daily rows into a durable Iceberg table (euler_daily) via MERGE on (vin, event_date) —
     idempotent, so re-running a day never duplicates (the bug that killed their `append` attempt).
  4. RECOMPUTES the trend/degradation features (Section 5) over each touched vehicle's FULL history read back
     from euler_daily — so cumulative_distance / vehicle_age_days / soh_degradation_* are lifetime-correct,
     not truncated to the window. Upserts into euler_features.
  5. TRAINING output = a full snapshot of euler_features (overwrite of the complete durable table — idempotent
     AND complete, unlike window-overwrite). INFERENCE = latest row per VIN over the full table.

REQUIRED JOB PARAMETERS (set on the Glue job, in addition to --JOB_NAME):
    --datalake-formats            iceberg
    --process_date                2025-01-05        (the day to ingest; scheduler passes yesterday)
    --raw_bucket                  oem-iot-data       (verify: prod may be oem-data-iot)
    --warehouse                   s3://rcs-mlops-data/iceberg/
    --training_output             s3://rcs-mlops-data/training/
    --inference_output            s3://rcs-mlops-data/inference_input/latest/
    --conf  spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            --conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog
            --conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog
            --conf spark.sql.catalog.glue_catalog.warehouse=s3://rcs-mlops-data/iceberg/
            --conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO
Also flip the job config to  bookmark: job-bookmark-enable  (used only if you switch to the catalog read in
_read_new_days; the parameterised-partition read below is already incremental without bookmarks).
"""
import sys
import builtins
from datetime import datetime, timedelta

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.storagelevel import StorageLevel

from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lit, when, trim, from_unixtime, to_timestamp, to_date, year, month, dayofmonth,
    hex, lag, lead, row_number, first, avg, min, max, stddev, sum, count, abs, datediff,
    unix_timestamp, broadcast, months,
)

############################################################
# Initialise Glue + parameters
############################################################

ARG_KEYS = ["JOB_NAME", "process_date", "raw_bucket", "warehouse",
            "training_output", "inference_output"]
_defaults = dict(
    raw_bucket="oem-iot-data",
    warehouse="s3://rcs-mlops-data/iceberg/",
    training_output="s3://rcs-mlops-data/training/",
    inference_output="s3://rcs-mlops-data/inference_input/latest/",
)
# getResolvedOptions requires all listed keys to be present; make the optional ones default cleanly.
_present = [k for k in ARG_KEYS if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, _present)
for k, v in _defaults.items():
    args.setdefault(k, v)
if "process_date" not in args:
    raise ValueError("--process_date=YYYY-MM-DD is required (the day to ingest). The scheduler passes it.")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

DEBUG = False
OEM = "euler"
CATALOG = "glue_catalog"
DB = "turno_ml"
DAILY_TABLE = f"{CATALOG}.{DB}.{OEM}_daily"
FEATURES_TABLE = f"{CATALOG}.{DB}.{OEM}_features"

# Iceberg catalog confs (safe to set here; the SQL-extensions conf must be a job --conf, see header).
spark.conf.set(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.warehouse", args["warehouse"])
spark.conf.set(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")

spark.conf.set("spark.sql.shuffle.partitions", "64")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

PROCESS_DATE = datetime.strptime(args["process_date"], "%Y-%m-%d").date()

############################################################
# Helpers
############################################################

def ensure_iceberg_table(df, table, partition_col):
    """Create an empty Iceberg table with df's schema on first run (partitioned by months(partition_col))."""
    if not spark.catalog.tableExists(table):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DB}")
        (df.limit(0).writeTo(table).using("iceberg")
           .partitionedBy(months(partition_col)).create())
        logger.info(f"created Iceberg table {table}")


def merge_upsert(df, table, keys):
    """Idempotent MERGE of df into `table` on `keys` — WHEN MATCHED UPDATE * / NOT MATCHED INSERT *."""
    view = f"_stage_{table.split('.')[-1]}"
    df.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    spark.sql(f"""
        MERGE INTO {table} t USING {view} s ON {on}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def _read_new_days(process_date):
    """Read ONLY the target day's raw partition (parameterised — no hardcoded window).

    Alternative (bookmarked) read, if the raw feed is catalogued — auto-skips already-processed files and
    picks up late arrivals; requires bookmark: job-bookmark-enable:
        dyf = glueContext.create_dynamic_frame.from_catalog(
            database="battery_raw", table_name="euler_vehicle_data",
            transformation_ctx="euler_raw",
            push_down_predicate=f"year='{process_date.year}' and month='{process_date.month:02d}' "
                                f"and day='{process_date.day:02d}'")
        return dyf.toDF()
    """
    path = (f"s3://{args['raw_bucket']}/battery-oem-data/parquet/{OEM}/vehicle-data/"
            f"year={process_date.year}/month={process_date.month:02d}/day={process_date.day:02d}/")
    logger.info(f"reading new-day partition: {path}")
    return spark.read.parquet(path)

############################################################
# SECTION 1-2 : read new day, clean, feature-prep (unchanged logic, new-day scope)
############################################################

df = _read_new_days(PROCESS_DATE)
df = df.filter(col("vin").isNotNull() & (trim(col("vin")) != ""))

df = (
    df.withColumn("event_timestamp", to_timestamp(from_unixtime(col("eventat") / 1000)))
      .withColumn("event_date", to_date(col("event_timestamp")))
      .withColumn("event_year", year(col("event_timestamp")))
      .withColumn("event_month", month(col("event_timestamp")))
      .withColumn("event_day", dayofmonth(col("event_timestamp")))
)

# Pin event_date to exactly the target day (timezone conversion can spill a few records to the neighbour day).
df = df.filter(col("event_date") == lit(str(PROCESS_DATE)))

RANGE_COLUMNS = [
    ("batterysoh", "battery_soh", 70, 100),
    ("batterysoc", "battery_soc", 0, 100),
    ("odometer", "odometer_clean", 0, 125000),
    ("speed", "speed_clean", 0, 60),
    ("batterycurrent", "battery_current", -150, 150),
    ("batterytemperature", "battery_temperature", -20, 60),
    ("batteryremainingcapacity", "battery_remaining_capacity", 0, 210),
    ("controllertemperature", "controller_temperature", -20, 120),
    ("motortemperature", "motor_temperature", -20, 150),
    ("cellimbalance", "cell_imbalance", 0, 200),
    ("batteryvoltage", "battery_voltage", 0, 80),
]
for raw_col, alias, low, high in RANGE_COLUMNS:
    df = df.withColumn(alias, when(col(raw_col).between(low, high), col(raw_col)))

VEHICLE_MODE_MAP = {"30": 0, "31": 1, "32": 2, "33": 3, "34": 4, "35": 5, "45636F6E6F6D79": 6}
df = df.withColumn("vehicle_mode_hex", hex(col("vehiclemode")))
mode_expr = None
for hex_val, mode_val in VEHICLE_MODE_MAP.items():
    cond = col("vehicle_mode_hex") == hex_val
    mode_expr = when(cond, mode_val) if mode_expr is None else mode_expr.when(cond, mode_val)
df = df.withColumn("vehicle_mode", mode_expr)

required_columns = [
    "vin", "event_timestamp", "event_date", "battery_soh", "battery_soc", "odometer_clean",
    "speed_clean", "battery_current", "battery_temperature", "battery_remaining_capacity",
    "controller_temperature", "motor_temperature", "cell_imbalance", "battery_voltage", "vehicle_mode",
]
df = df.na.drop(subset=required_columns).select(
    "vin", "event_timestamp", "event_date", "event_year", "event_month", "event_day",
    "battery_soh", "battery_soc", "odometer_clean", "speed_clean", "battery_current",
    "battery_temperature", "battery_remaining_capacity", "controller_temperature",
    "motor_temperature", "cell_imbalance", "battery_voltage", "vehicle_mode",
)
df.persist(StorageLevel.MEMORY_AND_DISK)
logger.info("Section 1-2 complete (clean, new-day scope)")

############################################################
# SECTION 3 : intra-day window features (per VIN, within the day)
# first_vehicle_date is NO LONGER computed here — it is a lifetime attribute and is derived in Section 5
# from the durable full history, not from a single day's read.
############################################################

vin_window = Window.partitionBy("vin").orderBy("event_timestamp")
df = (
    df.withColumn("previous_soc", lag("battery_soc").over(vin_window))
      .withColumn("next_timestamp", lead("event_timestamp").over(vin_window))
)
df = df.withColumn(
    "duration_minutes",
    (unix_timestamp("next_timestamp") - unix_timestamp("event_timestamp")) / 60,
).fillna({"duration_minutes": 0})
df = df.withColumn(
    "cycle_difference",
    when(col("previous_soc").isNotNull() & (abs(col("battery_soc") - col("previous_soc")) >= 1),
         abs(col("battery_soc") - col("previous_soc"))).otherwise(0),
)
df = df.withColumn("running_minutes", when(col("vehicle_mode") == 3, col("duration_minutes")).otherwise(0))
df = df.withColumn("high_temp_minutes", when(col("battery_temperature") > 45, col("duration_minutes")).otherwise(0))
df.persist(StorageLevel.MEMORY_AND_DISK)
logger.info("Section 3 complete")

############################################################
# SECTION 4 : daily aggregation (one row per vin, event_date) — new day only
############################################################

daily_new = df.groupBy("vin", "event_date").agg(
    max("battery_soh").alias("current_soh"),
    avg("battery_soc").alias("avg_soc"),
    min("battery_soc").alias("min_soc"),
    max("battery_soc").alias("max_soc"),
    (max("odometer_clean") - min("odometer_clean")).alias("daily_distance"),
    avg(when(col("vehicle_mode") == 3, col("speed_clean"))).alias("avg_speed"),
    max(when(col("vehicle_mode") == 3, col("speed_clean"))).alias("max_speed"),
    stddev(when(col("vehicle_mode") == 3, col("speed_clean"))).alias("std_speed"),
    avg(when(col("vehicle_mode") == 3, col("battery_current"))).alias("avg_current"),
    max(when(col("vehicle_mode") == 3, col("battery_current"))).alias("max_current"),
    stddev(when(col("vehicle_mode") == 3, col("battery_current"))).alias("std_current"),
    avg("battery_temperature").alias("avg_battery_temperature"),
    max("battery_temperature").alias("max_battery_temperature"),
    (max("battery_temperature") - min("battery_temperature")).alias("temperature_variation"),
    max("battery_remaining_capacity").alias("battery_remaining_capacity"),
    avg(when(col("vehicle_mode") == 3, col("controller_temperature"))).alias("avg_controller_temperature"),
    max(when(col("vehicle_mode") == 3, col("controller_temperature"))).alias("max_controller_temperature"),
    avg(when(col("vehicle_mode") == 3, col("motor_temperature"))).alias("avg_motor_temperature"),
    max(when(col("vehicle_mode") == 3, col("motor_temperature"))).alias("max_motor_temperature"),
    max("cell_imbalance").alias("max_cell_imbalance"),
    avg("battery_voltage").alias("avg_battery_voltage"),
    min("battery_voltage").alias("min_battery_voltage"),
    max("battery_voltage").alias("max_battery_voltage"),
    (sum("cycle_difference") / 100).alias("estimated_cycle_count"),
    (max("battery_soc") - min("battery_soc")).alias("soc_variation"),
    sum("high_temp_minutes").alias("high_temp_exposure_minutes"),
    (sum("running_minutes") / 60).alias("running_hours"),
)
daily_new = daily_new.withColumn(
    "driving_intensity",
    when(col("running_hours") > 0, col("daily_distance") / col("running_hours")),
)
daily_new.persist(StorageLevel.MEMORY_AND_DISK)
logger.info(f"Section 4 complete — {daily_new.count() if DEBUG else 'n'} new daily rows")

############################################################
# STATE : upsert the new daily rows into the durable Iceberg table (idempotent)
############################################################

ensure_iceberg_table(daily_new, DAILY_TABLE, "event_date")
merge_upsert(daily_new, DAILY_TABLE, keys=["vin", "event_date"])
logger.info(f"MERGED new daily rows into {DAILY_TABLE}")

############################################################
# SECTION 5 : trend & degradation over FULL history — only for touched VINs
# Read back each touched vehicle's complete daily history from the durable table, so cumulative / age /
# degradation features are lifetime-correct rather than truncated to the day just read.
############################################################

touched_vins = daily_new.select("vin").distinct()
hist = spark.table(DAILY_TABLE).join(broadcast(touched_vins), "vin")

vin_daily_window = Window.partitionBy("vin").orderBy("event_date")
expanding_window = vin_daily_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
ROLLING_WINDOW_DAYS = 7
rolling_window = vin_daily_window.rowsBetween(-(ROLLING_WINDOW_DAYS - 1), Window.currentRow)
vin_full_window = Window.partitionBy("vin")

hist = (
    hist
    .withColumn("first_vehicle_date", min("event_date").over(vin_full_window))   # true first OBSERVED day
    .withColumn("vehicle_age_days", datediff(col("event_date"), min("event_date").over(vin_full_window)))
    .withColumn("cumulative_distance", sum("daily_distance").over(expanding_window))
    .withColumn("avg_daily_distance", avg("daily_distance").over(expanding_window))
    .withColumn("rolling_temperature_exposure", sum("high_temp_exposure_minutes").over(rolling_window))
    .withColumn("rolling_cycle_count", avg("estimated_cycle_count").over(rolling_window))
)

soh_max = max("current_soh").over(vin_full_window)
soh_min = min("current_soh").over(vin_full_window)
vin_day_count = count("event_date").over(vin_full_window)
hist = hist.withColumn(
    "soh_degradation_per_day",
    when(vin_day_count > 1, (soh_max - soh_min) / (vin_day_count - 1)).otherwise(lit(0.0)),
)
cum_dist_max = max("cumulative_distance").over(vin_full_window)
cum_dist_min = min("cumulative_distance").over(vin_full_window)
total_km = cum_dist_max - cum_dist_min
hist = hist.withColumn(
    "soh_degradation_per_km",
    when(total_km > 0, (soh_max - soh_min) / total_km).otherwise(lit(0.0)),
)
logger.info("Section 5 complete (full-history trend features for touched VINs)")

############################################################
# SECTION 6 : final feature selection + upsert into euler_features
############################################################

FINAL_FEATURE_COLUMNS = [
    "vin", "event_date", "current_soh", "vehicle_age_days", "estimated_cycle_count",
    "cumulative_distance", "avg_daily_distance", "avg_battery_temperature", "max_battery_temperature",
    "avg_current", "max_current", "rolling_temperature_exposure", "high_temp_exposure_minutes",
    "soc_variation", "driving_intensity", "max_cell_imbalance", "soh_degradation_per_day",
    "avg_speed", "max_speed", "soh_degradation_per_km", "rolling_cycle_count",
]
features_new = hist.select(*FINAL_FEATURE_COLUMNS)
ensure_iceberg_table(features_new, FEATURES_TABLE, "event_date")
merge_upsert(features_new, FEATURES_TABLE, keys=["vin", "event_date"])
logger.info(f"MERGED features into {FEATURES_TABLE}")

############################################################
# SECTION 7 : training snapshot (Parquet) — full durable table, overwrite = idempotent AND complete
# Overwrite is now safe because the source is the ENTIRE accumulated history, not just this run's window.
############################################################

(spark.table(FEATURES_TABLE)
      .write.mode("overwrite").option("maxRecordsPerFile", 500000)
      .parquet(args["training_output"]))
logger.info(f"training snapshot written to {args['training_output']}")

############################################################
# SECTION 8 : inference input (JSON) — latest row per VIN across the FULL table
# A vehicle silent this run still gets its true last snapshot (not dropped because it had no new data).
############################################################

latest_window = Window.partitionBy("vin").orderBy(col("event_date").desc())
latest_df = (spark.table(FEATURES_TABLE)
             .withColumn("row_num", row_number().over(latest_window))
             .filter(col("row_num") == 1).drop("row_num"))
(latest_df.repartition("vin")
          .write.mode("overwrite").partitionBy("vin")
          .option("maxRecordsPerFile", 500000).option("compression", "gzip")
          .json(args["inference_output"]))
logger.info(f"inference snapshot written to {args['inference_output']}")

############################################################
# Cleanup
############################################################

df.unpersist()
daily_new.unpersist()
job.commit()
