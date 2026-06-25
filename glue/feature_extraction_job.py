"""Glue 4.0 (Spark) feature-extraction job — ONE OEM, full fleet.

Reads raw telemetry from S3, repartitions by VIN, runs the per-VIN pandas pipeline
(`feature_logic.vin_features`) via Spark `applyInPandas`, and writes the feature_table to S3.
Run once per OEM (the output columns are OEM-specific); orchestrate the three with a Glue Workflow.

Job parameters (--key value):
  --OEM           euler | mahindra | bajaj
  --RAW_S3        s3://.../   raw parquet root for this OEM's SOURCE feed
                  (bajaj/euler: their own vehicle-data feed; mahindra: the INTELLICAR feed,
                   ideally pre-filtered to Mahindra VINs — see README)
  --REG_CSV_S3    s3://.../<oem>_reg.csv   (cols: vin + regd_date | vehicle_registration_date)
  --OUT_S3        s3://.../features/oem=<oem>/   output prefix
  --VINS_CSV_S3   (optional) s3://.../cohort.csv with a `vin` column to restrict the fleet

Packaging: --extra-py-files s3://.../src.zip   (zip of repo src/ + glue/feature_logic.py)
Deps:      --additional-python-modules xgboost,lightgbm,scikit-learn,pyarrow
"""
import sys
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession, functions as F
import pandas as pd

import feature_logic

REQ = ["OEM", "RAW_S3", "REG_CSV_S3", "OUT_S3"]
args = getResolvedOptions(sys.argv, REQ)
OEM = args["OEM"].lower()
vins_csv = None
if "--VINS_CSV_S3" in sys.argv:
    vins_csv = getResolvedOptions(sys.argv, ["VINS_CSV_S3"])["VINS_CSV_S3"]

spark = (SparkSession.builder.appName(f"feature-extraction-{OEM}").getOrCreate())
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# 1) raw telemetry (ingest-date partitioned). For Mahindra's ~70M tiny files, point RAW_S3 at a
#    COMPACTED dataset instead of the raw firehose (see README) — a naive scan is the slow part.
raw = spark.read.parquet(args["RAW_S3"])
if "vin" not in raw.columns:
    raise SystemExit("raw data must expose a `vin` column")

# 2) optional cohort filter (broadcast small VIN list)
if vins_csv:
    keep = spark.read.csv(vins_csv, header=True).select("vin").distinct()
    raw = raw.join(F.broadcast(keep), "vin")

# 3) registration map -> broadcast (small)
regdf = spark.read.csv(args["REG_CSV_S3"], header=True)
rcol = next(c for c in ("regd_date", "vehicle_registration_date") if c in regdf.columns)
reg_map = {r["vin"]: pd.to_datetime(r[rcol], errors="coerce") for r in regdf.collect()}
reg_b = spark.sparkContext.broadcast(reg_map)

# 4) infer the OEM-specific output schema from one real vehicle (avoids hardcoding ~25 columns)
sample_vin = raw.select("vin").na.drop().limit(1).collect()[0]["vin"]
sample_out = feature_logic.vin_features(
    raw.filter(F.col("vin") == sample_vin).toPandas(), OEM, reg_b.value)
if sample_out is None or sample_out.empty:
    raise SystemExit(f"sample vehicle {sample_vin} yielded no features — check the source columns for {OEM}")
schema = spark.createDataFrame(sample_out).schema

# 5) distributed per-VIN feature extraction
def _udf(pdf):
    return feature_logic.vin_features(pdf, OEM, reg_b.value)

feats = raw.repartition("vin").groupBy("vin").applyInPandas(_udf, schema=schema)

# 6) write the feature_table (single OEM partition)
feats.write.mode("overwrite").parquet(args["OUT_S3"])
print(f"[feature-extraction] OEM={OEM} -> {args['OUT_S3']}")
