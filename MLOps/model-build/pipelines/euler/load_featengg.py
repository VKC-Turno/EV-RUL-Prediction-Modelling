"""Processing step — unload the `euler_featengg` Iceberg feature store to parquet for the training channel.

The feature store (Glue output) is all-vehicles / all-months with `soh` nullable. This step just materialises
it for training; the POINT-IN-TIME cohort selection (labelled rows, data-quality gate, in-service filter) and
the train/val/test split happen in train.py — deliberately NOT baked into the store. Add an as-of `--cutoff`
here if a retrain must reproduce the fleet exactly as it was on a past date.

    reads  glue_catalog.turno_ml.euler_featengg  ->  writes /opt/ml/processing/output/*.parquet
"""
import argparse
import os

from pyspark.sql import SparkSession, functions as F

TABLE = "glue_catalog.turno_ml.euler_featengg"
OUT = "/opt/ml/processing/output"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default=TABLE)
    p.add_argument("--output", default=OUT)
    p.add_argument("--cutoff", default="", help="optional YYYY-MM-DD: keep rows with ymd <= cutoff (as-of retrain)")
    a = p.parse_args()

    spark = (SparkSession.builder.appName("euler-load-featengg")
             .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
             .config("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
             .config("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
             .config("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
             .getOrCreate())

    df = spark.table(a.table).drop("month")            # 25 deployed cols (ymd + features + soh)
    if a.cutoff:
        df = df.filter(F.col("ymd") <= F.lit(a.cutoff))
    df.write.mode("overwrite").parquet(a.output)
    print(f"unloaded {a.table} -> {a.output} ({df.count()} rows)")


if __name__ == "__main__":
    main()
