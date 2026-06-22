#!/usr/bin/env python3
"""Import a MONTHLY SAMPLE of Mahindra telemetry for the oldest 10 vehicles.

Mahindra stores ~70M tiny Parquet files (no VIN index), so a full-history extract
is infeasible. Instead we:
  1. Find the oldest 10 VINs from a sample of the earliest partition.
  2. For each month in range, extract ONE representative day (capped file count),
     S3-Selecting just those 10 VINs.
This yields ~20 monthly snapshots — enough for a SoH-degradation trend — quickly.
Output: data/mahindra_extracted/ as Parquet (one file per source object with matches).
"""
import os, io, json, re, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv("/home/hj/Desktop/EULER_RUL_MODEL/.env")
BUCKET = os.environ["S3_BUCKET"]
PREFIX = "battery-oem-data/parquet/mahindra/vehicle-data/"
OUT_DIR = Path("data/mahindra_extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10
MAX_WORKERS = 48
DISCOVERY_SAMPLE = 4000      # files read from earliest day to identify oldest VINs
FILES_PER_DAY_CAP = 25000    # cap S3-Select requests per sampled day
TARGET_DOM = 15              # prefer the ~15th of each month as the representative day

s3 = boto3.client("s3", config=Config(max_pool_connections=MAX_WORKERS + 8,
                                      retries={"max_attempts": 5, "mode": "adaptive"}))

# Only the analytically useful columns (audited 2026-06-17): drops requestUUID,
# lat/long, lastConnected, gearPosition, licensePlate, valid, color, vehicleVariant.
ALL_COLS = ["eventAt", "vin", "soc", "odometer", "distanceToEmpty",
            "state", "batteryTemp", "kwh", "vehicleModel"]


def kids(p):
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def list_keys(prefix, cap=None):
    keys, pag = [], s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".parquet"):
                keys.append(o["Key"])
                if cap and len(keys) >= cap:
                    return keys
    return keys


# ---- 1. Load the target VINs (most-aged recurring vehicles) ------------------
# Selection is done separately (see manifest); the earliest partition is a single-
# vehicle backfill dump, so "oldest by first partition" is invalid. We instead use
# the 10 highest-odometer vehicles that recur across partitions.
years = kids(PREFIX)
man = pd.read_csv("data/mahindra_oldest10_manifest.csv")
top_vins = man["vin"].tolist()
print(f"Target {len(top_vins)} most-aged Mahindra VINs (from manifest):", flush=True)
for _, r in man.iterrows():
    print(f"  {r['vin']}  model_year={r.get('myear')}  max_odo={r.get('max_odo')}", flush=True)

# ---- 2. Pick one representative day per month --------------------------------
def dom(day_prefix):
    return int(re.search(r"day=(\d{2})", day_prefix).group(1))

sample_days = []
for y in years:
    for m in kids(y):
        days = kids(m)
        if not days:
            continue
        chosen = min(days, key=lambda d: abs(dom(d) - TARGET_DOM))
        sample_days.append(chosen)
print(f"\n{len(sample_days)} monthly sample days "
      f"({sample_days[0].split('vehicle-data/')[1].rstrip('/')} ... "
      f"{sample_days[-1].split('vehicle-data/')[1].rstrip('/')})", flush=True)

# ---- 3. S3-Select the oldest VINs from each sampled day ----------------------
_vins_sql = ", ".join(f"'{v}'" for v in top_vins)
_cols_sql = ", ".join(f"s.{c}" for c in ALL_COLS)   # project only useful columns
EXPRESSION = f"SELECT {_cols_sql} FROM s3object s WHERE s.vin IN ({_vins_sql})"

def out_path_for(key):
    rel = key[len(PREFIX):] if key.startswith(PREFIX) else key
    return OUT_DIR / (rel.replace("/", "__").replace(".parquet", "") + ".parquet")

def extract_one(key):
    op = out_path_for(key)
    if op.exists():
        return None
    resp = s3.select_object_content(
        Bucket=BUCKET, Key=key, ExpressionType="SQL", Expression=EXPRESSION,
        InputSerialization={"Parquet": {}},
        OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
    buf = bytearray()
    for ev in resp["Payload"]:
        if "Records" in ev:
            buf += ev["Records"]["Payload"]
    rows = [json.loads(l) for l in buf.decode().splitlines() if l.strip()]
    if not rows:
        return (0, Counter())
    df = pd.DataFrame(rows).reindex(columns=ALL_COLS)
    df.to_parquet(op, index=False)
    return (len(df), Counter(df["vin"]))

totals = Counter()
n_rows = n_files = errors = 0
for di, day in enumerate(sample_days, 1):
    keys = list_keys(day, cap=FILES_PER_DAY_CAP)
    tag = day.split("vehicle-data/")[1].rstrip("/")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(extract_one, k): k for k in keys}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  error {futs[fut]}: {type(e).__name__}: {e}", flush=True)
                continue
            if res is None:
                continue
            rows, per = res
            n_rows += rows
            if rows:
                n_files += 1
            totals.update(per)
    print(f"  [{di}/{len(sample_days)}] {tag}: {len(keys)} files scanned | "
          f"cumulative {n_rows:,} rows | err {errors}", flush=True)

print(f"\nDONE. {n_rows:,} rows across {n_files:,} files with matches. Errors: {errors}")
print("Rows per VIN:")
for v in top_vins:
    print(f"  {v}: {totals.get(v, 0):,}")
