#!/usr/bin/env python3
"""Import a MONTHLY SAMPLE of intellicar telemetry for the oldest 10 Mahindra vehicles.

Unlike the mahindra/ OEM feed, the intellicar table has real signed `current` (+ voltage),
which enables TRUE coulomb counting for SoH. We monthly-sample one representative day per
month, S3-Selecting just our 10 VINs and only the columns coulomb counting needs.
Output: data/intellicar_extracted/.
"""
import os, io, json, re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv("/home/hj/Desktop/EULER_RUL_MODEL/.env")
BUCKET = os.environ["S3_BUCKET"]
PREFIX = "battery-oem-data/parquet/intellicar/battery-data/"
OUT_DIR = Path("data/intellicar_extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 48
FILES_PER_DAY_CAP = 2500     # intellicar files are dense (~1-3k rows each)
TARGET_DOM = 15

s3 = boto3.client("s3", config=Config(max_pool_connections=MAX_WORKERS + 8,
                                      retries={"max_attempts": 5, "mode": "adaptive"}))

# Only what coulomb-counting SoH needs (+ make/model context). `current` is a reserved
# word in S3 Select SQL, so it must be double-quoted in the SELECT.
# Audited 2026-06-17: for Mahindra rows only 9 of intellicar's 84 columns hold data
# (all temp/fault/cycle/motor columns are empty for Mahindra). `current` is reserved -> quoted.
ALL_COLS = ["vin", "eventAt", "make", "model", "soc", "current", "batteryVoltage", "odometer", "dte"]


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


man = pd.read_csv("data/intellicar_mahindra_oldest10_manifest.csv")
top_vins = man["vin"].tolist()
print(f"Target {len(top_vins)} oldest Mahindra VINs (intellicar):", flush=True)
for _, r in man.iterrows():
    print(f"  {r['vin']}  {r.get('model')}  max_odo={r.get('max_odo')}", flush=True)

# Representative day per month (skip the junk year=0000 partition).
def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))

sample_days = []
for y in kids(PREFIX):
    if "year=0000" in y:
        continue
    for m in kids(y):
        days = kids(m)
        if days:
            sample_days.append(min(days, key=lambda d: abs(dom(d) - TARGET_DOM)))
print(f"\n{len(sample_days)} monthly sample days "
      f"({sample_days[0].split('battery-data/')[1].rstrip('/')} ... "
      f"{sample_days[-1].split('battery-data/')[1].rstrip('/')})", flush=True)

_vins_sql = ", ".join(f"'{v}'" for v in top_vins)
_cols_sql = ", ".join(f's."{c}"' if c == "current" else f"s.{c}" for c in ALL_COLS)
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
    tag = day.split("battery-data/")[1].rstrip("/")
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
    print(f"  [{di}/{len(sample_days)}] {tag}: {len(keys)} files | "
          f"cumulative {n_rows:,} rows | err {errors}", flush=True)

print(f"\nDONE. {n_rows:,} rows across {n_files:,} files. Errors: {errors}")
print("Rows per VIN:")
for v in top_vins:
    print(f"  {v}: {totals.get(v, 0):,}")
