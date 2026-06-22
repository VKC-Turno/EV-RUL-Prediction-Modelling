#!/usr/bin/env python3
"""Dense import: ALL days in a year for ONE vin from a chosen feed (for the SoH comparison).
Usage: import_dense.py <intellicar|mahindra> <YEAR>
"""
import os, io, sys, json, re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

FEED, YEAR = sys.argv[1], sys.argv[2]
VIN = "MB7F8CLLFNJH48488"
load_dotenv("/home/hj/Desktop/EULER_RUL_MODEL/.env")
B = os.environ["S3_BUCKET"]

CFG = {
    "intellicar": dict(
        prefix="battery-oem-data/parquet/intellicar/battery-data/",
        out="data/intellicar_dense", cap=4000,
        cols=["vin", "eventAt", "make", "model", "soc", "current", "batteryVoltage", "odometer", "dte"],
        split="battery-data/"),
    "mahindra": dict(
        prefix="battery-oem-data/parquet/mahindra/vehicle-data/",
        out="data/mahindra_dense", cap=12000,
        cols=["eventAt", "vin", "soc", "odometer", "distanceToEmpty", "state", "batteryTemp", "kwh", "vehicleModel"],
        split="vehicle-data/"),
}[FEED]

OUT = Path(CFG["out"]); OUT.mkdir(parents=True, exist_ok=True)
MW = 40
s3 = boto3.client("s3", config=Config(max_pool_connections=MW + 8, retries={"max_attempts": 5, "mode": "adaptive"}))
COLS = CFG["cols"]
_cols_sql = ", ".join(f's."{c}"' if c == "current" else f"s.{c}" for c in COLS)
EXPR = f"SELECT {_cols_sql} FROM s3object s WHERE s.vin = '{VIN}'"


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def list_keys(prefix, cap):
    keys, pag = [], s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=B, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".parquet"):
                keys.append(o["Key"])
                if len(keys) >= cap:
                    return keys
    return keys


def out_path_for(key):
    rel = key[len(CFG["prefix"]):] if key.startswith(CFG["prefix"]) else key
    return OUT / (rel.replace("/", "__").replace(".parquet", "") + ".parquet")


def extract_one(key):
    op = out_path_for(key)
    if op.exists():
        return None
    resp = s3.select_object_content(Bucket=B, Key=key, ExpressionType="SQL", Expression=EXPR,
        InputSerialization={"Parquet": {}}, OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
    buf = bytearray()
    for ev in resp["Payload"]:
        if "Records" in ev:
            buf += ev["Records"]["Payload"]
    rows = [json.loads(l) for l in buf.decode().splitlines() if l.strip()]
    if not rows:
        return 0
    pd.DataFrame(rows).reindex(columns=COLS).to_parquet(op, index=False)
    return len(rows)


year_prefix = f"{CFG['prefix']}year={YEAR}/"
days = []
for m in kids(year_prefix):
    days += kids(m)
print(f"{FEED} {YEAR}: {len(days)} day-partitions for {VIN}", flush=True)

tot = nfiles = err = 0
for di, day in enumerate(days, 1):
    keys = list_keys(day, CFG["cap"])
    tag = day.split(CFG["split"])[1].rstrip("/")
    with ThreadPoolExecutor(max_workers=MW) as pool:
        futs = {pool.submit(extract_one, k): k for k in keys}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    tot += r; nfiles += 1
            except Exception as e:
                err += 1
                if err <= 3:
                    print(f"  err {type(e).__name__}: {str(e)[:80]}", flush=True)
    if di % 10 == 0 or di == len(days):
        print(f"  [{di}/{len(days)}] {tag}: {len(keys)} files | cumulative {tot:,} rows | err {err}", flush=True)

print(f"\nDONE {FEED} {YEAR}. {tot:,} rows for {VIN} across {nfiles} files. Errors: {err}", flush=True)
