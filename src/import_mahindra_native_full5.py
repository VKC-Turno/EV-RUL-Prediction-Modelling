#!/usr/bin/env python3
"""Download COMPLETE native telemetry (every day, full within-day resolution) for the <=5 highest-data
Mahindra VINs — a full-resolution deep-dive to test whether a degradation signal that's invisible in the
monthly sample shows up at full resolution.

No VIN index exists, so this scans EVERY file of EVERY day partition and S3-Selects the 5 VINs' rows
(SELECT * handles the 2024->2025 schema migration). Output: one parquet per month in
data/mahindra/native_full5/. Tiny on disk (5 vins); the cost is the full S3 scan (~30-45 min).
Run: .venv/bin/python src/import_mahindra_native_full5.py
"""
import os, json, shutil
from pathlib import Path
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

os.chdir(Path(__file__).resolve().parent.parent); load_dotenv(".env")
B = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", config=Config(max_pool_connections=104, retries={"max_attempts": 5, "mode": "adaptive"}))
NAT = "battery-oem-data/parquet/mahindra/vehicle-data/"
OUT = Path("data/mahindra/native_full5"); OUT.mkdir(parents=True, exist_ok=True)

top5 = pd.read_csv("data/manifests/mahindra_native_top50.csv")["vin"].head(5).astype(str).tolist()
print("top-5 VINs:", top5, flush=True)
EXPR = "SELECT * FROM s3object s WHERE s.vin IN (" + ",".join(f"'{v}'" for v in top5) + ")"


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def allkeys(prefix):
    ks, pag = [], s3.get_paginator("list_objects_v2")
    for pg in pag.paginate(Bucket=B, Prefix=prefix):
        ks += [o["Key"] for o in pg.get("Contents", []) if o["Key"].endswith(".parquet")]
    return ks


def sel(k):
    try:
        resp = s3.select_object_content(Bucket=B, Key=k, ExpressionType="SQL", Expression=EXPR,
            InputSerialization={"Parquet": {}}, OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
        buf = bytearray()
        for ev in resp["Payload"]:
            if "Records" in ev:
                buf += ev["Records"]["Payload"]
        return [json.loads(l) for l in buf.decode().splitlines() if l.strip()]
    except Exception:
        return []


months = [m for y in kids(NAT) for m in kids(y)]
print(f"{len(months)} months to fully scan | free {shutil.disk_usage('.').free/1e9:.1f} GB", flush=True)
tot = 0
for m in months:
    ym = m.split("year=")[1].replace("/month=", "-").rstrip("/")
    outp = OUT / f"{ym}.parquet"
    if outp.exists():
        continue
    rows = []
    for day in kids(m):
        keys = allkeys(day)
        with ThreadPoolExecutor(max_workers=96) as pool:
            for res in pool.map(sel, keys):
                rows += res
    if rows:
        pd.DataFrame(rows).to_parquet(outp, index=False); tot += len(rows)
    print(f"  {ym}: {len(rows):,} rows (cum {tot:,}) | free {shutil.disk_usage('.').free/1e9:.1f} GB", flush=True)

print(f"\nDONE: {tot:,} rows for {len(top5)} vins across {len(list(OUT.glob('*.parquet')))} monthly files")
