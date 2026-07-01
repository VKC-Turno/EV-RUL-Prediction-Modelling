#!/usr/bin/env python3
"""Download COMPLETE (no file cap), ALL-FIELDS native data for the 100 longest-availability Mahindra vehicles.

All-days-all-files is infeasible (~18 h) on this no-index tiny-file feed, so we take the WHOLE day (NO 2500-file
cap) on 3 representative days/month across the vehicles' full window. That yields ~thousands of rows/vehicle/
month (vs ~90 in the old thin sample) — i.e. complete within-day resolution, enough to test whether a real
degradation signal exists at full resolution.

`SELECT *` -> every field (robust to the 2024->2025 schema migration). Consolidated one parquet/month.
  -> data/mahindra/native100/<YYYY-MM>.parquet
Run: .venv/bin/python src/import_mahindra_native_100.py
"""
import os, re, json, shutil
from pathlib import Path
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

os.chdir(Path(__file__).resolve().parent.parent); load_dotenv(".env")
B = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", config=Config(max_pool_connections=104, retries={"max_attempts": 5, "mode": "adaptive"}))
NAT = "battery-oem-data/parquet/mahindra/vehicle-data/"
OUT = Path("data/mahindra/native100"); OUT.mkdir(parents=True, exist_ok=True)
MIN_FREE_GB = 3.0
DAY_TARGETS = [8, 16, 24]     # 3 days/month; NO per-day file cap (complete days)

vins = pd.read_csv("data/manifests/mahindra_native_top100.csv")["vin"].astype(str).tolist()
EXPR = "SELECT * FROM s3object s WHERE s.vin IN (" + ",".join(f"'{v}'" for v in vins) + ")"
print(f"{len(vins)} vins | ALL fields (SELECT *) | {len(DAY_TARGETS)} days/month, NO file cap", flush=True)


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


def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))


months = [m for y in kids(NAT) for m in kids(y)]
print(f"{len(months)} months to sample", flush=True)
tot = 0
for m in months:
    ym = m.split("year=")[1].replace("/month=", "-").rstrip("/")
    outp = OUT / f"{ym}.parquet"
    if outp.exists():
        tot += len(pd.read_parquet(outp, columns=["vin"])); continue
    if shutil.disk_usage(".").free / 1e9 < MIN_FREE_GB:
        print(f"STOP — low disk ({shutil.disk_usage('.').free/1e9:.1f} GB)", flush=True); break
    dd = kids(m)
    if not dd:
        continue
    chosen = sorted({min(dd, key=lambda x: abs(dom(x) - t)) for t in DAY_TARGETS})
    rows = []
    for day in chosen:
        keys = allkeys(day)
        with ThreadPoolExecutor(max_workers=96) as pool:
            for res in pool.map(sel, keys):
                rows += res
        print(f"    {day.split('vehicle-data/')[1].rstrip('/')}: {len(keys):,} files scanned, {len(rows):,} rows so far", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(outp, index=False); tot += len(rows)
    print(f"  {ym}: {len(rows):,} rows -> {outp.name} | free {shutil.disk_usage('.').free/1e9:.1f} GB", flush=True)

print(f"\nDONE: {tot:,} rows for {len(vins)} vins across {len(list(OUT.glob('*.parquet')))} monthly files")
