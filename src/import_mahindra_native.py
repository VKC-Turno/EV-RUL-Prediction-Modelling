#!/usr/bin/env python3
"""Efficient WHOLE-FLEET Mahindra NATIVE downloader — monthly-sampled, CONSOLIDATED (one parquet per month).

The native feed now covers ~the whole Mahindra fleet (~12k VINs) and is dense (hundreds of rows/file), so
we pull a monthly sample of ALL vehicles into data/mahindra/native_monthly/<YYYY-MM>.parquet. Consolidating
(one file/month) avoids the per-source-file tiny-output explosion that would need ~50 GB for the fleet — this
fits in ~2 GB.

Schema migrated 2024->2025 (dropped batteryTemp/kwh/state; added vehicleStatus/vehicleMode/vehicleSpeed/
keyStatus), so per-year column sets are pulled with a common-cols fallback so no file is lost to a schema
mismatch. eventAt/soc/odometer/distanceToEmpty/gps exist in every year.

Disk-guarded: stops cleanly before free space falls below MIN_FREE_GB. Resumable (skips months already done).
Run: .venv/bin/python src/import_mahindra_native.py
"""
import os, re, json, shutil
from pathlib import Path
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

os.chdir(Path(__file__).resolve().parent.parent); load_dotenv(".env")
B = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", config=Config(max_pool_connections=60, retries={"max_attempts": 5, "mode": "adaptive"}))
NAT = "battery-oem-data/parquet/mahindra/vehicle-data/"
OUT = Path("data/mahindra/native_monthly"); OUT.mkdir(parents=True, exist_ok=True)

MIN_FREE_GB = 4.0             # hard stop before the disk gets dangerously full
CAP = 2500                    # files/day sampled (enough to cover ~the whole active fleet that day)
DAY_TARGET = 15              # 1 representative day/month, nearest the 15th
COMMON = ["vin", "eventAt", "soc", "odometer", "distanceToEmpty", "gearPosition", "latitude", "longitude"]
COLS_2024 = COMMON + ["batteryTemp", "kwh", "state"]
COLS_NEW = COMMON + ["vehicleStatus", "vehicleMode", "vehicleSpeed", "keyStatus"]


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def free_gb():
    return shutil.disk_usage(".").free / 1e9


def _select(k, cols):
    expr = "SELECT " + ", ".join(f"s.{c}" for c in cols) + " FROM s3object s"
    resp = s3.select_object_content(Bucket=B, Key=k, ExpressionType="SQL", Expression=expr,
        InputSerialization={"Parquet": {}}, OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
    buf = bytearray()
    for ev in resp["Payload"]:
        if "Records" in ev:
            buf += ev["Records"]["Payload"]
    return [json.loads(l) for l in buf.decode().splitlines() if l.strip()]


def sel(k, cols):
    try:
        return _select(k, cols)
    except Exception:
        try:
            return _select(k, COMMON)     # fall back to columns present in every batch
        except Exception:
            return []


def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))


months = [m for y in kids(NAT) for m in kids(y)]
print(f"{len(months)} months to sample | free {free_gb():.1f} GB | guard {MIN_FREE_GB} GB", flush=True)
seen = set(); rows_total = 0
for m in months:
    ym = m.split("year=")[1].replace("/month=", "-").rstrip("/")
    outp = OUT / f"{ym}.parquet"
    if outp.exists():
        seen |= set(pd.read_parquet(outp, columns=["vin"])["vin"].astype(str)); continue
    if free_gb() < MIN_FREE_GB:
        print(f"STOP — free {free_gb():.1f} GB < {MIN_FREE_GB} GB guard", flush=True); break
    cols = COLS_2024 if int(ym[:4]) <= 2024 else COLS_NEW
    dd = kids(m)
    if not dd:
        continue
    day = min(dd, key=lambda x: abs(dom(x) - DAY_TARGET))
    r = s3.list_objects_v2(Bucket=B, Prefix=day)
    keys = [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")][:CAP]
    rows = []
    with ThreadPoolExecutor(max_workers=56) as pool:
        for res in pool.map(lambda k: sel(k, cols), keys):
            rows += res
    if not rows:
        continue
    df = pd.DataFrame(rows)
    df.to_parquet(outp, index=False)
    seen |= set(df["vin"].astype(str)); rows_total += len(df)
    print(f"  {ym}: {len(df):,} rows, {df['vin'].nunique()} vins ({day.split('vehicle-data/')[1].rstrip('/')}) "
          f"-> cumulative {len(seen)} vins | free {free_gb():.1f} GB", flush=True)

print(f"\nDONE: {rows_total:,} rows, {len(seen)} distinct vins across "
      f"{len(list(OUT.glob('*.parquet')))} monthly files, {sum(f.stat().st_size for f in OUT.glob('*.parquet'))/1e6:.0f} MB")
