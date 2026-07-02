#!/usr/bin/env python3
"""Download NATIVE data for the ~220 BOTH-FEEDS Mahindra vehicles (they also have an intellicar coulomb SoH),
to cross-validate the native distance-per-SoC proxy against the coulomb ground truth.

Moderate density (cap files/day) so it finishes fast; soc+odometer+eventAt only. 3 sampled days/month.
  -> data/mahindra/bothfeeds_native/<YYYY-MM>.parquet
Run: .venv/bin/python src/import_mahindra_bothfeeds.py
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
OUT = Path("data/mahindra/bothfeeds_native"); OUT.mkdir(parents=True, exist_ok=True)
CAP = 15000; DAY_TARGETS = [8, 16, 24]; MIN_FREE_GB = 3.0

vins = pd.read_csv("data/manifests/mahindra_bothfeeds_vins.csv")["vin"].astype(str).tolist()
EXPR = "SELECT s.vin, s.eventAt, s.soc, s.odometer FROM s3object s WHERE s.vin IN (" + ",".join(f"'{v}'" for v in vins) + ")"
print(f"{len(vins)} both-feeds vins | 3 days/month | cap {CAP} files/day", flush=True)


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


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
tot = 0
for m in months:
    ym = m.split("year=")[1].replace("/month=", "-").rstrip("/")
    outp = OUT / f"{ym}.parquet"
    if outp.exists():
        continue
    if shutil.disk_usage(".").free / 1e9 < MIN_FREE_GB:
        print("STOP low disk", flush=True); break
    dd = kids(m)
    if not dd:
        continue
    chosen = sorted({min(dd, key=lambda x: abs(dom(x) - t)) for t in DAY_TARGETS})
    rows = []
    for day in chosen:
        r = s3.list_objects_v2(Bucket=B, Prefix=day)
        keys = [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")][:CAP]
        with ThreadPoolExecutor(max_workers=96) as pool:
            for res in pool.map(sel, keys):
                rows += res
    if rows:
        pd.DataFrame(rows).to_parquet(outp, index=False); tot += len(rows)
    print(f"  {ym}: {len(rows):,} rows (cum {tot:,}) | free {shutil.disk_usage('.').free/1e9:.1f} GB", flush=True)

print(f"\nDONE: {tot:,} rows across {len(list(OUT.glob('*.parquet')))} months, {len(vins)} vins target")
