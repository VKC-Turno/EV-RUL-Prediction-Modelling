#!/usr/bin/env python3
"""Dense import of ONE Bajaj vehicle for reported-SoH + degradation features.

Bajaj's native feed (battery-oem-data/parquet/bajaj/vehicle-data/) uses verbose signal names
(essBms*/etsVcu*/hmiIcl*), one Value per signal per row, multi-VIN dense files (~400 rows/file,
~2k files/day). Crucially it has NO current, NO voltage, NO remaining-capacity — so neither
coulomb counting nor Euler's BMS-capacity SoH is possible. The usable target is the BMS-reported
SoH `essBmsSohcEstPercValue` (clean, monotone; see bajaj_features.py). History is short
(2025-09..2026-06, ~10 months), so we sample a few days/month and pull ALL files of those days
(a VIN's rows are spread ~1/file across the day partition).

Usage: import_bajaj_dense.py <VIN> [days_per_month] [max_workers]  -> data/bajaj/dense/<VIN>.parquet
"""
import os, sys, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

os.chdir(Path(__file__).resolve().parent.parent)
load_dotenv(".env", override=True)
B = os.environ["S3_BUCKET"]
PREFIX = "battery-oem-data/parquet/bajaj/vehicle-data/"
VIN = sys.argv[1] if len(sys.argv) > 1 else "MD2B94301RWA17513"
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TARGET_DOMS = [8, 16, 24][:N_DAYS]
MW = int(sys.argv[3]) if len(sys.argv) > 3 else 64

# Bajaj-native columns. *Value holds the signal; *Time twins are ignored. No current/voltage/remcap.
COLS = ["eventAt", "vin",
        "essBmsSocEstPercValue", "essBmsSohcEstPercValue", "essBmsChgcycleActCountValue",
        "essBmsTemperatureActDegcValue", "etsVcuAmbienttempActDegcValue",
        "etsVcuDriveeffEstWhpkmValue", "evcChgInputenergycountActKwhValue", "hmiIclOdoActMValue"]
NUM_COLS = COLS[2:]
EXPR = f"SELECT {', '.join('s.' + c for c in COLS)} FROM s3object s WHERE s.vin='{VIN}'"

s3 = boto3.client("s3", config=Config(max_pool_connections=MW + 8,
                                      retries={"max_attempts": 6, "mode": "adaptive"}))


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))


def day_files(day):
    keys = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=day):
        keys += [o["Key"] for o in pg.get("Contents", []) if o["Key"].endswith(".parquet")]
    return keys


def sel(k):
    r = s3.select_object_content(Bucket=B, Key=k, ExpressionType="SQL", Expression=EXPR,
                                 InputSerialization={"Parquet": {}},
                                 OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
    buf = bytearray()
    for ev in r["Payload"]:
        if "Records" in ev:
            buf += ev["Records"]["Payload"]
    return [json.loads(l) for l in buf.decode().splitlines() if l.strip()]


def main():
    # pick N representative days per month across the vehicle's full history
    days = []
    for y in kids(PREFIX):
        for mo in kids(y):
            dd = kids(mo)
            if not dd:
                continue
            for target in TARGET_DOMS:
                days.append(min(dd, key=lambda d: abs(dom(d) - target)))
    days = sorted(set(days))
    print(f"{VIN}: {len(days)} sample days ({N_DAYS}/month), pulling all files each", flush=True)

    rows, done = [], 0
    for day in days:
        keys = day_files(day)
        with ThreadPoolExecutor(max_workers=MW) as pool:
            for f in as_completed([pool.submit(sel, k) for k in keys]):
                try:
                    rows += f.result()
                except Exception:
                    pass
        done += 1
        if done % 6 == 0 or done == len(days):
            print(f"  [{done}/{len(days)}] {day.split('vehicle-data/')[1].rstrip('/')}: "
                  f"cum rows {len(rows):,}", flush=True)

    df = pd.DataFrame(rows).reindex(columns=COLS)   # S3 Select drops null fields -> reindex
    df["t"] = pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("t").reset_index(drop=True)
    out = Path("data/bajaj/dense"); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{VIN}.parquet"
    df.to_parquet(fp, index=False)
    soh = df["essBmsSohcEstPercValue"]
    print(f"DONE {len(df):,} rows -> {fp}  | {df['t'].min().date()}..{df['t'].max().date()} | "
          f"SoH fill {100*soh.notna().mean():.0f}% range {soh.min():.0f}-{soh.max():.0f}", flush=True)


if __name__ == "__main__":
    main()
