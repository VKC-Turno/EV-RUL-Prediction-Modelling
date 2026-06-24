#!/usr/bin/env python3
"""Batch dense import of MANY Bajaj vehicles in ONE S3-Select pass (WHERE vin IN ...).

The per-VIN import (import_bajaj_dense.py) samples the SAME partition days for every vehicle, so
scanning each sampled day's files once and selecting `vin IN (...)` extracts all requested VINs at
once — ~Nx cheaper. Bajaj-native schema (essBms*/etsVcu*/hmiIcl*); NO current/voltage/remaining-
capacity, so the target is the reported SoH (essBmsSohcEstPercValue). Feed span ~2025-09..2026-06.

Usage: python src/import_bajaj_batch.py <vin_list_file> [days_per_month] [max_workers]
  -> data/bajaj/dense/<VIN>.parquet   (one per VIN that returned rows)
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
VIN_FILE = sys.argv[1]
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MW = int(sys.argv[3]) if len(sys.argv) > 3 else 64
TARGET_DOMS = [8, 16, 24][:N_DAYS]

VINS = [v.strip() for v in open(VIN_FILE) if v.strip()]
COLS = ["eventAt", "vin",
        "essBmsSocEstPercValue", "essBmsSohcEstPercValue", "essBmsChgcycleActCountValue",
        "essBmsTemperatureActDegcValue", "etsVcuAmbienttempActDegcValue",
        "etsVcuDriveeffEstWhpkmValue", "evcChgInputenergycountActKwhValue", "hmiIclOdoActMValue"]
NUM_COLS = COLS[2:]
vin_in = ", ".join(f"'{v}'" for v in VINS)
EXPR = f"SELECT {', '.join('s.' + c for c in COLS)} FROM s3object s WHERE s.vin IN ({vin_in})"

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
    days = []
    for y in kids(PREFIX):
        for mo in kids(y):
            dd = kids(mo)
            if not dd:
                continue
            for target in TARGET_DOMS:
                days.append(min(dd, key=lambda d: abs(dom(d) - target)))
    days = sorted(set(days))
    print(f"{len(VINS)} VINs | {len(days)} sample days ({N_DAYS}/month), pulling all files each", flush=True)

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
        if done % 4 == 0 or done == len(days):
            print(f"  [{done}/{len(days)}] {day.split('vehicle-data/')[1].rstrip('/')}: "
                  f"cum rows {len(rows):,}, VINs seen {len({r['vin'] for r in rows})}", flush=True)

    if not rows:
        print("NO rows returned — check VINs / partitions."); return
    df = pd.DataFrame(rows).reindex(columns=COLS)            # S3 Select drops null fields
    df["t"] = pd.to_datetime(df["eventAt"].astype("int64"), unit="ms")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = Path("data/bajaj/dense"); out.mkdir(parents=True, exist_ok=True)
    got = 0
    for vin, g in df.groupby("vin"):
        g.sort_values("t").reset_index(drop=True).to_parquet(out / f"{vin}.parquet", index=False)
        got += 1
    miss = sorted(set(VINS) - set(df["vin"].unique()))
    print(f"DONE: {len(df):,} rows -> {got}/{len(VINS)} VINs written to data/bajaj/dense/", flush=True)
    print(f"  no data returned for {len(miss)} VINs", flush=True)


if __name__ == "__main__":
    main()
