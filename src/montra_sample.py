#!/usr/bin/env python3
"""Download a 10-VEHICLE SAMPLE of the Montra battery feed from S3 and stash it locally.

Montra (new OEM, VIN prefix P60…) has ~4 months (2024-07..10) in battery-oem-data/parquet/montra/battery-data/,
partitioned by day, tiny files (~2 VINs each). To onboard a POC we sample DAYS across the window (events for a
vin scatter across all daily files, so a day-sample ≈ a monthly sample per vehicle), pick the 10 best-covered
vehicles, and keep only their rows.  -> data/montra/sample_raw.parquet  (+ picks list)
Run: .venv/bin/python src/montra_sample.py
"""
import os, io, pathlib
from concurrent.futures import ThreadPoolExecutor
import boto3, pandas as pd, pyarrow.parquet as pq

os.chdir(pathlib.Path(__file__).resolve().parent.parent)
for ln in pathlib.Path(".env").read_text().splitlines():
    if "=" in ln and not ln.strip().startswith("#"):
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
S3 = boto3.client("s3", region_name="ap-south-1"); B = os.environ["S3_BUCKET"]
BASE = "battery-oem-data/parquet/montra/battery-data/"
COLS = ["vin", "eventAt", "soc", "current", "batteryPackVoltage", "resCapacity", "temperature", "odometer"]
SAMPLE_EVERY = 3          # take ~every 3rd available day per month (dense enough for monthly coulomb SoH)
N_VEHICLES = 10


def days_of(prefix):
    r = S3.list_objects_v2(Bucket=B, Prefix=prefix, Delimiter="/")
    return [x["Prefix"].split("/")[-2] for x in r.get("CommonPrefixes", [])]


def list_files(prefix):
    out = []
    for pg in S3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix):
        out += [o["Key"] for o in pg.get("Contents", []) if o["Key"].endswith(".parquet")]
    return out


def read_key(k):
    try:
        t = pq.read_table(io.BytesIO(S3.get_object(Bucket=B, Key=k)["Body"].read())).to_pandas()
        return t[[c for c in COLS if c in t.columns]]
    except Exception:
        return None


# 1) pick sample days across the window
keys = []
for y in days_of(BASE):
    for mth in days_of(f"{BASE}{y}/"):
        dd = sorted(days_of(f"{BASE}{y}/{mth}/"))
        for d in dd[::SAMPLE_EVERY]:
            keys += list_files(f"{BASE}{y}/{mth}/{d}/")
print(f"sampled {len(keys)} files across the window", flush=True)

# 2) threaded download + concat
frames = []
with ThreadPoolExecutor(max_workers=32) as ex:
    for i, df in enumerate(ex.map(read_key, keys), 1):
        if df is not None and len(df):
            frames.append(df)
        if i % 500 == 0:
            print(f"  read {i}/{len(keys)}", flush=True)
raw = pd.concat(frames, ignore_index=True)
raw["vin"] = raw["vin"].astype(str)
raw["t"] = pd.to_datetime(pd.to_numeric(raw["eventAt"], errors="coerce"), unit="ms")
raw = raw.dropna(subset=["t"])
raw["month"] = raw["t"].dt.to_period("M").dt.to_timestamp()
print(f"downloaded {len(raw):,} rows, {raw['vin'].nunique()} vehicles", flush=True)

# 3) pick the 10 best-covered vehicles (most months, then most rows)
cov = raw.groupby("vin").agg(rows=("t", "size"), months=("month", "nunique")).sort_values(
    ["months", "rows"], ascending=False)
picks = list(cov.head(N_VEHICLES).index)
print("picked 10 vehicles:", picks)
print(cov.head(N_VEHICLES).to_string())

out = raw[raw["vin"].isin(picks)].drop(columns=["month"]).sort_values(["vin", "t"]).reset_index(drop=True)
pathlib.Path("data/montra").mkdir(parents=True, exist_ok=True)
out.to_parquet("data/montra/sample_raw.parquet", index=False)
pd.Series(picks, name="vin").to_csv("data/montra/sample_vins.csv", index=False)
print(f"wrote data/montra/sample_raw.parquet ({len(out):,} rows) + sample_vins.csv")
print("DONE")
