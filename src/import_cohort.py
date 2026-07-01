#!/usr/bin/env python3
"""Cohort import for SoH FORECASTING: the most-aged vehicles present in BOTH feeds.

Pulls forecasting-relevant columns (not just SoH-target columns) for the cohort in
data/forecast_cohort.csv, monthly-sampled across each feed's full history.
Usage: import_cohort.py <intellicar|mahindra>
  intellicar -> electrical signals (current/voltage/soc) for coulomb counting, C-rate,
                Ah-throughput, DoD, SoC-dwell features. 3 days/month (dense files, cheap).
  mahindra   -> usage + environment (odometer, location, gearPosition, sparse temp/state)
                for thermal/usage/driving features. 1 day/month (tiny files, expensive).
"""
import os, sys, json, re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as _cfg                                 # OEM registry (prefixes, per-OEM columns, cadence)

FEED = sys.argv[1]                                   # 'intellicar' | the OEM's own feed name
OEM = sys.argv[2] if len(sys.argv) > 2 else "mahindra"
os.chdir(Path(__file__).resolve().parent.parent)    # repo root -> data/ paths resolve
load_dotenv(".env")
B = os.environ["S3_BUCKET"]
COHORT = pd.read_csv(f"data/manifests/{OEM}_cohort.csv")["vin"].tolist()

# Config-driven so a new OEM is a src/config.py entry, not a code fork. Columns/prefix/cadence come from
# INTELLICAR (shared feed) or OEM_FEEDS[<oem>] (native feed). `current` is reserved -> quoted in SELECT below.
_IC_DEFAULT = ["vin", "eventAt", "make", "model", "soc", "current", "batteryVoltage", "odometer", "dte"]
if FEED == "intellicar":
    CFG = dict(
        prefix=_cfg.INTELLICAR["prefix"], out=f"data/{OEM}/intellicar",
        cap=_cfg.FILES_PER_DAY_CAP.get("intellicar", 2500),
        days_per_month=_cfg.DAYS_PER_MONTH.get("intellicar", [8, 16, 24]),
        skip_year=_cfg.INTELLICAR.get("skip_year", "0000"),
        cols=_cfg.INTELLICAR["cols_by_oem"].get(OEM, _IC_DEFAULT),
        split=_cfg.INTELLICAR["split"])
elif FEED in _cfg.OEM_FEEDS:
    f = _cfg.OEM_FEEDS[FEED]
    CFG = dict(
        prefix=f["prefix"], out=f"data/{OEM}/feed", cap=_cfg.FILES_PER_DAY_CAP.get("default", 15000),
        days_per_month=([8, 16, 24] if f.get("dense_files") else _cfg.DAYS_PER_MONTH.get("default", [15])),
        skip_year=None, cols=f["cols"], split=f["split"])
else:
    raise SystemExit(f"unknown feed '{FEED}' — add it to src/config.py OEM_FEEDS")

OUT = Path(CFG["out"]); OUT.mkdir(parents=True, exist_ok=True)
MW = 44
s3 = boto3.client("s3", config=Config(max_pool_connections=MW + 8, retries={"max_attempts": 5, "mode": "adaptive"}))
COLS = CFG["cols"]
_cols_sql = ", ".join(f's."{c}"' if c == "current" else f"s.{c}" for c in COLS)
_vins_sql = ", ".join(f"'{v}'" for v in COHORT)
EXPR = f"SELECT {_cols_sql} FROM s3object s WHERE s.vin IN ({_vins_sql})"


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


def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))


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
        return (0, Counter())
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_parquet(op, index=False)
    return (len(df), Counter(df["vin"]))


# Build the list of sample days: days_per_month nearest to each target DOM, per month.
sample_days = []
for y in kids(CFG["prefix"]):
    if CFG["skip_year"] and f"year={CFG['skip_year']}" in y:
        continue
    for m in kids(y):
        dd = kids(m)
        if not dd:
            continue
        chosen = set()
        for target in CFG["days_per_month"]:
            chosen.add(min(dd, key=lambda x: abs(dom(x) - target)))
        sample_days += sorted(chosen)
print(f"{FEED}: cohort {len(COHORT)} VINs | {len(sample_days)} sample days "
      f"({CFG['days_per_month']} days/month)", flush=True)

totals = Counter(); n_rows = err = 0
for di, day in enumerate(sample_days, 1):
    keys = list_keys(day, CFG["cap"])
    tag = day.split(CFG["split"])[1].rstrip("/")
    with ThreadPoolExecutor(max_workers=MW) as pool:
        futs = {pool.submit(extract_one, k): k for k in keys}
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:
                err += 1
                if err <= 3:
                    print(f"  err {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            if res is None:
                continue
            rows, per = res
            n_rows += rows
            totals.update(per)
    if di % 12 == 0 or di == len(sample_days):
        print(f"  [{di}/{len(sample_days)}] {tag}: cumulative {n_rows:,} rows | "
              f"VINs seen {len(totals)} | err {err}", flush=True)

print(f"\nDONE {FEED}. {n_rows:,} rows, {len(totals)}/{len(COHORT)} cohort VINs. Errors: {err}")
for v in COHORT:
    print(f"  {v}: {totals.get(v, 0):,}")
