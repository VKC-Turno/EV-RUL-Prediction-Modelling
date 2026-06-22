#!/usr/bin/env python3
"""Monthly-sample import of SPECIFIC Mahindra vins from intellicar or the OEM feed.

intellicar -> has signed current (coulomb SoH); feed -> soc/odometer only (distance-per-SoC SoH).
Usage: import_mahindra_vins.py <intellicar|feed> <vin1,vin2,...> [days_per_month] [mw]
-> data/mahindra/extra/<source>/<vin>.parquet
"""
import os, sys, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

os.chdir(Path(__file__).resolve().parent.parent)
load_dotenv(".env")
B = os.environ["S3_BUCKET"]
SRC = sys.argv[1]
VINS = sys.argv[2].split(",")
NDAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MW = int(sys.argv[4]) if len(sys.argv) > 4 else 48

CFG = {
    "intellicar": dict(prefix="battery-oem-data/parquet/intellicar/battery-data/", cap=2500, skip="0000",
                       cols=["vin", "eventAt", "make", "model", "soc", "current", "batteryVoltage", "odometer", "dte"]),
    "feed": dict(prefix="battery-oem-data/parquet/mahindra/vehicle-data/", cap=15000, skip=None,
                 cols=["eventAt", "vin", "soc", "odometer", "distanceToEmpty", "batteryTemp", "state", "kwh", "vehicleModel"]),
}[SRC]
s3 = boto3.client("s3", config=Config(max_pool_connections=MW + 8, retries={"max_attempts": 6, "mode": "adaptive"}))
COLS = CFG["cols"]
colsql = ", ".join(f's."{c}"' if c == "current" else f"s.{c}" for c in COLS)
vinsql = ", ".join(f"'{v}'" for v in VINS)
EXPR = f"SELECT {colsql} FROM s3object s WHERE s.vin IN ({vinsql})"


def kids(p):
    r = s3.list_objects_v2(Bucket=B, Prefix=p, Delimiter="/")
    return sorted(x["Prefix"] for x in r.get("CommonPrefixes", []))


def dom(d):
    return int(re.search(r"day=(\d{2})", d).group(1))


def sel(k):
    r = s3.select_object_content(Bucket=B, Key=k, ExpressionType="SQL", Expression=EXPR,
                                 InputSerialization={"Parquet": {}}, OutputSerialization={"JSON": {"RecordDelimiter": "\n"}})
    buf = bytearray()
    for ev in r["Payload"]:
        if "Records" in ev:
            buf += ev["Records"]["Payload"]
    return [json.loads(l) for l in buf.decode().splitlines() if l.strip()]


def main():
    days = []
    for y in kids(CFG["prefix"]):
        if CFG["skip"] and f"year={CFG['skip']}" in y:
            continue
        for mo in kids(y):
            dd = kids(mo)
            if dd:
                for t in [15, 8, 22][:NDAYS]:
                    days.append(min(dd, key=lambda d: abs(dom(d) - t)))
    days = sorted(set(days))
    print(f"{SRC}: {len(VINS)} vins, {len(days)} sample days", flush=True)
    rows, done = [], 0
    for day in days:
        keys = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=day)
                for o in pg.get("Contents", []) if o["Key"].endswith(".parquet")][:CFG["cap"]]
        with ThreadPoolExecutor(max_workers=MW) as pool:
            for f in as_completed([pool.submit(sel, k) for k in keys]):
                try:
                    rows += f.result()
                except Exception:
                    pass
        done += 1
        if done % 6 == 0 or done == len(days):
            print(f"  [{done}/{len(days)}] cum {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows)
    out = Path(f"data/mahindra/extra/{SRC}"); out.mkdir(parents=True, exist_ok=True)
    if len(df):
        df["t"] = pd.to_datetime(df["eventAt"].astype("int64"), unit="ms", errors="coerce")
        for v, g in df.groupby("vin"):
            g.to_parquet(out / f"{v}.parquet", index=False)
            print(f"  saved {v}: {len(g)} rows ({g['t'].min().date()}..{g['t'].max().date()})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
