#!/usr/bin/env python3
"""Import ONE vehicle (present in both feeds) from the mahindra OEM feed, for SoH comparison
against its intellicar coulomb-counting SoH. Monthly sample, useful 9 cols. -> data/compare_mahindra/"""
import os, json, re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv("/home/hj/Desktop/EULER_RUL_MODEL/.env")
B=os.environ["S3_BUCKET"]
PREFIX="battery-oem-data/parquet/mahindra/vehicle-data/"
OUT=Path("data/compare_mahindra"); OUT.mkdir(parents=True,exist_ok=True)
VIN="MB7F8CLLFNJH48488"
CAP=25000; MW=48; TARGET=15
s3=boto3.client("s3",config=Config(max_pool_connections=MW+8,retries={"max_attempts":5,"mode":"adaptive"}))
COLS=["eventAt","vin","soc","odometer","distanceToEmpty","state","batteryTemp","kwh","vehicleModel"]
def kids(p):
    r=s3.list_objects_v2(Bucket=B,Prefix=p,Delimiter="/"); return sorted(x["Prefix"] for x in r.get("CommonPrefixes",[]))
def list_keys(p,cap=None):
    ks=[];pag=s3.get_paginator("list_objects_v2")
    for pg in pag.paginate(Bucket=B,Prefix=p):
        for o in pg.get("Contents",[]):
            if o["Key"].endswith(".parquet"):
                ks.append(o["Key"])
                if cap and len(ks)>=cap: return ks
    return ks
def dom(d): return int(re.search(r"day=(\d{2})",d).group(1))
days=[]
for y in kids(PREFIX):
    for m in kids(y):
        dd=kids(m)
        if dd: days.append(min(dd,key=lambda x:abs(dom(x)-TARGET)))
print(f"VIN {VIN}: {len(days)} monthly sample days",flush=True)
cs=", ".join(f"s.{c}" for c in COLS)
EXPR=f"SELECT {cs} FROM s3object s WHERE s.vin = '{VIN}'"
def outp(k):
    rel=k[len(PREFIX):] if k.startswith(PREFIX) else k
    return OUT/(rel.replace("/","__").replace(".parquet","")+".parquet")
def one(k):
    op=outp(k)
    if op.exists(): return None
    r=s3.select_object_content(Bucket=B,Key=k,ExpressionType="SQL",Expression=EXPR,
        InputSerialization={"Parquet":{}},OutputSerialization={"JSON":{"RecordDelimiter":"\n"}})
    buf=bytearray()
    for e in r["Payload"]:
        if "Records" in e: buf+=e["Records"]["Payload"]
    rows=[json.loads(l) for l in buf.decode().splitlines() if l.strip()]
    if not rows: return 0
    pd.DataFrame(rows).reindex(columns=COLS).to_parquet(op,index=False); return len(rows)
tot=0;err=0
for i,day in enumerate(days,1):
    ks=list_keys(day,cap=CAP); tag=day.split("vehicle-data/")[1].rstrip("/")
    with ThreadPoolExecutor(max_workers=MW) as pool:
        futs={pool.submit(one,k):k for k in ks}
        for f in as_completed(futs):
            try:
                r=f.result()
                if r: tot+=r
            except Exception as e:
                err+=1
                if err<=3: print("  err",type(e).__name__,str(e)[:80],flush=True)
    print(f"  [{i}/{len(days)}] {tag}: {len(ks)} files | cumulative {tot:,} rows | err {err}",flush=True)
print(f"\nDONE. {tot:,} rows for {VIN}. Errors: {err}")
