#!/usr/bin/env python3
"""Precompute the both-feed coulomb cross-validation for the explorer (Section 11):
  - overlapping (vin, month) pairs: native distance-per-SoC proxy vs intellicar COULOMB SoH (ground truth),
    each normalised within-vehicle.
  - monthly feed coverage (coulomb vs native vehicle-months) — shows WHY the overlap is thin (feeds cover
    different periods; the cohort migrated intellicar -> native).
  -> data/mahindra/crossval_pairs.parquet, data/mahindra/crossval_coverage.csv
Run: .venv/bin/python src/crossval_prep.py   (after src/import_mahindra_bothfeeds.py)
"""
import glob
import numpy as np, pandas as pd

fs = sorted(glob.glob("data/mahindra/bothfeeds_native/*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
df["t"] = pd.to_datetime(pd.to_numeric(df.eventAt, errors="coerce"), unit="ms")
df["soc"] = pd.to_numeric(df.soc, errors="coerce"); df["odo"] = pd.to_numeric(df.odometer, errors="coerce")
df["vin"] = df.vin.astype(str)
df = df[df.soc.between(0, 100) & df.odo.between(0, 300000)].dropna(subset=["t"]).sort_values(["vin", "t"])
df["d_odo"] = df.groupby("vin")["odo"].diff(); df["d_soc"] = -df.groupby("vin")["soc"].diff()
df["dtm"] = df.groupby("vin")["t"].diff().dt.total_seconds() / 60
seg = df[df.d_odo.between(0.1, 80) & df.d_soc.between(0.5, 40) & df.dtm.between(0.1, 180)].copy()
seg["month"] = seg["t"].dt.to_period("M").dt.to_timestamp()
vm = seg.groupby(["vin", "month"]).agg(odo=("d_odo", "sum"), soc=("d_soc", "sum"), n=("d_odo", "size")).reset_index()
vm = vm[vm.n >= 3].copy(); vm["nrange"] = 100 * vm.odo / vm.soc; vm = vm[vm.nrange.between(20, 400)]

c = pd.read_parquet("data/redshift/mahindra_featengg.parquet"); c["vin"] = c["vin"].astype(str)
c["month"] = pd.to_datetime(c["ymd"].astype(str))
bf = set(pd.read_csv("data/manifests/mahindra_bothfeeds_vins.csv")["vin"].astype(str))
cc = c[c.vin.isin(bf)]
cs = cc[["vin", "month", "soh"]].rename(columns={"soh": "coulomb"})

M = vm.merge(cs, on=["vin", "month"], how="inner")
cnt = M.groupby("vin").size(); M = M[M.vin.isin(cnt[cnt >= 3].index)].copy()
z = lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
M["nz"] = M.groupby("vin")["nrange"].transform(z); M["cz"] = M.groupby("vin")["coulomb"].transform(z)
M[["vin", "month", "nrange", "coulomb", "nz", "cz"]].to_parquet("data/mahindra/crossval_pairs.parquet", index=False)

cov = pd.concat([cc["month"].dt.to_period("M").value_counts().rename("coulomb"),
                 vm["month"].dt.to_period("M").value_counts().rename("native")], axis=1).fillna(0).astype(int).sort_index()
cov.index = cov.index.astype(str)
cov.to_csv("data/mahindra/crossval_coverage.csv")
print(f"wrote crossval_pairs ({len(M)} pairs, {M.vin.nunique()} vehicles) + crossval_coverage ({len(cov)} months)")
