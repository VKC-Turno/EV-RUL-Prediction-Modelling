#!/usr/bin/env python3
"""Derive the native-style proxies (km/%SoC discharge range, %SoC/hr charge rate) from the INTELLICAR feed and
test them against the coulomb SoH the SAME feed produces — the one place we can validate the proxies WITHOUT the
native/intellicar temporal-disjointness problem (intellicar carries soc+current+odometer+time simultaneously, so
proxy inputs and coulomb ground truth co-exist over the full 2022-2025 degradation).

Two questions this answers:
  1. Validation — does either proxy track the coulomb SoH? (within-vehicle-normalized pooled correlation + CI)
  2. Continuity — the per-(vin,month) proxy series it writes lets the dashboard overlay intellicar-era vs
     native-era proxies on one timeline for the vehicles that have both feeds.

Intellicar gotchas handled: timestamps are burst-duplicated (cadence ~0s) and soc carries out-of-range garbage
(seen up to 866) -> clamp soc to [0,100] and RESAMPLE to a fixed 2-min cadence (matching the native feed's
natural rate) before any per-step diff/crossing logic.

-> data/mahindra/intellicar_proxy_monthly.parquet   (vin, month, ic_km_soc, ic_chg_rate, coulomb)
-> data/mahindra/intellicar_proxy_summary.json       (fleet-level proxy-vs-coulomb correlation + 95% CI)
Run: .venv/bin/python src/intellicar_proxy_prep.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds

os.chdir(Path(__file__).resolve().parent.parent)
IC_DIR = "data/mahindra/intellicar"
FEAT = "data/redshift/mahindra_featengg.parquet"
OUT_MONTHLY = "data/mahindra/intellicar_proxy_monthly.parquet"
OUT_SUMMARY = "data/mahindra/intellicar_proxy_summary.json"


def km_soc_monthly(R):
    """km per %SoC over driving segments (odo up, soc down) on the 2-min-resampled series, aggregated monthly."""
    R = R.copy()
    R["do"] = R.odo.diff(); R["ds"] = -R.soc.diff(); R["dm"] = R.t.diff().dt.total_seconds() / 60
    s = R[R.do.between(0.1, 80) & R.ds.between(0.5, 40) & R.dm.between(0.1, 180)].copy()
    if not len(s):
        return pd.Series(dtype=float)
    s["month"] = s.t.dt.to_period("M").dt.to_timestamp()
    r = s.groupby("month").agg(o=("do", "sum"), sc=("ds", "sum"), n=("do", "size"))
    r = r[r.n >= 3]; v = 100 * r.o / r.sc
    return v[v.between(20, 400)]


def chg_rate_monthly(R, LO=30, HI=70):
    """%SoC per hour crossing a fixed CC-phase window (up through LO then HI in one charge), consistent-charger
    filtered (within +-25-33% of the vehicle's median rate), aggregated monthly."""
    soc = R.soc.values; ts = R.t.values.astype("datetime64[s]").astype("int64"); tt = R.t.values
    if len(soc) < 10:
        return pd.Series(dtype=float)
    sh = np.empty_like(soc); sh[0] = soc[0]; sh[1:] = soc[:-1]
    ulo = np.where((soc >= LO) & (sh < LO))[0]
    uhi = np.where((soc >= HI) & (sh < HI))[0]
    dn = np.where(soc < sh - 3)[0]
    ev = []
    for lo in ulo:
        da = dn[dn > lo]; nd = da[0] if len(da) else len(soc) + 1
        ha = uhi[(uhi > lo) & (uhi < nd)]
        if len(ha):
            hrs = (ts[ha[0]] - ts[lo]) / 3600.0
            if 0.15 < hrs < 10:
                ev.append((tt[lo], (HI - LO) / hrs))
    if not ev:
        return pd.Series(dtype=float)
    e = pd.DataFrame(ev, columns=["t", "rate"]); md = e.rate.median()
    e = e[(e.rate / md).between(0.75, 1.33)]
    return e.assign(m=pd.to_datetime(e.t).dt.to_period("M").dt.to_timestamp()).groupby("m")["rate"].median()


print("loading intellicar raw (all files, columns pruned)...", flush=True)
tbl = ds.dataset(IC_DIR, format="parquet").to_table(columns=["vin", "eventAt", "soc", "odometer"])
D = tbl.to_pandas(); del tbl
D["vin"] = D.vin.astype(str)
D["t"] = pd.to_datetime(pd.to_numeric(D.eventAt, errors="coerce"), unit="ms")
D["soc"] = pd.to_numeric(D.soc, errors="coerce").astype("float32")
D["odo"] = pd.to_numeric(D.odometer, errors="coerce").astype("float32")
D = D[D.soc.between(0, 100)].dropna(subset=["t"])[["vin", "t", "soc", "odo"]]
print(f"loaded {len(D):,} rows across {D.vin.nunique()} vins", flush=True)

rows = []
vins = list(D.vin.unique())
for i, v in enumerate(vins):
    g = D[D.vin == v].sort_values("t")
    R = g.set_index("t").resample("2min").agg(odo=("odo", "last"), soc=("soc", "last")).dropna().reset_index()
    if len(R) < 20:
        continue
    km = km_soc_monthly(R); cr = chg_rate_monthly(R)
    m = pd.concat([km.rename("ic_km_soc"), cr.rename("ic_chg_rate")], axis=1)
    m.index.name = "month"; m = m.reset_index(); m["vin"] = v
    rows.append(m)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(vins)} vins", flush=True)

P = pd.concat(rows, ignore_index=True)
c = pd.read_parquet(FEAT); c["vin"] = c["vin"].astype(str); c["month"] = pd.to_datetime(c["ymd"].astype(str))
P = P.merge(c[["vin", "month", "soh"]].rename(columns={"soh": "coulomb"}), on=["vin", "month"], how="left")
P.to_parquet(OUT_MONTHLY, index=False)
print(f"wrote {OUT_MONTHLY}: {len(P):,} vin-months, {P.vin.nunique()} vins", flush=True)


def fleet_corr(P, proxy):
    """within-vehicle-normalized pooled correlation of proxy vs coulomb (a real proxy => strongly POSITIVE)."""
    d = P.dropna(subset=[proxy, "coulomb"]).copy()
    cnt = d.groupby("vin").size(); d = d[d.vin.isin(cnt[cnt >= 4].index)]
    z = lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
    d["pz"] = d.groupby("vin")[proxy].transform(z); d["cz"] = d.groupby("vin")["coulomb"].transform(z)
    d = d.dropna(subset=["pz", "cz"])
    d = d[np.isfinite(d.pz) & np.isfinite(d.cz)]
    if len(d) < 10:
        return dict(proxy=proxy, vehicles=int(d.vin.nunique()), pairs=int(len(d)), r=None, ci=None)
    x = d.pz.values; y = d.cz.values; r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(0)
    boot = [np.corrcoef(x[(idx := rng.integers(0, len(x), len(x)))], y[idx])[0, 1] for _ in range(2000)]
    return dict(proxy=proxy, vehicles=int(d.vin.nunique()), pairs=int(len(d)), r=r,
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


summ = {p: fleet_corr(P, p) for p in ["ic_km_soc", "ic_chg_rate"]}
json.dump(summ, open(OUT_SUMMARY, "w"), indent=2)
print("\n=== fleet-level proxy vs coulomb (within-vehicle normalized) ===")
print(json.dumps(summ, indent=2))
