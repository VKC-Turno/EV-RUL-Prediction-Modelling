#!/usr/bin/env python3
"""Shared, source-agnostic definitions of the two behavioural SoH proxies, so every feed (Euler, Bajaj, Piaggio,
Mahindra-intellicar, Mahindra-native) computes them IDENTICALLY. Used by the cross-source probabilistic-SoH
experiment (train proxy->SoH on feeds that HAVE a SoH, apply to Mahindra-native which does not).

Proxies (both need only soc, odometer, timestamp — the columns the native feed also has):
  * km_soc   — km driven per %SoC over discharge segments (a capacity/efficiency range proxy).
  * chg_rate — %SoC per hour crossing a fixed 30->70% CC-phase charge window, consistent-charger filtered
               (a charge-acceptance proxy; rises as capacity fades since a constant current fills a smaller pack).

Feed gotchas handled once, here: soc clamped to [0,100] (some feeds emit garbage > 100), and every series is
RESAMPLED to a fixed 2-min cadence before any per-step diff/crossing — intellicar-style feeds burst many rows on
one millisecond, which destroys naive per-row deltas.
"""
import numpy as np, pandas as pd


def _km_soc_month(R):
    R = R.copy()
    R["do"] = R.odo.diff(); R["ds"] = -R.soc.diff(); R["dm"] = R.t.diff().dt.total_seconds() / 60
    s = R[R.do.between(0.1, 80) & R.ds.between(0.5, 40) & R.dm.between(0.1, 180)].copy()
    if not len(s):
        return pd.Series(dtype=float)
    s["month"] = s.t.dt.to_period("M").dt.to_timestamp()
    r = s.groupby("month").agg(o=("do", "sum"), sc=("ds", "sum"), n=("do", "size"))
    r = r[r.n >= 3]; v = 100 * r.o / r.sc
    return v[v.between(20, 400)]


def _chg_rate_month(R, LO=30, HI=70):
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


def compute_proxies(raw, resample="2min", min_rows=20):
    """raw: DataFrame with columns [vin, t(datetime64), soc, odo].  Returns per-(vin,month) DataFrame
    [vin, month, km_soc, chg_rate].  soc is clamped to [0,100]; each vehicle is resampled to `resample`
    cadence (last value per bucket) before proxy extraction."""
    d = raw[["vin", "t", "soc", "odo"]].copy()
    d["soc"] = pd.to_numeric(d.soc, errors="coerce")
    d["odo"] = pd.to_numeric(d.odo, errors="coerce")
    d = d[d.soc.between(0, 100)].dropna(subset=["t"])
    out = []
    for v, g in d.groupby("vin", sort=False):
        g = g.sort_values("t")
        R = g.set_index("t").resample(resample).agg(odo=("odo", "last"), soc=("soc", "last")).dropna().reset_index()
        if len(R) < min_rows:
            continue
        km = _km_soc_month(R); cr = _chg_rate_month(R)
        m = pd.concat([km.rename("km_soc"), cr.rename("chg_rate")], axis=1)
        m.index.name = "month"; m = m.reset_index(); m["vin"] = str(v)
        out.append(m)
    if not out:
        return pd.DataFrame(columns=["vin", "month", "km_soc", "chg_rate"])
    return pd.concat(out, ignore_index=True)[["vin", "month", "km_soc", "chg_rate"]]


def age_controlled_signal(df, proxy, soh="soh", age="age_months", vin="vin"):
    """Does `proxy` carry SoH signal BEYOND the age confound?  Returns dict with the raw within-vehicle
    correlation vs SoH and the partial correlation controlling for age (the honest number)."""
    d = df.dropna(subset=[proxy, soh, age]).copy()
    cnt = d.groupby(vin).size(); d = d[d[vin].isin(cnt[cnt >= 4].index)]
    if d[vin].nunique() < 3 or len(d) < 12:
        return dict(proxy=proxy, vehicles=int(d[vin].nunique()), pairs=int(len(d)), r=None, partial_r=None)
    z = lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
    for col in [proxy, soh, age]:
        d[col + "_z"] = d.groupby(vin)[col].transform(z)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[proxy + "_z", soh + "_z", age + "_z"])
    r = float(np.corrcoef(d[proxy + "_z"], d[soh + "_z"])[0, 1])
    # partial corr(proxy, soh | age): correlate residuals after regressing out age
    def resid(y):
        a = d[age + "_z"].values
        b = np.polyfit(a, y, 1)
        return y - np.polyval(b, a)
    pr = float(np.corrcoef(resid(d[proxy + "_z"].values), resid(d[soh + "_z"].values))[0, 1])
    return dict(proxy=proxy, vehicles=int(d[vin].nunique()), pairs=int(len(d)), r=r, partial_r=pr)
