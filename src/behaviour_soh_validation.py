#!/usr/bin/env python3
"""Proof-of-validation for the Mahindra native SoH model, on the vehicles where we HAVE ground truth.

The native fleet has no measured SoH — but the Mahindra-INTELLICAR cohort has both a behaviour fingerprint AND a
real coulomb SoH curve. So we can test the model's core claim directly: does a vehicle's driving/charging behaviour
predict its actual SoH trajectory? Two pieces of evidence:

  (a) PER-VEHICLE — for a few contrasting vehicles (light vs heavy users), predict the SoH curve from behaviour with
      the vehicle HELD OUT of training, and overlay its actual measured coulomb SoH. Report band coverage.
  (b) SYSTEMATIC — across all well-observed Mahindra-IC vehicles, correlate km/month (behaviour) with the actual SoH
      decline rate (no model in the loop — pure behaviour-feature vs outcome).

-> data/mahindra/behaviour_validation_curves.parquet  (demo predicted bands on an age grid)
-> data/mahindra/behaviour_validation_actual.parquet  (demo actual coulomb SoH points)
-> data/mahindra/behaviour_validation_scatter.parquet (km_month vs actual rate, all vehicles + demo flag)
-> data/mahindra/behaviour_validation_report.json
Run: .venv/bin/python src/behaviour_soh_validation.py   (after src/behaviour_soh_experiment.py)
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
import sys; sys.path.insert(0, "src")
from bayes_degradation import fit_gibbs, predict_curve

BEH = ["km_month", "soc_mean", "frac_soc_low"]
SRC = {"euler": "data/euler/features/feature_table.parquet", "bajaj": "data/bajaj/features/feature_table.parquet",
       "piaggio": "data/piaggio/features/feature_table.parquet", "mahindra_ic": "data/redshift/mahindra_featengg.parquet"}
KM_LO, KM_HI = 10.0, 8000.0
# contrasting, well-observed Mahindra-IC vehicles: one light-flat, two heavier-declining (roughly monotone in km)
DEMOS = [("MA1FW22DDUP3B16058", "light use"), ("MA1FL2DDUN3L22121", "moderate use"), ("MA1FL2DDUP3A15262", "heavy use")]


def rkm(g):
    o = g.sort_values("month")["odo_max"].values; sp = g.age_months.max() - g.age_months.min()
    if sp <= 0:
        return np.nan
    k = (o[-1] - o[0]) / sp
    if not (KM_LO <= k <= KM_HI):
        k = (np.nanmax(o) - np.nanmin(o)) / sp
    return k if KM_LO <= k <= KM_HI else np.nan


def load(src, p):
    d = pd.read_parquet(p); d["vin"] = d.vin.astype(str); mc = "month" if "month" in d.columns else "ymd"
    d[mc] = pd.to_datetime(d[mc].astype(str)); d = d.rename(columns={mc: "month"})
    d = d[["vin", "month", "soh", "age_months", "odo_max", "soc_mean", "frac_soc_low"]].dropna(subset=["soh", "age_months"])
    obs, beh = [], []
    for v, g in d.groupby("vin"):
        g = g.sort_values("month")
        if len(g) < 5 or g.age_months.max() - g.age_months.min() < 6:
            continue
        k = rkm(g)
        if not np.isfinite(k):
            continue
        for a, s in zip(g.age_months, g.soh):
            obs.append((src, v, float(a), float(s)))
        beh.append(dict(src=src, vin=v, km_month=k, soc_mean=float(g.soc_mean.mean()), frac_soc_low=float(g.frac_soc_low.mean())))
    return pd.DataFrame(obs, columns=["src", "vin", "age", "soh"]), pd.DataFrame(beh)


OBS, BEHV = [], []
for s, p in SRC.items():
    o, b = load(s, p); OBS.append(o); BEHV.append(b)
obs = pd.concat(OBS, ignore_index=True); behv = pd.concat(BEHV, ignore_index=True)
SCALER = {f: (behv[f].mean(), behv[f].std()) for f in BEH}
for f in BEH:
    behv[f + "_z"] = (behv[f] - SCALER[f][0]) / SCALER[f][1]

demo_vins = [v for v, _ in DEMOS]
fitv = behv[~behv.vin.isin(demo_vins)].copy()                       # genuine hold-out of the demo vehicles
srcs = sorted(fitv.src.unique()); mah = srcs.index("mahindra_ic")
key = fitv.set_index(["src", "vin"]); vlist = list(key.index); vidx = {vk: i for i, vk in enumerate(vlist)}
Ds = np.array([[1.0 if vk[0] == s else 0 for s in srcs] for vk in vlist]); Z = key[[f + "_z" for f in BEH]].values
X = np.hstack([Ds, Z]); Pb = len(srcs); group = np.array([srcs.index(vk[0]) for vk in vlist])
o2 = obs[~obs.vin.isin(demo_vins)].copy(); o2["vi"] = list(zip(o2.src, o2.vin)); o2 = o2[o2.vi.isin(vidx)]
draws = fit_gibbs(o2.soh.values, o2.age.values, o2.vi.map(vidx).values, X, group=group, n_iter=5000, burn=1500, seed=0, beta_prior_sd=2.0)

# (a) per-vehicle predicted band vs actual
cur_rows, act_rows, cov = [], [], {}
for v, lab in DEMOS:
    br = behv[behv.vin == v].iloc[0]; av = obs[obs.vin == v].sort_values("age")
    xj = np.zeros(X.shape[1]); xj[mah] = 1.0
    for j, f in enumerate(BEH):
        xj[Pb + j] = br[f + "_z"]
    grid = np.arange(0, av.age.max() + 3, 1.5)
    g = predict_curve(draws, xj, grid, group=mah, anchor_intercept=100.0, intercept_sd=1.0)
    for k, ag in enumerate(grid):
        cur_rows.append(dict(vin=v, label=lab, km_month=round(float(br.km_month)), age=float(ag),
                             p10=g["q10"][k], p50=g["q50"][k], p90=g["q90"][k]))
    pa = predict_curve(draws, xj, av.age.values, group=mah, anchor_intercept=100.0, intercept_sd=1.0)
    inside = (av.soh.values >= pa["q10"]) & (av.soh.values <= pa["q90"])
    cov[v] = float(inside.mean())
    for ag, s in zip(av.age.values, av.soh.values):
        act_rows.append(dict(vin=v, label=lab, age=float(ag), actual_soh=float(s)))
pd.DataFrame(cur_rows).to_parquet("data/mahindra/behaviour_validation_curves.parquet", index=False)
pd.DataFrame(act_rows).to_parquet("data/mahindra/behaviour_validation_actual.parquet", index=False)

# (b) systematic: km_month vs actual decline rate, all well-observed Mahindra-IC vehicles (no model)
c = pd.read_parquet(SRC["mahindra_ic"]); c["vin"] = c["vin"].astype(str); c["month"] = pd.to_datetime(c["ymd"].astype(str))
sc = []
for v, g in c.groupby("vin"):
    g = g.sort_values("month")
    if len(g) < 8 or g.age_months.max() - g.age_months.min() < 9:
        continue
    k = rkm(g)
    if not np.isfinite(k):
        continue
    rate = float(np.polyfit(g.age_months.values, g.soh.values, 1)[0])
    sc.append(dict(vin=v, km_month=round(k), actual_rate=rate, is_demo=v in demo_vins))
SC = pd.DataFrame(sc); SC.to_parquet("data/mahindra/behaviour_validation_scatter.parquet", index=False)
rho = float(SC.km_month.corr(SC.actual_rate))

report = dict(demos=[dict(vin=v, label=lab, km_month=round(float(behv[behv.vin == v].km_month.iloc[0])),
                          actual_first=round(float(obs[obs.vin == v].sort_values("age").soh.iloc[0]), 1),
                          actual_last=round(float(obs[obs.vin == v].sort_values("age").soh.iloc[-1]), 1),
                          band_coverage=round(cov[v], 2)) for v, lab in DEMOS],
              systematic=dict(vehicles=int(len(SC)), km_vs_rate_corr=round(rho, 3),
                              light_median_rate=round(float(SC[SC.km_month < SC.km_month.quantile(.3)].actual_rate.median()), 3),
                              heavy_median_rate=round(float(SC[SC.km_month > SC.km_month.quantile(.7)].actual_rate.median()), 3)))
json.dump(report, open("data/mahindra/behaviour_validation_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
print("wrote behaviour_validation_{curves,actual,scatter}.parquet + _report.json")
