#!/usr/bin/env python3
"""Probable SoH curves for Mahindra-NATIVE by behaviour cohort — the Bayesian version (v2, post-verification).

Hypothesis (not "km/%SoC is a SoH proxy", which failed): vehicles that CHARGE & DRIVE alike DEGRADE alike. Model the
degradation TRAJECTORY on feeds that have a real SoH (Euler / Bajaj / Piaggio / Mahindra-intellicar), let a
native-computable behaviour fingerprint tilt the degradation rate, then posterior-predict native curves.

Model = src/bayes_degradation.py (hierarchical Bayesian, Gibbs):
    SoH_ij ~ N(a_i + b_i*age_ij, s^2);  a_i ~ N(mu_a, .);  b_i ~ N(x_i.beta, sigma_b[source]^2)
X = [per-source baseline dummies] + [GLOBALLY-standardised behaviour]. Source dummies absorb OEM/chemistry rate
differences; the SHARED behaviour slopes transfer to native. sigma_b is per-SOURCE so each OEM's band is calibrated.
Native rides the MAHINDRA baseline + Mahindra's own heterogeneity.

Fixes applied after adversarial verification (workflow wbrarbxyx):
  * km_month recomputed IDENTICALLY everywhere from odometer ENDPOINT-SPAN (last-first)/age-span with sane guards —
    the feature-table km_month summed odometer resets (Euler up to 73M km/mo); good-source vs native used different
    definitions; per-source z-scoring made the slope scale-arbitrary. Now one physical definition + GLOBAL scaling.
  * dropped frac_soc_high (good sources threshold soc>90, native used soc>80 — incomparable; also not credible).
  * per-source sigma_b (single global sigma_b under/over-covered non-Mahindra OEMs).
  * report the PER-SOURCE km->rate slope (it is credible in Bajaj -0.84 and Mahindra-intellicar -0.28 but ~0 in Euler).

-> data/mahindra/native_behaviour_soh.parquet, data/mahindra/behaviour_soh_report.json
Run: .venv/bin/python src/behaviour_soh_experiment.py
"""
import os, glob, json
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
import sys; sys.path.insert(0, "src")
from bayes_degradation import fit_gibbs, predict_curve, predict_rate

BEH = ["km_month", "soc_mean", "frac_soc_low"]              # native-computable, consistently-defined, cross-OEM comparable
SRC = {"euler": "data/euler/features/feature_table.parquet",
       "bajaj": "data/bajaj/features/feature_table.parquet",
       "piaggio": "data/piaggio/features/feature_table.parquet",
       "mahindra_ic": "data/redshift/mahindra_featengg.parquet"}
MIN_MONTHS, MIN_SPAN = 5, 6.0
KM_LO, KM_HI = 10.0, 8000.0                                 # plausible km/month for a commercial 3W; guards odo glitches


def robust_km_month(g):
    """km/month from odometer ENDPOINT span over the observed age window (robust to reset-diff corruption)."""
    o = g.sort_values("month")["odo_max"].values; span = g.age_months.max() - g.age_months.min()
    if span <= 0:
        return np.nan
    km = (o[-1] - o[0]) / span
    if not (KM_LO <= km <= KM_HI):
        km = (np.nanmax(o) - np.nanmin(o)) / span            # fall back to max-min if endpoints reset
    return km if KM_LO <= km <= KM_HI else np.nan


def load_source(src, p):
    d = pd.read_parquet(p); d["vin"] = d["vin"].astype(str)
    mc = "month" if "month" in d.columns else "ymd"
    d[mc] = pd.to_datetime(d[mc].astype(str)); d = d.rename(columns={mc: "month"})
    d = d[["vin", "month", "soh", "age_months", "odo_max", "soc_mean", "frac_soc_low"]].dropna(subset=["soh", "age_months"])
    obs, beh = [], []
    for v, g in d.groupby("vin"):
        g = g.sort_values("month")
        if len(g) < MIN_MONTHS or g.age_months.max() - g.age_months.min() < MIN_SPAN:
            continue
        km = robust_km_month(g)
        if not np.isfinite(km):
            continue
        for a, s in zip(g.age_months, g.soh):
            obs.append((src, v, float(a), float(s)))
        beh.append(dict(src=src, vin=v, km_month=km, soc_mean=float(g.soc_mean.mean()), frac_soc_low=float(g.frac_soc_low.mean())))
    return pd.DataFrame(obs, columns=["src", "vin", "age", "soh"]), pd.DataFrame(beh)


# 1. assemble good sources
OBS, BEHV = [], []
for s, p in SRC.items():
    if not os.path.exists(p):
        print(f"[skip] {s}"); continue
    o, b = load_source(s, p); OBS.append(o); BEHV.append(b)
    print(f"[{s}] {b.vin.nunique()} vehicles, {len(o)} obs, median km/mo {b.km_month.median():.0f}")
obs = pd.concat(OBS, ignore_index=True); behv = pd.concat(BEHV, ignore_index=True)

# 2. GLOBAL standardisation (one physical beta; native uses the SAME scaler)
SCALER = {f: (behv[f].mean(), behv[f].std()) for f in BEH}
for f in BEH:
    mu, sd = SCALER[f]; behv[f + "_z"] = (behv[f] - mu) / sd

# 3. design: per-source baseline dummies + shared behaviour slopes; group = source (for per-source sigma_b)
srcs = sorted(behv.src.unique())
key = behv.set_index(["src", "vin"]); vlist = list(key.index); vidx = {vk: i for i, vk in enumerate(vlist)}
Dsrc = np.array([[1.0 if vk[0] == s else 0.0 for s in srcs] for vk in vlist])
Zbeh = key[[f + "_z" for f in BEH]].values
X = np.hstack([Dsrc, Zbeh]); Pb = len(srcs); P = X.shape[1]
group = np.array([srcs.index(vk[0]) for vk in vlist])
obs["vi"] = list(zip(obs.src, obs.vin)); obs = obs[obs.vi.isin(vidx)]
vin_idx = obs.vi.map(vidx).values; soh = obs.soh.values; age = obs.age.values
N = X.shape[0]
print(f"\nfitting: {N} vehicles, {len(soh)} obs, {P} params ({Pb} source baselines + {len(BEH)} behaviour), per-source sigma_b")

draws = fit_gibbs(soh, age, vin_idx, X, group=group, n_iter=6000, burn=2000, thin=2, seed=0, beta_prior_sd=2.0)
bm = draws["beta"].mean(0); blo, bhi = np.percentile(draws["beta"], [2.5, 97.5], axis=0)
print("\nsource baseline rates (SoH/mo):")
for i, s in enumerate(srcs):
    print(f"  {s:12s} {bm[i]:+.3f}  [{blo[i]:+.3f},{bhi[i]:+.3f}]  sigma_b {np.sqrt(draws['sigma_b2'][:, i].mean()):.3f}")
print("behaviour slopes (per +1 global SD, SoH/mo):")
beh_sig = {}
for j, f in enumerate(BEH):
    i = Pb + j; sig = (blo[i] > 0) or (bhi[i] < 0)
    abs_sens = bm[i] / SCALER[f][1] * (1000 if f == "km_month" else 1)          # per 1000 km/mo for km
    beh_sig[f] = dict(mean=float(bm[i]), lo=float(blo[i]), hi=float(bhi[i]), credible=bool(sig),
                      abs_sensitivity=float(abs_sens))
    unit = "per +1000 km/mo" if f == "km_month" else "per unit"
    print(f"  {f:14s} {bm[i]:+.3f}  [{blo[i]:+.3f},{bhi[i]:+.3f}]  ({abs_sens:+.4f} {unit}) {'<-- credible' if sig else ''}")

# 3b. PER-SOURCE km_month -> rate (the honest transparency the verifier asked for; signs differ across OEMs)
per_src_km = {}
for s in srcs:
    bb = behv[behv.src == s]
    rr = []
    for r in bb.itertuples():
        g = obs[obs.vi == (s, r.vin)]
        if len(g) >= MIN_MONTHS:
            rr.append((r.km_month, np.polyfit(g.age.values, g.soh.values, 1)[0]))
    if len(rr) >= 8:
        km_arr, rate_arr = np.array(rr).T
        per_src_km[s] = dict(rho=float(np.corrcoef(km_arr, rate_arr)[0, 1]), n=len(rr))
print("per-source km_month->rate corr:", {s: round(v["rho"], 2) for s, v in per_src_km.items()})

# 4. honesty: 5-fold vehicle-held-out RATE prediction — behaviour vs source-baseline
rng = np.random.default_rng(0); orderv = rng.permutation(N); folds = np.array_split(orderv, 5)
true_rate = np.full(N, np.nan)
for vk, i in vidx.items():
    g = obs[obs.vi == vk]
    if len(g) >= MIN_MONTHS:
        true_rate[i] = np.polyfit(g.age.values, g.soh.values, 1)[0]
mae_beh, mae_base = [], []
for fold in folds:
    te = set(fold.tolist()); tr = sorted(i for i in range(N) if i not in te)
    tr_mask = np.isin(vin_idx, tr); remap = {old: k for k, old in enumerate(tr)}
    dr = fit_gibbs(soh[tr_mask], age[tr_mask], np.array([remap[i] for i in vin_idx[tr_mask]]),
                   X[tr], group=group[tr], n_iter=2500, burn=800, thin=2, seed=1, beta_prior_sd=2.0)
    bmean = dr["beta"].mean(0); base_rate = {s: bmean[k] for k, s in enumerate(srcs)}
    for i in fold:
        if np.isnan(true_rate[i]):
            continue
        mae_beh.append(abs(float(X[i] @ bmean) - true_rate[i]))
        mae_base.append(abs(base_rate[srcs[int(np.argmax(X[i, :Pb]))]] - true_rate[i]))
mae_beh, mae_base = float(np.mean(mae_beh)), float(np.mean(mae_base))
improve = 100 * (mae_base - mae_beh) / mae_base
print(f"\nheld-out RATE MAE (SoH/mo): behaviour {mae_beh:.4f} vs source-baseline {mae_base:.4f} -> {improve:+.1f}%")

# 4b. band decomposition for native (Mahindra group)
mah_i = srcs.index("mahindra_ic")
xavg = np.zeros(P); xavg[mah_i] = 1.0
param_sd = float((draws["beta"] @ xavg).std())
het_sd = float(np.sqrt(draws["sigma_b2"][:, mah_i].mean()))
print(f"band decomposition (Mahindra rate SD): parameter {param_sd:.4f} vs heterogeneity {het_sd:.4f}")

# 5. NATIVE behaviour — SAME definitions + SAME global scaler; age from reg dates
fs = sorted(glob.glob("data/mahindra/native100/*.parquet"))
nat = pd.concat([pd.read_parquet(f, columns=["vin", "eventAt", "soc", "odometer"]) for f in fs], ignore_index=True)
nat["vin"] = nat.vin.astype(str)
nat["t"] = pd.to_datetime(pd.to_numeric(nat.eventAt, errors="coerce"), unit="ms")
nat["soc"] = pd.to_numeric(nat.soc, errors="coerce"); nat["odo"] = pd.to_numeric(nat.odometer, errors="coerce")
nat = nat[nat.soc.between(0, 100)].dropna(subset=["t"])
reg = pd.read_csv("Mh_Regd_Date.csv"); rvin = next(c for c in reg.columns if c.lower() == "vin")
reg["rd"] = pd.to_datetime(reg["vehicle_registration_date"], errors="coerce"); REG = dict(zip(reg[rvin].astype(str), reg["rd"]))
nrows = []
for v, g in nat.groupby("vin"):
    span_mo = (g.t.max() - g.t.min()).days / 30.4
    if span_mo < 1:
        continue
    km = (g.odo.max() - g.odo.min()) / span_mo               # SAME endpoint-span definition as good sources
    if not (KM_LO <= km <= KM_HI):
        continue
    nrows.append(dict(vin=v, km_month=km, soc_mean=g.soc.mean(), frac_soc_low=(g.soc < 20).mean(),
                      last_age=((g.t.max() - REG[v]).days / 30.4) if REG.get(v) is not None and pd.notna(REG.get(v)) else np.nan))
natb = pd.DataFrame(nrows)
for f in BEH:                                                # SAME global scaler as training
    mu, sd = SCALER[f]; natb[f + "_z"] = (natb[f] - mu) / sd
print(f"\nnative: {len(natb)} vehicles | median km/mo {natb.km_month.median():.0f} (good-source pooled median {behv.km_month.median():.0f})")

# 6. posterior-predict each native curve (Mahindra group heterogeneity + small intercept spread)
age_grid = np.arange(0, 49, 3.0); out = []
for r in natb.itertuples():
    xj = np.zeros(P); xj[mah_i] = 1.0
    for j, f in enumerate(BEH):
        xj[Pb + j] = getattr(r, f + "_z")
    cur = predict_curve(draws, xj, age_grid, group=mah_i, anchor_intercept=100.0, intercept_sd=1.0)
    rate = predict_rate(draws, xj, group=mah_i)
    for k, ag in enumerate(age_grid):
        out.append(dict(vin=r.vin, age_months=float(ag), soh_p10=cur["q10"][k], soh_p50=cur["q50"][k],
                        soh_p90=cur["q90"][k], pred_rate=rate["mean"], km_month=r.km_month, last_age=r.last_age))
ND = pd.DataFrame(out); ND.to_parquet("data/mahindra/native_behaviour_soh.parquet", index=False)

b36 = ND[ND.age_months == 36.0]
report = dict(
    n_good_vehicles=int(N), n_obs=int(len(soh)), sources=srcs,
    source_baseline_rate={s: float(bm[i]) for i, s in enumerate(srcs)},
    source_sigma_b={s: float(np.sqrt(draws["sigma_b2"][:, i].mean())) for i, s in enumerate(srcs)},
    behaviour_slopes=beh_sig, per_source_km_rate=per_src_km,
    heldout_rate_mae=dict(behaviour=mae_beh, source_baseline=mae_base, behaviour_improvement_pct=improve),
    band_decomposition=dict(parameter_sd=param_sd, heterogeneity_sd=het_sd),
    native=dict(vehicles=int(natb.vin.nunique()), km_month_median=float(natb.km_month.median()),
                pred_rate_median=float(ND.groupby("vin").pred_rate.first().median()),
                band_width_at_36mo_median=float((b36.soh_p90 - b36.soh_p10).median()),
                soh50_at_36mo_median=float(b36.soh_p50.median())))
json.dump(report, open("data/mahindra/behaviour_soh_report.json", "w"), indent=2, default=float)
print("\n=== native probable-SoH summary ==="); print(json.dumps(report["native"], indent=2))
print(f"wrote native_behaviour_soh.parquet ({ND.vin.nunique()} vehicles) + behaviour_soh_report.json")
