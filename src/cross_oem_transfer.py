#!/usr/bin/env python3
"""Cross-OEM degradation transfer: can a model trained on an AGED oem forecast a YOUNG oem?

We transfer the degradation RATE (monthly SoH loss), not the absolute SoH level — Euler/Mahindra are
renormalized to 100 at registration while Bajaj uses absolute BMS-reported SoH, so only the rate is
comparable across OEMs. Predictors are the feature-CONCEPTS shared by all three feeds (no current/
voltage, since Bajaj lacks them). We roll the rate model forward over each target vehicle's observed
months and score the reconstructed SoH trajectory (MAE pp) against that vehicle's actual SoH.

For each (source -> target) we report transfer MAE next to two yardsticks on the SAME target vehicles:
  - WITHIN (LOVO): model trained on the target OEM itself, leave-one-vehicle-out = home-field ceiling.
  - PERSIST: assume SoH stays flat = the floor a model must beat to be worth anything.

Run: .venv/bin/python src/cross_oem_transfer.py
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent.parent)

OEMS = ["Euler", "Mahindra", "Bajaj"]
FT = {o: f"data/{o.lower()}/features/feature_table.parquet" for o in OEMS}
# feature CONCEPTS present in all three feeds (verified by inventory). No current/voltage (Bajaj lacks).
SHARED = ["age_months", "soc_mean", "frac_soc_high", "frac_soc_low",
          "temp_mean", "temp_max", "km_month", "odo_max", "cum_km"]
CURV = ["inv_sqrt_age", "soh_deficit"]
FEATS = SHARED + CURV


def curv(age, soh):
    return 1.0 / np.sqrt(np.maximum(age, 0) + 1.0), 100.0 - soh


def load(o):
    m = pd.read_parquet(FT[o])[["vin", "month", "soh"] + SHARED].copy()
    m["month"] = pd.to_datetime(m["month"])
    return m.sort_values(["vin", "month"]).reset_index(drop=True)


def transitions(m, max_gap=3.0):
    parts = []
    for vin, g in m.groupby("vin"):
        g = g.sort_values("month").reset_index(drop=True)
        gap = g["month"].diff().shift(-1).dt.days / 30.4
        loss = ((g["soh"] - g["soh"].shift(-1)) / gap).clip(lower=0)
        r = g[SHARED].copy()
        r["inv_sqrt_age"], r["soh_deficit"] = curv(g["age_months"].to_numpy(), g["soh"].to_numpy())
        r["vin"] = vin; r["loss"] = loss.values; r["gap"] = gap.values
        parts.append(r)
    t = pd.concat(parts, ignore_index=True)
    t = t[(t["gap"] <= max_gap) & t["loss"].notna()].copy()
    t["w"] = 1.0 + t["loss"].clip(0, 5)                 # up-weight real-decline months
    return t


def fit(t):
    return XGBRegressor(n_estimators=350, learning_rate=0.03, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, n_jobs=8, verbosity=0).fit(
        t[FEATS].to_numpy(), t["loss"].to_numpy(), sample_weight=t["w"].to_numpy())


def free_run(g, model):
    """Roll predicted SoH over g's observed months using actual per-month shared stress."""
    g = g.sort_values("month").reset_index(drop=True)
    gapv = (g["month"].diff().dt.days / 30.4).to_numpy()
    pred = [float(g["soh"].iloc[0])]
    for i in range(1, len(g)):
        row = g.iloc[i - 1]
        isa, dfc = curv(float(row["age_months"]), pred[-1])
        x = np.array([[*[float(row[s]) for s in SHARED], isa, dfc]])
        step = max(float(model.predict(x)[0]), 0.0) * (gapv[i] if gapv[i] > 0 else 1.0)
        pred.append(pred[-1] - step)
    return np.array(pred)


def traj_mae(m, model):
    """Mean per-vehicle trajectory MAE (pp) over a target OEM using a given rate model."""
    es = []
    for vin, g in m.groupby("vin"):
        if len(g) < 3:
            continue
        g = g.sort_values("month")
        es.append(np.mean(np.abs(free_run(g, model) - g["soh"].to_numpy())))
    return float(np.mean(es)), len(es)


def persist_mae(m):
    es = []
    for vin, g in m.groupby("vin"):
        if len(g) < 3:
            continue
        s = g.sort_values("month")["soh"].to_numpy()
        es.append(np.mean(np.abs(s - s[0])))
    return float(np.mean(es))


def within_lovo(m, tr):
    es = []
    for vin, g in m.groupby("vin"):
        if len(g) < 3:
            continue
        model = fit(tr[tr["vin"] != vin])
        es.append(np.mean(np.abs(free_run(g.sort_values("month"), model) - g.sort_values("month")["soh"].to_numpy())))
    return float(np.mean(es))


print("loading + building transitions ...")
M = {o: load(o) for o in OEMS}
T = {o: transitions(M[o]) for o in OEMS}
full = {o: fit(T[o]) for o in OEMS}                     # model trained on ALL of each OEM
for o in OEMS:
    print(f"  {o}: {M[o]['vin'].nunique()} veh, {len(T[o])} transitions")

print("\n=== TRANSFER: source-trained rate model -> target trajectory MAE (pp, lower=better) ===")
print(f"{'target':9} {'PERSIST':>8} {'WITHIN':>8} | " + " ".join(f"{('from '+s):>11}" for s in OEMS))
rows = {}
for tgt in OEMS:
    persist = persist_mae(M[tgt])
    within = within_lovo(M[tgt], T[tgt])
    cells = []
    for src in OEMS:
        if src == tgt:
            cells.append("    —(self)")          # self handled by WITHIN
        else:
            mae, n = traj_mae(M[tgt], full[src])
            cells.append(f"{mae:11.2f}")
    rows[tgt] = (persist, within)
    print(f"{tgt:9} {persist:8.2f} {within:8.2f} | " + " ".join(cells))

# combined aged-source (Euler+Mahindra, the electrically-rich/aged pair) -> Bajaj (young target)
combo = fit(pd.concat([T["Euler"], T["Mahindra"]], ignore_index=True))
cmae, _ = traj_mae(M["Bajaj"], combo)
print(f"\nEuler+Mahindra (combined aged source) -> Bajaj: {cmae:.2f} pp  "
      f"(vs Bajaj WITHIN {rows['Bajaj'][1]:.2f}, PERSIST {rows['Bajaj'][0]:.2f})")

# degrader-only view for the headline aged->young routes (transfer matters most on real decline)
print("\n=== degraders-only (target vehicles with >=3pp total drop) ===")
def deg_subset(m):
    keep = [v for v, g in m.groupby("vin") if (g.sort_values("month")["soh"].iloc[0]
            - g.sort_values("month")["soh"].iloc[-1]) >= 3.0 and len(g) >= 3]
    return m[m["vin"].isin(keep)], len(keep)
for tgt in OEMS:
    md, nd = deg_subset(M[tgt])
    pmae = persist_mae(md); wmae = within_lovo(md, transitions(md))
    e = traj_mae(md, full["Euler"])[0] if tgt != "Euler" else float("nan")
    mh = traj_mae(md, full["Mahindra"])[0] if tgt != "Mahindra" else float("nan")
    print(f"{tgt:9} (n={nd:3d} degraders)  PERSIST {pmae:5.2f}  WITHIN {wmae:5.2f}  "
          f"fromEuler {e:5.2f}  fromMahindra {mh:5.2f}")
