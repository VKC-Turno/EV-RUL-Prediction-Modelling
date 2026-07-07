#!/usr/bin/env python3
"""Euler SoH from BMS full-capacity — the real fix (coulomb was NOISIER; see euler_full_charge_soh.py negative test).

Euler's BMS reports batteryRemainingCapacity directly, and at near-full SoC that reading (≈5.7% raw CV) is
cleaner than anything we can coulomb-count. Production SoH already uses it, but pushes it through a heavy
isotonic-envelope + 100-clip that collapses it into flat / cliff / stuck-floor artifacts (30% flat, 24% clipped).

This exposes the SAME signal raw and monthly, so the dashboard can apply the light monotone-decreasing fit +
outlier-greying (the treatment we built for Mahindra) instead of the flattening envelope.

full_cap = batteryRemainingCapacity / (SoC/100) at SoC 95-100%, monthly median per vehicle. One fast pass over
the 231 dense files.  Output: data/euler/bms_soh.parquet (+ _summary.parquet, _report.json).
Run: .venv/bin/python src/euler_bms_soh.py
"""
import os, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import theilslopes

os.chdir(Path(__file__).resolve().parent.parent)
DENSE = "data/euler/dense/*.parquet"
FEAT = "data/euler/features/feature_table.parquet"
FULL_CHARGE = "data/euler/full_charge_soh.parquet"   # INDEPENDENT coulomb full-charge SoH (src/euler_full_charge_soh.py)
OUT, OUT_SUM, OUT_REP = "data/euler/bms_soh.parquet", "data/euler/bms_soh_summary.parquet", "data/euler/bms_soh_report.json"
DECL_PPY = 3.0                        # a vehicle is a "confirmed decliner" at >= this fade rate (pp/year)
SOC_LO, SOC_HI = 95.0, 100.0
RC_LO, RC_HI = 1.0, 500.0
MIN_MON_N = 20                        # a month needs >= this many near-full readings
BASE_LO, BASE_HI = 0.5, 12.0          # baseline cap0 window (months)


def _cv(s):
    s = pd.Series(s).dropna()
    return float(s.std() / s.mean() * 100.0) if (len(s) > 1 and s.mean()) else np.nan


def recovery_inliers(age_m, soh, margin=3.5, max_rate=10.0):
    """Temporal/physical outlier mask (uses MONOTONICITY): a reading is a transient BMS under-estimate — greyed —
    if a SUSTAINED later-higher level exists (median of readings after it exceeds it by `margin`; capacity can't
    recover) OR it implies a >max_rate pp/yr fade from a 100% BOL. A genuine sustained decline has no higher
    future, so it is KEPT. This catches the re-estimation dips/cliffs that threshold filters can't."""
    a = np.asarray(age_m, float); s = np.asarray(soh, float); n = len(s)
    ex = np.zeros(n, dtype=bool)
    for i in range(n - 2):
        if np.median(s[i + 1:]) > s[i] + margin:
            ex[i] = True
    ex = ex | ((100.0 - s) / np.maximum(a / 12.0, 0.25) > max_rate)
    return ~ex


def clean_label(age_m, soh):
    """The RETRAINING TARGET: recovery-aware cleaning + isotonic-decreasing smoothing -> a monotone, <=100,
    per-month SoH label (transient dips removed, real declines kept). NaN-age months (no registration anchor)
    are left as NaN (not trainable). Returns (is_inlier_mask, soh_label), both full length."""
    from sklearn.isotonic import IsotonicRegression
    a = np.asarray(age_m, float); s = np.clip(np.asarray(soh, float), None, 100.0)
    n = len(s); is_inlier = np.zeros(n, bool); soh_label = np.full(n, np.nan)
    fin = np.isfinite(a) & np.isfinite(s)
    if fin.sum() < 3:                                    # too few anchored months to smooth
        is_inlier[fin] = True; soh_label[fin] = s[fin]
        return is_inlier, soh_label
    idx = np.where(fin)[0]; af, sf = a[fin], s[fin]
    inl = recovery_inliers(af, sf)
    if inl.sum() < 3:
        inl = np.ones(len(sf), bool)
    iso = IsotonicRegression(increasing=False, y_max=100.0, out_of_bounds="clip").fit(af[inl], sf[inl])
    is_inlier[idx[inl]] = True
    soh_label[idx] = np.clip(iso.predict(af), None, 100.0)
    return is_inlier, soh_label


def _slope_ppy(age, y, minn=4, minspan=4.0):
    """Robust decline rate (pp/YEAR, positive = losing SoH) via Theil-Sen; NaN if coverage is too thin."""
    a = np.asarray(age, float); v = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(v); a, v = a[m], v[m]
    if len(a) < minn or (a.max() - a.min()) < minspan:
        return np.nan
    return float(-theilslopes(v, a)[0] * 12.0)


def hybrid_target(M, full_charge_path=FULL_CHARGE):
    """The DEPLOYABLE target: soh_label for flat/healthy vehicles (where it robustly beats production), but the
    incumbent production SoH (soh_prod) for CONFIRMED DECLINERS, where the clean label is provably too optimistic
    against the independent coulomb yardstick (see euler_soh_label_retrain-gated finding).

    A vehicle is a confirmed decliner if the PHYSICALLY INDEPENDENT coulomb full-charge SoH fades >= DECL_PPY pp/yr
    (arbiter of first resort), OR — where coulomb coverage is thin — both dense BMS signals (remaining-capacity SoH
    and BMS-native batterySoh) AGREE on >= DECL_PPY (agreement rules out either being an isolated artifact).

    Adds per-vehicle columns to M: soh_target (the target), confirmed_decliner; and returns a per-vin flag frame."""
    coul = {}
    if Path(full_charge_path).exists():
        C = pd.read_parquet(full_charge_path)[["vin", "age_months", "soh_full"]].copy()
        C["vin"] = C["vin"].astype(str)
        C["c"] = np.clip(pd.to_numeric(C["soh_full"], errors="coerce"), None, 100.0)
        for v, c in C.dropna(subset=["age_months", "c"]).groupby("vin"):
            coul[v] = _slope_ppy(c["age_months"], c["c"])
    flags = []
    for vin, g in M.groupby("vin"):
        g = g.sort_values("age_months")
        r_rem = _slope_ppy(g["age_months"], g["soh_full"])                       # remaining-capacity SoH slope
        r_nat = _slope_ppy(g["age_months"], pd.to_numeric(g["soh_reported"], errors="coerce"))  # batterySoh slope
        r_cou = coul.get(vin, np.nan)
        decliner = bool((np.isfinite(r_cou) and r_cou >= DECL_PPY) or
                        (np.isfinite(r_rem) and np.isfinite(r_nat) and r_rem >= DECL_PPY and r_nat >= DECL_PPY))
        flags.append(dict(vin=vin, coul_ppy=r_cou, remcap_ppy=r_rem, native_ppy=r_nat, confirmed_decliner=decliner))
    F = pd.DataFrame(flags)
    M = M.merge(F[["vin", "confirmed_decliner"]], on="vin", how="left")
    # target: production for decliners (fall back to label if production missing), clean label otherwise; <=100
    prod = pd.to_numeric(M["soh_prod"], errors="coerce")
    tgt = np.where(M["confirmed_decliner"].fillna(False), prod.where(prod.notna(), M["soh_label"]), M["soh_label"])
    M["soh_target"] = np.clip(pd.to_numeric(pd.Series(tgt, index=M.index), errors="coerce"), None, 100.0)
    M.loc[M["soh_label"].isna(), "soh_target"] = np.nan                          # only trainable where label exists
    return M, F


def main():
    fs = glob.glob(DENSE)
    d = pd.concat([pd.read_parquet(f, columns=["vin", "eventAt", "batterySoc", "batteryRemainingCapacity", "batterySoh"])
                   for f in fs], ignore_index=True)
    d["vin"] = d["vin"].astype(str)
    for c in ("batterySoc", "batteryRemainingCapacity", "batterySoh"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(pd.to_numeric(d["eventAt"], errors="coerce"), unit="ms")
    d["month"] = d["t"].dt.to_period("M").dt.to_timestamp()
    # BMS-REPORTED SoH (batterySoh): monthly median of valid readings over ALL SoC (garbage sentinels >110 gated)
    rep = (d[d["batterySoh"].between(40, 110)].groupby(["vin", "month"])["batterySoh"]
           .median().clip(upper=100.0).rename("soh_reported").reset_index())   # SoH <= 100 (physical bound)
    # BMS full-capacity from near-full readings
    nf = d[(d["batterySoc"].between(SOC_LO, SOC_HI)) & (d["batteryRemainingCapacity"].between(RC_LO, RC_HI))].copy()
    nf["full_cap"] = nf["batteryRemainingCapacity"] / (nf["batterySoc"] / 100.0)
    mon = nf.groupby(["vin", "month"]).agg(full_cap=("full_cap", "median"), n=("full_cap", "size"),
                                           raw_cv=("full_cap", _cv)).reset_index()
    mon = mon[mon["n"] >= MIN_MON_N].merge(rep, on=["vin", "month"], how="left")

    fe = pd.read_parquet(FEAT); fe["vin"] = fe["vin"].astype(str); fe["month"] = pd.to_datetime(fe["month"])
    reg = {}
    for vin, g in fe.groupby("vin"):
        g = g.sort_values("month")
        reg[vin] = g["month"].iloc[0] - pd.DateOffset(months=int(round(float(g["age_months"].iloc[0]))))
    mon["age_months"] = [((m - reg[v]).days / 30.4 if v in reg else np.nan) for v, m in zip(mon["vin"], mon["month"])]
    mon = mon.merge(fe[["vin", "month", "soh"]].rename(columns={"soh": "soh_prod"}), on=["vin", "month"], how="left")

    parts, summ = [], []
    for vin, g in mon.groupby("vin"):
        g = g.sort_values("month")
        base = g[g["age_months"].between(BASE_LO, BASE_HI)]["full_cap"]
        cap0 = float(base.median()) if len(base) >= 2 else float(g["full_cap"].head(3).median())
        if not (np.isfinite(cap0) and cap0 > 0):
            continue
        g = g.assign(cap0=cap0, soh_full=np.clip(100.0 * g["full_cap"] / cap0, None, 100.0))   # SoH <= 100 (physical bound)
        _inl, _label = clean_label(g["age_months"].values, g["soh_full"].values)               # THE cleaned target
        g = g.assign(is_inlier=_inl, soh_label=_label)
        parts.append(g)
        sp = pd.to_numeric(fe[fe["vin"] == vin].sort_values("month")["soh"], errors="coerce").dropna()
        _ages = g["age_months"].values; _fin = np.isfinite(_label)
        _lf, _af = _label[_fin], _ages[_fin]                                                    # finite (trainable) label pts
        _drop = float(_lf[0] - _lf[-1]) if len(_lf) > 1 else np.nan
        _span = max((_af[-1] - _af[0]) / 12.0, 0.1) if len(_af) > 1 else np.nan
        summ.append(dict(vin=vin, n_months=int(len(g)), n_label=int(_fin.sum()), n_inlier=int(_inl.sum()),
                         cap0=cap0, cv_full=_cv(g["full_cap"]), raw_cv_median=float(g["raw_cv"].median()),
                         label_drop=_drop, label_rate=(_drop / _span if len(_lf) > 1 else np.nan),
                         prod_drop=float(sp.iloc[0] - sp.iloc[-1]) if len(sp) > 1 else np.nan))
    M = pd.concat(parts, ignore_index=True); S = pd.DataFrame(summ)
    M, F = hybrid_target(M)                               # soh_target (deployable) + confirmed_decliner flag
    S = S.merge(F, on="vin", how="left")
    M.to_parquet(OUT, index=False); S.to_parquet(OUT_SUM, index=False)
    n_decl = int(S["confirmed_decliner"].sum())
    rep = dict(oem="euler", vehicles=int(S["vin"].nunique()), vehicles_ge4_months=int((S["n_months"] >= 4).sum()),
               median_months=float(S["n_months"].median()),
               cv_monthly_median=round(float(S["cv_full"].median()), 1),
               cv_raw_reading_median=round(float(S["raw_cv_median"].median()), 1),
               confirmed_decliners=n_decl, hybrid_target="soh_target = production for decliners, soh_label otherwise")
    json.dump(rep, open(OUT_REP, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {OUT}, {OUT_SUM}, {OUT_REP}")


if __name__ == "__main__":
    main()
