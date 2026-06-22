"""Customer-Centric Insights + Fleet Behavior Analysis for the SoH dashboard.

Two public functions, both accepting a preloaded feature-table DataFrame (so the Streamlit app can
pass its cached frame) and both returning JSON-serializable dicts:

  customer_insights(feature_table_df, vin) -> dict
      One vehicle, operator-facing. Usage intensity, charging behaviour, depth-of-discharge habits,
      energy efficiency and thermal exposure — each compared to the fleet (percentile + z) — plus a
      0-100 "battery-care score" (higher = gentler usage), human-readable insight strings, and the
      2-3 biggest behavioural levers that would extend this vehicle's life.

  fleet_behavior(feature_table_df) -> dict
      Whole fleet. Distribution summaries of the key behaviour metrics; behaviour SEGMENTS (KMeans
      k=3 on standardized behaviour features, each labelled in plain language with size + mean SoH-
      loss rate); the ranked correlation between each behaviour metric and the observed monthly SoH-
      loss rate (which behaviours actually predict faster fade in THIS data); and actionable findings.

This COMPLEMENTS build_dashboard.py's per-vehicle risk-cause attribution (which z-scores recent
stress into charging/driving/thermal/deep-discharge): here we add fleet context, percentile ranking,
a care score, behavioural levers, and fleet-wide segmentation + degradation correlation.

Run standalone:  .venv/bin/python dashboard/insights.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── nominal pack capacity (Ah) for C-rate proxies. Fleet early-life capacity_ah ~136 Ah median;
#    Treo/Zor Grand LFP packs are ~ this size. Used only to turn currents into C-rates. ──────────
NOMINAL_AH = 136.0
DAYS_PER_MONTH = 30.4375

# ── behaviour metrics: column in the feature table + how to read/clean it. Sparse/garbage-prone
#    columns (km, wh_per_km, temp_max) get a physical valid-range so contaminated rows are dropped
#    rather than averaged in. "higher_is_harsher" drives the care-score direction. ───────────────
# label, source-expr, valid (lo,hi) or None, higher_is_harsher, unit, human name
BEH_SPECS = {
    "charge_crate":   dict(name="Charge C-rate",        unit="C",     harsh=True,  vmin=0.0,  vmax=2.0),
    "discharge_crate":dict(name="Discharge C-rate",     unit="C",     harsh=True,  vmin=0.0,  vmax=2.0),
    "peak_crate":     dict(name="Peak C-rate (p95)",    unit="C",     harsh=True,  vmin=0.0,  vmax=3.0),
    "frac_soc_high":  dict(name="High-SoC dwell",       unit="frac",  harsh=True,  vmin=0.0,  vmax=1.0),
    "frac_soc_low":   dict(name="Low-SoC / deep cycle", unit="frac",  harsh=True,  vmin=0.0,  vmax=1.0),
    "dod_mean":       dict(name="Depth of discharge",   unit="%-ish", harsh=True,  vmin=0.0,  vmax=25.0),
    "ah_throughput":  dict(name="Monthly Ah throughput",unit="Ah/mo", harsh=True,  vmin=0.0,  vmax=1200.0),
    "km_month":       dict(name="Monthly distance",     unit="km/mo", harsh=True,  vmin=0.0,  vmax=3000.0),
    "wh_per_km":      dict(name="Energy use",           unit="Wh/km", harsh=True,  vmin=20.0, vmax=600.0),
    "temp_max":       dict(name="Peak temperature",     unit="degC",  harsh=True,  vmin=-10.0,vmax=70.0),
}

# metrics safe to use for the care score / KMeans (well-populated for ~all 95 vins)
CORE_BEH = ["charge_crate", "discharge_crate", "peak_crate", "frac_soc_high",
            "frac_soc_low", "dod_mean", "ah_throughput"]


# ───────────────────────────── helpers ─────────────────────────────
def _clip_valid(s: pd.Series, lo, hi) -> pd.Series:
    if lo is None and hi is None:
        return s
    return s.where((s >= lo) & (s <= hi))


def _f(x):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 4)


def _per_vehicle_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """One row per vin: lifetime-mean of each behaviour metric (robustly cleaned), plus the observed
    monthly SoH-loss rate and a few descriptors. C-rates are derived from currents / NOMINAL_AH."""
    rows = []
    for vin, g in df.groupby("vin"):
        g = g.sort_values("month")
        chg = _clip_valid(g["cur_chg_mean"].abs(), 0, 200)
        dis = _clip_valid(g["cur_dis_mean"].abs(), 0, 300)
        pk = _clip_valid(g["cur_abs_p95"], 0, 400)
        rec = {
            "vin": vin,
            "charge_crate":    (chg / NOMINAL_AH).mean(),
            "discharge_crate": (dis / NOMINAL_AH).mean(),
            "peak_crate":      (pk / NOMINAL_AH).mean(),
            "frac_soc_high":   g["frac_soc_high"].mean(),
            "frac_soc_low":    g["frac_soc_low"].mean(),
            "dod_mean":        _clip_valid(g["dod_mean"], 0, 25).mean(),
            "ah_throughput":   g["ah_throughput"].mean(),
            "km_month":        _clip_valid(g["km_month"], 0, 3000).mean(),
            "wh_per_km":       _clip_valid(g["wh_per_km"], 20, 600).mean(),
            "temp_max":        _clip_valid(g["temp_max"], -10, 70).mean(),
            "soc_mean":        g["soc_mean"].mean(),
            "soh_now":         g["soh"].iloc[-1],
            "age_months":      g["age_months"].max(),
            "n_months":        len(g),
            "loss_rate":       _loss_rate(g),
        }
        rows.append(rec)
    return pd.DataFrame(rows).set_index("vin")


def _loss_rate(g: pd.DataFrame):
    """Observed monthly SoH-loss rate (%/month) for one vehicle: sort by month, take the per-step
    drop (prev - curr) over the month-gap, keep gaps <= 3 months, clip negatives (SoH can only fade),
    average. Returns NaN if fewer than 2 usable transitions."""
    g = g.sort_values("month")
    if len(g) < 3:
        return np.nan
    gap = g["month"].diff().dt.days / DAYS_PER_MONTH
    loss = (g["soh"].shift(1) - g["soh"]) / gap          # prev - curr => positive when SoH fades
    m = (gap > 0) & (gap <= 3) & loss.notna()
    if m.sum() < 2:
        return np.nan
    return float(loss[m].clip(lower=0).mean())


def _pct_rank(series: pd.Series, value) -> float:
    """Percentile (0-100) of `value` within the non-null fleet distribution."""
    s = series.dropna()
    if len(s) == 0 or value is None or not np.isfinite(value):
        return None
    return round(float((s < value).mean() * 100.0), 1)


def _zscore(series: pd.Series, value):
    s = series.dropna()
    if len(s) < 3 or value is None or not np.isfinite(value):
        return None
    sd = s.std(ddof=0)
    if sd == 0:
        return 0.0
    return round(float((value - s.mean()) / sd), 2)


# ───────────────────────────── 1. customer_insights ─────────────────────────────
def customer_insights(feature_table_df: pd.DataFrame, vin: str) -> dict:
    """Per-vehicle, operator-facing insights for `vin`, benchmarked against the fleet.

    Returns dict with: vin, label, current SoH / age / observed loss-rate; a `metrics` block (each
    metric: value, unit, fleet percentile, z-score, plain verdict); a 0-100 `care_score`
    (higher = gentler usage); ranked `levers` (biggest behavioural changes that would help); and
    human-readable `insights` strings. All values JSON-serializable.
    """
    fb = _per_vehicle_behaviour(feature_table_df)
    if vin not in fb.index:
        return {"vin": vin, "error": "vin not found in feature table"}
    row = fb.loc[vin]

    # ── per-metric: value, fleet percentile, z, verdict ──────────────────────────
    metrics = {}
    for key, spec in BEH_SPECS.items():
        val = row[key]
        pct = _pct_rank(fb[key], val)
        z = _zscore(fb[key], val)
        verdict = _verdict(pct, spec["harsh"]) if pct is not None else "no fleet-comparable data"
        metrics[key] = {
            "name": spec["name"], "unit": spec["unit"],
            "value": _f(val), "fleet_pctile": pct, "z": z, "verdict": verdict,
        }

    # ── 0-100 battery-care score: average over core metrics of a per-metric gentleness score, where
    #    gentleness = 100 - harshness-percentile (so a vehicle in the gentlest fleet quartile scores
    #    ~85+, the harshest ~15). Only well-populated core metrics count. ──────────────────────────
    gentle = []
    for key in CORE_BEH:
        pct = metrics[key]["fleet_pctile"]
        if pct is None:
            continue
        gentle.append((100.0 - pct) if BEH_SPECS[key]["harsh"] else pct)
    care_score = round(float(np.mean(gentle)), 1) if gentle else None

    # ── behavioural levers: the harsh metrics where this vehicle is worst vs fleet (highest harsh
    #    percentile) AND that actually correlate with faster fade fleet-wide — ranked. ─────────────
    corr = _behaviour_degradation_corr(fb)            # {metric: spearman vs loss_rate}
    levers = []
    for key in CORE_BEH + ["km_month", "wh_per_km", "temp_max"]:
        pct = metrics[key]["fleet_pctile"]
        if pct is None or not BEH_SPECS[key]["harsh"] or pct < 60:
            continue                                   # only flag metrics this vehicle is HIGH on
        c = corr.get(key, 0.0) or 0.0
        # severity = how far above the fleet this vehicle is, weighted by how predictive the metric is
        severity = (pct - 50) / 50.0 * (0.3 + abs(c))
        levers.append({
            "metric": key, "name": BEH_SPECS[key]["name"], "fleet_pctile": pct,
            "degradation_corr": _f(c), "severity": round(float(severity), 3),
            "action": _lever_action(key),
        })
    levers = sorted(levers, key=lambda d: -d["severity"])[:3]

    # ── human-readable insight strings ───────────────────────────────────────────
    insights = _customer_strings(row, metrics, care_score, levers, fb)

    return {
        "vin": vin,
        "label": vin[-6:],
        "current_soh": _f(row["soh_now"]),
        "age_months": _f(row["age_months"]),
        "months_observed": int(row["n_months"]),
        "observed_loss_rate_pct_per_month": _f(row["loss_rate"]),
        "fleet_size": int(len(fb)),
        "care_score": care_score,
        "care_grade": _grade(care_score),
        "metrics": metrics,
        "levers": levers,
        "insights": insights,
    }


def _verdict(pct, harsh):
    """Plain word for where a vehicle sits in the fleet for a metric (harsh=True => high pct is bad)."""
    if harsh:
        if pct >= 80:  return "much harsher than fleet"
        if pct >= 60:  return "harsher than fleet"
        if pct <= 20:  return "much gentler than fleet"
        if pct <= 40:  return "gentler than fleet"
        return "typical for fleet"
    else:
        if pct >= 80:  return "much better than fleet"
        if pct <= 20:  return "much worse than fleet"
        return "typical for fleet"


def _grade(score):
    if score is None: return None
    if score >= 75:  return "Gentle"
    if score >= 55:  return "Moderate"
    if score >= 40:  return "Firm"
    return "Hard"


def _lever_action(key):
    return {
        "charge_crate":   "Avoid high-power / fast charging where possible — lower charge current reduces lithium-plating stress.",
        "discharge_crate":"Encourage smoother driving (less hard acceleration) to cut average discharge current.",
        "peak_crate":     "Trim peak-current events (heavy loads / hard launches) — peak C-rate drives heat and stress.",
        "frac_soc_high":  "Reduce time parked at full charge — charge to ~80-90% for daily use and avoid long high-SoC dwell.",
        "frac_soc_low":   "Avoid running the pack down to very low SoC — recharge before deep depletion.",
        "dod_mean":       "Use shallower discharge cycles (top up mid-shift) instead of full deep cycles.",
        "ah_throughput":  "Very high charge/discharge throughput — consider rotating this vehicle to lighter routes.",
        "km_month":       "High monthly mileage accelerates cycle aging — balance route assignment across the fleet.",
        "wh_per_km":      "High energy-per-km suggests heavy loading / aggressive driving — coach for efficiency.",
        "temp_max":       "High peak temperatures — improve charging-bay ventilation / avoid charging in peak heat.",
    }.get(key, "Review operating pattern.")


def _customer_strings(row, metrics, care_score, levers, fb):
    out = []
    soh = row["soh_now"]; age = row["age_months"]; lr = row["loss_rate"]
    out.append(f"Current SoH {soh:.1f}% at {age:.0f} months in service "
               f"(observed fade ~{(lr if np.isfinite(lr) else 0):.2f}%/month).")
    if care_score is not None:
        out.append(f"Battery-care score {care_score:.0f}/100 ({_grade(care_score)} usage) — "
                   f"higher means gentler treatment than the rest of the fleet.")
    # charging
    cc = metrics["charge_crate"]; hi = metrics["frac_soc_high"]
    if cc["value"] is not None:
        out.append(f"Charging: ~{cc['value']:.2f}C average charge rate ({cc['verdict']}, "
                   f"{cc['fleet_pctile']:.0f}th pctile); sits at high SoC (>90%) "
                   f"{(hi['value'] or 0)*100:.0f}% of the time ({hi['verdict']}).")
    # driving / usage intensity
    dc = metrics["discharge_crate"]; ah = metrics["ah_throughput"]; km = metrics["km_month"]
    drv = (f"Driving: ~{dc['value']:.2f}C average discharge ({dc['verdict']}); "
           f"{ah['value']:.0f} Ah/month throughput ({ah['fleet_pctile']:.0f}th pctile)")
    if km["value"] is not None:
        drv += f"; ~{km['value']:.0f} km/month"
    out.append(drv + ".")
    # depth of discharge
    dod = metrics["dod_mean"]; lo = metrics["frac_soc_low"]
    if dod["value"] is not None:
        out.append(f"Cycling depth: DoD index {dod['value']:.1f} ({dod['verdict']}); "
                   f"dwells below 20% SoC {(lo['value'] or 0)*100:.0f}% of the time.")
    # efficiency / thermal where available
    wh = metrics["wh_per_km"]; tp = metrics["temp_max"]
    if wh["value"] is not None:
        out.append(f"Efficiency: {wh['value']:.0f} Wh/km ({wh['verdict']}).")
    if tp["value"] is not None:
        out.append(f"Thermal: peak battery temp ~{tp['value']:.0f}°C ({tp['verdict']}).")
    # levers
    if levers:
        lv = "; ".join(f"{l['name']} ({l['fleet_pctile']:.0f}th pctile)" for l in levers)
        out.append(f"Biggest life-extension levers for this vehicle: {lv}.")
    else:
        out.append("No single behaviour stands out as harsh — fade is mostly calendar/cycle aging.")
    return out


# ───────────────────────────── 2. fleet_behavior ─────────────────────────────
def fleet_behavior(feature_table_df: pd.DataFrame) -> dict:
    """Fleet-wide behaviour analysis. Returns dict with:
      fleet_size, distributions (per metric summary stats + percentiles),
      segments (KMeans k=3 behaviour groups, each labelled + size + mean SoH-loss rate),
      degradation_drivers (ranked Spearman corr of each behaviour vs monthly SoH-loss rate),
      findings (actionable strings). All JSON-serializable.
    """
    fb = _per_vehicle_behaviour(feature_table_df)
    n = len(fb)

    # ── distributions ────────────────────────────────────────────────────────────
    distributions = {}
    for key, spec in BEH_SPECS.items():
        s = fb[key].dropna()
        if len(s) == 0:
            continue
        distributions[key] = {
            "name": spec["name"], "unit": spec["unit"], "n": int(len(s)),
            "mean": _f(s.mean()), "median": _f(s.median()), "std": _f(s.std()),
            "p10": _f(s.quantile(0.10)), "p25": _f(s.quantile(0.25)),
            "p75": _f(s.quantile(0.75)), "p90": _f(s.quantile(0.90)),
            "min": _f(s.min()), "max": _f(s.max()),
        }
    distributions["loss_rate"] = {
        "name": "Monthly SoH-loss rate", "unit": "%/mo", "n": int(fb["loss_rate"].notna().sum()),
        "mean": _f(fb["loss_rate"].mean()), "median": _f(fb["loss_rate"].median()),
        "p75": _f(fb["loss_rate"].quantile(0.75)), "p90": _f(fb["loss_rate"].quantile(0.90)),
        "max": _f(fb["loss_rate"].max()),
    }

    # ── degradation drivers: Spearman corr of each behaviour metric vs loss_rate ──
    corr = _behaviour_degradation_corr(fb, with_n=True)
    drivers = sorted(
        [{"metric": k, "name": BEH_SPECS[k]["name"], "spearman": _f(v["rho"]), "n": v["n"]}
         for k, v in corr.items()],
        key=lambda d: -abs(d["spearman"] if d["spearman"] is not None else 0.0),
    )

    # ── behaviour segments via KMeans (k=3) on standardized core behaviour ────────
    segments = _segment_fleet(fb, k=3)

    # ── actionable findings ──────────────────────────────────────────────────────
    findings = _fleet_findings(fb, distributions, drivers, segments)

    return {
        "fleet_size": int(n),
        "distributions": distributions,
        "degradation_drivers": drivers,
        "segments": segments,
        "findings": findings,
    }


def _behaviour_degradation_corr(fb: pd.DataFrame, with_n: bool = False):
    """Spearman correlation of each behaviour metric with the per-vehicle monthly SoH-loss rate.
    Returns {metric: rho} (or {metric: {rho,n}} if with_n). Skips metrics with <10 paired vins."""
    out = {}
    for key in BEH_SPECS:
        sub = fb[[key, "loss_rate"]].dropna()
        if len(sub) < 10 or sub[key].nunique() <= 3:
            rho, nn = None, int(len(sub))
        else:
            rho = float(sub[key].corr(sub["loss_rate"], method="spearman"))
            rho = rho if np.isfinite(rho) else None
            nn = int(len(sub))
        out[key] = {"rho": rho, "n": nn} if with_n else rho
    return out


def _segment_fleet(fb: pd.DataFrame, k: int = 3) -> list:
    """KMeans (k clusters) on standardized core behaviour features. Each cluster is labelled in plain
    language from its standout standardized features and ordered gentlest -> hardest by a harshness
    index. Returns a list of segment dicts (label, size, share, mean care/loss, profile)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = fb[CORE_BEH].dropna()
    if len(X) < k + 2:
        return []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    Z = pd.DataFrame(Xs, index=X.index, columns=CORE_BEH)
    Z["seg"] = labels
    joined = X.copy()
    joined["seg"] = labels
    joined = joined.join(fb[["loss_rate", "soh_now", "age_months"]])

    # harshness index = mean standardized value across harsh core metrics (all CORE_BEH are harsh)
    harsh_idx = Z.groupby("seg")[CORE_BEH].mean().mean(axis=1)
    order = harsh_idx.sort_values().index.tolist()       # gentlest first

    segs = []
    for rank, seg in enumerate(order):
        members = joined[joined["seg"] == seg]
        zmean = Z[Z["seg"] == seg][CORE_BEH].mean()
        label, desc = _segment_label(rank, len(order), zmean)
        segs.append({
            "segment_id": int(rank),
            "label": label,
            "description": desc,
            "size": int(len(members)),
            "share_pct": round(100.0 * len(members) / len(X), 1),
            "mean_loss_rate_pct_per_month": _f(members["loss_rate"].mean()),
            "mean_soh_now": _f(members["soh_now"].mean()),
            "mean_age_months": _f(members["age_months"].mean()),
            "harshness_index": _f(harsh_idx[seg]),
            "profile": {key: _f(members[key].mean()) for key in CORE_BEH},
            "standout_traits": _standout_traits(zmean),
        })
    return segs


def _standout_traits(zmean: pd.Series):
    """Plain-language standout features of a segment from its standardized means (|z|>=0.5)."""
    NICE = {
        "charge_crate": "fast charging", "discharge_crate": "hard discharge",
        "peak_crate": "high peak current", "frac_soc_high": "high-SoC dwell",
        "frac_soc_low": "deep discharging", "dod_mean": "deep cycles",
        "ah_throughput": "high throughput",
    }
    traits = []
    for key, z in zmean.sort_values(ascending=False).items():
        if z >= 0.5:
            traits.append(f"high {NICE[key]}")
        elif z <= -0.5:
            traits.append(f"low {NICE[key]}")
    return traits[:4]


def _segment_label(rank, k, zmean):
    """Name a segment. Rank 0 = gentlest. Use standout traits to specialise the middle/hard groups."""
    traits = _standout_traits(zmean)
    base = {0: "Gentle usage", k - 1: "Hard usage"}.get(rank, "Average usage")
    # specialise if a clear dominant harsh trait exists
    dom = zmean.idxmax()
    domz = zmean.max()
    special = {
        "charge_crate": "Heavy fast-chargers", "peak_crate": "High-current haulers",
        "discharge_crate": "Hard drivers", "frac_soc_high": "High-SoC dwellers",
        "frac_soc_low": "Deep-dischargers", "dod_mean": "Deep cyclers",
        "ah_throughput": "High-throughput workhorses",
    }
    if rank not in (0,) and domz >= 0.7 and dom in special:
        label = special[dom]
    else:
        label = base
    desc = (", ".join(traits) if traits else "no standout traits") + "."
    return label, desc.capitalize()


def _fleet_findings(fb, distributions, drivers, segments):
    out = []
    n = len(fb)
    lr = fb["loss_rate"].dropna()
    if len(lr):
        out.append(f"Fleet median fade is {lr.median():.2f}%/month; the worst-decile vehicles fade "
                   f"~{lr.quantile(0.90):.2f}%/month — roughly {lr.quantile(0.90)/max(lr.median(),1e-6):.1f}x "
                   f"the median, so a small tail drives most of the warranty risk.")
    # top driver
    top = [d for d in drivers if d["spearman"] is not None]
    if top:
        d0 = top[0]
        direction = "faster" if d0["spearman"] > 0 else "slower"
        if abs(d0["spearman"]) < 0.2:
            out.append(f"No behaviour metric strongly predicts degradation in this fleet "
                       f"(top is {d0['name']}, Spearman {d0['spearman']:+.2f}) — consistent with the "
                       f"SoH model's finding that calendar/cycle aging dominates over operating style here.")
        else:
            out.append(f"{d0['name']} is the strongest behavioural predictor of {direction} fade "
                       f"(Spearman {d0['spearman']:+.2f}, n={d0['n']}) — worth targeting operationally.")
    # segments
    if segments:
        hard = max(segments, key=lambda s: s["harshness_index"] if s["harshness_index"] is not None else -9)
        gentle = min(segments, key=lambda s: s["harshness_index"] if s["harshness_index"] is not None else 9)
        if hard["mean_loss_rate_pct_per_month"] and gentle["mean_loss_rate_pct_per_month"]:
            out.append(f"The '{hard['label']}' segment ({hard['size']} vehicles, {hard['share_pct']:.0f}% "
                       f"of fleet) fades {hard['mean_loss_rate_pct_per_month']:.2f}%/mo vs "
                       f"{gentle['mean_loss_rate_pct_per_month']:.2f}%/mo for '{gentle['label']}' "
                       f"({gentle['size']} vehicles) — a usage-driven gap worth coaching/route-balancing.")
    # high-SoC dwell prevalence (charging hygiene)
    hi = fb["frac_soc_high"].dropna()
    if len(hi):
        share = float((hi > 0.4).mean() * 100)
        out.append(f"{share:.0f}% of vehicles spend >40% of their time above 90% SoC — a fleet-wide "
                   f"charging-hygiene opportunity (charge-to-80% policy / avoid long full-charge dwell).")
    # peak C-rate spread
    pk = fb["peak_crate"].dropna()
    if len(pk):
        out.append(f"Peak C-rate ranges {pk.quantile(0.1):.2f}-{pk.quantile(0.9):.2f}C across the fleet; "
                   f"the high-current tail is where thermal/plating stress concentrates.")
    return out


# ───────────────────────────── standalone / self-test ─────────────────────────────
def _load_default_frame() -> pd.DataFrame:
    here = Path(__file__).resolve().parent.parent
    fp = here / "data" / "mahindra" / "features" / "feature_table.parquet"
    return pd.read_parquet(fp)


if __name__ == "__main__":
    import json

    df = _load_default_frame()
    print(f"Loaded feature table: {df.shape[0]} rows, {df['vin'].nunique()} vehicles\n")

    print("=" * 78)
    print("FLEET BEHAVIOR")
    print("=" * 78)
    fleet = fleet_behavior(df)
    print(f"fleet_size = {fleet['fleet_size']}\n")
    print("-- behaviour segments --")
    for s in fleet["segments"]:
        print(f"  [{s['label']}] n={s['size']} ({s['share_pct']}%)  "
              f"meanLoss={s['mean_loss_rate_pct_per_month']}%/mo  "
              f"meanSoH={s['mean_soh_now']}%  traits={s['standout_traits']}")
    print("\n-- degradation drivers (Spearman vs monthly SoH-loss) --")
    for d in fleet["degradation_drivers"]:
        print(f"  {d['name']:24s} {d['spearman']}  (n={d['n']})")
    print("\n-- findings --")
    for f in fleet["findings"]:
        print(f"  * {f}")

    print("\n" + "=" * 78)
    print("CUSTOMER INSIGHTS (sample vehicles)")
    print("=" * 78)
    # pick a spread: lowest SoH, a mid vehicle, highest throughput
    fb = _per_vehicle_behaviour(df)
    sample = [
        fb["soh_now"].idxmin(),
        fb.sort_values("soh_now").index[len(fb) // 2],
        fb["ah_throughput"].idxmax(),
    ]
    for vin in dict.fromkeys(sample):           # dedupe, keep order
        ci = customer_insights(df, vin)
        print(f"\n--- {vin}  (SoH {ci['current_soh']}%, age {ci['age_months']}mo, "
              f"care {ci['care_score']}/100 [{ci['care_grade']}]) ---")
        for ins in ci["insights"]:
            print(f"  * {ins}")
        if ci["levers"]:
            print("  levers:", [l["name"] for l in ci["levers"]])

    # confirm JSON-serializable
    json.dumps({"fleet": fleet, "customer": customer_insights(df, sample[0])})
    print("\n[OK] outputs are JSON-serializable.")
