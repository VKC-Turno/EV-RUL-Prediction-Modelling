#!/usr/bin/env python3
"""Build a per-vehicle DATA-QUALITY manifest so we never train on vehicles whose SoH trend is unprovable,
and so we know which dense files to delete & re-fill with better vehicles.

For every vehicle in each OEM feature table it records observation depth (valid SoH months, age span,
per-month sample density), the SoH trajectory (start/end/drop, degrader-vs-flat), registration/model, and
a quality verdict:
  quality = "trainable"  -> enough data to trust its trend (use for training)
            "thin"       -> too few valid months / too short a span -> EXCLUDE from training; free the space

Thresholds are OEM-aware (Bajaj's feed window is short for everyone, so it gets no span bar). Tune in QUAL.
-> data/manifests/vehicle_data_quality.csv   (one row per vehicle)
Run: .venv/bin/python src/build_data_quality.py
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

os.chdir(Path(__file__).resolve().parent.parent); sys.path.insert(0, "src")

FT = {"Euler": "data/euler/features/feature_table.parquet",
      "Mahindra": "data/mahindra/features/feature_table.parquet",
      "Bajaj": "data/bajaj/features/feature_table.parquet"}
REG_FILES = {"Euler": ("data/euler/Euler_Regd_Details.csv", "regd_date", "%d/%m/%y"),
             "Mahindra": ("Mh_Regd_Date.csv", "vehicle_registration_date", None),
             "Bajaj": ("Bajaj_Regd_Details.csv", "regd_date", None)}
# min valid SoH months AND min age span (months) to trust a vehicle's trend
QUAL = {"Euler": dict(min_months=6, min_span=9.0),
        "Mahindra": dict(min_months=6, min_span=9.0),
        "Bajaj": dict(min_months=6, min_span=0.0)}   # Bajaj feed is ~9 mo for all -> no span bar
DEG = 2.0
# A vehicle that reached end-of-life, OR dropped >= EXEMPT_DROP pp, has PROVEN a real degradation trend —
# it is never "thin" no matter how short the window (the span rule only exists to confirm a trend exists).
EOL = {"Euler": 80, "Mahindra": 80, "Bajaj": 70}
EXEMPT_DROP = 5.0


def reg_dates(oem):
    f, col, fmt = REG_FILES[oem]
    if not Path(f).exists():
        return {}
    r = pd.read_csv(f)
    d = pd.to_datetime(r[col], format=fmt, errors="coerce") if fmt else pd.to_datetime(r[col], errors="coerce")
    vc = next((c for c in r.columns if c.lower() == "vin"), "vin")
    return dict(zip(r[vc], d))


def model_map():
    p = "Vin_Model_Details.csv"
    if not Path(p).exists():
        return {}
    v = pd.read_csv(p); v = v[v["model"].notna()].drop_duplicates("vin")
    return dict(zip(v["vin"], v["make"].fillna("") + " " + v["model"].fillna("")))


def load_cohort(oem, ft):
    """Mirror the dashboard's load_cohort: run the manifest on the FULL Redshift STORE feature table
    (data/redshift/<oem>_featengg.parquet) when it's larger than the local one, so data-quality reflects
    the cohort the pipeline actually trains on — not the older, smaller local download subset."""
    p = f"data/redshift/{oem.lower()}_featengg.parquet"
    if Path(p).exists():
        s = pd.read_parquet(p).rename(columns={"ymd": "month"}); s["vin"] = s["vin"].astype(str)
        if s["vin"].nunique() > pd.read_parquet(ft)["vin"].nunique():
            return s
    return pd.read_parquet(ft)


VMODEL = model_map()
rows = []
for oem, ft in FT.items():
    m = load_cohort(oem, ft); m["month"] = pd.to_datetime(m["month"].astype(str))
    reg = reg_dates(oem); q = QUAL[oem]
    dens = next((c for c in ["n_rows", "n_sessions", "n_rows_ic", "n_soh"] if c in m.columns), None)
    for vin, g in m.sort_values("month").groupby("vin"):
        months = len(g); fa = float(g.age_months.min()); la = float(g.age_months.max()); span = la - fa
        s0, s1 = float(g.soh.iloc[0]), float(g.soh.iloc[-1]); drop = s0 - s1
        thin = []
        if months < q["min_months"]:
            thin.append(f"<{q['min_months']}_valid_months")
        if span < q["min_span"]:
            thin.append(f"<{q['min_span']:.0f}mo_span")
        proven = drop >= EXEMPT_DROP or s1 <= EOL.get(oem, 80)   # big drop / reached EoL => keep regardless
        if thin and proven:
            quality, reasons = "trainable", "proven-degrader(overrides:" + ";".join(thin) + ")"
        elif thin:
            quality, reasons = "thin", ";".join(thin)
        else:
            quality, reasons = "trainable", ""
        rd = reg.get(vin)
        rows.append(dict(
            oem=oem, vin=vin, model=VMODEL.get(vin, "").strip(),
            reg_date=(rd.date().isoformat() if pd.notna(rd) else ""),
            months=months, span_months=round(span, 1), first_age_mo=round(fa, 1),
            current_age_mo=round(la, 1), obs_per_month=(int(g[dens].median()) if dens else ""),
            soh_first=round(s0, 1), soh_last=round(s1, 1), soh_drop=round(drop, 1),
            vehicle_class=("degrader" if drop >= DEG else "flat"),
            quality=quality, reasons=reasons,
            dense_file=f"data/{oem.lower()}/dense/{vin}.parquet"))

df = pd.DataFrame(rows).sort_values(["oem", "quality", "vehicle_class", "vin"])
out = "data/manifests/vehicle_data_quality.csv"
Path("data/manifests").mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} vehicles\n")
print(f"{'OEM':9} {'trainable':>9} {'thin':>5} {'thin degraders':>15} {'thin flat':>10}")
for oem, d in df.groupby("oem"):
    thin = d[d.quality == "thin"]
    print(f"{oem:9} {(d.quality=='trainable').sum():>9} {len(thin):>5} "
          f"{(thin.vehicle_class=='degrader').sum():>15} {(thin.vehicle_class=='flat').sum():>10}")
print(f"\nTOTAL trainable {(df.quality=='trainable').sum()} / {len(df)} | "
      f"thin (free space + re-import): {(df.quality=='thin').sum()}")
