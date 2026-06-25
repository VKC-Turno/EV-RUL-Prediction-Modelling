"""Shared data-quality gate: which vehicles are trustworthy enough to train on / predict for.

Single source of truth for "don't train on unprovable vehicles". Reads the per-vehicle manifest
`data/manifests/vehicle_data_quality.csv` (built by `build_data_quality.py`) and drops ONLY vehicles
EXPLICITLY flagged `quality == "thin"` for the OEM. Vehicles ABSENT from the manifest are KEPT — presumed
OK until the manifest is regenerated for the full fleet — so the gate never silently excludes new fleet
vehicles (important at AWS/fleet scale, where the shipped manifest only knows the local cohort).

Every exclusion is logged with its reason (logger `data_quality`, WARNING level → shows in Glue
CloudWatch and the local console). If the manifest is missing, nothing is dropped.

Used by the dashboard, `euler_train`, and the Glue training/prediction jobs.
"""
import logging
from pathlib import Path
import pandas as pd

MANIFEST = "data/manifests/vehicle_data_quality.csv"
_log = logging.getLogger("data_quality")


def thin_map(oem, manifest=MANIFEST):
    """{vin: reason} for vehicles EXPLICITLY flagged data-thin for `oem`; empty dict if no manifest."""
    p = Path(manifest)
    if not p.exists():
        return {}
    q = pd.read_csv(p)
    t = q[(q["oem"].astype(str).str.lower() == str(oem).lower()) & (q["quality"] == "thin")]
    reasons = (t["reasons"] if "reasons" in t.columns
               else pd.Series(index=t.index, dtype=object)).fillna("data-thin")
    return dict(zip(t["vin"], reasons))


def trainable_vins(oem, manifest=MANIFEST):
    """Back-compat helper: the set of VINs NOT flagged thin for `oem` (None if no manifest)."""
    p = Path(manifest)
    if not p.exists():
        return None
    q = pd.read_csv(p)
    return set(q[(q["oem"].astype(str).str.lower() == str(oem).lower()) & (q["quality"] != "thin")]["vin"])


def excluded(m, oem, manifest=MANIFEST):
    """The vehicles in `m` that WOULD be dropped, as a DataFrame [vin, reason] — the audit log of *why*."""
    thin = thin_map(oem, manifest)
    rows = [(v, thin[v]) for v in pd.unique(m["vin"]) if v in thin]
    return pd.DataFrame(rows, columns=["vin", "reason"])


def apply_quality(m, oem, manifest=MANIFEST):
    """Drop ONLY vehicles explicitly flagged `thin`; KEEP everything else — including vehicles absent from
    the manifest (presumed OK until it's regenerated for the full fleet). Logs each exclusion + its reason."""
    thin = thin_map(oem, manifest)
    excl = [v for v in pd.unique(m["vin"]) if v in thin]
    if excl:
        _log.warning("data-quality gate [%s]: excluding %d data-thin vehicle(s) of %d:",
                     oem, len(excl), m["vin"].nunique())
        for v in excl:
            _log.warning("  exclude %s  reason=%s", v, thin[v])
    return m[~m["vin"].isin(thin)].copy()
