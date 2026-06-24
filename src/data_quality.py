"""Shared data-quality gate: which vehicles are trustworthy enough to train on.

Single source of truth for "don't train on unprovable vehicles". Reads the per-vehicle manifest
`data/manifests/vehicle_data_quality.csv` (built by `build_data_quality.py`) and returns the set of
TRAINABLE vins (quality != "thin") for an OEM, or filters a feature frame to them. If the manifest is
missing it returns None / passes the frame through unchanged, so callers degrade gracefully.

Used by the dashboard training path and (on next retrain) the production model builders, so a vehicle
flagged data-thin can never be trained on by mistake.
"""
from pathlib import Path
import pandas as pd

MANIFEST = "data/manifests/vehicle_data_quality.csv"


def trainable_vins(oem, manifest=MANIFEST):
    """Set of trainable vins for `oem`, or None if no manifest exists."""
    p = Path(manifest)
    if not p.exists():
        return None
    q = pd.read_csv(p)
    return set(q[(q["oem"].str.lower() == oem.lower()) & (q["quality"] != "thin")]["vin"])


def apply_quality(m, oem, manifest=MANIFEST):
    """Drop data-thin vehicles from feature frame `m` for `oem` (no-op if the manifest is absent)."""
    keep = trainable_vins(oem, manifest)
    return m if keep is None else m[m["vin"].isin(keep)].copy()
