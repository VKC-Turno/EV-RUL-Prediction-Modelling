"""Path helpers — keep all data under data/<oem>/<source>/ so the layout scales to many OEMs.

Layout:
  data/
    manifests/                 # <oem>_cohort.csv, <oem>_overlap.csv, vin manifests
    <oem>/                     # e.g. mahindra/, piaggio/
      intellicar/              # this OEM's vehicles extracted from the intellicar table
      feed/                    # this OEM's own native-feed extract
      features/                # feature_table.parquet
      soh/                     # soh series, rul outputs
"""
import os
from pathlib import Path


def repo_root() -> Path:
    """Walk up from CWD to the repo root (marked by requirements.txt) and return it."""
    p = Path.cwd()
    while not (p / "requirements.txt").exists() and p != p.parent:
        p = p.parent
    return p


def chdir_root() -> Path:
    """Chdir to repo root so relative data/ paths resolve from anywhere (use atop notebooks)."""
    root = repo_root()
    os.chdir(root)
    return root


ROOT = repo_root()
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"


def oem_dir(oem: str, source: str = "") -> Path:
    """data/<oem>/<source>/ — source in {intellicar, feed, features, soh}. Creates it."""
    d = DATA / oem / source if source else DATA / oem
    d.mkdir(parents=True, exist_ok=True)
    return d


def cohort_csv(oem: str) -> Path:
    return MANIFESTS / f"{oem}_cohort.csv"


def overlap_csv(oem: str) -> Path:
    return MANIFESTS / f"{oem}_overlap.csv"
