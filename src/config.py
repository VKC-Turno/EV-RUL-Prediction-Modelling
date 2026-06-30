"""OEM / data-source registry — the single place to add a new OEM.

To onboard a new OEM:
  1. Add an entry to OEM_FEEDS with its S3 prefix and useful columns (audit first!).
  2. If it appears in the intellicar table, list which intellicar columns are populated for it.
  3. Everything downstream (extract, SoH, features) reads from here — no per-OEM code forks.

Schema/availability facts below were established by auditing the data (see docs/SOH_COLUMN_USABILITY.md).
"""

# ── Shared multi-OEM telematics table (the only source with real `current`) ──────────────
INTELLICAR = {
    "prefix": "battery-oem-data/parquet/intellicar/battery-data/",
    "skip_year": "0000",                 # junk partition
    "reserved_cols": {"current"},        # must be quoted as s."current" in S3 Select
    "split": "battery-data/",
    "dense_files": True,                 # ~1-3k rows/file -> cheap to scan
    # Columns populated FOR MAHINDRA rows (other OEMs differ — audit per OEM).
    "cols_by_oem": {
        "mahindra": ["vin", "eventAt", "make", "model", "soc", "current",
                     "batteryVoltage", "odometer", "dte"],
        # "piaggio": [...],  # TODO audit; piaggio additionally has chargeCycle, batteryTemp
    },
}

# ── Per-OEM native feeds ─────────────────────────────────────────────────────────────────
OEM_FEEDS = {
    "mahindra": {
        "prefix": "battery-oem-data/parquet/mahindra/vehicle-data/",
        "split": "vehicle-data/",
        "dense_files": False,            # ~5 rows/file, ~70M files -> monthly-sample only
        "reserved_cols": set(),
        "has_current": False,
        "has_reported_soh": False,
        # forecasting columns (usage + environment); temp/state/kwh are sparse (~30%, status rows)
        "cols": ["eventAt", "vin", "soc", "odometer", "distanceToEmpty", "latitude",
                 "longitude", "gearPosition", "batteryTemp", "state", "kwh", "vehicleModel"],
        "notes": "No current. `kwh` is signed/instantaneous (NOT integrable). soc/odo have garbage.",
    },
    "euler": {
        "prefix": "battery-oem-data/parquet/euler/vehicle-data/",
        "split": "vehicle-data/",
        "dense_files": False,
        "reserved_cols": set(),
        "has_current": False,
        "has_reported_soh": True,        # `batterySoh` exists (coarse, has 0.0 garbage)
        "cols": ["eventAt", "vin", "batterySoc", "batterySoh", "odometer", "batteryTemperature"],
        "notes": "Reported batterySoh present; batteryVoltage is 100% NULL. No current.",
    },
    "bajaj": {
        "prefix": "battery-oem-data/parquet/bajaj/vehicle-data/",
        "split": "vehicle-data/",
        "dense_files": True,             # ~400 rows/file, multi-VIN -> dense; ~2k files/day
        "reserved_cols": set(),
        "has_current": False,            # no current AND no voltage AND no remaining-capacity
        "has_reported_soh": True,        # `essBmsSohcEstPercValue` is a clean, monotone reported SoH
        # Bajaj-native signal names (one Value per signal per row; *Time twins ignored).
        # SoH target = reported essBmsSohcEstPercValue (78-100, well-behaved). No coulomb / no BMS-capacity.
        "cols": ["eventAt", "vin",
                 "essBmsSocEstPercValue",        # SoC %  (=batterySoc)
                 "essBmsSohcEstPercValue",       # reported SoH %  (=batterySoh) -> TARGET
                 "essBmsChgcycleActCountValue",  # charge-cycle count (direct aging driver)
                 "essBmsTemperatureActDegcValue",# pack temp degC (=batteryTemperature)
                 "etsVcuAmbienttempActDegcValue",# ambient temp degC (climate proxy)
                 "etsVcuDriveeffEstWhpkmValue",  # drive efficiency Wh/km (range-fade proxy)
                 "evcChgInputenergycountActKwhValue",  # charge input energy kWh (per-row, NOT cumulative)
                 "hmiIclOdoActMValue"],          # odometer in METRES (=odometer*1000)
        "notes": ("Bajaj-native verbose schema (essBms*/etsVcu*/hmiIcl*). NO current, NO voltage, "
                  "NO remaining-capacity -> coulomb & BMS-capacity SoH both impossible; use reported "
                  "essBmsSohcEstPercValue. odometer is in metres. Span 2025-09..2026-06 (~10 mo)."),
    },
    # Not yet audited — add prefix + audited cols before extracting:
    # "piaggio": {"prefix": "battery-oem-data/parquet/piaggio/...", ...},
    # "montra": {...}, "jbm": {...},
}

# Warranty (years, km) by OEM + model keyword — whichever limit is hit first.
# Source: "0. Tech Specs for OEMs.xlsx" (docs). km in absolute km (1.2 lakh = 120000).
WARRANTY = {
    "mahindra": [("grand", (5, 120000)), ("treo", (3, 120000)), ("zor", (3, 80000))],
    # Euler: HiLoad (our cargo cohort) corrected to 3yr/80k 2026-06-30 — the bare-default 5yr/125k was the
    # passenger HiCity term mis-applied to HiLoad (HiLoad warranty 'unverified' in OEM_Model_Specs.csv;
    # independent search: HiLoad vehicle = 3yr, 80-100k km, +2yr extended battery). PROVISIONAL — get official doc.
    "euler":    [("hirange", (6, 150000)), ("hi range", (6, 150000)), ("hicity", (5, 125000)),
                 ("hi city", (5, 125000)), ("", (3, 80000))],
    # Bajaj RE E-TEC: BATTERY warranty 5yr / 120k km (OEM_Model_Specs.csv, spec-sheet verified) — the relevant
    # term for SoH at-risk. (The ~3yr the pricing-CSV warranty_end_date implies is the shorter VEHICLE warranty.)
    # The 120k-km limit usually binds BEFORE 5yr for high-use vehicles -> use the km-bound effective deadline.
    "bajaj":    [("", (5, 120000))],
    "piaggio":  [("", (3, 100000))],
    "montra":   [("", (5, 175000))],
}
DEFAULT_WARRANTY = (5, 120000)

# Fleet-level representative warranty (years, km) per OEM — the cohort-majority variant. SINGLE SOURCE for
# the one-warranty-line-per-OEM dashboard views (so config and the dashboard never drift apart again).
# Euler = HiLoad (3yr, provisional); Mahindra = Treo (3yr); Bajaj = RE E-TEC battery (5yr; km usually binds first).
FLEET_WARRANTY = {"euler": (3, 80000), "mahindra": (3, 120000), "bajaj": (5, 120000)}


def warranty_for(oem, model):
    """Return (years, km) for an OEM+model. Keyword rules checked in order; 'grand'/'treo'
    take precedence over bare 'zor' (cargo Treo Zor) for Mahindra."""
    ml = str(model).lower()
    for kw, val in WARRANTY.get(str(oem).lower(), []):
        if kw == "" or kw in ml:
            return val
    return DEFAULT_WARRANTY


# Default SoH method per source: 'coulomb' needs current (intellicar); 'distance_per_soc' for native
# Mahindra; 'bms_capacity' = BMS remaining-capacity -> full-capacity -> SoH (what euler_features actually
# builds — high-SoC band, isotonic fit); 'reported' = the BMS-reported SoH field directly (Bajaj).
# (Euler's 'batterySoh' reported field is garbage 0/>70000, so Euler uses bms_capacity, NOT 'reported'.)
SOH_METHOD = {"intellicar": "coulomb", "mahindra": "distance_per_soc", "euler": "bms_capacity",
              "bajaj": "reported"}

# Monthly-sample cadence: dense feeds tolerate more days/month.
DAYS_PER_MONTH = {"intellicar": [8, 16, 24], "default": [15]}
FILES_PER_DAY_CAP = {"intellicar": 2500, "default": 15000}
