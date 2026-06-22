# EULER_RUL_MODEL — Battery SoH & Degradation Forecasting

Pipeline to extract EV battery telemetry from S3, compute **State of Health (SoH)**, and train an
**SoH forecasting / RUL model**. Designed to scale across **multiple OEMs** (Mahindra first; Bajaj,
Euler, Montra, Piaggio, JBM to follow).

## Folder structure

```
EULER_RUL_MODEL/
├── README.md                 # this file
├── requirements.txt
├── .env / .env.example       # AWS creds + S3_BUCKET (gitignored .env)
├── src/                      # code (run from repo root: `python src/<script>.py`)
│   ├── config.py             # ★ OEM registry — add a new OEM here (prefixes, columns, quirks)
│   ├── paths.py              # repo-root + per-OEM path helpers (data/<oem>/<source>/)
│   ├── soh.py                # vectorized coulomb-counting SoH (runs on cuDF/GPU or CPU)
│   ├── import_cohort.py      # cohort extractor:  python src/import_cohort.py <feed> <oem>
│   └── import_*.py           # earlier per-feed / dense / single-vin extractors
├── notebooks/                # analysis, organized as the pipeline FLOW (see notebooks/README.md)
│   ├── 00_extract/           # pull telemetry from S3
│   ├── 01_soh_target/        # compute the SoH target (coulomb / BMS-capacity / proxy)
│   ├── 02_features_model/    # feature table + condition-aware model (P10–P90)
│   ├── 03_validation/        # leave-one-vehicle-out backtests
│   ├── 04_forecast_decisions/# warranty risk, RUL, resale forecasts
│   ├── 05_insights/          # degradation-driver & km analyses
│   └── _archive/             # superseded / exploratory notebooks
├── docs/
│   ├── PROCEDURE.md                   # full runbook + lessons learned + modeling options
│   └── SOH_COLUMN_USABILITY.md        # per-column audit (SoH + forecasting features)
└── data/                     # gitignored working data
    ├── manifests/            # <oem>_cohort.csv, <oem>_overlap.csv, vin manifests
    └── <oem>/                # one folder per OEM, e.g. mahindra/
        ├── intellicar/       # this OEM's vehicles from the intellicar table (current!)
        ├── feed/             # this OEM's own native-feed extract
        ├── features/         # feature_table.parquet
        ├── soh/              # SoH series + RUL outputs
        └── _archive/         # exploratory / superseded extracts
```

## Add a new OEM (e.g. Piaggio)

1. **Audit columns** for that OEM in both the intellicar table and its native feed
   (`docs/SOH_COLUMN_USABILITY.md` shows the method). Update `src/config.py`:
   - add an `OEM_FEEDS["piaggio"]` entry (prefix, useful `cols`, `has_current`, notes),
   - add its populated intellicar columns under `INTELLICAR["cols_by_oem"]["piaggio"]`.
2. **Build the cohort:** intersect the OEM's vehicles in intellicar vs its native feed, rank by
   odometer → `data/manifests/piaggio_cohort.csv`.
3. **Extract:** `python src/import_cohort.py intellicar piaggio` and
   `python src/import_cohort.py piaggio piaggio` → `data/piaggio/{intellicar,feed}/`.
4. **SoH + features + model:** reuse the notebooks (point them at `data/piaggio/...`).

> Note: only the **intellicar** table has pack `current` → coulomb-counting SoH. OEM native feeds
> fall back to a distance-per-SoC proxy (or a reported `batterySoh`, for Euler). See PROCEDURE.md.

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env            # fill AWS creds + S3_BUCKET
python src/import_cohort.py intellicar mahindra
python src/import_cohort.py mahindra   mahindra
jupyter notebook notebooks/02_features_model/mahindra_features_baseline.ipynb
```

> Notebooks read top-to-bottom as the pipeline runs (`00_extract` → `05_insights`). Start at
> [`notebooks/README.md`](notebooks/README.md) for the flow map, and
> [`docs/OEM_MODELING_PLAYBOOK.md`](docs/OEM_MODELING_PLAYBOOK.md) for the methods & assumptions.
> Run notebooks with the `euler-rul` kernel.
