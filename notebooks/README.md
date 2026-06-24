# Notebooks — end-to-end pipeline flow

These notebooks are organized to **read top-to-bottom as the pipeline runs**: extract raw telemetry →
compute the SoH target → engineer features & train the model → validate it → forecast & make decisions →
deep-dive analyses. The numbered folders are the stages; within a stage, filenames are prefixed by OEM.

For the *why* behind every method here (the SoH decision tree, the LOVO protocol, the assumptions and
data-quality findings), read [`../docs/OEM_MODELING_PLAYBOOK.md`](../docs/OEM_MODELING_PLAYBOOK.md). For
the operational extract runbook, see [`../docs/PROCEDURE.md`](../docs/PROCEDURE.md).

> **Note on outputs & data.** Cell outputs are stripped (lean, diff-friendly repo). The `data/` folder
> is gitignored, so these notebooks **need the local extracts / S3 access to run** — they document the
> method and code, not a runnable demo on a fresh clone. Re-extract with `src/import_cohort.py` first
> (see PROCEDURE.md). All notebooks auto-`chdir` to the repo root and add `src/` to the path in cell 1,
> so they work from any folder depth.
>
> **Kernel:** run with the **`euler-rul`** kernel (the repo `.venv`, which has xgboost/lightgbm/plotly).
> The default `python3` kernel is broken for this repo.

---

## The flow

### `00_extract/` — pull telemetry from S3
| Notebook | OEM | What it does |
|---|---|---|
| `euler_import_telemetry.ipynb` | Euler | S3-Select extract of the cohort VINs from the date-partitioned Parquet lake. |

> Mahindra/Bajaj extraction is **script-driven**, not a notebook: `python src/import_cohort.py <feed> <oem>`
> (and `src/import_euler_dense.py` / `src/import_bajaj_dense.py` for the dense cohorts).

### `01_soh_target/` — compute the SoH target (Playbook §3)
| Notebook | OEM | Method (tier) |
|---|---|---|
| `mahindra_soh_coulomb.ipynb` | Mahindra | **Coulomb counting** (Tier A) on the intellicar feed — the only source with `current`. |
| `mahindra_soh_distance_proxy.ipynb` | Mahindra | **Distance-per-SoC** proxy (Tier D) on the native feed (no current). |
| `mahindra_soh_method_compare.ipynb` | Mahindra | Cross-validates coulomb vs proxy on a vehicle present in both feeds (agreed ~3 pp). |
| `euler_soh_bms_capacity.ipynb` | Euler | **BMS remaining-capacity** SoH (Tier B), high-SoC band + isotonic decreasing fit. |

### `02_features_model/` — features & model training (Playbook §5–6)
| Notebook | OEM | What it does |
|---|---|---|
| `mahindra_features_baseline.ipynb` | Mahindra | Builds the per-(vin,month) feature table; baseline SoH-estimation + RUL. |
| `mahindra_degradation_model.ipynb` | Mahindra | Condition-aware monthly-fade model (the production rate model, `src/model.py`). |
| `mahindra_feature_analysis.ipynb` | Mahindra | Feature pruning + SHAP — which drivers the model actually uses. |
| `euler_forecast_model.ipynb` | Euler | Condition-aware SoH forecast with **P10–P90** bands (`src/euler_model.py`). |

### `03_validation/` — does the model generalize? (Playbook §4.3)
| Notebook | OEM | What it does |
|---|---|---|
| `mahindra_actual_vs_predicted_lovo.ipynb` | Mahindra | **Leave-one-vehicle-out** actual-vs-predicted backtest. |
| `mahindra_model_comparison.ipynb` | Mahindra | XGBoost vs Chronos-2 vs TimesFM-2.5 vs persistence, fair per-vehicle holdout. |

> Euler's LOVO backtest is script-driven: `python src/euler_backtest.py`.

### `04_forecast_decisions/` — turn forecasts into decisions (Playbook §6.3)
| Notebook | OEM | What it does |
|---|---|---|
| `mahindra_warranty_risk.ipynb` | Mahindra | Fleet warranty-risk table — projected SoH at warranty expiry, split by model/term. |
| `mahindra_degraded_grid.ipynb` | Mahindra | Oldest genuinely-degrading vehicles forecast to the 80% EoFL line. |
| `mahindra_forecast_report.ipynb` | Mahindra | Multi-page PDF: actual vs model vs √t to warranty, one page per vehicle. |
| `euler_resold_cohorts.ipynb` | Euler | Resale-date cohorts (≥3 well-observed vehicles), forecast to the in-house warranty window. |
| `predictions_june2023.ipynb` | Both | SoH predictions for the June-2023 registration cohort (Euler + Mahindra). |
| `bajaj_warranty_km_cycle_rul.ipynb` | Bajaj | km-warranty runway + cycle-based RUL. For Bajaj the **km limit binds** (high-mileage) — opposite of Mahindra. |
| `remaining_km_to_eol.ipynb` | All | **RUL in km** — remaining km = km/month × months-to-EoL (80/70/60% SoH). Mahindra n/a (sparse odometer). |

### `05_insights/` — the substantive findings (Playbook §7)
| Notebook | OEM | What it shows |
|---|---|---|
| `degradation_vs_km.ipynb` | Both | **Same km ≠ same degradation** (~9–15 pp spread); why Mahindra's odometer is unusable for this. |
| `euler_soh_vs_usage_proof.ipynb` | Euler | Lines up SoH drops against usage — steep drops = heavy use, flat = idle. |
| `euler_degraders.ipynb` | Euler | Well-observed degraders grouped by 100%-SoH start month. |
| `cycle_features_rejected.ipynb` | Bajaj+ | **Why charge cycles aren't a health signal** — and the age-control re-check that caught a +0.88 artifact. |

---

## `_archive/` — superseded / exploratory

Kept for reference, **not part of the canonical flow** (near-duplicates, single-vehicle explorations, or
niche variants): GPU pipeline, single-vehicle forecasts, the distance-per-SoC exploration, the Euler
2022 reported-SoH path, the June-2025 resale trio (superseded by `euler_resold_cohorts`), and the
smoothed / 2024 prediction variants.
