#!/usr/bin/env bash
# Quarterly Euler retrain: refresh the cohort's latest telemetry, rebuild features, then
# retrain + recalibrate + version the model (src/euler_train.py). Run from anywhere.
#
# Install as a quarterly cron job (1st of Jan/Apr/Jul/Oct, 03:00):
#   0 3 1 1,4,7,10 *  /home/hj/Desktop/EULER_RUL_MODEL/scripts/retrain_euler.sh >> /tmp/euler_retrain.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "==================== Euler retrain $(date) ===================="

# 1. Refresh the latest telemetry for the CURRENT cohort (captures new months as vehicles age
#    toward 80%). Skips gracefully if S3/creds are unavailable — then we retrain on existing data.
if [ -f data/euler/features/feature_table.parquet ]; then
  $PY -c "import pandas as pd; pd.read_parquet('data/euler/features/feature_table.parquet')['vin'].drop_duplicates().to_csv('/tmp/euler_cohort.txt', index=False, header=False)"
  $PY src/import_euler_batch.py /tmp/euler_cohort.txt || echo "WARN: S3 refresh failed — retraining on existing data"
fi

# 2. Rebuild the feature table from the (refreshed) dense parquets.
$PY src/euler_features.py || { echo "ERROR: feature build failed"; exit 1; }

# 3. Retrain + LOVO backtest + recalibrate bands + version into the model registry.
$PY src/euler_train.py || { echo "ERROR: training failed"; exit 1; }

echo "==================== done $(date) ===================="
