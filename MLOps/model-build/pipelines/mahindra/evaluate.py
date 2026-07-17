"""Mahindra evaluation entry point (Processing step) — held-out backtest -> evaluation.json.

Delegates to common.backtest_lib.run. Canonical per-OEM backtest lives in src/oem_train.py::backtest
(overall + degrading-cohort RMSE vs a persistence baseline).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pipelines/
from common import backtest_lib

if __name__ == "__main__":
    backtest_lib.run("mahindra")
