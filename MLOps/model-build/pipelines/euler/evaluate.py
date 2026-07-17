"""Euler evaluation entry point (Processing step) — held-out backtest -> evaluation.json.

Delegates to common.backtest_lib.run. For Euler the canonical version is a LOVO backtest with a
recalibrated P10/P90 band (src/euler_train.py); this shared runner is the pipeline-facing equivalent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pipelines/
from common import backtest_lib

if __name__ == "__main__":
    backtest_lib.run("euler")
