"""Piaggio preprocessing entry point (Processing step) -> the piaggio_featengg table.

Delegates to common.preprocess_lib; the SoH method (coulomb (via intellicar) -> isotonic) and all params come from the OEM
registry. Packaging: the processor uploads pipelines/ as source_dir so `common/` is importable here
(see common/pipeline_factory.py and model-build/README.md).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pipelines/
from common import preprocess_lib

if __name__ == "__main__":
    preprocess_lib.cli("piaggio")
