"""Unit tests for the shared pipeline logic — runnable locally, no AWS required.

The SageMaker SDK graph (pipeline_factory.get_pipeline) needs credentials/role, so it is not built here;
CI validates it separately via `get-pipeline-definition`. These tests cover the science-bearing common code.
"""
import numpy as np
import pandas as pd
import pytest

from pipelines.common import config, soh, features, forecaster, backtest_lib


def test_registry_covers_five_oems():
    assert set(config.all_oems()) == {"euler", "mahindra", "bajaj", "piaggio", "montra"}


def test_only_euler_is_gated():
    gated = [o for o in config.all_oems() if config.get(o).has_gate]
    assert gated == ["euler"]


def test_soh_methods_registered():
    assert set(soh.METHODS) == {"coulomb", "bms_capacity", "reported"}


def _synthetic_bms(vin="P60X", months=12, start_cap=100.0, fade=0.5):
    rows = []
    for k in range(months):
        cap = start_cap - fade * k + np.random.normal(0, 0.3)
        for _ in range(8):
            rows.append(dict(vin=vin, t=pd.Timestamp("2024-01-01") + pd.DateOffset(months=k),
                             soc=98.0, resCapacity=cap * 0.98, age_months=k))
    return pd.DataFrame(rows)


def test_bms_capacity_soh_is_non_increasing():
    df = _synthetic_bms()
    out = soh.bms_capacity(df)
    s = out.sort_values("month")["soh"].to_numpy()
    assert np.all(np.diff(s) <= 1e-6)          # isotonic envelope never rises
    assert s.max() <= 100.0 + 1e-9


def test_reported_soh_drops_garbage_and_is_monotone():
    df = pd.DataFrame({
        "vin": ["b"] * 6,
        "t": pd.date_range("2025-01-01", periods=6, freq="MS"),
        "batterySoh": [99, 0.0, 98, 97, 70000, 96],   # 0.0 and 70000 are garbage
        "age_months": range(6),
    })
    out = soh.reported(df)
    assert out["soh"].between(30, 100).all()
    assert np.all(np.diff(out.sort_values("month")["soh"]) <= 1e-6)


def test_assemble_emits_full_schema():
    df = _synthetic_bms()
    soh_df = soh.bms_capacity(df)
    feat = features.electrical_features(df)
    m = features.assemble(soh_df, feat)
    assert list(m.columns) == features.SCHEMA


def test_forecaster_simulates_a_band():
    df = _synthetic_bms(months=10)
    m = features.assemble(soh.bms_capacity(df), features.electrical_features(df))
    model = forecaster.fit(m)
    sim = model.simulate(m.iloc[:6], horizon=4)
    assert list(sim.columns) == ["h", "q10", "q50", "q90"]
    assert len(sim) == 4
    assert (sim["q10"] <= sim["q90"]).all()


def test_backtest_reports_metrics():
    frames = []
    for i in range(6):
        d = _synthetic_bms(vin=f"v{i}", months=10, fade=0.4 + 0.1 * i)
        frames.append(features.assemble(soh.bms_capacity(d), features.electrical_features(d)))
    m = pd.concat(frames, ignore_index=True)
    model = forecaster.fit(m)
    metrics = backtest_lib.evaluate(model, m, [f"v{i}" for i in range(6)])
    assert "overall_rmse" in metrics and metrics["n_forecasts"] > 0
