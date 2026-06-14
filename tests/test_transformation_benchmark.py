"""Smoke tests for the transformation benchmark (correctness thresholds)."""

from __future__ import annotations

import pytest

from experiments.benchmark_transformations import (
    N_RUNS,
    _make_data,
    build_cases,
    run_benchmark,
)


@pytest.fixture
def benchmark_summary():
    data = _make_data(n_samples=24, n_timestamps=61, seed=42)
    cases = build_cases(
        data,
        gaf_params={"method": "summation", "overlapping": True, "image_size": 0.7},
        mtf_params={
            "image_size": 0.7,
            "n_bins": 8,
            "strategy": "quantile",
            "overlapping": True,
            "flatten": False,
        },
        stft_params={
            "window_size": 16,
            "hop_length": 4,
            "n_fft": 16,
            "window_type": "hann",
            "center": False,
            "power": 2.0,
            "normalized": False,
        },
    )
    summary, _ = run_benchmark(cases, n_warmup=1, n_runs=2)
    return summary


def test_transformation_benchmark_rmse(benchmark_summary) -> None:
    for _, row in benchmark_summary.iterrows():
        assert row["rmse"] < 1e-6, f"{row['method']} RMSE too high: {row['rmse']}"


def test_transformation_benchmark_speedup_positive(benchmark_summary) -> None:
    for _, row in benchmark_summary.iterrows():
        assert row["speedup"] > 1.0, f"{row['method']} expected speedup > 1, got {row['speedup']}"


def test_transformation_benchmark_default_runs() -> None:
    assert N_RUNS == 10
