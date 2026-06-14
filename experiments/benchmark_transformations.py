"""Benchmark GAF, STFT, and MTF against reference implementations on CPU.

Reference libraries:
  - GAF: pyts.image.GramianAngularField
  - MTF: pyts.image.MarkovTransitionField
  - STFT: scipy.signal.stft (per-sample loop)

Each method is timed over ``N_RUNS`` repetitions. Outputs:
  - boxplot PNG with reference mean time and custom-implementation distribution
  - CSV summary table (time, speedup, RMSE)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyts.image import GramianAngularField, MarkovTransitionField
from scipy import signal

from src.image_transformation.methods.gaf_transformation import GAF
from src.image_transformation.methods.mtf_transformation import MTF
from src.image_transformation.methods.stft_transformation import STFTSpectrogram


N_RUNS = 10
N_WARMUP = 2
DEFAULT_OUTPUT_DIR = Path("results/benchmarks")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    reference_label: str
    custom_label: str
    reference_fn: Callable[[], np.ndarray]
    custom_fn: Callable[[], np.ndarray]


def _rmse(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - actual) ** 2)))


def _time_runs(fn: Callable[[], object], *, n_warmup: int, n_runs: int) -> list[float]:
    for _ in range(n_warmup):
        fn()
    timings: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def _make_data(n_samples: int, n_timestamps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_timestamps))


def _reference_stft_scipy(data: np.ndarray, model: STFTSpectrogram) -> np.ndarray:
    if model.n_fft != model.window_size:
        raise ValueError(
            "SciPy STFT reference is only shape-compatible with this torch "
            "implementation when 'n_fft' equals 'window_size'."
        )
    if model.center:
        raise ValueError("SciPy STFT reference expects center=False.")
    if model.normalized:
        raise ValueError("SciPy STFT reference expects normalized=False.")

    if model.window_type == "hann":
        window = signal.windows.hann(model.window_size, sym=False)
    elif model.window_type == "hamming":
        window = signal.windows.hamming(model.window_size, sym=False)
    else:
        sigma = model.sigma if model.sigma is not None else model.window_size / 6.0
        n = np.arange(model.window_size, dtype=np.float64)
        center = (model.window_size - 1) / 2.0
        window = np.exp(-0.5 * ((n - center) / sigma) ** 2)

    noverlap = model.window_size - model.hop_length
    outputs: list[np.ndarray] = []
    for row in data:
        _, _, stft = signal.stft(
            row,
            nperseg=model.window_size,
            noverlap=noverlap,
            nfft=model.n_fft,
            window=window,
            boundary=None,
            padded=False,
            return_onesided=True,
            scaling="spectrum",
        )
        # scipy.signal.stft with scaling="spectrum" divides by sum(window).
        # Undo that normalization to match torch.stft's unnormalized magnitude.
        outputs.append((np.abs(stft) * np.sum(window)) ** model.power)
    return np.stack(outputs)


def build_cases(
    data: np.ndarray,
    *,
    gaf_params: dict,
    mtf_params: dict,
    stft_params: dict,
) -> list[BenchmarkCase]:
    data_tensor = torch.tensor(data, dtype=torch.float64)

    gaf_pyts = GramianAngularField(
        method="s" if gaf_params["method"] == "summation" else "d",
        overlapping=gaf_params["overlapping"],
        image_size=gaf_params["image_size"],
    )
    gaf_torch = GAF(gaf_params)

    mtf_pyts = MarkovTransitionField(
        image_size=mtf_params["image_size"],
        n_bins=mtf_params["n_bins"],
        strategy=mtf_params["strategy"],
        overlapping=mtf_params["overlapping"],
        flatten=mtf_params["flatten"],
    )
    mtf_torch = MTF(mtf_params)

    stft_model = STFTSpectrogram(stft_params)

    def gaf_reference() -> np.ndarray:
        return gaf_pyts.fit_transform(data)

    def gaf_custom() -> np.ndarray:
        return gaf_torch.transform(data_tensor).detach().cpu().numpy()

    def mtf_reference() -> np.ndarray:
        return mtf_pyts.fit_transform(data)

    def mtf_custom() -> np.ndarray:
        return mtf_torch.transform(data_tensor).detach().cpu().numpy()

    def stft_reference() -> np.ndarray:
        return _reference_stft_scipy(data, stft_model)

    def stft_custom() -> np.ndarray:
        out = stft_model.transform(data_tensor)
        return out.detach().cpu().numpy()

    return [
        BenchmarkCase("GAF", "pyts", "torch (ours)", gaf_reference, gaf_custom),
        BenchmarkCase("STFT", "scipy", "torch (ours)", stft_reference, stft_custom),
        BenchmarkCase("MTF", "pyts", "torch (ours)", mtf_reference, mtf_custom),
    ]


def run_benchmark(
    cases: list[BenchmarkCase],
    *,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    timing_rows: list[dict] = []

    for case in cases:
        ref_out = case.reference_fn()
        custom_out = case.custom_fn()
        rmse = _rmse(ref_out, custom_out)

        ref_times = _time_runs(case.reference_fn, n_warmup=n_warmup, n_runs=n_runs)
        custom_times = _time_runs(case.custom_fn, n_warmup=n_warmup, n_runs=n_runs)

        ref_mean = float(np.mean(ref_times))
        custom_mean = float(np.mean(custom_times))
        speedup = ref_mean / custom_mean if custom_mean > 0 else np.nan

        rows.append(
            {
                "method": case.name,
                "reference": case.reference_label,
                "implementation": case.custom_label,
                "ref_time_s_mean": ref_mean,
                "ref_time_s_std": float(np.std(ref_times, ddof=1)),
                "custom_time_s_mean": custom_mean,
                "custom_time_s_std": float(np.std(custom_times, ddof=1)),
                "speedup": speedup,
                "rmse": rmse,
            }
        )

        for run_idx, elapsed in enumerate(ref_times, start=1):
            timing_rows.append(
                {
                    "method": case.name,
                    "implementation": case.reference_label,
                    "run": run_idx,
                    "time_s": elapsed,
                }
            )
        for run_idx, elapsed in enumerate(custom_times, start=1):
            timing_rows.append(
                {
                    "method": case.name,
                    "implementation": case.custom_label,
                    "run": run_idx,
                    "time_s": elapsed,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(timing_rows)


def plot_benchmark(summary: pd.DataFrame, timing: pd.DataFrame, output_path: Path) -> None:
    methods = summary["method"].tolist()
    n_methods = len(methods)
    positions = np.arange(1, n_methods + 1, dtype=float)
    box_width = 0.3
    ref_offset = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))

    custom_color = "#4C72B0"
    ref_color = "#C44E52"
    all_positive_times: list[float] = []

    for method in methods:
        x = positions[methods.index(method)]
        custom_label = summary.loc[summary["method"] == method, "implementation"].iloc[0]
        custom_times = timing.loc[
            (timing["method"] == method) & (timing["implementation"] == custom_label),
            "time_s",
        ].to_numpy()
        ref_mean = float(summary.loc[summary["method"] == method, "ref_time_s_mean"].iloc[0])

        all_positive_times.extend(custom_times.tolist())
        all_positive_times.append(ref_mean)

        ax.boxplot(
            [custom_times],
            positions=[x - ref_offset / 2],
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            medianprops={"color": "black", "linewidth": 1.2},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": custom_color,
                "markersize": 6,
            },
            boxprops={"facecolor": custom_color, "alpha": 0.35, "edgecolor": custom_color},
            whiskerprops={"color": custom_color},
            capprops={"color": custom_color},
            flierprops={"marker": "o", "markerfacecolor": custom_color, "alpha": 0.5, "markersize": 4},
        )

        ax.scatter(
            [x + ref_offset / 2],
            [ref_mean],
            color=ref_color,
            s=70,
            zorder=5,
            marker="s",
            clip_on=False,
        )

    y_min = min(all_positive_times) * 0.7
    y_max = max(all_positive_times) * 1.4
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)

    custom_handle = plt.Line2D(
        [0],
        [0],
        color=custom_color,
        lw=8,
        alpha=0.35,
        label="torch implementation (10 runs)",
    )
    ref_handle = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=ref_color,
        markersize=8,
        label="reference mean time",
    )
    ax.legend(handles=[custom_handle, ref_handle], loc="upper left")

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title("Transformation benchmark on CPU (10 runs per method)")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-timestamps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    data = _make_data(args.n_samples, args.n_timestamps, args.seed)
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

    summary, timing = run_benchmark(cases, n_warmup=args.n_warmup, n_runs=args.n_runs)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "transformation_benchmark_summary.csv"
    timing_path = output_dir / "transformation_benchmark_timings.csv"
    plot_path = output_dir / "transformation_benchmark_boxplot.png"

    summary.to_csv(summary_path, index=False)
    timing.to_csv(timing_path, index=False)
    plot_benchmark(summary, timing, plot_path)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print()
    print(f"Saved summary: {summary_path}")
    print(f"Saved timings: {timing_path}")
    print(f"Saved plot:    {plot_path}")


if __name__ == "__main__":
    main()
