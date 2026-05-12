from statistical.stat_features import (mean_torch, median_torch, 
std_torch, max_torch, min_torch, q5_torch, q25_torch, q75_torch, 
q95_torch, skewness_torch, kurtosis_torch, n_peaks_torch, slope_torch, 
ben_corr_torch, interquantile_range_torch, energy_torch, 
zero_crossing_rate_torch, autocorrelation_torch, shannon_entropy_torch, 
ptp_amp_torch, mean_ptp_distance_torch, crest_factor_torch, mean_ema_torch, 
mean_moving_median_torch, hjorth_mobility_torch, hjorth_complexity_torch, 
hurst_exponent_torch, pfd_torch)
from enum import Enum


class FeatureConstant(Enum):
    STAT_METHODS_TORCH = {
        'mean_': mean_torch,
        'median_': median_torch,
        'std_': std_torch,
        'max_': max_torch,
        'min_': min_torch,
        'q5_': q5_torch,
        'q25_': q25_torch,
        'q75_': q75_torch,
        'q95_': q95_torch
    }

    STAT_METHODS_GLOBAL_TORCH = {
        'skewness_': skewness_torch,
        'kurtosis_': kurtosis_torch,
        'n_peaks_': n_peaks_torch,
        'slope_': slope_torch,
        'ben_corr_': ben_corr_torch,
        'interquartile_range_': interquantile_range_torch,
        'energy_': energy_torch,
        'cross_rate_': zero_crossing_rate_torch,
        'autocorrelation_': autocorrelation_torch,
        'shannon_entropy_': shannon_entropy_torch,
        'ptp_amplitude_': ptp_amp_torch,
        'mean_ptp_distance_': mean_ptp_distance_torch,
        'crest_factor_': crest_factor_torch,
        'mean_ema_': mean_ema_torch,
        'mean_moving_median_': mean_moving_median_torch,
        'hjorth_mobility_': hjorth_mobility_torch,
        'hjorth_complexity_': hjorth_complexity_torch,
        'hurst_exponent_': hurst_exponent_torch,
        'petrosian_fractal_dimension_': pfd_torch
    }
