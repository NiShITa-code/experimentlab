"""Tests for Power analysis."""

import pytest
import numpy as np

from src.design.power import (
    compute_sample_size_binary,
    compute_sample_size_continuous,
    compute_mde,
    run_power_analysis,
    PowerResult,
)


class TestBinarySampleSize:
    def test_returns_positive_int(self):
        n = compute_sample_size_binary(0.10, 0.02)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_effect_needs_fewer_samples(self):
        n_small = compute_sample_size_binary(0.10, 0.01)
        n_large = compute_sample_size_binary(0.10, 0.05)
        assert n_large < n_small

    def test_higher_power_needs_more_samples(self):
        n_80 = compute_sample_size_binary(0.10, 0.02, power=0.80)
        n_90 = compute_sample_size_binary(0.10, 0.02, power=0.90)
        assert n_90 > n_80

    def test_lower_alpha_needs_more_samples(self):
        n_05 = compute_sample_size_binary(0.10, 0.02, alpha=0.05)
        n_01 = compute_sample_size_binary(0.10, 0.02, alpha=0.01)
        assert n_01 > n_05

    def test_zero_mde_returns_large(self):
        n = compute_sample_size_binary(0.10, 0.0)
        assert n == 999999


class TestContinuousSampleSize:
    def test_returns_positive_int(self):
        n = compute_sample_size_continuous(50, 10, 2.0)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_effect_needs_more_samples(self):
        n_big = compute_sample_size_continuous(50, 10, 5.0)
        n_small = compute_sample_size_continuous(50, 10, 1.0)
        assert n_small > n_big

    def test_higher_variance_needs_more_samples(self):
        n_low = compute_sample_size_continuous(50, 5, 2.0)
        n_high = compute_sample_size_continuous(50, 20, 2.0)
        assert n_high > n_low


class TestMDE:
    def test_binary_mde_positive(self):
        mde = compute_mde(5000, baseline_rate=0.10, metric_type="binary")
        assert mde > 0

    def test_larger_sample_smaller_mde(self):
        mde_small = compute_mde(1000, baseline_rate=0.10, metric_type="binary")
        mde_large = compute_mde(10000, baseline_rate=0.10, metric_type="binary")
        assert mde_large < mde_small

    def test_continuous_mde(self):
        mde = compute_mde(5000, baseline_rate=50, baseline_std=10, metric_type="continuous")
        assert mde > 0


class TestRunPowerAnalysis:
    def test_returns_power_result(self):
        result = run_power_analysis(baseline_rate=0.10, mde=0.02)
        assert isinstance(result, PowerResult)

    def test_duration_estimate(self):
        result = run_power_analysis(baseline_rate=0.10, mde=0.02, daily_traffic=1000)
        assert result.estimated_days > 0
        assert result.estimated_weeks > 0

    def test_total_is_double_per_arm(self):
        result = run_power_analysis(baseline_rate=0.10, mde=0.02)
        assert result.total_sample == result.sample_per_arm * 2

    def test_summary_string(self):
        result = run_power_analysis(baseline_rate=0.10, mde=0.02)
        summary = result.summary()
        assert "POWER ANALYSIS" in summary
