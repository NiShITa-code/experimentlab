"""Tests for Synthetic Control Method."""

import pytest
import numpy as np
import pandas as pd

from src.data.simulator import simulate_geo_experiment
from src.analysis.synthetic_control import (
    _find_weights,
    run_synthetic_control,
    SyntheticControlResult,
)


@pytest.fixture
def geo_data():
    result = simulate_geo_experiment(n_markets=15, n_days=60, true_effect=0.15, seed=42)
    return result


class TestFindWeights:
    def test_weights_sum_to_one(self):
        treated_pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        donor_pre = np.column_stack([
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [0.9, 1.9, 2.9, 3.9, 4.9],
            [1.5, 2.5, 3.5, 4.5, 5.5],
        ])
        w = _find_weights(treated_pre, donor_pre)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        treated_pre = np.array([1.0, 2.0, 3.0])
        donor_pre = np.column_stack([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ])
        w = _find_weights(treated_pre, donor_pre)
        assert (w >= -1e-8).all()

    def test_perfect_donor_gets_full_weight(self):
        treated_pre = np.array([10.0, 20.0, 30.0])
        donor_pre = np.column_stack([
            [10.0, 20.0, 30.0],   # perfect match
            [100.0, 200.0, 300.0],  # very different
        ])
        w = _find_weights(treated_pre, donor_pre)
        assert w[0] > 0.9  # most weight on perfect donor


class TestSyntheticControl:
    def test_returns_result(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        assert isinstance(result, SyntheticControlResult)

    def test_has_donor_weights(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        assert len(result.donor_weights) > 0
        assert all(v > 0 for v in result.donor_weights.values())

    def test_has_time_series(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        assert len(result.actual) == len(result.synthetic)
        assert len(result.gap) == len(result.actual)

    def test_pre_rmspe_reasonable(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        assert result.pre_rmspe >= 0

    def test_effect_is_numeric(self, geo_data):
        """Estimated effect should be a valid number."""
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        assert isinstance(result.estimated_effect, float)
        assert not np.isnan(result.estimated_effect)

    def test_summary_string(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=False,
        )
        summary = result.summary()
        assert "SYNTHETIC CONTROL" in summary

    def test_placebo_tests(self, geo_data):
        df = geo_data.data
        treatment_markets = geo_data.config["treatment_markets"]
        treatment_start = df[df.post_treatment == 1].date.min()
        result = run_synthetic_control(
            df, treatment_unit=treatment_markets[0],
            treatment_start=str(treatment_start),
            run_placebo=True,
        )
        assert result.placebo_effects is not None
        assert result.placebo_p_value is not None
        assert 0 <= result.placebo_p_value <= 1
