"""Tests for Diff-in-Diff analysis."""

import pytest
import numpy as np
import pandas as pd

from src.data.simulator import simulate_geo_experiment
from src.analysis.diff_in_diff import (
    _test_parallel_trends,
    _compute_event_study,
    run_diff_in_diff,
    DIDResult,
    ParallelTrendsResult,
)


@pytest.fixture
def geo_data():
    """Generate geo experiment data for DiD tests."""
    result = simulate_geo_experiment(n_markets=20, n_days=60, true_effect=0.15, seed=42)
    return result.data


@pytest.fixture
def geo_data_no_effect():
    result = simulate_geo_experiment(n_markets=20, n_days=60, true_effect=0.0, seed=42)
    return result.data


class TestDIDResult:
    def test_returns_did_result(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert isinstance(result, DIDResult)

    def test_has_att(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert isinstance(result.att, float)

    def test_has_confidence_interval(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert result.ci_lower < result.ci_upper

    def test_has_p_value(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert 0 <= result.p_value <= 1

    def test_parallel_trends_result(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert isinstance(result.parallel_trends, ParallelTrendsResult)
        assert 0 <= result.parallel_trends.p_value <= 1

    def test_event_study_exists(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert isinstance(result.event_study, pd.DataFrame)
        assert "relative_period" in result.event_study.columns
        assert "effect" in result.event_study.columns

    def test_positive_effect_detected(self, geo_data):
        """Large positive effect should give positive ATT."""
        result = run_diff_in_diff(geo_data)
        assert result.att > 0

    def test_zero_effect_small_att(self, geo_data_no_effect):
        """No treatment effect should give small ATT."""
        result = run_diff_in_diff(geo_data_no_effect)
        assert abs(result.att) < 50  # should be small relative to metric values

    def test_unit_counts(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.n_treated + result.n_control == geo_data.market_id.nunique()

    def test_period_counts(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert result.n_pre_periods > 0
        assert result.n_post_periods > 0

    def test_summary_string(self, geo_data):
        result = run_diff_in_diff(geo_data)
        summary = result.summary()
        assert "DIFFERENCE-IN-DIFFERENCES" in summary
        assert "ATT" in summary

    def test_elapsed_time(self, geo_data):
        result = run_diff_in_diff(geo_data)
        assert result.elapsed_seconds > 0
