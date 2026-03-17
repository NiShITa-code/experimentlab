"""Tests for Metric definition and validation framework."""

import pytest
import numpy as np
import pandas as pd

from src.metrics.builder import (
    validate_metric,
    validate_metric_suite,
    MetricValidation,
    MetricSuite,
)
from src.data.schema import MetricDefinition, MetricType


@pytest.fixture
def ab_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "converted": np.random.binomial(1, 0.10, n),
        "revenue": np.random.exponential(10, n),
        "sessions": np.random.poisson(5, n),
    })
    return df


@pytest.fixture
def binary_metric():
    return MetricDefinition(
        name="conversion_rate",
        metric_type=MetricType.BINARY,
        numerator_col="converted",
        denominator_col=None,
    )


@pytest.fixture
def continuous_metric():
    return MetricDefinition(
        name="avg_revenue",
        metric_type=MetricType.CONTINUOUS,
        numerator_col="revenue",
        denominator_col=None,
    )


class TestValidateMetric:
    def test_returns_validation(self, ab_data, binary_metric):
        result = validate_metric(ab_data, binary_metric)
        assert isinstance(result, MetricValidation)

    def test_valid_metric_passes(self, ab_data, binary_metric):
        result = validate_metric(ab_data, binary_metric)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_column_fails(self, ab_data):
        bad_metric = MetricDefinition(
            name="missing", metric_type=MetricType.BINARY,
            numerator_col="nonexistent", denominator_col=None,
        )
        result = validate_metric(ab_data, bad_metric)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_observed_stats(self, ab_data, binary_metric):
        result = validate_metric(ab_data, binary_metric)
        assert result.observed_mean is not None
        assert result.observed_std is not None
        assert result.n_observations == len(ab_data)

    def test_continuous_metric(self, ab_data, continuous_metric):
        result = validate_metric(ab_data, continuous_metric)
        assert result.valid is True
        assert result.observed_mean > 0

    def test_missing_data_warning(self):
        df = pd.DataFrame({
            "x": [1, 2, None, None, None, None, None, None, None, None],
        })
        metric = MetricDefinition(
            name="test", metric_type=MetricType.CONTINUOUS,
            numerator_col="x", denominator_col=None,
        )
        result = validate_metric(df, metric)
        assert len(result.warnings) > 0

    def test_all_null_fails(self):
        df = pd.DataFrame({"x": [None, None, None]})
        metric = MetricDefinition(
            name="test", metric_type=MetricType.CONTINUOUS,
            numerator_col="x", denominator_col=None,
        )
        result = validate_metric(df, metric)
        assert result.valid is False


class TestValidateMetricSuite:
    def test_returns_suite(self, ab_data, binary_metric, continuous_metric):
        result = validate_metric_suite(
            ab_data, primary=binary_metric, guardrails=[continuous_metric],
        )
        assert isinstance(result, MetricSuite)

    def test_all_valid(self, ab_data, binary_metric, continuous_metric):
        result = validate_metric_suite(
            ab_data, primary=binary_metric, guardrails=[continuous_metric],
        )
        assert result.all_valid is True

    def test_invalid_guardrail_flags(self, ab_data, binary_metric):
        bad_guardrail = MetricDefinition(
            name="missing", metric_type=MetricType.BINARY,
            numerator_col="missing_col", denominator_col=None,
        )
        result = validate_metric_suite(
            ab_data, primary=binary_metric, guardrails=[bad_guardrail],
        )
        assert result.all_valid is False

    def test_summary_text(self, ab_data, binary_metric, continuous_metric):
        result = validate_metric_suite(
            ab_data, primary=binary_metric, guardrails=[continuous_metric],
        )
        assert "METRIC SUITE VALIDATION" in result.summary_text
