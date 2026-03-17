"""Tests for Privacy-preserving analysis."""

import pytest
import numpy as np
import pandas as pd

from src.privacy.aggregation import (
    check_k_anonymity,
    add_dp_noise,
    private_mean,
    run_privacy_audit,
    PrivacyReport,
)


@pytest.fixture
def grouped_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "group": np.random.choice(["A", "B", "C"], n),
        "segment": np.random.choice(["new", "returning"], n),
        "metric": np.random.normal(100, 20, n),
    })
    return df


@pytest.fixture
def small_group_data():
    """Data with some very small groups."""
    data = {
        "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 3,
        "segment": ["x"] * 100 + ["x"] * 100 + ["x"] * 3,
        "metric": np.random.normal(100, 20, 203),
    }
    return pd.DataFrame(data)


class TestKAnonymity:
    def test_large_groups_pass(self, grouped_data):
        result = check_k_anonymity(grouped_data, ["group"], k=5)
        assert result["passed"] is True

    def test_small_groups_fail(self, small_group_data):
        result = check_k_anonymity(small_group_data, ["group"], k=5)
        assert result["passed"] is False

    def test_returns_min_size(self, grouped_data):
        result = check_k_anonymity(grouped_data, ["group"], k=5)
        assert result["min_size"] > 0

    def test_violations_listed(self, small_group_data):
        result = check_k_anonymity(small_group_data, ["group"], k=5)
        assert result["groups_below_k"] > 0
        assert len(result["violations"]) > 0


class TestDPNoise:
    def test_noise_changes_value(self):
        np.random.seed(42)
        noised = add_dp_noise(100.0, sensitivity=1.0, epsilon=1.0)
        assert noised != 100.0

    def test_lower_epsilon_more_noise(self):
        """Lower epsilon = more privacy = more noise (on average)."""
        np.random.seed(42)
        diffs_high_eps = []
        diffs_low_eps = []
        for _ in range(1000):
            diffs_high_eps.append(abs(add_dp_noise(100, 1.0, epsilon=10.0) - 100))
            diffs_low_eps.append(abs(add_dp_noise(100, 1.0, epsilon=0.1) - 100))
        assert np.mean(diffs_low_eps) > np.mean(diffs_high_eps)

    def test_gaussian_mechanism(self):
        np.random.seed(42)
        noised = add_dp_noise(100.0, sensitivity=1.0, epsilon=1.0, mechanism="gaussian")
        assert isinstance(noised, float)

    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError):
            add_dp_noise(100.0, sensitivity=1.0, mechanism="invalid")


class TestPrivateMean:
    def test_returns_float(self):
        np.random.seed(42)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = private_mean(values, 0, 10, epsilon=1.0)
        assert isinstance(result, float)

    def test_close_to_true_mean_with_high_epsilon(self):
        np.random.seed(42)
        values = np.random.normal(50, 1, 10000)
        result = private_mean(values, 0, 100, epsilon=100.0)
        assert abs(result - 50) < 1  # should be very close


class TestPrivacyAudit:
    def test_returns_report(self, grouped_data):
        result = run_privacy_audit(
            grouped_data, metric_col="metric",
            group_cols=["group"], k=5, epsilon=1.0,
        )
        assert isinstance(result, PrivacyReport)

    def test_report_fields(self, grouped_data):
        result = run_privacy_audit(
            grouped_data, metric_col="metric",
            group_cols=["group"], k=5, epsilon=1.0,
        )
        assert result.raw_mean is not None
        assert result.noisy_mean is not None
        assert result.dp_applied is True

    def test_no_dp(self, grouped_data):
        result = run_privacy_audit(
            grouped_data, metric_col="metric",
            group_cols=["group"], apply_dp=False,
        )
        assert result.dp_applied is False
        assert result.noisy_mean is None

    def test_summary_string(self, grouped_data):
        result = run_privacy_audit(
            grouped_data, metric_col="metric",
            group_cols=["group"],
        )
        summary = result.summary()
        assert "PRIVACY REPORT" in summary
