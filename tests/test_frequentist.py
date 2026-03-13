"""Tests for Frequentist A/B testing and SRM."""

import pytest
import numpy as np

from src.analysis.frequentist import (
    run_frequentist_ab,
    _apply_cuped,
    check_srm,
    FrequentistABResult,
)


@pytest.fixture
def binary_significant():
    np.random.seed(42)
    control = np.random.binomial(1, 0.10, 5000)
    treatment = np.random.binomial(1, 0.15, 5000)
    return control, treatment


@pytest.fixture
def binary_null():
    np.random.seed(42)
    control = np.random.binomial(1, 0.10, 5000)
    treatment = np.random.binomial(1, 0.10, 5000)
    return control, treatment


@pytest.fixture
def continuous_data():
    np.random.seed(42)
    control = np.random.normal(50, 10, 2000)
    treatment = np.random.normal(55, 10, 2000)
    return control, treatment


class TestFrequentistBinary:
    def test_returns_result(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        assert isinstance(result, FrequentistABResult)

    def test_significant_with_real_effect(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        assert result.significant == True
        assert result.p_value < 0.05

    def test_not_significant_under_null(self, binary_null):
        c, t = binary_null
        result = run_frequentist_ab(c, t, metric_type="binary")
        # Under null, may or may not be significant, but p should be large
        assert result.p_value > 0.01

    def test_test_name_binary(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        assert result.test_name == "two-proportion z-test"

    def test_positive_lift(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        assert result.absolute_lift > 0
        assert result.relative_lift > 0

    def test_ci_contains_lift(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        assert result.ci_lower <= result.absolute_lift <= result.ci_upper


class TestFrequentistContinuous:
    def test_welchs_t_test(self, continuous_data):
        c, t = continuous_data
        result = run_frequentist_ab(c, t, metric_type="continuous")
        assert result.test_name == "Welch's t-test"
        assert result.significant == True

    def test_continuous_lift(self, continuous_data):
        c, t = continuous_data
        result = run_frequentist_ab(c, t, metric_type="continuous")
        assert result.absolute_lift > 0


class TestCUPED:
    def test_cuped_reduces_variance(self):
        np.random.seed(42)
        n = 3000
        # Correlated pre/post
        pre = np.random.normal(0, 1, n * 2)
        post = 0.7 * pre + np.random.normal(0, 0.5, n * 2)
        post[n:] += 0.1  # treatment effect

        c_adjusted, t_adjusted, reduction = _apply_cuped(
            post[:n], post[n:], pre[:n], pre[n:]
        )
        assert reduction > 0  # should reduce variance

    def test_cuped_integrated(self):
        np.random.seed(42)
        n = 3000
        pre = np.random.normal(0, 1, n * 2)
        post = 0.7 * pre + np.random.normal(0, 0.5, n * 2)
        post[n:] += 0.3

        result = run_frequentist_ab(
            post[:n], post[n:], metric_type="continuous",
            pre_experiment_control=pre[:n], pre_experiment_treatment=pre[n:],
        )
        assert result.cuped_applied is True
        assert result.variance_reduction > 0


class TestSRM:
    def test_balanced_passes(self):
        result = check_srm(5000, 5000)
        assert result["passed"] == True

    def test_severely_imbalanced_fails(self):
        result = check_srm(5000, 4000)
        assert result["passed"] == False

    def test_has_p_value(self):
        result = check_srm(5000, 5000)
        assert 0 <= result["p_value"] <= 1

    def test_returns_expected_fields(self):
        result = check_srm(5000, 5100)
        assert "chi_squared" in result
        assert "actual_ratio" in result
        assert "message" in result

    def test_summary_string(self, binary_significant):
        c, t = binary_significant
        result = run_frequentist_ab(c, t, metric_type="binary")
        summary = result.summary()
        assert "FREQUENTIST A/B TEST" in summary
