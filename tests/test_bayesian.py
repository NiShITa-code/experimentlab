"""Tests for Bayesian A/B testing."""

import pytest
import numpy as np

from src.analysis.bayesian import run_bayesian_ab, BayesianABResult


@pytest.fixture
def binary_data_positive():
    """Treatment is clearly better."""
    np.random.seed(42)
    control = np.random.binomial(1, 0.10, 5000)
    treatment = np.random.binomial(1, 0.15, 5000)
    return control, treatment


@pytest.fixture
def binary_data_equal():
    """A/A test — no difference."""
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


class TestBayesianBinary:
    def test_returns_result(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert isinstance(result, BayesianABResult)

    def test_prob_better_high_with_real_effect(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert result.prob_treatment_better > 0.90

    def test_recommends_treatment_with_effect(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert result.recommendation == "treatment"

    def test_lift_positive_with_effect(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert result.lift > 0

    def test_aa_test_inconclusive(self, binary_data_equal):
        c, t = binary_data_equal
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert result.prob_treatment_better < 0.95
        assert result.prob_treatment_better > 0.05

    def test_posterior_samples_exist(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert len(result.control_posterior_samples) > 0
        assert len(result.treatment_posterior_samples) > 0
        assert len(result.lift_samples) > 0

    def test_expected_loss_correct_direction(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        # If treatment is better, loss from picking treatment should be small
        assert result.expected_loss_treatment < result.expected_loss_control

    def test_hdi_contains_lift(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        assert result.lift_ci_lower <= result.lift <= result.lift_ci_upper


class TestBayesianContinuous:
    def test_continuous_metric(self, continuous_data):
        c, t = continuous_data
        result = run_bayesian_ab(c, t, metric_type="continuous")
        assert result.metric_type == "continuous"
        assert result.prob_treatment_better > 0.90

    def test_continuous_lift_positive(self, continuous_data):
        c, t = continuous_data
        result = run_bayesian_ab(c, t, metric_type="continuous")
        assert result.lift > 0

    def test_summary_string(self, binary_data_positive):
        c, t = binary_data_positive
        result = run_bayesian_ab(c, t, metric_type="binary")
        summary = result.summary()
        assert "BAYESIAN A/B TEST" in summary
        assert "P(treatment > control)" in summary
