"""Tests for experiment data simulators."""

import pytest
import numpy as np
import pandas as pd

from src.data.simulator import (
    simulate_ab_test,
    simulate_geo_experiment,
    simulate_feedback_data,
)
from src.data.schema import ExperimentType


class TestABSimulator:
    def test_returns_simulation_result(self):
        result = simulate_ab_test(n_users=500)
        assert result.experiment_type == ExperimentType.AB_TEST

    def test_correct_user_count(self):
        result = simulate_ab_test(n_users=1000)
        assert len(result.data) == 1000

    def test_two_groups(self):
        result = simulate_ab_test(n_users=1000)
        assert set(result.data.group.unique()) == {0, 1}

    def test_conversion_is_binary(self):
        result = simulate_ab_test(n_users=1000)
        assert set(result.data.converted.unique()).issubset({0, 1})

    def test_revenue_non_negative(self):
        result = simulate_ab_test(n_users=1000)
        assert (result.data.revenue >= 0).all()

    def test_treatment_effect_direction(self):
        """Positive effect should increase treatment conversion rate."""
        result = simulate_ab_test(n_users=10000, true_effect=0.10, seed=42)
        df = result.data
        c_rate = df[df.group == 0].converted.mean()
        t_rate = df[df.group == 1].converted.mean()
        assert t_rate > c_rate

    def test_zero_effect_similar_rates(self):
        """A/A test should have similar rates."""
        result = simulate_ab_test(n_users=10000, true_effect=0.0, seed=42)
        df = result.data
        c_rate = df[df.group == 0].converted.mean()
        t_rate = df[df.group == 1].converted.mean()
        assert abs(t_rate - c_rate) < 0.03

    def test_reproducibility(self):
        r1 = simulate_ab_test(n_users=500, seed=7)
        r2 = simulate_ab_test(n_users=500, seed=7)
        assert r1.data.equals(r2.data)

    def test_different_seeds_differ(self):
        r1 = simulate_ab_test(n_users=500, seed=1)
        r2 = simulate_ab_test(n_users=500, seed=2)
        assert not r1.data.equals(r2.data)

    def test_has_segments(self):
        result = simulate_ab_test(n_users=1000)
        assert "segment" in result.data.columns
        assert set(result.data.segment.unique()) == {"new", "returning", "power"}

    def test_has_signup_day(self):
        result = simulate_ab_test(n_users=1000)
        assert "signup_day" in result.data.columns
        assert (result.data.signup_day >= 0).all()


class TestGeoSimulator:
    def test_returns_simulation_result(self):
        result = simulate_geo_experiment(n_markets=20, n_days=60)
        assert result.experiment_type == ExperimentType.GEO_EXPERIMENT

    def test_correct_market_count(self):
        result = simulate_geo_experiment(n_markets=30, n_days=60)
        assert result.data.market_id.nunique() == 30

    def test_correct_day_count(self):
        result = simulate_geo_experiment(n_markets=20, n_days=60)
        assert result.data.date.nunique() == 60

    def test_has_treatment_column(self):
        result = simulate_geo_experiment(n_markets=20, n_days=60)
        assert "is_treatment" in result.data.columns
        assert set(result.data.is_treatment.unique()) == {0, 1}

    def test_metric_positive(self):
        result = simulate_geo_experiment(n_markets=20, n_days=60)
        assert (result.data.metric_value >= 0).all()

    def test_treatment_markets_in_config(self):
        result = simulate_geo_experiment(n_markets=20, n_days=60)
        assert "treatment_markets" in result.config
        assert len(result.config["treatment_markets"]) > 0

    def test_effect_recoverable(self):
        """Large effect should be visible in raw means."""
        result = simulate_geo_experiment(
            n_markets=30, n_days=90, true_effect=0.20, seed=42
        )
        df = result.data
        post = df[df.post_treatment == 1]
        t_mean = post[post.is_treatment == 1].metric_value.mean()
        c_mean = post[post.is_treatment == 0].metric_value.mean()
        assert t_mean > c_mean  # treatment should be higher


class TestFeedbackSimulator:
    def test_correct_count(self):
        df = simulate_feedback_data(n_reviews=200)
        assert len(df) == 200

    def test_has_periods(self):
        df = simulate_feedback_data(n_reviews=200)
        assert set(df.period.unique()) == {"pre", "post"}

    def test_has_text(self):
        df = simulate_feedback_data(n_reviews=200)
        assert df.text.str.len().min() > 0

    def test_ratings_in_range(self):
        df = simulate_feedback_data(n_reviews=200)
        assert df.rating.between(1, 5).all()

    def test_positive_shift_visible(self):
        df = simulate_feedback_data(n_reviews=1000, true_sentiment_shift=0.3)
        pre_rating = df[df.period == "pre"].rating.mean()
        post_rating = df[df.period == "post"].rating.mean()
        assert post_rating > pre_rating
