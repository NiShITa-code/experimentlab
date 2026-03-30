"""
Experiment data simulators.

Generates realistic experiment datasets with known ground-truth effects.
This lets users:
  1. Explore the tool without needing real data
  2. Validate that each analysis method recovers the true effect
  3. Understand how sample size / noise / effect size interact

Three simulators:
  - A/B test: user-level with binary conversion + continuous revenue
  - Geo experiment: city-level time series with treatment period
  - Before/after: single time series with intervention point
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CFG
from src.data.schema import (
    ExperimentType,
    MetricDefinition,
    MetricType,
    ExperimentConfig,
    SimulationResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def simulate_ab_test(
    n_users: int = CFG.simulation.n_units,
    baseline_rate: float = CFG.simulation.baseline_rate,
    true_effect: float = CFG.simulation.true_effect,
    seed: int = CFG.simulation.seed,
    split_ratio: float = 0.5,
) -> SimulationResult:
    """
    Simulate a user-level A/B test.

    Generates users with:
      - user_id
      - group: 0 (control) or 1 (treatment)
      - converted: binary outcome (Bernoulli)
      - revenue: continuous outcome (converted * log-normal)
      - signup_day: day of signup (for sequential analysis)
      - segment: user segment (new/returning/power)

    The true_effect is an additive lift on the conversion rate:
      P(convert | treatment) = baseline_rate + true_effect

    Parameters
    ----------
    n_users : int
        Total users (split between control/treatment).
    baseline_rate : float
        Control group conversion rate.
    true_effect : float
        Additive treatment effect (can be 0 for A/A test).
    seed : int
        Random seed for reproducibility.
    split_ratio : float
        Fraction assigned to treatment.

    Returns
    -------
    SimulationResult
    """
    logger.info("Simulating A/B test: %d users, effect=%.3f", n_users, true_effect)
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)

    # Assignment
    group = rng.binomial(1, split_ratio, n_users)
    n_treatment = group.sum()
    n_control = n_users - n_treatment

    # Segments
    segment = rng.choice(
        ["new", "returning", "power"],
        n_users,
        p=[0.50, 0.35, 0.15],
    )

    # Segment-specific baseline adjustments
    segment_adjust = np.where(
        segment == "new", -0.02,
        np.where(segment == "power", 0.05, 0.0)
    )

    # Conversion (binary)
    prob = baseline_rate + segment_adjust + group * true_effect
    prob = np.clip(prob, 0.001, 0.999)
    converted = rng.binomial(1, prob)

    # Revenue (continuous, conditional on conversion)
    base_revenue = rng.lognormal(mean=3.0, sigma=1.0, size=n_users)
    revenue = converted * base_revenue * (1 + group * 0.1 * true_effect)
    revenue = revenue.round(2)

    # Signup day (for sequential testing)
    signup_day = rng.exponential(scale=10, size=n_users).astype(int)
    signup_day = np.clip(signup_day, 0, 30)

    df = pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(n_users)],
        "group": group,
        "segment": segment,
        "signup_day": signup_day,
        "converted": converted,
        "revenue": revenue,
    })

    elapsed = time.perf_counter() - t0
    logger.info("  Control: %d (rate=%.3f) | Treatment: %d (rate=%.3f) | %.2fs",
                n_control, df.loc[df.group == 0, "converted"].mean(),
                n_treatment, df.loc[df.group == 1, "converted"].mean(),
                elapsed)

    return SimulationResult(
        data=df,
        experiment_type=ExperimentType.AB_TEST,
        true_effect=true_effect,
        n_units=n_users,
        config={
            "baseline_rate": baseline_rate,
            "true_effect": true_effect,
            "split_ratio": split_ratio,
            "seed": seed,
        },
        elapsed_seconds=elapsed,
    )


def simulate_geo_experiment(
    n_markets: int = CFG.simulation.n_geo_units,
    n_days: int = CFG.simulation.n_time_periods,
    true_effect: float = CFG.simulation.true_effect,
    treatment_fraction: float = 0.3,
    seed: int = CFG.simulation.seed,
) -> SimulationResult:
    """
    Simulate a geo-level experiment (city-level time series).

    Generates daily metric data for n_markets cities over n_days.
    A fraction of markets receive treatment starting at day
    int(n_days * pre_fraction).

    Each market has:
      - A market-specific baseline (intercept)
      - A common time trend
      - Weekly seasonality
      - Gaussian noise
      - Treatment effect (additive, post-treatment only)

    This is the standard DGP for diff-in-diff and synthetic control.

    Parameters
    ----------
    n_markets : int
        Number of geographic markets (cities/regions).
    n_days : int
        Total observation period in days.
    true_effect : float
        True treatment effect (additive on daily metric).
    treatment_fraction : float
        Fraction of markets assigned to treatment.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    logger.info("Simulating geo experiment: %d markets × %d days, effect=%.3f",
                n_markets, n_days, true_effect)
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)

    pre_fraction = CFG.geo.synthetic_control.get("pre_periods_fraction", 0.7)
    treatment_start_day = int(n_days * pre_fraction)

    # Market characteristics
    market_ids = [f"M{i:03d}" for i in range(n_markets)]
    market_baselines = rng.uniform(50, 200, n_markets)  # daily metric baseline
    market_trends = rng.uniform(0.01, 0.05, n_markets)  # daily growth rate

    # Treatment assignment
    n_treated = max(2, int(n_markets * treatment_fraction))
    treated_idx = rng.choice(n_markets, n_treated, replace=False)
    treatment_markets = [market_ids[i] for i in treated_idx]

    # Population for each market (affects noise scale)
    market_pop = rng.lognormal(mean=10, sigma=1.5, size=n_markets).astype(int)
    market_pop = np.clip(market_pop, 1000, 10_000_000)

    records = []
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    for m_idx in range(n_markets):
        mid = market_ids[m_idx]
        baseline = market_baselines[m_idx]
        trend = market_trends[m_idx]
        pop = market_pop[m_idx]
        is_treated = mid in treatment_markets

        for d_idx, date in enumerate(dates):
            # Base value: baseline + trend + weekly seasonality
            day_of_week = date.dayofweek
            seasonality = 5 * np.sin(2 * np.pi * day_of_week / 7)
            value = baseline + trend * d_idx + seasonality

            # Treatment effect (only post-treatment, only treated markets)
            effect = 0.0
            if is_treated and d_idx >= treatment_start_day:
                # Effect ramps up over first 7 days, then stabilises
                days_since = d_idx - treatment_start_day
                ramp = min(1.0, days_since / 7)
                effect = true_effect * baseline * ramp  # proportional effect

            # Noise (smaller for larger markets)
            noise_scale = baseline * 0.1 / np.sqrt(pop / 100000)
            noise = rng.normal(0, noise_scale)

            value = max(0, value + effect + noise)

            records.append({
                "market_id": mid,
                "date": date,
                "metric_value": round(value, 2),
                "population": pop,
                "is_treatment": int(is_treated),
                "post_treatment": int(d_idx >= treatment_start_day),
            })

    df = pd.DataFrame(records)

    elapsed = time.perf_counter() - t0
    logger.info("  %d treated markets, treatment starts day %d | %.2fs",
                n_treated, treatment_start_day, elapsed)

    return SimulationResult(
        data=df,
        experiment_type=ExperimentType.GEO_EXPERIMENT,
        true_effect=true_effect,
        n_units=n_markets,
        config={
            "n_markets": n_markets,
            "n_days": n_days,
            "true_effect": true_effect,
            "treatment_fraction": treatment_fraction,
            "treatment_start_day": treatment_start_day,
            "treatment_markets": treatment_markets,
            "seed": seed,
        },
        elapsed_seconds=elapsed,
    )


def simulate_feedback_data(
    n_reviews: int = 500,
    true_sentiment_shift: float = 0.15,
    seed: int = CFG.simulation.seed,
) -> pd.DataFrame:
    """
    Simulate user feedback/review data for NLP sentiment analysis.

    Generates pre/post text reviews with a controllable sentiment shift.

    Parameters
    ----------
    n_reviews : int
        Total number of reviews (split pre/post).
    true_sentiment_shift : float
        How much average sentiment improves post-treatment.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with columns: review_id, period, text, rating
    """
    rng = np.random.RandomState(seed)

    positive_phrases = [
        "Great experience, very smooth",
        "Fast and reliable service",
        "Love the new features, much improved",
        "Easy to use and intuitive",
        "Best app in its category",
        "Excellent customer support",
        "Very satisfied with the update",
        "Works perfectly every time",
    ]
    negative_phrases = [
        "App crashes frequently",
        "Very slow and buggy",
        "Terrible user experience",
        "Customer support is unhelpful",
        "Too many ads and interruptions",
        "Not worth the subscription price",
        "Confusing interface, hard to navigate",
        "Keeps logging me out randomly",
    ]
    neutral_phrases = [
        "It works okay, nothing special",
        "Average experience overall",
        "Some good features but room for improvement",
        "Decent app for basic needs",
        "Does what it says, no more no less",
    ]

    records = []
    n_pre = n_reviews // 2
    n_post = n_reviews - n_pre

    for i in range(n_reviews):
        is_post = i >= n_pre
        period = "post" if is_post else "pre"

        # Sentiment probability: post-treatment gets a boost
        p_positive = 0.3 + (true_sentiment_shift if is_post else 0)
        p_negative = 0.3 - (true_sentiment_shift * 0.5 if is_post else 0)
        p_neutral = 1 - p_positive - p_negative
        p_neutral = max(0.1, p_neutral)

        # Normalize
        total = p_positive + p_negative + p_neutral
        probs = [p_positive/total, p_negative/total, p_neutral/total]

        sentiment = rng.choice(["positive", "negative", "neutral"], p=probs)

        if sentiment == "positive":
            text = rng.choice(positive_phrases)
            rating = rng.choice([4, 5])
        elif sentiment == "negative":
            text = rng.choice(negative_phrases)
            rating = rng.choice([1, 2])
        else:
            text = rng.choice(neutral_phrases)
            rating = 3

        records.append({
            "review_id": f"R{i:05d}",
            "period": period,
            "text": text,
            "rating": rating,
        })

    return pd.DataFrame(records)
