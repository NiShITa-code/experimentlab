"""
Bayesian A/B testing.

Instead of p-values, answers the question practitioners actually care about:
"What is the probability that treatment is better than control?"

Advantages over frequentist:
  - Direct probability statements (not "fail to reject H0")
  - Works with small samples (posterior updates continuously)
  - ROPE (Region of Practical Equivalence) for practical significance
  - No multiple testing correction needed for peeking

Models:
  - Binary metrics: Beta-Binomial conjugate model
  - Continuous metrics: Normal-Normal conjugate model

Reference: Kruschke (2013), "Bayesian estimation supersedes the t-test"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BayesianABResult:
    """Complete Bayesian A/B test output."""
    # Posterior summaries
    control_mean: float
    treatment_mean: float
    lift: float                         # (treatment - control) / control
    lift_ci_lower: float                # 95% HDI lower
    lift_ci_upper: float                # 95% HDI upper

    # Decision metrics
    prob_treatment_better: float        # P(B > A)
    prob_lift_gt_0: float               # P(lift > 0)
    prob_in_rope: float                 # P(|lift| < rope_width)
    expected_loss_treatment: float      # E[loss if we pick treatment]
    expected_loss_control: float        # E[loss if we pick control]

    # Recommendation
    recommendation: str                 # "treatment", "control", "inconclusive"
    confidence: str                     # "high", "medium", "low"

    # Posteriors for plotting
    control_posterior_samples: np.ndarray
    treatment_posterior_samples: np.ndarray
    lift_samples: np.ndarray

    # Metadata
    metric_type: str                    # "binary" or "continuous"
    n_control: int
    n_treatment: int
    rope_width: float
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            "═" * 55,
            "BAYESIAN A/B TEST",
            "═" * 55,
            f"  Control mean:            {self.control_mean:.4f}",
            f"  Treatment mean:          {self.treatment_mean:.4f}",
            f"  Relative lift:           {self.lift:+.2%}",
            f"  95% HDI on lift:         [{self.lift_ci_lower:+.2%}, {self.lift_ci_upper:+.2%}]",
            "",
            f"  P(treatment > control):  {self.prob_treatment_better:.1%}",
            f"  P(lift > 0):             {self.prob_lift_gt_0:.1%}",
            f"  P(in ROPE ±{self.rope_width:.1%}):    {self.prob_in_rope:.1%}",
            "",
            f"  Expected loss (pick treatment): {self.expected_loss_treatment:.4f}",
            f"  Expected loss (pick control):   {self.expected_loss_control:.4f}",
            "",
            f"  Recommendation:          {self.recommendation.upper()}",
            f"  Confidence:              {self.confidence}",
            f"  Samples:                 control={self.n_control:,} | treatment={self.n_treatment:,}",
            "═" * 55,
        ]
        return "\n".join(lines)


def run_bayesian_ab(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    metric_type: str = "binary",
    prior_alpha: float = CFG.bayesian.prior_alpha,
    prior_beta: float = CFG.bayesian.prior_beta,
    n_samples: int = CFG.bayesian.n_posterior_samples,
    rope_width: float = CFG.bayesian.rope_width,
) -> BayesianABResult:
    """
    Run Bayesian A/B test.

    Parameters
    ----------
    control_data : np.ndarray
        Observed outcomes for control group.
    treatment_data : np.ndarray
        Observed outcomes for treatment group.
    metric_type : str
        "binary" (conversion) or "continuous" (revenue).
    prior_alpha, prior_beta : float
        Beta prior parameters (for binary metrics).
    n_samples : int
        Number of posterior samples to draw.
    rope_width : float
        Region of practical equivalence (relative).

    Returns
    -------
    BayesianABResult
    """
    logger.info("Running Bayesian A/B test (metric=%s)...", metric_type)
    t0 = time.perf_counter()

    n_control = len(control_data)
    n_treatment = len(treatment_data)

    if metric_type == "binary":
        # Beta-Binomial conjugate model
        # Posterior: Beta(alpha + successes, beta + failures)
        c_successes = control_data.sum()
        c_failures = n_control - c_successes
        t_successes = treatment_data.sum()
        t_failures = n_treatment - t_successes

        control_posterior = np.random.beta(
            prior_alpha + c_successes,
            prior_beta + c_failures,
            n_samples,
        )
        treatment_posterior = np.random.beta(
            prior_alpha + t_successes,
            prior_beta + t_failures,
            n_samples,
        )

    elif metric_type == "continuous":
        # Normal-Normal conjugate model (known variance approximation)
        c_mean, c_std = control_data.mean(), control_data.std()
        t_mean, t_std = treatment_data.mean(), treatment_data.std()

        c_se = c_std / np.sqrt(n_control)
        t_se = t_std / np.sqrt(n_treatment)

        control_posterior = np.random.normal(c_mean, c_se, n_samples)
        treatment_posterior = np.random.normal(t_mean, t_se, n_samples)

    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    # Lift samples
    # Avoid division by zero
    safe_control = np.where(control_posterior == 0, 1e-10, control_posterior)
    lift_samples = (treatment_posterior - control_posterior) / np.abs(safe_control)

    # Decision metrics
    prob_treatment_better = float(np.mean(treatment_posterior > control_posterior))
    prob_lift_gt_0 = float(np.mean(lift_samples > 0))
    prob_in_rope = float(np.mean(np.abs(lift_samples) < rope_width))

    # Expected loss (opportunity cost of wrong decision)
    loss_if_treatment = np.maximum(control_posterior - treatment_posterior, 0).mean()
    loss_if_control = np.maximum(treatment_posterior - control_posterior, 0).mean()

    # HDI on lift
    lift_sorted = np.sort(lift_samples)
    ci_idx = int(0.025 * n_samples)
    lift_ci_lower = float(lift_sorted[ci_idx])
    lift_ci_upper = float(lift_sorted[-ci_idx])

    # Recommendation
    if prob_treatment_better > 0.95:
        recommendation = "treatment"
        confidence = "high"
    elif prob_treatment_better > 0.90:
        recommendation = "treatment"
        confidence = "medium"
    elif prob_treatment_better < 0.05:
        recommendation = "control"
        confidence = "high"
    elif prob_treatment_better < 0.10:
        recommendation = "control"
        confidence = "medium"
    elif prob_in_rope > 0.90:
        recommendation = "no difference (ROPE)"
        confidence = "high"
    else:
        recommendation = "inconclusive"
        confidence = "low"

    elapsed = time.perf_counter() - t0

    result = BayesianABResult(
        control_mean=float(control_posterior.mean()),
        treatment_mean=float(treatment_posterior.mean()),
        lift=float(np.mean(lift_samples)),
        lift_ci_lower=lift_ci_lower,
        lift_ci_upper=lift_ci_upper,
        prob_treatment_better=prob_treatment_better,
        prob_lift_gt_0=prob_lift_gt_0,
        prob_in_rope=prob_in_rope,
        expected_loss_treatment=float(loss_if_treatment),
        expected_loss_control=float(loss_if_control),
        recommendation=recommendation,
        confidence=confidence,
        control_posterior_samples=control_posterior,
        treatment_posterior_samples=treatment_posterior,
        lift_samples=lift_samples,
        metric_type=metric_type,
        n_control=n_control,
        n_treatment=n_treatment,
        rope_width=rope_width,
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result
