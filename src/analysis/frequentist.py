"""
Frequentist A/B test analysis.

Standard hypothesis testing for A/B experiments:
  - Two-proportion z-test (binary metrics)
  - Welch's t-test (continuous metrics)
  - Confidence intervals on lift
  - Multiple testing correction (Bonferroni, Holm)

Also includes CUPED (Controlled-experiment Using Pre-Experiment Data)
for variance reduction when pre-experiment covariates are available.

Reference: Deng et al. (2013), "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FrequentistABResult:
    """Complete frequentist A/B test output."""
    # Core results
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float            # (treatment - control) / control
    ci_lower: float                 # 95% CI on absolute lift
    ci_upper: float
    ci_relative_lower: float        # 95% CI on relative lift
    ci_relative_upper: float

    # Test statistics
    test_stat: float                # z or t statistic
    p_value: float
    significant: bool
    test_name: str                  # "two-proportion z-test" or "Welch's t-test"

    # Sample info
    n_control: int
    n_treatment: int
    control_std: float
    treatment_std: float
    pooled_se: float

    # CUPED (if applied)
    cuped_applied: bool
    variance_reduction: float       # % reduction from CUPED

    elapsed_seconds: float

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        lines = [
            "═" * 55,
            "FREQUENTIST A/B TEST",
            "═" * 55,
            f"  Test:                    {self.test_name}",
            f"  Control mean:            {self.control_mean:.4f} (n={self.n_control:,})",
            f"  Treatment mean:          {self.treatment_mean:.4f} (n={self.n_treatment:,})",
            "",
            f"  Absolute lift:           {self.absolute_lift:+.4f}",
            f"  Relative lift:           {self.relative_lift:+.2%}",
            f"  95% CI (absolute):       [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"  95% CI (relative):       [{self.ci_relative_lower:+.2%}, {self.ci_relative_upper:+.2%}]",
            "",
            f"  {self.test_name} stat:   {self.test_stat:.3f}",
            f"  p-value:                 {self.p_value:.4f} ({sig})",
        ]
        if self.cuped_applied:
            lines.append(f"  CUPED variance reduction: {self.variance_reduction:.1f}%")
        lines.append("═" * 55)
        return "\n".join(lines)


def run_frequentist_ab(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    metric_type: str = "binary",
    alpha: float = 0.05,
    pre_experiment_control: Optional[np.ndarray] = None,
    pre_experiment_treatment: Optional[np.ndarray] = None,
) -> FrequentistABResult:
    """
    Run frequentist A/B test.

    Parameters
    ----------
    control_data : np.ndarray
        Observed outcomes for control.
    treatment_data : np.ndarray
        Observed outcomes for treatment.
    metric_type : str
        "binary" or "continuous".
    alpha : float
        Significance level.
    pre_experiment_control, pre_experiment_treatment : np.ndarray, optional
        Pre-experiment covariate for CUPED variance reduction.

    Returns
    -------
    FrequentistABResult
    """
    logger.info("Running frequentist A/B test (metric=%s)...", metric_type)
    t0 = time.perf_counter()

    n_c = len(control_data)
    n_t = len(treatment_data)

    # CUPED variance reduction
    cuped_applied = False
    variance_reduction = 0.0

    if pre_experiment_control is not None and pre_experiment_treatment is not None:
        control_data, treatment_data, variance_reduction = _apply_cuped(
            control_data, treatment_data,
            pre_experiment_control, pre_experiment_treatment,
        )
        cuped_applied = True

    c_mean = float(control_data.mean())
    t_mean = float(treatment_data.mean())
    c_std = float(control_data.std(ddof=1))
    t_std = float(treatment_data.std(ddof=1))

    absolute_lift = t_mean - c_mean
    relative_lift = absolute_lift / c_mean if c_mean != 0 else 0

    if metric_type == "binary":
        # Two-proportion z-test
        p_pooled = (control_data.sum() + treatment_data.sum()) / (n_c + n_t)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_c + 1/n_t))
        z_stat = absolute_lift / se if se > 0 else 0
        p_value = 2 * stats.norm.sf(abs(z_stat))
        test_name = "two-proportion z-test"
        test_stat = z_stat

    elif metric_type == "continuous":
        # Welch's t-test
        se = np.sqrt(c_std**2 / n_c + t_std**2 / n_t)
        t_stat_val = absolute_lift / se if se > 0 else 0
        # Welch-Satterthwaite degrees of freedom
        df_num = (c_std**2/n_c + t_std**2/n_t)**2
        df_den = (c_std**2/n_c)**2/(n_c-1) + (t_std**2/n_t)**2/(n_t-1)
        df = df_num / df_den if df_den > 0 else n_c + n_t - 2
        p_value = 2 * stats.t.sf(abs(t_stat_val), df)
        test_name = "Welch's t-test"
        test_stat = t_stat_val

    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = absolute_lift - z_crit * se
    ci_upper = absolute_lift + z_crit * se

    # Relative CI (delta method)
    if c_mean != 0:
        rel_se = se / abs(c_mean)
        ci_rel_lower = relative_lift - z_crit * rel_se
        ci_rel_upper = relative_lift + z_crit * rel_se
    else:
        ci_rel_lower = ci_rel_upper = 0

    elapsed = time.perf_counter() - t0

    result = FrequentistABResult(
        control_mean=c_mean,
        treatment_mean=t_mean,
        absolute_lift=round(float(absolute_lift), 6),
        relative_lift=round(float(relative_lift), 6),
        ci_lower=round(float(ci_lower), 6),
        ci_upper=round(float(ci_upper), 6),
        ci_relative_lower=round(float(ci_rel_lower), 6),
        ci_relative_upper=round(float(ci_rel_upper), 6),
        test_stat=round(float(test_stat), 4),
        p_value=round(float(p_value), 6),
        significant=p_value < alpha,
        test_name=test_name,
        n_control=n_c,
        n_treatment=n_t,
        control_std=round(float(c_std), 6),
        treatment_std=round(float(t_std), 6),
        pooled_se=round(float(se), 6),
        cuped_applied=cuped_applied,
        variance_reduction=round(float(variance_reduction), 1),
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result


def _apply_cuped(
    control_post: np.ndarray,
    treatment_post: np.ndarray,
    control_pre: np.ndarray,
    treatment_pre: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    CUPED: reduce variance using pre-experiment covariates.

    Y_cuped = Y - theta * (X - E[X])
    where theta = Cov(Y, X) / Var(X)

    Returns adjusted control, treatment, and % variance reduction.
    """
    # Combine all data for theta estimation
    all_post = np.concatenate([control_post, treatment_post])
    all_pre = np.concatenate([control_pre, treatment_pre])

    # Estimate theta
    cov_xy = np.cov(all_post, all_pre)[0, 1]
    var_x = np.var(all_pre)
    theta = cov_xy / var_x if var_x > 0 else 0

    # Adjust
    pre_mean = all_pre.mean()
    c_adjusted = control_post - theta * (control_pre - pre_mean)
    t_adjusted = treatment_post - theta * (treatment_pre - pre_mean)

    # Variance reduction
    var_before = np.var(all_post)
    var_after = np.var(np.concatenate([c_adjusted, t_adjusted]))
    reduction = (1 - var_after / var_before) * 100 if var_before > 0 else 0

    logger.info("  CUPED: theta=%.3f, variance reduction=%.1f%%", theta, reduction)

    return c_adjusted, t_adjusted, float(reduction)


def check_srm(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 0.5,
    threshold: float = CFG.experiment.srm_threshold,
) -> dict:
    """
    Sample Ratio Mismatch (SRM) check.

    Tests if the actual split matches the expected split.
    SRM indicates a bug in the randomization — results are unreliable.

    Parameters
    ----------
    n_control, n_treatment : int
        Observed group sizes.
    expected_ratio : float
        Expected fraction in treatment (default 0.5 for 50/50).
    threshold : float
        p-value below which SRM is flagged.

    Returns
    -------
    dict with keys: passed, p_value, expected_treatment, actual_treatment, message
    """
    total = n_control + n_treatment
    expected_treatment = int(total * expected_ratio)
    expected_control = total - expected_treatment

    # Chi-squared goodness of fit
    observed = [n_control, n_treatment]
    expected = [expected_control, expected_treatment]
    chi2, p_value = stats.chisquare(observed, expected)

    passed = p_value > threshold
    actual_ratio = n_treatment / total

    msg = (
        f"SRM check: expected {expected_ratio:.1%}/{1-expected_ratio:.1%}, "
        f"observed {actual_ratio:.1%}/{1-actual_ratio:.1%}. "
        f"Chi²={chi2:.2f}, p={p_value:.4f}. "
        + ("PASSED — no mismatch detected." if passed
           else "FAILED — sample ratio mismatch detected! Results may be unreliable.")
    )

    return {
        "passed": passed,
        "p_value": round(float(p_value), 6),
        "chi_squared": round(float(chi2), 3),
        "expected_ratio": expected_ratio,
        "actual_ratio": round(float(actual_ratio), 4),
        "n_control": n_control,
        "n_treatment": n_treatment,
        "message": msg,
    }
