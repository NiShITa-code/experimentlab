"""
Privacy-preserving experiment analysis.

Enables measurement without individual-level tracking. Critical for:
  - Post-cookie advertising measurement
  - Compliance with GDPR/CCPA
  - Geo experiments where you only have aggregate data

Methods:
  - k-anonymity checks (group sizes ≥ k)
  - Differential privacy noise (Laplace mechanism)
  - Aggregate-only analysis pipeline

Reference: Dwork & Roth (2014), "The Algorithmic Foundations of
Differential Privacy"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PrivacyReport:
    """Privacy audit output."""
    k_anonymity_passed: bool
    k_threshold: int
    min_group_size: int
    groups_below_threshold: int
    total_groups: int

    dp_applied: bool
    dp_epsilon: float
    noise_magnitude: float

    # Utility metrics
    raw_mean: Optional[float]
    noisy_mean: Optional[float]
    utility_loss: float             # relative error from noise

    message: str

    def summary(self) -> str:
        k_status = "PASSED" if self.k_anonymity_passed else "FAILED"
        lines = [
            "─" * 50,
            "PRIVACY REPORT",
            "─" * 50,
            f"  k-anonymity (k={self.k_threshold}): {k_status}",
            f"  Min group size:          {self.min_group_size}",
            f"  Groups below threshold:  {self.groups_below_threshold}/{self.total_groups}",
        ]
        if self.dp_applied:
            lines.append(f"  DP noise applied:        ε={self.dp_epsilon}")
            lines.append(f"  Noise magnitude:         {self.noise_magnitude:.4f}")
            lines.append(f"  Utility loss:            {self.utility_loss:.2%}")
        lines.append("─" * 50)
        return "\n".join(lines)


def check_k_anonymity(
    df: pd.DataFrame,
    group_cols: List[str],
    k: int = CFG.privacy.k_anonymity_threshold,
) -> dict:
    """
    Check if all groups have at least k members.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check.
    group_cols : list[str]
        Columns that define groups (e.g., treatment + segment).
    k : int
        Minimum group size required.

    Returns
    -------
    dict with keys: passed, min_size, groups_below_k, violations
    """
    group_sizes = df.groupby(group_cols).size()
    min_size = int(group_sizes.min())
    below_k = int((group_sizes < k).sum())
    total = len(group_sizes)

    violations = []
    if below_k > 0:
        small = group_sizes[group_sizes < k]
        for idx, size in small.items():
            violations.append({"group": str(idx), "size": int(size)})

    return {
        "passed": below_k == 0,
        "k": k,
        "min_size": min_size,
        "groups_below_k": below_k,
        "total_groups": total,
        "violations": violations,
    }


def add_dp_noise(
    value: float,
    sensitivity: float,
    epsilon: float = CFG.privacy.dp_epsilon,
    mechanism: str = "laplace",
) -> float:
    """
    Add differentially private noise to a value.

    Parameters
    ----------
    value : float
        True aggregate value.
    sensitivity : float
        L1 sensitivity (max change from adding/removing one record).
    epsilon : float
        Privacy budget (smaller = more private, more noisy).
    mechanism : str
        "laplace" (ε-DP) or "gaussian" (approximate (ε,δ)-DP).

    Returns
    -------
    float : noised value
    """
    if mechanism == "laplace":
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
    elif mechanism == "gaussian":
        delta = CFG.privacy.dp_delta
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return float(value + noise)


def private_mean(
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    epsilon: float = CFG.privacy.dp_epsilon,
) -> float:
    """
    Compute differentially private mean.

    Clips values to [lower_bound, upper_bound], computes mean,
    then adds calibrated Laplace noise.
    """
    clipped = np.clip(values, lower_bound, upper_bound)
    true_mean = float(clipped.mean())
    sensitivity = (upper_bound - lower_bound) / len(values)
    return add_dp_noise(true_mean, sensitivity, epsilon)


def run_privacy_audit(
    df: pd.DataFrame,
    metric_col: str,
    group_cols: List[str],
    apply_dp: bool = True,
    epsilon: float = CFG.privacy.dp_epsilon,
    k: int = CFG.privacy.k_anonymity_threshold,
) -> PrivacyReport:
    """
    Run full privacy audit on experiment data.

    Checks k-anonymity and optionally applies DP noise.
    """
    logger.info("Running privacy audit (k=%d, ε=%.1f)...", k, epsilon)

    # k-anonymity check
    k_check = check_k_anonymity(df, group_cols, k)

    # DP noise
    raw_mean = float(df[metric_col].mean())
    noisy_mean = None
    noise_mag = 0.0
    utility_loss = 0.0

    if apply_dp:
        values = df[metric_col].values
        lower = float(values.min())
        upper = float(values.max())
        noisy_mean = private_mean(values, lower, upper, epsilon)
        noise_mag = abs(noisy_mean - raw_mean)
        utility_loss = noise_mag / abs(raw_mean) if raw_mean != 0 else 0

    msg = (
        f"k-anonymity (k={k}): {'PASSED' if k_check['passed'] else 'FAILED'}. "
        f"Min group size: {k_check['min_size']}. "
    )
    if apply_dp:
        msg += f"DP noise applied (ε={epsilon}): utility loss {utility_loss:.2%}."

    result = PrivacyReport(
        k_anonymity_passed=k_check["passed"],
        k_threshold=k,
        min_group_size=k_check["min_size"],
        groups_below_threshold=k_check["groups_below_k"],
        total_groups=k_check["total_groups"],
        dp_applied=apply_dp,
        dp_epsilon=epsilon,
        noise_magnitude=round(noise_mag, 4),
        raw_mean=round(raw_mean, 4),
        noisy_mean=round(noisy_mean, 4) if noisy_mean is not None else None,
        utility_loss=round(utility_loss, 4),
        message=msg,
    )

    logger.info(result.summary())
    return result
