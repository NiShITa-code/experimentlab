"""
Statistical power analysis for experiment design.

Answers: "How many users/markets do I need, and how long will it take?"

Supports:
  - Binary metrics (proportion test)
  - Continuous metrics (t-test)
  - Geo experiments (cluster-level)
  - Inverse: given N, what's the minimum detectable effect?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PowerResult:
    """Output of power analysis."""
    sample_per_arm: int
    total_sample: int
    baseline_rate: float
    mde: float
    alpha: float
    power: float
    metric_type: str
    daily_traffic: int
    estimated_days: int
    estimated_weeks: float

    def summary(self) -> str:
        lines = [
            "─" * 50,
            "POWER ANALYSIS",
            "─" * 50,
            f"  Metric type:             {self.metric_type}",
            f"  Baseline:                {self.baseline_rate:.4f}",
            f"  MDE:                     {self.mde:+.4f}",
            f"  α={self.alpha}, power={self.power}",
            "",
            f"  Sample per arm:          {self.sample_per_arm:,}",
            f"  Total sample:            {self.total_sample:,}",
            f"  Daily traffic:           {self.daily_traffic:,}",
            f"  Duration:                {self.estimated_days} days ({self.estimated_weeks:.1f} weeks)",
            "─" * 50,
        ]
        return "\n".join(lines)


def compute_sample_size_binary(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Sample size per arm for two-proportion z-test."""
    p1 = baseline_rate
    p2 = max(0.001, min(0.999, baseline_rate + mde))

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1*(1-p1) + p2*(1-p2))
    denominator = (p2 - p1) ** 2

    if denominator == 0:
        return 999999
    return max(10, math.ceil(numerator / denominator))


def compute_sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Sample size per arm for Welch's t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Assume equal variance in both arms
    numerator = 2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2
    denominator = mde ** 2

    if denominator == 0:
        return 999999
    return max(10, math.ceil(numerator / denominator))


def compute_mde(
    sample_per_arm: int,
    baseline_rate: float = 0.10,
    baseline_std: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binary",
) -> float:
    """Given N, compute the minimum detectable effect."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if metric_type == "binary":
        # Binary search
        lo, hi = 0.001, 0.5
        for _ in range(50):
            mid = (lo + hi) / 2
            n = compute_sample_size_binary(baseline_rate, mid, alpha, power)
            if n <= sample_per_arm:
                hi = mid
            else:
                lo = mid
        return round((lo + hi) / 2, 4)
    else:
        if baseline_std is None:
            baseline_std = baseline_rate * 0.5  # rough estimate
        mde = (z_alpha + z_beta) * baseline_std * np.sqrt(2 / sample_per_arm)
        return round(float(mde), 4)


def run_power_analysis(
    baseline_rate: float = 0.10,
    baseline_std: Optional[float] = None,
    mde: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binary",
    daily_traffic: int = 1000,
) -> PowerResult:
    """Run full power analysis with duration estimate."""
    if metric_type == "binary":
        n = compute_sample_size_binary(baseline_rate, mde, alpha, power)
    else:
        std = baseline_std or baseline_rate * 0.5
        n = compute_sample_size_continuous(baseline_rate, std, mde, alpha, power)

    total = n * 2
    days = math.ceil(total / daily_traffic) if daily_traffic > 0 else 999

    result = PowerResult(
        sample_per_arm=n,
        total_sample=total,
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power,
        metric_type=metric_type,
        daily_traffic=daily_traffic,
        estimated_days=days,
        estimated_weeks=round(days / 7, 1),
    )
    logger.info(result.summary())
    return result
