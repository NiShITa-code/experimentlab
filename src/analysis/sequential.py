"""
Sequential testing for A/B experiments.

Allows valid early stopping — check results at multiple points
without inflating false positive rate.

Methods:
  - O'Brien-Fleming spending function (conservative early, liberal late)
  - Pocock spending function (equal at each look)
  - Group sequential boundaries

Used at Google for: experiments that run for weeks where you want
to stop early if the result is obviously positive or futile.

Reference: Jennison & Turnbull (1999), Group Sequential Methods
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SequentialBoundary:
    """Boundaries at a single interim look."""
    look_number: int
    information_fraction: float     # fraction of total sample observed
    n_observed: int
    z_upper: float                  # efficacy boundary (stop for success)
    z_lower: float                  # futility boundary (stop for no effect)
    alpha_spent: float              # cumulative alpha spent so far


@dataclass
class SequentialLook:
    """Result of checking at one interim analysis."""
    look_number: int
    n_control: int
    n_treatment: int
    z_statistic: float
    p_value: float
    boundary: SequentialBoundary
    decision: str                   # "continue", "stop_efficacy", "stop_futility"
    control_mean: float
    treatment_mean: float


@dataclass
class SequentialResult:
    """Complete sequential analysis output."""
    boundaries: List[SequentialBoundary]
    looks: List[SequentialLook]
    spending_function: str          # "obrien_fleming" or "pocock"
    total_alpha: float
    max_looks: int
    stopped_early: bool
    final_decision: str
    final_look: int

    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            "═" * 55,
            "SEQUENTIAL TESTING",
            "═" * 55,
            f"  Spending function:       {self.spending_function}",
            f"  Total α:                 {self.total_alpha}",
            f"  Planned looks:           {self.max_looks}",
            f"  Looks conducted:         {len(self.looks)}",
            f"  Stopped early:           {'Yes' if self.stopped_early else 'No'}",
            f"  Decision:                {self.final_decision}",
            "",
            "  Boundaries:",
            f"  {'Look':>4} {'Info%':>6} {'z_upper':>8} {'z_lower':>8} {'α_spent':>8}",
        ]
        for b in self.boundaries:
            lines.append(
                f"  {b.look_number:>4} {b.information_fraction:>6.0%} "
                f"{b.z_upper:>8.3f} {b.z_lower:>8.3f} {b.alpha_spent:>8.4f}"
            )
        if self.looks:
            lines.append("")
            lines.append("  Results at each look:")
            for look in self.looks:
                lines.append(
                    f"  Look {look.look_number}: z={look.z_statistic:+.3f}, "
                    f"p={look.p_value:.4f} → {look.decision}"
                )
        lines.append("═" * 55)
        return "\n".join(lines)


def _obrien_fleming_boundary(alpha: float, info_fraction: float) -> float:
    """O'Brien-Fleming spending function: α(t) = 2 - 2*Φ(z_α/2 / sqrt(t))"""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return z_alpha / np.sqrt(info_fraction)


def _pocock_boundary(alpha: float, n_looks: int) -> float:
    """Pocock spending function: constant boundary at each look."""
    # Approximate Pocock boundary
    # Exact requires numerical integration; this is the standard approximation
    z_alpha = stats.norm.ppf(1 - alpha / (2 * n_looks))
    return z_alpha


def compute_boundaries(
    n_looks: int = CFG.experiment.sequential_looks,
    alpha: float = CFG.experiment.default_alpha,
    spending_function: str = "obrien_fleming",
    total_sample: int = 10000,
) -> List[SequentialBoundary]:
    """
    Compute group sequential boundaries for planned interim analyses.

    Parameters
    ----------
    n_looks : int
        Number of planned interim looks (including final).
    alpha : float
        Total Type I error rate to spend.
    spending_function : str
        "obrien_fleming" (conservative early) or "pocock" (equal).
    total_sample : int
        Total planned sample size.

    Returns
    -------
    List[SequentialBoundary]
    """
    boundaries = []
    cumulative_alpha = 0.0

    for i in range(1, n_looks + 1):
        info_frac = i / n_looks
        n_obs = int(total_sample * info_frac)

        if spending_function == "obrien_fleming":
            z_upper = _obrien_fleming_boundary(alpha, info_frac)
            # Incremental alpha spending
            alpha_at_look = 2 * (1 - stats.norm.cdf(z_upper))
            cumulative_alpha = min(alpha, cumulative_alpha + alpha_at_look)
        elif spending_function == "pocock":
            z_upper = _pocock_boundary(alpha, n_looks)
            alpha_at_look = alpha / n_looks
            cumulative_alpha = min(alpha, cumulative_alpha + alpha_at_look)
        else:
            raise ValueError(f"Unknown spending function: {spending_function}")

        # Futility boundary (non-binding): z < 0 at early looks
        z_lower = -z_upper * 0.5 if i < n_looks else 0

        boundaries.append(SequentialBoundary(
            look_number=i,
            information_fraction=info_frac,
            n_observed=n_obs,
            z_upper=round(float(z_upper), 4),
            z_lower=round(float(z_lower), 4),
            alpha_spent=round(float(cumulative_alpha), 4),
        ))

    return boundaries


def run_sequential_analysis(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    n_looks: int = CFG.experiment.sequential_looks,
    alpha: float = CFG.experiment.default_alpha,
    spending_function: str = "obrien_fleming",
    metric_type: str = "binary",
) -> SequentialResult:
    """
    Run sequential analysis with interim looks.

    Simulates checking the experiment at n_looks equally-spaced points.
    At each look, computes z-statistic and compares to boundaries.

    Parameters
    ----------
    control_data, treatment_data : np.ndarray
        Full experiment data (we simulate peeking at subsets).
    n_looks : int
        Number of interim analyses.
    alpha : float
        Total Type I error rate.
    spending_function : str
        "obrien_fleming" or "pocock".
    metric_type : str
        "binary" or "continuous".

    Returns
    -------
    SequentialResult
    """
    logger.info("Running sequential analysis (%d looks, %s)...",
                n_looks, spending_function)
    t0 = time.perf_counter()

    total_sample = len(control_data) + len(treatment_data)
    boundaries = compute_boundaries(n_looks, alpha, spending_function, total_sample)

    n_c_total = len(control_data)
    n_t_total = len(treatment_data)
    looks = []
    stopped_early = False
    final_decision = "continue"

    for boundary in boundaries:
        if stopped_early:
            break

        # Fraction of data to use at this look
        frac = boundary.information_fraction
        n_c = max(10, int(n_c_total * frac))
        n_t = max(10, int(n_t_total * frac))
        c_data = control_data[:n_c]
        t_data = treatment_data[:n_t]

        # Compute z-statistic
        c_mean = c_data.mean()
        t_mean = t_data.mean()

        if metric_type == "binary":
            p_pool = (c_data.sum() + t_data.sum()) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        else:
            se = np.sqrt(c_data.var()/n_c + t_data.var()/n_t)

        z_stat = (t_mean - c_mean) / se if se > 0 else 0
        p_value = 2 * stats.norm.sf(abs(z_stat))

        # Decision
        if abs(z_stat) >= boundary.z_upper:
            decision = "stop_efficacy"
            stopped_early = boundary.look_number < n_looks
            final_decision = "significant"
        elif z_stat <= boundary.z_lower and boundary.look_number < n_looks:
            decision = "stop_futility"
            stopped_early = True
            final_decision = "futile"
        else:
            decision = "continue"

        looks.append(SequentialLook(
            look_number=boundary.look_number,
            n_control=n_c,
            n_treatment=n_t,
            z_statistic=round(float(z_stat), 4),
            p_value=round(float(p_value), 6),
            boundary=boundary,
            decision=decision,
            control_mean=round(float(c_mean), 6),
            treatment_mean=round(float(t_mean), 6),
        ))

    if final_decision == "continue":
        final_decision = "not significant (ran to completion)"

    elapsed = time.perf_counter() - t0

    result = SequentialResult(
        boundaries=boundaries,
        looks=looks,
        spending_function=spending_function,
        total_alpha=alpha,
        max_looks=n_looks,
        stopped_early=stopped_early,
        final_decision=final_decision,
        final_look=looks[-1].look_number if looks else 0,
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result
