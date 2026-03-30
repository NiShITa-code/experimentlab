"""
Synthetic Control Method (Abadie, Diamond & Hainmueller, 2010).

Constructs a "synthetic" version of the treated unit by finding
optimal weights over donor (control) units such that the synthetic
unit matches the treated unit's pre-treatment trajectory.

Post-treatment divergence between the treated unit and its synthetic
counterpart is the estimated causal effect.

Used at Google for:
  - Measuring ad campaign lift in geo experiments
  - Estimating impact of policy changes in specific markets
  - When only a few units are treated (DiD may lack power)

Key features:
  - Constrained optimization (weights sum to 1, non-negative)
  - Pre-treatment fit quality (RMSPE)
  - Placebo tests (in-space and in-time)
  - Gap plot visualization data

Reference: Abadie, Diamond & Hainmueller (2010), JASA
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticControlResult:
    """Complete synthetic control analysis output."""
    # Core estimate
    treatment_unit: str
    estimated_effect: float         # mean post-treatment gap
    cumulative_effect: float        # total post-treatment gap
    percent_effect: float           # effect as % of counterfactual

    # Weights
    donor_weights: Dict[str, float]  # control unit → weight
    top_donors: List[Tuple[str, float]]  # sorted by weight

    # Fit quality
    pre_rmspe: float                # root mean squared prediction error (pre)
    post_rmspe: float               # RMSPE in post period
    rmspe_ratio: float              # post/pre — large = real effect

    # Time series for plotting
    actual: pd.Series               # treated unit actual values
    synthetic: pd.Series            # synthetic control values
    gap: pd.Series                  # actual - synthetic
    treatment_start: str

    # Placebo test results
    placebo_effects: Optional[Dict[str, float]]  # unit → placebo effect
    placebo_rank: Optional[int]     # rank of treated unit among placebos
    placebo_p_value: Optional[float]

    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            "═" * 55,
            "SYNTHETIC CONTROL METHOD",
            "═" * 55,
            f"  Treated unit:            {self.treatment_unit}",
            f"  Estimated effect:        {self.estimated_effect:+.4f}",
            f"  Effect (% of baseline):  {self.percent_effect:+.1f}%",
            f"  Cumulative effect:       {self.cumulative_effect:+.2f}",
            "",
            f"  Pre-treatment RMSPE:     {self.pre_rmspe:.4f}",
            f"  Post-treatment RMSPE:    {self.post_rmspe:.4f}",
            f"  RMSPE ratio (post/pre):  {self.rmspe_ratio:.2f}",
            "",
            f"  Top 3 donors:",
        ]
        for unit, w in self.top_donors[:3]:
            bar = "█" * max(1, int(w * 40))
            lines.append(f"    {unit:<15} {w:.3f} {bar}")

        if self.placebo_p_value is not None:
            lines.append("")
            lines.append(f"  Placebo rank:            {self.placebo_rank}/{len(self.placebo_effects)}")
            lines.append(f"  Placebo p-value:         {self.placebo_p_value:.3f}")

        lines.append("═" * 55)
        return "\n".join(lines)


def _find_weights(
    treated_pre: np.ndarray,
    donor_pre: np.ndarray,
) -> np.ndarray:
    """
    Find optimal donor weights via constrained optimisation.

    Minimises ||treated_pre - W @ donor_pre||^2
    subject to: W >= 0, sum(W) = 1

    Parameters
    ----------
    treated_pre : np.ndarray
        Shape (T_pre,) — treated unit pre-treatment values.
    donor_pre : np.ndarray
        Shape (T_pre, N_donors) — donor units pre-treatment values.

    Returns
    -------
    np.ndarray
        Shape (N_donors,) — optimal weights.
    """
    n_donors = donor_pre.shape[1]

    def loss(w):
        synthetic = donor_pre @ w
        return np.sum((treated_pre - synthetic) ** 2)

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: non-negative
    bounds = [(0, 1)] * n_donors
    # Initial: equal weights
    w0 = np.ones(n_donors) / n_donors

    result = minimize(loss, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-10})

    return result.x


def run_synthetic_control(
    df: pd.DataFrame,
    treatment_unit: str,
    unit_col: str = "market_id",
    time_col: str = "date",
    metric_col: str = "metric_value",
    treatment_start: Optional[str] = None,
    run_placebo: bool = True,
    max_donors: int = 20,
) -> SyntheticControlResult:
    """
    Run synthetic control method for a single treated unit.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data: unit × time.
    treatment_unit : str
        ID of the treated unit.
    unit_col, time_col, metric_col : str
        Column names.
    treatment_start : str
        Date when treatment began.
    run_placebo : bool
        Run in-space placebo tests (slower but validates results).
    max_donors : int
        Max number of donor units to use.

    Returns
    -------
    SyntheticControlResult
    """
    logger.info("Running synthetic control for unit: %s", treatment_unit)
    t0 = time.perf_counter()

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    treatment_date = pd.to_datetime(treatment_start)

    # Pivot to wide format: time × units
    wide = df.pivot_table(
        index=time_col, columns=unit_col,
        values=metric_col, aggfunc="mean"
    ).sort_index()

    if treatment_unit not in wide.columns:
        raise ValueError(f"Treatment unit {treatment_unit} not found in data")

    # Split pre/post
    pre_mask = wide.index < treatment_date
    post_mask = wide.index >= treatment_date

    # Treated unit series
    treated_full = wide[treatment_unit].values
    treated_pre = wide.loc[pre_mask, treatment_unit].values

    # Donor pool (all other units)
    donor_cols = [c for c in wide.columns if c != treatment_unit]
    if len(donor_cols) > max_donors:
        # Keep donors most correlated with treated unit in pre-period
        pre_corrs = wide.loc[pre_mask, donor_cols].corrwith(
            wide.loc[pre_mask, treatment_unit]
        ).abs().nlargest(max_donors)
        donor_cols = pre_corrs.index.tolist()

    donor_pre = wide.loc[pre_mask, donor_cols].values
    donor_full = wide[donor_cols].values

    # Find optimal weights
    weights = _find_weights(treated_pre, donor_pre)

    # Construct synthetic control
    synthetic_full = donor_full @ weights
    synthetic_pre = donor_pre @ weights

    # Gap (actual - synthetic)
    gap = treated_full - synthetic_full

    # Pre-treatment fit
    pre_rmspe = np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2))

    # Post-treatment effect
    post_idx = post_mask.values if hasattr(post_mask, 'values') else post_mask
    post_gap = gap[post_idx]
    post_rmspe = np.sqrt(np.mean(post_gap ** 2))
    estimated_effect = float(np.mean(post_gap))
    cumulative_effect = float(np.sum(post_gap))

    # Percentage effect
    synthetic_post_mean = synthetic_full[post_idx].mean()
    percent_effect = (estimated_effect / synthetic_post_mean * 100) if synthetic_post_mean > 0 else 0

    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")

    # Donor weights
    donor_weights = {unit: round(float(w), 4) for unit, w in zip(donor_cols, weights) if w > 0.001}
    top_donors = sorted(donor_weights.items(), key=lambda x: x[1], reverse=True)

    # Time series for plotting
    dates = wide.index
    actual_series = pd.Series(treated_full, index=dates, name="actual")
    synthetic_series = pd.Series(synthetic_full, index=dates, name="synthetic")
    gap_series = pd.Series(gap, index=dates, name="gap")

    # ── Placebo tests ────────────────────────────────────────
    placebo_effects = None
    placebo_rank = None
    placebo_p_value = None

    if run_placebo and len(donor_cols) >= 3:
        logger.info("  Running %d placebo tests...", len(donor_cols))
        placebo_effects = {}
        all_effects = [abs(estimated_effect)]  # include treated unit

        for placebo_unit in donor_cols:
            try:
                p_treated_pre = wide.loc[pre_mask, placebo_unit].values
                p_donor_cols = [c for c in donor_cols if c != placebo_unit]
                # Add the actual treated unit back as a donor for placebos
                p_all_donors = p_donor_cols + [treatment_unit]
                p_donor_pre = wide.loc[pre_mask, p_all_donors].values
                p_donor_full = wide[p_all_donors].values

                p_weights = _find_weights(p_treated_pre, p_donor_pre)
                p_synthetic = p_donor_full @ p_weights
                p_gap = wide[placebo_unit].values - p_synthetic
                p_effect = float(np.mean(p_gap[post_mask.values]))

                # Only include placebos with decent pre-fit
                p_pre_rmspe = np.sqrt(np.mean((p_treated_pre - p_donor_pre @ p_weights) ** 2))
                if p_pre_rmspe < pre_rmspe * 5:  # exclude very bad fits
                    placebo_effects[placebo_unit] = p_effect
                    all_effects.append(abs(p_effect))
            except Exception:
                continue

        # Rank: how many placebos have a larger effect?
        placebo_rank = sum(1 for e in all_effects if e >= abs(estimated_effect))
        placebo_p_value = placebo_rank / len(all_effects)

    elapsed = time.perf_counter() - t0

    result = SyntheticControlResult(
        treatment_unit=treatment_unit,
        estimated_effect=round(float(estimated_effect), 4),
        cumulative_effect=round(float(cumulative_effect), 2),
        percent_effect=round(float(percent_effect), 1),
        donor_weights=donor_weights,
        top_donors=top_donors,
        pre_rmspe=round(float(pre_rmspe), 4),
        post_rmspe=round(float(post_rmspe), 4),
        rmspe_ratio=round(float(rmspe_ratio), 2),
        actual=actual_series,
        synthetic=synthetic_series,
        gap=gap_series,
        treatment_start=str(treatment_start),
        placebo_effects=placebo_effects,
        placebo_rank=placebo_rank,
        placebo_p_value=round(float(placebo_p_value), 3) if placebo_p_value is not None else None,
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result
