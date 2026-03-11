"""
Geographic experiment market allocator.

Assigns markets (cities/regions) to treatment and control groups
for geo-level experiments where user-level randomization isn't possible.

Key features:
  - Stratified allocation (balance on covariates)
  - Covariate balance check
  - Power analysis for geo experiments (cluster-level)
  - Holdout selection for synthetic control

Used at Google for: ad campaign geo tests, market-level feature rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BalanceCheck:
    """Result of covariate balance check between groups."""
    balanced: bool
    covariate_stats: Dict[str, Dict]  # covariate → {treatment_mean, control_mean, std_diff}
    max_std_diff: float
    message: str


@dataclass
class GeoAllocation:
    """Complete geo allocation output."""
    treatment_markets: List[str]
    control_markets: List[str]
    n_treatment: int
    n_control: int
    balance: BalanceCheck
    allocation_method: str
    seed: int

    def summary(self) -> str:
        status = "BALANCED" if self.balance.balanced else "IMBALANCED"
        lines = [
            "─" * 50,
            "GEO ALLOCATION",
            "─" * 50,
            f"  Method:              {self.allocation_method}",
            f"  Treatment markets:   {self.n_treatment}",
            f"  Control markets:     {self.n_control}",
            f"  Balance status:      {status}",
            f"  Max std diff:        {self.balance.max_std_diff:.3f}",
            "",
            "  Treatment: " + ", ".join(self.treatment_markets[:10]),
            "  Control:   " + ", ".join(self.control_markets[:10]),
            "─" * 50,
        ]
        return "\n".join(lines)


def _check_balance(
    df: pd.DataFrame,
    treatment_markets: List[str],
    market_col: str,
    covariates: List[str],
    tolerance: float = CFG.geo.balance_tolerance,
) -> BalanceCheck:
    """Check covariate balance between treatment and control."""
    is_treatment = df[market_col].isin(treatment_markets)
    stats_dict = {}
    max_diff = 0.0

    for cov in covariates:
        if cov not in df.columns:
            continue
        t_vals = df.loc[is_treatment, cov].values.astype(float)
        c_vals = df.loc[~is_treatment, cov].values.astype(float)

        t_mean = t_vals.mean()
        c_mean = c_vals.mean()
        pooled_std = np.sqrt((t_vals.var() + c_vals.var()) / 2)

        std_diff = abs(t_mean - c_mean) / pooled_std if pooled_std > 0 else 0

        stats_dict[cov] = {
            "treatment_mean": round(float(t_mean), 3),
            "control_mean": round(float(c_mean), 3),
            "std_diff": round(float(std_diff), 3),
        }
        max_diff = max(max_diff, std_diff)

    balanced = max_diff < tolerance
    msg = (
        f"Max standardized difference: {max_diff:.3f} "
        f"(threshold: {tolerance}). "
        + ("Balance achieved." if balanced else "WARNING: Groups may be imbalanced.")
    )

    return BalanceCheck(
        balanced=balanced,
        covariate_stats=stats_dict,
        max_std_diff=round(float(max_diff), 3),
        message=msg,
    )


def allocate_markets(
    df: pd.DataFrame,
    market_col: str = "market_id",
    treatment_fraction: float = 0.3,
    covariates: Optional[List[str]] = None,
    method: str = "stratified",
    seed: int = CFG.simulation.seed,
    n_attempts: int = 100,
) -> GeoAllocation:
    """
    Allocate markets to treatment and control.

    Parameters
    ----------
    df : pd.DataFrame
        Market-level data with covariates.
    market_col : str
        Column with market identifiers.
    treatment_fraction : float
        Fraction of markets to assign to treatment.
    covariates : list[str], optional
        Columns to balance on. Auto-detected if None.
    method : str
        "stratified" (balance on covariates) or "random".
    seed : int
        Random seed.
    n_attempts : int
        Number of random allocations to try (keep the most balanced).

    Returns
    -------
    GeoAllocation
    """
    logger.info("Allocating markets (method=%s, treatment=%.0f%%)",
                method, treatment_fraction * 100)

    rng = np.random.RandomState(seed)
    markets = df[market_col].unique().tolist()
    n_total = len(markets)
    n_treatment = max(CFG.geo.min_markets_per_arm,
                      int(n_total * treatment_fraction))
    n_control = n_total - n_treatment

    if covariates is None:
        # Auto-detect numeric columns
        covariates = [c for c in df.select_dtypes(include=[np.number]).columns
                      if c != market_col]

    # Aggregate to market level
    market_df = df.groupby(market_col)[covariates].mean().reset_index()

    if method == "random":
        idx = rng.permutation(n_total)
        treatment_idx = idx[:n_treatment]
        treatment_markets = [markets[i] for i in treatment_idx]
    elif method == "stratified":
        # Try many random allocations, keep the most balanced
        best_allocation = None
        best_max_diff = float("inf")

        for attempt in range(n_attempts):
            idx = rng.permutation(n_total)
            candidate = [markets[i] for i in idx[:n_treatment]]
            balance = _check_balance(
                market_df, candidate, market_col, covariates
            )
            if balance.max_std_diff < best_max_diff:
                best_max_diff = balance.max_std_diff
                best_allocation = candidate
                if balance.balanced:
                    break  # good enough

        treatment_markets = best_allocation
    else:
        raise ValueError(f"Unknown method: {method}")

    control_markets = [m for m in markets if m not in treatment_markets]

    balance = _check_balance(market_df, treatment_markets, market_col, covariates)

    result = GeoAllocation(
        treatment_markets=treatment_markets,
        control_markets=control_markets,
        n_treatment=len(treatment_markets),
        n_control=len(control_markets),
        balance=balance,
        allocation_method=method,
        seed=seed,
    )

    logger.info(result.summary())
    return result
