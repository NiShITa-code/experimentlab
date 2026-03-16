"""
Metric definition and validation framework.

Google DS responsibility #1: define what to measure.
This module lets users:
  - Define metrics with clear semantics (ratio, binary, continuous)
  - Validate that metrics are computable from the data
  - Check sensitivity (can we detect a meaningful change?)
  - Set guardrail metrics (things that shouldn't degrade)

Demonstrates metric design thinking — not just analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.schema import MetricDefinition, MetricType
from src.design.power import compute_sample_size_binary, compute_sample_size_continuous
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValidation:
    """Result of validating a metric against data."""
    metric_name: str
    valid: bool
    errors: List[str]
    warnings: List[str]
    observed_mean: Optional[float]
    observed_std: Optional[float]
    n_observations: int
    n_missing: int
    sensitivity: Optional[float]    # min detectable effect at 80% power


@dataclass
class MetricSuite:
    """Complete set of experiment metrics with validation."""
    primary: MetricValidation
    guardrails: List[MetricValidation]
    all_valid: bool
    summary_text: str


def validate_metric(
    df: pd.DataFrame,
    metric: MetricDefinition,
    sample_size: int = 5000,
    alpha: float = 0.05,
    power: float = 0.80,
) -> MetricValidation:
    """
    Validate a metric definition against actual data.

    Checks:
      1. Required columns exist
      2. No excessive nulls
      3. Correct data types
      4. Computes sensitivity (MDE at given sample size)

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data.
    metric : MetricDefinition
        The metric to validate.
    sample_size : int
        Expected per-arm sample size (for sensitivity).
    alpha, power : float
        For sensitivity calculation.

    Returns
    -------
    MetricValidation
    """
    errors = []
    warnings = []

    # Column existence
    if metric.numerator_col not in df.columns:
        errors.append(f"Column '{metric.numerator_col}' not found in data")
        return MetricValidation(
            metric_name=metric.name, valid=False, errors=errors,
            warnings=warnings, observed_mean=None, observed_std=None,
            n_observations=0, n_missing=0, sensitivity=None,
        )

    if metric.metric_type == MetricType.RATIO and metric.denominator_col:
        if metric.denominator_col not in df.columns:
            errors.append(f"Denominator column '{metric.denominator_col}' not found")

    col = df[metric.numerator_col]
    n_total = len(col)
    n_missing = int(col.isna().sum())

    if n_missing > 0.1 * n_total:
        warnings.append(f"{n_missing}/{n_total} values missing ({n_missing/n_total:.0%})")
    if n_missing == n_total:
        errors.append("All values are null")
        return MetricValidation(
            metric_name=metric.name, valid=False, errors=errors,
            warnings=warnings, observed_mean=None, observed_std=None,
            n_observations=n_total, n_missing=n_missing, sensitivity=None,
        )

    # Compute stats
    values = col.dropna().values.astype(float)
    obs_mean = float(values.mean())
    obs_std = float(values.std())

    # Binary validation
    if metric.metric_type == MetricType.BINARY:
        unique = set(values)
        if not unique.issubset({0, 0.0, 1, 1.0}):
            errors.append(f"Binary metric has non-binary values: {unique - {0, 1}}")

    # Sensitivity (MDE at given sample size)
    try:
        if metric.metric_type == MetricType.BINARY:
            from src.design.power import compute_mde
            sensitivity = compute_mde(sample_size, obs_mean, metric_type="binary")
        else:
            from src.design.power import compute_mde
            sensitivity = compute_mde(sample_size, obs_mean, obs_std, metric_type="continuous")
    except Exception:
        sensitivity = None

    valid = len(errors) == 0

    return MetricValidation(
        metric_name=metric.name,
        valid=valid,
        errors=errors,
        warnings=warnings,
        observed_mean=round(obs_mean, 4),
        observed_std=round(obs_std, 4),
        n_observations=n_total,
        n_missing=n_missing,
        sensitivity=sensitivity,
    )


def validate_metric_suite(
    df: pd.DataFrame,
    primary: MetricDefinition,
    guardrails: List[MetricDefinition],
    sample_size: int = 5000,
) -> MetricSuite:
    """
    Validate a complete metric suite for an experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data.
    primary : MetricDefinition
        Primary success metric.
    guardrails : list[MetricDefinition]
        Metrics that should not degrade.
    sample_size : int
        Expected per-arm sample size.

    Returns
    -------
    MetricSuite
    """
    logger.info("Validating metric suite: primary='%s', %d guardrails",
                primary.name, len(guardrails))

    primary_val = validate_metric(df, primary, sample_size)
    guardrail_vals = [validate_metric(df, g, sample_size) for g in guardrails]

    all_valid = primary_val.valid and all(g.valid for g in guardrail_vals)

    lines = [
        "─" * 55,
        "METRIC SUITE VALIDATION",
        "─" * 55,
        f"  PRIMARY: {primary.name}",
        f"    Valid: {primary_val.valid}",
        f"    Mean: {primary_val.observed_mean}",
        f"    MDE (at n={sample_size:,}): {primary_val.sensitivity}",
    ]
    for gv in guardrail_vals:
        status = "✓" if gv.valid else "✗"
        lines.append(f"  GUARDRAIL: {gv.metric_name} [{status}] mean={gv.observed_mean}")

    summary = "\n".join(lines)

    return MetricSuite(
        primary=primary_val,
        guardrails=guardrail_vals,
        all_valid=all_valid,
        summary_text=summary,
    )
