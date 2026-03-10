"""
Typed data contracts for experiments.

Every data structure flowing through ExperimentLab is defined here.
Validation happens at the boundary — bad data fails loudly before
reaching any analysis method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ExperimentType(Enum):
    AB_TEST = "ab_test"
    GEO_EXPERIMENT = "geo_experiment"
    BEFORE_AFTER = "before_after"


class MetricType(Enum):
    BINARY = "binary"         # conversion (0/1)
    CONTINUOUS = "continuous"  # revenue, time-on-page
    RATIO = "ratio"           # conversion rate, CTR


@dataclass
class MetricDefinition:
    """Defines what we're measuring and how."""
    name: str
    metric_type: MetricType
    numerator_col: str                # column for numerator
    denominator_col: Optional[str]    # column for denominator (ratio metrics)
    direction: str = "increase"       # "increase" or "decrease" is good
    min_detectable_effect: float = 0.0
    description: str = ""


@dataclass
class ExperimentConfig:
    """Defines an experiment's structure."""
    name: str
    experiment_type: ExperimentType
    unit_col: str                     # user_id, city_id, etc.
    group_col: str                    # treatment/control column
    timestamp_col: Optional[str]      # for time-series methods
    treatment_start: Optional[str]    # when treatment began (geo/before-after)
    primary_metric: MetricDefinition
    guardrail_metrics: List[MetricDefinition] = field(default_factory=list)
    alpha: float = 0.05
    power: float = 0.80


@dataclass
class ABTestData:
    """Validated A/B test dataset."""
    data: pd.DataFrame
    config: ExperimentConfig
    n_control: int
    n_treatment: int
    n_total: int


@dataclass
class GeoExperimentData:
    """Validated geo experiment dataset."""
    data: pd.DataFrame              # long format: unit × time
    config: ExperimentConfig
    treatment_units: List[str]
    control_units: List[str]
    pre_period: pd.DatetimeIndex
    post_period: pd.DatetimeIndex
    n_pre_periods: int
    n_post_periods: int


@dataclass
class SimulationResult:
    """Output of any data simulator."""
    data: pd.DataFrame
    experiment_type: ExperimentType
    true_effect: float              # ground truth for validation
    n_units: int
    config: dict
    elapsed_seconds: float


# ── Validation ──────────────────────────────────────────────


def validate_ab_data(df: pd.DataFrame, config: ExperimentConfig) -> ABTestData:
    """Validate an A/B test DataFrame against its config."""
    errors = []

    if config.unit_col not in df.columns:
        errors.append(f"Missing unit column: {config.unit_col}")
    if config.group_col not in df.columns:
        errors.append(f"Missing group column: {config.group_col}")
    if config.primary_metric.numerator_col not in df.columns:
        errors.append(f"Missing metric column: {config.primary_metric.numerator_col}")

    if errors:
        raise ValueError("AB test validation failed:\n" + "\n".join(errors))

    groups = df[config.group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(groups)}: {groups}")

    # Identify control/treatment (0/1 or control/treatment)
    group_vals = sorted(df[config.group_col].unique())
    control_mask = df[config.group_col] == group_vals[0]

    if df[config.unit_col].duplicated().any():
        raise ValueError("Duplicate unit IDs found")

    if df[config.primary_metric.numerator_col].isna().any():
        raise ValueError("Null values in primary metric column")

    return ABTestData(
        data=df,
        config=config,
        n_control=int(control_mask.sum()),
        n_treatment=int((~control_mask).sum()),
        n_total=len(df),
    )


def validate_geo_data(
    df: pd.DataFrame,
    config: ExperimentConfig,
    treatment_units: List[str],
) -> GeoExperimentData:
    """Validate a geo experiment DataFrame."""
    errors = []

    for col in [config.unit_col, config.timestamp_col,
                config.primary_metric.numerator_col]:
        if col and col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        raise ValueError("Geo experiment validation failed:\n" + "\n".join(errors))

    if config.treatment_start is None:
        raise ValueError("treatment_start is required for geo experiments")

    df = df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    treatment_date = pd.to_datetime(config.treatment_start)

    all_units = df[config.unit_col].unique().tolist()
    control_units = [u for u in all_units if u not in treatment_units]

    pre_mask = df[config.timestamp_col] < treatment_date
    post_mask = df[config.timestamp_col] >= treatment_date

    return GeoExperimentData(
        data=df,
        config=config,
        treatment_units=treatment_units,
        control_units=control_units,
        pre_period=df.loc[pre_mask, config.timestamp_col].unique(),
        post_period=df.loc[post_mask, config.timestamp_col].unique(),
        n_pre_periods=int(pre_mask.groupby(df[config.unit_col]).sum().iloc[0]) if pre_mask.any() else 0,
        n_post_periods=int(post_mask.groupby(df[config.unit_col]).sum().iloc[0]) if post_mask.any() else 0,
    )
