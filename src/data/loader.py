"""
CSV data loader with validation.

Lets users bring their own experiment data.
Validates column requirements and provides helpful error messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataRequirement:
    """Describes what columns a tab needs."""
    required_cols: List[str]
    optional_cols: List[str] = field(default_factory=list)
    description: str = ""
    example_format: str = ""


# ── Column specs for each analysis type ──────────────────────

AB_TEST_REQUIREMENTS = DataRequirement(
    required_cols=["group", "converted"],
    optional_cols=["revenue", "segment", "user_id", "signup_day"],
    description="A/B test data: one row per user with group assignment and outcome.",
    example_format=(
        "user_id,group,converted,revenue,segment\n"
        "u001,0,1,12.50,new\n"
        "u002,1,0,0.00,returning\n"
        "u003,0,1,8.75,power"
    ),
)

GEO_EXPERIMENT_REQUIREMENTS = DataRequirement(
    required_cols=["market_id", "date", "metric_value", "is_treatment"],
    optional_cols=["post_treatment"],
    description="Geo experiment data: one row per market × date with metric value.",
    example_format=(
        "market_id,date,metric_value,is_treatment,post_treatment\n"
        "NYC,2025-01-01,1250.5,1,0\n"
        "LAX,2025-01-01,980.3,0,0\n"
        "NYC,2025-01-02,1300.2,1,0"
    ),
)

FEEDBACK_REQUIREMENTS = DataRequirement(
    required_cols=["text", "period"],
    optional_cols=["rating", "user_id", "date"],
    description="Feedback data: one row per review with text and pre/post period.",
    example_format=(
        "text,period,rating\n"
        '"Great app, love the new feature!",post,5\n'
        '"Too slow, keeps crashing",pre,2\n'
        '"Decent experience overall",pre,3'
    ),
)

PRIVACY_REQUIREMENTS = DataRequirement(
    required_cols=["group"],
    optional_cols=["segment", "converted", "revenue"],
    description="Data with group columns for k-anonymity checking.",
    example_format=(
        "group,segment,converted,revenue\n"
        "0,new,1,12.50\n"
        "1,returning,0,0.00"
    ),
)


def validate_upload(
    df: pd.DataFrame,
    requirement: DataRequirement,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate uploaded CSV against requirements.

    Returns
    -------
    (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Check required columns
    missing = [c for c in requirement.required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    # Check optional columns
    available_optional = [c for c in requirement.optional_cols if c in df.columns]
    missing_optional = [c for c in requirement.optional_cols if c not in df.columns]
    if missing_optional:
        warnings.append(
            f"Optional columns not found (features will be limited): "
            f"{', '.join(missing_optional)}"
        )

    # Check for empty data
    if len(df) == 0:
        errors.append("Uploaded file is empty")

    # Check for too few rows
    if len(df) < 10:
        warnings.append(f"Only {len(df)} rows — results may be unreliable")

    # Check for nulls in required columns
    for col in requirement.required_cols:
        if col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 50:
                errors.append(f"Column '{col}' is {null_pct:.0f}% null")
            elif null_pct > 5:
                warnings.append(f"Column '{col}' has {null_pct:.1f}% missing values")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def load_and_validate(
    uploaded_file,
    requirement: DataRequirement,
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Load a CSV file and validate it.

    Parameters
    ----------
    uploaded_file : UploadedFile (from Streamlit)
    requirement : DataRequirement

    Returns
    -------
    (df or None, errors, warnings)
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, [f"Could not read CSV: {str(e)}"], []

    is_valid, errors, warnings = validate_upload(df, requirement)

    if is_valid:
        logger.info("Loaded %d rows × %d columns from uploaded file", len(df), len(df.columns))
        return df, errors, warnings
    else:
        return None, errors, warnings
