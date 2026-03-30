"""
Configuration loader.

Reads config/config.yaml into frozen dataclasses.
Single source of truth — no magic numbers in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    n_units: int
    n_geo_units: int
    n_time_periods: int
    true_effect: float
    baseline_rate: float
    noise_std: float


@dataclass(frozen=True)
class ExperimentConfig:
    default_alpha: float
    default_power: float
    default_mde: float
    srm_threshold: float
    sequential_looks: int


@dataclass(frozen=True)
class GeoConfig:
    min_markets_per_arm: int
    balance_tolerance: float
    synthetic_control: dict


@dataclass(frozen=True)
class BayesianConfig:
    prior_alpha: float
    prior_beta: float
    n_posterior_samples: int
    rope_width: float


@dataclass(frozen=True)
class DIDConfig:
    min_pre_periods: int
    min_post_periods: int
    robust_se: bool


@dataclass(frozen=True)
class PrivacyConfig:
    k_anonymity_threshold: int
    dp_epsilon: float
    dp_delta: float


@dataclass(frozen=True)
class NLPConfig:
    sentiment_model: str
    min_reviews: int


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    experiment: ExperimentConfig
    geo: GeoConfig
    bayesian: BayesianConfig
    diff_in_diff: DIDConfig
    privacy: PrivacyConfig
    nlp: NLPConfig


def _load_config(path: Optional[Path] = None) -> AppConfig:
    if path is None:
        path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        simulation=SimulationConfig(**raw["simulation"]),
        experiment=ExperimentConfig(**raw["experiment"]),
        geo=GeoConfig(
            min_markets_per_arm=raw["geo"]["min_markets_per_arm"],
            balance_tolerance=raw["geo"]["balance_tolerance"],
            synthetic_control=raw["geo"]["synthetic_control"],
        ),
        bayesian=BayesianConfig(**raw["bayesian"]),
        diff_in_diff=DIDConfig(**raw["diff_in_diff"]),
        privacy=PrivacyConfig(**raw["privacy"]),
        nlp=NLPConfig(**raw["nlp"]),
    )


CFG = _load_config()
