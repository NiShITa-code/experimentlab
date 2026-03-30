"""Tests for Geo market allocator."""

import pytest
import numpy as np
import pandas as pd

from src.design.geo_allocator import (
    _check_balance,
    allocate_markets,
    GeoAllocation,
    BalanceCheck,
)


@pytest.fixture
def market_data():
    """Create sample market-level data."""
    np.random.seed(42)
    n_markets = 30
    df = pd.DataFrame({
        "market_id": [f"market_{i}" for i in range(n_markets)],
        "population": np.random.lognormal(10, 1, n_markets),
        "avg_revenue": np.random.normal(1000, 200, n_markets),
        "growth_rate": np.random.normal(0.05, 0.02, n_markets),
    })
    return df


class TestCheckBalance:
    def test_balanced_groups(self, market_data):
        # Split roughly in half — should be balanced by chance
        treatment = market_data.market_id[:15].tolist()
        result = _check_balance(
            market_data, treatment, "market_id",
            ["population", "avg_revenue", "growth_rate"],
        )
        assert isinstance(result, BalanceCheck)

    def test_imbalanced_groups(self, market_data):
        # Put only extreme markets in treatment
        sorted_df = market_data.sort_values("population", ascending=False)
        treatment = sorted_df.market_id[:5].tolist()  # top 5 by population
        result = _check_balance(
            market_data, treatment, "market_id", ["population"],
        )
        assert result.max_std_diff > 0


class TestAllocateMarkets:
    def test_returns_allocation(self, market_data):
        result = allocate_markets(market_data)
        assert isinstance(result, GeoAllocation)

    def test_all_markets_assigned(self, market_data):
        result = allocate_markets(market_data)
        total = result.n_treatment + result.n_control
        assert total == market_data.market_id.nunique()

    def test_no_overlap(self, market_data):
        result = allocate_markets(market_data)
        assert len(set(result.treatment_markets) & set(result.control_markets)) == 0

    def test_treatment_fraction(self, market_data):
        result = allocate_markets(market_data, treatment_fraction=0.3)
        expected = int(30 * 0.3)
        assert abs(result.n_treatment - expected) <= 2

    def test_stratified_balanced(self, market_data):
        result = allocate_markets(market_data, method="stratified", n_attempts=200)
        assert result.balance.max_std_diff < 1.0  # should be roughly balanced

    def test_random_method(self, market_data):
        result = allocate_markets(market_data, method="random")
        assert result.allocation_method == "random"
        assert result.n_treatment > 0

    def test_reproducible_with_seed(self, market_data):
        r1 = allocate_markets(market_data, seed=42)
        r2 = allocate_markets(market_data, seed=42)
        assert set(r1.treatment_markets) == set(r2.treatment_markets)

    def test_summary_string(self, market_data):
        result = allocate_markets(market_data)
        summary = result.summary()
        assert "GEO ALLOCATION" in summary
