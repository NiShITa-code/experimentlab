"""Tests for Sequential testing."""

import pytest
import numpy as np

from src.analysis.sequential import (
    _obrien_fleming_boundary,
    _pocock_boundary,
    compute_boundaries,
    run_sequential_analysis,
    SequentialBoundary,
    SequentialResult,
)


class TestBoundaryFunctions:
    def test_obf_decreases_over_time(self):
        """O'Brien-Fleming boundary should decrease as information increases."""
        b1 = _obrien_fleming_boundary(0.05, 0.25)
        b2 = _obrien_fleming_boundary(0.05, 0.50)
        b3 = _obrien_fleming_boundary(0.05, 1.00)
        assert b1 > b2 > b3

    def test_pocock_constant(self):
        """Pocock boundary is constant across looks."""
        b = _pocock_boundary(0.05, 4)
        assert b > 0

    def test_obf_boundary_positive(self):
        assert _obrien_fleming_boundary(0.05, 0.5) > 0


class TestComputeBoundaries:
    def test_returns_correct_count(self):
        boundaries = compute_boundaries(n_looks=5)
        assert len(boundaries) == 5

    def test_boundary_fields(self):
        boundaries = compute_boundaries(n_looks=3)
        b = boundaries[0]
        assert isinstance(b, SequentialBoundary)
        assert b.look_number == 1
        assert 0 < b.information_fraction <= 1

    def test_info_fraction_increases(self):
        boundaries = compute_boundaries(n_looks=4)
        fracs = [b.information_fraction for b in boundaries]
        assert fracs == sorted(fracs)
        assert fracs[-1] == 1.0

    def test_alpha_spent_increases(self):
        boundaries = compute_boundaries(n_looks=4)
        alphas = [b.alpha_spent for b in boundaries]
        assert alphas == sorted(alphas)

    def test_pocock_spending(self):
        boundaries = compute_boundaries(n_looks=4, spending_function="pocock")
        assert len(boundaries) == 4
        # Pocock: all z_upper should be equal
        z_uppers = [b.z_upper for b in boundaries]
        assert all(abs(z - z_uppers[0]) < 0.01 for z in z_uppers)


class TestSequentialAnalysis:
    def test_returns_result(self):
        np.random.seed(42)
        c = np.random.binomial(1, 0.10, 5000)
        t = np.random.binomial(1, 0.10, 5000)
        result = run_sequential_analysis(c, t, n_looks=4)
        assert isinstance(result, SequentialResult)

    def test_looks_conducted(self):
        np.random.seed(42)
        c = np.random.binomial(1, 0.10, 5000)
        t = np.random.binomial(1, 0.10, 5000)
        result = run_sequential_analysis(c, t, n_looks=4)
        assert len(result.looks) > 0
        assert len(result.looks) <= 4

    def test_early_stop_with_big_effect(self):
        """Very large effect should stop early."""
        np.random.seed(42)
        c = np.random.binomial(1, 0.10, 10000)
        t = np.random.binomial(1, 0.30, 10000)
        result = run_sequential_analysis(c, t, n_looks=5)
        assert result.stopped_early is True
        assert result.final_decision == "significant"

    def test_no_early_stop_aa_test(self):
        """A/A test shouldn't stop early for efficacy."""
        np.random.seed(42)
        c = np.random.binomial(1, 0.10, 5000)
        t = np.random.binomial(1, 0.10, 5000)
        result = run_sequential_analysis(c, t, n_looks=4)
        # Should either continue to end or stop for futility
        assert result.final_decision != "significant" or not result.stopped_early

    def test_summary_string(self):
        np.random.seed(42)
        c = np.random.binomial(1, 0.10, 5000)
        t = np.random.binomial(1, 0.10, 5000)
        result = run_sequential_analysis(c, t, n_looks=3)
        summary = result.summary()
        assert "SEQUENTIAL TESTING" in summary
