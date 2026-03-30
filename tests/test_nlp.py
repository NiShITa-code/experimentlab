"""Tests for NLP sentiment analysis."""

import pytest
import numpy as np
import pandas as pd

from src.data.simulator import simulate_feedback_data
from src.nlp.feedback import (
    _keyword_sentiment,
    _extract_word_shifts,
    analyze_sentiment,
    SentimentShift,
)


@pytest.fixture
def feedback_positive_shift():
    return simulate_feedback_data(n_reviews=500, true_sentiment_shift=0.3)


@pytest.fixture
def feedback_no_shift():
    return simulate_feedback_data(n_reviews=500, true_sentiment_shift=0.0)


class TestKeywordSentiment:
    def test_positive_text(self):
        scores = _keyword_sentiment(["This is great and amazing"])
        assert scores[0] > 0

    def test_negative_text(self):
        scores = _keyword_sentiment(["This is terrible and awful"])
        assert scores[0] < 0

    def test_neutral_text(self):
        scores = _keyword_sentiment(["The sky is blue"])
        assert scores[0] == 0.0

    def test_returns_array(self):
        scores = _keyword_sentiment(["good", "bad", "neutral"])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3


class TestWordShifts:
    def test_returns_tuples(self):
        pre = ["the app is slow and buggy"]
        post = ["the app is fast and great"]
        pos, neg = _extract_word_shifts(pre, post)
        assert isinstance(pos, list)
        assert isinstance(neg, list)

    def test_shifted_words_detected(self):
        pre = ["slow slow slow"] * 100
        post = ["fast fast fast"] * 100
        pos, neg = _extract_word_shifts(pre, post)
        pos_words = [w for w, _ in pos]
        neg_words = [w for w, _ in neg]
        assert "fast" in pos_words
        assert "slow" in neg_words


class TestAnalyzeSentiment:
    def test_returns_result(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        assert isinstance(result, SentimentShift)

    def test_positive_shift_detected(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        assert result.sentiment_lift > 0

    def test_no_shift_small_lift(self, feedback_no_shift):
        result = analyze_sentiment(feedback_no_shift)
        assert abs(result.sentiment_lift) < 0.3

    def test_has_percentages(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        assert 0 <= result.pre_positive_pct <= 100
        assert 0 <= result.post_positive_pct <= 100
        assert 0 <= result.pre_negative_pct <= 100
        assert 0 <= result.post_negative_pct <= 100

    def test_has_p_value(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        assert 0 <= result.p_value <= 1

    def test_counts_correct(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        assert result.n_pre + result.n_post == len(feedback_positive_shift)

    def test_summary_string(self, feedback_positive_shift):
        result = analyze_sentiment(feedback_positive_shift)
        summary = result.summary()
        assert "SENTIMENT ANALYSIS" in summary
