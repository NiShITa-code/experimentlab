"""
NLP sentiment analysis for pre/post experiment feedback.

Measures whether an intervention changed how users feel — not just
what they do. Sentiment is a leading indicator that often moves
before behavioural metrics.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for
lightweight, no-GPU sentiment analysis. Works out of the box
without downloading large models.

Reference: Hutto & Gilbert (2014), ICWSM
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentShift:
    """Result of pre/post sentiment comparison."""
    pre_mean_sentiment: float
    post_mean_sentiment: float
    sentiment_lift: float
    p_value: float
    significant: bool

    pre_positive_pct: float
    post_positive_pct: float
    pre_negative_pct: float
    post_negative_pct: float

    n_pre: int
    n_post: int

    top_positive_shift_words: List[Tuple[str, float]]
    top_negative_shift_words: List[Tuple[str, float]]

    elapsed_seconds: float

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        direction = "improved" if self.sentiment_lift > 0 else "worsened"
        lines = [
            "═" * 55,
            "SENTIMENT ANALYSIS",
            "═" * 55,
            f"  Pre-treatment sentiment:   {self.pre_mean_sentiment:+.3f}",
            f"  Post-treatment sentiment:  {self.post_mean_sentiment:+.3f}",
            f"  Sentiment shift:           {self.sentiment_lift:+.3f} ({direction})",
            f"  p-value:                   {self.p_value:.4f} ({sig})",
            "",
            f"  Pre:  {self.pre_positive_pct:.0f}% positive, {self.pre_negative_pct:.0f}% negative (n={self.n_pre})",
            f"  Post: {self.post_positive_pct:.0f}% positive, {self.post_negative_pct:.0f}% negative (n={self.n_post})",
            "═" * 55,
        ]
        return "\n".join(lines)


def _get_vader_scores(texts: List[str]) -> np.ndarray:
    """Compute VADER compound sentiment scores."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return np.array([analyzer.polarity_scores(t)["compound"] for t in texts])
    except ImportError:
        # Fallback: simple keyword-based scoring
        logger.warning("vaderSentiment not installed. Using keyword fallback.")
        return _keyword_sentiment(texts)


def _keyword_sentiment(texts: List[str]) -> np.ndarray:
    """Simple keyword-based sentiment as VADER fallback."""
    positive_words = {
        "great", "excellent", "love", "best", "smooth", "fast", "easy",
        "amazing", "perfect", "reliable", "satisfied", "improved", "good",
        "intuitive", "wonderful", "fantastic",
    }
    negative_words = {
        "terrible", "worst", "hate", "slow", "buggy", "crash", "confusing",
        "awful", "horrible", "broken", "unhelpful", "frustrating", "bad",
        "annoying", "useless", "disappointing",
    }

    scores = []
    for text in texts:
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        if total == 0:
            scores.append(0.0)
        else:
            scores.append((pos - neg) / total)

    return np.array(scores)


def _extract_word_shifts(
    pre_texts: List[str],
    post_texts: List[str],
    top_n: int = 10,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Find words that shifted most between pre and post."""
    from collections import Counter

    def word_freq(texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        total = sum(counter.values())
        return {w: c/total for w, c in counter.items()}

    pre_freq = word_freq(pre_texts)
    post_freq = word_freq(post_texts)

    all_words = set(pre_freq.keys()) | set(post_freq.keys())

    shifts = {}
    for w in all_words:
        if len(w) < 3:  # skip tiny words
            continue
        shifts[w] = post_freq.get(w, 0) - pre_freq.get(w, 0)

    positive_shifts = sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    negative_shifts = sorted(shifts.items(), key=lambda x: x[1])[:top_n]

    return positive_shifts, negative_shifts


def analyze_sentiment(
    df: pd.DataFrame,
    text_col: str = "text",
    period_col: str = "period",
    pre_label: str = "pre",
    post_label: str = "post",
    alpha: float = 0.05,
) -> SentimentShift:
    """
    Compare sentiment between pre and post periods.

    Parameters
    ----------
    df : pd.DataFrame
        Reviews/feedback with text and period columns.
    text_col : str
        Column containing review text.
    period_col : str
        Column indicating pre/post period.
    pre_label, post_label : str
        Values in period_col for each period.
    alpha : float
        Significance level.

    Returns
    -------
    SentimentShift
    """
    logger.info("Analyzing sentiment shift...")
    t0 = time.perf_counter()

    pre_mask = df[period_col] == pre_label
    post_mask = df[period_col] == post_label

    pre_texts = df.loc[pre_mask, text_col].tolist()
    post_texts = df.loc[post_mask, text_col].tolist()

    if len(pre_texts) < CFG.nlp.min_reviews or len(post_texts) < CFG.nlp.min_reviews:
        logger.warning("Insufficient reviews for analysis (need %d per period)",
                       CFG.nlp.min_reviews)

    # Compute sentiment scores
    pre_scores = _get_vader_scores(pre_texts)
    post_scores = _get_vader_scores(post_texts)

    # Statistical comparison (Welch's t-test on sentiment scores)
    t_stat, p_value = stats.ttest_ind(post_scores, pre_scores, equal_var=False)

    # Sentiment categories
    pre_pos = (pre_scores > 0.05).mean() * 100
    pre_neg = (pre_scores < -0.05).mean() * 100
    post_pos = (post_scores > 0.05).mean() * 100
    post_neg = (post_scores < -0.05).mean() * 100

    # Word shifts
    pos_shifts, neg_shifts = _extract_word_shifts(pre_texts, post_texts)

    elapsed = time.perf_counter() - t0

    result = SentimentShift(
        pre_mean_sentiment=round(float(pre_scores.mean()), 4),
        post_mean_sentiment=round(float(post_scores.mean()), 4),
        sentiment_lift=round(float(post_scores.mean() - pre_scores.mean()), 4),
        p_value=round(float(p_value), 4),
        significant=p_value < alpha,
        pre_positive_pct=round(float(pre_pos), 1),
        post_positive_pct=round(float(post_pos), 1),
        pre_negative_pct=round(float(pre_neg), 1),
        post_negative_pct=round(float(post_neg), 1),
        n_pre=len(pre_texts),
        n_post=len(post_texts),
        top_positive_shift_words=pos_shifts,
        top_negative_shift_words=neg_shifts,
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result
