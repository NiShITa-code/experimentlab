"""Structured logging with file handler."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(ARTIFACTS_DIR / "experimentlab.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
