"""Stable cache targets for expensive, deterministic notebook operations."""

from __future__ import annotations


def fit_estimator(estimator, features, target):
    """Fit and return an estimator; designed for ``joblib.Memory.cache``."""
    return estimator.fit(features, target)
