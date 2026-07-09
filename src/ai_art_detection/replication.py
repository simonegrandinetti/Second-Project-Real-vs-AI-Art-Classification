from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .evaluation import binary_metrics

BOOTSTRAP_STRATA = ("source_label", "style_label")


def split_path_hash(frame: pd.DataFrame) -> str:
    """Hash sorted image paths so split membership can be audited."""
    payload = "\n".join(sorted(frame["image_path"].astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratified_bootstrap_scores(
    predictions: pd.DataFrame,
    *,
    metric: str = "f1",
    n_resamples: int = 1_000,
    seed: int = 4242,
    threshold: float = 0.5,
    strata: Sequence[str] = BOOTSTRAP_STRATA,
) -> np.ndarray:
    """Bootstrap a binary metric while preserving source/style composition."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    required = {"label", "logit", *strata}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    groups = [
        part.index.to_numpy()
        for _, part in predictions.groupby(list(strata), sort=True)
    ]
    if not groups:
        raise ValueError("Predictions must not be empty.")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        metrics = binary_metrics(
            predictions.loc[sampled, "label"].to_numpy(),
            predictions.loc[sampled, "logit"].to_numpy(),
            threshold=threshold,
        )
        if metric not in metrics:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        scores[index] = metrics[metric]
    return scores


def percentile_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    alpha = (1 - confidence) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)


def f1_audit_intervals(
    train_predictions: pd.DataFrame,
    original_predictions: pd.DataFrame,
    replication_predictions: pd.DataFrame,
    *,
    n_resamples: int = 1_000,
    seed: int = 4242,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return replication F1 and independent-bootstrap gap intervals."""
    train_scores = stratified_bootstrap_scores(
        train_predictions,
        n_resamples=n_resamples,
        seed=seed,
        threshold=threshold,
    )
    original_scores = stratified_bootstrap_scores(
        original_predictions,
        n_resamples=n_resamples,
        seed=seed + 1,
        threshold=threshold,
    )
    replication_scores = stratified_bootstrap_scores(
        replication_predictions,
        n_resamples=n_resamples,
        seed=seed + 2,
        threshold=threshold,
    )
    replication_low, replication_high = percentile_interval(replication_scores)
    train_gap_low, train_gap_high = percentile_interval(
        train_scores - replication_scores
    )
    replication_delta_low, replication_delta_high = percentile_interval(
        replication_scores - original_scores
    )
    return {
        "replication_f1_ci_low": replication_low,
        "replication_f1_ci_high": replication_high,
        "train_replication_f1_gap_ci_low": train_gap_low,
        "train_replication_f1_gap_ci_high": train_gap_high,
        "replication_original_f1_delta_ci_low": replication_delta_low,
        "replication_original_f1_delta_ci_high": replication_delta_high,
    }
