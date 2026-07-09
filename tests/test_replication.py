import numpy as np
import pandas as pd

from ai_art_detection.replication import (
    f1_audit_intervals,
    split_path_hash,
    stratified_bootstrap_scores,
)


def predictions(logits: list[float]) -> pd.DataFrame:
    labels = [0, 0, 1, 1]
    return pd.DataFrame(
        {
            "image_path": [f"/image/{index}.jpg" for index in range(4)],
            "label": labels,
            "logit": logits,
            "source_label": ["Human", "Human", "Latent_Diffusion", "Stable_Diffusion"],
            "style_label": ["Baroque"] * 4,
        }
    )


def test_stratified_bootstrap_is_deterministic():
    frame = predictions([-2.0, -1.0, 1.0, 2.0])
    first = stratified_bootstrap_scores(frame, n_resamples=50, seed=4242)
    second = stratified_bootstrap_scores(frame, n_resamples=50, seed=4242)
    assert np.array_equal(first, second)
    assert np.all(first == 1.0)


def test_f1_audit_intervals_and_split_hashes():
    train = predictions([-3.0, -2.0, 2.0, 3.0])
    original = predictions([-2.0, -1.0, 1.0, 2.0])
    replication = predictions([-2.0, 1.0, 1.0, 2.0])
    intervals = f1_audit_intervals(
        train,
        original,
        replication,
        n_resamples=100,
        seed=4242,
    )
    assert intervals["replication_f1_ci_low"] <= intervals[
        "replication_f1_ci_high"
    ]
    assert intervals["train_replication_f1_gap_ci_high"] >= 0
    assert split_path_hash(train) == split_path_hash(train.sample(frac=1))
