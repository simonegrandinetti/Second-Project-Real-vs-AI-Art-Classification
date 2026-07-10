import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from PIL import Image

from ai_art_detection.evaluation import (
    ROBUSTNESS_CONDITIONS,
    apply_robustness_perturbation,
    binary_metrics,
    evaluate_robustness,
    robustness_condition_label,
    robustness_transform,
    robustness_value_label,
    source_error_summary,
)
from ai_art_detection.gradcam import GradCAM
from ai_art_detection.models import (
    build_model,
    count_trainable_parameters,
    gradcam_target_layer,
)


def test_binary_metrics_are_correct():
    metrics = binary_metrics([0, 0, 1, 1], [-3, -2, 2, 3])
    assert metrics["accuracy"] == 1
    assert metrics["f1"] == 1
    assert metrics["roc_auc"] == 1


def test_mobilenet_binary_output_and_freezing():
    model = build_model("mobilenet_v2", mode="frozen", pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1)
    assert 0 < count_trainable_parameters(model) < sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_invalid_model_rejected():
    try:
        build_model("made_up_model", pretrained=False)
    except ValueError as error:
        assert "Unknown model" in str(error)
    else:
        raise AssertionError("Invalid model name should fail")


def test_source_error_summary_uses_class_appropriate_rates():
    predictions = pd.DataFrame(
        {
            "source_label": ["Human", "Human", "Latent_Diffusion"] * 2,
            "label": [0, 0, 1] * 2,
            "pred": [0, 1, 0, 0, 0, 1],
        }
    )
    summary = source_error_summary(predictions).set_index("source_label")
    assert summary.loc["Human", "error_type"] == "false_positive_rate"
    assert summary.loc["Human", "error_rate"] == 0.25
    assert summary.loc["Latent_Diffusion", "error_type"] == "false_negative_rate"
    assert summary.loc["Latent_Diffusion", "error_rate"] == 0.5


def test_all_robustness_transforms_return_normalized_tensors():
    image = Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8))
    seen = set()
    for condition in ROBUSTNESS_CONDITIONS:
        seen.add((condition.kind, condition.value))
        assert robustness_condition_label(condition.kind)
        assert robustness_value_label(condition.kind, condition.value)
        tensor = robustness_transform(condition.kind, condition.value, 32)(image)
        assert tensor.shape == (3, 32, 32)
    assert ("resample", 64.0) in seen
    assert ("noise", 0.08) in seen


def test_invalid_robustness_transform_is_rejected():
    with pytest.raises(ValueError, match="Unknown robustness condition"):
        robustness_transform("made_up", 1.0, 32)


def test_gaussian_noise_perturbation_is_content_deterministic():
    image = Image.fromarray(np.full((24, 24, 3), 120, dtype=np.uint8))
    other = Image.fromarray(np.full((24, 24, 3), 121, dtype=np.uint8))
    first = np.asarray(apply_robustness_perturbation(image, "noise", 0.08))
    second = np.asarray(apply_robustness_perturbation(image, "noise", 0.08))
    third = np.asarray(apply_robustness_perturbation(other, "noise", 0.08))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


class TinyBrightnessModel(nn.Module):
    """Small deterministic model for CPU robustness smoke tests."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3)).unsqueeze(1)


def test_evaluate_robustness_smoke_on_synthetic_images(tmp_path):
    rows = []
    for index in range(4):
        label = index % 2
        path = tmp_path / f"{index}.png"
        Image.fromarray(
            np.full((24, 24, 3), 40 + index * 50, dtype=np.uint8)
        ).save(path)
        rows.append(
            {
                "image_path": str(path),
                "source_label": "Human" if label == 0 else "Latent_Diffusion",
                "binary_label": label,
                "style_label": "Baroque",
            }
        )
    frame = pd.DataFrame(rows)
    results = evaluate_robustness(
        TinyBrightnessModel(),
        frame,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        image_size=16,
        batch_size=2,
        num_workers=0,
    )
    expected = [(condition.kind, condition.value) for condition in ROBUSTNESS_CONDITIONS]
    actual = list(zip(results["condition"], results["value"]))
    assert actual == expected
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "loss"} <= set(
        results.columns
    )
    assert np.isfinite(results["loss"]).all()


def test_gradcam_works_with_a_frozen_backbone():
    model = build_model("mobilenet_v2", mode="frozen", pretrained=False)
    target = gradcam_target_layer(model, "mobilenet_v2")
    with GradCAM(model, target) as engine:
        cam, probability = engine(
            torch.randn(3, 64, 64), target_class=1, device=torch.device("cpu")
        )
    assert cam.shape == (64, 64)
    assert 0 <= probability <= 1
