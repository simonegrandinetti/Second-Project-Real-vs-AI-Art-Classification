import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai_art_detection.evaluation import (
    binary_metrics,
    robustness_transform,
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
    for kind, value in (
        ("clean", 1.0),
        ("contrast", 0.5),
        ("jpeg", 50),
        ("resample", 16),
    ):
        tensor = robustness_transform(kind, value, 32)(image)
        assert tensor.shape == (3, 32, 32)


def test_gradcam_works_with_a_frozen_backbone():
    model = build_model("mobilenet_v2", mode="frozen", pretrained=False)
    target = gradcam_target_layer(model, "mobilenet_v2")
    with GradCAM(model, target) as engine:
        cam, probability = engine(
            torch.randn(3, 64, 64), target_class=1, device=torch.device("cpu")
        )
    assert cam.shape == (64, 64)
    assert 0 <= probability <= 1
