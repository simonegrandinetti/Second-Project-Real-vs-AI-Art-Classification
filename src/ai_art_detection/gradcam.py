"""Grad-CAM utilities for qualitative model inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional

from .data import get_transforms
from .models import gradcam_target_layer


class GradCAM:
    """Small context-managed Grad-CAM engine for a selected target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Hooks are registered once and removed by `close()` / context manager
        # exit so notebook reruns do not accumulate stale hooks.
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __call__(
        self,
        image: torch.Tensor,
        target_class: int,
        device: torch.device,
    ) -> tuple[np.ndarray, float]:
        """Return a normalized CAM and fake probability for one image tensor."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        # Input gradients keep hooks active even when the backbone is frozen.
        batch = image.unsqueeze(0).to(device).requires_grad_(True)
        logit = self.model(batch).flatten()[0]

        # Binary models emit one logit for the fake class.  For real-class
        # examples we backpropagate the negative logit.
        score = logit if target_class == 1 else -logit
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive tensors.")
        activations = self.activations[0]
        gradients = self.gradients[0]

        # Standard Grad-CAM: average gradients over space, weight activations,
        # clamp to positive evidence, then resize to input resolution.
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0, keepdim=True))
        cam = functional.interpolate(
            cam[None],
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), torch.sigmoid(logit).item()


def save_gradcam_panels(
    model: nn.Module,
    model_name: str,
    predictions: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    *,
    image_size: int = 224,
    seed: int = 42,
    examples_per_group: int = 4,
) -> list[Path]:
    """Save representative correct-real, correct-fake, and error panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, transform = get_transforms(image_size, augment=False)
    target_layer = gradcam_target_layer(model, model_name)
    predictions = predictions.copy()
    if predictions["correct"].dtype != bool:
        predictions["correct"] = (
            predictions["correct"].astype(str).str.lower().eq("true")
        )
    predictions["label"] = predictions["label"].astype(int)
    predictions["pred"] = predictions["pred"].astype(int)

    # Three groups give a compact qualitative check: what the model uses for
    # real successes, fake successes, and mistakes.
    groups = {
        "gradcam_correct_real": predictions[
            predictions["correct"] & (predictions["label"] == 0)
        ],
        "gradcam_correct_fake": predictions[
            predictions["correct"] & (predictions["label"] == 1)
        ],
        "gradcam_misclassified": predictions[~predictions["correct"]],
    }
    written = []
    with GradCAM(model, target_layer) as cam_engine:
        for name, frame in groups.items():
            if frame.empty:
                continue

            sample = frame.sample(
                min(examples_per_group, len(frame)), random_state=seed
            )
            figure, axes = plt.subplots(
                len(sample), 2, figsize=(9, 3 * len(sample)), squeeze=False
            )
            for row_axes, (_, row) in zip(axes, sample.iterrows()):
                with Image.open(row["image_path"]) as source:
                    image = source.convert("RGB")

                tensor = transform(image)
                cam, probability = cam_engine(tensor, int(row["pred"]), device)
                rendered = image.resize((image_size, image_size))
                row_axes[0].imshow(rendered)
                row_axes[0].set_title(
                    f"True={row['label']}, Pred={row['pred']}, "
                    f"P(fake)={probability:.2f}"
                )
                row_axes[1].imshow(rendered)
                row_axes[1].imshow(cam, cmap="jet", alpha=0.45)
                row_axes[1].set_title("Grad-CAM")
                for axis in row_axes:
                    axis.axis("off")
            figure.tight_layout()
            output_path = output_dir / f"{name}.png"
            figure.savefig(output_path, dpi=200)
            plt.close(figure)
            written.append(output_path)
    return written
