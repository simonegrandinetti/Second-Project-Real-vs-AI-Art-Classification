"""Model builders and transfer-learning freeze policies."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    convnext_tiny,
    mobilenet_v2,
    resnet18,
)


def set_trainable(module: nn.Module, trainable: bool) -> None:
    """Enable or disable gradients for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will receive optimizer updates."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class ConvNeXtTinySEBinary(nn.Module):
    """Final-feature SE channel-attention variant.

    This is a small insertion experiment on top of ConvNeXt-Tiny, not the
    paper's full multi-level AttentionConvNeXt model.
    """

    def __init__(
        self,
        pretrained: bool = True,
        mode: str = "last_stage",
        reduction: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = convnext_tiny(weights=weights)

        self.features = base.features
        channels = base.classifier[2].in_features

        # The SE block gates only the final ConvNeXt feature channels.  This is
        # intentionally a small insertion experiment, not a full reproduction.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )
        self.configure_training(mode)

    def configure_training(self, mode: str) -> None:
        """Apply the same freeze policy used by the standard backbones."""
        set_trainable(self, False)
        if mode == "frozen":
            set_trainable(self.se, True)
            set_trainable(self.classifier, True)
        elif mode == "last_stage":
            set_trainable(self.features[7], True)
            set_trainable(self.se, True)
            set_trainable(self.classifier, True)
        elif mode == "full":
            set_trainable(self, True)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        channel_weights = self.se(self.pool(features).flatten(1))[:, :, None, None]
        return self.classifier(self.pool(features * channel_weights))


def build_model(
    model_name: str,
    mode: str = "frozen",
    pretrained: bool = True,
) -> nn.Module:
    """Build a binary classifier and apply the requested training mode."""
    if mode not in {"frozen", "last_stage", "full"}:
        raise ValueError(f"Unknown training mode: {mode}")

    if model_name == "mobilenet_v2":
        # MobileNetV2 is the lightweight baseline.  Its classifier is replaced
        # with a single-logit binary head.
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        set_trainable(model, mode == "full")
        set_trainable(model.classifier, True)
        if mode == "last_stage":
            set_trainable(model.features[-1], True)
        return model

    if model_name == "resnet18":
        # ResNet18 is kept for compatibility/smoke tests even though it is not
        # part of the final E0--E4 coursework matrix.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        set_trainable(model, mode == "full")
        set_trainable(model.fc, True)
        if mode == "last_stage":
            set_trainable(model.layer4, True)
        return model

    if model_name == "convnext_tiny":
        # ConvNeXt-Tiny is the main paper-inspired backbone.
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        set_trainable(model, mode == "full")
        set_trainable(model.classifier, True)
        if mode == "last_stage":
            set_trainable(model.features[7], True)
        return model

    if model_name == "convnext_tiny_se":
        return ConvNeXtTinySEBinary(pretrained=pretrained, mode=mode)

    raise ValueError(f"Unknown model: {model_name}")


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last spatial feature block used for Grad-CAM."""
    if model_name.startswith("convnext"):
        return model.features[7]  # type: ignore[attr-defined]
    if model_name == "mobilenet_v2":
        return model.features[-1]  # type: ignore[attr-defined]
    if model_name == "resnet18":
        return model.layer4[-1]  # type: ignore[attr-defined]
    raise ValueError(f"No Grad-CAM target is defined for {model_name}")
