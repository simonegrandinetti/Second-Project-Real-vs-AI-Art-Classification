"""Build the binary classifiers and apply their transfer-learning policies.

The experiment matrix uses torchvision implementations rather than local
reimplementations of the backbones. When pretraining is enabled, the builders
request the installed torchvision release's ``DEFAULT`` ImageNet-1K weights and
replace the multiclass classifier with a single-logit human-versus-AI head.
"""

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
    """Set one gradient policy recursively across a module.

    Args:
        module: PyTorch module whose parameters should be updated.
        trainable: Desired value of ``requires_grad`` for every parameter.
    """
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    """Count scalar parameters currently eligible for optimization.

    Args:
        model: Model whose ``requires_grad`` flags have already been configured.

    Returns:
        Sum of ``numel()`` over trainable parameters only.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class ConvNeXtTinySEBinary(nn.Module):
    """Final-feature SE channel-attention variant.

    This is a small insertion experiment on top of ConvNeXt-Tiny, not the
    paper's full multi-level AttentionConvNeXt model.

    A global average first summarizes each final ConvNeXt channel. The two-layer
    squeeze-and-excitation (SE) gate turns that summary into per-image channel
    weights, which rescale the final spatial feature map before the binary head.
    The gate changes channel emphasis but does not provide spatial attention.
    """

    def __init__(
        self,
        pretrained: bool = True,
        mode: str = "last_stage",
        reduction: int = 16,
        dropout: float = 0.2,
    ):
        """Initialize the ConvNeXt-Tiny backbone, SE gate, and binary head.

        Args:
            pretrained: Load ``ConvNeXt_Tiny_Weights.DEFAULT`` from torchvision when
                true; otherwise initialize the backbone randomly.
            mode: Initial freeze policy: ``frozen``, ``last_stage``, or ``full``.
            reduction: Channel reduction ratio inside the SE bottleneck.
            dropout: Probability used by the classifier's dropout layer.
        """
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
        """Apply the experiment's transfer-learning freeze policy.

        Args:
            mode: ``frozen`` trains only SE and classifier parameters;
                ``last_stage`` additionally trains final feature stage ``features[7]``;
                ``full`` trains the complete network.

        Raises:
            ValueError: If ``mode`` is not one of the supported policies.
        """
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
        """Return one fake-class logit for every input image.

        Args:
            images: ImageNet-normalized tensor with shape ``(N, 3, H, W)``.

        Returns:
            Logit tensor with shape ``(N, 1)``. Positive values favor the
            AI-generated class and negative values favor the human class.
        """
        features = self.features(images)
        channel_weights = self.se(self.pool(features).flatten(1))[:, :, None, None]
        return self.classifier(self.pool(features * channel_weights))


def build_model(
    model_name: str,
    mode: str = "frozen",
    pretrained: bool = True,
) -> nn.Module:
    """Build a supported one-logit classifier and configure trainable layers.

    Args:
        model_name: ``mobilenet_v2``, ``convnext_tiny``, ``convnext_tiny_se``, or
            the ``resnet18`` compatibility model used by smoke tests.
        mode: Transfer-learning policy. ``frozen`` trains the new head,
            ``last_stage`` also trains the final backbone stage, and ``full`` trains
            all parameters. The SE model also trains its newly inserted gate.
        pretrained: Request torchvision's model-specific ``DEFAULT`` ImageNet-1K
            weights. False avoids downloads and is intended for tests or diagnostics.

    Returns:
        A PyTorch model whose forward pass produces shape ``(N, 1)`` and whose
        ``requires_grad`` flags reflect ``mode``.

    Raises:
        ValueError: If the model name or training mode is unknown.
    """
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
    """Select the final spatial feature block suitable for Grad-CAM hooks.

    Args:
        model: Model produced by :func:`build_model`.
        model_name: Architecture identifier used to locate its final spatial block.

    Returns:
        The ConvNeXt final feature stage, MobileNetV2 final feature layer, or ResNet18
        final residual block. Its output still has spatial dimensions, unlike the
        classifier head.

    Raises:
        ValueError: If no target-layer mapping exists for ``model_name``.
    """
    if model_name.startswith("convnext"):
        return model.features[7]  # type: ignore[attr-defined]
    if model_name == "mobilenet_v2":
        return model.features[-1]  # type: ignore[attr-defined]
    if model_name == "resnet18":
        return model.layer4[-1]  # type: ignore[attr-defined]
    raise ValueError(f"No Grad-CAM target is defined for {model_name}")
