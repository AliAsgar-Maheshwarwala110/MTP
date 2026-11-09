"""Lightweight MobileNetV3 multimodal classifier adapted to Sen12Landslides."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


def _make_backbone(in_channels: int, pretrained: bool):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    first = backbone.features[0][0]
    replacement = nn.Conv2d(
        in_channels,
        first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            replacement.weight.copy_(
                first.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            )
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out")
    backbone.features[0][0] = replacement
    feature_dim = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


class LightweightMultimodalClassifier(nn.Module):
    """Two MobileNetV3-Small branches with a compact fusion head.

    Inputs are ``s2`` and ``s1`` tensors with shapes ``[B, C, H, W]``.
    The branch widths are configurable because the local NetCDF release has
    different channel counts from the reference repository's NumPy export.
    """

    def __init__(
        self,
        s2_channels: int,
        s1_channels: int,
        pretrained: bool = True,
        img_size: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.img_size = img_size
        self.s2_backbone, feature_dim = _make_backbone(s2_channels, pretrained)
        self.s1_backbone, _ = _make_backbone(s1_channels, pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        if s2.shape[-2:] != (self.img_size, self.img_size):
            s2 = F.interpolate(s2, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        if s1.shape[-2:] != (self.img_size, self.img_size):
            s1 = F.interpolate(s1, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        features = torch.cat([self.s2_backbone(s2), self.s1_backbone(s1)], dim=1)
        return self.classifier(features).squeeze(1)
