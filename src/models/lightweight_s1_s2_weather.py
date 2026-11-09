"""Lightweight S1/S2 classifier with temporal weather and cross-modal attention."""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


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


class LightweightS1S2WeatherClassifier(nn.Module):
    """Image encoders plus a GRU weather encoder and cross-modal attention."""

    def __init__(
        self,
        ablation: str = "s1_s2_weather",
        s1_channels: int = 3,
        s2_channels: int = 11,
        weather_features: int = 27,
        weather_sequence_features: int = 11,
        pretrained: bool = True,
        fusion_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        if ablation not in ("s2", "s1_s2", "s1_s2_weather"):
            raise ValueError(f"Unknown ablation: {ablation}")
        self.ablation = ablation
        self.s2_backbone, feature_dim = _make_backbone(s2_channels, pretrained)
        if ablation != "s2":
            self.s1_backbone, _ = _make_backbone(s1_channels, pretrained)
            self.s1_projection = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, fusion_dim),
            )
        self.s2_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, fusion_dim),
        )
        self.image_norm = nn.LayerNorm(fusion_dim)

        if ablation == "s1_s2_weather":
            self.weather_gru = nn.GRU(
                input_size=weather_sequence_features,
                hidden_size=32,
                batch_first=True,
                bidirectional=True,
            )
            self.weather_attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
            self.aggregate_encoder = nn.Sequential(
                nn.Linear(weather_features, 32),
                nn.LayerNorm(32),
                nn.Hardswish(),
                nn.Dropout(dropout),
            )
            self.weather_projection = nn.Sequential(
                nn.Linear(96, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.Hardswish(),
            )
            self.cross_modal_attention = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=4,
                dim_feedforward=2 * fusion_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            head_features = 2 * fusion_dim
        else:
            self.image_attention = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=4,
                dim_feedforward=2 * fusion_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            head_features = fusion_dim

        self.fusion_gate = (
            nn.Sequential(nn.Linear(2 * fusion_dim, fusion_dim), nn.Sigmoid())
            if ablation == "s1_s2_weather" else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_features),
            nn.Linear(head_features, 64),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image_tokens = [self.s2_projection(self.s2_backbone(batch["s2"]))]
        if self.ablation != "s2":
            asc = self.s1_backbone(batch["s1asc"])
            dsc = self.s1_backbone(batch["s1dsc"])
            s1 = self.s1_projection((asc + dsc) * 0.5)
            image_tokens.append(s1)
        image_tokens = self.image_norm(torch.stack(image_tokens, dim=1))

        if self.ablation == "s1_s2_weather":
            sequence, _ = self.weather_gru(batch["weather_sequence"])
            attention = torch.softmax(self.weather_attention(sequence), dim=1)
            temporal = (sequence * attention).sum(dim=1)
            aggregate = self.aggregate_encoder(batch["weather"])
            weather = self.weather_projection(torch.cat([temporal, aggregate], dim=1))
            tokens = torch.cat([image_tokens, weather.unsqueeze(1)], dim=1)
            contextual = self.cross_modal_attention(tokens)
            image_context = contextual[:, :-1].mean(dim=1)
            weather_context = contextual[:, -1]
            gate = self.fusion_gate(torch.cat([image_context, weather_context], dim=1))
            fused = gate * image_context + (1 - gate) * weather_context
            features = torch.cat([fused, image_context * weather_context], dim=1)
        else:
            contextual = self.image_attention(image_tokens)
            features = contextual.mean(dim=1)
        return self.classifier(features).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
