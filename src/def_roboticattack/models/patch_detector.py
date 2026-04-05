"""Lightweight CNN for adversarial patch detection in VLA image inputs.

Architecture: multi-scale edge + frequency features → binary classifier.
Detects whether an image contains an adversarial patch (UADA/UPA/TMA style).
Designed for real-time inference on edge devices (L4, Jetson Orin).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBranch(nn.Module):
    """Extract high-frequency anomalies via DCT-like learned filters."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.high_pass = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual high-pass: subtract blurred version
        blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
        high_freq = x - blurred
        feat = F.relu(self.bn(self.high_pass(high_freq)))
        return self.pool(feat).flatten(1)


class EdgeBranch(nn.Module):
    """Multi-scale Sobel edge energy analysis."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Learnable edge filters at multiple scales
        self.edge_3x3 = nn.Conv2d(in_channels, 8, 3, padding=1, bias=False)
        self.edge_5x5 = nn.Conv2d(in_channels, 8, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e3 = self.edge_3x3(x)
        e5 = self.edge_5x5(x)
        feat = F.relu(self.bn(torch.cat([e3, e5], dim=1)))
        return self.pool(feat).flatten(1)


class SpatialBranch(nn.Module):
    """Spatial consistency analysis for patch boundary detection."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.pool(x).flatten(1)


class PatchDetectorNet(nn.Module):
    """Multi-branch adversarial patch detector.

    Three branches analyze different signal domains:
    - Frequency: high-pass anomalies (patches have sharp spectral signatures)
    - Edge: multi-scale edge energy (patch boundaries create artificial edges)
    - Spatial: patch region consistency (adversarial patches break spatial coherence)

    Output: binary score [0, 1] indicating patch presence probability.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.freq_branch = FrequencyBranch(in_channels)  # -> 16
        self.edge_branch = EdgeBranch(in_channels)  # -> 16
        self.spatial_branch = SpatialBranch(in_channels)  # -> 32

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, C, H, W], Output: [B, 1] logits."""
        freq_feat = self.freq_branch(x)
        edge_feat = self.edge_branch(x)
        spatial_feat = self.spatial_branch(x)
        combined = torch.cat([freq_feat, edge_feat, spatial_feat], dim=1)
        return self.classifier(combined)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of adversarial patch presence."""
        return torch.sigmoid(self.forward(x))
