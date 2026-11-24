"""Deepfake voice detection CNN model architecture for VOXGUARD.

Implements a 4-block 2D Convolutional Neural Network with Batch Normalization,
Global Average Pooling, and Dropout for robust acoustic artifact classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional building block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class VoxGuardCNN(nn.Module):
    """Convolutional Neural Network for Deepfake Voice Audio Classification.

    Input shape: (Batch, 1, n_mels=128, frames=64)
    Output shape: (Batch, 2) -> [logits_genuine, logits_fake]
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        # Conv Block 1: (1, 128, 64) -> (32, 64, 32)
        self.block1 = ConvBlock(in_channels, 32)

        # Conv Block 2: (32, 64, 32) -> (64, 32, 16)
        self.block2 = ConvBlock(32, 64)

        # Conv Block 3: (64, 32, 16) -> (128, 16, 8)
        self.block3 = ConvBlock(64, 128)

        # Conv Block 4: (128, 16, 8) -> (256, 8, 4)
        self.block4 = ConvBlock(128, 256)

        # Global Average Pooling -> (256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully-connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions [P(Genuine), P(Fake)]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def get_fake_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the probability scalar/tensor of the Fake class (index 1)."""
        probs = self.predict_proba(x)
        return probs[:, 1]
