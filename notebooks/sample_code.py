# AquaYOLO Reference Architecture
# Modified YOLO-based underwater fish detection framework
# -------------------------------------------------------
# Changes:
# 1. Modified Backbone for enhanced feature extraction
# 2. Reduced C2f blocks
# 3. Lightweight Neck design
# 4. Multi-scale detection head
# 5. Optimized for underwater environments

import torch
import torch.nn as nn

# -------------------------------------------------------
# Basic Conv Block
# -------------------------------------------------------

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel=3, stride=1, padding=1):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

# -------------------------------------------------------
# Lightweight Feature Extraction Block
# Replaces heavy C2f blocks
# -------------------------------------------------------

class FeatureExtractionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layer = nn.Sequential(

            ConvBlock(in_channels, out_channels, 3, 1, 1),

            ConvBlock(out_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.layer(x)

# -------------------------------------------------------
# AquaYOLO Backbone
# Modified backbone with reduced complexity
# -------------------------------------------------------

class AquaYOLO_Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = ConvBlock(3, 32, 3, 2, 1)

        self.stage2 = FeatureExtractionBlock(32, 64)

        self.stage3 = FeatureExtractionBlock(64, 128)

        self.stage4 = FeatureExtractionBlock(128, 256)

        self.stage5 = FeatureExtractionBlock(256, 512)

    def forward(self, x):

        x1 = self.stage1(x)

        x2 = self.stage2(x1)

        x3 = self.stage3(x2)

        x4 = self.stage4(x3)

        x5 = self.stage5(x4)

        return x3, x4, x5

# -------------------------------------------------------
# Neck
# Lightweight Feature Fusion Neck
# -------------------------------------------------------

class AquaYOLO_Neck(nn.Module):

    def __init__(self):

        super().__init__()

        self.fusion1 = ConvBlock(512, 256, 1, 1, 0)

        self.fusion2 = ConvBlock(256, 128, 1, 1, 0)

    def forward(self, x3, x4, x5):

        p5 = self.fusion1(x5)

        p4 = self.fusion2(x4)

        return x3, p4, p5

# -------------------------------------------------------
# Detection Head
# Multi-scale prediction head
# -------------------------------------------------------

class AquaYOLO_Head(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        self.detect_small = nn.Conv2d(
            128,
            num_classes + 5,
            kernel_size=1
        )

        self.detect_medium = nn.Conv2d(
            128,
            num_classes + 5,
            kernel_size=1
        )

        self.detect_large = nn.Conv2d(
            256,
            num_classes + 5,
            kernel_size=1
        )

    def forward(self, x_small, x_medium, x_large):

        out_small = self.detect_small(x_small)

        out_medium = self.detect_medium(x_medium)

        out_large = self.detect_large(x_large)

        return out_small, out_medium, out_large

# -------------------------------------------------------
# Complete AquaYOLO Model
# -------------------------------------------------------

class AquaYOLO(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        self.backbone = AquaYOLO_Backbone()

        self.neck = AquaYOLO_Neck()

        self.head = AquaYOLO_Head(num_classes)

    def forward(self, x):

        x3, x4, x5 = self.backbone(x)

        p3, p4, p5 = self.neck(x3, x4, x5)

        outputs = self.head(p3, p4, p5)

        return outputs

# -------------------------------------------------------
# Model Testing
# -------------------------------------------------------

if __name__ == "__main__":

    model = AquaYOLO(num_classes=1)

    dummy_input = torch.randn(1, 3, 640, 640)

    outputs = model(dummy_input)

    print("AquaYOLO Architecture Loaded Successfully")

    print(f"Small Scale Output Shape : {outputs[0].shape}")
    print(f"Medium Scale Output Shape: {outputs[1].shape}")
    print(f"Large Scale Output Shape : {outputs[2].shape}")
