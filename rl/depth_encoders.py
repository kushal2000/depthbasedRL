"""Pluggable depth encoder registry for wrist-mounted depth camera.

Each encoder is an ``nn.Module`` that maps ``(B, 1, H, W)`` depth images
to ``(B, feature_dim)`` feature vectors.  A common wrapper handles the
optional projection layer and freeze/unfreeze logic.

Usage::

    from rl.depth_encoders import build_depth_encoder
    encoder = build_depth_encoder('resnet18', feature_dim=512, freeze=True)
    features = encoder(depth_images)          # (B, 1, 64, 64) → (B, 512)

Adding a new encoder: decorate a factory with ``@register_depth_encoder('name')``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH_ENCODERS = {}  # name → factory(feature_dim, freeze) → nn.Module


def register_depth_encoder(name):
    """Decorator to register an encoder factory."""
    def decorator(fn):
        DEPTH_ENCODERS[name] = fn
        return fn
    return decorator


def build_depth_encoder(encoder_type, feature_dim, freeze):
    """Public API — called from ActorNetwork."""
    if encoder_type not in DEPTH_ENCODERS:
        raise ValueError(
            f"Unknown depth encoder '{encoder_type}'. "
            f"Available: {list(DEPTH_ENCODERS.keys())}"
        )
    return DEPTH_ENCODERS[encoder_type](feature_dim=feature_dim, freeze=freeze)


# ── ResNet-18 (pretrained on ImageNet) ──────────────────────────────────


class _ResNet18Encoder(nn.Module):
    """ResNet-18 backbone adapted for single-channel depth input."""

    def __init__(self, feature_dim, freeze):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt conv1 from 3-channel to 1-channel by averaging pretrained weights
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(
            1, old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        resnet.conv1 = new_conv1

        # Build backbone: everything up through avgpool (output 512-dim)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        backbone_out_dim = 512

        # Projection head
        if feature_dim != backbone_out_dim:
            self.projection = nn.Sequential(
                nn.Linear(backbone_out_dim, feature_dim),
                nn.ELU(),
            )
        else:
            self.projection = None

        # Freeze backbone if requested (projection stays trainable)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """x: (B, 1, H, W) → (B, feature_dim)"""
        feat = self.backbone(x)
        if self.projection is not None:
            feat = self.projection(feat)
        return feat


@register_depth_encoder('resnet18')
def _build_resnet18(feature_dim, freeze):
    return _ResNet18Encoder(feature_dim=feature_dim, freeze=freeze)


# ── DINOv2-small (ViT-S/14) ────────────────────────────────────────────


class _DINOv2Encoder(nn.Module):
    """DINOv2-small backbone for depth images."""

    def __init__(self, feature_dim, freeze):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True,
        )
        backbone_out_dim = 384  # ViT-S CLS token dim

        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ELU(),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """x: (B, 1, H, W) → (B, feature_dim)"""
        # DINOv2 expects 224×224 3-channel input
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.expand(-1, 3, -1, -1)  # 1ch → 3ch
        feat = self.backbone(x)  # CLS token: (B, 384)
        feat = self.projection(feat)
        return feat


@register_depth_encoder('dinov2')
def _build_dinov2(feature_dim, freeze):
    return _DINOv2Encoder(feature_dim=feature_dim, freeze=freeze)


# ── ScratchCNN (lightweight, trained from scratch) ─────────────────────


class _ScratchCNNEncoder(nn.Module):
    """4-layer CNN for depth encoding, trained from scratch."""

    def __init__(self, feature_dim):
        super().__init__()
        # 4 conv layers: 1→32→64→128→256, stride 2, 3×3, BN+ELU
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ELU(),
        )

    def forward(self, x):
        """x: (B, 1, H, W) → (B, feature_dim)"""
        feat = self.backbone(x)
        feat = self.projection(feat)
        return feat


@register_depth_encoder('scratch_cnn')
def _build_scratch_cnn(feature_dim, freeze):
    # freeze is a no-op for scratch CNN (no pretrained weights)
    return _ScratchCNNEncoder(feature_dim=feature_dim)
