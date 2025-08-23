"""
MobileViT Hybrid Backbone implementation for YOLOv8.
This module provides a lightweight vision transformer backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MV2Block(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.
    """
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    """
    MobileViT Block combining local and global features.
    """
    def __init__(self, d_model, d_ffn, n_heads, patch_size=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Local feature extraction
        self.local_rep = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )
        
        # Global feature extraction (Transformer)
        self.global_rep = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        
        # Fusion
        self.fusion = nn.Conv2d(d_model * 2, d_model, 1, 1, 0, bias=False)

    def forward(self, x):
        # Local features
        local_features = self.local_rep(x)
        
        # Global features
        b, c, h, w = x.shape
        # Unfold into patches
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.view(b, c, self.patch_size**2, -1)
        patches = patches.permute(0, 3, 1, 2).contiguous()
        patches = patches.view(b * patches.shape[1], c, self.patch_size**2)
        patches = patches.permute(0, 2, 1).contiguous()
        
        # Apply transformer
        global_features, _ = self.global_rep(patches)
        global_features = global_features.permute(0, 2, 1).contiguous()
        global_features = global_features.view(b, -1, c, self.patch_size**2)
        global_features = global_features.permute(0, 2, 3, 1).contiguous()
        
        # Fold back
        global_features = global_features.view(b, c * self.patch_size**2, -1)
        global_features = F.fold(global_features, (h, w), kernel_size=self.patch_size, stride=self.patch_size)
        
        # Fusion
        combined = torch.cat([local_features, global_features], dim=1)
        output = self.fusion(combined)
        
        return output


class MobileViT(nn.Module):
    """
    MobileViT Hybrid Backbone.
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super().__init__()
        
        # Initial convolution
        input_channel = int(32 * width_multiplier)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.SiLU(inplace=True),
        )
        
        # MV2 blocks
        self.mv2_1 = MV2Block(input_channel, int(64 * width_multiplier), stride=1, expansion=1)
        input_channel = int(64 * width_multiplier)
        
        self.mv2_2 = MV2Block(input_channel, int(128 * width_multiplier), stride=2, expansion=4)
        input_channel = int(128 * width_multiplier)
        
        # MobileViT blocks
        self.mvit_1 = MobileViTBlock(
            d_model=input_channel,
            d_ffn=int(256 * width_multiplier),
            n_heads=4,
            patch_size=2
        )
        
        self.mv2_3 = MV2Block(input_channel, int(256 * width_multiplier), stride=2, expansion=4)
        input_channel = int(256 * width_multiplier)
        
        self.mvit_2 = MobileViTBlock(
            d_model=input_channel,
            d_ffn=int(512 * width_multiplier),
            n_heads=8,
            patch_size=2
        )
        
        self.mv2_4 = MV2Block(input_channel, int(512 * width_multiplier), stride=2, expansion=4)
        input_channel = int(512 * width_multiplier)
        
        self.mvit_3 = MobileViTBlock(
            d_model=input_channel,
            d_ffn=int(1024 * width_multiplier),
            n_heads=8,
            patch_size=2
        )
        
        # Output layers for YOLO integration
        self.feature_channels = [input_channel, int(256 * width_multiplier), int(128 * width_multiplier)]
        
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # MV2 blocks
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        
        # MobileViT block 1
        x = self.mvit_1(x)
        feat1 = x
        
        # MV2 block 3
        x = self.mv2_3(x)
        
        # MobileViT block 2
        x = self.mvit_2(x)
        feat2 = x
        
        # MV2 block 4
        x = self.mv2_4(x)
        
        # MobileViT block 3
        x = self.mvit_3(x)
        feat3 = x
        
        return [feat1, feat2, feat3]


def create_mobilevit_backbone(width_multiplier=1.0):
    """
    Create MobileViT backbone for YOLOv8 integration.
    
    Args:
        width_multiplier: Width multiplier for model scaling
    
    Returns:
        MobileViT backbone model
    """
    return MobileViT(width_multiplier=width_multiplier)


def integrate_mobilevit_with_yolo(yolo_model, width_multiplier=1.0):
    """
    Integrate MobileViT backbone with YOLOv8.
    
    Args:
        yolo_model: YOLOv8 model
        width_multiplier: Width multiplier for MobileViT
    
    Returns:
        Modified YOLOv8 model with MobileViT backbone
    """
    # This function would need to be customized based on the specific YOLOv8 architecture
    # For now, we provide a template for integration
    print("⚠️  MobileViT integration requires manual modification of YOLOv8 architecture")
    print("📝 This is a placeholder for Model C implementation")
    return yolo_model
