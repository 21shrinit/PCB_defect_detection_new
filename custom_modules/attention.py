"""
CBAM (Convolutional Block Attention Module) implementation for YOLOv8.
This module enhances the model's ability to focus on important features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Generates channel attention maps using both max and average pooling.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling outputs
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        
        # Combine both branches
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Generates spatial attention maps using channel-wise pooling.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module).
    Combines channel and spatial attention mechanisms.
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class CBAMBlock(nn.Module):
    """
    CBAM Block that can be integrated into YOLOv8 backbone.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        
        # Standard convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
        # CBAM attention
        self.cbam = CBAM(out_channels, reduction_ratio)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.cbam(x)
        return x


def add_cbam_to_model(model, reduction_ratio=16):
    """
    Add CBAM attention modules to a YOLOv8 model.
    
    Args:
        model: YOLOv8 model
        reduction_ratio: Reduction ratio for channel attention
    
    Returns:
        Modified model with CBAM attention
    """
    # This function would need to be customized based on the specific YOLOv8 architecture
    # For now, we provide a template for integration
    print("⚠️  CBAM integration requires manual modification of YOLOv8 architecture")
    print("📝 This is a placeholder for Model B implementation")
    return model
