# ultralytics/nn/modules/attention.py
"""
Attention mechanisms for deep convolutional neural networks.

This module implements various attention mechanisms including ECA-Net, CBAM, and Coordinate Attention.
All implementations are mathematically accurate and follow the original papers.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('ECA', 'CBAM', 'CoordAtt', 'ChannelAttention', 'SpatialAttention', 'h_sigmoid', 'h_swish')


class ECA(nn.Module):
    """
    Efficient Channel Attention module.
    
    Implementation based on:
    "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Paper: https://arxiv.org/abs/1910.03151
    
    The ECA module captures cross-channel interaction with only a few parameters by using
    adaptive kernel size selection for 1D convolution.
    
    Args:
        c1 (int): Number of input channels
        b (int): Bias parameter for adaptive kernel size calculation. Default: 1
        gamma (int): Gamma parameter for adaptive kernel size calculation. Default: 2
        
    Mathematical Formula:
        k = |log2(c1) + b| / gamma  (adaptive kernel size)
        where k is odd-valued to ensure proper padding
    """
    
    def __init__(self, c1: int, b: int = 1, gamma: int = 2) -> None:
        super().__init__()
        
        # Adaptive kernel size calculation as per ECA-Net paper: k = |⌊(log₂(C) + b) / γ⌋|
        t = int(abs((math.log(c1, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # Ensure odd kernel size
        k = max(3, k)  # Handle edge case for very small channel counts (minimum k=3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D convolution across channels with adaptive kernel size
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ECA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with channel attention applied
        """
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        
        # Reshape for 1D convolution: (B, C, 1, 1) -> (B, 1, C) -> (B, C, 1) -> (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid activation
        y = self.sigmoid(y)
        
        # Element-wise multiplication with broadcasting
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention module for CBAM.
    
    Implements the channel attention mechanism that focuses on 'what' is meaningful
    given an input feature map. Uses both average and max pooling to capture different
    aspects of channel-wise information.
    
    Args:
        in_planes (int): Number of input channels
        ratio (int): Reduction ratio for the intermediate layer. Default: 16
    """
    
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP with reduction ratio
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Channel Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Channel attention weights of shape (B, C, 1, 1)
        """
        # Apply shared MLP to both average and max pooled features
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Element-wise sum and sigmoid activation
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention module for CBAM.
    
    Implements the spatial attention mechanism that focuses on 'where' is meaningful
    given a feature map. Uses channel-wise statistics (average and max) to generate
    spatial attention map.
    
    Args:
        kernel_size (int): Kernel size for the convolutional layer. Default: 7
    """
    
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        # Single convolutional layer to generate spatial attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Spatial Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Spatial attention weights of shape (B, 1, H, W)
        """
        # Generate channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and process with convolution
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        out = self.conv1(x_cat)  # (B, 1, H, W)
        
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Implementation based on:
    "CBAM: Convolutional Block Attention Module"
    Paper: https://arxiv.org/abs/1807.06521
    
    CBAM is a simple yet effective attention module that can be integrated into any CNN.
    It sequentially applies channel attention and spatial attention to refine features.
    
    Args:
        c1 (int): Number of input channels
        ratio (int): Reduction ratio for channel attention. Default: 16
        kernel_size (int): Kernel size for spatial attention. Default: 7
    """
    
    def __init__(self, c1: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with attention applied
        """
        # Apply channel attention first
        x = x * self.channel_attention(x)
        
        # Then apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class h_sigmoid(nn.Module):
    """
    Hard Sigmoid activation function.
    
    A computationally efficient approximation of sigmoid using ReLU6.
    Used in mobile-friendly attention mechanisms.
    
    Formula: h_sigmoid(x) = ReLU6(x + 3) / 6
    
    Args:
        inplace (bool): If True, performs operation in-place. Default: True
    """
    
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Hard Sigmoid.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with hard sigmoid applied
        """
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """
    Hard Swish activation function.
    
    A computationally efficient approximation of Swish activation using hard sigmoid.
    Used in mobile-friendly networks and attention mechanisms.
    
    Formula: h_swish(x) = x * h_sigmoid(x)
    
    Args:
        inplace (bool): If True, performs operation in-place. Default: True
    """
    
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Hard Swish.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with hard swish applied
        """
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention module.
    
    Implementation based on:
    "Coordinate Attention for Efficient Mobile Network Design"
    Paper: https://arxiv.org/abs/2103.02907
    
    Coordinate Attention factorizes channel attention into two 1D feature encoding
    processes that aggregate features along two spatial directions separately.
    This allows the attention module to capture long-range dependencies with precise positional information.
    
    Args:
        inp (int): Number of input channels
        oup (int): Number of output channels (should equal inp for residual connection)
        reduction (int): Reduction ratio for the intermediate channels. Default: 32
    """
    
    def __init__(self, inp: int, oup: int, reduction: int = 32) -> None:
        super().__init__()
        
        # Coordinate information embedding
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool along width
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # Pool along height
        
        # Calculate intermediate channels with minimum constraint
        mip = max(8, inp // reduction)
        
        # Coordinate information transformation
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # Coordinate attention generation
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Coordinate Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with coordinate attention applied
        """
        identity = x
        n, c, h, w = x.size()
        
        # Coordinate information embedding
        x_h = self.pool_h(x)                                    # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)               # (B, C, W, 1)
        
        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)                        # (B, C, H+W, 1)
        
        # Coordinate information transformation
        y = self.conv1(y)                                       # (B, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back to height and width components
        x_h, x_w = torch.split(y, [h, w], dim=2)               # (B, mip, H, 1), (B, mip, W, 1)
        x_w = x_w.permute(0, 1, 3, 2)                          # (B, mip, 1, W)
        
        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()                       # (B, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()                       # (B, oup, 1, W)
        
        # Apply coordinate attention
        out = identity * a_w * a_h                              # (B, C, H, W)
        
        return out
