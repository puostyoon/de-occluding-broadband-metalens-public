from typing import Callable, List, Union, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import kornia.filters as KF

from . import base
from .base import Conv2d, Linear, ChAttn2d, SpAttn2d


_DEBUG_IMAGES_: Union[List[torch.Tensor], None] = None


def safe_pow(x: torch.Tensor, gamma: Union[torch.Tensor, float], eps: float = 1e-12) -> torch.Tensor:
    """ Apply pow function to posibly negative values. """
    return x.sign() * (x.abs() + eps).pow(gamma)


def soft_histogram(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """ Compute soft histogram of each pixels.

    References:
        - Liu, Y.-L., et al. (2020). "Single-image HDR reconstruction by learning to reverse the camera pipeline." CVPR.

    Args:
        x: input tensor, `(*, H, W)`.
        n_bins: number of bins.

    Returns:
        output tensor, `(*, n_bins, H, W)`.
    """
    assert x.ndim >= 2, f"Expected image, got {x.shape}-dim"

    hists = torch.stack([
        F.relu(1. - torch.abs(x - i/(n_bins-1.)) * (n_bins-1.))
        for i in range(n_bins)
    ], dim=-3)
    assert hists.shape == (*x.shape[:-2], n_bins, *x.shape[-2:])

    return hists


def over_exposed_mask(x: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """ Compute soft over-exposed mask.

    Args:
        x: input tensor, `(*, H, W)`.
        threshold: threshold value.

    Returns:
        output tensor, `(*, H, W)`.
    """
    assert x.ndim >= 2, f"Expected image, got {x.shape}-dim"

    return (x - threshold).clip(min=0.) / (1.-threshold)


def content_features(
    x: torch.Tensor,
    use_gradient: bool = True,
    histogram_bins: List[int] = [4, 8, 16],
):
    """ Create content features.

    Args:
        x: input tensor, `(N, C, H, W)`.
        use_gradient: whether to use gradient features.
        histogram_bins: list of of histogram bins for each channel.

    Returns:
        output tensor, `(N, C', H, W)`. `C' = C * (2 * use_gradient + sum(histogram_bins))`.
    """
    
    assert x.ndim >= 3, f"Expected batch image, got {x.shape}-dim"
    features: list[torch.Tensor] = []

    if use_gradient:
        features.append(KF.spatial_gradient(x).flatten(-4, -3))  # (N, C*2, H, W)

    for n_bins in histogram_bins:
        features.append(soft_histogram(x, n_bins).flatten(-4, -3))  # (N, C*n_bins, H, W)

    return torch.cat(features, dim=-3)  # (N, C', H, W)


class LocalBlock(nn.Module):
    """ Residual convolution-attention block.

    Reference:
        - Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV.

    Args:
        n_channels: number of feature channels.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        kernel_size: size of the convolution kernel.
    """

    def __init__(self, n_channels: int, reduction: int = 2, *, kernel_size: int = 3, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, f"`kernel_size` must be odd, got {kernel_size}"
        self.conv1 = Conv2d(n_channels, n_channels//reduction, kernel_size, padding_mode="replicate", **kwargs)
        self.conv2 = Conv2d(n_channels//reduction, n_channels, kernel_size, padding_mode="replicate", **kwargs)
        self.chattn = ChAttn2d(n_channels, reduction, **kwargs)
        self.spattn = SpAttn2d(padding_mode="replicate", **kwargs)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        r = self.conv1(r)
        r = self.act(r)
        r = self.conv2(r)
        r = self.chattn(r)
        r = self.spattn(r)
        return x + r


class LocalNet(nn.Module):
    """ Local residual network with U-Net style structure.

    Args:
        feat_channels: number of channels in input feature.
        mid_channels: number of channels in middle layers.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        kernel_size: size of the convolution kernel.
        num_block: number of residual blocks in each stage.
        num_scale: number of scales in U-Net structure, 0 for no down/up-sampling.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """
    color_channels: int = 3

    def __init__(
        self,
        feat_channels: int,
        mid_channels: int = 64,
        reduction: int = 2,
        kernel_size: int = 3,
        num_block: int = 2,
        num_scale: int = 2,
        **kwargs,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"`kernel_size` must be odd, got {kernel_size}"
        assert num_block >= 1, f"`num_block` must be positive, got {num_block}"
        assert num_scale >= 0, f"`num_scale` must be non-negative, got {num_scale}"

        self.feat_channels = feat_channels
        self.mid_channels = mid_channels
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.num_block = num_block
        self.num_scale = num_scale

        self.conv_head = Conv2d(feat_channels, mid_channels, kernel_size, padding_mode="replicate", **kwargs)
        self.conv_down = nn.ModuleList([
            nn.Conv2d(mid_channels, mid_channels, 2, stride=2)
            for _ in range(num_scale)
        ])
        self.conv_up = nn.ModuleList([
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2)
            for _ in range(num_scale)
        ])
        self.conv_tail = Conv2d(mid_channels, self.color_channels, kernel_size, padding_mode="replicate", **kwargs)

        self.resblks = nn.ModuleList([
            LocalBlock(mid_channels, reduction, kernel_size=kernel_size, **kwargs)
            for _ in range(num_block * (2*num_scale + 1))
        ])

        if "init_weights" not in kwargs:
            init.xavier_normal_(self.conv_tail.weight, 1e-2)
            if self.conv_tail.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv_tail.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.conv_tail.bias, -bound, bound)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3], f"Expected same batch size, got {x.shape[:-3]} and {z.shape[:-3]}"
        assert x.shape[-3] == self.color_channels, f"Expected {self.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.feat_channels, f"Expected {self.feat_channels}-channel feature, got {z.shape[-3]}"
        assert x.shape[-1] % 2**self.num_scale == 0 and x.shape[-2] % 2**self.num_scale == 0, \
            f"Expected image size divisible by 2^{self.num_scale}, got {x.shape[-2:]}"

        z_skip: dict[int, torch.Tensor] = {}

        z = self.conv_head(z)  # (*, mid_channels, H, W)

        for stage in range(self.num_scale):
            for i in range(stage * self.num_block, (stage + 1) * self.num_block):
                z = self.resblks[i](z)
            z_skip[stage] = z
            z = self.conv_down[stage](z)  # (*, mid_channels, H/2, W/2)

        for i in range(self.num_scale * self.num_block, (self.num_scale + 1) * self.num_block):
            z = self.resblks[i](z)

        for stage in range(self.num_scale-1, -1, -1):
            z = self.conv_up[stage](z)
            z = z_skip[stage] + z
            for i in range((self.num_scale+stage+1) * self.num_block, (self.num_scale+stage+2) * self.num_block):
                z = self.resblks[i](z)

        z = self.conv_tail(z)  # (*, 3, H, W)

        return x + z
