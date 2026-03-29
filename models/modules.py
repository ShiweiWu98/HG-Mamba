import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torchvision.ops.deform_conv import DeformConv2d
from typing import Optional, Callable, Any
import torch.utils.checkpoint as checkpoint
try:
    from .conv_custom import FRBlock
except:
    from conv_custom import FRBlock
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DFFN(nn.Module):
    def __init__(self, in_features, ffn_expansion_factor=4.0, bias=False, patch_size=8, drop=0.0):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        # Input projection for channel expansion
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)

        # Parameters for frequency-domain processing
        self.patch_size = patch_size
        self.fft_weight = nn.Parameter(
            torch.ones((hidden_features * 2, 1, 1, patch_size, patch_size // 2 + 1))
        )

        # Depth-wise convolution for spatial mixing
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias
        )

        # Output projection for channel reduction
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

        # Dropout after gating
        self.drop = nn.Dropout(drop)

        # Keep the same interface as the original MLP
        self.channels_first = False

    def forward(self, x):
        # Convert [B, H, W, C] to [B, C, H, W] if needed
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2)

        # Channel expansion
        x = self.project_in(x)

        # Frequency-domain filtering on local patches
        if self.patch_size > 0:
            x_patch = rearrange(
                x, 'b c (h p1) (w p2) -> b c h w p1 p2',
                p1=self.patch_size, p2=self.patch_size
            )
            x_patch_fft = torch.fft.rfft2(x_patch.float())
            x_patch_fft = x_patch_fft * self.fft_weight
            x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
            x = rearrange(
                x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)',
                p1=self.patch_size, p2=self.patch_size
            )

        # Gated feature interaction
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.drop(x)

        # Channel reduction
        x = self.project_out(x)

        # Convert [B, C, H, W] back to [B, H, W, C] if needed
        if not self.channels_first:
            x = x.permute(0, 2, 3, 1)

        return x

    
class FRFFN(nn.Module):
    def __init__(self, in_features, ffn_expansion_factor=4.0, bias=False, drop=0.0):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.freq_select = FRBlock(
            in_channels=in_features,
            k_list=[2,4,8],
            lowfreq_att=False,
            lp_type='freq',
            act='sigmoid',
            spatial='conv',
            spatial_group=1, 
            spatial_kernel=3
        )
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, 
                               stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)
        self.channels_first = False

    def forward(self, x):
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2)
            
        x = self.project_in(x)
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.drop(x)
        x = self.project_out(x)
        x = self.freq_select(x)
        if not self.channels_first:
            x = x.permute(0, 2, 3, 1)
        return x


class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1,
                      padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(n_feat // 2),
            # nn.Hardswish(),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1,
                      padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(n_feat*2),
            # nn.Hardswish(),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class ChannelAttention(nn.Module):
    """Channel attention.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y