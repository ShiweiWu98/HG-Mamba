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
    from .util import selective_scan_state_flop_jit, selective_scan_fn
    from .modules import *
except:
    from util import selective_scan_state_flop_jit, selective_scan_fn
    from modules import *



class StateFusion(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=5,
                 ):
        super(StateFusion, self).__init__()
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                                        stride=1, padding=kernel_size//2, bias=False, groups=dim),
                                              ChannelAttention(num_feat=dim),
                                              nn.GELU(),
                                              nn.Conv2d(in_channels=dim, out_channels=2*kernel_size*kernel_size,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)

                                              )
        self.dcn = DeformConv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False,
            groups=dim,
        )
        
        
    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, x, h):
        offset = self.offset_generator(x)
        out = self.dcn(h, offset)
        return out

# class StateFusion(nn.Module):
#     """
#     Replaces torchvision DeformConv2d with F.grid_sample.
#     Numerically identical to the original (max|Δ| < 1e-5), but ~3-5x faster.

#     NOTE: Please use this version for training instead of the DeformConv2d version.

#     Initialization:
#         - Main conv weight / offset first layer : kaiming_uniform_
#         - Offset output layer                   : weight=0, bias=0
#           (all offsets start at 0, equivalent to a plain depthwise conv at init)
#     """

#     def __init__(self, dim: int, kernel_size: int = 5):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#         K  = kernel_size
#         K2 = K * K

#         self.offset_generator = nn.Sequential(
#             nn.Conv2d(dim, dim, K, padding=K // 2, bias=False, groups=dim),
#             ChannelAttention(num_feat=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, 2 * K2, kernel_size=1, bias=True),
#         )

#         # depthwise conv weight [dim, 1, K, K]
#         self.weight = nn.Parameter(torch.empty(dim, 1, K, K))

#         # fixed kernel grid offsets (buffer: no grad, moves with device)
#         ky, kx = torch.meshgrid(
#             torch.arange(-(K // 2), K // 2 + 1, dtype=torch.float32),
#             torch.arange(-(K // 2), K // 2 + 1, dtype=torch.float32),
#             indexing='ij',
#         )
#         self.register_buffer('ky', ky.reshape(1, K2, 1, 1))
#         self.register_buffer('kx', kx.reshape(1, K2, 1, 1))

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.offset_generator[0].weight, a=math.sqrt(5))

#         # zero-init offset output layer so offsets start at exactly 0
#         last_conv = self.offset_generator[-1]
#         nn.init.zeros_(last_conv.weight)
#         nn.init.zeros_(last_conv.bias)

#     def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = h.shape
#         K2 = self.kernel_size ** 2

#         # generate offsets [B, 2·K², H, W]
#         offset   = self.offset_generator(x)
#         offset_y = offset[:, 0::2]   # [B, K², H, W]
#         offset_x = offset[:, 1::2]   # [B, K², H, W]

#         # normalised sampling coordinates (align_corners=True: p -> 2p/(S-1) - 1)
#         y_base = torch.arange(H, device=h.device, dtype=h.dtype).view(1, 1, H, 1)
#         x_base = torch.arange(W, device=h.device, dtype=h.dtype).view(1, 1, 1, W)
#         sy = (y_base + self.ky + offset_y) * (2.0 / max(H - 1, 1)) - 1.0
#         sx = (x_base + self.kx + offset_x) * (2.0 / max(W - 1, 1)) - 1.0

#         # stack K² grids along H axis -> single grid_sample call (1 CUDA kernel)
#         grid    = torch.stack([sx, sy], dim=-1).reshape(B, K2 * H, W, 2)
#         sampled = F.grid_sample(
#             h, grid,
#             mode='bilinear',
#             padding_mode='zeros',
#             align_corners=True,
#         ).view(B, C, K2, H, W)

#         # depthwise weighted sum over K² positions
#         out = (sampled * self.weight.view(1, C, K2, 1, 1)).sum(dim=2)
#         return out

class HGSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        kernel_size=7,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(
            self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(
            self.d_inner, kernel_size=kernel_size)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) *
                (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1,
                             dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(x, h)

        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x))

        y = self.ssm(x)

        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class HGMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        kernel_size=7,
        **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3,
                              padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = HGSSM(
            d_model=hidden_dim, dropout=attn_drop_rate,
            d_state=d_state, dt_init=dt_init,
            kernel_size=kernel_size,
            **kwargs)
        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3,
                              padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = FRFFN(in_features=hidden_dim, ffn_expansion_factor=2.0)

    def forward(self, x: torch.Tensor):
        x = x + self.cpe1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x_cpe2 = x.permute(0, 3, 1, 2)
        x = x + self.cpe2(x_cpe2).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class HGMambaLayer(nn.Module):
    """ 
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth=1,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        kernel_size=7,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            HGMambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                kernel_size=kernel_size,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Desmoke_Net(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=[48, 64, 128, 64, 48],
                 num_blocks=[2, 3, 4, 3, 2],
                 kernels=(5, 5, 5, 5, 5),
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 ):

        super(Desmoke_Net, self).__init__()
        
        self.img_range = 1.0
        if inp_channels == 3:
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
            
        self.inp_channels = inp_channels
        self.stem_in = nn.Sequential(nn.Conv2d(inp_channels, dim[0], kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Hardswish(),
                                     )
        self.layer1 = HGMambaLayer(
            dim[0], d_state=1, depth=num_blocks[0], 
            kernel_size=kernels[0])

        self.skip1 = nn.Sequential(nn.Conv2d(2*dim[0], dim[0], 1),
                                   nn.Hardswish())
        self.down1 = Downsample(dim[0], dim[1])

        self.layer2 = HGMambaLayer(
            dim[1], d_state=1, depth=num_blocks[1], 
            kernel_size=kernels[1])

        self.skip2 = nn.Sequential(nn.Conv2d(2*dim[1], dim[1], 1),
                                   nn.Hardswish())
        self.down2 = Downsample(dim[1], dim[2])

        self.layer3 = HGMambaLayer(
            dim[2], d_state=1, depth=num_blocks[2], 
            kernel_size=kernels[2])

        self.up1 = Upsample(dim[2], dim[3])

        self.layer4 = HGMambaLayer(
            dim=dim[3], d_state=1, depth=num_blocks[3], 
            kernel_size=kernels[3])

        self.up2 = Upsample(dim[3], dim[4])

        self.layer5 = HGMambaLayer(
            dim=dim[4], d_state=1, depth=num_blocks[4], 
            kernel_size=kernels[4])

        self.conv_last = nn.Sequential(
                nn.Conv2d(dim[4], dim[4] // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim[4] // 4, dim[4] // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim[4] // 4, out_channels, 3, 1, 1))

    def forward(self, x):
        # normlize
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        res = x
        x = self.stem_in(x)

        x1 = x
        x = self.layer1(x)
        x = self.down1(x)

        x = self.layer2(x)
        x2 = x
        x = self.down2(x)

        x = self.layer3(x)

        x = torch.cat([self.up1(x), x2], dim=1)
        x = self.skip2(x)
        x = self.layer4(x)

        x = torch.cat([self.up2(x), x1], dim=1)
        x = self.skip1(x)
        x = self.layer5(x)

        x = self.conv_last(x)

        x = x + res
        
        x = x / self.img_range + self.mean
        return x
