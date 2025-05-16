# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ConvBnTanh2d(nn.Module):
    #convolution batch normalization tanh
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3(in_chsnnels, out_channels):
    "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out

class Feature_extract(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Feature_extract, self).__init__()
        self.SFEB1 = nn.Sequential(
            #            1         16
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
            nn.Conv2d(int(out_channels/2), int(out_channels/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
        )
        self.SFEB2= nn.Sequential(
            nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels),
            nn.Conv2d(out_channels,  out_channels, kernel_size=3, stride=1, padding=1),)

    def forward(self, x):        
        high_x = self.SFEB1(x)
        x = self.SFEB2(high_x)
        return high_x, x
    
class CMDFB(nn.Module):
    
    # iterative dual-branch attention module (IDATM)

    def __init__(self, in_channels, r=4):
        super(CMDFB, self).__init__()
        inter_channels = int(in_channels // r)
        self.pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.pw3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)

        self.first_local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )


        self.first_global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        self.second_local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
 
        self.second_global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()
          
    def forward(self, x, y):
        
       
        x_y_add = x + y
        x_y_pw1 = self.pw1(x_y_add)
        conv1 = self.bn(x_y_pw1)
        conv_at = self.relu(conv1)
        conv_out = self.bn(self.pw2(conv_at))
        x_y_local = self.first_local_att(conv_out)
        x_y_global = self.first_global_att(conv_out)
        x_y_lg = x_y_local + x_y_global
        
       
        x_input2 = x + x_y_lg
        y_input2 = y + x_y_lg
        x_y_add2 = torch.cat((x_input2,y_input2),dim=1)
        x_y_pw2 = self.pw3(x_y_add2)
        x_y_local2 = self.second_local_att(x_y_pw2)
        x_y_global2 = self.second_global_att(x_y_pw2)
        xlg2 = x_y_local2 + x_y_global2
        A_weight2 = self.sigmoid(xlg2)
        x_out = x * A_weight2
        y_out = y * (1 - A_weight2)

        return x_out, y_out
    
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops
    
class Mlp_shallow(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_shallow(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    def forward(self, x, x_size):
        B,C,H,W= x.shape
        x=x.view(B,H,W,C)
        shortcut = x
        shape=x.view(H*W*B,C)
        x = self.norm1(shape)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        B,H,W,C=x.shape
        x=x.view(B,C,H,W)
        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops
    
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# =============================================================================
import numbers
##########################################################################
## Layer Norm
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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
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
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim*ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
       
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
 
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x  
    
class Atten_dis_vis(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Atten_dis_vis, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv_x = self.qkv_dwconv(self.qkv(x))
        qkv_y = self.qkv_dwconv(self.qkv(y))
 
        q_x, k_x, v_x = qkv_x.chunk(3, dim=1)
        q_y, k_y, v_y = qkv_y.chunk(3, dim=1)
        q_x = rearrange(q_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q_y = rearrange(q_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_x = rearrange(k_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_y = rearrange(k_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_x = rearrange(v_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_y = rearrange(v_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        #归一化
        q_x = torch.nn.functional.normalize(q_x, dim=-1)
        q_y = torch.nn.functional.normalize(q_y, dim=-1)
        k_x = torch.nn.functional.normalize(k_x, dim=-1)
        k_y = torch.nn.functional.normalize(k_y, dim=-1)

        attn = (q_y @ k_x.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = v_x - (attn @ v_x)

        out_x = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out_x = self.project_out(out_x)
        return out_x
    
class Atten_dis_inf(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Atten_dis_inf, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)

        self. qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv_x = self.qkv_dwconv(self.qkv(x))
        qkv_y = self.qkv_dwconv(self.qkv(y))

        q_x, k_x, v_x = qkv_x.chunk(3, dim=1)
        q_y, k_y, v_y = qkv_y.chunk(3, dim=1)
    
        q_x = rearrange(q_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q_y = rearrange(q_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_x = rearrange(k_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_y = rearrange(k_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_x = rearrange(v_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_y = rearrange(v_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q_x = torch.nn.functional.normalize(q_x, dim=-1)
        q_y = torch.nn.functional.normalize(q_y, dim=-1)
        k_x = torch.nn.functional.normalize(k_x, dim=-1)
        k_y = torch.nn.functional.normalize(k_y, dim=-1)

        attn = (q_x @ k_y.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = v_y - (attn @ v_y)
        out_y = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out_y)
        return out_y
    
    
class DICM_vis(nn.Module):
    def __init__(self,dim,num_heads, bias, LayerNorm_type):
        super(DICM_vis,self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.atten_dis = Atten_dis_vis(dim, num_heads, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.mlp = Mlp(dim, hidden_features,ffn_expansion_factor, bias)
        
    def forward(self, x, y):
        x = x + self.atten_dis(self.norm1(x), self.norm1(y))
        # x = x + self.mlp(self.norm2(x))
        
        return x
    
class DICM_inf(nn.Module):
    def __init__(self,dim,num_heads, bias, LayerNorm_type):
        super(DICM_inf,self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.atten_inf = Atten_dis_inf(dim, num_heads, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.mlp = Mlp(dim, hidden_features,ffn_expansion_factor, bias)
        
    def forward(self, x, y):
        y = y + self.atten_inf(self.norm1(x), self.norm1(y))
        # y = y + self.mlp(self.norm2(y))
        
        return y
        
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__() 
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode_rescon1(nn.Module):
    def __init__(self,input):
        super(DetailNode_rescon1, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=int(input/2), oup=int(input/2), expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=int(input/2), oup=int(input/2), expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=int(input/2), oup=int(input/2), expand_ratio=2)
        self.shffleconv = nn.Conv2d(input, input, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2  
    
class DetailFeatureExtraction1(nn.Module):

    def __init__(self,in_put,num_layers=1):
        super(DetailFeatureExtraction1, self).__init__()
        INNmodules = [DetailNode_rescon1(input=in_put) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)

        # return torch.cat((z1, z2), dim=1)
        z = z1 + z2
        return z
    

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class FusionNet(nn.Module):
    def __init__(self,img_size=128,patch_size=4, window_size=1, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True,
                 depth=2, downsample=None, drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False,inputs=224,dim_swin=48,dim_restormer=64,
                 inp_channels=1,out_conv1=16,out_conv2=32,out_conv3=64,
                 out_channels=1,dim=112,num_blocks=[2, 4],heads=[4, 8],
                 ffn_expansion_factor=2,bias=False,LayerNorm_type='WithBias',):
        super(FusionNet, self).__init__()
        
        self.sefm_vis = Feature_extract(inp_channels,out_conv2)
        self.sefm_inf = Feature_extract(inp_channels,out_conv2)
        self.cmdf1 = CMDFB(out_conv1)
        self.cmdf2 = CMDFB(out_conv2)
        self.cmdf3 = CMDFB(out_conv3)
        # self.sim = SIM(norm_nc=32,label_nc=64,nhidden=32)
        self.patch_embed_vis = OverlapPatchEmbed(out_conv2,out_conv3)
        self.patch_embed_inf = OverlapPatchEmbed(out_conv2,out_conv3)
        
        self.relu=nn.ReLU(True)
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer=BasicLayer(dim= dim_swin,
                                   input_resolution=(patches_resolution[0],patches_resolution[1]),
                                         depth=depth,
                                         num_heads=heads[0],
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        self.restormer = nn.Sequential(*[TransformerBlock(dim=dim_restormer, num_heads=heads[0], 
                        ffn_expansion_factor=ffn_expansion_factor,bias=bias, 
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.dicm_vis = DICM_vis(dim=dim, num_heads=heads[1], bias=bias, LayerNorm_type=LayerNorm_type)
        self.dicm_inf = DICM_inf(dim=dim, num_heads=heads[1], bias=bias, LayerNorm_type=LayerNorm_type)
        
        # 特征重建部分
        self.rescon1 = DetailFeatureExtraction1(in_put=inputs) 
        self.rescon2 = DetailFeatureExtraction1(in_put=int(inputs/2))
        self.rescon3 = DetailFeatureExtraction1(in_put=int(inputs/4))
        self.rescon4 = DetailFeatureExtraction1(in_put=int(inputs/8)) 
        self.rescon5 = ConvBnTanh2d(14, out_channels)
        
    def forward(self, image_vis, image_ir):
        
        vis_out1, vis_out2 = self.sefm_vis(image_vis)
        inf_out1, inf_out2 = self.sefm_inf(image_ir)
        
        vis_cmdf1, inf_cmdf1 = self.cmdf1(vis_out1, inf_out1)
        vis_cmdf2, inf_cmdf2 = self.cmdf2(vis_out2, inf_out2)
        vis_inf_cross1 = torch.cat((vis_cmdf1,inf_cmdf2),dim=1)
        vis_inf_cross2 = torch.cat((vis_cmdf2,inf_cmdf1),dim=1)
        msa1=self.relu(vis_inf_cross1)
        msa2=self.relu(vis_inf_cross2)
        msa1_input_size = (msa1.shape[2], msa1.shape[3])
        msa2_input_size = (msa2.shape[2], msa2.shape[3])
        vis_inf_msa1=self.basicLayer(msa1, msa1_input_size)
        vis_inf_msa2=self.basicLayer(msa2, msa2_input_size)
        
        vis_patch = self.patch_embed_vis(vis_out2)
        inf_patch = self.patch_embed_vis(inf_out2)
        
        vis_out_restormer = self.restormer(vis_patch)
        inf_out_restormer = self.restormer(inf_patch)
        
        vis_cmdf3,inf_cmdf3= self.cmdf3(vis_out_restormer, inf_out_restormer)
        
        vis_inf_cross3 = torch.cat((vis_cmdf3,vis_inf_msa1),dim=1)
        vis_inf_cross4 = torch.cat((inf_cmdf3,vis_inf_msa2),dim=1)
        
        vis_dicm_restormer = self.dicm_vis(vis_inf_cross3, vis_inf_cross4)
        inf_dicm_restormer = self.dicm_inf(vis_inf_cross3, vis_inf_cross4)
        vis_inf_dicm = torch.cat((vis_dicm_restormer,inf_dicm_restormer),dim=1)

        # rescon
        x_1 = self.rescon1(vis_inf_dicm)
        x_2 = self.rescon2(x_1)
        x_3 = self.rescon3(x_2)
        x_4 = self.rescon4(x_3)
        x = self.rescon5(x_4)

        return x


