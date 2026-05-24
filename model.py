import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """Overlapped patch embedding used by SegFormer."""

    def __init__(self, in_ch, embed_dim, patch_size, stride):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """SegFormer efficient self-attention with spatial reduction."""

    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """MLP with depth-wise convolution for positional mixing."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, h, w):
        b, n, _ = x.shape
        x = self.fc1(x)
        hidden_dim = x.shape[-1]
        x = x.transpose(1, 2).reshape(b, hidden_dim, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SegFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class SegFormerStage(nn.Module):
    def __init__(self, in_ch, embed_dim, depth, num_heads, sr_ratio, patch_size, stride):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [
                SegFormerBlock(embed_dim, num_heads, sr_ratio)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, h, w)
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class SegFormerB0(nn.Module):
    """Lightweight SegFormer-B0 for binary nucleus segmentation."""

    def __init__(self, in_ch=3, out_ch=1, decoder_dim=128):
        super().__init__()
        dims = [32, 64, 160, 256]
        depths = [2, 2, 2, 2]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]

        self.stage1 = SegFormerStage(
            in_ch, dims[0], depths[0], heads[0], sr_ratios[0], patch_size=7, stride=4
        )
        self.stage2 = SegFormerStage(
            dims[0], dims[1], depths[1], heads[1], sr_ratios[1], patch_size=3, stride=2
        )
        self.stage3 = SegFormerStage(
            dims[1], dims[2], depths[2], heads[2], sr_ratios[2], patch_size=3, stride=2
        )
        self.stage4 = SegFormerStage(
            dims[2], dims[3], depths[3], heads[3], sr_ratios[3], patch_size=3, stride=2
        )

        self.linear_c1 = nn.Conv2d(dims[0], decoder_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(dims[1], decoder_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(dims[2], decoder_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(dims[3], decoder_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(decoder_dim, out_ch, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        target_size = c1.shape[2:]
        c1 = self.linear_c1(c1)
        c2 = F.interpolate(self.linear_c2(c2), size=target_size, mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.linear_c3(c3), size=target_size, mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.linear_c4(c4), size=target_size, mode="bilinear", align_corners=False)

        x = self.fuse(torch.cat([c1, c2, c3, c4], dim=1))
        x = self.pred(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
