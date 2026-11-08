import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out))
        self.norm1 = nn.GroupNorm(groups, dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)       # (B, HW, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, dim=16, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )

        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))           # [(16,16),(16,32),(32,64),(64,128)]
        n = len(in_out)

        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResBlock(d_in, d_out, time_dim),
                ResBlock(d_out, d_out, time_dim),
                nn.Conv2d(d_out, d_out, 4, stride=2, padding=1) if i < n - 1 else nn.Identity(),
            ]))

        mid = dims[-1]
        self.mid1 = ResBlock(mid, mid, time_dim)
        self.mid_attn = Attention(mid)
        self.mid2 = ResBlock(mid, mid, time_dim)

        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResBlock(d_out * 2, d_in, time_dim),      # concat with skip
                ResBlock(d_in, d_in, time_dim),
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(d_in, d_in, 3, padding=1)) if i < n - 1 else nn.Identity(),
            ]))

        self.final_res = ResBlock(dim, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, t):
        x = self.init_conv(x)
        t_emb = self.time_mlp(t)

        skips = []
        for res1, res2, down in self.downs:
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for res1, res2, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = up(x)

        x = self.final_res(x, t_emb)
        return self.final_conv(x)
