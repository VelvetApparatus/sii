import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Time embedding ---------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float()

        if half <= 1:
            # безопасный случай для маленьких dim
            emb = torch.zeros((t.shape[0], self.dim), device=device, dtype=t.dtype)
            if self.dim >= 1:
                emb[:, 0] = t
            return emb

        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / (half - 1)
        )  # (half,)
        args = t[:, None] * freqs[None, :]  # (B, half)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# --------- Blocks ---------
def norm(num_channels: int) -> nn.GroupNorm:
    # GroupNorm требует num_channels % groups == 0
    g = min(32, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        t_add = self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = h + t_add
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# --------- UNet ---------
class UNetDDPM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- Down path (по уровням) ----
        self.down_levels = nn.ModuleList()
        ch = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch

            downsample = Downsample(ch) if i != len(channel_mults) - 1 else None
            self.down_levels.append(nn.ModuleDict({"blocks": blocks, "down": downsample}))

        # ---- Middle ----
        self.mid1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ---- Up path (зеркало down) ----
        self.up_levels = nn.ModuleList()
        # важно: идём по уровням в обратном порядке
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult

            upsample = Upsample(ch) if i != len(channel_mults) - 1 else None

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                # сюда будем конкатить skip с таким же out_ch, поэтому in = ch + out_ch
                blocks.append(ResBlock(ch + out_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch

            self.up_levels.append(nn.ModuleDict({"up": upsample, "blocks": blocks}))

        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x)

        skips = []

        # Down: сохраняем skip только после ResBlock
        for level in self.down_levels:
            for block in level["blocks"]:
                h = block(h, t_emb)
                skips.append(h)
            if level["down"] is not None:
                h = level["down"](h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Up: на уровне (если надо) upsample, затем (concat + ResBlock)×N
        for level in self.up_levels:
            if level["up"] is not None:
                h = level["up"](h)

            for block in level["blocks"]:
                s = skips.pop()

                # страховка по spatial (на случай нечётных H/W)
                if h.shape[-2:] != s.shape[-2:]:
                    # лучше подгонять h под s (или наоборот) — главное одинаково
                    h = F.interpolate(h, size=s.shape[-2:], mode="nearest")

                h = torch.cat([h, s], dim=1)
                h = block(h, t_emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
