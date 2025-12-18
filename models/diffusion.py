import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# Utils: Sinusoidal Embedding
# -------------------------
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int):
    """
    timesteps: (B,) int64 or float tensor
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------
# MLP Model for DDPM
# ---------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, t_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_dim),
        )

    def forward(self, x, t_emb):
        h = F.silu(self.fc1(x))
        h = h + self.time_mlp(t_emb)
        h = F.silu(self.fc2(h))
        h = self.fc3(h)
        return h


class DDPM_MLP(nn.Module):
    def __init__(self, input_dim=784, base_dim=512, t_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.t_dim = t_dim
        self.base_dim = base_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # MLP blocks
        self.block1 = MLPBlock(input_dim, base_dim, t_dim)
        self.block2 = MLPBlock(base_dim, base_dim, t_dim)
        self.block3 = MLPBlock(base_dim, base_dim, t_dim)

        self.fc_out = nn.Linear(base_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # Embed time
        h = self.block1(x, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        out = self.fc_out(h)
        return out

