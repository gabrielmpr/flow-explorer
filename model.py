"""The U-Net from the MIT 6.S184 lab-2 script (flow_matching_dogs.py), unchanged except for width.

One network, u_theta(x, t). The score is not learned separately: for the straight path
x_t = t z + (1 - t) eps, it follows from the velocity in closed form (see score_from_u).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('freqs', torch.exp(torch.linspace(0, math.log(1000), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, 4 * dim))

    def forward(self, t):                       # t: (B, 1)
        args = t * self.freqs                   # (B, dim/2)
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, te):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(te)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, n_ch=64, dropout=0.1):
        super().__init__()
        c, t_dim = n_ch, 4 * n_ch
        self.n_ch = n_ch
        self.time_emb = TimeEmbedding(c)
        self.inp = nn.Conv2d(3, c, 3, padding=1)
        self.down1 = ResBlock(c, c, t_dim, dropout)            # 32x32
        self.down2 = ResBlock(c, 2 * c, t_dim, dropout)        # 16x16
        self.down3 = ResBlock(2 * c, 4 * c, t_dim, dropout)    # 8x8
        self.mid = ResBlock(4 * c, 4 * c, t_dim, dropout)      # 4x4
        self.up3 = ResBlock(8 * c, 2 * c, t_dim, dropout)      # 8x8
        self.up2 = ResBlock(4 * c, c, t_dim, dropout)          # 16x16
        self.up1 = ResBlock(2 * c, c, t_dim, dropout)          # 32x32
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x, t):
        te = self.time_emb(t)
        h1 = self.down1(self.inp(x), te)
        h2 = self.down2(F.avg_pool2d(h1, 2), te)
        h3 = self.down3(F.avg_pool2d(h2, 2), te)
        h = self.mid(F.avg_pool2d(h3, 2), te)
        h = self.up3(torch.cat([F.interpolate(h, scale_factor=2), h3], dim=1), te)
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2), h2], dim=1), te)
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=2), h1], dim=1), te)
        return self.out(h)


def score_from_u(u, x, t):
    """Score of p_t from the velocity, for alpha_t = t, beta_t = 1 - t.

    u = E[z - eps | x] and x = t z + (1 - t) eps give E[eps | x] = x - t u,
    and the Gaussian path's score is -E[eps | x] / (1 - t). So score = (t u - x) / (1 - t).
    """
    t = t[:, :, None, None]
    return (t * u - x) / (1 - t)


def predicted_endpoint(u, x, t):
    """E[z | x_t]: where the straight line through x with velocity u lands at t = 1."""
    return x + (1 - t[:, :, None, None]) * u


@torch.no_grad()
def sample(model, x, n_steps=100, sigma=0.0, t0=0.0, gen=None, return_path=False):
    """Euler-Maruyama for dX = [u + s_t^2/2 * score] dt + s_t dW from t0 to 1, with s_t = sigma * sqrt(1 - t).

    Any noise schedule s_t keeps the same marginals p_t. This one fades the noise out by t = 1, and
    s_t^2/2 * score = sigma^2/2 * (t u - x) has no 1/(1 - t) left in it. sigma = 0 is the ODE.
    """
    h = (1 - t0) / n_steps
    path = [x]
    for i in range(n_steps):
        tv = t0 + i * h
        t = torch.full((x.shape[0], 1), tv, device=x.device)
        u = model(x, t)
        if sigma > 0:
            x = x + h * (u + 0.5 * sigma ** 2 * (tv * u - x)) \
                + sigma * math.sqrt((1 - tv) * h) * torch.randn(x.shape, device=x.device, generator=gen)
        else:
            x = x + h * u
        if return_path:
            path.append(x)
    return (x, path) if return_path else x
