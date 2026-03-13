import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t: (S,) or (B, S)
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(math.log(10000) / (half_dim - 1)))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # (..., dim)

class TransformerDiffusion(nn.Module):
    def __init__(self,
                 feature_dim: int = 1+4+64,
                 emb_size: int = 128,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # input projection
        self.input_proj = nn.Linear(feature_dim, emb_size)
        # positional/time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(emb_size),
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        # condition embedding
        self.cond_proj = nn.Linear(emb_size, emb_size)
        # transformer encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=emb_size*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # output projection back to features
        self.output_proj = nn.Linear(emb_size, feature_dim)

    def forward(self, x, t, cond):
        # x: (B, S, feature_dim), t: (S,) or (B, S), cond: (B, cond_dim)
        B, S, _ = x.shape
        # project input
        h = self.input_proj(x)  # (B, S, emb_size)
        # time embedding
        t_in = t
        if t.ndim == 1:
            t_in = t.unsqueeze(1).expand(-1, S)
        t_emb = self.time_emb(t_in)  # (B, S, emb_size)
        # condition embedding broadcast
        cond_emb = self.cond_proj(cond)  # (B, emb_size)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, S, -1)
        # sum all embeddings
        h = h + t_emb + cond_emb
        # transformer expects (B, S, emb_size) with batch_first=True
        h = self.transformer(h)  # (B, S, emb_size)
        # project to output
        out = self.output_proj(h)  # (B, S, feature_dim)
        return out

# 如何使用:
# model = TransformerDiffusion(feature_dim=6, seq_len=1000, emb_size=128,
#                              n_layers=6, n_heads=8, mlp_dim=512, cond_dim=256)
# noise_pred = model(noisy_sequence, timesteps, condition_vector)
