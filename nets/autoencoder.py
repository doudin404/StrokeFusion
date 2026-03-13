import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from einops import rearrange


# class VectorEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim: int = 3,
#         d_h: int = 64,
#         out_dim: int | None = None,
#         n_layers: int = 6,
#         n_heads: int = 8,
#         max_len: int = 64
#     ):
#         super().__init__()
#         # 投射输入到隐藏维
#         self.linear = nn.Linear(input_dim, d_h)
#         # 位置编码
#         self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_h))
#         # 可学习的[CLS]标记
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_h))
#         # Transformer 层，支持 batch_first
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_h,
#             nhead=n_heads,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         # 可选的投射到输出维度
#         self.project_out = nn.Linear(d_h, out_dim) if out_dim and out_dim != d_h else None

#     def forward(self, x, mask=None):
#         # x: (B, N_p, input_dim)
#         B, N, _ = x.shape
#         h = self.linear(x) + self.pos_enc[:, :N, :]
#         # 构造 batch 首位的 CLS token 并拼接
#         cls = self.cls_token.expand(B, -1, -1)  # (B,1,d_h)
#         seq = torch.cat([cls, h], dim=1)        # (B, N_p+1, d_h)
#         # Transformer 编码
#         out = self.transformer(seq, src_key_padding_mask=mask)
#         cls_rep = out[:, 0, :]                  # 取 CLS 位置表示
#         # 投射到目标维度
#         if self.project_out:
#             cls_rep = self.project_out(cls_rep)
#         return cls_rep  # (B, out_dim or d_h)


class VectorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        d_h: int = 64,
        out_dim: int | None = None,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 64
    ):
        super().__init__()
        # 投射输入到隐藏维
        self.linear = nn.Linear(input_dim, d_h)
        # 位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_h))
        # Transformer 层，支持 batch_first
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 注意力池化用的可训练查询向量
        self.read_query = nn.Parameter(torch.randn(1, 1, d_h))
        self.attn = nn.MultiheadAttention(embed_dim=d_h, num_heads=n_heads, batch_first=True)
        # 可选的投射到输出维度
        self.project_out = nn.Linear(d_h, out_dim) if out_dim and out_dim != d_h else None

    def forward(self, x, mask=None):
        # x: (B, N_p, input_dim)
        B, N, _ = x.shape
        # 输入投射并添加位置信息
        h = self.linear(x) + self.pos_enc[:, :N, :]
        # 直接对全部位置进行 Transformer 编码
        out = self.transformer(h, src_key_padding_mask=mask)
        # 使用注意力池化聚合序列表示
        query = self.read_query.expand(B, -1, -1)  # (B,1,d_h)
        attn_out, _ = self.attn(query, out, out, key_padding_mask=mask)
        rep = attn_out.squeeze(1)  # (B,d_h)
        # 投射到目标维度
        if self.project_out:
            rep = self.project_out(rep)
        return rep  # (B, out_dim or d_h)

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, dims=(128, 64, 32, 16, 8, 4), d_img=64):
        super().__init__()
        layers = []
        c = in_channels
        # 6层下采样: 64->32->16->8->4->2->1
        for d in dims:
            layers += [nn.Conv2d(c, d, 3, stride=2, padding=1), nn.ReLU()]
            c = d
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[-1], d_img)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).view(x.size(0), -1)
        return self.fc(h)

class VectorDecoder(nn.Module):
    def __init__(self, d_f=128, n_layers=6, n_heads=8, output_dim=3, max_len=48):
        super().__init__()
        self.fc = nn.Linear(d_f, d_f)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_f,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.pos_dec = nn.Parameter(torch.randn(1, max_len, d_f))
        self.out_lin = nn.Linear(d_f, output_dim)

    def forward(self, z):
        seq = self.fc(z).unsqueeze(1).repeat(1, self.pos_dec.size(1), 1)
        seq = seq + self.pos_dec
        out = self.transformer(seq)
        return self.out_lin(out)

class ImageDecoder(nn.Module):
    def __init__(self, d_f=128, dims=(4, 8, 16, 32, 64, 128), out_channel=1):
        super().__init__()
        layers = []
        in_d = d_f
        # 对应 ImageEncoder 的下采样 dims 反向上采样
        for d in dims:
            layers.append(nn.ConvTranspose2d(in_d, d, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_d = d
        layers.append(nn.Conv2d(in_d, out_channel, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        h = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(h)