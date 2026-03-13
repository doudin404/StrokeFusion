import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from einops import rearrange
from nets.autoencoder import VectorEncoder,VectorDecoder,ImageEncoder,ImageDecoder
from torchvision.models import vgg16, VGG16_Weights
import os
import matplotlib.pyplot as plt


class DualModalModel(pl.LightningModule):
    def __init__(
        self,
        d_h: int = 64,
        out_seq: int = 64,
        d_img: int = 64,
        d_f: int = 64,
        lambda_CE: float = 0.1,
        lambda_L1: float = 10,
        lambda_img: float = 10,
        lambda_KL: float = 0.001,
        lambda_smooth:float = 0,
        lr: float = 1e-4,
        in_channels: int = 1,
        N_p: int = 64,
        max_len:int=64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vec_enc = VectorEncoder(d_h=d_h, out_dim=out_seq, max_len=N_p)
        self.img_enc = ImageEncoder(in_channels=in_channels, d_img=d_img)
        fused_dim = out_seq + d_img
        self.fc_mu = nn.Linear(fused_dim, d_f)
        self.fc_logvar = nn.Linear(fused_dim, d_f)
        self.vec_dec = VectorDecoder(d_f=d_f, max_len=N_p)
        self.img_dec = ImageDecoder(d_f=d_f)
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters(): p.requires_grad = False
        self.percep_net = vgg

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, strokes, fields):
        z_seq = self.vec_enc(strokes)
        z_img = self.img_enc(fields)
        fused = torch.cat([z_seq, z_img], dim=-1)
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        z = self.reparameterize(mu, logvar)
        s_hat = self.vec_dec(z)
        I_hat = self.img_dec(z)
        return s_hat, I_hat, mu, logvar

    def encode(self, strokes, fields):
        """
        编码器：输入strokes和fields，输出mu, logvar, z
        """
        z_seq = self.vec_enc(strokes)
        z_img = self.img_enc(fields)
        fused = torch.cat([z_seq, z_img], dim=-1)
        z = self.fc_mu(fused)
        return z

    def decode(self, z):
        """
        解码器：输入z，输出s_hat和I_hat
        """
        s_hat = self.vec_dec(z)
        I_hat = self.img_dec(z)
        return s_hat, I_hat

    def perceptual_loss(self, pred, target):
        pred_rgb = pred.repeat(1,3,1,1)
        tgt_rgb = target.repeat(1,3,1,1)
        f_p = self.percep_net(pred_rgb)
        f_t = self.percep_net(tgt_rgb)
        return F.l1_loss(f_p, f_t)

    def compute_losses(self, strokes, fields, s_hat, I_hat, mu, logvar):
        # 1. 向量损失
        m = strokes[..., 2]
        ce = F.binary_cross_entropy_with_logits(s_hat[..., 2], m)
        l1 = ((strokes[..., :2] - s_hat[..., :2])**2 * m.unsqueeze(-1)).mean()

        # 2. 图像损失
        rec = F.mse_loss(I_hat, fields)
        percep = self.perceptual_loss(I_hat, fields)
        img_loss = rec + percep

        # 3. KL 散度
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 4. 基于曲率的条件平滑正则(未启用)
        # coords_gt = strokes[..., :2]
        # d1_gt = coords_gt[:, 1:, :] - coords_gt[:, :-1, :]
        # d2_gt = d1_gt[:, 1:, :] - d1_gt[:, :-1, :]
        # curv_gt = d2_gt.norm(dim=-1)
        # coords_pred = s_hat[..., :2]
        # d1_pred = coords_pred[:, 1:, :] - coords_pred[:, :-1, :]
        # d2_pred = d1_pred[:, 1:, :] - d1_pred[:, :-1, :]
        # curv_pred = d2_pred.norm(dim=-1)
        # diff = (curv_pred - curv_gt)
        # smooth_loss = torch.clamp(diff, min=0.0).pow(2).mean()

        # 总损失
        total_loss = (
            self.hparams.lambda_CE  * ce + 
            self.hparams.lambda_L1  * l1 +
            self.hparams.lambda_img * img_loss +
            self.hparams.lambda_KL  * kl
            # self.hparams.lambda_smooth * smooth_loss
        )
        return {
            'ce': ce,
            'l1': l1,
            'img': img_loss,
            'kl': kl,
            # 'smooth': smooth_loss,
            'total': total_loss
        }

    def training_step(self, batch, batch_idx):
        strokes = batch['strokes']
        fields = batch['dist_fields'].unsqueeze(1)
        s_hat, I_hat, mu, logvar = self(strokes, fields)

        losses = self.compute_losses(strokes, fields, s_hat, I_hat, mu, logvar)

        batch_size = batch['sample_idx'].max().item() + 1
        log_kwargs = {
            'prog_bar':   True,
            'on_step':    True,
            'on_epoch':   False,
            'batch_size': batch_size,
            'sync_dist':  True,
        }
        for k, v in losses.items():
            self.log(k, v, **log_kwargs)

        return losses['total']

    def validation_step(self, batch, batch_idx):
        strokes = batch['strokes']
        fields = batch['dist_fields'].unsqueeze(1)
        s_hat, I_hat, mu, logvar = self(strokes, fields)
        self._last_pred = {
            's_hat': s_hat.detach(),
            'I_hat': I_hat.detach(),
            'strokes_gt': strokes.detach(),
            'fields_gt': batch['dist_fields'].detach()
        }
        losses = self.compute_losses(strokes, fields, s_hat, I_hat, mu, logvar)

        batch_size = batch['sample_idx'].max().item() + 1
        log_kwargs = {
            'prog_bar':   False,
            'on_step':    False,
            'on_epoch':   True,
            'batch_size': batch_size,
            'sync_dist':  True,
        }
        for k, v in losses.items():
            self.log(f'val_{k}', v, **log_kwargs)

        return losses['total']

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % 200 == 0 and hasattr(self, '_last_pred') and self.trainer.is_global_zero:
            pred = self._last_pred
            strokes_gt = pred['strokes_gt'][0].cpu().numpy()
            field_gt   = pred['fields_gt'][0].cpu().numpy()
            strokes_pt = pred['s_hat'][0].cpu().numpy()
            field_pt   = pred['I_hat'][0,0].cpu().numpy()
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes[0,0].plot(strokes_gt[:, 0], strokes_gt[:, 1]); axes[0,0].set_title('GT Stroke'); axes[0,0].axis('equal'); axes[0,0].axis('off')
            axes[0,1].plot(strokes_pt[:, 0], strokes_pt[:, 1]); axes[0,1].set_title('Pred Stroke'); axes[0,1].axis('equal'); axes[0,1].axis('off')
            axes[1,0].imshow(field_gt, cmap='gray', origin='lower', aspect='equal'); axes[1,0].set_title('GT Distance Field'); axes[1,0].axis('off')
            axes[1,1].imshow(field_pt, cmap='gray', origin='lower', aspect='equal'); axes[1,1].set_title('Pred Distance Field'); axes[1,1].axis('off')
            plt.tight_layout()
            self.logger.experiment.add_figure(
                'train/batch0_GT_vs_Pred', fig, self.current_epoch
            )
            os.makedirs("output", exist_ok=True)
            fig_path = f"output/train_GT_vs_Pred_epoch{self.current_epoch}_batch{batch_idx}.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)