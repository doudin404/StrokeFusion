import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from nets.diffusion import TransformerDiffusion
from models.stroke_fusion import DualModalModel
from diffusers import DDPMScheduler
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

# 假设 TransformerDiffusion 已经定义，如上文 canvas 中
# from transformer_diffusion import TransformerDiffusion

class SketchDiffusion(pl.LightningModule):
    def __init__(self,
                 feature_dim: int = 1+4+64,
                 emb_size: int = 128,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 timesteps: int = 1000,
                 lr: float = 1e-4,
                 warmup_epochs: int = 10,
                 num_classes: int = 1):  # 新增num_classes参数
        super().__init__()
        # 条件嵌入
        self.cond_embd = nn.Embedding(num_classes, emb_size)
        # 定义去噪模型
        self.model = TransformerDiffusion(
            feature_dim=feature_dim,
            emb_size=emb_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        # 定义调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="linear"
        )
        self.timesteps = timesteps
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes

    def test(self,t,x,cond):
        """使用add_noise加噪,使用_predict_noise预测噪声,计算均方误差"""
        noise = torch.randn_like(x)
        t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(x, noise, t_batch)
        noise_pred = self(noisy, t_batch,cond)
        mse = F.mse_loss(noise_pred, noise, reduction='mean')
        print(f"1:t={t}, MSE={mse.item()}")

    def forward(self, x, t, cond):
        return self.model(x, t, cond)

    def compute_loss(self, batch, stage):
        seqs = batch['seqs']  # (B, S, feature_dim)
        cond = batch['cond']  # (B,)
        
        seqs[...,0:1]*=0.5
        #seqs[...,1:]=seqs[...,0:1]
        B, S, _ = seqs.shape
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(seqs)
        noisy = self.scheduler.add_noise(seqs, noise, t)
        cond_embd = self.cond_embd(cond)
        noise_pred = self(noisy, t, cond_embd)
        
        mse = F.mse_loss(noise_pred, noise, reduction='none')  # (B, S, D)
        weights = 1#torch.where(seqs[...,0]>0, 1, 0.0005).unsqueeze(-1)  # (B, S, 1)
        loss = (mse * weights).mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # 定义一个 LambdaLR，用来在前 warmup_epochs 内做线性增大
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                # 线性 warm-up
                return float(current_epoch + 1) / float(self.warmup_epochs)
            return 1

        warmup_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "epoch",      # 按 epoch 更新
                "frequency": 1,
            },
        }
    




class SketchDiffusionSampler:
    def __init__(self,
            diffusion_module,  # 传入 SketchDiffusion LightningModule 实例
            dual_model=None,   # DualModalModel 实例（仅用于 decode）
            samples_dir: str = 'samples',
            device: torch.device | str = None,
            rdp_simplify: bool = True,  # 新增 rdp_simplify 参数
            viz_intermediate: bool = False,              # 新增：是否保存中间过程
            viz_steps: list[int] | None = None):         # 新增：保存的步点
        # 设备设置
        self.viz_intermediate = viz_intermediate
        default_steps = [1000,500, 200, 100, 50, 20, 10, 0]
        self.viz_steps = set(default_steps if viz_steps is None else viz_steps)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(samples_dir, exist_ok=True)
        self.samples_dir = Path(samples_dir)

        # 扩散模型及调度器
        self.diffusion_module = diffusion_module.to(self.device)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_module.timesteps,
            beta_schedule='linear',
            clip_sample=False,
        )
        self.cond_emb_layer = self.diffusion_module.cond_embd
        self.feature_dim = self.diffusion_module.model.input_proj.in_features
        self.cond_dim = self.cond_emb_layer.embedding_dim
        self.timesteps = self.diffusion_module.timesteps
        self.dual_model = dual_model.to(self.device) if dual_model is not None else None
        self.rdp_simplify = rdp_simplify

    def _rdp(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """Ramer-Douglas-Peucker 简化算法"""
        if points.shape[0] < 3:
            return points
        start, end = points[0], points[-1]
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len == 0:
            dists = np.linalg.norm(points - start, axis=1)
        else:
            proj = ((points - start) @ seg) / (seg_len ** 2)
            proj_point = np.outer(proj, seg) + start
            dists = np.linalg.norm(points - proj_point, axis=1)
        idx = np.argmax(dists)
        dmax = dists[idx]
        if dmax > epsilon:
            left = self._rdp(points[:idx+1], epsilon)
            right = self._rdp(points[idx:], epsilon)
            return np.vstack((left[:-1], right))
        else:
            return np.vstack((start, end))

    @torch.no_grad()
    def _sample_rec(self,
                    cond_ids=None,
                    batch_size: int = 1,
                    max_len: int = 32,
                    return_intermediates: bool = False):
        """
        根据 cond_ids 生成编码。
        return_intermediates=True 时，额外返回 {t: rec_numpy} 的字典，t 为整数时间步。
        """
        cond_emb = None
        if cond_ids is not None:
            cond_ids = torch.tensor(cond_ids, device=self.device)
            cond_emb = self.cond_emb_layer(cond_ids)

        x = torch.randn(batch_size, max_len, self.feature_dim, device=self.device)

        # 依据实际总步数过滤/归一化可视化步点（保证都在 [0, timesteps-1]）
        if return_intermediates:
            max_t = self.timesteps - 1
            viz_steps = {int(max(0, min(max_t, s))) for s in self.viz_steps}
        intermediates = {} if return_intermediates else None

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            inp = cond_emb if cond_emb is not None else torch.zeros(batch_size, self.cond_dim, device=self.device)
            noise_pred = self.diffusion_module(x, t_batch, inp)
            step = self.scheduler.step(noise_pred, t, x)
            x = step.prev_sample

            if return_intermediates and (t in viz_steps):
                # 存的是当前步完成后的样本（x 已更新为 prev_sample）
                intermediates[t] = x.detach().cpu().numpy()

        final = x.detach().cpu().numpy()
        return (final, intermediates) if return_intermediates else final

    def render_and_save(self,
                        rec: np.ndarray,
                        prefix: str = '',
                        flip_vertical: bool = False,
                        color: bool = False) -> None:
        """
        保存 rec 编码并渲染草图，图片命名含 prefix。
        """
        # 保存编码
        npz_path = self.samples_dir / f'{prefix}_encoding.npz'
        np.savez(npz_path, encoding=rec)

        if self.dual_model is None:
            return

        seqs = torch.from_numpy(rec).to(self.device)
        lengths = torch.full((rec.shape[0],), rec.shape[1], dtype=torch.long)
        for i in range(rec.shape[0]):
            fig, ax = plt.subplots()
            seq = seqs[i]
            for j in range(lengths[i]):
                flag = seq[j, 0].item()
                if flag <= 0:
                    continue
                bbox = seq[j, 1:5].cpu().numpy()
                enc = seq[j,5:].unsqueeze(0).to(next(self.dual_model.parameters()).device)
                with torch.no_grad():
                    strokes_hat, _ = self.dual_model.decode(enc)
                pts = strokes_hat.squeeze(0).cpu().numpy()

                # RDP 简化: 先缩放到 256×256，再简化，然后归一化回 [0,1]
                if self.rdp_simplify:
                    pts_scaled = pts * 256.0
                    pts_simpl = self._rdp(pts_scaled, epsilon=2.0)
                    pts = pts_simpl / 256.0

                cx, cy, w, h = bbox.tolist()
                scale = max(w, h)
                xs = pts[:,0]*scale + cx
                ys = pts[:,1]*scale + cy
                if flip_vertical:
                    ys = -ys
                ax.plot(xs, ys, color=None if color else 'black')
            ax.axis('equal')
            ax.axis('off')
            fig.tight_layout()
            img_path = self.samples_dir / f'{prefix}_sketch_{i}.png'
            fig.savefig(str(img_path), dpi=300,)
            img_path = self.samples_dir / f'{prefix}_sketch_{i}.svg'
            fig.savefig(str(img_path), dpi=300, format='svg')
            plt.close(fig)

    def sample(self,
               dataloader,
               num_samples: int,
               mode: str = 'random',
               epoch_prefix: str = 'run',
               max_len: int = None,
               flip_vertical: bool = False,
               color: bool = False,
               cond_rw=None) -> None:
        """
        一步方法：
        mode='random' 时，从 dataloader 中读取 cond 并随机生成 num_samples 个样本；
        mode='dataset' 时，从 dataloader 中读取 bboxes 和 strokes 并绘制真实数据；
        mode='recon' 时，从 dataloader 中读取 seqs 并用 render_and_save 渲染。
        输出文件名包含 epoch_prefix 和样本序号。
        """
        count = 0

        if mode == 'random':
            total = num_samples
            with tqdm(total=total, desc="Sampling (random mode)") as pbar:
                for batch in dataloader:
                    cond = batch.get('cond')
                    if cond is None:
                        raise KeyError('Batch 中缺少 cond 键')#hambuger
                    if cond_rw is not None:
                        cond = torch.ones_like(cond) * cond_rw
                    remaining = num_samples - count
                    if remaining <= 0:
                        break
                    curr = cond[:remaining]
                    B = curr.size(0)
                    L = max_len or curr.size(1)

                    if self.viz_intermediate and (self.dual_model is not None):
                        rec_final, snaps = self._sample_rec(curr.tolist(), B, L, return_intermediates=True)
                        # 先保存最终结果
                        self.render_and_save(rec_final, f"{epoch_prefix}_gen_{count}", flip_vertical, color)
                        # 再按时间步从大到小保存中间过程
                        for t in sorted(snaps.keys(), reverse=True):
                            self.render_and_save(snaps[t], f"{epoch_prefix}_gen_{count}_t{t}", flip_vertical, color)
                    else:
                        rec_final = self._sample_rec(curr.tolist(), B, L)
                        self.render_and_save(rec_final, f"{epoch_prefix}_gen_{count}", flip_vertical, color)

                    count += B
                    pbar.update(B)
        elif mode == 'dataset':
            total = len(dataloader.dataset) if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__') else num_samples
            with tqdm(total=min(total, num_samples), desc="Sampling (dataset mode)") as pbar:
                for batch in dataloader:
                    bboxes = batch.get('bboxes')
                    strokes = batch.get('strokes')
                    if bboxes is None or strokes is None:
                        raise KeyError('Batch 中缺少 bboxes 或 strokes 键')
                    B = bboxes.size(0)
                    for i in range(B):
                        if count >= num_samples:
                            return
                        fig, ax = plt.subplots()
                        box = bboxes[i]
                        stk = strokes[i].cpu().numpy()
                        for j in range(box.size(0)):
                            bbox = box[j].cpu().numpy()
                            if (bbox == 0).all():
                                continue
                            pts = stk[j]

                            # RDP 简化
                            if self.rdp_simplify:
                                pts_scaled = pts * 256.0
                                pts_simpl = self._rdp(pts_scaled, epsilon=2.0)
                                pts = pts_simpl / 256.0

                            cx, cy, w, h = bbox.tolist()
                            scale = max(w, h)
                            xs = pts[:,0]*scale + cx
                            ys = pts[:,1]*scale + cy
                            if flip_vertical:
                                ys = -ys
                            ax.plot(xs, ys, color=None if color else 'black')
                        ax.axis('equal')
                        ax.axis('off')
                        fig.tight_layout()
                        #img_path = self.samples_dir / f"{epoch_prefix}_data_{count}.png"
                        img_path = self.samples_dir / f"{epoch_prefix}_data_{count}.svg"
                        fig.savefig(str(img_path), format='svg', dpi=300)
                        plt.close(fig)
                        count += 1
                        pbar.update(1)
        elif mode == 'recon':
            total = num_samples
            with tqdm(total=total, desc="Sampling (recon mode)") as pbar:
                for batch in dataloader:
                    seqs = batch.get('seqs')
                    if seqs is None:
                        raise KeyError('Batch 中缺少 seqs 键')
                    B = seqs.shape[0]
                    remaining = num_samples - count
                    if remaining <= 0:
                        break
                    curr = seqs[:remaining].cpu().numpy()
                    self.render_and_save(curr, f"{epoch_prefix}_recon_{count}", flip_vertical, color)
                    count += curr.shape[0]
                    pbar.update(curr.shape[0])
        else:
            raise ValueError("mode 必须是 'random'、'dataset' 或 'recon'。")
    def _render_rec_to_svg(self,
                           rec: np.ndarray,
                           prefix: str = '',
                           flip_vertical: bool = False,
                           color: bool = False) -> None:
        """
        内部复用渲染逻辑，但只保存为 SVG（不再保存 encoding.npz）。
        """
        if self.dual_model is None:
            return

        seqs = torch.from_numpy(rec).to(self.device)
        lengths = torch.full((rec.shape[0],), rec.shape[1], dtype=torch.long)
        for i in range(rec.shape[0]):
            fig, ax = plt.subplots()
            seq = seqs[i]
            for j in range(lengths[i]):
                flag = seq[j, 0].item()
                if flag <= 0:
                    continue
                bbox = seq[j, 1:5].cpu().numpy()
                enc = seq[j,5:].unsqueeze(0).to(next(self.dual_model.parameters()).device)
                with torch.no_grad():
                    strokes_hat, _ = self.dual_model.decode(enc)
                pts = strokes_hat.squeeze(0).cpu().numpy()

                if self.rdp_simplify:
                    pts_scaled = pts * 256.0
                    pts_simpl = self._rdp(pts_scaled, epsilon=2.0)
                    pts = pts_simpl / 256.0

                cx, cy, w, h = bbox.tolist()
                scale = max(w, h)
                xs = pts[:,0]*scale + cx
                ys = pts[:,1]*scale + cy
                if flip_vertical:
                    ys = -ys
                ax.plot(xs, ys, color=None if color else 'black')
            ax.axis('equal')
            ax.axis('off')
            fig.tight_layout()
            svg_path = self.samples_dir/"svg"/ f'{prefix}_sketch_{i}.svg'
            svg_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            fig.savefig(str(svg_path), format='svg', dpi=300)
            plt.close(fig)

    def export_encodings_to_svg(self,
                                encodings_dir: str | Path,
                                flip_vertical: bool = False,
                                color: bool = False,
                                pattern: str = "*_encoding.npz") -> None:
        """
        批量读取某个目录下的 encoding .npz 文件，解码并以 SVG 保存草图。
        """
        enc_dir = Path(encodings_dir)
        if not enc_dir.is_dir():
            raise ValueError(f"{encodings_dir} 不是一个合法目录")

        files = sorted(enc_dir.glob(pattern))
        if not files:
            print(f"在 {encodings_dir} 下没有找到匹配 {pattern} 的文件。")
            return

        for npz_path in tqdm(files):
            try:
                data = np.load(npz_path)
            except Exception as e:
                print(f"加载 {npz_path} 失败: {e}")
                continue
            if 'encoding' not in data:
                print(f"{npz_path} 中不含 'encoding' 字段，跳过。")
                continue
            rec = data['encoding']  # 形状 (B, S, F)
            stem = npz_path.stem  # 例如 "run1_gen_encoding"
            base_prefix = stem.replace('_encoding', '')
            # 用 SVG 保存（不再重复存 encoding）
            self._render_rec_to_svg(rec, f"{base_prefix}", flip_vertical, color)
