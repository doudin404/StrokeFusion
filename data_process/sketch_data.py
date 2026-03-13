import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import matplotlib
from abc import abstractmethod
from typing import Optional

# 使用非交互式后端，避免弹出图形窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from svgpathtools import svg2paths2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def visualize_strokes(
    strokes: np.ndarray,
    bboxes: np.ndarray,
    save_path: Path | str = "output/output.png",
) -> None:
    """
    使用 scale = max(w, h) 均匀缩放并以中心坐标还原原始位置并绘制笔画。
    strokes: np.ndarray [num_strokes, path_points, 2]
    bboxes: np.ndarray [num_strokes, 4] (cx, cy, w, h)
    """
    plt.ioff()
    plt.figure()
    for stroke_norm, (cx, cy, w, h) in zip(strokes, bboxes):
        scale = max(w, h)
        xs = stroke_norm[:, 0] * scale + cx
        ys = stroke_norm[:, 1] * scale + cy
        plt.plot(xs, ys)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=300)
    plt.close()


def visualize_distance_fields(
    dist_fields: np.ndarray, save_dir: Path | str = "output/fields"
) -> None:
    """
    基于局部坐标系绘制居中且按 display_scale 放大后的距离场矩阵并分别保存。
    dist_fields: np.ndarray [num_strokes, grid, grid]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.ioff()
    for i, field in enumerate(dist_fields):
        plt.figure()
        plt.imshow(field, origin="lower", aspect="equal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(save_dir / f"field_{i}.png"), dpi=300)
        plt.close()


class BaseSketchDataset(Dataset):
    """
    基类：统一处理 seq -> strokes -> normalized strokes & distance fields，并支持缓存与可选的编码器。
    子类需实现 `_get_seq(idx)` 从不同数据源加载原始序列。
    """

    EPS = 1e-6

    def __init__(
        self,
        path_points: int = 64,
        grid_size: int = 64,
        gamma: float = 100.0,
        display_scale: float = 0.8,
        cache_dir: str | Path | None = None,
        encoder_model: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.path_points = path_points
        self.grid_size = grid_size
        self.gamma = gamma
        self.display_scale = display_scale
        self.encoder = encoder_model

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    @abstractmethod
    def _get_seq(self, idx: int) -> np.ndarray:
        """子类实现：返回 shape [L,3] 的增量 + 笔触标志序列。"""
        ...

    @abstractmethod
    def _get_class(self, idx: int) -> int:
        """子类实现：返回序号对应的草图的类型整数。"""
        ...

    def __getitem__(self, idx: int):
        strokes, fields, bboxes, encoding = self._process_seq(idx)

        result = {
            "strokes": torch.from_numpy(strokes).float(),
            "dist_fields": torch.from_numpy(fields).float(),
            "bboxes": torch.from_numpy(bboxes).float(),
        }
        # 只有子类实现了 _get_class 时才加入 "class"
        if type(self)._get_class is not BaseSketchDataset._get_class:
            result["class"] = self._get_class(idx)
        if encoding is not None:
            result["encoding"] = torch.from_numpy(encoding).float()
        return result

    def _process_seq(self, idx: int):
        """
        统一的缓存 & 计算逻辑：
         - parse_seq_to_strokes
         - normalize_strokes
         - compute_distance_fields
         - 可选 encoder 前向 + 缓存编码
        """
        # 定义缓存文件路径
        if self.cache_dir:
            strokes_p = self.cache_dir / f"strokes_{idx}.npy"
            fields_p = self.cache_dir / f"fields_{idx}.npy"
            bboxes_p = self.cache_dir / f"bboxes_{idx}.npy"
            enc_p = self.cache_dir / f"encoding_{idx}.npy"
        else:
            strokes_p = fields_p = bboxes_p = enc_p = None

        # 加载或计算 strokes, bboxes, fields
        if strokes_p and strokes_p.exists() and fields_p.exists() and bboxes_p.exists():
            strokes = np.load(strokes_p)
            fields = np.load(fields_p)
            bboxes = np.load(bboxes_p)
        else:
            # 1) 加载 seq
            seq = self._get_seq(idx)
            # 2) 从 seq 解析重采样笔画
            raw_strokes = self.parse_seq_to_strokes(seq)
            # 3) 全局 & per-stroke 归一化
            strokes, bboxes = self.normalize_strokes(raw_strokes)
            # 4) 距离场计算
            fields = self.compute_distance_fields(strokes, bboxes)
            # 5) 缓存到磁盘
            if self.cache_dir:
                np.save(strokes_p, strokes)
                np.save(fields_p, fields)
                np.save(bboxes_p, bboxes)

        # 编码器处理
        encoding = None
        if self.encoder:
            # 若已有缓存，则直接加载
            if enc_p and enc_p.exists():
                encoding = np.load(enc_p)
            else:
                # strokes: (b, 64, 2) -> (b, 64, 3) with third dim all 1s
                b, n, _ = strokes.shape
                ones = np.ones((b, n, 1), dtype=strokes.dtype)
                strokes_with_m = np.concatenate([strokes, ones], axis=-1)
                stroke_t = (
                    torch.from_numpy(strokes_with_m)
                    .float()
                    .to(next(self.encoder.parameters()).device)
                )
                # fields: (b, 64, 64) -> (b, 1, 64, 64)
                field_t = (
                    torch.from_numpy(fields).float().unsqueeze(1).to(stroke_t.device)
                )
                with torch.no_grad():
                    s_hat, I_hat, mu, logvar = self.encoder(stroke_t, field_t)
                    encoding_t = mu
                encoding = encoding_t.cpu().numpy()
                if self.cache_dir:
                    np.save(enc_p, encoding)

        return strokes, fields, bboxes, encoding

    # 原有方法保留：
    def resample_seg(self, seg: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
        cum = np.concatenate(([0], np.cumsum(d)))
        total = cum[-1]
        if total < self.EPS:
            return np.repeat(seg[:1], self.path_points, axis=0)
        ts = np.linspace(0, total, self.path_points)
        xs = np.interp(ts, cum, seg[:, 0])
        ys = np.interp(ts, cum, seg[:, 1])
        return np.stack([xs, ys], axis=1)

    def parse_seq_to_strokes(self, seq: np.ndarray) -> np.ndarray:
        deltas = seq[:, :2].astype(float)
        pen = seq[:, 2].astype(int)
        coords = np.cumsum(deltas, axis=0)
        coords[:, 1] *= -1
        strokes = []
        start = 0
        for i, p in enumerate(pen):
            if p == 1:
                seg = coords[start : i + 1]
                if len(seg) > 1:
                    strokes.append(self.resample_seg(seg))
                start = i + 1
        if start < len(coords):
            seg = coords[start:]
            if len(seg) > 1:
                strokes.append(self.resample_seg(seg))
        return (
            np.stack(strokes, axis=0) if strokes else np.zeros((0, self.path_points, 2))
        )

    def normalize_strokes(self, strokes: np.ndarray):
        if strokes.size == 0:
            return strokes, np.zeros((0, 4), dtype=float)
        pts = strokes.reshape(-1, 2)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        half = (mx - mn) / 2
        scale = max(half.max(), self.EPS)
        norm_all = (strokes - center) / scale
        bboxes, norms = [], []
        for s in norm_all:
            xs, ys = s[:, 0], s[:, 1]
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            w, h = (maxx - minx) / 2, (maxy - miny) / 2
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            bboxes.append([cx, cy, w, h])
            m = max(max(w, h), self.EPS)
            norms.append((s - [cx, cy]) / m)
        return np.stack(norms), np.array(bboxes, dtype=float)

    def compute_distance_fields(
        self, strokes_norm: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        G = self.grid_size
        xs = np.linspace(0, 1, G)
        Gx, Gy = np.meshgrid(xs, xs)
        fields = []
        for s, (cx, cy, w, h) in zip(strokes_norm, bboxes):
            s01 = (s * self.display_scale + 1) / 2
            field = np.zeros((G, G), dtype=float)
            for (x0, y0), (x1, y1) in zip(s01[:-1], s01[1:]):
                vx, vy = x1 - x0, y1 - y0
                wx, wy = Gx - x0, Gy - y0
                denom = vx * vx + vy * vy or self.EPS
                t = np.clip((wx * vx + wy * vy) / denom, 0, 1)
                px, py = x0 + t * vx, y0 + t * vy
                d2 = (Gx - px) ** 2 + (Gy - py) ** 2
                field = np.maximum(field, np.exp(-self.gamma * d2))
            fields.append(field)
        return np.stack(fields, axis=0) if fields else np.zeros((0, G, G))


class NpzDataset(BaseSketchDataset):
    """QuickDraw .npz 数据集"""

    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        split: str = "train",
        encoder_model: torch.nn.Module | None = None,
        lengths: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, encoder_model=encoder_model, **kwargs)
        self.npz_path = Path(root_dir)
        if not str(self.npz_path).endswith(".npz"):
            self.npz_path = Path(str(self.npz_path) + ".npz")
        data = np.load(str(self.npz_path), allow_pickle=True, encoding="latin1")
        self.seqs = data[split]
        self.lengths = (
            min(lengths, len(self.seqs)) if lengths is not None else len(self.seqs)
        )

    def __len__(self):
        return self.lengths  # len(self.seqs)

    def _get_seq(self, idx: int) -> np.ndarray:
        return self.seqs[idx]
    # def _get_class(self, idx: int) -> int:
    #     # QuickDraw .npz 文件通常不包含类别信息，返回 0 或其他默认值
    #     return 0


class SvgDataset(BaseSketchDataset):
    """SVG 文件夹数据集"""

    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        encoder_model: torch.nn.Module | None = None,
        lengths: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, encoder_model=encoder_model, **kwargs)
        self.files = sorted(Path(root_dir).rglob("*.svg"))
        self.lengths = (
            min(lengths, len(self.files)) if lengths is not None else len(self.files)
        )
        # 收集类别名（父文件夹名）并建立类别到整数的映射表，带缓存
        self.class_map_path = Path(cache_dir) / "class_map.npy" if cache_dir else None
        if self.class_map_path and self.class_map_path.exists():
            # 直接读取缓存
            class_map = np.load(self.class_map_path, allow_pickle=True).item()
        else:
            # 统计所有类别名
            class_names = sorted({f.parent.name for f in self.files})
            class_map = {name: idx for idx, name in enumerate(class_names)}
            if self.class_map_path:
                np.save(self.class_map_path, class_map)
        self.class_map = class_map
        self.class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    def _get_class(self, idx: int) -> int:
        # 返回当前文件所属类别的整数编号
        folder_name = self.files[idx].parent.name
        return self.class_map.get(folder_name, 0)

    def __len__(self):
        return self.lengths  # len(self.files)

    def _get_seq(self, idx: int) -> np.ndarray:
        try:
            paths_raw, _, attrs = svg2paths2(str(self.files[idx]))
        except Exception:
            #
            print(f"Error reading SVG file: {path}. Returning empty sequence.")
        vb = attrs.get("viewBox")
        vb_h = (
            float(vb.split()[3])
            if vb
            else max(seg.end.imag for p in paths_raw for seg in p)
        )
        coords_list, pen_list = [], []
        for p in paths_raw:
            if len(p) == 0 or p.length() < 1e-6:
                # 跳过没有段，或者长度近乎为 0 的 Path
                continue
            pts = np.array([p.point(t) for t in np.linspace(0, 1, self.path_points)])
            xs = pts.real.astype(float)
            ys = (vb_h - pts.imag).astype(float)
            for i in range(len(xs)):
                coords_list.append((xs[i], ys[i]))
                pen_list.append(1 if i == len(xs) - 1 else 0)
        diffs = np.diff(coords_list, axis=0, prepend=[coords_list[0]])
        pen_arr = np.array(pen_list, dtype=int)[:, None]
        seq = np.concatenate([diffs, pen_arr], axis=1)
        return seq


def strokes_collate_fn(batch):
    all_strokes, all_fields, all_bboxes, sample_idx = [], [], [], []
    for idx, item in enumerate(batch):
        strokes, fields, bboxes = item["strokes"], item["dist_fields"], item["bboxes"]
        for stroke, field, bbox in zip(strokes, fields, bboxes):
            # stroke: Tensor[N_p, 2]  -> 添加 m 标志 Tensor[N_p, 1]
            m = torch.ones(stroke.size(0), dtype=stroke.dtype, device=stroke.device)
            stroke_with_m = torch.cat([stroke, m.unsqueeze(-1)], dim=-1)  # [N_p,3]

            all_strokes.append(stroke_with_m)
            all_fields.append(field)
            all_bboxes.append(bbox)
            sample_idx.append(idx)

    return {
        "strokes": torch.stack(all_strokes, dim=0),  # [总笔画数, N_p, 3]
        "dist_fields": torch.stack(all_fields, dim=0),  # [总笔画数, G, G]
        "bboxes": torch.stack(all_bboxes, dim=0),  # [总笔画数, 4]
        "sample_idx": torch.tensor(
            sample_idx, dtype=torch.long, device=all_strokes[0].device  # [总笔画数]
        ),
    }


class StrokesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        cache_dir: str | None = None,
        data_name="quick_draw/airplane",
        data_type: str = "npz",  # 'svg' or 'npz'
        path_points: int = 64,
        grid_size: int = 64,
        gamma: float = 50.0,
        display_scale: float = 0.8,
        batch_size: int = 4,
        num_workers: int = 4,
        encoder_model: torch.nn.Module | None = None,  # 新增参数
    ):
        super().__init__()
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.data_name = data_name
        self.data_type = data_type
        self.path_points = path_points
        self.grid_size = grid_size
        self.gamma = gamma
        self.display_scale = display_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_model = encoder_model  # 保存参数

    def setup(self, stage=None):
        ds_kwargs = {
            "path_points": self.path_points,
            "grid_size": self.grid_size,
            "gamma": self.gamma,
            "display_scale": self.display_scale,
            "encoder_model": self.encoder_model,  # 传递参数
        }

        if self.data_type == "svg":
            root_dir = Path(self.root_dir) / self.data_name
            cache_dir = (
                Path(self.cache_dir) / self.data_name if self.cache_dir else None
            )
            full_ds = SvgDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                **ds_kwargs,
            )
            total = len(full_ds)
            train_len = int(total * 0.99)
            val_len = total - train_len
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(
                full_ds, [train_len, val_len],generator=generator
            )
        elif self.data_type == "npz":
            root_dir = Path(self.root_dir) / self.data_name
            cache_dir = (
                Path(self.cache_dir) / self.data_name if self.cache_dir else None
            )
            self.train_dataset = NpzDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                split="train",
                **ds_kwargs,
            )
            self.val_dataset = NpzDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                split="valid",
                **ds_kwargs,
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=strokes_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=strokes_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=strokes_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class SketchsDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root_dir: str,
        cache_dir: Optional[str] = None,
        data_name: str = "quick_draw/airplane",
        data_type: str = "npz",  # 'svg' or 'npz'
        path_points: int = 64,
        grid_size: int = 64,
        gamma: float = 50.0,
        display_scale: float = 0.8,
        batch_size: int = 4,
        num_workers: int = 0,
        encoder_model: Optional[torch.nn.Module] = None,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.data_name = data_name
        self.data_type = data_type
        self.path_points = path_points
        self.grid_size = grid_size
        self.gamma = gamma
        self.display_scale = display_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_model = encoder_model
        self.max_seq_len = max_seq_len  # 如果 None 则按 batch 内最长笔画序列填充

    def setup(self, stage=None):
        ds_kwargs = {
            "path_points": self.path_points,
            "grid_size": self.grid_size,
            "gamma": self.gamma,
            "display_scale": self.display_scale,
            "encoder_model": self.encoder_model,
        }

        if self.data_type == "svg":
            root_dir = Path(self.root_dir) / self.data_name
            cache_dir = (
                Path(self.cache_dir) / self.data_name if self.cache_dir else None
            )
            full_ds = SvgDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                **ds_kwargs,
            )
            total = len(full_ds)
            train_len = int(total * 0.99)
            val_len = total - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_ds, [train_len, val_len]
            )
        elif self.data_type == "npz":
            root_dir = Path(self.root_dir) / self.data_name
            cache_dir = (
                Path(self.cache_dir) / self.data_name if self.cache_dir else None
            )
            self.train_dataset = NpzDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                split="train",
                **ds_kwargs,
            )
            self.val_dataset = NpzDataset(
                root_dir=root_dir,
                cache_dir=cache_dir,
                split="valid",
                lengths=self.batch_size,
                **ds_kwargs,
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._sketchs_collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._sketchs_collate,
            persistent_workers=self.num_workers > 0,
        )

    def _sketchs_collate(self, batch):
        # batch: list of dicts, each dict has 'encoding': Tensor[num_strokes, D], 'bboxes': Tensor[num_strokes,4], 'strokes': Tensor[num_strokes, path_points, 2]
        batch_seqs = []
        batch_lens = []
        conds = []
        batch_bboxes = []
        batch_strokes = []
        has_class = all(
            "class" in item and isinstance(item["class"], int) for item in batch
        )
        max_len = self.max_seq_len
        for item in batch:
            enc = item["encoding"]
            bboxes = item["bboxes"]
            strokes = item["strokes"]
            n = enc.shape[0]
            # 跳过超出最大长度的数据
            if max_len is not None and n > max_len:
                continue
            flags = enc.new_ones(n, 1)
            seq = torch.cat([flags, bboxes, enc], dim=1)  # [n, 1+4+D]
            batch_seqs.append(seq)
            batch_lens.append(n)
            batch_bboxes.append(bboxes)
            batch_strokes.append(strokes)
            if has_class:
                conds.append(item["class"])
        if not batch_seqs:
            # 若全部被跳过，返回空 batch
            return {
                "seqs": torch.empty(0),
                "lengths": torch.empty(0, dtype=torch.long),
                "cond": torch.empty(0, dtype=torch.long),
                "bboxes": torch.empty(0),
                "strokes": torch.empty(0),
            }
        # determine pad length
        L = max_len if max_len is not None else max(batch_lens)
        B = len(batch_seqs)
        feat_dim = batch_seqs[0].shape[1]
        # allocate tensor: [B, L, feat_dim]
        out = batch_seqs[0].new_zeros(B, L, feat_dim)
        out_bboxes = batch_bboxes[0].new_zeros(B, L, 4)
        out_strokes = batch_strokes[0].new_zeros(B, L, batch_strokes[0].shape[1], batch_strokes[0].shape[2])
        for i, (seq, bboxes, strokes) in enumerate(zip(batch_seqs, batch_bboxes, batch_strokes)):
            l = seq.shape[0]
            out[i, :l] = seq
            out_bboxes[i, :l] = bboxes
            out_strokes[i, :l] = strokes
            if l < L:
                # pad flags to -1 for invalid positions
                out[i, l:, 0] = -1
        if has_class:
            cond = torch.tensor(conds, dtype=torch.long, device=out.device)
        else:
            cond = torch.zeros(B, dtype=torch.long, device=out.device)
        return {
            "seqs": out,
            "lengths": torch.tensor(batch_lens, dtype=torch.long, device=out.device),
            "cond": cond,
            "bboxes": out_bboxes,
            "strokes": out_strokes,
        }
