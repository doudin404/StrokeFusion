#!/usr/bin/env python3
"""
脚本用途：加载Sketch-RNN生成的npz数据文件，将其中的草图序列渲染为图片并保存。
用法示例：
    python render_sketches.py \
      --input outputs/sketches.npz \
      --output_dir samples \
      [--dpi 300] [--flip_vertical]
如果不提供参数，将使用下列默认值：
    input: outputs/sketches.npz
    output_dir: samples
    dpi: 300
    flip_vertical: False
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker 算法：对二维点集简化
    points: (N,2) 数组
    epsilon: 最大允许误差（同坐标尺度）
    返回简化后的点集
    """
    if points.shape[0] < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        proj = ((points - start) @ seg) / (seg_len**2)
        proj_point = np.outer(proj, seg) + start
        dists = np.linalg.norm(points - proj_point, axis=1)
    idx = np.argmax(dists)
    dmax = dists[idx]
    if dmax > epsilon:
        left = rdp(points[:idx+1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((start, end))


def render_sequence(seq,
                    png_dir: Path = None,
                    svg_dir: Path = None,
                    idx: int = None,
                    flip_vertical=False,
                    color=False,
                    dpi=300,
                    rdp_simplify=True):
    """
    渲染单条草图序列并保存为图像（PNG 和/或 SVG）。
    seq: numpy array of shape (T, 3) —— [delta_x, delta_y, pen_flag]
    png_dir: PNG保存目录 (Path)，为None则不保存PNG
    svg_dir: SVG保存目录 (Path)，为None则不保存SVG
    idx: 文件编号，用于命名
    flip_vertical: 是否垂直翻转 Y 轴
    color: 是否用彩色绘制；False 则统一黑色
    dpi: 图像分辨率（对 PNG 生效，SVG 可忽略但保留）
    rdp_simplify: 是否启用 RDP 简化
    """
    deltas = seq[:, :2].astype(float)
    pen = seq[:, 2].astype(int)
    coords = np.cumsum(deltas, axis=0)
    coords[:, 1] *= -1  # Y 方向翻转以符合常见画图习惯

    strokes = []
    start = 0
    for i, p in enumerate(pen):
        if p == 1:
            seg = coords[start:i+1]
            if len(seg) > 1:
                strokes.append(seg)
            start = i + 1
    if start < len(coords):
        seg = coords[start:]
        if len(seg) > 1:
            strokes.append(seg)

    fig, ax = plt.subplots()
    for seg in strokes:
        pts = seg.copy()
        if rdp_simplify:
            pts_scaled = pts * 256.0
            pts_simpl = rdp(pts_scaled, epsilon=2.0)
            pts = pts_simpl / 256.0
        x = pts[:, 0]
        y = pts[:, 1]
        if flip_vertical:
            y = -y
        ax.plot(x, y, color=None if color else 'black')
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout()

    if png_dir is not None and idx is not None:
        png_path = png_dir / f'sketch_{idx:05d}.png'
        fig.savefig(str(png_path), dpi=dpi)
    if svg_dir is not None and idx is not None:
        svg_path = svg_dir / f'sketch_{idx:05d}.svg'
        fig.savefig(str(svg_path), format='svg')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='渲染Sketch-RNN序列为图片')
    parser.add_argument('--input', '-i', default='data/download/facex_chirodiff.npz',)
    parser.add_argument('--output_dir', '-o', default='sample/chirodiff/facex',)
    parser.add_argument('--dpi', type=int, default=300,)
    parser.add_argument('--flip_vertical', action='store_true', default=False,)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if 'sequences' not in data:
        raise KeyError(f"找不到键 'sequences' 于文件 {args.input}")
    sequences = data['sequences']

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / 'png'
    svg_dir = out_dir / 'svg'
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    for idx, seq in enumerate(sequences):
        #if idx!=185:continue
        save_path = out_dir / f'sketch_{idx:05d}.png'
        render_sequence(
            seq,
            #png_dir=png_dir,
            svg_dir=svg_dir,
            idx=idx,
            flip_vertical=args.flip_vertical,
            color=False,
            dpi=args.dpi,
            rdp_simplify=True
        )
        if idx % 1000 == 0:
            print(f"已渲染 {idx}/{len(sequences)} 条草图")
    print(f"渲染完成，共保存 {len(sequences)} 张草图，目录：{out_dir}")

if __name__ == '__main__':
    main()
