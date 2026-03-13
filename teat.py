import torch
from pathlib import Path
import numpy as np

# 直接复用你定义的 SvgDataset
from data_process.sketch_data import SvgDataset,NpzDataset  # 请替换成实际保存上面代码的文件名

def count_long_strokes(root_dir="data/raw", data_name="facex", max_seq_len=32, path_points=64):
    ds = SvgDataset(root_dir=Path(root_dir)/data_name, path_points=path_points)

    total = len(ds)
    over_count = 0

    for i in range(total):
        seq = ds._get_seq(i)  # shape [L,3]
        raw_strokes = ds.parse_seq_to_strokes(seq)  # [num_strokes, path_points, 2]
        num_strokes = raw_strokes.shape[0]
        if num_strokes > max_seq_len:
            over_count += 1
    
    ratio = over_count / total if total > 0 else 0
    print(f"总样本数: {total}")
    print(f"超过 {max_seq_len} 笔画的样本数: {over_count}")
    print(f"占比: {ratio:.4f}")

def count_long_strokes_npz(root_dir="data/raw", data_name="creative\creative_creatures.npz",
                           split="train", max_seq_len=32, path_points=64):
    ds = NpzDataset(root_dir=Path(root_dir)/data_name,
                    split=split,
                    path_points=path_points)

    total = len(ds)
    over_count = 0

    for i in range(total):
        seq = ds._get_seq(i)   # shape [L,3]
        strokes = ds.parse_seq_to_strokes(seq)  # [num_strokes, path_points, 2]
        num_strokes = strokes.shape[0]
        if num_strokes > max_seq_len:
            over_count += 1
    
    ratio = over_count / total if total > 0 else 0
    print(f"数据 split: {split}")
    print(f"总样本数: {total}")
    print(f"超过 {max_seq_len} 笔画的样本数: {over_count}")
    print(f"占比: {ratio:.4f}")

if __name__ == "__main__":
    count_long_strokes()