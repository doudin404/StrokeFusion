#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算生成数据和真实数据之间的 FID、Precision、Recall
优化：使用 DataLoader 并行加载图片，加速特征提取
用法:
    python evaluate_metrics.py <真实图片文件夹> <生成图片文件夹> [--batch_size B] [--k K] [--num_workers W]
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    """自定义 Dataset：并行加载指定文件夹下的 PNG 图像"""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Failed to read image: {path}")
            raise e


def get_activations(files, model, device, batch_size=50, num_workers=4):
    """使用 DataLoader 并行加载与预处理，提取 Inception 特征"""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: 
            transforms.functional.pad(
                img,
                padding=(
                    (max(img.size) - img.size[0]) // 2,
                    (max(img.size) - img.size[1]) // 2,
                    (max(img.size) - img.size[0] + 1) // 2,
                    (max(img.size) - img.size[1] + 1) // 2,
                ),
                fill=(255, 255, 255),
                padding_mode='constant'
            )
        ),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolderDataset(files, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True,
    )
    activations = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="提取特征", ncols=80, leave=False):
            imgs = batch.to(device)
            pred = model(imgs)
            activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations


def calculate_fid(act1, act2):
    """计算 Fréchet Inception Distance"""
    mu1, mu2 = np.mean(act1, axis=0), np.mean(act2, axis=0)
    sigma1, sigma2 = np.cov(act1, rowvar=False), np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_pr(act_real, act_gen, k=3):
    """计算 Precision 和 Recall"""
    # 真实流形半径
    nbrs_r = NearestNeighbors(n_neighbors=k+1).fit(act_real)
    radii_r = nbrs_r.kneighbors(act_real)[0][:, k]

    # Precision
    dist_rg, idx_rg = NearestNeighbors(n_neighbors=1).fit(act_real).kneighbors(act_gen)
    precision = np.mean(dist_rg[:, 0] <= radii_r[idx_rg[:, 0]])

    # 生成流形半径
    nbrs_g = NearestNeighbors(n_neighbors=k+1).fit(act_gen)
    radii_g = nbrs_g.kneighbors(act_gen)[0][:, k]

    # Recall
    dist_gr, idx_gr = NearestNeighbors(n_neighbors=1).fit(act_gen).kneighbors(act_real)
    recall = np.mean(dist_gr[:, 0] <= radii_g[idx_gr[:, 0]])

    return precision, recall


def main(class_name):
    path_real = f'/root/repo/stroke_fusion/sample/dataset_{class_name}/'  # 替换为真实图片文件夹路径
    path_gen = f'/root/repo/stroke_fusion/sample/rnn/{class_name}_flip/'   # 替换为生成图片文件sample_RDP_  rnn/{class_name}/' chirodiff/{class_name}/
    parser = argparse.ArgumentParser(description='计算 FID, Precision, Recall')
    parser.add_argument('--path_real', type=str, help='真实图片文件夹路径',default=path_real)
    parser.add_argument('--path_gen', type=str, help='生成图片文件夹路径', default=path_gen)
    parser.add_argument('--batch_size', type=int, default=512, help='批量大小')
    parser.add_argument('--k', type=int, default=20, help='kNN 中的 k 值')
    args = parser.parse_args()

    files_real = [os.path.join(args.path_real, f) for f in os.listdir(args.path_real) if f.lower().endswith('.png')]
    files_gen = [os.path.join(args.path_gen, f) for f in os.listdir(args.path_gen) if f.lower().endswith('.png')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载 Inception v3 模型
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.to(device)

    #print('正在提取真实数据特征...')
    act_real = get_activations(files_real, model, device, args.batch_size, 20)
    #print('正在提取生成数据特征...')
    act_gen = get_activations(files_gen, model, device, args.batch_size, 20)

    #print('正在计算 FID...')
    fid_value = calculate_fid(act_real, act_gen)
    print(f'FID: {fid_value:.4f}')

    #print('正在计算 Precision/Recall...')
    precision, recall = calculate_pr(act_real, act_gen, args.k)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')


if __name__ == '__main__':
    # airplane apple television spider shoe bus cat chair face fish moon pizza train umbrella['apple', 'shoe','chair', 'moon', 'train', 'umbrella']
    ls=['facex', 'tu_berlin']
    for i in ls:
        print(f"Processing {i}...")
        main(i)
    # main("cat")
# 