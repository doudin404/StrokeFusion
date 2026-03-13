import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# 注意根据你工程的目录结构修改导入路径
from data_process.sketch_data import StrokesDataModule,SketchsDataModule
from models.stroke_fusion import DualModalModel
from models.sketch_diffusion import SketchDiffusion

if __name__ == '__main__':
    # 提高 float32 矩阵乘精度（可选）
    torch.set_float32_matmul_precision('high')

    # TensorBoard 日志
    tb_logger = TensorBoardLogger(
        save_dir="logs/sketch_log",
        name="dual_modal"
    )
    # 加载已训练的 DualModalModel 作为 encoder_model
    encoder_ckpt = "checkpoints/best_tu_berlin.ckpt"##################
    encoder_model = DualModalModel.load_from_checkpoint(encoder_ckpt).eval()

    # 数据模块
    data_dir = "data/raw"               # 替换为你的数据根目录
    save_dir = "data/processed"
    dm = SketchsDataModule(
        root_dir=data_dir,
        cache_dir=save_dir,
        data_name="tu_berlin",               # 子文件夹名##############################
        data_type="svg",               # 数据类型：'svg', 'npz', 'json'
        path_points=64,
        gamma=200.0,
        display_scale=0.8,
        #subset=None,
        batch_size=128,#8192,  # 每个 GPU 的 batch size
        num_workers=20,#20,#############################
        encoder_model=encoder_model,
        max_seq_len=64,  # 设置最大序列长度
    )

    # 模型：使用默认超参数（已在 __init__ 中通过 save_hyperparameters 保存）
    # 定义两个不同的参数设置
    model_configs = [
        {
            "emb_size": 512,
            "n_layers": 16,
            "n_heads": 16,
            "warmup_epochs": 10,
            "num_classes": 250,  # 新增 num_classes 参数###########################################
        },
        {
            "emb_size": 1024,
            "n_layers": 16,
            "n_heads": 16,
            "warmup_epochs": 100,
            "num_classes": 250,  # 新增 num_classes 参数###########################################
        },
    ]

    devices= [[0],[0]]  # 根据实际 GPU 数量修改

    # 根据索引选择参数设置
    config_idx = 1
    model = SketchDiffusion(**model_configs[config_idx])

    # Checkpoint 回调：保存最优和最后一次模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best_tu_berlin_diff_{config_idx}",######################
        save_top_k=1,
        monitor="val_loss",     # 或者你训练时记录的其他指标
        mode="min",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_tu_berlin_diff_{config_idx}"########################

    # Trainer 配置
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices[config_idx],  # 根据 config_idx 选择设备
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # 如果有已存的 checkpoint，就从上次训练继续
    last_ckpt = "checkpoints/best_tu_berlin_diff_1-v1.ckpt"##############################
    if last_ckpt and os.path.exists(last_ckpt):
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)

    # 输出模型保存路径
    print("最佳模型路径：", checkpoint_callback.best_model_path)
    print("最后一次模型路径：", checkpoint_callback.last_model_path)