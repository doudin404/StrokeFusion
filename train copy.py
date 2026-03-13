import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# 注意根据你工程的目录结构修改导入路径
from data_process.sketch_data import StrokesDataModule
from models.stroke_fusion import DualModalModel

if __name__ == '__main__':
    # 提高 float32 矩阵乘精度（可选）
    torch.set_float32_matmul_precision('high')

    # TensorBoard 日志
    tb_logger = TensorBoardLogger(
        save_dir="logs/sketch_log",
        name="dual_modal"
    )

    # 数据模块
    data_dir = "data/raw"               # 替换为你的数据根目录
    save_dir = "data/processed"
    dm = StrokesDataModule(
        root_dir=data_dir,
        cache_dir=save_dir,
        data_name="quick_draw/moon",               # 子文件夹名#############################
        data_type="npz",               # 数据类型：'svg', 'npz', 'json'#################################
        path_points=64,
        gamma=200.0,
        display_scale=0.8,
        #subset=None,
        batch_size=32,  # 每个 GPU 的 batch size##########################
        num_workers=20,#20,
    )

    # 模型：使用默认超参数（已在 __init__ 中通过 save_hyperparameters 保存）
    model = DualModalModel()

    # Checkpoint 回调：保存最优和最后一次模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_moon",
        save_top_k=1,
        monitor="val_total",     # 或者你训练时记录的其他指标
        mode="min",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_moon"
    print("Checkpoint will be saved as:", checkpoint_callback.CHECKPOINT_NAME_LAST)

    # mooner 配置
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0,1,2,3],
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # 如果有已存的 checkpoint，就从上次训练继续
    last_ckpt = "checkpoints/ready/best_moon.ckpt"
    if last_ckpt and os.path.exists(last_ckpt):
        print(f"加载已有 checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("没有找到已有的 checkpoint，开始新的训练")
        trainer.fit(model, datamodule=dm)

    # 输出模型保存路径
    print("最佳模型路径：", checkpoint_callback.best_model_path)
    print("最后一次模型路径：", checkpoint_callback.last_model_path)