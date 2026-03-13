import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Ensure import paths match your project directory structure
from data_process.sketch_data import StrokesDataModule, SketchsDataModule
from models.stroke_fusion import DualModalModel
from models.sketch_diffusion import SketchDiffusion

if __name__ == '__main__':
    # Set matrix multiplication precision (Optional: 'high' or 'medium')
    torch.set_float32_matmul_precision('high')

    # TensorBoard Logger configuration
    tb_logger = TensorBoardLogger(
        save_dir="logs/sketch_log",
        name="dual_modal"
    )

    # Load a pre-trained DualModalModel to serve as the encoder
    encoder_ckpt = "checkpoints/ready/best_tu_berlin.ckpt"
    encoder_model = DualModalModel.load_from_checkpoint(encoder_ckpt).eval()

    # Data Module initialization
    data_dir = "data/raw"            # Path to your raw dataset root
    save_dir = "data/processed"      # Path for processed data cache
    dm = SketchsDataModule(
        root_dir=data_dir,
        cache_dir=save_dir,
        data_name="tu_berlin",       # Dataset sub-folder name
        data_type="svg",             # Data format: 'svg', 'npz', or 'json'
        path_points=64,
        gamma=200.0,
        display_scale=0.8,
        # subset=None,
        batch_size=256,              # Batch size per GPU
        num_workers=20,
        encoder_model=encoder_model, # Pass the pre-trained encoder
        max_seq_len=32,              # Maximum sequence length constraint
    )

    # Model configuration: hyperparameters are stored via save_hyperparameters in __init__
    # Define different parameter settings for experiments or ablation
    model_configs = [
        {
            "emb_size": 512,
            "n_layers": 16,
            "n_heads": 16,
            "warmup_epochs": 10,
            "num_classes": 250,      # Number of target classes
        },
    ]
    
    # GPU device allocation
    devices = [[0, 1]]  # Adjust based on your available hardware

    # Select configuration by index
    config_idx = 0
    model = SketchDiffusion(**model_configs[config_idx])

    # Checkpoint Callback: Save the best model and the latest state
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best_tu_berlin_diff_{config_idx}_32",
        save_top_k=1,
        monitor="val_loss",          # Metric used to determine the 'best' model
        mode="min",
        save_last=True,
    )
    
    # Customize the filename for the latest checkpoint
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_tu_berlin_diff_{config_idx}_32"

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=-1,               # Unlimited training epochs
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices[config_idx], # Select devices based on config index
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # Automatically resume training if a valid checkpoint exists
    last_ckpt = "checkpoints/ready/last_tu_berlin_diff_0.ckpt"
    if last_ckpt and os.path.exists(last_ckpt):
        print(f"Resuming training from checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("No checkpoint found. Starting a new training session.")
        trainer.fit(model, datamodule=dm)

    # Output the paths for the saved models
    print("Best model saved at:", checkpoint_callback.best_model_path)
    print("Latest model saved at:", checkpoint_callback.last_model_path)