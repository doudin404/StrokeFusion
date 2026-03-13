import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os

# Note: Adjust these import paths according to your project structure
from data_process.sketch_data import StrokesDataModule
from models.stroke_fusion import DualModalModel

if __name__ == '__main__':
    # Increase float32 matrix multiplication precision (optional)
    torch.set_float32_matmul_precision('high')

    # TensorBoard Logger setup
    tb_logger = TensorBoardLogger(
        save_dir="logs/sketch_log",
        name="dual_modal"
    )

    # Data Module initialization
    data_dir = "data/raw"            # Root directory of your dataset
    save_dir = "data/processed"      # Directory for processed data cache
    dm = StrokesDataModule(
        root_dir=data_dir,
        cache_dir=save_dir,
        data_name="tu_berlin",       # Name of the sub-folder/dataset
        data_type="svg",             # Supported formats: 'svg', 'npz', 'json'
        path_points=64,
        gamma=200.0,
        display_scale=0.8,
        # subset=None,
        batch_size=18,               # Batch size per GPU
        num_workers=20,
    )

    # Model definition: uses default hyperparameters 
    # (Saved via save_hyperparameters in __init__)
    model = DualModalModel()

    # Checkpoint Callback: Save the best model and the latest training state
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_creative_creatures",
        save_top_k=1,
        monitor="val_total",         # The metric to monitor for the 'best' model
        mode="min",
        save_last=True,
    )
    
    # Customize the filename for the latest checkpoint
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_creative_creatures"
    print("Checkpoint will be saved as:", checkpoint_callback.CHECKPOINT_NAME_LAST)

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=-1,               # Train indefinitely
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[1],                 # Specify GPU index
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # Automatically resume training if a checkpoint exists
    last_ckpt = "checkpoints/last_creative_creatures-v1.ckpt"
    if last_ckpt and os.path.exists(last_ckpt):
        print(f"Resuming training from checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("No existing checkpoint found. Starting fresh training session.")
        trainer.fit(model, datamodule=dm)

    # Output saved model paths upon completion or interruption
    print("Best model path:", checkpoint_callback.best_model_path)
    print("Latest model path:", checkpoint_callback.last_model_path)