import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_process.sketch_data import SketchsDataModule
from models.stroke_fusion import DualModalModel
from models.sketch_diffusion import SketchDiffusion
from models.sketch_diffusion import SketchDiffusionSampler

if __name__ == '__main__':
    # Define model configurations (Ablation study or different versions)
    model_configs = [ 
        {
            "emb_size": 512,         # Final architecture parameter
            "n_layers": 16,
            "n_heads": 16,
            "warmup_epochs": 10,
            "num_classes": 1,        # Placeholder, will be updated based on dataset
        },
    ]
    config_idx = 0

    # Configure sampling parameters
    sample_configs = {
        # Options: "tu_berlin", "facex", "creative_birds", "creative_creatures", etc.
        "data_name": "facex",  
        "prefix": "",                # Prefix path for saving results (e.g., "quick_draw/")
        "mode": "random",            # Selection: "dataset", "recon" (reconstruction), or "random"
        "device": "cuda:0",          # Target computation device
        "num_samples": 64 * 4,       # Total number of samples to generate
        "max_len": 32,               # Maximum sequence length
        "flip_vertical": False,      # Vertical flip toggle
        "color": False,              # Color rendering toggle
        "cond": None,                # Value for conditional sampling (None for unconditional)
    }
    print(f"Sampling configuration: {sample_configs}")

    # Set data type and class count based on the selected dataset
    if sample_configs["data_name"] in ["tu_berlin", "facex"]:
        data_type = "svg"
        flip_vertical = True
        model_configs[config_idx]["num_classes"] = 250  # Update class count for TU-Berlin
    else:
        data_type = "npz"
        flip_vertical = False
        model_configs[config_idx]["num_classes"] = 1

    # Define output directory based on sampling mode
    if sample_configs["mode"] == "dataset":
        samples_dir = f"sample/dataset_{sample_configs['data_name']}"
    elif sample_configs["mode"] == "recon":
        samples_dir = f"sample/recon_{sample_configs['data_name']}"
    else:
        # Default save path for random or conditional sampling
        samples_dir = f"sample/sample_{sample_configs['cond']}_{sample_configs['data_name']}"

    # Load pre-trained DualModalModel as the encoder (used for latent decoding)
    encoder_ckpt = f'checkpoints/ready/best_{sample_configs["data_name"]}.ckpt'
    encoder_model = DualModalModel.load_from_checkpoint(encoder_ckpt).eval()

    # Load pre-trained SketchDiffusion LightningModule
    diff_ckpt = f'checkpoints/ready/last_{sample_configs["data_name"]}_diff_0.ckpt'
    sketch_diffusion_module = SketchDiffusion.load_from_checkpoint(
        diff_ckpt, 
        **model_configs[config_idx]
    ).eval()

    # Initialize DataModule to retrieve condition IDs (cond_ids)
    dm = SketchsDataModule(
        root_dir='data/raw',
        cache_dir='data/processed',
        data_name=sample_configs["prefix"] + sample_configs["data_name"],
        data_type=data_type,
        path_points=32,
        gamma=200.0,
        display_scale=0.8,
        batch_size=64 * 4,
        num_workers=0,
        encoder_model=encoder_model,
        max_seq_len=32,
    )
    dm.setup()
    train_loader = dm.train_dataloader()

    # Instantiate the Sampler
    sampler = SketchDiffusionSampler(
        diffusion_module=sketch_diffusion_module,
        dual_model=encoder_model,
        samples_dir=samples_dir,
        device=sample_configs["device"],
        # viz_intermediate=True # Uncomment to visualize the diffusion steps
    )

    # Execute sampling process
    sampler.sample(
        dataloader=train_loader,
        num_samples=sample_configs["num_samples"],
        mode=sample_configs["mode"],
        max_len=sample_configs["max_len"],
        flip_vertical=flip_vertical,
        color=sample_configs["color"],
        cond_rw=sample_configs["cond"],
    )

    # Optional: Export latent encodings directly to SVG
    # sampler.export_encodings_to_svg(samples_dir, flip_vertical=flip_vertical)

    print('Sampling complete.')