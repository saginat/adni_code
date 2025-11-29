"""
Configuration for SwiFT pretraining
"""

# Model architecture parameters
# Using SwiFT's original hyperparameters from sample_script.sh
# Reference: https://github.com/Transconnectome/SwiFT
# Adapted for custom img_size (36, 36, 36, 20) - ADNI data
# patch_size (6,6,6,1) -> grid (6, 6, 6, 20)
# Using 2 layers to support 36x36x36 spatial size (6->3 with 1 downsample)
MODEL_CONFIG = {
    "img_size": (36, 36, 36, 20),  # ADNI data: 36x36x36 spatial, 20 time frames
    "in_chans": 1,
    "embed_dim": 36,  # SwiFT default: 36
    "window_size": (3, 3, 3, 4),  # Adjusted for smaller spatial grid
    "first_window_size": (3, 3, 3, 4),  # Adjusted for smaller spatial grid
    "patch_size": (6, 6, 6, 1),  # 36/6=6 -> grid (6,6,6,20)
    "depths": [2, 6],  # 2 layers (1 downsample: 6->3)
    "num_heads": [3, 6],  # 2 layers
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,  # SwiFT default: 0
    "drop_path_rate": 0.1,
    "patch_norm": False,
    "use_checkpoint": False,
    "spatial_dims": 4,
    "c_multiplier": 2,  # SwiFT default: 2
    "last_layer_full_MSA": True,  # SwiFT default: True
    "downsample": "mergingv2",
}

# Calculate output feature dimension
# embed_dim * (c_multiplier ** (num_layers - 1))
# 36 * (2 ** 1) = 36 * 2 = 72 (with 2 layers)
OUTPUT_DIM = 72

# Contrastive head config
CONTRASTIVE_CONFIG = {
    "num_features": OUTPUT_DIM,
    "embedding_dim": 128,
}

# Training parameters
# SwiFT uses batch_size=8, learning_rate=5e-5
TRAIN_CONFIG = {
    "batch_size": 8,  # SwiFT default: 8
    "learning_rate": 5e-5,  # SwiFT default: 5e-5
    "weight_decay": 0.01,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "temperature": 0.1,  # Common default for NTXentLoss
    "use_cosine_similarity": True,
}

# Data preprocessing parameters
# Custom for ADNI data: 36x36x36 spatial with 20 time points
DATA_CONFIG = {
    "target_spatial_size": (36, 36, 36),  # Custom: ADNI data resized to 36x36x36
    "sequence_length": 20,  # Custom: 20 time points
    "window_size": 20,  # Temporal window size
    "stride": 5,  # Temporal stride
    "normalize": True,
    "to_float16": True,
}

# Optimizer parameters
OPTIMIZER_CONFIG = {
    "name": "AdamW",
    "lr": TRAIN_CONFIG["learning_rate"],
    "weight_decay": TRAIN_CONFIG["weight_decay"],
    "betas": (0.9, 0.999),
}

# Scheduler parameters
SCHEDULER_CONFIG = {
    "name": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 1e-6,
}

# Paths
PATHS = {
    "data_dir": "./swift_pipeline/data/preprocessed",
    "checkpoint_dir": "./swift_pipeline/checkpoints",
    "log_dir": "./swift_pipeline/logs",
}

# Device configuration
DEVICE = "cuda"  # or 'cpu'
MIXED_PRECISION = True  # Use automatic mixed precision training
