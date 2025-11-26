"""
Configuration for SwiFT pretraining
"""

# Model architecture parameters
MODEL_CONFIG = {
    "img_size": (96, 96, 96, 20),
    "in_chans": 1,
    "embed_dim": 36,
    "window_size": (4, 4, 4, 4),
    "first_window_size": (4, 4, 4, 4),
    "patch_size": (6, 6, 6, 1),
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "patch_norm": False,
    "use_checkpoint": False,
    "spatial_dims": 4,
    "c_multiplier": 2,
    "last_layer_full_MSA": False,
    "downsample": "mergingv2",
}

# Calculate output feature dimension
# embed_dim * (c_multiplier ** (num_layers - 1))
# 36 * (2 ** 3) = 36 * 8 = 288
OUTPUT_DIM = 288

# Contrastive head config
CONTRASTIVE_CONFIG = {
    "num_features": OUTPUT_DIM,
    "embedding_dim": 128,
}

# Training parameters
TRAIN_CONFIG = {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "temperature": 0.5,
    "use_cosine_similarity": True,
}

# Data preprocessing parameters
DATA_CONFIG = {
    "original_size": (91, 109, 91, 140),
    "target_spatial_size": (96, 96, 96),
    "window_size": 20,
    "stride": 10,
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
