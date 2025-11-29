"""
Configuration for SwiFT fine-tuning on downstream tasks
"""

# Import pretrained model config
from .config_pretrain import (
    MODEL_CONFIG,
    OUTPUT_DIM,
    DATA_CONFIG,
    DEVICE,
    MIXED_PRECISION,
)

# Task-specific parameters
TASK_CONFIG = {
    "task_type": "binary_classification",  # 'binary_classification', 'multiclass_classification', or 'regression'
    "num_classes": 2,  # For classification tasks
    "freeze_encoder": True,  # Freeze pretrained encoder weights
    "freeze_layers": None,  # None = freeze all, or list of layer indices to freeze
}

# Fine-tuning head config
HEAD_CONFIG = {
    "num_features": OUTPUT_DIM,  # 288
    "num_classes": TASK_CONFIG["num_classes"],
}

# Training parameters (often use smaller lr and fewer epochs for fine-tuning)
# SwiFT uses batch_size=8, learning_rate=5e-5 for training
TRAIN_CONFIG = {
    "batch_size": 8,  # SwiFT default: 8
    "learning_rate": 5e-5,  # SwiFT default: 5e-5
    "weight_decay": 0.01,
    "num_epochs": 50,
    "warmup_epochs": 5,
}

# Loss function
if TASK_CONFIG["task_type"] == "binary_classification":
    LOSS_FUNCTION = "BCEWithLogitsLoss"
elif TASK_CONFIG["task_type"] == "multiclass_classification":
    LOSS_FUNCTION = "CrossEntropyLoss"
elif TASK_CONFIG["task_type"] == "regression":
    LOSS_FUNCTION = "MSELoss"
else:
    raise ValueError(f"Unknown task type: {TASK_CONFIG['task_type']}")

# Optimizer parameters
OPTIMIZER_CONFIG = {
    "name": "AdamW",
    "lr": TRAIN_CONFIG["learning_rate"],
    "weight_decay": TRAIN_CONFIG["weight_decay"],
    "betas": (0.9, 0.999),
}

# Scheduler parameters
SCHEDULER_CONFIG = {
    "name": "CosineAnnealingLR",
    "T_max": TRAIN_CONFIG["num_epochs"],
    "eta_min": 1e-7,
}

# Paths
PATHS = {
    "pretrained_encoder": "./swift_pipeline/checkpoints/pretrained_encoder.pth",
    "data_dir": "./swift_pipeline/data/preprocessed",
    "checkpoint_dir": "./swift_pipeline/checkpoints",
    "log_dir": "./swift_pipeline/logs",
}

# Evaluation metrics
METRICS = {
    "binary_classification": ["accuracy", "precision", "recall", "f1", "auc"],
    "multiclass_classification": [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ],
    "regression": ["mse", "mae", "r2"],
}
