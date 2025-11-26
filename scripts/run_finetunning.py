import os

import torch
import torch.optim as optim

from ..configs import config_finetune as config

from ..models.heads import PredictionHead
from ..models.ag_vit import TransformerAutoEncoder
from ..training.losses import get_loss_fns
from ..training.trainer import FineTuner
from ..utils.metrics import MetricsTracker
from ..utils.tracking import RunTracker
from ..data.prepare_data import prepare_dataloaders
from ..utils.general import set_seed


def main():
    # --- Setup ---
    set_seed(config.SEED, config.DEVICE)
    device = torch.device(config.DEVICE)

    # --- Load Data ---
    data_components = prepare_dataloaders(config, stage="finetune")

    train_dataloader = data_components["train_dataloader"]
    val_dataloader = data_components["val_dataloader"]

    # --- Load Pretrained Model ---
    print(f"Loading pretrained model from: {config.PRETRAINED_CHECKPOINT_PATH}")
    pretrained_model = torch.load(
        config.PRETRAINED_CHECKPOINT_PATH, map_location=device
    )
    pretrained_model.eval()

    # --- Initialize Prediction Head ---
    prediction_head = PredictionHead(
        embedding_dim=config.HEAD_INPUT_DIM,
        output_dims=config.OUTPUT_DIMS,
        p_dropout=config.HEAD_P_DROPOUT,
    ).to(device)

    # --- Setup Training Components ---
    optimizer = optim.AdamW(
        prediction_head.parameters(),
        lr=config.LR,
        weight_decay=config.OPTIMIZER_WEIGHT_DECAY,
    )
    loss_fns = get_loss_fns(config, class_weights=None)
    metrics_tracker_tr = MetricsTracker(config.TASKS_TYPES)
    metrics_tracker_val = MetricsTracker(config.TASKS_TYPES)
    tracker = RunTracker(base_dir=config.FINETUNE_MODEL_DIR)

    # --- Initialize and Run FineTuner ---
    finetuner = FineTuner(
        model=pretrained_model,
        prediction_head=prediction_head,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        tracker=tracker,
        hyperparams=vars(config),
        metrics_tracker_tr=metrics_tracker_tr,
        metrics_tracker_test=metrics_tracker_val,
        loss_fns=loss_fns,
        freeze_model=True,
    )

    print("Starting fine-tuning...")
    finetuner.train()
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
