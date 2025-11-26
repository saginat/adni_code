import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


from ..configs import config_pretrain as config
from ..models.ag_vit import TransformerAutoEncoder
from ..training.losses import calculate_balanced_weights, get_loss_fns
from ..training.trainer import MultiTaskTrainer
from ..utils.metrics import MetricsTracker
from ..utils.plotting import MetricsPlotter
from ..utils.tracking import RunTracker, save_training_results
from ..data.prepare_data import prepare_dataloaders, compute_num_patches_3d
from ..utils.general import set_seed, lr_lambda_scheduler


def main():
    # --- Set Seed ---
    set_seed(config.SEED, config.DEVICE)
    # --- Load Data ---

    data_components = prepare_dataloaders(config, stage="pretrain")

    train_dataloader = data_components["train_dataloader"]
    val_dataloader = data_components["val_dataloader"]
    test_dataloader = data_components["test_dataloader"]
    imageID_to_labels = data_components["imageID_to_labels"]
    dataset_tr = data_components["dataset_tr"]

    # --- Initialize Model ---
    print("Initializing model...")
    input_shape = dataset_tr[0][0].shape

    num_spatial_patches = compute_num_patches_3d(input_shape[:-1], config.PATCH_SIZE)

    model = TransformerAutoEncoder(
        input_size=input_shape,
        patch_size=config.PATCH_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        output_dims={
        },
        num_of_spatial_patches=num_spatial_patches,
        num_cls_tokens=config.NUM_CLS_TOKENS,
        reduced_patches_factor_percent=config.REDUCED_PATCHES_FACTOR_PERCENT,
        reduce_time_factor_percent=config.REDUCE_TIME_FACTOR_PERCENT,
        p_dropout=config.P_DROPOUT,
        custom_decoder=config.CUSTOM_RECON_BOOL,
        merge_patches=config.MERGE_PATCHES,
        use_patch_merger=config.USE_PATCH_MERGER,
    ).to(config.DEVICE)

    # --- Setup Training Components ---
    print("Setting up training components...")
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.OPTIMIZER_WEIGHT_DECAY
    )

    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    warmup_steps = int(config.WARMUP_STEPS_PERCENT * total_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda_scheduler(step, warmup_steps, total_steps),
    )

    class_weights = calculate_balanced_weights(
        imageID_to_labels, config.TASKS_TYPES, config.DEVICE
    )
    loss_fns = get_loss_fns(config, class_weights)

    metrics_tracker_tr = MetricsTracker(config.TASKS_TYPES)
    metrics_tracker_val = MetricsTracker(config.TASKS_TYPES)

    chosen_labels = config.TASKS + ["Reconstruction"]
    hyperparams = {
        key: getattr(config, key) for key in dir(config) if not key.startswith("__")
    }
    hyperparams['num_epochs'] = config.NUM_EPOCHS
    hyperparams['max_norm'] = config.MAX_NORM
    hyperparams["chosen_labels"] = chosen_labels
    hyperparams["len_train_dataset"] = len(dataset_tr)

    tracker = RunTracker(base_dir=config.PRETRAINED_MODEL_DIR)
    run_id = tracker.create_run(description="", hparams={})

    # --- Initialize Trainer and Start Training ---
    print(f"Starting training for run ID: {run_id}")
    trainer = MultiTaskTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        tracker=tracker,
        hyperparams=hyperparams,
        metrics_tracker_tr=metrics_tracker_tr,
        metrics_tracker_test=metrics_tracker_val,
        loss_fns=loss_fns,
        scheduler=scheduler,
        tracker_description="ICLR submission run",
        run_id=run_id,
        baselines=None,


    )

    trainer.train()

    # --- Save Final Results ---
    print("Training finished. Saving results...")
    plotter = MetricsPlotter(
        train_metrics=metrics_tracker_tr.get_metrics(),
        test_metrics=metrics_tracker_val.get_metrics(),
        baseline_metrics_dict={},
    )
    save_training_results(
        trainer=trainer,
        tracker=tracker,
        metrics_tracker_tr=metrics_tracker_tr,
        metrics_tracker_test=metrics_tracker_val,
        model=model,
        plotter=plotter,
    )
    print("Done.")


if __name__ == "__main__":
    main()
