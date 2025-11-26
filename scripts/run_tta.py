import torch
from ..configs import config_tta as config
from ..evaluation.tta import TestTimeOptimizer
from ..models.heads import PredictionHead
from ..models.ag_vit import TransformerAutoEncoder
from ..training.losses import get_loss_fns
from ..utils.metrics import MetricsTracker
from ..data.prepare_data import prepare_dataloaders
from ..utils.general import set_seed


def main():
    # --- Setup ---
    set_seed(config.SEED, config.DEVICE)
    device = torch.device(config.DEVICE)

    # --- Load Data ---
    print("Loading test data for TTA...")
    data_components = prepare_dataloaders(config, stage="tta")

    test_dataloader = data_components["test_dataloader"]

    # --- Load Pretrained and Fine-tuned Models ---
    print(
        f"Loading pretrained encoder/decoder from: {config.PRETRAINED_CHECKPOINT_PATH}"
    )
    pretrained_model = torch.load(
        config.PRETRAINED_CHECKPOINT_PATH, map_location=device
    )

    print(f"Loading fine-tuned head from: {config.FINETUNED_HEAD_PATH}")
    prediction_head = torch.load(config.FINETUNED_HEAD_PATH, map_location=device)

    # --- Setup TTA Components ---
    loss_fns = get_loss_fns(config, class_weights=None)
    metrics_tracker_test = MetricsTracker(config.TASKS_TYPES)

    # --- Initialize and Run TestTimeOptimizer ---
    tta_optimizer = TestTimeOptimizer(
        self_supervised_model=pretrained_model,
        prediction_head=prediction_head,
        test_dataloader=test_dataloader,
        metrics_tracker_test=metrics_tracker_test,
        loss_fns=loss_fns,
        hyperparams=vars(config),
        tta_steps=config.TTA_STEPS,
        tta_lr=config.TTA_LR,
        device=device,
        verbose=config.VERBOSE,
    )

    print(
        f"Starting TTA evaluation for task: {config.TTA_TASK} with K={config.TTA_STEPS} steps..."
    )

    comparison_results = tta_optimizer.compare_with_baseline()

    print("\n--- TTA Evaluation Finished ---")

    # with open('tta_results.json', 'w') as f:
    #     json.dump(comparison_results, f, indent=4)


if __name__ == "__main__":
    main()
