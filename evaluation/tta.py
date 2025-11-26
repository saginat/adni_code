import torch
import torch.nn as nn
import torch.optim as optim
import einops
from tqdm import tqdm

from ..training.losses import (
    handle_null_and_dtypes,
    calculate_task_losses,
    choose_labels,
)


class TestTimeOptimizer:
    """
    Performs Test-Time Adaptation by optimizing the bottleneck representation
    of a test sample to improve atlas reconstruction before making a final prediction.
    """

    def __init__(
        self,
        self_supervised_model,
        prediction_head,
        test_dataloader,
        metrics_tracker_test,
        loss_fns,
        hyperparams,
        tta_steps=50,
        tta_lr=5e-4,
        device="cuda",
        verbose=1,
    ):
        self.self_supervised_model = self_supervised_model.to(device)
        self.prediction_head = prediction_head.to(device)
        self.test_dataloader = test_dataloader
        self.metrics_tracker_test = metrics_tracker_test
        self.loss_fns = loss_fns
        self.hyperparams = hyperparams
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.device = device
        self.verbose = verbose

        # Set models to evaluation mode and freeze parameters
        self.self_supervised_model.eval()
        self.prediction_head.eval()
        for param in self.self_supervised_model.parameters():
            param.requires_grad = False
        for param in self.prediction_head.parameters():
            param.requires_grad = False

        self.task_weights = hyperparams.get("TASK_WEIGHTS", {})
        self.chosen_labels = hyperparams.get("CHOSEN_LABELS", [])
        self.atlas_loss_fn = nn.MSELoss()

    def _optimize_bottleneck(self, initial_bottleneck, target_atlas):
        # Make a clone of the bottleneck that requires gradients
        optimized_bottleneck = initial_bottleneck.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([optimized_bottleneck], lr=self.tta_lr)

        # The decoder needs to be in a state that allows gradient flow
        decoder_was_training = self.self_supervised_model.custom_recon_decoder.training
        self.self_supervised_model.custom_recon_decoder.train()

        for step in range(self.tta_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                reconstructed_atlas = self.self_supervised_model.custom_recon_decoder(
                    optimized_bottleneck
                )
                atlas_loss = self.atlas_loss_fn(reconstructed_atlas, target_atlas)
                if not atlas_loss.requires_grad:
                    break
                atlas_loss.backward()
                optimizer.step()

        # Restore the decoder's original state
        self.self_supervised_model.custom_recon_decoder.train(decoder_was_training)
        return optimized_bottleneck.detach()

    def _handle_batch(self, data, labels_dict, custom_recon, tta_enabled=True):
        """
        Handles a single test batch, performing TTA if enabled.
        """
        data = data.to(self.device)
        custom_recon = (
            custom_recon.to(self.device) if custom_recon is not None else None
        )
        target = choose_labels(labels_dict, self.chosen_labels)
        target = {
            k: v.to(self.device)
            for k, v in target.items()
            if isinstance(v, torch.Tensor)
        }

        # Get the initial bottleneck representation
        with torch.no_grad():
            _, other_tokens = self.self_supervised_model.encode(data)
            initial_bottleneck = other_tokens

        if tta_enabled and custom_recon is not None:
            final_bottleneck = self._optimize_bottleneck(
                initial_bottleneck, custom_recon
            )
        else:
            final_bottleneck = initial_bottleneck

        with torch.no_grad():
            final_bottleneck_flat = einops.rearrange(
                final_bottleneck, "b t t_d -> b (t t_d)"
            )
            outputs = self.prediction_head(final_bottleneck_flat)

            # Calculate and track metrics
            try:
                outputs, target, _ = handle_null_and_dtypes(
                    outputs, target, self.loss_fns
                )
                task_losses = calculate_task_losses(outputs, target, self.loss_fns)
                total_loss = sum(
                    self.task_weights.get(task, 1.0) * loss
                    for task, loss in task_losses.items()
                )

                for task_name, task_loss in task_losses.items():
                    self.metrics_tracker_test.update(
                        task_name, outputs[task_name], target[task_name], loss=task_loss
                    )
                self.metrics_tracker_test.update_total_loss(total_loss)
            except Exception as e:
                if self.verbose:
                    print(f" Warning: Skipping batch due to invalid labels: {e}")

    def run_evaluation(self, tta_enabled=True):
        """Runs evaluation on the entire test set with or without TTA."""
        if tta_enabled:
            print(
                f"--- Starting evaluation WITH Test-Time Adaptation (K={self.tta_steps} steps) ---"
            )
        else:
            print("--- Starting evaluation for BASELINE (No TTA) ---")

        self.metrics_tracker_test.start_epoch()

        loop = tqdm(
            self.test_dataloader,
            desc=f"Evaluation ({'TTA' if tta_enabled else 'Baseline'})",
        )
        for data, labels_dict, custom_recon in loop:
            self._handle_batch(data, labels_dict, custom_recon, tta_enabled=tta_enabled)

        epoch_metrics = self.metrics_tracker_test.calculate_epoch_metrics()
        print("--- Evaluation complete ---")
        print(epoch_metrics)
        return epoch_metrics

    def compare_with_baseline(self):
        """Runs both baseline and TTA evaluations and returns a comparison."""
        # Run baseline evaluation (TTA disabled)
        baseline_results = self.run_evaluation(tta_enabled=False)

        # Run TTA evaluation (TTA enabled)
        tta_results = self.run_evaluation(tta_enabled=True)

        # Compare and print results
        comparison = {
            "baseline": baseline_results,
            "tta": tta_results,
            "improvement": {},
        }
        print("\n" + "=" * 60)
        print("COMPARISON: TTA vs. Baseline")
        print("=" * 60)
        for task in tta_results:
            if isinstance(tta_results[task], dict):
                comparison["improvement"][task] = {}
                print(f"\nTask: {task}")
                for metric, tta_val in tta_results[task].items():
                    baseline_val = baseline_results.get(task, {}).get(metric)
                    if baseline_val is not None:
                        improvement = tta_val - baseline_val
                        comparison["improvement"][task][metric] = improvement
                        print(
                            f"  {metric:<12}: Baseline={baseline_val:.4f}, TTA={tta_val:.4f}, Change={improvement:+.4f}"
                        )

        return comparison
