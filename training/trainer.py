import glob
import os

import einops
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.metrics import print_metrics
from utils.plotting import plot_grid_recon
from .losses import calculate_task_losses, choose_labels, handle_null_and_dtypes


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        tracker,
        hyperparams,
        scheduler=None,
        tracker_description=None,
        run_id=None,
    ):
        """
        Initialize the base trainer with core training components.

        Args:
            model (torch.nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for test/validation data.
            tracker (object): Tracker for logging and tracking experiments.
            hyperparams (dict): Comprehensive dictionary of experiment hyperparameters.
            scheduler (optional): Learning rate scheduler. Default is None.
            tracker_description (str, optional): Description for the tracker run. Default is None.
            run_id(str, optional), instead of creating a new run id, load a previous one
        """

        self.default_values_used = {}
        self._initialize_core_components(
            model, optimizer, train_dataloader, test_dataloader, tracker, scheduler
        )

        self._setup_hyperparameters(hyperparams)
        self._calculate_intervals()
        self._initialize_tracking(tracker_description, run_id, hyperparams)
        # _ = self._set_loops()
        self._print_default_values()

    def _initialize_tracking(self, tracker_description, run_id, hyperparams):
        """Initialize experiment tracking and run ID."""
        self.tracker_description = self._get_param_with_default(
            hyperparams, "tracker_description", tracker_description
        )

        if run_id is None:
            self.run_id = self._create_new_run()
        else:
            self.run_id = self._load_or_create_run(run_id)

        self.run_checkpoint_dir = self.tracker.runs_index[self.run_id]["checkpoints"]

    def _create_new_run(self, run_id=None):
        """Create a new run with the tracker."""
        run_id = self.tracker.create_run(
            description=self.tracker_description,
            hparams=self.hyperparams,
            run_id=run_id,
        )
        print(f"A new run id has been created {run_id}")
        return run_id

    def _load_or_create_run(self, run_id):
        """Try to load an existing run, or create a new one if not found."""
        try:
            self.tracker.get_run(run_id)
            print("Loading previous run id...")
            return run_id
        except KeyError:
            print("Run ID not found; creating a new one...")
            return self._create_new_run(run_id)

    def _get_param_with_default(self, params_dict, key, default_value):
        """Helper method to get parameter and track if default was used."""
        value = params_dict.get(key, default_value)
        if key not in params_dict:
            self.default_values_used[key] = default_value
        return value

    def _initialize_core_components(
        self, model, optimizer, train_dataloader, test_dataloader, tracker, scheduler
    ):
        """Initialize the core training components."""
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tracker = tracker
        self.scheduler = scheduler
        self.current_epoch = 1
        self.grad_norms = []
        self.grad_norms_clipped = []

    def _setup_hyperparameters(self, hyperparams):
        """Extract and set up training hyperparameters."""
        self.hyperparams = hyperparams
        self.num_epochs = self._get_param_with_default(hyperparams, "num_epochs", 1)
        self.device = self._get_param_with_default(hyperparams, "device", "cuda")
        self.max_norm = self._get_param_with_default(hyperparams, "max_norm", None)
        self.track_grad = self._get_param_with_default(hyperparams, "track_grad", False)
        self.verbose = self._get_param_with_default(hyperparams, "verbose", 1)
        self.best_metric_name = self._get_param_with_default(
            hyperparams, "best_metric_name", None
        )
        self.best_metric_value = self._initialize_best_metric_value()
        self.best_epoch = 1
        self.is_maximization_metric = self.best_metric_value == float("-inf")

    def _calculate_intervals(self):
        """Calculate various intervals for printing, plotting, and checkpointing."""
        print_percentage = self._get_param_with_default(
            self.hyperparams, "print_percentage", 0.1
        )
        plot_percentage = self._get_param_with_default(
            self.hyperparams, "plot_percentage", None
        )
        checkpoints_percentage = self._get_param_with_default(
            self.hyperparams, "checkpoints_percentage", None
        )

        total_prints = round(self.num_epochs * print_percentage)
        total_plots = round(self.num_epochs * plot_percentage) if plot_percentage else 0
        total_checkpoints = (
            round(self.num_epochs * checkpoints_percentage)
            if checkpoints_percentage
            else 1
        )

        self.print_interval = (
            max(1, self.num_epochs // total_prints)
            if total_prints > 0
            else self.num_epochs
        )
        self.plot_interval = (
            max(1, self.num_epochs // total_plots) if total_plots > 0 else None
        )
        self.checkpoint_interval = (
            max(1, self.num_epochs // total_checkpoints)
            if total_checkpoints > 0
            else self.num_epochs
        )

    def _print_default_values(self):
        """Print all hyperparameters that used default values."""
        if self.default_values_used:
            print(
                "\nThe following hyperparameters were not provided and are using default values:"
            )
            for param, default_value in self.default_values_used.items():
                print(f"  - {param}: {default_value}")

    def _initialize_best_metric_value(self):
        """
        Initialize the best metric value based on the metric's nature.

        Returns:
            float: Initial best metric value
        """
        if not self.best_metric_name:
            return None

        maximization_keywords = ["accuracy", "score", "f1", "precision", "recall"]

        minimization_keywords = ["loss", "error", "mse", "mae", "rmse"]

        is_maximization_metric = any(
            keyword in self.best_metric_name.lower()
            for keyword in maximization_keywords
        )

        is_minimization_metric = any(
            keyword in self.best_metric_name.lower()
            for keyword in minimization_keywords
        )

        if is_maximization_metric:
            return float("-inf")
        elif is_minimization_metric:
            return float("inf")
        else:
            # If we can't determine, default to minimization
            print(
                f"Warning: Could not determine optimization direction for metric {self.best_metric_name}. Defaulting to minimization."
            )
            return float("inf")

    def _set_loops(self):
        self.epoch_loop = (
            tqdm(range(self.current_epoch, self.num_epochs + 1), desc="epoch loop")
            if self.verbose >= 1
            else range(self.current_epoch, self.num_epochs + 1)
        )
        self.train_loop = (
            tqdm(self.train_dataloader, colour="#1167b1", desc="train dataloader loop")
            if self.verbose == 2
            else self.train_dataloader
        )
        self.test_loop = (
            tqdm(self.test_dataloader, colour="red", desc="test dataloader loop")
            if self.verbose == 2
            else self.test_dataloader
        )
        return None

    def _save_checkpoint(self, epoch):
        """
        Save a checkpoint of the current training state.

        Args:
            epoch (int): Current epoch number
        """
        checkpoint_path = os.path.join(
            self.run_checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_norms": self.grad_norms,
            "current_lr": self.optimizer.param_groups[0]["lr"],
        }

        checkpoint["scheduler_state_dict"] = (
            self.scheduler.state_dict() if self.scheduler else None
        )

        torch.save(checkpoint, checkpoint_path)

        if self.verbose >= 1:
            print(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path=None, epoch=None):
        """
        Load a checkpoint for continuing training.
        if no checkpoint path and no epoch is given then it takes the latest checkpoint
        Args:
            checkpoint_path (str, optional): Specific checkpoint file to load
            epoch (int, optional): Epoch number to load (will find the corresponding checkpoint)

        Returns:
            dict: Loaded checkpoint information
        """
        # If no specific path is provided, find the latest or specific epoch checkpoint
        if checkpoint_path is None:
            # Get all checkpoints for this run
            checkpoints = sorted(
                glob.glob(
                    os.path.join(self.run_checkpoint_dir, "checkpoint_epoch_*.pth")
                )
            )

            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.run_checkpoint_dir}"
                )

            # If specific epoch is provided, find its checkpoint
            if epoch is not None:
                checkpoint_path = os.path.join(
                    self.run_checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth"
                )
                if checkpoint_path not in checkpoints:
                    raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")
            else:
                # Default to the latest checkpoint (last in sorted list)
                checkpoint_path = checkpoints[-1]

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.grad_norms = checkpoint.get("grad_norms", [])

        if self.verbose >= 1:
            print(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint

    def _get_grad_norm(self, norm=2):
        """
        Calculate the gradient norm of model parameters.

        Args:
            norm (int): Type of norm to calculate (default: 2 for L2 norm)

        Returns:
            float: Gradient norm
        """
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(norm)
                total_norm += param_norm.item() ** norm
        return total_norm ** (1 / norm)

    def _handle_batch(self, data, metrics_tracker, is_training=True):
        """
        Process a single batch of data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def train(self):
        """
        Main training loop for the model.
        Trains for specified number of epochs,
        alternating between training and validation.
        """
        self._set_loops()

        for epoch in self.epoch_loop:
            self.model.train()
            if self.verbose >= 1:
                self.epoch_loop.set_postfix(
                    {"lr": self.optimizer.param_groups[0]["lr"]}
                )
            self._train_epoch(self.train_loop, epoch)
            if (
                self.checkpoint_interval is not None
                and epoch % self.checkpoint_interval == 0
            ):
                self._save_checkpoint(epoch=epoch)

            torch.cuda.empty_cache()
            self.model.eval()
            self._validate_epoch(self.test_loop, epoch)
        self._save_checkpoint(epoch=epoch)

        return self.metrics_tracker_tr, self.metrics_tracker_test

    def continue_training(
        self, checkpoint_path=None, epoch=None, new_total_epochs=None
    ):
        """
        Continue training from the last saved checkpoint.

        Args:
            new_total_epochs (int, optional): New total number of epochs to train.
                                              If None, uses the original num_epochs.
            checkpoint_path (str, optional): Specific checkpoint file to load
            epoch (int, optional): Epoch number to load (will find the corresponding checkpoint)
        """

        self.load_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch)
        print(f"Resuming from epoch {self.current_epoch}")

        if new_total_epochs is None:
            new_total_epochs = self.num_epochs

        original_epochs = self.num_epochs
        self.num_epochs = new_total_epochs

        if self.verbose >= 1:
            print(f"Continuing training from epoch {self.current_epoch}")
            print(f"Will train until epoch {new_total_epochs}")

        try:
            self.epoch_loop = (
                tqdm(
                    range(self.current_epoch + 1, new_total_epochs + 1),
                    desc="epoch loop",
                )
                if self.verbose >= 1
                else range(self.current_epoch + 1, new_total_epochs + 1)
            )
            self.train()

        except Exception as e:
            print(f"Training interrupted: {e}")

            self._save_checkpoint(epoch)

        finally:
            self.num_epochs = original_epochs

    def _update_best_checkpoint(self, epoch, current_metric_value):
        """
        Update the best checkpoint if the specified metric has improved.

        Args:
            epoch (int): Current epoch number

        Returns:
            bool: Whether a new best checkpoint was saved
        """

        is_better = (
            (current_metric_value > self.best_metric_value)
            if self.is_maximization_metric
            else (current_metric_value < self.best_metric_value)
        )

        if is_better:
            if hasattr(self, "best_checkpoint_path") and os.path.exists(
                self.best_checkpoint_path
            ):
                os.remove(self.best_checkpoint_path)

            new_checkpoint_path = os.path.join(
                self.tracker.runs_index[self.run_id]["checkpoints"],
                f"best_checkpoint_epoch_{epoch}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metric_value": current_metric_value,
                },
                new_checkpoint_path,
            )

            self.best_metric_value = current_metric_value
            self.best_epoch = epoch
            self.best_checkpoint_path = new_checkpoint_path

            return True

        return False

    def _train_epoch(self, train_dataloader, epoch):
        """
        Perform training for a single epoch.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _validate_epoch(self, test_dataloader, epoch):
        """
        Perform validation for a single epoch.
        """
        raise NotImplementedError("Subclasses must implement this method")


class MultiTaskTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        tracker,
        hyperparams,
        metrics_tracker_tr,
        metrics_tracker_test,
        loss_fns,
        scheduler=None,
        tracker_description=None,
        run_id=None,
        baselines=None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            hyperparams=hyperparams,
            tracker=tracker,
            scheduler=scheduler,
            tracker_description=tracker_description,
            run_id=run_id,
        )
        self.metrics_tracker_tr = metrics_tracker_tr
        self.metrics_tracker_test = metrics_tracker_test
        self.chosen_labels = hyperparams["chosen_labels"]
        self.task_weights = hyperparams["TASK_WEIGHTS"]
        self.loss_fns = {k: v for k, v in loss_fns.items() if k in self.chosen_labels}
        self.checkpoint_task = hyperparams.get("CHECKPOINT_TASK", "Reconstruction")
        self.baselines = baselines

    def _handle_batch(
        self, data, labels_dict, custom_recon, metrics_tracker, is_training=True
    ):
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
        target["Reconstruction"] = (
            custom_recon if self.hyperparams.get("CUSTOM_RECON_BOOL") else data
        )

        if is_training:
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = self.model(data, custom_recon)  # Pass custom_recon to model
            original_recons = outputs.get("Reconstruction")

            outputs, target, _ = handle_null_and_dtypes(outputs, target, self.loss_fns)
            task_losses = calculate_task_losses(outputs, target, self.loss_fns)
            total_loss = sum(
                self.task_weights.get(task, 1.0) * loss
                for task, loss in task_losses.items()
            )

            for task_name, task_loss in task_losses.items():
                metrics_tracker.update(
                    task_name,
                    outputs.get(task_name),
                    target.get(task_name),
                    loss=task_loss,
                )
            metrics_tracker.update_total_loss(total_loss)

            if is_training:
                total_loss.backward()
                if self.track_grad:
                    self.grad_norms.append(self._get_grad_norm())
                if self.max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
                if self.track_grad and self.max_norm:
                    self.grad_norms_clipped.append(self._get_grad_norm())
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        return total_loss, task_losses, original_recons

    def _train_epoch(self, train_dataloader, epoch):
        self.metrics_tracker_tr.start_epoch()
        recons_for_plot, custom_recon_for_plot = None, None
        for data, labels_dict, custom_recon in train_dataloader:
            _, _, recons = self._handle_batch(
                data,
                labels_dict,
                custom_recon,
                self.metrics_tracker_tr,
                is_training=True,
            )
            if recons is not None:
                recons_for_plot, custom_recon_for_plot = recons, custom_recon

        epoch_metrics_tr = self.metrics_tracker_tr.calculate_epoch_metrics()
        epoch_total_loss = np.mean(self.metrics_tracker_tr.total_losses)

        if self.print_interval and (epoch % self.print_interval == 0):
            print_metrics(epoch, epoch_metrics_tr, epoch_total_loss, source="Train")
            print(f" LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        if (
            self.plot_interval
            and (epoch % self.plot_interval == 0)
            and recons_for_plot is not None
        ):
            plot_grid_recon(
                originals=custom_recon_for_plot,
                reconstructeds=recons_for_plot,
                max_rows=self.hyperparams.get("MAX_ROWS_FOR_PLOT", 2),
                save_path=os.path.join(
                    self.run_checkpoint_dir,
                    "..",
                    "plots",
                    f"recon_train_epoch_{epoch}.png",
                ),
                title="Train Recon (last batch)",
            )
        return epoch_metrics_tr, epoch_total_loss

    def _validate_epoch(self, test_dataloader, epoch):
        self.metrics_tracker_test.start_epoch()
        recons_for_plot, custom_recon_for_plot = None, None
        with torch.no_grad():
            for data, labels_dict, custom_recon in test_dataloader:
                _, _, recons = self._handle_batch(
                    data,
                    labels_dict,
                    custom_recon,
                    self.metrics_tracker_test,
                    is_training=False,
                )
                if recons is not None:
                    recons_for_plot, custom_recon_for_plot = recons, custom_recon

        epoch_metrics_test = self.metrics_tracker_test.calculate_epoch_metrics()
        epoch_total_loss = np.mean(self.metrics_tracker_test.total_losses)

        if self.print_interval and (epoch % self.print_interval == 0):
            print_metrics(epoch, epoch_metrics_test, epoch_total_loss, source="Test")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"Memory Allocated: {allocated:.2f} GB")

        if (
            self.plot_interval
            and (epoch % self.plot_interval == 0)
            and recons_for_plot is not None
        ):
            plot_grid_recon(
                originals=custom_recon_for_plot,
                reconstructeds=recons_for_plot,
                max_rows=self.hyperparams.get("MAX_ROWS_FOR_PLOT", 2),
                save_path=os.path.join(
                    self.run_checkpoint_dir,
                    "..",
                    "plots",
                    f"recon_test_epoch_{epoch}.png",
                ),
                title="Test Recon (last batch)",
            )
        return epoch_metrics_test, epoch_total_loss

    def train(self):
        self._set_loops()
        for epoch in self.epoch_loop:
            if self.verbose >= 1:
                self.epoch_loop.set_postfix(
                    {"lr": self.optimizer.param_groups[0]["lr"]}
                )

            self.model.train()
            self._train_epoch(self.train_loop, epoch)
            torch.cuda.empty_cache()

            self.model.eval()
            metrics_test, _ = self._validate_epoch(self.test_loop, epoch)

            if self.best_metric_name:
                metric_value = metrics_test.get(self.checkpoint_task, {}).get(
                    self.best_metric_name
                )
                if self._update_best_checkpoint(epoch, metric_value):
                    if self.verbose >= 1:
                        print(
                            f"** New best checkpoint saved at epoch {epoch} with {self.best_metric_name} of {metric_value:.4f} **"
                        )

            if (
                self.checkpoint_interval
                and (epoch % self.checkpoint_interval == 0)
                and epoch > 0
            ):
                self._save_checkpoint(epoch=epoch)

        self._save_checkpoint(epoch=self.num_epochs)
        return self.metrics_tracker_tr, self.metrics_tracker_test


class FineTuner(MultiTaskTrainer):
    """
    A specialized trainer for Stage 2: Supervised Fine-tuning.
    This trainer freezes a pre-trained model and trains a new prediction head
    on its bottleneck features.
    """

    def __init__(
        self,
        model,
        prediction_head,
        optimizer,
        train_dataloader,
        test_dataloader,
        tracker,
        hyperparams,
        metrics_tracker_tr,
        metrics_tracker_test,
        loss_fns,
        freeze_model=True,
        scheduler=None,
        tracker_description=None,
        run_id=None,
        baselines=None,
    ):
        """
        Args:
            model (torch.nn.Module): The pre-trained self-supervised model (e.g., TransformerAutoEncoder).
            prediction_head (torch.nn.Module): The new prediction head to be trained.
            freeze_model (bool): If True, freezes all parameters of the `model`.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            hyperparams=hyperparams,
            tracker=tracker,
            metrics_tracker_tr=metrics_tracker_tr,
            metrics_tracker_test=metrics_tracker_test,
            loss_fns=loss_fns,
            scheduler=scheduler,
            tracker_description=tracker_description,
            run_id=run_id,
            baselines=baselines,
        )
        self.self_supervised_model = self.model
        self.model = prediction_head  # The trainable part is now the prediction head
        self.first_pass = True

        if freeze_model:
            self._freeze_model_parameters()

    def _freeze_model_parameters(self):
        """Freezes all parameters of the self-supervised encoder/decoder."""
        print("Freezing pre-trained model parameters for fine-tuning.")
        for param in self.self_supervised_model.parameters():
            param.requires_grad = False

    def _handle_batch(
        self, data, labels_dict, custom_recon, metrics_tracker, is_training=True
    ):
        data = data.to(self.device)
        target = choose_labels(labels_dict, self.chosen_labels)
        target = {
            k: v.to(self.device)
            for k, v in target.items()
            if isinstance(v, torch.Tensor)
        }

        if is_training:
            self.optimizer.zero_grad()

        with torch.no_grad():
            self.self_supervised_model.first_pass = False  # Prevent internal prints
            _, other_tokens = self.self_supervised_model.encode(data)
            bottleneck = einops.rearrange(other_tokens, "b t d -> b (t d)")

        with torch.set_grad_enabled(is_training):
            outputs = self.model(bottleneck)
            del bottleneck

            outputs, target, _ = handle_null_and_dtypes(outputs, target, self.loss_fns)
            task_losses = calculate_task_losses(outputs, target, self.loss_fns)
            total_loss = sum(
                self.task_weights.get(task, 1.0) * loss
                for task, loss in task_losses.items()
            )

            for task_name, task_loss in task_losses.items():
                metrics_tracker.update(
                    task_name,
                    outputs.get(task_name),
                    target.get(task_name),
                    loss=task_loss,
                )
            metrics_tracker.update_total_loss(total_loss)

            if is_training:
                total_loss.backward()
                if self.track_grad:
                    self.grad_norms.append(self._get_grad_norm())
                if self.max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
                if self.track_grad and self.max_norm:
                    self.grad_norms_clipped.append(self._get_grad_norm())
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        return total_loss, task_losses, None
