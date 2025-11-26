from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tabulate import tabulate
from tqdm.auto import tqdm


def print_combined_metrics(
    epoch: int,
    task_metrics: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, pd.DataFrame],
    total_loss: float,
    source: str,
    models_to_show: List[str] = None,
    decimals: int = 5,
) -> None:
    """
    Prints compact epoch and baseline metrics side by side.

    Args:
        epoch (int): Current epoch number.
        task_metrics (Dict[str, Dict[str, float]]): Current metrics per task.
        baseline_metrics (Dict[str, pd.DataFrame]): Baseline metrics per model.
        total_loss (float): Total loss value.
        source (str): Source name (tr/val/tst).
        models_to_show (List[str]): List of baseline models to display. If None, shows all.
        decimals (int): Number of decimal places (default: 5).
    """

    metric_map = {
        "accuracy": "acc",
        "precision": "prc",
        "recall": "rcl",
        "f1": "f1",
        "loss": "loss",
    }

    source = source.lower()

    source_map = {"train": "tr", "val": "val", "test": "tst"}

    colors = {
        "tr": ("\033[92m", "\033[32m"),  # Bright green, Dark green
        "val": ("\033[94m", "\033[34m"),  # Bright blue, Dark blue
        "tst": ("\033[93m", "\033[33m"),  # Bright yellow, Dark yellow
    }
    reset = "\033[0m"

    src_abbr = source_map.get(source, source)
    source_color, baseline_color = colors.get(src_abbr, ("\033[97m", "\033[37m"))

    if models_to_show is None:
        models_to_show = list(baseline_metrics.keys())

    header = ["Task"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:
        abbr = metric_map[metric]
        header.append(f"{abbr}_{src_abbr}")
        for model in models_to_show:
            header.append(f"{abbr}_{model[:3]}")

    header.append(f"{metric_map['loss']}_{src_abbr}")

    table_data = []
    for task in task_metrics.keys():
        current_metrics = task_metrics[task]

        short_task = task.split("_")[0] if "_" in task else task
        short_task = short_task[:10] if len(short_task) > 10 else short_task

        row = [f"{source_color}{short_task}{reset}"]

        for metric in metrics:
            curr_value = current_metrics.get(metric, float("nan"))
            row.append(f"{source_color}{curr_value:.{decimals}f}{reset}")

            for model in models_to_show:
                base_value = (
                    baseline_metrics[model].loc[task, metric]
                    if task in baseline_metrics[model].index
                    else float("nan")
                )
                row.append(f"{baseline_color}{base_value:.{decimals}f}{reset}")

        #
        row.append(
            f"{source_color}{current_metrics.get('loss', float('nan')):.{decimals}f}{reset}"
        )

        table_data.append(row)

    print(
        f"\033[1mEpoch:{epoch}, Loss:{total_loss:.{decimals}f} ({source_color}{src_abbr}{reset})\033[0m"
    )

    table = tabulate(
        table_data,
        headers=header,
        tablefmt="fancy_grid",
        stralign="right",
        numalign="right",
    )
    print(table)
    return table


def print_metrics(
    epoch: int,
    task_metrics: Dict[str, Dict[str, float]],
    total_loss: float,
    source: str,
) -> None:
    colors = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
    ]
    reset = "\033[0m"  # Reset color
    bold = "\033[1m"  # Bold text

    task_colors = {
        task: colors[i % len(colors)] for i, task in enumerate(task_metrics.keys())
    }

    print(f"{bold}Epoch: {epoch}, Total Loss: {total_loss:.4f}({source}){reset}")

    for task, metrics in task_metrics.items():
        color = task_colors[task]

        task_str = f"{bold}{color}{task}({source}){reset}"

        metrics_str = (
            ", ".join(
                f"{color}{metric} ({source}): {value:.4f}{reset}"
                for metric, value in metrics.items()
            )
            if task.lower() != "reconstruction"
            else f"{color}{'loss'} ({source}): {metrics['loss']:.4f}{reset}"
        )
        print(f"{task_str}: {metrics_str}")


class MetricsTracker:
    """
    Tracks metrics, predictions, and losses for multiple tasks across training epochs.

    Attributes:
        tasks_types (dict): A dictionary where keys are task names and values are task types
            ('Categorical' or 'Binary').
        epoch_data (dict): Stores calculated metrics for each epoch.
        predictions (dict): Stores predictions for each task in the current epoch.
        ground_truth (dict): Stores ground truth values for each task in the current epoch.
        current_epoch (int): Tracks the current epoch number.
        task_losses (dict): Tracks batch-level losses for each task.
        total_losses (list): Tracks total losses for each batch.
        zero_division

    Methods:
        start_epoch(): Resets tracking data for a new epoch.
        update(task_name, preds, targets, loss=None): Updates predictions, ground truth, and losses for a task.
        update_total_loss(total_loss): Tracks the total loss for the current batch.
        calculate_epoch_metrics(): Computes and stores metrics for all tasks at the end of an epoch.
        get_metrics(): Returns the stored metrics for all epochs.
    """

    def __init__(self, tasks_types, zero_division=0):
        """
        Initialize the MetricsTracker.

        Args:
            tasks_types (dict): Dictionary mapping task names to task types ('Categorical' or 'Binary').
        """
        self.epoch_data = {}
        self.predictions = {}
        self.ground_truth = {}
        self.tasks_types = tasks_types
        self.tasks = [task for task in self.tasks_types.keys() if task != "Total_Loss"]
        self.current_epoch = 1
        self.zero_division = zero_division  # {“warn”, 0.0, 1.0, np.nan}
        # Initialize loss tracking
        self.task_losses = {}
        self.total_losses = []

    def start_epoch(self):
        """Prepares the tracker for a new epoch by resetting predictions, ground truth, and losses."""
        self.predictions = {}
        self.ground_truth = {}

        # Reset loss tracking for the new epoch
        self.task_losses = {task: [] for task in self.tasks_types.keys()}
        self.total_losses = []

    def update(self, task_name, preds=None, targets=None, loss=None):
        """
        Updates the tracker with new predictions, ground truth, and optional loss for a task.

        Args:
            task_name (str): Name of the task.
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.
            loss (torch.Tensor, optional): Loss for the task. Default is None.
        """
        assert (
            preds is not None and targets is not None
        ) or task_name.lower() == "reconstruction", (
            "preds and targets can only be None if task_name is 'reconstruction'."
        )

        if loss is not None:
            self.task_losses[task_name].append(loss.item())

        if task_name not in self.predictions:
            self.predictions[task_name] = []
            self.ground_truth[task_name] = []

        if task_name.lower() == "reconstruction":
            return

        # Detach predictions and targets from the graph and move to CPU
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        type_of_task = self.tasks_types[task_name]

        if type_of_task == "categorical":  # Categorical task
            preds = preds.argmax(dim=1)

        elif type_of_task == "binary":  # Binary task
            preds = (torch.sigmoid(preds) > 0.5).int()

        self.predictions[task_name].extend(
            preds.numpy()
        ) if task_name.lower() != "reconstruction" else None

        self.ground_truth[task_name].extend(
            targets.numpy()
        ) if task_name.lower() != "reconstruction" else None

    def update_total_loss(self, total_loss):
        """
        Records the total loss for the current batch.


        Args:
            total_loss (torch.Tensor): Total loss for the batch.

        """

        self.total_losses.append(total_loss.item())

    def calculate_epoch_metrics(self):
        """
        Calculates metrics (accuracy, precision, recall, f1, loss) for each task at the end of an epoch.

        Returns:
            dict: A dictionary of metrics for all tasks in the current epoch.
        """
        epoch_metrics = {}
        for task_name in self.predictions.keys():
            type_of_task = self.tasks_types[task_name]

            if type_of_task == "regression":
                task_metrics = {
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "loss": np.mean(self.task_losses[task_name]),
                }

            else:
                preds = np.stack(self.predictions[task_name])
                targets = np.stack(self.ground_truth[task_name])
                task_metrics = {
                    "accuracy": accuracy_score(targets, preds),
                    "precision": precision_score(
                        targets,
                        preds,
                        zero_division=self.zero_division,
                        average="binary" if type_of_task == "binary" else "macro",
                    ),
                    "recall": recall_score(
                        targets,
                        preds,
                        zero_division=self.zero_division,
                        average="binary" if type_of_task == "binary" else "macro",
                    ),
                    "f1": f1_score(
                        targets,
                        preds,
                        zero_division=self.zero_division,
                        average="binary" if type_of_task == "binary" else "macro",
                    ),
                    "loss": np.mean(self.task_losses[task_name]),
                }

            if task_name not in self.epoch_data:
                self.epoch_data[task_name] = {}
            self.epoch_data[task_name][self.current_epoch] = task_metrics
            epoch_metrics[task_name] = task_metrics

        self.epoch_data.setdefault("Total_Loss", {})[self.current_epoch] = np.mean(
            self.total_losses
        )

        self.current_epoch += 1
        return epoch_metrics

    def get_metrics(self):
        """
        Retrieves the stored metrics for all epochs.

        Returns:
            dict: Metrics for all tasks across all epochs.
        """
        return self.epoch_data

    def dict_to_dataframe(self, save_path: str = None) -> pd.DataFrame:
        """
        Converts a nested dictionary structure of metrics to a DataFrame.

        Parameters:
        - save_path (Optional[str]): If provided, saves the resulting DataFrame to the specified file path.

        Returns:
        - pd.DataFrame: A DataFrame where rows represent epochs, and columns are
          combinations of task names and metrics.

        """
        data = self.get_metrics()
        flattened_data = [
            (epoch, f"{task}_{metric}", value)
            if isinstance(metrics, dict)
            else (epoch, f"{task}", metrics)
            for task, epochs in data.items()
            for epoch, metrics in epochs.items()
            for metric, value in (
                metrics.items() if isinstance(metrics, dict) else [(None, metrics)]
            )
        ]

        df = pd.DataFrame(flattened_data, columns=["epoch", "metric", "value"])

        result_df = df.pivot(index="epoch", columns="metric", values="value")
        result_df.columns.name = None

        result_df.index.name = "Epoch"
        if save_path:
            result_df.to_csv(save_path)
            print(f"DataFrame saved to {save_path}")
        return result_df
