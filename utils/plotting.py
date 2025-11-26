import itertools
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix


class MetricsPlotter:
    def __init__(
        self,
        train_metrics: Dict[str, Dict[int, Any]],
        val_metrics: Optional[Dict[str, Dict[int, Any]]] = None,
        test_metrics: Optional[Dict[str, Dict[int, Any]]] = None,
        baseline_metrics_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Initialize the MetricsPlotter.

        Args:
            train_metrics (Dict[str, Dict[int, Any]]): Training metrics dictionary.
            val_metrics (Optional[Dict[str, Dict[int, Any]]]): Validation metrics dictionary. Default is None.
            test_metrics (Optional[Dict[str, Dict[int, Any]]]): Test metrics dictionary. Default is None.
            baseline_metrics_dict (Optional[Dict[str, pd.DataFrame]]): Dictionary of baseline metrics DataFrames. Default is None.
        """
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.baseline_metrics_dict = baseline_metrics_dict or {}
        self.colors: Dict[str, str] = {
            "train": "blue",
            "validation": "orange",
            "test": "green",
        }
        self.baseline_color = "gray"
        self.baseline_markers: Dict[str, str] = self._generate_baseline_markers()
        self.tasks = [
            task for task in self.train_metrics.keys() if task != "Total_Loss"
        ]

    def _generate_baseline_markers(self) -> Dict[str, Dict[str, str]]:
        """
        Generate distinct markers for each baseline for both Matplotlib and Plotly.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping baseline names to markers for Matplotlib and Plotly.
        """
        matplotlib_markers = itertools.cycle(
            ["o", "s", "D", "x", "+", "^", "v", "<", ">", "p", "*"]
        )
        plotly_markers = itertools.cycle(
            [
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "triangle-down",
                "triangle-left",
                "triangle-right",
                "pentagon",
                "star",
            ]
        )

        return {
            baseline: {
                "matplotlib": next(matplotlib_markers),
                "plotly": next(plotly_markers),
            }
            for baseline in self.baseline_metrics_dict.keys()
        }

    def _save_figure_plotly(self, fig: go.Figure, save_path: str) -> None:
        """Helper function to save a Plotly figure to a file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """Helper function to save a Matplotlib figure to a file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    def _get_dataset_metrics(self, dataset_name):
        metrics_map = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

        if dataset_name not in metrics_map:
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. Choose from ['train', 'val', 'test']."
            )

        metrics = metrics_map[dataset_name]

        if metrics is None:
            raise ValueError(f"Metrics for {dataset_name} are Empty, choose another.")

        return metrics

    def plot_task_metrics_plotly(
        self, task_name: str, save: Optional[str] = None, plot: bool = True
    ) -> None:
        """
        Plots metrics for a specific task and Total_Loss in a 3x2 grid.

        Args:
            task_name (str): Name of the task to plot metrics for.
            save (Optional[str]): Path to save the plot. Default is None.
            plot (bool): Whether to display the plot. Default is True.
        """

        train_task_metrics = self.train_metrics.get(task_name, {})
        val_task_metrics = (
            self.val_metrics.get(task_name, {}) if self.val_metrics else {}
        )
        test_task_metrics = (
            self.test_metrics.get(task_name, {}) if self.test_metrics else {}
        )

        train_total_loss = self.train_metrics.get("Total_Loss", {})
        val_total_loss = (
            self.val_metrics.get("Total_Loss", {}) if self.val_metrics else {}
        )
        test_total_loss = (
            self.test_metrics.get("Total_Loss", {}) if self.test_metrics else {}
        )

        metrics = list(next(iter(train_task_metrics.values())).keys())
        metrics.append("Total_Loss")  # Add Total_Loss as a metric

        rows, cols = 3, 2
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                f"{metric.capitalize()} for {task_name}" for metric in metrics
            ],
        )

        legend_shown = {
            "train": False,
            "validation": False,
            "test": False,
            "logistic": False,
            "ST-NN": False,
            "MT-NN": False,
        }

        for idx, metric in enumerate(metrics):
            row, col = divmod(idx, cols)
            row += 1
            col += 1

            if metric == "Total_Loss":
                x_train = list(train_total_loss.keys())
                y_train = [train_total_loss[epoch] for epoch in x_train]
                fig.add_trace(
                    go.Scatter(
                        x=x_train,
                        y=y_train,
                        mode="lines+markers",
                        name="Train",
                        line=dict(color=self.colors["train"]),
                        legendgroup="train",
                        showlegend=not legend_shown["train"],
                    ),
                    row=row,
                    col=col,
                )
                legend_shown["train"] = True

                if val_total_loss:
                    x_val = list(val_total_loss.keys())
                    y_val = [val_total_loss[epoch] for epoch in x_val]
                    fig.add_trace(
                        go.Scatter(
                            x=x_val,
                            y=y_val,
                            mode="lines+markers",
                            name="Validation",
                            line=dict(color=self.colors["validation"]),
                            legendgroup="validation",
                            showlegend=not legend_shown["validation"],
                        ),
                        row=row,
                        col=col,
                    )
                    legend_shown["validation"] = True

                if test_total_loss:
                    x_test = list(test_total_loss.keys())
                    y_test = [test_total_loss[epoch] for epoch in x_test]
                    fig.add_trace(
                        go.Scatter(
                            x=x_test,
                            y=y_test,
                            mode="lines+markers",
                            name="Test",
                            line=dict(color=self.colors["test"]),
                            legendgroup="test",
                            showlegend=not legend_shown["test"],
                        ),
                        row=row,
                        col=col,
                    )
                    legend_shown["test"] = True
            else:
                x_train = list(train_task_metrics.keys())
                y_train = [train_task_metrics[epoch][metric] for epoch in x_train]
                fig.add_trace(
                    go.Scatter(
                        x=x_train,
                        y=y_train,
                        mode="lines+markers",
                        name="Train",
                        line=dict(color=self.colors["train"]),
                        legendgroup="train",
                        showlegend=not legend_shown["train"],
                    ),
                    row=row,
                    col=col,
                )
                legend_shown["train"] = True

                if val_task_metrics:
                    x_val = list(val_task_metrics.keys())
                    y_val = [val_task_metrics[epoch][metric] for epoch in x_val]
                    fig.add_trace(
                        go.Scatter(
                            x=x_val,
                            y=y_val,
                            mode="lines+markers",
                            name="Validation",
                            line=dict(color=self.colors["validation"]),
                            legendgroup="validation",
                            showlegend=not legend_shown["validation"],
                        ),
                        row=row,
                        col=col,
                    )
                    legend_shown["validation"] = True

                if test_task_metrics:
                    x_test = list(test_task_metrics.keys())
                    y_test = [test_task_metrics[epoch][metric] for epoch in x_test]
                    fig.add_trace(
                        go.Scatter(
                            x=x_test,
                            y=y_test,
                            mode="lines+markers",
                            name="Test",
                            line=dict(color=self.colors["test"]),
                            legendgroup="test",
                            showlegend=not legend_shown["test"],
                        ),
                        row=row,
                        col=col,
                    )
                    legend_shown["test"] = True

            for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
                if (
                    task_name in baseline_metrics.index
                    and metric in baseline_metrics.columns
                ):
                    baseline_value = baseline_metrics.loc[task_name, metric]
                    fig.add_trace(
                        go.Scatter(
                            x=x_train,
                            y=[baseline_value] * len(x_train),
                            mode="lines+markers",
                            name=baseline_name,
                            line=dict(color=self.baseline_color, dash="dot"),
                            marker=dict(
                                symbol=self.baseline_markers[baseline_name]["plotly"]
                            ),
                            showlegend=not legend_shown[baseline_name],
                        ),
                        row=row,
                        col=col,
                    )
                    legend_shown[baseline_name] = True

        fig.update_layout(
            height=800,
            width=1000,
            title_text=f"Metrics for Task: {task_name}",
            showlegend=True,
        )

        if save:
            self._save_figure_plotly(fig, save)
        if plot:
            fig.show()

    def plot_all_metrics_plotly(
        self, save_dir: Optional[str] = None, plot: bool = False
    ) -> None:
        """
        Plots metrics for all tasks in a 3x2 grid, including Total_Loss.

        Args:
            save_dir (Optional[str]): Directory to save all plots. Default is None.
            plot (bool): Whether to display the plots. Default is True.
        """
        for task_name in self.train_metrics.keys():
            if task_name == "Total_Loss":
                continue  # Skip Total_Loss; it will be plotted as part of each task

            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{task_name}_metrics.html")

            self.plot_task_metrics_plotly(task_name, save=save_path, plot=plot)

    # ----------- Matplotlib Functions -----------

    def plot_task_metrics(
        self, task_name: str, save: Optional[str] = None, plot: bool = True
    ) -> None:
        """
        Plots metrics for a specific task and Total_Loss in a 3x2 grid using Matplotlib.

        Args:
            task_name (str): Name of the task to plot metrics for.
            save (Optional[str]): Path to save the plot. Default is None.
            plot (bool): Whether to display the plot. Default is True.
        """
        train_task_metrics = self.train_metrics.get(task_name, {})
        val_task_metrics = (
            self.val_metrics.get(task_name, {}) if self.val_metrics else {}
        )
        test_task_metrics = (
            self.test_metrics.get(task_name, {}) if self.test_metrics else {}
        )

        train_total_loss = self.train_metrics.get("Total_Loss", {})
        val_total_loss = (
            self.val_metrics.get("Total_Loss", {}) if self.val_metrics else {}
        )
        test_total_loss = (
            self.test_metrics.get("Total_Loss", {}) if self.test_metrics else {}
        )

        metrics = list(next(iter(train_task_metrics.values())).keys())
        metrics.append("Total_Loss")

        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()

        # To collect handles and labels for a single legend
        handles = []
        labels = []

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x_train = list(train_task_metrics.keys())
            y_train = (
                [train_total_loss[epoch] for epoch in x_train]
                if metric == "Total_Loss"
                else [train_task_metrics[epoch][metric] for epoch in x_train]
            )
            (train_line,) = ax.plot(
                x_train, y_train, label="Train", color=self.colors["train"]
            )

            if "Train" not in labels:
                handles.append(train_line)
                labels.append("Train")

            if val_task_metrics:
                x_val = list(val_task_metrics.keys())
                y_val = (
                    [val_total_loss[epoch] for epoch in x_val]
                    if metric == "Total_Loss"
                    else [val_task_metrics[epoch][metric] for epoch in x_val]
                )
                (val_line,) = ax.plot(
                    x_val, y_val, label="Validation", color=self.colors["validation"]
                )

                if "Validation" not in labels:
                    handles.append(val_line)
                    labels.append("Validation")

            if test_task_metrics:
                x_test = list(test_task_metrics.keys())
                y_test = (
                    [test_total_loss[epoch] for epoch in x_test]
                    if metric == "Total_Loss"
                    else [test_task_metrics[epoch][metric] for epoch in x_test]
                )
                (test_line,) = ax.plot(
                    x_test, y_test, label="Test", color=self.colors["test"]
                )

                if "Test" not in labels:
                    handles.append(test_line)
                    labels.append("Test")

            # Add multiple baselines
            for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
                if (
                    task_name in baseline_metrics.index
                    and metric in baseline_metrics.columns
                ):
                    baseline_value = baseline_metrics.loc[task_name, metric]
                    (baseline_line,) = ax.plot(
                        x_train,
                        [baseline_value] * len(x_train),
                        linestyle="dotted",
                        color=self.baseline_color,
                        marker=self.baseline_markers[baseline_name]["matplotlib"],
                        label=f"{baseline_name}",
                    )
                    if baseline_name not in labels:
                        handles.append(baseline_line)
                        labels.append(baseline_name)

            ax.set_title(f"{metric.capitalize()} for {task_name}")

        # Remove individual legends and create a single shared legend
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize="medium",
            frameon=False,
        )
        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust layout to make space for the legend

        if save:
            self._save_figure(fig, save)
        if plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_all_metrics(
        self, save_dir: Optional[str] = None, plot: bool = False
    ) -> None:
        """
        Plots metrics for all tasks, including Total_Loss, in a 3x2 grid using Matplotlib.

        Args:
            save_dir (Optional[str]): Directory to save all plots. Default is None.
            plot (bool): Whether to display the plots. Default is True.
        """
        for task_name in self.train_metrics.keys():
            if task_name == "Total_Loss":
                continue  # Skip Total_Loss; it will be plotted as part of each task

            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{task_name}_metrics.png")

            self.plot_task_metrics(task_name, save=save_path, plot=plot)

    def save_all_plots(self, save_dir: str) -> None:
        """
        Saves all plots (both Matplotlib and Plotly versions).

        Args:
            save_dir (str): Directory to save all plots.
        """
        self.plot_all_metrics(save_dir=save_dir, plot=False)
        self.plot_all_metrics_plotly(save_dir=save_dir, plot=False)

    def plot_metric_plotly(
        self, metric: str, save: Optional[str] = None, plot: bool = True
    ) -> None:
        """
        Plots a specific metric with a subplot for each task using Plotly.

        Args:
            metric (str): The metric to plot ('accuracy', 'precision', 'recall', 'f1', 'loss').
            save (Optional[str]): Path to save the plot. Default is None.
            plot (bool): Whether to display the plot. Default is True.
        """
        tasks = self.tasks
        fig = make_subplots(
            rows=1,
            cols=len(tasks),
            subplot_titles=[f"{task.capitalize()} - {metric}" for task in tasks],
        )
        legend_shown = {
            "train": False,
            "validation": False,
            "test": False,
            "logistic": False,
            "ST-NN": False,
            "MT-NN": False,
        }

        for idx, task in enumerate(tasks):
            train_task_metrics = self.train_metrics.get(task, {})
            val_task_metrics = (
                self.val_metrics.get(task, {}) if self.val_metrics else {}
            )
            test_task_metrics = (
                self.test_metrics.get(task, {}) if self.test_metrics else {}
            )

            # Train metrics
            x_train = list(train_task_metrics.keys())
            y_train = [train_task_metrics[epoch][metric] for epoch in x_train]
            fig.add_trace(
                go.Scatter(
                    x=x_train,
                    y=y_train,
                    mode="lines+markers",
                    name="Train",
                    line=dict(color=self.colors["train"]),
                    legendgroup="train",
                    showlegend=not legend_shown["train"],
                ),
                row=1,
                col=idx + 1,
            )
            legend_shown["train"] = True

            # Validation metrics
            if val_task_metrics:
                x_val = list(val_task_metrics.keys())
                y_val = [val_task_metrics[epoch][metric] for epoch in x_val]
                fig.add_trace(
                    go.Scatter(
                        x=x_val,
                        y=y_val,
                        mode="lines+markers",
                        name="Validation",
                        line=dict(color=self.colors["validation"]),
                        legendgroup="validation",
                        showlegend=not legend_shown["validation"],
                    ),
                    row=1,
                    col=idx + 1,
                )
                legend_shown["validation"] = True

            # Test metrics
            if test_task_metrics:
                x_test = list(test_task_metrics.keys())
                y_test = [test_task_metrics[epoch][metric] for epoch in x_test]
                fig.add_trace(
                    go.Scatter(
                        x=x_test,
                        y=y_test,
                        mode="lines+markers",
                        name="Test",
                        line=dict(color=self.colors["test"]),
                        legendgroup="test",
                        showlegend=not legend_shown["test"],
                    ),
                    row=1,
                    col=idx + 1,
                )
                legend_shown["test"] = True

            # Multiple baselines
            for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
                if (
                    task in baseline_metrics.index
                    and metric in baseline_metrics.columns
                ):
                    baseline_value = baseline_metrics.loc[task, metric]
                    fig.add_trace(
                        go.Scatter(
                            x=x_train,  # Baseline is constant, use any x values (e.g., train epochs)
                            y=[baseline_value] * len(x_train),
                            mode="lines+markers",
                            name=baseline_name,
                            line=dict(color=self.baseline_color, dash="dot"),
                            marker=dict(
                                symbol=self.baseline_markers[baseline_name]["plotly"]
                            ),
                            legendgroup=f"baseline_{baseline_name}",
                            showlegend=not legend_shown[baseline_name],
                        ),
                        row=1,
                        col=idx + 1,
                    )
                    legend_shown[baseline_name] = True

        # Layout adjustments
        fig.update_layout(
            height=400,
            width=300 * len(tasks),
            title_text=f"Metric: {metric.capitalize()} Across Tasks",
            showlegend=True,
        )

        # Save or display the plot
        if save:
            self._save_figure_plotly(fig, save)
        if plot:
            fig.show()

    def plot_metric(
        self, metric: str, save: Optional[str] = None, plot: bool = True
    ) -> None:
        """
        Plots a specific metric with a subplot for each task using Matplotlib.

        Args:
            metric (str): The metric to plot ('accuracy', 'precision', 'recall', 'f1', 'loss').
            save (Optional[str]): Path to save the plot. Default is None.
            plot (bool): Whether to display the plot. Default is True.
        """
        tasks = self.tasks
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(5 * len(tasks), 4), sharey=True
        )

        if len(tasks) == 1:
            axes = [axes]  # Handle single subplot case

        # Track handles and labels for a single shared legend
        handles = []
        labels = []

        for ax, task in zip(axes, tasks):
            train_task_metrics = self.train_metrics.get(task, {})
            val_task_metrics = (
                self.val_metrics.get(task, {}) if self.val_metrics else {}
            )
            test_task_metrics = (
                self.test_metrics.get(task, {}) if self.test_metrics else {}
            )

            # Plot train metrics
            x_train = list(train_task_metrics.keys())
            y_train = [train_task_metrics[epoch][metric] for epoch in x_train]
            (train_line,) = ax.plot(
                x_train, y_train, label="Train", color=self.colors["train"]
            )
            if "Train" not in labels:
                handles.append(train_line)
                labels.append("Train")

            # Plot validation metrics
            if val_task_metrics:
                x_val = list(val_task_metrics.keys())
                y_val = [val_task_metrics[epoch][metric] for epoch in x_val]
                (val_line,) = ax.plot(
                    x_val, y_val, label="Validation", color=self.colors["validation"]
                )
                if "Validation" not in labels:
                    handles.append(val_line)
                    labels.append("Validation")

            # Plot test metrics
            if test_task_metrics:
                x_test = list(test_task_metrics.keys())
                y_test = [test_task_metrics[epoch][metric] for epoch in x_test]
                (test_line,) = ax.plot(
                    x_test, y_test, label="Test", color=self.colors["test"]
                )
                if "Test" not in labels:
                    handles.append(test_line)
                    labels.append("Test")

            # Plot multiple baselines
            for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
                if (
                    task in baseline_metrics.index
                    and metric in baseline_metrics.columns
                ):
                    baseline_value = baseline_metrics.loc[task, metric]
                    baseline_line = ax.plot(
                        x_train,  # Use train epochs for x-axis
                        [baseline_value] * len(x_train),
                        linestyle="dotted",
                        color=self.baseline_color,
                        marker=self.baseline_markers["matplotlib"][baseline_name],
                        label=baseline_name,
                    )[0]  # Retrieve the Line2D object
                    if baseline_name not in labels:
                        handles.append(baseline_line)
                        labels.append(baseline_name)

            # Configure subplot
            ax.set_title(f"{task.capitalize()} - {metric}")
            ax.set_xlabel("Epoch")
            ax.grid(True)

        # Configure shared ylabel and legend
        axes[0].set_ylabel(metric.capitalize())
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="upper center",
            ncol=4,
            fontsize="medium",
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout for legend

        # Save or display the plot
        if save:
            self._save_figure(fig, save)
        if plot:
            plt.show()
        else:
            plt.close(fig)


def plot_confusion_matrix(
    ground_truth,
    predictions,
    task_type="binary",
    task_name="",
    extra_title="",
    save_path=None,
):
    if task_type not in ["binary", "categorical"]:
        raise ValueError("task_type must be either 'binary' or 'categorical'.")

    # Compute the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100  # Compute percentages

    # Define labels for the confusion matrix
    if task_type == "binary":
        display_labels = ["Negative", "Positive"]
    else:  # For categorical tasks, assume unique labels from the ground truth
        display_labels = sorted(set(ground_truth))

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    # Set axis labels
    ax.set(
        xticks=np.arange(len(display_labels)),
        yticks=np.arange(len(display_labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        title=f"CM {task_name}  ({task_type.capitalize()} Task), {extra_title}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Annotate each cell with counts and percentages
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Tight layout to avoid clipping
    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def save_all_cm(Tracker, folder_path, dataset, extra_title=""):
    for task in Tracker.tasks:
        task_type = Tracker.tasks_types[task]
        if task_type != "regression":
            predictions = Tracker.predictions[task]
            ground_truth = Tracker.ground_truth[task]
            plot_confusion_matrix(
                ground_truth=ground_truth,
                predictions=predictions,
                task_name=task,
                task_type=task_type,
                save_path=os.path.join(folder_path, f"CM_{task}_{dataset}.png"),
                extra_title=extra_title,
            )


def plot_single_recon(
    original,
    reconstructed,
    ax=None,
    fig=None,
    show_axes=True,
    spacing=0.4,
    image_values_label="",
    titles=None,
):
    """
    Plots the original and reconstructed 2D arrays side-by-side with their difference on the provided axes.
    Places the colorbars next to each plot and optionally includes x and y axis notations.

    Args:
        original (np.ndarray or torch.Tensor): Original 2D array.
        reconstructed (np.ndarray or torch.Tensor): Reconstructed 2D array.
        ax (matplotlib.axes.Axes, optional): Axes on which to plot the images. If None, a new figure and axes will be created.
        fig (matplotlib.figure.Figure, optional): Figure object to which colorbars should be added. If None, a new figure will be created.
        show_axes (bool, optional): Whether to display the x and y axis notations (ticks and labels). Default is True.
        spacing (float, optional): Controls the horizontal spacing between plots inside the triplet. Default is 0.4.
        image_values_label (str, optional): Label for the colorbars representing the image values. Default is an empty string.
        titles (list of str, optional): Titles for the subplots. Should contain 3 titles: ["Original", "Reconstructed", "Difference"].

    Returns:
        tuple: The figure and axes objects for further manipulation or saving.

    Raises:
        ValueError: If the input arrays are not 2D or titles are incorrect.
    """
    # Ensure inputs are numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Validate input shapes
    if original.ndim != 2 or reconstructed.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    # Ensure titles are provided for each plot
    if titles is None:
        titles = ["Original", "Reconstructed", "Difference"]
    elif len(titles) != 3:
        raise ValueError(
            "Titles must be a list of 3 elements: ['Original', 'Reconstructed', 'Difference']"
        )

    # If ax and fig aren't provided, create them
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Adjust spacing between plots
    fig.subplots_adjust(wspace=spacing)

    # Compute global vmin and vmax
    vmin = min(original.min(), reconstructed.min())
    vmax = max(original.max(), reconstructed.max())

    # Plot original
    im1 = ax[0].imshow(original, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    if not show_axes:
        ax[0].axis("off")
    ax[0].set_title(titles[0])

    # Plot reconstructed
    im2 = ax[1].imshow(
        reconstructed, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto"
    )
    if not show_axes:
        ax[1].axis("off")
    ax[1].set_title(titles[1])

    # Plot difference
    diff = np.abs(original - reconstructed)
    im3 = ax[2].imshow(diff, cmap="Reds", vmin=0, aspect="auto")
    if not show_axes:
        ax[2].axis("off")
    ax[2].set_title(titles[2])

    # Colorbars for each subplot
    fig.colorbar(im1, ax=ax[0], shrink=0.8, label=image_values_label)
    fig.colorbar(im2, ax=ax[1], shrink=0.8, label=image_values_label)
    fig.colorbar(im3, ax=ax[2], shrink=0.8, label="difference")

    return (
        fig,
        ax,
    )  # Return the figure and axes in case you need to further manipulate them


def plot_grid_recon(
    originals,
    reconstructeds,
    max_rows=2,
    show_axes=True,
    spacing=0.4,
    save_path=None,
    title="",
    image_values_label="",
):
    """
    Plots a grid of original, reconstructed 2D arrays, and their differences using `plot_single_recon`.
    Each row contains one triplet: original, reconstructed, and difference.

    Args:
        originals (list of np.ndarray or torch.Tensor): List of original 2D arrays.
        reconstructeds (list of np.ndarray or torch.Tensor): List of reconstructed 2D arrays.
        max_rows (int, optional): Maximum number of rows to display in the grid. Default is 2.
        show_axes (bool, optional): Whether to display x and y axis notations. Default is True.
        spacing (float, optional): Controls the horizontal spacing between plots inside each triplet. Default is 0.4.
        save_path (str, optional): If provided, saves the figure to this path instead of displaying. Default is None.
        title (str, optional): Title for the entire plot grid. Default is an empty string.
        image_values_label (str, optional): Label for the colorbars representing the image values. Default is an empty string.

    Raises:
        ValueError: If lengths of `originals` and `reconstructeds` do not match or the arrays are not 2D.

    Returns:
        None: The function either displays or saves the figure, depending on the `save_path` parameter.
    """
    if len(originals) != len(reconstructeds):
        raise ValueError("Lengths of originals and reconstructeds must match.")

    # Limit number of rows to max_rows
    num_triplets = min(len(originals), max_rows)

    # Create a figure with subplots (one row per triplet)
    fig, axs = plt.subplots(num_triplets, 3, figsize=(12, 4 * num_triplets))

    # If there is only one row, axs won't be 2D, so we adjust it to make indexing easier
    if num_triplets == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, (original, reconstructed) in enumerate(
        zip(originals[:num_triplets], reconstructeds[:num_triplets])
    ):
        # Generate titles dynamically
        mse = torch.mean((original.detach().cpu() - reconstructed.detach().cpu()) ** 2)
        titles = [
            f"Original {idx + 1}",
            f"Reconstructed {idx + 1}",
            f"Difference {idx + 1} \n MSE = {mse:.3f}",
        ]

        # Call `plot_single_recon` to plot each triplet in the row
        plot_single_recon(
            original,
            reconstructed,
            ax=axs[idx],
            fig=fig,
            show_axes=show_axes,
            spacing=spacing,
            image_values_label=image_values_label,
            titles=titles,
        )

    # Set the overall title if provided
    if title:
        fig.suptitle(title)

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_gradients(
    gradients,
    clipped_gradients,
    total_epochs,
    batches_per_epoch,
    save_path=None,
    max_ticks=40,
):
    """
    Plot gradients and clipped gradients with a double x-axis.

    Parameters:
    - gradients: List of gradients.
    - clipped_gradients: List of clipped gradients.
    - total_epochs: Total number of epochs.
    - batches_per_epoch: Number of batches (training steps) per epoch.
    - save_path: (Optional) Path to save the figure. If None, the plot will not be saved.

    The first x-axis will represent training steps, and the second x-axis will represent epochs.
    """
    # Calculate the number of training steps
    training_steps = np.arange(len(gradients))

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot gradients on the primary x-axis
    ax1.plot(training_steps, gradients, label="Gradients", color="blue", alpha=0.7)
    ax1.plot(
        training_steps,
        clipped_gradients,
        label="Clipped Gradients",
        color="red",
        alpha=0.7,
    )

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Gradient Magnitude (norm p=2)")
    ax1.set_title("Gradients and Clipped Gradients over Training Steps and Epochs")

    # Create secondary x-axis for epochs
    ax2 = ax1.twiny()

    # Determine the number of ticks to show (e.g., show every 5th epoch)
    num_epoch_ticks = min(max_ticks, total_epochs)  # Show max 10 ticks
    epoch_tick_indices = np.linspace(0, len(gradients) - 1, num_epoch_ticks).astype(int)

    ax2.set_xlim(ax1.get_xlim())  # Ensure both axes have the same range
    ax2.set_xticks(epoch_tick_indices)

    # Epoch numbers start from 1, so we adjust the tick labels accordingly
    epoch_labels = [
        f"{int(i / len(gradients) * total_epochs) + 1}" for i in epoch_tick_indices
    ]
    ax2.set_xticklabels(epoch_labels, rotation=45, ha="left")

    ax2.set_xlabel("Epochs")

    # Add a legend
    ax1.legend(loc="upper left")

    # Show the plot
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close(fig)  # Close the figure to prevent it from being shown

    else:
        # Show the plot if no save path is provided
        plt.tight_layout()
        plt.show()
