import json
import os
import uuid
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
import shutil
from .plotting import plot_gradients, save_all_cm


def remove_non_complete_runs(base_directory=""):
    """
    Removes folders inside the base directory that do not contain either a 'model.pth' or 'model.pt' file.
    Args:
        base_directory (str): The path to the 'model_runs' directory.
    """
    if not os.path.exists(base_directory):
        print(f"Directory '{base_directory}' does not exist.")
        return

    for folder in tqdm(os.listdir(base_directory)):
        folder_path = os.path.join(base_directory, folder)

        if os.path.isdir(folder_path):
            # Check if either 'model.pth' or 'model.pt' exists in the folder
            model_pth_path = os.path.join(folder_path, "model.pth")
            model_pt_path = os.path.join(folder_path, "model.pt")

            if not (os.path.isfile(model_pth_path) or os.path.isfile(model_pt_path)):
                # Delete the folder if neither file is found
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
        else:
            print(f"Skipping non-folder item: {folder_path}")

    print("Cleanup completed.")


class RunTracker:
    def __init__(self, base_dir: str):
        """
        Initialize the RunTracker class.

        Args:
        - base_dir (str): The base directory where all runs will be stored.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.index_file = os.path.join(self.base_dir, "runs_index.json")
        self.runs_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load or create the index file."""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the index to a file."""
        with open(self.index_file, "w") as f:
            # print(self.runs_index)
            json.dump(self.runs_index, f, indent=4)

    def create_run(self, description: str, hparams: Dict, run_id: Optional[str] = None):
        """
        Create a new run.

        Args:
        - description (str): A short description of the run.
        - hparams (Dict): A dictionary of hyperparameters for the run.
        - run_id (str): Custom run ID. If None, a UUID will be generated.

        Returns:
        - str: The unique ID of the created run.
        """

        run_id = run_id if run_id is not None else str(uuid.uuid4())
        run_folder = os.path.join(self.base_dir, run_id)
        subfolders = ['plots', 'checkpoints', 'metrics']
        folders = {name: os.path.join(run_folder, name) for name in subfolders}
        
        os.makedirs(run_folder, exist_ok=True)
        for folder in subfolders:
            os.makedirs(os.path.join(run_folder, folder), exist_ok=True)

        # Save description and hyperparameters

            
        if 'loss_fn_hyper_parameters' in hparams:
            
            hparams['loss_fn_hyper_parameters'] = {
                key: {
                    **value,
                    'weight': value['weight'].tolist() if isinstance(value['weight'], (np.ndarray, torch.Tensor)) else value['weight']
                }
                for key, value in hparams['loss_fn_hyper_parameters'].items()
            }
        # if ('class_weights' in hparams) and hparams['class_weights'] is not None:
            
        #     hparams['class_weights'] = hparams['class_weights'].tolist()
            
     

        metadata = {
            "description": description,
            "hparams": hparams,
            "run_folder": run_folder,
            **folders,
            
        }
        self.runs_index[run_id] = metadata

        self._save_index()

        # Save metadata to the run folder
        with open(os.path.join(run_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return run_id

    def search_runs(self, keyword: Optional[str] = None) -> List[Dict]:
        """
        Search for runs by keyword in the description.

        Args:
        - keyword (str): The keyword to search for in descriptions.

        Returns:
        - List[Dict]: A list of runs matching the keyword.
        """
        if not keyword:
            return list(self.runs_index.values())
        
        return [
            run for run in self.runs_index.values()
            if keyword.lower() in run["description"].lower()
        ]

    def get_run(self, run_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific run.

        Args:
        - run_id (str): The unique ID of the run.

        Returns:
        - Dict: Metadata of the run, or None if not found.
        """
        return self.runs_index[run_id]

    def save_text_to_run(self, run_id: str, filename: str, content: str):
        """
        Save a file to a specific run's folder.

        Args:
        - run_id (str): The unique ID of the run.
        - filename (str): The name of the file to save.
        - content (str): The content to write to the file.
        """
        run_metadata = self.get_run(run_id)
        if not run_metadata:
            raise ValueError(f"Run ID {run_id} does not exist.")

        run_folder = run_metadata["run_folder"]
        file_path = os.path.join(run_folder, filename)
        with open(file_path, "w") as f:
            f.write(content)

    def _clean_up_index(self):
        """
        Remove runs from the index if their directories no longer exist.
        """
        keys_to_remove = [
            run_id for run_id, metadata in self.runs_index.items()
            if not os.path.exists(metadata["run_folder"])
        ]
        for run_id in keys_to_remove:
            del self.runs_index[run_id]
        self._save_index()

    def cleanup(self):
        remove_non_complete_runs(self.base_dir)
        self._clean_up_index()

    def gather_metrics(self, file_name: str = None, run_ids: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Gather metrics DataFrames from all or specified runs.

        Args:
        - file_name (str, optional): Name of the metric file to look for (e.g., 'train', 'test').
                                     If None, will gather all CSV files.
        - run_ids (List[str], optional): List of specific run IDs to gather metrics from.
                                       If None, will gather from all runs.

        Returns:
        - Dict[str, pd.DataFrame]: Dictionary mapping run IDs to their respective metrics DataFrames
        """
        files_dict = {}
        runs_to_process = run_ids if run_ids is not None else self.runs_index.keys()

        for run_id in runs_to_process:
            run_metadata = self.get_run(run_id)
            if not run_metadata:
                continue

            metrics_folder = run_metadata.get('metrics')
            if not metrics_folder or not os.path.exists(metrics_folder):
                continue

            # Get all CSV files in the metrics folder
            metric_files = [f for f in os.listdir(metrics_folder) if f.endswith('.csv')]
            
            for metric_file in metric_files:
                # If file_name is specified, only process matching files
                if file_name and not metric_file.startswith(f"{file_name}"):
                    continue
                
                file_path = os.path.join(metrics_folder, metric_file)
                try:
                    df = pd.read_csv(file_path)
                    files_dict[f"{run_id}_{metric_file[:-4]}"] = df
                except Exception as e:
                    print(f"Error reading metrics file {file_path}: {str(e)}")

        return files_dict

    def process_metrics(self, 
                       metrics_dict: Dict[str, pd.DataFrame],
                       columns: List[str] = None,
                       row_range: tuple = None,
                       last_row_only: bool = False,
                       aggregation: str = None,
                       filters: Dict = None,
                       sort_by: str = None,
                       ascending: bool = True,
                       as_df: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Process gathered metrics DataFrames with various operations.

        Args:
        - metrics_dict (Dict[str, pd.DataFrame]): Dictionary of metrics DataFrames from gather_metrics
        - columns (List[str], optional): Specific columns to include
        - row_range (tuple, optional): Range of rows to include (start, end)
        - last_row_only (bool): If True, only returns the last row of each DataFrame
        - aggregation (str, optional): Aggregation function to apply ('mean', 'sum', 'max', 'min')
        - filters (Dict, optional): Dictionary of column:value pairs to filter rows
        - sort_by (str, optional): Column name to sort by
        - ascending (bool): Sort order if sort_by is specified
        - as_df (bool): to return a dict or df

        Returns:
        - Dict[str, pd.DataFrame] or pd.DataFrame: Processed metrics DataFrames
        """
        processed_dict = {}

        for run_id, df in metrics_dict.items():
            # Make a copy to avoid modifying original
            processed_df = df.copy()

            # Apply column selection
            if columns:
                available_cols = [col for col in columns if col in processed_df.columns]
                processed_df = processed_df[available_cols]

            # Apply row range
            if row_range and not last_row_only:  # Skip if last_row_only is True
                start, end = row_range
                processed_df = processed_df.iloc[start:end]

            # Apply last row selection (takes precedence over row_range)
            if last_row_only:
                processed_df = processed_df.iloc[[-1]]

            # Apply filters
            if filters:
                for col, value in filters.items():
                    if col in processed_df.columns:
                        if isinstance(value, (list, tuple)):
                            processed_df = processed_df[processed_df[col].isin(value)]
                        else:
                            processed_df = processed_df[processed_df[col] == value]

            # Apply aggregation (only if not getting last row)
            if aggregation and not last_row_only:
                agg_map = {
                    'mean': processed_df.mean(),
                    'sum': processed_df.sum(),
                    'max': processed_df.max(),
                    'min': processed_df.min()
                }
                if aggregation.lower() in agg_map:
                    processed_df = pd.DataFrame([agg_map[aggregation.lower()]])

            # Apply sorting
            if sort_by and sort_by in processed_df.columns:
                processed_df = processed_df.sort_values(by=sort_by, ascending=ascending)

            processed_dict[run_id] = processed_df
        if as_df:
            dfs = []
            for key, df in processed_dict.items():
                df = df.copy()  # To avoid modifying the original
                df['Category'] = key  # Add the dictionary key as a new column
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        return processed_dict

    def get_metric_summary(self, metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a summary of metrics across all runs.

        Args:
        - metrics_dict (Dict[str, pd.DataFrame]): Dictionary of metrics DataFrames

        Returns:
        - pd.DataFrame: Summary DataFrame with basic statistics for each metric
        """
        summaries = []
        
        for run_id, df in metrics_dict.items():
            summary = df.describe()
            summary['run_id'] = run_id
            summaries.append(summary)
        
        return pd.concat(summaries, axis=0)

    def gather_and_display_plots(self, 
                               task_name: str, 
                               run_ids: List[str] = None,
                               figsize: tuple = (15, 10)) -> None:
        """
        Gather and display plots for a specific task across multiple runs.

        Args:
        - task_name (str): Name of the task (e.g., 'Reconstruction')
        - run_ids (List[str], optional): List of specific run IDs to gather plots from.
                                       If None, will gather from all runs.
        - figsize (tuple): Figure size for the combined plot display
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np

        runs_to_process = run_ids if run_ids is not None else self.runs_index.keys()
        plot_images = []
        run_labels = []

        for run_id in runs_to_process:
            run_metadata = self.get_run(run_id)
            if not run_metadata:
                continue

            plots_folder = run_metadata.get('plots')
            if not plots_folder or not os.path.exists(plots_folder):
                continue

            # Look for matching plot file
            plot_filename = f"{task_name}_metrics.png"
            plot_path = os.path.join(plots_folder, plot_filename)

            if os.path.exists(plot_path):
                try:
                    # Read the image
                    img = Image.open(plot_path)
                    plot_images.append(np.array(img))
                    run_labels.append(run_id)
                except Exception as e:
                    print(f"Error reading plot file {plot_path}: {str(e)}")

        if not plot_images:
            print(f"No plots found for task: {task_name}")
            return

        # Calculate grid dimensions
        n_plots = len(plot_images)
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

        # Plot each image
        for idx, (img, run_id) in enumerate(zip(plot_images, run_labels)):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Run: {run_id}")
            axes[row, col].axis('off')

        # Turn off any empty subplots
        for idx in range(len(plot_images), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        return fig

def save_training_results(
    trainer, tracker, metrics_tracker_tr, metrics_tracker_test, model, plotter
):
    """
    Save training results including metrics, plots, and the model.

    Args:
        trainer: Trainer object containing run_id.
        tracker: Tracker object with run indexing information.
        metrics_tracker_tr: Metrics tracker object for training data.
        metrics_tracker_test: Metrics tracker object for test data.
        model: Trained PyTorch model to save.
        plotter: Plotter class

    Returns:
        None
    """
    # Retrieve folders for saving results
    run_id = trainer.run_id
    plots_folder = tracker.runs_index[run_id]["plots"]
    metrics_folder = tracker.runs_index[run_id]["metrics"]
    run_folder = tracker.runs_index[run_id]["run_folder"]

    # Save metrics as CSV
    metrics_tracker_tr.dict_to_dataframe(os.path.join(metrics_folder, "train.csv"))
    metrics_tracker_test.dict_to_dataframe(os.path.join(metrics_folder, "test.csv"))

    # Initialize plotter and save plots
    plotter.save_all_plots(save_dir=plots_folder)

    # Save confusion matrices
    save_all_cm(metrics_tracker_test, plots_folder, dataset="Test", extra_title="")
    # Save gradients plot:

    plot_gradients(
        trainer.grad_norms,
        trainer.grad_norms_clipped,
        total_epochs=trainer.hyperparams["num_epochs"],
        batches_per_epoch=len(trainer.train_dataloader),
        save_path=os.path.join(plots_folder, "gradients_plot.png"),
    )
    # Save the trained model
    torch.save(model, os.path.join(run_folder, "model.pt"))

    print(f"Training results saved in {run_folder}")
