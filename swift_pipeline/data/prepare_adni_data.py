"""
Data preparation utilities for ADNI dataset with SwiFT pipeline
Handles data loading, preprocessing, and dataset creation
"""

import torch
import numpy as np
import json
import sys
import os
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from pathlib import Path

# Add path to access transforms from the parent adni_code/data directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'))
from transforms import Resize3D

from .dataset_adni import (
    ADNISwiFTPretrainDataset,
    ADNISwiFTFinetuneDataset,
    load_adni_data,
)
from .preprocessing import preprocess_scan


class ADNIDataSplitter:
    """Split ADNI data into train/val/test sets"""

    def __init__(
        self,
        data: torch.Tensor,
        index_to_info: Dict,
        val_split: float = 0.1,
        test_split: float = 0.2,
        seed: int = 42,
    ):
        self.data = data
        self.index_to_info = index_to_info
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Group indices by subject_id to avoid data leakage
        self.subject_to_indices = {}
        for idx, info in index_to_info.items():
            subject_id = info.get("subject_id", info.get("image_id"))
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)

        self.subject_ids = list(self.subject_to_indices.keys())

    def split(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Split data by subjects to avoid data leakage

        Returns:
            train_indices, val_indices, test_indices
        """
        # Shuffle subjects
        np.random.seed(self.seed)
        shuffled_subjects = np.random.permutation(self.subject_ids)

        # Calculate split sizes
        n_subjects = len(shuffled_subjects)
        n_test = int(n_subjects * self.test_split)
        n_val = int(n_subjects * self.val_split)
        n_train = n_subjects - n_test - n_val

        # Split subjects
        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train : n_train + n_val]
        test_subjects = shuffled_subjects[n_train + n_val :]

        # Get all indices for each split
        train_indices = []
        val_indices = []
        test_indices = []

        for subject in train_subjects:
            train_indices.extend(self.subject_to_indices[subject])

        for subject in val_subjects:
            val_indices.extend(self.subject_to_indices[subject])

        for subject in test_subjects:
            test_indices.extend(self.subject_to_indices[subject])

        print(f"\nData split by subjects (no data leakage):")
        print(f"  Train: {n_train} subjects, {len(train_indices)} samples")
        print(f"  Val:   {n_val} subjects, {len(val_indices)} samples")
        print(f"  Test:  {n_test} subjects, {len(test_indices)} samples")

        return train_indices, val_indices, test_indices


def prepare_adni_datasets(
    data_path: str,
    labels_path: str,
    info_path: str,
    task_names: List[str],
    stage: str = "finetune",
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
    target_spatial_size: Tuple[int, int, int] = (35, 37, 35),
    scale_factor = None,
    window_size: int = 10,
    stride: int = 5,
    normalize: bool = True,
    handle_nan: str = "skip",
) -> Dict:
    """
    Prepare ADNI datasets for SwiFT pipeline

    Args:
        data_path: Path to preprocessed tensor data
        labels_path: Path to imageID_to_labels.json
        info_path: Path to index_to_name.json
        task_names: List of task names (e.g., ['degradation_binary_1year'])
        stage: 'pretrain' or 'finetune'
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
        target_spatial_size: Target spatial size after Resize3D transform
        scale_factor: Scale factor for Resize3D (default: 0.7 for 35x37x35)
        window_size: Temporal window size (default: 10)
        stride: Temporal stride (default: 5)
        normalize: Whether to normalize data
        handle_nan: How to handle NaN labels - 'skip', 'zero', or 'keep'

    Returns:
        Dictionary containing datasets and metadata
    """
    print("=" * 80)
    print("PREPARING ADNI DATA FOR SWIFT PIPELINE")
    print("=" * 80)

    # Load data
    data, imageID_to_labels, index_to_info = load_adni_data(
        data_path, labels_path, info_path
    )

    print(f"\nLoaded data shape: {data.shape}")
    print(f"Expected format: [N_scans, H, W, D, T]")

    # Apply spatial resizing using Resize3D
    if scale_factor is not None:
        
        print(f"\nApplying spatial resize with scale_factor={scale_factor}:")
        resize_transform = Resize3D(scale_factor=scale_factor, align_corners=False)
    
        resized_scans = []
        for i in range(data.shape[0]):
            scan = data[i:i+1]  # [1, H, W, D, T]
            resized_scan = resize_transform(scan)
            resized_scans.append(resized_scan)
        
        data = torch.cat(resized_scans, dim=0)  # [N_scans, H_new, W_new, D_new, T]
    print(f"  Resized data shape: {data.shape}")
    print(f"  Target spatial size: {target_spatial_size}")

    # Create temporal windows from full scans
    print(f"\nCreating temporal windows:")
    print(f"  - Window size: {window_size}")
    print(f"  - Stride: {stride}")

    windowed_data = []
    windowed_info = {}
    window_idx_counter = 0

    for scan_idx in range(data.shape[0]):
        scan = data[scan_idx]  # [H, W, D, T]
        temporal_length = scan.shape[-1]

        # Get original scan info
        original_info = index_to_info[scan_idx]
        image_id = original_info["image_id"]

        # Create windows for this scan
        num_windows = (temporal_length - window_size) // stride + 1

        for win_idx in range(num_windows):
            start_t = win_idx * stride
            end_t = start_t + window_size

            # Extract window
            window = scan[:, :, :, start_t:end_t]  # [H, W, D, window_size]
            windowed_data.append(window)

            # Create window info
            window_info = original_info.copy()
            window_info["window_index"] = win_idx
            window_info["start_time"] = start_t
            window_info["end_time"] = end_t
            window_info["original_scan_index"] = scan_idx

            windowed_info[window_idx_counter] = window_info
            window_idx_counter += 1

    # Stack all windows
    windowed_data = torch.stack(
        windowed_data, dim=0
    )  # [N_windows, H, W, D, window_size]

    print(f"✓ Created {len(windowed_data)} windows from {data.shape[0]} scans")
    print(f"  - Windowed data shape: {windowed_data.shape}")
    print(f"  - Windows per scan: ~{len(windowed_data) // data.shape[0]}")

    # Split data by subjects (using windowed data now)
    splitter = ADNIDataSplitter(
        windowed_data,
        windowed_info,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    train_indices, val_indices, test_indices = splitter.split()

    # Split data
    train_data = windowed_data[train_indices]
    val_data = windowed_data[val_indices]
    test_data = windowed_data[test_indices]

    # Split index_to_info
    train_info = {i: windowed_info[idx] for i, idx in enumerate(train_indices)}
    val_info = {i: windowed_info[idx] for i, idx in enumerate(val_indices)}
    test_info = {i: windowed_info[idx] for i, idx in enumerate(test_indices)}

    print(f"\nData shapes:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val:   {val_data.shape}")
    print(f"  Test:  {test_data.shape}")

    # Preprocess data (data is now in windowed format [N_windows, H, W, D, window_size])
    # No additional preprocessing needed as windows are already created

    # Create metadata for padding
    metadata = {
        "background_value": 0.0,  # Will be computed per scan if needed
        "normalize": normalize,
    }

    result = {
        "imageID_to_labels": imageID_to_labels,
        "train_info": train_info,
        "val_info": val_info,
        "test_info": test_info,
    }

    if stage == "pretrain":
        print(f"\n{'=' * 80}")
        print("CREATING CONTRASTIVE PRETRAINING DATASETS")
        print(f"{'=' * 80}")

        train_dataset = ADNISwiFTPretrainDataset(
            train_data,
            train_info,
            window_size=window_size,  # Not used for windowing, just for reference
            stride=stride,  # Not used for windowing, just for reference
            metadata=metadata,
            target_spatial_size=target_spatial_size,
        )

        val_dataset = ADNISwiFTPretrainDataset(
            val_data,
            val_info,
            window_size=window_size,
            stride=stride,
            metadata=metadata,
            target_spatial_size=target_spatial_size,
        )

        result.update(
            {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
            }
        )

    elif stage == "finetune":
        print(f"\n{'=' * 80}")
        print("CREATING SUPERVISED FINETUNING DATASETS")
        print(f"{'=' * 80}")

        train_dataset = ADNISwiFTFinetuneDataset(
            train_data,
            train_info,
            imageID_to_labels,
            task_names,
            metadata=metadata,
            target_spatial_size=target_spatial_size,
            handle_nan=handle_nan,
        )

        val_dataset = ADNISwiFTFinetuneDataset(
            val_data,
            val_info,
            imageID_to_labels,
            task_names,
            metadata=metadata,
            target_spatial_size=target_spatial_size,
            handle_nan=handle_nan,
        )

        test_dataset = ADNISwiFTFinetuneDataset(
            test_data,
            test_info,
            imageID_to_labels,
            task_names,
            metadata=metadata,
            target_spatial_size=target_spatial_size,
            handle_nan=handle_nan,
        )

        result.update(
            {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
            }
        )

    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'pretrain' or 'finetune'")

    print(f"\n{'=' * 80}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 80}\n")

    return result


def create_adni_dataloaders(
    datasets_dict: Dict,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Dict:
    """
    Create DataLoaders from datasets

    Args:
        datasets_dict: Dictionary containing datasets
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Dictionary containing dataloaders
    """
    result = {}

    if "train_dataset" in datasets_dict:
        result["train_loader"] = DataLoader(
            datasets_dict["train_dataset"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )

    if "val_dataset" in datasets_dict:
        result["val_loader"] = DataLoader(
            datasets_dict["val_dataset"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    if "test_dataset" in datasets_dict:
        result["test_loader"] = DataLoader(
            datasets_dict["test_dataset"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return result


if __name__ == "__main__":
    # Example usage
    print("Example: Prepare ADNI data for degradation task")
    print("=" * 80)

    # Example paths (adjust to your actual paths)
    data_path = "../../data/all_4d_downsampled.pt"
    labels_path = "../../imageID_to_labels.json"
    info_path = "../../index_to_name.json"

    # Prepare for finetuning on degradation task
    datasets = prepare_adni_datasets(
        data_path=data_path,
        labels_path=labels_path,
        info_path=info_path,
        task_names=["degradation_binary_1year"],
        stage="finetune",
        val_split=0.1,
        test_split=0.2,
        seed=42,
    )

    print("\nDatasets created:")
    for key in datasets:
        if "dataset" in key:
            print(f"  - {key}: {len(datasets[key])} samples")
