"""
ADNI-specific dataset classes for SwiFT pipeline
Integrates with existing ADNI data structure (imageID_to_labels, index_to_info)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List
import random
import json


def resize_to_target_size(
    data: torch.Tensor,
    target_size: Tuple[int, int, int] = (96, 96, 96),
    background_value: float = 0.0,
) -> torch.Tensor:
    """
    Resize data to target spatial size using padding and/or cropping

    Args:
        data: Input tensor of shape [C, H, W, D, T]
        target_size: Target spatial dimensions (H, W, D)
        background_value: Value to use for padding

    Returns:
        Resized tensor of shape [C, target_H, target_W, target_D, T]
    """
    C, H, W, D, T = data.shape
    target_H, target_W, target_D = target_size

    # Handle each dimension: crop if larger, pad if smaller
    result = data

    # Process H dimension
    if H > target_H:
        # Center crop
        start_h = (H - target_H) // 2
        result = result[:, start_h : start_h + target_H, :, :, :]
    elif H < target_H:
        # Center pad
        pad_h = target_H - H
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        result = F.pad(
            result,
            (0, 0, 0, 0, 0, 0, pad_h_top, pad_h_bottom),
            mode="constant",
            value=background_value,
        )

    # Process W dimension
    _, H_new, W_new, D_new, T_new = result.shape
    if W_new > target_W:
        # Center crop
        start_w = (W_new - target_W) // 2
        result = result[:, :, start_w : start_w + target_W, :, :]
    elif W_new < target_W:
        # Center pad
        pad_w = target_W - W_new
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
        result = F.pad(
            result,
            (0, 0, 0, 0, pad_w_left, pad_w_right, 0, 0),
            mode="constant",
            value=background_value,
        )

    # Process D dimension
    _, H_new, W_new, D_new, T_new = result.shape
    if D_new > target_D:
        # Center crop
        start_d = (D_new - target_D) // 2
        result = result[:, :, :, start_d : start_d + target_D, :]
    elif D_new < target_D:
        # Center pad
        pad_d = target_D - D_new
        pad_d_front = pad_d // 2
        pad_d_back = pad_d - pad_d_front
        result = F.pad(
            result,
            (0, 0, pad_d_front, pad_d_back, 0, 0, 0, 0),
            mode="constant",
            value=background_value,
        )

    return result


class ADNISwiFTPretrainDataset(Dataset):
    """
    ADNI dataset for contrastive pretraining with SwiFT
    Returns two random non-overlapping temporal windows for contrastive learning

    Integrates with existing ADNI data structure:
    - Uses index_to_info for scan metadata
    - No labels needed (self-supervised)
    """

    def __init__(
        self,
        data: torch.Tensor,
        index_to_info: Dict,
        window_size: int = 20,
        stride: int = 10,
        metadata: Optional[Dict] = None,
        target_spatial_size: Tuple[int, int, int] = (96, 96, 96),
        augmentation: Optional[Callable] = None,
    ):
        """
        Args:
            data: Preprocessed data [N, H, W, D, T] (already windowed)
            index_to_info: Dict mapping {idx: {'image_id': ..., 'window_index': ...}}
            window_size: Size of temporal window
            stride: Stride for temporal windows
            metadata: Dict with 'background_value' for padding
            target_spatial_size: Target size for padding (default: 96x96x96)
            augmentation: Optional augmentation function
        """
        self.data = data
        self.index_to_info = index_to_info
        self.window_size = window_size
        self.stride = stride
        self.metadata = metadata or {}
        self.target_spatial_size = target_spatial_size
        self.augmentation = augmentation

        # Get background value for padding
        self.background_value = self.metadata.get("background_value", 0.0)

        # Check if resizing is needed
        current_size = data.shape[1:4]  # [H, W, D]
        self.needs_padding = any(
            current_size[i] != target_spatial_size[i] for i in range(3)
        )

        if self.needs_padding:
            print(f"  Dataset will resize from {current_size} to {target_spatial_size}")
            print(f"  Using background value: {self.background_value:.6f}")

        # Group windows by original scan (image_id)
        self.scan_to_windows = {}
        for idx, info in index_to_info.items():
            image_id = info["image_id"]
            if image_id not in self.scan_to_windows:
                self.scan_to_windows[image_id] = []
            self.scan_to_windows[image_id].append(idx)

        # Create list of valid samples (need at least 2 windows per scan)
        self.valid_samples = []
        for image_id, window_indices in self.scan_to_windows.items():
            if len(window_indices) >= 2:
                for window_idx in window_indices:
                    self.valid_samples.append((image_id, window_idx))

        print(
            f"Contrastive dataset: {len(self.valid_samples)} valid samples from {len(self.scan_to_windows)} scans"
        )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        image_id, window_idx = self.valid_samples[idx]

        # Get first view - add channel dimension [H, W, D, T] -> [1, H, W, D, T]
        view1 = self.data[window_idx].unsqueeze(0)

        # Resize if needed
        if self.needs_padding:
            view1 = resize_to_target_size(
                view1, self.target_spatial_size, self.background_value
            )

        # Find all windows from same scan, excluding overlapping windows
        available_windows = []
        window_info1 = self.index_to_info[window_idx]
        window_idx1 = window_info1.get("window_index", 0)

        for other_idx in self.scan_to_windows[image_id]:
            if other_idx == window_idx:
                continue

            other_info = self.index_to_info[other_idx]
            other_window_idx = other_info.get("window_index", 0)

            # Check if windows overlap (based on window_index)
            if abs(window_idx1 - other_window_idx) >= 2:  # Non-adjacent windows
                available_windows.append(other_idx)

        # If no non-overlapping windows available, use any different window
        if not available_windows:
            available_windows = [
                idx for idx in self.scan_to_windows[image_id] if idx != window_idx
            ]

        # Sample second view
        if available_windows:
            view2_idx = random.choice(available_windows)
            view2 = self.data[view2_idx].unsqueeze(0)

            # Resize if needed
            if self.needs_padding:
                view2 = resize_to_target_size(
                    view2, self.target_spatial_size, self.background_value
                )
        else:
            # Fallback: use the same window (should rarely happen)
            view2 = view1.clone()

        # Apply augmentation if provided
        if self.augmentation is not None:
            view1 = self.augmentation(view1)
            view2 = self.augmentation(view2)

        return view1, view2


class ADNISwiFTFinetuneDataset(Dataset):
    """
    ADNI dataset for supervised fine-tuning with SwiFT
    Returns windows with labels from imageID_to_labels

    Supports degradation tasks and other ADNI labels
    """

    def __init__(
        self,
        data: torch.Tensor,
        index_to_info: Dict,
        imageID_to_labels: Dict,
        task_names: List[str],
        metadata: Optional[Dict] = None,
        target_spatial_size: Tuple[int, int, int] = (96, 96, 96),
        augmentation: Optional[Callable] = None,
        handle_nan: str = "skip",  # 'skip', 'zero', or 'keep'
    ):
        """
        Args:
            data: Preprocessed data [N, H, W, D, T] (already windowed)
            index_to_info: Dict mapping {idx: {'image_id': ..., 'window_index': ...}}
            imageID_to_labels: Dict mapping {image_id: {task: label, ...}}
            task_names: List of task names to use (e.g., ['degradation_binary_1year'])
            metadata: Dict with 'background_value' for padding
            target_spatial_size: Target size for padding (default: 96x96x96)
            augmentation: Optional augmentation function
            handle_nan: How to handle NaN labels - 'skip', 'zero', or 'keep'
        """
        self.data = data
        self.index_to_info = index_to_info
        self.imageID_to_labels = imageID_to_labels
        self.task_names = task_names
        self.metadata = metadata or {}
        self.target_spatial_size = target_spatial_size
        self.augmentation = augmentation
        self.handle_nan = handle_nan

        # Get background value for padding
        self.background_value = self.metadata.get("background_value", 0.0)

        # Check if resizing is needed
        current_size = data.shape[1:4]  # [H, W, D]
        self.needs_padding = any(
            current_size[i] != target_spatial_size[i] for i in range(3)
        )

        if self.needs_padding:
            print(f"  Dataset will resize from {current_size} to {target_spatial_size}")
            print(f"  Using background value: {self.background_value:.6f}")

        # Filter valid samples (those with labels and non-NaN if needed)
        self.valid_indices = []
        self.labels = []

        nan_count = 0
        missing_count = 0

        for idx, info in index_to_info.items():
            image_id = info["image_id"]

            # Check if image_id has labels
            if image_id not in imageID_to_labels:
                missing_count += 1
                continue

            labels_dict = imageID_to_labels[image_id]

            # Extract labels for specified tasks
            sample_labels = []
            has_nan = False

            for task in task_names:
                if task not in labels_dict:
                    label = float("nan")
                    has_nan = True
                else:
                    label = labels_dict[task]
                    # Check for NaN values
                    if isinstance(label, float) and np.isnan(label):
                        has_nan = True

                sample_labels.append(label)

            # Handle NaN based on strategy
            if has_nan:
                nan_count += 1
                if self.handle_nan == "skip":
                    continue
                elif self.handle_nan == "zero":
                    sample_labels = [0.0 if np.isnan(x) else x for x in sample_labels]

            self.valid_indices.append(idx)
            self.labels.append(sample_labels)

        print(f"Fine-tuning dataset: {len(self.valid_indices)} valid samples")
        print(f"  - Tasks: {task_names}")
        print(f"  - Skipped {nan_count} samples with NaN labels")
        print(f"  - Skipped {missing_count} samples without labels")

        # Convert labels to tensor
        if len(self.labels) > 0:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # Print label statistics
            for i, task in enumerate(task_names):
                task_labels = self.labels[:, i]
                if not torch.isnan(task_labels).all():
                    unique_labels = torch.unique(task_labels[~torch.isnan(task_labels)])
                    print(f"  - {task}: {len(unique_labels)} unique values")
                    if len(unique_labels) <= 10:
                        print(
                            f"    Distribution: {dict(zip(unique_labels.tolist(), [(task_labels == l).sum().item() for l in unique_labels]))}"
                        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]

        # Get data - add channel dimension [H, W, D, T] -> [1, H, W, D, T]
        x = self.data[data_idx].unsqueeze(0)

        # Resize if needed
        if self.needs_padding:
            x = resize_to_target_size(
                x, self.target_spatial_size, self.background_value
            )

        # Get labels
        y = self.labels[idx]

        # Apply augmentation if provided (typically only during training)
        if self.augmentation is not None:
            x = self.augmentation(x)

        return x, y


def load_adni_data(
    data_path: str,
    labels_path: str,
    info_path: str,
) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    Load ADNI data files

    Args:
        data_path: Path to preprocessed tensor data (e.g., 'all_4d_downsampled.pt')
        labels_path: Path to imageID_to_labels.json
        info_path: Path to index_to_name.json

    Returns:
        data: Tensor of shape [N, H, W, D, T]
        imageID_to_labels: Dict mapping image_id to labels
        index_to_info: Dict mapping idx to scan info
    """
    print(f"Loading ADNI data...")

    # Load data tensor
    data = torch.load(data_path)
    print(f"  - Data shape: {data.shape}")

    # Load labels
    with open(labels_path, "r") as f:
        imageID_to_labels = json.load(f)
    print(f"  - Loaded labels for {len(imageID_to_labels)} scans")

    # Load scan info
    with open(info_path, "r") as f:
        index_to_info = json.load(f)
    index_to_info = {int(k): v for k, v in index_to_info.items()}
    print(f"  - Loaded info for {len(index_to_info)} samples")

    return data, imageID_to_labels, index_to_info


if __name__ == "__main__":
    # Test datasets
    print("Testing ADNI SwiFT datasets...")
    print("=" * 80)

    # This is just a structure test - real data would be loaded from files
    print("Note: This is a structure test with dummy data")
    print("Real usage requires loading actual ADNI data files")
