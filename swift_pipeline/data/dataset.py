"""
Dataset classes for SwiFT pretraining and fine-tuning

Following SwiFT paper's approach:
- Padding is done HERE in the dataset, not during preprocessing
- Use background values for padding (passed via metadata)
- Different datasets may need different padding strategies
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Callable, Dict
import random


def pad_to_target_size(
    data: torch.Tensor,
    target_size: Tuple[int, int, int] = (96, 96, 96),
    background_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad data to target spatial size using background value

    This follows SwiFT's approach of padding in the Dataset class
    with the background value computed during preprocessing.

    Args:
        data: Input tensor of shape [C, H, W, D, T]
        target_size: Target spatial dimensions (H, W, D)
        background_value: Value to use for padding (from preprocessing metadata)

    Returns:
        Padded tensor of shape [C, target_H, target_W, target_D, T]
    """
    C, H, W, D, T = data.shape
    target_H, target_W, target_D = target_size

    # Calculate padding needed
    pad_H = max(0, target_H - H)
    pad_W = max(0, target_W - W)
    pad_D = max(0, target_D - D)

    if pad_H == 0 and pad_W == 0 and pad_D == 0:
        return data

    # Distribute padding evenly
    pad_H_top = pad_H // 2
    pad_H_bottom = pad_H - pad_H_top

    pad_W_left = pad_W // 2
    pad_W_right = pad_W - pad_W_left

    pad_D_front = pad_D // 2
    pad_D_back = pad_D - pad_D_front

    # Apply padding
    # F.pad order for [C, H, W, D, T]: (T_left, T_right, D_left, D_right, W_left, W_right, H_left, H_right)
    padded = F.pad(
        data,
        (
            0,
            0,
            pad_D_front,
            pad_D_back,
            pad_W_left,
            pad_W_right,
            pad_H_top,
            pad_H_bottom,
        ),
        mode="constant",
        value=background_value,
    )

    return padded


class SwiFTPretrainDataset(Dataset):
    """
    Dataset for contrastive pretraining
    Returns two random non-overlapping temporal windows for contrastive learning
    """

    def __init__(
        self,
        data: torch.Tensor,
        window_indices: torch.Tensor,
        window_size: int = 20,
        metadata: Optional[Dict] = None,
        target_spatial_size: Tuple[int, int, int] = (96, 96, 96),
        augmentation: Optional[Callable] = None,
    ):
        """
        Args:
            data: Preprocessed windows [N, C, H, W, D, T] (may not be exactly 96x96x96)
            window_indices: Window indices [N, 2] (batch_idx, start_frame)
            window_size: Size of temporal window
            metadata: Dict with 'background_value', 'needs_padding', etc.
            target_spatial_size: Target size for padding (default: 96x96x96)
            augmentation: Optional augmentation function
        """
        self.data = data
        self.window_indices = window_indices
        self.window_size = window_size
        self.metadata = metadata or {}
        self.target_spatial_size = target_spatial_size
        self.augmentation = augmentation

        # Get background value for padding
        self.background_value = self.metadata.get("background_value", 0.0)

        # Check if padding is needed
        current_size = data.shape[2:5]  # [H, W, D]
        self.needs_padding = any(
            current_size[i] < target_spatial_size[i] for i in range(3)
        )

        if self.needs_padding:
            print(f"  Dataset will pad from {current_size} to {target_spatial_size}")
            print(f"  Using background value: {self.background_value:.6f}")

        # Group windows by original scan (batch_idx)
        self.scan_to_windows = {}
        for idx, (batch_idx, start_frame) in enumerate(window_indices):
            batch_idx = int(batch_idx.item())
            if batch_idx not in self.scan_to_windows:
                self.scan_to_windows[batch_idx] = []
            self.scan_to_windows[batch_idx].append((idx, int(start_frame.item())))

        # Create list of valid pairs (scan_idx, window_idx)
        self.valid_samples = []
        for batch_idx, windows in self.scan_to_windows.items():
            if len(windows) >= 2:  # Need at least 2 windows for contrastive learning
                for window_idx, _ in windows:
                    self.valid_samples.append((batch_idx, window_idx))

        print(
            f"Contrastive dataset: {len(self.valid_samples)} valid samples from {len(self.scan_to_windows)} scans"
        )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        batch_idx, window_idx = self.valid_samples[idx]

        # Get first view
        view1 = self.data[window_idx]

        # Pad if needed
        if self.needs_padding:
            view1 = pad_to_target_size(
                view1, self.target_spatial_size, self.background_value
            )

        # Find all windows from same scan, excluding overlapping windows
        available_windows = []
        _, start_frame1 = self.window_indices[window_idx]
        start_frame1 = int(start_frame1.item())

        for other_idx, other_start in self.scan_to_windows[batch_idx]:
            if other_idx == window_idx:
                continue

            # Check if windows overlap
            # Windows overlap if |start1 - start2| < window_size
            if abs(start_frame1 - other_start) >= self.window_size:
                available_windows.append(other_idx)

        # If no non-overlapping windows available, use a different window anyway
        if not available_windows:
            available_windows = [
                idx for idx, _ in self.scan_to_windows[batch_idx] if idx != window_idx
            ]

        # Sample second view
        if available_windows:
            view2_idx = random.choice(available_windows)
            view2 = self.data[view2_idx]

            # Pad if needed
            if self.needs_padding:
                view2 = pad_to_target_size(
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


class SwiFTFinetuneDataset(Dataset):
    """
    Dataset for supervised fine-tuning
    Returns single windows with labels
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        window_indices: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        target_spatial_size: Tuple[int, int, int] = (96, 96, 96),
        augmentation: Optional[Callable] = None,
    ):
        """
        Args:
            data: Preprocessed windows [N, C, H, W, D, T] (may not be exactly 96x96x96)
            labels: Labels for each window [N] or [N, num_classes]
            window_indices: Optional window indices [N, 2]
            metadata: Dict with 'background_value', 'needs_padding', etc.
            target_spatial_size: Target size for padding (default: 96x96x96)
            augmentation: Optional augmentation function
        """
        self.data = data
        self.labels = labels
        self.window_indices = window_indices
        self.metadata = metadata or {}
        self.target_spatial_size = target_spatial_size
        self.augmentation = augmentation

        # Get background value for padding
        self.background_value = self.metadata.get("background_value", 0.0)

        # Check if padding is needed
        current_size = data.shape[2:5]  # [H, W, D]
        self.needs_padding = any(
            current_size[i] < target_spatial_size[i] for i in range(3)
        )

        if self.needs_padding:
            print(f"  Dataset will pad from {current_size} to {target_spatial_size}")
            print(f"  Using background value: {self.background_value:.6f}")

        assert len(data) == len(labels), (
            f"Data and labels must have same length: {len(data)} vs {len(labels)}"
        )

        print(f"Fine-tuning dataset: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # Pad if needed
        if self.needs_padding:
            x = pad_to_target_size(x, self.target_spatial_size, self.background_value)

        # Apply augmentation if provided (typically only during training)
        if self.augmentation is not None:
            x = self.augmentation(x)

        return x, y


class SimpleDataset(Dataset):
    """
    Simple dataset for single scan testing
    """

    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            data: Preprocessed windows [N, C, H, W, D, T]
            labels: Optional labels
        """
        self.data = data
        self.labels = labels if labels is not None else torch.zeros(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def create_dummy_labels(num_samples: int, task_type: str = "binary") -> torch.Tensor:
    """
    Create dummy labels for testing

    Args:
        num_samples: Number of samples
        task_type: 'binary' or 'regression'

    Returns:
        Tensor of labels
    """
    if task_type == "binary":
        return torch.randint(0, 2, (num_samples,))
    elif task_type == "regression":
        return torch.randn(num_samples)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


if __name__ == "__main__":
    # Test datasets
    print("Testing datasets...")
    print("=" * 80)

    # Create dummy preprocessed data (not exactly 96x96x96)
    num_windows = 12
    data = torch.randn(num_windows, 1, 90, 92, 88, 20)  # Smaller than target
    indices = torch.tensor([[0, i * 10] for i in range(num_windows)])

    # Create metadata (simulating preprocessing output)
    metadata = {
        "background_value": -2.5,
        "brain_bbox": {"height": (5, 85), "width": (8, 100), "depth": (5, 83)},
        "needs_padding": (6, 4, 8),
        "current_spatial_size": (90, 92, 88),
    }

    print(f"\nTesting Contrastive Dataset with padding:")
    print(f"Input data shape: {data.shape}")
    contrastive_dataset = SwiFTPretrainDataset(
        data, indices, metadata=metadata, target_spatial_size=(96, 96, 96)
    )
    print(f"Dataset length: {len(contrastive_dataset)}")

    view1, view2 = contrastive_dataset[0]
    print(f"✓ View 1 shape: {view1.shape} (should be [1, 96, 96, 96, 20])")
    print(f"✓ View 2 shape: {view2.shape} (should be [1, 96, 96, 96, 20])")
    assert view1.shape == (1, 96, 96, 96, 20), (
        f"Expected (1, 96, 96, 96, 20), got {view1.shape}"
    )

    print(f"\nTesting Fine-tuning Dataset with padding:")
    labels = create_dummy_labels(num_windows, task_type="binary")
    finetune_dataset = SwiFTFinetuneDataset(
        data, labels, indices, metadata=metadata, target_spatial_size=(96, 96, 96)
    )
    print(f"Dataset length: {len(finetune_dataset)}")

    x, y = finetune_dataset[0]
    print(f"✓ Input shape: {x.shape} (should be [1, 96, 96, 96, 20])")
    print(f"✓ Label: {y.item()}")
    assert x.shape == (1, 96, 96, 96, 20), (
        f"Expected (1, 96, 96, 96, 20), got {x.shape}"
    )

    print("\n" + "=" * 80)
    print("✓ All dataset tests passed!")
