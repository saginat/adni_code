"""
Data preprocessing module for SwiFT pipeline
Handles transformation from raw fMRI data to SwiFT-compatible format

IMPORTANT: Following SwiFT paper's preprocessing approach:
1. Identify and preserve all brain regions
2. Crop/remove only non-brain (background) voxels
3. Apply dataset-specific padding during data loading (not here)
4. Use background value for padding (computed from non-brain regions)

Users should inspect their data first using visualization tools to determine
appropriate cropping levels that don't cut out brain regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


def detect_background_and_brain(
    data: torch.Tensor, threshold: float = 1e-5
) -> Tuple[torch.Tensor, float, Dict[str, Tuple[int, int]]]:
    """
    Detect background (non-brain) voxels and compute brain bounding box

    Args:
        data: Input tensor of shape [B, H, W, D, T] or [H, W, D, T]
        threshold: Threshold for considering voxels as background

    Returns:
        Tuple of:
            - background_mask: Boolean mask where True = background
            - background_value: Typical value of background voxels
            - brain_bbox: Dictionary with 'height', 'width', 'depth' tuples of (start, end)
    """
    # Handle different input shapes
    if data.ndim == 4:
        # [H, W, D, T] -> average over time
        mean_volume = data.mean(dim=-1)
    elif data.ndim == 5:
        # [B, H, W, D, T] -> average over batch and time
        mean_volume = data.mean(dim=(0, -1))
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")

    # Background is voxels that are zero or very close to zero across time
    background_mask = torch.abs(mean_volume) < threshold

    # Compute background value (for padding)
    if background_mask.any():
        background_value = float(mean_volume[background_mask].mean().item())
    else:
        background_value = 0.0

    # Find brain bounding box (non-background regions)
    brain_mask = ~background_mask

    # Find bounds for each dimension
    brain_indices = torch.where(brain_mask)

    brain_bbox = {
        "height": (int(brain_indices[0].min()), int(brain_indices[0].max()) + 1),
        "width": (int(brain_indices[1].min()), int(brain_indices[1].max()) + 1),
        "depth": (int(brain_indices[2].min()), int(brain_indices[2].max()) + 1),
    }

    return background_mask, background_value, brain_bbox


def add_channel_dimension(data: torch.Tensor) -> torch.Tensor:
    """
    Add channel dimension to fMRI data

    Args:
        data: Input tensor of shape [B, H, W, D, T] or [H, W, D, T]

    Returns:
        Tensor with channel dimension: [B, 1, H, W, D, T] or [1, H, W, D, T]
    """
    if data.ndim == 4:
        # Single scan: [H, W, D, T] -> [1, H, W, D, T]
        return data.unsqueeze(0)
    elif data.ndim == 5:
        # Batch: [B, H, W, D, T] -> [B, 1, H, W, D, T]
        return data.unsqueeze(1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")


def crop_to_brain_preserve_dims(
    data: torch.Tensor,
    target_size: Tuple[int, int, int] = (96, 96, 96),
    brain_bbox: Optional[Dict[str, Tuple[int, int]]] = None,
    auto_detect: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
    """
    Crop spatial dimensions to remove background while preserving brain regions

    This follows SwiFT's approach:
    - Inspect data to find brain regions
    - Crop only background voxels
    - Ensure no brain regions are cut out
    - Target is to get as close to 96x96x96 as possible

    Note: Padding to exactly 96x96x96 should be done in the Dataset class
    with background values, not here.

    Args:
        data: Input tensor of shape [B, C, H, W, D, T]
        target_size: Target spatial dimensions (should be ~96x96x96)
        brain_bbox: Optional pre-computed brain bounding box
        auto_detect: Whether to auto-detect brain regions

    Returns:
        Tuple of:
            - Cropped data (may not be exactly target_size)
            - Brain bounding box used for cropping
    """
    B, C, H, W, D, T = data.shape
    target_H, target_W, target_D = target_size

    # Auto-detect brain regions if needed
    if brain_bbox is None and auto_detect:
        # Remove channel dimension for detection
        data_for_detection = data[:, 0, :, :, :, :]  # [B, H, W, D, T]
        _, _, brain_bbox = detect_background_and_brain(data_for_detection)

    # If we have brain bbox, crop to it
    if brain_bbox is not None:
        h_start, h_end = brain_bbox["height"]
        w_start, w_end = brain_bbox["width"]
        d_start, d_end = brain_bbox["depth"]

        data = data[:, :, h_start:h_end, w_start:w_end, d_start:d_end, :]

        print(
            f"  Cropped to brain regions: H[{h_start}:{h_end}]={h_end - h_start}, "
            f"W[{w_start}:{w_end}]={w_end - w_start}, "
            f"D[{d_start}:{d_end}]={d_end - d_start}"
        )
    else:
        # No brain detection - do conservative center crop if needed
        H_new, W_new, D_new = H, W, D

        if H > target_H:
            h_crop = (H - target_H) // 2
            data = data[:, :, h_crop : h_crop + target_H, :, :, :]
            H_new = target_H

        if W > target_W:
            w_crop = (W - target_W) // 2
            data = data[:, :, :, w_crop : w_crop + target_W, :, :]
            W_new = target_W

        if D > target_D:
            d_crop = (D - target_D) // 2
            data = data[:, :, :, :, d_crop : d_crop + target_D, :]
            D_new = target_D

        brain_bbox = {
            "height": (0, H_new),
            "width": (0, W_new),
            "depth": (0, D_new),
        }

        print(f"  Center cropped to: {H_new}x{W_new}x{D_new}")

    return data, brain_bbox


def pad_or_crop_spatial(
    data: torch.Tensor, target_size: Tuple[int, int, int] = (96, 96, 96)
) -> torch.Tensor:
    """
    DEPRECATED: Use crop_to_brain_preserve_dims() and dataset-specific padding instead.

    This function is kept for backwards compatibility but should not be used
    in production. SwiFT's approach is to:
    1. Crop only background voxels (use crop_to_brain_preserve_dims)
    2. Apply padding in Dataset class with background values

    Pad or crop spatial dimensions to target size

    Original size: [91, 109, 91] -> Target: [96, 96, 96]
    - Height 91 -> 96: pad 2 on top, 3 on bottom
    - Width 109 -> 96: center crop (remove 13 voxels)
    - Depth 91 -> 96: pad 2 on front, 3 on back

    Args:
        data: Input tensor of shape [B, C, H, W, D, T]
        target_size: Target spatial dimensions (H, W, D)

    Returns:
        Tensor with spatial dimensions adjusted to target_size
    """
    print(
        "WARNING: pad_or_crop_spatial is deprecated. Use crop_to_brain_preserve_dims + dataset padding."
    )

    B, C, H, W, D, T = data.shape
    target_H, target_W, target_D = target_size

    # Handle height (91 -> 96): pad
    if H < target_H:
        pad_H = target_H - H
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
    else:
        # Crop if larger
        crop_H = H - target_H
        crop_top = crop_H // 2
        data = data[:, :, crop_top : crop_top + target_H, :, :, :]
        pad_top = pad_bottom = 0
        H = target_H

    # Handle width (109 -> 96): center crop
    if W > target_W:
        crop_W = W - target_W
        crop_left = crop_W // 2
        data = data[:, :, :, crop_left : crop_left + target_W, :, :]
        pad_left = pad_right = 0
        W = target_W
    else:
        # Pad if smaller
        pad_W = target_W - W
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

    # Handle depth (91 -> 96): pad
    if D < target_D:
        pad_D = target_D - D
        pad_front = pad_D // 2
        pad_back = pad_D - pad_front
    else:
        # Crop if larger
        crop_D = D - target_D
        crop_front = crop_D // 2
        data = data[:, :, :, :, crop_front : crop_front + target_D, :]
        pad_front = pad_back = 0
        D = target_D

    # Apply padding if needed
    # F.pad format: (left, right, top, bottom, front, back) for last 3 dimensions
    # Our dimensions: [B, C, H, W, D, T]
    # We need to pad D, W, H (in reverse order for F.pad)
    if any([pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_back]):
        # Note: F.pad pads from last dimension backwards
        # For [B, C, H, W, D, T], we want to pad H, W, D
        # F.pad order: (T_left, T_right, D_left, D_right, W_left, W_right, H_left, H_right)
        # We don't pad T, so: (0, 0, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom)

        # Get background value (use edge value for padding)
        pad_value = float(data[0, 0, 0, 0, 0, 0].item())

        data = F.pad(
            data,
            (0, 0, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=pad_value,
        )

    return data


def whole_brain_znormalization(
    data: torch.Tensor,
    background_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    """
    Apply whole-brain z-normalization following SwiFT's approach

    Important: Only normalize brain voxels, not background.
    Background voxels should retain their original value for padding.

    Args:
        data: Input tensor of shape [B, C, H, W, D, T]
        background_mask: Optional mask of background voxels [B, C, H, W, D, T] or [H, W, D]
        eps: Small constant for numerical stability

    Returns:
        Tuple of:
            - Z-normalized tensor
            - Background value (for later padding)
    """
    B, C, H, W, D, T = data.shape

    # Create or expand background mask
    if background_mask is None:
        # Auto-detect: background is near-zero voxels
        data_for_detection = data[:, 0, :, :, :, :]
        background_mask, background_value, _ = detect_background_and_brain(
            data_for_detection
        )
        # Expand to match data shape
        background_mask = (
            background_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )  # [1, 1, H, W, D, 1]
        background_mask = background_mask.expand(B, C, H, W, D, T)
    elif background_mask.ndim == 3:
        # [H, W, D] -> [B, C, H, W, D, T]
        background_mask = background_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        background_mask = background_mask.expand(B, C, H, W, D, T)

    # Get background value before normalization
    if background_mask.any():
        background_value = float(data[background_mask].mean().item())
    else:
        background_value = 0.0

    # Normalize only brain voxels
    brain_mask = ~background_mask

    if brain_mask.any():
        brain_voxels = data[brain_mask]
        global_mean = brain_voxels.mean()
        global_std = brain_voxels.std() + eps

        # Create normalized tensor
        data_normalized = data.clone()
        data_normalized[brain_mask] = (brain_voxels - global_mean) / global_std

        # Find minimum z-value for background regions (following SwiFT)
        min_z_value = data_normalized[brain_mask].min()
        data_normalized[background_mask] = min_z_value
    else:
        data_normalized = data
        min_z_value = 0.0

    return data_normalized, float(min_z_value.item()) if isinstance(
        min_z_value, torch.Tensor
    ) else min_z_value


def create_temporal_windows(
    data: torch.Tensor, window_size: int = 20, stride: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create overlapping temporal windows from long fMRI sequence

    Args:
        data: Input tensor of shape [B, C, H, W, D, T]
        window_size: Size of temporal window (default: 20)
        stride: Stride between windows (default: 10)

    Returns:
        Tuple of:
            - windowed_data: Tensor of shape [N, C, H, W, D, window_size] where N is number of windows
            - window_indices: Tensor of shape [N, 2] containing (batch_idx, start_frame) for each window
    """
    B, C, H, W, D, T = data.shape

    windows = []
    indices = []

    for b in range(B):
        num_windows = (T - window_size) // stride + 1

        for i in range(num_windows):
            start = i * stride
            end = start + window_size

            if end <= T:
                window = data[b : b + 1, :, :, :, :, start:end]
                windows.append(window)
                indices.append([b, start])

    if windows:
        windowed_data = torch.cat(windows, dim=0)
        window_indices = torch.tensor(indices)
    else:
        raise ValueError(
            f"No valid windows created. T={T}, window_size={window_size}, stride={stride}"
        )

    return windowed_data, window_indices


def preprocess_scan(
    data: torch.Tensor,
    target_spatial_size: Tuple[int, int, int] = (96, 96, 96),
    window_size: int = 20,
    stride: int = 10,
    normalize: bool = True,
    to_float16: bool = True,
    crop_background: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Complete preprocessing pipeline for fMRI scan (SwiFT-compatible)

    Following SwiFT paper's approach:
    1. Detect background regions
    2. Crop to brain regions (remove background, preserve brain)
    3. Normalize only brain voxels
    4. Create temporal windows
    5. Return metadata for dataset-specific padding

    Note: Final padding to 96x96x96 should be done in Dataset class!

    Args:
        data: Input tensor of shape [B, H, W, D, T] or [H, W, D, T]
        target_spatial_size: Target spatial dimensions (default: 96x96x96)
        window_size: Temporal window size (default: 20)
        stride: Stride between windows (default: 10)
        normalize: Whether to apply z-normalization (default: True)
        to_float16: Whether to convert to float16 (default: True)
        crop_background: Whether to crop background regions (default: True)

    Returns:
        Tuple of:
            - preprocessed_data: [N, 1, H', W', D', 20] (may not be exactly 96x96x96)
            - window_indices: [N, 2] containing (batch_idx, start_frame)
            - metadata: Dict with 'background_value', 'brain_bbox', 'needs_padding'
    """
    print(f"Input shape: {data.shape}")
    metadata = {}

    # Step 1: Add channel dimension
    data = add_channel_dimension(data)
    print(f"After adding channel: {data.shape}")

    # Step 2: Detect background (before normalization)
    data_for_detection = data[:, 0, :, :, :, :]
    background_mask, background_value, brain_bbox = detect_background_and_brain(
        data_for_detection
    )
    metadata["background_value"] = background_value
    metadata["brain_bbox"] = brain_bbox

    brain_size = (
        brain_bbox["height"][1] - brain_bbox["height"][0],
        brain_bbox["width"][1] - brain_bbox["width"][0],
        brain_bbox["depth"][1] - brain_bbox["depth"][0],
    )
    print(f"Detected brain regions: {brain_size[0]}x{brain_size[1]}x{brain_size[2]}")
    print(f"Background value: {background_value:.6f}")

    # Step 3: Crop to brain regions (remove background)
    if crop_background:
        data, brain_bbox = crop_to_brain_preserve_dims(
            data, target_spatial_size, brain_bbox, auto_detect=False
        )
        current_size = data.shape[2:5]
        print(f"After brain-aware cropping: {current_size}")

        # Check if padding will be needed
        needs_padding = tuple(
            (target_spatial_size[i] - current_size[i]) for i in range(3)
        )
        metadata["needs_padding"] = needs_padding
        metadata["current_spatial_size"] = current_size
    else:
        # Use old method (deprecated)
        data = pad_or_crop_spatial(data, target_spatial_size)
        metadata["needs_padding"] = (0, 0, 0)
        metadata["current_spatial_size"] = target_spatial_size
        print(f"After spatial adjustment: {data.shape}")

    # Step 4: Normalize if requested (only brain voxels)
    if normalize:
        data, min_z_value = whole_brain_znormalization(data, background_mask)
        metadata["background_value"] = (
            min_z_value  # Update with normalized background value
        )
        print(f"After normalization - background value updated to: {min_z_value:.6f}")

    # Step 5: Create temporal windows
    windowed_data, window_indices = create_temporal_windows(data, window_size, stride)
    print(f"After windowing: {windowed_data.shape}, {len(window_indices)} windows")

    # Step 6: Convert to float16 if requested
    if to_float16:
        windowed_data = windowed_data.half()
        print(f"Converted to float16")

    return windowed_data, window_indices, metadata


def save_preprocessed_data(data: torch.Tensor, indices: torch.Tensor, save_path: str):
    """
    Save preprocessed data to disk

    Args:
        data: Preprocessed tensor
        indices: Window indices
        save_path: Path to save the data
    """
    torch.save({"data": data, "indices": indices, "shape": data.shape}, save_path)
    print(f"Saved preprocessed data to {save_path}")


def load_preprocessed_data(load_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load preprocessed data from disk

    Args:
        load_path: Path to load the data

    Returns:
        Tuple of (data, indices)
    """
    checkpoint = torch.load(load_path)
    print(f"Loaded preprocessed data with shape: {checkpoint['shape']}")
    return checkpoint["data"], checkpoint["indices"]


if __name__ == "__main__":
    # Test preprocessing with dummy data
    print("Testing preprocessing pipeline...")
    print("=" * 80)

    # Create dummy data: [1, 91, 109, 91, 140]
    dummy_data = torch.randn(1, 91, 109, 91, 140)
    print(f"Created dummy data: {dummy_data.shape}")

    # Add some "brain-like" structure (non-zero regions)
    # Simulate brain in center region
    dummy_data[:, 10:85, 15:95, 10:85, :] += 5.0  # Brain regions have higher values

    print("\n" + "=" * 80)
    print("TEST 1: Brain-aware preprocessing (NEW SwiFT-compatible approach)")
    print("=" * 80)

    # Preprocess with brain-aware cropping
    preprocessed, indices, metadata = preprocess_scan(
        dummy_data,
        window_size=20,
        stride=10,
        normalize=True,
        to_float16=False,  # Keep float32 for testing
        crop_background=True,
    )

    print(f"\n✓ Preprocessing complete!")
    print(f"  - Output shape: {preprocessed.shape}")
    print(f"  - Number of windows: {len(indices)}")
    print(f"  - Background value: {metadata['background_value']:.6f}")
    print(f"  - Brain bbox: {metadata['brain_bbox']}")
    print(f"  - Current spatial size: {metadata['current_spatial_size']}")
    print(f"  - Needs padding: {metadata['needs_padding']}")
    print(f"  - Data range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    print(f"  - Data mean: {preprocessed.mean():.3f}, std: {preprocessed.std():.3f}")

    print("\nNOTE: Final padding to 96x96x96 should be done in Dataset class")
    print("      using the background_value from metadata!")
