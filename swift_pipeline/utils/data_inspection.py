"""
Data inspection utilities for fMRI preprocessing

Use these tools to visualize your fMRI data and determine appropriate
cropping levels before preprocessing, following SwiFT paper recommendations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import warnings

try:
    from nilearn import plotting
    from nilearn.image import mean_img

    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    warnings.warn(
        "nilearn not installed. Install with: pip install nilearn nibabel\n"
        "Visualization functions will not be available."
    )


def visualize_fmri_nifti(fmri_path: str, threshold: Optional[float] = None):
    """
    Visualize fMRI data from NIfTI file using nilearn

    This matches the SwiFT paper's recommendation:
    ```python
    from nilearn import plotting
    from nilearn.image import mean_img
    plotting.view_img(mean_img(fmri_filename), threshold=None)
    ```

    Args:
        fmri_path: Path to NIfTI file (.nii or .nii.gz)
        threshold: Optional threshold for visualization

    Returns:
        nilearn viewer object (interactive in Jupyter)
    """
    if not NILEARN_AVAILABLE:
        raise ImportError(
            "nilearn is required for this function. Install with: pip install nilearn nibabel"
        )

    print(f"Loading fMRI from: {fmri_path}")
    mean_image = mean_img(fmri_path)

    print(f"Mean image shape: {mean_image.shape}")
    print(f"Affine:\n{mean_image.affine}")

    # Interactive visualization (works in Jupyter)
    viewer = plotting.view_img(mean_image, threshold=threshold)

    # Also create static plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    plotting.plot_stat_map(
        mean_image, display_mode="x", cut_coords=5, axes=axes[0], title="Sagittal view"
    )
    plotting.plot_stat_map(
        mean_image, display_mode="y", cut_coords=5, axes=axes[1], title="Coronal view"
    )
    plotting.plot_stat_map(
        mean_image, display_mode="z", cut_coords=5, axes=axes[2], title="Axial view"
    )

    plt.tight_layout()
    plt.show()

    return viewer


def inspect_tensor_dimensions(
    data: torch.Tensor,
    threshold: float = 1e-5,
    show_slices: bool = True,
) -> dict:
    """
    Inspect tensor dimensions and identify brain regions

    Args:
        data: fMRI tensor [B, H, W, D, T] or [H, W, D, T]
        threshold: Threshold for background detection
        show_slices: Whether to show sample slices

    Returns:
        Dictionary with dimension information and recommended cropping
    """
    # Handle different input shapes
    if data.ndim == 4:
        mean_volume = data.mean(dim=-1)  # Average over time
        batch_size = 1
    elif data.ndim == 5:
        mean_volume = data.mean(dim=(0, -1))  # Average over batch and time
        batch_size = data.shape[0]
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")

    H, W, D = mean_volume.shape

    # Detect background
    background_mask = torch.abs(mean_volume) < threshold
    brain_mask = ~background_mask

    # Find brain bounding box
    brain_indices = torch.where(brain_mask)

    if len(brain_indices[0]) == 0:
        print("Warning: No brain voxels detected! Check your threshold.")
        brain_bbox = {"height": (0, H), "width": (0, W), "depth": (0, D)}
    else:
        brain_bbox = {
            "height": (int(brain_indices[0].min()), int(brain_indices[0].max()) + 1),
            "width": (int(brain_indices[1].min()), int(brain_indices[1].max()) + 1),
            "depth": (int(brain_indices[2].min()), int(brain_indices[2].max()) + 1),
        }

    # Calculate brain dimensions
    brain_h = brain_bbox["height"][1] - brain_bbox["height"][0]
    brain_w = brain_bbox["width"][1] - brain_bbox["width"][0]
    brain_d = brain_bbox["depth"][1] - brain_bbox["depth"][0]

    # Prepare report
    info = {
        "original_shape": (H, W, D),
        "brain_bbox": brain_bbox,
        "brain_dimensions": (brain_h, brain_w, brain_d),
        "background_voxels": background_mask.sum().item(),
        "brain_voxels": brain_mask.sum().item(),
        "background_percentage": (
            background_mask.sum() / background_mask.numel() * 100
        ).item(),
    }

    # Calculate what needs to be done to reach 96x96x96
    target_size = (96, 96, 96)
    crop_or_pad = []
    for i, dim_name in enumerate(["height", "width", "depth"]):
        brain_dim = info["brain_dimensions"][i]
        target_dim = target_size[i]

        if brain_dim > target_dim:
            action = f"⚠️  ERROR: Brain {dim_name} ({brain_dim}) > target ({target_dim})! You will lose brain data!"
        elif brain_dim == target_dim:
            action = f"✓ Perfect: Brain {dim_name} = {target_dim}"
        else:
            pad_needed = target_dim - brain_dim
            action = f"→ Pad {dim_name}: {brain_dim} + {pad_needed} = {target_dim}"

        crop_or_pad.append(action)

    info["recommended_actions"] = crop_or_pad

    # Print report
    print("=" * 80)
    print("fMRI DIMENSION INSPECTION REPORT")
    print("=" * 80)
    print(f"Original shape: {H} x {W} x {D}")
    print(f"Batch size: {batch_size}")
    print("\nBrain bounding box:")
    print(f"  Height: {brain_bbox['height']} → size = {brain_h}")
    print(f"  Width:  {brain_bbox['width']} → size = {brain_w}")
    print(f"  Depth:  {brain_bbox['depth']} → size = {brain_d}")
    print(f"\nBackground: {info['background_percentage']:.1f}% of voxels")
    print("\nTo reach target size 96x96x96:")
    for action in crop_or_pad:
        print(f"  {action}")

    # Show sample slices if requested
    if show_slices:
        plot_brain_slices(mean_volume, brain_bbox)

    return info


def plot_brain_slices(
    volume: torch.Tensor,
    brain_bbox: dict,
    num_slices: int = 5,
):
    """
    Plot sample slices from the volume with brain bounding box overlay

    Args:
        volume: 3D volume tensor [H, W, D]
        brain_bbox: Brain bounding box dict
        num_slices: Number of slices to show per dimension
    """
    volume_np = volume.cpu().numpy()
    H, W, D = volume_np.shape

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 9))

    # Sagittal slices (varying H)
    h_slices = np.linspace(0, H - 1, num_slices, dtype=int)
    for i, h in enumerate(h_slices):
        axes[0, i].imshow(volume_np[h, :, :].T, cmap="gray", origin="lower")
        axes[0, i].set_title(f"H={h}")
        axes[0, i].axis("off")
        # Mark if in brain region
        if brain_bbox["height"][0] <= h < brain_bbox["height"][1]:
            axes[0, i].set_title(f"H={h} [BRAIN]", color="green", fontweight="bold")

    # Coronal slices (varying W)
    w_slices = np.linspace(0, W - 1, num_slices, dtype=int)
    for i, w in enumerate(w_slices):
        axes[1, i].imshow(volume_np[:, w, :].T, cmap="gray", origin="lower")
        axes[1, i].set_title(f"W={w}")
        axes[1, i].axis("off")
        if brain_bbox["width"][0] <= w < brain_bbox["width"][1]:
            axes[1, i].set_title(f"W={w} [BRAIN]", color="green", fontweight="bold")

    # Axial slices (varying D)
    d_slices = np.linspace(0, D - 1, num_slices, dtype=int)
    for i, d in enumerate(d_slices):
        axes[2, i].imshow(volume_np[:, :, d].T, cmap="gray", origin="lower")
        axes[2, i].set_title(f"D={d}")
        axes[2, i].axis("off")
        if brain_bbox["depth"][0] <= d < brain_bbox["depth"][1]:
            axes[2, i].set_title(f"D={d} [BRAIN]", color="green", fontweight="bold")

    axes[0, 0].set_ylabel("Sagittal", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Coronal", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("Axial", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def compare_cropping_strategies(
    data: torch.Tensor,
    strategies: list = None,
):
    """
    Compare different cropping strategies visually

    Args:
        data: fMRI tensor [H, W, D, T] or [B, H, W, D, T]
        strategies: List of (name, crop_ranges) tuples
                   crop_ranges = {'height': (start, end), 'width': (start, end), 'depth': (start, end)}
    """
    # Handle different input shapes
    if data.ndim == 4:
        mean_volume = data.mean(dim=-1)
    elif data.ndim == 5:
        mean_volume = data.mean(dim=(0, -1))
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")

    if strategies is None:
        # Default: compare no crop vs auto-detect brain
        from .preprocessing import detect_background_and_brain

        _, _, brain_bbox = detect_background_and_brain(
            data if data.ndim == 4 else data[0]
        )

        strategies = [
            ("Original", None),
            ("Brain-aware", brain_bbox),
        ]

    num_strategies = len(strategies)
    fig, axes = plt.subplots(num_strategies, 3, figsize=(12, 4 * num_strategies))

    if num_strategies == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, crop_ranges) in enumerate(strategies):
        if crop_ranges is None:
            cropped = mean_volume
        else:
            h_start, h_end = crop_ranges["height"]
            w_start, w_end = crop_ranges["width"]
            d_start, d_end = crop_ranges["depth"]
            cropped = mean_volume[h_start:h_end, w_start:w_end, d_start:d_end]

        # Show middle slices
        H, W, D = cropped.shape

        axes[idx, 0].imshow(cropped[H // 2, :, :].T, cmap="gray", origin="lower")
        axes[idx, 0].set_title(f"{name}\nSagittal (middle)")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(cropped[:, W // 2, :].T, cmap="gray", origin="lower")
        axes[idx, 1].set_title(f"{name}\nCoronal (middle)")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(cropped[:, :, D // 2].T, cmap="gray", origin="lower")
        axes[idx, 2].set_title(f"{name}\nAxial (middle)\nShape: {H}x{W}x{D}")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Data Inspection Utilities for SwiFT Pipeline")
    print("=" * 80)
    print("\nExample usage:")
    print("""
    # For NIfTI files:
    from utils.data_inspection import visualize_fmri_nifti
    visualize_fmri_nifti('path/to/scan.nii.gz')
    
    # For PyTorch tensors:
    from utils.data_inspection import inspect_tensor_dimensions
    data = torch.load('your_data.pt')
    info = inspect_tensor_dimensions(data, show_slices=True)
    
    # Compare cropping strategies:
    from utils.data_inspection import compare_cropping_strategies
    compare_cropping_strategies(data)
    """)

    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_data = torch.randn(1, 91, 109, 91, 140)
    # Add brain-like structure
    dummy_data[:, 10:85, 15:95, 10:85, :] += 5.0

    info = inspect_tensor_dimensions(dummy_data[0], show_slices=False)
    print("\n✓ Inspection complete!")
