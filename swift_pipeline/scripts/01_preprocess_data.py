"""
Preprocess data for SwiFT pipeline
Converts raw fMRI data to SwiFT-compatible format
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.preprocessing import preprocess_scan, save_preprocessed_data
from configs.config_pretrain import DATA_CONFIG


def main():
    print("=" * 80)
    print("SwiFT Data Preprocessing")
    print("=" * 80)

    # For testing with dummy data
    print("\nCreating dummy fMRI scan...")
    print(f"Shape: {DATA_CONFIG['original_size']}")

    # Create dummy scan: [1, 91, 109, 91, 140]
    dummy_scan = torch.randn(*DATA_CONFIG["original_size"]).unsqueeze(0)
    print(f"✓ Dummy scan created: {dummy_scan.shape}")

    # Preprocess
    print("\nPreprocessing...")
    preprocessed, indices = preprocess_scan(
        dummy_scan,
        target_spatial_size=DATA_CONFIG["target_spatial_size"],
        window_size=DATA_CONFIG["window_size"],
        stride=DATA_CONFIG["stride"],
        normalize=DATA_CONFIG["normalize"],
        to_float16=DATA_CONFIG["to_float16"],
    )

    print(f"\n✓ Preprocessing complete!")
    print(f"  - Output shape: {preprocessed.shape}")
    print(f"  - Number of windows: {len(indices)}")

    # Save
    save_path = "./data/preprocessed/preprocessed_data.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_preprocessed_data(preprocessed, indices, save_path)

    print(f"\n✓ Data saved to {save_path}")
    print("\nNext step: Run 02_pretrain.py for contrastive pretraining")


if __name__ == "__main__":
    main()
