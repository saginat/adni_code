# ADNI Data Integration with SwiFT Pipeline

This directory contains ADNI-specific dataset classes and utilities for the SwiFT pipeline.

## Overview

The ADNI integration maintains your existing data structure while adapting it for SwiFT's architecture:

- **Your data format**: `[N, H, W, D, T]` tensors with `imageID_to_labels` and `index_to_name` JSON files
- **SwiFT format**: Requires `[N, C, H, W, D, T]` with channel dimension and proper temporal windowing
- **Integration**: Handles format conversion, label mapping, and NaN handling automatically

## Files

### `dataset_adni.py`
ADNI-specific dataset classes that integrate with your existing data structure:

- **`ADNISwiFTPretrainDataset`**: For contrastive pretraining (self-supervised, no labels needed)
  - Groups windows by `image_id` from `index_to_info`
  - Returns two non-overlapping temporal windows for contrastive learning
  - Handles padding and augmentation

- **`ADNISwiFTFinetuneDataset`**: For supervised fine-tuning with degradation labels
  - Loads labels from `imageID_to_labels` dictionary
  - Supports multiple task names (e.g., `degradation_binary_1year`)
  - Handles NaN values with configurable strategy ('skip', 'zero', 'keep')
  - Automatic label statistics and validation

- **`load_adni_data()`**: Utility to load your data files
  ```python
  data, imageID_to_labels, index_to_info = load_adni_data(
      data_path="path/to/all_4d_downsampled.pt",
      labels_path="path/to/imageID_to_labels.json",
      info_path="path/to/index_to_name.json"
  )
  ```

### `prepare_adni_data.py`
High-level utilities for data preparation:

- **`ADNIDataSplitter`**: Splits data by subjects (prevents data leakage)
  - Groups samples by `subject_id` from `index_to_info`
  - Ensures same subject doesn't appear in train and test sets
  - Configurable train/val/test splits

- **`prepare_adni_datasets()`**: Main function to prepare datasets
  ```python
  datasets = prepare_adni_datasets(
      data_path="path/to/data.pt",
      labels_path="path/to/labels.json",
      info_path="path/to/info.json",
      task_names=["degradation_binary_1year"],
      stage="finetune",  # or "pretrain"
      val_split=0.1,
      test_split=0.2,
      seed=42,
      handle_nan='skip'
  )
  ```

- **`create_adni_dataloaders()`**: Creates PyTorch DataLoaders

## Data Format

### Your Existing Format
```python
# Data tensor
data.shape  # [N, H, W, D, T] e.g., [4177, 64, 76, 63, 140]

# index_to_name.json
{
    "0": {
        "filename": "dswau012_S_4188_20120502_Resting_State_fMRI_79_I301834.nii",
        "subject_id": "012_S_4188",
        "date": "02/05/2012",
        "image_id": "I301834"
    },
    ...
}

# imageID_to_labels.json
{
    "I301221": {
        "Sex": "M",
        "Age": 74.6,
        "degradation_binary_1year": 0,
        "degradation_binary_2years": 0,
        "degradation_binary_3years": 0,
        ...
    },
    ...
}
```

### SwiFT Expected Format
After temporal windowing with your existing code, you have:
```python
# After TimeWindowSplitter (from your preprocessing.py)
windowed_data.shape  # [N * num_windows, H, W, D, window_size]
# e.g., [4177 * 14, 64, 76, 63, 10]  (140 timepoints → 14 windows of 10)

# Updated index_to_info with window information
{
    "0": {
        "filename": "...",
        "subject_id": "012_S_4188",
        "image_id": "I301834",
        "window_index": 0  # Which window this is
    },
    ...
}
```

The ADNI dataset classes automatically:
1. Add channel dimension: `[N, H, W, D, T]` → `[N, 1, H, W, D, T]`
2. Pad to target size (96×96×96) using background values
3. Map labels using `image_id` from `index_to_info`

## Usage Examples

### Example 1: Contrastive Pretraining (Self-Supervised)
```python
from data.prepare_adni_data import prepare_adni_datasets, create_adni_dataloaders

# Prepare datasets (no labels needed)
datasets = prepare_adni_datasets(
    data_path="../../data/all_4d_downsampled.pt",
    labels_path="../../imageID_to_labels.json",
    info_path="../../index_to_name.json",
    task_names=[],  # Not used in pretraining
    stage="pretrain",
    val_split=0.1,
    test_split=0.2,
)

# Create dataloaders
loaders = create_adni_dataloaders(datasets, batch_size=4)

# Use in training
for view1, view2 in loaders['train_loader']:
    # view1, view2: [B, 1, 96, 96, 96, 20]
    # Two non-overlapping temporal windows from same scan
    ...
```

### Example 2: Fine-tuning for Degradation Prediction
```python
# Prepare datasets with degradation labels
datasets = prepare_adni_datasets(
    data_path="../../data/all_4d_downsampled.pt",
    labels_path="../../imageID_to_labels.json",
    info_path="../../index_to_name.json",
    task_names=["degradation_binary_1year"],
    stage="finetune",
    val_split=0.1,
    test_split=0.2,
    handle_nan='skip',  # Skip samples with NaN labels
)

# Create dataloaders
loaders = create_adni_dataloaders(datasets, batch_size=8)

# Use in training
for inputs, labels in loaders['train_loader']:
    # inputs: [B, 1, 96, 96, 96, 20]
    # labels: [B, 1] for single task or [B, num_tasks] for multi-task
    ...
```

### Example 3: Multi-Task Learning
```python
# Train on multiple degradation horizons simultaneously
datasets = prepare_adni_datasets(
    data_path="../../data/all_4d_downsampled.pt",
    labels_path="../../imageID_to_labels.json",
    info_path="../../index_to_name.json",
    task_names=[
        "degradation_binary_1year",
        "degradation_binary_2years",
        "degradation_binary_3years",
    ],
    stage="finetune",
    handle_nan='skip',
)

# Labels will be [B, 3] for the three tasks
```

## Handling NaN Values

Your ADNI labels contain NaN values. The `handle_nan` parameter controls behavior:

- **`'skip'`** (default): Exclude samples with any NaN labels from dataset
- **`'zero'`**: Replace NaN values with 0.0
- **`'keep'`**: Keep NaN values (you'll need to handle them in your loss function)

Example output:
```
Fine-tuning dataset: 3245 valid samples
  - Tasks: ['degradation_binary_1year']
  - Skipped 932 samples with NaN labels
  - Skipped 0 samples without labels
  - degradation_binary_1year: 2 unique values
    Distribution: {0.0: 2180, 1.0: 1065}
```

## Subject-Level Splitting (No Data Leakage)

The `ADNIDataSplitter` prevents data leakage by splitting at the subject level:

```python
# Splits by subject_id, not by individual scans
# Ensures same patient doesn't appear in both train and test sets
splitter = ADNIDataSplitter(data, index_to_info, val_split=0.1, test_split=0.2)
train_indices, val_indices, test_indices = splitter.split()

# Output:
# Data split by subjects (no data leakage):
#   Train: 2000 subjects, 8545 samples
#   Val:   250 subjects, 1075 samples
#   Test:  500 subjects, 2145 samples
```

## Complete Pipeline Script

See `scripts/validate_adni_pipeline.py` for a complete working example that:
1. Loads your ADNI data
2. Performs contrastive pretraining
3. Fine-tunes for degradation prediction
4. Evaluates on test set

Run with:
```bash
cd scripts
python validate_adni_pipeline.py
```

## Integration with Your Existing Code

Your existing preprocessing pipeline from `my_code/data/preprocessing.py`:
```python
# Your existing code
train_data = TimeWindowSplitter(train_data, config.WINDOW_SIZE).split()
index_to_info_tr = TimeWindowSplitter.update_info_dict(
    index_to_info_tr, original_time_len, config.WINDOW_SIZE
)
```

Now feeds directly into SwiFT datasets:
```python
# SwiFT integration
train_dataset = ADNISwiFTFinetuneDataset(
    train_data,              # Already windowed [N, H, W, D, window_size]
    index_to_info_tr,        # Already updated with window_index
    imageID_to_labels,       # Your existing labels dict
    task_names=['degradation_binary_1year'],
)
```

## Notes

1. **Data Shape**: Your data is `[N, H, W, D, T]`. SwiFT expects `[N, C, H, W, D, T]`. The channel dimension is added automatically in `__getitem__`.

2. **Temporal Windows**: Your `TimeWindowSplitter` already creates windows. The datasets use `window_index` from `index_to_info` to avoid overlapping windows in contrastive learning.

3. **Spatial Size**: Your data is ~64×76×63. SwiFT expects 96×96×96. Padding is applied automatically using background values.

4. **Labels**: The dataset automatically maps `image_id` from `index_to_info` to labels in `imageID_to_labels`.

5. **Memory Efficiency**: Data is loaded once and indexed, not duplicated per dataset.

