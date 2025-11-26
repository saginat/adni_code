# SwiFT Preprocessing Update - Brain-Aware Approach

## Summary of Changes

You were absolutely correct! Our original implementation didn't follow SwiFT's critical preprocessing approach. We've now updated the pipeline to match the SwiFT paper exactly.

## Key Differences: Old vs. New Approach

### ❌ Old Approach (INCORRECT)
1. Generic padding/cropping without considering brain regions
2. All preprocessing including padding done in `preprocessing.py`
3. Could potentially crop out brain regions
4. Padding with zeros or edge values

### ✅ New Approach (SwiFT-COMPATIBLE)
1. **Brain-aware cropping**: Only remove background voxels, preserve ALL brain regions
2. **Two-stage approach**: 
   - Stage 1 (preprocessing.py): Detect background, crop to brain regions
   - Stage 2 (dataset.py): Pad to 96x96x96 with background values
3. **Background value tracking**: Use computed background value for padding
4. **Data inspection first**: Users should visualize data before preprocessing

## What Changed

### 1. `preprocessing.py` - New Functions

#### `detect_background_and_brain()`
```python
# Detects background (near-zero) voxels
# Computes brain bounding box
# Returns background value for later padding
background_mask, background_value, brain_bbox = detect_background_and_brain(data)
```

#### `crop_to_brain_preserve_dims()`
```python
# Crops ONLY background regions
# Preserves ALL brain voxels
# May result in dimensions < 96x96x96 (that's OK!)
data, brain_bbox = crop_to_brain_preserve_dims(data, target_size=(96,96,96))
```

#### `whole_brain_znormalization()` - Updated
```python
# NOW: Only normalizes brain voxels (not background)
# Returns background value for padding
data_normalized, background_value = whole_brain_znormalization(data, background_mask)
```

#### `preprocess_scan()` - Updated
```python
# NOW returns 3 values:
preprocessed_data, window_indices, metadata = preprocess_scan(data)

# metadata contains:
# - 'background_value': for padding in dataset
# - 'brain_bbox': brain regions identified
# - 'needs_padding': how much padding needed
# - 'current_spatial_size': actual size after brain cropping
```

### 2. `dataset.py` - New Functions

#### `pad_to_target_size()`
```python
# Pads data to 96x96x96 using background value
# Called during data loading (__getitem__)
padded = pad_to_target_size(data, target_size=(96,96,96), background_value=-2.5)
```

#### `SwiFTPretrainDataset` - Updated
```python
# NOW accepts metadata parameter
dataset = SwiFTPretrainDataset(
    data, 
    indices, 
    metadata=metadata,  # <-- NEW!
    target_spatial_size=(96, 96, 96)
)

# Padding happens in __getitem__() with background values
```

#### `SwiFTFinetuneDataset` - Updated
```python
# Same updates as SwiFTPretrainDataset
dataset = SwiFTFinetuneDataset(
    data, 
    labels, 
    indices,
    metadata=metadata,  # <-- NEW!
    target_spatial_size=(96, 96, 96)
)
```

### 3. `utils/data_inspection.py` - NEW FILE

Tools to inspect your fMRI data BEFORE preprocessing:

#### `visualize_fmri_nifti()`
```python
# Visualize NIfTI files using nilearn (matches SwiFT paper recommendation)
from utils.data_inspection import visualize_fmri_nifti
viewer = visualize_fmri_nifti('path/to/scan.nii.gz')
```

#### `inspect_tensor_dimensions()`
```python
# Inspect PyTorch tensors, get brain dimensions, see if cropping is needed
from utils.data_inspection import inspect_tensor_dimensions
info = inspect_tensor_dimensions(data, show_slices=True)
# Prints:
# - Brain bounding box
# - Current dimensions vs target (96x96x96)
# - Warnings if brain regions would be cut
```

#### `compare_cropping_strategies()`
```python
# Visually compare different cropping approaches
from utils.data_inspection import compare_cropping_strategies
compare_cropping_strategies(data)
```

## How SwiFT Paper Does It (from README)

From the SwiFT repository:

```python
# Step 1: Minimal preprocessing (fMRIprep or FSL)
# - Register to MNI space
# - Apply whole-brain mask

# Step 2: Additional preprocessing (preprocessing.py)
# - Whole-brain z-normalization (only brain voxels)
# - Convert to float16
# - Save each volume separately

# Step 3: Remove non-brain background voxels > 96
# IMPORTANT: "you should open your fMRI scans to determine the level 
#             that does not cut out the brain regions"
data = data[:, 14:-7, :, :]  # Example: crop background in width dimension

# Step 4: Padding in Dataset class (datasets.py)
# - Different padding for HCP, ABCD, UKB
# - Use background value for padding
y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
```

## Dataset-Specific Examples from SwiFT

### HCP Dataset
```python
# HCP image shape after cropping: 79, 97, 85
# Padding: (depth: 8+7=15, width: 0, height: 9+8=17)
y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
# Result: 96, 96, 96
```

### ABCD Dataset  
```python
# ABCD rest shape: 79, 97, 85
y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:]
# ABCD task shape: 96, 96, 95
y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value)
```

### UKB Dataset
```python
# UKB shape needs cropping in width, padding in others
y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value)
# Note: Negative padding = cropping
```

## Migration Guide

### For Existing Code

If you have existing preprocessing code, update it as follows:

#### Before:
```python
from data.preprocessing import preprocess_scan
preprocessed, indices = preprocess_scan(data)
dataset = SwiFTPretrainDataset(preprocessed, indices)
```

#### After:
```python
from data.preprocessing import preprocess_scan
preprocessed, indices, metadata = preprocess_scan(data)  # <-- Now returns 3 values
dataset = SwiFTPretrainDataset(preprocessed, indices, metadata=metadata)  # <-- Pass metadata
```

### For New Data

**STEP 1**: Inspect your data first!
```python
from utils.data_inspection import inspect_tensor_dimensions
data = torch.load('your_data.pt')
info = inspect_tensor_dimensions(data, show_slices=True)
```

Look at the output:
- Are brain dimensions > 96 in any axis? → **WARNING**: You'll lose brain data!
- Are brain dimensions < 96? → Good, padding will handle it
- Check the cropping recommendations

**STEP 2**: Preprocess with brain-aware cropping
```python
from data.preprocessing import preprocess_scan
preprocessed, indices, metadata = preprocess_scan(
    data,
    crop_background=True,  # Use brain-aware cropping
    normalize=True,
    to_float16=True,
)
```

**STEP 3**: Create datasets (padding happens automatically)
```python
from data.dataset import SwiFTPretrainDataset
dataset = SwiFTPretrainDataset(
    preprocessed, 
    indices, 
    metadata=metadata,  # Passes background value
    target_spatial_size=(96, 96, 96)
)
```

## Why This Matters

### Problem with Old Approach
```
Original: [91, 109, 91] → Center crop to [96, 96, 96]
Result: Brain regions at edges might be cut out! ❌
```

### New Brain-Aware Approach
```
Original: [91, 109, 91]
↓ Detect brain: [10:85, 15:95, 10:85] = [75, 80, 75]
↓ Crop to brain: [75, 80, 75]
↓ Pad with background: [96, 96, 96]
Result: ALL brain regions preserved! ✓
```

## Validation

The `validate_pipeline.ipynb` notebook will need updates to:
1. Handle the new 3-value return from `preprocess_scan()`
2. Pass `metadata` to dataset constructors
3. Verify that padding is working correctly

## Next Steps

1. ✅ Updated `preprocessing.py` with brain-aware functions
2. ✅ Updated `dataset.py` with padding logic
3. ✅ Created `data_inspection.py` utilities
4. ⏳ Update configs with brain region parameters
5. ⏳ Update validation notebook
6. ⏳ Update README with new approach
7. ⏳ Test with real data

## Questions?

The key insight from SwiFT paper:
> "you should open your fMRI scans to determine the level that does not cut out the brain regions"

This means:
- **Never blindly crop/resize** - you might cut out brain!
- **Always inspect first** - use visualization tools
- **Crop only background** - preserve all brain voxels
- **Pad in dataset** - use computed background values

This is a **critical difference** that affects model fairness and comparison validity!
