# SwiFT Pipeline for fMRI Analysis

Complete implementation of SwiFT (Swin 4D fMRI Transformer) adapted for your data dimensions.

## Overview

This pipeline implements:
- **Data preprocessing**: Transforms fMRI data from [91,109,91,140] to SwiFT-compatible format [1,96,96,96,20]
- **Contrastive pretraining**: Self-supervised learning using dual-view temporal windows
- **Supervised fine-tuning**: Transfer learning for downstream tasks (classification/regression)

## Directory Structure

```
swift_pipeline/
├── models/               # SwiFT architecture
│   ├── swin4d_transformer_ver7.py
│   ├── patchembedding.py
│   ├── heads.py
│   └── __init__.py
├── data/                 # Data processing
│   ├── preprocessing.py
│   ├── dataset.py
│   └── preprocessed/     # Processed data storage
├── training/             # Training components
│   └── losses.py
├── configs/              # Configuration files
│   ├── config_pretrain.py
│   └── config_finetune.py
├── scripts/              # Executable scripts
│   ├── validate_pipeline.py
│   ├── 01_preprocess_data.py
│   ├── 02_pretrain.py
│   └── 03_finetune.py
├── checkpoints/          # Saved models
└── logs/                 # Training logs
```

## Requirements

```bash
pip install torch torchvision
pip install monai
pip install einops
pip install numpy pandas
```

## Quick Start

### 1. Validate Pipeline

Test the complete pipeline with dummy data:

```bash
cd swift_pipeline/scripts
python validate_pipeline.py
```

This will test:
- Data preprocessing ([91,109,91,140] → [1,96,96,96,20])
- Model forward pass
- Contrastive training iteration
- Fine-tuning iteration
- Checkpoint save/load

### 2. Preprocess Your Data

```bash
python 01_preprocess_data.py
```

This creates temporal windows from your fMRI scan and applies:
- Channel dimension addition
- Spatial padding/cropping to 96×96×96
- Whole-brain z-normalization
- Temporal windowing (20 frames with stride 10)
- Float16 conversion

### 3. Contrastive Pretraining

```bash
python 02_pretrain.py
```

Trains SwiFT encoder with contrastive learning:
- Uses dual-view temporal windows
- NTXent contrastive loss
- Saves pretrained encoder to `checkpoints/pretrained_encoder.pth`

### 4. Fine-tuning

```bash
python 03_finetune.py
```

Fine-tunes on downstream task:
- Loads pretrained encoder
- Adds classification/regression head
- Trains with task-specific loss
- Saves final model to `checkpoints/finetuned_model.pth`

## Configuration

### Pretraining Config (`configs/config_pretrain.py`)

```python
MODEL_CONFIG = {
    'embed_dim': 36,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': (4, 4, 4, 4),
    'patch_size': (6, 6, 6, 1),
    'c_multiplier': 2,
}

TRAIN_CONFIG = {
    'batch_size': 4,
    'learning_rate': 5e-5,
    'num_epochs': 100,
}
```

### Fine-tuning Config (`configs/config_finetune.py`)

```python
TASK_CONFIG = {
    'task_type': 'binary_classification',  # or 'regression'
    'num_classes': 2,
    'freeze_encoder': True,
}

TRAIN_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-5,
    'num_epochs': 50,
}
```

## Data Format

### Input
- Raw fMRI: `[B, H, W, D, T]` = `[B, 91, 109, 91, 140]`
- B: Batch size (number of scans)
- H, W, D: Spatial dimensions (height, width, depth)
- T: Temporal dimension (time points)

### Preprocessed
- SwiFT format: `[N, 1, 96, 96, 96, 20]`
- N: Number of temporal windows
- 1: Channel dimension
- 96×96×96: Padded/cropped spatial dimensions
- 20: Temporal window size

### Window Creation
- Window size: 20 frames
- Stride: 10 frames
- From 140 frames → 12 overlapping windows per scan

## Model Architecture

**SwiFT (Swin 4D Transformer)**:
- Hierarchical architecture with 4 stages
- Window-based 4D attention (spatial + temporal)
- Output dimension: 288 (36 × 2³)

**Dimension Flow**:
```
Input:      [B, 1, 96, 96, 96, 20]
PatchEmbed: [B, 36, 16, 16, 16, 20]
Stage 1:    [B, 36, 16, 16, 16, 20]
Stage 2:    [B, 72, 8, 8, 8, 20]
Stage 3:    [B, 144, 4, 4, 4, 20]
Stage 4:    [B, 288, 2, 2, 2, 20]
Pool+Head:  [B, num_classes]
```

## Memory Requirements

**With 1x A40 (48GB GPU)**:
- Batch size 4-6 for pretraining (contrastive)
- Batch size 8-12 for fine-tuning
- Mixed precision (float16) recommended

## Customization

### For Different Data Dimensions
Edit `configs/config_pretrain.py`:
```python
DATA_CONFIG = {
    'original_size': (YOUR_H, YOUR_W, YOUR_D, YOUR_T),
    'target_spatial_size': (96, 96, 96),  # or adjust
    'window_size': 20,  # temporal window
}
```

### For Different Tasks
Edit `configs/config_finetune.py`:
```python
TASK_CONFIG = {
    'task_type': 'regression',  # or 'binary_classification', 'multiclass_classification'
    'num_classes': 5,  # for multiclass
}
```

### For Different Architectures
Edit `MODEL_CONFIG` in `configs/config_pretrain.py`:
- `embed_dim`: Base embedding dimension
- `depths`: Blocks per stage
- `num_heads`: Attention heads per stage
- `window_size`: Attention window size

## Troubleshooting

### Out of Memory
- Reduce batch_size in configs
- Enable gradient checkpointing: `use_checkpoint=True`
- Use smaller window_size or stride

### Slow Training
- Increase batch_size if memory allows
- Use multiple GPUs
- Reduce num_epochs

### Poor Performance
- Try longer pretraining
- Adjust learning rate
- Modify data augmentations
- Collect more data

## Citation

SwiFT paper:
```
@article{swift2023,
  title={SwiFT: Swin 4D fMRI Transformer},
  author={...},
  journal={...},
  year={2023}
}
```

## License

Based on SwiFT implementation: https://github.com/Transconnectome/SwiFT
