"""
Validate SwiFT pipeline with ADNI data for degradation prediction

This script demonstrates:
1. Loading ADNI data with degradation labels
2. Contrastive pretraining (self-supervised)
3. Fine-tuning for degradation prediction task
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# SwiFT pipeline imports
from data.prepare_adni_data import prepare_adni_datasets, create_adni_dataloaders
from models.swin4d_transformer_ver7 import SwinTransformer4D
from models.heads import ContrastiveHead, ClassificationHead
from training.losses import NTXentLoss
from configs.config_pretrain import (
    MODEL_CONFIG,
    CONTRASTIVE_CONFIG,
    TRAIN_CONFIG,
    DATA_CONFIG,
)
from configs.config_finetune import (
    TASK_CONFIG,
    HEAD_CONFIG,
    TRAIN_CONFIG as FINETUNE_TRAIN_CONFIG,
)

print("=" * 80)
print("SWIFT PIPELINE WITH ADNI DEGRADATION DATA")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to your ADNI data
BASE_DATA_PATH = "../../"  # Adjust to your data location
DATA_PATH = f"{BASE_DATA_PATH}/data/all_4d_downsampled.pt"
LABELS_PATH = f"{BASE_DATA_PATH}/imageID_to_labels.json"
INFO_PATH = f"{BASE_DATA_PATH}/index_to_name.json"

# Task configuration
DEGRADATION_TASKS = [
    "degradation_binary_1year",
    # "degradation_binary_2years",
    # "degradation_binary_3years",
]

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_EPOCHS = 5  # Increase for real training
FINETUNE_EPOCHS = 10  # Increase for real training

print(f"\nDevice: {DEVICE}")
print(f"Tasks: {DEGRADATION_TASKS}")

# ============================================================================
# STAGE 1: CONTRASTIVE PRETRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STAGE 1: CONTRASTIVE PRETRAINING (Self-Supervised)")
print("=" * 80)

# Prepare datasets for pretraining
pretrain_datasets = prepare_adni_datasets(
    data_path=DATA_PATH,
    labels_path=LABELS_PATH,
    info_path=INFO_PATH,
    task_names=DEGRADATION_TASKS,  # Not used in pretraining, but needed for data loading
    stage="pretrain",
    val_split=0.1,
    test_split=0.2,
    seed=42,
)

# Create dataloaders
pretrain_loaders = create_adni_dataloaders(
    pretrain_datasets,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle_train=True,
)

train_loader = pretrain_loaders["train_loader"]
val_loader = pretrain_loaders["val_loader"]

print(f"\nDataloaders created:")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")

# Initialize models
print("\nInitializing models for pretraining...")
encoder = SwinTransformer4D(**MODEL_CONFIG).to(DEVICE)
contrastive_head = ContrastiveHead(**CONTRASTIVE_CONFIG).to(DEVICE)

# Setup training
criterion = NTXentLoss(
    device=DEVICE,
    batch_size=TRAIN_CONFIG["batch_size"],
    temperature=TRAIN_CONFIG["temperature"],
)
optimizer = optim.AdamW(
    list(encoder.parameters()) + list(contrastive_head.parameters()),
    lr=TRAIN_CONFIG["learning_rate"],
)

print("✓ Models and optimizer initialized")

# Training loop
print(f"\nStarting pretraining ({PRETRAIN_EPOCHS} epochs)...")
encoder.train()
contrastive_head.train()

for epoch in range(PRETRAIN_EPOCHS):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (view1, view2) in enumerate(train_loader):
        view1, view2 = view1.to(DEVICE), view2.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()

        # Encode both views
        features1 = encoder(view1)
        features2 = encoder(view2)

        # Project to embedding space
        embeddings1 = contrastive_head(features1)
        embeddings2 = contrastive_head(features2)

        # Compute contrastive loss
        loss = criterion(embeddings1, embeddings2)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        # Limit batches for demo (remove for real training)
        if num_batches >= 10:
            break

    avg_loss = epoch_loss / num_batches
    print(f"  Epoch {epoch + 1}/{PRETRAIN_EPOCHS}: Loss = {avg_loss:.4f}")

print("\n✓ Pretraining complete!")

# Save pretrained model
pretrained_path = "../checkpoints/pretrained_adni.pth"
os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
torch.save(
    {
        "model_state_dict": encoder.state_dict(),
        "config": MODEL_CONFIG,
    },
    pretrained_path,
)
print(f"✓ Saved pretrained encoder to: {pretrained_path}")

# ============================================================================
# STAGE 2: SUPERVISED FINE-TUNING FOR DEGRADATION PREDICTION
# ============================================================================

print("\n" + "=" * 80)
print("STAGE 2: SUPERVISED FINE-TUNING (Degradation Prediction)")
print("=" * 80)

# Prepare datasets for finetuning
finetune_datasets = prepare_adni_datasets(
    data_path=DATA_PATH,
    labels_path=LABELS_PATH,
    info_path=INFO_PATH,
    task_names=DEGRADATION_TASKS,
    stage="finetune",
    val_split=0.1,
    test_split=0.2,
    seed=42,
    handle_nan="skip",  # Skip samples with NaN labels
)

# Create dataloaders
finetune_loaders = create_adni_dataloaders(
    finetune_datasets,
    batch_size=FINETUNE_TRAIN_CONFIG["batch_size"],
    shuffle_train=True,
)

train_loader_ft = finetune_loaders["train_loader"]
val_loader_ft = finetune_loaders["val_loader"]
test_loader_ft = finetune_loaders["test_loader"]

print(f"\nDataloaders created:")
print(f"  - Train batches: {len(train_loader_ft)}")
print(f"  - Val batches: {len(val_loader_ft)}")
print(f"  - Test batches: {len(test_loader_ft)}")

# Load pretrained encoder
print("\nLoading pretrained encoder...")
finetuned_encoder = SwinTransformer4D(**MODEL_CONFIG).to(DEVICE)
checkpoint = torch.load(pretrained_path, map_location=DEVICE)
finetuned_encoder.load_state_dict(checkpoint["model_state_dict"])
print("✓ Loaded pretrained weights")

# Freeze encoder if specified
if TASK_CONFIG["freeze_encoder"]:
    print("Freezing encoder weights...")
    for param in finetuned_encoder.parameters():
        param.requires_grad = False
    finetuned_encoder.eval()

# Create classification head
num_tasks = len(DEGRADATION_TASKS)
classification_head = ClassificationHead(
    num_classes=2,  # Binary classification
    num_features=HEAD_CONFIG["num_features"],
).to(DEVICE)

print(f"✓ Classification head created for {num_tasks} task(s)")

# Setup training
if TASK_CONFIG["freeze_encoder"]:
    params_to_optimize = classification_head.parameters()
else:
    params_to_optimize = list(finetuned_encoder.parameters()) + list(
        classification_head.parameters()
    )

optimizer_ft = optim.AdamW(
    params_to_optimize,
    lr=FINETUNE_TRAIN_CONFIG["learning_rate"],
    weight_decay=FINETUNE_TRAIN_CONFIG["weight_decay"],
)

# Loss function (binary classification)
criterion_ft = nn.BCEWithLogitsLoss()

print("✓ Optimizer and loss function initialized")

# Training loop
print(f"\nStarting fine-tuning ({FINETUNE_EPOCHS} epochs)...")
classification_head.train()

best_val_acc = 0.0

for epoch in range(FINETUNE_EPOCHS):
    # Training
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader_ft):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().squeeze()

        # Forward pass
        optimizer_ft.zero_grad()

        with torch.set_grad_enabled(not TASK_CONFIG["freeze_encoder"]):
            features = finetuned_encoder(inputs)

        outputs = classification_head(features).squeeze()

        # Compute loss
        loss = criterion_ft(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer_ft.step()

        # Metrics
        train_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    classification_head.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader_ft:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().squeeze()

            features = finetuned_encoder(inputs)
            outputs = classification_head(features).squeeze()

            loss = criterion_ft(outputs, labels)

            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    # Calculate metrics
    train_loss /= len(train_loader_ft)
    train_acc = 100 * train_correct / train_total
    val_loss /= len(val_loader_ft)
    val_acc = 100 * val_correct / val_total

    print(
        f"  Epoch {epoch + 1}/{FINETUNE_EPOCHS}: "
        f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
        f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1

    classification_head.train()

print(f"\n✓ Fine-tuning complete!")
print(f"  Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION ON TEST SET")
print("=" * 80)

classification_head.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader_ft:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().squeeze()

        features = finetuned_encoder(inputs)
        outputs = classification_head(features).squeeze()

        loss = criterion_ft(outputs, labels)

        test_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader_ft)
test_acc = 100 * test_correct / test_total

print(f"\nTest Results:")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - Accuracy: {test_acc:.2f}%")
print(f"  - Correct: {test_correct}/{test_total}")

# Save final model
final_path = "../checkpoints/finetuned_degradation.pth"
torch.save(
    {
        "encoder_state_dict": finetuned_encoder.state_dict(),
        "head_state_dict": classification_head.state_dict(),
        "config": {
            "model_config": MODEL_CONFIG,
            "task_config": TASK_CONFIG,
            "tasks": DEGRADATION_TASKS,
        },
        "results": {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
        },
    },
    final_path,
)

print(f"\n✓ Saved final model to: {final_path}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
print("\nSummary:")
print(
    f"  - Pretraining: Learned representations from {len(pretrain_datasets['train_dataset'])} samples"
)
print(
    f"  - Fine-tuning: Trained on {len(finetune_datasets['train_dataset'])} labeled samples"
)
print(f"  - Task: {DEGRADATION_TASKS[0]}")
print(f"  - Best validation accuracy: {best_val_acc:.2f}%")
print(f"  - Test accuracy: {test_acc:.2f}%")
print("\n" + "=" * 80)
