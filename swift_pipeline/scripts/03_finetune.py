"""
Simplified fine-tuning script for SwiFT
For detailed implementation with full training loop, see the SwiFT repository
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.preprocessing import load_preprocessed_data
from data.dataset import SwiFTFinetuneDataset, create_dummy_labels
from models.swin4d_transformer_ver7 import SwinTransformer4D
from models.heads import ClassificationHead
from configs.config_finetune import (
    MODEL_CONFIG,
    HEAD_CONFIG,
    TASK_CONFIG,
    TRAIN_CONFIG,
    OPTIMIZER_CONFIG,
    PATHS,
)


def train_epoch(model, head, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    head.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        if TASK_CONFIG["task_type"] == "binary_classification":
            y = y.float().unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        features = model(x)
        logits = head(features)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy for classification
        if "classification" in TASK_CONFIG["task_type"]:
            if TASK_CONFIG["task_type"] == "binary_classification":
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    print("=" * 80)
    print("SwiFT Fine-tuning")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data_path = "./data/preprocessed/preprocessed_data.pt"
    data, indices = load_preprocessed_data(data_path)

    # Create dummy labels for demonstration
    # TODO: Replace with your actual labels
    print("\nCreating dummy labels (replace with real labels)...")
    labels = create_dummy_labels(
        len(data), task_type=TASK_CONFIG["task_type"].replace("_classification", "")
    )

    # Create dataset
    print("\nCreating fine-tuning dataset...")
    dataset = SwiFTFinetuneDataset(data, labels, indices)
    dataloader = DataLoader(
        dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=0
    )
    print(f"✓ Dataset ready: {len(dataset)} samples")

    # Initialize model
    print("\nInitializing model...")
    model = SwinTransformer4D(**MODEL_CONFIG).to(device)

    # Load pretrained weights if available
    pretrained_path = PATHS["pretrained_encoder"]
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Pretrained weights loaded")
    else:
        print(f"⚠ Pretrained weights not found at {pretrained_path}")
        print("  Training from scratch...")

    # Freeze encoder if specified
    if TASK_CONFIG["freeze_encoder"]:
        print("\nFreezing encoder weights...")
        for param in model.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")

    # Initialize task head
    if TASK_CONFIG["task_type"] in [
        "binary_classification",
        "multiclass_classification",
    ]:
        head = ClassificationHead(**HEAD_CONFIG).to(device)
    else:
        from models.heads import RegressionHead

        head = RegressionHead(num_features=HEAD_CONFIG["num_features"]).to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) + sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Setup loss
    if TASK_CONFIG["task_type"] == "binary_classification":
        criterion = nn.BCEWithLogitsLoss()
    elif TASK_CONFIG["task_type"] == "multiclass_classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Setup optimizer
    params = list(head.parameters())
    if not TASK_CONFIG["freeze_encoder"]:
        params += list(model.parameters())

    optimizer = optim.AdamW(
        params,
        lr=OPTIMIZER_CONFIG["lr"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        betas=OPTIMIZER_CONFIG["betas"],
    )

    # Training loop
    print(f"\nStarting fine-tuning for {TRAIN_CONFIG['num_epochs']} epochs...")
    print("=" * 80)

    best_acc = 0.0
    for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
        avg_loss, accuracy = train_epoch(
            model, head, dataloader, criterion, optimizer, device, epoch
        )
        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']}")
        print(f"  Average Loss: {avg_loss:.4f}")
        if "classification" in TASK_CONFIG["task_type"]:
            print(f"  Accuracy: {accuracy:.4f}")
        print("-" * 80)

        # Save best model
        if "classification" in TASK_CONFIG["task_type"] and accuracy > best_acc:
            best_acc = accuracy
            best_path = "../checkpoints/best_finetuned_model.pth"
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                    "loss": avg_loss,
                },
                best_path,
            )
            print(f"✓ Best model saved: {best_path} (accuracy: {accuracy:.4f})")

    # Save final model
    final_path = "../checkpoints/finetuned_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "config": MODEL_CONFIG,
            "task_config": TASK_CONFIG,
        },
        final_path,
    )
    print(f"\n✓ Final model saved: {final_path}")
    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
