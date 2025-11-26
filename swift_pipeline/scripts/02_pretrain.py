"""
Simplified contrastive pretraining script for SwiFT
For detailed implementation with full training loop, see the SwiFT repository
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.preprocessing import load_preprocessed_data
from data.dataset import SwiFTPretrainDataset
from models.swin4d_transformer_ver7 import SwinTransformer4D
from models.heads import ContrastiveHead
from training.losses import NTXentLoss
from configs.config_pretrain import (
    MODEL_CONFIG,
    CONTRASTIVE_CONFIG,
    TRAIN_CONFIG,
    OPTIMIZER_CONFIG,
)


def train_epoch(
    model, contrastive_head, dataloader, criterion, optimizer, device, epoch
):
    """Train for one epoch"""
    model.train()
    contrastive_head.train()
    total_loss = 0.0

    for batch_idx, (view1, view2) in enumerate(dataloader):
        view1, view2 = view1.to(device), view2.to(device)

        # Forward pass
        optimizer.zero_grad()
        features1 = model(view1)
        features2 = model(view2)
        embeddings1 = contrastive_head(features1)
        embeddings2 = contrastive_head(features2)

        # Compute loss
        loss = criterion(embeddings1, embeddings2)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    print("=" * 80)
    print("SwiFT Contrastive Pretraining")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data_path = "./data/preprocessed/preprocessed_data.pt"
    data, indices = load_preprocessed_data(data_path)
    print(f"✓ Loaded {len(data)} windows")

    # Create dataset
    print("\nCreating contrastive dataset...")
    dataset = SwiFTPretrainDataset(data, indices, window_size=20)
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
    )
    print(f"✓ Dataset ready: {len(dataset)} samples")

    # Initialize models
    print("\nInitializing models...")
    model = SwinTransformer4D(**MODEL_CONFIG).to(device)
    contrastive_head = ContrastiveHead(**CONTRASTIVE_CONFIG).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in contrastive_head.parameters()
    )
    print(f"✓ Total parameters: {total_params:,}")

    # Setup loss and optimizer
    criterion = NTXentLoss(
        device=device,
        batch_size=TRAIN_CONFIG["batch_size"],
        temperature=TRAIN_CONFIG["temperature"],
        use_cosine_similarity=TRAIN_CONFIG["use_cosine_similarity"],
    )

    optimizer = optim.AdamW(
        list(model.parameters()) + list(contrastive_head.parameters()),
        lr=OPTIMIZER_CONFIG["lr"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        betas=OPTIMIZER_CONFIG["betas"],
    )

    # Training loop
    print(f"\nStarting training for {TRAIN_CONFIG['num_epochs']} epochs...")
    print("=" * 80)

    for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
        avg_loss = train_epoch(
            model, contrastive_head, dataloader, criterion, optimizer, device, epoch
        )
        print(
            f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']}, Average Loss: {avg_loss:.4f}"
        )
        print("-" * 80)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = f"../checkpoints/pretrain_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = "../checkpoints/pretrained_encoder.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": MODEL_CONFIG,
        },
        final_path,
    )
    print(f"\n✓ Final model saved: {final_path}")
    print("\nPretraining complete!")
    print("Next step: Run 03_finetune.py for downstream task fine-tuning")


if __name__ == "__main__":
    main()
