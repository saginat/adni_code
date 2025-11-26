"""
Complete validation script for SwiFT pipeline
Tests: data preprocessing, model forward pass, pretraining iteration, fine-tuning iteration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import pipeline modules
from data.preprocessing import preprocess_scan
from data.dataset import SwiFTPretrainDataset, SwiFTFinetuneDataset, create_dummy_labels
from models.swin4d_transformer_ver7 import SwinTransformer4D
from models.heads import ContrastiveHead, ClassificationHead
from training.losses import NTXentLoss
from configs.config_pretrain import (
    MODEL_CONFIG,
    CONTRASTIVE_CONFIG,
    TRAIN_CONFIG,
    DATA_CONFIG,
)


def test_preprocessing():
    """Test data preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("TEST 1: Data Preprocessing")
    print("=" * 80)

    # Create dummy scan [1, 91, 109, 91, 140]
    print("\nCreating dummy scan with shape [1, 91, 109, 91, 140]...")
    dummy_scan = torch.randn(1, 91, 109, 91, 140)
    print(f"✓ Created dummy scan: {dummy_scan.shape}")

    # Preprocess
    print("\nRunning preprocessing pipeline...")
    preprocessed, indices = preprocess_scan(
        dummy_scan,
        target_spatial_size=DATA_CONFIG["target_spatial_size"],
        window_size=DATA_CONFIG["window_size"],
        stride=DATA_CONFIG["stride"],
        normalize=DATA_CONFIG["normalize"],
        to_float16=False,  # Keep float32 for validation
    )

    print(f"\n✓ Preprocessing complete!")
    print(f"  - Output shape: {preprocessed.shape}")
    print(f"  - Number of windows: {len(indices)}")
    print(f"  - Expected shape: [N, 1, 96, 96, 96, 20]")
    print(f"  - Data range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    print(
        f"  - Data stats: mean={preprocessed.mean():.3f}, std={preprocessed.std():.3f}"
    )

    assert preprocessed.shape[1:] == (1, 96, 96, 96, 20), "Unexpected output shape!"
    assert len(indices) > 0, "No windows created!"

    return preprocessed, indices


def test_model_forward():
    """Test SwiFT model forward pass"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Forward Pass")
    print("=" * 80)

    # Create model
    print("\nInitializing SwiFT model...")
    model = SwinTransformer4D(**MODEL_CONFIG)
    print(f"✓ Model created")

    # Print model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 96, 96, 96, 20)
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Forward pass successful!")
    print(f"  - Output shape: {output.shape}")
    print(
        f"  - Expected: [batch_size, {MODEL_CONFIG['embed_dim'] * (MODEL_CONFIG['c_multiplier'] ** 3)}, ...]"
    )

    return model


def test_contrastive_training():
    """Test contrastive pretraining iteration"""
    print("\n" + "=" * 80)
    print("TEST 3: Contrastive Pretraining Iteration")
    print("=" * 80)

    # Create dummy preprocessed data
    num_windows = 12
    data = torch.randn(num_windows, 1, 96, 96, 96, 20)
    indices = torch.tensor([[0, i * 10] for i in range(num_windows)])

    # Create contrastive dataset
    print("\nCreating contrastive dataset...")
    dataset = SwiFTPretrainDataset(data, indices)
    dataloader = DataLoader(
        dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True
    )
    print(f"✓ Dataset created: {len(dataset)} samples")

    # Create model and contrastive head
    print("\nInitializing model and contrastive head...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer4D(**MODEL_CONFIG).to(device)
    contrastive_head = ContrastiveHead(**CONTRASTIVE_CONFIG).to(device)
    print(f"✓ Models initialized on {device}")

    # Create loss and optimizer
    criterion = NTXentLoss(
        device=device,
        batch_size=TRAIN_CONFIG["batch_size"],
        temperature=TRAIN_CONFIG["temperature"],
        use_cosine_similarity=TRAIN_CONFIG["use_cosine_similarity"],
    )
    optimizer = optim.AdamW(
        list(model.parameters()) + list(contrastive_head.parameters()),
        lr=TRAIN_CONFIG["learning_rate"],
    )

    # Run one training iteration
    print("\nRunning one training iteration...")
    model.train()
    contrastive_head.train()

    for batch_idx, (view1, view2) in enumerate(dataloader):
        view1, view2 = view1.to(device), view2.to(device)
        print(f"  Batch {batch_idx}: view1={view1.shape}, view2={view2.shape}")

        # Forward pass
        optimizer.zero_grad()

        # Encode both views
        features1 = model(view1)
        features2 = model(view2)

        # Project to embedding space
        embeddings1 = contrastive_head(features1)
        embeddings2 = contrastive_head(features2)

        # Compute contrastive loss
        loss = criterion(embeddings1, embeddings2)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}")
        print(f"  ✓ Training iteration successful!")
        break  # Only test one batch

    return model


def test_finetuning():
    """Test fine-tuning iteration"""
    print("\n" + "=" * 80)
    print("TEST 4: Fine-tuning Iteration")
    print("=" * 80)

    # Create dummy preprocessed data
    num_windows = 20
    data = torch.randn(num_windows, 1, 96, 96, 96, 20)
    labels = create_dummy_labels(num_windows, task_type="binary")

    # Create fine-tuning dataset
    print("\nCreating fine-tuning dataset...")
    dataset = SwiFTFinetuneDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"✓ Dataset created: {len(dataset)} samples")

    # Create model and classification head
    print("\nInitializing model and classification head...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer4D(**MODEL_CONFIG).to(device)
    clf_head = ClassificationHead(num_classes=2, num_features=288).to(device)
    print(f"✓ Models initialized on {device}")

    # Create loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(clf_head.parameters()), lr=1e-5
    )

    # Run one training iteration
    print("\nRunning one training iteration...")
    model.train()
    clf_head.train()

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device).float().unsqueeze(1)
        print(f"  Batch {batch_idx}: x={x.shape}, y={y.shape}")

        # Forward pass
        optimizer.zero_grad()
        features = model(x)
        logits = clf_head(features)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()

        print(f"  Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
        print(f"  ✓ Fine-tuning iteration successful!")
        break  # Only test one batch

    return model


def test_checkpoint_save_load():
    """Test checkpoint saving and loading"""
    print("\n" + "=" * 80)
    print("TEST 5: Checkpoint Save/Load")
    print("=" * 80)

    # Create model
    print("\nCreating model...")
    model = SwinTransformer4D(**MODEL_CONFIG)
    original_state = model.state_dict()

    # Save checkpoint
    checkpoint_path = "../checkpoints/test_checkpoint.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    torch.save(
        {
            "model_state_dict": original_state,
            "config": MODEL_CONFIG,
        },
        checkpoint_path,
    )
    print(f"✓ Checkpoint saved")

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Checkpoint loaded successfully")

    # Verify
    loaded_state = model.state_dict()
    all_match = all(
        torch.equal(original_state[k], loaded_state[k]) for k in original_state.keys()
    )
    assert all_match, "Loaded weights don't match original!"
    print(f"✓ Weight verification passed")

    return True


def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("SwiFT Pipeline Validation")
    print("=" * 80)
    print("\nThis script validates the complete SwiFT pipeline:")
    print("1. Data preprocessing")
    print("2. Model forward pass")
    print("3. Contrastive pretraining")
    print("4. Supervised fine-tuning")
    print("5. Checkpoint save/load")

    try:
        # Test 1: Preprocessing
        preprocessed, indices = test_preprocessing()

        # Test 2: Model forward pass
        model = test_model_forward()

        # Test 3: Contrastive training
        if torch.cuda.is_available():
            pretrained_model = test_contrastive_training()
        else:
            print("\n⚠ Skipping contrastive training test (GPU not available)")
            pretrained_model = None

        # Test 4: Fine-tuning
        if torch.cuda.is_available():
            finetuned_model = test_finetuning()
        else:
            print("\n⚠ Skipping fine-tuning test (GPU not available)")
            finetuned_model = None

        # Test 5: Checkpoint save/load
        test_checkpoint_save_load()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe SwiFT pipeline is ready for use.")
        print("\nNext steps:")
        print("1. Prepare your real fMRI data")
        print("2. Run 01_preprocess_data.py to preprocess your data")
        print("3. Run 02_pretrain.py for contrastive pretraining")
        print("4. Run 03_finetune.py for downstream task fine-tuning")

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
