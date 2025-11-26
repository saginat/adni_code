import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import collate_fn_corr, fmri_corr_dataset
from .preprocessing import load_and_process_data
from .transforms import NormalizeByRegion, Resize3D


def prepare_dataloaders(config, stage="pretrain"):
    """
    Prepare dataloaders for different training stages with consistent preprocessing.
    Args:
        config: Configuration object containing all hyperparameters
        stage: One of "pretrain", "finetune", or "tta"
    Returns:
        Dictionary containing dataloaders and other necessary components
    """
    print("Loading and preprocessing data...")

    # Load raw data
    train_set, val_set, test_set, imageID_to_labels, (full_4d, full_atlas), (scan_norm, region_norm) = (
        load_and_process_data(config)
    )
    train_data, regions_train, index_to_info_tr = train_set
    val_data, regions_val, index_to_info_val = val_set
    test_data, regions_test, index_to_info_test = test_set

    result = {
        "imageID_to_labels": imageID_to_labels,
        "train_info": index_to_info_tr,
        "val_info": index_to_info_val,
        "test_info": index_to_info_test,
    }
    det_transform = transforms.Compose(
    [
        
        scan_norm,
        Resize3D(scale_factor=0.7, align_corners=False),
    ]
)
    recon_transform = transforms.Compose([region_norm])

    if stage == "pretrain" or stage == "finetune":
        print(f"Creating datasets and dataloaders for {stage}...")

        dataset_tr = fmri_corr_dataset(
            train_data,
            index_to_info_tr,
            imageID_to_labels,
            det_transform=det_transform,
            custom_recon=regions_train,
            recon_transform=recon_transform,
        )

        dataset_val = fmri_corr_dataset(
            val_data,
            index_to_info_val,
            imageID_to_labels,
            det_transform=det_transform,
            custom_recon=regions_val,
            recon_transform=recon_transform,
        )

        dataset_test = fmri_corr_dataset(
            test_data,
            index_to_info_test,
            imageID_to_labels,
            det_transform=det_transform,
            custom_recon=regions_test,
            recon_transform=recon_transform,
        )

        train_dataloader = DataLoader(
            dataset_tr,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_corr,
        )
        val_dataloader = DataLoader(
            dataset_val,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_corr,
        )
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_corr,
        )

        result.update(
            {
                "train_dataloader": train_dataloader,
                "val_dataloader": val_dataloader,
                "test_dataloader": test_dataloader,
                "dataset_tr": dataset_tr,
                "dataset_val": dataset_val,
                "dataset_test": dataset_test,
            }
        )

    elif stage == "tta":
        print("Creating datasets and dataloaders for TTA...")

        # TTA only needs test data with custom_recon (atlas) for reconstruction loss
        dataset_test = fmri_corr_dataset(
            test_data,
            index_to_info_test,
            imageID_to_labels,
            det_transform=det_transform,
            custom_recon=regions_test,
            recon_transform=recon_transform,
        )

        test_dataloader = DataLoader(
            dataset_test,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_corr,
        )

        result.update(
            {
                "test_dataloader": test_dataloader,
                "dataset_test": dataset_test,
            }
        )

    else:
        raise ValueError(
            f"Unknown stage: {stage}. Must be one of 'pretrain', 'finetune', or 'tta'"
        )

    return result


def compute_num_patches_3d(input_shape, patch_size):
    """
    Compute the number of 3D patches given input shape and patch size.
    Moved here from run_pretraining.py for reusability.
    """
    H, W, D = input_shape
    pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
    pad_d = (patch_size[2] - D % patch_size[2]) % patch_size[2]
    return (
        ((H + pad_h) // patch_size[0])
        * ((W + pad_w) // patch_size[1])
        * ((D + pad_d) // patch_size[2])
    )
