import torch
import numpy as np
import json
from typing import Tuple, Union
from .transforms import NormalizeByRegion

class DataSplitter:
    """A class for splitting a dataset into training, validation, and test sets."""

    def __init__(
        self, data_tensor: torch.Tensor, val_split: float, test_split: float, seed: int
    ):
        self.data = data_tensor
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.indices = torch.randperm(
            self.data.size(0), generator=torch.Generator().manual_seed(self.seed)
        )

        total_samples = len(self.indices)
        self.test_size = int(total_samples * self.test_split)
        self.val_size = int(total_samples * self.val_split)
        self.train_size = total_samples - self.test_size - self.val_size

    def split_data(
        self,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Splits data indices."""
        print(
            f"Train samples: {self.train_size} ({self.train_size / len(self.indices):.2%})"
        )
        if self.val_size > 0:
            print(
                f"Validation samples: {self.val_size} ({self.val_size / len(self.indices):.2%})"
            )
        print(
            f"Test samples: {self.test_size} ({self.test_size / len(self.indices):.2%})"
        )

        train_indices = self.indices[: self.train_size].numpy()

        if self.val_size > 0:
            val_indices = self.indices[
                self.train_size : self.train_size + self.val_size
            ].numpy()
            test_indices = self.indices[self.train_size + self.val_size :].numpy()
            return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)
        else:
            test_indices = self.indices[self.train_size :].numpy()
            return np.sort(train_indices), np.sort(test_indices)


class TimeWindowSplitter:
    """Splits time-series data into windows."""

    def __init__(self, data: torch.Tensor, window_size: int):
        self.data = data
        self.window_size = window_size

    def split(self) -> torch.Tensor:
        """Performs the split."""
        n_samples, h, w, d, t = self.data.shape
        if t % self.window_size != 0:
            raise ValueError(
                f"Time dimension ({t}) must be divisible by window size ({self.window_size})."
            )

        num_windows = t // self.window_size

        # Reshape and permute to create windows
        # [N, H, W, D, T] -> [N, H, W, D, num_windows, window_size]
        data = self.data.view(n_samples, h, w, d, num_windows, self.window_size)
        # -> [N, num_windows, H, W, D, window_size]
        data = data.permute(0, 4, 1, 2, 3, 5).contiguous()
        # -> [N * num_windows, H, W, D, window_size]
        data = data.view(-1, h, w, d, self.window_size)

        return data

    @staticmethod
    def update_info_dict(info_dict: dict, original_len: int, window_size: int) -> dict:
        """Updates the info dictionary to reflect the windowed data."""
        num_windows = original_len // window_size
        new_info_dict = {}
        original_keys = sorted(info_dict.keys())

        new_idx = 0
        for old_idx in original_keys:
            info = info_dict[old_idx]
            for i in range(num_windows):
                new_info = info.copy()
                new_info["window_index"] = i
                new_info_dict[new_idx] = new_info
                new_idx += 1
        return new_info_dict


def load_and_process_data(config, atlas_name=None, default_final_run=True):
    """Main function to load, preprocess, and split the data.
    
    Args:
        config: Configuration object
        atlas_name: Optional atlas name (e.g., 'schaefer100', 'schaefer200'). 
                   If None, uses the old schaefer atlas file for backward compatibility.
    """
    import psutil
    import gc
    
    def print_memory_usage(stage):
        """Print current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1e9
        available_gb = psutil.virtual_memory().available / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        print(f"[{stage}] RAM: {mem_gb:.2f}GB used | {available_gb:.2f}GB available | {total_gb:.2f}GB total")
    
    print_memory_usage("START")

    with open(f"{config.BASE_DATA_PATH}/index_to_name.json", "r") as f:
        index_to_info = json.load(f)
    index_to_info = {int(k): v for k, v in index_to_info.items()}

    with open(f"{config.BASE_DATA_PATH}/imageID_to_labels.json", "r") as f:
        imageID_to_labels = json.load(f)
    
    print_memory_usage("After loading JSON metadata")

    all_data_4d = torch.load(f"{config.BASE_DATA_PATH}/all_4d_downsampled.pt")
    print_memory_usage(f"After loading 4D data - shape: {all_data_4d.shape}")
    
    # Load atlas based on atlas_name parameter
    if atlas_name is not None:
        atlas_path = f"{config.BASE_DATA_PATH}/atlas_{atlas_name}.pt"
        schaefer_atlas = torch.load(atlas_path)
        print(f"Loaded atlas: {atlas_name} from {atlas_path}")
        print_memory_usage(f"After loading atlas - shape: {schaefer_atlas.shape}")
    else:
        # Backward compatibility: use old schaefer file
        schaefer_atlas = torch.load(
            f"{config.BASE_DATA_PATH}/time_regions_tensor_not_normalized_schaefer.pt"
        )
        schaefer_atlas = schaefer_atlas.permute(0, 2, 1)  # samples, regions, time
        print_memory_usage(f"After loading legacy atlas - shape: {schaefer_atlas.shape}")

    std_data = np.std(all_data_4d.numpy(), axis=tuple(range(1, all_data_4d.data.ndim)))
    top_k_std_scans = np.argsort(std_data)[-config.REMOVE_TOP_K_STD :]
    mask_bad = np.isin(np.arange(all_data_4d.size(0)), top_k_std_scans)

    clean_indices = np.where(~mask_bad)[0]
    print_memory_usage("After computing outliers")
    
    all_data_4d = all_data_4d[clean_indices]
    schaefer_atlas = schaefer_atlas[clean_indices]
    print_memory_usage(f"After filtering outliers - 4D: {all_data_4d.shape}, Atlas: {schaefer_atlas.shape}")

    index_to_info = {
        i: index_to_info[clean_idx] for i, clean_idx in enumerate(clean_indices)
    }

    splitter = DataSplitter(
        all_data_4d, config.VAL_SPLIT, config.TEST_SPLIT, config.SEED
    )
    if not default_final_run:
        train_indices, val_indices, test_indices = splitter.split_data()
        print_memory_usage("After splitting indices")

        # Create normalizers BEFORE splitting data (they need the full dataset)
        region_normalize_4d = NormalizeByRegion(all_data_4d)
        region_normalize_atlas = NormalizeByRegion(schaefer_atlas)
        print_memory_usage("After creating normalizers")

        # Keep references to full data for return
        full_4d = all_data_4d
        full_atlas = schaefer_atlas

        train_data, val_data, test_data = (
            all_data_4d[train_indices],
            all_data_4d[val_indices],
            all_data_4d[test_indices],
        )
        regions_train, regions_val, regions_test = (
            schaefer_atlas[train_indices],
            schaefer_atlas[val_indices],
            schaefer_atlas[test_indices],
        )
        print_memory_usage("After splitting data into train/val/test")

        # Create index mappings for each split
        index_to_info_tr = {i: index_to_info[idx] for i, idx in enumerate(train_indices)}
        index_to_info_val = {i: index_to_info[idx] for i, idx in enumerate(val_indices)}
        index_to_info_test = {i: index_to_info[idx] for i, idx in enumerate(test_indices)}

        # Apply windowing
        print_memory_usage("Before windowing")
        original_time_len = train_data.shape[-1]
        train_data = TimeWindowSplitter(train_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing train_data - shape: {train_data.shape}")
        
        val_data = TimeWindowSplitter(val_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing val_data - shape: {val_data.shape}")
        
        test_data = TimeWindowSplitter(test_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing test_data - shape: {test_data.shape}")
        
        regions_train = (
            TimeWindowSplitter(regions_train.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_train - shape: {regions_train.shape}")
        
        regions_val = (
            TimeWindowSplitter(regions_val.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_val - shape: {regions_val.shape}")
        
        regions_test = (
            TimeWindowSplitter(regions_test.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_test - shape: {regions_test.shape}")

        index_to_info_tr = TimeWindowSplitter.update_info_dict(
            index_to_info_tr, original_time_len, config.WINDOW_SIZE
        )
        index_to_info_val = TimeWindowSplitter.update_info_dict(
            index_to_info_val, original_time_len, config.WINDOW_SIZE
        )
        index_to_info_test = TimeWindowSplitter.update_info_dict(
            index_to_info_test, original_time_len, config.WINDOW_SIZE
        )

        print_memory_usage("Before returning")
        gc.collect()
        print_memory_usage("After garbage collection")

        return (
            (train_data, regions_train, index_to_info_tr),
            (val_data, regions_val, index_to_info_val),
            (test_data, regions_test, index_to_info_test),
            imageID_to_labels,
            (full_4d, full_atlas),
            (region_normalize_4d, region_normalize_atlas)
        )

    else:
        train_indices,val_indices, test_indices = splitter.split_data()
        train_indices = np.concatenate([train_indices, val_indices])
        print_memory_usage("After splitting indices")

        # Create normalizers BEFORE splitting data (they need the full dataset)
        region_normalize_4d = NormalizeByRegion(all_data_4d)
        region_normalize_atlas = NormalizeByRegion(schaefer_atlas)
        print_memory_usage("After creating normalizers")

        # Keep references to full data for return
        full_4d = all_data_4d
        full_atlas = schaefer_atlas

        train_data, val_data, test_data = (
            all_data_4d[train_indices],
            all_data_4d[test_indices],
            all_data_4d[test_indices],
        )
        regions_train, regions_val, regions_test = (
            schaefer_atlas[train_indices],
            schaefer_atlas[test_indices],
            schaefer_atlas[test_indices],
        )
        print_memory_usage("After splitting data into train/val/test")

        # Create index mappings for each split
        index_to_info_tr = {i: index_to_info[idx] for i, idx in enumerate(train_indices)}
        index_to_info_val = {i: index_to_info[idx] for i, idx in enumerate(test_indices)}
        index_to_info_test = {i: index_to_info[idx] for i, idx in enumerate(test_indices)}

        # Apply windowing
        print_memory_usage("Before windowing")
        original_time_len = train_data.shape[-1]
        train_data = TimeWindowSplitter(train_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing train_data - shape: {train_data.shape}")
        
        val_data = TimeWindowSplitter(val_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing val_data - shape: {val_data.shape}")
        
        test_data = TimeWindowSplitter(test_data, config.WINDOW_SIZE).split()
        print_memory_usage(f"After windowing test_data - shape: {test_data.shape}")
        
        regions_train = (
            TimeWindowSplitter(regions_train.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_train - shape: {regions_train.shape}")
        
        regions_val = (
            TimeWindowSplitter(regions_val.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_val - shape: {regions_val.shape}")
        
        regions_test = (
            TimeWindowSplitter(regions_test.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
            .split()
            .squeeze()
        )
        print_memory_usage(f"After windowing regions_test - shape: {regions_test.shape}")

        index_to_info_tr = TimeWindowSplitter.update_info_dict(
            index_to_info_tr, original_time_len, config.WINDOW_SIZE
        )
        index_to_info_val = TimeWindowSplitter.update_info_dict(
            index_to_info_val, original_time_len, config.WINDOW_SIZE
        )
        index_to_info_test = TimeWindowSplitter.update_info_dict(
            index_to_info_test, original_time_len, config.WINDOW_SIZE
        )

        print_memory_usage("Before returning")
        gc.collect()
        print_memory_usage("After garbage collection")

        return (
            (train_data, regions_train, index_to_info_tr),
            (val_data, regions_val, index_to_info_val),
            (test_data, regions_test, index_to_info_test),
            imageID_to_labels,
            (full_4d, full_atlas),
            (region_normalize_4d, region_normalize_atlas)
        )

       

    

# Configuration for atlas pretraining experiment