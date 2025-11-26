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


def load_and_process_data(config):
    """Main function to load, preprocess, and split the data."""

    with open(f"{config.BASE_DATA_PATH}/index_to_name.json", "r") as f:
        index_to_info = json.load(f)
    index_to_info = {int(k): v for k, v in index_to_info.items()}

    with open(f"{config.BASE_DATA_PATH}/imageID_to_labels.json", "r") as f:
        imageID_to_labels = json.load(f)

    all_data_4d = torch.load(f"{config.BASE_DATA_PATH}/data/all_4d_downsampled.pt")
    schaefer_atlas = torch.load(
        f"{config.BASE_DATA_PATH}/data/time_regions_tensor_not_normalized_schaefer.pt"
    )
    schaefer_atlas = schaefer_atlas.permute(0, 2, 1)  # samples, regions, time

    std_data = np.std(all_data_4d.numpy(), axis=tuple(range(1, all_data_4d.data.ndim)))
    top_k_std_scans = np.argsort(std_data)[-config.REMOVE_TOP_K_STD :]
    mask_bad = np.isin(np.arange(all_data_4d.size(0)), top_k_std_scans)

    clean_indices = np.where(~mask_bad)[0]
    all_data_4d = all_data_4d[clean_indices]
    schaefer_atlas = schaefer_atlas[clean_indices]

    index_to_info = {
        i: index_to_info[clean_idx] for i, clean_idx in enumerate(clean_indices)
    }

    splitter = DataSplitter(
        all_data_4d, config.VAL_SPLIT, config.TEST_SPLIT, config.SEED
    )
    train_indices, val_indices, test_indices = splitter.split_data()

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

    index_to_info_tr = {i: index_to_info[idx] for i, idx in enumerate(train_indices)}
    index_to_info_val = {i: index_to_info[idx] for i, idx in enumerate(val_indices)}
    index_to_info_test = {i: index_to_info[idx] for i, idx in enumerate(test_indices)}

    region_normalize_4d = NormalizeByRegion(all_data_4d)
    region_normalize_atlas = NormalizeByRegion(schaefer_atlas)

    # Apply windowing
    original_time_len = train_data.shape[-1]
    train_data = TimeWindowSplitter(train_data, config.WINDOW_SIZE).split()
    val_data = TimeWindowSplitter(val_data, config.WINDOW_SIZE).split()
    test_data = TimeWindowSplitter(test_data, config.WINDOW_SIZE).split()
    regions_train = (
        TimeWindowSplitter(regions_train.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )
    regions_val = (
        TimeWindowSplitter(regions_val.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )
    regions_test = (
        TimeWindowSplitter(regions_test.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )

    index_to_info_tr = TimeWindowSplitter.update_info_dict(
        index_to_info_tr, original_time_len, config.WINDOW_SIZE
    )
    index_to_info_val = TimeWindowSplitter.update_info_dict(
        index_to_info_val, original_time_len, config.WINDOW_SIZE
    )
    index_to_info_test = TimeWindowSplitter.update_info_dict(
        index_to_info_test, original_time_len, config.WINDOW_SIZE
    )

    return (
        (train_data, regions_train, index_to_info_tr),
        (val_data, regions_val, index_to_info_val),
        (test_data, regions_test, index_to_info_test),
        imageID_to_labels,
        (all_data_4d, schaefer_atlas),
        (region_normalize_4d, region_normalize_atlas)
    )
