import torch
from torch.utils.data import Dataset


class fmri_corr_dataset(Dataset):
    def __init__(
        self,
        data,
        index_to_info,
        imageID_to_labels,
        det_transform=None,
        rnd_transform=None,
        custom_recon=None,
        recon_transform=None,
        aug_probability=0.0,
    ):
        self.det_transform = det_transform
        self.data = self.det_transform(data) if self.det_transform else data
        self.index_to_info = index_to_info
        self.imageID_to_labels = imageID_to_labels
        self.rnd_transform = rnd_transform
        self.custom_recon = (
            recon_transform(custom_recon)
            if recon_transform is not None and recon_transform is not None
            else custom_recon
        )
        self.aug_probability = aug_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        if self.rnd_transform and torch.rand(1).item() < self.aug_probability:
            sample_x = self.rnd_transform(sample_x)

        image_id = self.index_to_info[idx]["image_id"]
        labels_dict = self.imageID_to_labels.get(image_id, {})
        labels_dict["scan_id"] = image_id

        recon = self.custom_recon[idx] if self.custom_recon is not None else None

        return sample_x, labels_dict, recon


def collate_fn_corr(batch):
    data_samples = [item[0] for item in batch]
    labels_dicts = [item[1] for item in batch]
    custom_recons = [item[2] for item in batch]

    data_batch = torch.stack(data_samples)
    custom_recon_batch = (
        torch.stack(custom_recons) if custom_recons[0] is not None else None
    )

    batched_labels = {}
    if labels_dicts:
        for key in labels_dicts[0].keys():
            if isinstance(labels_dicts[0][key], str):
                batched_labels[key] = [d[key] for d in labels_dicts]
            else:
                # Pad with NaN for missing labels
                default_val = float("nan")
                tensor_list = [
                    torch.tensor(d.get(key, default_val)) for d in labels_dicts
                ]
                batched_labels[key] = torch.stack(tensor_list)

    return data_batch, batched_labels, custom_recon_batch
