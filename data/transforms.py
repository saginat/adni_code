import torch
import torch.nn.functional as F
import einops


class Resize3D:
    """Resize 3D spatial dimensions while keeping the time dimension intact."""

    def __init__(self, scale_factor=None, target_size=None, align_corners=False):
        assert (scale_factor is not None) or (target_size is not None), (
            "You must provide either scale_factor or target_size."
        )
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.align_corners = align_corners

    def __call__(self, tensor):
        # Assumes shape [N, H, W, D, T]
        x, y, z = tensor.shape[1:4]

        if self.scale_factor is not None:
            new_size = (
                int(x * self.scale_factor),
                int(y * self.scale_factor),
                int(z * self.scale_factor),
            )
        else:
            new_size = self.target_size

        tensor = tensor.permute(0, 4, 1, 2, 3)  # N, T, H, W, D
        resized = F.interpolate(
            tensor, size=new_size, mode="trilinear", align_corners=self.align_corners
        )
        resized = resized.permute(0, 2, 3, 4, 1)  # N, H, W, D, T
        return resized


class NormalizeByRegion:
    def __init__(self, tensor):
        """
        Initialize region-wise normalization for tensors with arbitrary spatial dimensions.
        
        Args:
            tensor: The full dataset tensor of shape [samples, spatial_1, ..., spatial_k, time]
                   where spatial_1 through spatial_k represent arbitrary spatial dimensions.
        """
        # Get the number of dimensions in the tensor
        n_dims = len(tensor.shape)
        # The dimensions we want to average over are samples (0) and time (-1)
        dims_to_reduce = [0, -1]
        
        # Calculate mean and std across samples and time dimension
        # This preserves the spatial dimensions while reducing samples and time
        self.mean = tensor.mean(dim=dims_to_reduce)
        self.std = tensor.std(dim=dims_to_reduce)
        
        # Add small epsilon to prevent division by zero
        self.std = torch.clamp(self.std, min=1e-6)

    def __call__(self, sample):
        """
        Normalize a single sample using precomputed region-wise statistics.
        
        Args:
            sample: A single sample tensor of shape [spatial_1, ..., spatial_k, time]
        Returns:
            Normalized sample tensor of the same shape
        """
        # Move time dimension to first position for normalization
        pattern = '... time -> time ...'
        sample = einops.rearrange(sample, pattern)
        
        # Apply normalization
        # Broadcasting will automatically handle arbitrary spatial dimensions
        sample = (sample - self.mean) / self.std
        
        # Move time dimension back to the end
        pattern = 'time ... -> ... time'
        sample = einops.rearrange(sample, pattern)
        
        return sample


class NormalizeGlobal:
    def __init__(self, tensor):
        """
        Initialize global normalization for tensors with arbitrary dimensions.
        
        Args:
            tensor: The full dataset tensor of any shape
        """
        self.mean = tensor.mean()
        self.std = torch.clamp(tensor.std(), min=1e-6)

    def __call__(self, sample):
        """
        Normalize a single sample using precomputed global statistics.
        
        Args:
            sample: A single sample tensor of any shape
        Returns:
            Normalized sample tensor of the same shape
        """
        return (sample - self.mean) / self.std


class NormalizeByTime:
    def __init__(self, tensor):
        """
        Initialize time-wise normalization for tensors with arbitrary spatial dimensions.
        
        Args:
            tensor: The full dataset tensor of shape [samples, spatial_1, ..., spatial_k, time]
                   where spatial_1 through spatial_k represent arbitrary spatial dimensions
        """
        # We want to average over all dimensions except time
        # This means reducing samples and all spatial dimensions
        dims_to_reduce = list(range(len(tensor.shape)-1))
        
        self.mean = tensor.mean(dim=dims_to_reduce)  # Mean by time
        self.std = tensor.std(dim=dims_to_reduce)    # Std by time
        self.std = torch.clamp(self.std, min=1e-6)

    def __call__(self, sample):
        """
        Normalize a single sample using precomputed time-based statistics.
        
        Args:
            sample: A single sample tensor of shape [spatial_1, ..., spatial_k, time]
        Returns:
            Normalized sample tensor of the same shape
        """
        return (sample - self.mean) / self.std
