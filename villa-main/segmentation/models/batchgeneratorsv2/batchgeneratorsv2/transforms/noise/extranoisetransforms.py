from typing import Union, Tuple, List, Callable
import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform

class ColorFunctionExtractor:
    def __init__(self, rectangle_value):
        self.rectangle_value = rectangle_value

    def __call__(self, x):
        if np.isscalar(self.rectangle_value):
            return self.rectangle_value
        elif callable(self.rectangle_value):
            return self.rectangle_value(x)
        elif isinstance(self.rectangle_value, (tuple, list)):
            return np.random.uniform(*self.rectangle_value)
        else:
            raise RuntimeError("unrecognized format for rectangle_value")

class BlankRectangleTransform(BasicTransform):
    """
    Overwrites areas in tensors with rectangles of specified intensity.
    Supports nD data.
    """
    def __init__(self, 
                 rectangle_size: Union[int, Tuple, List],
                 rectangle_value: Union[int, Tuple, List, Callable],
                 num_rectangles: Union[int, Tuple[int, int]],
                 force_square: bool = False,
                 p_per_sample: float = 0.5,
                 p_per_channel: float = 0.5):
        """
        Args:
            rectangle_size: Can be:
                - int: creates squares with edge length rectangle_size
                - tuple/list of int: constant size for rectangles
                - tuple/list of tuple/list: ranges for each dimension
            rectangle_value: Intensity value for rectangles. Can be:
                - int: constant value
                - tuple: range for uniform sampling
                - callable: function to determine value
            num_rectangles: Number of rectangles per image
            force_square: If True, only produces squares
            p_per_sample: Probability per sample
            p_per_channel: Probability per channel
        """
        super().__init__()
        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles
        self.force_square = force_square
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.color_fn = ColorFunctionExtractor(rectangle_value)

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _get_rectangle_size(self, img_shape: Tuple[int, ...]) -> List[int]:
        img_dim = len(img_shape)
        
        if isinstance(self.rectangle_size, int):
            return [self.rectangle_size] * img_dim
        
        elif isinstance(self.rectangle_size, (tuple, list)) and all([isinstance(i, int) for i in self.rectangle_size]):
            return list(self.rectangle_size)
        
        elif isinstance(self.rectangle_size, (tuple, list)) and all([isinstance(i, (tuple, list)) for i in self.rectangle_size]):
            if self.force_square:
                return [np.random.randint(self.rectangle_size[0][0], self.rectangle_size[0][1] + 1)] * img_dim
            else:
                return [np.random.randint(self.rectangle_size[d][0], self.rectangle_size[d][1] + 1) 
                        for d in range(img_dim)]
        else:
            raise RuntimeError("unrecognized format for rectangle_size")

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        result = img.clone()
        img_shape = img.shape[1:]  # DHW
        
        if np.random.uniform() < self.p_per_sample:
            for c in range(img.shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    # Number of rectangles
                    n_rect = (self.num_rectangles if isinstance(self.num_rectangles, int) 
                            else np.random.randint(self.num_rectangles[0], self.num_rectangles[1] + 1))
                    
                    for _ in range(n_rect):
                        rectangle_size = self._get_rectangle_size(img_shape)
                        
                        # Get random starting positions
                        lb = [np.random.randint(0, max(img_shape[i] - rectangle_size[i], 1)) 
                            for i in range(len(img_shape))]
                        ub = [i + j for i, j in zip(lb, rectangle_size)]
                        
                        # Create slice for the rectangle
                        my_slice = tuple([c, *[slice(i, j) for i, j in zip(lb, ub)]])
                        
                        # Get intensity value and convert to torch tensor before assignment
                        intensity = self.color_fn(result[my_slice].cpu().numpy())
                        intensity = torch.tensor(intensity, device=result.device, dtype=result.dtype)
                        result[my_slice] = intensity
        
        return result

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **kwargs) -> torch.Tensor:
        return segmentation  # Don't modify segmentations

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **kwargs) -> torch.Tensor:
        # DO NOT blank anything in the distance map
        # (this is an intensity transform, not geometric)
        return dist_map

    def _apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **kwargs):
        raise NotImplementedError

    def _apply_to_regr_target(self, regr_target: torch.Tensor, **kwargs) -> torch.Tensor:
        return regr_target  # Don't modify regression targets
    
import numpy as np
import torch
from typing import Union, Tuple
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

def augment_rician_noise(data: torch.Tensor, noise_variance: Tuple[float, float]) -> torch.Tensor:
    """
    Adds Rician noise to the input tensor.
    
    Args:
        data: Input tensor
        noise_variance: Range for variance of the Gaussian distributions
        
    Returns:
        Tensor with added Rician noise
    """
    variance = np.random.uniform(*noise_variance)
    
    # Generate two independent Gaussian distributions
    noise1 = torch.normal(0, np.sqrt(variance), size=data.shape)
    noise2 = torch.normal(0, np.sqrt(variance), size=data.shape)
    
    # Calculate Rician noise
    return torch.sqrt((data + noise1) ** 2 + noise2 ** 2)

class RicianNoiseTransform(BasicTransform):
    """
    Adds Rician noise with the given variance.
    The Noise of MRI data tends to have a Rician distribution: 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/
    
    Args:
        noise_variance: Tuple of (min, max) for variance of Gaussian distributions
        p_per_sample: Probability of applying the transform per sample
    """
    def __init__(self, 
                 noise_variance: Union[Tuple[float, float], float] = (0, 0.1),
                 p_per_sample: float = 1.0):
        super().__init__()
        self.noise_variance = noise_variance if isinstance(noise_variance, tuple) else (0, noise_variance)
        self.p_per_sample = p_per_sample

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        if np.random.uniform() < self.p_per_sample:
            return augment_rician_noise(img, self.noise_variance)
        return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **kwargs) -> torch.Tensor:
        return segmentation  # Don't apply noise to segmentations

    def _apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **kwargs):
        raise NotImplementedError

    def _apply_to_regr_target(self, regr_target: torch.Tensor, **kwargs) -> torch.Tensor:
        return regr_target  # Don't apply noise to regression targets


class SmearTransform(ImageOnlyTransform):
    def __init__(self, shift=(10, 0), alpha=0.5, num_prev_slices=1, smear_axis=1):
        """
        Args:
            shift : tuple of int
                The (row_shift, col_shift) to apply to each previous slice (wrap-around is used).
            alpha : float
                Blending factor for the aggregated shifted slices (0 = no influence, 1 = full replacement).
            num_prev_slices : int
                The number of previous slices to aggregate and use for blending.
            smear_axis : int
                The spatial axis (in the full tensor) along which to apply the smear.
                For an input image with shape (C, X, Y) or (C, X, Y, Z), spatial dimensions are indices 1,2,(3).
                Default: 1 (i.e. the first spatial axis).
        """
        super().__init__()
        self.shift = shift
        self.alpha = alpha
        self.num_prev_slices = num_prev_slices
        self.smear_axis = smear_axis

    def get_parameters(self, **data_dict) -> dict:
        # No extra parameters are needed.
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        # Ensure the image is on CPU and convert to numpy.
        device = img.device
        img_np = img.detach().cpu().numpy()
        # We assume the input image has shape (C, ...) where the remaining dimensions are spatial.
        C = img_np.shape[0]
        spatial_shape = img_np.shape[1:]
        num_spatial_dims = len(spatial_shape)
        # Validate smear_axis: must be between 1 and number of spatial dimensions.
        if not (1 <= self.smear_axis <= num_spatial_dims):
            raise ValueError(f"smear_axis must be between 1 and {num_spatial_dims} for input with shape {img_np.shape}")
        # For each channel, we want to operate on the corresponding spatial image.
        # Since the channel dimension is separate, we adjust the smear axis to a "local" axis in the channel image.
        # (For example, if smear_axis is 1 in the full tensor, then for the channel image it is 0.)
        local_smear_axis = self.smear_axis - 1

        transformed = np.copy(img_np)
        for ch in range(C):
            chan_img = img_np[ch]  # shape: spatial_shape (e.g., for a 3D image: (D, H, W))
            # Proceed only if the size along the smear axis is greater than num_prev_slices.
            if chan_img.shape[local_smear_axis] <= self.num_prev_slices:
                continue

            # To iterate easily along the smear axis, bring that axis to the front.
            moved = np.moveaxis(chan_img, local_smear_axis, 0)  # shape: (N, ...) where N = size along smear axis
            N = moved.shape[0]
            # Iterate over slices starting from index num_prev_slices.
            for i in range(self.num_prev_slices, N):
                aggregated = np.zeros_like(moved[i], dtype=np.float32)
                count = 0
                for j in range(i - self.num_prev_slices, i):
                    # Shift the previous slice by the given offset.
                    shifted = np.roll(moved[j], shift=self.shift, axis=(0, 1))
                    aggregated += shifted.astype(np.float32)
                    count += 1
                if count > 0:
                    aggregated /= count
                # Blend the aggregated slice with the current slice.
                moved[i] = ((1 - self.alpha) * moved[i].astype(np.float32) + self.alpha * aggregated).astype(moved[i].dtype)
            # Restore the original axis order.
            transformed[ch] = np.moveaxis(moved, 0, local_smear_axis)
        # Convert back to a torch tensor (preserving dtype and device).
        out = torch.from_numpy(transformed).to(device)
        return out