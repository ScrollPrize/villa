from typing import Union, Tuple, List, Callable
import numpy as np
import torch
from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform

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
                        
                        # Compute intensity value; prefer torch-native paths, fallback to numpy for custom callables
                        rv = getattr(self.color_fn, 'rectangle_value', None)
                        slice_t = result[my_slice]
                        intensity_t: torch.Tensor
                        if isinstance(rv, (int, float)):
                            intensity_t = torch.tensor(rv, device=result.device, dtype=result.dtype)
                        elif isinstance(rv, (tuple, list)) and len(rv) == 2 and all(isinstance(x, (int, float)) for x in rv):
                            low, high = float(rv[0]), float(rv[1])
                            r = torch.rand((), device=result.device, dtype=result.dtype)
                            intensity_t = r * (high - low) + low
                        elif callable(rv) and getattr(rv, '__name__', '') in ('mean', 'median'):
                            if rv.__name__ == 'mean':
                                intensity_t = slice_t.mean().to(dtype=result.dtype)
                            else:
                                intensity_t = slice_t.median().values.to(dtype=result.dtype)
                        else:
                            # Fallback to user-provided callable on CPU numpy, then convert back
                            intensity = self.color_fn(slice_t.detach().cpu().numpy())
                            intensity_t = torch.tensor(intensity, device=result.device, dtype=result.dtype)
                        result[my_slice] = intensity_t
        
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
    
    Rician noise occurs in MRI when taking the magnitude of complex data with 
    Gaussian noise in both real and imaginary parts.
    
    Args:
        data: Input tensor (treated as the underlying signal magnitude)
        noise_variance: Range for variance of the Gaussian distributions
        
    Returns:
        Tensor with added Rician noise
    """
    variance = np.random.uniform(*noise_variance)
    std_dev = np.sqrt(variance)
    
    # Generate independent Gaussian noise for real and imaginary components
    # The data is treated as the underlying signal magnitude
    real_noise = torch.normal(0, std_dev, size=data.shape, device=data.device, dtype=data.dtype)
    imag_noise = torch.normal(0, std_dev, size=data.shape, device=data.device, dtype=data.dtype)
    
    # In Rician noise, the signal is in the real component, imaginary starts at 0
    # Result is the magnitude of complex signal + noise
    return torch.sqrt((data + real_noise) ** 2 + imag_noise ** 2)

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
        self._skip_when_vector = True

    def get_parameters(self, **data_dict) -> dict:
        # No extra parameters are needed.
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        # Pure torch implementation; runs on CPU or GPU depending on img.device
        C = img.shape[0]
        spatial_shape = img.shape[1:]
        num_spatial_dims = len(spatial_shape)
        if not (1 <= self.smear_axis <= num_spatial_dims):
            raise ValueError(f"smear_axis must be between 1 and {num_spatial_dims} for input with shape {tuple(img.shape)}")

        local_smear_axis = self.smear_axis - 1
        transformed = img.clone()
        # Determine dims for moveaxis in torch
        for ch in range(C):
            chan_img = img[ch]
            if chan_img.shape[local_smear_axis] <= self.num_prev_slices:
                continue
            # Move the smear axis to front: build permutation
            dims = list(range(chan_img.ndim))
            if local_smear_axis != 0:
                dims = [local_smear_axis] + [d for d in dims if d != local_smear_axis]
                moved = chan_img.permute(*dims)
            else:
                moved = chan_img
            N = moved.shape[0]
            for i in range(self.num_prev_slices, N):
                aggregated = torch.zeros_like(moved[i], dtype=torch.float32, device=moved.device)
                count = 0
                for j in range(i - self.num_prev_slices, i):
                    slice_j = moved[j]
                    # Determine shift behavior based on dimensionality of slice
                    if slice_j.ndim == 0:
                        shifted = slice_j
                    elif slice_j.ndim == 1:
                        shifted = torch.roll(slice_j, shifts=self.shift[0], dims=0)
                    else:
                        # Shift along first two axes of the slice
                        s0 = self.shift[0] if len(self.shift) >= 1 else 0
                        s1 = self.shift[1] if len(self.shift) >= 2 else 0
                        shifted = torch.roll(slice_j, shifts=(s0, s1), dims=(0, 1))
                    aggregated += shifted.to(aggregated.dtype)
                    count += 1
                if count > 0:
                    aggregated = aggregated / float(count)
                moved[i] = ((1 - self.alpha) * moved[i].to(torch.float32) + self.alpha * aggregated).to(moved.dtype)
            # Restore original axis order
            if local_smear_axis != 0:
                inv_perm = list(range(1, moved.ndim))
                inv_perm.insert(local_smear_axis, 0)
                transformed[ch] = moved.permute(*inv_perm)
            else:
                transformed[ch] = moved
        return transformed
