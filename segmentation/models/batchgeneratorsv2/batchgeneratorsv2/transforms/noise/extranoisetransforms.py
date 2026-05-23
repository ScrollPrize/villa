from numbers import Integral, Real
from typing import Union, Tuple, List, Callable
import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
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


def _is_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_probability(value: float, name: str) -> None:
    if not _is_number(value) or not (0 <= float(value) <= 1):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")


def _validate_random_scalar_range(value: RandomScalar, name: str) -> None:
    if callable(value):
        return
    if _is_number(value):
        _validate_probability(float(value), name)
        return
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_number(v) for v in value):
        lo, hi = float(value[0]), float(value[1])
        if not (0 <= lo <= hi <= 1):
            raise ValueError(f"{name} range must satisfy 0 <= min <= max <= 1, got {value!r}")
        return
    raise ValueError(f"{name} must be a number, inclusive range, or callable, got {value!r}")


def _validate_num_prev_slices(value: Union[int, Tuple[int, int], List[int]]) -> None:
    if _is_int(value):
        if int(value) < 1:
            raise ValueError(f"num_prev_slices must be >= 1, got {value!r}")
        return
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_int(v) for v in value):
        lo, hi = int(value[0]), int(value[1])
        if not (1 <= lo <= hi):
            raise ValueError(f"num_prev_slices range must satisfy 1 <= min <= max, got {value!r}")
        return
    raise ValueError(f"num_prev_slices must be an int or inclusive int range, got {value!r}")


def _validate_shift_ranges(shift) -> None:
    if _is_int(shift):
        return
    if not isinstance(shift, (tuple, list)):
        raise ValueError(f"shift must be an int, tuple, or list, got {type(shift)}")
    for entry in shift:
        if _is_int(entry):
            continue
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and all(_is_int(v) for v in entry):
            lo, hi = int(entry[0]), int(entry[1])
            if lo > hi:
                raise ValueError(f"shift ranges must satisfy min <= max, got {entry!r}")
            continue
        raise ValueError(f"shift entries must be ints or inclusive int ranges, got {entry!r}")


def _validate_smear_axis(smear_axis: int) -> None:
    if not _is_int(smear_axis):
        raise ValueError(f"smear_axis must be a positive integer, got {smear_axis!r}")
    if int(smear_axis) < 1:
        raise ValueError(f"smear_axis must be >= 1, got {smear_axis!r}")


def _sample_int(value: Union[int, Tuple[int, int], List[int]]) -> int:
    if _is_int(value):
        return int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_int(v) for v in value):
        lo, hi = int(value[0]), int(value[1])
        if lo > hi:
            raise ValueError(f"Expected inclusive int range with min <= max, got {value!r}")
        if lo == hi:
            return lo
        return int(torch.randint(lo, hi + 1, (1,)).item())
    raise ValueError(f"Expected int or inclusive int range, got {value!r}")


def _sample_shift(shift, num_plane_dims: int) -> Tuple[int, ...]:
    if _is_int(shift):
        return (int(shift),) * num_plane_dims
    if not isinstance(shift, (tuple, list)):
        raise ValueError(f"Expected int, tuple, or list for shift, got {type(shift)}")
    if len(shift) != num_plane_dims:
        raise ValueError(f"Expected {num_plane_dims} shift entries, got {len(shift)}")
    return tuple(_sample_int(s) for s in shift)


class DecohesionTransform(ImageOnlyTransform):
    def __init__(
            self,
            shift: Union[int, Tuple[int, ...], Tuple[Tuple[int, int], ...]] = (10, 0),
            alpha: RandomScalar = 0.5,
            num_prev_slices: Union[int, Tuple[int, int]] = 1,
            smear_axis: int = 1,
            p_per_channel: float = 1.0,
    ):
        """
        Simulate scroll-specific decohesion by blending shifted previous slices into
        later slices. This is an image-only transform; labels are left untouched.

        Args:
            shift:
                Int shift applied to every in-slice dimension, a tuple of constant
                shifts, or a tuple of inclusive integer ranges sampled per call.
                For 2D (C, H, W) images, use an int shift because there is only one
                non-smear plane dimension.
            alpha:
                Blending factor for the aggregated shifted slices (0 = no influence, 1 = full replacement).
            num_prev_slices:
                The number of previous slices to aggregate and use for blending.
            smear_axis:
                The spatial axis (in the full tensor) along which to apply the smear.
                For an input image with shape (C, X, Y) or (C, X, Y, Z), spatial dimensions are indices 1,2,(3).
                Default: 1 (i.e. the first spatial axis).
            p_per_channel:
                Probability of applying the transform to each channel.
        """
        super().__init__()
        _validate_shift_ranges(shift)
        _validate_random_scalar_range(alpha, "alpha")
        _validate_num_prev_slices(num_prev_slices)
        _validate_smear_axis(smear_axis)
        _validate_probability(p_per_channel, "p_per_channel")
        self.shift = shift
        self.alpha = alpha
        self.num_prev_slices = num_prev_slices
        self.smear_axis = int(smear_axis)
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        num_spatial_dims = img.ndim - 1
        local_smear_axis = self.smear_axis - 1
        if not (0 <= local_smear_axis < num_spatial_dims):
            raise ValueError(f"smear_axis must be between 1 and {num_spatial_dims} for input with shape {tuple(img.shape)}")
        num_plane_dims = num_spatial_dims - 1
        alpha = float(sample_scalar(self.alpha, image=img))
        _validate_probability(alpha, "alpha")
        num_prev_slices = _sample_int(self.num_prev_slices)
        return {
            "apply_to_channel": torch.rand(img.shape[0], device=img.device) < self.p_per_channel,
            "alpha": alpha,
            "num_prev_slices": num_prev_slices,
            "shift": _sample_shift(self.shift, num_plane_dims),
            "local_smear_axis": local_smear_axis,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        apply_to_channel = params["apply_to_channel"]
        if not torch.any(apply_to_channel):
            return img

        num_prev_slices = int(params["num_prev_slices"])
        if num_prev_slices <= 0:
            return img

        local_smear_axis = params["local_smear_axis"]
        selected = torch.where(apply_to_channel.to(img.device))[0]
        work = img[selected]
        moved = torch.movedim(work, local_smear_axis + 1, 1)
        num_slices = moved.shape[1]
        if num_slices <= 1:
            return img

        work_dtype = moved.dtype if moved.is_floating_point() else torch.float32
        base = moved.to(work_dtype)
        accumulated = torch.zeros_like(base)
        counts = torch.zeros(num_slices, dtype=work_dtype, device=img.device)
        plane_dims = tuple(range(2, moved.ndim))
        max_lag = min(num_prev_slices, num_slices - 1)

        for lag in range(1, max_lag + 1):
            shifted_previous = torch.roll(base[:, :-lag], shifts=params["shift"], dims=plane_dims)
            accumulated[:, lag:] += shifted_previous
            counts[lag:] += 1

        view_shape = (1, num_slices, *([1] * (moved.ndim - 2)))
        mean_previous = accumulated / counts.clamp_min(1).view(view_shape)
        alpha = float(params["alpha"])
        blended = (1 - alpha) * base + alpha * mean_previous
        blended[:, counts == 0] = base[:, counts == 0]

        img[selected] = torch.movedim(blended.to(img.dtype), 1, local_smear_axis + 1)
        return img


class SmearTransform(DecohesionTransform):
    """Backward-compatible name for the scroll decohesion smear transform."""
