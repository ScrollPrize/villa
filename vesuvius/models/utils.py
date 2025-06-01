import importlib
import torch
from torch import optim
import torch.nn
from typing import Type
import numpy as np
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm

# a significant part of this code is copied from nnUNetv2
# https://github.com/MIC-DKFZ/nnUNet

def get_number_of_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == nn.Conv1d:
        return 1
    elif conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))


def get_matching_pool_op(conv_op: Type[_ConvNd] = None,
                         dimension: int = None,
                         adaptive=False,
                         pool_type: str = 'avg') -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if dimension == 1:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dimension == 2:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dimension == 3:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d


def get_matching_instancenorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_InstanceNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    if dimension is not None:
        assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.InstanceNorm1d
    elif dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d


def get_matching_convtransp(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.ConvTranspose1d
    elif dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d


def get_matching_batchnorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_BatchNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.BatchNorm1d
    elif dimension == 2:
        return nn.BatchNorm2d
    elif dimension == 3:
        return nn.BatchNorm3d


def get_matching_dropout(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_DropoutNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.Dropout
    elif dimension == 2:
        return nn.Dropout2d
    elif dimension == 3:
        return nn.Dropout3d


def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar

from copy import deepcopy
import numpy as np

def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    This is the same as your posted function.
    We assume `patch_size` is the user-provided shape,
    not one that we compute ourselves.
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # 1) check if we can still downsample by factor 2 on each axis
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        # 2) check spacing ratio constraints: only pool if spacing not >2× the smallest spacing
        min_spacing_of_valid = min(current_spacing[i] for i in valid_axes_for_pool)
        valid_axes_for_pool = [i for i in valid_axes_for_pool
                               if current_spacing[i] / min_spacing_of_valid < 2]

        # 3) check we haven't exceeded the max number of pooling ops
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]
        if len(valid_axes_for_pool) < 1:
            break

        # 4) update conv kernel sizes:
        #    once an axis is within factor 2 of the smallest spacing, we set kernel_size=3 for that axis.
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        pool_kernel_sizes = [1] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2).astype(int)

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))

    # finalize the patch shape so it is divisible by 2^(num_pool_per_axis)
    must_be_divisible_by = 2 ** np.array(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # add one more conv_kernel_size for the bottleneck
    conv_kernel_sizes.append([3]*dim)

    def _to_tuple(lst):
        return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

    return (
        num_pool_per_axis,
        _to_tuple(pool_op_kernel_sizes),
        _to_tuple(conv_kernel_sizes),
        tuple(patch_size),
        must_be_divisible_by
    )


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = []
    for i in range(len(shape)):
        remainder = shape[i] % must_be_divisible_by[i]
        if remainder == 0:
            # if shape[i] is already divisible, in nnU-Net we typically keep it
            # but to exactly emulate nnU-Net v1/v2 logic, we subtract the block size
            # so that e.g. 128 remains 128 (not 192). In practice, either approach can be used:
            new_shp.append(shape[i])
        else:
            # add however many voxels are needed to make it multiple
            new_shp.append(shape[i] + (must_be_divisible_by[i] - remainder))

    return tuple(new_shp)

def get_n_blocks_per_stage(num_stages):
    """
    Stage 0 -> 1 block
    Stage 1 -> 3 blocks
    Stage 2 -> 4 blocks
    Stages 3+ -> 6 blocks each
    """
    blocks = []
    for i in range(num_stages):
        if i == 0:
            blocks.append(1)
        elif i == 1:
            blocks.append(3)
        elif i == 2:
            blocks.append(4)
        else:
            blocks.append(6)
    return blocks

def determine_dimensionality(patch_size, verbose=False):
    """
    Centralized function to determine dimensionality and set appropriate operations
    based on patch size.
    
    Parameters
    ----------
    patch_size : tuple or list
        The patch size dimensions
    verbose : bool, optional
        Whether to print debug information
        
    Returns
    -------
    dict
        Dictionary containing dimensionality info and appropriate operations
    """
    if len(patch_size) == 2:
        if verbose:
            print(f"Detected 2D patch size {patch_size}, setting 2D operations")
        return {
            "op_dims": 2,
            "conv_op": "nn.Conv2d",
            "pool_op": "nn.AvgPool2d",
            "norm_op": "nn.InstanceNorm2d",
            "dropout_op": "nn.Dropout2d",
            "spacing": [1] * 2,
            "default_kernel": [3, 3],
            "default_pool": [1, 1],
            "default_stride": [1, 1]
        }
    elif len(patch_size) == 3:
        if verbose:
            print(f"Detected 3D patch size {patch_size}, setting 3D operations")
        return {
            "op_dims": 3,
            "conv_op": "nn.Conv3d",
            "pool_op": "nn.AvgPool3d",
            "norm_op": "nn.InstanceNorm3d",
            "dropout_op": "nn.Dropout3d",
            "spacing": [1] * 3,
            "default_kernel": [3, 3, 3],
            "default_pool": [1, 1, 1],
            "default_stride": [1, 1, 1]
        }
    else:
        raise ValueError(f"Patch size must have either 2 or 3 dimensions! Got {len(patch_size)}D: {patch_size}")
