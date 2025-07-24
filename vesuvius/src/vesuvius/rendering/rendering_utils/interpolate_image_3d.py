# Vesuvius Challenge Team 2024
# integrated from ThaumatoAnakalyptor
# Adjusted from https://github.com/teamtomo/libtilt/blob/b09dd9b245a3ca48354161cb6126d192a6af3e78/src/libtilt/interpolation/interpolate_image_3d.py, https://github.com/teamtomo/libtilt/blob/b09dd9b245a3ca48354161cb6126d192a6af3e78/src/libtilt/coordinate_utils.py#L7

from typing import Literal

import einops
import torch
import torch.nn.functional as F

from typing import Sequence

def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def extract_from_image_3d(
    image: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample a volume with linear interpolation.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` volume.
    coordinates: torch.Tensor
        `(..., zyx)` array of coordinates at which `image` should be sampled.
        Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
        Coordinates should be array coordinates, spanning `[0, N-1]` for a
        dimension of length N.
    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of complex valued samples from `image`.
    """
    device = image.device
    # pack coordinates into shape (b, 3)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # sample dft at coordinates
    image = einops.repeat(image, 'd h w -> b 1 d h w', b=n_samples)  # b c d h w
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )
    samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')

    # zero out samples from outside of volume
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:], device=device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0

    # pack data back up and return
    samples = samples.squeeze(-1)  # Remove the last dimension if it's 1
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (...)

def extract_from_image_4d(
    image: torch.Tensor,
    image_index: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample a volume with linear interpolation.

    Parameters
    ----------
    image: torch.Tensor
        `(n, d, h, w)` volume.
    image_index: torch.Tensor
        `(..., )` array of indices of the image to sample from.
    coordinates: torch.Tensor
        `(..., zyx)` array of coordinates at which `image` should be sampled.
        Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
        Coordinates should be array coordinates, spanning `[0, N-1]` for a
        dimension of length N.
    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of complex valued samples from `image`.
    """
    device = image.device

    # pack coordinates into shape (b, 4)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]
    # pack image_index into shape (b, 1)
    image_index, _ = einops.pack([image_index], pattern='*')
    image_index = image_index.long()  # Convert to long dtype

    # sample dft at coordinates
    image = einops.repeat(image, 'n d h w -> b n d h w', b=n_samples)  # b n d h w

    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )
    samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')
    # extract the image_index
    samples = samples[torch.arange(n_samples), image_index]

    # zero out samples from outside of volume
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:], device=device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0

    # pack data back up and return
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (...)


def insert_into_image_3d(
    data: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Insert values into a 3D volume with trilinear interpolation (rasterisation).

    Parameters
    ----------
    data: torch.Tensor
        `(...)` array of values to be inserted into the image.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
    image: torch.Tensor
        `(d, h, w)` volume containing the image into which data will be inserted.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    device = image.device
    if data.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')

    # linearise data and coordinates
    data, _ = einops.pack([data], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the volume
    inside = (coordinates >= 0) & (coordinates <= torch.tensor(image.shape, device=device) - 1)
    inside = torch.all(inside, dim=-1)
    data, coordinates = data[inside], coordinates[inside]

    # calculate and cache floor and ceil of coordinates for each piece of slice data
    _c = torch.empty(size=(data.shape[0], 2, 3), dtype=torch.long)
    _c[:, 0] = torch.floor(coordinates)  # for lower corners
    _c[:, 1] = torch.ceil(coordinates)  # for upper corners

    # cache linear interpolation weights for each data point being inserted
    _w = torch.empty(size=(data.shape[0], 2, 3))  # (b, 2, zyx)
    _w[:, 1] = coordinates - _c[:, 0]  # upper corner weights
    _w[:, 0] = 1 - _w[:, 1]  # lower corner weights

    def add_data_at_corner(z: Literal[0, 1], y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_w[:, [z, y, x], [0, 1, 2]], 'b zyx -> b', reduction='prod')
        zc, yc, xc = einops.rearrange(_c[:, [z, y, x], [0, 1, 2]], 'b zyx -> zyx b')
        image.index_put_(indices=(zc, yc, xc), values=w * data, accumulate=True)

    add_data_at_corner(0, 0, 0)
    add_data_at_corner(0, 0, 1)
    add_data_at_corner(0, 1, 0)
    add_data_at_corner(0, 1, 1)
    add_data_at_corner(1, 0, 0)
    add_data_at_corner(1, 0, 1)
    add_data_at_corner(1, 1, 0)
    add_data_at_corner(1, 1, 1)

    return image
