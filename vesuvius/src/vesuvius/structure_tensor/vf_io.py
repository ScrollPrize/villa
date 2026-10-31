# vf_io.py

import numpy as np
import torch
from vc_reader.reader import ZarrGroup, open_node, open_zarr
from typing import Tuple

def load_density_mask(
    input_zarr: str,
    volume_id: int,
    bounds: Tuple[int,int,int,int,int,int],
) -> torch.Tensor:
    """
    Read only the subvolume ρ_s in [z0:z1, y0:y1, x0:x1] from the original input Zarr,
    and return it as a float32 mask of shape (Dz, Dy, Dx).
    """
    z0, z1, y0, y1, x0, x1 = bounds
    arr = open_zarr(input_zarr)
    # if there's a channel axis, we assume it's the first one; drop it
    if arr.ndim == 4:
        sub = arr[0, z0:z1, y0:y1, x0:x1]
    else:
        sub = arr[z0:z1, y0:y1, x0:x1]

    # build mask
    mask = (sub == volume_id).astype(np.float32)
    return torch.from_numpy(mask)  # (Dz, Dy, Dx)


def load_eigenvector_field(
    eigen_zarr: str,
    eig_index: int,
    bounds: Tuple[int,int,int,int,int,int],
) -> torch.Tensor:
    """
    Read only eigenvector #eig_index (0 for principal, 1 for secondary)
    over [z0:z1, y0:y1, x0:x1] from eigen_zarr/eigenvectors,
    returning a tensor of shape (3, Dz, Dy, Dx).
    """
    z0, z1, y0, y1, x0, x1 = bounds
    root = open_node(eigen_zarr)

    if isinstance(root, ZarrGroup):
        try:
            arr = root['eigenvectors']
        except KeyError:
           raise KeyError(f"Zarr group at {eigen_zarr!r} has no 'eigenvectors' dataset")
    else:
        arr = root

    # eigenvectors is a 9×Z×Y×X array; slice out channels [start:start+3]
    start = eig_index * 3
    block = arr[
        start : start + 3,
        z0    : z1,
        y0    : y1,
        x0    : x1
    ].astype(np.float32)

    return torch.from_numpy(block)  # (3, Dz, Dy, Dx)
