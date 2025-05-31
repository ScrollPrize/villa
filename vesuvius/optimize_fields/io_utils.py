#optimize_fields/io_utils.py

import zarr
import numpy as np
from typing import Tuple, Optional, Union
from zarr import Blosc
from zarr.hierarchy import Group
from zarr.core import Array

def open_zarr(path: str, mode: str = 'r') -> Union[Array, Group]:
    """
    Open a Zarr store (either an Array or a Group), locally or on S3.
    """
    storage_opts = {'anon': False} if path.startswith('s3://') else None
    return zarr.open(path, mode=mode, storage_options=storage_opts)

def create_output_store(
    output_path: str,
    volume_shape: Tuple[int, int, int, int],
    chunk_size: Tuple[int, int, int, int],
    compressor: Optional[Blosc] = None
) -> Tuple[Group, Array, Array]:
    """
    Create a Zarr group at `output_path` with three datasets 'u', 'v' and 'n'.

    Args:
      output_path: path for the zarr (will get '.zarr' appended if needed)
      volume_shape: (3, Z, Y, X)
      chunk_size:   (3, cz, cy, cx)
      compressor:   a Blosc compressor or None

    Returns:
      (root_group, ds_u, ds_v, ds_n)
    """
    if not output_path.endswith('.zarr'):
        output_path = output_path + '.zarr'

    root = zarr.open_group(
        output_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None
    )

    ds_u = root.create_dataset(
        'u',
        shape=volume_shape,
        chunks=chunk_size,
        dtype=np.float32,
        compressor=compressor,
        write_empty_chunks=False
    )
    ds_v = root.create_dataset(
        'v',
        shape=volume_shape,
        chunks=chunk_size,
        dtype=np.float32,
        compressor=compressor,
        write_empty_chunks=False
    )
    ds_n = root.create_dataset(
        'n',
        shape=volume_shape,
        chunks=chunk_size,
        dtype=np.float32,
        compressor=compressor,
        write_empty_chunks=False
    )
    return root, ds_u, ds_v, ds_n
