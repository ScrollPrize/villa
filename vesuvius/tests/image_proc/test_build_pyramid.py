"""build_pyramid has to give each level the shape VC3D expects: ceil(level-0 size / 2**level).

It used Python's round(), which rounds half to even, so an odd size of the form 4k + 1 lost its last
sampled plane (5 -> 2 instead of 3). VC3D's Volume loader compares every level with ceiling division
and refused those volumes with "zarr level 1 shape ... does not match". The official open-data pyramids
use ceiling division too, and so does numpy's [::2].
"""
import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.utils import build_pyramid, create_level_dataset


@pytest.mark.parametrize(
    "shape",
    [
        (5, 9, 13),     # 4k + 1 on every axis: level 1 was (2, 4, 6)
        (6, 10, 14),    # even at level 0, short at level 2 (y: 10 -> 5 -> 2 instead of 3)
        (7, 11, 15),    # 4k + 3: was already right
        (1125, 3, 1),   # 1125 is a real level size (PHerc1451 8.640 um, level 4); 1 stays 1
    ],
)
def test_levels_use_ceiling_division_and_match_strided_level0(tmp_path, shape):
    root = str(tmp_path / "vol.zarr")
    level0 = (np.arange(np.prod(shape), dtype=np.uint32) % 251 + 1).astype(np.uint8).reshape(shape)
    create_level_dataset(root, "0", shape, (4, 4, 4), np.dtype("uint8"), Blosc(cname="zstd", clevel=1))[:] = level0

    num_levels = 4
    build_pyramid(root, ["z", "y", "x"], num_levels=num_levels, num_workers=1)

    for level in range(1, num_levels):
        step = 2 ** level
        arr = zarr.open(f"{root}/{level}", mode="r")
        expected_shape = tuple(-(-s // step) for s in shape)
        assert arr.shape == expected_shape, f"level {level}"
        np.testing.assert_array_equal(np.asarray(arr[:]), level0[::step, ::step, ::step])
