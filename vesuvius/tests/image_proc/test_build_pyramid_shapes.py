"""build_pyramid must size each level as ceil(n / 2**level), the rule VC3D checks and the published volumes follow.

round() is half to even, so an axis of 4k + 1 voxels came out one voxel short (1125 gave 562, not 563) and VC3D
refused the volume: "zarr level 1 shape ... does not match synthesized dimensions" (#2135).
"""
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.utils import build_pyramid


def _level0(tmp_path, shape, chunks):
    root = str(tmp_path / "vol.zarr")
    rng = np.random.default_rng(0)
    data = rng.integers(1, 250, size=shape, dtype=np.uint8)
    group = zarr.open_group(root, mode="w", zarr_format=2)
    arr = group.create_array("0", shape=shape, chunks=chunks, dtype=data.dtype)
    arr[:] = data
    return root, data


def _ceil_div_pow2(n, level):
    # Volume.cpp ceilDivPow2: (v + 2**level - 1) / 2**level in integer arithmetic
    denom = 1 << level
    return (n + denom - 1) // denom


@pytest.mark.parametrize(
    "shape",
    [
        (1125, 6, 6),  # the axis from #2135: 562 written, 563 expected, at every level
        (5, 7, 9),  # 2k + 1 with k even and odd on different axes
        (8, 8, 8),  # even sizes are unchanged
    ],
)
def test_levels_are_ceiling_division(tmp_path, shape):
    root, data = _level0(tmp_path, shape, chunks=(128, 6, 6))
    num_levels = 6
    build_pyramid(root, ["z", "y", "x"], num_levels, num_workers=1)
    for level in range(1, num_levels):
        arr = zarr.open(f"{root}/{level}", mode="r")
        expected = tuple(max(1, _ceil_div_pow2(n, level)) for n in shape)
        assert arr.shape == expected, f"level {level}: {arr.shape} != {expected}"


def test_level_one_is_nearest_neighbour_with_the_last_voxel_kept(tmp_path):
    root, data = _level0(tmp_path, (1125, 5, 5), chunks=(128, 5, 5))
    build_pyramid(root, ["z", "y", "x"], 2, num_workers=1)
    level1 = zarr.open(f"{root}/1", mode="r")[:]
    assert level1.shape == (563, 3, 3)
    # every level-1 voxel is the level-0 voxel at twice its index, including the last one (1124)
    np.testing.assert_array_equal(level1, data[::2, ::2, ::2])


def test_issue_2135_shapes(tmp_path):
    # the table from the report: (1125, 364, 364) must give 563, 282, 141, 71, 36 on z
    for level, z in zip(range(1, 6), (563, 282, 141, 71, 36)):
        assert _ceil_div_pow2(1125, level) == z
    root, _ = _level0(tmp_path, (1125, 4, 4), chunks=(256, 4, 4))
    build_pyramid(root, ["z", "y", "x"], 6, num_workers=1)
    assert [zarr.open(f"{root}/{l}", mode="r").shape[0] for l in range(1, 6)] == [563, 282, 141, 71, 36]
