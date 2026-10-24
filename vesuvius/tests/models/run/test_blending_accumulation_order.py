"""blend_logits gives the same merged logits whatever order the patches were predicted in.

vesuvius.predict --patch_order chunk writes the same patches as the row-major order,
only at different indices. process_chunk sums the patches overlapping a voxel in
float32, and float32 summation depends on the order of its terms, so
get_chunk_patches hands them over in coordinate order rather than index order.
These tests run the per-chunk accumulation the blend_logits workers run, in process,
on one synthetic part written in two index orders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vesuvius.data.patch_order import chunk_local_order
from vesuvius.data.utils import open_zarr
from vesuvius.models.run import blending
from vesuvius.models.run.blending import SpatialPatchGrid, calculate_chunks, generate_gaussian_map
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

PATCH = (16, 16, 16)
SHAPE = (64, 64, 64)
NUM_CLASSES = 2


def _row_major_positions() -> list[tuple[int, int, int]]:
    grids = [compute_steps_for_sliding_window(SHAPE[a], PATCH[a], 0.5) for a in range(3)]
    return [(z, y, x) for z in grids[0] for y in grids[1] for x in grids[2]]


def _write_part(parent: Path, coords, logits: np.ndarray) -> dict:
    """One part's logits and coordinates stores, laid out as vesuvius.predict writes them."""
    parent.mkdir(parents=True, exist_ok=True)
    logits_path = str(parent / "logits_part_0.zarr")
    store = open_zarr(logits_path, mode="w", shape=logits.shape, chunks=(1, NUM_CLASSES, *PATCH),
                      dtype=np.float16, write_empty_chunks=False)
    store[:] = logits
    store.attrs["patch_size"] = list(PATCH)
    store.attrs["original_volume_shape"] = list(SHAPE)
    coords_path = str(parent / "coordinates_part_0.zarr")
    coords_store = open_zarr(coords_path, mode="w", shape=(len(coords), 3), chunks=(len(coords), 3), dtype=np.int32)
    coords_store[:] = np.asarray(coords, dtype=np.int32)
    return {0: {"logits": logits_path, "coordinates": coords_path}}


def _spatial_index(parent: Path, part_files: dict) -> SpatialPatchGrid:
    grid = SpatialPatchGrid(patch_size=PATCH, grid_size=1000)
    grid.build(str(parent), part_files, [0], PATCH, chunks=None, tqdm_kwargs={"disable": True})
    return grid


def _blend_in_process(parent: Path, part_files: dict, output_path: str) -> np.ndarray:
    """Accumulate every chunk the way the blend_logits worker processes do, in this process."""
    open_zarr(output_path, mode="w", shape=(NUM_CLASSES, *SHAPE), chunks=(1, *PATCH), dtype=np.float16,
              fill_value=0, write_empty_chunks=False)
    grid = _spatial_index(parent, part_files)
    blending._init_worker(part_files, output_path, generate_gaussian_map(PATCH), PATCH, NUM_CLASSES, False)
    for chunk in calculate_chunks(SHAPE, output_chunks=PATCH):
        chunk_patches = grid.get_chunk_patches(chunk)
        if chunk_patches:
            blending.process_chunk(chunk, chunk_patches)
    return np.asarray(open_zarr(output_path, mode="r")[:])


def _two_index_orders(tmp_path: Path):
    """The same patches as a row-major part and as a chunk-ordered (Morton) part."""
    coords = _row_major_positions()
    logits = np.random.default_rng(0).standard_normal((len(coords), NUM_CLASSES, *PATCH)).astype(np.float16)
    morton = chunk_local_order(coords, (16, 16, 16))
    assert morton != list(range(len(coords)))
    row_major = _write_part(tmp_path / "row_major", coords, logits)
    chunk = _write_part(tmp_path / "chunk", [coords[i] for i in morton], logits[morton])
    return (tmp_path / "row_major", row_major), (tmp_path / "chunk", chunk)


def test_get_chunk_patches_lists_each_part_in_coordinate_order(tmp_path: Path) -> None:
    (rm_dir, rm_files), (ch_dir, ch_files) = _two_index_orders(tmp_path)
    chunks = calculate_chunks(SHAPE, output_chunks=PATCH)

    for parent, part_files in ((rm_dir, rm_files), (ch_dir, ch_files)):
        grid = _spatial_index(parent, part_files)
        for chunk in chunks:
            for patches in grid.get_chunk_patches(chunk).values():
                assert [p[1:] for p in patches] == sorted(p[1:] for p in patches)

    # Row-major index order is coordinate order already, so it is left as it was.
    grid = _spatial_index(rm_dir, rm_files)
    for chunk in chunks:
        for patches in grid.get_chunk_patches(chunk).values():
            indices = [p[0] for p in patches]
            assert indices == sorted(indices)


def test_blended_logits_do_not_depend_on_patch_index_order(tmp_path: Path) -> None:
    (rm_dir, rm_files), (ch_dir, ch_files) = _two_index_orders(tmp_path)
    merged_rm = _blend_in_process(rm_dir, rm_files, str(tmp_path / "merged_row_major.zarr"))
    merged_ch = _blend_in_process(ch_dir, ch_files, str(tmp_path / "merged_chunk.zarr"))
    assert merged_rm.shape == (NUM_CLASSES, *SHAPE)
    assert np.count_nonzero(merged_rm) > 0
    assert np.array_equal(merged_rm, merged_ch)
