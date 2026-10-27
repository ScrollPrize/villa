"""edt-dilate, remap and resize create their output levels on zarr 2 and zarr 3.

They called Group.create_dataset, which zarr 3 removed, so under zarr 3 each
crashed with AttributeError before writing anything (the same break as merge, #2240).
"""
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr, open_zarr_group
from vesuvius.image_proc.run.zarr_tasks.tasks.edt_dilate import EdtDilateConfig, EdtDilateTask
from vesuvius.image_proc.run.zarr_tasks.tasks.remap import RemapConfig, RemapTask
from vesuvius.image_proc.run.zarr_tasks.tasks.resize import ResizeConfig, ResizeTask


def _group(path, *levels):
    open_zarr_group(str(path), mode="w")
    for i, data in enumerate(levels):
        arr = open_zarr(f"{path}/{i}", mode="w", shape=data.shape, chunks=data.shape, dtype=data.dtype)
        arr[...] = data
    return str(path)


def test_edt_dilate_writes_its_level(tmp_path):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[4, 4, 4] = 255
    src = _group(tmp_path / "in.zarr", mask)
    out = str(tmp_path / "out.zarr")
    EdtDilateTask(EdtDilateConfig(input_zarr=src, output_zarr=out, num_workers=1, distance=1.5,
                                  chunk_size=(8, 8, 8), black_border=False)).run()
    got = zarr.open(f"{out}/0", mode="r")[...]
    assert got[4, 4, 4] > 0 and got[4, 4, 5] > 0 and got[0, 0, 0] == 0


def test_remap_writes_every_level(tmp_path):
    level0 = np.array([[[0, 1], [2, 1]]], dtype=np.uint8).repeat(2, axis=0)
    src = _group(tmp_path / "in.zarr", level0, level0[:1, :1, :1])
    out = str(tmp_path / "out.zarr")
    RemapTask(RemapConfig(input_zarr=src, output_zarr=out, num_workers=1, remap_dict={1: 7})).run()
    np.testing.assert_array_equal(zarr.open(f"{out}/0", mode="r")[...], np.where(level0 == 1, 7, level0))
    assert zarr.open(f"{out}/1", mode="r").shape == (1, 1, 1)


def test_resize_writes_reference_shape(tmp_path):
    src = _group(tmp_path / "in.zarr", np.full((4, 4, 4), 9, dtype=np.uint8))
    ref = _group(tmp_path / "ref.zarr", np.zeros((8, 8, 8), dtype=np.uint8))
    out = str(tmp_path / "out.zarr")
    ResizeTask(ResizeConfig(input_zarr=src, output_zarr=out, num_workers=1, reference_zarr=ref,
                            interpolation="nearest")).run()
    got = zarr.open(f"{out}/0", mode="r")[...]
    assert got.shape == (8, 8, 8) and (got == 9).all()
