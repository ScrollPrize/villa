"""Downsampled vector-field writes must tile the global grid.

`write_block` sampled each block from its own origin (`block[::ds]`) but wrote to
`[z0 // ds : z1 // ds]`, which counts multiples of `ds` from the volume origin.
The two agree only when every block boundary is a multiple of `ds`, and
`create_vf` partitions by a chunk size the user picks independently of
`--downsample`. On a volume whose size is not a multiple of `ds` the final block
raised `ValueError: could not broadcast input array ...`, and when the lengths
happened to match the samples were written shifted by up to `ds - 1` voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from vesuvius.structure_tensor.vf_format import OMEU8VectorWriter, encode_dir_to_u8


def _blocks(shape, chunk):
    """The partition create_vf feeds the writer."""
    Z, Y, X = shape
    cz, cy, cx = chunk
    for z0 in range(0, Z, cz):
        for y0 in range(0, Y, cy):
            for x0 in range(0, X, cx):
                yield z0, min(z0 + cz, Z), y0, min(y0 + cy, Y), x0, min(x0 + cx, X)


def _write_ramp(path, shape, chunk, downsample):
    """Write a field whose z component encodes the global z of each voxel."""
    writer = OMEU8VectorWriter(
        output_path=str(path),
        group_name="vector_field",
        vol_shape_zyx=shape,
        chunks_zyx=(min(64, shape[0]), min(64, shape[1]), min(64, shape[2])),
        downsample=downsample,
    )
    for z0, z1, y0, y1, x0, x1 in _blocks(shape, chunk):
        block = np.zeros((z1 - z0, y1 - y0, x1 - x0, 3), dtype=np.float32)
        zs = np.arange(z0, z1, dtype=np.float32) / max(shape[0] - 1, 1)  # -> [0, 1]
        block[..., 0] = zs[:, None, None]
        writer.write_block(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1,
                           directions_block_zyx=block)
    return writer


@pytest.mark.parametrize("shape, chunk, downsample", [
    ((600, 8, 8), (250, 8, 8), 4),     # chunk not a multiple of ds
    ((1001, 8, 8), (256, 8, 8), 2),    # volume not a multiple of ds
    ((100, 8, 8), (32, 8, 8), 3),      # neither is
    ((512, 8, 8), (256, 8, 8), 4),     # both are: was already fine
])
def test_every_block_writes(tmp_path, shape, chunk, downsample):
    _write_ramp(tmp_path / "vf.zarr", shape, chunk, downsample)


def test_samples_land_on_their_global_positions(tmp_path):
    shape, chunk, ds = (600, 8, 8), (250, 8, 8), 4
    writer = _write_ramp(tmp_path / "vf.zarr", shape, chunk, ds)

    written = np.asarray(writer.ds_z[:, 0, 0])
    expected = encode_dir_to_u8(
        np.arange(0, shape[0], ds, dtype=np.float32) / (shape[0] - 1))

    assert written.shape == expected.shape
    np.testing.assert_array_equal(written, expected)


def test_downsampled_array_is_fully_covered(tmp_path):
    """No row may be left at its fill value once every block has been written."""
    shape, chunk, ds = (1001, 8, 8), (256, 8, 8), 2
    writer = _write_ramp(tmp_path / "vf.zarr", shape, chunk, ds)

    written = np.asarray(writer.ds_z[:, 0, 0])
    assert written.shape[0] == -(-shape[0] // ds)
    assert written[-1] != 0            # the last row was written, not left empty


def test_downsample_one_is_the_identity(tmp_path):
    shape, chunk = (64, 8, 8), (32, 8, 8)
    writer = _write_ramp(tmp_path / "vf.zarr", shape, chunk, 1)

    written = np.asarray(writer.ds_z[:, 0, 0])
    expected = encode_dir_to_u8(np.arange(shape[0], dtype=np.float32) / (shape[0] - 1))
    np.testing.assert_array_equal(written, expected)


def test_confidence_follows_the_same_grid(tmp_path):
    shape, chunk, ds = (600, 8, 8), (250, 8, 8), 4
    writer = OMEU8VectorWriter(
        output_path=str(tmp_path / "vf.zarr"), group_name="vector_field",
        vol_shape_zyx=shape,
        chunks_zyx=(64, 8, 8), downsample=ds, make_confidence=True)

    for z0, z1, y0, y1, x0, x1 in _blocks(shape, chunk):
        d = np.zeros((z1 - z0, y1 - y0, x1 - x0, 3), dtype=np.float32)
        conf = np.full((z1 - z0, y1 - y0, x1 - x0),
                       (z0 // 250 + 1) / 4.0, dtype=np.float32)
        writer.write_block(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1,
                           directions_block_zyx=d, confidence_block=conf)

    conf_written = np.asarray(writer.ds_conf[:, 0, 0])
    assert conf_written.shape[0] == -(-shape[0] // ds)
    assert len(set(conf_written.tolist())) == 3      # one level per source block
