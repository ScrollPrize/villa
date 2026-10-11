import numpy as np
from vesuvius.structure_tensor.vf_format import encode_dir_to_u8, encode_conf_to_u8


def test_encode_dir_to_u8_roundtrip_edges():
    xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    u8 = encode_dir_to_u8(xs)
    # exact endpoints under spec d*127 + 128  ⇒  -1→1, 1→255
    assert u8[0] == 1 and u8[4] == 255
    # center maps to ~128 (allow 1 ulp tolerance due to rounding)
    assert abs(int(u8[2]) - 128) <= 1
    # NaN/±Inf are clamped
    assert u8[5] == 128 and u8[6] == 255 and u8[7] == 0


def test_encode_conf_to_u8_clamps():
    cs = np.array([-0.1, 0.0, 0.5, 1.0, 1.7, np.nan], dtype=np.float32)
    u8 = encode_conf_to_u8(cs)
    assert u8.tolist() == [0, 0, 128, 255, 255, 0]

# --- OMEU8VectorWriter downsampling ------------------------------------------------
# With downsample > 1 the writer used block bounds // ds for the target slice but took
# block[::ds] as the samples. When a block bound is not a multiple of ds -- always the
# case for the last block of an odd-length volume, which create_vf clips to the volume
# edge -- the target slice came out empty or one short and the last downsampled plane
# was silently left unwritten (0) while the run reported success.

import pytest
from vesuvius.structure_tensor.vf_format import OMEU8VectorWriter, downsample_axis


def _write_in_blocks(tmp_path, field, block, ds):
    Z, Y, X, _ = field.shape
    w = OMEU8VectorWriter(output_path=str(tmp_path / "vf.zarr"), group_name="vf",
                          vol_shape_zyx=(Z, Y, X), chunks_zyx=(4, 4, 4), downsample=ds)
    for z0 in range(0, Z, block):
        for y0 in range(0, Y, block):
            for x0 in range(0, X, block):
                z1, y1, x1 = min(z0 + block, Z), min(y0 + block, Y), min(x0 + block, X)
                w.write_block(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1,
                              directions_block_zyx=field[z0:z1, y0:y1, x0:x1])
    return w


@pytest.mark.parametrize("shape,block,ds", [
    ((5, 4, 4), 4, 2),      # odd length: last block [4, 5) holds one on-grid plane
    ((9, 7, 8), 4, 2),
    ((10, 10, 10), 4, 3),   # block starts (4, 8) are not multiples of ds
    ((8, 8, 8), 4, 2),      # aligned case keeps its old result
])
def test_downsampled_writer_matches_nearest_neighbour(tmp_path, shape, block, ds):
    rng = np.random.default_rng(0)
    field = rng.uniform(-1, 1, size=(*shape, 3)).astype(np.float32)
    w = _write_in_blocks(tmp_path, field, block, ds)
    expected = encode_dir_to_u8(field[::ds, ::ds, ::ds, 0])
    assert np.asarray(w.ds_z).shape == expected.shape
    np.testing.assert_array_equal(np.asarray(w.ds_z), expected)


def test_downsample_axis_matches_global_grid():
    for s in (1, 2, 3, 4):
        for a0 in range(0, 9):
            for a1 in range(a0 + 1, 13):
                sl, o0, o1 = downsample_axis(a0, a1, s)
                taken = list(range(a0, a1))[sl]
                assert all(v % s == 0 for v in taken)
                assert [v // s for v in taken] == list(range(o0, o1))
