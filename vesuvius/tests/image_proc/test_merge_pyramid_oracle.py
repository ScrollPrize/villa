"""Multi-level check of merge's "average" pyramid against an exact oracle.

The oracle evaluates the pyramid the way `_downsample_chunk_worker` defines it (2x2x2
mean, odd extent at the volume end padded with mode='edge'), but without intermediate
rounding: level L is S / 8**L with S an exact integer sum. Rounding each level to nearest
keeps every level within L/2 of that value (each level adds at most 1/2, and averaging
does not amplify earlier errors); truncation can lose up to 7/8 per level.

The test runs the real worker over several chunks per level, on inputs built so that
the rounding residual is as large as possible.
"""
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import _downsample_chunk_worker

LEVELS = 5
CHUNK = (8, 8, 8)


def _pyramid(tmp_path, level0):
    path = str(tmp_path / "vol.zarr")
    root = zarr.open_group(path, mode="w")
    root.zeros(name="0", shape=level0.shape, chunks=CHUNK, dtype=level0.dtype)
    root["0"][...] = level0
    shape = level0.shape
    for level in range(1, LEVELS + 1):
        out_shape = tuple((s + 1) // 2 for s in shape)
        root.zeros(name=str(level), shape=out_shape, chunks=CHUNK, dtype=level0.dtype)
        for cz in range(-(-out_shape[0] // CHUNK[0])):
            for cy in range(-(-out_shape[1] // CHUNK[1])):
                for cx in range(-(-out_shape[2] // CHUNK[2])):
                    _downsample_chunk_worker(((cz, cy, cx), path, level - 1, level,
                                              shape, out_shape, CHUNK, "average"))
        shape = out_shape
    root = zarr.open_group(path, mode="r")
    return {L: root[str(L)][...].astype(np.int64) for L in range(1, LEVELS + 1)}


def _weights_1d(n0, level):
    """Integer weight of each level-0 index in each level-`level` cell (edge replication)."""
    cells = [[i] for i in range(n0)]
    for _ in range(level):
        cells = [cells[j] + (cells[j + 1] if j + 1 < len(cells) else cells[j])
                 for j in range(0, len(cells), 2)]
    w = np.zeros((len(cells), n0), np.int64)
    for j, c in enumerate(cells):
        for i in c:
            w[j, i] += 1
    return w


def _exact_sums(level0, level):
    """S such that S / 8**level is the unrounded level value."""
    s = level0.astype(np.int64)
    for axis in range(3):
        w = _weights_1d(level0.shape[axis], level)
        s = np.moveaxis(np.tensordot(w, np.moveaxis(s, axis, 0), axes=1), 0, axis)
    return s


def _round_half_even(s, d):
    q, r = np.divmod(s, d)
    return q + ((2 * r > d) | ((2 * r == d) & (q % 2 == 1)))


def _cascade(shape, base, step, dtype, hit_all_zero_bits):
    """base + step * (number of scales k < LEVELS at which the voxel's index bits are,
    or are not, all zero). Every 2x2x2 block at every level then has the same fractional
    mean: 4/8 with an even floor (step 4, all-zero bits) or 7/8 (step 1, not all zero)."""
    g = np.indices(shape)
    c = np.zeros(shape, np.int64)
    for k in range(LEVELS):
        zero = ((g[0] >> k) & 1 == 0) & ((g[1] >> k) & 1 == 0) & ((g[2] >> k) & 1 == 0)
        c += zero if hit_all_zero_bits else ~zero
    return (base + step * c).astype(dtype)


def _inputs():
    rng = np.random.default_rng(0)
    yield "random_uint8", rng.integers(0, 256, (32, 32, 32), dtype=np.uint8)
    yield "random_uint16_odd", rng.integers(0, 65536, (33, 17, 9), dtype=np.uint16)
    yield "seven_eighths_uint8", _cascade((32, 32, 32), 100, 1, np.uint8, False)
    yield "seven_eighths_uint16_top", _cascade((32, 32, 32), 65530, 1, np.uint16, False)
    yield "seven_eighths_uint8_odd", _cascade((33, 17, 9), 100, 1, np.uint8, False)
    yield "even_ties_uint8", _cascade((32, 32, 32), 100, 4, np.uint8, True)
    yield "even_ties_uint16_top", _cascade((32, 32, 32), 65514, 4, np.uint16, True)


@pytest.mark.parametrize("name, level0", list(_inputs()), ids=[n for n, _ in _inputs()])
def test_every_level_within_half_a_level_per_level_of_exact(tmp_path, name, level0):
    out = _pyramid(tmp_path, level0)
    for level in range(1, LEVELS + 1):
        s, d = _exact_sums(level0, level), 8 ** level
        assert out[level].shape == s.shape
        # |out - S/d| <= level/2, in integers: |2*d*out - 2*s| <= level*d
        assert np.abs(2 * d * out[level] - 2 * s).max() <= level * d, (name, level)
        if level == 1:
            # One rounding step: level 1 is the correctly rounded mean.
            np.testing.assert_array_equal(out[1], _round_half_even(s, d))


def test_per_level_rounding_reaches_the_bound_on_even_ties(tmp_path):
    # Every block at every level is an exact .5 tie with an even floor, so np.rint
    # rounds down each time: the bound level/2 is attained. The exact value is
    # pipeline + level/2 with an even pipeline value, so the correctly rounded mean of
    # level 0 is higher by round_half_even(level/2) = 0, 1, 2, 2, 2 grey levels for
    # levels 1..5. This is expected for a per-level pyramid, not a rounding defect.
    level0 = _cascade((32, 32, 32), 100, 4, np.uint8, True)
    out = _pyramid(tmp_path, level0)
    for level in range(1, LEVELS + 1):
        s, d = _exact_sums(level0, level), 8 ** level
        assert np.abs(2 * d * out[level] - 2 * s).max() == level * d
        gap = np.abs(out[level] - _round_half_even(s, d)).max()
        assert gap == _round_half_even(level, 2), (level, gap)
