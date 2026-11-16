"""Rows of scans above each other (a grid of circular scans) in the slice reconstructor: a height is reconstructed
from the scans that see it, one at each place across the beam at least; where two rows hold a height both enter by
their weights; a scan that sees nothing of a chunk is left out of it; rows the beam does not light are not seen
through; the luminosity step takes the scans of each row to each other's level."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core
from test_sirt_pipeline import recon

NR, ND, NGRID = 12, 24, 64
LEVEL = 50000.0


def value(h):
    """what every scan records of the height h (the same at every angle and column)"""
    return np.rint(30000 + 500 * (np.asarray(h) + 10)).astype(np.uint16)


def circular_scan(tmp_path, name, shift, rows_from_first, lit=None, weight=None, gain=1.0):
    """a circular scan of NGRID + 1 projections whose row v holds the height v - rows_from_first; it sits `shift`
    columns right of the merged origin; lit: (first, end), the rows that hold data; gain: its level against the others'"""
    p = np.arange(NGRID + 1)
    frame = np.broadcast_to(np.rint(gain * value(np.arange(NR) - rows_from_first)).astype(np.uint16)[:, None], (NR, ND))
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.broadcast_to(frame, (len(p), NR, ND)).copy(), chunks=(1, NR, ND))
    reader = _core.RawReader([str(path)], "data", NR, ND)
    w = np.ones((NR, ND), np.float32) if weight is None else weight
    more = {} if lit is None else dict(row_first=lit[0], row_end=lit[1])
    # (the slice h is read at the row h - z_readjustment)
    return _core.make_scan_data(p * (360.0 / NGRID), np.zeros(len(p)), .01, NR, ND, shift, 0., float(-rows_from_first), np.full(len(p), .2), np.array([0], np.float32),
                                np.zeros((NR, ND), np.float32), np.full((1, NR, ND), LEVEL, np.float32), w, reader, np.zeros(len(p), np.int32), np.arange(len(p), dtype=np.int32), **more)


def grid(tmp_path, lit=None, weight=None, gains=(1.0, 1.0)):
    """two places across the beam (16 columns apart), two rows: the second holds what lies 8 rows above the first;
    gains: the level of each row's second scan against its first"""
    return [circular_scan(tmp_path, f"r{r}c{c}", 16 * c, 8 * r, lit, weight, gains[r] if c else 1.0) for r in range(2) for c in range(2)]


@pytest.mark.parametrize("device", helirec.devices())
def test_heights_are_the_union_over_the_rows(tmp_path, device):
    kw = dict(backprojector="fbp", grid_size=0, rot_center=11.5, luminosity_repartition=0.)
    r = recon(grid(tmp_path), device, **kw)
    # the first row sees the heights 0..11, the second -8..3: together -8..11 (the last valid height is left out, as ever)
    assert r.info["doable"] == (-8, 11) and r.info["merged_width"] == ND + 16
    for h in (-8, -3, 0, 2, 3, 4, 9, 10):
        sino = r.reconstruct(h, h + 1, debug=True)["sino"]
        expect = -np.log(value(h) / LEVEL)
        np.testing.assert_allclose(sino[:, 0, 2:-2], expect, atol=2e-5, err_msg=str(h))
    # chunks across the rows' overlap and beyond it give the slices that single heights give
    whole = r.reconstruct(-6, 8)
    for h in (-6, -1, 0, 3, 4, 7):
        np.testing.assert_allclose(whole[h + 6], r.reconstruct(h, h + 1)[0], atol=1e-5, err_msg=str(h))
    # one row alone sees its own heights only; scans side by side as before
    assert recon(grid(tmp_path)[:2], device, **kw).info["doable"] == (0, 11)
    # a place with no scan at a height: that height is not reconstructed (the second row's first column missing)
    assert recon(grid(tmp_path)[:2] + grid(tmp_path)[3:], device, **kw).info["doable"] == (0, 11)


@pytest.mark.parametrize("device", helirec.devices())
def test_the_luminosity_step_joins_the_scans_of_each_row(tmp_path, device):
    """The second scan of the first row reads 3 % above its neighbour and that of the second row 2 % below. The step
    compares a scan with the earlier scans of its own row at their seam (another row holds other rows of the detector
    around a height, so it is not compared with), and takes it to their level: per angle and row, the difference of
    the two levels in a window around the seam (12 rows either side), so what is left is the step x how far a row's
    value lies from the window's mean (here under a tenth of the step)."""
    kw = dict(backprojector="fbp", grid_size=0, rot_center=11.5)
    gains = (1.03, 0.98)
    own = slice(26, 38)                                   # merged columns the second scans hold alone
    off = recon(grid(tmp_path, gains=gains), device, **dict(kw, luminosity_repartition=0.))
    on = recon(grid(tmp_path, gains=gains), device, **dict(kw, luminosity_repartition=100.))
    assert on.info["doable"] == off.info["doable"] == (-8, 11)
    for h, row in ((-6, 1), (-3, 1), (6, 0), (9, 0)):      # heights one row of scans holds alone
        expect = -np.log(value(h) / LEVEL)
        step = -np.log(gains[row])
        left_off = off.reconstruct(h, h + 1, debug=True)["sino"][:, 0, own] - expect
        left_on = on.reconstruct(h, h + 1, debug=True)["sino"][:, 0, own] - expect
        np.testing.assert_allclose(left_off, step, atol=1e-4, err_msg=str(h))
        assert np.abs(left_on).max() < 0.12 * abs(step), (h, np.abs(left_on).max())
        np.testing.assert_allclose(on.reconstruct(h, h + 1, debug=True)["sino"][:, 0, 2:14], expect, atol=2e-5, err_msg=str(h))   # the first scans stay
    for h in (0, 2):                                      # heights both rows hold: each row joined in itself, then both by their weights
        expect = -np.log(value(h) / LEVEL)
        left_off = off.reconstruct(h, h + 1, debug=True)["sino"][:, 0, own] - expect
        left_on = on.reconstruct(h, h + 1, debug=True)["sino"][:, 0, own] - expect
        assert np.abs(left_off).max() > 0.004 and np.abs(left_on).max() < 0.12 * 0.03, (h, np.abs(left_off).max(), np.abs(left_on).max())
    # chunks give the slices that single heights give
    whole = on.reconstruct(-6, 8)
    for h in (-6, 0, 3, 7):
        np.testing.assert_allclose(whole[h + 6], on.reconstruct(h, h + 1)[0], atol=1e-5, err_msg=str(h))
    # the scans of a row must follow each other
    g = grid(tmp_path)
    with pytest.raises(RuntimeError, match="must follow each other"):
        recon([g[0], g[2], g[1], g[3]], device, **dict(kw, luminosity_repartition=100.))


@pytest.mark.parametrize("device", helirec.devices())
def test_rows_the_beam_does_not_light_are_not_seen_through(tmp_path, device):
    kw = dict(backprojector="fbp", grid_size=0, rot_center=11.5, luminosity_repartition=0.)
    weight = np.zeros((NR, ND), np.float32)
    weight[2:10] = 1.0
    r = recon(grid(tmp_path, lit=(2, 10), weight=weight), device, **kw)
    # the first row holds the heights 2..9 in its lit rows, the second -6..1
    assert r.info["doable"] == (-6, 9)
    for h in (-6, 0, 1, 2, 8):
        sino = r.reconstruct(h, h + 1, debug=True)["sino"]
        np.testing.assert_allclose(sino[:, 0, 2:-2], -np.log(value(h) / LEVEL), atol=2e-5, err_msg=str(h))
    # lit rows that leave a gap between the two rows of scans: the longer run of heights is what can be done
    apart = np.zeros((NR, ND), np.float32)
    apart[4:10] = 1.0
    assert recon(grid(tmp_path, lit=(4, 10), weight=apart), device, **kw).info["doable"] == (-4, 1) or \
        recon(grid(tmp_path, lit=(4, 10), weight=apart), device, **kw).info["doable"] == (4, 9)


@pytest.mark.parametrize("device", helirec.devices())
def test_the_fan_of_a_grids_columns(tmp_path, device):
    """A beam that diverges (fan_per_column): every merged column takes the fan of the detector column that holds
    it, counted from that scan's middle column, so the fan starts again at each place across the beam; over the
    columns two places share it goes from one's to the other's by their weights. Rows of scans at one place share
    their columns' fan. Only the exact backprojector takes it."""
    kw = dict(backprojector="fbp", grid_size=0, rot_center=11.5, luminosity_repartition=0.)
    k = 1e-3
    r = recon(grid(tmp_path), device, fan_per_column=k, **kw)
    fan = np.array(r.fan_of_columns)
    m = np.arange(ND + 16)
    first, second = k * (m - 0.5 * (ND - 1)), k * (m - 16 - 0.5 * (ND - 1))
    assert r.info["fan_per_column"] == k and fan.shape == (ND + 16,)
    np.testing.assert_allclose(fan[:16], first[:16], atol=1e-7)          # the first place alone
    np.testing.assert_allclose(fan[ND:], second[ND:], atol=1e-7)         # the second alone
    np.testing.assert_allclose(fan[16:ND], 0.5 * (first + second)[16:ND], atol=1e-7)   # both, at equal weights
    # weights that fall towards a scan's last columns: the shared columns go from the first's fan to the second's
    w = np.ones((NR, ND), np.float32)
    w[:, 16:] = np.linspace(1, 0, ND - 16, dtype=np.float32)
    ramped = np.array(recon([circular_scan(tmp_path, "a", 0, 0, weight=w), circular_scan(tmp_path, "b", 16, 0, weight=w[:, ::-1].copy())], device, fan_per_column=k, **kw).fan_of_columns)
    share = np.linspace(0, 1, ND - 16)
    np.testing.assert_allclose(ramped[17:ND - 1], ((1 - share) * first[16:ND] + share * second[16:ND])[1:-1], atol=1e-6)
    # the slices change with it, and a parallel beam's are as before
    off = recon(grid(tmp_path), device, **kw)
    assert off.info["fan_per_column"] == 0 and len(off.fan_of_columns) == 0
    np.testing.assert_array_equal(recon(grid(tmp_path), device, fan_per_column=0.0, **kw).reconstruct(0, 2), off.reconstruct(0, 2))
    # another backprojector takes the beam as parallel
    assert recon(grid(tmp_path), device, fan_per_column=k, **dict(kw, backprojector="ghbp", angle_interp="linear")).info["fan_per_column"] == 0
