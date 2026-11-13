"""helirec-composite (tools/helirec_composite.cpp) and BandComposite (helirec/surface_composite.hpp): a flat synthetic
surface over a synthetic OME-Zarr volume whose values rise along z (volume mode, through the tool), and the layers of
a surface reconstructed straight from the projections of the synthetic acquisition of test_reconstruct_tool.py, against
the same points of whole reconstructed slices."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from helirec import _core
from helirec.band import BandComposite
from helirec.recon import HelicalRecon

tifxyz = pytest.importorskip("vc_reader.tifxyz")
tifffile = pytest.importorskip("tifffile")
from test_reconstruct_tool import acquisition  # noqa: E402,F401  (the fixture)

TOOL = next((str(t) for t in (Path(_core.__file__).resolve().parent / "helirec-composite",) if t.exists()), None) or shutil.which("helirec-composite")


def _plane(path, z, rows=40, cols=60, x0=10.0, y0=12.0):
    """a flat surface at height z over (x0 .. x0 + cols - 1, y0 .. y0 + rows - 1), one vertex per voxel"""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    xyz = np.stack([xx + x0, yy + y0, np.full_like(xx, z)], -1)
    tifxyz.write(str(path), tifxyz.Surface(np.ascontiguousarray(xyz), uuid="plane"))
    return str(path)


@pytest.mark.skipif(TOOL is None, reason="helirec-composite is not built")
def test_volume_composites(tmp_path):
    vol = (np.arange(64, dtype=np.float32)[:, None, None] * 2 + np.zeros((64, 80, 96), np.float32)).astype(np.uint8)  # 2 z
    w = _core.OmeZarrWriter(str(tmp_path / "v.zarr"), vol.shape, np.uint8, chunks=(32, 32, 32))
    w.write(0, vol)
    w.close()
    surface = _plane(tmp_path / "plane", z=30.0)
    run = lambda *a: subprocess.run([TOOL, str(tmp_path / "v.zarr"), surface, "--out_root", str(tmp_path), *a], capture_output=True, text=True)
    p = run("--layers", "5", "--start", "-2", "--direction", "+1", "--xy_slices", "1", "--tag", "up")
    assert p.returncode == 0, p.stderr
    out = tmp_path / "up"
    meta = json.loads((out / "composite.json").read_text())
    assert meta["layer_offsets_voxels"] == [-2.0, -1.0, 0.0, 1.0, 2.0] and meta["direction"] == 1
    mean, mx = tifffile.imread(out / "mean.tif"), tifffile.imread(out / "max.tif")
    good = np.isfinite(mean)
    assert good.sum() > 0.8 * 40 * 60
    # the normal is +z or -z; either way the layers are 2 z for z = 28 .. 32 and their mean is 60
    assert np.allclose(mean[good], 60.0) and set(np.unique(mx[good])) <= {64.0, 56.0}
    layers = sorted((out / "layers").iterdir())
    assert [l.name for l in layers][0] == "00_offset-02.0.tif" and len(layers) == 5
    assert (out / "mean_u8.png").exists() and (out / "mask.png").exists() and len(list((out / "xy_slices").glob("*.tif"))) == 1
    # direction auto: toward the denser side, which here is +z or -z, whichever the normal points to
    p = run("--layers", "3", "--start", "1", "--tag", "auto")
    assert p.returncode == 0, p.stderr
    m = json.loads((tmp_path / "auto" / "composite.json").read_text())
    assert m["side_test"]["plus_side_mean"] != m["side_test"]["minus_side_mean"]
    assert np.nanmean(tifffile.imread(tmp_path / "auto" / "mean.tif")) > 60.0       # the layers went up the ramp
    assert run("--layers", "0", "--tag", "x").returncode != 0


def test_band_composite_against_whole_slices(acquisition):  # noqa: F811
    proj, geometry = acquisition
    cache = Path(geometry).parent / "band_cache"
    r = HelicalRecon(proj, cache_dir=str(cache), device="cpu", params=dict(geometry=geometry), log=lambda *a: None)
    N = r.info["grid_size"]
    z = (r.hi - r.lo) // 2
    # points on integer voxels (trilinear sampling is then the voxel itself): a 10 x 12 patch, three layers along z
    yy, xx = np.mgrid[0:10, 0:12].astype(np.float32)
    pts = np.stack([np.stack([xx + N // 2 - 6, yy + N // 2 - 5, np.full_like(xx, z + k)], -1) for k in (-1, 0, 1)])
    bc = BandComposite(r, pts, np.ones((10, 12), bool))
    assert bc.band and bc.n_points == 3 * 120 and bc.valid.all()
    bc.run(chunk=2)
    layers = bc.layers()
    vol = r.reconstruct_heights(z - 1 + r.lo, z + 2 + r.lo, roi=(N // 2 - 6, N // 2 + 6, N // 2 - 5, N // 2 + 5))
    np.testing.assert_allclose(layers, vol, rtol=1e-4, atol=1e-4 * np.abs(vol).max())
    mx, mean = bc.composites()
    np.testing.assert_allclose(mx, vol.max(0), rtol=1e-4, atol=1e-4 * np.abs(vol).max())
    np.testing.assert_allclose(mean, vol.mean(0), rtol=1e-4, atol=1e-4 * np.abs(vol).max())
    # the whole-slice methods: the points' box reconstructed slice by slice, sampled by vc_reader
    r.set_params(backprojector="gridrec")
    whole = BandComposite(r, pts, np.ones((10, 12), bool))
    assert not whole.band
    whole.run(chunk=2)
    ref = r.reconstruct_heights(z - 1 + r.lo, z + 2 + r.lo, roi=(N // 2 - 6, N // 2 + 6, N // 2 - 5, N // 2 + 5))
    np.testing.assert_allclose(whole.layers(), ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())
