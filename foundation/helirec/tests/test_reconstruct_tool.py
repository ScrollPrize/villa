"""helirec-reconstruct (tools/helirec_reconstruct.cpp) and helirec::Reconstruction end to end, on a small synthetic
two-scan helical acquisition written the way BLISS / Lima write one (a rest turn, a helix of four turns, a rest
turn; 200 x 240 frames) with a geometry calibration made up for it (tests/test_geometry.py): the outputs of both
modes and formats, uint8 alone, the reference volume's mask, previews, the config file, and what is refused."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from helirec import _core
from helirec import geometry as G
from helirec.recon import HelicalRecon

h5py = pytest.importorskip("h5py")
tifffile = pytest.importorskip("tifffile")
from test_geometry import _synthetic_geometry  # noqa: E402

# the executable: beside the extension in a build tree, else on the PATH (an installed package)
TOOL = next((str(t) for t in (Path(_core.__file__).resolve().parent / "helirec-reconstruct",) if t.exists()), None) or shutil.which("helirec-reconstruct")
pytestmark = pytest.mark.skipif(TOOL is None, reason="helirec-reconstruct is not built")
ROWS, COLS, PIXEL_UM = 200, 240, 8.64


def _write(root, name, angles, heights_rows, axis_offset_px):
    """root/name/name_0001/name_0001.h5 and its detector files: 20 darks, 20 flats, the projections, 20 flats"""
    d = root / name / f"{name}_0001"
    d.mkdir(parents=True)
    n = len(angles)
    rng = np.random.default_rng(len(name))
    col = np.arange(COLS) - (COLS - 1) / 2 - axis_offset_px
    entries = [("2.1", "dark images", 2, np.full((20, ROWS, COLS), 100, np.uint16)),
               ("3.1", "flat images 1", 1, (20100 + rng.integers(0, 50, (20, ROWS, COLS))).astype(np.uint16))]
    th = np.deg2rad(angles)[:, None]
    line = 0.6 * np.exp(-0.5 * ((col[None] - 30 * np.cos(th)) / 9.0) ** 2) + 0.4 * np.exp(-0.5 * ((col[None] + 20 * np.sin(th)) / 6.0) ** 2)
    proj = (100 + 20000 * np.exp(-line))[:, None, :] * np.ones((1, ROWS, 1))
    entries.append(("4.1", "projections 0 - 600", 0, proj.astype(np.uint16)))
    entries.append(("5.1", "flat images 600", 1, (20100 + rng.integers(0, 50, (20, ROWS, COLS))).astype(np.uint16)))
    with h5py.File(d / f"{name}_0001.h5", "w") as f:
        g = f.create_group("1.1")
        g["title"], g["start_time"], g["end_time"], g["sample/name"] = "tomo:helical", "2025-05-10T10:00:00+02:00", "2025-05-10T10:05:00+02:00", name
        tc = g.create_group("technique/tomoconfig")
        for k, v in (("detector", [b"det3"]), ("rotation", [b"mrsrot", b"srot"]), ("translation_z", [b"mrsz", b"sz"]), ("translation_y", [b"mryrot", b"yrot"])):
            tc[k] = np.array(v, dtype=object)
        for k, v, u in (("energy", 116.0, "keV"), ("effective_propagation_distance", 1000.0, "mm"), ("sample_pixel_size", PIXEL_UM, "um")):
            g[f"technique/scan/{k}"] = v
            g[f"technique/scan/{k}"].attrs["units"] = u
        for k, (entry, title, key, frames) in enumerate(entries):
            e = f.create_group(entry)
            e["title"], e["technique/image_key"] = title, key
            e["start_time"], e["end_time"] = f"2025-05-10T10:0{k}:00+02:00", f"2025-05-10T10:0{k}:50+02:00"
            e["measurement/current"] = np.array([80.0, 80.5])
            e["measurement/current"].attrs["units"] = "mA"
            e["instrument/positioners/yrot"] = axis_offset_px * PIXEL_UM * 1e-3
            e["instrument/positioners/yrot"].attrs["units"] = "mm"
            e["instrument/positioners/sz"] = 0.0
            sub = d / f"scan{k + 2:04d}"
            sub.mkdir()
            with h5py.File(sub / "det3_0000.h5", "w") as lf:
                lf.create_dataset("/entry_0000/instrument/det3/data", data=frames, chunks=(1, ROWS, COLS))
            layout = h5py.VirtualLayout(shape=frames.shape, dtype=np.uint16)
            layout[:] = h5py.VirtualSource(f"scan{k + 2:04d}/det3_0000.h5", "/entry_0000/instrument/det3/data", shape=frames.shape)
            e.create_virtual_dataset("instrument/det3/image", layout)
            if key == 0:
                e["measurement/mrsrot"] = angles
                e["measurement/mrsrot"].attrs["units"] = "deg"
                e["measurement/mrsz"] = -np.asarray(heights_rows) * PIXEL_UM * 1e-3        # (row_sign -1)
                e["measurement/mrsz"].attrs["units"] = "mm"
    return str(d / f"{name}_0001.h5")


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    """the projections folder, and the geometry calibration of its two scans"""
    root = tmp_path_factory.mktemp("acquisition")
    cal, ang, H = _synthetic_geometry(np.random.default_rng(3))
    masters = [_write(root / "projections", name, a, h, off) for name, a, h, off in (("s_HA", ang[0], H[0], 20.5), ("s_TA", ang[1], H[1], 90.5))]
    cal["scans"] = masters
    cal["map"] = G.empty_map(ROWS, COLS)
    path = root / "geometry.json"
    path.write_text(json.dumps(cal, default=float))
    return root / "projections", str(path)


def _run(tmp_path, acquisition, *args, check=True):
    proj, geometry = acquisition
    p = subprocess.run([str(TOOL), str(proj), "--set", f"geometry={geometry}", "--device", "cpu", "--cache_dir", str(tmp_path / "cache"),
                        "--out_root", str(tmp_path), *args], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    if check and p.returncode:
        raise AssertionError(p.stdout[-3000:] + p.stderr[-3000:])
    return p


def test_the_tool_and_the_python_api_agree(tmp_path, acquisition):
    proj, geometry = acquisition
    _run(tmp_path, acquisition, "--fractions", "0.5", "--uint8", "--tag", "f")
    m = json.loads((tmp_path / "f" / "manifest.json").read_text())
    r = HelicalRecon(proj, cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=geometry), log=lambda *a: None)
    s = m["slices"][0]
    assert s["view_height"] == r.slice_info(0.5).view_height and m["params"] == r.params
    img = r.reconstruct_heights(s["view_height"], s["view_height"] + 1)[0]
    assert np.isfinite(img).all() and img.max() > 1                              # (the synthetic sample)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "f" / "slice_f0.5000.tiff"), img)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "f" / "slice_f0.5000_uint8.tif"), _core.to_uint8(img, *m["uint8_window"]))
    assert m["calibration"]["fingerprint"] == r.cal.fingerprint and m["provenance"]["geometry"]["path"] == geometry
    assert not (tmp_path / "f" / "preview").exists()                             # previews are opted in
    # the run's config.json reruns it
    _run(tmp_path, acquisition, "--config", str(tmp_path / "f" / "config.json"), "--tag", "again", "--previews")
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "again" / "slice_f0.5000.tiff"), img)
    assert (tmp_path / "again" / "preview" / "slice_f0.5000.png").exists() and (tmp_path / "again" / "contact_sheet.png").exists()


@pytest.mark.parametrize("fmt", ["zarr", "tiff"])
@pytest.mark.parametrize("flags,kinds", [([], {"float32"}), (["--uint8"], {"float32", "uint8"}), (["--uint8_only"], {"uint8"})])
def test_range_outputs(tmp_path, acquisition, fmt, flags, kinds):
    _run(tmp_path, acquisition, "--z", "100", "103", "--roi", "100", "164", "90", "150", "--chunk", "2", "--format", fmt, "--u8_window", "0", "0.1",
         "--u8_compression", "none", "--tag", "t", *flags)
    out = tmp_path / "t"
    names = {"float32": "volume.zarr", "uint8": "volume_uint8.zarr"} if fmt == "zarr" else {"float32": "float32", "uint8": "uint8"}
    assert {k for k, n in names.items() if (out / n).exists()} == kinds
    m = json.loads((out / "manifest.json").read_text())
    assert set(m["outputs"]) == kinds and m["shape_zyx"] == [3, 60, 64] and m["z"] == [100, 103]
    if fmt == "tiff":
        stack = {k: [tifffile.imread(out / names[k] / f"z{z:05d}.{'tiff' if k == 'float32' else 'tif'}") for z in (100, 101, 102)] for k in kinds}
        if kinds == {"float32", "uint8"}:
            for f, u in zip(stack["float32"], stack["uint8"]):
                assert f.shape == (60, 64) and f.dtype == np.float32 and u.dtype == np.uint8
                N = m["provenance"]["reconstructor"]["grid_size"]
                np.testing.assert_array_equal(u, _core.to_uint8(f, 0, 0.1, (N, 100, 90)))   # 0 outside the reconstruction disk
    else:
        import zarr
        for k in kinds:
            assert zarr.open(str(out / names[k]), mode="r")["0"].shape == (3, 60, 64)


def test_reference_volume_masks_the_uint8_output(tmp_path, acquisition):
    proj, geometry = acquisition
    r = HelicalRecon(proj, cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=geometry), log=lambda *a: None)
    N, nz = r.info["grid_size"], r.hi - r.lo
    ref = np.zeros((nz, N, N), np.uint8)
    ref[:, 100:130, 110:150] = 200
    w = _core.OmeZarrWriter(str(tmp_path / "ref.zarr"), ref.shape, np.uint8, chunks=(32, 64, 64))
    w.write(0, ref)
    w.close()
    args = ["--z", "100", "103", "--roi", "100", "164", "90", "150", "--chunk", "1", "--format", "tiff", "--u8_window", "0", "0.1"]
    _run(tmp_path, acquisition, *args, "--uint8_only", "--tag", "plain")
    _run(tmp_path, acquisition, *args, "--uint8", "--reference_volume", str(tmp_path / "ref.zarr"), "--tag", "masked")
    for z in (100, 101, 102):     # three chunks: the masks are read ahead into two buffers in turn
        plain = tifffile.imread(tmp_path / "plain" / "uint8" / f"z{z:05d}.tif")
        np.testing.assert_array_equal(tifffile.imread(tmp_path / "masked" / "uint8" / f"z{z:05d}.tif"), np.where(ref[z, 90:150, 100:164], plain, 0))
    assert json.loads((tmp_path / "masked" / "manifest.json").read_text())["reference_volume"]["level_shape"] == [nz, N, N]
    p = _run(tmp_path, acquisition, *args, "--reference_volume", str(tmp_path / "ref.zarr"), "--tag", "float", check=False)
    assert p.returncode and "--uint8" in p.stderr


def test_what_is_refused(tmp_path, acquisition):
    for args, said in ((["--z", "1", "2", "--heights", "1", "2"], "either --z or --heights"), (["--fractions", "0.5", "--format", "zarr"], "contiguous range"),
                       (["--z", "0", "100000"], "outside the doable"), (["--set", "nonsense=1"], "unknown parameter"),
                       (["--set", "mode=legacy_reference", "--set", "margin=3"], "pins")):
        p = _run(tmp_path, acquisition, *args, "--tag", "x", "--overwrite", check=False)
        assert p.returncode and said in p.stderr, (args, p.stderr)
    _run(tmp_path, acquisition, "--fractions", "0.5", "--tag", "once")
    p = _run(tmp_path, acquisition, "--fractions", "0.5", "--tag", "once", check=False)
    assert p.returncode and "exists" in p.stderr


def test_bbox_computes_part_of_the_slice(tmp_path, acquisition):
    _run(tmp_path, acquisition, "--fractions", "0.5", "--tag", "whole")
    _run(tmp_path, acquisition, "--fractions", "0.5", "--bbox", "150", "250", "180", "260", "--tag", "box")
    whole, box = (tifffile.imread(tmp_path / t / "slice_f0.5000.tiff") for t in ("whole", "box"))
    np.testing.assert_array_equal(box[180:260, 150:250], whole[180:260, 150:250])
    box[180:260, 150:250] = 0
    assert not box.any()
    p = _run(tmp_path, acquisition, "--fractions", "0.5", "--bbox", "1", "9999", "2", "8", "--tag", "bad", check=False)
    assert p.returncode and "--bbox must lie within" in p.stderr
