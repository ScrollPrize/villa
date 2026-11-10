"""Baseline modes, output units and opt-in previews without raw-data dependencies."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import compare_methods as compare
from helirec import cli, to_uint8
from helirec.recon import DEFAULTS, HelicalRecon, LEGACY_REFERENCE


@pytest.mark.parametrize("recipe", ["343p_baselines.json", "343p_corrected_baselines.json"])
def test_baseline_paganin_uses_reference_sharpening(recipe):
    cfg = json.loads((Path(compare.__file__).parent / recipe).read_text())
    for method in cfg["methods"]:
        r = object.__new__(HelicalRecon)
        r.params = dict(DEFAULTS)
        r._update(compare.method_params(cfg["params"], method))
        if "no_paganin" in method["name"]:
            assert r.params["filters"] == []
        else:
            assert r.params["filters"] == LEGACY_REFERENCE["filters"]


def test_legacy_method_pins_settings_and_corrected_method_resets_them():
    r = object.__new__(HelicalRecon)
    r.params = dict(DEFAULTS)
    shared = {"mode": "corrected", "filters": [], "luminosity_repartition": 0}
    r._update(compare.method_params(shared, {"mode": "legacy_reference", "backprojector": "ghbp"}))
    assert r.params == {"mode": "legacy_reference", **LEGACY_REFERENCE}
    r._update(compare.method_params(shared, {"backprojector": "sirt"}))
    assert r.params["mode"] == "corrected"
    assert r.params["filters"] == []
    assert r.params["luminosity_repartition"] == 0
    assert compare.display_scale("reference_units (attenuation / 2)") == 2


class FakeRecon:
    def __init__(self, *args, **kwargs):
        self.params = dict(DEFAULTS)
        self.lo, self.hi = 0, 10
        self.info = {"grid_size": 8, "margin": 0}
        self.cal = SimpleNamespace(rot_center=3.5, to_json=lambda: "{}")
        self.regions = FakeRecon.regions = []
        self.main = SimpleNamespace(pixel_size_m=8.64e-6)

    def set_params(self, **kw):
        HelicalRecon._update(self, kw)

    def provenance(self):
        return {"params": dict(self.params), "output_units": "reference_units (attenuation / 2)"
                if self.params["mode"] == "legacy_reference" else "attenuation_cm-1"}

    def reconstruct_heights(self, lo, hi, roi=None, uint8=None, float32=True, progress=None, mask=None, region=None,
                            **kwargs):
        vol = np.broadcast_to(np.arange(64, dtype=np.float32).reshape(8, 8), (hi - lo, 8, 8)).copy()
        self.regions.append(region)
        if progress is not None:
            progress.update(hi - lo)
        if uint8 is None:
            return vol
        lo, hi, *zero_outside = uint8      # (as HelicalRecon: the outer circle is clipped unless told otherwise)
        u8 = to_uint8(vol, lo, hi, disk=(8, 0, 0) if not zero_outside or zero_outside[0] else None, mask=mask)
        return (vol, u8) if float32 else u8


@pytest.mark.parametrize("previews", [False, True])
def test_baseline_previews_opt_in(tmp_path, monkeypatch, previews):
    monkeypatch.setattr(compare, "HelicalRecon", FakeRecon)
    cfg = tmp_path / "methods.json"
    cfg.write_text(json.dumps({"projections": "unused", "slices": {"z": [2]}, "methods": [
        {"name": "fbp", "backprojector": "fbp"},
        {"name": "legacy", "mode": "legacy_reference", "backprojector": "ghbp"}]}))
    out = tmp_path / "out"
    compare.main([str(cfg), "--out", str(out), "--run-name", "new"] + (["--previews"] if previews else []))
    assert len(list(out.rglob("*.tif"))) == 2
    assert bool(list(out.rglob("*.png"))) == previews
    assert (out / "sheets_new").exists() == previews
    assert not list((out / "slices").rglob("*.png"))          # the slice folders hold TIFFs only
    manifest = json.loads((out / "manifest_new.json").read_text())
    assert manifest["legacy"]["2"]["output_units"] == "reference_units (attenuation / 2)"
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("flag,expected", [([], True), (["--previews"], False), (["--no_preview"], True)])
def test_slice_cli_previews_opt_in(tmp_path, monkeypatch, flag, expected):
    monkeypatch.setattr("helirec.recon.HelicalRecon", FakeRecon)
    seen = []
    monkeypatch.setattr(cli, "run_fractions", lambda a, *args: seen.append(a.no_preview))
    cli.main(["unused", "--out_root", str(tmp_path), "--tag", "test"] + flag)
    assert seen == [expected]


@pytest.mark.parametrize("fmt", ["zarr", "tiff"])
@pytest.mark.parametrize("flags,kinds", [([], {"float32"}), (["--uint8"], {"float32", "uint8"}),
                                         (["--uint8_only"], {"uint8"})])
def test_slice_cli_uint8_only_skips_float32(tmp_path, monkeypatch, fmt, flags, kinds):
    import tifffile
    monkeypatch.setattr("helirec.recon.HelicalRecon", FakeRecon)
    cli.main(["unused", "--out_root", str(tmp_path), "--tag", "t", "--z", "2", "5", "--format", fmt,
              "--u8_window", "0", "63", "--u8_compression", "none"] + flags)
    out = tmp_path / "t"
    names = {"float32": "volume.zarr", "uint8": "volume_uint8.zarr"} if fmt == "zarr" else {"float32": "float32", "uint8": "uint8"}
    assert {k for k, n in names.items() if (out / n).exists()} == kinds
    assert set(json.loads((out / "manifest.json").read_text())["outputs"]) == kinds
    if fmt == "tiff" and "uint8" in kinds:
        want = to_uint8(np.arange(64, dtype=np.float32).reshape(8, 8), 0, 63, disk=(8, 0, 0))
        assert sorted(f.name for f in (out / "uint8").iterdir()) == ["z00002.tif", "z00003.tif", "z00004.tif"]
        np.testing.assert_array_equal(tifffile.imread(out / "uint8" / "z00003.tif"), want)


def test_slice_cli_reference_volume_masks_the_uint8_output(tmp_path, monkeypatch):
    import tifffile
    from helirec.omezarr import OmeZarrWriter
    monkeypatch.setattr("helirec.recon.HelicalRecon", FakeRecon)
    ref = np.zeros((10, 8, 8), np.uint8)                                                # the volume's grid
    ref[1:8, 2:7, 1:6] = 200
    for name, vol in (("ref.zarr", ref), ("other.zarr", ref[:, :6])):
        w = OmeZarrWriter(str(tmp_path / name), vol.shape, np.uint8, chunks=(4, 4, 4), levels=2)
        w.write(0, vol)
        w.close()
    args = ["unused", "--out_root", str(tmp_path), "--z", "2", "7", "--chunk", "2", "--format", "tiff", "--u8_window", "0", "63"]
    cli.main(args + ["--tag", "plain", "--uint8_only"])
    cli.main(args + ["--tag", "masked", "--uint8", "--reference_volume", str(tmp_path / "ref.zarr")])
    for z in range(2, 7):      # three chunks: the masks are read ahead into two buffers in turn
        plain = tifffile.imread(tmp_path / "plain" / "uint8" / f"z{z:05d}.tif")
        masked = tifffile.imread(tmp_path / "masked" / "uint8" / f"z{z:05d}.tif")
        np.testing.assert_array_equal(masked, np.where(ref[z], plain, 0))
        assert tifffile.imread(tmp_path / "masked" / "float32" / f"z{z:05d}.tiff").dtype == np.float32   # not masked
    assert json.loads((tmp_path / "masked" / "manifest.json").read_text())["reference_volume"]["level_shape"] == [10, 8, 8]
    # a coarser level of the reference decides the same area to within its voxels
    w = OmeZarrWriter(str(tmp_path / "pyramid.zarr"), ref.shape, np.uint8, chunks=(4, 4, 4), levels=2)
    w.write(0, ref)
    w.close()
    coarse = np.repeat(np.repeat(np.repeat(w.read(1, 0, 5), 2, 0), 2, 1), 2, 2)
    cli.main(args + ["--tag", "coarse", "--uint8_only", "--reference_volume", str(tmp_path / "pyramid.zarr"),
                     "--reference_scale", "1"])
    for z in range(2, 7):
        plain = tifffile.imread(tmp_path / "plain" / "uint8" / f"z{z:05d}.tif")
        np.testing.assert_array_equal(tifffile.imread(tmp_path / "coarse" / "uint8" / f"z{z:05d}.tif"),
                                      np.where(coarse[z], plain, 0))
    assert json.loads((tmp_path / "coarse" / "manifest.json").read_text())["reference_volume"]["factors"] == [2, 2, 2]
    with pytest.raises(SystemExit, match="same grid"):
        cli.main(args + ["--tag", "wrong", "--uint8", "--reference_volume", str(tmp_path / "other.zarr")])
    with pytest.raises(SystemExit, match="--uint8"):
        cli.main(args + ["--tag", "float", "--reference_volume", str(tmp_path / "ref.zarr")])


def test_slice_cli_bbox_is_passed_to_the_reconstructor(tmp_path, monkeypatch):
    monkeypatch.setattr("helirec.recon.HelicalRecon", FakeRecon)
    args = ["unused", "--out_root", str(tmp_path), "--z", "2", "6", "--chunk", "2", "--format", "tiff"]
    cli.main(args + ["--tag", "box", "--bbox", "1", "5", "2", "8"])
    assert FakeRecon.regions == [[1, 5, 2, 8]] * 2
    assert json.loads((tmp_path / "box" / "manifest.json").read_text())["bbox_xyxy"] == [1, 5, 2, 8]
    cli.main(args + ["--tag", "whole"])
    assert FakeRecon.regions == [None] * 2
    with pytest.raises(SystemExit, match="--bbox"):
        cli.main(args + ["--tag", "bad", "--bbox", "1", "9", "2", "8"])
