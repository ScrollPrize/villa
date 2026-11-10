"""Baseline modes, output units and opt-in previews without raw-data dependencies."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import compare_methods as compare
from helirec import cli
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
        self.main = SimpleNamespace(pixel_size_m=8.64e-6)

    def set_params(self, **kw):
        HelicalRecon._update(self, kw)

    def provenance(self):
        return {"params": dict(self.params), "output_units": "reference_units (attenuation / 2)"
                if self.params["mode"] == "legacy_reference" else "attenuation_cm-1"}

    def reconstruct_heights(self, lo, hi, **kwargs):
        return np.broadcast_to(np.arange(64, dtype=np.float32).reshape(8, 8), (hi - lo, 8, 8)).copy()


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
    from helirec.omezarr import outside_disk, to_uint8
    monkeypatch.setattr("helirec.recon.HelicalRecon", FakeRecon)
    cli.main(["unused", "--out_root", str(tmp_path), "--tag", "t", "--z", "2", "5", "--format", fmt,
              "--u8_window", "0", "63", "--u8_compression", "none"] + flags)
    out = tmp_path / "t"
    names = {"float32": "volume.zarr", "uint8": "volume_uint8.zarr"} if fmt == "zarr" else {"float32": "float32", "uint8": "uint8"}
    assert {k for k, n in names.items() if (out / n).exists()} == kinds
    assert set(json.loads((out / "manifest.json").read_text())["outputs"]) == kinds
    if fmt == "tiff" and "uint8" in kinds:
        want = to_uint8(np.arange(64, dtype=np.float32).reshape(8, 8), 0, 63, outside=outside_disk(8))
        assert sorted(f.name for f in (out / "uint8").iterdir()) == ["z00002.tif", "z00003.tif", "z00004.tif"]
        np.testing.assert_array_equal(tifffile.imread(out / "uint8" / "z00003.tif"), want)
