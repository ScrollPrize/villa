"""Pyramid numerical reference, sparse/odd edges, native immutability and resume."""
import importlib
import json
from pathlib import Path

import numpy as np
from numcodecs import Blosc
import pytest
import zarr
from vesuvius.label_zarr import create_v2_array, downsample_mean


def reference_xy(a):
    out = np.empty((a.shape[0], (a.shape[1]+1)//2, (a.shape[2]+1)//2),dtype=a.dtype)
    for y in range(out.shape[1]):
        for x in range(out.shape[2]):
            out[:,y,x] = np.floor(a[:,2*y:2*y+2,2*x:2*x+2].mean(axis=(1,2)) + 0.5)
    return out


@pytest.mark.parametrize("depth", [3,65])
def test_parallel_pyramid_and_resume(tmp_path,monkeypatch,depth):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    module = importlib.import_module("build_surface3d_multiscales")
    path = tmp_path/"ink.zarr"
    group = zarr.open_group(str(path),mode="w",zarr_format=2)
    a = np.random.default_rng(17).integers(0,256,(depth,19,27),dtype=np.uint8)
    a[:,:8,:8] = 0
    native = create_v2_array(group,"0",shape=a.shape,chunks=(depth,4,4),dtype=np.uint8,
                             compressor=Blosc(),fill_value=0)
    native[:] = a
    group.attrs.update({"complete":True,"checkpoint_sha256":"abc","multiscales":[{
        "version":"0.4","axes":[{"name":n,"type":"space","unit":"micrometer"} for n in "zyx"],
        "datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[2.4]*3}]}]}]})
    before = {str(p):p.read_bytes() for p in (path/"0").rglob("*") if p.is_file()}
    original = module.make_metadata(dict(group.attrs))
    original[0]["metadata"] = {"downsampling_method":"mean"}
    for dataset in original[0]["datasets"]:
        dataset["coordinateTransformations"].append({"type":"translation","translation":[0.0,0.0,0.0]})
    (tmp_path/".zattrs").write_text(json.dumps({"multiscales":original,"slice_step":1.0}))
    item = {"id":"test","output":str(path),"input":str(tmp_path),"shape":list(a.shape)}
    result = module.build_one(item,tmp_path,workers=4)
    group = zarr.open_group(str(path),mode="r")
    expected = a
    for level in range(6):
        np.testing.assert_array_equal(group[str(level)][:],expected)
        assert group[str(level)].shape[0] == depth
        expected = reference_xy(expected)
    assert before == {str(p):p.read_bytes() for p in (path/"0").rglob("*") if p.is_file()}
    attrs = dict(group.attrs)
    assert attrs["multiscales_complete"] and len(attrs["multiscales"][0]["datasets"]) == 6
    assert attrs["multiscales"][0]["datasets"][4]["coordinateTransformations"][0]["scale"] == [2.4,38.4,38.4]
    monkeypatch.setattr(module,"populate_chunk",lambda *a:pytest.fail("Completed level rebuilt"))
    assert module.build_one(item,tmp_path,workers=4) == result
    assert set(result["level_inventory_sha256"]) == {str(level) for level in range(1, 6)}
    first_marker = tmp_path / "test-pyramid-vc6-level-1.json"
    assert json.loads(first_marker.read_text())["output_inventory_sha256"] == result["level_inventory_sha256"]["1"]
    # A missing compressed chunk would otherwise read as the fill value and be
    # silently accepted by the completed-receipt fast path.
    chunk = next(p for p in (path / "1").glob("*/*/*") if p.is_file())
    chunk.unlink()
    with pytest.raises(ValueError, match="Pyramid level 1 inventory mismatch"):
        module.build_one(item, tmp_path, workers=4)
    # The interrupted-build resume path must also reject the missing chunk.
    (tmp_path / "test-multiscales-complete.json").unlink()
    group = zarr.open_group(str(path), mode="a")
    group.attrs["multiscales_complete"] = False
    with pytest.raises(ValueError, match="Pyramid level 1 inventory mismatch"):
        module.build_one(item, tmp_path, workers=4)


def test_inventory_legacy_compatibility_is_explicit(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("build_surface3d_multiscales")
    module.verify_level_inventory(tmp_path / "1", None, required=False)
    with pytest.raises(ValueError, match="no completed inventory"):
        module.verify_level_inventory(tmp_path / "1", None, required=True)
    with pytest.raises(ValueError, match="inventory mismatch"):
        module.verify_level_inventory(tmp_path / "1", "missing", required=False)


def test_shared_xyz_pooling_matches_original_reference():
    a = np.random.default_rng(0).integers(0,256,(5,9,11),dtype=np.uint8)
    expected = np.empty((3,5,6),dtype=np.uint8)
    for z,y,x in np.ndindex(expected.shape):
        expected[z,y,x] = np.rint(a[2*z:2*z+2,2*y:2*y+2,2*x:2*x+2].mean())
    np.testing.assert_array_equal(downsample_mean(a),expected)


def test_half_up_matches_vc_integer_formula():
    a = np.array([[[0,1,2],[0,1,3],[4,5,6]]],dtype=np.uint8)
    np.testing.assert_array_equal(downsample_mean(a,(1,2,2),rounding="half_up"),
                                  np.array([[[1,3],[5,6]]],dtype=np.uint8))
    assert downsample_mean(a,(1,2,2))[0,0,0] == 0  # Legacy tie-to-even stays unchanged.


def test_metadata_independent_z_spacing_matches_vc(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    module=importlib.import_module("build_surface3d_multiscales")
    attrs={"multiscales":[{"version":"0.4","datasets":[{"path":"0", "coordinateTransformations":[
        {"type":"scale","scale":[24.0,4.0,4.0]}, {"type":"translation","translation":[0.0,0.0,0.0]}]}]}]}
    datasets=module.make_metadata(attrs)[0]["datasets"]
    assert len(datasets)==6
    for level,d in enumerate(datasets):
        assert d["coordinateTransformations"]==[
            {"type":"scale","scale":[24.0,4.0*2**level,4.0*2**level]},
            {"type":"translation","translation":[0.0,0.0,0.0]}]
