"""Release portability and strict content identity, without private asset fixtures."""
import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    return importlib.import_module("distilled_3d_ink")


def test_release_asset_cannot_escape_root(tmp_path, cli):
    with pytest.raises(ValueError, match="escapes"):
        cli.asset_path(tmp_path, "../private.json")


def test_release_alias_is_content_checked(tmp_path, cli):
    path = tmp_path/"student.pth"
    path.write_bytes(b"checkpoint")
    release = {"aliases":{"latest":"student.pth"},"files":{"student.pth":{"sha256":cli.sha256(path)}}}
    assert cli.checkpoint_path(tmp_path,release,"latest") == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="verified asset"):
        cli.checkpoint_path(tmp_path,release,"latest")


def test_relocation_preserves_resume_identity_but_not_changed_weights():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    source = {"model_config":{"canonical":{"source_dir":"/old/code", "with_norm":True}},
              "manifest_sha256":"dataset", "distillation":{"coarse":{"checkpoint":"/old/teacher", "sha256":"weights"}}}
    relocated = deepcopy(source)
    relocated["model_config"]["canonical"]["source_dir"] = "/new/code"
    relocated["distillation"]["coarse"]["checkpoint"] = "/new/teacher"
    assert not validate_resume_signature(source,relocated)
    relocated["distillation"]["coarse"]["sha256"] = "other weights"
    with pytest.raises(ValueError,match="Resume configuration"):
        validate_resume_signature(source,relocated)


def test_dataset_reads_use_relocated_root_without_changing_manifest(monkeypatch):
    import vesuvius.ink_detection.data.multiteacher as data
    dataset = object.__new__(data.FlatDistillationDataset)
    dataset._volumes = {}
    dataset.path_roots = {"/old/data":"/new/data"}
    reads = []
    monkeypatch.setattr(data,"open_volume",lambda path,level:reads.append(str(path)) or "array")
    assert dataset._open("/old/data/scroll/segment.zarr") == "array"
    assert reads == ["/new/data/scroll/segment.zarr"]


def test_queue_rejects_mixing_checkpoints(tmp_path, cli):
    module=importlib.import_module("run_surface3d_inference")
    plan={"root":str(tmp_path),"tile_size":256,"checkpoint":"one", "datasets":[{"id":"sample","shape":[65,256,256]}]}
    queue=module.initialize(plan);queue.close()
    with pytest.raises(ValueError,match="mix inference"):
        module.initialize({**plan,"checkpoint":"two"})


def test_pooling_rebuilds_geometry_from_native_scale(cli):
    module=importlib.import_module("build_surface3d_multiscales")
    attrs={"multiscales":[{"axes":[{"name":a,"type":"space"} for a in "zyx"],
           "datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[2,3,4]},
                        {"type":"translation","translation":[5,6,7]}]}]}]}
    result=module.make_metadata(attrs)[0]["datasets"]
    assert len(result)==6
    assert result[-1]["coordinateTransformations"] == [
        {"type":"scale","scale":[2,96,128]}, {"type":"translation","translation":[5,6,7]}]
