"""Opt-in validation against existing artifacts; no inference or uploads.

LAS_MANAGER_TEST_FIBER_RUN, LAS_MANAGER_TEST_NORMAL_RUN, and
LAS_MANAGER_TEST_FIBERLET_RUN name completed manager run directories.
LAS_MANAGER_TEST_SNAPSHOT optionally supplies the Fiber run's exact checkpoint.
Only the test's temporary directory is written; large Zarr payloads are not copied.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from lasagna.manager.config import ManagerConfig
from lasagna.manager.fiberlets import resolve_fiberlet_inputs
from lasagna.manager.open_data import resolve_model_id, validate_inference
from lasagna.manager.snapshots import import_snapshot, index_snapshots


def _run_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to validate existing artifacts")
    path = Path(value).resolve()
    assert (path / "metadata.json").is_file(), path
    return path


def test_existing_fiber_and_normal_inputs(tmp_path: Path) -> None:
    fiber_path = _run_path("LAS_MANAGER_TEST_FIBER_RUN")
    normal_path = _run_path("LAS_MANAGER_TEST_NORMAL_RUN")
    assert fiber_path.parent == normal_path.parent
    roots = tuple(filter(None, os.environ.get("LAS_MANAGER_TEST_SNAPSHOT_DIRS", "").split(os.pathsep)))
    config = ManagerConfig(
        output_dir=str(fiber_path.parent), cache_dir=str(tmp_path), snapshot_dirs=roots,
        model_dir=str(tmp_path / "models"),
    )
    # Historical bundles predate mandatory registered repository identity.
    # Register their exact checkpoints in the temporary test model directory;
    # never modify the production snapshots or their sidecars.
    snapshots = index_snapshots(config, write_cache=False)
    for run in (fiber_path, normal_path):
        record = json.loads((run / "metadata.json").read_text())
        provenance = json.loads((run / record["artifacts"]["provenance"]).read_text())
        model = provenance["model"]
        if model.get("model_identifier"):
            snapshot = next(item for item in snapshots if item.sha256 == model["sha256"])
            import_snapshot(config, snapshot.path, run=model["run"],
                            model_identifier=model["model_identifier"])
    fiber, normal = resolve_fiberlet_inputs(config, fiber_path.name, normal_path.name)
    assert fiber["source"]["sample_id"] == normal["source"]["sample_id"]
    assert fiber["source"]["volume_id"] == normal["source"]["volume_id"]
    assert fiber["manifest_sha256"] and normal["manifest_sha256"]
    assert fiber["model"]["atlas_model_id"] and normal["model"]["atlas_model_id"]


def test_existing_fiberlet_bundle_has_bounded_publication_plan(tmp_path: Path) -> None:
    run = _run_path("LAS_MANAGER_TEST_FIBERLET_RUN")
    config = ManagerConfig(
        output_dir=str(run.parent), cache_dir=str(tmp_path),
        upload_staging_s3="s3://test-only/not-uploaded",
    )
    record = json.loads((run / "metadata.json").read_text())
    attrs = json.loads((run / record["artifacts"]["dataset"] / ".zattrs").read_text())
    if attrs.get("format_version") != 3:
        with pytest.raises(ValueError):
            validate_inference(config, run.name)
        return
    plan = validate_inference(config, run.name)
    assert plan.files is None  # No materialized payload inventory.
    assert plan.model_id is None  # Derived artifact, not a new model.
    assert plan.provenance["fiberlet_dataset"]["format_version"] == 3


def test_existing_checkpoint_import_preserves_publication_identity(tmp_path: Path) -> None:
    run = _run_path("LAS_MANAGER_TEST_FIBER_RUN")
    checkpoint = os.environ.get("LAS_MANAGER_TEST_SNAPSHOT")
    if not checkpoint:
        pytest.skip("set LAS_MANAGER_TEST_SNAPSHOT to test checkpoint import")
    record = json.loads((run / "metadata.json").read_text())
    provenance = json.loads((run / record["artifacts"]["provenance"]).read_text())
    config = ManagerConfig(model_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"))
    model = provenance["model"]
    imported = import_snapshot(
        config, checkpoint, run=model["run"], model_identifier=model["model_identifier"],
    )
    assert imported.sha256 == model["sha256"]
    assert imported.model_identifier == model["model_identifier"]
    assert resolve_model_id(config, provenance).endswith("-lasagna")
    # Indexing a second, byte-identical source copy must not create an ambiguity.
    from lasagna.manager.snapshots import index_snapshots

    with_source = replace(config, snapshot_dirs=(str(Path(checkpoint).parent),))
    assert len(index_snapshots(with_source)) == 1
