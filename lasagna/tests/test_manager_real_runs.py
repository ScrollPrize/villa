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
import shutil

import pytest

from lasagna.manager.config import ManagerConfig
from lasagna.manager.fiberlets import finalize_fiberlet_provenance, resolve_fiberlet_inputs
from lasagna.manager.open_data import resolve_model_id, validate_inference
from lasagna.manager.snapshots import import_snapshot


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
    config = ManagerConfig(output_dir=str(fiber_path.parent), cache_dir=str(tmp_path))
    fiber, normal = resolve_fiberlet_inputs(config, fiber_path.name, normal_path.name)
    assert fiber["source"]["sample_id"] == normal["source"]["sample_id"]
    assert fiber["source"]["volume_id"] == normal["source"]["volume_id"]
    assert fiber["manifest_sha256"] and normal["manifest_sha256"]


def test_existing_fiberlet_bundle_has_bounded_publication_plan(tmp_path: Path) -> None:
    run = _run_path("LAS_MANAGER_TEST_FIBERLET_RUN")
    config = ManagerConfig(
        output_dir=str(run.parent), cache_dir=str(tmp_path),
        upload_staging_s3="s3://test-only/not-uploaded",
    )
    plan = validate_inference(config, run.name)
    assert plan.files is None  # No materialized payload inventory.
    assert plan.model_id is None  # Derived artifact, not a new model.
    record = json.loads((run / "metadata.json").read_text())
    local_dataset = tmp_path / record["artifacts"]["dataset"]
    for name in (".zgroup", ".zattrs", "anchors/.zarray", "prefix/.zarray", "routes/.zarray"):
        destination = local_dataset / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / record["artifacts"]["dataset"] / name, destination)
    finalize_fiberlet_provenance(tmp_path, record)
    rebuilt = json.loads((tmp_path / record["artifacts"]["provenance"]).read_text())
    assert rebuilt["fiberlet_dataset"] == plan.provenance["fiberlet_dataset"]


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
