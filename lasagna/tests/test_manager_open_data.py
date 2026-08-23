from __future__ import annotations

import json
from pathlib import Path

import pytest

from lasagna.manager.config import ManagerConfig
from lasagna.manager.open_data import (
    INCOMPLETE_MARKER,
    S3ObjectStore,
    stage_upload,
    upload_inference,
    validate_inference,
    resolve_model_id,
)
from lasagna.manager.snapshots import SnapshotRecord
from lasagna.manager.runs import atomic_json


class FakeStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.events: list[tuple[str, str]] = []

    def put_file(self, key: str, path: Path) -> None:
        self.events.append(("file", key))
        self.objects[key] = path.read_bytes()

    def put_bytes(self, key: str, value: bytes) -> None:
        self.events.append(("bytes", key))
        self.objects[key] = value

    def delete(self, key: str) -> None:
        self.events.append(("delete", key))
        self.objects.pop(key, None)


class FakeS3Client:
    def upload_file(self, *_args) -> None:
        raise AssertionError("bulk upload must use rclone")


def test_s3_bulk_upload_uses_configured_rclone_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a").write_bytes(b"a")
    (bundle / "nested").mkdir()
    (bundle / "nested" / "b").write_bytes(b"b")
    calls = []

    def fake_run(command, *, check):
        file_list = Path(command[command.index("--files-from-raw") + 1])
        calls.append((command, check, file_list.read_text(encoding="utf-8")))

    monkeypatch.setattr("lasagna.manager.open_data.subprocess.run", fake_run)
    store = S3ObjectStore(
        "stage", client=FakeS3Client(),
        rclone_params=("--transfers", "512", "--buffer-size", "2M"),
    )
    store.put_files(
        "root/inference/run", bundle,
        ("a", "nested/b"),
    )
    command, check, file_list = calls[0]
    assert command[:4] == ["rclone", "copy", str(bundle), ":s3:stage/root/inference/run/"]
    assert command[4:8] == ["--transfers", "512", "--buffer-size", "2M"]
    assert check is True
    assert file_list == "a\nnested/b\n"


def _snapshot_record(path: Path, sha256: str) -> SnapshotRecord:
    return SnapshotRecord(
        backend="fiber3d", run="s1_128_20260801_084232", checkpoint="best.pt",
        selector="fiber3d/s1_128_20260801_084232/best.pt", path=str(path),
        size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha256,
        step=None, metric_name=None, metric_value=None, patch_shape=None,
        architecture="fiber_trace_3d", option_count=None, precision_policy=None,
        atlas_model_id=None, model_creation_utc=None, process="fiber_trace_3d.train",
        task="lasagna", output_schema=None, code_revision=None,
    )


def test_model_resolution_rehashes_snapshot_and_derives_numeric_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "s1_128_20260801_084232" / "snapshots" / "best.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"checkpoint")
    import hashlib
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    record = _snapshot_record(snapshot, digest)
    monkeypatch.setattr("lasagna.manager.open_data.index_snapshots", lambda _config, **_kwargs: [record])
    config = ManagerConfig(snapshot_dirs=(str(tmp_path),), cache_dir=str(tmp_path / "cache"))
    assert resolve_model_id(config, {"model": {
        "sha256": digest, "run": record.run, "checkpoint": record.checkpoint,
    }}) == "20260801084232-lasagna"
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no configured snapshot"):
        resolve_model_id(config, {"model": {
            "sha256": digest, "atlas_model_id": "20260801084232",
        }})


def test_model_creation_offset_is_normalized_to_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "run" / "snapshots" / "best.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"checkpoint")
    import hashlib
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    record = SnapshotRecord(**{
        **_snapshot_record(snapshot, digest).__dict__,
        "model_creation_utc": "2026-08-01T10:42:32+02:00",
    })
    monkeypatch.setattr("lasagna.manager.open_data.index_snapshots", lambda _config, **_kwargs: [record])
    config = ManagerConfig(snapshot_dirs=(str(tmp_path),), cache_dir=str(tmp_path / "cache"))
    assert resolve_model_id(config, {"model": {"sha256": digest}}) == "20260801084232-lasagna"
    naive = SnapshotRecord(**{**record.__dict__, "model_creation_utc": "2026-08-01T08:42:32"})
    monkeypatch.setattr("lasagna.manager.open_data.index_snapshots", lambda _config, **_kwargs: [naive])
    with pytest.raises(ValueError, match="UTC offset"):
        resolve_model_id(config, {"model": {"sha256": digest}})


def test_model_resolution_accepts_byte_identical_checkpoint_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "s1_128_20260801_084232" / "snapshots"
    run.mkdir(parents=True)
    best = run / "best.pt"
    named = run / "best91_5k.pt"
    best.write_bytes(b"same")
    named.write_bytes(b"same")
    import hashlib
    digest = hashlib.sha256(b"same").hexdigest()
    records = [_snapshot_record(best, digest), _snapshot_record(named, digest)]
    records[1] = SnapshotRecord(**{
        **records[1].__dict__,
        "checkpoint": "best91_5k.pt",
        "selector": "fiber3d/s1_128_20260801_084232/best91_5k.pt",
    })
    monkeypatch.setattr("lasagna.manager.open_data.index_snapshots", lambda _config, **_kwargs: records)
    config = ManagerConfig(snapshot_dirs=(str(tmp_path),), cache_dir=str(tmp_path / "cache"))
    assert resolve_model_id(config, {"model": {
        "sha256": digest, "run": records[0].run, "checkpoint": "best91_5k.pt",
    }}) == "20260801084232-lasagna"


def test_model_resolution_rejects_tampered_atlas_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "run_20260801_084232" / "snapshots" / "best.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"checkpoint")
    import hashlib
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    record = _snapshot_record(snapshot, digest)
    monkeypatch.setattr("lasagna.manager.open_data.index_snapshots", lambda _config, **_kwargs: [record])
    config = ManagerConfig(snapshot_dirs=(str(tmp_path),), cache_dir=str(tmp_path / "cache"))
    base = {
        "sha256": digest, "run": record.run, "checkpoint": record.checkpoint,
        "snapshot": f"{record.run}/snapshots/{record.checkpoint}",
        "architecture": "fiber3d/unet",
    }
    assert resolve_model_id(config, {"model": base}) == "20260801084232-lasagna"
    with pytest.raises(ValueError, match="model.snapshot"):
        resolve_model_id(config, {"model": {**base, "snapshot": "other/snapshots/best.pt"}})
    with pytest.raises(ValueError, match="model.architecture"):
        resolve_model_id(config, {"model": {**base, "architecture": "unet"}})


def _completed_run(
    tmp_path: Path, *, license_name: str = "CC BY-NC 4.0",
    artifact_kind: str = "fiber3d-prediction",
) -> tuple[ManagerConfig, Path]:
    torch = pytest.importorskip("torch")
    snapshot = tmp_path / "runs" / "run_20260806_120000" / "snapshots" / "best.pt"
    snapshot.parent.mkdir(parents=True)
    torch.save({
        "model": {},
        "config": {"model_3d": {"output_channels": 7}},
    }, snapshot)
    import hashlib
    snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    output = tmp_path / "outputs"
    run = output / "fiber-run"
    bundle = run / "artifacts"
    channel = bundle / "presence.ome.zarr" / "3"
    channel.mkdir(parents=True)
    (channel / ".zarray").write_text("{}", encoding="utf-8")
    manifest = bundle / "result.lasagna.json"
    manifest.write_text(json.dumps({
        "groups": {"presence": {"zarr": "presence.ome.zarr/3"}},
    }), encoding="utf-8")
    atomic_json(bundle / "inference.json", {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "status": "completed",
        "run_uuid": "run-uuid",
        "source": {
            "sample_id": "PHerc0001", "volume_id": "20260101000001",
            "requested_group": 2,
            "license": {"name": license_name},
        },
        "model": {
            "atlas_model_id": "20260806120000", "task": "lasagna",
            "architecture": "fiber3d/unet", "run": "run_20260806_120000",
            "checkpoint": "best.pt",
            "snapshot": "run_20260806_120000/snapshots/best.pt",
            "sha256": snapshot_sha256,
        },
        "artifacts": [
            {"kind": "manifest", "path": "result.lasagna.json"},
            {"kind": "ome-zarr-channel", "path": "presence.ome.zarr"},
        ],
    })
    atomic_json(run / "metadata.json", {
        "run_uuid": "run-uuid", "run_name": "fiber-run", "status": "completed",
        "created_at": "2026-08-06T00:00:00Z",
        "artifacts": {"root": "artifacts", "provenance": "artifacts/inference.json"},
        "lifecycle": {
            "inference": "completed", "staging_upload": "not_started",
            "atlas_ingest": "not_started", "atlas_publication": "not_started",
        },
    })
    config = ManagerConfig(
        output_dir=str(output), cache_dir=str(tmp_path / "cache"),
        atlas_dir=str(tmp_path / "atlas"), upload_staging_s3="s3://stage/prefix",
        snapshot_dirs=(str(tmp_path / "runs"),),
    )
    return config, run


def _completed_fiberlet_run(tmp_path: Path) -> tuple[ManagerConfig, Path]:
    output = tmp_path / "outputs"
    run = output / "fiberlet-run"
    bundle = run / "artifacts"
    dataset = bundle / "fiberlets.zarr"
    for relative in ("anchors/.zarray", "prefix/.zarray", "routes/.zarray"):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (dataset / ".zgroup").write_text("{}", encoding="utf-8")
    (dataset / ".zattrs").write_text(json.dumps({
        "vc_format": "fiberlet_dataset",
        "dataset_kind": "combined",
    }), encoding="utf-8")
    atomic_json(bundle / "fiberlets.json", {
        "schema_version": 1,
        "artifact_kind": "fiberlets",
        "status": "completed",
        "run_uuid": "fiberlet-uuid",
        "source": {
            "sample_id": "PHerc0001", "volume_id": "20260101000001",
            "license": {"name": "CC BY-NC 4.0"},
        },
        "dependencies": {},
        "artifacts": [{"kind": "fiberlet-zarr", "path": "fiberlets.zarr"}],
    })
    atomic_json(run / "metadata.json", {
        "run_uuid": "fiberlet-uuid", "run_name": "fiberlet-run",
        "job_kind": "fiberlet", "active_lifecycle_phase": "fiberlet_preprocess",
        "artifact_kind": "fiberlets", "status": "completed",
        "artifacts": {
            "root": "artifacts", "provenance": "artifacts/fiberlets.json",
        },
        "lifecycle": {
            "fiberlet_preprocess": "completed", "staging_upload": "not_started",
            "atlas_ingest": "not_started", "atlas_publication": "not_started",
        },
    })
    return ManagerConfig(
        output_dir=str(output), cache_dir=str(tmp_path / "cache"),
        atlas_dir=str(tmp_path / "atlas"), upload_staging_s3="s3://stage/prefix",
    ), run


def test_fiberlet_upload_streams_tree_in_distinct_namespace(tmp_path: Path) -> None:
    config, _run = _completed_fiberlet_run(tmp_path)
    plan = validate_inference(config, "fiberlet-run")
    assert plan.prefix == "prefix/fiberlets/fiberlet-uuid"
    assert plan.model_id is None
    assert plan.files is None

    class TreeStore(FakeStore):
        def put_tree(self, prefix: str, bundle: Path) -> None:
            self.events.append(("tree", f"{prefix}:{bundle.name}"))

        def put_file(self, _key: str, _path: Path) -> None:
            raise AssertionError("streamed Fiberlet upload must not enumerate files")

    store = TreeStore()
    assert stage_upload(plan, store) == "s3://stage/prefix/fiberlets/fiberlet-uuid/"
    assert store.events[0] == ("bytes", f"{plan.prefix}/{INCOMPLETE_MARKER}")
    assert store.events[1] == ("tree", f"{plan.prefix}:artifacts")
    assert store.events[-1] == ("delete", f"{plan.prefix}/{INCOMPLETE_MARKER}")


def test_validate_requires_completed_cc_bundle_and_model(tmp_path: Path) -> None:
    config, _run = _completed_run(tmp_path)
    assert not (Path(config.cache_dir) / "snapshots" / "index.json").exists()
    plan = validate_inference(config, "fiber-run")
    assert plan.prefix == "prefix/inference/run-uuid"
    assert plan.model_id == "20260806120000-lasagna"
    assert not (Path(config.cache_dir) / "snapshots" / "index.json").exists()
    assert "presence.ome.zarr/3/.zarray" in plan.files

    bad_config, _ = _completed_run(tmp_path / "bad", license_name="private")
    with pytest.raises(ValueError, match="CC BY-NC"):
        validate_inference(bad_config, "fiber-run")


def test_validate_accepts_lasagna_through_same_upload_path(tmp_path: Path) -> None:
    config, _run = _completed_run(tmp_path, artifact_kind="lasagna")
    plan = validate_inference(config, "fiber-run")
    assert plan.provenance["artifact_kind"] == "lasagna"


def test_validate_rejects_reserved_incomplete_marker_in_bundle(tmp_path: Path) -> None:
    config, run = _completed_run(tmp_path)
    (run / "artifacts" / INCOMPLETE_MARKER).write_bytes(b"")
    with pytest.raises(ValueError, match="reserved upload filename"):
        validate_inference(config, "fiber-run")


@pytest.mark.parametrize("artifact_kind", ["fiber3d-prediction", "lasagna"])
def test_marker_guards_each_resumable_upload_attempt(
    tmp_path: Path, artifact_kind: str,
) -> None:
    config, _run = _completed_run(tmp_path, artifact_kind=artifact_kind)
    plan = validate_inference(config, "fiber-run")
    store = FakeStore()
    url = stage_upload(plan, store)
    assert url == "s3://stage/prefix/inference/run-uuid/"
    marker = f"{plan.prefix}/{INCOMPLETE_MARKER}"
    assert marker not in store.objects
    assert store.events[0] == ("bytes", marker)
    assert store.events[-1] == ("delete", marker)

    event_count = len(store.events)
    assert stage_upload(plan, store) == url
    assert len(store.events) > event_count
    assert store.events[event_count] == ("bytes", marker)
    assert store.events[-1] == ("delete", marker)


def test_failed_upload_retains_marker_and_never_ingests(tmp_path: Path) -> None:
    config, run = _completed_run(tmp_path)

    class FailingStore(FakeStore):
        def put_files(self, _prefix: str, _bundle: Path, _files) -> None:
            raise RuntimeError("transfer failed")

    store = FailingStore()
    with pytest.raises(RuntimeError, match="transfer failed"):
        upload_inference(
            config, "fiber-run", store=store,
            validator=lambda **_kwargs: {"validated": True},
            ingester=lambda **_kwargs: pytest.fail("ingest called after failed staging"),
        )
    plan = validate_inference(config, "fiber-run")
    marker = f"{plan.prefix}/{INCOMPLETE_MARKER}"
    assert marker in store.objects
    persisted = json.loads((run / "metadata.json").read_text())
    assert persisted["lifecycle"]["staging_upload"] == "failed"
    assert persisted["lifecycle"]["atlas_ingest"] == "not_started"


@pytest.mark.parametrize("artifact_kind", ["fiber3d-prediction", "lasagna"])
def test_upload_updates_independent_lifecycle_and_calls_ingester(
    tmp_path: Path, artifact_kind: str,
) -> None:
    config, run = _completed_run(tmp_path, artifact_kind=artifact_kind)
    store = FakeStore()
    calls = []

    def ingest(**kwargs):
        calls.append(kwargs)
        return {"volume_metadata": "data/samples/PHerc0001/volumes/v.json", "model_id": kwargs["model_id"]}

    record = upload_inference(
        config, "fiber-run", store=store,
        validator=lambda **kwargs: {"validated": True}, ingester=ingest,
    )
    assert record["lifecycle"] == {
        "inference": "completed", "staging_upload": "completed",
        "atlas_ingest": "completed", "atlas_publication": "not_started",
    }
    ingested_provenance = json.loads(
        (Path(calls[0]["bundle_dir"]) / "inference.json").read_text(encoding="utf-8")
    )
    assert ingested_provenance["artifact_kind"] == artifact_kind
    assert "register_model" not in calls[0]
    persisted = json.loads((run / "metadata.json").read_text())
    assert persisted["upload"] == {
        "staging_url": "s3://stage/prefix/inference/run-uuid/",
    }


def test_atlas_preflight_failure_happens_before_staging(tmp_path: Path) -> None:
    config, _run = _completed_run(tmp_path)
    store = FakeStore()

    def reject(**_kwargs):
        raise ValueError("unknown Atlas model")

    with pytest.raises(ValueError, match="unknown Atlas model"):
        upload_inference(
            config, "fiber-run", store=store, validator=reject,
            ingester=lambda **_kwargs: pytest.fail("ingest called"),
        )
    assert store.events == []
