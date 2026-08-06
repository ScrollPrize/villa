from __future__ import annotations

import json
from pathlib import Path

import pytest

from lasagna.manager.config import ManagerConfig
from lasagna.manager.open_data import (
    INCOMPLETE_MARKER,
    UPLOAD_MANIFEST,
    stage_upload,
    upload_inference,
    validate_inference,
)
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

    def get_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def delete(self, key: str) -> None:
        self.events.append(("delete", key))
        self.objects.pop(key, None)


def _completed_run(
    tmp_path: Path, *, license_name: str = "CC BY-NC 4.0",
    artifact_kind: str = "fiber3d-prediction",
) -> tuple[ManagerConfig, Path]:
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
        "model": {"atlas_model_id": "20260806120000", "task": "lasagna"},
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
    return ManagerConfig(
        output_dir=str(output), cache_dir=str(tmp_path / "cache"),
        atlas_dir=str(tmp_path / "atlas"), upload_staging_s3="s3://stage/prefix",
    ), run


def test_validate_requires_completed_cc_bundle_and_model(tmp_path: Path) -> None:
    config, _run = _completed_run(tmp_path)
    plan = validate_inference(config, "fiber-run")
    assert plan.prefix == "prefix/inference/run-uuid"
    assert plan.model_id == "20260806120000"
    assert any(item["path"] == "presence.ome.zarr/3/.zarray" for item in plan.files)

    bad_config, _ = _completed_run(tmp_path / "bad", license_name="private")
    with pytest.raises(ValueError, match="CC BY-NC"):
        validate_inference(bad_config, "fiber-run")


def test_validate_accepts_lasagna_through_same_upload_path(tmp_path: Path) -> None:
    config, _run = _completed_run(tmp_path, artifact_kind="lasagna")
    plan = validate_inference(config, "fiber-run")
    assert plan.provenance["artifact_kind"] == "lasagna"


@pytest.mark.parametrize("artifact_kind", ["fiber3d-prediction", "lasagna"])
def test_atomic_upload_marker_manifest_order_and_idempotency(
    tmp_path: Path, artifact_kind: str,
) -> None:
    config, _run = _completed_run(tmp_path, artifact_kind=artifact_kind)
    plan = validate_inference(config, "fiber-run")
    store = FakeStore()
    url, uploaded = stage_upload(plan, store)
    assert uploaded and url == "s3://stage/prefix/inference/run-uuid/"
    marker = f"{plan.prefix}/{INCOMPLETE_MARKER}"
    manifest = f"{plan.prefix}/{UPLOAD_MANIFEST}"
    assert marker not in store.objects
    assert store.events[0] == ("bytes", marker)
    assert store.events[-2:] == [("bytes", manifest), ("delete", marker)]
    assert json.loads(store.objects[manifest])["bundle_digest"] == plan.bundle_digest

    event_count = len(store.events)
    assert stage_upload(plan, store) == (url, False)
    assert len(store.events) == event_count

    remote = json.loads(store.objects[manifest])
    remote["bundle_digest"] = "different"
    store.objects[manifest] = json.dumps(remote).encode()
    with pytest.raises(ValueError, match="collision"):
        stage_upload(plan, store)


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
    assert calls[0]["register_model"] is False
    persisted = json.loads((run / "metadata.json").read_text())
    assert persisted["upload"]["bundle_digest"]


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
