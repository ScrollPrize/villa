from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

from lasagna.inference_provenance import code_commit, base_document, finalize_document, validate_portable_bundle


def test_code_commit_is_full_git_hash() -> None:
    value = code_commit(Path(__file__))
    assert len(value) == 40
    assert set(value) <= set("0123456789abcdef")


def test_code_commit_accepts_packaged_build_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VILLA_CODE_COMMIT", "a" * 40)
    assert code_commit(Path("/outside/checkout")) == "a" * 40


def test_portable_provenance_is_redacted_and_structural(tmp_path: Path) -> None:
    root = tmp_path / "presence.ome.zarr"
    for level, shape in ((3, [8, 7, 6]), (4, [4, 4, 3])):
        path = root / str(level)
        path.mkdir(parents=True)
        (path / ".zarray").write_text(json.dumps({
            "zarr_format": 2,
            "shape": shape,
            "chunks": [4, 4, 4],
            "dtype": "|u1",
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3},
        }), encoding="utf-8")
    manifest = tmp_path / "fiber.lasagna.json"
    manifest.write_text(json.dumps({
        "version": 2,
        "groups": {
            "presence": {
                "zarr": "presence.ome.zarr/3",
                "scaledown": 3,
                "channels": ["presence"],
            }
        },
    }), encoding="utf-8")
    document = base_document(
        artifact_kind="fiber3d-prediction",
        context={
            "run_uuid": "run-1",
            "source": {"volume_id": "v1"},
            "hostname": "private-host",
            "checkpoint_path": "/private/model.pt",
        },
    )
    written = finalize_document(
        document, path=tmp_path / "inference.json", status="completed",
        manifest_path=manifest,
    )

    assert written["status"] == "completed"
    assert "hostname" not in written
    assert "checkpoint_path" not in written
    zarr_entry = written["artifacts"][1]
    assert zarr_entry["path"] == "presence.ome.zarr"
    assert [level["level"] for level in zarr_entry["levels"]] == [3, 4]
    assert zarr_entry["levels"][0]["compressor"]["cname"] == "zstd"


@pytest.mark.parametrize("artifact_kind", ["fiber3d-prediction", "lasagna"])
def test_bundle_is_backend_neutral_and_self_contained_after_move(tmp_path: Path, artifact_kind: str) -> None:
    source = tmp_path / "source"
    root = source / "channel.ome.zarr" / "0"
    root.mkdir(parents=True)
    (root / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [2, 2, 2], "chunks": [2, 2, 2],
        "dtype": "|u1", "compressor": None,
    }), encoding="utf-8")
    manifest = source / "result.lasagna.json"
    manifest.write_text(json.dumps({
        "groups": {"channel": {"zarr": "channel.ome.zarr/0", "channels": ["channel"]}},
    }), encoding="utf-8")
    document = base_document(artifact_kind=artifact_kind, context={"run_uuid": "portable"})
    finalize_document(document, path=source / "inference.json", status="completed", manifest_path=manifest)
    moved = tmp_path / "moved" / "artifacts"
    shutil.copytree(source, moved)

    validated = validate_portable_bundle(moved)
    assert validated["artifact_kind"] == artifact_kind
    assert all(not Path(item["path"]).is_absolute() for item in validated["artifacts"])
