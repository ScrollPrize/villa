"""Opt-in contract validation using copied real PCL, patch and fiber inputs.

This exercises transport/catalog/publication, not resident fitting or editors.
Never writes to SPIRAL_REVISION_DATASET; all publication targets are temporary.
"""

import io
import json
import os
from pathlib import Path
import shutil
import threading
from uuid import uuid4

import pytest

from input_publication import Output, PublicationTransaction, fingerprint
from input_workspace import Catalog, Change, Content, InputIdentity, MutationCoordinator
from service_http import sha256_file
from service_uploads import UploadEnvironment, UploadManager


@pytest.mark.skipif(not os.environ.get("SPIRAL_REVISION_DATASET"),
                    reason="set SPIRAL_REVISION_DATASET to a real scroll dataset")
def test_real_mixed_inputs_transfer_catalog_and_publication(tmp_path, monkeypatch):
    dataset = Path(os.environ["SPIRAL_REVISION_DATASET"]).resolve()
    patch_name = os.environ.get("SPIRAL_REVISION_PATCH")
    patch = (dataset / "verified_patches" / patch_name if patch_name
             else next(path for path in sorted((dataset / "verified_patches").iterdir())
                       if (path / "meta.json").is_file()))
    fiber = next(iter(sorted((dataset / "fibers").glob("*.json"))))
    pcl = dataset / "same_windings.json"
    original = {path: fingerprint(path) for path in (patch, fiber, pcl)}
    manager = UploadManager(UploadEnvironment(
        lock=threading.RLock(), output_root=lambda: tmp_path / "output",
        session_id=lambda: "real-input-workspace",
        ephemeral_dir=lambda: tmp_path / "content",
        require_session=lambda: None))
    catalog = Catalog()
    document = json.loads(pcl.read_text())
    collection_id = min(document["collections"], key=int)
    # The runtime identity is one logical collection, preserving its source id.
    document["collections"] = {collection_id: document["collections"][collection_id]}
    selected_pcl = tmp_path / "source-pcl.json"
    selected_pcl.write_text(json.dumps(document))
    inputs = [("pcl", selected_pcl), ("patch", patch), ("fiber", fiber)]
    outputs, source_copies, batch = [], [], []
    targets = tmp_path / "dataset-copy"
    targets.mkdir()
    for kind, source in inputs:
        name = source.name if kind == "patch" else f"{kind}.json"
        target = targets / name
        if source.is_dir():
            shutil.copytree(source, target)
            files = sorted(path for path in source.rglob("*") if path.is_file())
        else:
            shutil.copy2(source, target)
            files = [source]
        input_id = str(uuid4())
        logical = InputIdentity(input_id, kind, str(target),
                                "same_winding" if kind == "pcl" else None,
                                int(collection_id) if kind == "pcl" else None)
        catalog.register_base(logical, Content.from_json({"fingerprint": fingerprint(target)}))
        if kind == "pcl":
            # Only the temporary selected document is edited. Coordinates,
            # ordering and annotations remain exactly as in the real source.
            edited = json.loads(source.read_text())
            edited["collections"][collection_id]["name"] += " (validation draft)"
            source.write_text(json.dumps(edited))
        manifest = {
            "upload_id": uuid4().hex, "kind": kind, "id": input_id,
            "files": [{"name": path.relative_to(source).as_posix()
                       if source.is_dir() else path.name,
                       "size": path.stat().st_size, "sha256": sha256_file(path)}
                      for path in files],
        }
        if kind == "pcl":
            manifest["role"] = "same_winding"
        upload_id = manager.begin(manifest)["upload_id"]
        assert manager.begin(manifest)["upload_id"] == upload_id
        for path, entry in zip(files, manifest["files"]):
            data = path.read_bytes()
            split = len(data) // 2
            manager.receive(upload_id, entry["name"], io.BytesIO(data[:split]), split, offset=0)
            # Reconnect status confirms progress after a lost chunk response.
            status = manager.status(upload_id)
            actual = next(item for item in status["files"] if item["name"] == entry["name"])
            assert actual["offset"] == split
            manager.receive(upload_id, entry["name"], io.BytesIO(data[split:]), len(data) - split,
                            offset=split)
        finalized = manager.finalize(upload_id)
        assert manager.finalize(upload_id).record == finalized.record
        staged = Path(finalized.record["path"])
        assert fingerprint(staged) == fingerprint(source)
        source_copies.append(staged)
        outputs.append(Output(target, staged, fingerprint(target)))
        batch.append(Change(logical, 1, Content.from_json({"upload_id": upload_id})))

    # Explicitly only catalog bookkeeping: no claim that a fitter applied it.
    accepted = catalog.accept(batch)
    catalog.mark_applied(accepted)
    transaction = PublicationTransaction.prepare(outputs)
    coordinator = MutationCoordinator()
    real_replace = os.replace

    def lose_first_publish_response(src, dst):
        real_replace(src, dst)
        raise OSError("injected publication response loss")

    def publish(_):
        transaction.resume()
        catalog.mark_persisted(accepted)
        return {"persisted": [(revision.id, revision.number) for revision in accepted]}

    selection = [[revision.id, revision.number] for revision in accepted]
    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", lose_first_publish_response)
        with pytest.raises(OSError):
            coordinator.execute("commit", "commit", selection, publish, recoverable=True)
    result = coordinator.execute("commit", "commit", selection, publish, recoverable=True)
    assert coordinator.execute("commit", "commit", selection, publish, recoverable=True) == result
    assert all(entry.persisted == 2 for entry in catalog.entries())
    for output, staged in zip(outputs, source_copies):
        assert fingerprint(output.target) == fingerprint(staged)
    transaction.release()

    # Deletion and restore remain available in the catalog after application;
    # persistence deletes only the isolated managed entries.
    deleted = catalog.accept([Change(entry.identity, 2, None) for entry in catalog.entries()])
    catalog.mark_applied(deleted)
    assert all(entry["can_restore"] for entry in catalog.status())
    deletion = PublicationTransaction.prepare([
        Output(output.target, None, fingerprint(output.target)) for output in outputs])
    deletion.resume()
    catalog.mark_persisted(deleted)
    deletion.release()
    assert all(not output.target.exists() for output in outputs)
    assert all(not entry["can_restore"] for entry in catalog.status())
    assert {path: fingerprint(path) for path in original} == original
