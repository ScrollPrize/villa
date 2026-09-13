"""Transport retries converge without accepting duplicate logical inputs."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import threading
import zipfile

import pytest

from service_http import ApiError
from service_uploads import UploadEnvironment, UploadManager, UPLOAD_GC_SECONDS


@pytest.fixture
def manager(tmp_path):
    return UploadManager(UploadEnvironment(
        lock=threading.RLock(), output_root=lambda: tmp_path,
        session_id=lambda: "workspace", ephemeral_dir=lambda: tmp_path / "inputs",
        require_session=lambda: None))


DATA = b'{"type":"vc3d_fiber","version":1,"points":[]}'


def request(data=DATA):
    return {"upload_id": "a" * 32, "kind": "fiber", "id": "fiber-1",
            "files": [{"name": "fiber.json", "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}]}


def test_lost_create_and_finalize_responses_return_one_transfer(manager):
    upload_id = manager.begin(request())["upload_id"]
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA), len(DATA))
    finalized = manager.finalize(upload_id)
    assert manager.begin(request())["upload_id"] == upload_id
    assert manager.finalize(upload_id).record == finalized.record
    assert len(manager.uploads) == 1
    assert manager.status(upload_id)["state"] == "finalized"


def test_concurrent_create_and_changed_manifest(manager):
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: manager.begin(request()), range(16)))
    assert all(result == results[0] for result in results)
    with pytest.raises(ApiError, match="different content"):
        manager.begin(request(b"different"))
    assert len(manager.uploads) == 1


def test_stable_id_cannot_replace_a_legacy_transfer(manager):
    legacy = request()
    del legacy["upload_id"]
    upload_id = manager.begin(legacy)["upload_id"]
    replacement = dict(request(b"different"), upload_id=upload_id)
    with pytest.raises(ApiError, match="already exists"):
        manager.begin(replacement)
    assert manager.uploads[upload_id].declared_bytes() == len(DATA)


def test_cancelled_pcl_transfer_releases_legacy_target_reservation(manager):
    first = dict(request(), kind="pcl", role="same_winding",
                 operation="replace_collection", target_collection_id="0",
                 base_source_revision="0" * 64)
    manager.begin(first)
    manager.cancel(first["upload_id"])
    second = dict(first, upload_id="b" * 32)
    assert manager.begin(second)["accepted"]


def test_resume_partial_body_and_lost_chunk_response(manager):
    upload_id = manager.begin(request())["upload_id"]
    with pytest.raises(ApiError, match="ended early"):
        manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[:7]), 10, offset=0)
    assert manager.status(upload_id)["files"][0]["offset"] == 7
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[7:14]), 7, offset=7)
    # The caller never sees that response; an identical retry cannot append.
    with pytest.raises(ApiError) as error:
        manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[7:14]), 7, offset=7)
    assert error.value.payload["offset"] == 14
    assert manager.status(upload_id)["files"][0]["offset"] == 14
    with pytest.raises(ApiError, match="missing declared files"):
        manager.finalize(upload_id)
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[14:]), len(DATA) - 14, offset=14)
    assert manager.status(upload_id)["files"][0]["received"]
    manager.finalize(upload_id)


def test_bad_resumed_digest_resets_file_without_finalizing(manager):
    upload_id = manager.begin(request())["upload_id"]
    manager.receive(upload_id, "fiber.json", io.BytesIO(b"!" * 5), 5, offset=0)
    with pytest.raises(ApiError, match="SHA-256"):
        manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[5:]), len(DATA) - 5, offset=5)
    assert manager.status(upload_id)["files"][0]["offset"] == 0
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA), len(DATA), offset=0)
    manager.finalize(upload_id)


def test_whole_file_retry_after_partial_transfer_cleans_partial_file(manager):
    upload_id = manager.begin(request())["upload_id"]
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA[:5]), 5, offset=0)
    manager.receive(upload_id, "fiber.json", io.BytesIO(DATA), len(DATA))
    manager.finalize(upload_id)  # Single-JSON validation must see one file.


def test_cancellation_retains_receipt_and_never_recreates_transfer(manager):
    upload_id = manager.begin(request())["upload_id"]
    manager.cancel(upload_id)
    manager.cancel(upload_id)
    assert manager.begin(request())["upload_id"] == upload_id
    assert manager.status(upload_id)["state"] == "cancelled"
    assert not manager.uploads[upload_id].staging_dir.exists()
    assert manager.staged_ephemeral_bytes() == 0
    with pytest.raises(ApiError, match="cancelled"):
        manager.receive(upload_id, "fiber.json", io.BytesIO(DATA), len(DATA))
    with pytest.raises(ApiError, match="cancelled"):
        manager.finalize(upload_id)


def test_stable_uploads_remain_reconcilable_after_legacy_gc_deadline(manager):
    upload_id = manager.begin(request())["upload_id"]
    manager.uploads[upload_id].created -= UPLOAD_GC_SECONDS + 1
    manager.collect_garbage()
    assert manager.status(upload_id)["state"] == "transferring"


def test_deduplicated_checkpoint_has_a_status_receipt_with_stable_id(manager):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", b"checkpoint")
    data = buffer.getvalue()
    manifest = dict(request(data), kind="checkpoint")
    upload_id = manager.begin(manifest)["upload_id"]
    manager.receive(upload_id, "fiber.json", io.BytesIO(data), len(data))
    manager.finalize(upload_id)
    manifest["upload_id"] = "b" * 32
    reused = manager.begin(manifest)
    assert reused["deduplicated"]
    assert manager.begin(manifest) == reused
    assert manager.status("b" * 32)["state"] == "finalized"
    assert manager.finalize("b" * 32).record == reused["input"]


def test_finalize_waits_for_a_file_writer(manager):
    upload_id = manager.begin(request())["upload_id"]
    started, finish = threading.Event(), threading.Event()

    class Stream(io.BytesIO):
        def read(self, length):
            started.set()
            assert finish.wait(5)
            return super().read(length)

    with ThreadPoolExecutor(2) as pool:
        receive = pool.submit(manager.receive, upload_id, "fiber.json", Stream(DATA), len(DATA))
        assert started.wait(5)
        finalize = pool.submit(manager.finalize, upload_id)
        finish.set()
        receive.result(5)
        assert finalize.result(5).record["id"] == "fiber-1"
