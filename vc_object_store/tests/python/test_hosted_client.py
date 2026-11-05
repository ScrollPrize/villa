"""HostedStore from Python against a vc_object_server subprocess (VC_OBJECT_SERVER names the binary)."""
import json
import os
import re
import subprocess
import uuid

import pytest

import vc_object_store as vos

SERVER = os.environ.get("VC_OBJECT_SERVER")
pytestmark = pytest.mark.skipif(not SERVER, reason="set VC_OBJECT_SERVER to a vc_object_server binary")


@pytest.fixture()
def hosted(tmp_path):
    path = tmp_path / "hosted.vcobj"
    vos.ObjectStore.create(str(path), author="pytest", hostname="testhost").close()
    server = subprocess.Popen([SERVER, str(path), "--port", "0"], stderr=subprocess.PIPE, text=True)
    try:
        line = server.stderr.readline()
        port = re.search(r":(\d+)\s*$", line)
        assert port, line
        yield vos.HostedStore(f"http://127.0.0.1:{port.group(1)}", retries=1), path
    finally:
        server.terminate()
        assert server.wait(timeout=30) == 0


def request(status, operations, expected, author="py"):
    return dict(command_version=1, request=str(uuid.uuid4()), store=status["store"], generation=status["generation"],
                author=author, expected_objects=expected, expected_relationships={}, read_dependencies={},
                expected_commit_sequence=None, operations=operations)


def test_hosted_requests_queries_and_artifacts(hosted, tmp_path):
    store, path = hosted
    status = store.status()
    obj = str(uuid.uuid4())
    create = request(status, [dict(op="create_object", record=dict(uuid=obj, kind="note", name="n"))], {obj: None})
    receipt = store.submit(create)
    assert store.submit(create)["replayed"] and store.receipt(create["request"]) == dict(receipt, replayed=False)
    with pytest.raises(RuntimeError):
        store.submit(request(status, [dict(op="set_meta", uuid=obj, key="k", value=1)], {obj: None}))

    revision, as_of = store.current_object_revision(obj, with_status=True)
    assert revision["version"] == 1 and as_of["commit_sequence"] == status["commit_sequence"] + 1
    rows = store.select_all_objects(dict(selection=dict(kind="note")), page_size=1)
    assert [r["uuid"] for r in rows] == [obj] and rows[0]["version"] == 1
    page, page_status = store.select_objects(dict(selection=dict(kind="note", limit=10)), with_status=True)
    assert page == rows and page_status == as_of
    ref = dict(store=status["store"], generation=status["generation"], object=obj, version=1)
    assert store.objects_at_versions([ref])[0]["record"]["name"] == "n"
    record = store.record_selection(dict(selection=dict(kind="note")))
    assert record["objects"] == [ref] and vos.decode_selection_record(vos.encode_selection_record(record)) == record
    changes = store.changes_since(dict(store=status["store"], generation=status["generation"], sequence=0))
    assert changes["commits"][-1]["changes"][0]["uuid"] == obj

    source = tmp_path / "model.bin"
    source.write_bytes(os.urandom(4096))
    digest = store.put_artifact(source)
    files = [dict(name="model.bin", sha256=digest, length=4096)]
    store.submit(request(status, [dict(op="set_meta", uuid=obj, key=vos.ARTIFACTS_META_KEY,
                                       value=json.loads(vos.artifact_manifest_json(files)))], {obj: 1}))
    assert store.read_artifact(digest, 10, 5) == source.read_bytes()[10:15]
    store.fetch_artifact(files[0], tmp_path / "copy.bin")
    assert (tmp_path / "copy.bin").read_bytes() == source.read_bytes()
    assert [r["version"] for r in store.object_history(obj)] == [1, 2]
