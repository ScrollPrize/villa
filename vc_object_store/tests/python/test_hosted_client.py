"""HostedStore from Python against a vc_object_server subprocess (VC_OBJECT_SERVER names the binary)."""
import json
import os
import re
import subprocess
import uuid
import http.client
import http.server
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from object_helpers import fixture_guards
import numpy as np

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
    return dict(command_version=3, request=str(uuid.uuid4()), store=status["store"], generation=status["generation"],
                author=author, expected_objects=fixture_guards(status, expected), expected_relationships={}, read_dependencies={},
                expected_commit_sequence=None, operations=operations)


@pytest.mark.parametrize("cache_bytes", [1 << 20, 8 << 20])
def test_copy_transfers_each_geometry_byte_once(hosted, tmp_path, cache_bytes):
    from vc_object_store.surfaces import create_surface, read_surface

    store, _ = hosted
    points = np.arange(512 * 512 * 3, dtype=np.float32).reshape(512, 512, 3) + 1
    surface = create_surface(store.endpoint, points, name="large", scale=1, volume="a", tool="pytest")
    destination = tmp_path / "copy.vcobj"
    vos.ObjectStore.create(str(destination), author="pytest").close()
    upstream = urlsplit(store.endpoint)
    transfers = []

    class CountingProxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=10)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                connection.request(self.command, self.path, body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                data = response.read()
                if "/arrays/points?" in self.path:
                    transfers.append(len(data))
                self.send_response(response.status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                connection.close()

        do_GET = forward
        do_POST = forward

    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CountingProxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    session = vos.ObjectStoreSession()
    try:
        remote = session.attach_hosted(f"http://127.0.0.1:{proxy.server_port}", poll_interval_ms=3600000,
                                       retries=0, cache_bytes=cache_bytes)
        local = session.attach(str(destination), "pytest")
        copied = session.copy_object(remote, local, surface)
        np.testing.assert_array_equal(read_surface(destination, copied, volume="a").points, points)
        assert sum(transfers) == points.nbytes
        if cache_bytes < points.nbytes:
            assert max(transfers) <= 1 << 20
        else:
            assert len(transfers) == 1
    finally:
        for attachment in session.attachments():
            session.detach(attachment)
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=10)


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


def test_session_typed_operations_on_hosted_and_local(hosted, tmp_path):
    store, path = hosted
    local_path = tmp_path / "local.vcobj"
    vos.ObjectStore.create(str(local_path), author="pytest", hostname="testhost").close()
    session = vos.ObjectStoreSession()
    remote = session.attach_hosted(store.endpoint, poll_interval_ms=3600000, retries=1)
    local = session.attach(str(local_path), "pytest")
    assert session.hosted(remote) and not session.hosted(local)
    assert session.freshness(remote)["error"] == "" and session.freshness(local) is None
    session.take_events()
    for attachment in (remote, local):
        status = session.status(attachment)
        obj = str(uuid.uuid4())
        receipt = session.submit(attachment, request(status, [dict(op="create_object", record=dict(uuid=obj, kind="note", name="n"))],
                                                     {obj: None}))
        assert receipt["objects"][obj] == 1 and session.current_object(attachment, obj)["version"] == 1
        ref = dict(store=status["store"], generation=status["generation"], object=obj, version=1)
        assert session.objects_at_versions(attachment, [ref])[0]["record"]["name"] == "n"
        source = tmp_path / "w.bin"
        source.write_bytes(b"w" * 100)
        session.publish_artifacts(attachment, ref, {"w.bin": str(source)}, "pytest")
        if attachment == remote:
            session.refresh(remote)
            events = [e for e in session.take_events() if e["kind"] == "committed"]
            assert events and not events[-1]["revalidate"] and events[-1]["changes"][-1]["kind"] == "note"
        files = session.materialize_artifacts(attachment, obj, str(tmp_path / f"out-{attachment.store_uuid}"))
        assert files[0]["length"] == 100 and (tmp_path / f"out-{attachment.store_uuid}" / "w.bin").read_bytes() == b"w" * 100
        historical = vos.object_reference(session.current_object(attachment, obj)["record"])
        source.write_bytes(b"replacement")
        session.publish_artifacts(attachment, historical, {"w.bin": str(source)}, "pytest")
        session.submit(attachment, request(session.status(attachment), [dict(op="delete_object", uuid=obj)],
                                           {obj: historical["version"] + 1}))
        assert session.artifacts(attachment, obj, reference=historical) == files
        selected = session.artifact_file(attachment, obj, "w.bin", str(tmp_path / "exact-cache"), reference=historical)
        assert Path(selected).read_bytes() == b"w" * 100
        output = tmp_path / f"exact-{attachment.store_uuid}"
        session.materialize_artifacts(attachment, obj, str(output), reference=historical)
        assert (output / "w.bin").read_bytes() == b"w" * 100
    with pytest.raises(ValueError):
        session.path(remote)  # hosted attachments have no local database

    # Durable saves: the request is written before it is sent.
    drafts = tmp_path / "drafts"
    assert session.use_draft_directory(str(drafts))["checkpoint"] == 0
    status = session.status(remote)
    obj = str(uuid.uuid4())
    session.submit_durably(remote, request(status, [dict(op="create_object", record=dict(uuid=obj, kind="note"))], {obj: None}))
    assert session.pending_submissions() == [] and session.draft_checkpoint_error() == ""
    assert session.checkpoint_drafts() >= 2


def test_surface_and_document_helpers_take_an_endpoint(hosted):
    """The location helpers give the same results through a served store as through its file."""
    import numpy as np
    from vc_object_store.documents import attach_document, attached_document, point_collections
    from vc_object_store.surfaces import create_surface, read_surface
    store, path = hosted
    endpoint, status = store.endpoint, store.status()
    xyz = np.stack([*np.mgrid[:3, :4][::-1], np.full((3, 4), 5)], axis=-1).astype(np.float32)
    xyz[0, 0] = -1
    surface = create_surface(endpoint, xyz, name="served", scale=2.0, volume="vol", tool="test",
                             channels={"d": np.arange(12, dtype="<f4").reshape(3, 4)}, author="remote")
    grid = read_surface(endpoint, surface, volume="vol")
    local = read_surface(path, surface, volume="vol")  # read-only opens are allowed while hosted
    assert np.array_equal(grid.points, local.points) and np.array_equal(grid.valid, local.valid)
    assert grid.points[0, 0].tolist() == [-1, -1, -1] and not grid.valid[0, 0]
    assert grid.scale == local.scale and grid.channels == frozenset({"d"}) and grid.version == 1
    assert grid.record["meta"]["provenance"]["tool"] == "test" and grid.generation == status["generation"]
    document = {"summary": 1, "points": [{"id": 1, "p": [1, 2, 3]}]}
    collection = attach_document(endpoint, surface, document, role="results", list_key="points",
                                 position_key="p", name="results", volume="vol", author="remote")
    assert attached_document(endpoint, surface, "results") == document == attached_document(path, surface, "results")
    [read] = point_collections(endpoint, [collection], "vol")
    assert read["uuid"] == collection and [list(p) for p in read["points"]] == [[1, 2, 3]]
    assert read_surface(endpoint, surface, volume="vol").version == 2  # the attachment versioned the surface
    session = vos.ObjectStoreSession()
    attachment = session.attach_location(endpoint, "remote")
    try:
        assert session.hosted(attachment)
        names = session.select_objects(attachment, {"selection": {"kind": "surface"}, "fields": ["name"]})
        assert [row["name"] for row in names] == ["served"]
        channel = np.array(session.read_geometry_arrays(attachment, [grid.geometry_revision], "d").values)
        assert channel.reshape(-1).tolist() == list(range(12))
    finally:
        session.detach(attachment)
