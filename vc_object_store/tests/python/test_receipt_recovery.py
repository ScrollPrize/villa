"""Rejected legacy receipts must not turn an unknown outcome into a new request."""
import http.client
import http.server
import json
import sqlite3
import threading
import uuid
from urllib.parse import urlsplit

import pytest
import vc_object_store as vos
from test_hosted_client import SERVER, hosted, request  # noqa: F401 - shared server fixture


@pytest.mark.parametrize("references", [None, {}])
def test_local_unreadable_replay_stays_pending(tmp_path, references):
    path = str(tmp_path / "local.vcobj")
    vos.ObjectStore.create(path, author="test").close()
    session = vos.ObjectStoreSession()
    attachment = session.attach(path, "test")
    session.use_draft_directory(tmp_path / "drafts")
    uid = str(uuid.uuid4())
    command = request(session.status(attachment), [dict(op="create_object", record=dict(uuid=uid, kind="note"))], {uid: None})
    try:
        session.submit(attachment, command)
        # Fixture-only fault injection: simulate a saved pre-contract receipt.
        with sqlite3.connect(path) as sql:
            original = sql.execute("SELECT receipt FROM transaction_receipts").fetchone()[0]
            invalid = json.loads(original)
            if references is None:
                invalid.pop("object_references")
            else:
                invalid["object_references"] = references
            sql.execute("DROP TRIGGER transaction_receipts_UPDATE")
            sql.execute("UPDATE transaction_receipts SET receipt=?", (json.dumps(invalid),))
        with pytest.raises(vos.ReceiptDecodeError, match="preserve the pending request"):
            session.submit_durably(attachment, command)
        assert [p["request_uuid"] for p in session.pending_submissions()] == [command["request"]]
        with sqlite3.connect(path) as sql:
            sql.execute("UPDATE transaction_receipts SET receipt=?", (original,))
        assert session.resolve_pending(attachment, command["request"])["replayed"] is False
        assert session.pending_submissions() == []
        assert session.current_object(attachment, uid)["version"] == 1
    finally:
        session.detach(attachment)


@pytest.mark.skipif(not SERVER, reason="set VC_OBJECT_SERVER to a vc_object_server binary")
def test_hosted_unreadable_success_stays_pending(hosted, tmp_path):
    store, _ = hosted
    upstream = urlsplit(store.endpoint)
    corrupt = [True]

    class Proxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=10)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                connection.request(self.command, self.path, body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                data = response.read()
                if corrupt[0] and self.path == "/v1/transactions" and response.status == 200:
                    receipt = json.loads(data)
                    receipt.pop("object_references")
                    data = json.dumps(receipt).encode()
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                connection.close()

        do_GET = forward
        do_POST = forward

    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    session = vos.ObjectStoreSession()
    attachment = None
    try:
        attachment = session.attach_hosted(f"http://127.0.0.1:{proxy.server_port}", author="test", retries=0)
        session.use_draft_directory(tmp_path / "drafts")
        uid = str(uuid.uuid4())
        command = request(session.status(attachment), [dict(op="create_object", record=dict(uuid=uid, kind="note"))], {uid: None})
        with pytest.raises(vos.ReceiptDecodeError, match="preserve the pending request"):
            session.submit_durably(attachment, command)
        assert store.current_object_revision(uid)["version"] == 1  # success was real
        assert [p["request_uuid"] for p in session.pending_submissions()] == [command["request"]]
        corrupt[0] = False
        session.resolve_pending(attachment, command["request"])
        assert session.pending_submissions() == []
    finally:
        if attachment is not None:
            session.detach(attachment)
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=10)
