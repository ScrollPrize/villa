"""Zarr writes to S3, against a local server that checks every request's signature."""

import hashlib
import hmac
import http.server
import pickle
import re
import threading
import urllib.parse
from xml.sax.saxutils import escape

import numpy as np
import pytest
import zarr

from vc_reader.reader import ObjectStore, S3Credentials, ZarrWriter, create_group, open_zarr

KEYS = S3Credentials("access", "secret-key", region="us-east-1")


def _signature(request, secret):
    authorization = request.headers["Authorization"]
    names = re.search(r"SignedHeaders=([^,]+)", authorization).group(1).split(";")
    credential = re.search(r"Credential=[^/]+/(\d+)/([^/]+)/", authorization)
    day, region = credential.group(1), credential.group(2)
    path, _, query = request.path.partition("?")
    pairs = sorted(urllib.parse.parse_qsl(query, keep_blank_values=True))
    canonical_query = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in pairs)
    headers = "".join(f"{name}:{request.headers[name].strip()}\n" for name in names)
    canonical = "\n".join([request.command, path, canonical_query, headers, ";".join(names), request.headers["x-amz-content-sha256"]])
    date = request.headers["x-amz-date"]
    to_sign = "\n".join(["AWS4-HMAC-SHA256", date, f"{day}/{region}/s3/aws4_request", hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + secret).encode()
    for part in (day, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()


class FakeS3(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    objects: dict[str, bytes]
    secret: str
    fail_puts: list[int]
    methods: list[str]

    def log_message(self, *args):
        pass

    def _respond(self, status, body=b""):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self):
        expected = _signature(self, self.secret) if "Authorization" in self.headers else None
        if expected is None or f"Signature={expected}" not in self.headers["Authorization"]:
            self._respond(403, b"<Error><Code>SignatureDoesNotMatch</Code></Error>")
            return False
        return True

    def _key(self):
        return urllib.parse.unquote(self.path.partition("?")[0]).removeprefix("/bucket/")

    def _query(self):
        return dict(urllib.parse.parse_qsl(self.path.partition("?")[2], keep_blank_values=True))

    def do_POST(self):
        if not self._authorized():
            return self._respond(403)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        query = self._query()
        if "uploads" in query:
            return self._respond(200, b"<InitiateMultipartUploadResult><UploadId>the-id</UploadId></InitiateMultipartUploadResult>")
        numbers = [int(n) for n in re.findall(r"<PartNumber>(\d+)</PartNumber>", body.decode())]
        self.objects[self._key()] = b"".join(self.parts[n] for n in numbers)
        type(self).completed_uploads = getattr(type(self), "completed_uploads", 0) + 1
        self._respond(200, b"<CompleteMultipartUploadResult></CompleteMultipartUploadResult>")

    def do_PUT(self):
        if "partNumber" in self._query():
            if not self._authorized():
                return self._respond(403)
            part = self.rfile.read(int(self.headers["Content-Length"]))
            type(self).parts = {**getattr(type(self), "parts", {}), int(self._query()["partNumber"]): part}
            self.send_response(200)
            self.send_header("ETag", f'"{hashlib.md5(part).hexdigest()}"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.methods.append("PUT")
        if not self._authorized():
            return
        if hashlib.sha256(body).hexdigest() != self.headers["x-amz-content-sha256"]:
            return self._respond(400, b"<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
        if self.fail_puts:
            return self._respond(self.fail_puts.pop())
        self.objects[self._key()] = body
        self._respond(200)

    def do_DELETE(self):
        self.methods.append("DELETE")
        if self._authorized():
            self.objects.pop(self._key(), None)
            self._respond(204)

    def do_HEAD(self):
        self.methods.append("HEAD")
        if self._authorized():
            self._respond(200 if self._key() in self.objects else 404)

    def do_GET(self):
        self.methods.append("GET")
        if not self._authorized():
            return
        query = dict(urllib.parse.parse_qsl(self.path.partition("?")[2]))
        if "list-type" not in query:
            key = self._key()
            if key not in self.objects:
                return self._respond(404)
            data = self.objects[key]
            wanted = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if not wanted:
                return self._respond(200, data)
            first = int(wanted[1]) if wanted[1] else len(data) - int(wanted[2])
            last = int(wanted[2]) if wanted[1] and wanted[2] else len(data) - 1
            part = data[first:last + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(part)))
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
            self.end_headers()
            return self.wfile.write(part)
        prefix, delimiter = query.get("prefix", ""), query.get("delimiter", "")
        items = []  # (key or common prefix, is_prefix) in key order, as S3 pages them
        for key in sorted(self.objects):
            if key.startswith(prefix):
                rest = key[len(prefix):]
                if delimiter and delimiter in rest:
                    common = prefix + rest.split(delimiter)[0] + delimiter
                    if not items or items[-1] != (common, True):
                        items.append((common, True))
                else:
                    items.append((key, False))
        start = int(query.get("continuation-token", "0"))
        end = min(len(items), start + int(query.get("max-keys", "1000")))
        contents = "".join(f"<Contents><Key>{escape(k)}</Key><Size>{len(self.objects[k])}</Size></Contents>" for k, p in items[start:end] if not p)
        common = "".join(f"<CommonPrefixes><Prefix>{escape(k)}</Prefix></CommonPrefixes>" for k, p in items[start:end] if p)
        truncated = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{end}</NextContinuationToken>" if end < len(items) else "<IsTruncated>false</IsTruncated>"
        self._respond(200, f"<ListBucketResult>{truncated}{contents}{common}</ListBucketResult>".encode())


@pytest.fixture
def s3():
    handler = type("Handler", (FakeS3,), {"objects": {}, "secret": KEYS.secret_access_key, "fail_puts": [], "methods": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    handler.auth = S3Credentials(KEYS.access_key_id, KEYS.secret_access_key, region=KEYS.region,
                                 endpoint_url=f"http://127.0.0.1:{server.server_address[1]}")
    yield handler
    server.shutdown()
    thread.join()


def _local_copy(s3, root, prefix):
    for key, data in s3.objects.items():
        if key.startswith(prefix):
            path = root / key[len(prefix):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def test_array_objects_are_what_a_local_write_stores(s3, tmp_path):
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    kwargs = dict(shape=(5, 6), chunks=(2, 3), dtype="u2", cname="zstd", shuffle=2, fill_value=7)
    remote = ZarrWriter.create("s3://bucket/run/a.zarr", auth=s3.auth, **kwargs)
    remote[:] = data
    local = ZarrWriter.create(tmp_path / "run" / "a.zarr", **kwargs)
    local[:] = data
    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert s3.objects["run/" + path.relative_to(tmp_path / "run").as_posix()] == path.read_bytes()
    assert sorted(s3.objects) == sorted("run/" + p.relative_to(tmp_path / "run").as_posix()
                                        for p in (tmp_path / "run").rglob("*") if p.is_file())
    _local_copy(s3, tmp_path / "copy", "run/a.zarr/")
    np.testing.assert_array_equal(zarr.open_array(tmp_path / "copy", mode="r")[:], data)


def test_regions_read_back_and_partial_chunks_are_patched(s3):
    w = ZarrWriter.create("s3://bucket/a", shape=(4, 4), chunks=(2, 2), dtype="f4", auth=s3.auth)
    w[:] = 1.5
    w[1, 1] = 9
    expected = np.full((4, 4), 1.5, np.float32)
    expected[1, 1] = 9
    np.testing.assert_array_equal(w[:], expected)
    np.testing.assert_array_equal(open_zarr("s3://bucket/a", auth=s3.auth)[:], expected)
    reopened = ZarrWriter.open("s3://bucket/a", auth=s3.auth)
    assert reopened.shape == (4, 4) and reopened.dtype == np.float32
    np.testing.assert_array_equal(reopened[:], expected)


def test_skip_empty_overwrite_and_chunks(s3):
    w = ZarrWriter.create("s3://bucket/a", shape=(4, 4), chunks=(2, 2), dtype="u1", auth=s3.auth, skip_empty=True)
    w[:] = 3
    assert w.chunk_exists((0, 0))
    w[0:2, 0:2] = 0
    assert not w.chunk_exists((0, 0)) and "a/0.0" not in s3.objects
    assert w.remove_chunk((1, 1)) and not w.remove_chunk((1, 1))
    w = ZarrWriter.create("s3://bucket/a", shape=(2, 2), chunks=(2, 2), dtype="u1", auth=s3.auth, overwrite=True)
    assert not any(k.startswith("a/0") or k.startswith("a/1") for k in s3.objects)
    w.write_chunk((0, 0), np.ones((2, 2), np.uint8))
    np.testing.assert_array_equal(w[:], 1)


def test_groups_attributes_and_fill_value(s3):
    g = create_group("s3://bucket/g", attrs={"x": 1}, auth=s3.auth)
    a = g.create_array("0", shape=(2,), chunks=(2,), dtype="u1")
    g.create_group("sub").attrs["y"] = 2
    assert g.keys() == ["0", "sub"] and "0" in g and "nothing" not in g
    assert dict(g.attrs) == {"x": 1} and dict(g["sub"].attrs) == {"y": 2}
    a.attrs["k"] = [1, 2]
    assert dict(g["0"].attrs) == {"k": [1, 2]}
    a.set_fill_value(5)
    np.testing.assert_array_equal(a[:], 5)
    del g["0"]
    assert g.keys() == ["sub"] and not any(k.startswith("g/0/") for k in s3.objects)
    assert list(create_group("s3://bucket/g", overwrite=True, auth=s3.auth).keys()) == []
    assert "g/sub/.zgroup" not in s3.objects


@pytest.mark.parametrize("version,shards", [(3, None), (3, (4, 4))])
def test_v3_and_shards(s3, tmp_path, version, shards):
    w = ZarrWriter.create("s3://bucket/v3", shape=(8, 8), chunks=(2, 2), dtype="u2", version=version, shards=shards, auth=s3.auth)
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    w[:] = data
    w[0, 0] = 99
    data[0, 0] = 99
    np.testing.assert_array_equal(w[:], data)
    assert w.chunk_exists((0, 0))
    assert w.remove_chunk((0, 0)) and not w.chunk_exists((0, 0))
    _local_copy(s3, tmp_path / "copy", "v3/")
    data[0:2, 0:2] = 0
    np.testing.assert_array_equal(zarr.open_array(tmp_path / "copy", mode="r")[:], data)


def test_writer_pickles_with_its_access(s3):
    w = ZarrWriter.create("s3://bucket/a", shape=(2,), chunks=(2,), dtype="u1", auth=s3.auth)
    copy = pickle.loads(pickle.dumps(w))
    copy[:] = 4
    np.testing.assert_array_equal(w[:], 4)


def test_object_store(s3, tmp_path):
    for location, auth in (("s3://bucket/root", s3.auth), (tmp_path / "root", None)):
        store = ObjectStore(location, auth=auth)
        assert store.read("a/b.json") is None and not store.exists("a/b.json")
        store.write("a/b.json", '{"x": 1}')
        store.write("c", b"\x00\x01")
        assert store.read("a/b.json") == b'{"x": 1}' and store.exists("c")
        assert store.keys() == ["a/b.json", "c"] and store.sizes()["c"] == 2 and store.children() == ["a"]
        assert store.keys("a/") == ["a/b.json"] and store.keys("c") == ["c"] and store.keys("z") == []
        assert [o["key"] for o in store.objects("c")] == ["c"] and store.objects("c")[0]["size"] == 2
        # One level, paged: children and objects in key order, relative keys, a token between pages.
        store.write("a/d", b"1")
        store.write("e/f", b"2")
        store.write("g", b"3")
        first = store.list_page(max_keys=2)
        assert first["children"] == ["a"] and [o["key"] for o in first["objects"]] == ["c"]
        assert first["next_continuation_token"]
        second = store.list_page(continuation_token=first["next_continuation_token"], max_keys=2)
        assert second["children"] == ["e"] and [o["key"] for o in second["objects"]] == ["g"]
        assert second["next_continuation_token"] == ""
        below = store.list_page("a/")
        assert [o["key"] for o in below["objects"]] == ["a/b.json", "a/d"] and below["children"] == []
        assert store.list_page("z") == {"objects": [], "children": [], "next_continuation_token": ""}
        for key in ("a/d", "e/f", "g"):
            store.remove(key)
        assert pickle.loads(pickle.dumps(store)).read("c") == b"\x00\x01"
        assert store.remove("c") and not store.remove("c")
        assert store.remove_all() and store.keys() == []


def test_object_store_files(s3, tmp_path):
    store = ObjectStore("s3://bucket/files", auth=s3.auth)
    small = tmp_path / "small.bin"
    small.write_bytes(b"small")
    store.write_file("small.bin", small)
    assert store.read("small.bin") == b"small"
    # Past one part: a multipart upload, read back whole and by streaming.
    large = tmp_path / "large.bin"
    payload = np.random.default_rng(0).integers(0, 255, (65 << 20) + 17, dtype=np.uint8).tobytes()
    large.write_bytes(payload)
    store.write_file("large.bin", large)
    assert store.read_file("large.bin", tmp_path / "back.bin") is True
    assert (tmp_path / "back.bin").read_bytes() == payload
    assert s3.completed_uploads == 1
    assert store.read_file("absent.bin", tmp_path / "absent.bin") is False
    assert not list(tmp_path.glob("absent*"))
    local = ObjectStore(tmp_path / "local", auth=None)
    local.write_file("a/b.bin", small)
    assert local.read("a/b.bin") == b"small"
    assert local.read_file("a/b.bin", tmp_path / "c.bin") and (tmp_path / "c.bin").read_bytes() == b"small"


def test_transient_failures_are_retried_and_denials_raise(s3):
    store = ObjectStore("s3://bucket/x", auth=s3.auth)
    s3.fail_puts.extend([503, 500])
    store.write("a", b"1")
    assert store.read("a") == b"1"
    wrong = S3Credentials("access", "other", region="us-east-1", endpoint_url=s3.auth.endpoint_url)
    with pytest.raises(PermissionError):
        ObjectStore("s3://bucket/x", auth=wrong).write("a", b"2")
    assert store.read("a") == b"1"


def test_writing_needs_s3_credentials(s3, monkeypatch):
    for variable in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises((ValueError, RuntimeError)):
        ZarrWriter.create("https://example.com/a", shape=(1,), chunks=(1,), dtype="u1", auth=None)
    with pytest.raises((ValueError, RuntimeError)):
        ObjectStore("https://example.com/a", auth=None).write("a", b"1")
