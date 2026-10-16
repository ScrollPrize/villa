"""Credentials (S3 SigV4, HTTP basic auth) and offline (cache-only) reads."""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import http.server
import json
import os
import pickle
import threading
import urllib.parse
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native", reason="the native extension is not built")

from test_zarr_reader import (  # noqa: E402
    BLOSC, CHUNKS, MISSING, _expected, _foreign_cache, _serve, _vc3d_mirror, _write_zarr)

from vc_reader import _native  # noqa: E402
from vc_reader.errors import CacheMissError  # noqa: E402
from vc_reader.reader import S3Credentials, ZarrReader, load_basic_auth  # noqa: E402
from vc_reader.reader import auth as auth_mod  # noqa: E402
from vc_reader.reader.auth import Access, resolve_access  # noqa: E402

SHAPE = (37, 50, 45)
KEY, SECRET, TOKEN = "AKIDTEST", "sekrit/key+with=chars", "tok/en+1="


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(0).integers(0, 255, SHAPE, dtype=np.uint8)


@pytest.fixture(autouse=True)
def no_aws_environment(monkeypatch, tmp_path):
    """Tests never see the developer's AWS configuration or reach instance metadata."""
    for var in list(os.environ):
        if var.startswith("AWS_"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


# -- SigV4 -----------------------------------------------------------------------------------


def test_sigv4_matches_aws_documentation_examples():
    # "Signature Calculations for the Authorization Header", Amazon S3 API reference.
    key, secret = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    get = dict(_native.sigv4_headers("https://examplebucket.s3.amazonaws.com/test.txt", key, secret, "",
                                     "us-east-1", "20130524T000000Z", [("range", "bytes=0-9")]))
    assert get["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")
    listing = dict(_native.sigv4_headers("https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J", key,
                                         secret, "", "us-east-1", "20130524T000000Z"))
    assert listing["authorization"].endswith(
        "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


@pytest.mark.parametrize("url", [
    "http://bucket.s3.eu-west-2.amazonaws.com/vol.zarr/0/1.2.3",
    "http://s3.eu-west-2.amazonaws.com/my.bucket/a%20b/c~d_e-f.g",
    "http://127.0.0.1:9000/bucket/?list-type=2&delimiter=%2F&prefix=vol.zarr%2F&continuation-token=a%2Bb%3D",
])
@pytest.mark.parametrize("token", ["", TOKEN])
def test_sigv4_matches_botocore(monkeypatch, url, token):
    auth = pytest.importorskip("botocore.auth")
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    when = datetime.datetime(2026, 10, 16, 12, 34, 56)
    monkeypatch.setattr(auth, "get_current_datetime", lambda: when)
    req = AWSRequest(method="GET", url=url)
    auth.S3SigV4Auth(Credentials(KEY, SECRET, token or None), "s3", "eu-west-2").add_auth(req)
    ours = dict(_native.sigv4_headers(url, KEY, SECRET, token, "eu-west-2", when.strftime("%Y%m%dT%H%M%SZ")))
    assert ours["authorization"] == req.headers["Authorization"]
    assert ours.get("x-amz-security-token") == req.headers.get("X-Amz-Security-Token")


# -- a fake S3 that checks signatures ---------------------------------------------------------


def _q(s: str) -> str:
    return urllib.parse.quote(s, safe="-_.~")


def _expected_signature(handler, secret: str, scope: str, signed: list[str], amz_date: str) -> str:
    """SigV4 of the received request, computed from the specification (not by the reader's code)."""
    url = urllib.parse.urlsplit(handler.path)
    query = sorted((_q(k), _q(v)) for k, v in urllib.parse.parse_qsl(url.query, keep_blank_values=True))
    canonical = "\n".join([
        "GET",
        urllib.parse.quote(urllib.parse.unquote(url.path), safe="/-_.~"),
        "&".join(f"{k}={v}" for k, v in query),
        "".join(f"{h}:{handler.headers[h].strip()}\n" for h in signed),
        ";".join(signed),
        handler.headers["x-amz-content-sha256"],
    ])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + secret).encode()
    for part in scope.split("/"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()


class _S3Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_message = lambda *a: None  # noqa: E731

    def _reply(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("AWS4-HMAC-SHA256 "):
            return False
        fields = dict(part.strip().split("=", 1) for part in auth[len("AWS4-HMAC-SHA256 "):].split(","))
        access_key, scope = fields["Credential"].split("/", 1)
        keys = self.server.keys
        if access_key not in keys:
            return False
        secret, token = keys[access_key]
        signed = fields["SignedHeaders"].split(";")
        if "host" not in signed or "x-amz-date" not in signed or (token and self.headers.get("x-amz-security-token") != token):
            return False
        if not scope.endswith(f"/{self.server.region}/s3/aws4_request"):
            return False
        expected = _expected_signature(self, secret, scope, signed, self.headers["x-amz-date"])
        return hmac.compare_digest(expected, fields["Signature"])

    def do_GET(self):  # noqa: N802
        self.server.requests.append((self.path, self.headers.get("Authorization", "")))
        if not self._authorized():
            return self._reply(403, b"<Error><Code>SignatureDoesNotMatch</Code></Error>")
        url = urllib.parse.urlsplit(self.path)
        bucket, _, key = urllib.parse.unquote(url.path).lstrip("/").partition("/")
        root = Path(self.server.root, bucket)
        q = dict(urllib.parse.parse_qsl(url.query, keep_blank_values=True))
        if q.get("list-type") == "2":
            return self._reply(200, self._listing(root, q))
        path = root / key
        if path.is_file():
            return self._reply(200, path.read_bytes())
        self._reply(404, b"<Error><Code>NoSuchKey</Code></Error>")

    def _listing(self, root: Path, q: dict) -> bytes:
        """ListObjectsV2 with delimiter "/", one common prefix per page (to exercise continuation)."""
        prefix = q.get("prefix", "")
        d = root / prefix
        subdirs = sorted(e.name for e in os.scandir(d) if e.is_dir()) if d.is_dir() else []
        start = int(q.get("continuation-token") or 0)
        page = subdirs[start:start + 1]
        more = start + 1 < len(subdirs)
        ns = "http://s3.amazonaws.com/doc/2006-03-01/"
        items = "".join(f"<CommonPrefixes><Prefix>{prefix}{name}/</Prefix></CommonPrefixes>" for name in page)
        token = f"<NextContinuationToken>{start + 1}</NextContinuationToken>" if more else ""
        return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{ns}"><Prefix>{prefix}</Prefix>'
                f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>{token}{items}</ListBucketResult>").encode()


@pytest.fixture(scope="module")
def s3(tmp_path_factory, data):
    """A bucket "bkt" holding vol.zarr (OME-like group, level 0 blosc) behind signature checks."""
    pytest.importorskip("numcodecs")
    root = tmp_path_factory.mktemp("s3")
    group = root / "bkt" / "vol.zarr"
    group.mkdir(parents=True)
    (group / ".zgroup").write_text('{"zarr_format": 2}')
    _write_zarr(str(group / "0"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    (group / "labels").mkdir()
    (group / "labels" / ".zgroup").write_text('{"zarr_format": 2}')
    (group / "notes").mkdir()  # listed by S3, but not a zarr node
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _S3Handler)
    httpd.root, httpd.region, httpd.requests = str(root), "eu-west-2", []
    httpd.keys = {KEY: (SECRET, TOKEN), "AKIDOTHER": ("other-secret", "")}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _creds(base, **kw):
    return S3Credentials(kw.pop("key", KEY), kw.pop("secret", SECRET), kw.pop("token", TOKEN),
                         region="eu-west-2", endpoint_url=base, **kw)


def test_signed_s3_reads(s3, data, tmp_path):
    httpd, base = s3
    exp = _expected(data, MISSING, CHUNKS)
    with ZarrReader.open("s3://bkt/vol.zarr/0", auth=_creds(base), cache_dir=tmp_path) as r:
        np.testing.assert_array_equal(r.read((0, 0, 0), SHAPE), exp)
        assert r.access.kind == "s3"
        assert r.stats()["remote_missing"] == len(MISSING)  # 404s, not 403s
        r.flush()
    # Signed caches are keyed by the s3:// URL, like vesuvius' RemoteChunkedArray caches.
    assert (tmp_path / "s3" / "bkt" / "vol.zarr" / "0" / ".zarray").is_file()
    assert all(a.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/") for _, a in httpd.requests)


def test_signed_reads_survive_pickle(s3, data):
    _, base = s3
    exp = _expected(data, MISSING, CHUNKS)
    r = pickle.loads(pickle.dumps(ZarrReader.open("s3://bkt/vol.zarr/0", auth=_creds(base))))
    np.testing.assert_array_equal(r.read((3, 4, 5), (20, 20, 20)), exp[3:23, 4:24, 5:25])
    r.close()


@pytest.mark.parametrize("bad", [dict(secret="wrong"), dict(token="stale"), dict(key="AKIDNOBODY")])
def test_bad_credentials_raise_permission_error(s3, bad):
    _, base = s3
    with pytest.raises(PermissionError, match="403"):
        ZarrReader.open("s3://bkt/vol.zarr/0", auth=_creds(base, **bad))


def test_anonymous_request_to_private_bucket_hints_at_auth(s3):
    _, base = s3
    with pytest.raises(PermissionError, match="pass auth="):
        ZarrReader.open(f"{base}/bkt/vol.zarr/0")


def test_refreshed_credentials_reach_the_native_reader(s3, data, monkeypatch):
    httpd, base = s3

    class Rotating:  # stands in for botocore's refreshing credentials
        def __init__(self):
            self.keys = S3Credentials(KEY, SECRET, TOKEN)

        def current(self):
            return self.keys

    source = Rotating()
    access = Access("s3", source=source, region="eu-west-2", endpoint_url=base)
    monkeypatch.setattr(auth_mod, "REFRESH_INTERVAL_S", 0.0)
    r = ZarrReader.open("s3://bkt/vol.zarr/0", auth=access)
    r.read((0, 0, 0), (1, 1, 1))
    source.keys = S3Credentials("AKIDOTHER", "other-secret")  # the old keys expire
    httpd.keys = {"AKIDOTHER": ("other-secret", "")}
    try:
        httpd.requests.clear()
        np.testing.assert_array_equal(r.read((20, 20, 20), (5, 5, 5)), _expected(data, MISSING, CHUNKS)[20:25, 20:25, 20:25])
        assert httpd.requests and all("Credential=AKIDOTHER/" in a for _, a in httpd.requests)
    finally:
        httpd.keys = {KEY: (SECRET, TOKEN), "AKIDOTHER": ("other-secret", "")}
        r.close()


# -- credential resolution --------------------------------------------------------------------


def test_resolve_access_rules(monkeypatch):
    assert resolve_access("/local/path", ("u", "p")).kind == "anonymous"
    assert resolve_access("s3://vesuvius-challenge-open-data/x.zarr").kind == "anonymous"  # known public bucket
    assert resolve_access("s3://somebody-elses-bucket/x.zarr").kind == "anonymous"  # auto, no credentials
    assert resolve_access("https://host/x.zarr", ("u", "p")).kind == "basic"
    assert resolve_access("https://host/x.zarr", None).kind == "anonymous"
    with pytest.raises(ValueError, match="basic auth"):
        resolve_access("s3://b/x", ("u", "p"))
    with pytest.raises(ValueError, match="s3://"):
        resolve_access("https://host/x", S3Credentials("a", "b"))
    with pytest.raises(PermissionError, match="no AWS credentials"):
        resolve_access("s3://b/x", "aws")
    with pytest.raises(TypeError):
        resolve_access("s3://b/x", 42)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://127.0.0.1:1")  # no region lookup at amazonaws.com
    a = resolve_access("s3://private-bucket/x.zarr")  # auto: credentials found, so signed
    assert a.kind == "s3" and a.native()[1] == "AKIDENV"
    assert a.url("s3://private-bucket/x.zarr/.zarray") == "http://127.0.0.1:1/private-bucket/x.zarr/.zarray"


def test_s3_request_urls():
    a = Access("s3", keys=S3Credentials("a", "b"), region="eu-west-2")
    assert a.url("s3://bkt/a b/0") == "https://bkt.s3.eu-west-2.amazonaws.com/a%20b/0"
    assert a.url("s3://my.bkt/k") == "https://s3.eu-west-2.amazonaws.com/my.bkt/k"  # dots: path style
    keys = S3Credentials("a", "the-secret", "the-token")
    assert "the-secret" not in repr(keys) + repr(Access("s3", keys=keys, region="r"))
    assert "the-token" not in repr(keys)


def test_credentials_from_files_without_botocore(tmp_path, monkeypatch):
    (tmp_path / "creds").write_text("[default]\naws_access_key_id = AKIDDEF\naws_secret_access_key = s1\n"
                                    "[work]\naws_access_key_id = AKIDWORK\naws_secret_access_key = s2\n"
                                    "aws_session_token = t2\n")
    (tmp_path / "config").write_text("[profile cfg]\naws_access_key_id = AKIDCFG\naws_secret_access_key = s3\n"
                                     "region = ap-south-1\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "creds"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    files = lambda profile=None: auth_mod._AwsSource(profile)._from_files()  # noqa: E731
    assert files() == S3Credentials("AKIDDEF", "s1")
    assert files("work") == S3Credentials("AKIDWORK", "s2", "t2")
    assert files("cfg") == S3Credentials("AKIDCFG", "s3")
    assert files("missing") is None
    monkeypatch.setenv("AWS_PROFILE", "work")
    assert files() == S3Credentials("AKIDWORK", "s2", "t2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
    assert files() == S3Credentials("AKIDENV", "envsecret")  # the environment wins
    assert files("cfg") == S3Credentials("AKIDCFG", "s3")  # unless a profile is named


def test_credentials_from_environment(tmp_path, monkeypatch):
    assert S3Credentials.from_environment() is None
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "envtoken")
    assert S3Credentials.from_environment() == S3Credentials("AKIDENV", "envsecret", "envtoken")


# -- HTTP basic auth ------------------------------------------------------------------------------


class _BasicHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_message = lambda *a: None  # noqa: E731

    def do_GET(self):  # noqa: N802
        good = "Basic " + base64.b64encode(b"alice:s3cret").decode()
        if self.headers.get("Authorization") != good:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="scrolls"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


def test_basic_auth(tmp_path, data):
    import functools

    pytest.importorskip("numcodecs")
    _write_zarr(str(tmp_path / "www" / "a.zarr"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                            functools.partial(_BasicHandler, directory=str(tmp_path / "www")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/a.zarr"
    try:
        with ZarrReader.open(url, auth=("alice", "s3cret")) as r:
            np.testing.assert_array_equal(r.read((0, 0, 0), SHAPE), _expected(data, MISSING, CHUNKS))
        for auth in [None, ("alice", "wrong")]:
            with pytest.raises(PermissionError, match="401"):
                ZarrReader.open(url, auth=auth)
    finally:
        httpd.shutdown()


def test_load_basic_auth(tmp_path):
    (tmp_path / "a.json").write_text('{"username": "u", "password": "p"}')
    (tmp_path / "b.json").write_text('{"auth": {"user": "u2", "password": ""}}')
    (tmp_path / "c.json").write_text('{"username": "u"}')
    assert load_basic_auth(tmp_path / "a.json") == ("u", "p")
    assert load_basic_auth(tmp_path / "b.json") == ("u2", "")
    with pytest.raises(ValueError, match="password"):
        load_basic_auth(tmp_path / "c.json")


# -- offline ----------------------------------------------------------------------------------------


@pytest.fixture()
def www(tmp_path, data):
    pytest.importorskip("numcodecs")
    root = tmp_path / "www"
    (root / "vol.zarr").mkdir(parents=True)
    (root / "vol.zarr" / ".zgroup").write_text('{"zarr_format": 2}')
    _write_zarr(str(root / "vol.zarr" / "0"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    httpd, base = _serve(root)
    yield httpd, base, root / "vol.zarr"
    httpd.shutdown()


def test_offline_reads_only_the_cache(www, data, tmp_path):
    httpd, base, _ = www
    exp = _expected(data, MISSING, CHUNKS)
    url, cache = f"{base}/vol.zarr/0", tmp_path / "cache"
    with ZarrReader.open(url, cache_dir=cache) as r:  # fills chunks (0..1, 0..1, 0..1), marks (1, 1, 1) missing
        r.read((0, 0, 0), (32, 32, 32))
    httpd.requests.clear()
    with ZarrReader.open(url, cache_dir=cache, offline=True) as r:
        assert r.offline and r.write_layer is None and r.shape == SHAPE
        np.testing.assert_array_equal(r.read((0, 0, 0), (32, 32, 32)), exp[:32, :32, :32])
        np.testing.assert_array_equal(r.read_points([[20, 20, 20], [3, 4, 5]]), [7, exp[3, 4, 5]])
        with pytest.raises(CacheMissError, match="offline"):
            r.read((0, 0, 0), (33, 1, 1))  # chunk (2, 0, 0) was never fetched
        r2 = pickle.loads(pickle.dumps(r))
        np.testing.assert_array_equal(r2.read((5, 5, 5), (3, 3, 3)), exp[5:8, 5:8, 5:8])
        r2.close()
    assert httpd.requests == []


def test_offline_needs_a_cache(www, tmp_path):
    _, base, _ = www
    with pytest.raises(ValueError, match="cache_dir"):
        ZarrReader.open(f"{base}/vol.zarr/0", offline=True)
    with pytest.raises(CacheMissError):
        ZarrReader.open(f"{base}/vol.zarr/0", offline=True, cache_dir=tmp_path / "empty")


def test_offline_from_foreign_caches(www, data, tmp_path):
    httpd, base, src = www
    exp = _expected(data, MISSING, CHUNKS)
    _vc3d_mirror(tmp_path / "vc3d", f"{base}/vol.zarr", src, ["0/0/0"])
    _foreign_cache(tmp_path / "vesuvius", f"{base}/vol.zarr/0", data)  # chunk (0,0,0) holds 99s
    httpd.requests.clear()
    with ZarrReader.open(f"{base}/vol.zarr/0", cache_dir=tmp_path / "vc3d", offline=True) as r:
        np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    with ZarrReader.open(f"{base}/vol.zarr/0", cache_dir=tmp_path / "vesuvius", offline=True) as r:
        assert (r.read((0, 0, 0), (16, 16, 16)) == 99).all()
        with pytest.raises(CacheMissError):
            r.read((16, 0, 0), (1, 1, 1))
    assert httpd.requests == []


def test_offline_local_array_is_unaffected(tmp_path, data):
    pytest.importorskip("numcodecs")
    _write_zarr(str(tmp_path / "a.zarr"), data, CHUNKS, missing=MISSING)
    with ZarrReader.open(tmp_path / "a.zarr", offline=True) as r:
        np.testing.assert_array_equal(r.read((0, 0, 0), SHAPE), _expected(data, MISSING, CHUNKS))


def test_offline_signed_s3_cache(s3, data, tmp_path):
    httpd, base = s3
    with ZarrReader.open("s3://bkt/vol.zarr/0", auth=_creds(base), cache_dir=tmp_path) as r:
        r.read((0, 0, 0), (16, 16, 16))
    httpd.requests.clear()
    with ZarrReader.open("s3://bkt/vol.zarr/0", auth=_creds(base), cache_dir=tmp_path, offline=True) as r:
        np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), data[:16, :16, :16])
    assert httpd.requests == []
