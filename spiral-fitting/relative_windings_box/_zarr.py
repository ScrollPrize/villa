"""Read chunks of a zarr v2 or v3 array (or OME-Zarr group), from a local folder or over anonymous http(s).

A 404 from S3 is a missing chunk only when its body names NoSuchKey, or carries no S3 error code at all. The
open-data bucket sometimes answers 404 NoSuchBucket for chunks that exist, and reading those as empty would blank
real data, so they are asked again. A chunk that cannot be fetched raises; it never reads as fill. Connections are
kept alive and reused across requests and threads. Sharded zarr v3 arrays, zarr filters and the transpose codec are
not read. Standard library and numpy; blosc, zstd and lz4 chunks need numcodecs.
"""

import base64
import collections
import gzip
import http.client
import io
import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np

USER_AGENT = "relative_windings_box"
DEFAULT_RATE = 20.0          # requests per second to a remote store (0: no pace)
TRIES = 10                   # attempts for a key that keeps answering a false 404
MISSING_CODE = "NoSuchKey"
MAX_IDLE = 32                # kept connections per origin
_CODE = re.compile(rb"<Code>\s*([A-Za-z0-9_.-]+)\s*</Code>")
_STALE = (http.client.RemoteDisconnected, http.client.CannotSendRequest, http.client.BadStatusLine,
          ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


class ZarrError(ValueError):
    pass


# ---------------------------------------------------------------- S3 answers

def s3_https(url):
    """s3://bucket/key as the bucket's anonymous https URL."""
    bucket, _, key = url[5:].partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}".rstrip("/")


def s3_listing(url):
    """(endpoint, key prefix) for listing under an S3 URL, virtual-host or path style; (None, None) otherwise."""
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    if not host.endswith("amazonaws.com"):
        return None, None
    path = u.path.strip("/")
    if re.match(r"^[^.]+(\.[^.]+)*\.s3([.-][a-z0-9-]+)?\.amazonaws\.com$", host) and not host.startswith("s3."):
        return f"{u.scheme}://{u.netloc}", path + "/" if path else ""
    bucket, _, rest = path.partition("/")
    return f"{u.scheme}://{u.netloc}/{bucket}", rest + "/" if rest else ""


def error_code(body):
    """The <Code> of an S3 XML error body, or None."""
    if not body:
        return None
    m = _CODE.search(body if isinstance(body, bytes) else str(body).encode("utf-8", "replace"))
    return m.group(1).decode() if m else None


def false_404(status, body):
    """True when a 404 is not a missing key: its S3 error body names another code than NoSuchKey."""
    code = error_code(body)
    return status == 404 and code is not None and code != MISSING_CODE


def open_url(req, opener=None, timeout=60, tries=TRIES, sleep=time.sleep):
    """opener(req, timeout=timeout) with a false 404 retried, up to `tries` attempts in all."""
    if opener is None:
        opener = urllib.request.urlopen
    tries = max(1, tries)
    for attempt in range(tries):
        try:
            return opener(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            body = exc.read() or b""
            exc.close()
            if not false_404(404, body):
                raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.hdrs, io.BytesIO(body)) from None
            if attempt == tries - 1:
                raise urllib.error.HTTPError(
                    exc.url, 404, f"S3 answered 404 {error_code(body)} {tries} times; not taken as a "
                                  "missing key", exc.hdrs, io.BytesIO(body)) from None
            sleep(min(2.0, 0.1 * 2 ** attempt))


# ---------------------------------------------------------------- kept-alive connections

class _Response:
    def __init__(self, status, body, headers, url):
        self.status, self.headers, self.url = status, headers, url
        self._body = body

    def read(self):
        return self._body

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class KeepAlive:
    """urlopen(request, timeout=...) with finished connections kept per origin and reused from any thread. A
    request on a kept connection the server has closed is sent once more on a new one (GET and HEAD only). An
    http proxy for https (HTTPS_PROXY) is used through CONNECT; redirects and other proxies go through urllib."""

    def __init__(self, max_idle=MAX_IDLE):
        self.max_idle = max(0, int(max_idle))
        self.context = None
        self._idle = {}
        self._lock = threading.Lock()

    def _ctx(self):
        if self.context is None:
            self.context = ssl.create_default_context()
        return self.context

    def _route(self, parts):
        scheme, host = parts.scheme.lower(), parts.hostname
        if scheme not in ("http", "https") or not host:
            return None
        port = parts.port or (443 if scheme == "https" else 80)
        proxy = urllib.request.getproxies().get(scheme)
        if proxy and urllib.request.proxy_bypass(host):
            proxy = None
        if proxy and (scheme != "https" or urllib.parse.urlsplit(proxy).scheme.lower() != "http"):
            return None
        return scheme, host, port, proxy or None

    def _new(self, route, timeout):
        scheme, host, port, proxy = route
        if proxy:
            p = urllib.parse.urlsplit(proxy)
            conn = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=timeout, context=self._ctx())
            headers = {}
            if p.username is not None:
                cred = f"{urllib.parse.unquote(p.username)}:{urllib.parse.unquote(p.password or '')}"
                headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
            conn.set_tunnel(host, port, headers=headers)
        elif scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=self._ctx())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        return conn

    def _take(self, route, timeout):
        with self._lock:
            idle = self._idle.get(route)
            conn = idle.pop() if idle else None
        if conn is None:
            return self._new(route, timeout), True
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn, False

    def _give(self, route, conn):
        with self._lock:
            idle = self._idle.setdefault(route, [])
            if len(idle) < self.max_idle:
                idle.append(conn)
                return
        conn.close()

    def _drop(self, route):
        with self._lock:
            conns = self._idle.pop(route, [])
        for c in conns:
            c.close()

    def __call__(self, req, timeout=None):
        if isinstance(req, str):
            req = urllib.request.Request(req)
        url = req.full_url
        parts = urllib.parse.urlsplit(url)
        route = self._route(parts)
        if route is None:
            return urllib.request.urlopen(req, timeout=timeout)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        method = req.get_method()
        headers = dict(req.header_items())
        resp = body = None
        for attempt in (0, 1):
            # the second try is always on a new connection: after a pause every kept one may be closed
            conn, fresh = (self._new(route, timeout), True) if attempt else self._take(route, timeout)
            try:
                conn.request(method, path, body=req.data, headers=headers)
                resp = conn.getresponse()
                body = resp.read()
            except _STALE:
                conn.close()
                if fresh:
                    raise
                self._drop(route)
                if method not in ("GET", "HEAD"):
                    raise
                continue
            except BaseException:
                conn.close()
                raise
            if resp.will_close:
                conn.close()
            else:
                self._give(route, conn)
            break
        if 300 <= resp.status < 400:
            return urllib.request.urlopen(req, timeout=timeout)
        if resp.status >= 400:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, io.BytesIO(body))
        return _Response(resp.status, body, resp.headers, url)


_POOL = None
_POOL_LOCK = threading.Lock()


def default_opener():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = KeepAlive()
        return _POOL


# ---------------------------------------------------------------- stores

class LocalStore:
    """A zarr store on disk; keys are paths relative to its root."""

    def __init__(self, root):
        self.root = Path(root)

    def __str__(self):
        return str(self.root)

    def read(self, key):
        try:
            return (self.root / key).read_bytes()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None


class HttpStore:
    """A zarr store behind http(s), read anonymously and paced to `rate` requests per second; network errors, 429
    and 5xx are retried, and so is a false 404 (see the module docstring), up to false_404_tries attempts."""

    def __init__(self, url, rate=DEFAULT_RATE, timeout=30.0, retries=2, opener=None, false_404_tries=TRIES):
        if url.lower().startswith("s3://"):
            url = s3_https(url)
        self.base = url.rstrip("/")
        self.rate, self.timeout, self.retries = rate, timeout, retries
        self.false_404_tries = max(1, false_404_tries)
        self.opener = opener or default_opener()
        self._lock = threading.Lock()
        self._next = 0.0

    def __str__(self):
        return self.base

    def _pace(self):
        with self._lock:
            if not self.rate:
                return
            now = time.monotonic()
            at = max(now, self._next)
            self._next = at + 1.0 / self.rate
        if at > now:
            time.sleep(at - now)

    def _request(self, url):
        """(status, body) of one GET; status None (and the error as body) when every attempt failed."""
        last, failed, false = None, 0, 0
        while True:
            if failed or false:
                time.sleep(min(8.0, 0.5 * 2 ** failed) if failed else min(2.0, 0.1 * 2 ** false))
            self._pace()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with self.opener(req, timeout=self.timeout) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                body = exc.read() if exc.code == 404 else b""
                exc.close()
                if exc.code == 404 and false_404(404, body):
                    false += 1
                    last = f"HTTP 404 {error_code(body)} on {false} attempts (S3 lists such keys as present)"
                    if false >= self.false_404_tries:
                        return None, last
                    continue
                if exc.code == 429 or exc.code >= 500:
                    last = f"HTTP {exc.code}"
                    failed += 1
                    if failed > self.retries:
                        return None, last
                    continue
                return exc.code, body if exc.code == 404 else None
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                last = str(getattr(exc, "reason", exc)) or type(exc).__name__
                failed += 1
                if failed > self.retries:
                    return None, last

    def _url(self, key):
        return self.base + "/" + urllib.parse.quote(key) if key else self.base

    def read(self, key):
        status, body = self._request(self._url(key))
        if status == 200:
            return body
        if status == 404:
            return None
        raise ZarrError(f"{self._url(key)}: " + (f"HTTP {status}" if status else f"request failed ({body})"))


def open_store(location, **kw):
    text = str(location)
    if re.match(r"^(https?|s3)://", text, re.I):
        return HttpStore(text, **kw)
    root = Path(text)
    if not root.is_dir():
        raise ZarrError(f"{root}: not a directory")
    return LocalStore(root)


# ---------------------------------------------------------------- headers and levels

def _json(store, key):
    raw = store.read(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ZarrError(f"{key}: not valid JSON ({exc})") from exc


def array_meta(store, path):
    """The parsed header of the array at `path` ("" for the store root), or None when there is none."""
    pre = path.strip("/") + "/" if path else ""
    meta = _json(store, pre + ".zarray")
    if meta is not None:
        return {"zarr_format": 2, "path": pre.rstrip("/"), "shape": list(meta.get("shape") or []),
                "chunks": list(meta.get("chunks") or []), "dtype": meta.get("dtype"),
                "fill_value": meta.get("fill_value"), "order": meta.get("order", "C"),
                "sep": meta.get("dimension_separator") or ".", "key_prefix": "",
                "compressor": meta.get("compressor"), "filters": meta.get("filters") or None, "inner": None}
    meta = _json(store, pre + "zarr.json")
    if meta is None or meta.get("node_type") != "array":
        return None
    conf = (meta.get("chunk_grid") or {}).get("configuration") or {}
    enc = meta.get("chunk_key_encoding") or {"name": "default"}
    default = enc.get("name", "default") == "default"
    sep = (enc.get("configuration") or {}).get("separator", "/" if default else ".")
    out = {"zarr_format": 3, "path": pre.rstrip("/"), "shape": list(meta.get("shape") or []),
           "chunks": list(conf.get("chunk_shape") or []), "dtype": meta.get("data_type"),
           "fill_value": meta.get("fill_value"), "order": "C", "sep": sep, "key_prefix": "c" + sep if default else "",
           "codecs": meta.get("codecs") or [], "inner": None}
    for codec in out["codecs"]:
        if codec.get("name") == "sharding_indexed":
            out["inner"] = list((codec.get("configuration") or {}).get("chunk_shape") or [])
    return out


def _group_attrs(store):
    """(zarr format, attributes) of the store root, (None, None) when it holds no zarr metadata."""
    attrs = _json(store, ".zattrs")
    if attrs is not None:
        return 2, attrs
    root = _json(store, "zarr.json")
    if root is not None:
        a = root.get("attributes") or {}
        return 3, a.get("ome", a) if isinstance(a.get("ome"), dict) else a
    if _json(store, ".zgroup") is not None:
        return 2, {}
    return None, None


def levels(store):
    """[{name, meta}] for an OME-Zarr group, finest level first, or a bare array as one level named ''. A level
    listed in the multiscales whose header is missing has meta None."""
    root = array_meta(store, "")
    if root is not None:
        return [{"name": "", "meta": root}]
    version, attrs = _group_attrs(store)
    if version is None:
        raise ZarrError(f"{store}: no zarr metadata at its root (.zarray, .zgroup, .zattrs or zarr.json)")
    out = []
    ms = (attrs or {}).get("multiscales") or []
    if ms:
        for d in ms[0].get("datasets", []):
            name = str(d.get("path"))
            out.append({"name": name, "meta": array_meta(store, name)})
    else:
        for k in range(16):  # no multiscales: numbered levels until one has no header
            meta = array_meta(store, str(k))
            if meta is None:
                break
            out.append({"name": str(k), "meta": meta})
    if not out:
        raise ZarrError(f"{store}: a zarr group with no levels (no multiscales and no array named 0)")
    return out


def chunk_key(meta, idx):
    pre = meta["path"] + "/" if meta["path"] else ""
    return pre + meta["key_prefix"] + meta["sep"].join(str(i) for i in idx)


# ---------------------------------------------------------------- decoding

def _numcodecs(spec):
    try:
        import numcodecs
    except ImportError as exc:
        raise ZarrError(f"compressor {spec.get('id')!r} needs numcodecs") from exc
    return numcodecs.get_codec(spec)


def _decompress(spec, raw):
    cid = (spec or {}).get("id") if isinstance(spec, dict) else None
    if not cid:
        return raw
    if cid == "zlib":
        return zlib.decompress(raw)
    if cid == "gzip":
        return gzip.decompress(raw)
    return bytes(_numcodecs(spec).decode(raw))


def _v3_bytes_codecs(codecs, raw):
    for codec in reversed(codecs):
        name, conf = codec.get("name"), codec.get("configuration") or {}
        if name in ("bytes", "transpose", "endian"):
            continue
        if name == "crc32c":
            raw = raw[:-4]
        elif name == "gzip":
            raw = gzip.decompress(raw)
        elif name in ("zstd", "blosc", "lz4"):
            spec = dict(conf, id=name)
            if name == "blosc":
                spec["shuffle"] = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}.get(conf.get("shuffle"), 1)
                spec.pop("typesize", None)
            raw = bytes(_numcodecs(spec).decode(raw))
        else:
            raise ZarrError(f"codec {name} not decoded here")
    return raw


def _dtype(meta):
    if meta["zarr_format"] == 2:
        return np.dtype(meta["dtype"])
    big = any(c.get("name") == "bytes" and (c.get("configuration") or {}).get("endian") == "big"
              for c in meta.get("codecs") or [])
    return np.dtype(meta["dtype"]).newbyteorder(">" if big else "<")


def _permutes(codecs):
    """Whether a zarr v3 codec list holds a transpose codec that changes the axis order."""
    for c in codecs or []:
        if c.get("name") == "transpose":
            order = (c.get("configuration") or {}).get("order")
            if order != "C" and not (isinstance(order, list) and order == list(range(len(order)))):
                return True
    return False


class Volume:
    """One OME-Zarr volume (or bare array), read chunk by chunk through a small cache of decoded chunks."""

    def __init__(self, location=None, store=None, cache_chunks=192, workers=8, **store_kw):
        if store is None and location is None:
            raise ZarrError("a volume needs a location or a store")
        if store is None and re.match(r"^(https?|s3)://", str(location), re.I):
            store_kw.setdefault("rate", 0)          # no pace for the prediction's reads
        self.store = store or open_store(str(location), **store_kw)
        self.levels = levels(self.store)
        if any(lv["meta"] is None for lv in self.levels):
            raise ZarrError(f"{self.store}: a pyramid level has no header")
        self.workers = max(1, workers)
        self.missing = 0
        self.chunks_read = 0
        self._cache = collections.OrderedDict()
        self._cache_n = cache_chunks
        self._lock = threading.Lock()

    def __str__(self):
        return str(self.store)

    def _chunk(self, level, idx):
        """The decoded chunk at chunk index idx (z, y, x) of a level, or None when it is missing."""
        key = (level, idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        meta = self.levels[level]["meta"]
        if meta["inner"] is not None:
            raise ZarrError(f"{self.store} level {level}: sharded arrays are not read here")
        if meta.get("filters"):
            raise ZarrError(f"{self.store} level {level}: zarr filters {meta['filters']} are not decoded here")
        if meta["zarr_format"] == 3 and _permutes(meta["codecs"]):
            raise ZarrError(f"{self.store} level {level}: the transpose codec is not read here")
        raw = self.store.read(chunk_key(meta, idx))
        if raw is None:
            arr = None
        else:
            try:
                data = (_decompress(meta["compressor"], raw) if meta["zarr_format"] == 2
                        else _v3_bytes_codecs(meta["codecs"], raw))
            except Exception as exc:  # noqa: BLE001 - any codec failure is a read failure here
                raise ZarrError(f"{self.store} level {level} chunk {idx}: does not decode ({exc})") from exc
            arr = np.frombuffer(data, dtype=_dtype(meta))
            if arr.size != int(np.prod(meta["chunks"])):
                raise ZarrError(f"{self.store} level {level} chunk {idx}: {arr.size} values, expected "
                                f"{int(np.prod(meta['chunks']))}")
            arr = arr.reshape(meta["chunks"], order=meta.get("order") or "C")
        with self._lock:
            self.chunks_read += 1
            self.missing += arr is None
            self._cache[key] = arr
            while len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
        return arr
