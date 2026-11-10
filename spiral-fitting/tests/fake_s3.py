"""A loopback HTTP server standing in for an S3-compatible endpoint: the
files of a local directory as the objects of one bucket (path-style URLs),
with ListObjectsV2 listings. Request signatures are not checked, so any
fixed keys reach it (FakeS3Server.access)."""

import http.server
from pathlib import Path
import threading
from urllib.parse import parse_qsl, quote, unquote, urlparse
from xml.sax.saxutils import escape

BUCKET = "bucket"


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root: Path
    gets: list

    def log_message(self, *args):
        pass

    def _respond(self, status, body=b""):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        key = unquote(parsed.path).removeprefix(f"/{BUCKET}").lstrip("/")
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if query.get("list-type") == "2":
            prefix = query.get("prefix", "")
            keys = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")
                          if path.is_file() and str(path.relative_to(self.root)).startswith(prefix))
            contents = "".join(
                f"<Contents><Key>{escape(quote(key))}</Key>"
                f"<Size>{(self.root / key).stat().st_size}</Size></Contents>" for key in keys)
            return self._respond(200, (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<EncodingType>url</EncodingType>{contents}"
                "<IsTruncated>false</IsTruncated></ListBucketResult>").encode())
        self.gets.append(key)
        path = self.root / key
        if path.is_file():
            return self._respond(200, path.read_bytes())
        self._respond(404, b"<Error><Code>NoSuchKey</Code></Error>")


class FakeS3Server:
    """Serves ``root`` as bucket BUCKET on a loopback port; ``gets`` records
    the keys of object requests."""

    def __init__(self, root, gets=None):
        handler = type("Handler", (_Handler,),
                       {"root": Path(root), "gets": gets if gets is not None else []})
        self.gets = handler.gets
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def location(self, key=""):
        """The ``s3://`` location of ``key`` in the bucket."""
        return f"s3://{BUCKET}/{key}".rstrip("/")

    def access(self):
        """A vc_reader Access reaching this server for the bucket."""
        from vc_reader import _reader
        return _reader.s3_access(self.location(), "key", "secret", "", "us-east-1", self.endpoint)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
