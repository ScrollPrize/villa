"""A requests.Session stand-in serving a local directory as an anonymous
virtual-hosted S3 bucket: objects by URL, and ListObjectsV2 listings."""

from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

BUCKET = "https://bucket.s3.example.com"


class Response:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content

    def json(self):
        import json
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeS3Session:
    """Serves ``root`` at BUCKET; ``gets`` counts object requests."""

    def __init__(self, root, gets=None):
        self.root = Path(root)
        self.gets = gets if gets is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def get(self, url, params=None, timeout=None):
        parsed = urlparse(url)
        key = parsed.path.lstrip("/")
        if params and params.get("list-type") == "2":
            prefix = params["prefix"]
            keys = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")
                          if path.is_file()
                          and str(path.relative_to(self.root)).startswith(prefix))
            body = "".join(f"<Contents><Key>{escape(k)}</Key></Contents>" for k in keys)
            return Response(200, (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"{body}<IsTruncated>false</IsTruncated></ListBucketResult>").encode())
        self.gets.append(key)
        path = self.root / key
        return Response(200, path.read_bytes()) if path.is_file() else Response(404)
