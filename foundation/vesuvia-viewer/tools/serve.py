import argparse
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

MIME = {
    ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
    ".wasm": "application/wasm", ".tif": "image/tiff", ".tiff": "image/tiff", ".css": "text/css",
    ".html": "text/html; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
    ".zarray": "application/json", ".zattrs": "application/json",
    ".zgroup": "application/json", ".jpg": "image/jpeg",
}

parser = argparse.ArgumentParser(description="Static file server for the viewer")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--dir", default=".")
args = parser.parse_args()
root = Path(args.dir).resolve()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *rest):
        pass

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

    def serve(self, head):
        path = Path(os.path.normpath(root / unquote(urlsplit(self.path).path).lstrip("/")))
        if path.is_dir():
            path = path / "index.html"
        if root not in path.parents or not path.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        total = path.stat().st_size
        first, last = 0, total - 1
        asked = self.headers.get("Range")
        if asked is not None:
            span = re.fullmatch(r"bytes=(\d+)-(\d+)", asked)
            if span is None or not int(span[1]) <= int(span[2]) < total:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            first, last = int(span[1]), int(span[2])
        self.send_response(HTTPStatus.OK if asked is None else HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(last - first + 1))
        if asked is not None:
            self.send_header("Content-Range", f"bytes {first}-{last}/{total}")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if head:
            return
        with path.open("rb") as f:
            f.seek(first)
            remaining = last - first + 1
            while remaining > 0:
                block = f.read(min(1 << 20, remaining))
                self.wfile.write(block)
                remaining -= len(block)


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
print(f"serving {root} on http://127.0.0.1:{server.server_address[1]}/", flush=True)
server.serve_forever()
