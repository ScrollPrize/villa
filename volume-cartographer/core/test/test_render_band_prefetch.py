"""CLI regression: a TIFF render must only fetch the chunks its samples read, not their bounding box.

Run with: python test_render_band_prefetch.py /path/to/vc_render_tifxyz

The TIFF path renders a segment in bands of rows that each run the full width of the segment. The
band readers prefetch a bounding box around the band (readMultiSlice: all its samples;
readCompositeFast: its surface points). On a winding that box covers the area inside the winding
while the samples touch a thin ring of it: on a band crop of the published PHercParis4 mesh
20260701183128-w053-058, each full band queued 2,772-3,087 chunks where its samples read 369-397.
The renderer now queues exactly the chunks each band samples.

The volume is synthetic and served over HTTP by this script, which records every chunk requested.
The surface is a diagonal wall, x = y, so one band's bounding box covers every chunk column of the
volume while its samples stay within one chunk of the diagonal. A request for a chunk far from the
diagonal can only come from a bounding-box prefetch.

Standard library only, like test_render_fetch_failure.py. A few seconds.
"""

from pathlib import Path
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

from render_test_tiff import write_float_tiff

if len(sys.argv) < 2:
    raise SystemExit("usage: test_render_band_prefetch.py /path/to/vc_render_tifxyz")
RENDERER = str(Path(sys.argv.pop(1)).resolve())

# Volume shape and chunk size, in voxels: 8 x 8 chunk columns. The wall runs x = y = 0..255 at
# z = 8..23, so every sample stays in chunk layer 0.
Z, Y, X, C = 64, 256, 256, 32
WALL_N, WALL_Z0, WALL_H = 256, 8, 16


def chunk_bytes(cz, cy, cx):
    """Deterministic, non-constant content so the render has something to show."""
    plane = C * C
    out = bytearray(C * plane)
    for z in range(C):
        v = (40 + 5 * z + (cz * 7 + cy * 13 + cx * 17) % 11) & 0xFF
        out[z * plane : (z + 1) * plane] = bytes([v]) * plane
    return bytes(out)


class Volume(http.server.BaseHTTPRequestHandler):
    requested = None
    lock = None

    def log_message(self, *a):
        pass

    def _send(self, body, code=200):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        p = self.path.lstrip("/")
        if p in ("", ".zgroup"):
            return self._send(json.dumps({"zarr_format": 2}).encode())
        if p == ".zattrs":
            return self._send(
                json.dumps(
                    {
                        "multiscales": [
                            {
                                "version": "0.4",
                                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                                "datasets": [{"path": "0"}],
                            }
                        ]
                    }
                ).encode()
            )
        if p == "0/.zarray":
            return self._send(
                json.dumps(
                    {
                        "zarr_format": 2,
                        "shape": [Z, Y, X],
                        "chunks": [C, C, C],
                        "dtype": "|u1",
                        "compressor": None,
                        "fill_value": 0,
                        "order": "C",
                        "filters": None,
                    }
                ).encode()
            )
        try:
            level, key = p.split("/", 1)
            cz, cy, cx = (int(v) for v in key.split("."))
        except ValueError:
            return self.send_error(404)
        if level != "0":
            return self.send_error(404)
        with self.lock:
            self.requested.add((cz, cy, cx))
        return self._send(chunk_bytes(cz, cy, cx))


def serve():
    cls = type("V", (Volume,), {"requested": set(), "lock": threading.Lock()})
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), cls)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, cls.requested, "http://127.0.0.1:%d/" % httpd.server_address[1]


def write_wall(root):
    """A tifxyz wall along the diagonal x = y, one grid step per voxel, WALL_H rows tall."""
    d = root / "wall"
    d.mkdir()
    n = WALL_N * WALL_H
    xs = [float(i % WALL_N) for i in range(n)]
    zs = [float(WALL_Z0 + i // WALL_N) for i in range(n)]
    for name, vals in (("x", xs), ("y", xs), ("z", zs)):
        write_float_tiff(d / ("%s.tif" % name), WALL_N, WALL_H, vals)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "format": "tifxyz",
                "type": "seg",
                "uuid": "wall",
                "scale": [1.0, 1.0],
                "bbox": [[0, 0, min(zs)], [WALL_N - 1, WALL_N - 1, max(zs)]],
            }
        )
    )
    return d


class BandPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vc render band prefetch ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wall = write_wall(self.root)

    def render(self, extra=(), tifs=3):
        httpd, requested, url = serve()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        out = self.root / ("out%d" % len(list(self.root.glob("out*"))))
        args = [RENDERER, "-v", url, "--remote-url", url, "-s", str(self.wall)]
        args += "--scale 1 -g 0 --num-slices 3 --slice-step 1 --cache-gb 1 --timeout 1".split()
        args += ["--tif-output", str(out)] + list(extra)
        # --remote-url stages chunks under $HOME/.VC3D/remote_cache; keep that inside the temp dir.
        env = dict(os.environ, HOME=str(self.root), USERPROFILE=str(self.root))
        r = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 0, "render failed: %s" % (r.stdout + r.stderr)[-600:])
        self.assertEqual(len(sorted(out.glob("*.tif"))), tifs, "unexpected number of TIFFs")
        return set(requested)

    def check_only_near_the_wall(self, requested):
        # Guard: the render must have fetched the chunks along the wall, or the checks below pass
        # without anything being read.
        on_wall = {(0, c, c) for c in range(X // C)}
        self.assertTrue(
            on_wall <= requested,
            "chunks on the wall were not fetched: %s" % sorted(on_wall - requested),
        )
        # Samples sit within a voxel or two of x = y (trilinear neighbours, +-1 slice along the
        # normal), so they never leave the chunks next to the diagonal.
        far = sorted(k for k in requested if abs(k[1] - k[2]) > 1 or k[0] != 0)
        self.assertEqual(
            far,
            [],
            "%d of %d requested chunks are away from the wall, e.g. %s: the "
            "band prefetched its bounding box" % (len(far), len(requested), far[:4]),
        )

    def test_band_fetches_only_the_chunks_it_samples(self):
        self.check_only_near_the_wall(self.render())

    def test_composite_band_fetches_only_the_chunks_it_samples(self):
        # --composite-collapse reads through readCompositeFast, the other band reader.
        extra = ["--composite-collapse", "--composite-start", "-1", "--composite-end", "1"]
        self.check_only_near_the_wall(self.render(extra, tifs=1))

    def test_prefetch_remote_fetches_only_the_chunks_it_samples(self):
        # --prefetch-remote plans the whole render exactly up front (#1657); keep it that way.
        self.check_only_near_the_wall(self.render(["--prefetch-remote"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
