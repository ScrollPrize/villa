"""CLI regressions for a remote chunk fetch that fails; no scroll downloads are required.

Run with: python test_render_fetch_failure.py /path/to/vc_render_tifxyz

Everything the renderer reads is built in a temporary directory: a small synthetic OME-Zarr volume
whose chunk bytes are generated on demand, and a flat tifxyz surface inside it. An HTTP server serves
the volume and decides, per request, whether to answer. Five cases:

  every fetch fails      the run must stop with a non-zero exit code rather than a core dump (#1809),
                         and the TIFF it leaves behind must not open as an image
  the zarr output path   the same, in the other output mode, where the leftover DOES open: a zarr with
                         every .zarray written and no chunks reads back as zeros, so the exit code is
                         the only signal a caller gets
  the first N fail       the retry path must still recover, exit 0, and produce pixels identical to a
                         run against a server that never fails
  a failure partway      the first two requests are served and the rest refused, so the failure lands
                         inside the sampling loop, which is the situation #1809 describes
  the error message      one line naming the chunk and the HTTP status, exit code 1, and no sign of
                         the abort path

Needs numpy and tifffile, which the vesuvius tooling already depends on; it exits 77 (the ctest skip
code) if either is missing. No network, no credentials, fifteen to twenty seconds.
"""

from pathlib import Path
import http.server
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

try:
    import numpy as np
    import tifffile
except ImportError:                                            # pragma: no cover
    print("numpy and tifffile are required for this test; skipping"); raise SystemExit(77)

RENDERER = str(Path(sys.argv.pop(1)).resolve())

Z, Y, X, C = 64, 128, 128, 64          # volume shape and chunk size, in voxels
NEVER_FAIL = 1 << 30                   # serve_first value for a server that answers every request


def chunk_bytes(cz, cy, cx):
    """Deterministic content for one chunk: a plane of bright material with darker gaps."""
    a = np.zeros((C, C, C), np.uint8)
    zz = np.arange(C)[:, None, None]
    a += (200 - 8 * np.abs(zz - (C // 2 - cz * 0)) % 200).astype(np.uint8)
    a += np.uint8((cz * 7 + cy * 13 + cx * 17) % 5)
    return a.tobytes()


class Volume(http.server.BaseHTTPRequestHandler):
    fail_first = 0                      # 0 means fail every chunk request
    separator = "."                     # zarr v2's default chunk-key style, "0/0.0.0"; the bucket's
                                        # volumes use "/", so the error path sees both in practice
    serve_first = 0                     # answer this many chunk requests before failing, so that the
                                        # failure lands inside the sampling loop rather than a preflight
    seen: dict = {}
    lock = threading.Lock()

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
            return self._send(json.dumps({"multiscales": [{"version": "0.4", "axes": [
                {"name": "z"}, {"name": "y"}, {"name": "x"}],
                "datasets": [{"path": "0"}]}]}).encode())
        if p == "0/.zarray":
            meta = {"zarr_format": 2, "shape": [Z, Y, X], "chunks": [C, C, C], "dtype": "|u1",
                    "compressor": None, "fill_value": 0, "order": "C", "filters": None}
            if self.separator == "/":
                meta["dimension_separator"] = "/"
            return self._send(json.dumps(meta).encode())
        if p.endswith((".zarray", ".zattrs", ".zgroup", ".zmetadata", ".json")) or p.endswith("/"):
            return self.send_error(404)
        with self.lock:
            self.seen["_n"] = self.seen.get("_n", 0) + 1
            total = self.seen["_n"]
            n = self.seen.get(p, 0) + 1
            self.seen[p] = n
        if self.serve_first and total <= self.serve_first:
            try:
                tail = p.split("/", 1)[1]
                cz, cy, cx = (int(v) for v in tail.replace("/", ".").split("."))
            except (ValueError, IndexError):
                return self.send_error(404)
            return self._send(chunk_bytes(cz, cy, cx))
        if self.fail_first and n > self.fail_first:
            try:
                tail = p.split("/", 1)[1]
                cz, cy, cx = (int(v) for v in tail.replace("/", ".").split("."))
            except (ValueError, IndexError):
                return self.send_error(404)
            return self._send(chunk_bytes(cz, cy, cx))
        try:                                                   # the failure under test: no reply at all
            self.connection.close()
        except (OSError, ValueError):                          # the peer may already be gone
            pass


def serve(fail_first, serve_first=0, separator="."):
    """fail_first=0 fails every chunk request; serve_first=NEVER_FAIL answers all of them.

    separator picks the chunk-key style the synthetic volume advertises: "." is zarr v2's default,
    "/" is what the published volumes use.
    """
    cls = type("V", (Volume,), {"fail_first": fail_first, "serve_first": serve_first,
                                "separator": separator, "seen": {}, "lock": threading.Lock()})
    # bind port 0 on the server itself: probing for a free port and then reopening it races another
    # process for that port, which on a busy CI runner is a flake nobody will be able to reproduce
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), cls)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def write_surface(root):
    """A flat 32 by 32 tifxyz sitting in the middle of the volume, one grid step per voxel."""
    d = root / "surface"
    d.mkdir()
    g = np.arange(32, dtype=np.float32)
    xs, ys = np.meshgrid(g + X // 2 - 16, g + Y // 2 - 16)
    zs = np.full_like(xs, Z // 2)
    for name, arr in (("x", xs), ("y", ys), ("z", zs)):
        tifffile.imwrite(d / f"{name}.tif", arr.astype(np.float32))
    (d / "meta.json").write_text(json.dumps(
        {"format": "tifxyz", "type": "seg", "uuid": "test", "scale": [1.0, 1.0],
         "bbox": [[float(xs.min()), float(ys.min()), float(zs.min())],
                  [float(xs.max()), float(ys.max()), float(zs.max())]]}))
    return d


class FetchFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vc render fetch ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.surface = write_surface(self.root)

    def render(self, url, out_flag, out_path, extra=()):
        args = [RENDERER, "-v", url, "--remote-url", url, "-s", str(self.surface),
                "--scale", "1", "-g", "0", "--num-slices", "3", "--slice-step", "1",
                "--cache-gb", "1", "--timeout", "1", out_flag, str(out_path), *extra]
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:                      # shorter than the 300 s ctest timeout,
            self.fail("the renderer hung for 120 s against a server that never answers a chunk; "
                      "a hang is a different failure from a crash and the test should say which")

    def test_tif_output_fails_cleanly(self):
        httpd, url = serve(0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        out = self.root / "tif_fail"
        r = self.render(url, "--tif-output", out)
        # subprocess reports a process killed by a signal as a NEGATIVE code, so "not zero" is not
        # enough and neither is "< 128": an abort comes back as -6. The fix under test is that this
        # is a positive exit code.
        self.assertGreater(r.returncode, 0,
                           f"exit {r.returncode}: the process died on a signal instead of exiting")
        for f in sorted(out.glob("*.tif")):
            with self.assertRaises(Exception, msg=f"{f.name} was left readable"):
                tifffile.imread(f)

    def test_failure_after_the_render_has_started(self):
        """#1809 is a fetch that fails partway, not one that fails before any data arrives.

        Serving the first two chunk requests and then refusing everything puts the failure inside the
        sampling loop, which is where the exception had nowhere to go. Without this case a fix that
        only handles an immediate failure would pass.
        """
        httpd, url = serve(0, serve_first=2)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        out = self.root / "midway"
        r = self.render(url, "--tif-output", out)
        self.assertGreater(r.returncode, 0,
                           f"exit {r.returncode}: the process died on a signal instead of exiting")
        self.assertIn("Error:", r.stdout + r.stderr,
                      "a failure partway through must still be reported as an error")

    def test_the_error_names_the_chunk_that_failed(self):
        """The PR's user-facing promise: one line naming the chunk and the transport error.

        Run for both chunk-key styles. A synthetic zarr v2 defaults to "0/0.0.0"; the published
        volumes are written with "/" separators, so the message path sees "0/34/29/21" in the wild.
        Both go through the same formatting, and a test that only ever saw one of them would not
        notice if that changed.
        """
        for sep in (".", "/"):
            with self.subTest(separator=sep):
                httpd, url = serve(0, separator=sep)
                self.addCleanup(httpd.server_close)
                self.addCleanup(httpd.shutdown)
                r = self.render(url, "--tif-output", self.root / f"named{sep == '/'}")
                both = r.stdout + r.stderr
                self.assertRegex(both, r"Error: HTTP \d+ fetching [\d]+[/.][\d./]+",
                                 "the error should name the chunk and the HTTP status")
                self.assertNotIn("terminating due to", both, "that is the abort path this PR removes")
                self.assertEqual(r.returncode, 1,
                                 "the PR documents exit code 1; another non-zero code is still a change")

    def test_zarr_output_leftover_is_silent(self):
        httpd, url = serve(0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        out = self.root / "zarr_fail.zarr"
        r = self.render(url, "--zarr-output", out)
        self.assertGreater(r.returncode, 0,
                           f"exit {r.returncode}: the process died on a signal instead of exiting")
        try:
            import zarr
        except ImportError:                                    # pragma: no cover
            return
        if (out / "0" / ".zarray").exists():                   # the leftover reads back as zeros
            a = np.asarray(zarr.open(str(out), mode="r")["0"][:])
            self.assertTrue((a == 0).all(),
                            "the leftover zarr is readable, so the exit code is the only signal")

    def test_transient_failures_still_recover(self):
        httpd, url = serve(2)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        out = self.root / "recovered"
        r = self.render(url, "--tif-output", out)
        self.assertEqual(r.returncode, 0, f"the retry path did not recover: {r.stderr[-400:]}")
        httpd2, url2 = serve(0, NEVER_FAIL)                    # fails nothing: the reference render
        self.addCleanup(httpd2.server_close)
        self.addCleanup(httpd2.shutdown)
        clean = self.root / "clean"
        self.assertEqual(self.render(url2, "--tif-output", clean).returncode, 0)
        got = sorted(out.glob("*.tif"))
        want = sorted(clean.glob("*.tif"))
        self.assertTrue(got and len(got) == len(want), f"{len(got)} against {len(want)} slices")
        for a, b in zip(got, want):
            self.assertEqual(a.read_bytes(), b.read_bytes(), f"{a.name} differs after recovery")


if __name__ == "__main__":
    unittest.main(verbosity=2)
