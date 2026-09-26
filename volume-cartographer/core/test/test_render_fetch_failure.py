"""CLI regressions for a remote chunk fetch that fails; no downloads, no third-party packages.

Run with: python test_render_fetch_failure.py /path/to/vc_render_tifxyz

Everything the renderer reads is built in a temporary directory: a small synthetic OME-Zarr volume
whose chunk bytes are generated on demand, and a flat tifxyz surface inside it. An HTTP server serves
the volume and decides, per request, whether to answer. Six cases:

  every fetch fails      the run must stop with exit code 1 rather than a core dump (#1809), and
                         the TIFF it leaves behind must not contain an image
  the zarr output path   the same, in the other output mode, where the leftover is metadata with no
                         chunks: a reader gets fill_value everywhere, so the exit code is the only
                         signal a caller gets
  transient failures     the retry path must still recover, exit 0, and produce bytes identical to a
                         run against a server that never fails
  a failure partway      one chunk column is served and the next refused, so the failure lands inside
                         the sampling loop, which is the situation #1809 describes
  the error message      one line naming the chunk and the HTTP status, exit code 1, and no sign of
                         the abort path, for both zarr chunk-key styles
  a working render       a guard: a run against a server that answers everything must come back with
                         real voxels in it, or every case above would pass without rendering anything

Standard library only: it writes the surface TIFFs and reads back the leftover TIFF itself, so there
is nothing to install, nothing to skip over, and it runs on any machine that can build the renderer.
About fifteen seconds.
"""

from pathlib import Path
import http.server
import json
import os
import struct
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

if len(sys.argv) < 2:
    raise SystemExit("usage: test_render_fetch_failure.py /path/to/vc_render_tifxyz")
RENDERER = str(Path(sys.argv.pop(1)).resolve())

Z, Y, X, C = 64, 128, 128, 64          # volume shape and chunk size, in voxels
SURF_N, SURF_0 = 32, 48                # the surface: 32 x 32 at x,y = 48..80, so it straddles the
                                       # chunk boundary at 64 and needs chunk columns 0 and 1
NEVER_FAIL = 1 << 30                   # serve_first value for a server that answers every request


def chunk_bytes(cz, cy, cx):
    """Deterministic content for one chunk: bright material with darker gaps, offset per chunk."""
    plane = C * C
    out = bytearray(C * plane)
    for z in range(C):
        v = (200 - 8 * abs(z - C // 2) + (cz * 7 + cy * 13 + cx * 17) % 5) & 0xFF
        out[z * plane:(z + 1) * plane] = bytes([v]) * plane
    return bytes(out)


class Volume(http.server.BaseHTTPRequestHandler):
    fail_first = 0                      # 0 means fail every chunk request
    separator = "."                     # zarr v2's default chunk-key style, "0/0.0.0"; the bucket's
                                        # volumes use "/", so the error path sees both in practice
    serve_first = 0                     # answer this many chunk requests before failing
    deny_cx = None                      # refuse chunk columns at or past this one, whatever the
                                        # order the requests arrive in: the same failure serially
                                        # (the clang OpenMP shim) and in parallel (gcc, real OpenMP)
    seen = {}
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

    def _coords(self, path):
        """(cz, cy, cx) from a chunk key in either style, or None if this is not a chunk key."""
        try:
            tail = path.split("/", 1)[1]
            cz, cy, cx = (int(v) for v in tail.replace("/", ".").split("."))
        except (ValueError, IndexError):
            return None
        return cz, cy, cx

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
        coords = self._coords(p)
        if coords is None:
            return self.send_error(404)
        with self.lock:
            self.seen["_n"] = self.seen.get("_n", 0) + 1
            total = self.seen["_n"]
            n = self.seen.get(p, 0) + 1
            self.seen[p] = n
        if self.deny_cx is not None:
            if coords[2] < self.deny_cx:
                return self._send(chunk_bytes(*coords))
            return self._send(b"<html><body>refused</body></html>", 503)
        if self.serve_first and total <= self.serve_first:
            return self._send(chunk_bytes(*coords))
        if self.fail_first and n > self.fail_first:
            return self._send(chunk_bytes(*coords))
        try:                                                   # the failure under test: no reply at all
            self.connection.close()
        except (OSError, ValueError):                          # the peer may already be gone
            pass


def serve(fail_first, serve_first=0, separator=".", deny_cx=None):
    """fail_first=0 fails every chunk request; serve_first=NEVER_FAIL answers all of them.

    separator picks the chunk-key style the synthetic volume advertises: "." is zarr v2's default,
    "/" is what the published volumes use. deny_cx refuses whole chunk columns instead of counting
    requests, so the same chunks fail whatever order the threads ask in.
    """
    cls = type("V", (Volume,), {"fail_first": fail_first, "serve_first": serve_first,
                                "separator": separator, "deny_cx": deny_cx,
                                "seen": {}, "lock": threading.Lock()})
    # bind port 0 on the server itself: probing for a free port and then reopening it races another
    # process for that port, which on a busy CI runner is a flake nobody will be able to reproduce
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), cls)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d/" % httpd.server_address[1]


# --- the two pieces that used to need numpy and tifffile ---------------------------------------

TIFF_TAGS = {256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample", 259: "Compression",
             262: "Photometric", 273: "StripOffsets", 277: "SamplesPerPixel", 278: "RowsPerStrip",
             279: "StripByteCounts", 324: "TileOffsets", 325: "TileByteCounts", 339: "SampleFormat"}


def write_float_tiff(path, width, height, values):
    """A baseline single-strip float32 greyscale TIFF, which is what a tifxyz coordinate plane is."""
    data = struct.pack("<%df" % len(values), *values)
    entries = [(256, 3, 1, width), (257, 3, 1, height), (258, 3, 1, 32), (259, 3, 1, 1),
               (262, 3, 1, 1), (273, 4, 1, 0), (277, 3, 1, 1), (278, 3, 1, height),
               (279, 4, 1, len(data)), (339, 3, 1, 3)]          # 339 SampleFormat 3 = IEEE float
    ifd_offset = 8
    pixel_offset = ifd_offset + 2 + 12 * len(entries) + 4
    entries = [(t, ty, n, pixel_offset if t == 273 else v) for (t, ty, n, v) in entries]
    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offset))
    out += struct.pack("<H", len(entries))
    for tag, typ, count, value in sorted(entries):
        # a value of 2 bytes or 4 bytes lives in the entry itself, left-justified
        raw = struct.pack("<I", value) if typ == 4 else struct.pack("<HH", value, 0)
        out += struct.pack("<HHI", tag, typ, count) + raw
    out += struct.pack("<I", 0)                                 # no second IFD
    out += data
    Path(path).write_bytes(bytes(out))


TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
TYPE_FMT = {1: "B", 3: "H", 4: "I"}


def read_tiff_tags(path):
    """Every IFD in a TIFF, as {tag name: [values]}. [] means the file declares no image at all.

    This replaces "tifffile.imread raises": the point is not that some library dislikes the file, it
    is that the file the renderer leaves behind has no pixel data, and this says which tag is missing.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 8 or raw[:2] not in (b"II", b"MM"):
        return []
    end = "<" if raw[:2] == b"II" else ">"
    magic, offset = struct.unpack(end + "HI", raw[2:8])
    if magic != 42:
        return []
    pages = []
    while offset and offset + 2 <= len(raw):
        (count,) = struct.unpack(end + "H", raw[offset:offset + 2])
        page = {}
        for i in range(count):
            at = offset + 2 + 12 * i
            if at + 12 > len(raw):
                return pages
            tag, typ, n = struct.unpack(end + "HHI", raw[at:at + 8])
            size = TYPE_SIZE.get(typ, 4) * n
            if size <= 4:                                      # small values live in the entry
                data = raw[at + 8:at + 8 + size]
            else:                                              # larger ones are an offset to an array
                (off,) = struct.unpack(end + "I", raw[at + 8:at + 12])
                data = raw[off:off + size]
            fmt = TYPE_FMT.get(typ)
            if fmt is None or len(data) < size:
                continue
            page[TIFF_TAGS.get(tag, tag)] = list(struct.unpack("%s%d%s" % (end, n, fmt), data))
        pages.append(page)
        nxt = offset + 2 + 12 * count
        if nxt + 4 > len(raw):
            return pages
        (offset,) = struct.unpack(end + "I", raw[nxt:nxt + 4])
    return pages


def pixel_bytes(path):
    """All the pixel data of a TIFF, strips or tiles, or b"" if the file declares none.

    vc_render_tifxyz writes tiled TIFFs, so a reader that only understood strips would report an
    ordinary output file as having no pixels.
    """
    pages = read_tiff_tags(path)
    if not pages:
        return b""
    p0 = pages[0]
    for offs, counts in (("TileOffsets", "TileByteCounts"), ("StripOffsets", "StripByteCounts")):
        if offs in p0 and counts in p0:
            raw = Path(path).read_bytes()
            return b"".join(raw[o:o + n] for o, n in zip(p0[offs], p0[counts]))
    return b""


def write_surface(root):
    """A flat 32 by 32 tifxyz straddling the chunk boundary, one grid step per voxel."""
    d = root / "surface"
    d.mkdir()
    xs = [float(SURF_0 + i % SURF_N) for i in range(SURF_N * SURF_N)]
    ys = [float(SURF_0 + i // SURF_N) for i in range(SURF_N * SURF_N)]
    zs = [float(Z // 2)] * (SURF_N * SURF_N)
    for name, vals in (("x", xs), ("y", ys), ("z", zs)):
        write_float_tiff(d / ("%s.tif" % name), SURF_N, SURF_N, vals)
    (d / "meta.json").write_text(json.dumps(
        {"format": "tifxyz", "type": "seg", "uuid": "test", "scale": [1.0, 1.0],
         "bbox": [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]}))
    return d


class FetchFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vc render fetch ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.surface = write_surface(self.root)

    def serving(self, *a, **kw):
        httpd, url = serve(*a, **kw)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return url

    def render(self, url, out_flag, out_path, extra=()):
        args = [RENDERER, "-v", url, "--remote-url", url, "-s", str(self.surface),
                "--scale", "1", "-g", "0", "--num-slices", "3", "--slice-step", "1",
                "--cache-gb", "1", "--timeout", "1", out_flag, str(out_path)] + list(extra)
        # --remote-url makes the renderer stage chunks under homeDirectory()/.VC3D/remote_cache
        # (core/src/RemoteCacheSettings.cpp), and it reads ~/.VC3D/VC3D.ini, where a user may have
        # pointed that cache somewhere else. Without this the test writes into the real home
        # directory of whoever runs it and depends on their settings file, so "hermetic" would not
        # be true. Pointing HOME at the temp directory makes it true.
        env = dict(os.environ, HOME=str(self.root), USERPROFILE=str(self.root))
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=120, env=env)
        except subprocess.TimeoutExpired:                      # shorter than the 300 s ctest timeout,
            self.fail("the renderer hung for 120 s against a server that never answers a chunk; "
                      "a hang is a different failure from a crash and the test should say which")

    def test_tif_output_fails_cleanly(self):
        url = self.serving(0)
        out = self.root / "tif_fail"
        r = self.render(url, "--tif-output", out)
        # subprocess reports a process killed by a signal as a NEGATIVE code, so "not zero" is not
        # enough and neither is "< 128": an abort comes back as -6. Nor is "> 0" enough on its own:
        # --timeout starts a watchdog thread that _exit(2)s after a minute (vc_render_tifxyz.cpp),
        # so a renderer that hung would also pass that. The fix under test is exit code 1.
        self.assertEqual(r.returncode, 1,
                         "exit %d: expected a clean exit 1 (-6 is the abort this PR removes, "
                         "2 is the --timeout watchdog, which means it hung instead)" % r.returncode)
        for f in sorted(out.glob("*.tif")):
            for page in read_tiff_tags(f):
                self.assertNotIn("StripOffsets", page, "%s was left with pixel data" % f.name)
                self.assertNotIn("TileOffsets", page, "%s was left with pixel data" % f.name)

    def test_failure_after_the_render_has_started(self):
        """#1809 is a fetch that fails partway, not one that fails before any data arrives.

        The surface straddles the chunk boundary at x = 64, so chunk column 0 is served and column 1
        refused: the tile needs both, part of it resolves, and the failure is on a chunk the render
        has already committed to. Refusing by chunk column rather than by a request count keeps the
        same chunks failing whether the tile loop runs serially or across threads, which a count does
        not. Measured on main by request count for comparison: refusing everything, or serving one or
        two first, all abort; serving four or more finishes.
        """
        url = self.serving(0, deny_cx=1)
        r = self.render(url, "--tif-output", self.root / "midway")
        self.assertEqual(r.returncode, 1,
                         "exit %d: expected a clean exit 1 (-6 is the abort, 2 is the watchdog)"
                         % r.returncode)
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
                url = self.serving(0, separator=sep)
                r = self.render(url, "--tif-output", self.root / ("named%s" % (sep == "/")))
                both = r.stdout + r.stderr
                self.assertRegex(both, r"Error: HTTP \d+ fetching [\d]+[/.][\d./]+",
                                 "the error should name the chunk and the HTTP status")
                self.assertNotIn("terminating due to", both, "that is the abort path this PR removes")
                self.assertEqual(r.returncode, 1,
                                 "the PR documents exit code 1; another non-zero code is still a change")

    def test_zarr_output_leftover_is_silent(self):
        """The leftover store is metadata with no chunks, so a reader sees fill_value, not an error.

        That is deliberate (--resume may need the store), which is exactly why it is worth pinning:
        the exit code is the only signal, and a change that started writing partial chunks here would
        make --resume skip tiles that were never rendered.
        """
        url = self.serving(0)
        out = self.root / "zarr_fail.zarr"
        r = self.render(url, "--zarr-output", out)
        self.assertGreater(r.returncode, 0,
                           "exit %d: the process died on a signal instead of exiting" % r.returncode)
        if not (out / "0" / ".zarray").exists():
            return
        meta = json.loads((out / "0" / ".zarray").read_text())
        self.assertEqual(meta.get("fill_value", 0), 0, "a reader of the leftover gets fill_value")
        chunks = [f for f in (out / "0").rglob("*") if f.is_file() and not f.name.startswith(".")]
        self.assertEqual(chunks, [], "the failed run left chunk data: %s" % chunks[:3])

    def test_the_harness_actually_renders_data(self):
        """Guard against every other case passing vacuously.

        If the surface ever stops landing inside the synthetic volume, the renderer writes empty
        tiles without fetching a single chunk and exits 0, and then the failure cases here would be
        testing nothing at all: no fetch, no failure, no crash to catch. This pins that a successful
        render against a server that answers everything really does come back with voxels in it.
        """
        url = self.serving(0, NEVER_FAIL)
        out = self.root / "sanity"
        r = self.render(url, "--tif-output", out)
        self.assertEqual(r.returncode, 0, "the reference render failed: %s" % r.stderr[-400:])
        tifs = sorted(out.glob("*.tif"))
        self.assertTrue(tifs, "the reference render produced no TIFFs")
        px = pixel_bytes(tifs[0])
        self.assertTrue(px, "%s has no pixel data" % tifs[0].name)
        self.assertGreater(len(set(px)), 1,
                           "every pixel of the reference render is the same value, so the surface "
                           "is probably outside the volume and nothing was sampled")

    def test_zarr_output_records_render_provenance(self):
        url = self.serving(0, NEVER_FAIL)
        out = self.root / "provenance.zarr"
        r = self.render(url, "--zarr-output", out, extra=("--pyramid", "false"))
        self.assertEqual(r.returncode, 0, "the provenance render failed: %s" % r.stderr[-400:])
        attrs = json.loads((out / ".zattrs").read_text())
        p = attrs["render_provenance"]
        self.assertEqual(attrs["source_zarr"], url)
        self.assertEqual(p["schema_version"], 1)
        self.assertEqual(p["tool"], "vc_render_tifxyz")
        # A build without git history records "Untracked" instead of a commit.
        self.assertRegex(p["tool_git_commit"], r"^([0-9a-f]{40}|Untracked)$")
        self.assertEqual(p["source_volume"]["input_url"], url)
        self.assertFalse(p["source_volume"]["input_url_redacted"])
        self.assertNotIn("local_path", p["source_volume"])
        self.assertEqual(p["source_volume"]["remote_url"], url)
        self.assertFalse(p["source_volume"]["remote_url_redacted"])
        self.assertEqual(p["source_volume"]["group_index"], 0)
        self.assertEqual(p["surface"]["path"], str(self.surface))
        self.assertRegex(p["surface"]["effective_geometry_sha1"], r"^[0-9a-f]{40}$")
        self.assertEqual(p["surface"]["grid_width"], SURF_N)
        self.assertEqual(p["surface"]["grid_height"], SURF_N)
        self.assertEqual(p["render"]["pixels_per_level_voxel"], 1.0)
        self.assertEqual(p["render"]["num_slices"], 3)
        self.assertFalse(p["render"]["composite"])
        self.assertFalse(p["render"]["flip_normals"])
        self.assertFalse(p["render"]["flatten"])
        self.assertEqual(p["render"]["output_dtype"], "uint8")
        self.assertEqual(p["affine"]["composed_matrix"], [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def test_transient_failures_still_recover(self):
        url = self.serving(2)
        out = self.root / "recovered"
        r = self.render(url, "--tif-output", out)
        self.assertEqual(r.returncode, 0, "the retry path did not recover: %s" % r.stderr[-400:])
        url2 = self.serving(0, NEVER_FAIL)                     # fails nothing: the reference render
        clean = self.root / "clean"
        self.assertEqual(self.render(url2, "--tif-output", clean).returncode, 0)
        got = sorted(out.glob("*.tif"))
        want = sorted(clean.glob("*.tif"))
        self.assertTrue(got and len(got) == len(want), "%d against %d slices" % (len(got), len(want)))
        for a, b in zip(got, want):
            self.assertEqual(a.read_bytes(), b.read_bytes(), "%s differs after recovery" % a.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
