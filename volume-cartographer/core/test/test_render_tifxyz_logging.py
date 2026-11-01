"""CLI regressions for file-log cleanup; no scroll downloads are required.

Run with: python test_render_tifxyz_logging.py /path/to/vc_render_tifxyz
Real-scroll reproduction and pixel comparisons are separate validation.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


RENDERER = str(Path(sys.argv.pop(1)).resolve())


def usable_memory_gb():
    """Physical RAM, or a lower cgroup v2 memory limit on this process's cgroup or one
    above it. Never below what vc_render_tifxyz finds, which also reads cgroup v1."""
    gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    try:
        with open("/proc/self/cgroup") as f:
            rel = next(l.split(":", 2)[2].strip() for l in f if l.startswith("0::"))
    except (OSError, StopIteration):
        return gb
    root = Path("/sys/fs/cgroup")
    for d in [root / rel.lstrip("/"), *(root / rel.lstrip("/")).parents]:
        try:
            gb = min(gb, int((d / "memory.max").read_text()) / 2**30)
        except (OSError, ValueError):  # absent, or "max"
            pass
        if d == root:
            break
    return gb


class RenderLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vc render logging ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "render.log"
        self.args = [
            RENDERER, "-v", str(self.root / "absent.zarr"),
            "--scale", "1", "-g", "0",
            "--tif-output", str(self.root / "output"),
        ]

    def run_renderer(self, extra=(), logged=True):
        args = self.args + list(extra)
        if logged:
            args += ["--log-path", str(self.log)]
        # These cases fail before rendering starts. Allow startup overhead, but
        # fail if shutdown waits for the flusher's five-second periodic timer.
        return subprocess.run(args, capture_output=True, text=True, timeout=3)

    def assert_logged_error(self, extra, message):
        # Logging must preserve both the diagnostic and the normal error code,
        # rather than aborting during destruction of a joinable flush thread.
        control = self.run_renderer(extra, logged=False)
        self.assertEqual(control.returncode, 1, control.stderr)
        self.assertIn(message, control.stderr)
        result = self.run_renderer(extra)
        self.assertEqual(result.returncode, control.returncode, result.stderr)
        self.assertIn(message, self.log.read_text())
        self.assertNotIn("terminate", result.stderr.lower())

    def test_option_validation_preserves_error_and_append_mode(self):
        self.log.write_text("earlier run\n")
        self.assert_logged_error(["--num-parts", "0"], "need 0 <= part-id < num-parts")
        self.assertTrue(self.log.read_text().startswith("earlier run\n"))

    def test_invalid_surface_interpolation_is_rejected(self):
        # Validated in the same place as --accum-type: after --segmentation is
        # checked, but before the volume is opened.
        seg = ["-s", str(self.root / "absent.tifxyz")]
        self.assert_logged_error(
            seg + ["--surface-interpolation", "bogus"],
            "invalid --surface-interpolation",
        )

    def test_valid_surface_interpolation_values_are_accepted(self):
        # These get past flag parsing and fail later, on the absent volume.
        seg = ["-s", str(self.root / "absent.tifxyz")]
        for value in ("linear", "smooth", "bicubic", "SMOOTH"):
            with self.subTest(value=value):
                result = self.run_renderer(
                    seg + ["--surface-interpolation", value], logged=False
                )
                self.assertNotIn("invalid --surface-interpolation", result.stderr)
                self.assertIn("Error opening local zarr:", result.stderr)

    def test_missing_segmentation_preserves_error(self):
        self.assert_logged_error([], "give --segmentation, or --store with one --surface")

    def test_caught_volume_error_preserves_diagnostic(self):
        self.assert_logged_error(
            ["-s", str(self.root / "absent.tifxyz")], "Error opening local zarr:"
        )

    def test_default_cache_is_at_most_half_the_memory(self):
        # Without --cache-gb the chunk cache is 16 GB, or half the memory the process can
        # use if that is less (#1404). Printed before the volume is opened.
        if not hasattr(os, "sysconf"):
            self.skipTest("no sysconf")
        mem_gb = usable_memory_gb()
        seg = ["-s", str(self.root / "absent.tifxyz")]
        result = self.run_renderer(seg, logged=False)
        self.assertIn("Error opening local zarr:", result.stderr)
        sized = re.search(r"Chunk cache: ([0-9.]+) GB, half the memory", result.stdout)
        if mem_gb < 32:
            self.assertIsNotNone(sized, result.stdout)
        if sized:
            self.assertLessEqual(float(sized.group(1)), min(16, mem_gb / 2) + 0.05)
        explicit = self.run_renderer(seg + ["--cache-gb", "1"], logged=False)
        self.assertNotIn("Chunk cache:", explicit.stdout)
        self.assertIn("Error opening local zarr:", explicit.stderr)

    def test_failed_log_open_exits_cleanly(self):
        self.log = self.root / "absent-directory" / "render.log"
        result = self.run_renderer()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("cannot open log file:", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
