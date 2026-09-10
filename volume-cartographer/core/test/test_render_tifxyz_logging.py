"""CLI regressions for file-log cleanup; no scroll downloads are required.

Run with: python test_render_tifxyz_logging.py /path/to/vc_render_tifxyz
Real-scroll reproduction and pixel comparisons are separate validation.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RENDERER = str(Path(sys.argv.pop(1)).resolve())


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

    def test_missing_segmentation_preserves_error(self):
        self.assert_logged_error([], "--segmentation required")

    def test_caught_volume_error_preserves_diagnostic(self):
        self.assert_logged_error(
            ["-s", str(self.root / "absent.tifxyz")], "Error opening local zarr:"
        )

    def test_failed_log_open_exits_cleanly(self):
        self.log = self.root / "absent-directory" / "render.log"
        result = self.run_renderer()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("cannot open log file:", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
