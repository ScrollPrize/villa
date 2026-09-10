import json
import subprocess
import sys
import unittest
from pathlib import Path

from click.testing import CliRunner


SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import render_ink


class RenderInkPathTests(unittest.TestCase):
    def test_default_lasagna_dir_is_sibling_of_spiral_fitting(self):
        script = Path("/checkout/spiral-fitting/render_ink.py")

        actual = Path(render_ink.default_lasagna_dir(script))

        self.assertEqual(actual, Path("/checkout/lasagna"))

    def test_failed_full_scroll_flatten_fails_when_no_strips_are_rendered(self):
        with CliRunner().isolated_filesystem():
            meshes_dir = Path("meshes")
            mesh = meshes_dir / "w001_spliced"
            mesh.mkdir(parents=True)
            (mesh / "meta.json").write_text(json.dumps({"format": "tifxyz"}))

            original_read = render_ink.read_step_and_voxel
            original_build = render_ink.build_full_concat
            original_flatten = render_ink.lasagna_flatten
            try:
                render_ink.read_step_and_voxel = lambda _path: (1, 1.0)
                render_ink.build_full_concat = lambda *_args: (
                    "w001-001", "meshes/concat/w001-001", 10)

                def fail_flatten(*_args):
                    raise subprocess.CalledProcessError(1, ["lasagna"])

                render_ink.lasagna_flatten = fail_flatten
                result = CliRunner().invoke(render_ink.main, [
                    str(meshes_dir), "--volume", "ink.zarr",
                ])
            finally:
                render_ink.read_step_and_voxel = original_read
                render_ink.build_full_concat = original_build
                render_ink.lasagna_flatten = original_flatten

        self.assertEqual(result.exit_code, 1)
        self.assertIn("render produced no ink strip images", result.output)


if __name__ == "__main__":
    unittest.main()
