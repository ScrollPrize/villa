import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
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


class ScaleSegmentationPassThroughTests(unittest.TestCase):
    """--scale-segmentation reaches vc_render_tifxyz unchanged, and nothing else moves.

    The strip path is driven with concat/save/render stubbed out, so what is
    checked is the renderer command render_ink builds, not the renderer.
    """

    def _render_strip(self, extra_args):
        commands = []
        with tempfile.TemporaryDirectory() as td:
            meshes_dir = Path(td) / "meshes"
            mesh = meshes_dir / "w010_spliced"
            mesh.mkdir(parents=True)
            (mesh / "meta.json").write_text(json.dumps({
                "format": "tifxyz", "scale": [1.0, 1.0], "area_cm2": 1.0, "area_vx2": 1.0,
            }))

            original_concat = render_ink.concat_meshes
            original_save = render_ink.save_tifxyz
            original_run = render_ink.subprocess.run
            try:
                render_ink.concat_meshes = lambda _paths: np.zeros((2, 2, 3), np.float32)

                def fake_save(_zyxs, out_dir, *, uuid, **_kwargs):
                    (Path(out_dir) / uuid).mkdir(parents=True, exist_ok=True)

                def fake_run(cmd, check=True, **_kwargs):
                    commands.append([str(part) for part in cmd])
                    out = Path(cmd[cmd.index("--tif-output") + 1])
                    out.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(out / "00.tif", np.ones((2, 2), np.uint8))

                render_ink.save_tifxyz = fake_save
                render_ink.subprocess.run = fake_run
                result = CliRunner().invoke(render_ink.main, [
                    str(meshes_dir), "--volume", "ink.zarr", "--vc-render-bin", "fake-renderer",
                    "--strips", "--no-full-scroll", "--no-flatten",
                    "--pre-erode", "0", "--no-keep-largest",
                    *extra_args,
                ])
            finally:
                render_ink.concat_meshes = original_concat
                render_ink.save_tifxyz = original_save
                render_ink.subprocess.run = original_run
        return result, commands

    def test_explicit_scale_segmentation_is_passed_to_the_renderer(self):
        result, commands = self._render_strip(["--scale-segmentation", "4"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[0], "fake-renderer")
        self.assertEqual(command[command.index("--scale-segmentation") + 1], "4.0")
        # --scale is the output density and is a different knob.
        self.assertEqual(command[command.index("--scale") + 1], "0.25")

    def test_default_keeps_the_renderer_default(self):
        result, commands = self._render_strip([])

        self.assertEqual(result.exit_code, 0, result.output)
        command = commands[0]
        self.assertEqual(command[command.index("--scale-segmentation") + 1], "1.0")

    def test_nonpositive_or_nonfinite_scale_fails_before_anything_is_rendered(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                result, commands = self._render_strip(["--scale-segmentation", value])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("must be finite and greater than zero", result.output)
                self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
