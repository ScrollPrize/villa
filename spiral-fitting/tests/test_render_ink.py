import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner
from PIL import Image


SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import render_ink


class RenderInkPathTests(unittest.TestCase):
    def _invoke_render(self, array, extra_args=(), *, stale_outputs=False, render_failure=False):
        meshes_dir = Path("meshes")
        mesh = meshes_dir / "w001_spliced"
        mesh.mkdir(parents=True, exist_ok=True)
        (mesh / "meta.json").write_text(json.dumps({"format": "tifxyz"}))
        commands = []

        def fake_build_full_concat(*_args):
            concat_path = Path("meshes/concat/w001-001")
            if stale_outputs:
                old_ink = concat_path / "ink"
                old_ink.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((4, 10), 200, dtype=np.uint8)).save(
                    old_ink / "stale.tif"
                )
            return "w001-001", str(concat_path), 10

        if stale_outputs:
            old_collect = meshes_dir / "ink"
            old_collect.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 10), 200, dtype=np.uint8)).save(
                old_collect / "w001-001_flat.jpg"
            )
            Image.fromarray(np.full((4, 10), 200, dtype=np.uint8)).save(
                old_collect / "w001-001_flat.000.jpg"
            )

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            if render_failure:
                raise subprocess.CalledProcessError(1, cmd)
            output_dir = Path(cmd[cmd.index("--tif-output") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array.astype(np.uint8)).save(output_dir / "slice.tif")

        with patch.object(render_ink, "read_step_and_voxel", return_value=(1, 1.0)), \
             patch.object(
                 render_ink,
                 "build_full_concat",
                 side_effect=fake_build_full_concat,
             ), \
             patch.object(
                 render_ink,
                 "lasagna_flatten",
                 side_effect=lambda *_args: (
                     Path("meshes/concat/w001-001").mkdir(parents=True, exist_ok=True)
                     or "meshes/concat/w001-001"
                 ),
             ), \
             patch.object(render_ink.subprocess, "run", side_effect=fake_run):
            result = CliRunner().invoke(
                render_ink.main,
                [
                    str(meshes_dir),
                    "--volume",
                    "ink.zarr",
                    "--no-full-scroll-trim",
                    *extra_args,
                ],
            )
        return result, commands, meshes_dir

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

    def test_all_zero_strip_fails_without_writing_jpg(self):
        with CliRunner().isolated_filesystem():
            result, commands, meshes_dir = self._invoke_render(
                np.zeros((4, 10), dtype=np.uint8),
                ("--scale-segmentation", "4"),
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("rendered entirely zero", result.output)
            self.assertIn("w001-001_flat", result.output)
            self.assertIn("--scale-segmentation", result.output)
            self.assertIn(["--scale-segmentation", "4.0"], [
                commands[0][i:i + 2] for i in range(len(commands[0]) - 1)
            ])
            self.assertFalse((meshes_dir / "ink" / "w001-001_flat.jpg").exists())

    def test_empty_rerender_cannot_pass_from_stale_tifs_or_leave_old_jpg(self):
        with CliRunner().isolated_filesystem():
            result, _commands, meshes_dir = self._invoke_render(
                np.zeros((4, 10), dtype=np.uint8),
                stale_outputs=True,
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("rendered entirely zero", result.output)
            self.assertFalse(
                (meshes_dir / "concat" / "w001-001" / "ink" / "stale.tif").exists()
            )
            self.assertFalse((meshes_dir / "ink" / "w001-001_flat.jpg").exists())
            self.assertFalse((meshes_dir / "ink" / "w001-001_flat.000.jpg").exists())

    def test_successful_render_replaces_stale_tiles_with_current_width(self):
        with CliRunner().isolated_filesystem():
            result, _commands, meshes_dir = self._invoke_render(
                np.full((4, 10), 200, dtype=np.uint8),
                ("--max-strip-width", "4"),
                stale_outputs=True,
            )
            self.assertEqual(result.exit_code, 0)
            output_dir = meshes_dir / "ink"
            self.assertFalse((output_dir / "w001-001_flat.jpg").exists())
            self.assertTrue((output_dir / "w001-001_flat.000.jpg").exists())
            self.assertTrue((output_dir / "w001-001_flat.001.jpg").exists())
            self.assertTrue((output_dir / "w001-001_flat.002.jpg").exists())
            self.assertFalse((output_dir / "w001-001_flat.003.jpg").exists())

    def test_failed_renderer_preserves_previous_tifs_and_jpgs(self):
        with CliRunner().isolated_filesystem():
            result, _commands, meshes_dir = self._invoke_render(
                np.zeros((4, 10), dtype=np.uint8),
                stale_outputs=True,
                render_failure=True,
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertTrue(
                (meshes_dir / "concat" / "w001-001" / "ink" / "stale.tif").exists()
            )
            self.assertTrue((meshes_dir / "ink" / "w001-001_flat.jpg").exists())
            self.assertTrue((meshes_dir / "ink" / "w001-001_flat.000.jpg").exists())

    def test_no_fail_on_empty_writes_black_jpg_with_warning(self):
        with CliRunner().isolated_filesystem():
            result, _commands, meshes_dir = self._invoke_render(
                np.zeros((4, 10), dtype=np.uint8),
                ("--no-fail-on-empty",),
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("WARNING rendered strip is entirely zero", result.output)
            self.assertTrue((meshes_dir / "ink" / "w001-001_flat.jpg").exists())

    def test_sparse_nonzero_strip_still_writes_jpg(self):
        array = np.zeros((4, 10), dtype=np.uint8)
        array[0, 0] = 200
        with CliRunner().isolated_filesystem():
            result, _commands, meshes_dir = self._invoke_render(array)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("p95=0.0", result.output)
            self.assertTrue((meshes_dir / "ink" / "w001-001_flat.jpg").exists())

    def test_scale_segmentation_passthrough_and_default_omission(self):
        with CliRunner().isolated_filesystem():
            result, commands, _meshes_dir = self._invoke_render(
                np.full((4, 10), 200, dtype=np.uint8),
                ("--scale-segmentation", "4"),
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn(["--scale-segmentation", "4.0"], [
                commands[0][i:i + 2] for i in range(len(commands[0]) - 1)
            ])
            result, commands, _meshes_dir = self._invoke_render(
                np.full((4, 10), 200, dtype=np.uint8),
            )
            self.assertEqual(result.exit_code, 0)
            self.assertNotIn("--scale-segmentation", commands[0])

    def test_max_composite_by_bands_matches_whole_slices(self):
        rng = np.random.default_rng(0)
        for dtype in (np.uint8, np.uint16):
            slices = [rng.integers(0, 60000, size=(9, 3)).astype(dtype) for _ in range(3)]
            with CliRunner().isolated_filesystem():
                paths = []
                for i, array in enumerate(slices):
                    Image.fromarray(array).save(f"{i:02d}.tif")
                    paths.append(f"{i:02d}.tif")
                with patch.object(render_ink, "BLOCK_PIXELS", 7):
                    composite = render_ink.max_composite(paths)
            self.assertEqual(composite.dtype, dtype)
            np.testing.assert_array_equal(composite, np.maximum.reduce(slices))

    def test_scale_to_uint8_matches_float_normalisation(self):
        rng = np.random.default_rng(0)
        strips = [
            rng.integers(0, 256, size=(37, 41)).astype(np.uint8),
            rng.integers(0, 65536, size=(37, 41)).astype(np.uint16),
            np.where(rng.random((37, 41)) < 0.03, 200, 0).astype(np.uint16),  # p95 == 0
            np.arange(20, dtype=np.uint8).reshape(4, 5),  # p95 between two values
        ]
        with patch.object(render_ink, "BLOCK_PIXELS", 7):
            for comp in strips:
                strip = comp.astype(np.float32)
                p95 = np.percentile(strip, 95)
                strip = np.clip(strip / p95, 0, 1) * 255 if p95 > 0 else strip
                strip8, actual_p95 = render_ink.scale_to_uint8(comp)
                np.testing.assert_array_equal(strip8, strip.astype(np.uint8))
                self.assertEqual(actual_p95.tobytes(), p95.tobytes())


if __name__ == "__main__":
    unittest.main()
