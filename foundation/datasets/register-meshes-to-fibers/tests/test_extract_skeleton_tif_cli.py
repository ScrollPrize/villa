import importlib.util
import subprocess
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "extract_skeleton_tif.py"


def _load_extract_skeleton_tif():
    module_name = "_test_extract_skeleton_tif"
    sys.modules.pop(module_name, None)

    open3d_stub = types.ModuleType("open3d")
    open3d_stub.geometry = types.SimpleNamespace(LineSet=object)
    open3d_stub.utility = types.SimpleNamespace(
        Vector3dVector=lambda value: value,
        Vector2iVector=lambda value: value,
    )
    open3d_stub.visualization = types.SimpleNamespace(
        draw_geometries=lambda *_args, **_kwargs: None,
    )

    skimage_stub = types.ModuleType("skimage")
    skimage_io_stub = types.ModuleType("skimage.io")
    skimage_io_stub.imread = lambda *_args, **_kwargs: None
    skimage_stub.io = skimage_io_stub

    kimimaro_stub = types.ModuleType("kimimaro")
    kimimaro_stub.skeletonize = lambda *_args, **_kwargs: {}

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "open3d": open3d_stub,
            "skimage": skimage_stub,
            "skimage.io": skimage_io_stub,
            "kimimaro": kimimaro_stub,
            module_name: module,
        },
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The NumPy module was reloaded.*",
                category=UserWarning,
            )
            spec.loader.exec_module(module)
    return module


class ExtractSkeletonTifCliTest(unittest.TestCase):
    def test_help_does_not_import_heavy_optional_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cube_label", result.stdout)
        self.assertIn("--fiber_type", result.stdout)

    def test_parser_exposes_label_mask_and_fiber_options(self):
        extract_cli = _load_extract_skeleton_tif()

        parser = extract_cli.build_arg_parser()
        args = parser.parse_args(
            [
                "--tif",
                "fiber.tif",
                "--cube_label",
                "mask.tif",
                "--label",
                "7",
                "--fiber_type",
                "vt",
                "--z_threshold",
                "0.75",
                "--axis_order",
                "xyz",
            ]
        )

        self.assertEqual(args.tif, "fiber.tif")
        self.assertEqual(args.cube_label, "mask.tif")
        self.assertEqual(args.label, 7)
        self.assertEqual(args.fiber_type, "vt")
        self.assertEqual(args.z_threshold, 0.75)
        self.assertEqual(args.axis_order, "xyz")

    def test_classify_curve_pca_defaults_to_zyx_compatible_axis(self):
        extract_cli = _load_extract_skeleton_tif()

        zyx_vertical = extract_cli.classify_curve_pca(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
        )
        xyz_vertical = extract_cli.classify_curve_pca(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32),
            axis_order="xyz",
        )
        xyz_horizontal = extract_cli.classify_curve_pca(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
            axis_order="xyz",
        )

        self.assertEqual(zyx_vertical, "vertical")
        self.assertEqual(xyz_vertical, "vertical")
        self.assertEqual(xyz_horizontal, "horizontal")

    def test_extract_skeleton_rejects_unknown_fiber_type_before_reading_images(self):
        extract_cli = _load_extract_skeleton_tif()
        skimage_stub = types.ModuleType("skimage")
        skimage_io_stub = types.ModuleType("skimage.io")
        skimage_io_stub.imread = mock.Mock()
        skimage_stub.io = skimage_io_stub
        kimimaro_stub = types.ModuleType("kimimaro")
        kimimaro_stub.skeletonize = mock.Mock(return_value={})

        with mock.patch.dict(
            sys.modules,
            {"skimage": skimage_stub, "skimage.io": skimage_io_stub, "kimimaro": kimimaro_stub},
        ):
            with self.assertRaisesRegex(ValueError, "fiber_type must be 'hz' or 'vt'"):
                extract_cli.extract_skeleton_from_tif("fiber.tif", "mask.tif", fiber_type="diag")

        skimage_io_stub.imread.assert_not_called()

    def test_main_passes_cli_options_to_extractor(self):
        extract_cli = _load_extract_skeleton_tif()
        captured = {}

        def fake_extract(tif_file, original_file, **kwargs):
            captured["tif_file"] = tif_file
            captured["original_file"] = original_file
            captured.update(kwargs)
            return {"vertical": ["curve"], "horizontal": []}

        with mock.patch.object(extract_cli.os.path, "isfile", return_value=True), \
             mock.patch.object(extract_cli, "extract_skeleton_from_tif", side_effect=fake_extract), \
             mock.patch.object(extract_cli, "visualize_curves") as visualize:
            result = extract_cli.main(
                [
                    "--tif",
                    "fiber.tif",
                    "--cube_label",
                    "mask.tif",
                    "--label",
                    "7",
                    "--fiber_type",
                    "vt",
                    "--z_threshold",
                    "0.75",
                    "--axis_order",
                    "xyz",
                    "--visualize",
                ]
            )

        self.assertEqual(result, {"vertical": ["curve"], "horizontal": []})
        self.assertEqual(captured["tif_file"], "fiber.tif")
        self.assertEqual(captured["original_file"], "mask.tif")
        self.assertEqual(captured["label"], 7)
        self.assertEqual(captured["fiber_type"], "vt")
        self.assertEqual(captured["z_threshold"], 0.75)
        self.assertEqual(captured["axis_order"], "xyz")
        visualize.assert_called_once_with(result)


if __name__ == "__main__":
    unittest.main()
