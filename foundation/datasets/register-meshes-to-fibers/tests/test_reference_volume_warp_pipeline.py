import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "reference_volume_warp_pipeline.py"


def _import_real_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise unittest.SkipTest("zarr is not installed") from exc
    return zarr


class FakeZarrArray:
    def __init__(self, array, fail_on_array=False):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = array.ndim
        self.fail_on_array = fail_on_array
        self.materialize_count = 0

    def __array__(self, dtype=None):
        self.materialize_count += 1
        if self.fail_on_array:
            raise AssertionError("zarr input should not be materialized as a full numpy array")
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, item, value):
        self.array[item] = value


class FakeZarrModule:
    def __init__(self):
        self.stores = {}

    def open(self, path, mode="r", shape=None, dtype=None, chunks=None):
        if mode == "r":
            return self.stores[str(path)]
        if mode in {"w", "w+"}:
            store = FakeZarrArray(np.zeros(shape, dtype=dtype))
            self.stores[str(path)] = store
            return store
        raise AssertionError(f"unexpected zarr mode: {mode}")


class FakeZarrGroup:
    def __init__(self, arrays):
        self.arrays = arrays

    def keys(self):
        return self.arrays.keys()

    def __getitem__(self, key):
        return self.arrays[str(key)]


def _load_pipeline():
    module_name = "_test_reference_volume_warp_pipeline"
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(MODULE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(MODULE_DIR))


class ReferenceVolumeWarpPipelineTest(unittest.TestCase):
    def test_infer_grid_shape_from_volume_uses_zyx_volume_and_xyz_spacing(self):
        pipeline = _load_pipeline()

        shape_xyz = pipeline.infer_grid_shape_from_volume(
            volume_shape_zyx=(3, 4, 5),
            field_spacing_xyz=(2.0, 3.0, 4.0),
            volume_spacing_xyz=(1.0, 2.0, 3.0),
        )

        self.assertEqual(shape_xyz, (3, 3, 3))

    def test_parser_rejects_non_positive_grid_spacing_and_metrics_step(self):
        pipeline = _load_pipeline()
        parser = pipeline.build_arg_parser()
        base_args = [
            "--mesh-pair",
            "source.obj",
            "registered.obj",
            "--volume",
            "volume.npy",
            "--output-dir",
            "outputs",
            "--grid-shape",
            "1,1,1",
            "--field-spacing",
            "1,1,1",
        ]

        with self.assertRaises(SystemExit):
            parser.parse_args(base_args[: base_args.index("--field-spacing") + 1] + ["0,1,1"])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args[: base_args.index("--grid-shape") + 1] + ["1,0,1"] + base_args[base_args.index("--field-spacing") :])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--metrics-sample-step", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--chunk-depth", "-1"])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--field-query-chunk-size", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--field-control-chunk-size", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--max-points-per-mesh", "-1"])

    def test_run_pipeline_rejects_negative_chunk_depth_before_io(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(2, dtype=np.float32).reshape(1, 1, 2))

            with self.assertRaisesRegex(ValueError, "chunk_depth must be non-negative"):
                pipeline.run_pipeline(
                    mesh_pairs=[(source_path, registered_path)],
                    volume_path=volume_path,
                    output_dir=output_dir,
                    grid_shape_xyz=(2, 1, 1),
                    field_spacing_xyz=(1.0, 1.0, 1.0),
                    chunk_depth=-1,
                )

            self.assertFalse(output_dir.exists())

    def test_run_pipeline_rejects_non_positive_k_before_io(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(2, dtype=np.float32).reshape(1, 1, 2))

            with self.assertRaisesRegex(ValueError, "k must be positive"):
                pipeline.run_pipeline(
                    mesh_pairs=[(source_path, registered_path)],
                    volume_path=volume_path,
                    output_dir=output_dir,
                    grid_shape_xyz=(2, 1, 1),
                    field_spacing_xyz=(1.0, 1.0, 1.0),
                    k=0,
                )

            self.assertFalse(output_dir.exists())

    def test_run_pipeline_rejects_non_positive_field_query_chunk_size_before_io(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "out"
            source_path.write_text("v 0 0 0\n", encoding="utf-8")
            registered_path.write_text("v 1 0 0\n", encoding="utf-8")
            np.save(volume_path, np.zeros((1, 1, 1), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "field_query_chunk_size must be positive"):
                pipeline.run_pipeline(
                    mesh_pairs=[(source_path, registered_path)],
                    volume_path=volume_path,
                    output_dir=output_dir,
                    field_query_chunk_size=0,
                )

            self.assertFalse(output_dir.exists())

    def test_run_pipeline_rejects_non_positive_field_control_chunk_size_before_io(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\n", encoding="utf-8")
            np.save(volume_path, np.zeros((1, 1, 1), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "field_control_chunk_size must be positive"):
                pipeline.run_pipeline(
                    mesh_pairs=[(source_path, registered_path)],
                    volume_path=volume_path,
                    output_dir=output_dir,
                    field_control_chunk_size=0,
                )

            self.assertFalse(output_dir.exists())

    def test_run_pipeline_rejects_negative_max_points_before_io(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(2, dtype=np.float32).reshape(1, 1, 2))

            with self.assertRaisesRegex(ValueError, "max_points_per_mesh must be non-negative"):
                pipeline.run_pipeline(
                    mesh_pairs=[(source_path, registered_path)],
                    volume_path=volume_path,
                    output_dir=output_dir,
                    grid_shape_xyz=(2, 1, 1),
                    field_spacing_xyz=(1.0, 1.0, 1.0),
                    max_points_per_mesh=-1,
                )

            self.assertFalse(output_dir.exists())

    def test_cli_writes_controls_field_and_warped_volume(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text(
                "\n".join(["v 0 0 0", "v 4 0 0"]),
                encoding="utf-8",
            )
            registered_path.write_text(
                "\n".join(["v 1 0 0", "v 5 0 0"]),
                encoding="utf-8",
            )
            np.save(volume_path, np.arange(5, dtype=np.float32).reshape(1, 1, 5))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "5,1,1",
                    "--field-spacing",
                    "1,1,1",
                    "--field-origin",
                    "0,0,0",
                    "--volume-spacing",
                    "1,1,1",
                    "--volume-origin",
                    "0,0,0",
                    "--max-points-per-mesh",
                    "0",
                    "--field-query-chunk-size",
                    "1",
                    "--field-control-chunk-size",
                    "1",
                    "--fill-value",
                    "-1",
                ]
            )

            controls_path = output_dir / "deformation-controls.json"
            field_path = output_dir / "deformation-field.npz"
            warped_path = output_dir / "warped-volume.npy"
            manifest_path = output_dir / "deformation-run-manifest.json"
            metrics_path = output_dir / "deformation-run-metrics.json"
            controls = json.loads(controls_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(field_path) as field_data:
                field = field_data["displacement_xyz"]
            warped = np.load(warped_path)

        self.assertEqual(len(controls["control_points"]), 2)
        self.assertEqual(field.shape, (1, 1, 5, 3))
        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertEqual(manifest["input_volume"], str(volume_path))
        self.assertEqual(manifest["output_paths"]["controls"], str(controls_path))
        self.assertEqual(manifest["output_paths"]["field"], str(field_path))
        self.assertEqual(manifest["output_paths"]["warped"], str(warped_path))
        self.assertEqual(manifest["output_paths"]["metrics"], str(metrics_path))
        self.assertEqual(manifest["volume_shape_zyx"], [1, 1, 5])
        self.assertEqual(manifest["actual_grid_shape_xyz"], [5, 1, 1])
        self.assertEqual(manifest["effective_volume_spacing_xyz"], [1.0, 1.0, 1.0])
        self.assertEqual(manifest["effective_volume_origin_xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(manifest["chunk_depth"], 0)
        self.assertEqual(manifest["field_query_chunk_size"], 1)
        self.assertEqual(manifest["field_control_chunk_size"], 1)
        self.assertEqual(metrics["control_point_count"], 2)
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["in_bounds_sample_count"], 4)
        self.assertAlmostEqual(metrics["out_of_bounds_fraction"], 0.2)
        self.assertAlmostEqual(metrics["displacement_magnitude"]["max"], 1.0)
        np.testing.assert_allclose(field[0, 0, :, 0], np.ones(5))
        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_cli_accepts_custom_output_filenames(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\n", encoding="utf-8")
            np.save(volume_path, np.zeros((1, 1, 1), dtype=np.float32))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--controls-output",
                    "controls.custom.json",
                    "--field-output",
                    "field.custom.npz",
                    "--warped-output",
                    "warped.custom.npy",
                    "--manifest-output",
                    "manifest.custom.json",
                    "--metrics-output",
                    "metrics.custom.json",
                    "--grid-shape",
                    "1,1,1",
                    "--field-spacing",
                    "1,1,1",
                ]
            )

            self.assertTrue((output_dir / "controls.custom.json").exists())
            self.assertTrue((output_dir / "field.custom.npz").exists())
            self.assertTrue((output_dir / "warped.custom.npy").exists())
            self.assertTrue((output_dir / "manifest.custom.json").exists())
            self.assertTrue((output_dir / "metrics.custom.json").exists())

    def test_cli_auto_grid_shape_covers_volume_extent_at_field_spacing(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
            registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(5, dtype=np.float32).reshape(1, 1, 5))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "auto",
                    "--field-spacing",
                    "2,1,1",
                    "--volume-spacing",
                    "1,1,1",
                    "--volume-origin",
                    "0,0,0",
                    "--fill-value",
                    "-1",
                ]
            )

            with np.load(output_dir / "deformation-field.npz") as field_data:
                field = field_data["displacement_xyz"]
            manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))
            warped = np.load(output_dir / "warped-volume.npy")

        self.assertEqual(field.shape, (1, 1, 3, 3))
        self.assertEqual(manifest["requested_grid_shape"], "auto")
        self.assertEqual(manifest["actual_grid_shape_xyz"], [3, 1, 1])
        self.assertEqual(manifest["effective_volume_spacing_xyz"], [1.0, 1.0, 1.0])
        self.assertEqual(manifest["effective_volume_origin_xyz"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(field[0, 0, :, 0], np.ones(3))
        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_manifest_records_effective_volume_defaults(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            registered_path.write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(2, dtype=np.float32).reshape(1, 1, 2))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "2,1,1",
                    "--field-spacing",
                    "2,3,4",
                    "--field-origin",
                    "5,6,7",
                ]
            )

            manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))

        self.assertIsNone(manifest["volume_spacing_xyz"])
        self.assertIsNone(manifest["volume_origin_xyz"])
        self.assertEqual(manifest["effective_volume_spacing_xyz"], [2.0, 3.0, 4.0])
        self.assertEqual(manifest["effective_volume_origin_xyz"], [5.0, 6.0, 7.0])

    def test_cli_can_subsample_run_metrics(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
            registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(5, dtype=np.float32).reshape(1, 1, 5))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "5,1,1",
                    "--field-spacing",
                    "1,1,1",
                    "--volume-spacing",
                    "1,1,1",
                    "--fill-value",
                    "-1",
                    "--metrics-sample-step",
                    "2",
                ]
            )

            metrics = json.loads((output_dir / "deformation-run-metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["metrics_sample_step"], 2)
        self.assertEqual(metrics["volume_voxel_count"], 5)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["in_bounds_sample_count"], 2)
        self.assertAlmostEqual(metrics["in_bounds_fraction"], 2.0 / 3.0)

    def test_cli_passes_chunk_depth_to_reference_warper(self):
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
            registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(10, dtype=np.float32).reshape(2, 1, 5))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "auto",
                    "--field-spacing",
                    "1,1,1",
                    "--volume-spacing",
                    "1,1,1",
                    "--fill-value",
                    "-1",
                    "--chunk-depth",
                    "1",
                ]
            )

            warped = np.load(output_dir / "warped-volume.npy")
            manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])
        self.assertEqual(manifest["chunk_depth"], 1)
        self.assertEqual(manifest["output_write_mode"], "chunked_npy_memmap")

    def test_cli_chunked_npy_output_bypasses_full_array_save(self):
        pipeline = _load_pipeline()

        def fail_save_volume(path, volume):
            raise AssertionError("chunked .npy output should write directly through a memmap")

        pipeline.save_volume = fail_save_volume
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.npy"
            output_dir = temp / "outputs"
            source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
            registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
            np.save(volume_path, np.arange(10, dtype=np.float32).reshape(2, 1, 5))

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--grid-shape",
                    "auto",
                    "--field-spacing",
                    "1,1,1",
                    "--fill-value",
                    "-1",
                    "--chunk-depth",
                    "1",
                ]
            )

            warped = np.load(output_dir / "warped-volume.npy")

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_cli_can_read_and_write_tiff_volumes(self):
        calls = []

        def fake_imwrite(path, array):
            calls.append(("imwrite", str(path), array.shape))
            with Path(path).open("wb") as f:
                np.save(f, array)

        def fake_imread(path):
            calls.append(("imread", str(path)))
            with Path(path).open("rb") as f:
                return np.load(f)

        fake_tifffile = types.SimpleNamespace(imread=fake_imread, imwrite=fake_imwrite)
        previous_tifffile = sys.modules.get("tifffile")
        sys.modules["tifffile"] = fake_tifffile
        try:
            pipeline = _load_pipeline()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                source_path = temp / "source.obj"
                registered_path = temp / "registered.obj"
                volume_path = temp / "volume.tif"
                output_dir = temp / "outputs"
                source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
                registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
                with volume_path.open("wb") as f:
                    np.save(f, np.arange(5, dtype=np.float32).reshape(1, 1, 5))

                pipeline.main(
                    [
                        "--mesh-pair",
                        str(source_path),
                        str(registered_path),
                        "--volume",
                        str(volume_path),
                        "--output-dir",
                        str(output_dir),
                        "--warped-output",
                        "warped-volume.tif",
                        "--grid-shape",
                        "5,1,1",
                        "--field-spacing",
                        "1,1,1",
                        "--fill-value",
                        "-1",
                    ]
                )

                with (output_dir / "warped-volume.tif").open("rb") as f:
                    warped = np.load(f)
        finally:
            if previous_tifffile is None:
                sys.modules.pop("tifffile", None)
            else:
                sys.modules["tifffile"] = previous_tifffile

        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])
        self.assertEqual([call[0] for call in calls], ["imread", "imwrite"])

    def test_cli_can_read_and_write_zarr_volumes(self):
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            pipeline = _load_pipeline()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                source_path = temp / "source.obj"
                registered_path = temp / "registered.obj"
                volume_path = temp / "volume.zarr"
                output_dir = temp / "outputs"
                output_path = output_dir / "warped-volume.zarr"
                source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
                registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
                input_store = FakeZarrArray(
                    np.arange(10, dtype=np.float32).reshape(2, 1, 5),
                    fail_on_array=True,
                )
                fake_zarr.stores[str(volume_path)] = input_store

                pipeline.main(
                    [
                        "--mesh-pair",
                        str(source_path),
                        str(registered_path),
                        "--volume",
                        str(volume_path),
                        "--output-dir",
                        str(output_dir),
                        "--warped-output",
                        "warped-volume.zarr",
                        "--grid-shape",
                        "auto",
                        "--field-spacing",
                        "1,1,1",
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
                manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])
        self.assertEqual(input_store.materialize_count, 0)
        self.assertEqual(manifest["output_write_mode"], "chunked_zarr")
        self.assertEqual(manifest["output_paths"]["warped"], str(output_path))

    def test_cli_can_read_requested_zarr_group_array(self):
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            pipeline = _load_pipeline()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                source_path = temp / "source.obj"
                registered_path = temp / "registered.obj"
                volume_path = temp / "volume.zarr"
                output_dir = temp / "outputs"
                output_path = output_dir / "warped-volume.zarr"
                source_path.write_text("v 0 0 0\nv 4 0 0\n", encoding="utf-8")
                registered_path.write_text("v 1 0 0\nv 5 0 0\n", encoding="utf-8")
                level0 = FakeZarrArray(np.zeros((2, 1, 5), dtype=np.float32), fail_on_array=True)
                level1 = FakeZarrArray(
                    np.arange(10, dtype=np.float32).reshape(2, 1, 5),
                    fail_on_array=True,
                )
                fake_zarr.stores[str(volume_path)] = FakeZarrGroup({"0": level0, "1": level1})

                pipeline.main(
                    [
                        "--mesh-pair",
                        str(source_path),
                        str(registered_path),
                        "--volume",
                        str(volume_path),
                        "--output-dir",
                        str(output_dir),
                        "--warped-output",
                        "warped-volume.zarr",
                        "--zarr-array-key",
                        "1",
                        "--grid-shape",
                        "auto",
                        "--field-spacing",
                        "1,1,1",
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
                manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])
        self.assertEqual(level0.materialize_count, 0)
        self.assertEqual(level1.materialize_count, 0)
        self.assertEqual(manifest["zarr_array_key"], "1")
        self.assertEqual(manifest["output_write_mode"], "chunked_zarr")

    def test_cli_can_read_real_zarr_group_array(self):
        zarr = _import_real_zarr()
        pipeline = _load_pipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.obj"
            registered_path = temp / "registered.obj"
            volume_path = temp / "volume.zarr"
            output_dir = temp / "outputs"
            source_path.write_text("\n".join(["v 0 0 0", "v 4 0 0"]), encoding="utf-8")
            registered_path.write_text("\n".join(["v 0 0 0", "v 4 0 0"]), encoding="utf-8")
            group = zarr.open_group(str(volume_path), mode="w")
            group.create_dataset("0", shape=(1, 1, 5), chunks=(1, 1, 5), dtype="float32")[:] = 999
            group.create_dataset("1", shape=(1, 1, 5), chunks=(1, 1, 5), dtype="float32")[:] = np.arange(
                5, dtype=np.float32
            ).reshape(1, 1, 5)

            pipeline.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--volume",
                    str(volume_path),
                    "--output-dir",
                    str(output_dir),
                    "--warped-output",
                    "warped-volume.zarr",
                    "--zarr-array-key",
                    "1",
                    "--grid-shape",
                    "5,1,1",
                    "--field-spacing",
                    "1,1,1",
                    "--field-origin",
                    "0,0,0",
                    "--volume-spacing",
                    "1,1,1",
                    "--volume-origin",
                    "0,0,0",
                    "--max-points-per-mesh",
                    "0",
                    "--chunk-depth",
                    "1",
                ]
            )

            manifest = json.loads((output_dir / "deformation-run-manifest.json").read_text(encoding="utf-8"))
            warped = np.asarray(zarr.open(str(output_dir / "warped-volume.zarr"), mode="r"))

        self.assertEqual(manifest["zarr_array_key"], "1")
        self.assertEqual(manifest["output_write_mode"], "chunked_zarr")
        np.testing.assert_allclose(warped[0, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_help_mentions_tiff_and_zarr_volume_support(self):
        pipeline = _load_pipeline()

        help_text = pipeline.build_arg_parser().format_help()

        self.assertIn(".tif", help_text)
        self.assertIn(".zarr", help_text)
        self.assertIn("--zarr-array-key", help_text)
        self.assertIn("auto", help_text)
        self.assertIn("--chunk-depth", help_text)
        self.assertIn("--field-query-chunk-size", help_text)
        self.assertIn("--field-control-chunk-size", help_text)
        self.assertIn("--manifest-output", help_text)
        self.assertIn("--metrics-output", help_text)
        self.assertIn("--metrics-sample-step", help_text)


if __name__ == "__main__":
    unittest.main()
