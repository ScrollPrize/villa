import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "warp_volume_with_field.py"


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
        self.open_calls = []

    def open(self, path, mode="r", shape=None, dtype=None, chunks=None):
        self.open_calls.append((str(path), mode, shape, dtype, chunks))
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


def _load_warper():
    module_name = "_test_warp_volume_with_field"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WarpVolumeWithFieldTest(unittest.TestCase):
    def test_trilinear_sample_returns_exact_values_at_integer_points(self):
        warper = _load_warper()

        volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        samples = warper.trilinear_sample_zyx(
            volume,
            np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], dtype=np.float64),
            fill_value=-1.0,
        )

        np.testing.assert_allclose(samples, [0.0, volume[2, 1, 1]])

    def test_trilinear_sample_interpolates_fractional_points(self):
        warper = _load_warper()

        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        samples = warper.trilinear_sample_zyx(
            volume,
            np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
            fill_value=-1.0,
        )

        self.assertAlmostEqual(float(samples[0]), float(volume.mean()))

    def test_warp_volume_identity_field_returns_input(self):
        warper = _load_warper()

        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        field = np.zeros((2, 2, 2, 3), dtype=np.float32)

        warped = warper.warp_volume_with_displacement_field(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(1.0, 1.0, 1.0),
            fill_value=-1.0,
        )

        np.testing.assert_allclose(warped, volume)

    def test_warp_volume_uses_backward_sampling_convention(self):
        warper = _load_warper()

        volume = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
        field = np.zeros((1, 1, 5, 3), dtype=np.float32)
        field[..., 0] = 1.0

        warped = warper.warp_volume_with_displacement_field(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(1.0, 1.0, 1.0),
            fill_value=-1.0,
        )

        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_warp_sampling_diagnostics_reports_in_bounds_fraction(self):
        warper = _load_warper()

        volume = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
        field = np.zeros((1, 1, 5, 3), dtype=np.float32)
        field[..., 0] = 1.0

        diagnostics = warper.warp_sampling_diagnostics(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(1.0, 1.0, 1.0),
            volume_origin_xyz=(0.0, 0.0, 0.0),
            volume_spacing_xyz=(1.0, 1.0, 1.0),
            chunk_depth=2,
        )

        self.assertEqual(diagnostics["sample_count"], 5)
        self.assertEqual(diagnostics["in_bounds_sample_count"], 4)
        self.assertAlmostEqual(diagnostics["in_bounds_fraction"], 0.8)
        self.assertAlmostEqual(diagnostics["out_of_bounds_fraction"], 0.2)
        self.assertAlmostEqual(diagnostics["displacement_magnitude"]["max"], 1.0)

    def test_warp_sampling_diagnostics_can_subsample_large_volumes(self):
        warper = _load_warper()

        volume = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
        field = np.zeros((1, 1, 5, 3), dtype=np.float32)
        field[..., 0] = 1.0

        diagnostics = warper.warp_sampling_diagnostics(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(1.0, 1.0, 1.0),
            volume_origin_xyz=(0.0, 0.0, 0.0),
            volume_spacing_xyz=(1.0, 1.0, 1.0),
            sample_step=2,
        )

        self.assertEqual(diagnostics["metrics_sample_step"], 2)
        self.assertEqual(diagnostics["volume_voxel_count"], 5)
        self.assertEqual(diagnostics["sample_count"], 3)
        self.assertEqual(diagnostics["in_bounds_sample_count"], 2)
        self.assertAlmostEqual(diagnostics["in_bounds_fraction"], 2.0 / 3.0)

    def test_warp_volume_resamples_lower_resolution_field_to_volume_grid(self):
        warper = _load_warper()

        volume = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
        field = np.zeros((1, 1, 3, 3), dtype=np.float32)
        field[..., 0] = 1.0

        warped = warper.warp_volume_with_displacement_field(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(2.0, 1.0, 1.0),
            volume_origin_xyz=(0.0, 0.0, 0.0),
            volume_spacing_xyz=(1.0, 1.0, 1.0),
            fill_value=-1.0,
        )

        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_warp_volume_rejects_non_positive_spacing(self):
        warper = _load_warper()

        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        field = np.zeros((2, 2, 2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "field spacing values must be positive"):
            warper.warp_volume_with_displacement_field(
                volume,
                field,
                origin_xyz=(0.0, 0.0, 0.0),
                spacing_xyz=(1.0, 0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "volume spacing values must be positive"):
            warper.warp_volume_with_displacement_field(
                volume,
                field,
                origin_xyz=(0.0, 0.0, 0.0),
                spacing_xyz=(1.0, 1.0, 1.0),
                volume_spacing_xyz=(1.0, -1.0, 1.0),
            )

    def test_chunked_warp_matches_full_volume_warp(self):
        warper = _load_warper()

        volume = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
        field = np.zeros((2, 2, 3, 3), dtype=np.float32)
        field[..., 0] = 0.5
        field[..., 2] = 0.25

        full = warper.warp_volume_with_displacement_field(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(2.0, 2.0, 2.0),
            volume_origin_xyz=(0.0, 0.0, 0.0),
            volume_spacing_xyz=(1.0, 1.0, 1.0),
            fill_value=-1.0,
        )
        chunked = warper.warp_volume_with_displacement_field_chunked(
            volume,
            field,
            origin_xyz=(0.0, 0.0, 0.0),
            spacing_xyz=(2.0, 2.0, 2.0),
            volume_origin_xyz=(0.0, 0.0, 0.0),
            volume_spacing_xyz=(1.0, 1.0, 1.0),
            fill_value=-1.0,
            chunk_depth=2,
        )

        np.testing.assert_allclose(chunked, full)

    def test_cli_writes_warped_npy_volume(self):
        warper = _load_warper()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            volume_path = temp / "volume.npy"
            field_path = temp / "field.npz"
            output_path = temp / "warped.npy"
            np.save(volume_path, np.arange(5, dtype=np.float32).reshape(1, 1, 5))
            field = np.zeros((1, 1, 5, 3), dtype=np.float32)
            field[..., 0] = 1.0
            np.savez_compressed(
                field_path,
                displacement_xyz=field,
                origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            )

            warper.main(
                [
                    "--volume",
                    str(volume_path),
                    "--field",
                    str(field_path),
                    "--output",
                    str(output_path),
                    "--volume-spacing",
                    "1,1,1",
                    "--volume-origin",
                    "0,0,0",
                    "--fill-value",
                    "-1",
                ]
            )

            warped = np.load(output_path)

        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_parser_rejects_non_positive_volume_spacing(self):
        warper = _load_warper()
        parser = warper.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--volume",
                    "volume.npy",
                    "--field",
                    "field.npz",
                    "--output",
                    "warped.npy",
                    "--volume-spacing",
                    "0,1,1",
                ]
            )

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--volume",
                    "volume.npy",
                    "--field",
                    "field.npz",
                    "--output",
                    "warped.npy",
                    "--chunk-depth",
                    "-1",
                ]
            )

    def test_cli_writes_chunked_warped_npy_volume(self):
        warper = _load_warper()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            volume_path = temp / "volume.npy"
            field_path = temp / "field.npz"
            output_path = temp / "warped.npy"
            np.save(volume_path, np.arange(10, dtype=np.float32).reshape(2, 1, 5))
            field = np.zeros((2, 1, 5, 3), dtype=np.float32)
            field[..., 0] = 1.0
            np.savez_compressed(
                field_path,
                displacement_xyz=field,
                origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            )

            warper.main(
                [
                    "--volume",
                    str(volume_path),
                    "--field",
                    str(field_path),
                    "--output",
                    str(output_path),
                    "--volume-spacing",
                    "1,1,1",
                    "--volume-origin",
                    "0,0,0",
                    "--fill-value",
                    "-1",
                    "--chunk-depth",
                    "1",
                ]
            )

            warped = np.load(output_path)

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_cli_chunked_npy_output_bypasses_full_array_save(self):
        warper = _load_warper()

        def fail_save_volume(path, volume):
            raise AssertionError("chunked .npy output should write directly through a memmap")

        warper.save_volume = fail_save_volume
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            volume_path = temp / "volume.npy"
            field_path = temp / "field.npz"
            output_path = temp / "warped.npy"
            np.save(volume_path, np.arange(10, dtype=np.float32).reshape(2, 1, 5))
            field = np.zeros((2, 1, 5, 3), dtype=np.float32)
            field[..., 0] = 1.0
            np.savez_compressed(
                field_path,
                displacement_xyz=field,
                origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            )

            warper.main(
                [
                    "--volume",
                    str(volume_path),
                    "--field",
                    str(field_path),
                    "--output",
                    str(output_path),
                    "--fill-value",
                    "-1",
                    "--chunk-depth",
                    "1",
                ]
            )

            warped = np.load(output_path)

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])


    def test_tiff_volume_io_uses_lazy_tifffile_dependency(self):
        warper = _load_warper()

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
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                tiff_path = temp / "warped.tif"
                expected = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)

                warper.save_volume(tiff_path, expected)
                loaded = warper.load_volume(tiff_path)
        finally:
            if previous_tifffile is None:
                sys.modules.pop("tifffile", None)
            else:
                sys.modules["tifffile"] = previous_tifffile

        np.testing.assert_array_equal(loaded, expected)
        self.assertEqual(calls[0][0], "imwrite")
        self.assertEqual(calls[1][0], "imread")

    def test_zarr_volume_io_uses_lazy_zarr_dependency(self):
        warper = _load_warper()
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                zarr_path = Path(temp_dir) / "warped.zarr"
                expected = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)

                warper.save_volume(zarr_path, expected)
                loaded = warper.load_volume(zarr_path)
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        np.testing.assert_array_equal(loaded, expected)
        self.assertEqual(fake_zarr.open_calls[0][1], "w")
        self.assertEqual(fake_zarr.open_calls[1][1], "r")

    def test_chunked_zarr_input_is_not_materialized(self):
        warper = _load_warper()
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                volume_path = temp / "volume.zarr"
                field_path = temp / "field.npz"
                output_path = temp / "warped.zarr"
                input_store = FakeZarrArray(
                    np.arange(10, dtype=np.float32).reshape(2, 1, 5),
                    fail_on_array=True,
                )
                fake_zarr.stores[str(volume_path)] = input_store
                field = np.zeros((2, 1, 5, 3), dtype=np.float32)
                field[..., 0] = 1.0
                np.savez_compressed(
                    field_path,
                    displacement_xyz=field,
                    origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                )

                warper.main(
                    [
                        "--volume",
                        str(volume_path),
                        "--field",
                        str(field_path),
                        "--output",
                        str(output_path),
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        self.assertEqual(input_store.materialize_count, 0)
        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_chunked_zarr_group_input_uses_requested_array_key(self):
        warper = _load_warper()
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                volume_path = temp / "volume.zarr"
                field_path = temp / "field.npz"
                output_path = temp / "warped.zarr"
                level0 = FakeZarrArray(np.zeros((2, 1, 5), dtype=np.float32), fail_on_array=True)
                level1 = FakeZarrArray(
                    np.arange(10, dtype=np.float32).reshape(2, 1, 5),
                    fail_on_array=True,
                )
                fake_zarr.stores[str(volume_path)] = FakeZarrGroup({"0": level0, "1": level1})
                field = np.zeros((2, 1, 5, 3), dtype=np.float32)
                field[..., 0] = 1.0
                np.savez_compressed(
                    field_path,
                    displacement_xyz=field,
                    origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                )

                warper.main(
                    [
                        "--volume",
                        str(volume_path),
                        "--field",
                        str(field_path),
                        "--output",
                        str(output_path),
                        "--zarr-array-key",
                        "1",
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        self.assertEqual(level0.materialize_count, 0)
        self.assertEqual(level1.materialize_count, 0)
        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_chunked_zarr_group_input_accepts_array_key_in_path(self):
        warper = _load_warper()
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                volume_path = temp / "volume.zarr"
                field_path = temp / "field.npz"
                output_path = temp / "warped.zarr"
                level0 = FakeZarrArray(np.zeros((2, 1, 5), dtype=np.float32), fail_on_array=True)
                level1 = FakeZarrArray(
                    np.arange(10, dtype=np.float32).reshape(2, 1, 5),
                    fail_on_array=True,
                )
                fake_zarr.stores[str(volume_path)] = FakeZarrGroup({"0": level0, "1": level1})
                field = np.zeros((2, 1, 5, 3), dtype=np.float32)
                field[..., 0] = 1.0
                np.savez_compressed(
                    field_path,
                    displacement_xyz=field,
                    origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                )

                warper.main(
                    [
                        "--volume",
                        str(volume_path / "1"),
                        "--field",
                        str(field_path),
                        "--output",
                        str(output_path),
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        self.assertEqual(level0.materialize_count, 0)
        self.assertEqual(level1.materialize_count, 0)
        self.assertEqual(fake_zarr.open_calls[0][0], str(volume_path))
        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_real_zarr_group_input_uses_requested_array_key(self):
        zarr = _import_real_zarr()
        warper = _load_warper()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            volume_path = temp / "volume.zarr"
            field_path = temp / "field.npz"
            output_path = temp / "warped.zarr"
            group = zarr.open_group(str(volume_path), mode="w")
            group.create_dataset("0", shape=(1, 1, 5), chunks=(1, 1, 5), dtype="float32")[:] = 999
            group.create_dataset("1", shape=(1, 1, 5), chunks=(1, 1, 5), dtype="float32")[:] = np.arange(
                5, dtype=np.float32
            ).reshape(1, 1, 5)
            field = np.zeros((1, 1, 5, 3), dtype=np.float32)
            field[..., 0] = 1.0
            np.savez_compressed(
                field_path,
                displacement_xyz=field,
                origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            )

            warper.main(
                [
                    "--volume",
                    str(volume_path),
                    "--field",
                    str(field_path),
                    "--output",
                    str(output_path),
                    "--zarr-array-key",
                    "1",
                    "--fill-value",
                    "-1",
                    "--chunk-depth",
                    "1",
                ]
            )

            warped = np.asarray(zarr.open(str(output_path), mode="r"))

        np.testing.assert_allclose(warped[0, 0], [-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_cli_writes_chunked_zarr_output_without_full_array_save(self):
        warper = _load_warper()
        fake_zarr = FakeZarrModule()
        previous_zarr = sys.modules.get("zarr")
        sys.modules["zarr"] = fake_zarr

        def fail_save_volume(path, volume):
            raise AssertionError("chunked .zarr output should write directly to the zarr store")

        warper.save_volume = fail_save_volume
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                volume_path = temp / "volume.npy"
                field_path = temp / "field.npz"
                output_path = temp / "warped.zarr"
                np.save(volume_path, np.arange(10, dtype=np.float32).reshape(2, 1, 5))
                field = np.zeros((2, 1, 5, 3), dtype=np.float32)
                field[..., 0] = 1.0
                np.savez_compressed(
                    field_path,
                    displacement_xyz=field,
                    origin_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    spacing_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                )

                warper.main(
                    [
                        "--volume",
                        str(volume_path),
                        "--field",
                        str(field_path),
                        "--output",
                        str(output_path),
                        "--fill-value",
                        "-1",
                        "--chunk-depth",
                        "1",
                    ]
                )

                warped = np.asarray(fake_zarr.stores[str(output_path)])
        finally:
            if previous_zarr is None:
                sys.modules.pop("zarr", None)
            else:
                sys.modules["zarr"] = previous_zarr

        np.testing.assert_allclose(warped[:, 0], [[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 5.0, 6.0, 7.0, 8.0]])

    def test_help_mentions_tiff_and_zarr_volume_support(self):
        warper = _load_warper()

        help_text = warper.build_arg_parser().format_help()

        self.assertIn(".tif", help_text)
        self.assertIn(".zarr", help_text)
        self.assertIn("--zarr-array-key", help_text)
        self.assertIn("--chunk-depth", help_text)


if __name__ == "__main__":
    unittest.main()
