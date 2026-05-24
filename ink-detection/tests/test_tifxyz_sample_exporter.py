import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
VESUVIUS_SRC = REPO_ROOT / "vesuvius" / "src"
for path in (PROJECT_ROOT, VESUVIUS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _lightweight_dependency_modules():
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_utils_data.Dataset = Dataset
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    vesuvius = types.ModuleType("vesuvius")
    tifxyz = types.ModuleType("vesuvius.tifxyz")
    tifxyz.load_folder = lambda *args, **kwargs: []
    tifxyz.interpolate_at_points = None

    datasets_common = types.ModuleType("vesuvius.neural_tracing.datasets.common")
    datasets_common.normalize_zscore = lambda arr: arr
    datasets_common._parse_z_range = lambda value: value
    datasets_common._segment_overlaps_z_range = lambda *args, **kwargs: True
    datasets_common.open_zarr = lambda *args, **kwargs: None

    cover_bboxes = types.ModuleType(
        "vesuvius.neural_tracing.inference.generate_segment_cover_bboxes"
    )
    cover_bboxes._generate_segment_cover_records = lambda *args, **kwargs: []

    training_transforms = types.ModuleType(
        "vesuvius.models.augmentation.pipelines.training_transforms"
    )
    training_transforms.create_training_transforms = lambda *args, **kwargs: None

    return {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "vesuvius": vesuvius,
        "vesuvius.tifxyz": tifxyz,
        "vesuvius.neural_tracing": types.ModuleType("vesuvius.neural_tracing"),
        "vesuvius.neural_tracing.datasets": types.ModuleType("vesuvius.neural_tracing.datasets"),
        "vesuvius.neural_tracing.datasets.common": datasets_common,
        "vesuvius.neural_tracing.inference": types.ModuleType("vesuvius.neural_tracing.inference"),
        "vesuvius.neural_tracing.inference.generate_segment_cover_bboxes": cover_bboxes,
        "vesuvius.models": types.ModuleType("vesuvius.models"),
        "vesuvius.models.augmentation": types.ModuleType("vesuvius.models.augmentation"),
        "vesuvius.models.augmentation.pipelines": types.ModuleType("vesuvius.models.augmentation.pipelines"),
        "vesuvius.models.augmentation.pipelines.training_transforms": training_transforms,
    }


_DEPENDENCY_PATCHER = patch.dict(sys.modules, _lightweight_dependency_modules())
_DEPENDENCY_PATCHER.start()


def tearDownModule():
    for name in list(sys.modules):
        if name == "tifxyz_dataset" or name.startswith("tifxyz_dataset."):
            sys.modules.pop(name, None)
    _DEPENDENCY_PATCHER.stop()


import tifxyz_dataset.export_samples as export_samples_module  # noqa: E402
from tifxyz_dataset.export_samples import (  # noqa: E402
    export_sample,
    export_samples,
    main,
    parse_indices,
    sample_to_numpy,
)


class FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeDataset:
    def __init__(self):
        self.samples = [
            self._sample(0, 11.0),
            self._sample(1, 22.0),
            self._sample(2, 33.0),
        ]

    def _sample(self, idx, fill_value):
        return {
            "vol": FakeTensor(np.full((2, 3, 4), fill_value, dtype=np.float32)),
            "labeled_vox_at_surface": np.full((2, 3, 4), idx, dtype=np.float32),
            "surface_vox": np.ones((2, 3, 4), dtype=np.float32),
            "projected_loss_mask": np.zeros((2, 3, 4), dtype=np.float32),
            "patch": {
                "world_bbox": [0, 1, 2, 4, 5, 8],
                "dataset_idx": np.int64(idx),
                "segment_uuid": f"segment-{idx}",
                "segment": object(),
            },
            "idx": idx,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TifxyzSampleExporterTest(unittest.TestCase):
    def test_sample_to_numpy_converts_tensor_like_arrays(self):
        arrays = sample_to_numpy(FakeDataset()[0])

        self.assertEqual(set(arrays), {
            "vol",
            "labeled_vox_at_surface",
            "surface_vox",
            "projected_loss_mask",
        })
        self.assertEqual(arrays["vol"].shape, (2, 3, 4))
        self.assertEqual(arrays["vol"].dtype, np.float32)
        self.assertEqual(float(arrays["vol"][0, 0, 0]), 11.0)

    def test_export_sample_writes_arrays_and_json_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = export_sample(FakeDataset(), 1, Path(tmpdir))

            self.assertEqual(sample_dir.name, "sample_000001")
            self.assertTrue((sample_dir / "vol.npy").is_file())
            self.assertTrue((sample_dir / "labeled_vox_at_surface.npy").is_file())
            self.assertTrue((sample_dir / "surface_vox.npy").is_file())
            self.assertTrue((sample_dir / "projected_loss_mask.npy").is_file())

            vol = np.load(sample_dir / "vol.npy")
            self.assertEqual(float(vol[0, 0, 0]), 22.0)

            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_index"], 1)
            self.assertEqual(metadata["sample_idx"], 1)
            self.assertEqual(metadata["arrays"]["vol"]["shape"], [2, 3, 4])
            self.assertEqual(metadata["arrays"]["vol"]["dtype"], "float32")
            self.assertEqual(metadata["patch"]["dataset_idx"], 1)
            self.assertIn("segment", metadata["patch"])

    def test_export_sample_refuses_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_sample(FakeDataset(), 0, Path(tmpdir))

            with self.assertRaisesRegex(FileExistsError, "sample_000000"):
                export_sample(FakeDataset(), 0, Path(tmpdir))

    def test_export_samples_writes_manifest_for_explicit_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = export_samples(FakeDataset(), Path(tmpdir), indices=[2, 0])

            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(
                [sample["dataset_index"] for sample in manifest["samples"]],
                [2, 0],
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["sample_count"],
                2,
            )

    def test_parse_indices_accepts_comma_separated_non_negative_ints(self):
        self.assertEqual(parse_indices("0, 4,9"), [0, 4, 9])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_indices("0,-1")

    def test_cli_rejects_count_and_indices_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    main([
                        "--config",
                        "config.json",
                        "--output",
                        tmpdir,
                        "--count",
                        "1",
                        "--indices",
                        "0",
                    ])

        self.assertNotEqual(cm.exception.code, 0)

    def test_cli_requires_count_or_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(export_samples_module, "load_config", return_value={"datasets": []}), patch.object(
                export_samples_module,
                "build_dataset",
                return_value=FakeDataset(),
            ), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    main([
                        "--config",
                        "config.json",
                        "--output",
                        tmpdir,
                    ])

        self.assertNotEqual(cm.exception.code, 0)

    def test_cli_exports_with_injected_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(export_samples_module, "load_config", return_value={"datasets": []}), patch.object(
                export_samples_module,
                "build_dataset",
                return_value=FakeDataset(),
            ):
                exit_code = main([
                    "--config",
                    "config.json",
                    "--output",
                    tmpdir,
                    "--indices",
                    "1",
                ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmpdir) / "sample_000001" / "vol.npy").is_file())


if __name__ == "__main__":
    unittest.main()
