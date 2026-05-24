import sys
import types
import unittest
from contextlib import contextmanager
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

from tifxyz_dataset.common import (  # noqa: E402
    _build_surface_label_volume,
    _build_surface_supervision_from_ink_mask,
    _fix_known_bottom_right_padding,
    _normalize_distance_pair,
)
from tifxyz_dataset import patch_finding  # noqa: E402
from tifxyz_dataset.tifxyz_dataset import TifxyzInkDataset  # noqa: E402


class TifxyzLabelHelpersTest(unittest.TestCase):
    def _dataset_config(self, auto_fix_padding_multiples=unittest.mock.sentinel.default):
        config = {
            "patch_size": [8, 8, 8],
            "bg_distance": 1,
            "label_distance": 1,
            "datasets": [],
        }
        if auto_fix_padding_multiples is not unittest.mock.sentinel.default:
            config["auto_fix_padding_multiples"] = auto_fix_padding_multiples
        return config

    @contextmanager
    def _patched_find_patches(self):
        with patch(
            "tifxyz_dataset.tifxyz_dataset.find_patches",
            return_value=([], {"kept_patches": 0}),
        ) as find_patches:
            yield find_patches

    def test_normalize_distance_pair_accepts_scalar_and_clamps_negative_values(self):
        self.assertEqual(_normalize_distance_pair(3, "label_distance"), (3.0, 3.0))
        self.assertEqual(
            _normalize_distance_pair([4, -2], "label_distance"),
            (4.0, 0.0),
        )

    def test_surface_supervision_marks_ink_near_background_and_unknown(self):
        ink_mask = np.zeros((7, 7), dtype=bool)
        ink_mask[3, 3] = True

        supervision = _build_surface_supervision_from_ink_mask(
            ink_mask,
            bg_dilate_distance=1,
        )

        self.assertEqual(int(supervision[3, 3]), 1)
        self.assertEqual(int(supervision[3, 4]), 0)
        self.assertEqual(int(supervision[0, 0]), 100)
        self.assertEqual(supervision.dtype, np.uint8)

    def test_surface_label_volume_prefers_positive_labels_over_background(self):
        positive = np.zeros((3, 3, 3), dtype=np.float32)
        background = np.zeros((3, 3, 3), dtype=np.float32)
        background[1, 1, 1] = 1.0
        positive[1, 1, 1] = 1.0

        label_volume = _build_surface_label_volume(positive, background, (3, 3, 3))

        self.assertEqual(float(label_volume[1, 1, 1]), 1.0)
        self.assertEqual(float(label_volume[0, 0, 0]), 2.0)

    def test_known_bottom_right_padding_crops_empty_padded_edges(self):
        label = np.zeros((8, 8), dtype=np.uint8)
        label[:5, :6] = 1

        fixed, matched_multiple = _fix_known_bottom_right_padding(
            label,
            expected_shape=(5, 6),
            multiples=[4],
        )

        self.assertEqual(matched_multiple, 4)
        self.assertEqual(fixed.shape, (5, 6))
        self.assertTrue(np.all(fixed == 1))

    def test_dataset_uses_configured_auto_fix_padding_multiples(self):
        config = self._dataset_config(auto_fix_padding_multiples=[32])

        with self._patched_find_patches() as find_patches:
            dataset = TifxyzInkDataset(config, apply_augmentation=False)

        self.assertEqual(dataset.auto_fix_padding_multiples, [32])
        self.assertEqual(find_patches.call_args.kwargs["auto_fix_padding_multiples"], [32])

    def test_dataset_defaults_auto_fix_padding_multiples_when_missing(self):
        with self._patched_find_patches() as find_patches:
            dataset = TifxyzInkDataset(self._dataset_config(), apply_augmentation=False)

        self.assertEqual(dataset.auto_fix_padding_multiples, [64, 256])
        self.assertEqual(find_patches.call_args.kwargs["auto_fix_padding_multiples"], [64, 256])

    def test_dataset_allows_disabling_auto_fix_padding(self):
        config = self._dataset_config(auto_fix_padding_multiples=None)

        with self._patched_find_patches() as find_patches:
            dataset = TifxyzInkDataset(config, apply_augmentation=False)

        self.assertEqual(dataset.auto_fix_padding_multiples, [])
        self.assertEqual(find_patches.call_args.kwargs["auto_fix_padding_multiples"], [])

    def test_dataset_accepts_scalar_auto_fix_padding_multiple(self):
        config = self._dataset_config(auto_fix_padding_multiples=64)

        with self._patched_find_patches() as find_patches:
            dataset = TifxyzInkDataset(config, apply_augmentation=False)

        self.assertEqual(dataset.auto_fix_padding_multiples, [64])
        self.assertEqual(find_patches.call_args.kwargs["auto_fix_padding_multiples"], [64])

    def test_dataset_rejects_invalid_auto_fix_padding_multiples(self):
        invalid_values = ([0], ["64"], [-8], True, [True])

        for value in invalid_values:
            with self.subTest(value=value), self._patched_find_patches():
                with self.assertRaisesRegex(ValueError, "auto_fix_padding_multiples"):
                    TifxyzInkDataset(
                        self._dataset_config(auto_fix_padding_multiples=value),
                        apply_augmentation=False,
                    )

    def test_patch_cache_key_includes_auto_fix_padding_multiples(self):
        self.assertTrue(hasattr(patch_finding, "_build_patch_cache_key"))
        dataset = {
            "volume_path": "volume.zarr",
            "volume_scale": 0,
            "segments_path": "segments",
            "z_range": None,
        }

        key_64 = patch_finding._build_patch_cache_key(
            dataset,
            patch_size_zyx=(8, 8, 8),
            min_positive_fraction=0.01,
            min_span_ratio=0.5,
            overlap_fraction=0.25,
            auto_fix_padding_multiples=[64],
        )
        key_disabled = patch_finding._build_patch_cache_key(
            dataset,
            patch_size_zyx=(8, 8, 8),
            min_positive_fraction=0.01,
            min_span_ratio=0.5,
            overlap_fraction=0.25,
            auto_fix_padding_multiples=[],
        )

        self.assertNotEqual(key_64, key_disabled)


if __name__ == "__main__":
    unittest.main()
