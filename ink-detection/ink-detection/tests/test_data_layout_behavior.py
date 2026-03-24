from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import time

import zarr

from ink.recipes.data.layout import NestedZarrLayout


def _write_zarr_array(path: Path, *, shape=(1, 1), dtype="uint8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=shape, dtype=dtype)
    store[:] = 0


class NestedZarrLayoutTests(unittest.TestCase):
    def test_resolve_paths_uses_nested_group_segment_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels.zarr")
            _write_zarr_array(segment_dir / "1234_supervision_mask.zarr")

            layout = NestedZarrLayout(root)
            paths = layout.resolve_paths("1234")

            self.assertEqual(paths.segment_dir, segment_dir)
            self.assertEqual(paths.volume_path, segment_dir / "1234.zarr")
            self.assertEqual(paths.inklabels_path, segment_dir / "1234_inklabels.zarr")
            self.assertEqual(paths.supervision_mask_path, segment_dir / "1234_supervision_mask.zarr")

    def test_resolve_paths_does_not_fall_back_to_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_zarr_array(root / "1234.zarr", shape=(4, 8, 8))

            layout = NestedZarrLayout(root)

            with self.assertRaisesRegex(FileNotFoundError, "segment directory"):
                layout.resolve_paths("1234")

    def test_resolve_paths_requires_supervision_mask_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels.zarr")
            _write_zarr_array(segment_dir / "1234_mask.zarr")

            layout = NestedZarrLayout(root)

            with self.assertRaisesRegex(FileNotFoundError, "supervision_mask"):
                layout.resolve_paths("1234")

    def test_resolve_paths_requires_exact_segment_id_and_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "20240303120000_0034"
            _write_zarr_array(segment_dir / "20240303120000_0034.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "20240303120000_0034_inklabels.zarr")
            _write_zarr_array(segment_dir / "20240303120000_0034_supervision_mask.zarr")

            layout = NestedZarrLayout(root)

            with self.assertRaisesRegex(FileNotFoundError, "segment directory"):
                layout.resolve_paths("34")

    def test_resolve_paths_supports_validation_mask_name_with_version_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels_v2.zarr")
            _write_zarr_array(segment_dir / "1234_validation_mask_v2.zarr")

            layout = NestedZarrLayout(root)
            paths = layout.resolve_paths(
                "1234",
                label_suffix="_v2",
                mask_suffix="_v2",
                mask_name="validation_mask",
            )

            self.assertEqual(paths.inklabels_path, segment_dir / "1234_inklabels_v2.zarr")
            self.assertEqual(paths.mask_path, segment_dir / "1234_validation_mask_v2.zarr")
            self.assertEqual(paths.supervision_mask_path, segment_dir / "1234_validation_mask_v2.zarr")

    def test_resolve_paths_uses_latest_version_when_suffix_is_unspecified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels.zarr")
            _write_zarr_array(segment_dir / "1234_inklabels_v2.zarr")
            _write_zarr_array(segment_dir / "1234_inklabels_v10.zarr")
            _write_zarr_array(segment_dir / "1234_supervision_mask.zarr")
            _write_zarr_array(segment_dir / "1234_supervision_mask_v3.zarr")

            layout = NestedZarrLayout(root)
            paths = layout.resolve_paths("1234")

            self.assertEqual(paths.inklabels_path, segment_dir / "1234_inklabels_v10.zarr")
            self.assertEqual(paths.mask_path, segment_dir / "1234_supervision_mask_v3.zarr")

    def test_label_mask_fingerprint_is_portable_across_dataset_relocation(self):
        with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
            for root_str in (tmpdir_a, tmpdir_b):
                root = Path(root_str)
                group_dir = root / "PHerc0332"
                segment_dir = group_dir / "1234"
                _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
                _write_zarr_array(segment_dir / "1234_inklabels.zarr", shape=(8, 8))
                _write_zarr_array(segment_dir / "1234_supervision_mask.zarr", shape=(8, 8))

            fingerprint_a = NestedZarrLayout(tmpdir_a).label_mask_fingerprint("1234")
            fingerprint_b = NestedZarrLayout(tmpdir_b).label_mask_fingerprint("1234")

        self.assertEqual(fingerprint_a, fingerprint_b)

    def test_label_mask_fingerprint_changes_when_label_tree_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels.zarr", shape=(8, 8))
            _write_zarr_array(segment_dir / "1234_supervision_mask.zarr", shape=(8, 8))

            layout = NestedZarrLayout(root)
            fingerprint_before = layout.label_mask_fingerprint("1234")

            time.sleep(0.01)
            zarr.open(str(segment_dir / "1234_inklabels.zarr"), mode="r+")[:] = 1
            fingerprint_after = layout.label_mask_fingerprint("1234")

        self.assertNotEqual(fingerprint_before, fingerprint_after)

    def test_label_mask_metadata_fingerprint_is_portable_across_dataset_relocation(self):
        with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
            for root_str in (tmpdir_a, tmpdir_b):
                root = Path(root_str)
                group_dir = root / "PHerc0332"
                segment_dir = group_dir / "1234"
                _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
                _write_zarr_array(segment_dir / "1234_inklabels.zarr", shape=(8, 8))
                _write_zarr_array(segment_dir / "1234_supervision_mask.zarr", shape=(8, 8))

            fingerprint_a = NestedZarrLayout(tmpdir_a).label_mask_metadata_fingerprint("1234")
            fingerprint_b = NestedZarrLayout(tmpdir_b).label_mask_metadata_fingerprint("1234")

        self.assertEqual(fingerprint_a, fingerprint_b)

    def test_label_mask_metadata_fingerprint_changes_when_zarr_metadata_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "PHerc0332"
            segment_dir = group_dir / "1234"
            _write_zarr_array(segment_dir / "1234.zarr", shape=(4, 8, 8))
            _write_zarr_array(segment_dir / "1234_inklabels.zarr", shape=(8, 8))
            _write_zarr_array(segment_dir / "1234_supervision_mask.zarr", shape=(8, 8))

            layout = NestedZarrLayout(root)
            fingerprint_before = layout.label_mask_metadata_fingerprint("1234")

            zarr.open(str(segment_dir / "1234_inklabels.zarr"), mode="r+").attrs["cache_version"] = "v2"
            fingerprint_after = layout.label_mask_metadata_fingerprint("1234")

        self.assertNotEqual(fingerprint_before, fingerprint_after)

if __name__ == "__main__":
    unittest.main()
