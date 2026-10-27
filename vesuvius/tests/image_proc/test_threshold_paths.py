"""Prepared PBS-only tiny-store controls for the current-main threshold guard.

Uses selected-array inputs on valid runs; PR2295's separate group-worker routing
and physical-scale improvements are not part of this guard's proof.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import ThresholdConfig, ThresholdTask
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def tree(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class ThresholdPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def args(self, source, output):
        return SimpleNamespace(input_zarr=str(source), output_zarr=str(output),
            num_workers=1, inplace=False, level=0, threshold=127.0,
            erase_blank=False, num_levels=1)

    def source(self, constant=False):
        root = self.root / "volume.zarr"
        values = (np.full((8, 12, 16), 200, dtype=np.uint16) if constant else
                  np.arange(8*12*16, dtype=np.uint16).reshape(8, 12, 16))
        array = create_level_dataset(str(root), "0", values.shape, (4, 4, 4), values.dtype, None)
        array[:] = values
        zarr.open_group(str(root), mode="a").attrs["preserve_note"] = "generated-input"
        (root / "sentinel.txt").write_text("preserve", encoding="utf-8")
        return root, root / "0", values

    def task(self, source, output, erase_blank=False):
        return ThresholdTask(ThresholdConfig(input_zarr=str(source), output_zarr=str(output),
            num_workers=1, level=0, threshold=127.0, erase_blank=erase_blank, num_levels=1))

    def test_cli_selected_array_parent_rejects(self):
        root = self.root / "volume.zarr"
        with self.assertRaisesRegex(SystemExit, "overlaps"):
            ThresholdTask.validate_args(self.args(root / "0", root))

    def test_cli_nested_output_rejects(self):
        source = self.root / "volume.zarr/0"
        with self.assertRaisesRegex(SystemExit, "overlaps"):
            ThresholdTask.validate_args(self.args(source, source / "out.zarr"))

    def test_from_args_rejects_alias_before_constructing_task(self):
        root = self.root / "volume.zarr"
        with self.assertRaisesRegex(SystemExit, "overlaps"):
            ThresholdTask.from_args(self.args(root / "0", root))

    def test_prepare_parent_output_preserves_whole_input_store(self):
        root, source, _ = self.source()
        before = tree(root)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.task(source, root).prepare()
        self.assertEqual(tree(root), before)

    def test_prepare_same_array_tree_output_preserves_store(self):
        root, source, _ = self.source()
        before = tree(root)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.task(source, source).prepare()
        self.assertEqual(tree(root), before)

    def completed(self, suffix, erase_blank=False):
        root, source, values = self.source(constant=erase_blank)
        before = tree(root)
        output = self.root / suffix
        self.task(source, output, erase_blank=erase_blank).run()
        expected = (np.zeros(values.shape, dtype=np.uint8) if erase_blank else
                    (values > 127).astype(np.uint8)*255)
        actual = zarr.open_group(str(output), mode="r")["0"]
        self.assertEqual(actual.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(actual[:], expected)
        self.assertEqual(tree(root), before)

    def test_distinct_selected_array_threshold_values_control(self):
        self.completed("output.zarr")

    def test_shared_prefix_output_values_control(self):
        self.completed("volume.zarr-output")

    def test_erase_blank_distinct_output_control(self):
        self.completed("erased.zarr", erase_blank=True)

    def test_remote_input_argument_spelling_control(self):
        # Argument policy only; no connection/store is opened.
        self.assertIsNone(ThresholdTask.validate_args(
            self.args("s3://bucket/volume.zarr", self.root / "output.zarr")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
