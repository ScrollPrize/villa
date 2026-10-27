"""Standalone transpose-path controls; no coordinate/routing patch is common.

Valid cases use explicit array paths so the separate group-worker routing
proposal is outside this test's contract.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest

import numpy as np
import zarr
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import TransposeConfig, TransposeTask


def tree(root):
    return {p.relative_to(root).as_posix(): ("link", str(p.readlink())) if p.is_symlink()
            else ("file", hashlib.sha256(p.read_bytes()).hexdigest()) if p.is_file()
            else ("directory",) for p in root.rglob("*")}


class TransposeStandaloneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "in.zarr"
        self.values = np.arange(8*12*16, dtype=np.uint16).reshape(8, 12, 16)
        group = zarr.open_group(str(self.source), mode="w", zarr_format=2)
        group.create_array("0", data=self.values, chunks=(4, 4, 4))
        (self.source / "keep.txt").write_text("input marker", encoding="utf-8")

    def task(self, source, output):
        return TransposeTask(TransposeConfig(input_zarr=str(source), output_zarr=str(output),
                             num_workers=1, level=0, num_levels=1, transpose_order="xzy"))

    def reject(self, source, output):
        before = tree(self.root)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.task(source, output).prepare()
        self.assertEqual(tree(self.root), before)

    def test_same_root_rejects_before_writes(self):
        self.reject(self.source, self.source)

    def test_parent_output_rejects_before_writes(self):
        self.reject(self.source / "0", self.source)

    def test_nested_output_rejects_before_writes(self):
        self.reject(self.source, self.source / "nested.zarr")

    def test_symlink_alias_rejects_before_writes(self):
        alias = self.root / "alias.zarr"
        alias.symlink_to(self.source, target_is_directory=True)
        self.reject(self.source, alias)

    def completed(self, output):
        before = tree(self.source)
        self.task(self.source / "0", output).run()
        result = np.asarray(zarr.open(str(output / "0"), mode="r")[:])
        self.assertTrue(np.array_equal(result, self.values.transpose(2, 0, 1)))
        self.assertEqual(result.dtype, self.values.dtype)
        self.assertEqual(tree(self.source), before)

    def test_distinct_selected_array_values(self):
        self.completed(self.root / "out.zarr")

    def test_shared_prefix_selected_array_values(self):
        self.completed(self.root / "in.zarr-out")

    def test_remote_argument_spelling(self):
        TransposeTask.validate_args(SimpleNamespace(input_zarr="https://example.invalid/input.zarr",
                                    output_zarr=str(self.root / "out.zarr"), transpose_order="xzy"))

    def test_invalid_permutation(self):
        with self.assertRaises(SystemExit):
            TransposeTask.validate_args(SimpleNamespace(input_zarr=str(self.source),
                                        output_zarr=str(self.root / "out.zarr"), transpose_order="zzx"))
