"""Merge must reject overlapping input/output trees before modifying files.

Run on PBS only in this research workspace. Full valid-output controls need a
working merge API: zarr 2, or the separately credited PR2244 compatibility fix
applied identically to both arms. The path guard must also work on current main.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig, MergeTask
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def tree(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


class TestMergeInputPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = np.arange(1, 65, dtype=np.uint8).reshape(4, 4, 4)

    def inputs(self, name):
        sandbox = self.root / name
        paths = [sandbox / "inputs" / filename for filename in ("a.zarr", "b.zarr")]
        for index, path in enumerate(paths):
            array = create_level_dataset(str(path), "0", self.values.shape,
                                         (2, 2, 2), self.values.dtype, None)
            array[:] = self.values + index
            (path / "keep.txt").write_bytes(b"preserve original input\n")
        return sandbox, paths

    def task(self, paths, output):
        return MergeTask(MergeConfig(input_zarr=str(paths[0]), input2_zarr=str(paths[1]),
            output_zarr=None if output is None else str(output), num_workers=1, num_levels=1))

    def assert_rejected_unchanged(self, sandbox, paths, output):
        before = tree(sandbox)
        with self.assertRaisesRegex(ValueError, "merge output overlaps an input"):
            self.task(paths, output).prepare()
        self.assertEqual(tree(sandbox), before)

    def test_overlapping_paths_reject_before_input_or_sidecar_changes(self):
        for kind in ("first", "second", "ancestor", "child_first", "child_second", "relative_alias"):
            with self.subTest(kind=kind):
                sandbox, paths = self.inputs(kind)
                outputs = {"first": paths[0], "second": paths[1],
                    "ancestor": paths[0].parent, "child_first": paths[0] / "merged.zarr",
                    "child_second": paths[1] / "merged.zarr",
                    "relative_alias": paths[0].parent / "b.zarr" / ".." / "a.zarr"}
                self.assert_rejected_unchanged(sandbox, paths, outputs[kind])

    def test_symlink_alias_rejects_without_changing_the_target(self):
        sandbox, paths = self.inputs("symlink")
        alias = sandbox / "alias.zarr"
        try:
            alias.symlink_to(paths[0], target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        self.assert_rejected_unchanged(sandbox, paths, alias)
        self.assertTrue(alias.is_symlink())

    def test_cli_validation_rejects_before_the_overwrite_prompt(self):
        sandbox, paths = self.inputs("cli")
        before = tree(sandbox)
        args = argparse.Namespace(input_zarr=str(paths[0]), input2=str(paths[1]), output_zarr=str(paths[0]))
        with self.assertRaisesRegex(SystemExit, "merge output overlaps an input"):
            MergeTask.validate_args(args)
        self.assertEqual(tree(sandbox), before)

    def test_missing_output_rejects_without_writes(self):
        sandbox, paths = self.inputs("missing")
        before = tree(sandbox)
        for output in (None, ""):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "separate output path"):
                    self.task(paths, output).prepare()
                self.assertEqual(tree(sandbox), before)

    def assert_merge(self, paths, output, expected):
        snapshots = [tree(path) for path in paths]
        task = self.task(paths, output)
        task.prepare()
        task._results = [task.process_item(item) for item in task.generate_work_items()]
        task.finalize()
        np.testing.assert_array_equal(zarr.open(str(output / "0"), mode="r")[:], expected)
        self.assertEqual([tree(path) for path in paths], snapshots)

    def test_distinct_sibling_with_shared_prefix_keeps_valid_merge(self):
        _, paths = self.inputs("valid")
        output = paths[0].with_name("a.zarr.merged")
        output.mkdir()
        (output / "stale.txt").write_bytes(b"stale output\n")
        self.assert_merge(paths, output, self.values + 1)
        self.assertFalse((output / "stale.txt").exists())

    def test_reading_the_same_input_twice_is_still_allowed(self):
        sandbox, paths = self.inputs("same_source")
        self.assert_merge([paths[0], paths[0]], sandbox / "out.zarr", self.values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
