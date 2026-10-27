"""Reject an output argument before an in-place zero-range task touches data."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr

from tests.zarr_utils import create_v2_array
from vesuvius.image_proc.run.zarr_tasks import __main__ as cli
from vesuvius.image_proc.run.zarr_tasks.tasks.zero_range import (
    ZeroRangeConfig,
    ZeroRangeTask,
)


def tree(path):
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*")) if p.is_file()
    }


class TestZeroRangeOutput(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.zarr"
        self.values = np.arange(1, 65, dtype=np.uint16).reshape(4, 4, 4)
        array = create_v2_array(
            self.source, shape=self.values.shape, chunks=(2, 2, 2), dtype=np.uint16
        )
        array[:] = self.values

    def argv(self, output=None, *, inplace=False):
        result = ["zarr_tasks", str(self.source)]
        if output is not None:
            result.append(str(output))
        result += ["--task", "zero-range", "--z-start", "1", "--z-end", "2", "-n", "1"]
        if inplace:
            result.append("--inplace")
        return result

    def refused_output(self, exists):
        output = self.root / "output.zarr"
        if exists:
            output.mkdir()
            (output / "keep.txt").write_text("existing output", encoding="utf-8")
        before_source, before_output = tree(self.source), tree(output)
        with patch.object(sys, "argv", self.argv(output)):
            with patch("builtins.input", return_value="y") as question:
                with self.assertRaisesRegex(SystemExit, "zero-range operates in place"):
                    cli.main()
                question.assert_not_called()
        self.assertEqual(tree(self.source), before_source)
        self.assertEqual(tree(output), before_output)
        self.assertEqual(output.exists(), exists)

    def test_new_output_is_rejected_before_source_changes(self):
        self.refused_output(False)

    def test_existing_output_is_rejected_before_overwrite_prompt(self):
        self.refused_output(True)

    def test_direct_constructor_rejects_output_without_opening_data(self):
        before = tree(self.source)
        config = ZeroRangeConfig(
            input_zarr=str(self.source), output_zarr=str(self.root / "new.zarr"),
            num_workers=1, z_start=1, z_end=2,
        )
        with self.assertRaisesRegex(ValueError, "output_zarr is not supported"):
            ZeroRangeTask(config)
        self.assertEqual(tree(self.source), before)
        self.assertFalse((self.root / "new.zarr").exists())

    def valid_inplace(self, explicit):
        with patch.object(sys, "argv", self.argv(inplace=explicit)):
            with patch("builtins.input") as question:
                cli.main()
                question.assert_not_called()
        expected = self.values.copy()
        expected[1:3] = 0
        np.testing.assert_array_equal(zarr.open(str(self.source), mode="r")[:], expected)

    def test_explicit_inplace_still_changes_exact_requested_planes(self):
        self.valid_inplace(True)

    def test_existing_implicit_inplace_usage_is_preserved(self):
        self.valid_inplace(False)

    def test_backwards_range_still_rejected(self):
        from argparse import Namespace
        with self.assertRaisesRegex(SystemExit, "must be <="):
            ZeroRangeTask.validate_args(Namespace(output_zarr=None, z_start=2, z_end=1))


if __name__ == "__main__":
    unittest.main()
