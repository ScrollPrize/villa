"""Prepared merge-grid controls; run only on PBS in this research workspace.

Full task execution under Zarr3 requires the separately credited PR2244 API fix
in both arms. The path-safety guard0029 must also be common in the future job.
These generated stores are engineering controls, not real scroll validation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig, MergeTask
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def header(scale=(9.0, 6.0, 3.0), translation=(12.0, 24.0, 36.0), unit="micrometer"):
    axes = [{"name": name, "type": "space"} for name in ("z", "y", "x")]
    if unit is not None:
        for axis in axes:
            axis["unit"] = unit
    return {"multiscales": [{"version": "0.4", "axes": axes, "datasets": [{"path": "0",
        "coordinateTransformations": [{"type": "scale", "scale": list(scale)},
                                      {"type": "translation", "translation": list(translation)}]}]}]}


def tree(path):
    return {item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(path.rglob("*")) if item.is_file()}


class MergeGridTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.values = np.arange(1, 65, dtype=np.uint16).reshape(4, 4, 4)

    def inputs(self, first, second):
        paths = [self.root / "first.zarr", self.root / "second.zarr"]
        for index, (path, attrs) in enumerate(zip(paths, (first, second))):
            array = create_level_dataset(str(path), "0", (4, 4, 4), (2, 2, 2), self.values.dtype, None)
            array[:] = self.values + index
            if attrs is not None:
                zarr.open_group(str(path), mode="a").attrs.update(attrs)
            (path / "keep.txt").write_text("preserve input\n")
        return paths

    def task(self, paths, levels=1):
        return MergeTask(MergeConfig(input_zarr=str(paths[0]), input2_zarr=str(paths[1]),
            output_zarr=str(self.root / "output.zarr"), num_workers=1, num_levels=levels))

    def run_merge(self, first, second, levels=1):
        paths = self.inputs(first, second)
        before = [tree(path) for path in paths]
        task = self.task(paths, levels)
        task.prepare()
        task._results = [task.process_item(item) for item in task.generate_work_items()]
        task.finalize()
        output = self.root / "output.zarr"
        np.testing.assert_array_equal(zarr.open(str(output / "0"), mode="r")[:], self.values + 1)
        self.assertEqual([tree(path) for path in paths], before)
        return json.loads((output / ".zattrs").read_text())["multiscales"][0]

    def reject(self, first, second):
        paths = self.inputs(first, second)
        output = self.root / "output.zarr"
        output.mkdir()
        (output / "keep.txt").write_text("existing output must remain\n")
        before = tree(self.root)
        with self.assertRaisesRegex(ValueError, "merge"):
            self.task(paths).prepare()
        self.assertEqual(tree(self.root), before)

    def test_preserves_physical_level0_grid(self):
        attrs = header()
        metadata = self.run_merge(attrs, attrs)
        self.assertEqual(metadata["axes"], attrs["multiscales"][0]["axes"])
        self.assertEqual(metadata["datasets"][0]["coordinateTransformations"],
                         attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"])

    def test_pyramid_scales_preserve_pitch_and_origin(self):
        metadata = self.run_merge(header(), header(), levels=3)
        for dataset, expected in zip(metadata["datasets"], ([9., 6., 3.], [18., 12., 6.], [36., 24., 12.])):
            self.assertEqual(dataset["coordinateTransformations"], [
                {"type": "scale", "scale": expected},
                {"type": "translation", "translation": [12., 24., 36.]}])

    def test_composes_dataset_and_global_transforms(self):
        first = header((2., 3., 4.), (5., 7., 11.))
        first["multiscales"][0]["coordinateTransformations"] = [
            {"type": "scale", "scale": [3., 4., 5.]},
            {"type": "translation", "translation": [13., 17., 19.]}]
        second = header((6., 12., 20.), (28., 45., 74.))
        metadata = self.run_merge(first, second)
        self.assertEqual(metadata["datasets"][0]["coordinateTransformations"], [
            {"type": "scale", "scale": [6., 12., 20.]},
            {"type": "translation", "translation": [28., 45., 74.]}])

    def test_unitless_legacy_grid_stays_unitless(self):
        metadata = self.run_merge(None, None)
        self.assertTrue(all("unit" not in axis for axis in metadata["axes"]))

    def test_unitless_declared_grid_stays_unitless(self):
        metadata = self.run_merge(header(unit=None), header(unit=None))
        self.assertTrue(all("unit" not in axis for axis in metadata["axes"]))

    def test_non_micrometer_units_are_preserved(self):
        metadata = self.run_merge(header(unit="nanometer"), header(unit="nanometer"))
        self.assertTrue(all(axis.get("unit") == "nanometer" for axis in metadata["axes"]))

    def test_shifted_origin_rejects_before_output_writes(self):
        self.reject(header(), header(translation=(12., 24., 39.)))

    def test_large_origin_shift_rejects_before_output_writes(self):
        self.reject(header(scale=(1., 1., 1.), translation=(1e9, 0., 0.)),
                    header(scale=(1., 1., 1.), translation=(1e9 + .5, 0., 0.)))

    def test_pitch_mismatch_rejects_before_output_writes(self):
        self.reject(header(), header(scale=(18., 6., 3.)))

    def test_axis_order_mismatch_rejects_before_output_writes(self):
        second = header()
        second["multiscales"][0]["axes"].reverse()
        self.reject(header(), second)

    def test_unit_mismatch_rejects_before_output_writes(self):
        self.reject(header(), header(unit="nanometer"))

    def test_partly_declared_pair_rejects_before_output_writes(self):
        self.reject(header(), None)

    def test_zero_pitch_rejects_before_output_writes(self):
        self.reject(header(scale=(0., 6., 3.)), header())

    def test_boolean_pitch_rejects_before_output_writes(self):
        self.reject(header(scale=(True, 6., 3.)), header())

    def test_missing_transform_rejects_before_output_writes(self):
        first = header()
        del first["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        self.reject(first, header())

    def test_duplicate_level0_grid_rejects_before_output_writes(self):
        first = header()
        first["multiscales"].append(deepcopy(first["multiscales"][0]))
        self.reject(first, header())

    def test_wrong_transform_order_rejects_before_output_writes(self):
        first = header()
        first["multiscales"][0]["datasets"][0]["coordinateTransformations"].reverse()
        self.reject(first, header())

    def test_same_input_twice_keeps_values_control(self):
        paths = self.inputs(None, None)
        before = tree(paths[0])
        task = self.task([paths[0], paths[0]])
        task.prepare()
        task._results = [task.process_item(item) for item in task.generate_work_items()]
        task.finalize()
        np.testing.assert_array_equal(zarr.open(str(self.root / "output.zarr/0"), mode="r")[:], self.values)
        self.assertEqual(tree(paths[0]), before)

    def test_small_roundoff_does_not_reject_matching_grid_control(self):
        first, second = header(), header()
        second["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]["translation"][0] += 1e-10
        self.run_merge(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
