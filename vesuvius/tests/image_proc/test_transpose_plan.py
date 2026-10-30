"""Prepared PBS-only generated-store controls for the PR2295 successor.

No scans or human annotations. Import scientific modules and run these methods
only in a separately frozen PBS job, never on the local machine/frontend.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import zarr

from vesuvius.image_proc.run.zarr_tasks.base import describe_plan
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import TransposeConfig, TransposeTask


def _tree(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def _physical(ms, level, index):
    values = list(index)
    dataset = next(d for d in ms["datasets"] if d["path"] == str(level))
    for transform in dataset["coordinateTransformations"] + ms.get("coordinateTransformations", []):
        if transform["type"] == "scale":
            values = [a*b for a, b in zip(values, transform["scale"])]
        else:
            values = [a+b for a, b in zip(values, transform["translation"])]
    return dict(zip([a["name"] for a in ms["axes"]], values))


class TransposePlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.values = np.arange(8*12*16, dtype=np.uint16).reshape(8, 12, 16)

    def source(self, *, names="zyx", translation=False, global_frame=False,
               signed=False, invalid=False, nested=False):
        path = self.root / "source.zarr"
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        perm = ["zyx".index(name) for name in names]
        for level in range(2):
            group.create_array(str(level), data=self.values[::2**level, ::2**level, ::2**level].transpose(perm),
                               chunks=(4, 4, 4))
        pitch = {"z": -3.0 if signed else 3.0, "y": 2.0, "x": 1.0}
        offset = {"z": 9.0, "y": 8.0, "x": 7.0}
        ms = {"version": "0.4", "name": "generated-contract",
              "axes": [{"name": name, "type": "space", "unit": "micrometer"} for name in names],
              "datasets": []}
        for level in range(2):
            scales = [pitch[name]*2**level for name in names]
            transforms = [{"type": "scale", "scale": scales[:2] if invalid else scales}]
            if translation:
                transforms.append({"type": "translation", "translation": [offset[name] for name in names]})
            ms["datasets"].append({"path": str(level), "coordinateTransformations": transforms})
        if global_frame:
            ms["coordinateTransformations"] = [
                {"type": "scale", "scale": [0.5, 2.0, -1.0]},
                {"type": "translation", "translation": [19.0, 20.0, 21.0]}]
        group.attrs["ome" if nested else "multiscales"] = {"multiscales": [ms]} if nested else [ms]
        return path, group, ms

    def args(self, source, output, order, level):
        return SimpleNamespace(input_zarr=str(source), output_zarr=str(output),
            task="transpose", inplace=False, level=level, num_workers=1,
            transpose_order=order, num_levels=1)

    def run_case(self, source, group, ms, *, order="xzy", level=0,
                 perm=(2, 0, 1), selected=False, physical=True):
        selected_values = group[str(level)][:]
        input_path = source / str(level) if selected else source
        config_level = None if selected else level
        output = self.root / "output.zarr"
        args = self.args(input_path, output, order, config_level)
        before = _tree(self.root)
        lines = describe_plan(TransposeTask, args)
        expected = selected_values.transpose(perm)
        self.assertTrue(any("level 0 shape " + str(expected.shape) in line for line in lines))
        self.assertEqual(_tree(self.root), before)
        task = TransposeTask(TransposeConfig(input_zarr=str(input_path), output_zarr=str(output),
            num_workers=1, level=config_level, transpose_order=order, num_levels=1))
        task.run()
        result = zarr.open_group(str(output), mode="r")
        np.testing.assert_array_equal(result["0"][:], expected)
        self.assertEqual(_tree(source), {k.removeprefix(source.name + "/"): v for k, v in before.items()
                                       if k.startswith(source.name + "/")})
        out_ms = result.attrs["multiscales"][0]
        self.assertEqual([a["name"] for a in out_ms["axes"]], list(order))
        if physical:
            self.assertEqual({a["name"]: a["unit"] for a in ms["axes"]},
                             {a["name"]: a["unit"] for a in out_ms["axes"]})
            points = [tuple(0 for _ in selected_values.shape),
                      tuple(s-1 for s in selected_values.shape),
                      tuple(min(i+1, s-1) for i, s in enumerate(selected_values.shape))]
            names = [a["name"] for a in ms["axes"]]
            for point in points:
                by_name = dict(zip(names, point))
                actual_point = [by_name[name] for name in order]
                self.assertEqual(_physical(out_ms, 0, actual_point), _physical(ms, level, point))
        return result

    def test_preoriented_plan_and_values_follow_named_axes(self):
        source, group, ms = self.source(names="xzy")
        self.run_case(source, group, ms, order="zyx", perm=(1, 2, 0))

    def test_dataset_translation_survives_plan_and_execution(self):
        self.run_case(*self.source(translation=True))

    def test_global_frame_and_signed_scale_survive_execution(self):
        self.run_case(*self.source(translation=True, global_frame=True, signed=True))

    def test_selected_array_parent_frame_matches_plan(self):
        source, group, ms = self.source(names="xzy", translation=True)
        self.run_case(source, group, ms, order="zyx", level=1, perm=(1, 2, 0), selected=True)

    def test_malformed_frame_rejects_plan_and_prepare_without_writes(self):
        source, _, _ = self.source(invalid=True)
        output = self.root / "existing-output.zarr"
        output.mkdir()
        (output / "sentinel.txt").write_text("preserve", encoding="utf-8")
        before = _tree(self.root)
        with self.assertRaises(ValueError):
            describe_plan(TransposeTask, self.args(source, output, "xzy", 0))
        self.assertEqual(_tree(self.root), before)
        task = TransposeTask(TransposeConfig(input_zarr=str(source), output_zarr=str(output),
            num_workers=1, level=0, transpose_order="xzy", num_levels=1))
        with self.assertRaises(ValueError):
            task.prepare()
        self.assertEqual(_tree(self.root), before)

    def test_zyx_pitch_units_and_values_control(self):
        self.run_case(*self.source())

    def test_selected_group_level_pitch_and_values_control(self):
        self.run_case(*self.source(), level=1)

    def test_uncalibrated_bare_array_legacy_control(self):
        path = self.root / "bare.zarr"
        zarr.open_array(str(path), mode="w", shape=self.values.shape, chunks=(4, 4, 4),
                        dtype="uint16", zarr_format=2)[:] = self.values
        # Supply only an input accessor to the common assertion helper. The
        # task and planner themselves still open the real bare store.
        holder = {"0": zarr.open_array(str(path), mode="r")}
        self.run_case(path, holder, None, physical=False)

    def test_nested_ome_pitch_fallback_legacy_control(self):
        self.run_case(*self.source(nested=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
