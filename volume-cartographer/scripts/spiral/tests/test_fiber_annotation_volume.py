import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fiber_annotation_volume import (load_fiber_display_geometry,
                                     write_fiber_annotation_volume)


def test_loads_full_display_geometry_and_applies_spiral_scale(tmp_path):
    path = tmp_path / "fiber.json"
    path.write_text(json.dumps({
        "control_points": [[4, 8, 12], [20, 24, 28]],
        "line_points": [[4, 8, 12], [8, 12, 16], [20, 24, 28]],
    }))
    result = load_fiber_display_geometry(
        path, identity="fiber-a", coordinate_scale=0.25)
    assert result["identity"] == "fiber-a"
    np.testing.assert_array_equal(
        result["line_points_xyz"],
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]])
    np.testing.assert_array_equal(
        result["control_points_xyz"], [[1, 2, 3], [5, 6, 7]])


def test_control_points_are_the_display_centerline_fallback(tmp_path):
    path = tmp_path / "fiber.json"
    path.write_text(json.dumps({"control_points": [[1, 2, 3], [4, 5, 6]]}))
    result = load_fiber_display_geometry(path, coordinate_scale=1.0)
    np.testing.assert_array_equal(
        result["line_points_xyz"], result["control_points_xyz"])


def test_builds_deterministic_batches_and_caller_metadata(tmp_path, monkeypatch):
    calls = {}

    class PyramidLevel:
        def __init__(self, shape, scale, translation):
            self.shape = shape
            self.scale = scale
            self.translation = translation

    class Spec:
        def __init__(self, shape, *, pyramid_levels, root_attributes):
            self.shape = shape
            self.pyramid_levels = pyramid_levels
            self.root_attributes = root_attributes

    class Batch:
        def __init__(self, label, coordinates, order, mode, radius):
            self.label = label
            self.coordinates = coordinates
            self.order = order
            self.mode = mode
            self.radius = radius

    fake = types.SimpleNamespace(
        PyramidLevel=PyramidLevel,
        SparseAnnotationVolumeSpec=Spec,
        AnnotationPointBatch=Batch,
        CoordinateOrder=types.SimpleNamespace(XYZ="xyz"),
        GeometryMode=types.SimpleNamespace(
            ORDERED_POLYLINE="polyline", POINTS="points"),
        write_sparse_annotation_volume=lambda spec, batches, destination:
            calls.update(spec=spec, batches=batches, destination=destination),
    )
    monkeypatch.setitem(sys.modules, "vc", types.SimpleNamespace(
        annotation_volume=fake))

    fibers = {
        "z.json": {
            "source_path": "/z.json",
            "line_points_xyz": np.array([[0, 0, 0], [1, 1, 1]]),
            "control_points_xyz": np.array([[0, 0, 0]]),
        },
        "a.json": {
            "source_path": "/a.json",
            "line_points_xyz": np.array([[2, 2, 2], [3, 3, 3]]),
            "control_points_xyz": np.empty((0, 3)),
        },
    }
    destination = tmp_path / "generation"
    meta = write_fiber_annotation_volume(destination, {
        "shape_zyx": [32, 24, 16],
        "pyramid_levels": [{
            "shape_zyx": [16, 12, 8],
            "scale_zyx": [2, 2, 2],
            "translation_zyx": [0, 0, 0],
        }],
        "fiber_coordinate_scale": 0.25,
        "coordinate_identity": {"vc_open_data_coordinate_space": "scroll-a"},
    }, fibers)

    assert meta == str(destination / "meta.json")
    assert calls["spec"].shape == (32, 24, 16)
    assert [batch.label for batch in calls["batches"]] == [1, 2, 2]
    assert [batch.mode for batch in calls["batches"]] == [
        "polyline", "polyline", "points"]
    assert [batch.radius for batch in calls["batches"]] == [1.0, 1.0, 3.0]
    assert [entry["identity"] for entry in
            calls["spec"].root_attributes["fiber_labels"]] == [
                "a.json", "z.json"]
    assert calls["spec"].root_attributes[
        "vc_open_data_coordinate_space"] == "scroll-a"


def test_rejects_nonfinite_display_geometry(tmp_path):
    path = tmp_path / "fiber.json"
    path.write_text('{"control_points": [[1, 2, NaN]]}')
    with pytest.raises(ValueError, match="non-finite"):
        load_fiber_display_geometry(path)
