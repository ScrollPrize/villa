import argparse
import json

import dask.array as da
import numpy as np
import pytest
import zarr

from vesuvius.scripts.view_fiber_presence import (
    CropSelection,
    clipping_plane_in_layer_data,
    common_shape_edge_width,
    crop_clipping_planes_in_base,
    crop_clipping_planes_in_layer_data,
    fiberlet_colormap_names,
    fiberlet_quality_colormap_spec,
    load_fiber_replay_bundle,
    mask_presence_by_distance,
    open_lazy_crop,
    parse_crop,
    read_anchor_cell_obj,
    read_line_obj,
    replay_distance_transform_base,
    resolve_ome_zarr_level,
    select_base_crop,
    set_common_shape_edge_width,
)


def _fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _write_nonfailure_replay(tmp_path):
    generation = tmp_path / "runs" / "abc" / "replay"
    generation.mkdir(parents=True)
    reference = (
        b"# vc_fiber_replay_reference version 1\n"
        b"v 0 0 0\nv 8 0 0\nl 1 2\n"
    )
    trace = (
        b"# vc_fiber_replay_trace version 1\n"
        b"v 0 0 0\nv 4 0 0\nl 1 2\n"
    )
    (generation / "reference.obj").write_bytes(reference)
    (generation / "trace.obj").write_bytes(trace)
    bundle = {
        "format": "vc_fiber_replay",
        "version": 1,
        "coordinates": {
            "position_order": "XYZ",
            "position_space": "base_volume",
            "distance_unit": "base_voxels",
        },
        "sources": {
            "fiber_manifest": "/fiber.json",
            "fiber_manifest_content_hash": "fnv1a64:1",
            "normal_manifest": "/normal.json",
            "normal_manifest_content_hash": "fnv1a64:2",
            "fiber_json": "/reference.json",
            "fiber_json_content_hash": "fnv1a64:3",
        },
        "bindings": {
            "trace": {
                "mode": "trace_options",
                "trace_to_base_scale": 2.0,
                "prediction_to_base_scale": 2.0,
                "prediction_spacing_trace_voxels": 1.0,
            },
            "prediction": {
                "mode": "canonical_stored_grid",
                "prediction_to_base_scale": 2.0,
                "prediction_shape_zyx": [4, 4, 4],
            }
        },
        "trace_config": {
            "requested": {"beam_width": 8, "beam_lookahead_steps": 2},
            "effective": {"beam_width": 1, "beam_lookahead_steps": 1},
        },
        "status": "no_failure",
        "termination_reason": "reference_end",
        "reference_points_base_xyz": [[0, 0, 0], [8, 0, 0]],
        "trace_points_base_xyz": [[0, 0, 0], [4, 0, 0]],
        "trace_cumulative_losses": [0.0, 1.0],
        "matching": {
            "failure_threshold_base_voxels": 20.0,
            "refine_steps": 1.0,
            "records": [],
        },
        "postroll": {
            "requested_steps": 100,
            "completed_steps": 0,
            "maximum_trace_steps": 10,
        },
        "failure_trace_point_index": None,
        "failure_reference_arc_base": None,
        "tube": None,
        "volume_crop_base_xyzwhd": None,
        "artifacts": {
            "replay/reference.obj": {
                "path": "runs/abc/replay/reference.obj",
                "content_hash": _fnv1a64(reference),
            },
            "replay/trace.obj": {
                "path": "runs/abc/replay/trace.obj",
                "content_hash": _fnv1a64(trace),
            },
        },
    }
    path = tmp_path / "fiber_replay.json"
    path.write_text(json.dumps(bundle))
    return path


def test_loads_strict_nonfailure_replay_bundle(tmp_path):
    replay = load_fiber_replay_bundle(_write_nonfailure_replay(tmp_path))

    assert replay.status == "no_failure"
    assert replay.crop_xyzwhd == (0, 0, 0, 9, 1, 1)
    assert replay.prediction_shape_zyx == (4, 4, 4)
    np.testing.assert_array_equal(replay.reference_zyx, [[0, 0, 0], [0, 0, 8]])
    assert replay.anchors_obj is None
    assert replay.anchor_cells_obj is None
    assert replay.failure_zyx is None
    assert replay.tube_radius_base_voxels is None


def test_reads_anchor_cell_centers_and_accepted_offsets(tmp_path):
    path = tmp_path / "anchor_cells.obj"
    path.write_text(
        "# vc_fiberlet_anchor_cells version 1\n"
        "g cell_0_0_0\n"
        "v 1 2 3\n"
        "p 1\n"
        "v 4 5 6\n"
        "l 1 2\n"
        "g cell_0_0_1\n"
        "v 7 8 9\n"
        "p 3\n"
    )

    geometry = read_anchor_cell_obj(path)

    np.testing.assert_array_equal(geometry.centers_zyx, [[3, 2, 1], [9, 8, 7]])
    assert len(geometry.displacements_zyx) == 1
    np.testing.assert_array_equal(
        geometry.displacements_zyx[0], [[3, 2, 1], [6, 5, 4]]
    )


def test_replay_distance_mask_uses_reference_and_trace_in_base_voxels():
    selection = CropSelection(
        requested_base_xyzwhd=(0, 0, 0, 10, 10, 10),
        slices_zyx=(slice(0, 5), slice(0, 5), slice(0, 5)),
        origin_base_zyx=(0.0, 0.0, 0.0),
        shape_zyx=(5, 5, 5),
    )
    distance = replay_distance_transform_base(
        np.asarray([[0.0, 4.0, 0.0], [0.0, 4.0, 8.0]]),
        np.asarray([[0.0, 8.0, 0.0], [0.0, 8.0, 8.0]]),
        selection,
        (2.0, 2.0, 2.0),
    )

    np.testing.assert_array_equal(distance[0, 2], np.zeros(5))
    np.testing.assert_array_equal(distance[0, 4], np.zeros(5))
    assert distance[0, 3, 2] == pytest.approx(2.0)
    presence = da.from_array(np.ones((5, 5, 5), dtype=np.float32), chunks=(2, 2, 2))
    distance_data = da.from_array(distance, chunks=presence.chunks)
    masked = mask_presence_by_distance(presence, distance_data, 1.0).compute()
    np.testing.assert_array_equal(masked[0, 2], np.ones(5))
    np.testing.assert_array_equal(masked[0, 4], np.ones(5))
    assert masked[0, 3, 2] == 0.0


def test_replay_bundle_rejects_hash_mismatch(tmp_path):
    path = _write_nonfailure_replay(tmp_path)
    (tmp_path / "runs" / "abc" / "replay" / "trace.obj").write_text("changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_fiber_replay_bundle(path)


def test_replay_bundle_rejects_lexical_escape(tmp_path):
    path = _write_nonfailure_replay(tmp_path)
    bundle = json.loads(path.read_text())
    bundle["artifacts"]["replay/trace.obj"]["path"] = "../trace.obj"
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="escapes"):
        load_fiber_replay_bundle(path)


def test_replay_bundle_rejects_symlink_escape(tmp_path):
    path = _write_nonfailure_replay(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.obj"
    outside.write_text("outside")
    symlink = tmp_path / "runs" / "abc" / "replay" / "escape.obj"
    symlink.symlink_to(outside)
    bundle = json.loads(path.read_text())
    bundle["artifacts"]["replay/trace.obj"] = {
        "path": "runs/abc/replay/escape.obj",
        "content_hash": _fnv1a64(b"outside"),
    }
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="escapes"):
        load_fiber_replay_bundle(path)
    outside.unlink()


def test_fiberlet_quality_colormap_spec_is_red_yellow_green():
    name, colors, controls = fiberlet_quality_colormap_spec()

    assert name == "red-yellow-green"
    np.testing.assert_allclose(
        colors,
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_allclose(controls, [0.0, 0.5, 1.0])


def test_fiberlet_colormap_names_are_custom_first_sorted_and_unique():
    assert fiberlet_colormap_names(
        ["viridis", "magma", "viridis", "red-yellow-green"]
    ) == ("red-yellow-green", "magma", "viridis")


@pytest.mark.parametrize(
    ("edge_width", "expected"),
    [([2.0, 2.0, 2.0], 2.0), (np.asarray([0.25]), 0.25), (1.5, 1.5)],
)
def test_reads_common_width_from_napari_per_shape_values(edge_width, expected):
    class Layer:
        pass

    layer = Layer()
    layer.edge_width = edge_width

    assert common_shape_edge_width(layer) == expected


def test_common_width_uses_default_for_missing_layer_or_empty_shapes():
    class Layer:
        def __init__(self):
            self.edge_width = []

    assert common_shape_edge_width(None) == 2.0
    assert common_shape_edge_width(Layer()) == 2.0


def test_sets_common_width_and_emits_napari_edge_width_event():
    class Event:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1

    class Events:
        def __init__(self):
            self.edge_width = Event()

    class Layer:
        def __init__(self):
            self.edge_width = [2.0, 2.0]
            self.events = Events()

    layer = Layer()
    set_common_shape_edge_width(layer, 0.25)

    assert layer.edge_width == 0.25
    assert layer.events.edge_width.calls == 1


def test_clipping_plane_is_transformed_from_base_to_layer_data():
    class Layer:
        @staticmethod
        def world_to_data(world):
            return (np.asarray(world) - [100.0, 200.0, 300.0]) / [8.0, 4.0, 2.0]

    plane = clipping_plane_in_layer_data(
        Layer(),
        position_base_zyx=(132.0, 220.0, 306.0),
        normal_base_zyx=(0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(plane["position"], [4.0, 5.0, 3.0])
    np.testing.assert_allclose(plane["normal"], [0.0, 1.0, 0.0])
    assert plane["enabled"] is True


def test_base_crop_planes_remain_in_world_coordinates_for_volume_clipping():
    planes = crop_clipping_planes_in_base(
        lower_base_zyx=(10_000.0, 20_000.0, 30_000.0),
        upper_base_zyx=(10_100.0, 20_200.0, 30_300.0),
    )

    assert len(planes) == 6
    np.testing.assert_allclose(planes[0]["position"], [10_000, 20_000, 30_000])
    np.testing.assert_allclose(planes[1]["position"], [10_100, 20_200, 30_300])
    np.testing.assert_allclose(planes[0]["normal"], [1, 0, 0])
    np.testing.assert_allclose(planes[1]["normal"], [-1, 0, 0])


def test_crop_clipping_planes_bound_all_six_sides_in_layer_data():
    class Layer:
        @staticmethod
        def world_to_data(world):
            return (np.asarray(world) - [100.0, 200.0, 300.0]) / [8.0, 4.0, 2.0]

    planes = crop_clipping_planes_in_layer_data(
        Layer(),
        lower_base_zyx=(108.0, 208.0, 304.0),
        upper_base_zyx=(132.0, 220.0, 312.0),
    )

    assert len(planes) == 6
    np.testing.assert_allclose(
        [plane["position"] for plane in planes],
        [
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
        ],
    )
    np.testing.assert_allclose(
        [plane["normal"] for plane in planes],
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )


def test_reads_and_crops_grouped_fiberlet_obj(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
# trace_quality_population successful_scored_fiberlets
# trace_loss_density_unit prediction_voxel
# trace_quality_formula inverse_min_max_low_loss_is_one
# trace_quality_count 2
# trace_loss_density_min 1
# trace_loss_density_max 3
g inside_explicit_edges
# trace_loss_total 4
# trace_loss_per_prediction_voxel 1
# trace_quality_relative 1
v 10 20 30
v 11 21 31
v 12 22 32
l 1 2
l 2 3
g outside_explicit_edges
# trace_loss_total 12
# trace_loss_per_prediction_voxel 3
# trace_quality_relative 0
v 100 200 300
v 101 201 301
v 102 202 302
l 4 5
l 5 6
"""
    )
    geometry = read_line_obj(obj, "paths", (0, 0, 0, 50, 50, 50))

    assert geometry.total_groups == 2
    assert len(geometry.paths_zyx) == 1
    assert geometry.trace_loss_total == [4.0]
    assert geometry.loss_per_prediction_voxel == [1.0]
    assert geometry.relative_quality == [1.0]
    np.testing.assert_array_equal(
        geometry.paths_zyx[0],
        np.asarray([[30, 20, 10], [31, 21, 11], [32, 22, 12]], dtype=np.float32),
    )


def test_rejects_obsolete_fiberlet_material_records(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
mtllib fiberlets.mtl
"""
    )

    with pytest.raises(ValueError, match="unsupported OBJ record 'mtllib'"):
        read_line_obj(obj, "paths", (0, 0, 0, 10, 10, 10))


def test_fiberlet_crop_keeps_geometry_and_metrics_aligned(tmp_path):
    groups = [
        ("outside_first", 100.0, 4.0, 0.0),
        ("inside_first", 10.0, 2.0, 2.0 / 3.0),
        ("outside_between", 200.0, 3.0, 1.0 / 3.0),
        ("partially_inside", -1.0, 1.0, 1.0),
    ]
    obj_lines = [
        "# vc_fiberlets version 1",
        "# trace_quality_population successful_scored_fiberlets",
        "# trace_loss_density_unit prediction_voxel",
        "# trace_quality_formula inverse_min_max_low_loss_is_one",
        "# trace_quality_count 4",
        "# trace_loss_density_min 1",
        "# trace_loss_density_max 4",
    ]
    vertex = 1
    for name, x, density, quality in groups:
        obj_lines.extend(
            [
                f"g {name}",
                f"# trace_loss_total {density * 2}",
                f"# trace_loss_per_prediction_voxel {density}",
                f"# trace_quality_relative {quality!r}",
                f"v {x} 1 1",
                f"v {x + 2} 1 1",
                f"l {vertex} {vertex + 1}",
            ]
        )
        vertex += 2
    (tmp_path / "fiberlets.obj").write_text("\n".join(obj_lines) + "\n")

    geometry = read_line_obj(
        tmp_path / "fiberlets.obj", "paths", (0, 0, 0, 50, 50, 50)
    )

    assert geometry.total_groups == 4
    assert len(geometry.paths_zyx) == 2
    assert geometry.trace_loss_total == [4.0, 2.0]
    assert geometry.loss_per_prediction_voxel == [2.0, 1.0]
    np.testing.assert_allclose(geometry.relative_quality, [2.0 / 3.0, 1.0])


def test_rejects_disconnected_line_records(tmp_path):
    obj = tmp_path / "anchors.obj"
    obj.write_text(
        """# vc_fiberlet_anchors version 1
g broken
v 0 0 0
v 1 1 1
v 2 2 2
v 3 3 3
l 1 2
l 3 4
"""
    )

    with pytest.raises(ValueError, match="do not form one ordered path"):
        read_line_obj(obj, "anchors", (0, 0, 0, 10, 10, 10))


def test_anchor_geometry_has_no_fiberlet_metrics(tmp_path):
    obj = tmp_path / "anchors.obj"
    obj.write_text(
        """# vc_fiberlet_anchors version 1
g anchor
v 0 0 0
v 1 0 0
l 1 2
"""
    )
    geometry = read_line_obj(obj, "anchors", (0, 0, 0, 10, 10, 10))
    assert geometry.trace_loss_total == []
    assert geometry.loss_per_prediction_voxel == []
    assert geometry.relative_quality == []


def test_empty_fiberlet_geometry_has_no_metrics(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
# trace_quality_population successful_scored_fiberlets
# trace_loss_density_unit prediction_voxel
# trace_quality_formula inverse_min_max_low_loss_is_one
# trace_quality_count 0
# trace_loss_density_min none
# trace_loss_density_max none
"""
    )
    geometry = read_line_obj(obj, "paths", (0, 0, 0, 10, 10, 10))

    assert geometry.paths_zyx == []
    assert geometry.trace_loss_total == []
    assert geometry.loss_per_prediction_voxel == []
    assert geometry.relative_quality == []


def make_presence_pyramid(tmp_path):
    root = tmp_path / "presence.ome.zarr"
    root.mkdir()
    (root / ".zgroup").write_text('{"zarr_format": 2}')
    (root / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "3",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [8.0, 8.0, 8.0]}
                                ],
                            },
                            {
                                "path": "4",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [16.0, 16.0, 16.0]}
                                ],
                            },
                        ],
                    }
                ]
            }
        )
    )
    level3 = zarr.open_array(
        root / "3", mode="w", shape=(5, 6, 7), chunks=(2, 3, 4), dtype="u1"
    )
    level3[:] = np.arange(level3.size, dtype=np.uint8).reshape(level3.shape)
    zarr.open_array(root / "4", mode="w", shape=(3, 3, 4), chunks=(2, 2, 2), dtype="u1")
    return root


def test_resolves_finest_level_and_direct_array(tmp_path):
    root = make_presence_pyramid(tmp_path)

    finest = resolve_ome_zarr_level(root)
    direct = resolve_ome_zarr_level(root / "4")

    assert finest.path == "3"
    assert finest.scale_zyx == (8.0, 8.0, 8.0)
    assert direct.path == "4"
    assert direct.scale_zyx == (16.0, 16.0, 16.0)


def test_base_crop_is_clipped_and_keeps_world_origin(tmp_path):
    root = make_presence_pyramid(tmp_path)
    level = resolve_ome_zarr_level(root)

    selection = select_base_crop((5, 6, 7), level, (9, 7, 1, 40, 30, 50))

    assert selection.slices_zyx == (slice(1, 5), slice(1, 5), slice(2, 7))
    assert selection.shape_zyx == (4, 4, 5)
    assert selection.origin_base_zyx == (8.0, 8.0, 16.0)


def test_lazy_crop_does_not_require_napari(tmp_path):
    root = make_presence_pyramid(tmp_path)
    level = resolve_ome_zarr_level(root, "3")

    data, selection = open_lazy_crop(level, (8, 8, 8, 16, 16, 16))

    assert data.shape == (2, 2, 2)
    assert selection.origin_base_zyx == (8.0, 8.0, 8.0)
    np.testing.assert_array_equal(
        data.compute(), zarr.open_array(root / "3", mode="r")[1:3, 1:3, 1:3]
    )


@pytest.mark.parametrize(
    "value",
    ["1,2,3,4,5", "1,2,3,0,5,6", "-1,2,3,4,5,6", "one,2,3,4,5,6"],
)
def test_parse_crop_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_crop(value)
