import argparse
import json

import numpy as np
import pytest
import zarr
from vesuvius.scripts.view_fiber_presence import (
    clipping_plane_in_layer_data,
    common_shape_edge_width,
    crop_clipping_planes_in_base,
    crop_clipping_planes_in_layer_data,
    open_lazy_crop,
    parse_crop,
    read_line_obj,
    resolve_ome_zarr_level,
    select_base_crop,
    set_common_shape_edge_width,
)


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
g inside_explicit_edges
v 10 20 30
v 11 21 31
v 12 22 32
l 1 2
l 2 3
g outside_multi_index
v 100 200 300
v 101 201 301
v 102 202 302
l 4 5 6
"""
    )

    geometry = read_line_obj(obj, "paths", (0, 0, 0, 50, 50, 50))

    assert geometry.total_groups == 2
    assert len(geometry.paths_zyx) == 1
    np.testing.assert_array_equal(
        geometry.paths_zyx[0],
        np.asarray([[30, 20, 10], [31, 21, 11], [32, 22, 12]], dtype=np.float32),
    )


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
