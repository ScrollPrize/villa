from pathlib import Path

import numpy as np
import pytest
import vesuvius.scripts.view_fiber_windings as viewer_module
from vesuvius.scripts.ordered_polyline_obj import read_ordered_polyline_obj
from vesuvius.scripts.view_fiber_windings import (
    ReferenceGeometry,
    WindingArtifact,
    WindingGeometry,
    WindingLayerKey,
    add_reference_layer,
    add_winding_layers,
    animation_interval_milliseconds,
    build_parser,
    complete_winding_layer_keys,
    discover_reference_artifact,
    discover_winding_artifacts,
    format_visible_windings,
    load_reference_geometry,
    navigable_windings,
    nonempty_layer_keys,
    read_reference_artifact,
    read_winding_artifact,
    rotate_visible_winding_mask,
    rotate_winding_layer_visibility,
    visible_winding_layers,
    winding_layer_color,
    winding_layer_colors,
)

HEADERS = {
    "h": "VC3D Fiberlet crop traces: BP horizontal argmax",
    "v": "VC3D Fiberlet crop traces: BP vertical argmax",
    "err": "VC3D Fiberlet crop traces: BP error/Mixed argmax",
    "tie": "VC3D Fiberlet crop traces: BP exact argmax tie",
}


def _write_state(
    base: Path,
    winding: int,
    state: str,
    body: str = "",
) -> Path:
    path = base.parent / f"{base.name}_w_{winding}_{state}.obj"
    path.write_text(f"# {HEADERS[state]}\n{body}")
    return path


def _write_quartet(base: Path, winding: int, *, h_body: str = "") -> None:
    for state in HEADERS:
        _write_state(base, winding, state, h_body if state == "h" else "")


def _write_reference(base: Path, body: str) -> Path:
    path = base.parent / f"{base.name}_reference.obj"
    path.write_text(f"# VC3D tagged reference fibers\n{body}")
    return path


def test_shared_obj_reader_reconstructs_global_indexed_objects_and_singleton(tmp_path):
    path = tmp_path / "lines.obj"
    path.write_text(
        "# header\n"
        "o first\n"
        "v 1 2 3\n"
        "v 4 5 6\n"
        "v 7 8 9\n"
        "l 1 2\n"
        "l 2 3\n"
        "o second\n"
        "v 10 11 12\n"
        "p 4\n"
    )
    parsed = read_ordered_polyline_obj(
        path,
        container_records=("o",),
        allow_singletons=True,
        require_segment_lines=True,
    )
    assert tuple(comment.text for comment in parsed.preamble_comments) == ("header",)
    assert [group.name for group in parsed.groups] == ["first", "second"]
    np.testing.assert_array_equal(
        parsed.groups[0].points_xyz,
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )
    assert not parsed.groups[0].singleton
    assert parsed.groups[1].singleton


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "o a\nv 0 0 0\nv 1 0 0\nl 1 2\no b\nv 2 0 0\nv 3 0 0\nl 2 4\n",
            "does not reference a vertex",
        ),
        ("o a\nv 0 0 0\nv 1 0 0\nv 2 0 0\nl 1 2\n", "unused or disconnected"),
        ("o a\nv 0 0 0\nv 1 0 0\nl 1 2\no a\n", "duplicate container"),
        ("o a\nv 0 0 0\nv 1 0 0\nf 1 2\n", "unsupported OBJ record"),
        (
            "o a\nv 0 0 0\nv 1 0 0\nv 2 0 0\nv 3 0 0\n"
            + "l 1 2\nl 2 3\nl 2 4\n",
            "do not form one ordered path",
        ),
        (
            "o a\nv 0 0 0\nv 1 0 0\nv 2 0 0\nv 3 0 0\n"
            + "l 1 2\nl 3 4\n",
            "do not form one ordered path",
        ),
        ("o a\nv 0 0 0\nv 1 0 0\nl 1 2\nl 2 1\n", "branches or cycles"),
    ],
)
def test_shared_obj_reader_rejects_invalid_topology(tmp_path, body, message):
    path = tmp_path / "invalid.obj"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        read_ordered_polyline_obj(
            path,
            container_records=("o",),
            allow_singletons=True,
            require_segment_lines=True,
        )


def test_winding_discovery_uses_numeric_order_complete_quartets_and_ignores_siblings(
    tmp_path,
):
    base = tmp_path / "fibers"
    _write_quartet(base, 10)
    _write_quartet(base, 2)
    (tmp_path / "fibers_w_2.obj").write_text("# aggregate\n")
    (tmp_path / "fibers_winding_factors.csv").write_text("header\n")
    (tmp_path / "fibers_w_m1_h.obj").write_text("# obsolete\n")

    artifacts = discover_winding_artifacts(tmp_path / "fibers.obj")
    assert [(item.key.winding, item.key.state) for item in artifacts] == [
        (winding, state) for winding in (2, 10) for state in ("h", "v", "err", "tie")
    ]


def test_winding_discovery_accepts_sparse_state_artifacts(tmp_path):
    base = tmp_path / "fibers"
    for state in ("h", "v", "err"):
        _write_state(base, 0, state)
    artifacts = discover_winding_artifacts(base)
    assert [artifact.key for artifact in artifacts] == [
        WindingLayerKey(0, state) for state in ("h", "v", "err")
    ]


def test_winding_discovery_requires_at_least_one_quartet(tmp_path):
    with pytest.raises(ValueError, match="no winding state OBJ artifacts"):
        discover_winding_artifacts(tmp_path / "fibers")


def test_winding_reader_converts_xyz_to_zyx_and_accepts_empty_state(tmp_path):
    base = tmp_path / "fibers"
    path = _write_state(
        base,
        3,
        "h",
        "o fiber\nv 1 2 3\nv 4 5 6\nl 1 2\n",
    )
    geometry = read_winding_artifact(
        WindingArtifact(WindingLayerKey(3, "h"), path)
    )
    assert geometry.paths_zyx[0].dtype == np.float32
    np.testing.assert_array_equal(geometry.paths_zyx[0], [[3, 2, 1], [6, 5, 4]])

    empty_path = _write_state(base, 3, "v")
    empty = read_winding_artifact(
        WindingArtifact(WindingLayerKey(3, "v"), empty_path)
    )
    assert empty.paths_zyx == ()


def test_winding_reader_rejects_wrong_header_and_singleton(tmp_path):
    wrong = tmp_path / "wrong.obj"
    wrong.write_text("# wrong\n")
    with pytest.raises(ValueError, match="expected exactly"):
        read_winding_artifact(WindingArtifact(WindingLayerKey(0, "h"), wrong))

    point = _write_state(
        tmp_path / "fibers",
        0,
        "h",
        "o point\nv 1 2 3\np 1\n",
    )
    with pytest.raises(ValueError, match="is a singleton"):
        read_winding_artifact(WindingArtifact(WindingLayerKey(0, "h"), point))


def test_reference_artifact_is_optional_and_converts_xyz_to_zyx(tmp_path):
    base = tmp_path / "fibers"
    assert discover_reference_artifact(base) is None
    assert load_reference_geometry(base) is None

    path = _write_reference(
        base,
        "o reference_0_a\n"
        "v 1 2 3\n"
        "v 4 5 6\n"
        "l 1 2\n",
    )
    assert discover_reference_artifact(base) == path
    loaded = read_reference_artifact(path)
    assert loaded.path == path
    assert loaded.paths_zyx[0].dtype == np.float32
    np.testing.assert_array_equal(loaded.paths_zyx[0], [[3, 2, 1], [6, 5, 4]])
    reloaded = load_reference_geometry(base)
    assert reloaded is not None
    assert reloaded.path == loaded.path
    np.testing.assert_array_equal(reloaded.paths_zyx[0], loaded.paths_zyx[0])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# wrong\n", "tagged reference-fiber header"),
        ("# VC3D tagged reference fibers\n", "artifact is empty"),
        (
            "# VC3D tagged reference fibers\n"
            + "o one\nv 0 0 0\nv 1 0 0\nl 1 2\n"
            + "o one\nv 2 0 0\nv 3 0 0\nl 3 4\n",
            "duplicate container",
        ),
    ],
)
def test_reference_artifact_rejects_malformed_present_output(
    tmp_path, content, message
):
    path = tmp_path / "fibers_reference.obj"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_reference_artifact(path)


def test_winding_colors_are_stable_shared_for_hv_bright_and_opaque():
    keys = [
        WindingLayerKey(winding, state)
        for winding in range(20)
        for state in ("h", "v", "err", "tie")
    ]
    colors = [winding_layer_color(key) for key in keys]
    for color in colors:
        assert color[3] == 1.0
        assert max(color[:3]) == 1.0
        assert min(color[:3]) >= 0.27
    assert winding_layer_color(WindingLayerKey(7, "h")) == winding_layer_color(
        WindingLayerKey(7, "v")
    )
    assert winding_layer_color(WindingLayerKey(7, "err")) != winding_layer_color(
        WindingLayerKey(7, "h")
    )
    assert winding_layer_color(WindingLayerKey(7, "tie")) != winding_layer_color(
        WindingLayerKey(7, "err")
    )


def test_winding_layer_colors_are_explicit_per_shape_rgba_arrays():
    key = WindingLayerKey(7, "h")
    colors = winding_layer_colors(key, 3)
    assert colors.shape == (3, 4)
    assert colors.dtype == np.float32
    np.testing.assert_array_equal(colors[0], colors[1])
    np.testing.assert_allclose(colors[0], winding_layer_color(key))


def test_add_winding_layers_passes_identical_hv_per_shape_colors(tmp_path):
    class FakeLayer:
        editable = True

    class FakeViewer:
        def __init__(self):
            self.calls = []

        def add_shapes(self, data, **kwargs):
            self.calls.append((data, kwargs))
            return FakeLayer()

    key = WindingLayerKey(2, "v")
    paths = (
        np.zeros((2, 3), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
    )
    geometry = (
        WindingGeometry(WindingArtifact(key, tmp_path / "v.obj"), paths),
        WindingGeometry(
            WindingArtifact(WindingLayerKey(2, "h"), tmp_path / "h.obj"),
            (np.full((2, 3), 2.0, dtype=np.float32),),
        ),
    )
    viewer = FakeViewer()
    layers, fiber_count = add_winding_layers(viewer, geometry, 3.0)
    assert tuple(layers) == tuple(
        WindingLayerKey(2, state) for state in ("h", "v", "err", "tie")
    )
    assert fiber_count == 3
    assert len(viewer.calls) == 2
    data, kwargs = viewer.calls[1]
    assert len(data) == 2
    assert kwargs["edge_color"].shape == (2, 4)
    assert kwargs["edge_color"].dtype == np.float32
    assert kwargs["edge_width"] == 3.0
    assert kwargs["blending"] == "opaque"
    assert all(not getattr(layer, "editable", False) for layer in layers.values())
    np.testing.assert_array_equal(
        viewer.calls[0][1]["edge_color"][0], kwargs["edge_color"][0]
    )
    assert not layers[WindingLayerKey(2, "err")].visible
    assert not layers[WindingLayerKey(2, "tie")].visible


def test_add_winding_layers_materializes_missing_states_and_windings(tmp_path):
    class FakeViewer:
        def __init__(self):
            self.calls = []

        def add_shapes(self, data, **kwargs):
            layer = type("FakeLayer", (), {"visible": kwargs["visible"]})()
            self.calls.append((data, kwargs, layer))
            return layer

    geometry = (
        WindingGeometry(
            WindingArtifact(WindingLayerKey(1, "h"), tmp_path / "w1_h.obj"),
            (np.zeros((2, 3), dtype=np.float32),),
        ),
        WindingGeometry(
            WindingArtifact(WindingLayerKey(3, "v"), tmp_path / "w3_v.obj"),
            (),
        ),
    )
    viewer = FakeViewer()
    layers, fiber_count = add_winding_layers(viewer, geometry, 2.0)
    assert tuple(layers) == tuple(
        WindingLayerKey(winding, state)
        for winding in range(1, 4)
        for state in ("h", "v", "err", "tie")
    )
    assert fiber_count == 1
    assert len(viewer.calls) == 1
    assert viewer.calls[0][0]
    assert all(not layer.visible for layer in layers.values())


def test_materialized_grid_rotates_missing_empty_and_absent_slots(tmp_path):
    class FakeLayer:
        def __init__(self, visible: bool):
            self.visible = visible

    class FakeViewer:
        def add_shapes(self, _data, **kwargs):
            return FakeLayer(kwargs["visible"])

    geometry = (
        WindingGeometry(
            WindingArtifact(WindingLayerKey(1, "h"), tmp_path / "w1_h.obj"),
            (np.zeros((2, 3), dtype=np.float32),),
        ),
        WindingGeometry(
            WindingArtifact(WindingLayerKey(3, "v"), tmp_path / "w3_v.obj"),
            (),
        ),
    )
    layers, _ = add_winding_layers(FakeViewer(), geometry, 2.0)
    original = {
        WindingLayerKey(1, "v"),  # Missing state file.
        WindingLayerKey(2, "h"),  # Completely absent winding.
        WindingLayerKey(3, "v"),  # Present but empty state file.
    }
    for key, layer in layers.items():
        layer.visible = key in original
    assert rotate_winding_layer_visibility(layers, 1) == {
        WindingLayerKey(2, "v"),
        WindingLayerKey(3, "h"),
        WindingLayerKey(1, "v"),
    }
    assert rotate_winding_layer_visibility(layers, -1) == original


def test_reference_layer_is_independent_bright_and_visible(tmp_path):
    class FakeLayer:
        visible = True

    class FakeViewer:
        def __init__(self):
            self.calls = []

        def add_shapes(self, data, **kwargs):
            self.calls.append((data, kwargs))
            return FakeLayer()

    geometry = ReferenceGeometry(
        tmp_path / "fibers_reference.obj",
        (np.zeros((2, 3), dtype=np.float32),),
    )
    viewer = FakeViewer()
    reference_layer = add_reference_layer(viewer, geometry, 4.0)
    assert reference_layer.visible
    assert len(viewer.calls) == 1
    _, kwargs = viewer.calls[0]
    assert kwargs["name"] == "Reference fibers"
    assert kwargs["visible"] is True
    assert kwargs["edge_width"] == 4.0
    assert kwargs["blending"] == "opaque"
    assert reference_layer.editable is False
    assert kwargs["edge_color"].shape == (1, 4)
    assert np.max(kwargs["edge_color"][0, :3]) == 1.0

    keys = (WindingLayerKey(0, "h"), WindingLayerKey(0, "v"))
    assert visible_winding_layers(keys, "none") == set()
    assert reference_layer.visible
    assert add_reference_layer(viewer, None, 4.0) is None


def test_visibility_presets_and_winding_navigation_group_ties_as_broken():
    keys = tuple(
        WindingLayerKey(winding, state)
        for winding in (2, 10)
        for state in ("h", "v", "err", "tie")
    )
    assert visible_winding_layers(keys, "h") == {
        WindingLayerKey(winding, "h") for winding in range(2, 11)
    }
    assert visible_winding_layers(keys, "v") == {
        WindingLayerKey(winding, "v") for winding in range(2, 11)
    }
    assert visible_winding_layers(keys, "broken") == {
        WindingLayerKey(winding, state)
        for winding in range(2, 11)
        for state in ("err", "tie")
    }
    assert visible_winding_layers(keys, "winding", winding=10) == {
        WindingLayerKey(10, "h"),
        WindingLayerKey(10, "v"),
    }
    assert visible_winding_layers(keys, "all") == set(
        complete_winding_layer_keys(keys)
    )
    assert visible_winding_layers(keys, "none") == set()


def test_winding_visibility_rotation_preserves_arbitrary_mask_and_state():
    keys = tuple(
        WindingLayerKey(winding, state)
        for winding in range(2, 7)
        for state in ("h", "v", "err", "tie")
    )
    visible = {
        WindingLayerKey(3, "h"),
        WindingLayerKey(5, "v"),
        WindingLayerKey(5, "h"),
        WindingLayerKey(6, "h"),
        WindingLayerKey(4, "err"),
    }
    assert rotate_visible_winding_mask(keys, visible, -1) == {
        WindingLayerKey(2, "h"),
        WindingLayerKey(4, "v"),
        WindingLayerKey(4, "h"),
        WindingLayerKey(5, "h"),
        WindingLayerKey(3, "err"),
    }


def test_winding_visibility_rotation_wraps_and_handles_sparse_states():
    keys = (
        WindingLayerKey(0, "h"),
        WindingLayerKey(1, "v"),
        WindingLayerKey(2, "h"),
        WindingLayerKey(4, "h"),
        WindingLayerKey(2, "err"),
        WindingLayerKey(2, "tie"),
    )
    assert rotate_visible_winding_mask(
        keys,
        (WindingLayerKey(0, "h"), WindingLayerKey(1, "v")),
        -1,
    ) == {WindingLayerKey(4, "h"), WindingLayerKey(0, "v")}
    assert rotate_visible_winding_mask(
        keys,
        (
            WindingLayerKey(2, "h"),
            WindingLayerKey(2, "err"),
        ),
        1,
    ) == {WindingLayerKey(3, "h"), WindingLayerKey(3, "err")}
    assert rotate_visible_winding_mask(
        keys, (WindingLayerKey(4, "h"),), 1
    ) == {WindingLayerKey(0, "h")}
    one_winding = (
        WindingLayerKey(7, "h"),
        WindingLayerKey(7, "err"),
    )
    assert rotate_visible_winding_mask(
        one_winding, (WindingLayerKey(7, "err"),), 1
    ) == {WindingLayerKey(7, "err")}


def test_winding_visibility_rotation_moves_an_empty_winding_with_wrap():
    keys = tuple(
        WindingLayerKey(winding, state)
        for winding in (0, 1, 2, 3)
        for state in ("h", "v", "err", "tie")
    )
    visible = {key for key in keys if key.winding != 1}
    rotated = rotate_visible_winding_mask(keys, visible, 1)
    assert {key for key in keys if key not in rotated} == {
        key for key in keys if key.winding == 2
    }
    restored = rotate_visible_winding_mask(keys, rotated, -1)
    assert restored == visible


def test_layer_visibility_rotation_reads_live_complete_mask_only():
    class FakeLayer:
        def __init__(self, visible: bool):
            self.visible = visible

    keys = tuple(
        WindingLayerKey(winding, state)
        for winding in range(2, 6)
        for state in ("h", "v", "err", "tie")
    )
    visible = {
        WindingLayerKey(3, "h"),
        WindingLayerKey(4, "v"),
        WindingLayerKey(3, "err"),
    }
    layers = {key: FakeLayer(key in visible) for key in keys}
    unmanaged = FakeLayer(True)
    rotated = rotate_winding_layer_visibility(layers, -1)
    assert rotated == {
        WindingLayerKey(2, "h"),
        WindingLayerKey(3, "v"),
        WindingLayerKey(2, "err"),
    }
    assert layers[WindingLayerKey(2, "h")].visible
    assert not layers[WindingLayerKey(3, "h")].visible
    assert not layers[WindingLayerKey(4, "v")].visible
    assert layers[WindingLayerKey(3, "v")].visible
    assert layers[WindingLayerKey(2, "err")].visible
    assert not layers[WindingLayerKey(3, "err")].visible
    assert not layers[WindingLayerKey(3, "tie")].visible
    assert unmanaged.visible
    restored = rotate_winding_layer_visibility(layers, 1)
    assert restored == visible


def test_navigation_uses_only_nonempty_h_or_v_geometry(tmp_path):
    def geometry(winding: int, state: str, nonempty: bool) -> WindingGeometry:
        points = (np.zeros((2, 3), dtype=np.float32),) if nonempty else ()
        artifact = WindingArtifact(
            WindingLayerKey(winding, state),
            tmp_path / f"{winding}_{state}.obj",
        )
        return WindingGeometry(artifact, points)

    loaded = (
        geometry(0, "err", True),
        geometry(1, "h", True),
        geometry(2, "v", False),
        geometry(2, "tie", True),
        geometry(3, "v", True),
    )
    keys = nonempty_layer_keys(loaded)
    assert navigable_windings(tuple(keys)) == (1, 3)


def test_animation_interval_uses_seconds_and_rejects_invalid_values():
    assert animation_interval_milliseconds(0.5) == 500
    assert animation_interval_milliseconds(0.125) == 125
    assert animation_interval_milliseconds(0.0001) == 1
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            animation_interval_milliseconds(invalid)


def test_visible_winding_label_compacts_ranges_and_bounds_sparse_lists():
    assert format_visible_windings(()) == "No visible winding"
    assert format_visible_windings((4,)) == "Winding 4"
    assert format_visible_windings(range(18)) == "Windings 0-17"
    assert format_visible_windings((0, 1, 3, 5, 6, 7)) == (
        "Windings 0-1, 3, 5-7"
    )
    assert format_visible_windings((0, 2, 4, 6, 8, 10, 12)) == (
        "Windings 0, 2, ..., 10, 12"
    )


def test_notification_timer_guard_configures_existing_and_future_instances():
    class Timer:
        def __init__(self):
            self.interval = 0
            self.single_shot = False
            self.active = False

        def isActive(self):  # noqa: N802 - Qt API
            return self.active

        def stop(self):
            self.active = False

        def start(self):
            self.active = True

        def setInterval(self, value):  # noqa: N802 - Qt API
            self.interval = value

        def setSingleShot(self, value):  # noqa: N802 - Qt API
            self.single_shot = value

    class Notification:
        DISMISS_AFTER = 4000
        _instances = []

        def __init__(self):
            self.timer = Timer()

        def timer_start(self):
            if self.DISMISS_AFTER > 0:
                self.timer.start()

    existing = Notification()
    Notification._instances.append(existing)
    viewer_module._install_napari_notification_timer_guard(Notification)

    assert existing.timer.interval == 4000
    assert existing.timer.single_shot is True
    assert existing.timer.active is False

    future = Notification()
    future.timer_start()
    assert future.timer.interval == 4000
    assert future.timer.single_shot is True
    assert future.timer.active is True


def test_module_import_is_gui_independent_and_cli_accepts_obj_base():
    assert "napari" not in viewer_module.__dict__
    args = build_parser().parse_args(["fibers.obj", "--width", "3.5"])
    assert args.base == "fibers.obj"
    assert args.width == 3.5
