from pathlib import Path

import numpy as np
import pytest
import vesuvius.scripts.view_fiber_windings as viewer_module
from vesuvius.scripts.ordered_polyline_obj import read_ordered_polyline_obj
from vesuvius.scripts.view_fiber_windings import (
    WindingArtifact,
    WindingGeometry,
    WindingLayerKey,
    advance_winding,
    build_parser,
    discover_winding_artifacts,
    navigable_windings,
    nonempty_layer_keys,
    read_winding_artifact,
    visible_winding_layers,
    winding_layer_color,
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


def test_winding_discovery_rejects_missing_quartet_member(tmp_path):
    base = tmp_path / "fibers"
    for state in ("h", "v", "err"):
        _write_state(base, 0, state)
    with pytest.raises(ValueError, match="missing state artifacts: tie"):
        discover_winding_artifacts(base)


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


def test_winding_colors_are_stable_distinct_bright_and_opaque():
    keys = [
        WindingLayerKey(winding, state)
        for winding in range(20)
        for state in ("h", "v", "err", "tie")
    ]
    colors = [winding_layer_color(key) for key in keys]
    assert len(set(colors)) == len(colors)
    for color in colors:
        assert color[3] == 1.0
        assert max(color[:3]) == 1.0
        assert min(color[:3]) >= 0.27
    assert winding_layer_color(WindingLayerKey(7, "h")) == winding_layer_color(
        WindingLayerKey(7, "h")
    )


def test_visibility_presets_and_winding_navigation_group_ties_as_broken():
    keys = tuple(
        WindingLayerKey(winding, state)
        for winding in (2, 10)
        for state in ("h", "v", "err", "tie")
    )
    assert visible_winding_layers(keys, "h") == {
        WindingLayerKey(2, "h"),
        WindingLayerKey(10, "h"),
    }
    assert visible_winding_layers(keys, "v") == {
        WindingLayerKey(2, "v"),
        WindingLayerKey(10, "v"),
    }
    assert visible_winding_layers(keys, "broken") == {
        WindingLayerKey(winding, state)
        for winding in (2, 10)
        for state in ("err", "tie")
    }
    assert visible_winding_layers(keys, "winding", winding=10) == {
        WindingLayerKey(10, "h"),
        WindingLayerKey(10, "v"),
    }
    assert visible_winding_layers(keys, "all") == set(keys)
    assert visible_winding_layers(keys, "none") == set()
    assert advance_winding((2, 10), 2, 1) == 10
    assert advance_winding((2, 10), 10, 1) == 2
    assert advance_winding((2, 10), 2, -1) == 10


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


def test_module_import_is_gui_independent_and_cli_accepts_obj_base():
    assert "napari" not in viewer_module.__dict__
    args = build_parser().parse_args(["fibers.obj", "--width", "3.5"])
    assert args.base == "fibers.obj"
    assert args.width == 3.5
