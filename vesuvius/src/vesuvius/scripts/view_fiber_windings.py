"""View per-winding H, V, and broken fiber OBJ layers in Napari."""

from __future__ import annotations

import argparse
import colorsys
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vesuvius.scripts.ordered_polyline_obj import read_ordered_polyline_obj

_STATE_ORDER = ("h", "v", "err", "tie")
_BROKEN_STATES = frozenset(("err", "tie"))
_STATE_HEADERS = {
    "h": "VC3D Fiberlet crop traces: BP horizontal argmax",
    "v": "VC3D Fiberlet crop traces: BP vertical argmax",
    "err": "VC3D Fiberlet crop traces: BP error/Mixed argmax",
    "tie": "VC3D Fiberlet crop traces: BP exact argmax tie",
}
_STATE_NAMES = {
    "h": "H",
    "v": "V",
    "err": "Broken",
    "tie": "Tie",
}
_STATE_HUE_OFFSETS = {
    "err": 0.38,
    "tie": 0.57,
}
_REFERENCE_HEADER = "VC3D tagged reference fibers"
_REFERENCE_COLOR = (1.0, 0.88, 0.12, 1.0)


@dataclass(frozen=True, order=True)
class WindingLayerKey:
    """One published winding and orientation-state partition."""

    winding: int
    state: str


@dataclass(frozen=True)
class WindingArtifact:
    """One discovered winding-state OBJ artifact."""

    key: WindingLayerKey
    path: Path


@dataclass(frozen=True)
class WindingGeometry:
    """One winding-state artifact converted to Napari ZYX paths."""

    artifact: WindingArtifact
    paths_zyx: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ReferenceGeometry:
    """Optional tagged VC3D reference fibers converted to Napari ZYX paths."""

    path: Path
    paths_zyx: tuple[np.ndarray, ...]


def normalize_winding_output_base(path: str | Path) -> Path:
    """Normalize either `/path/base` or `/path/base.obj` to `/path/base`."""
    base = Path(path).expanduser()
    if base.suffix == ".obj":
        base = base.with_suffix("")
    if not base.name:
        raise ValueError("winding output base must include a filename")
    if not base.parent.is_dir():
        raise ValueError(f"winding output directory does not exist: {base.parent}")
    return base


def discover_winding_artifacts(path: str | Path) -> tuple[WindingArtifact, ...]:
    """Discover complete current-format state quartets for an output base."""
    base = normalize_winding_output_base(path)
    pattern = re.compile(
        rf"^{re.escape(base.name)}_w_(?P<winding>[0-9]+)_"
        rf"(?P<state>{'|'.join(_STATE_ORDER)})\.obj$"
    )
    by_winding: dict[int, dict[str, Path]] = {}
    for candidate in base.parent.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.fullmatch(candidate.name)
        if match is None:
            continue
        winding = int(match.group("winding"))
        state = match.group("state")
        states = by_winding.setdefault(winding, {})
        if state in states:
            raise ValueError(
                f"duplicate winding artifact for winding {winding} state {state}"
            )
        states[state] = candidate

    if not by_winding:
        raise ValueError(f"no winding state OBJ artifacts found for base {base}")
    expected = set(_STATE_ORDER)
    for winding, states in sorted(by_winding.items()):
        missing = expected - set(states)
        if missing:
            raise ValueError(
                f"winding {winding} is missing state artifacts: "
                + ", ".join(sorted(missing))
            )

    return tuple(
        WindingArtifact(WindingLayerKey(winding, state), by_winding[winding][state])
        for winding in sorted(by_winding)
        for state in _STATE_ORDER
    )


def read_winding_artifact(artifact: WindingArtifact) -> WindingGeometry:
    """Read and validate one C++ winding-state OBJ artifact."""
    parsed = read_ordered_polyline_obj(
        artifact.path,
        container_records=("o",),
        allow_singletons=True,
        require_segment_lines=True,
    )
    expected_header = _STATE_HEADERS[artifact.key.state]
    comments = tuple(comment.text for comment in parsed.preamble_comments)
    if comments != (expected_header,):
        raise ValueError(
            f"{artifact.path}: expected exactly the {artifact.key.state!r} "
            "winding-state header"
        )
    paths: list[np.ndarray] = []
    for group in parsed.groups:
        if group.comments:
            raise ValueError(
                f"{artifact.path}:{group.comments[0].line_number}: "
                "winding objects may not contain metadata comments"
            )
        if group.singleton:
            raise ValueError(
                f"{artifact.path}:{group.line_number}: winding fiber "
                f"{group.name!r} is a singleton"
            )
        paths.append(group.points_xyz[:, ::-1].astype(np.float32, copy=True))
    return WindingGeometry(artifact=artifact, paths_zyx=tuple(paths))


def load_winding_geometry(path: str | Path) -> tuple[WindingGeometry, ...]:
    """Discover and read every winding-state artifact for one output base."""
    return tuple(read_winding_artifact(item) for item in discover_winding_artifacts(path))


def discover_reference_artifact(path: str | Path) -> Path | None:
    """Return the optional CLI-owned tagged-reference sibling artifact."""
    base = normalize_winding_output_base(path)
    candidate = base.parent / f"{base.name}_reference.obj"
    return candidate if candidate.is_file() else None


def read_reference_artifact(path: str | Path) -> ReferenceGeometry:
    """Read and validate the tagged VC3D reference-fiber OBJ artifact."""
    artifact = Path(path)
    parsed = read_ordered_polyline_obj(
        artifact,
        container_records=("o",),
        allow_singletons=False,
        require_segment_lines=True,
    )
    comments = tuple(comment.text for comment in parsed.preamble_comments)
    if comments != (_REFERENCE_HEADER,):
        raise ValueError(
            f"{artifact}: expected exactly the tagged reference-fiber header"
        )
    paths: list[np.ndarray] = []
    for group in parsed.groups:
        if group.comments:
            raise ValueError(
                f"{artifact}:{group.comments[0].line_number}: "
                "reference fiber objects may not contain metadata comments"
            )
        paths.append(group.points_xyz[:, ::-1].astype(np.float32, copy=True))
    if not paths:
        raise ValueError(f"{artifact}: reference fiber artifact is empty")
    return ReferenceGeometry(path=artifact, paths_zyx=tuple(paths))


def load_reference_geometry(path: str | Path) -> ReferenceGeometry | None:
    """Load the optional tagged-reference sibling for one winding output base."""
    artifact = discover_reference_artifact(path)
    return None if artifact is None else read_reference_artifact(artifact)


def winding_layer_color(key: WindingLayerKey) -> tuple[float, float, float, float]:
    """Return a stable bright color, sharing one color for H/V per winding."""
    if key.winding < 0 or key.state not in _STATE_ORDER:
        raise ValueError("invalid winding layer key")
    golden_ratio_conjugate = 0.6180339887498949
    state_offset = 0.0 if key.state in {"h", "v"} else _STATE_HUE_OFFSETS[key.state]
    hue = (key.winding * golden_ratio_conjugate + state_offset) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return red, green, blue, 1.0


def winding_layer_colors(key: WindingLayerKey, count: int) -> np.ndarray:
    """Return Napari's explicit per-shape RGBA color array."""
    if count < 0:
        raise ValueError("winding layer color count must be nonnegative")
    color = np.asarray(winding_layer_color(key), dtype=np.float32)
    return np.broadcast_to(color, (count, 4)).copy()


def winding_layer_name(key: WindingLayerKey) -> str:
    """Return a compact stable Napari layer name."""
    return f"w{key.winding} {_STATE_NAMES[key.state]}"


def nonempty_layer_keys(
    geometry: Sequence[WindingGeometry],
) -> frozenset[WindingLayerKey]:
    """Return the keys that contain displayable paths."""
    return frozenset(item.artifact.key for item in geometry if item.paths_zyx)


def navigable_windings(keys: Sequence[WindingLayerKey]) -> tuple[int, ...]:
    """Return sorted windings having at least one nonempty H or V layer."""
    return tuple(sorted({key.winding for key in keys if key.state in {"h", "v"}}))


def visible_winding_layers(
    keys: Sequence[WindingLayerKey],
    preset: str,
    *,
    winding: int | None = None,
) -> frozenset[WindingLayerKey]:
    """Select visible keys for one category or one H+V winding preset."""
    available = frozenset(keys)
    if preset == "all":
        return available
    if preset == "none":
        return frozenset()
    if preset == "h":
        states = frozenset(("h",))
    elif preset == "v":
        states = frozenset(("v",))
    elif preset == "broken":
        states = _BROKEN_STATES
    elif preset == "winding":
        if winding is None or winding < 0:
            raise ValueError("winding preset requires a nonnegative winding")
        return frozenset(
            key
            for key in available
            if key.winding == winding and key.state in {"h", "v"}
        )
    else:
        raise ValueError(f"unknown winding visibility preset: {preset!r}")
    return frozenset(key for key in available if key.state in states)


def rotate_visible_winding_mask(
    keys: Sequence[WindingLayerKey],
    visible: Sequence[WindingLayerKey],
    delta: int,
) -> frozenset[WindingLayerKey]:
    """Circularly rotate the complete managed visibility mask by winding."""
    if delta == 0:
        raise ValueError("winding visibility rotation must be nonzero")
    available = frozenset(keys)
    selected = frozenset(visible) & available
    windings = tuple(sorted({key.winding for key in available}))
    if len(windings) <= 1:
        return selected
    winding_index = {winding: index for index, winding in enumerate(windings)}
    rotated: set[WindingLayerKey] = set()
    for source in selected:
        destination_index = (winding_index[source.winding] + delta) % len(windings)
        destination = WindingLayerKey(windings[destination_index], source.state)
        if destination in available:
            rotated.add(destination)
    return frozenset(rotated)


def rotate_winding_layer_visibility(
    layers: Mapping[WindingLayerKey, object], delta: int
) -> frozenset[WindingLayerKey]:
    """Rotate a snapshot of every managed layer's live visibility bit."""
    visible = tuple(key for key, layer in layers.items() if layer.visible)
    rotated = rotate_visible_winding_mask(tuple(layers), visible, delta)
    for key, layer in layers.items():
        layer.visible = key in rotated
    return rotated


def add_winding_controls(
    viewer,
    layers: Mapping[WindingLayerKey, object],
) -> None:
    """Add grouped category and winding navigation controls to a viewer."""
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    keys = tuple(layers)
    windings = tuple(sorted({key.winding for key in keys}))
    initial_windings = navigable_windings(keys)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    category_row = QHBoxLayout()
    navigation_row = QHBoxLayout()
    winding_label = QLabel()

    def update_label() -> None:
        visible_windings = sorted(
            {
                key.winding
                for key, layer in layers.items()
                if layer.visible
            }
        )
        if not visible_windings:
            winding_label.setText("No visible winding")
            return
        prefix = "Winding" if len(visible_windings) == 1 else "Windings"
        winding_label.setText(
            f"{prefix} {', '.join(str(value) for value in visible_windings)}"
        )

    def apply(preset: str, winding: int | None = None) -> None:
        selected = visible_winding_layers(keys, preset, winding=winding)
        for key, layer in layers.items():
            layer.visible = key in selected
        update_label()

    for label, preset in (
        ("H", "h"),
        ("V", "v"),
        ("Broken", "broken"),
        ("All", "all"),
        ("None", "none"),
    ):
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False, value=preset: apply(value))
        category_row.addWidget(button)

    previous_button = QPushButton("Previous")
    previous_button.setToolTip("Rotate visibility to the previous winding")
    next_button = QPushButton("Next")
    next_button.setToolTip("Rotate visibility to the next winding")
    previous_button.setEnabled(bool(windings))
    next_button.setEnabled(bool(windings))

    def move(delta: int) -> None:
        rotate_winding_layer_visibility(layers, delta)
        update_label()

    previous_button.clicked.connect(lambda _checked=False: move(-1))
    next_button.clicked.connect(lambda _checked=False: move(1))
    navigation_row.addWidget(previous_button)
    navigation_row.addWidget(winding_label, stretch=1)
    navigation_row.addWidget(next_button)
    layout.addLayout(category_row)
    layout.addLayout(navigation_row)
    for layer in layers.values():
        visible_event = getattr(getattr(layer, "events", None), "visible", None)
        if visible_event is not None:
            visible_event.connect(lambda _event=None: update_label())
    update_label()
    if not initial_windings:
        apply("all")
    else:
        apply("winding", initial_windings[0])
    viewer.window.add_dock_widget(widget, area="right", name="Winding visibility")


def add_winding_layers(
    viewer,
    geometry: Sequence[WindingGeometry],
    edge_width: float,
) -> tuple[dict[WindingLayerKey, object], int]:
    """Add nonempty winding artifacts using explicit per-path colors."""
    layers: dict[WindingLayerKey, object] = {}
    fiber_count = 0
    hv_palette = {
        item.artifact.key.winding: np.asarray(
            winding_layer_color(WindingLayerKey(item.artifact.key.winding, "h")),
            dtype=np.float32,
        )
        for item in geometry
        if item.artifact.key.state in {"h", "v"}
    }
    for item in geometry:
        if not item.paths_zyx:
            continue
        key = item.artifact.key
        color = (
            hv_palette[key.winding]
            if key.state in {"h", "v"}
            else np.asarray(winding_layer_color(key), dtype=np.float32)
        )
        colors = np.broadcast_to(color, (len(item.paths_zyx), 4)).copy()
        layers[key] = viewer.add_shapes(
            list(item.paths_zyx),
            ndim=3,
            shape_type="path",
            name=winding_layer_name(key),
            edge_color=colors,
            edge_width=edge_width,
            face_color="transparent",
            visible=False,
        )
        fiber_count += len(item.paths_zyx)
    return layers, fiber_count


def add_reference_layer(viewer, geometry: ReferenceGeometry | None, edge_width: float):
    """Add the optional independent tagged-reference comparison layer."""
    if geometry is None:
        return None
    colors = np.broadcast_to(
        np.asarray(_REFERENCE_COLOR, dtype=np.float32),
        (len(geometry.paths_zyx), 4),
    ).copy()
    return viewer.add_shapes(
        list(geometry.paths_zyx),
        ndim=3,
        shape_type="path",
        name="Reference fibers",
        edge_color=colors,
        edge_width=edge_width,
        face_color="transparent",
        visible=True,
    )


def launch_viewer(path: str | Path, edge_width: float = 2.0) -> None:
    """Load winding geometry and launch the interactive 3D viewer."""
    if not np.isfinite(edge_width) or edge_width <= 0:
        raise ValueError("edge width must be finite and positive")
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed; install the vesuvius GUI extra"
        ) from exc

    base = normalize_winding_output_base(path)
    geometry = load_winding_geometry(base)
    reference = load_reference_geometry(base)
    viewer = napari.Viewer(ndisplay=3, title=f"Fiber windings: {base.name}")
    layers, fiber_count = add_winding_layers(viewer, geometry, edge_width)
    if not layers:
        raise ValueError(f"all winding state OBJ artifacts are empty for base {base}")
    add_reference_layer(viewer, reference, edge_width)
    add_winding_controls(viewer, layers)
    viewer.reset_view()
    print(
        f"fiber winding viewer windings={len({key.winding for key in layers})} "
        f"layers={len(layers)} fibers={fiber_count} "
        f"reference_fibers={0 if reference is None else len(reference.paths_zyx)}"
    )
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base",
        help="Winding output base, with or without the .obj suffix",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=2.0,
        help="Displayed line width [2]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        launch_viewer(args.base, args.width)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
