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
    "h": 0.0,
    "v": 0.19,
    "err": 0.38,
    "tie": 0.57,
}


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


def winding_layer_color(key: WindingLayerKey) -> tuple[float, float, float, float]:
    """Return a stable bright color derived only from winding and state."""
    if key.winding < 0 or key.state not in _STATE_HUE_OFFSETS:
        raise ValueError("invalid winding layer key")
    golden_ratio_conjugate = 0.6180339887498949
    hue = (
        key.winding * golden_ratio_conjugate + _STATE_HUE_OFFSETS[key.state]
    ) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return red, green, blue, 1.0


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


def advance_winding(
    windings: Sequence[int], current: int | None, delta: int
) -> int:
    """Move cyclically through a nonempty ordered winding collection."""
    ordered = tuple(windings)
    if not ordered:
        raise ValueError("cannot navigate an empty winding collection")
    if len(set(ordered)) != len(ordered) or tuple(sorted(ordered)) != ordered:
        raise ValueError("navigable windings must be unique and sorted")
    if delta == 0:
        raise ValueError("winding navigation delta must be nonzero")
    if current not in ordered:
        return ordered[0] if delta > 0 else ordered[-1]
    return ordered[(ordered.index(current) + delta) % len(ordered)]


def add_winding_controls(
    viewer,
    layers: Mapping[WindingLayerKey, object],
) -> None:
    """Add grouped category and winding navigation controls to a viewer."""
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    keys = tuple(layers)
    windings = navigable_windings(keys)
    current_winding = windings[0] if windings else None

    widget = QWidget()
    layout = QVBoxLayout(widget)
    category_row = QHBoxLayout()
    navigation_row = QHBoxLayout()
    winding_label = QLabel()

    def apply(preset: str, winding: int | None = None) -> None:
        selected = visible_winding_layers(keys, preset, winding=winding)
        for key, layer in layers.items():
            layer.visible = key in selected

    def update_label() -> None:
        if current_winding is None:
            winding_label.setText("No H/V winding")
            return
        winding_label.setText(
            f"Winding {current_winding} "
            f"({windings.index(current_winding) + 1}/{len(windings)})"
        )

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

    previous_button = QToolButton()
    previous_button.setText("<")
    previous_button.setToolTip("Previous winding")
    next_button = QToolButton()
    next_button.setText(">")
    next_button.setToolTip("Next winding")
    previous_button.setEnabled(bool(windings))
    next_button.setEnabled(bool(windings))

    def move(delta: int) -> None:
        nonlocal current_winding
        current_winding = advance_winding(windings, current_winding, delta)
        apply("winding", current_winding)
        update_label()

    previous_button.clicked.connect(lambda _checked=False: move(-1))
    next_button.clicked.connect(lambda _checked=False: move(1))
    navigation_row.addWidget(previous_button)
    navigation_row.addWidget(winding_label, stretch=1)
    navigation_row.addWidget(next_button)
    layout.addLayout(category_row)
    layout.addLayout(navigation_row)
    update_label()
    if current_winding is None:
        apply("all")
    else:
        apply("winding", current_winding)
    viewer.window.add_dock_widget(widget, area="right", name="Winding visibility")


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
    viewer = napari.Viewer(ndisplay=3, title=f"Fiber windings: {base.name}")
    layers: dict[WindingLayerKey, object] = {}
    fiber_count = 0
    for item in geometry:
        if not item.paths_zyx:
            continue
        key = item.artifact.key
        layers[key] = viewer.add_shapes(
            list(item.paths_zyx),
            ndim=3,
            shape_type="path",
            name=winding_layer_name(key),
            edge_color=winding_layer_color(key),
            edge_width=edge_width,
            face_color="transparent",
            visible=False,
        )
        fiber_count += len(item.paths_zyx)
    if not layers:
        raise ValueError(f"all winding state OBJ artifacts are empty for base {base}")
    add_winding_controls(viewer, layers)
    viewer.reset_view()
    print(
        f"fiber winding viewer windings={len({key.winding for key in layers})} "
        f"layers={len(layers)} fibers={fiber_count}"
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

