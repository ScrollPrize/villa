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
_DEFAULT_ANIMATION_INTERVAL_SECONDS = 0.5


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


@dataclass
class _EmptyWindingLayer:
    """Logical visibility slot for winding geometry absent from Napari."""

    visible: bool = False


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
    """Discover current-format state artifacts that physically exist."""
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

    return tuple(
        WindingArtifact(
            WindingLayerKey(winding, state), by_winding[winding][state]
        )
        for winding in sorted(by_winding)
        for state in _STATE_ORDER
        if state in by_winding[winding]
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


def complete_winding_layer_keys(
    keys: Sequence[WindingLayerKey],
) -> tuple[WindingLayerKey, ...]:
    """Expand observed artifacts to the complete contiguous managed grid."""
    observed = tuple(keys)
    if not observed:
        return ()
    if any(key.winding < 0 or key.state not in _STATE_ORDER for key in observed):
        raise ValueError("invalid winding layer key")
    first = min(key.winding for key in observed)
    last = max(key.winding for key in observed)
    return tuple(
        WindingLayerKey(winding, state)
        for winding in range(first, last + 1)
        for state in _STATE_ORDER
    )


def visible_winding_layers(
    keys: Sequence[WindingLayerKey],
    preset: str,
    *,
    winding: int | None = None,
) -> frozenset[WindingLayerKey]:
    """Select visible keys for one category or one H+V winding preset."""
    available = frozenset(complete_winding_layer_keys(keys))
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
    available = frozenset(complete_winding_layer_keys(keys))
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
        target = key in rotated
        if layer.visible != target:
            layer.visible = target
    return rotated


def format_visible_windings(windings: Sequence[int]) -> str:
    """Format a winding selection compactly without an unbounded label."""
    values = tuple(sorted(set(windings)))
    if not values:
        return "No visible winding"

    ranges: list[tuple[int, int]] = []
    first = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((first, previous))
        first = previous = value
    ranges.append((first, previous))

    def render(item: tuple[int, int]) -> str:
        return str(item[0]) if item[0] == item[1] else f"{item[0]}-{item[1]}"

    rendered = [render(item) for item in ranges]
    if len(rendered) > 5:
        rendered = [*rendered[:2], "...", *rendered[-2:]]
    prefix = "Winding" if len(values) == 1 else "Windings"
    return f"{prefix} {', '.join(rendered)}"


def animation_interval_milliseconds(seconds: float) -> int:
    """Convert a positive finite animation interval to Qt milliseconds."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("animation interval must be finite and positive")
    return max(1, round(seconds * 1000.0))


def _configure_napari_notification_timer(notification: object) -> None:
    """Apply the timer state intended by Napari's notification show path."""
    timer = notification.timer
    dismiss_after = int(notification.DISMISS_AFTER)
    was_active = timer.isActive()
    if was_active:
        timer.stop()
    if dismiss_after <= 0:
        return
    timer.setInterval(max(1, dismiss_after))
    timer.setSingleShot(True)
    if was_active:
        timer.start()


def _install_napari_notification_timer_guard(notification_type=None) -> None:
    """Prevent an unconfigured notification timer from spinning at 0 ms."""
    if notification_type is None:
        from napari._qt.dialogs.qt_notification import NapariQtNotification

        notification_type = NapariQtNotification

    marker = "_fiber_winding_timer_guard_installed"
    if not getattr(notification_type, marker, False):
        original_timer_start = notification_type.timer_start

        def guarded_timer_start(notification) -> None:
            _configure_napari_notification_timer(notification)
            original_timer_start(notification)

        notification_type.timer_start = guarded_timer_start
        setattr(notification_type, marker, True)

    for notification in tuple(notification_type._instances):
        _configure_napari_notification_timer(notification)


def add_winding_controls(
    viewer,
    layers: Mapping[WindingLayerKey, object],
    initial_windings: Sequence[int] | None = None,
) -> None:
    """Add grouped category and winding navigation controls to a viewer."""
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    keys = tuple(layers)
    windings = tuple(sorted({key.winding for key in keys}))
    if initial_windings is None:
        initial_windings = navigable_windings(keys)
    initial_windings = tuple(
        winding for winding in initial_windings if winding in windings
    )

    widget = QWidget()
    layout = QVBoxLayout(widget)
    category_row = QHBoxLayout()
    navigation_row = QHBoxLayout()
    animation_row = QHBoxLayout()
    winding_label = QLabel()
    winding_label.setMinimumWidth(80)
    winding_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label_update_timer = QTimer(widget)
    label_update_timer.setObjectName("fiber_winding_label_update")
    label_update_timer.setSingleShot(True)
    label_update_timer.setInterval(0)

    def update_label() -> None:
        visible_windings = sorted(
            {
                key.winding
                for key, layer in layers.items()
                if layer.visible
            }
        )
        winding_label.setText(format_visible_windings(visible_windings))
        winding_label.setToolTip(
            "No visible winding"
            if not visible_windings
            else ", ".join(str(value) for value in visible_windings)
        )

    label_update_timer.timeout.connect(update_label)

    def schedule_label_update(_event=None) -> None:
        if not label_update_timer.isActive():
            label_update_timer.start()

    def apply(preset: str, winding: int | None = None) -> None:
        selected = visible_winding_layers(keys, preset, winding=winding)
        for key, layer in layers.items():
            target = key in selected
            if layer.visible != target:
                layer.visible = target
        schedule_label_update()

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
        schedule_label_update()

    previous_button.clicked.connect(lambda _checked=False: move(-1))
    next_button.clicked.connect(lambda _checked=False: move(1))
    navigation_row.addWidget(previous_button)
    navigation_row.addWidget(winding_label, stretch=1)
    navigation_row.addWidget(next_button)

    animation_button = QPushButton("Animate")
    animation_button.setCheckable(True)
    animation_button.setToolTip("Advance the visible winding mask automatically")
    animation_button.setEnabled(bool(windings))
    animation_interval = QDoubleSpinBox()
    animation_interval.setRange(0.05, 60.0)
    animation_interval.setDecimals(2)
    animation_interval.setSingleStep(0.05)
    animation_interval.setSuffix(" s")
    animation_interval.setValue(_DEFAULT_ANIMATION_INTERVAL_SECONDS)
    animation_interval.setToolTip("Time between winding-mask steps")
    animation_timer = QTimer(widget)
    animation_timer.setObjectName("fiber_winding_animation")
    animation_timer.setInterval(
        animation_interval_milliseconds(animation_interval.value())
    )
    animation_timer.timeout.connect(lambda: move(1))

    def set_animation_running(enabled: bool) -> None:
        animation_button.setText("Stop" if enabled else "Animate")
        if enabled:
            animation_timer.start()
        else:
            animation_timer.stop()

    animation_button.toggled.connect(set_animation_running)
    animation_interval.valueChanged.connect(
        lambda seconds: animation_timer.setInterval(
            animation_interval_milliseconds(seconds)
        )
    )
    animation_row.addWidget(animation_button)
    animation_row.addWidget(QLabel("Interval"))
    animation_row.addWidget(animation_interval)
    animation_row.addStretch(1)
    layout.addLayout(category_row)
    layout.addLayout(navigation_row)
    layout.addLayout(animation_row)
    for layer in layers.values():
        visible_event = getattr(getattr(layer, "events", None), "visible", None)
        if visible_event is not None:
            visible_event.connect(schedule_label_update)
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
    """Add the complete managed grid, with geometry where it exists."""
    layers: dict[WindingLayerKey, object] = {}
    fiber_count = 0
    geometry_by_key: dict[WindingLayerKey, tuple[np.ndarray, ...]] = {}
    for item in geometry:
        key = item.artifact.key
        if key in geometry_by_key:
            raise ValueError(f"duplicate winding geometry for {key}")
        geometry_by_key[key] = item.paths_zyx
    for key in complete_winding_layer_keys(tuple(geometry_by_key)):
        paths = geometry_by_key.get(key, ())
        if not paths:
            layers[key] = _EmptyWindingLayer()
            continue
        color = np.asarray(winding_layer_color(key), dtype=np.float32)
        layer = viewer.add_shapes(
            list(paths),
            ndim=3,
            shape_type="path",
            name=winding_layer_name(key),
            edge_color=np.broadcast_to(color, (len(paths), 4)).copy(),
            edge_width=edge_width,
            face_color="transparent",
            blending="opaque",
            visible=False,
        )
        layer.editable = False
        layers[key] = layer
        fiber_count += len(paths)
    return layers, fiber_count


def add_reference_layer(viewer, geometry: ReferenceGeometry | None, edge_width: float):
    """Add the optional independent tagged-reference comparison layer."""
    if geometry is None:
        return None
    colors = np.broadcast_to(
        np.asarray(_REFERENCE_COLOR, dtype=np.float32),
        (len(geometry.paths_zyx), 4),
    ).copy()
    layer = viewer.add_shapes(
        list(geometry.paths_zyx),
        ndim=3,
        shape_type="path",
        name="Reference fibers",
        edge_color=colors,
        edge_width=edge_width,
        face_color="transparent",
        blending="opaque",
        visible=True,
    )
    layer.editable = False
    return layer


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

    _install_napari_notification_timer_guard()

    base = normalize_winding_output_base(path)
    geometry = load_winding_geometry(base)
    initial_windings = navigable_windings(tuple(nonempty_layer_keys(geometry)))
    reference = load_reference_geometry(base)
    viewer = napari.Viewer(ndisplay=3, title=f"Fiber windings: {base.name}")
    layers, fiber_count = add_winding_layers(viewer, geometry, edge_width)
    if fiber_count == 0:
        raise ValueError(f"all winding state OBJ artifacts are empty for base {base}")
    add_reference_layer(viewer, reference, edge_width)
    add_winding_controls(viewer, layers, initial_windings)
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
