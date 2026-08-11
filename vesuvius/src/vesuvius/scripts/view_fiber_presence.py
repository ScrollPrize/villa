"""View a base-coordinate crop of a fiber-presence OME-Zarr in napari."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OmeZarrLevel:
    root: Path
    path: str
    scale_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]

    @property
    def array_path(self) -> Path:
        return self.root / self.path


@dataclass(frozen=True)
class CropSelection:
    requested_base_xyzwhd: tuple[int, int, int, int, int, int]
    slices_zyx: tuple[slice, slice, slice]
    origin_base_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class LineObjGeometry:
    paths_zyx: list[np.ndarray]
    total_groups: int


_LINE_OBJ_HEADERS = {
    "anchors": "# vc_fiberlet_anchors version 1",
    "paths": "# vc_fiberlets version 1",
}


def parse_crop(value: str) -> tuple[int, int, int, int, int, int]:
    try:
        crop = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must contain six comma-separated integers: X,Y,Z,W,H,D"
        ) from exc
    if len(crop) != 6:
        raise argparse.ArgumentTypeError(
            "crop must contain six comma-separated integers: X,Y,Z,W,H,D"
        )
    if any(item < 0 for item in crop[:3]):
        raise argparse.ArgumentTypeError("crop origin must be non-negative")
    if any(item <= 0 for item in crop[3:]):
        raise argparse.ArgumentTypeError("crop dimensions must be positive")
    return crop


def _path_intersects_crop(
    path_zyx: np.ndarray,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> bool:
    x, y, z, width, height, depth = crop_xyzwhd
    low_zyx = np.asarray([z, y, x], dtype=np.float32)
    high_zyx = low_zyx + np.asarray([depth, height, width], dtype=np.float32)
    return bool(
        np.all(np.max(path_zyx, axis=0) >= low_zyx)
        and np.all(np.min(path_zyx, axis=0) < high_zyx)
    )


def read_line_obj(
    path: str | Path,
    kind: str,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> LineObjGeometry:
    """Read one ordered base-XYZ line per group from a fiberlet OBJ."""
    if kind not in _LINE_OBJ_HEADERS:
        raise ValueError(f"unknown line OBJ kind: {kind!r}")

    obj_path = Path(path).expanduser()
    expected_header = _LINE_OBJ_HEADERS[kind]
    paths_zyx: list[np.ndarray] = []
    total_groups = 0
    vertex_count = 0
    group_name: str | None = None
    group_vertices: dict[int, tuple[float, float, float]] = {}
    group_lines: list[list[int]] = []
    header_seen = False

    def fail(line_number: int, message: str) -> ValueError:
        return ValueError(f"{obj_path}:{line_number}: {message}")

    def finish_group(line_number: int) -> None:
        nonlocal total_groups
        if group_name is None:
            return
        total_groups += 1
        if not group_lines:
            raise fail(line_number, f"group {group_name!r} has no line record")

        ordered_indices: list[int] = []
        for indices in group_lines:
            if not ordered_indices:
                ordered_indices.extend(indices)
            elif ordered_indices[-1] == indices[0]:
                ordered_indices.extend(indices[1:])
            else:
                raise fail(
                    line_number,
                    f"group {group_name!r} line records do not form one ordered path",
                )
        try:
            xyz = np.asarray(
                [group_vertices[index] for index in ordered_indices],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise fail(
                line_number,
                f"group {group_name!r} references vertex {exc.args[0]} outside the group",
            ) from exc
        if xyz.shape[0] < 2:
            raise fail(line_number, f"group {group_name!r} has fewer than two points")
        path_zyx = xyz[:, ::-1].copy()
        if _path_intersects_crop(path_zyx, crop_xyzwhd):
            paths_zyx.append(path_zyx)

    try:
        with obj_path.open() as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    header_seen = header_seen or line == expected_header
                    continue

                fields = line.split()
                record = fields[0]
                if record == "g":
                    if len(fields) != 2:
                        raise fail(
                            line_number, "group record must contain exactly one name"
                        )
                    finish_group(line_number)
                    group_name = fields[1]
                    group_vertices = {}
                    group_lines = []
                elif record == "v":
                    if group_name is None:
                        raise fail(line_number, "vertex appears before the first group")
                    if len(fields) != 4:
                        raise fail(line_number, "vertex record must contain X Y Z")
                    try:
                        xyz = tuple(float(value) for value in fields[1:])
                    except ValueError as exc:
                        raise fail(
                            line_number, "vertex coordinates must be numeric"
                        ) from exc
                    if not np.isfinite(xyz).all():
                        raise fail(line_number, "vertex coordinates must be finite")
                    vertex_count += 1
                    group_vertices[vertex_count] = xyz
                elif record == "l":
                    if group_name is None:
                        raise fail(line_number, "line appears before the first group")
                    if len(fields) < 3:
                        raise fail(
                            line_number,
                            "line record must reference at least two vertices",
                        )
                    try:
                        indices = [int(value) for value in fields[1:]]
                    except ValueError as exc:
                        raise fail(
                            line_number, "line indices must be integers"
                        ) from exc
                    if any(index <= 0 for index in indices):
                        raise fail(line_number, "line indices must be positive")
                    group_lines.append(indices)
                else:
                    raise fail(line_number, f"unsupported OBJ record {record!r}")
            finish_group(line_number + 1 if "line_number" in locals() else 1)
    except OSError as exc:
        raise ValueError(f"cannot read line OBJ {obj_path}: {exc}") from exc

    if not header_seen:
        raise ValueError(f"{obj_path} is not a supported {kind} OBJ")
    return LineObjGeometry(paths_zyx=paths_zyx, total_groups=total_groups)


def clipping_plane_in_layer_data(
    layer,
    position_base_zyx: Sequence[float],
    normal_base_zyx: Sequence[float],
) -> dict:
    """Transform a base-coordinate clipping plane into layer data coordinates."""
    position_world = np.asarray(position_base_zyx, dtype=np.float64)
    normal_world = np.asarray(normal_base_zyx, dtype=np.float64)
    if position_world.shape != (3,) or normal_world.shape != (3,):
        raise ValueError("clipping-plane position and normal must be 3D")
    if not np.isfinite(position_world).all() or not np.isfinite(normal_world).all():
        raise ValueError("clipping-plane position and normal must be finite")
    if np.linalg.norm(normal_world) == 0:
        raise ValueError("clipping-plane normal must be nonzero")

    position_data = np.asarray(layer.world_to_data(position_world), dtype=np.float64)
    normal_tip_data = np.asarray(
        layer.world_to_data(position_world + normal_world), dtype=np.float64
    )
    normal_data = normal_tip_data - position_data
    normal_length = np.linalg.norm(normal_data)
    if not np.isfinite(normal_length) or normal_length == 0:
        raise ValueError("layer transform makes the clipping-plane normal invalid")
    normal_data /= normal_length
    return {
        "position": position_data,
        "normal": normal_data,
        "enabled": True,
    }


def crop_clipping_planes_in_layer_data(
    layer,
    lower_base_zyx: Sequence[float],
    upper_base_zyx: Sequence[float],
) -> list[dict]:
    """Build the six inward-facing planes of a base-coordinate crop box."""
    return [
        clipping_plane_in_layer_data(layer, plane["position"], plane["normal"])
        for plane in crop_clipping_planes_in_base(lower_base_zyx, upper_base_zyx)
    ]


def crop_clipping_planes_in_base(
    lower_base_zyx: Sequence[float],
    upper_base_zyx: Sequence[float],
) -> list[dict]:
    """Build six inward-facing crop planes in base/world coordinates."""
    lower = np.asarray(lower_base_zyx, dtype=np.float64)
    upper = np.asarray(upper_base_zyx, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("crop bounds must be 3D")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("crop bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("crop lower bounds must not exceed upper bounds")

    planes: list[dict] = []
    for axis in range(3):
        lower_position = lower.copy()
        upper_position = upper.copy()
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        planes.append({"position": lower_position, "normal": normal, "enabled": True})
        planes.append({"position": upper_position, "normal": -normal, "enabled": True})
    return planes


def common_shape_edge_width(layer, default: float = 2.0) -> float:
    """Read the common width from napari's per-shape edge-width collection."""
    if layer is None:
        return default
    widths = np.asarray(layer.edge_width, dtype=np.float64).reshape(-1)
    if widths.size == 0:
        return default
    width = float(widths[0])
    if not math.isfinite(width) or width <= 0:
        raise ValueError("shape edge width must be positive and finite")
    return width


def set_common_shape_edge_width(layer, width: float) -> None:
    """Set every shape width and notify napari's VisPy layer explicitly."""
    width = float(width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("shape edge width must be positive and finite")
    layer.edge_width = width
    layer.events.edge_width()


def add_clipping_controls(
    viewer,
    volume_layer,
    anchors_layer,
    paths_layer,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QPushButton,
        QSlider,
        QSpinBox,
        QWidget,
    )

    x, y, z, width, height, depth = crop_xyzwhd
    axes = {
        "X": (2, x, x + width),
        "Y": (1, y, y + height),
        "Z": (0, z, z + depth),
    }

    widget = QWidget()
    form = QFormLayout(widget)
    bound_controls: dict[tuple[str, str], tuple[QSlider, QSpinBox]] = {}

    for axis_name, (_, minimum, maximum) in axes.items():
        for side, initial in (("min", minimum), ("max", maximum)):
            control = QWidget()
            layout = QHBoxLayout(control)
            layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            spin = QSpinBox()
            slider.setRange(minimum, maximum)
            spin.setRange(minimum, maximum)
            slider.setValue(initial)
            spin.setValue(initial)
            layout.addWidget(slider, stretch=1)
            layout.addWidget(spin)
            form.addRow(f"{axis_name} {side}", control)
            bound_controls[(axis_name, side)] = (slider, spin)

    def add_width_control(label: str, layer) -> tuple[QSlider, QDoubleSpinBox]:
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 1000)
        slider.setTracking(False)
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        initial = common_shape_edge_width(layer)
        slider.setValue(round(initial * 100))
        spin.setValue(initial)
        slider.setEnabled(layer is not None)
        spin.setEnabled(layer is not None)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)
        form.addRow(label, control)
        return slider, spin

    anchors_width = add_width_control("Anchor width", anchors_layer)
    paths_width = add_width_control("Path width", paths_layer)
    reset_button = QPushButton("Reset")
    form.addRow(reset_button)

    shape_layers = tuple(
        layer for layer in (anchors_layer, paths_layer) if layer is not None
    )
    updating_bounds = False

    def current_bounds() -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(3, dtype=np.float64)
        upper = np.zeros(3, dtype=np.float64)
        for axis_name, (axis_index, _, _) in axes.items():
            lower[axis_index] = bound_controls[(axis_name, "min")][0].value()
            upper[axis_index] = bound_controls[(axis_name, "max")][0].value()
        return lower, upper

    def update_clipping(*_args) -> None:
        if updating_bounds:
            return
        lower, upper = current_bounds()
        # VisPy's volume clipper consumes scene coordinates after napari reverses
        # ZYX to XYZ; passing crop-local image coordinates clips translated crops.
        volume_layer.experimental_clipping_planes = crop_clipping_planes_in_base(
            lower, upper
        )
        for layer in shape_layers:
            layer.experimental_clipping_planes = crop_clipping_planes_in_layer_data(
                layer, lower, upper
            )

    def bound_changed(axis_name: str, side: str, value: int) -> None:
        nonlocal updating_bounds
        if updating_bounds:
            return
        updating_bounds = True
        slider, spin = bound_controls[(axis_name, side)]
        slider.setValue(value)
        spin.setValue(value)
        other_side = "max" if side == "min" else "min"
        other_slider, other_spin = bound_controls[(axis_name, other_side)]
        if (side == "min" and value > other_slider.value()) or (
            side == "max" and value < other_slider.value()
        ):
            other_slider.setValue(value)
            other_spin.setValue(value)
        updating_bounds = False
        update_clipping()

    for (axis_name, side), (slider, spin) in bound_controls.items():
        slider.valueChanged.connect(
            lambda value, axis_name=axis_name, side=side: bound_changed(
                axis_name, side, value
            )
        )
        spin.valueChanged.connect(
            lambda value, axis_name=axis_name, side=side: bound_changed(
                axis_name, side, value
            )
        )

    def connect_width_control(control, layer) -> None:
        slider, spin = control
        if layer is None:
            return
        updating_width = False

        def slider_changed(value: int) -> None:
            nonlocal updating_width
            if updating_width:
                return
            updating_width = True
            width = value / 100.0
            spin.setValue(width)
            updating_width = False
            set_common_shape_edge_width(layer, width)

        def spin_changed(value: float) -> None:
            nonlocal updating_width
            if updating_width:
                return
            updating_width = True
            slider.setValue(round(value * 100))
            updating_width = False
            set_common_shape_edge_width(layer, value)

        slider.valueChanged.connect(slider_changed)
        spin.valueChanged.connect(spin_changed)

    connect_width_control(anchors_width, anchors_layer)
    connect_width_control(paths_width, paths_layer)

    def reset_bounds(*_args) -> None:
        nonlocal updating_bounds
        updating_bounds = True
        for axis_name, (_, minimum, maximum) in axes.items():
            for side, value in (("min", minimum), ("max", maximum)):
                slider, spin = bound_controls[(axis_name, side)]
                slider.setValue(value)
                spin.setValue(value)
        updating_bounds = False
        update_clipping()

    reset_button.clicked.connect(reset_bounds)
    update_clipping()

    viewer.window.add_dock_widget(widget, area="right", name="Clip")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing OME-Zarr metadata: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OME-Zarr metadata {path}: {exc}") from exc


def _find_multiscale_root(path: Path) -> tuple[Path, dict]:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        attrs_path = candidate / ".zattrs"
        if not attrs_path.is_file():
            continue
        attrs = _read_json(attrs_path)
        if isinstance(attrs.get("multiscales"), list):
            return candidate, attrs
    raise ValueError(f"{path} is not an OME-Zarr pyramid root or an array inside one")


def _compose_transforms(
    transforms: Sequence[dict], ndim: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    scale = np.ones(ndim, dtype=np.float64)
    translation = np.zeros(ndim, dtype=np.float64)
    for transform in transforms:
        transform_type = transform.get("type")
        if transform_type == "scale":
            values = np.asarray(transform.get("scale"), dtype=np.float64)
            if (
                values.shape != (ndim,)
                or not np.isfinite(values).all()
                or np.any(values <= 0)
            ):
                raise ValueError(
                    "OME-Zarr scale must contain one positive value per axis"
                )
            scale *= values
            translation *= values
        elif transform_type == "translation":
            values = np.asarray(transform.get("translation"), dtype=np.float64)
            if values.shape != (ndim,) or not np.isfinite(values).all():
                raise ValueError(
                    "OME-Zarr translation must contain one finite value per axis"
                )
            translation += values
        else:
            raise ValueError(
                f"unsupported OME-Zarr coordinate transform: {transform_type!r}"
            )
    return tuple(float(item) for item in scale), tuple(
        float(item) for item in translation
    )


def resolve_ome_zarr_level(
    zarr_path: str | Path, level: str | None = None
) -> OmeZarrLevel:
    requested = Path(zarr_path).expanduser().resolve()
    root, attrs = _find_multiscale_root(requested)
    multiscales = attrs["multiscales"]
    if len(multiscales) != 1 or not isinstance(multiscales[0], dict):
        raise ValueError(
            "fiber presence OME-Zarr must contain exactly one multiscale image"
        )

    multiscale = multiscales[0]
    axes = multiscale.get("axes")
    axis_names = tuple(
        axis.get("name") if isinstance(axis, dict) else axis for axis in axes or ()
    )
    if axis_names != ("z", "y", "x"):
        raise ValueError(f"fiber presence axes must be Z,Y,X, got {axis_names}")

    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr multiscale metadata has no datasets")

    if level is not None:
        selected_path = level.strip("/")
    elif requested == root:
        selected_path = str(datasets[0].get("path", "")).strip("/")
    else:
        try:
            selected_path = requested.relative_to(root).as_posix().strip("/")
        except ValueError as exc:
            raise ValueError(f"{requested} is not inside OME-Zarr root {root}") from exc

    matches = [
        item
        for item in datasets
        if str(item.get("path", "")).strip("/") == selected_path
    ]
    if len(matches) != 1:
        available = ", ".join(str(item.get("path")) for item in datasets)
        raise ValueError(
            f"OME-Zarr level {selected_path!r} not found; available levels: {available}"
        )
    if not (root / selected_path / ".zarray").is_file():
        raise ValueError(
            f"OME-Zarr level is not a local Zarr v2 array: {root / selected_path}"
        )

    root_transforms = multiscale.get("coordinateTransformations", [])
    dataset_transforms = matches[0].get("coordinateTransformations", [])
    scale, translation = _compose_transforms(
        [*dataset_transforms, *root_transforms], ndim=3
    )
    return OmeZarrLevel(
        root=root,
        path=selected_path,
        scale_zyx=scale,
        translation_zyx=translation,
    )


def _ceil_lattice_coordinate(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.ceil(value)


def select_base_crop(
    shape_zyx: Sequence[int],
    level: OmeZarrLevel,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> CropSelection:
    if len(shape_zyx) != 3:
        raise ValueError(
            f"fiber presence array must be 3D, got shape {tuple(shape_zyx)}"
        )

    x, y, z, width, height, depth = crop_xyzwhd
    low_base = (z, y, x)
    high_base = (z + depth, y + height, x + width)
    lows: list[int] = []
    highs: list[int] = []
    for axis in range(3):
        scale = level.scale_zyx[axis]
        translation = level.translation_zyx[axis]
        low = _ceil_lattice_coordinate((low_base[axis] - translation) / scale)
        high = _ceil_lattice_coordinate((high_base[axis] - translation) / scale)
        lows.append(max(0, min(int(shape_zyx[axis]), low)))
        highs.append(max(0, min(int(shape_zyx[axis]), high)))

    if any(low >= high for low, high in zip(lows, highs, strict=True)):
        raise ValueError(
            "crop does not contain any samples from the selected OME-Zarr level"
        )

    slices = tuple(slice(low, high) for low, high in zip(lows, highs, strict=True))
    origin = tuple(
        level.translation_zyx[axis] + lows[axis] * level.scale_zyx[axis]
        for axis in range(3)
    )
    return CropSelection(
        requested_base_xyzwhd=crop_xyzwhd,
        slices_zyx=slices,
        origin_base_zyx=origin,
        shape_zyx=tuple(high - low for low, high in zip(lows, highs, strict=True)),
    )


def open_lazy_crop(
    level: OmeZarrLevel, crop_xyzwhd: tuple[int, int, int, int, int, int]
):
    try:
        import dask.array as da
        import zarr
    except ImportError as exc:
        raise RuntimeError("fiber presence viewing requires dask and zarr") from exc

    array = zarr.open_array(str(level.array_path), mode="r")
    selection = select_base_crop(array.shape, level, crop_xyzwhd)
    lazy_array = da.from_zarr(array)[selection.slices_zyx]
    return lazy_array, selection


def launch_viewer(
    level: OmeZarrLevel,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
    anchors_obj: str | Path | None = None,
    paths_obj: str | Path | None = None,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed; install the vesuvius GUI extra"
        ) from exc

    data, selection = open_lazy_crop(level, crop_xyzwhd)
    dense_gib = int(np.prod(selection.shape_zyx)) * data.dtype.itemsize / 1024**3
    stored_bounds = ",".join(
        f"{axis_slice.start}:{axis_slice.stop}" for axis_slice in selection.slices_zyx
    )
    print(f"OME-Zarr: {level.root}")
    print(f"Level: {level.path} scale_zyx={level.scale_zyx}")
    print(f"Stored crop ZYX: {stored_bounds} shape={selection.shape_zyx}")
    print(f"Dense crop size: {dense_gib:.3f} GiB")

    anchors = (
        read_line_obj(anchors_obj, "anchors", crop_xyzwhd)
        if anchors_obj is not None
        else None
    )
    fiberlets = (
        read_line_obj(paths_obj, "paths", crop_xyzwhd)
        if paths_obj is not None
        else None
    )
    if anchors is not None:
        print(
            f"Anchors: {len(anchors.paths_zyx)}/{anchors.total_groups} groups intersect crop"
        )
    if fiberlets is not None:
        print(
            f"Fiberlets: {len(fiberlets.paths_zyx)}/{fiberlets.total_groups} groups intersect crop"
        )

    viewer = napari.Viewer(ndisplay=3, title="Fiber presence")
    if np.issubdtype(data.dtype, np.integer):
        contrast_limits = (0, np.iinfo(data.dtype).max)
    else:
        contrast_limits = (0.0, 1.0)
    volume_layer = viewer.add_image(
        data,
        name=f"fiber presence [{level.path}]",
        scale=level.scale_zyx,
        translate=selection.origin_base_zyx,
        colormap="HiLo",
        contrast_limits=contrast_limits,
        rendering="attenuated_mip",
    )
    anchors_layer = None
    if anchors is not None and anchors.paths_zyx:
        anchors_layer = viewer.add_shapes(
            anchors.paths_zyx,
            shape_type="line",
            name="fiber anchors",
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
    paths_layer = None
    if fiberlets is not None and fiberlets.paths_zyx:
        paths_layer = viewer.add_shapes(
            fiberlets.paths_zyx,
            shape_type="path",
            name="fiberlet paths",
            edge_color="magenta",
            edge_width=2,
            face_color="transparent",
        )
    add_clipping_controls(
        viewer,
        volume_layer,
        anchors_layer,
        paths_layer,
        crop_xyzwhd,
    )
    viewer.reset_view()
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr",
        help="Local fiber-presence OME-Zarr pyramid root or array level",
    )
    parser.add_argument(
        "--crop",
        required=True,
        type=parse_crop,
        metavar="X,Y,Z,W,H,D",
        help="Half-open crop in base voxels",
    )
    parser.add_argument(
        "--level",
        help="OME-Zarr dataset path; defaults to the finest (first) level",
    )
    parser.add_argument(
        "--anchors",
        help="Fiberlet anchors OBJ to show as a separate line layer",
    )
    parser.add_argument(
        "--paths",
        help="Fiberlet paths OBJ to show as a separate path layer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        resolved = resolve_ome_zarr_level(args.zarr, args.level)
        launch_viewer(resolved, args.crop, args.anchors, args.paths)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
