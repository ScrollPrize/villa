from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from napari.utils.notifications import show_info, show_warning
from numba import njit

import colorcet
import napari
import numpy as np
import tifffile
from scipy import ndimage as ndi
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget
from skimage import measure


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}
OFFSETS_6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
OFFSETS_26 = np.array(
    [
        [i, j, k]
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if not (i == 0 and j == 0 and k == 0)
    ]
)


def _list_tiffs(folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def _hex_to_rgba(hex_color: str) -> Tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def _glasbey_mapping(unique_labels: Sequence[int]) -> Dict[int, Tuple[float, float, float, float]]:
    palette = colorcet.glasbey
    mapping: Dict[int, Tuple[float, float, float, float]] = {0: (0, 0, 0, 0)}
    label_ids = [label for label in unique_labels if label != 0]

    for idx, label in enumerate(label_ids):
        mapping[int(label)] = _hex_to_rgba(palette[idx % len(palette)])
    return mapping


def _connected_components(label_volume: np.ndarray, target_value: int = 1) -> np.ndarray:
    """
    Return connected components for a single label value; all other labels are treated as background.
    """
    mask = label_volume == target_value
    # Use 26-connectivity in 3D (faces + edges + corners).
    labeled = measure.label(mask, connectivity=3)
    return labeled.astype(np.int32)


def _load_volume(path: Path) -> np.ndarray:
    data = tifffile.imread(path)
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    return data


def _text_for_sample(name: str, index: int, total: int) -> str:
    return f"{name} ({index + 1}/{total})"


def _rgba_to_qcolor(rgba: Tuple[float, float, float, float]) -> QColor:
    r, g, b, a = rgba
    return QColor.fromRgbF(float(r), float(g), float(b), float(a))


def _normalize_rgba(rgba: np.ndarray) -> Tuple[float, float, float, float]:
    arr = np.asarray(rgba, dtype=float)
    if arr.shape != (4,):
        arr = arr.reshape((4,))
    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    if max_value > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _is_direct_color_mode(mode: object) -> bool:
    if isinstance(mode, str):
        return mode.lower() == "direct"
    if hasattr(mode, "value") and isinstance(getattr(mode, "value"), str):
        return getattr(mode, "value").lower() == "direct"
    if hasattr(mode, "name") and isinstance(getattr(mode, "name"), str):
        return getattr(mode, "name").lower() == "direct"
    return "direct" in str(mode).lower()


BBox = Tuple[int, int, int, int, int, int]


@dataclass
class ComponentTopology:
    cavities: int
    tunnels: int
    cavity_bboxes: Optional[List[BBox]] = None
    tunnel_bboxes: Optional[List[BBox]] = None


class ComponentPaletteWidget(QWidget):
    component_clicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

        self._component_ids: List[int] = []
        self._component_colors: List[QColor] = []
        self._selected_component: Optional[int] = None

        self._swatch_size = 18
        self._gap = 4
        self._margin = 6

    def set_components(
        self,
        component_ids: Sequence[int],
        component_colors: Sequence[QColor],
        selected_component: Optional[int],
    ) -> None:
        self._component_ids = list(component_ids)
        self._component_colors = list(component_colors)
        self._selected_component = selected_component
        self._update_minimum_height()
        self.update()

    def set_selected_component(self, selected_component: Optional[int]) -> None:
        if selected_component == self._selected_component:
            return
        self._selected_component = selected_component
        self.update()

    def _columns(self) -> int:
        if not self._component_ids:
            return 1
        available_width = max(self.width() - (2 * self._margin), 1)
        cell = self._swatch_size + self._gap
        cols = max(1, (available_width + self._gap) // cell)
        return int(cols)

    def _update_minimum_height(self) -> None:
        cols = self._columns()
        n = len(self._component_ids)
        rows = (n + cols - 1) // cols if n else 1
        height = (2 * self._margin) + (rows * self._swatch_size) + ((rows - 1) * self._gap)
        self.setMinimumHeight(int(height))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_minimum_height()

    def _component_at(self, x: int, y: int) -> Optional[int]:
        if not self._component_ids:
            return None
        x0 = x - self._margin
        y0 = y - self._margin
        if x0 < 0 or y0 < 0:
            return None

        cell = self._swatch_size + self._gap
        col = x0 // cell
        row = y0 // cell
        if col < 0 or row < 0:
            return None
        if (x0 % cell) >= self._swatch_size or (y0 % cell) >= self._swatch_size:
            return None

        cols = self._columns()
        idx = int(row * cols + col)
        if idx < 0 or idx >= len(self._component_ids):
            return None
        return self._component_ids[idx]

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if hasattr(event, "position"):
                pos = event.position()
                x = int(pos.x())
                y = int(pos.y())
            elif hasattr(event, "localPos"):
                pos = event.localPos()
                x = int(pos.x())
                y = int(pos.y())
            else:
                pos = event.pos()
                x = int(pos.x())
                y = int(pos.y())
            component_id = self._component_at(x, y)
            if component_id is not None:
                self.component_clicked.emit(int(component_id))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if hasattr(event, "position"):
            pos = event.position()
            x = int(pos.x())
            y = int(pos.y())
        elif hasattr(event, "localPos"):
            pos = event.localPos()
            x = int(pos.x())
            y = int(pos.y())
        else:
            pos = event.pos()
            x = int(pos.x())
            y = int(pos.y())
        component_id = self._component_at(x, y)
        self.setToolTip(f"Component {component_id}" if component_id is not None else "")
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        cols = self._columns()
        cell = self._swatch_size + self._gap
        for idx, (component_id, color) in enumerate(zip(self._component_ids, self._component_colors)):
            row = idx // cols
            col = idx % cols
            x = self._margin + (col * cell)
            y = self._margin + (row * cell)

            painter.fillRect(x, y, self._swatch_size, self._swatch_size, color)

            border_pen = QPen(Qt.black)
            border_pen.setWidth(1)
            painter.setPen(border_pen)
            painter.drawRect(x, y, self._swatch_size, self._swatch_size)

            if component_id == self._selected_component:
                selected_pen = QPen(QColor(255, 255, 255))
                selected_pen.setWidth(2)
                painter.setPen(selected_pen)
                painter.drawRect(x + 1, y + 1, self._swatch_size - 2, self._swatch_size - 2)


class ComponentLegendWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.count_label = QLabel("Connected components: 0")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        self.count_label.setFont(font)

        self.sample_label = QLabel("")
        self.sample_topology_label = QLabel("")
        self.selected_topology_label = QLabel("")
        self.selected_topology_label.setVisible(False)

        self.palette_widget = ComponentPaletteWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.palette_widget)
        scroll.setMinimumHeight(90)
        scroll.setMaximumHeight(220)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.count_label)
        layout.addWidget(self.sample_label)
        layout.addWidget(self.sample_topology_label)
        layout.addWidget(self.selected_topology_label)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def set_components(
        self,
        sample_id: str,
        component_ids: Sequence[int],
        color_mapping: Dict[int, Tuple[float, float, float, float]],
        selected_component: Optional[int],
        *,
        label_opacity: float = 1.0,
    ) -> None:
        n = len(component_ids)
        self.count_label.setText(f"Connected components: {n}")
        self.sample_label.setText(f"Sample: {sample_id}")

        opacity = float(np.clip(label_opacity, 0.0, 1.0))
        colors: List[QColor] = []
        for component_id in component_ids:
            qcolor = _rgba_to_qcolor(color_mapping[component_id])
            qcolor.setAlphaF(qcolor.alphaF() * opacity)
            colors.append(qcolor)
        self.palette_widget.set_components(component_ids, colors, selected_component)

    def set_selected_component(self, selected_component: Optional[int]) -> None:
        self.palette_widget.set_selected_component(selected_component)

    def set_topology(
        self,
        *,
        sample_cavities: int,
        sample_tunnels: int,
        selected_component: Optional[int],
        selected_cavities: int,
        selected_tunnels: int,
        isolate_mode: bool,
    ) -> None:
        self.sample_topology_label.setText(
            f"Sample — cavities: {sample_cavities} | tunnels: {sample_tunnels}"
        )
        if isolate_mode and selected_component:
            self.selected_topology_label.setText(
                f"Selected component {selected_component} — cavities: {selected_cavities} | tunnels: {selected_tunnels}"
            )
            self.selected_topology_label.setVisible(True)
        else:
            self.selected_topology_label.setVisible(False)


@njit(cache=True)
def _bridge_voxels(mask: np.ndarray, labeled6: np.ndarray, offsets6: np.ndarray, offsets26: np.ndarray) -> np.ndarray:
    to_remove = np.zeros(mask.shape, dtype=np.uint8)
    z_max, y_max, x_max = mask.shape
    coords = np.argwhere(mask)

    for idx in range(coords.shape[0]):
        z, y, x = coords[idx]
        base_id = labeled6[z, y, x]
        if base_id == 0:
            continue

        # If any 6-neighbor is a different component, keep it.
        skip = False
        for k in range(offsets6.shape[0]):
            dz, dy, dx = offsets6[k]
            zz = z + dz
            yy = y + dy
            xx = x + dx
            if 0 <= zz < z_max and 0 <= yy < y_max and 0 <= xx < x_max:
                nid = labeled6[zz, yy, xx]
                if nid != 0 and nid != base_id:
                    skip = True
                    break
        if skip:
            continue

        # Count distinct neighboring component ids via 26-neighborhood.
        neighbor1 = 0
        neighbor2 = 0
        for k in range(offsets26.shape[0]):
            dz, dy, dx = offsets26[k]
            zz = z + dz
            yy = y + dy
            xx = x + dx
            if 0 <= zz < z_max and 0 <= yy < y_max and 0 <= xx < x_max:
                nid = labeled6[zz, yy, xx]
                if nid != 0 and nid != base_id:
                    if neighbor1 == 0:
                        neighbor1 = nid
                    elif nid != neighbor1:
                        neighbor2 = nid
                        break
        if neighbor2 != 0:
            to_remove[z, y, x] = 1

    return to_remove


def _prune_diagonal_bridges(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Remove voxels that only connect multiple 6-connected components via 26-connectivity.
    Uses numba-accelerated inner loop for speed.
    """
    mask = mask.astype(bool)
    for _ in range(max(iterations, 1)):
        labeled6 = measure.label(mask, connectivity=1)
        to_remove = _bridge_voxels(mask, labeled6.astype(np.int32), OFFSETS_6.astype(np.int32), OFFSETS_26.astype(np.int32))
        if not np.any(to_remove):
            break
        mask[to_remove.astype(bool)] = False

    return mask.astype(np.uint8)


def _prune_diagonal_bridges_all_planes(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Apply diagonal bridge pruning across XY, ZX, and ZY orientations to catch plane-specific connectors.
    """
    perms = [
        (0, 1, 2),  # original Z, Y, X
        (1, 2, 0),  # rotate so original Z maps to X
        (0, 2, 1),  # rotate so original Z maps to Y
    ]

    pruned = mask.astype(np.uint8)
    for perm in perms:
        transposed = np.transpose(pruned, perm)
        cleaned = _prune_diagonal_bridges(transposed, iterations=iterations)
        # invert permutation to restore original axis order
        inv_perm = np.argsort(perm)
        pruned = np.transpose(cleaned, inv_perm)
    return pruned


def _write_fixed_label(output_path: Path, mask: np.ndarray) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, mask.astype(np.uint8))
    except Exception as exc:
        show_warning(f"Failed to write fixed label to {output_path}: {exc}")


@dataclass
class SamplePair:
    image_path: Path
    label_path: Path

    @property
    def name(self) -> str:
        return self.image_path.stem


class PairedDatasetViewer:
    def __init__(
        self,
        train_dir: Path,
        label_dir: Path,
        log_mergers: Optional[Path] = None,
        log_tiny: Optional[Path] = None,
    ) -> None:
        self.pairs = self._collect_pairs(train_dir, label_dir)
        if not self.pairs:
            raise ValueError("No matching .tif/.tiff files found between the two folders.")

        self.viewer = napari.Viewer()
        self.component_legend = ComponentLegendWidget()
        self.viewer.window.add_dock_widget(self.component_legend, area="top", name="Components")
        self.component_legend.palette_widget.component_clicked.connect(self._select_component_from_palette)
        self.viewer.text_overlay.visible = True
        try:
            self.viewer.text_overlay.font_size = 24
        except Exception:
            pass
        try:
            self.viewer.text_overlay.position = "top_left"
        except Exception:
            pass
        self.image_layer = None
        self.label_layer = None
        self.bbox_layer = None
        self.feature_bbox_layer = None
        self.component_bboxes: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        self.current_volume_shape: Optional[Tuple[int, int, int]] = None
        self.component_topology: Dict[int, ComponentTopology] = {}
        self.sample_cavities = 0
        self.sample_tunnels = 0
        self.index = 0
        self.current_sample_id: Optional[str] = None
        self.component_ids: List[int] = []
        self.component_index = 0
        self.isolate_component = False
        self._fallback_color_mapping: Dict[int, Tuple[float, float, float, float]] = {}
        self.label_source: str = "auto"  # auto -> use fixed if available, else raw; can be raw/fixed via toggle
        self.log_mergers_path = log_mergers
        self.log_tiny_path = log_tiny
        self.logged_mergers: Set[str] = self._load_existing_logs(log_mergers) if log_mergers else set()
        self.logged_tiny: Set[str] = self._load_existing_logs(log_tiny) if log_tiny else set()

        # Use letter keys that are broadly available across keyboard layouts.
        self.viewer.bind_key("n", overwrite=True)(self._next_sample)
        self.viewer.bind_key("b", overwrite=True)(self._previous_sample)
        # Component inspection: toggle isolation and cycle components.
        self.viewer.bind_key("v", overwrite=True)(self._toggle_isolate_component)
        self.viewer.bind_key("k", overwrite=True)(self._next_component)
        self.viewer.bind_key("j", overwrite=True)(self._previous_component)
        # Log current sample ID to CSV for different mistake types.
        self.viewer.bind_key("g", overwrite=True)(self._log_merger_sample)
        self.viewer.bind_key("t", overwrite=True)(self._log_tiny_sample)
        # Cycle label source (auto/raw/fixed) if available.
        self.viewer.bind_key("c")(self._cycle_label_source)

        self._load_current()
        self.viewer.text_overlay.visible = True
        self.viewer.dims.ndisplay = 3

    def show(self) -> None:
        napari.run()

    def _collect_pairs(self, train_dir: Path, label_dir: Path) -> List[SamplePair]:
        train_files = _list_tiffs(train_dir)
        label_files = {path.stem: path for path in _list_tiffs(label_dir)}

        pairs: List[SamplePair] = []
        for image_path in train_files:
            label_path = label_files.get(image_path.stem)
            if label_path:
                pairs.append(SamplePair(image_path=image_path, label_path=label_path))

        return pairs

    def _load_existing_logs(self, log_path: Optional[Path]) -> Set[str]:
        if not log_path or not log_path.exists():
            return set()
        try:
            with log_path.open("r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                # Skip header if present.
                if rows and rows[0] and rows[0][0].lower() == "sample_id":
                    rows = rows[1:]
                return {row[0] for row in rows if row}
        except Exception:
            return set()

    def _load_current(self) -> None:
        pair = self.pairs[self.index]
        self.current_sample_id = pair.name

        image_volume = _load_volume(pair.image_path)
        fixed_label_path = pair.label_path.with_name(f"{pair.label_path.stem}_fixed{pair.label_path.suffix}")
        raw_label = _load_volume(pair.label_path)
        raw_mask = (raw_label == 1).astype(np.uint8)
        fixed_mask = None
        fixed_used = False

        if fixed_label_path.exists():
            fixed_mask = (_load_volume(fixed_label_path) == 1).astype(np.uint8)
            fixed_used = True
            show_info(f"Using existing fixed label: {fixed_label_path.name}")
        else:
            pruned_mask = _prune_diagonal_bridges_all_planes(raw_mask, iterations=2)
            if not np.array_equal(raw_mask, pruned_mask):
                fixed_mask = pruned_mask.astype(np.uint8)
                _write_fixed_label(fixed_label_path, fixed_mask)
                fixed_used = True
                show_info(f"Pruned diagonal bridges and saved fixed label: {fixed_label_path.name}")

        label_volume, selected_source = self._select_label_volume(raw_mask, fixed_mask)
        if selected_source == "fixed" and fixed_used:
            show_info(f"Using fixed label for {pair.name}")

        labeled_components = _connected_components(label_volume, target_value=1)
        self.component_ids = [int(x) for x in np.unique(labeled_components) if x != 0]
        self.component_index = 0 if self.component_ids else -1
        self._compute_component_bboxes(labeled_components)
        self._compute_component_topology(labeled_components)

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(
                image_volume,
                name="image",
                colormap="gray",
                contrast_limits=(0, 255),
                blending="additive",
            )
            self.image_layer.bind_key("b", self._previous_sample, overwrite=True)
            self.image_layer.bind_key("n", self._next_sample, overwrite=True)
            self.image_layer.bind_key("v", self._toggle_isolate_component, overwrite=True)
            self.image_layer.bind_key("k", self._next_component, overwrite=True)
            self.image_layer.bind_key("j", self._previous_component, overwrite=True)
        else:
            self.image_layer.data = image_volume

        color_mapping = _glasbey_mapping(np.unique(labeled_components))
        self._fallback_color_mapping = color_mapping
        if self.label_layer is None:
            self.label_layer = self.viewer.add_labels(
                labeled_components,
                name="labels",
                opacity=0.5,
            )
            self._apply_label_colors(color_mapping)
            self._connect_label_layer_events()
            # Bind navigation keys on the layer to avoid layer-level defaults overriding viewer bindings.
            self.label_layer.bind_key("b", self._previous_sample, overwrite=True)
            self.label_layer.bind_key("n", self._next_sample, overwrite=True)
            self.label_layer.bind_key("v", self._toggle_isolate_component, overwrite=True)
            self.label_layer.bind_key("k", self._next_component, overwrite=True)
            self.label_layer.bind_key("j", self._previous_component, overwrite=True)
        else:
            # Recreate the labels layer so napari refreshes internal max labels.
            current_show_selected = self.isolate_component and bool(self.component_ids)
            self.viewer.layers.remove(self.label_layer)
            self.label_layer = self.viewer.add_labels(
                labeled_components,
                name="labels",
                opacity=0.5,
            )
            self._apply_label_colors(color_mapping)
            self._connect_label_layer_events()
            self.label_layer.show_selected_label = current_show_selected
            self.label_layer.bind_key("b", self._previous_sample, overwrite=True)
            self.label_layer.bind_key("n", self._next_sample, overwrite=True)
            self.label_layer.bind_key("v", self._toggle_isolate_component, overwrite=True)
            self.label_layer.bind_key("k", self._next_component, overwrite=True)
            self.label_layer.bind_key("j", self._previous_component, overwrite=True)

        self._apply_selected_component()
        self._refresh_component_legend()
        component_count = len(self.component_ids)
        self.viewer.text_overlay.text = f"{_text_for_sample(pair.name, self.index, len(self.pairs))}\ncomponents: {component_count}"
        # Surface the current ID in the viewer UI.
        self.viewer.title = f"{pair.name} [{self.index + 1}/{len(self.pairs)}]"
        self.viewer.status = f"Current sample: {pair.name}"

    def _next_sample(self, _viewer=None) -> None:
        self.index = (self.index + 1) % len(self.pairs)
        self._load_current()

    def _previous_sample(self, _viewer=None) -> None:
        self.index = (self.index - 1) % len(self.pairs)
        self._load_current()

    def _apply_label_colors(self, color_mapping: Dict[int, Tuple[float, float, float, float]]) -> None:
        if not self.label_layer:
            return
        try:
            from napari.utils.colormaps import direct_colormap

            color_dict = {**color_mapping, None: (0.0, 0.0, 0.0, 0.0)}
            self.label_layer.colormap = direct_colormap(color_dict)
            return
        except Exception as exc:
            show_warning(f"Failed to apply direct label colormap: {exc}")
        # Fallback for older napari: `color` is deprecated but may exist.
        try:
            self.label_layer.color = color_mapping
        except Exception as exc:
            show_warning(f"Failed to apply label colors: {exc}")

    def _connect_label_layer_events(self) -> None:
        if not self.label_layer:
            return
        def _try_connect(event_name: str, callback) -> None:
            try:
                emitter = getattr(self.label_layer.events, event_name, None)
                if emitter is not None and hasattr(emitter, "connect"):
                    emitter.connect(callback)
            except Exception:
                return

        _try_connect("colormap", lambda event: self._refresh_component_legend())
        _try_connect("opacity", lambda event: self._refresh_component_legend())
        _try_connect("selected_label", lambda event: self._sync_selected_from_layer())
        _try_connect("show_selected_label", lambda event: self._sync_isolate_from_layer())

    def _sync_selected_from_layer(self) -> None:
        if not self.label_layer:
            return
        self._update_bounding_boxes(min_size=10)
        self._update_feature_bounding_boxes(min_size=10)
        self._refresh_component_legend()

    def _sync_isolate_from_layer(self) -> None:
        if not self.label_layer:
            return
        try:
            self.isolate_component = bool(self.label_layer.show_selected_label)
        except Exception:
            return
        self._update_bounding_boxes(min_size=10)
        self._update_feature_bounding_boxes(min_size=10)
        self._refresh_component_legend()

    def _refresh_component_legend(self) -> None:
        if not self.current_sample_id:
            return
        selected = None
        opacity = 1.0
        isolate_mode = False
        if self.label_layer is not None:
            selected = int(self.label_layer.selected_label) if self.label_layer.selected_label else None
            try:
                opacity = float(self.label_layer.opacity)
            except Exception:
                opacity = 1.0
            try:
                isolate_mode = bool(self.label_layer.show_selected_label)
            except Exception:
                isolate_mode = bool(self.isolate_component)

        layer_colors = {
            int(component_id): self._fallback_color_mapping.get(int(component_id), (1.0, 1.0, 1.0, 1.0))
            for component_id in self.component_ids
        }
        self.component_legend.set_components(
            sample_id=self.current_sample_id,
            component_ids=self.component_ids,
            color_mapping=layer_colors,
            selected_component=selected,
            label_opacity=opacity,
        )
        selected_topology = self.component_topology.get(int(selected)) if (isolate_mode and selected) else None
        self.component_legend.set_topology(
            sample_cavities=int(self.sample_cavities),
            sample_tunnels=int(self.sample_tunnels),
            selected_component=int(selected) if (isolate_mode and selected) else None,
            selected_cavities=int(selected_topology.cavities) if selected_topology else 0,
            selected_tunnels=int(selected_topology.tunnels) if selected_topology else 0,
            isolate_mode=isolate_mode,
        )

    def _component_color_mapping_from_layer(
        self,
        component_ids: Sequence[int],
        fallback: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        mapping: Dict[int, Tuple[float, float, float, float]] = {}
        if not self.label_layer:
            for component_id in component_ids:
                if component_id in fallback:
                    mapping[int(component_id)] = fallback[int(component_id)]
            return mapping

        colormap = getattr(self.label_layer, "colormap", None)
        if colormap is not None and hasattr(colormap, "map"):
            try:
                colormap_no_selection = colormap
                if hasattr(colormap, "_cmap_without_selection"):
                    try:
                        colormap_no_selection = colormap._cmap_without_selection()
                    except Exception:
                        colormap_no_selection = colormap
                rgba_from_colormap = np.asarray(
                    colormap_no_selection.map(np.asarray(component_ids, dtype=np.int32)), dtype=float
                )
                if rgba_from_colormap.shape == (len(component_ids), 4):
                    for idx, component_id in enumerate(component_ids):
                        mapping[int(component_id)] = _normalize_rgba(rgba_from_colormap[idx])
                    return mapping
            except Exception:
                pass

        for component_id in component_ids:
            label_value = int(component_id)
            if label_value in fallback:
                mapping[label_value] = fallback[label_value]
            else:
                mapping[label_value] = (1.0, 1.0, 1.0, 1.0)

        return mapping

    def _compute_component_bboxes(self, labeled_components: np.ndarray) -> None:
        self.component_bboxes = {}
        self.current_volume_shape = tuple(int(x) for x in labeled_components.shape[:3])  # type: ignore[assignment]
        for prop in measure.regionprops(labeled_components):
            label_value = int(getattr(prop, "label", 0))
            if label_value == 0:
                continue
            bbox = getattr(prop, "bbox", None)
            if bbox is None or len(bbox) != 6:
                continue
            self.component_bboxes[label_value] = tuple(int(x) for x in bbox)  # type: ignore[assignment]

    def _compute_component_topology(self, labeled_components: np.ndarray) -> None:
        self.component_topology = {}
        self.sample_cavities = 0
        self.sample_tunnels = 0

        for component_id, bbox in self.component_bboxes.items():
            z0, y0, x0, z1, y1, x1 = bbox
            component_mask = labeled_components[z0:z1, y0:y1, x0:x1] == component_id
            cavities = self._count_cavities(component_mask)
            try:
                euler = int(measure.euler_number(component_mask.astype(np.uint8), connectivity=3))
            except Exception:
                euler = int(measure.euler_number(component_mask.astype(np.uint8)))
            tunnels = max(0, 1 + int(cavities) - int(euler))

            topology = ComponentTopology(cavities=int(cavities), tunnels=int(tunnels))
            self.component_topology[int(component_id)] = topology
            self.sample_cavities += int(cavities)
            self.sample_tunnels += int(tunnels)

    def _count_cavities(self, component_mask: np.ndarray) -> int:
        component_mask = component_mask.astype(bool)
        if not np.any(component_mask):
            return 0
        pad = 1
        padded = np.pad(component_mask, pad_width=pad, mode="constant", constant_values=False)
        background = ~padded
        structure = ndi.generate_binary_structure(3, 1)
        labeled_bg, num = ndi.label(background, structure=structure)
        if num == 0:
            return 0
        border_labels = np.unique(
            np.concatenate(
                [
                    labeled_bg[0, :, :].ravel(),
                    labeled_bg[-1, :, :].ravel(),
                    labeled_bg[:, 0, :].ravel(),
                    labeled_bg[:, -1, :].ravel(),
                    labeled_bg[:, :, 0].ravel(),
                    labeled_bg[:, :, -1].ravel(),
                ]
            )
        )
        outside_nonzero = int(np.count_nonzero(border_labels))
        return max(0, int(num) - outside_nonzero)

    def _cavity_mask(self, component_mask: np.ndarray) -> np.ndarray:
        component_mask = component_mask.astype(bool)
        if not np.any(component_mask):
            return np.zeros(component_mask.shape, dtype=bool)
        pad = 1
        padded = np.pad(component_mask, pad_width=pad, mode="constant", constant_values=False)
        background = ~padded
        structure = ndi.generate_binary_structure(3, 1)
        labeled_bg, num = ndi.label(background, structure=structure)
        if num == 0:
            return np.zeros(component_mask.shape, dtype=bool)
        border_labels = np.unique(
            np.concatenate(
                [
                    labeled_bg[0, :, :].ravel(),
                    labeled_bg[-1, :, :].ravel(),
                    labeled_bg[:, 0, :].ravel(),
                    labeled_bg[:, -1, :].ravel(),
                    labeled_bg[:, :, 0].ravel(),
                    labeled_bg[:, :, -1].ravel(),
                ]
            )
        )
        is_outside = np.zeros(int(num) + 1, dtype=bool)
        is_outside[border_labels] = True
        cavity_mask_padded = (labeled_bg != 0) & ~is_outside[labeled_bg]
        return cavity_mask_padded[pad:-pad, pad:-pad, pad:-pad]

    def _tunnel_proxy_mask(self, filled_component: np.ndarray) -> np.ndarray:
        """
        Return a *proxy* voxel mask for tunnels/handles by detecting 2D holes in
        planar slices of a cavity-filled component.

        This is used only for visualization (blue boxes), not for counting.
        """
        filled_component = filled_component.astype(bool)
        if filled_component.ndim != 3 or not np.any(filled_component):
            return np.zeros(filled_component.shape, dtype=bool)

        axis = int(np.argmin(np.asarray(filled_component.shape)))
        perm = {
            0: (0, 1, 2),  # slice along Z
            1: (1, 2, 0),  # slice along Y
            2: (2, 0, 1),  # slice along X
        }[axis]
        transposed = np.transpose(filled_component, perm)
        proxy_t = np.zeros(transposed.shape, dtype=bool)
        for idx in range(transposed.shape[0]):
            slice_mask = transposed[idx]
            filled_2d = ndi.binary_fill_holes(slice_mask)
            proxy_t[idx] = filled_2d & ~slice_mask
        inv_perm = np.argsort(perm)
        return np.transpose(proxy_t, inv_perm)

    def _ensure_topology_bboxes(self, component_id: int) -> None:
        if self.label_layer is None:
            return
        topology = self.component_topology.get(int(component_id))
        bbox = self.component_bboxes.get(int(component_id))
        if topology is None or bbox is None:
            return

        if topology.cavity_bboxes is not None and topology.tunnel_bboxes is not None:
            return

        z0, y0, x0, z1, y1, x1 = bbox
        labels = self.label_layer.data
        component_mask = np.asarray(labels[z0:z1, y0:y1, x0:x1] == int(component_id))
        cavity_mask = self._cavity_mask(component_mask)

        cavity_bboxes: List[BBox] = []
        if topology.cavities > 0 and np.any(cavity_mask):
            cavity_bboxes = self._bboxes_from_mask(cavity_mask, offset=(z0, y0, x0))

        tunnel_bboxes: List[BBox] = []
        if topology.tunnels > 0:
            filled = component_mask.astype(bool) | cavity_mask.astype(bool)
            tunnel_proxy = self._tunnel_proxy_mask(filled)
            if np.any(tunnel_proxy):
                tunnel_bboxes = self._bboxes_from_mask(tunnel_proxy, offset=(z0, y0, x0))

        topology.cavity_bboxes = cavity_bboxes
        topology.tunnel_bboxes = tunnel_bboxes

    def _bboxes_from_mask(self, mask: np.ndarray, *, offset: Tuple[int, int, int]) -> List[BBox]:
        if not np.any(mask):
            return []
        labeled = measure.label(mask.astype(bool), connectivity=1)
        bboxes: List[BBox] = []
        dz, dy, dx = offset
        for prop in measure.regionprops(labeled):
            bbox = getattr(prop, "bbox", None)
            if bbox is None or len(bbox) != 6:
                continue
            z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox)
            bboxes.append((z0 + dz, y0 + dy, x0 + dx, z1 + dz, y1 + dy, x1 + dx))
        return bboxes

    def _update_bounding_boxes(self, *, min_size: int = 10) -> None:
        if self.label_layer is None:
            return
        try:
            show_boxes = bool(self.label_layer.show_selected_label)
        except Exception:
            show_boxes = bool(self.isolate_component)

        selected_label = int(self.label_layer.selected_label) if self.label_layer.selected_label else 0
        if not show_boxes or selected_label == 0 or selected_label not in self.component_bboxes:
            if self.bbox_layer is not None:
                self.bbox_layer.visible = False
            return

        if not self.current_volume_shape:
            if self.bbox_layer is not None:
                self.bbox_layer.visible = False
            return

        volume_shape = np.asarray(self.current_volume_shape, dtype=float)
        edges: List[np.ndarray] = []
        edge_pairs = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        bbox = self.component_bboxes[selected_label]
        start = np.asarray(bbox[:3], dtype=float)
        end = np.asarray(bbox[3:], dtype=float)

        size = end - start
        target_size = np.maximum(size, float(min_size))
        center = (start + end) / 2.0
        start = center - (target_size / 2.0)
        end = center + (target_size / 2.0)

        start = np.maximum(start, 0.0)
        end = np.minimum(end, volume_shape)

        z0, y0, x0 = start.tolist()
        z1, y1, x1 = end.tolist()
        corners = np.array(
            [
                [z0, y0, x0],
                [z0, y0, x1],
                [z0, y1, x1],
                [z0, y1, x0],
                [z1, y0, x0],
                [z1, y0, x1],
                [z1, y1, x1],
                [z1, y1, x0],
            ],
            dtype=float,
        )
        for a, b in edge_pairs:
            segment = corners[[a, b], :]
            if np.allclose(segment[0], segment[1]):
                continue
            edges.append(segment)

        if not edges:
            if self.bbox_layer is not None:
                self.bbox_layer.visible = False
            return

        if self.bbox_layer is None:
            self.bbox_layer = self.viewer.add_shapes(
                edges,
                name="bboxes",
                shape_type="path",
                edge_color="red",
                face_color="transparent",
                edge_width=3,
                opacity=1.0,
            )
            try:
                self.bbox_layer.editable = False
            except Exception:
                pass
            try:
                self.bbox_layer.bind_key("v", self._toggle_isolate_component, overwrite=True)
                self.bbox_layer.bind_key("k", self._next_component, overwrite=True)
                self.bbox_layer.bind_key("j", self._previous_component, overwrite=True)
                self.bbox_layer.bind_key("b", self._previous_sample, overwrite=True)
                self.bbox_layer.bind_key("n", self._next_sample, overwrite=True)
            except Exception:
                pass
        else:
            self.bbox_layer.data = edges
            try:
                self.bbox_layer.shape_type = ["path"] * len(edges)
            except Exception:
                pass
        self.bbox_layer.visible = True

    def _bboxes_to_edge_shapes(self, bboxes: Sequence[BBox], *, min_size: int = 10) -> List[np.ndarray]:
        if not self.current_volume_shape:
            return []
        volume_shape = np.asarray(self.current_volume_shape, dtype=float)
        edge_pairs = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        edges: List[np.ndarray] = []
        for bbox in bboxes:
            start = np.asarray(bbox[:3], dtype=float)
            end = np.asarray(bbox[3:], dtype=float)

            size = end - start
            target_size = np.maximum(size, float(min_size))
            center = (start + end) / 2.0
            start = center - (target_size / 2.0)
            end = center + (target_size / 2.0)

            start = np.maximum(start, 0.0)
            end = np.minimum(end, volume_shape)

            z0, y0, x0 = start.tolist()
            z1, y1, x1 = end.tolist()
            corners = np.array(
                [
                    [z0, y0, x0],
                    [z0, y0, x1],
                    [z0, y1, x1],
                    [z0, y1, x0],
                    [z1, y0, x0],
                    [z1, y0, x1],
                    [z1, y1, x1],
                    [z1, y1, x0],
                ],
                dtype=float,
            )
            for a, b in edge_pairs:
                segment = corners[[a, b], :]
                if np.allclose(segment[0], segment[1]):
                    continue
                edges.append(segment)

        return edges

    def _update_feature_bounding_boxes(self, *, min_size: int = 10) -> None:
        if self.label_layer is None:
            return
        try:
            show_boxes = bool(self.label_layer.show_selected_label)
        except Exception:
            show_boxes = bool(self.isolate_component)

        selected_label = int(self.label_layer.selected_label) if self.label_layer.selected_label else 0
        topology = self.component_topology.get(int(selected_label))
        if not show_boxes or selected_label == 0 or topology is None:
            if self.feature_bbox_layer is not None:
                self.feature_bbox_layer.visible = False
            return

        self._ensure_topology_bboxes(int(selected_label))
        bboxes: List[BBox] = []
        if topology.cavity_bboxes:
            bboxes.extend(topology.cavity_bboxes)
        if topology.tunnel_bboxes:
            bboxes.extend(topology.tunnel_bboxes)
        edges = self._bboxes_to_edge_shapes(bboxes, min_size=min_size)
        if not edges:
            if self.feature_bbox_layer is not None:
                self.feature_bbox_layer.visible = False
            return

        if self.feature_bbox_layer is None:
            self.feature_bbox_layer = self.viewer.add_shapes(
                edges,
                name="topology",
                shape_type="path",
                edge_color="blue",
                face_color="transparent",
                edge_width=2,
                opacity=1.0,
            )
            try:
                self.feature_bbox_layer.editable = False
            except Exception:
                pass
            try:
                self.feature_bbox_layer.bind_key("v", self._toggle_isolate_component, overwrite=True)
                self.feature_bbox_layer.bind_key("k", self._next_component, overwrite=True)
                self.feature_bbox_layer.bind_key("j", self._previous_component, overwrite=True)
                self.feature_bbox_layer.bind_key("b", self._previous_sample, overwrite=True)
                self.feature_bbox_layer.bind_key("n", self._next_sample, overwrite=True)
            except Exception:
                pass
        else:
            self.feature_bbox_layer.data = edges
            try:
                self.feature_bbox_layer.shape_type = ["path"] * len(edges)
            except Exception:
                pass
        self.feature_bbox_layer.visible = True

    def _select_component_from_palette(self, component_id: int) -> None:
        if not self.component_ids or component_id not in self.component_ids:
            return
        self.component_index = self.component_ids.index(int(component_id))
        self._apply_selected_component()

    def _select_label_volume(self, raw_mask: np.ndarray, fixed_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
        if self.label_source == "raw" or (self.label_source == "fixed" and fixed_mask is None):
            return raw_mask, "raw"
        if self.label_source == "fixed" and fixed_mask is not None:
            return fixed_mask, "fixed"
        # auto: prefer fixed if available
        if fixed_mask is not None:
            return fixed_mask, "fixed"
        return raw_mask, "raw"

    def _cycle_label_source(self, _viewer=None) -> None:
        if self.label_source == "auto":
            self.label_source = "raw"
        elif self.label_source == "raw":
            self.label_source = "fixed"
        else:
            self.label_source = "auto"
        self._load_current()
        show_info(f"Label source: {self.label_source}")

    def _log_sample(self, sample_id: str, path: Optional[Path], cache: Set[str], label: str) -> None:
        if not path:
            show_warning(f"No log file configured for {label}. Use the corresponding CLI flag.")
            return
        if sample_id in cache:
            show_info(f"Sample already logged in {label}: {sample_id}")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["sample_id"])
                writer.writerow([sample_id])
            cache.add(sample_id)
            msg = f"Logged {label}: {sample_id} -> {path}"
            show_info(msg)
            print(f"[kaggle-visualizer] {msg}")
        except Exception as exc:
            show_warning(f"Failed to log {label} for {sample_id}: {exc}")
            print(f"[kaggle-visualizer] failed to log {label} for {sample_id}: {exc}")

    def _log_merger_sample(self, _viewer=None) -> None:
        sample_id = self.pairs[self.index].name
        self._log_sample(sample_id, self.log_mergers_path, self.logged_mergers, "merger")

    def _log_tiny_sample(self, _viewer=None) -> None:
        sample_id = self.pairs[self.index].name
        self._log_sample(sample_id, self.log_tiny_path, self.logged_tiny, "tiny")

    def _apply_selected_component(self) -> None:
        if not self.label_layer:
            return
        has_components = bool(self.component_ids)
        if has_components and self.component_index >= 0:
            self.component_index = self.component_index % len(self.component_ids)
            selected = self.component_ids[self.component_index]
            self.label_layer.selected_label = selected
        else:
            self.label_layer.selected_label = 0
            self.component_index = -1
        # show_selected_label isolates a single component when True.
        self.label_layer.show_selected_label = self.isolate_component and has_components
        self.component_legend.set_selected_component(
            int(self.label_layer.selected_label) if self.label_layer.selected_label else None
        )
        self._update_bounding_boxes(min_size=10)
        self._update_feature_bounding_boxes(min_size=10)

    def _toggle_isolate_component(self, _viewer=None) -> None:
        if not self.label_layer:
            return
        if not self.component_ids:
            self.label_layer.show_selected_label = False
            self.isolate_component = False
            return
        self.isolate_component = not self.isolate_component
        if self.isolate_component:
            # On first toggle per sample, start from the first component (label 1).
            self.component_index = 0
        self._apply_selected_component()

    def _next_component(self, _viewer=None) -> None:
        if not self.component_ids:
            return
        self.component_index = (self.component_index + 1) % len(self.component_ids)
        self._apply_selected_component()

    def _previous_component(self, _viewer=None) -> None:
        if not self.component_ids:
            return
        self.component_index = (self.component_index - 1) % len(self.component_ids)
        self._apply_selected_component()


def launch_viewer(
    train_dir: str,
    label_dir: str,
    log_mergers: Optional[Path] = None,
    log_tiny: Optional[Path] = None,
) -> None:
    """
    Launch a napari viewer showing 3D TIFF volumes with connected-component labels.
    """
    resolved_mergers = Path(log_mergers).expanduser() if log_mergers else None
    resolved_tiny = Path(log_tiny).expanduser() if log_tiny else None
    viewer = PairedDatasetViewer(
        Path(train_dir),
        Path(label_dir),
        log_mergers=resolved_mergers,
        log_tiny=resolved_tiny,
    )
    viewer.show()


__all__ = ["launch_viewer", "PairedDatasetViewer"]
