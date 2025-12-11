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
        self.image_layer = None
        self.label_layer = None
        self.index = 0
        self.component_ids: List[int] = []
        self.component_index = 0
        self.isolate_component = False
        self.label_source: str = "auto"  # auto -> use fixed if available, else raw; can be raw/fixed via toggle
        self.log_mergers_path = log_mergers
        self.log_tiny_path = log_tiny
        self.logged_mergers: Set[str] = self._load_existing_logs(log_mergers) if log_mergers else set()
        self.logged_tiny: Set[str] = self._load_existing_logs(log_tiny) if log_tiny else set()

        # Use letter keys that are broadly available across keyboard layouts.
        self.viewer.bind_key("n", overwrite=True)(self._next_sample)
        self.viewer.bind_key("b", overwrite=True)(self._previous_sample)
        # Component inspection: toggle isolation and cycle components.
        self.viewer.bind_key("v")(self._toggle_isolate_component)
        self.viewer.bind_key("k")(self._next_component)
        self.viewer.bind_key("j")(self._previous_component)
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
        self.isolate_component = False

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
        else:
            self.image_layer.data = image_volume

        color_mapping = _glasbey_mapping(np.unique(labeled_components))
        if self.label_layer is None:
            self.label_layer = self.viewer.add_labels(
                labeled_components,
                name="labels",
                opacity=0.5,
            )
            self.label_layer.color = color_mapping
            # Bind navigation keys on the layer to avoid layer-level defaults overriding viewer bindings.
            self.label_layer.bind_key("b", self._previous_sample, overwrite=True)
            self.label_layer.bind_key("n", self._next_sample, overwrite=True)
        else:
            # Recreate the labels layer so napari refreshes internal max labels.
            current_show_selected = self.isolate_component and bool(self.component_ids)
            self.viewer.layers.remove(self.label_layer)
            self.label_layer = self.viewer.add_labels(
                labeled_components,
                name="labels",
                opacity=0.5,
            )
            self.label_layer.color = color_mapping
            self.label_layer.show_selected_label = current_show_selected
            self.label_layer.bind_key("b", self._previous_sample, overwrite=True)
            self.label_layer.bind_key("n", self._next_sample, overwrite=True)

        self._apply_selected_component()
        self.viewer.text_overlay.text = _text_for_sample(pair.name, self.index, len(self.pairs))

    def _next_sample(self, _viewer=None) -> None:
        self.index = (self.index + 1) % len(self.pairs)
        self._load_current()

    def _previous_sample(self, _viewer=None) -> None:
        self.index = (self.index - 1) % len(self.pairs)
        self._load_current()

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
