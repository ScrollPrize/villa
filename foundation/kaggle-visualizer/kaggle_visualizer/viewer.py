from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from napari.utils.notifications import show_info, show_warning

import colorcet
import napari
import numpy as np
import tifffile
from skimage import measure


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}


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


@dataclass
class SamplePair:
    image_path: Path
    label_path: Path

    @property
    def name(self) -> str:
        return self.image_path.stem


class PairedDatasetViewer:
    def __init__(self, train_dir: Path, label_dir: Path, log_path: Optional[Path] = None) -> None:
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
        self.log_path = log_path
        self.logged_ids: Set[str] = self._load_existing_logs(log_path) if log_path else set()

        # Use letter keys that are broadly available across keyboard layouts.
        self.viewer.bind_key("n")(self._next_sample)
        self.viewer.bind_key("b")(self._previous_sample)
        # Component inspection: toggle isolation and cycle components.
        self.viewer.bind_key("v")(self._toggle_isolate_component)
        self.viewer.bind_key("k")(self._next_component)
        self.viewer.bind_key("j")(self._previous_component)
        # Log current sample ID to CSV. Use an unused key (`g` for "flag") with overwrite to avoid conflicts.
        self.viewer.bind_key("g", overwrite=True)(self._log_current_sample)

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
        label_volume = _load_volume(pair.label_path)
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

        self._apply_selected_component()
        self.viewer.text_overlay.text = _text_for_sample(pair.name, self.index, len(self.pairs))

    def _next_sample(self, _viewer=None) -> None:
        self.index = (self.index + 1) % len(self.pairs)
        self._load_current()

    def _previous_sample(self, _viewer=None) -> None:
        self.index = (self.index - 1) % len(self.pairs)
        self._load_current()

    def _log_current_sample(self, _viewer=None) -> None:
        if not self.log_path:
            show_warning("No log file configured. Use --log-csv to enable logging.")
            return
        sample_id = self.pairs[self.index].name
        if sample_id in self.logged_ids:
            show_info(f"Sample already logged: {sample_id}")
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
            with self.log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["sample_id"])
                writer.writerow([sample_id])
            self.logged_ids.add(sample_id)
            msg = f"Logged sample: {sample_id} -> {self.log_path}"
            show_info(msg)
            print(f"[kaggle-visualizer] {msg}")
        except Exception as exc:
            show_warning(f"Failed to log sample {sample_id}: {exc}")
            print(f"[kaggle-visualizer] failed to log sample {sample_id}: {exc}")

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


def launch_viewer(train_dir: str, label_dir: str, log_csv: Optional[Path] = None) -> None:
    """
    Launch a napari viewer showing 3D TIFF volumes with connected-component labels.
    """
    resolved_log = Path(log_csv).expanduser() if log_csv else None
    viewer = PairedDatasetViewer(Path(train_dir), Path(label_dir), log_path=resolved_log)
    viewer.show()


__all__ = ["launch_viewer", "PairedDatasetViewer"]
