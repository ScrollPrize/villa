#!/usr/bin/env python3
"""
Napari-based viewer/editor for hole detection and repair of label volumes.

Usage:
    python hole_viewer.py \
        --root-dir /path/to/samples \
        --suffix1 hole_mask_alpha0.5_close3.tif \
        --suffix2 _alpha_wrap_a0p05_closed_k2_md3.tif
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile as tiff

import napari
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Label

import cc3d

from detect_holes import compute_component_topology, localize_holes
from fix_holes import _fix_2d_hole_fill, _fix_morph_close, _fix_alpha_wrap


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    folders: List[Path] = field(default_factory=list)
    current_index: int = 0
    args: Optional[argparse.Namespace] = None
    _saved_ignore_mask: Optional[np.ndarray] = None


STATE = State()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file_by_suffix(folder: Path, suffix: str) -> Optional[Path]:
    """Find first file in *folder* whose name contains *suffix*."""
    # Try exact endswith first, then contains, to handle suffixes
    # given with or without extension (e.g. "alpha0.5_close3" matches
    # "hole_mask_alpha0.5_close3.tif").
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name.endswith(suffix):
            return p
    for p in sorted(folder.iterdir()):
        if p.is_file() and suffix in p.name:
            return p
    return None


def load_folder(viewer: napari.Viewer, index: int) -> None:
    """Clear viewer and load the folder at *index*."""
    STATE.current_index = index
    STATE._saved_ignore_mask = None
    folder = STATE.folders[index]
    args = STATE.args

    viewer.layers.clear()

    # --- suffix1: Labels layer (editable) ---
    path1 = _find_file_by_suffix(folder, args.suffix1)
    if path1 is not None:
        data1 = tiff.imread(str(path1))
        viewer.add_labels(data1, name=args.suffix1, metadata={"source_path": str(path1)})
    else:
        print(f"[WARN] No file matching suffix1 '{args.suffix1}' in {folder}")

    # --- suffix2: Labels layer ---
    path2 = _find_file_by_suffix(folder, args.suffix2)
    if path2 is not None:
        data2 = tiff.imread(str(path2))
        viewer.add_labels(data2, name=args.suffix2, metadata={"source_path": str(path2)})
    else:
        print(f"[WARN] No file matching suffix2 '{args.suffix2}' in {folder}")

    _update_nav_label()


# ---------------------------------------------------------------------------
# Navigation widget
# ---------------------------------------------------------------------------

_nav_label = Label(value="—")


def _update_nav_label() -> None:
    total = len(STATE.folders)
    idx = STATE.current_index
    name = STATE.folders[idx].name if total else "—"
    _nav_label.value = f"{idx + 1}/{total}: {name}"


def _on_prev(viewer: napari.Viewer) -> None:
    if STATE.current_index > 0:
        load_folder(viewer, STATE.current_index - 1)


def _on_next(viewer: napari.Viewer) -> None:
    if STATE.current_index < len(STATE.folders) - 1:
        load_folder(viewer, STATE.current_index + 1)


def make_nav_widget(viewer: napari.Viewer) -> Container:
    btn_prev = PushButton(text="Previous")
    btn_next = PushButton(text="Next")
    btn_prev.changed.connect(lambda: _on_prev(viewer))
    btn_next.changed.connect(lambda: _on_next(viewer))
    return Container(widgets=[btn_prev, btn_next, _nav_label], labels=False)


# ---------------------------------------------------------------------------
# Save widget
# ---------------------------------------------------------------------------

def make_save_widget(viewer: napari.Viewer) -> Container:
    from qtpy.QtWidgets import QComboBox, QWidget, QHBoxLayout, QLabel

    qt_combo = QComboBox()
    combo_wrapper = QWidget()
    combo_layout = QHBoxLayout(combo_wrapper)
    combo_layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel("Layer")
    combo_layout.addWidget(lbl)
    combo_layout.addWidget(qt_combo)

    btn_save = PushButton(text="Save")

    def _refresh_choices(_=None):
        current = qt_combo.currentText()
        qt_combo.clear()
        names = [l.name for l in viewer.layers]
        qt_combo.addItems(names)
        if current in names:
            qt_combo.setCurrentText(current)

    viewer.layers.events.inserted.connect(_refresh_choices)
    viewer.layers.events.removed.connect(_refresh_choices)

    def _on_save() -> None:
        layer_name = qt_combo.currentText()
        if not layer_name:
            print("[WARN] No layer selected")
            return
        layer = None
        for l in viewer.layers:
            if l.name == layer_name:
                layer = l
                break
        if layer is None:
            print(f"[WARN] Layer '{layer_name}' not found")
            return
        data = np.asarray(layer.data).copy()
        # Restore ignore label before saving if it was removed
        args = STATE.args
        if (STATE._saved_ignore_mask is not None
                and args.ignore_value is not None
                and getattr(STATE, '_saved_ignore_layer_name', None) == layer_name):
            data[STATE._saved_ignore_mask] = args.ignore_value
            print(f"[INFO] Restored ignore voxels (value={args.ignore_value}) before saving")

        src = layer.metadata.get("source_path")
        folder = STATE.folders[STATE.current_index]
        if src is not None:
            stem = Path(src).stem
        else:
            stem = layer.name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = folder / f"{stem}_{ts}.tif"
        tiff.imwrite(str(out_path), data)
        print(f"[OK] Saved {out_path}")

    btn_save.changed.connect(_on_save)
    _refresh_choices()

    container = Container(widgets=[btn_save], labels=False)
    container.native.layout().insertWidget(0, combo_wrapper)
    return container


# ---------------------------------------------------------------------------
# Detect Holes widget
# ---------------------------------------------------------------------------

def make_detect_holes_widget(viewer: napari.Viewer) -> Container:
    from qtpy.QtWidgets import QComboBox, QWidget, QHBoxLayout, QLabel
    from magicgui.widgets import SpinBox, ComboBox as MgComboBox

    # Layer chooser (Qt combo box)
    qt_layer_combo = QComboBox()
    layer_wrapper = QWidget()
    layer_layout = QHBoxLayout(layer_wrapper)
    layer_layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel("Layer")
    layer_layout.addWidget(lbl)
    layer_layout.addWidget(qt_layer_combo)

    backend_combo = MgComboBox(
        label="Backend",
        choices=["betti_matching", "cripser"],
        value="betti_matching",
    )
    dilation_spin = SpinBox(label="Dilation radius", min=0, max=50, value=3)
    btn = PushButton(text="Detect Holes")

    def _refresh_layer_choices(_=None):
        current = qt_layer_combo.currentText()
        qt_layer_combo.clear()
        names = [l.name for l in viewer.layers if isinstance(l, napari.layers.Labels)]
        qt_layer_combo.addItems(names)
        if current in names:
            qt_layer_combo.setCurrentText(current)

    viewer.layers.events.inserted.connect(_refresh_layer_choices)
    viewer.layers.events.removed.connect(_refresh_layer_choices)

    def _on_detect():
        args = STATE.args
        layer_name = qt_layer_combo.currentText()
        if not layer_name:
            print("[WARN] No layer selected")
            return
        label_layer = None
        for l in viewer.layers:
            if l.name == layer_name:
                label_layer = l
                break
        if label_layer is None or not isinstance(label_layer, napari.layers.Labels):
            print(f"[WARN] Labels layer '{layer_name}' not found")
            return

        backend = backend_combo.value
        dilation_radius = dilation_spin.value

        label_data = np.asarray(label_layer.data)
        mask = (label_data == args.fg_value)
        if args.ignore_value is not None:
            mask &= (label_data != args.ignore_value)

        if backend == "betti_matching":
            b1 = None
            print(f"[INFO] Localizing holes (backend={backend}) …")
            hole_mask, coords = localize_holes(
                mask,
                dilation_radius=dilation_radius,
                expected_b1=None,
                backend=backend,
            )
        else:
            print("[INFO] Computing topology …")
            b1, _b2, _chi = compute_component_topology(
                mask,
                fg_connectivity=args.fg_connectivity,
                bg_connectivity=args.bg_connectivity,
                use_cpu=True,
            )
            print(f"[INFO] Euler-based b1={b1}")
            if b1 > 0:
                print(f"[INFO] Localizing holes (backend={backend}) …")
                hole_mask, coords = localize_holes(
                    mask,
                    dilation_radius=dilation_radius,
                    expected_b1=b1,
                    backend=backend,
                )
            else:
                hole_mask = np.zeros(mask.shape, dtype=bool)
                coords = []

        print(f"[INFO] Found {len(coords)} hole location(s)")

        # Remove old hole_mask layer if present
        existing = [l for l in viewer.layers if l.name == "hole_mask"]
        for l in existing:
            viewer.layers.remove(l)

        viewer.add_labels(
            hole_mask.astype(np.uint8),
            name="hole_mask",
            metadata={"hole_coords": coords},
        )

    btn.changed.connect(_on_detect)
    _refresh_layer_choices()

    container = Container(widgets=[backend_combo, dilation_spin, btn], labels=False)
    container.native.layout().insertWidget(0, layer_wrapper)
    return container


# ---------------------------------------------------------------------------
# Fix Holes widget
# ---------------------------------------------------------------------------

def make_fix_holes_widget(viewer: napari.Viewer) -> Container:
    from qtpy.QtWidgets import QComboBox, QWidget, QHBoxLayout, QLabel as QLabel_
    from magicgui.widgets import ComboBox as MgComboBox, SpinBox, FloatSpinBox

    # Label layer dropdown
    qt_label_combo = QComboBox()
    label_wrapper = QWidget()
    label_layout = QHBoxLayout(label_wrapper)
    label_layout.setContentsMargins(0, 0, 0, 0)
    label_layout.addWidget(QLabel_("Label layer"))
    label_layout.addWidget(qt_label_combo)

    # Hole mask layer dropdown
    qt_hole_combo = QComboBox()
    hole_wrapper = QWidget()
    hole_layout = QHBoxLayout(hole_wrapper)
    hole_layout.setContentsMargins(0, 0, 0, 0)
    hole_layout.addWidget(QLabel_("Hole mask"))
    hole_layout.addWidget(qt_hole_combo)

    method_combo = MgComboBox(label="Method", choices=["2d_hole_fill", "close", "alpha_wrap"], value="2d_hole_fill")
    closing_kernel_spin = SpinBox(label="Closing kernel", min=1, max=20, value=2)
    alpha_spin = FloatSpinBox(label="Alpha", min=0.001, max=10.0, value=0.05, step=0.01)
    mask_dilate_spin = SpinBox(label="Mask dilate", min=0, max=50, value=0)
    iterations_spin = SpinBox(label="Iterations", min=1, max=20, value=1)
    btn = PushButton(text="Fix")

    def _refresh_choices(_=None):
        for combo in (qt_label_combo, qt_hole_combo):
            current = combo.currentText()
            combo.clear()
            names = [l.name for l in viewer.layers if isinstance(l, napari.layers.Labels)]
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)

    viewer.layers.events.inserted.connect(_refresh_choices)
    viewer.layers.events.removed.connect(_refresh_choices)

    def _on_fix():
        args = STATE.args
        label_name = qt_label_combo.currentText()
        hole_name = qt_hole_combo.currentText()
        if not label_name:
            print("[WARN] No label layer selected")
            return
        if not hole_name:
            print("[WARN] No hole mask layer selected")
            return

        label_layer = None
        hole_layer = None
        for l in viewer.layers:
            if l.name == label_name:
                label_layer = l
            if l.name == hole_name:
                hole_layer = l
        if label_layer is None:
            print(f"[WARN] Layer '{label_name}' not found")
            return
        if hole_layer is None:
            print(f"[WARN] Layer '{hole_name}' not found")
            return

        method = method_combo.value
        closing_kernel = closing_kernel_spin.value
        alpha = alpha_spin.value
        mask_dilate = mask_dilate_spin.value
        iterations = iterations_spin.value

        label_data = np.asarray(label_layer.data)
        hole_data = np.asarray(hole_layer.data).astype(bool)

        fixed = label_data
        for _ in range(iterations):
            if method == "2d_hole_fill":
                fixed = _fix_2d_hole_fill(
                    fixed, hole_data,
                    fg_value=args.fg_value,
                    ignore_value=args.ignore_value,
                    mask_dilate=mask_dilate,
                )
            elif method == "close":
                fixed = _fix_morph_close(
                    fixed, hole_data,
                    fg_value=args.fg_value,
                    ignore_value=args.ignore_value,
                    kernel=closing_kernel,
                    mask_dilate=mask_dilate,
                )
            elif method == "alpha_wrap":
                fixed = _fix_alpha_wrap(
                    fixed, hole_data,
                    fg_value=args.fg_value,
                    ignore_value=args.ignore_value,
                    alpha=alpha,
                    mask_dilate=mask_dilate,
                )

        label_layer.data = fixed
        print(f"[OK] Applied {method} x{iterations}")

    btn.changed.connect(_on_fix)
    _refresh_choices()

    container = Container(
        widgets=[method_combo, closing_kernel_spin, alpha_spin, mask_dilate_spin, iterations_spin, btn],
        labels=True,
    )
    container.native.layout().insertWidget(0, hole_wrapper)
    container.native.layout().insertWidget(0, label_wrapper)
    return container


# ---------------------------------------------------------------------------
# Connected Components widget
# ---------------------------------------------------------------------------

def make_components_widget(viewer: napari.Viewer) -> Container:
    from qtpy.QtWidgets import QComboBox, QWidget, QHBoxLayout
    from magicgui.widgets import create_widget

    # Use a raw Qt combo box to avoid magicgui categorical issues.
    qt_combo = QComboBox()
    combo_wrapper = QWidget()
    combo_layout = QHBoxLayout(combo_wrapper)
    combo_layout.setContentsMargins(0, 0, 0, 0)
    combo_layout.addWidget(qt_combo)

    btn = PushButton(text="Compute Components")

    def _refresh_choices(_=None):
        current = qt_combo.currentText()
        qt_combo.clear()
        names = [l.name for l in viewer.layers if isinstance(l, napari.layers.Labels)]
        qt_combo.addItems(names)
        if current in names:
            qt_combo.setCurrentText(current)

    viewer.layers.events.inserted.connect(_refresh_choices)
    viewer.layers.events.removed.connect(_refresh_choices)

    def _on_compute():
        layer_name = qt_combo.currentText()
        if not layer_name:
            print("[WARN] No layer selected")
            return
        layer = None
        for l in viewer.layers:
            if l.name == layer_name:
                layer = l
                break
        if layer is None or not isinstance(layer, napari.layers.Labels):
            print(f"[WARN] Labels layer '{layer_name}' not found")
            return

        data = np.asarray(layer.data)
        binary = (data > 0).astype(np.uint8)
        labels, n = cc3d.connected_components(binary, connectivity=26, return_N=True)
        print(f"[INFO] {n} connected component(s) (26-conn) in '{layer_name}'")

        out_name = f"{layer_name}_cc26"
        existing = [l for l in viewer.layers if l.name == out_name]
        for l in existing:
            viewer.layers.remove(l)
        viewer.add_labels(labels, name=out_name)

    btn.changed.connect(_on_compute)
    _refresh_choices()

    container = Container(widgets=[btn], labels=False)
    container.native.layout().insertWidget(0, combo_wrapper)
    return container


# ---------------------------------------------------------------------------
# Ignore Label widget
# ---------------------------------------------------------------------------

def make_ignore_widget(viewer: napari.Viewer) -> Container:
    btn_remove = PushButton(text="Remove Ignore")
    btn_restore = PushButton(text="Restore Ignore")

    def _get_selected_labels_layer():
        sel = list(viewer.layers.selection)
        for layer in sel:
            if isinstance(layer, napari.layers.Labels):
                return layer
        return None

    def _on_remove() -> None:
        args = STATE.args
        if args.ignore_value is None:
            print("[WARN] No ignore_value configured")
            return
        label_layer = _get_selected_labels_layer()
        if label_layer is None:
            print("[WARN] Select a Labels layer first")
            return
        data = np.asarray(label_layer.data)
        mask = data == args.ignore_value
        if not mask.any():
            print(f"[INFO] No ignore voxels (value={args.ignore_value}) in '{label_layer.name}'")
            return
        STATE._saved_ignore_mask = mask
        STATE._saved_ignore_layer_name = label_layer.name
        data = data.copy()
        data[mask] = 0
        label_layer.data = data
        print(f"[OK] Removed {int(mask.sum())} ignore voxels (value={args.ignore_value}) from '{label_layer.name}'")

    def _on_restore() -> None:
        args = STATE.args
        if STATE._saved_ignore_mask is None:
            print("[WARN] No saved ignore mask to restore")
            return
        layer_name = getattr(STATE, '_saved_ignore_layer_name', None)
        label_layer = None
        if layer_name is not None:
            for layer in viewer.layers:
                if layer.name == layer_name:
                    label_layer = layer
                    break
        if label_layer is None:
            print(f"[WARN] Original layer '{layer_name}' no longer exists")
            return
        data = np.asarray(label_layer.data).copy()
        data[STATE._saved_ignore_mask] = args.ignore_value
        label_layer.data = data
        print(f"[OK] Restored ignore voxels (value={args.ignore_value}) in '{label_layer.name}'")

    btn_remove.changed.connect(_on_remove)
    btn_restore.changed.connect(_on_restore)
    return Container(widgets=[btn_remove, btn_restore], labels=False)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Napari viewer/editor for hole detection & repair.")
    ap.add_argument("--root-dir", type=Path, required=True,
                    help="Folder containing sample subfolders.")
    ap.add_argument("--suffix1", type=str, required=True,
                    help="Filename suffix for the label layer (editable).")
    ap.add_argument("--suffix2", type=str, required=True,
                    help="Filename suffix for the image layer.")
    ap.add_argument("--fg-value", type=int, default=1,
                    help="Foreground label value (default: 1).")
    ap.add_argument("--ignore-value", type=int, default=2,
                    help="Ignore label value (default: 2).")
    ap.add_argument("--fg-connectivity", type=int, default=6, choices=[6, 18, 26],
                    help="FG connectivity for hole detection (default: 6).")
    ap.add_argument("--bg-connectivity", type=int, default=6, choices=[6, 18, 26],
                    help="BG connectivity for hole detection (default: 6).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root_dir
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"No subfolders found in {root}")

    STATE.folders = folders
    STATE.current_index = 0
    STATE.args = args

    viewer = napari.Viewer()

    # Load first folder
    load_folder(viewer, 0)

    # Dock widgets
    viewer.window.add_dock_widget(make_nav_widget(viewer), area="right", name="Navigation")
    viewer.window.add_dock_widget(make_save_widget(viewer), area="right", name="Save")
    viewer.window.add_dock_widget(make_detect_holes_widget(viewer), area="right", name="Detect Holes")
    viewer.window.add_dock_widget(make_fix_holes_widget(viewer), area="right", name="Fix Holes")
    viewer.window.add_dock_widget(make_ignore_widget(viewer), area="right", name="Ignore Label")
    viewer.window.add_dock_widget(make_components_widget(viewer), area="right", name="Components")

    napari.run()


if __name__ == "__main__":
    main()
