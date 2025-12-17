#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import napari
import numpy as np
from skimage import io
from skimage.segmentation import expand_labels
import cc3d
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure, convolve
from magicgui import magicgui
from napari.utils.notifications import show_info
from napari._qt.qt_viewer import QtViewer
from napari.components.viewer_model import ViewerModel
from numba import njit, prange
from qtpy.QtWidgets import QSplitter, QWidget, QVBoxLayout, QGridLayout, QCheckBox, QShortcut, QApplication, QPushButton
from qtpy.QtGui import QKeySequence
from qtpy.QtCore import Qt

# Numba-accelerated function to find voxels touching 2+ different component labels
@njit(parallel=True, cache=True)
def _find_multi_touching_numba(candidate_coords_z, candidate_coords_y, candidate_coords_x,
                                component_labels):
    """Find which candidate voxels touch 2+ different non-zero component labels.

    Args:
        candidate_coords_z/y/x: Arrays of candidate voxel coordinates
        component_labels: 3D array of component labels

    Returns:
        Boolean array indicating which candidates touch 2+ components
    """
    n_candidates = len(candidate_coords_z)
    result = np.zeros(n_candidates, dtype=np.bool_)

    shape_z, shape_y, shape_x = component_labels.shape

    # 6-connected face offsets
    offsets = np.array([
        [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]
    ], dtype=np.int32)

    for i in prange(n_candidates):
        z, y, x = candidate_coords_z[i], candidate_coords_y[i], candidate_coords_x[i]

        first_label = 0
        has_second = False

        for j in range(6):
            nz = z + offsets[j, 0]
            ny = y + offsets[j, 1]
            nx = x + offsets[j, 2]

            if 0 <= nz < shape_z and 0 <= ny < shape_y and 0 <= nx < shape_x:
                lbl = component_labels[nz, ny, nx]
                if lbl > 0:
                    if first_label == 0:
                        first_label = lbl
                    elif lbl != first_label:
                        has_second = True
                        break

        result[i] = has_second

    return result


# 8 corner neighbors in 3D (differ by ±1 in ALL three axes)
CORNER_OFFSETS = [
    (-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
    (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1),
]


class QtViewerWrap(QtViewer):
    """Wrapper for secondary QtViewer that delegates file operations to main viewer."""
    def __init__(self, main_window, viewer_model):
        super().__init__(viewer_model)
        self.main_window = main_window

    def _qt_open(self, filenames, stack, plugin=None, layer_type=None, **kwargs):
        """Delegate drag-and-drop to main viewer."""
        self.main_window._qt_viewer._qt_open(filenames, stack, plugin, layer_type, **kwargs)


class MultiViewerWidget(QWidget):
    """
    Widget containing 2 napari viewers side by side:
    - Left: XY plane (main viewer's canvas) - top-down slice view
    - Right: 3D view
    """

    def __init__(self, main_viewer, sync_enabled=True):
        super().__init__()
        self.main_viewer = main_viewer
        self.sync_enabled = sync_enabled
        self._block = False  # Prevent feedback loops

        # Create 3D viewer model
        self.viewer_model_3d = ViewerModel(title='3D View')

        # Create QtViewer wrapper for 3D view
        self.qt_viewer_3d = QtViewerWrap(main_viewer.window, self.viewer_model_3d)

        # Setup layout with QSplitter
        self._setup_layout()

        # Connect synchronization events
        self._connect_sync_events()

    def _setup_layout(self):
        """Create side-by-side layout with resizable splitter."""
        # Horizontal splitter: main view on left, 3D on right
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Get the main viewer's canvas widget
        main_canvas = self.main_viewer.window._qt_viewer
        main_splitter.addWidget(main_canvas)  # XY view (left)
        main_splitter.addWidget(self.qt_viewer_3d)  # 3D view (right)

        # Set equal sizes
        main_splitter.setSizes([500, 500])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)

        self.main_splitter = main_splitter

    def _connect_sync_events(self):
        """Connect events for layer and dimension synchronization."""
        # Layer events on main viewer
        self.main_viewer.layers.events.inserted.connect(self._on_layer_added)
        self.main_viewer.layers.events.removed.connect(self._on_layer_removed)
        self.main_viewer.layers.events.moved.connect(self._on_layer_moved)

        # Dimension synchronization
        self.main_viewer.dims.events.current_step.connect(self._on_dims_changed)
        self.viewer_model_3d.dims.events.current_step.connect(self._on_dims_changed)

    def _update_orthogonal_orders(self):
        """Set 3D view to show all dimensions."""
        order = list(self.main_viewer.dims.order)
        if len(order) < 3:
            return

        # 3D view: same order but set ndisplay=3
        self.viewer_model_3d.dims.order = tuple(order)
        self.viewer_model_3d.dims.ndisplay = 3

    def _on_dims_changed(self, event):
        """Synchronize dimension position across viewers."""
        if not self.sync_enabled or self._block:
            return

        try:
            self._block = True
            source = event.source
            new_step = event.value

            for model in [self.main_viewer, self.viewer_model_3d]:
                if model.dims is source:
                    continue
                # Only sync if same number of dimensions
                if len(model.dims.current_step) == len(new_step):
                    model.dims.current_step = new_step
        finally:
            self._block = False

    def _on_layer_added(self, event):
        """Add layer to 3D viewer when added to main."""
        layer = event.value
        index = event.index

        # Create copy for 3D viewer (sharing data array)
        layer_data = layer.as_layer_data_tuple()
        from napari.layers import Layer
        copy = Layer.create(*layer_data)
        copy.metadata['viewer_name'] = '3d'
        self.viewer_model_3d.layers.insert(index, copy)

        # Connect data sync for label edits
        layer.events.set_data.connect(self._on_layer_data_changed)
        layer.events.data.connect(self._on_layer_data_changed)

        # Sync layer properties (one-way from main to 3D)
        for prop in ['visible', 'opacity', 'blending']:
            if hasattr(layer.events, prop):
                getattr(layer.events, prop).connect(
                    lambda evt, p=prop: self._sync_property(evt, p)
                )

        # Sync mode and selected_label bidirectionally (tool changes should sync across all viewers)
        if hasattr(layer.events, 'mode'):
            layer.events.mode.connect(self._sync_mode_all)
        if hasattr(layer.events, 'selected_label'):
            layer.events.selected_label.connect(self._sync_selected_label_all)

        # Also connect from 3D viewer back
        for sec_layer in self.viewer_model_3d.layers:
            if sec_layer.name == layer.name:
                if hasattr(sec_layer.events, 'mode'):
                    sec_layer.events.mode.connect(self._sync_mode_all)
                if hasattr(sec_layer.events, 'selected_label'):
                    sec_layer.events.selected_label.connect(self._sync_selected_label_all)

        self._update_orthogonal_orders()

    def _on_layer_data_changed(self, event):
        """Sync data changes (edits) to 3D viewer."""
        if self._block:
            return
        try:
            self._block = True
            source_layer = event.source
            layer_name = source_layer.name
            for target in self.viewer_model_3d.layers:
                if target.name == layer_name:
                    target.data = source_layer.data
                    target.refresh()
                    break
        finally:
            self._block = False

    def _sync_property(self, event, prop_name):
        """Sync layer property changes."""
        if self._block:
            return
        try:
            self._block = True
            source = event.source
            value = getattr(source, prop_name)
            for target in self.viewer_model_3d.layers:
                if target.name == source.name:
                    setattr(target, prop_name, value)
                    break
        finally:
            self._block = False

    def _sync_mode_all(self, event):
        """Sync layer mode (tool) changes across ALL viewers bidirectionally."""
        if self._block:
            return
        try:
            self._block = True
            source = event.source
            mode = source.mode
            layer_name = source.name

            # Sync to all viewers including main
            all_models = [self.main_viewer, self.viewer_model_3d]
            for model in all_models:
                for target in model.layers:
                    if target.name == layer_name and target is not source:
                        target.mode = mode
                        break
        finally:
            self._block = False

    def _sync_selected_label_all(self, event):
        """Sync selected_label changes across ALL viewers bidirectionally."""
        if self._block:
            return
        try:
            self._block = True
            source = event.source
            selected_label = source.selected_label
            layer_name = source.name

            # Sync to all viewers including main
            all_models = [self.main_viewer, self.viewer_model_3d]
            for model in all_models:
                for target in model.layers:
                    if target.name == layer_name and target is not source:
                        target.selected_label = selected_label
                        break
        finally:
            self._block = False

    def _on_layer_removed(self, event):
        """Remove layer from 3D viewer."""
        index = event.index
        if index < len(self.viewer_model_3d.layers):
            self.viewer_model_3d.layers.pop(index)

    def _on_layer_moved(self, event):
        """Sync layer order changes."""
        dest = event.new_index if event.new_index < event.index else event.new_index + 1
        if event.index < len(self.viewer_model_3d.layers):
            self.viewer_model_3d.layers.move(event.index, dest)

    def set_sync_enabled(self, enabled):
        """Enable or disable position synchronization."""
        if isinstance(enabled, int):  # Qt checkbox state (0=unchecked, 2=checked)
            enabled = enabled == 2
        self.sync_enabled = enabled

    def get_all_viewer_models(self):
        """Return all viewer models for operations that need to affect all."""
        return [self.main_viewer, self.viewer_model_3d]


class ImageLabelViewer:
    def __init__(self, image_dir, label_dir, label_suffix="", output_dir=None,
                 mergers_csv=None, tiny_csv=None, zero_ignore_label=True, copy_on_skip=True,
                 dust_max_size=250):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_suffix = label_suffix
        # Default to timestamped folder in cwd if no output dir specified
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path.cwd() / f"output_{timestamp}"
        self.zero_ignore_label = zero_ignore_label
        self.copy_on_skip = copy_on_skip
        self.dust_max_size = dust_max_size

        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load CSV sample IDs
        self.mergers_ids = self._load_csv_ids(mergers_csv) if mergers_csv else set()
        self.tiny_ids = self._load_csv_ids(tiny_csv) if tiny_csv else set()

        # Get all tif files
        self.image_files = sorted([f for f in self.image_dir.glob("*.tif")
                                  if f.is_file()])
        if not self.image_files:
            self.image_files = sorted([f for f in self.image_dir.glob("*.tiff")
                                      if f.is_file()])

        # Filter to only images that have corresponding labels
        self.image_files = [f for f in self.image_files if self.get_label_path(f) is not None]

        self.current_index = 0
        self.viewer = None
        self.current_label_layer = None
        self.multi_viewer_widget = None
        self.viewer_models = []  # [main, xz, yz, 3d]
        self.sync_enabled = True
        self.ignore_mask = None  # Store ignore mask for toggle functionality
        self.ignore_visible = True  # Track ignore label visibility

        # Component navigation state
        self.component_index = 0  # Current component index (0-based)
        self.component_list = []  # List of unique component labels
        self.bbox_layer = None  # Reference to shapes layer for bounding box
        self.component_counter_label = None  # Label widget for "Component X of Y"

        # Flagged/skipped samples CSV paths (written to output directory)
        self.flagged_csv_path = self.output_dir / "flagged.csv" if self.output_dir else None
        self.skipped_csv_path = self.output_dir / "skipped.csv" if self.output_dir else None

    def _load_csv_ids(self, csv_path):
        """Load sample IDs from a CSV file."""
        import csv
        ids = set()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(row['sample_id'])
        return ids

    def _get_sample_id(self, image_path):
        """Extract sample ID from image filename."""
        # Assume the stem is the sample ID or contains it
        return image_path.stem.split('_')[0]

    def get_csv_membership(self):
        """Get which CSVs the current sample belongs to."""
        if self.current_index >= len(self.image_files):
            return []
        image_path = self.image_files[self.current_index]
        sample_id = self._get_sample_id(image_path)

        membership = []
        if sample_id in self.mergers_ids:
            membership.append("MERGERS")
        if sample_id in self.tiny_ids:
            membership.append("TINY")
        return membership

    def update_csv_label(self):
        """Update the CSV membership in the viewer title and widget."""
        membership = self.get_csv_membership()
        csv_str = f" - [{', '.join(membership)}]" if membership else ""
        self.viewer.title = f"Image {self.current_index + 1}/{len(self.image_files)}{csv_str}"
        # Update the widget if it exists
        if hasattr(self, 'csv_label_widget'):
            if membership:
                self.csv_label_widget.setText(", ".join(membership))
            else:
                self.csv_label_widget.setText("(none)")

    def update_component_count_display(self):
        """Update the component count text overlay in viewer windows."""
        if self.current_label_layer is None:
            count = 0
        else:
            labels = self.current_label_layer.data
            unique = np.unique(labels)
            # Exclude 0 (background) and 150 (ignore label)
            count = len([l for l in unique if l != 0 and l != 150])

        text = f"Components: {count}"

        # Update main viewer
        self.viewer.text_overlay.text = text
        self.viewer.text_overlay.visible = True

        # Update 3D viewer
        if self.multi_viewer_widget:
            self.multi_viewer_widget.viewer_model_3d.text_overlay.text = text
            self.multi_viewer_widget.viewer_model_3d.text_overlay.visible = True

    def compute_connected_components(self, label_data):
        """Compute 26-connected components on label volume."""
        # Binarize the label data (non-zero -> 1)
        binary = (label_data > 0).astype(np.uint8)
        # Compute connected components with 26-connectivity
        labeled = cc3d.connected_components(binary, connectivity=26)
        return labeled

    def recompute_labels(self):
        """Recompute connected components on current label layer."""
        if self.current_label_layer is None:
            show_info("No label layer to recompute")
            return

        label_data = self.current_label_layer.data
        new_labels = self.compute_connected_components(label_data)
        self.current_label_layer.data = new_labels
        num_components = len(np.unique(new_labels)) - 1  # Subtract 1 for background (0)
        show_info(f"Recomputed: found {num_components} connected components")
        self.update_component_count_display()
        self.update_component_list()

    # ==================== Component Navigation Methods ====================

    def update_component_list(self):
        """Update the list of unique component labels (excluding background and ignore)."""
        if self.current_label_layer is None:
            self.component_list = []
            self.component_index = 0
            self.update_component_bbox()
            return

        labels = self.current_label_layer.data
        unique = np.unique(labels)
        # Exclude 0 (background) and 150 (ignore label)
        self.component_list = sorted([int(l) for l in unique if l != 0 and l != 150])

        # Clamp index to valid range
        if len(self.component_list) == 0:
            self.component_index = 0
        elif self.component_index >= len(self.component_list):
            self.component_index = len(self.component_list) - 1

        self.update_component_bbox()

    def update_component_bbox(self):
        """Update the bounding box display for the current component."""
        # Update counter label
        if self.component_counter_label is not None:
            if len(self.component_list) == 0:
                self.component_counter_label.setText("No components")
            else:
                self.component_counter_label.setText(f"Component {self.component_index + 1} of {len(self.component_list)}")

        # Clear bbox if no components
        if len(self.component_list) == 0 or self.current_label_layer is None:
            if self.bbox_layer is not None:
                self.bbox_layer.data = []
            return

        # Get current component label
        current_label = self.component_list[self.component_index]
        label_data = self.current_label_layer.data

        # Find voxels belonging to this component
        coords = np.where(label_data == current_label)
        if len(coords[0]) == 0:
            if self.bbox_layer is not None:
                self.bbox_layer.data = []
            return

        # Compute bounding box (z, y, x order for napari)
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()

        # Create 3D bounding box as a rectangular prism (8 corners, 12 edges)
        # Define the 8 corners of the box
        corners = np.array([
            [z_min, y_min, x_min],
            [z_min, y_min, x_max],
            [z_min, y_max, x_min],
            [z_min, y_max, x_max],
            [z_max, y_min, x_min],
            [z_max, y_min, x_max],
            [z_max, y_max, x_min],
            [z_max, y_max, x_max],
        ])

        # Define the 12 edges as line segments (pairs of corner indices)
        edges = [
            # Bottom face (z_min)
            [corners[0], corners[1]],
            [corners[1], corners[3]],
            [corners[3], corners[2]],
            [corners[2], corners[0]],
            # Top face (z_max)
            [corners[4], corners[5]],
            [corners[5], corners[7]],
            [corners[7], corners[6]],
            [corners[6], corners[4]],
            # Vertical edges
            [corners[0], corners[4]],
            [corners[1], corners[5]],
            [corners[2], corners[6]],
            [corners[3], corners[7]],
        ]

        # Create or update the shapes layer
        if self.bbox_layer is None:
            self.bbox_layer = self.viewer.add_shapes(
                edges,
                shape_type='line',
                edge_color='red',
                edge_width=2,
                name='component_bbox',
            )
        else:
            self.bbox_layer.data = edges

    def next_component(self):
        """Navigate to the next component."""
        if len(self.component_list) == 0:
            show_info("No components to navigate")
            return

        self.component_index = (self.component_index + 1) % len(self.component_list)
        self.update_component_bbox()

    def previous_component(self):
        """Navigate to the previous component."""
        if len(self.component_list) == 0:
            show_info("No components to navigate")
            return

        self.component_index = (self.component_index - 1) % len(self.component_list)
        self.update_component_bbox()

    def delete_current_component(self):
        """Delete the currently selected component."""
        if len(self.component_list) == 0:
            show_info("No component selected to delete")
            return

        if self.current_label_layer is None:
            return

        current_label = self.component_list[self.component_index]
        label_data = self.current_label_layer.data.copy()

        # Zero out the current component
        label_data[label_data == current_label] = 0
        self.current_label_layer.data = label_data

        show_info(f"Deleted component {current_label}")

        # Recompute connected components (this will also update component list)
        self.recompute_labels()

    # ==================== End Component Navigation ====================

    def flag_current_sample(self):
        """Flag the current sample by appending its ID to flagged.csv."""
        print(f"flag_current_sample called, flagged_csv_path={self.flagged_csv_path}")
        if self.flagged_csv_path is None:
            show_info("No output directory specified - cannot flag sample")
            return

        if self.current_index >= len(self.image_files):
            show_info("No sample to flag")
            return

        import csv
        image_path = self.image_files[self.current_index]
        sample_id = self._get_sample_id(image_path)
        print(f"Flagging sample_id={sample_id}")

        # Check if file exists and if sample is already flagged
        file_exists = self.flagged_csv_path.exists()
        existing_ids = set()
        if file_exists:
            with open(self.flagged_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_ids = {row['sample_id'] for row in reader}

        if sample_id in existing_ids:
            show_info(f"Sample {sample_id} already flagged")
            return

        # Append to CSV (create with header if new)
        with open(self.flagged_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'sample_id': sample_id})

        print(f"Wrote {sample_id} to {self.flagged_csv_path}")
        show_info(f"Flagged sample: {sample_id}")

    def record_skipped_sample(self):
        """Record the current sample as skipped by appending its ID to skipped.csv."""
        if self.skipped_csv_path is None:
            return  # Silently skip if no output directory

        if self.current_index >= len(self.image_files):
            return

        import csv
        image_path = self.image_files[self.current_index]
        sample_id = self._get_sample_id(image_path)

        # Check if file exists and if sample is already recorded
        existing_ids = set()
        if self.skipped_csv_path.exists():
            with open(self.skipped_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_ids = {row['sample_id'] for row in reader}

        if sample_id in existing_ids:
            return  # Already recorded

        # Append to CSV (create with header if new)
        write_header = not self.skipped_csv_path.exists()
        with open(self.skipped_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id'])
            if write_header:
                writer.writeheader()
            writer.writerow({'sample_id': sample_id})

    def find_small_components(self, connectivity, max_size):
        """Find connected components smaller than max_size.

        Returns a mask of small components and the labeled array.
        """
        if self.current_label_layer is None:
            return None, None

        label_data = self.current_label_layer.data
        binary = (label_data > 0).astype(np.uint8)
        labeled = cc3d.connected_components(binary, connectivity=connectivity)

        # Get component sizes
        stats = cc3d.statistics(labeled)
        component_sizes = stats['voxel_counts']

        # Find small component IDs (skip 0 which is background)
        small_ids = []
        for comp_id, size in enumerate(component_sizes):
            if comp_id > 0 and size < max_size:
                small_ids.append(comp_id)

        # Create mask of small components
        small_mask = np.isin(labeled, small_ids).astype(np.uint8)

        return small_mask, labeled, small_ids

    def highlight_small_components(self, connectivity, max_size):
        """Highlight small components by creating a new label layer."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        result = self.find_small_components(connectivity, max_size)
        if result[0] is None:
            return

        small_mask, labeled, small_ids = result

        # Remove existing small components layer if present
        for layer in list(self.viewer.layers):
            if layer.name == "Small Components":
                self.viewer.layers.remove(layer)

        if len(small_ids) == 0:
            show_info(f"No components found with size < {max_size}")
            return

        # Create labels from small mask (relabel for visibility)
        small_labeled = cc3d.connected_components(small_mask, connectivity=connectivity)

        self.viewer.add_labels(small_labeled, name="Small Components", opacity=0.7)
        total_voxels = np.sum(small_mask)
        show_info(f"Found {len(small_ids)} small components ({total_voxels} voxels)")

    def remove_small_components(self, connectivity, max_size):
        """Remove small components from the label layer."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        result = self.find_small_components(connectivity, max_size)
        if result[0] is None:
            return

        small_mask, labeled, small_ids = result

        if len(small_ids) == 0:
            show_info(f"No components found with size < {max_size}")
            return

        # Remove small components from label data
        label_data = self.current_label_layer.data.copy()
        label_data[small_mask > 0] = 0
        self.current_label_layer.data = label_data

        # Remove highlight layer if present
        for layer in list(self.viewer.layers):
            if layer.name == "Small Components":
                self.viewer.layers.remove(layer)

        total_voxels = np.sum(small_mask)
        show_info(f"Removed {len(small_ids)} small components ({total_voxels} voxels)")
        self.update_component_count_display()
        self.update_component_list()

    def delete_selected_component(self):
        """Delete the currently selected label's 26-connected component."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        selected_label = self.current_label_layer.selected_label
        if selected_label == 0:
            show_info("No label selected (background selected)")
            return

        label_data = self.current_label_layer.data
        # Check if the selected label exists in the data
        if selected_label not in label_data:
            show_info(f"Label {selected_label} not found in current data")
            return

        # Create a mask of just this label value
        mask = (label_data == selected_label).astype(np.uint8)

        # Find 26-connected components within this mask
        labeled = cc3d.connected_components(mask, connectivity=26)

        # Get the component at the cursor position (if available) or remove all with this label
        # For simplicity, we remove all voxels with the selected label value
        label_data = label_data.copy()
        label_data[mask > 0] = 0
        self.current_label_layer.data = label_data

        num_voxels = np.sum(mask)
        show_info(f"Deleted label {selected_label} ({num_voxels} voxels)")
        self.update_component_count_display()
        self.update_component_list()

    def expand_current_labels(self, distance=2):
        """Expand labels and create new 'expanded' layer."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        label_data = self.current_label_layer.data

        expanded = expand_labels(label_data, distance=distance)

        # Add 3-voxel border with value 200 on all faces
        expanded[:3, :, :] = 200
        expanded[-3:, :, :] = 200
        expanded[:, :3, :] = 200
        expanded[:, -3:, :] = 200
        expanded[:, :, :3] = 200
        expanded[:, :, -3:] = 200

        # Remove existing expanded layer if present
        for layer in list(self.viewer.layers):
            if layer.name == "expanded":
                self.viewer.layers.remove(layer)

        self.viewer.add_labels(expanded, name="expanded", opacity=0.7)
        show_info(f"Expanded labels with distance={distance}")

    def copy_selected_from_expanded(self):
        """Copy selected label from 'expanded' layer back to original."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        # Find expanded layer
        expanded_layer = None
        for layer in self.viewer.layers:
            if layer.name == "expanded":
                expanded_layer = layer
                break

        if expanded_layer is None:
            show_info("No 'expanded' layer found")
            return

        # Get selected label from the expanded layer (which is active after Ctrl+G)
        selected = expanded_layer.selected_label
        if selected == 0:
            show_info("No label selected")
            return

        # Copy selected label from expanded to original
        expanded_data = expanded_layer.data
        label_data = self.current_label_layer.data.copy()

        mask = expanded_data == selected
        label_data[mask] = selected

        # Also copy the 3-voxel border with value 200
        label_data[:3, :, :] = 200
        label_data[-3:, :, :] = 200
        label_data[:, :3, :] = 200
        label_data[:, -3:, :] = 200
        label_data[:, :, :3] = 200
        label_data[:, :, -3:] = 200

        self.current_label_layer.data = label_data

        # Remove the expanded layer and make the label layer active
        self.viewer.layers.remove(expanded_layer)
        self.viewer.layers.selection.active = self.current_label_layer

        show_info(f"Copied label {selected} from expanded layer")
        self.update_component_count_display()
        self.update_component_list()

    def _shift_labels(self, labels, dz, dy, dx):
        """Shift label array and zero out wrapped boundaries."""
        shifted = np.roll(np.roll(np.roll(labels, -dz, axis=0), -dy, axis=1), -dx, axis=2)

        # Zero out wrapped regions to prevent false comparisons
        if dz > 0:
            shifted[-dz:, :, :] = 0
        elif dz < 0:
            shifted[:-dz, :, :] = 0
        if dy > 0:
            shifted[:, -dy:, :] = 0
        elif dy < 0:
            shifted[:, :-dy, :] = 0
        if dx > 0:
            shifted[:, :, -dx:] = 0
        elif dx < 0:
            shifted[:, :, :-dx] = 0

        return shifted

    def _find_multi_component_touching_vectorized(self, candidate_mask, component_labels):
        """Vectorized detection of voxels touching 2+ different components.

        Uses array shifts instead of per-voxel loops.

        Args:
            candidate_mask: Boolean mask of candidate voxels to check
            component_labels: Labeled array of components

        Returns:
            Boolean mask where candidate voxels touch 2+ different component labels
        """
        face_offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

        # Get neighbor labels at each face direction
        neighbor_labels = []
        for dz, dy, dx in face_offsets:
            shifted = self._shift_labels(component_labels, dz, dy, dx)
            neighbor_labels.append(shifted)

        # Stack to shape (Z, Y, X, 6)
        stacked = np.stack(neighbor_labels, axis=-1)

        # Mask to only consider candidate positions
        candidate_neighbors = stacked * candidate_mask[..., np.newaxis].astype(stacked.dtype)

        # Count unique non-zero labels per voxel using sorting
        sorted_neighbors = np.sort(candidate_neighbors, axis=-1)

        # A voxel touches multiple components if sorted array has 2+ distinct non-zero values
        nonzero = sorted_neighbors > 0
        # Count transitions to different values (excluding 0->nonzero transitions handled separately)
        changes = np.diff(sorted_neighbors, axis=-1) != 0
        changes = changes & nonzero[..., 1:]  # Only count changes between non-zero values

        # First non-zero value counts as 1
        first_nonzero = nonzero[..., 0]
        unique_count = changes.sum(axis=-1) + first_nonzero.astype(int)

        # Return mask where count >= 2
        return (unique_count >= 2) & candidate_mask

    def _find_articulation_voxels_batch(self, eroded_labels, candidates_mask):
        """Find which candidate voxels are true articulation points (bridges).

        Uses dilation: for each eroded component, find candidates that would
        connect it to other components when re-added.

        Args:
            eroded_labels: Labeled array of eroded components
            candidates_mask: Boolean mask of candidate bridge voxels

        Returns:
            Boolean mask of articulation (bridge) voxels
        """
        n_components = eroded_labels.max()
        articulation = np.zeros_like(candidates_mask, dtype=bool)

        if n_components <= 1:
            return articulation

        struct_6 = generate_binary_structure(3, 1)  # 6-connectivity

        # Pre-compute dilated masks for each component
        component_dilated = {}
        for comp_id in range(1, n_components + 1):
            comp_mask = eroded_labels == comp_id
            component_dilated[comp_id] = binary_dilation(comp_mask, structure=struct_6)

        # For each pair of components, find candidates that bridge them
        for comp_id in range(1, n_components + 1):
            # Candidates adjacent to this component (but not part of it)
            comp_mask = eroded_labels == comp_id
            adjacent_candidates = component_dilated[comp_id] & candidates_mask & ~comp_mask

            # Check if these candidates also touch other components
            for other_id in range(comp_id + 1, n_components + 1):
                # Voxels that bridge these two specific components
                bridges = adjacent_candidates & component_dilated[other_id]
                articulation |= bridges

        return articulation

    def _extend_bridge_mask(self, bridge_mask, component_labels, binary, max_extensions=1):
        """Extend bridge mask to include adjacent voxels that also bridge components.

        Uses Numba-accelerated neighbor checking for efficiency.

        Args:
            bridge_mask: Initial bridge detection mask
            component_labels: Labeled array from lower connectivity
            binary: Binary foreground mask
            max_extensions: Maximum dilation iterations (default: 1, conservative for thin sheets)

        Returns:
            Extended bridge_mask
        """
        struct_6 = generate_binary_structure(3, 1)  # 6-connectivity
        extended = bridge_mask.copy()

        for _ in range(max_extensions):
            # Dilate current bridges
            dilated = binary_dilation(extended > 0, structure=struct_6)
            candidates = dilated & binary & (extended == 0)

            if not candidates.any():
                break

            # Get candidate coordinates
            candidate_coords = np.where(candidates)
            if len(candidate_coords[0]) == 0:
                break

            # Numba-accelerated: find candidates that touch 2+ different component labels
            is_bridge = _find_multi_touching_numba(
                candidate_coords[0].astype(np.int32),
                candidate_coords[1].astype(np.int32),
                candidate_coords[2].astype(np.int32),
                component_labels
            )

            if not is_bridge.any():
                break

            # Mark new bridges
            new_bridge_coords = (
                candidate_coords[0][is_bridge],
                candidate_coords[1][is_bridge],
                candidate_coords[2][is_bridge]
            )
            extended[new_bridge_coords] = 1

        return extended

    def find_corner_bridges(self, label_data, labeled_18=None, labeled_26=None):
        """Find corner-only bridges using vectorized neighbor comparison.

        Detects 26-but-not-18 connectivity (corner bridges).
        Uses fast vectorized array shifts instead of per-component dilation.

        Args:
            label_data: The label array to analyze
            labeled_18: Optional pre-computed 18-connected labels
            labeled_26: Optional pre-computed 26-connected labels

        Returns:
            bridge_mask: Binary mask of voxels to remove
            merged_component_mask: Mask of all voxels in merged components
            num_mergers: Count of merged components found
        """
        binary = (label_data > 0).astype(np.uint8)
        if labeled_18 is None:
            labeled_18 = cc3d.connected_components(binary, connectivity=18)

        bridge_mask = np.zeros_like(binary)

        # Corner offsets: differ by 1 in all 3 axes (8 neighbors)
        corner_offsets = [
            (-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
            (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1),
        ]

        # For each corner direction, find voxels with different 18-labels
        for dz, dy, dx in corner_offsets:
            shifted = self._shift_labels(labeled_18, dz, dy, dx)

            # Find where current voxel and neighbor both exist but have different labels
            different = (labeled_18 > 0) & (shifted > 0) & (labeled_18 != shifted)
            bridge_mask[different] = 1

        # Extend bridge mask to catch thicker bridges
        if np.sum(bridge_mask) > 0:
            bridge_mask = self._extend_bridge_mask(bridge_mask, labeled_18, binary)

        # Find merged components (26-components containing bridge voxels)
        if labeled_26 is None:
            labeled_26 = cc3d.connected_components(binary, connectivity=26)
        bridge_labels = np.unique(labeled_26[bridge_mask > 0])
        bridge_labels = bridge_labels[bridge_labels > 0]

        merged_component_mask = np.isin(labeled_26, bridge_labels).astype(np.uint8)
        num_mergers = len(bridge_labels)

        return bridge_mask, merged_component_mask, num_mergers

    def find_edge_bridges(self, label_data, labeled_6=None, labeled_18=None):
        """Find edge-only bridges using vectorized neighbor comparison.

        Detects 18-but-not-6 connectivity (diagonal/edge bridges).
        Uses fast vectorized array shifts instead of per-component dilation.

        Args:
            label_data: The label array to analyze
            labeled_6: Optional pre-computed 6-connected labels
            labeled_18: Optional pre-computed 18-connected labels

        Returns:
            bridge_mask: Binary mask of voxels to remove
            merged_component_mask: Mask of all voxels in merged components
            num_mergers: Count of merged components found
        """
        binary = (label_data > 0).astype(np.uint8)
        if labeled_6 is None:
            labeled_6 = cc3d.connected_components(binary, connectivity=6)

        bridge_mask = np.zeros_like(binary)

        # Edge offsets: differ by 1 in exactly 2 axes (12 neighbors)
        edge_offsets = [
            (0, -1, -1), (0, -1, +1), (0, +1, -1), (0, +1, +1),  # Y-Z plane
            (-1, 0, -1), (-1, 0, +1), (+1, 0, -1), (+1, 0, +1),  # X-Z plane
            (-1, -1, 0), (-1, +1, 0), (+1, -1, 0), (+1, +1, 0),  # X-Y plane
        ]

        # For each edge direction, find voxels with different 6-labels
        for dz, dy, dx in edge_offsets:
            shifted = self._shift_labels(labeled_6, dz, dy, dx)

            # Find where current voxel and neighbor both exist but have different labels
            different = (labeled_6 > 0) & (shifted > 0) & (labeled_6 != shifted)
            bridge_mask[different] = 1

        # Extend bridge mask to catch thicker bridges
        if np.sum(bridge_mask) > 0:
            bridge_mask = self._extend_bridge_mask(bridge_mask, labeled_6, binary)

        # Find merged components (18-components containing bridge voxels)
        if labeled_18 is None:
            labeled_18 = cc3d.connected_components(binary, connectivity=18)
        bridge_labels = np.unique(labeled_18[bridge_mask > 0])
        bridge_labels = bridge_labels[bridge_labels > 0]

        merged_component_mask = np.isin(labeled_18, bridge_labels).astype(np.uint8)
        num_mergers = len(bridge_labels)

        return bridge_mask, merged_component_mask, num_mergers

    def find_minimal_bridges(self, label_data, max_erosion=3, labeled_6=None):
        """Find minimal bridge voxels using multi-level erosion + Numba-accelerated detection.

        Algorithm:
        1. Try multiple erosion levels (1 to max_erosion) to handle varying bridge thickness
        2. For each level where erosion increases component count, find removed voxels
        3. Use Numba-accelerated function to find voxels touching 2+ eroded components
        4. Voxels touching 2+ components ARE bridges - no further verification needed

        Args:
            label_data: The label array to analyze
            max_erosion: Maximum number of erosion iterations to try (default: 3)
            labeled_6: Optional pre-computed 6-connected labels

        Returns:
            bridge_mask: Binary mask of minimal voxels to remove
        """
        binary = (label_data > 0).astype(np.uint8)
        struct_6 = generate_binary_structure(3, 1)  # 6-connectivity

        # Count initial 6-connected components
        if labeled_6 is None:
            initial_labels = cc3d.connected_components(binary, connectivity=6)
        else:
            initial_labels = labeled_6
        n_initial = initial_labels.max()

        if n_initial == 0:
            return np.zeros_like(binary)

        bridge_mask = np.zeros_like(binary, dtype=np.uint8)
        working_binary = binary.copy()

        # Try multiple erosion levels to handle varying bridge thickness
        for erosion_iter in range(1, max_erosion + 1):
            eroded = binary_erosion(working_binary, structure=struct_6, iterations=erosion_iter)

            if not eroded.any():
                break

            eroded_labels = cc3d.connected_components(eroded.astype(np.uint8), connectivity=6)
            n_eroded = eroded_labels.max()

            if n_eroded <= n_initial:
                continue  # No splits at this erosion level

            # Find removed voxels at this erosion level
            removed = working_binary.astype(bool) & ~eroded

            # Get coordinates of removed voxels
            removed_coords = np.where(removed)
            if len(removed_coords[0]) == 0:
                continue

            # Numba-accelerated: find which removed voxels touch 2+ different eroded components
            # Voxels touching 2+ components ARE bridges by definition
            is_bridge = _find_multi_touching_numba(
                removed_coords[0].astype(np.int32),
                removed_coords[1].astype(np.int32),
                removed_coords[2].astype(np.int32),
                eroded_labels
            )

            if is_bridge.any():
                # Mark bridge voxels
                bridge_coords = (
                    removed_coords[0][is_bridge],
                    removed_coords[1][is_bridge],
                    removed_coords[2][is_bridge]
                )
                bridge_mask[bridge_coords] = 1
                # Update working binary for next iteration
                working_binary[bridge_coords] = 0

        return bridge_mask

    def find_skeleton_junctions(self, label_data, min_neighbors=3):
        """Find junction voxels in 1vx skeleton structures by counting 26-connected neighbors.

        For thin skeletons, junction points where separate lines touch/merge will have
        3+ neighbors, while normal line voxels have only 1-2 neighbors.

        Args:
            label_data: The label array to analyze
            min_neighbors: Minimum neighbor count to be considered a junction (default: 3)

        Returns:
            junction_mask: Binary mask of junction voxels to remove
        """
        binary = (label_data > 0).astype(np.uint8)

        # 3x3x3 kernel for 26-connectivity (center=0 to exclude self)
        struct_26 = generate_binary_structure(3, 3).astype(np.uint8)
        struct_26[1, 1, 1] = 0

        neighbor_count = convolve(binary, struct_26, mode='constant', cval=0)
        junction_mask = (binary > 0) & (neighbor_count >= min_neighbors)

        return junction_mask.astype(np.uint8)

    def split_merges(self, max_iterations=5, skeleton_mode=False):
        """Find and remove thin bridges or junction voxels in one operation.

        Args:
            max_iterations: Maximum number of passes (default: 5). Each pass removes bridges/junctions
                           that may have been revealed by previous passes.
            skeleton_mode: If True, use skeleton junction detection for 1vx lines (removes voxels
                          with 3+ neighbors). If False, use standard bridge detection for thick
                          structures (compares connectivity levels).
        """
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        label_data = self.current_label_layer.data.copy()

        # Loop until no more bridges/junctions found or max iterations reached
        total_removed = 0
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            removed_this_iter = 0

            if skeleton_mode:
                # Skeleton mode: find and remove junction voxels (3+ neighbors)
                junction_mask = self.find_skeleton_junctions(label_data, min_neighbors=3)
                junctions_removed = np.sum(junction_mask)
                if junctions_removed > 0:
                    removed_this_iter += junctions_removed
                    label_data[junction_mask > 0] = 0
            else:
                # Normal mode: detect bridges via connectivity comparison
                # Compute all connectivity levels once per iteration
                binary = (label_data > 0).astype(np.uint8)
                labeled_6 = cc3d.connected_components(binary, connectivity=6)
                labeled_18 = cc3d.connected_components(binary, connectivity=18)
                labeled_26 = cc3d.connected_components(binary, connectivity=26)

                # Try corner bridges (26-but-not-18) with extension
                bridge_mask, _, _ = self.find_corner_bridges(
                    label_data, labeled_18=labeled_18, labeled_26=labeled_26
                )
                corner_removed = np.sum(bridge_mask)
                if corner_removed > 0:
                    removed_this_iter += corner_removed
                    label_data[bridge_mask > 0] = 0
                    # Recompute only what's needed for next stage
                    binary = (label_data > 0).astype(np.uint8)
                    labeled_6 = cc3d.connected_components(binary, connectivity=6)
                    labeled_18 = cc3d.connected_components(binary, connectivity=18)

                # Try edge bridges (18-but-not-6) with extension
                bridge_mask, _, _ = self.find_edge_bridges(
                    label_data, labeled_6=labeled_6, labeled_18=labeled_18
                )
                edge_removed = np.sum(bridge_mask)
                if edge_removed > 0:
                    removed_this_iter += edge_removed
                    label_data[bridge_mask > 0] = 0
                    # Recompute only what's needed for next stage
                    binary = (label_data > 0).astype(np.uint8)
                    labeled_6 = cc3d.connected_components(binary, connectivity=6)

                # NOTE: Erosion-based minimal bridge detection is disabled by default
                # as it can damage thin sheet-like structures (e.g., papyrus layers).
                # The corner and edge bridge detection above handles most merge cases.
                # Uncomment below only for thick/blocky segmentation data:
                #
                # if corner_removed + edge_removed < 100:
                #     bridge_mask = self.find_minimal_bridges(label_data, max_erosion=1, labeled_6=labeled_6)
                #     minimal_removed = np.sum(bridge_mask)
                #     if minimal_removed > 0:
                #         removed_this_iter += minimal_removed
                #         label_data[bridge_mask > 0] = 0

            if removed_this_iter == 0:
                break
            total_removed += removed_this_iter

        if total_removed == 0:
            show_info("No bridges found")
            return

        # Recompute connected components
        new_labels = cc3d.connected_components((label_data > 0).astype(np.uint8), connectivity=26)
        self.current_label_layer.data = new_labels

        num_components = len(np.unique(new_labels)) - 1
        show_info(f"Removed {total_removed} bridge voxels in {iteration} passes, now {num_components} components")
        self.update_component_count_display()
        self.update_component_list()

        # Run dust with current widget parameters
        if hasattr(self, 'small_components_widget'):
            connectivity = self.small_components_widget.connectivity.value
            max_size = self.small_components_widget.max_size.value
            self.remove_small_components(connectivity, max_size)

    def get_label_path(self, image_path):
        """Get corresponding label path for an image."""
        stem = image_path.stem
        
        # Try with provided suffix first
        if self.label_suffix:
            label_name = f"{stem}{self.label_suffix}.tif"
            label_path = self.label_dir / label_name
            if label_path.exists():
                return label_path
            # Try .tiff extension
            label_name = f"{stem}{self.label_suffix}.tiff"
            label_path = self.label_dir / label_name
            if label_path.exists():
                return label_path
        
        # Search for any file that starts with the stem
        # This handles cases like "image_surface.tif" for "image.tif"
        possible_labels = list(self.label_dir.glob(f"{stem}*.tif")) + \
                         list(self.label_dir.glob(f"{stem}*.tiff"))
        
        if possible_labels:
            # Return the first match (you could also implement logic to choose the best match)
            return possible_labels[0]
        
        return None

    def save_current_label(self):
        """Save the current label layer data to output directory."""
        if self.output_dir is None:
            return

        if self.current_label_layer is None:
            show_info("No label to save")
            return

        # Get the current image path to derive output filename
        if self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]
        # Use the image stem with .tif extension for the output label
        output_path = self.output_dir / f"{image_path.stem}.tif"

        # Get the exact current label data from the viewer
        label_data = self.current_label_layer.data

        label_data = label_data.astype(np.uint8)

        io.imsave(str(output_path), label_data, compression='lzw')
        show_info(f"Saved label to: {output_path.name}")

    def output_exists(self, image_path):
        """Check if output already exists for given image."""
        if self.output_dir is None:
            return False
        output_path = self.output_dir / f"{image_path.stem}.tif"
        return output_path.exists()

    def skip_to_next_unprocessed(self):
        """Skip to next image that doesn't have output yet. Returns True if found."""
        while self.current_index < len(self.image_files):
            if not self.output_exists(self.image_files[self.current_index]):
                return True
            self.current_index += 1
        return False

    def load_current_pair(self):
        """Load current image-label pair into viewer."""
        if self.current_index >= len(self.image_files):
            show_info("No more images to display")
            return False

        # Save camera/view state before clearing (if we have layers)
        saved_camera_main = None
        saved_camera_3d = None
        saved_dims = None
        if len(self.viewer.layers) > 0:
            saved_camera_main = {
                'center': self.viewer.camera.center,
                'zoom': self.viewer.camera.zoom,
                'angles': self.viewer.camera.angles,
            }
            saved_dims = self.viewer.dims.current_step
            if self.multi_viewer_widget:
                saved_camera_3d = {
                    'center': self.multi_viewer_widget.viewer_model_3d.camera.center,
                    'zoom': self.multi_viewer_widget.viewer_model_3d.camera.zoom,
                    'angles': self.multi_viewer_widget.viewer_model_3d.camera.angles,
                }

        # Clear existing layers
        self.viewer.layers.clear()
        self.current_label_layer = None
        self.bbox_layer = None  # Reset bbox layer reference

        # Load image
        image_path = self.image_files[self.current_index]
        image = io.imread(str(image_path))

        # Add image layer with lower 25% of values suppressed to black
        img_min, img_max = image.min(), image.max()
        contrast_min = img_min + 0.25 * (img_max - img_min)
        self.viewer.add_image(image, name=f"Image: {image_path.name}", opacity=0.5,
                              contrast_limits=(contrast_min, img_max))

        # Load and add label if exists
        label_path = self.get_label_path(image_path)
        if label_path and label_path.exists():
            label = io.imread(str(label_path))
            # Handle ignore label (2) before computing components
            if self.zero_ignore_label:
                label[label == 2] = 0
                # Compute 26-connected components
                label = self.compute_connected_components(label)
                self.ignore_mask = None  # No ignore mask when zeroing
            else:
                # Save ignore mask, zero it out for CC computation, then restore as 150
                ignore_mask = label == 2
                label[ignore_mask] = 0
                label = self.compute_connected_components(label)
                label[ignore_mask] = 150
                # Store ignore mask for toggle functionality
                self.ignore_mask = ignore_mask.copy()
                # Apply current visibility state (don't reset user's selection)
                if not self.ignore_visible:
                    label[ignore_mask] = 0
            self.current_label_layer = self.viewer.add_labels(
                label, name=f"Label: {label_path.name}"
            )
            # Set 3D editing, disable contiguous fill, and make label layer active
            self.current_label_layer.n_edit_dimensions = 3
            self.current_label_layer.contiguous = False
            self.current_label_layer.selected_label = 0
            self.current_label_layer.mode = 'fill'
            self.viewer.layers.selection.active = self.current_label_layer

            # Connect callback to update ignore_mask on any edit
            self.current_label_layer.events.set_data.connect(self._on_label_data_changed)

            # Configure 3D viewer's label layer for 3D editing
            if self.multi_viewer_widget:
                for layer in self.multi_viewer_widget.viewer_model_3d.layers:
                    if layer.name == self.current_label_layer.name:
                        layer.n_edit_dimensions = 3
                        layer.contiguous = False
                        layer.mode = 'fill'
                        break

            num_components = len(np.unique(label)) - 1
            show_info(f"Loaded {num_components} connected components")

            # Remove small components on load
            self.remove_small_components(connectivity=26, max_size=self.dust_max_size)
        else:
            show_info(f"No label found for {image_path.name}")

        # Restore camera/view state if we saved it
        if saved_camera_main is not None:
            self.viewer.camera.center = saved_camera_main['center']
            self.viewer.camera.zoom = saved_camera_main['zoom']
            self.viewer.camera.angles = saved_camera_main['angles']
        if saved_dims is not None:
            # Clamp dims to valid range for new image
            # dims.range returns list of (min, max, step) tuples
            new_step = tuple(
                min(int(s), int(r[1]) - 1) if r[1] > 0 else 0
                for s, r in zip(saved_dims, self.viewer.dims.range)
            )
            self.viewer.dims.current_step = new_step
        if saved_camera_3d is not None and self.multi_viewer_widget:
            self.multi_viewer_widget.viewer_model_3d.camera.center = saved_camera_3d['center']
            self.multi_viewer_widget.viewer_model_3d.camera.zoom = saved_camera_3d['zoom']
            self.multi_viewer_widget.viewer_model_3d.camera.angles = saved_camera_3d['angles']

        # Update title (includes CSV membership)
        self.update_csv_label()
        self.update_component_count_display()
        self.update_component_list()  # Initialize component navigation
        return True

    def next_image(self):
        """Move to next image, saving current label first."""
        # Save current label to output directory before moving on
        self.save_current_label()
        # Auto-flag saved samples
        self.flag_current_sample()

        self.current_index += 1
        # Skip samples that already exist in output dir
        if not self.skip_to_next_unprocessed():
            show_info("Reached end of images (all remaining already processed)")
            self.current_index = len(self.image_files)
            return
        if not self.load_current_pair():
            show_info("Reached end of images")
            self.current_index = len(self.image_files)

    def copy_label_to_output(self):
        """Copy the on-disk label file to output directory without modifications."""
        if self.output_dir is None:
            return

        if self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]
        label_path = self.get_label_path(image_path)

        if label_path is None or not label_path.exists():
            show_info(f"No label found to copy for {image_path.name}")
            return

        output_path = self.output_dir / f"{image_path.stem}.tif"

        # Read and write the original label (preserving original data)
        label_data = io.imread(str(label_path))
        label_data = label_data.astype(np.uint8)
        io.imsave(str(output_path), label_data, compression='lzw')
        show_info(f"Copied original label to: {output_path.name}")

    def skip_image(self):
        """Move to next image, optionally copying the on-disk label to output."""
        if self.copy_on_skip:
            self.copy_label_to_output()
        # Record skipped sample
        self.record_skipped_sample()
        self.current_index += 1
        # Skip samples that already exist in output dir
        if not self.skip_to_next_unprocessed():
            show_info("Reached end of images (all remaining already processed)")
            self.current_index = len(self.image_files)
            return
        if not self.load_current_pair():
            show_info("Reached end of images")
            self.current_index = len(self.image_files)

    def previous_image(self):
        """Move to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()
        else:
            show_info("Already at first image")

    def reset_current(self):
        """Reset current image-label pair by reloading from disk."""
        self.load_current_pair()
        show_info("Reset to original from disk")

    def toggle_ignore_visibility(self, visible):
        """Toggle visibility of the ignore label (value 150)."""
        if self.ignore_mask is None or self.current_label_layer is None:
            return

        label_data = self.current_label_layer.data.copy()
        if visible:
            # Restore ignore label
            label_data[self.ignore_mask] = 150
            self.ignore_visible = True
        else:
            # Hide ignore label
            label_data[self.ignore_mask] = 0
            self.ignore_visible = False
        self.current_label_layer.data = label_data

    def _on_label_data_changed(self, event=None):
        """Update ignore_mask when label data is modified."""
        if self.zero_ignore_label or self.current_label_layer is None:
            return
        label_data = self.current_label_layer.data
        self.ignore_mask = label_data >= 150

    def finalize_label(self):
        """Finalize label by mapping values >100 to 2, and values >0 and <=100 to 1."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        label_data = self.current_label_layer.data

        # Create output array
        finalized = np.zeros_like(label_data)
        finalized[(label_data > 0) & (label_data <= 100)] = 1
        finalized[label_data > 100] = 2

        self.current_label_layer.data = finalized
        show_info("Finalized label: values >100→2, values 1-100→1")

    def delete_current(self):
        """Delete current image-label pair from disk."""
        if self.current_index >= len(self.image_files):
            show_info("No image to delete")
            return
        
        image_path = self.image_files[self.current_index]
        label_path = self.get_label_path(image_path)
        
        # Delete image
        try:
            image_path.unlink()
            show_info(f"Deleted: {image_path.name}")
        except Exception as e:
            show_info(f"Error deleting image: {e}")
            return
        
        # Delete label if exists
        if label_path and label_path.exists():
            try:
                label_path.unlink()
                show_info(f"Deleted: {label_path.name}")
            except Exception as e:
                show_info(f"Error deleting label: {e}")
        
        # Remove from list and load next
        del self.image_files[self.current_index]
        
        # Adjust index if needed
        if self.current_index >= len(self.image_files) and self.current_index > 0:
            self.current_index = len(self.image_files) - 1
        
        # Load next pair
        if self.image_files:
            self.load_current_pair()
        else:
            self.viewer.layers.clear()
            show_info("No more images")

    
    def run(self):
        """Run the viewer with multi-viewer layout."""
        # Critical: Set OpenGL context sharing BEFORE any viewer creation
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

        # Create main viewer (don't show yet - we'll replace central widget)
        self.viewer = napari.Viewer(show=False)

        # Create multi-viewer widget with 4 views
        self.multi_viewer_widget = MultiViewerWidget(self.viewer, self.sync_enabled)
        self.viewer_models = self.multi_viewer_widget.get_all_viewer_models()

        # Configure text overlay appearance for component count display
        self.viewer.text_overlay.font_size = 20
        self.viewer.text_overlay.color = 'white'
        self.multi_viewer_widget.viewer_model_3d.text_overlay.font_size = 20
        self.multi_viewer_widget.viewer_model_3d.text_overlay.color = 'white'

        # Skip to first unprocessed sample
        if not self.skip_to_next_unprocessed():
            show_info("All images already processed")
            return

        # Load first pair
        if not self.load_current_pair():
            show_info("No images found")
            return

        # Create navigation widget with grid layout
        nav_widget = QWidget()
        nav_grid = QGridLayout(nav_widget)
        nav_grid.setContentsMargins(2, 2, 2, 2)
        nav_grid.setSpacing(2)

        prev_btn = QPushButton("< Prev (A)")
        prev_btn.clicked.connect(self.previous_image)
        next_btn = QPushButton("Next > (Space)")
        next_btn.clicked.connect(self.next_image)
        skip_btn = QPushButton("Skip (S)")
        skip_btn.clicked.connect(self.skip_image)
        delete_btn = QPushButton("Delete (Ctrl+D)")
        delete_btn.clicked.connect(self.delete_current)
        reset_btn = QPushButton("Reset (Shift+R)")
        reset_btn.clicked.connect(self.reset_current)

        # Arrange in 2x3 grid
        nav_grid.addWidget(prev_btn, 0, 0)
        nav_grid.addWidget(next_btn, 0, 1)
        nav_grid.addWidget(skip_btn, 0, 2)
        nav_grid.addWidget(delete_btn, 1, 0)
        nav_grid.addWidget(reset_btn, 1, 1)

        @magicgui(
            call_button="Recompute (R)",
            auto_call=False,
        )
        def recompute_button():
            self.recompute_labels()

        @magicgui(
            connectivity={"choices": [6, 18, 26], "value": 26, "label": "Connectivity"},
            max_size={"value": self.dust_max_size, "min": 1, "max": 1000000, "label": "Max Size"},
            layout="vertical",
        )
        def small_components_widget(connectivity: int, max_size: int):
            pass

        # Store reference for use in split_merges
        self.small_components_widget = small_components_widget

        @small_components_widget.connectivity.changed.connect
        def _on_connectivity_changed(value):
            pass

        @small_components_widget.max_size.changed.connect
        def _on_max_size_changed(value):
            pass

        @magicgui(call_button="Dust (D)")
        def dust_button():
            connectivity = small_components_widget.connectivity.value
            max_size = small_components_widget.max_size.value
            self.remove_small_components(connectivity, max_size)

        @magicgui(call_button="Delete Selected (Shift+X)")
        def delete_selected_button():
            self.delete_selected_component()

        # Create containers for widget groups
        from magicgui.widgets import Container, Label
        from qtpy.QtWidgets import QLabel

        # Create sync toggle checkbox
        sync_checkbox = QCheckBox("Sync Position")
        sync_checkbox.setChecked(self.sync_enabled)
        sync_checkbox.stateChanged.connect(self.multi_viewer_widget.set_sync_enabled)

        # Create ignore label toggle checkbox (only functional with --keep-ignore-label)
        self.ignore_checkbox = QCheckBox("Show Ignore Label")
        self.ignore_checkbox.setChecked(True)
        self.ignore_checkbox.setEnabled(not self.zero_ignore_label)  # Only enable if keeping ignore
        self.ignore_checkbox.stateChanged.connect(
            lambda state: self.toggle_ignore_visibility(state == 2)
        )

        # Create CSV membership label widget (large red text)
        self.csv_label_widget = QLabel("(none)")
        self.csv_label_widget.setStyleSheet("color: red; font-size: 48px; font-weight: bold;")
        self.csv_label_widget.setAlignment(Qt.AlignCenter)

        # Create keybinds reference widget
        keybinds_text = "\n".join([
            "Space: Next",
            "S: Skip (no save)",
            "A: Previous",
            "Ctrl+D: Delete",
            "Shift+R: Reset",
            "D: Dust",
            "R: Recompute",
            "Shift+X: Delete Selected",
            "Shift+F: Expand",
            "Shift+E: Copy Expanded",
            "Shift+S: Split",
            "Shift+Q: Toggle Ignore",
            "Shift+Z: Finalize",
            "Shift+D: Next Component",
            "Shift+A: Prev Component",
            "Shift+T: Flag Sample",
        ])
        keybinds_widget = QLabel(keybinds_text)
        keybinds_widget.setStyleSheet("font-size: 12px; padding: 5px;")
        keybinds_widget.setAlignment(Qt.AlignLeft)


        # Component navigation widget with grid layout
        component_nav_widget = QWidget()
        component_nav_grid = QGridLayout(component_nav_widget)
        component_nav_grid.setContentsMargins(2, 2, 2, 2)
        component_nav_grid.setSpacing(2)

        self.component_counter_label = QLabel("Component 0 of 0")
        self.component_counter_label.setAlignment(Qt.AlignCenter)
        prev_comp_btn = QPushButton("<< Prev (Shift+A)")
        prev_comp_btn.clicked.connect(self.previous_component)
        next_comp_btn = QPushButton("Next >> (Shift+D)")
        next_comp_btn.clicked.connect(self.next_component)
        delete_comp_btn = QPushButton("Delete Component")
        delete_comp_btn.clicked.connect(self.delete_current_component)

        # Arrange in grid: label on top, buttons below
        component_nav_grid.addWidget(self.component_counter_label, 0, 0, 1, 2)
        component_nav_grid.addWidget(prev_comp_btn, 1, 0)
        component_nav_grid.addWidget(next_comp_btn, 1, 1)
        component_nav_grid.addWidget(delete_comp_btn, 2, 0, 1, 2)

        small_components_container = Container(
            widgets=[small_components_widget, dust_button, delete_selected_button],
            labels=False,
        )

        # Create expand labels widget
        @magicgui(
            distance={"value": 2, "min": 1, "max": 50, "label": "Distance"},
            call_button="Expand Labels (Shift+F)",
        )
        def expand_widget(distance: int):
            self.expand_current_labels(distance)

        @magicgui(call_button="Copy Selected from Expanded")
        def copy_back_button():
            self.copy_selected_from_expanded()

        expand_container = Container(widgets=[expand_widget, copy_back_button], labels=False)

        # Create split merges widget with skeleton mode option
        @magicgui(
            skeleton_mode={"value": False, "label": "Skeleton Mode (1vx)"},
            call_button="Split Merges (Shift+S)",
        )
        def split_merges_widget(skeleton_mode: bool):
            self.split_merges(skeleton_mode=skeleton_mode)

        self.split_merges_widget = split_merges_widget

        @magicgui(call_button="Finalize (Shift+Z)")
        def finalize_button():
            self.finalize_label()

        @magicgui(call_button="Flag Sample (Shift+T)")
        def flag_button():
            self.flag_current_sample()

        # Add widgets to viewer
        self.viewer.window.add_dock_widget(sync_checkbox, area='right', name='View Sync')
        self.viewer.window.add_dock_widget(self.ignore_checkbox, area='right', name='Ignore Label')
        self.viewer.window.add_dock_widget(self.csv_label_widget, area='right', name='CSV Membership')
        self.viewer.window.add_dock_widget(keybinds_widget, area='right', name='Keybinds', tabify=True)
        self.viewer.window.add_dock_widget(nav_widget, area='right', name='Navigation')
        self.viewer.window.add_dock_widget(recompute_button, area='right')
        self.viewer.window.add_dock_widget(
            component_nav_widget, area='right', name='Component Navigation'
        )
        self.viewer.window.add_dock_widget(
            small_components_container, area='right', name='Small Components'
        )
        self.viewer.window.add_dock_widget(
            expand_container, area='right', name='Expand Labels'
        )
        self.viewer.window.add_dock_widget(split_merges_widget, area='right', name='Split Merges')
        self.viewer.window.add_dock_widget(finalize_button, area='right', name='Finalize')
        self.viewer.window.add_dock_widget(flag_button, area='right', name='Flag Sample')
        # Update CSV label for initial load
        self.update_csv_label()

        # Replace main window central widget with multi-viewer
        main_window = self.viewer.window._qt_window
        main_window.setCentralWidget(self.multi_viewer_widget)

        # Add global keyboard shortcuts (work regardless of focused viewer)
        def make_shortcut(key, callback):
            shortcut = QShortcut(QKeySequence(key), main_window)
            shortcut.activated.connect(callback)
            return shortcut

        make_shortcut('Space', self.next_image)
        make_shortcut('S', self.skip_image)
        make_shortcut('Ctrl+D', self.delete_current)
        make_shortcut('D', lambda: self.remove_small_components(
            small_components_widget.connectivity.value,
            small_components_widget.max_size.value
        ))
        make_shortcut('A', self.previous_image)
        make_shortcut('R', self.recompute_labels)
        make_shortcut('Shift+R', self.reset_current)
        make_shortcut('Shift+X', self.delete_selected_component)

        def expand_and_select():
            distance = expand_widget.distance.value
            self.expand_current_labels(distance)
            # Select the expanded layer and configure it
            for layer in self.viewer.layers:
                if layer.name == "expanded":
                    self.viewer.layers.selection.active = layer
                    layer.selected_label = 150
                    layer.mode = 'fill'
                    layer.n_edit_dimensions = 3
                    layer.contiguous = True
                    break
        make_shortcut('Shift+F', expand_and_select)

        make_shortcut('Shift+E', self.copy_selected_from_expanded)
        make_shortcut('Shift+S', lambda: self.split_merges(
            skeleton_mode=self.split_merges_widget.skeleton_mode.value
        ))

        def toggle_ignore_shortcut():
            if not self.zero_ignore_label and self.ignore_mask is not None:
                new_state = not self.ignore_visible
                self.toggle_ignore_visibility(new_state)
                self.ignore_checkbox.setChecked(new_state)
        make_shortcut('Shift+Q', toggle_ignore_shortcut)

        make_shortcut('Shift+Z', self.finalize_label)

        # Component navigation shortcuts
        make_shortcut('Shift+D', self.next_component)
        make_shortcut('Shift+A', self.previous_component)

        # Flag sample shortcut
        make_shortcut('Shift+T', self.flag_current_sample)

        # Show window and start application
        self.viewer.window.show()
        napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="View and manage image-label pairs in napari"
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "label_dir",
        type=str,
        help="Path to directory containing labels"
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default="",
        help="Suffix for label files (e.g., '_label')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save labels when pressing Next (saves current viewer state)"
    )
    parser.add_argument(
        "--mergers-csv",
        type=str,
        default=None,
        help="Path to mergers.csv file containing sample IDs"
    )
    parser.add_argument(
        "--tiny-csv",
        type=str,
        default=None,
        help="Path to tiny.csv file containing sample IDs"
    )
    parser.add_argument(
        "--keep-ignore-label",
        action="store_true",
        help="Don't zero out the ignore label (value 2) on load"
    )
    parser.add_argument(
        "--no-copy-skip",
        action="store_true",
        help="Don't copy the label to output when using the skip button"
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable position synchronization between viewers by default"
    )
    parser.add_argument(
        "--dust-max-size",
        type=int,
        default=250,
        help="Initial max size for dust removal on image load (default: 250)"
    )

    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        return 1
    
    if not os.path.isdir(args.label_dir):
        print(f"Error: Label directory does not exist: {args.label_dir}")
        return 1
    
    # Run viewer
    viewer = ImageLabelViewer(
        args.image_dir, args.label_dir, args.label_suffix, args.output_dir,
        args.mergers_csv, args.tiny_csv,
        zero_ignore_label=not args.keep_ignore_label,
        copy_on_skip=not args.no_copy_skip,
        dust_max_size=args.dust_max_size
    )
    viewer.sync_enabled = not args.no_sync
    viewer.run()
    
    return 0


if __name__ == "__main__":
    exit(main())