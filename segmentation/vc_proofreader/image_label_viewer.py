#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import napari
import numpy as np
from skimage import io
from skimage.segmentation import expand_labels
import cc3d
from magicgui import magicgui
from napari.utils.notifications import show_info

# 8 corner neighbors in 3D (differ by ±1 in ALL three axes)
CORNER_OFFSETS = [
    (-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
    (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1),
]


class ImageLabelViewer:
    def __init__(self, image_dir, label_dir, label_suffix="", output_dir=None,
                 mergers_csv=None, tiny_csv=None, zero_ignore_label=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_suffix = label_suffix
        self.output_dir = Path(output_dir) if output_dir else None
        self.zero_ignore_label = zero_ignore_label

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

        self.current_index = 0
        self.viewer = None
        self.current_label_layer = None

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

    def find_corner_bridges(self, label_data):
        """Find corner-only bridges using vectorized neighbor comparison.

        Detects 26-but-not-18 connectivity (corner bridges).
        Uses fast vectorized array shifts instead of per-component dilation.

        Args:
            label_data: The label array to analyze

        Returns:
            bridge_mask: Binary mask of voxels to remove
            merged_component_mask: Mask of all voxels in merged components
            num_mergers: Count of merged components found
        """
        binary = (label_data > 0).astype(np.uint8)
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

        # Find merged components (26-components containing bridge voxels)
        labeled_26 = cc3d.connected_components(binary, connectivity=26)
        bridge_labels = np.unique(labeled_26[bridge_mask > 0])
        bridge_labels = bridge_labels[bridge_labels > 0]

        merged_component_mask = np.isin(labeled_26, bridge_labels).astype(np.uint8)
        num_mergers = len(bridge_labels)

        return bridge_mask, merged_component_mask, num_mergers

    def find_edge_bridges(self, label_data):
        """Find edge-only bridges using vectorized neighbor comparison.

        Detects 18-but-not-6 connectivity (diagonal/edge bridges).
        Uses fast vectorized array shifts instead of per-component dilation.

        Args:
            label_data: The label array to analyze

        Returns:
            bridge_mask: Binary mask of voxels to remove
            merged_component_mask: Mask of all voxels in merged components
            num_mergers: Count of merged components found
        """
        binary = (label_data > 0).astype(np.uint8)
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

        # Find merged components (18-components containing bridge voxels)
        labeled_18 = cc3d.connected_components(binary, connectivity=18)
        bridge_labels = np.unique(labeled_18[bridge_mask > 0])
        bridge_labels = bridge_labels[bridge_labels > 0]

        merged_component_mask = np.isin(labeled_18, bridge_labels).astype(np.uint8)
        num_mergers = len(bridge_labels)

        return bridge_mask, merged_component_mask, num_mergers

    def split_merges(self):
        """Find and remove all thin bridges (corner and edge) in one operation."""
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        label_data = self.current_label_layer.data.copy()

        # Loop until no more bridges found (both corner and edge)
        total_removed = 0
        iteration = 0
        while True:
            iteration += 1
            removed_this_iter = 0

            # Try corner bridges (26-but-not-18)
            bridge_mask, _, _ = self.find_corner_bridges(label_data)
            if np.sum(bridge_mask) > 0:
                removed_this_iter += np.sum(bridge_mask)
                label_data[bridge_mask > 0] = 0

            # Try edge bridges (18-but-not-6)
            bridge_mask, _, _ = self.find_edge_bridges(label_data)
            if np.sum(bridge_mask) > 0:
                removed_this_iter += np.sum(bridge_mask)
                label_data[bridge_mask > 0] = 0

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
        show_info(f"Removed {total_removed} bridge voxels in {iteration-1} passes, now {num_components} components")

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

        io.imsave(str(output_path), label_data)
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

        # Clear existing layers
        self.viewer.layers.clear()
        self.current_label_layer = None

        # Load image
        image_path = self.image_files[self.current_index]
        image = io.imread(str(image_path))

        # Add image layer
        self.viewer.add_image(image, name=f"Image: {image_path.name}")

        # Load and add label if exists
        label_path = self.get_label_path(image_path)
        if label_path and label_path.exists():
            label = io.imread(str(label_path))
            # Set ignore label (2) to background before computing components
            if self.zero_ignore_label:
                label[label == 2] = 0
            # Compute 26-connected components
            label = self.compute_connected_components(label)
            self.current_label_layer = self.viewer.add_labels(
                label, name=f"Label: {label_path.name}"
            )
            # Set 3D editing, disable contiguous fill, and make label layer active
            self.current_label_layer.n_edit_dimensions = 3
            self.current_label_layer.contiguous = False
            self.current_label_layer.selected_label = 0
            self.current_label_layer.mode = 'fill'
            self.viewer.layers.selection.active = self.current_label_layer
            num_components = len(np.unique(label)) - 1
            show_info(f"Loaded {num_components} connected components")

            # Remove small components (< 250 voxels) on load
            self.remove_small_components(connectivity=26, max_size=250)
        else:
            show_info(f"No label found for {image_path.name}")

        # Update title (includes CSV membership)
        self.update_csv_label()
        return True

    def next_image(self):
        """Move to next image, saving current label first."""
        # Save current label to output directory before moving on
        self.save_current_label()

        self.current_index += 1
        # Skip samples that already exist in output dir
        if not self.skip_to_next_unprocessed():
            show_info("Reached end of images (all remaining already processed)")
            self.current_index = len(self.image_files)
            return
        if not self.load_current_pair():
            show_info("Reached end of images")
            self.current_index = len(self.image_files)

    def skip_image(self):
        """Move to next image without saving anything."""
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
        """Run the viewer."""
        self.viewer = napari.Viewer()

        # Skip to first unprocessed sample
        if not self.skip_to_next_unprocessed():
            show_info("All images already processed")
            return

        # Load first pair
        if not self.load_current_pair():
            show_info("No images found")
            return
        
        # Create buttons widget
        @magicgui(
            call_button="Next (Space)",
            auto_call=False,
        )
        def next_button():
            self.next_image()
        
        @magicgui(
            call_button="Previous (A)",
            auto_call=False,
        )
        def previous_button():
            self.previous_image()

        @magicgui(
            call_button="Delete (Ctrl+D)",
            auto_call=False,
        )
        def delete_button():
            self.delete_current()

        @magicgui(
            call_button="Reset (Shift+R)",
            auto_call=False,
        )
        def reset_button():
            self.reset_current()

        @magicgui(
            call_button="Skip (S)",
            auto_call=False,
        )
        def skip_button():
            self.skip_image()

        @magicgui(
            call_button="Recompute (R)",
            auto_call=False,
        )
        def recompute_button():
            self.recompute_labels()

        @magicgui(
            connectivity={"choices": [6, 18, 26], "value": 26, "label": "Connectivity"},
            max_size={"value": 250, "min": 1, "max": 1000000, "label": "Max Size"},
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
        from magicgui.widgets import Container
        from qtpy.QtWidgets import QLabel
        from qtpy.QtCore import Qt

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
        ])
        keybinds_widget = QLabel(keybinds_text)
        keybinds_widget.setStyleSheet("font-size: 12px; padding: 5px;")
        keybinds_widget.setAlignment(Qt.AlignLeft)

        # Navigation container
        nav_container = Container(
            widgets=[previous_button, next_button, skip_button, delete_button, reset_button],
            labels=False,
        )

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

        # Create split merges widget
        @magicgui(call_button="Split Merges (Shift+S)")
        def split_merges_button():
            self.split_merges()

        # Add widgets to viewer
        self.viewer.window.add_dock_widget(self.csv_label_widget, area='right', name='CSV Membership')
        self.viewer.window.add_dock_widget(keybinds_widget, area='right', name='Keybinds', tabify=True)
        self.viewer.window.add_dock_widget(nav_container, area='right', name='Navigation')
        self.viewer.window.add_dock_widget(recompute_button, area='right')
        self.viewer.window.add_dock_widget(
            small_components_container, area='right', name='Small Components'
        )
        self.viewer.window.add_dock_widget(
            expand_container, area='right', name='Expand Labels'
        )
        self.viewer.window.add_dock_widget(split_merges_button, area='right')
        # Update CSV label for initial load
        self.update_csv_label()
        
        # Add keyboard bindings
        @self.viewer.bind_key('Space')
        def next_key(viewer):
            self.next_image()

        @self.viewer.bind_key('s')
        def skip_key(viewer):
            self.skip_image()

        @self.viewer.bind_key('Control-d')
        def delete_key(viewer):
            self.delete_current()

        @self.viewer.bind_key('d')
        def dust_key(viewer):
            connectivity = small_components_widget.connectivity.value
            max_size = small_components_widget.max_size.value
            self.remove_small_components(connectivity, max_size)

        @self.viewer.bind_key('a')
        def previous_key(viewer):
            self.previous_image()

        @self.viewer.bind_key('r')
        def recompute_key(viewer):
            self.recompute_labels()

        @self.viewer.bind_key('Shift-R')
        def reset_key(viewer):
            self.reset_current()

        @self.viewer.bind_key('Shift-X')
        def delete_selected_key(viewer):
            self.delete_selected_component()

        @self.viewer.bind_key('Shift-F')
        def expand_key(viewer):
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

        @self.viewer.bind_key('Shift-E')
        def copy_from_expanded_key(viewer):
            self.copy_selected_from_expanded()

        @self.viewer.bind_key('Shift-S')
        def split_merges_key(viewer):
            self.split_merges()

        # Start the application
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
        zero_ignore_label=not args.keep_ignore_label
    )
    viewer.run()
    
    return 0


if __name__ == "__main__":
    exit(main())