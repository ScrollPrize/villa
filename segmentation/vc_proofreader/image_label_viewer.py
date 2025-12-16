#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import napari
import numpy as np
from skimage import io
from skimage.segmentation import expand_labels
import cc3d
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from magicgui import magicgui
from napari.utils.notifications import show_info
from numba import njit, prange

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


class ImageLabelViewer:
    def __init__(self, image_dir, label_dir, label_suffix="", output_dir=None,
                 mergers_csv=None, tiny_csv=None, zero_ignore_label=True, copy_on_skip=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_suffix = label_suffix
        self.output_dir = Path(output_dir) if output_dir else None
        self.zero_ignore_label = zero_ignore_label
        self.copy_on_skip = copy_on_skip

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

    def split_merges(self, max_iterations=5):
        """Find and remove all thin bridges (corner, edge, and minimal erosion-based) in one operation.

        Optimized to compute connected components once per iteration and share across stages.

        Args:
            max_iterations: Maximum number of passes (default: 5). Each pass removes thin bridges
                           that may have been revealed by previous passes.
        """
        if self.current_label_layer is None:
            show_info("No label layer loaded")
            return

        label_data = self.current_label_layer.data.copy()

        # Loop until no more bridges found or max iterations reached
        total_removed = 0
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            removed_this_iter = 0

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

        # Clear existing layers
        self.viewer.layers.clear()
        self.current_label_layer = None

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
            else:
                # Save ignore mask, zero it out for CC computation, then restore as 150
                ignore_mask = label == 2
                label[ignore_mask] = 0
                label = self.compute_connected_components(label)
                label[ignore_mask] = 150
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
    parser.add_argument(
        "--no-copy-skip",
        action="store_true",
        help="Don't copy the label to output when using the skip button"
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
        copy_on_skip=not args.no_copy_skip
    )
    viewer.run()
    
    return 0


if __name__ == "__main__":
    exit(main())