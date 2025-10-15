#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import napari
import numpy as np
from skimage import io
from skimage.morphology import (
    ball,
    binary_dilation,
    binary_erosion,
    dilation,
    disk,
    erosion,
)
from magicgui import magicgui
from napari.utils.notifications import show_info
from napari.layers import Image as NapariImageLayer, Labels as NapariLabelsLayer


class ImageLabelViewer:
    def __init__(self, image_dir, label_dir, label_suffix=""):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_suffix = label_suffix
        
        # Get all tif files
        self.image_files = sorted([f for f in self.image_dir.glob("*.tif") 
                                  if f.is_file()])
        if not self.image_files:
            self.image_files = sorted([f for f in self.image_dir.glob("*.tiff") 
                                      if f.is_file()])
        
        self.current_index = 0
        self.viewer = None
        self.current_image_path = None
        self.current_label_path = None
        self.morphology_widget = None
        
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
    
    def load_current_pair(self):
        """Load current image-label pair into viewer."""
        if self.current_index >= len(self.image_files):
            show_info("No more images to display")
            return False
        
        # Clear existing layers
        self.viewer.layers.clear()
        
        # Load image
        image_path = self.image_files[self.current_index]
        image = io.imread(str(image_path))
        self.current_image_path = image_path
        
        # Add image layer
        self.viewer.add_image(image, name=f"Image: {image_path.name}")
        
        # Load and add label if exists
        label_path = self.get_label_path(image_path)
        if label_path and label_path.exists():
            label = io.imread(str(label_path))
            self.viewer.add_labels(label, name=f"Label: {label_path.name}")
            self.current_label_path = label_path
        else:
            show_info(f"No label found for {image_path.name}")
            self.current_label_path = None

        # Update title
        self.viewer.title = f"Image {self.current_index + 1}/{len(self.image_files)}"
        if self.morphology_widget is not None:
            choices = [choice[1] for choice in self._morphology_target_choices()]
            if choices:
                default_choice = "label" if "label" in choices else choices[0]
                if self.morphology_widget.target.value not in choices:
                    self.morphology_widget.target.value = default_choice
        return True
    
    def next_image(self):
        """Move to next image."""
        self.current_index += 1
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
        
        # Load first pair
        if not self.load_current_pair():
            show_info("No images found")
            return

        @magicgui(
            target={"choices": self._morphology_target_choices},
            operation={"choices": ["dilation", "erosion"]},
            radius={"min": 1, "max": 25, "step": 1, "value": 1},
            call_button="Apply Morphology",
        )
        def morphology_widget(target="label", operation="dilation", radius=1):
            self.apply_morphology(target, operation, radius)

        self.morphology_widget = morphology_widget
        self.viewer.window.add_dock_widget(morphology_widget, area='right')

        # Create buttons widget
        @magicgui(
            call_button="Next (Space)",
            auto_call=False,
        )
        def next_button():
            self.next_image()
        
        @magicgui(
            call_button="Delete (D)",
            auto_call=False,
        )
        def delete_button():
            self.delete_current()
        
        # Add widgets to viewer
        self.viewer.window.add_dock_widget(next_button, area='right')
        self.viewer.window.add_dock_widget(delete_button, area='right')
        
        # Add keyboard bindings
        @self.viewer.bind_key('Space')
        def next_key(viewer):
            self.next_image()
        
        @self.viewer.bind_key('d')
        def delete_key(viewer):
            self.delete_current()
        
        @self.viewer.bind_key('a')
        def previous_key(viewer):
            self.previous_image()
        
        # Start the application
        napari.run()

    def _morphology_target_choices(self, widget=None):
        """Return available targets for morphology widget."""
        choices = []
        if self.current_image_path is not None:
            choices.append(("image", "image"))
        if self.current_label_path is not None:
            choices.append(("label", "label"))
        if not choices:
            choices.append(("image", "image"))
        # magicgui expects iterable of choices; value, str
        return choices

    def _get_structuring_element(self, radius, ndim):
        """Generate a circular (or spherical) structuring element for given radius."""
        if radius < 1:
            return None
        if ndim == 2:
            return disk(radius)
        if ndim == 3:
            return ball(radius)

        # Fallback: n-dimensional hypersphere mask
        grid_shape = (2 * radius + 1,) * ndim
        coords = np.indices(grid_shape) - radius
        squared_dist = np.sum(coords ** 2, axis=0)
        return squared_dist <= radius ** 2

    def _get_layer_for_target(self, target):
        """Return napari layer instance for provided target name."""
        if target == "image":
            for layer in self.viewer.layers:
                if isinstance(layer, NapariImageLayer):
                    return layer
        elif target == "label":
            for layer in self.viewer.layers:
                if isinstance(layer, NapariLabelsLayer):
                    return layer
        return None

    def apply_morphology(self, target, operation, radius):
        """Apply morphology operation to selected layer and persist the result."""
        layer = self._get_layer_for_target(target)
        if layer is None:
            show_info("Selected layer is not available")
            return

        struct_elem = self._get_structuring_element(radius, layer.data.ndim)
        if struct_elem is None:
            show_info("Radius must be at least 1")
            return

        try:
            if isinstance(layer, NapariLabelsLayer):
                new_data = self._morph_labels(layer.data, operation, struct_elem)
            else:
                new_data = self._morph_image(layer.data, operation, struct_elem)
        except ValueError as exc:
            show_info(str(exc))
            return

        layer.data = new_data
        self._save_layer_data(target, new_data)

    def _morph_image(self, data, operation, struct_elem):
        """Morph grayscale image data using provided structuring element."""
        morph = dilation if operation == "dilation" else erosion
        result = morph(data, footprint=struct_elem)
        return result.astype(data.dtype, copy=False)

    def _morph_labels(self, data, operation, struct_elem):
        """Morph each label independently using binary operations."""
        unique_labels = np.unique(data)
        if unique_labels.size <= 1:
            return data

        labels_sorted = sorted([val for val in unique_labels if val != 0])
        if operation == "dilation":
            result = data.copy()
            occupied = result != 0
            for label_value in labels_sorted:
                mask = data == label_value
                if not np.any(mask):
                    continue
                transformed = binary_dilation(mask, footprint=struct_elem)
                new_area = np.logical_and(transformed, ~occupied)
                if np.any(new_area):
                    result[new_area] = label_value
                    occupied[new_area] = True
            return result

        # Erosion path
        result = np.zeros_like(data)
        for label_value in labels_sorted:
            mask = data == label_value
            if not np.any(mask):
                continue
            transformed = binary_erosion(mask, footprint=struct_elem)
            if np.any(transformed):
                result[transformed] = label_value
        return result

    def _save_layer_data(self, target, data):
        """Persist updated layer data to disk."""
        if target == "image":
            path = self.current_image_path
        else:
            path = self.current_label_path

        if not path:
            show_info("No file path available to save data")
            return

        try:
            io.imsave(str(path), data, check_contrast=False)
            show_info(f"Saved {path.name}")
        except Exception as exc:
            show_info(f"Failed to save {path.name}: {exc}")


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
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        return 1
    
    if not os.path.isdir(args.label_dir):
        print(f"Error: Label directory does not exist: {args.label_dir}")
        return 1
    
    # Run viewer
    viewer = ImageLabelViewer(args.image_dir, args.label_dir, args.label_suffix)
    viewer.run()
    
    return 0


if __name__ == "__main__":
    exit(main())
