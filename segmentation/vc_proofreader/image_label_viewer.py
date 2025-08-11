#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import napari
import numpy as np
from skimage import io
from magicgui import magicgui
from napari.utils.notifications import show_info


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
        
        # Add image layer
        self.viewer.add_image(image, name=f"Image: {image_path.name}")
        
        # Load and add label if exists
        label_path = self.get_label_path(image_path)
        if label_path and label_path.exists():
            label = io.imread(str(label_path))
            self.viewer.add_labels(label, name=f"Label: {label_path.name}")
        else:
            show_info(f"No label found for {image_path.name}")
        
        # Update title
        self.viewer.title = f"Image {self.current_index + 1}/{len(self.image_files)}"
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