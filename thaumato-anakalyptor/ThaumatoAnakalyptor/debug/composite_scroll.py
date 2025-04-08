#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image

def process_folder(folder, output_dir):
    """
    Process a single folder: in the folder, locate the 'layers'
    subfolder that contains images named '00.jpg' ... '64.jpg'.
    From these, select the images from 22.jpg to 42.jpg (i.e. 32 ± 10),
    compute the pixel-wise maximum image, and save the result.
    """
    layers_dir = os.path.join(folder, "layers")
    if not os.path.isdir(layers_dir):
        print(f"Skipping {folder}: 'layers' folder not found.")
        return

    # Define the range: 32 - 10 to 32 + 10 inclusive.
    start_index, end_index = 22, 42

    image_max = None
    for idx in range(start_index, end_index + 1):
        filename = f"{idx:02d}.jpg"
        filepath = os.path.join(layers_dir, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: {filepath} not found. Skipping this file.")
            continue
        try:
            with Image.open(filepath) as img:
                # Ensure image is in RGB mode
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        if image_max is None:
            # Initialize the image_max with the first valid image array
            image_max = img_array
        else:
            # Compute pixel-wise maximum
            image_max = np.maximum(image_max, img_array)

    if image_max is not None:
        # Convert the maximum array back to an image.
        max_image = Image.fromarray(image_max)
        # Use the folder's basename as the filename.
        folder_name = os.path.basename(os.path.normpath(folder))
        output_file = os.path.join(output_dir, f"{folder_name}.jpg")
        max_image.save(output_file)
        print(f"Saved maximum image for '{folder_name}' to {output_file}")
    else:
        print(f"No valid images found in {layers_dir} to process.")

def main():
    # Set up argparse arguments.
    parser = argparse.ArgumentParser(
        description="Traverse folders to compute and save a pixel-wise max image from layers (32 ± 10)."
    )
    parser.add_argument("input_dir", type=str, help="Input directory containing subfolders.")
    parser.add_argument("output_dir", type=str, help="Output directory to store processed images.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Traverse each item in the input directory.
    for entry in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, entry)
        if os.path.isdir(folder_path):
            process_folder(folder_path, args.output_dir)

if __name__ == "__main__":
    main()
