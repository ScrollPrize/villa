### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
from tqdm import tqdm

def cut_and_save_image(filepath, destination_folder, rectangle):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with Image.open(filepath) as img:
        # Cut out the rectangle
        x, y, width, height = rectangle
        cropped_img = img.crop((x, y, x + width, y + height))

        # Save the cutout in the destination folder
        base_filename = os.path.basename(filepath)
        cropped_img.save(os.path.join(destination_folder, base_filename))

def process_source_folder(source_folder, destination_folder, rectangle):
    # Iterate over all .tif files in the source folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(".tif"):
            filepath = os.path.join(source_folder, filename)
            cut_and_save_image(filepath, destination_folder, rectangle)

def process_inklabels(source_folder, destination_folder, rectangle):
    # Check for inklabels.png in the parent folder
    parent_folder = os.path.dirname(source_folder)
    inklabels_path = os.path.join(parent_folder, 'inklabels.png')

    if os.path.exists(inklabels_path):
        cut_and_save_image(inklabels_path, destination_folder, rectangle)

def process_maks(source_folder, destination_folder, rectangle):
    # Check for inklabels.png in the parent folder
    parent_folder = os.path.dirname(source_folder)
    maks_path = os.path.join(parent_folder, 'mask.png')
    mak_train_path = os.path.join(parent_folder, 'mask_train.png')
    thaumato_mask = os.path.join(parent_folder, 'thaumato_mask.png')

    if os.path.exists(maks_path):
        cut_and_save_image(maks_path, destination_folder, rectangle)

    if os.path.exists(mak_train_path):
        cut_and_save_image(mak_train_path, destination_folder, rectangle)

    if os.path.exists(thaumato_mask):
        cut_and_save_image(thaumato_mask, destination_folder, rectangle)

def main():
    parser = argparse.ArgumentParser(description='Cut out rectangles from TIFF images and an inklabels.png file.')
    parser.add_argument('source_folder', type=str, help='Path to the folder containing TIFF images')
    parser.add_argument('destination_folder', type=str, help='Path to the folder where cropped images will be saved')
    parser.add_argument('x', type=int, help='X coordinate of the top-left corner of the rectangle')
    parser.add_argument('y', type=int, help='Y coordinate of the top-left corner of the rectangle')
    parser.add_argument('width', type=int, help='Width of the rectangle')
    parser.add_argument('height', type=int, help='Height of the rectangle')

    args = parser.parse_args()

    # Create a tuple for the rectangle
    rectangle = (args.x, args.y, args.width, args.height)

    # Process the TIFF images in the source folder
    process_source_folder(args.source_folder, args.destination_folder, rectangle)

    # Process the inklabels.png if it exists
    process_inklabels(args.source_folder, args.destination_folder, rectangle)

    # Process the mask_train.png and thaumato_mask.png if they exist
    process_maks(args.source_folder, args.destination_folder, rectangle)

if __name__ == "__main__":
    main()
