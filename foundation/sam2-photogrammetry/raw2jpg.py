# Giorgio Angelotti, 2025

import os
import argparse
import rawpy
import multiprocessing
import subprocess
from tqdm import tqdm
from PIL import Image

VALID_RAW_EXTENSIONS = {'.cr2', '.nef', '.arw', '.dng', '.rw2', '.orf', '.raf'}

def convert_raw_to_jpg(args):
    """
    Convert a single RAW image to JPG only if it hasn't been converted already, preserving metadata.
    
    :param args: A tuple (input_path, output_path, quality, recompute)
    """
    input_path, output_path, quality, recompute = args
    
    if os.path.exists(output_path) and not recompute:
        return  

    try:
        # Read the RAW file
        with rawpy.imread(input_path) as raw:
            #rgb = raw.postprocess(use_camera_wb=True, highlight_mode=2, gamma=(1.4, 3.0), bright=0.95)
            # highlight_mode=2, gamma=(0.8, 4.5), bright=1.10  darkens , highlight_mode=2, gamma=(3.5, 4.5), bright=1.10
            rgb = raw.postprocess(use_camera_wb=True, highlight_mode=2, gamma=(3.5, 4.5), bright=1.10)
        
        # Convert to a PIL image
        img = Image.fromarray(rgb)

        # Save JPG without metadata first
        img.save(output_path, "jpeg", subsampling=0, quality=quality)

        # Copy metadata from RAW to JPG using ExifTool
        copy_exif_metadata(input_path, output_path)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def copy_exif_metadata(raw_path, jpg_path):
    """
    Use ExifTool to copy metadata from RAW to JPG.
    """
    try:
        subprocess.run(["exiftool", "-overwrite_original", "-TagsFromFile", raw_path, jpg_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error copying EXIF metadata: {e}")

def process_directory_recursively(root_dir, quality=100, recompute=False):
    """
    Traverse the directory tree starting from `root_dir`. Whenever RAW files
    are found in a directory, create a `JPGEnhanced` folder and convert only
    new RAW files to JPEG there using multiprocessing.
    
    :param root_dir:   The root directory to start searching from.
    :param quality:    JPEG quality (1–100).
    :param recompute:  Whether to overwrite existing JPGs (default: False).
    """
    tasks = []

    for current_dir, _, files in os.walk(root_dir):
        if "corrupted" in current_dir.lower().split(os.sep):
            continue
        
        #if "orientation2" in current_dir:
        #    continue

        raw_files = [
            f for f in files if os.path.splitext(f.lower())[1] in VALID_RAW_EXTENSIONS
        ]

        if not raw_files:
            continue  

        enhanced_dir = os.path.join(current_dir, "JPGEnhanced")
        os.makedirs(enhanced_dir, exist_ok=True)

        for raw_file in raw_files:
            input_path = os.path.join(current_dir, raw_file)
            base_name = os.path.splitext(raw_file)[0]
            output_file = base_name + ".jpg"
            output_path = os.path.join(enhanced_dir, output_file)
            
            tasks.append((input_path, output_path, quality, recompute))

    if tasks:
        cpu_count = min(multiprocessing.cpu_count(), len(tasks))
        with multiprocessing.Pool(processes=cpu_count) as pool:
            list(tqdm(pool.imap_unordered(convert_raw_to_jpg, tasks), total=len(tasks), desc="Processing Images"))

def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert RAW images in a directory tree to JPEG using multiprocessing, preserving metadata."
    )
    parser.add_argument("root_dir", help="Root directory to search for RAW files.")
    parser.add_argument("--quality", type=int, default=100, help="JPEG quality (1–100). Default=100")
    parser.add_argument("--recompute", action="store_true", help="Recompute and overwrite existing JPGs.")

    args = parser.parse_args()
    process_directory_recursively(args.root_dir, args.quality, args.recompute)

if __name__ == "__main__":
    main()
