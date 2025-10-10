#!/usr/bin/env python3
"""
Batch script to remove padding from multiple inference output images.
Simplified to match files between labels and images folders by filename.
"""

import os
import glob
import cv2
import numpy as np
from tap import Tap


class BatchRemovePaddingArgumentParser(Tap):
    labels_dir: str  # Directory containing padded prediction/label images
    images_dir: str  # Directory containing original images
    output_dir: str = None  # Output directory for unpadded images (if not provided, uses labels_dir)
    extensions: str = 'png,tif,jpg'  # Comma-separated list of file extensions to process


def process_padded_image(padded_image_path, images_dir, output_dir):
    """Process a single padded image by finding its matching original image"""
    print(f"\nProcessing: {padded_image_path}")
    
    # Get just the filename without path
    padded_filename = os.path.basename(padded_image_path)
    
    # Look for matching file in images directory
    # Try exact match first
    original_image_path = os.path.join(images_dir, padded_filename)
    
    # If exact match doesn't exist, try common variations
    if not os.path.exists(original_image_path):
        # Remove common suffixes like '_prediction', '_label', '_inklabels' etc
        base_name = padded_filename
        for suffix in ['_prediction', '_label', '_inklabels', '_padded']:
            if suffix in base_name:
                base_name = base_name.replace(suffix, '')
                break
        
        # Try to find a file that starts with the base name
        possible_matches = glob.glob(os.path.join(images_dir, f"{base_name}*"))
        if possible_matches:
            # Use the first match
            original_image_path = possible_matches[0]
        else:
            # Try specifically with _0000 suffix
            base_name_without_ext = os.path.splitext(base_name)[0]
            extensions_to_try = ['png', 'tif', 'jpg', 'jpeg']
            
            found = False
            for ext in extensions_to_try:
                test_path = os.path.join(images_dir, f"{base_name_without_ext}_0000.{ext}")
                if os.path.exists(test_path):
                    original_image_path = test_path
                    found = True
                    break
            
            if not found:
                print(f"Warning: No matching original image found for {padded_filename}")
                print(f"  Tried base name: {base_name}")
                print(f"  Also tried with _0000 suffix: {base_name_without_ext}_0000.*")
                return False
    
    try:
        # Read images
        padded_img = cv2.imread(padded_image_path, cv2.IMREAD_GRAYSCALE)
        if padded_img is None:
            print(f"Error: Could not read padded image: {padded_image_path}")
            return False
        
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Error: Could not read original image: {original_image_path}")
            return False
        
        # Get original dimensions and remove padding
        original_shape = original_img.shape
        unpadded_img = padded_img[:original_shape[0], :original_shape[1]]
        
        # Ensure the image is uint8
        if unpadded_img.dtype != np.uint8:
            # Scale to 0-255 range if needed
            img_min = unpadded_img.min()
            img_max = unpadded_img.max()
            if img_max > img_min:
                unpadded_img = ((unpadded_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                unpadded_img = unpadded_img.astype(np.uint8)
        
        # Save unpadded image as TIF
        basename = os.path.basename(padded_image_path)
        base, _ = os.path.splitext(basename)
        output_filename = f"{base}_unpadded.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, unpadded_img)
        print(f"Saved: {output_path}")
        print(f"Shape: {padded_img.shape} -> {unpadded_img.shape}")
        print(f"Matched with: {os.path.basename(original_image_path)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {padded_image_path}: {e}")
        return False


def main():
    args = BatchRemovePaddingArgumentParser().parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.labels_dir
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse extensions and find all matching files
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    padded_files = []
    
    for ext in extensions:
        pattern_path = os.path.join(args.labels_dir, f'*.{ext}')
        padded_files.extend(glob.glob(pattern_path))
    
    if not padded_files:
        print(f"No files found with extensions: {', '.join(extensions)}")
        return
    
    print(f"Found {len(padded_files)} files to process")
    print(f"Looking for original images in: {args.images_dir}")
    
    # Process each file
    success_count = 0
    for padded_file in sorted(padded_files):
        if process_padded_image(padded_file, args.images_dir, args.output_dir):
            success_count += 1
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(padded_files)} files")


if __name__ == "__main__":
    main()
