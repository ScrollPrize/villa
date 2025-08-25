import numpy as np
import tifffile
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool


def apply_mask(args):
    img_path, mask_path, output_path, fill_value = args
    
    try:
        # Read image and mask
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        
        # Check if shapes match
        if img.shape != mask.shape:
            return f"Error: Shape mismatch for {img_path.name} - Image: {img.shape}, Mask: {mask.shape}"
        
        # Apply mask (set to fill_value where mask is 0)
        masked_img = np.where(mask > 0, img, fill_value)
        
        # Convert to uint8 before saving
        masked_img = masked_img.astype(np.uint8)
        
        # Save the masked image
        tifffile.imwrite(output_path, masked_img, compression='lzw')
        
        return f"Processed {img_path.name} -> {output_path.name}"
    
    except Exception as e:
        return f"Error processing {img_path.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Apply masks to TIF images')
    parser.add_argument('image_dir', type=str, help='Directory containing input image TIF files')
    parser.add_argument('mask_dir', type=str, help='Directory containing mask TIF files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for masked TIF files')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes (default: 16)')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to match mask files (e.g., "_mask" if masks are named "image_mask.tif")')
    parser.add_argument('--fill-value', type=float, default=0, help='Value to set for pixels outside the mask (default: 0)')
    
    args = parser.parse_args()
    
    # Set up paths
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist")
        return
    
    if not mask_dir.exists():
        print(f"Error: Mask directory {mask_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all TIF files from image directory
    image_files = sorted(list(image_dir.glob('*.tif')) + list(image_dir.glob('*.tiff')))
    
    if not image_files:
        print(f"No TIF files found in {image_dir}")
        return
    
    # Match images with masks
    worker_args = []
    missing_masks = []
    
    for img_path in image_files:
        # Try to find corresponding mask
        mask_name = img_path.stem + args.suffix
        
        # Look for mask with .tif or .tiff extension
        mask_path = mask_dir / f"{mask_name}.tif"
        if not mask_path.exists():
            mask_path = mask_dir / f"{mask_name}.tiff"
        
        if not mask_path.exists():
            # Try without suffix if suffix was provided
            if args.suffix:
                mask_name = img_path.stem
                mask_path = mask_dir / f"{mask_name}.tif"
                if not mask_path.exists():
                    mask_path = mask_dir / f"{mask_name}.tiff"
        
        if mask_path.exists():
            output_path = output_dir / img_path.name
            worker_args.append((img_path, mask_path, output_path, args.fill_value))
        else:
            missing_masks.append(img_path.name)
    
    # Report missing masks
    if missing_masks:
        print(f"\nWarning: No masks found for {len(missing_masks)} images:")
        for name in missing_masks[:10]:  # Show first 10
            print(f"  - {name}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")
        print()
    
    if not worker_args:
        print("No matching image-mask pairs found")
        return
    
    print(f"Found {len(worker_args)} image-mask pairs to process")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.workers} workers")
    print(f"Fill value for masked areas: {args.fill_value}")
    
    # Process with multiprocessing
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(apply_mask, worker_args),
            total=len(worker_args),
            desc="Applying masks"
        ))
    
    # Print summary
    print("\n=== Processing Complete ===")
    successful = sum(1 for r in results if r.startswith("Processed"))
    errors = sum(1 for r in results if r.startswith("Error"))
    
    print(f"Successfully processed: {successful}/{len(worker_args)}")
    if errors > 0:
        print(f"Errors: {errors}")
        print("\nError details:")
        for result in results:
            if result.startswith("Error"):
                print(f"  {result}")


if __name__ == "__main__":
    main()