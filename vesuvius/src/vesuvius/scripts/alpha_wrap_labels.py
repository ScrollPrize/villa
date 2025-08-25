import alphashape
import numpy as np
import tifffile
from tqdm import tqdm
from PIL import Image, ImageDraw
import argparse
import os
from pathlib import Path
from multiprocessing import Pool
from functools import partial


def create_alphashape(indices, alpha=0.1):
    alpha_shape = alphashape.alphashape(indices, alpha)
    return alpha_shape

def tif_to_indices(img_path, subsample_rate=20, pts_over=0):
    img = tifffile.imread(img_path)
    pts = np.where(img > pts_over)

    # returns x,y coordinates of nonzero pts
    pts_2d = np.column_stack((pts[1][::subsample_rate], pts[0][::subsample_rate]))

    return pts_2d, img.shape

def alpha_shape_to_image(shape, output_path, alpha_shape):
    output_img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(output_img)

    if alpha_shape.geom_type == 'Polygon':
        coords = list(alpha_shape.exterior.coords)
        draw.polygon(coords, fill=255)
    elif alpha_shape.geom_type == 'MultiPolygon':
        for poly in alpha_shape.geoms:
            coords = list(poly.exterior.coords)
            draw.polygon(coords, fill=255)

    output_array = np.array(output_img)
    tifffile.imwrite(output_path, output_array)

def worker_fn(args):
    img_path, output_dir, subsample_rate, pts_over, alpha = args
    
    try:
        # Get indices and image shape
        pts_2d, img_shape = tif_to_indices(img_path, subsample_rate, pts_over)
        
        # Skip if no points found
        if len(pts_2d) == 0:
            print(f"Warning: No points found in {img_path}")
            return f"Skipped {img_path} - no points"
        
        # Create alpha shape
        alpha_shape = create_alphashape(pts_2d, alpha)
        
        # Generate output path
        img_name = Path(img_path).stem
        output_path = Path(output_dir) / f"{img_name}_alpha.tif"
        
        # Save the alpha shape as image
        alpha_shape_to_image(img_shape, output_path, alpha_shape)
        
        return f"Processed {img_path} -> {output_path}"
    
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Process TIF images to create alpha shapes')
    parser.add_argument('input_dir', type=str, help='Directory containing input TIF files')
    parser.add_argument('--output-dir', type=str, help='Output directory for alpha shape TIFs (default: input_dir/alpha_shapes)')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes (default: 16)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha parameter for alpha shape (default: 0.01)')
    parser.add_argument('--subsample-rate', type=int, default=20, help='Subsample rate for points (default: 20)')
    parser.add_argument('--pts-over', type=int, default=0, help='Threshold for point selection (default: 0)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'alpha_shapes'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all TIF files
    tif_files = list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff'))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} TIF files to process")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.workers} workers")
    print(f"Alpha: {args.alpha}, Subsample rate: {args.subsample_rate}, Threshold: {args.pts_over}")
    
    # Prepare arguments for worker function
    worker_args = [(str(tif_file), str(output_dir), args.subsample_rate, args.pts_over, args.alpha) 
                   for tif_file in tif_files]
    
    # Process with multiprocessing
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, worker_args),
            total=len(tif_files),
            desc="Processing TIF files"
        ))
    
    # Print summary
    print("\n=== Processing Complete ===")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
