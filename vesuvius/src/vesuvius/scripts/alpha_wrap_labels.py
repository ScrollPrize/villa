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

def tif_to_indices(img_path, subsample_rate=20, pts_over=0, downsample_factor=1):
    """Extract foreground point indices from a TIF image.

    - Reads the image at full resolution.
    - Optionally downsamples the grid by an integer factor before thresholding.
    - Optionally subsamples the resulting point list by taking every Nth point.
    - Returns points in the ORIGINAL image coordinate system and the original shape.
    """
    img = tifffile.imread(img_path)

    if downsample_factor is None or downsample_factor < 1:
        downsample_factor = 1

    if downsample_factor > 1:
        # Simple decimation for speed (sufficient for label masks)
        img_ds = img[::downsample_factor, ::downsample_factor]
        pts = np.where(img_ds > pts_over)
        xs = pts[1]
        ys = pts[0]
        # Map back to original coordinate system
        xs = xs * downsample_factor
        ys = ys * downsample_factor
    else:
        pts = np.where(img > pts_over)
        xs = pts[1]
        ys = pts[0]

    # Subsample the point list to thin it further
    if subsample_rate is None or subsample_rate < 1:
        subsample_rate = 1
    xs = xs[::subsample_rate]
    ys = ys[::subsample_rate]

    # returns x,y coordinates of selected pts in original scale
    pts_2d = np.column_stack((xs, ys))

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
    tifffile.imwrite(output_path, output_array, compression='packbits')

def worker_fn(args):
    img_path, output_dir, subsample_rate, pts_over, alpha, downsample_factor = args

    try:
        # Compute output path and early‑exit if it already exists (resume support)
        img_name = Path(img_path).stem
        output_path = Path(output_dir) / f"{img_name}.tif"
        if output_path.exists():
            return f"Skipped {img_path} - output exists"

        # Get indices and image shape
        pts_2d, img_shape = tif_to_indices(img_path, subsample_rate, pts_over, downsample_factor)

        # Skip if no points found
        if len(pts_2d) == 0:
            print(f"Warning: No points found in {img_path}")
            return f"Skipped {img_path} - no points"

        # Create alpha shape
        alpha_shape = create_alphashape(pts_2d, alpha)

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
    parser.add_argument('--downsample-factor', type=int, default=1, help='Integer factor to downsample image grid before point extraction (default: 1 = no downsampling)')
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
    
    # Determine which outputs already exist so we can resume
    existing = []
    pending_files = []
    for f in tif_files:
        out_path = output_dir / f"{f.stem}.tif"
        if out_path.exists():
            existing.append(f)
        else:
            pending_files.append(f)

    print(f"Found {len(tif_files)} TIF files total")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.workers} workers")
    print(f"Alpha: {args.alpha}, Subsample rate: {args.subsample_rate}, Downsample factor: {args.downsample_factor}, Threshold: {args.pts_over}")
    print(f"Already done: {len(existing)} | To process: {len(pending_files)}")
    
    # Prepare arguments for worker function
    worker_args = [
        (str(tif_file), str(output_dir), args.subsample_rate, args.pts_over, args.alpha, args.downsample_factor)
        for tif_file in pending_files
    ]
    
    # Process with multiprocessing
    results = []
    if worker_args:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, worker_args),
                total=len(worker_args),
                desc="Processing TIF files"
            ))
    else:
        print("No pending files; everything is up to date.")
    
    # Print summary
    print("\n=== Processing Complete ===")
    for result in results:
        print(result)

    # Also report skipped due to existing outputs for visibility
    for f in existing:
        print(f"Skipped {f} - output exists")


if __name__ == "__main__":
    main()
