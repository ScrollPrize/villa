import alphashape
import tifffile
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool

from vesuvius.image_proc.helpers import arr_to_indices
from vesuvius.image_proc.geometry.alpha_wrap import fill_alpha_shape

def worker_fn(args):
    img_path, output_dir, subsample_rate, pts_over, alpha, downsample_factor = args

    try:
        # Compute output path and early‑exit if it already exists (resume support)
        img_name = Path(img_path).stem
        output_path = Path(output_dir) / f"{img_name}.tif"
        if output_path.exists():
            return f"Skipped {img_path} - output exists"

        # Load image and derive sample points for the alpha shape
        img = tifffile.imread(img_path)
        pts_2d, img_shape = arr_to_indices(
            img,
            subsample_rate=subsample_rate,
            pts_over=pts_over,
            downsample_factor=downsample_factor,
        )

        # Skip if no points found
        if len(pts_2d) == 0:
            print(f"Warning: No points found in {img_path}")
            return f"Skipped {img_path} - no points"

        # Create alpha shape
        alpha_shape = alphashape.alphashape(pts_2d, alpha)

        # Save the alpha shape as image
        output_array = fill_alpha_shape(img_shape, alpha_shape, fill_value=255)
        tifffile.imwrite(output_path, output_array, compression='packbits')

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
