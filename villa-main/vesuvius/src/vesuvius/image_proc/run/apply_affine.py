#!/usr/bin/env python3
"""
Script to scale and/or transform OBJ files.

- Supports uniform vertex scaling and affine transforms loaded from JSON.
- Accepts 3x4 or 4x4 row-major matrices under key 'transformation_matrix'.
  If 3x4 is provided, it is promoted to a 4x4 by appending [0,0,0,1].
- Optionally inverts the given affine before applying (like the C++ path).
- Transforms vertex normals using (A^{-1})^T (A = linear 3x3 of affine) and renormalizes.

Reads all OBJ files from an input folder and saves scaled/transformed versions to an output folder.
"""

import argparse
import os
from pathlib import Path
import numpy as np

from vesuvius.image_proc.mesh.affine import (
    axis_perm,
    load_transform_from_json,
    compute_inv_transpose,
    apply_affine_to_points,
    transform_normals,
)


def process_obj_with_axis(input_path,
                          output_path,
                          scale_factor=None,
                          transform_matrix_4x4=None,
                          perm=(0, 1, 2),
                          axis_order='xyz'):
    """
    The matrix is assumed to act on coordinates ordered as `perm` (e.g., for 'zyx', perm=(2,1,0)).
    We reorder vertex and normal into that order before applying, then un-permute back to xyz.
    """
    # Precompute normal transform in the matrix's axis order space
    linear = None
    inv_transpose = None
    if transform_matrix_4x4 is not None:
        linear = transform_matrix_4x4[:3, :3]
        inv_transpose = compute_inv_transpose(linear)

    init_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    init_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    out_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    out_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    # Track initial bounds (unscaled, untransformed)
                    init_min = np.minimum(init_min, v)
                    init_max = np.maximum(init_max, v)
                    if scale_factor is not None:
                        v = v * float(scale_factor)
                    v_t = v
                    if transform_matrix_4x4 is not None:
                        v_t = apply_affine_to_points(v_t, transform_matrix_4x4, perm)
                    # Track output bounds
                    out_min = np.minimum(out_min, v_t)
                    out_max = np.maximum(out_max, v_t)
                    outfile.write(f'v {v_t[0]} {v_t[1]} {v_t[2]}\n')
                else:
                    outfile.write(line)
            elif line.startswith('vn '):
                parts = line.split()
                if len(parts) >= 4:
                    n = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    if linear is not None:
                        n_t = transform_normals(n, linear, perm=perm, inv_transpose=inv_transpose)
                        outfile.write(f'vn {n_t[0]} {n_t[1]} {n_t[2]}\n')
                    else:
                        outfile.write(line)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)

    # Print bounds summary for this file
    if np.all(np.isfinite(init_min)) and np.all(np.isfinite(init_max)):
        def fmt_bounds(lo, hi):
            return f"[{lo[0]:.3f}, {hi[0]:.3f}] x [{lo[1]:.3f}, {hi[1]:.3f}] x [{lo[2]:.3f}, {hi[2]:.3f}]"
        print(f"Axis order: {axis_order} | {os.path.basename(str(input_path))}")
        print(f"  Initial bounds (x,y,z): {fmt_bounds(init_min, init_max)}")
        if np.all(np.isfinite(out_min)) and np.all(np.isfinite(out_max)):
            print(f"  Result bounds  (x,y,z): {fmt_bounds(out_min, out_max)}")
        else:
            print("  Result bounds  (x,y,z): n/a")


def process_obj(input_path, output_path, scale_factor=None, transform_matrix_4x4=None):
    """Process an OBJ file assuming default axis order (XYZ)."""
    process_obj_with_axis(
        input_path,
        output_path,
        scale_factor=scale_factor,
        transform_matrix_4x4=transform_matrix_4x4,
        perm=(0, 1, 2),
        axis_order='xyz',
    )


def main():
    parser = argparse.ArgumentParser(description='Scale and/or transform OBJ files')
    parser.add_argument('input_folder', help='Path to folder containing OBJ files')
    parser.add_argument('-o', '--output', help='Output folder (default: input_folder_processed)',
                        default=None)
    parser.add_argument('-s', '--scale', type=float, default=None,
                        help='Scale factor (e.g., 2.0 for 2x scaling)')
    parser.add_argument('-t', '--transform', help='Path to JSON file containing affine transformation matrix (3x4 or 4x4)',
                        default=None)
    parser.add_argument('--invert', action='store_true', help='Invert the given affine before applying')
    parser.add_argument('--axis-order', default='xyz', choices=['xyz','xzy','yxz','yzx','zxy','zyx'],
                        help='Axis order the transform is defined in (default: xyz)')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return 1
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        return 1
    
    perm, _ = axis_perm(args.axis_order)

    # Load transformation matrix if specified
    transform_matrix_4x4 = None
    if args.transform:
        transform_path = Path(args.transform)
        if not transform_path.exists():
            print(f"Error: Transform file '{transform_path}' does not exist")
            return 1
        try:
            transform_matrix_4x4 = load_transform_from_json(transform_path)
            if args.invert:
                try:
                    transform_matrix_4x4 = np.linalg.inv(transform_matrix_4x4)
                    print("Note: Inverting affine as requested (--invert).")
                except np.linalg.LinAlgError:
                    print("Error: affine matrix is non-invertible")
                    return 1
            print(f"Loaded transformation matrix from {transform_path}")
            print(f"Axis order for transform: {args.axis_order}")
        except Exception as e:
            print(f"Error loading transformation matrix: {e}")
            return 1
    
    # Check if at least one operation is specified
    if args.scale is None and transform_matrix_4x4 is None:
        print("Error: Must specify at least one of --scale or --transform")
        return 1
    
    # Set output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        suffix = []
        if args.scale:
            suffix.append("scaled")
        if transform_matrix_4x4 is not None:
            suffix.append("transformed")
        output_folder = input_folder.parent / f"{input_folder.name}_{'_'.join(suffix)}"
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all OBJ files
    obj_files = list(input_folder.glob('*.obj'))
    if not obj_files:
        print(f"No OBJ files found in '{input_folder}'")
        return 1
    
    print(f"Found {len(obj_files)} OBJ file(s)")
    if args.scale:
        print(f"Scaling by factor: {args.scale}")
    if transform_matrix_4x4 is not None:
        print(f"Applying affine transformation")
    print(f"Output folder: {output_folder}")
    
    # Process each OBJ file
    for obj_file in obj_files:
        output_file = output_folder / obj_file.name
        print(f"Processing: {obj_file.name} -> {output_file}")
        process_obj_with_axis(obj_file, output_file, args.scale, transform_matrix_4x4, perm, args.axis_order)
    
    print(f"Done! Processed {len(obj_files)} file(s)")
    return 0


if __name__ == '__main__':
    exit(main())
