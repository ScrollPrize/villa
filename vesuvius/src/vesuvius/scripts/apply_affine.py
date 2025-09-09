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
import json
import os
from pathlib import Path
import numpy as np


def apply_affine_transform(vertex, transform_matrix_4x4):
    """
    Apply affine transformation to a vertex.
    
    Args:
        vertex: [x, y, z] coordinates
        transform_matrix_4x4: 4x4 affine transformation matrix
    
    Returns:
        Transformed [x, y, z] coordinates
    """
    # Convert to homogeneous coordinates
    homogeneous = np.array([vertex[0], vertex[1], vertex[2], 1.0])
    
    # Apply transformation
    transformed = transform_matrix_4x4 @ homogeneous
    
    return transformed[:3]


def transform_normal(normal, linear_3x3, inv_transpose_3x3=None):
    """
    Transform a normal vector using the inverse-transpose of the linear part.

    Args:
        normal: [nx, ny, nz]
        linear_3x3: 3x3 linear part A of the affine
        inv_transpose_3x3: precomputed (A^{-1})^T (optional)

    Returns:
        Transformed and normalized normal [nx, ny, nz]
    """
    if inv_transpose_3x3 is None:
        try:
            inv_transpose_3x3 = np.linalg.inv(linear_3x3).T
        except np.linalg.LinAlgError:
            # Non-invertible; fall back to applying A and renormalizing
            inv_transpose_3x3 = linear_3x3
    n = inv_transpose_3x3 @ np.asarray(normal, dtype=np.float64)
    # Normalize
    L = np.linalg.norm(n)
    if L > 0:
        n = n / L
    return n


def _reorder_vec3(v, perm):
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[perm[0]], v[perm[1]], v[perm[2]]], dtype=np.float64)


def _unpermute_vec3(v_ord, perm):
    out = np.zeros(3, dtype=np.float64)
    out[perm[0]] = v_ord[0]
    out[perm[1]] = v_ord[1]
    out[perm[2]] = v_ord[2]
    return out


def process_obj_with_axis(input_path, output_path, scale_factor=None, transform_matrix_4x4=None, perm=(0,1,2), inv_perm=(0,1,2), axis_order='xyz'):
    """
    Like process_obj, but interprets the transform matrix in a given axis order.

    The matrix is assumed to act on coordinates ordered as `perm` (e.g., for 'zyx', perm=(2,1,0)).
    We reorder vertex and normal into that order before applying, then un-permute back to xyz.
    """
    # Precompute normal transform in the matrix's axis order space
    A = None
    A_inv_T = None
    if transform_matrix_4x4 is not None:
        A = transform_matrix_4x4[:3, :3]
        try:
            A_inv_T = np.linalg.inv(A).T
        except np.linalg.LinAlgError:
            A_inv_T = None

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
                    if transform_matrix_4x4 is not None:
                        v_ord = _reorder_vec3(v, perm)
                        v_ord_t = apply_affine_transform(v_ord, transform_matrix_4x4)
                        v_t = _unpermute_vec3(v_ord_t, perm)
                    else:
                        v_t = v
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
                    if A is not None:
                        n_ord = _reorder_vec3(n, perm)
                        n_ord_t = transform_normal(n_ord, A, A_inv_T)
                        n_t = _unpermute_vec3(n_ord_t, perm)
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
    """
    Process an OBJ file with scaling and/or affine transformation.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
        scale_factor: Scale multiplier (optional)
        transform_matrix_4x4: 4x4 affine transformation matrix (optional)
    """
    # Precompute linear part and its inverse-transpose for normals
    A = None
    A_inv_T = None
    if transform_matrix_4x4 is not None:
        A = transform_matrix_4x4[:3, :3]
        try:
            A_inv_T = np.linalg.inv(A).T
        except np.linalg.LinAlgError:
            A_inv_T = None  # handled per-normal

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('v '):  # Vertex position
                parts = line.split()
                if len(parts) >= 4:
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    
                    # Apply scale if specified
                    if scale_factor is not None:
                        vertex = [v * scale_factor for v in vertex]
                    
                    # Apply affine transform if specified
                    if transform_matrix_4x4 is not None:
                        vertex = apply_affine_transform(vertex, transform_matrix_4x4)
                    
                    outfile.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
                else:
                    outfile.write(line)
            elif line.startswith('vn '):  # Vertex normal
                parts = line.split()
                if len(parts) >= 4:
                    normal = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    # For uniform scale, normals unaffected after normalization; only use A
                    if A is not None:
                        normal = transform_normal(normal, A, A_inv_T)
                        outfile.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
                    else:
                        # No transform matrix: keep as-is
                        outfile.write(line)
                else:
                    outfile.write(line)
            else:
                # Copy all other lines unchanged (faces, normals, texture coords, etc.)
                outfile.write(line)


def load_transform_from_json(json_path):
    """
    Load affine transformation matrix from JSON file.
    
    Args:
        json_path: Path to JSON file containing transformation matrix
    
    Returns:
        4x4 numpy array representing the affine transformation
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'transformation_matrix' not in data:
        raise ValueError(f"JSON file must contain 'transformation_matrix' field")
    
    matrix = np.array(data['transformation_matrix'], dtype=np.float64)
    
    if matrix.shape == (3, 4):
        # Promote to 4x4 by appending [0,0,0,1]
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        matrix4 = np.vstack([matrix, bottom])
    elif matrix.shape == (4, 4):
        matrix4 = matrix
        # Optional: sanity-check bottom row is [0,0,0,1] within tolerance
        if not (np.allclose(matrix4[3, :3], 0.0) and np.isclose(matrix4[3, 3], 1.0)):
            raise ValueError("Bottom affine row must be [0,0,0,1]")
    else:
        raise ValueError(f"Transformation matrix must be 3x4 or 4x4, got {matrix.shape}")
    
    return matrix4


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
    
    # Helper to parse axis order into a permutation and its inverse
    def axis_perm(order: str):
        order = order.lower()
        mapping = {'x':0,'y':1,'z':2}
        perm = tuple(mapping[c] for c in order)
        inv = [0,0,0]
        for i,p in enumerate(perm):
            inv[p] = i
        return perm, tuple(inv)

    perm, inv_perm = axis_perm(args.axis_order)

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
        process_obj_with_axis(obj_file, output_file, args.scale, transform_matrix_4x4, perm, inv_perm, args.axis_order)
    
    print(f"Done! Processed {len(obj_files)} file(s)")
    return 0


if __name__ == '__main__':
    exit(main())
