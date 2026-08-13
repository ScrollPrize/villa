#!/usr/bin/env python3
"""
This script processes OBJ files in a given directory tree.
For each OBJ file, it loads the mesh using trimesh and prints detailed
debugging information (extents, bounding box volume, computed volume before/after repair).
It optionally attempts to repair non-watertight meshes (using basic or advanced repair via PyMeshFix),
and allows a scale factor (e.g., converting cm to m).
The final results (file path and volume) are written to a CSV file.
"""

import os
import csv
import argparse
import trimesh
import trimesh.repair

# Try to import PyMeshFix for advanced repair.
try:
    import pymeshfix
except ImportError:
    pymeshfix = None

def basic_repair(mesh, debug=False):
    """
    Perform several basic repair steps on the mesh.
    """
    if debug:
        print("    [DEBUG] Performing basic repair: fill holes.")
    trimesh.repair.fill_holes(mesh)
    return mesh

def advanced_repair(mesh, debug=False):
    """
    Use PyMeshFix to attempt a robust repair.
    """
    if pymeshfix is None:
        if debug:
            print("    [DEBUG] PyMeshFix not available; falling back to basic repair.")
        return basic_repair(mesh, debug=debug)
    
    if debug:
        print("    [DEBUG] Performing advanced repair using PyMeshFix.")
    mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mf.repair()
    repaired_mesh = trimesh.Trimesh(vertices=mf.v, faces=mf.f, process=False)
    repaired_mesh.fix_normals()
    return repaired_mesh

def process_obj_files(root_dir, csv_file, fix_holes=False,
                      advanced_repair_flag=False, scale_factor=1.0, debug=False):
    """
    Process all OBJ files under the specified root directory.
    
    For each OBJ file:
      - Load the mesh.
      - Print debugging info (initial extents, bounding box volume, volume before repair).
      - Optionally apply a scaling factor (e.g., convert from cm to m).
      - If the mesh is watertight, compute its volume.
      - If not, and repair is enabled, attempt repair before computing volume.
    
    The file's relative path and computed volume are saved to a CSV file.
    """
    results = []  # List to store results as [relative file path, computed volume]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.obj'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing: {file_path}")
                try:
                    mesh = trimesh.load(file_path, force='mesh')
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
                    continue

                # Debug: Print initial mesh information.
                if debug:
                    print(f"    [DEBUG] Initial mesh extents: {mesh.extents}")
                    print(f"    [DEBUG] Initial bounding box volume: {mesh.bounding_box.volume}")
                    print(f"    [DEBUG] Volume before any repair: {mesh.volume}")

                # Apply scale factor if provided.
                if scale_factor != 1.0:
                    if debug:
                        print(f"    [DEBUG] Applying scale factor: {scale_factor}")
                    mesh.apply_scale(scale_factor)
                    if debug:
                        print(f"    [DEBUG] Mesh extents after scaling: {mesh.extents}")

                # Ensure normals are consistent.
                mesh.fix_normals()

                rel_path = os.path.basename(os.path.dirname(file_path))
                
                # If the mesh is already watertight, calculate its volume.
                if mesh.is_watertight:
                    volume = mesh.volume
                    if volume < 0:
                        if debug:
                            print("    [DEBUG] Negative volume computed; taking absolute value.")
                        volume = abs(volume)
                    print(f"  Watertight! Volume = {volume}")
                    results.append([rel_path, volume])
                else:
                    print("  Mesh is not watertight.")
                    if fix_holes:
                        # Save the volume before repair (for comparison).
                        vol_before = mesh.volume
                        print("  Attempting repair...")
                        if advanced_repair_flag:
                            mesh = advanced_repair(mesh, debug=debug)
                        else:
                            mesh = basic_repair(mesh, debug=debug)
                        mesh.fix_normals()
                        vol_after = mesh.volume
                        if debug:
                            print(f"    [DEBUG] Volume after repair: {vol_after}")
                            # Warn if the volume changed drastically.
                            if vol_before > 0 and (vol_after / vol_before) < 0.1:
                                print("    [DEBUG] Warning: Repaired volume is less than 10% of the pre-repair volume.")
                        if mesh.is_watertight:
                            volume = vol_after
                            if volume < 0:
                                if debug:
                                    print("    [DEBUG] Negative volume after repair; taking absolute value.")
                                volume = abs(volume)
                            print(f"  Repaired! Now watertight. Volume = {volume}")
                            results.append([rel_path, volume])
                        else:
                            print("  Mesh is still not watertight after repair; skipping volume calculation.")
                            vol_after = mesh.volume
                            print(f"Volume {vol_after}")
                            results.append([rel_path, vol_after])
                    else:
                        print("  Non-watertight mesh.")
                        vol_after = mesh.volume
                        print(f"Volume {vol_after}")
                        results.append([rel_path, vol_after])
    
    # Write results to a CSV file.
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Volume"])
        writer.writerows(results)
    print(f"Results written to {csv_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Process OBJ files to compute volumes for watertight meshes and output results to a CSV file. "
                    "Includes debugging output to help diagnose unit and repair issues."
    )
    parser.add_argument("root_dir", help="Root directory to search for OBJ files (recursive).")
    parser.add_argument("output_csv", help="Path to the output CSV file (e.g., results.csv).")
    parser.add_argument("--fix-holes", action="store_true", help="Attempt to repair non-watertight meshes.")
    parser.add_argument("--advanced-repair", action="store_true", help="Use advanced repair methods (via PyMeshFix) if available.")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="Scale factor to apply to mesh vertices (e.g., 0.01 to convert from cm to m).")
    parser.add_argument("--debug", action="store_true", help="Print debugging information.")
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        parser.error(f"Directory '{args.root_dir}' does not exist or is not a directory.")

    process_obj_files(
        args.root_dir,
        args.output_csv,
        fix_holes=args.fix_holes,
        advanced_repair_flag=args.advanced_repair,
        scale_factor=args.scale_factor,
        debug=args.debug
    )

if __name__ == '__main__':
    main()
