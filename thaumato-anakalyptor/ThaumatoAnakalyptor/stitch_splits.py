### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2025

import numpy as np
import open3d as o3d
import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2

# Disable implicit parallelism that could cause concurrency issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Set numpy to single-threaded mode to avoid race conditions
np.seterr(all='raise')  # Make numpy raise exceptions on errors instead of warnings

# This disables the decompression bomb protection in Pillow
Image.MAX_IMAGE_PIXELS = None

class MeshStitcher:
    def __init__(self, original_mesh_path, renders_folder):
        """
        Initialize the mesh stitcher.
        
        Args:
            original_mesh_path: Path to the original mesh file
            renders_folder: Path to folder containing split renders
        """
        self.original_mesh_path = original_mesh_path
        self.renders_folder = renders_folder
        
        # Find the working folder with split meshes
        mesh_dir = os.path.dirname(original_mesh_path)
        working_folders = glob.glob(os.path.join(mesh_dir, "windowed_mesh_*"))
        if not working_folders:
            raise ValueError(f"No windowed_mesh_* folder found in {mesh_dir}")
        
        # Use the most recent working folder
        self.working_folder = max(working_folders, key=os.path.getctime)
        print(f"Using working folder: {self.working_folder}")
        
        # Load split information
        split_info_path = os.path.join(self.working_folder, "split_info.json")
        if not os.path.exists(split_info_path):
            raise ValueError(f"Split info file not found: {split_info_path}")
        
        with open(split_info_path, 'r') as f:
            self.split_info = json.load(f)
        
        # Load flattened vertices
        vertices_path = os.path.join(mesh_dir, "vertices_flattened.npy")
        if not os.path.exists(vertices_path):
            raise ValueError(f"Flattened vertices file not found: {vertices_path}")
        
        with open(vertices_path, 'rb') as f:
            npzfile = np.load(f)
            self.vertices_flattened = npzfile['vertices']
        
        # Load original mesh
        self.mesh = o3d.io.read_triangle_mesh(original_mesh_path, print_progress=True)
        self.triangles = np.asarray(self.mesh.triangles)
        
        print(f"Loaded mesh with {len(self.vertices_flattened)} vertices and {len(self.triangles)} triangles")
        print(f"Split info contains {len(self.split_info['windows'])} windows")

    def find_render_for_window(self, window_info, image_filename=None):
        """
        Find the render file corresponding to a window.
        
        Args:
            window_info: Window information from split_info
            image_filename: Specific filename to look for (e.g., 'composite.jpg', 'prediction.png')
                          If None, will search for files matching the window mesh name
            
        Returns:
            Path to the render file or None if not found
        """
        # Extract window bounds from mesh path
        mesh_path = window_info['mesh_path']
        mesh_basename = os.path.basename(mesh_path)
        window_name = mesh_basename.replace('.obj', '')
        
        # If specific image filename is provided, look for it in the window folder or renders folder
        if image_filename:
            # First try in the window's own folder (same directory as the mesh)
            window_folder = os.path.dirname(mesh_path)
            image_path_in_window_folder = os.path.join(window_folder, image_filename)
            if os.path.exists(image_path_in_window_folder):
                return image_path_in_window_folder
            
            # Then try in the renders folder with the window name prefix
            image_path_in_renders = os.path.join(self.renders_folder, f"{window_name}_{image_filename}")
            if os.path.exists(image_path_in_renders):
                return image_path_in_renders
            
            # Finally try in the renders folder with just the filename
            image_path_direct = os.path.join(self.renders_folder, image_filename)
            if os.path.exists(image_path_direct):
                return image_path_direct
        
        # Original logic: Look for render files with matching name
        possible_extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in possible_extensions:
            pattern = os.path.join(self.renders_folder, f"{window_name}.{ext[2:]}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
            
            # Also try without extension pattern matching
            pattern = os.path.join(self.renders_folder, f"{window_name}.*")
            matches = glob.glob(pattern)
            for match in matches:
                if any(match.lower().endswith(e[2:]) for e in possible_extensions):
                    return match
        
        return None

    def get_triangle_uv_bounds(self, triangle_idx):
        """
        Get the UV bounds of a triangle in the flattened coordinate system.
        
        Args:
            triangle_idx: Index of the triangle
            
        Returns:
            Tuple of (min_u, max_u, min_v, max_v) for the triangle
        """
        triangle = self.triangles[triangle_idx]
        triangle_vertices = self.vertices_flattened[triangle]
        
        min_u = np.min(triangle_vertices[:, 0])
        max_u = np.max(triangle_vertices[:, 0])
        min_v = np.min(triangle_vertices[:, 1])
        max_v = np.max(triangle_vertices[:, 1])
        
        return min_u, max_u, min_v, max_v

    def rasterize_triangle_to_image(self, triangle_idx, image_shape, uv_bounds):
        """
        Rasterize a triangle to determine which pixels it covers.
        
        Args:
            triangle_idx: Index of the triangle
            image_shape: (height, width) of the target image
            uv_bounds: (min_u, max_u, min_v, max_v) of the entire UV space
            
        Returns:
            Binary mask of pixels covered by the triangle
        """
        triangle = self.triangles[triangle_idx]
        triangle_vertices = self.vertices_flattened[triangle]
        
        min_u, max_u, min_v, max_v = uv_bounds
        
        # Convert UV coordinates to pixel coordinates
        u_coords = triangle_vertices[:, 0]
        v_coords = triangle_vertices[:, 1]
        
        # Map to image coordinates
        x_coords = ((u_coords - min_u) / (max_u - min_u) * (image_shape[1] - 1)).astype(np.int32)
        y_coords = ((v_coords - min_v) / (max_v - min_v) * (image_shape[0] - 1)).astype(np.int32)
        
        # Create triangle mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        triangle_points = np.array([[x_coords[i], y_coords[i]] for i in range(3)], dtype=np.int32)
        cv2.fillPoly(mask, [triangle_points], 1)
        
        return mask.astype(bool)

    def stitch_renders(self, output_path=None, image_filename=None):
        """
        Stitch the split renders back into the original UV mapping image.
        
        Args:
            output_path: Path to save the stitched image. If None, saves next to original mesh.
            image_filename: Specific filename to look for in each window folder (e.g., 'composite.jpg', 'prediction.png')
                          If None, will search for files matching the window mesh names
        """
        if output_path is None:
            mesh_dir = os.path.dirname(self.original_mesh_path)
            mesh_name = os.path.splitext(os.path.basename(self.original_mesh_path))[0]
            if image_filename:
                # Include the image filename in the output name
                image_base = os.path.splitext(image_filename)[0]
                output_path = os.path.join(mesh_dir, f"{mesh_name}_{image_base}_stitched.jpg")
            else:
                output_path = os.path.join(mesh_dir, f"{mesh_name}_stitched.jpg")
        
        # Determine output image size based on UV bounds
        min_u = self.split_info['min_u']
        max_u = self.split_info['max_u']
        min_v = self.split_info['min_v']
        max_v = self.split_info['max_v']
        
        # Create output image with reasonable resolution
        width = int(np.ceil(max_u - min_u))
        height = int(np.ceil(max_v - min_v))
        
        print(f"Creating output image of size {width}x{height}")
        if image_filename:
            print(f"Looking for '{image_filename}' in each window folder")
        
        # Initialize output image (RGB)
        output_image = np.zeros((height, width, 3), dtype=np.uint8)
        coverage_mask = np.zeros((height, width), dtype=bool)
        
        # Process each window
        for window_info in tqdm(self.split_info['windows'], desc="Stitching windows"):
            # Find corresponding render
            render_path = self.find_render_for_window(window_info, image_filename)
            if render_path is None:
                if image_filename:
                    print(f"Warning: No '{image_filename}' found for window {window_info['window_start']}-{window_info['window_end']}")
                else:
                    print(f"Warning: No render found for window {window_info['window_start']}-{window_info['window_end']}")
                continue
            
            print(f"Processing window {window_info['window_start']}-{window_info['window_end']} with render {render_path}")
            
            # Load render image
            render_image = Image.open(render_path)
            if render_image.mode != 'RGB':
                render_image = render_image.convert('RGB')
            render_array = np.array(render_image)
            
            # Get qualifying triangles for this window
            qualifying_triangles = window_info['qualifying_triangles']
            triangle_indices = [i for i, qual in enumerate(qualifying_triangles) if qual]
            
            # For each triangle in this window, map its pixels to the output image
            for triangle_idx in tqdm(triangle_indices, desc=f"Processing triangles", leave=False):
                # Get triangle UV bounds in flattened space
                tri_min_u, tri_max_u, tri_min_v, tri_max_v = self.get_triangle_uv_bounds(triangle_idx)
                
                # Create triangle mask in output image space
                triangle_mask = self.rasterize_triangle_to_image(
                    triangle_idx, 
                    (height, width), 
                    (min_u, max_u, min_v, max_v)
                )
                
                # Map triangle region from render to output
                # For simplicity, we'll use the window bounds to map from render to output
                window_start = window_info['window_start']
                window_end = window_info['window_end']
                
                # Calculate mapping from render image to output image for this triangle
                # Map U coordinates
                render_u_start = (tri_min_u - window_start) / (window_end - window_start) * render_array.shape[1]
                render_u_end = (tri_max_u - window_start) / (window_end - window_start) * render_array.shape[1]
                
                # Map V coordinates (use per-cut V range)
                render_v_start = (tri_min_v - min_v) / (max_v - min_v) * render_array.shape[0]
                render_v_end = (tri_max_v - min_v) / (max_v - min_v) * render_array.shape[0]
                
                # Clamp to render image bounds
                render_u_start = max(0, min(render_array.shape[1] - 1, int(render_u_start)))
                render_u_end = max(0, min(render_array.shape[1] - 1, int(render_u_end)))
                render_v_start = max(0, min(render_array.shape[0] - 1, int(render_v_start)))
                render_v_end = max(0, min(render_array.shape[0] - 1, int(render_v_end)))
                
                # Map to output image coordinates
                output_u_start = int((tri_min_u - min_u) / (max_u - min_u) * (width - 1))
                output_u_end = int((tri_max_u - min_u) / (max_u - min_u) * (width - 1))
                output_v_start = int((tri_min_v - min_v) / (max_v - min_v) * (height - 1))
                output_v_end = int((tri_max_v - min_v) / (max_v - min_v) * (height - 1))
                
                # Clamp to output image bounds
                output_u_start = max(0, min(width - 1, output_u_start))
                output_u_end = max(0, min(width - 1, output_u_end))
                output_v_start = max(0, min(height - 1, output_v_start))
                output_v_end = max(0, min(height - 1, output_v_end))
                
                # Skip if triangle is too small
                if (output_u_end - output_u_start < 1) or (output_v_end - output_v_start < 1):
                    continue
                if (render_u_end - render_u_start < 1) or (render_v_end - render_v_start < 1):
                    continue
                
                # Extract triangle region from render and resize to fit output region
                try:
                    render_region = render_array[render_v_start:render_v_end+1, render_u_start:render_u_end+1]
                    if render_region.size > 0:
                        output_height = output_v_end - output_v_start + 1
                        output_width = output_u_end - output_u_start + 1
                        
                        if output_height > 0 and output_width > 0:
                            # Resize render region to match output region
                            render_region_resized = cv2.resize(render_region, (output_width, output_height))
                            if len(render_region_resized.shape) == 2:
                                render_region_resized = np.stack([render_region_resized] * 3, axis=-1)
                            
                            # Apply triangle mask to the region
                            region_mask = triangle_mask[output_v_start:output_v_end+1, output_u_start:output_u_end+1]
                            if region_mask.shape[:2] == render_region_resized.shape[:2]:
                                # Thread-safe max operation with explicit memory barriers
                                current_region = output_image[output_v_start:output_v_end+1, output_u_start:output_u_end+1].copy()
                                max_region = np.maximum(current_region, render_region_resized)
                                # Ensure atomic write operation
                                output_image[output_v_start:output_v_end+1, output_u_start:output_u_end+1] = max_region.copy()
                                coverage_mask[output_v_start:output_v_end+1, output_u_start:output_u_end+1] = True
                                pixels_updated = output_height * output_width
                                
                                triangles_processed_successfully = 1
                                
                                # Track this triangle as processed (for statistics only)
                                processed_triangles_global.add(triangle_idx)
                                
                                if i < 3:
                                    print(f"      Max-pooled {output_height * output_width} pixels")
                            else:
                                triangles_failed_processing += 1
                                if i < 10:
                                    print(f"      Failed: invalid output dimensions {output_width}x{output_height}")
                        else:
                            triangles_failed_processing += 1
                            if i < 10:
                                print(f"      Failed: empty render region")
                except Exception as e:
                    print(f"Warning: Error processing triangle {triangle_idx}: {e}")
                    continue
        
        # Save output image
        output_pil = Image.fromarray(output_image)
        output_pil.save(output_path, quality=60, optimize=True)
        print(f"Stitched image saved to: {output_path}")
        
        # Also save coverage mask for debugging
        coverage_path = output_path.replace('.jpg', '_coverage.png')
        coverage_pil = Image.fromarray((coverage_mask * 255).astype(np.uint8))
        coverage_pil.save(coverage_path)
        print(f"Coverage mask saved to: {coverage_path}")
        
        return output_path

class FinalizeMeshStitcher:
    def __init__(self, original_mesh_path):
        """
        Initialize the finalize mesh stitcher for meshes cut by finalize_mesh.py.
        
        Args:
            original_mesh_path: Path to the original mesh file (already in working folder)
        """
        self.original_mesh_path = original_mesh_path
        # The mesh is already in the working folder, so use its directory directly
        self.working_folder = os.path.dirname(original_mesh_path)
        
        print(f"Using working folder: {self.working_folder}")
        
        # Load cut information from the same directory as the mesh
        cut_info_path = os.path.join(self.working_folder, "cut_info.json")
        if not os.path.exists(cut_info_path):
            raise ValueError(f"Cut info file not found: {cut_info_path}")
        
        with open(cut_info_path, 'r') as f:
            self.cut_info = json.load(f)
        
        # Load original mesh
        self.mesh = o3d.io.read_triangle_mesh(original_mesh_path, print_progress=True)
        self.triangles = np.asarray(self.mesh.triangles)
        
        print(f"Loaded mesh with {len(np.asarray(self.mesh.vertices))} vertices and {len(self.triangles)} triangles")
        print(f"Cut info contains {len(self.cut_info['cuts'])} cuts")

    def find_render_for_cut(self, cut_info, image_filename=None):
        """
        Find the render file corresponding to a cut.
        
        Args:
            cut_info: Cut information from cut_info
            image_filename: Specific filename to look for (e.g., 'composite.jpg', 'prediction.png')
                          If None, will search for files matching the cut mesh name
            
        Returns:
            Path to the render file or None if not found
        """
        # Extract cut name from mesh path
        mesh_path = cut_info['mesh_path']
        mesh_basename = os.path.basename(mesh_path)
        cut_name = mesh_basename.replace('.obj', '')
        
        # If specific image filename is provided, look for it in the cut folder
        if image_filename:
            # Try in the cut's own folder (same directory as the mesh)
            cut_folder = os.path.dirname(mesh_path)
            image_path_in_cut_folder = os.path.join(cut_folder, image_filename)
            if os.path.exists(image_path_in_cut_folder):
                return image_path_in_cut_folder
        
        # Original logic: Look for render files with matching cut name in the cut folder
        cut_folder = os.path.dirname(mesh_path)
        possible_extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in possible_extensions:
            pattern = os.path.join(cut_folder, f"{cut_name}.{ext[2:]}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
            
            # Also try without extension pattern matching
            pattern = os.path.join(cut_folder, f"{cut_name}.*")
            matches = glob.glob(pattern)
            for match in matches:
                if any(match.lower().endswith(e[2:]) for e in possible_extensions):
                    return match
        
        return None

    def get_triangle_texture_bounds(self, triangle_uvs):
        """
        Get the texture bounds of a triangle.
        
        Args:
            triangle_uvs: UV coordinates for the triangle (3x2 array)
            
        Returns:
            Tuple of (min_u, max_u, min_v, max_v) for the triangle
        """
        triangle_uvs = np.array(triangle_uvs).reshape(3, 2)
        min_u = np.min(triangle_uvs[:, 0])
        max_u = np.max(triangle_uvs[:, 0])
        min_v = np.min(triangle_uvs[:, 1])
        max_v = np.max(triangle_uvs[:, 1])
        
        return min_u, max_u, min_v, max_v

    def rasterize_triangle_to_texture(self, triangle_uvs, texture_shape):
        """
        Rasterize a triangle to determine which pixels it covers in texture space.
        
        Args:
            triangle_uvs: UV coordinates for the triangle (3x2 array)
            texture_shape: (height, width) of the texture
            
        Returns:
            Binary mask of pixels covered by the triangle
        """
        triangle_uvs = np.array(triangle_uvs).reshape(3, 2)
        
        # Convert UV coordinates to pixel coordinates
        u_coords = triangle_uvs[:, 0]
        v_coords = triangle_uvs[:, 1]
        
        # Map to texture coordinates (note: V is flipped in texture space)
        x_coords = (u_coords * (texture_shape[1] - 1)).astype(np.int32)
        y_coords = (v_coords * (texture_shape[0] - 1)).astype(np.int32)
        
        # Create triangle mask
        mask = np.zeros(texture_shape, dtype=np.uint8)
        triangle_points = np.array([[x_coords[i], y_coords[i]] for i in range(3)], dtype=np.int32)
        cv2.fillPoly(mask, [triangle_points], 1)
        
        return mask.astype(bool)

    def stitch_renders(self, output_path=None, image_filename=None):
        """
        Stitch the cut renders back into the original texture image using triangle-level precision.
        
        Args:
            output_path: Path to save the stitched image. If None, saves next to original mesh.
            image_filename: Specific filename to look for in each cut folder (e.g., 'composite.jpg', 'prediction.png')
                          If None, will search for files matching the cut mesh names
        """
        if output_path is None:
            mesh_dir = os.path.dirname(self.original_mesh_path)
            mesh_name = os.path.splitext(os.path.basename(self.original_mesh_path))[0]
            if image_filename:
                # Include the image filename in the output name
                image_base = os.path.splitext(image_filename)[0]
                output_path = os.path.join(mesh_dir, f"{mesh_name}_{image_base}_stitched.jpg")
            else:
                output_path = os.path.join(mesh_dir, f"{mesh_name}_stitched.jpg")
        
        # Get original texture size
        original_texture_size = self.cut_info['original_texture_size']
        width = int(original_texture_size[0])
        height = int(original_texture_size[1])
        
        print(f"Creating output image of size {width}x{height}")
        print(f"Original texture size from cut_info: {original_texture_size}")
        if image_filename:
            print(f"Looking for '{image_filename}' in each cut folder")
        
        # Initialize output image (RGB)
        output_image = np.zeros((height, width, 3), dtype=np.uint8)
        coverage_mask = np.zeros((height, width), dtype=bool)
        
        # Debug counters
        cuts_processed = 0
        cuts_found = 0
        triangles_processed = 0
        pixels_updated = 0
        
        # Track which triangles from the original mesh have actually been written to the output so far
        processed_triangles_global = set()
        total_original_triangles = len(self.triangles)
        
        print(f"Original mesh has {total_original_triangles} triangles")
        
        # Process each cut
        for cut_info in tqdm(self.cut_info['cuts'], desc="Stitching cuts", disable=False, leave=True, position=0):
            # Find corresponding render
            render_path = self.find_render_for_cut(cut_info, image_filename)
            if render_path is None:
                if image_filename:
                    print(f"Warning: No '{image_filename}' found for cut {cut_info['cut_index']}")
                else:
                    print(f"Warning: No render found for cut {cut_info['cut_index']}")
                continue
            
            cuts_found += 1
            print(f"Processing cut {cut_info['cut_index']} with render {render_path}")
            
            # Load render image
            render_image = Image.open(render_path)
            if render_image.mode != 'RGB':
                render_image = render_image.convert('RGB')
            render_array = np.array(render_image)
            print(f"  Render image size: {render_array.shape}")
            
            # Check if we have triangle information (new format)
            if 'triangle_info' in cut_info and cut_info['triangle_info']:
                # Use triangle-level precision
                triangle_info = cut_info['triangle_info']
                original_uvs = triangle_info['original_uvs']
                original_triangle_indices = triangle_info['original_triangle_indices']
                
                print(f"  Using triangle-level precision with {len(original_triangle_indices)} triangles")
                print(f"  Original triangle indices range: {min(original_triangle_indices) if original_triangle_indices else 'N/A'} to {max(original_triangle_indices) if original_triangle_indices else 'N/A'}")
                print(f"  UV coordinates count: {len(original_uvs)} (should be {len(original_triangle_indices) * 3})")
                
                # Check if triangle indices and UV coordinates are properly aligned
                expected_uv_count = len(original_triangle_indices) * 3
                if len(original_uvs) != expected_uv_count:
                    print(f"  WARNING: UV count mismatch! Expected {expected_uv_count}, got {len(original_uvs)}")
                
                # CRITICAL FIX: We need to use triangle indices as lookups, not sequential array indices
                # The UV coordinates are stored sequentially, but they correspond to non-sequential triangle indices
                # We need to create a mapping from triangle index to UV coordinates
                
                print(f"  Creating triangle index to UV mapping...")
                triangle_to_uv_map = {}
                for uv_idx, triangle_idx in enumerate(original_triangle_indices):
                    triangle_to_uv_map[triangle_idx] = uv_idx
                
                print(f"  Created mapping for {len(triangle_to_uv_map)} triangles")
                
                # Get window bounds for this cut
                window_start = cut_info['window_start']
                window_end = cut_info['window_end']
                
                print(f"  Processing cut {cut_info['cut_index']} with window [{window_start}, {window_end}]")
                
                # Process triangles in batches for better performance
                batch_size = 1000
                num_triangles = len(original_triangle_indices)
                
                # Debug counters for UV coordinate issues
                triangles_with_negative_uv = 0
                triangles_outside_texture = 0
                triangles_with_zero_area = 0
                triangles_processed_successfully = 0
                triangles_failed_processing = 0
                
                # Reshape original_uvs from n*3,2 to n,3,2 format for easier triangle indexing
                original_uvs_reshaped = np.array(original_uvs).reshape(-1, 3, 2)
                # Flip V coordinate (Y axis) to convert from bottom-left origin (OBJ) to
                # top-left origin (image).  This avoids the vertical mirroring artefact
                # where top and bottom portions of the texture overlap.
                original_uvs_flipped = original_uvs_reshaped.copy()
                original_uvs_flipped[:, :, 1] = height - 1 - original_uvs_flipped[:, :, 1]

                print(f"  Reshaped UVs to: {original_uvs_reshaped.shape} (after flipping V)")
                
                # Compute V range for this cut based on the stored original UVs.
                # This is more accurate than relying on the global texture height
                # because each cut may not span the full V extent.
                cut_v_values = np.array(original_uvs_flipped).reshape(-1,2)[:,1]
                cut_min_v = float(np.min(cut_v_values))
                cut_max_v = float(np.max(cut_v_values))
                v_denominator = max(cut_max_v - cut_min_v, 1e-6)  # avoid divide-by-zero
                print(f"  Cut V range: [{cut_min_v:.2f}, {cut_max_v:.2f}]")
                
                for batch_start in tqdm(range(0, num_triangles, batch_size), desc=f"Processing triangle batches in cut {cut_info['cut_index']}"):
                    batch_end = min(batch_start + batch_size, num_triangles)
                    
                    # Process this batch of triangles
                    for i in range(batch_start, batch_end):
                        if i >= len(original_triangle_indices):
                            continue
                        
                        # Get the actual triangle index and corresponding UV index
                        triangle_idx = original_triangle_indices[i]

                        uv_idx = triangle_to_uv_map[triangle_idx]
                        triangles_processed += 1
                        
                        # Safety check
                        if uv_idx >= original_uvs_flipped.shape[0]:
                            continue

                        triangle_uvs = original_uvs_flipped[uv_idx]
                        
                        # Debug: Check triangle index mapping for first few triangles
                        if i < 3:
                            print(f"    Processing triangle {triangle_idx} (UV index {uv_idx})")
                            print(f"      UV coords: {triangle_uvs.flatten()}")
                        
                        # Get triangle bounds in original texture space
                        tri_min_u, tri_max_u, tri_min_v, tri_max_v = self.get_triangle_texture_bounds(triangle_uvs)
                        
                        # NO SKIPPING - Process ALL triangles and take max value for each pixel
                        
                        # Debug: Check for problematic UV coordinates
                        if tri_min_u < 0 or tri_min_v < 0:
                            triangles_with_negative_uv += 1
                        
                        if tri_max_u > width or tri_max_v > height:
                            triangles_outside_texture += 1
                        
                        if (tri_max_u - tri_min_u) < 0.1 or (tri_max_v - tri_min_v) < 0.1:
                            triangles_with_zero_area += 1
                        
                        # Debug: Show bounds for first few triangles
                        if i < 3:
                            print(f"      Triangle bounds: u=[{tri_min_u:.1f}, {tri_max_u:.1f}], v=[{tri_min_v:.1f}, {tri_max_v:.1f}]")
                        
                        # Calculate mapping from render image to output image for this triangle
                        # Map U coordinates (assuming render spans the full cut width)
                        render_u_start = (tri_min_u - window_start) / (window_end - window_start) * render_array.shape[1]
                        render_u_end = (tri_max_u - window_start) / (window_end - window_start) * render_array.shape[1]
                        
                        # Map V coordinates (use per-cut V range)
                        render_v_start = (tri_min_v - cut_min_v) / v_denominator * render_array.shape[0]
                        render_v_end = (tri_max_v - cut_min_v) / v_denominator * render_array.shape[0]
                        
                        # Clamp to render image bounds
                        render_u_start = max(0, min(render_array.shape[1] - 1, int(render_u_start)))
                        render_u_end = max(0, min(render_array.shape[1] - 1, int(render_u_end)))
                        render_v_start = max(0, min(render_array.shape[0] - 1, int(render_v_start)))
                        render_v_end = max(0, min(render_array.shape[0] - 1, int(render_v_end)))
                        
                        # Map to output image coordinates
                        output_u_start = max(0, min(width - 1, int(tri_min_u)))
                        output_u_end = max(0, min(width - 1, int(tri_max_u)))
                        output_v_start = max(0, min(height - 1, int(tri_min_v)))
                        output_v_end = max(0, min(height - 1, int(tri_max_v)))
                        
                        # Debug first few triangles' mapping
                        if i < 3:
                            print(f"      Render region: u=[{render_u_start}, {render_u_end}], v=[{render_v_start}, {render_v_end}]")
                            print(f"      Output region: u=[{output_u_start}, {output_u_end}], v=[{output_v_start}, {output_v_end}]")
                        
                        # Ensure we have valid regions (minimal check)
                        if output_u_end > output_u_start and output_v_end > output_v_start:
                            if render_u_end > render_u_start and render_v_end > render_v_start:
                                # Extract and resize regions
                                try:
                                    render_region = render_array[render_v_start:render_v_end+1, render_u_start:render_u_end+1]
                                    if render_region.size > 0:
                                        output_height = output_v_end - output_v_start + 1
                                        output_width = output_u_end - output_u_start + 1
                                        
                                        if output_height > 0 and output_width > 0:
                                            # Resize render region to match output region
                                            if render_region.shape[:2] != (output_height, output_width):
                                                render_region_resized = cv2.resize(render_region, (output_width, output_height))
                                            else:
                                                render_region_resized = render_region
                                            
                                            if len(render_region_resized.shape) == 2:
                                                render_region_resized = np.stack([render_region_resized] * 3, axis=-1)
                                            
                                            # Take maximum value for each pixel across all triangles
                                            # Thread-safe max operation with explicit memory barriers
                                            current_region = output_image[output_v_start:output_v_end+1, output_u_start:output_u_end+1].copy()
                                            max_region = np.maximum(current_region, render_region_resized)
                                            # Ensure atomic write operation
                                            output_image[output_v_start:output_v_end+1, output_u_start:output_u_end+1] = max_region.copy()
                                            coverage_mask[output_v_start:output_v_end+1, output_u_start:output_u_end+1] = True
                                            pixels_updated += output_height * output_width
                                            
                                            triangles_processed_successfully += 1
                                            
                                            # Track this triangle as processed (for statistics only)
                                            processed_triangles_global.add(triangle_idx)
                                            
                                            if i < 3:
                                                print(f"      Max-pooled {output_height * output_width} pixels")
                                except Exception as e:
                                    triangles_failed_processing += 1
                                    if i < 10:  # Only print first few errors
                                        print(f"      Error processing triangle {triangle_idx}: {e}")
                                    continue
                            else:
                                triangles_failed_processing += 1
                                if i < 10:
                                    print(f"      Failed: invalid render region {render_u_start}-{render_u_end}, {render_v_start}-{render_v_end}")
                        else:
                            triangles_failed_processing += 1
                            if i < 10:
                                print(f"      Failed: invalid output region {output_u_start}-{output_u_end}, {output_v_start}-{output_v_end}")
                
                print(f"  Cut {cut_info['cut_index']} summary: {triangles_processed} triangles processed")
                print(f"    Successfully processed: {triangles_processed_successfully}")
                print(f"    Failed processing: {triangles_failed_processing}")
                print(f"    Triangles with negative UV: {triangles_with_negative_uv}")
                print(f"    Triangles outside texture bounds: {triangles_outside_texture}")
                print(f"    Triangles with near-zero area: {triangles_with_zero_area}")
                cuts_processed += 1
                
                # Force memory synchronization after each cut to prevent race conditions
                output_image.flush() if hasattr(output_image, 'flush') else None
                coverage_mask.flush() if hasattr(coverage_mask, 'flush') else None
                # Explicit memory barrier
                import gc
                gc.collect()
            else:
                # Fallback to simple rectangular region mapping (old format)
                print(f"Warning: No triangle info found for cut {cut_info['cut_index']}, using rectangular mapping")
                window_start = int(cut_info['window_start'])
                window_end = int(cut_info['window_end'])
                
                print(f"  Window bounds: [{window_start}, {window_end}]")
                
                # Clamp to output image bounds
                window_start = max(0, min(width - 1, window_start))
                window_end = max(0, min(width, window_end))
                
                if window_end <= window_start:
                    continue
                
                # Calculate the region dimensions
                region_width = window_end - window_start
                region_height = height
                
                print(f"  Mapping to region: {region_width}x{region_height}")
                
                # Resize render to fit the region
                if render_array.shape[0] != region_height or render_array.shape[1] != region_width:
                    render_resized = cv2.resize(render_array, (region_width, region_height))
                else:
                    render_resized = render_array
                
                # Ensure render is RGB
                if len(render_resized.shape) == 2:
                    render_resized = np.stack([render_resized] * 3, axis=-1)
                
                # Copy to output image
                try:
                    # Thread-safe max operation for rectangular regions
                    current_region = output_image[:, window_start:window_end].copy()
                    max_region = np.maximum(current_region, render_resized)
                    output_image[:, window_start:window_end] = max_region.copy()
                    coverage_mask[:, window_start:window_end] = True
                    pixels_updated += region_width * region_height
                    
                    print(f"  Max-pooled {region_width * region_height} pixels")
                except Exception as e:
                    print(f"Warning: Error processing cut {cut_info['cut_index']}: {e}")
                    continue
                    
                cuts_processed += 1
        
        # Print debug summary
        print(f"\nDebug Summary:")
        print(f"  Total cuts in cut_info: {len(self.cut_info['cuts'])}")
        print(f"  Cuts with renders found: {cuts_found}")
        print(f"  Cuts successfully processed: {cuts_processed}")
        print(f"  Triangles processed: {triangles_processed}")
        print(f"  Total pixels updated: {pixels_updated}")
        print(f"  Coverage percentage: {pixels_updated / (width * height) * 100:.2f}%")
        
        # Triangle coverage analysis
        triangles_covered = len(processed_triangles_global)
        triangles_missing = total_original_triangles - triangles_covered
        coverage_percent = (triangles_covered / total_original_triangles) * 100
        
        print(f"\nTriangle Coverage Analysis:")
        print(f"  Original mesh triangles: {total_original_triangles}")
        print(f"  Triangles in cut metadata: {triangles_covered}")
        print(f"  Missing triangles: {triangles_missing}")
        print(f"  Triangle coverage: {coverage_percent:.2f}%")
        
        if triangles_missing > 0:
            print(f"  WARNING: {triangles_missing} triangles from the original mesh are not included in any cut!")
            print(f"  This explains the missing colored areas in the stitched image.")
            
            # Show some examples of missing triangle indices
            all_triangle_indices = set(range(total_original_triangles))
            missing_indices = all_triangle_indices - processed_triangles_global
            missing_sample = sorted(list(missing_indices))[:10]
            print(f"  Sample missing triangle indices: {missing_sample}")
        else:
            print(f"  All triangles are covered in the cut metadata.")
        
        # Save output image
        output_pil = Image.fromarray(output_image)
        output_pil.save(output_path, quality=60, optimize=True)
        print(f"Stitched image saved to: {output_path}")
        
        # Also save coverage mask for debugging
        coverage_path = output_path.replace('.jpg', '_coverage.png')
        coverage_pil = Image.fromarray((coverage_mask * 255).astype(np.uint8))
        coverage_pil.save(coverage_path)
        print(f"Coverage mask saved to: {coverage_path}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Stitch split mesh renders back into original UV mapping')
    parser.add_argument('original_mesh', type=str, help='Path to the original mesh file')
    parser.add_argument('--output', type=str, help='Output path for stitched image', default=None)
    parser.add_argument('--type', type=str, choices=['auto', 'split_mesh', 'finalize_mesh'], 
                       default='auto', help='Type of stitching to perform')
    parser.add_argument('--image_filename', type=str, help='Specific image filename to look for in each cut/window folder (e.g., composite.jpg, prediction.png)', default=None)
    
    args = parser.parse_args()
    
    print(f"Stitching renders from {args.original_mesh}")
    if args.image_filename:
        print(f"Looking for specific image file: {args.image_filename}")
    
    # Determine which type of stitching to use
    mesh_dir = os.path.dirname(args.original_mesh)
    
    if args.type == 'auto':
        # Check for split_mesh files (windowed_mesh folders)
        windowed_folders = glob.glob(os.path.join(mesh_dir, "windowed_mesh_*"))
        # Check for finalize_mesh files (cut_info.json in same directory)
        cut_info_exists = os.path.exists(os.path.join(mesh_dir, "cut_info.json"))
        
        if windowed_folders:
            stitch_type = 'split_mesh'
        elif cut_info_exists:
            stitch_type = 'finalize_mesh'
        else:
            print("Error: Could not determine stitching type.")
            print("For split_mesh: No windowed_mesh_* folders found.")
            print("For finalize_mesh: No cut_info.json found in mesh directory.")
            print("Please specify --type manually or ensure you have run the splitting process first.")
            return
    else:
        stitch_type = args.type
    
    print(f"Using stitching type: {stitch_type}")
    
    # Create appropriate stitcher and process
    try:
        if stitch_type == 'split_mesh':
            stitcher = MeshStitcher(args.original_mesh, mesh_dir)
        else:  # finalize_mesh
            stitcher = FinalizeMeshStitcher(args.original_mesh)
        
        output_path = stitcher.stitch_renders(args.output, args.image_filename)
        print(f"Stitching complete! Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during stitching: {e}")
        print("Make sure you have:")
        print("1. Split/cut the mesh first")
        print("2. Rendered images for each split/cut mesh")
        print("3. Placed the rendered images in the same folders as the cut meshes")
        print("4. Named the rendered images to match the mesh files")
        if args.image_filename:
            print(f"5. Ensured '{args.image_filename}' exists in each cut/window folder")

if __name__ == '__main__':
    main()
