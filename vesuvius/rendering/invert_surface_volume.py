#!/usr/bin/env python3
import os
import glob
import shutil
import tempfile
import logging
import argparse

import numpy as np
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
import open3d as o3d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import cv2
import zarr

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#################################################################
# MeshLoader
#################################################################
class MeshLoader:
    def __init__(self, obj_path, image_rows, image_cols, max_side_triangle=10):
        """
        Loads an OBJ mesh, extracts vertices, normals, triangle indices,
        and triangle UV coordinates. Splits any UV‐space triangles whose
        bounding box exceeds max_side_triangle in either U or V.
        """
        self.obj_path = obj_path
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.max_side_triangle = max_side_triangle

        self._load_mesh()
        self._adjust_triangle_sizes()

    def _load_mesh(self):
        # Copy to a temporary location and read with Open3D
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpobj = os.path.join(tmpdir, os.path.basename(self.obj_path))
            shutil.copyfile(self.obj_path, tmpobj)
            mesh = o3d.io.read_triangle_mesh(tmpobj)

        mesh.compute_vertex_normals()
        self.vertices = np.asarray(mesh.vertices)          # (V, 3)
        self.normals  = np.asarray(mesh.vertex_normals)    # (V, 3)
        self.triangles = np.asarray(mesh.triangles)        # (T, 3)

        # Open3D stores triangle_uvs in a flat array of length T*3
        uv_flat = np.asarray(mesh.triangle_uvs).reshape(-1, 2)
        if uv_flat.shape[0] != self.triangles.shape[0] * 3:
            raise ValueError("Unexpected number of UVs in mesh.")
        self.uv = uv_flat.reshape(-1, 3, 2)  # (T, 3, 2), normalized [0..1]

        # Build per‐triangle vertex and normal arrays
        self.triangle_vertices = self.vertices[self.triangles]  # (T, 3, 3)
        self.triangle_normals  = self.normals[self.triangles]   # (T, 3, 3)

        # Scale UV into pixel‐space so that U ∈ [0..image_cols−1], V ∈ [0..image_rows−1]
        scale = np.array([self.image_cols - 1, self.image_rows - 1], dtype=np.float32)
        self.uv_pixels = (self.uv * scale).astype(np.float32)   # (T, 3, 2)

        logger.info(f"Loaded mesh: {self.triangle_vertices.shape[0]} triangles.")
        logger.info(f"Scaled UV to pixel coords using (cols={self.image_cols}, rows={self.image_rows}).")

    def _adjust_triangle_sizes(self):
        """
        If a triangle’s
        UV‐bbox (ceil(max_uv) − floor(min_uv)) exceeds max_side_triangle
        in either dimension, split it along the longer axis until
        no triangle is bigger than max_side_triangle×max_side_triangle.
        """
        uv     = self.uv_pixels.copy()            # (T, 3, 2)
        verts  = self.triangle_vertices.copy()     # (T, 3, 3)
        norms  = self.triangle_normals.copy()      # (T, 3, 3)

        uv_good_list     = []
        verts_good_list  = []
        norms_good_list  = []

        def progress_score(maxU, maxV):
            return (np.log(2) - np.log(self.max_side_triangle / maxU)) \
                 + (np.log(2) - np.log(self.max_side_triangle / maxV))

        start_score = None
        with tqdm(total=100, desc="Adjusting triangle sizes") as pbar:
            while True:
                # 1) Compute per‐triangle UV min & max
                tri_min_uv = uv.min(axis=1)   # (N_current, 2)
                tri_max_uv = uv.max(axis=1)   # (N_current, 2)

                # 2) bounding‐box side lengths in pixels
                side_lens = np.ceil(tri_max_uv) - np.floor(tri_min_uv)  # (N_current, 2)
                max_side  = np.max(side_lens, axis=0)  # (maxU, maxV)

                if start_score is None:
                    start_score = progress_score(max_side[0], max_side[1])
                    prog = 0.0
                else:
                    now = progress_score(max_side[0], max_side[1])
                    prog = max(1.0 - now / start_score, 0.0)

                pbar.n = int(prog * 100)
                pbar.refresh()

                # 3) Which triangles exceed max_side_triangle in either U or V?
                mask_large = np.any(side_lens > self.max_side_triangle, axis=1)  # (N_current,)
                if not mask_large.any():
                    # All triangles are small enough → done
                    break

                # 4) Collect “small” triangles this iteration
                uv_good_list.append(uv[~mask_large])      # (N_small, 3, 2)
                verts_good_list.append(verts[~mask_large])  # (N_small, 3, 3)
                norms_good_list.append(norms[~mask_large])  # (N_small, 3, 3)

                # 5) Split the “large” triangles along their longer UV axis
                uv_large     = uv[mask_large]      # (L, 3, 2)
                verts_large  = verts[mask_large]   # (L, 3, 3)
                norms_large  = norms[mask_large]   # (L, 3, 3)
                lens_large   = side_lens[mask_large]  # (L, 2)

                L = uv_large.shape[0]
                split_U = lens_large[:, 0] >= lens_large[:, 1]  # (L,) True=split along U
                split_V = ~split_U

                min_uv_large = tri_min_uv[mask_large]  # (L, 2)
                max_uv_large = tri_max_uv[mask_large]  # (L, 2)

                mask_uv_min = np.zeros((L, 3), dtype=bool)
                mask_uv_max = np.zeros((L, 3), dtype=bool)

                for i in range(L):
                    if split_U[i]:
                        ucoords = uv_large[i, :, 0]  # (3,)
                        umin = min_uv_large[i, 0]
                        umax = max_uv_large[i, 0]
                        mask_uv_min[i] = np.isclose(ucoords, umin)
                        mask_uv_max[i] = np.isclose(ucoords, umax)
                    else:
                        vcoords = uv_large[i, :, 1]  # (3,)
                        vmin = min_uv_large[i, 1]
                        vmax = max_uv_large[i, 1]
                        mask_uv_min[i] = np.isclose(vcoords, vmin)
                        mask_uv_max[i] = np.isclose(vcoords, vmax)

                idx_min = np.argmax(mask_uv_min, axis=1)  # (L,)
                idx_max = np.argmax(mask_uv_max, axis=1)  # (L,)
                ix = np.arange(L)

                new_verts = (verts_large[ix, idx_min] + verts_large[ix, idx_max]) / 2.0  # (L, 3)
                new_norms = (norms_large[ix, idx_min] + norms_large[ix, idx_max]) / 2.0  # (L, 3)
                new_uvs   = (uv_large[ix, idx_min] + uv_large[ix, idx_max]) / 2.0      # (L, 2)

                tri0_uv    = uv_large.copy()
                tri1_uv    = uv_large.copy()
                tri0_verts = verts_large.copy()
                tri1_verts = verts_large.copy()
                tri0_norms = norms_large.copy()
                tri1_norms = norms_large.copy()

                # Replace “min‐vertex” in tri0
                tri0_uv[ix, idx_min]    = new_uvs
                tri0_verts[ix, idx_min] = new_verts
                tri0_norms[ix, idx_min] = new_norms

                # Replace “max‐vertex” in tri1
                tri1_uv[ix, idx_max]    = new_uvs
                tri1_verts[ix, idx_max] = new_verts
                tri1_norms[ix, idx_max] = new_norms

                # Concatenate to form the next‐iteration list of triangles
                uv   = np.concatenate((tri0_uv,    tri1_uv),    axis=0)  # (2L, 3, 2)
                verts = np.concatenate((tri0_verts, tri1_verts), axis=0)  # (2L, 3, 3)
                norms = np.concatenate((tri0_norms, tri1_norms), axis=0)  # (2L, 3, 3)

            # End while
            pbar.n = 100
            pbar.refresh()
            pbar.close()

        # Collect all “small” triangles we saved along the way
        self.uv_pixels         = np.concatenate(uv_good_list,    axis=0)  # (ΣN_small, 3, 2)
        self.triangle_vertices = np.concatenate(verts_good_list,  axis=0)  # (ΣN_small, 3, 3)
        self.triangle_normals  = np.concatenate(norms_good_list,  axis=0)  # (ΣN_small, 3, 3)
        logger.info(f"Adjusted triangles: {self.uv_pixels.shape[0]} total.")


#################################################################
# PPM class: per‐triangle grid → barycentric (O(B·N))
#################################################################
class PPM:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def point_in_triangle(self, grid, tri_uv):
        """
        Compute barycentric coords & inside‐triangle mask *per triangle*.

        Inputs:
          grid:   (B, N, 2)   → N = w*h  (each triangle’s pixel‐centers)
          tri_uv: (B, 3, 2)   → u/v coords of the 3 vertices for each triangle

        Returns:
          bary:    (B, N, 3)  barycentric coords
          inside:  (B, N)     boolean mask per triangle
        """
        # Number of triangles = B, number of points per triangle = N
        B, N, _ = grid.shape

        # Extract per‐triangle edges in UV
        # v0, v1 → each (B, 2)
        v0 = tri_uv[:, 2, :] - tri_uv[:, 0, :]  # (B, 2)
        v1 = tri_uv[:, 1, :] - tri_uv[:, 0, :]  # (B, 2)

        # For each triangle i, we need v2[i] = grid[i] - tri_uv[i, 0]
        # tri_uv[:,0,:].unsqueeze(1) → (B,1,2), broadcast to (B,N,2)
        v2 = grid - tri_uv[:, 0, :].unsqueeze(1)  # (B, N, 2)

        # Dot products:
        # dot00 = (v0·v0), dot01 = (v0·v1), dot11 = (v1·v1)  → (B,)
        dot00 = (v0 * v0).sum(dim=1)  # (B,)
        dot01 = (v0 * v1).sum(dim=1)  # (B,)
        dot11 = (v1 * v1).sum(dim=1)  # (B,)

        # Now dot02 = (v2·v0), dot12 = (v2·v1)  → (B, N)
        # v0.unsqueeze(1) is (B,1,2), broadcast to (B,N,2)
        dot02 = (v2 * v0.unsqueeze(1)).sum(dim=2)  # (B, N)
        dot12 = (v2 * v1.unsqueeze(1)).sum(dim=2)  # (B, N)

        # invDenom: (B,) → unsqueeze to (B,1) for broadcasting
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-20)  # (B,)
        inv_denom = inv_denom.unsqueeze(1)  # (B,1)

        # Compute u, v, w per‐triangle per‐point → (B, N)
        #   u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        #   v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        u = (dot11.unsqueeze(1) * dot02 - dot01.unsqueeze(1) * dot12) * inv_denom  # (B, N)
        v = (dot00.unsqueeze(1) * dot12 - dot01.unsqueeze(1) * dot02) * inv_denom  # (B, N)
        w = 1.0 - u - v                                                            # (B, N)

        # inside iff u>=−ϵ, v>=−ϵ, u+v ≤ 1 + ϵ
        inside = (u >= -self.epsilon) & (v >= -self.epsilon) & ((u + v) <= 1 + self.epsilon)  # (B, N)

        # Stack barycentric coords → (B, N, 3)
        bary = torch.stack([u, v, w], dim=2)  # (B, N, 3)
        bary = normalize(bary.clamp(min=0.0), p=1, dim=2)  # normalize along last dim

        return bary, inside

    def make_grid(self, min_uv, w, h):
        """
        Given:
          min_uv: (B, 2)  floored minimum UV‐coordinate of each triangle
          w, h:   ints   (max_side_triangle)

        Returns:
          grid: (B, w*h, 2) tensor of pixel‐centers:
                for each i in [0..B−1], grid[i] = min_uv[i] + offsets over [0..w−1]×[0..h−1]
        """
        device = min_uv.device
        B = min_uv.shape[0]

        dx = torch.arange(w, device=device)  # [0..w−1]
        dy = torch.arange(h, device=device)  # [0..h−1]
        mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing="xy")  # each (w,h)
        offsets = torch.stack([mesh_dx.reshape(-1), mesh_dy.reshape(-1)], dim=1)  # (w*h, 2)

        base = min_uv.unsqueeze(1)         # (B, 1, 2)
        grid = base + offsets.unsqueeze(0)  # (B, w*h, 2)
        return grid  # (B, w*h, 2)


#################################################################
# InverseMapper
#################################################################
class InverseMapper:
    def __init__(
        self,
        mesh_loader: MeshLoader,
        image_folder: str,
        output_zarr: str,
        step_size: float = 1.0,
        dtype: str = "uint8",
        batch_triangles: int = 200,
    ):
        """
        mesh_loader:       pre‐loaded MeshLoader (with uv_pixels, triangle_vertices, triangle_normals)
        image_folder:      folder containing sorted PNG layers (“00.png”, “01.png”, …)
        output_zarr:       folder path in which to create “volume.zarr”
        step_size:         distance (in world‐units) between adjacent layers
        dtype:             'uint8' or 'uint16' for zarr
        batch_triangles:   how many triangles to process in one PPM batch
                           (B * max_side_triangle^2 points). Smaller → less RAM, more time.
        """
        self.mesh = mesh_loader
        self.image_folder = image_folder
        self.output_zarr = output_zarr
        self.step_size = step_size
        self.dtype = dtype
        self.batch_triangles = batch_triangles

        # Find all PNG layers
        pattern = os.path.join(self.image_folder, "*.png")
        self.layer_paths = sorted(glob.glob(pattern))
        if not self.layer_paths:
            raise FileNotFoundError(f"No PNG files found in {self.image_folder}")
        self.num_layers = len(self.layer_paths)
        logger.info(f"Found {self.num_layers} image layers.")

        # Middle‐layer index
        self.middle_idx = self.num_layers // 2

        # Choose device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            logger.info("Using CUDA for PPM.")
        else:
            logger.info("CUDA not available; using CPU for PPM.")

        # Step 1: compute PPM only for middle layer
        self._compute_middle_layer_ppm()

        # Step 2: allocate zarr based on these points + normals
        self._compute_bounds_and_initialize_zarr()

    def _compute_middle_layer_ppm(self):
        """
        Build the Per‐Pixel Map *only* for the middle layer’s projection.
        For each batch of B triangles:
          1) floor min UV
          2) form a (w×h) grid around that min_uv  → (B, w*h, 2)
          3) run barycentric on each triangle against its own grid
          4) collect only those pixels that lie inside each triangle
        """
        logger.info("Computing PPM for middle layer …")
        tri_uv_np    = self.mesh.uv_pixels             # (T, 3, 2) as numpy
        tri_verts_np = self.mesh.triangle_vertices     # (T, 3, 3) as numpy
        tri_norms_np = self.mesh.triangle_normals      # (T, 3, 3) as numpy

        T = tri_uv_np.shape[0]
        ppm = PPM(epsilon=1e-7)

        all_pixels = []
        all_points = []
        all_norms  = []

        w = h = self.mesh.max_side_triangle

        for start in tqdm(range(0, T, self.batch_triangles), desc="Middle‐layer PPM batches"):
            end = min(start + self.batch_triangles, T)
            B = end - start

            # 1) Load this batch to device
            tri_uv_b    = torch.from_numpy(tri_uv_np[start:end]).float().to(self.device)    # (B,3,2)
            tri_verts_b = torch.from_numpy(tri_verts_np[start:end]).float().to(self.device)  # (B,3,3)
            tri_norms_b = torch.from_numpy(tri_norms_np[start:end]).float().to(self.device)  # (B,3,3)

            # 2) floor the per‐triangle UV min → (B, 2)
            min_uv_b = torch.floor(tri_uv_b.min(dim=1)[0])  # (B,2)

            # 3) form a local (w×h) grid around each min_uv_b → (B, w*h, 2)
            grid_b = ppm.make_grid(min_uv_b, w, h)  # (B, w*h, 2)

            # 4) run barycentric test (per triangle, per its own grid) → bary (B,w*h,3), inside (B,w*h)
            bary, inside = ppm.point_in_triangle(grid_b, tri_uv_b)  # shapes: (B,w*h,3), (B,w*h)

            # 5) find all (triangle_idx, pixel_idx) pairs where inside=True
            tri_idx, pix_idx = torch.nonzero(inside, as_tuple=True)  # each is (N_in,)

            if tri_idx.numel() == 0:
                # No pixels in this batch → skip
                del tri_uv_b, tri_verts_b, tri_norms_b, grid_b, bary, inside
                torch.cuda.empty_cache()
                continue

            # 6) gather those pixel coordinates in (col, row)
            sel_pixels = grid_b[tri_idx, pix_idx]  # (N_in, 2) on device

            # 7) gather corresponding 3D vertices & normals
            chosen_verts = tri_verts_b[tri_idx]  # (N_in, 3, 3)
            chosen_norms = tri_norms_b[tri_idx]  # (N_in, 3, 3)
            chosen_bary  = bary[tri_idx, pix_idx, :]  # (N_in, 3)

            # 8) compute world coordinates and normals
            # world_pts = Σ_j bary_j * chosen_verts[:, j, :]
            world_pts = torch.einsum("ij,ijk->ik", chosen_bary, chosen_verts)  # (N_in, 3)
            world_n   = torch.einsum("ij,ijk->ik", chosen_bary, chosen_norms)  # (N_in, 3)
            world_n   = normalize(world_n, p=2, dim=1)

            # 9) move back to CPU & numpy
            sel_pixels_np = sel_pixels.cpu().numpy().astype(np.int32)     # (N_in, 2)
            world_pts_np  = world_pts.cpu().numpy().astype(np.float32)    # (N_in, 3)
            world_n_np    = world_n.cpu().numpy().astype(np.float32)      # (N_in, 3)

            all_pixels.append(sel_pixels_np)
            all_points.append(world_pts_np)
            all_norms.append(world_n_np)

            # Free GPU memory for next batch
            del (
                tri_uv_b,
                tri_verts_b,
                tri_norms_b,
                grid_b,
                bary,
                inside,
                chosen_verts,
                chosen_norms,
                chosen_bary,
                world_pts,
                world_n,
            )
            torch.cuda.empty_cache()

        if not all_pixels:
            raise RuntimeError("No pixels found inside any triangle for the middle layer!")

        # Concatenate everything
        self.pixel_coords    = np.concatenate(all_pixels, axis=0)   # (N_total, 2) = (col, row)
        self.surface_points  = np.concatenate(all_points,  axis=0)   # (N_total, 3)
        self.surface_normals = np.concatenate(all_norms,   axis=0)   # (N_total, 3)

        logger.info(f"Middle‐layer PPM: found {self.pixel_coords.shape[0]} pixels inside mesh.")

    def _compute_bounds_and_initialize_zarr(self):
        """
        Optimized two‐pass version:

        Pass 1: for each layer k, compute
                 coords_k = surface_points + normals * offset_k
               then find layer_min, layer_max, and update global vol_min/vol_max.

        Pass 2: knowing vol_min and vol_max, allocate Zarr, then
                for each layer k, recompute coords_k, quantize to ints,
                subtract vol_min, and fill self.voxel_coords[k].
        """
        pts_mid   = self.surface_points    # (N, 3) float32
        norms_mid = self.surface_normals   # (N, 3) float32
        N = pts_mid.shape[0]

        # Initialize global min/max to large/small values
        # We'll store as float64 to be safe, then floor/ceil later.
        global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        # Pass 1: determine global bounds
        for k in range(self.num_layers):
            offset = (k - self.middle_idx) * self.step_size
            # coords_k: (N,3)
            coords_k = pts_mid + norms_mid * offset  # broadcast (N,3)

            layer_min = coords_k.min(axis=0)  # shape (3,)
            layer_max = coords_k.max(axis=0)  # shape (3,)

            # Update global min/max
            global_min = np.minimum(global_min, layer_min)
            global_max = np.maximum(global_max, layer_max)

        # Floor/ceil and add padding (–1/+1) exactly as before
        vol_min = np.floor(global_min).astype(np.int32) - 1  # shape (3,)
        vol_max = np.ceil(global_max).astype(np.int32) + 1   # shape (3,)
        dims = (vol_max - vol_min + 1).astype(np.int32)      # shape (3,)

        self.vol_min = vol_min
        self.vol_max = vol_max
        self.vol_shape = tuple(dims.tolist())  # (X_dim, Y_dim, Z_dim)

        logger.info(f"Computed volume bounds: min={self.vol_min}, max={self.vol_max}, shape={self.vol_shape}")

        # Allocate Zarr array
        os.makedirs(self.output_zarr, exist_ok=True)
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

        # zarr shape = (Z, Y, X)
        zarr_shape = (dims[2], dims[1], dims[0])
        self.z = zarr.open_array(
            os.path.join(self.output_zarr, "volume.zarr"),
            mode="w",
            shape=zarr_shape,
            dtype=self.dtype,
            chunks=(
                min(64, zarr_shape[0]),
                min(64, zarr_shape[1]),
                min(64, zarr_shape[2]),
            ),
            compressor=compressor,
        )

        # Prepare array for voxel coordinates
        self.voxel_coords = np.zeros((self.num_layers, N, 3), dtype=np.int32)

        # Pass 2: compute and store voxel indices per layer
        for k in range(self.num_layers):
            offset = (k - self.middle_idx) * self.step_size
            coords_k = pts_mid + norms_mid * offset  # (N,3)
            # Round to nearest integer
            rounded = np.floor(coords_k + 0.5).astype(np.int32)  # (N,3)
            idx = rounded - vol_min  # shift into [0..dims-1]
            self.voxel_coords[k] = idx

        # Save vol_min/vol_max for later use
        self.vol_min = vol_min
        self.vol_max = vol_max


    def run(self):
        """
        For each layer k:
          - Read “XX.png” into a 2D grayscale
          - Extract pixel values at self.pixel_coords (row, col)
          - Write those values into self.z at self.voxel_coords[k]
        """
        # pixel_coords was stored as (col, row). For indexing, convert to (row, col).
        pc = self.pixel_coords[:, [1, 0]]  # (N, 2)

        for k, layer_path in enumerate(tqdm(self.layer_paths, desc="Writing layers to volume")):
            img = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image: {layer_path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Extract the pixel values at (row, col)
            vals = img[pc[:, 0], pc[:, 1]]  # (N,)

            vox = self.voxel_coords[k]  # (N, 3)
            x_idx = vox[:, 0]
            y_idx = vox[:, 1]
            z_idx = vox[:, 2]
            # zarr is indexed as (Z, Y, X)
            self.z[z_idx, y_idx, x_idx] = vals

        logger.info("Finished writing all layers to the zarr volume.")


#################################################################
# Main entry point
#################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Inverse PPM: from a stack of PNG layers to a 3D zarr volume via a UV‐mapped mesh."
    )
    parser.add_argument(
        "--obj", type=str, required=True,
        help="Path to the OBJ mesh file (must contain UV coordinates)."
    )
    parser.add_argument(
        "--images", type=str, required=True,
        help="Folder containing ordered PNG layers (e.g. '00.png', '01.png', …)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output folder in which to create 'volume.zarr'."
    )
    parser.add_argument(
        "--step", type=float, default=1.0,
        help="Distance (in world units) between consecutive layers along mesh normal."
    )
    parser.add_argument(
        "--max_side_triangle", type=int, default=10,
        help="Maximum allowed triangle side in UV‐space (pixels) before splitting."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["uint8", "uint16"], default="uint8",
        help="dtype for the zarr volume (e.g. 'uint8' or 'uint16')."
    )
    parser.add_argument(
        "--batch_triangles", type=int, default=4000,
        help="Number of triangles per PPM batch. Smaller → less RAM, more time; larger → faster if you have more memory."
    )
    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    # Read "00.png" (first layer) to extract (cols, rows)
    example = os.path.join(args.images, "00.png")
    if not os.path.isfile(example):
        raise FileNotFoundError(f"Expected '00.png' not found in {args.images}")
    with Image.open(example) as ex:
        ex_cols, ex_rows = ex.size  # PIL gives (width, height)

    logger.info(f"Extracted image dimensions from '00.png': rows={ex_rows}, cols={ex_cols}")

    # 1) Load the mesh, splitting large UV triangles
    mesh_loader = MeshLoader(
        obj_path=args.obj,
        image_rows=ex_rows,
        image_cols=ex_cols,
        max_side_triangle=args.max_side_triangle
    )

    # 2) Build the InverseMapper (compute middle‐layer PPM + allocate zarr)
    mapper = InverseMapper(
        mesh_loader=mesh_loader,
        image_folder=args.images,
        output_zarr=args.output,
        step_size=args.step,
        dtype=args.dtype,
        batch_triangles=args.batch_triangles
    )

    # 3) Run the layer‐to‐volume pass
    mapper.run()


if __name__ == "__main__":
    main()
