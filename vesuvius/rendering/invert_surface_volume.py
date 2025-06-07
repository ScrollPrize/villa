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
        uv_scaled = (self.uv * scale).astype(np.float32)       # still (T, 3, 2)
        # flip V (so that v=0 becomes row=(rows-1), i.e. bottom of the image)
        uv_scaled[..., 1] = (self.image_rows - 1) - uv_scaled[..., 1]
        self.uv_pixels = uv_scaled

        logger.info(f"Loaded mesh: {self.triangle_vertices.shape[0]} triangles.")
        logger.info(f"Scaled UV to pixel coords using (cols={self.image_cols}, rows={self.image_rows}).")

    def _adjust_triangle_sizes(self):
        """
        If a triangle’s UV‐bbox (ceil(max_uv) − floor(min_uv)) exceeds max_side_triangle
        in either dimension, split it along the longer axis until no triangle is bigger
        than max_side_triangle×max_side_triangle.
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
# PPM class
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
        B, N, _ = grid.shape

        # Extract per‐triangle edges in UV
        v0 = tri_uv[:, 2, :] - tri_uv[:, 0, :]  # (B, 2)
        v1 = tri_uv[:, 1, :] - tri_uv[:, 0, :]  # (B, 2)

        v2 = grid - tri_uv[:, 0].unsqueeze(1)    # (B, N, 2)

        dot00 = (v0 * v0).sum(dim=1)   # (B,)
        dot01 = (v0 * v1).sum(dim=1)   # (B,)
        dot11 = (v1 * v1).sum(dim=1)   # (B,)

        dot02 = (v2 * v0.unsqueeze(1)).sum(dim=2)  # (B, N)
        dot12 = (v2 * v1.unsqueeze(1)).sum(dim=2)  # (B, N)

        invDen = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-20)  # (B,)
        invDen = invDen.unsqueeze(1)  # (B, 1)

        u = (dot11.unsqueeze(1) * dot02 - dot01.unsqueeze(1) * dot12) * invDen  # (B, N)
        v = (dot00.unsqueeze(1) * dot12 - dot01.unsqueeze(1) * dot02) * invDen  # (B, N)
        w = 1.0 - u - v                                                           # (B, N)

        inside = (u >= -self.epsilon) & (v >= -self.epsilon) & ((u + v) <= 1 + self.epsilon)  # (B, N)

        bary = torch.stack([u, v, w], dim=2)  # (B, N, 3)
        bary = normalize(bary.clamp(min=0.0), p=1, dim=2)

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
# InverseMapperFull: builds two datasets 'labels' and 'mask'
#################################################################
class InverseMapperFull:
    def __init__(
        self,
        mesh_loader: MeshLoader,
        image_folder: str,
        output_zarr: str,
        dims: tuple,
        mask_2d: np.ndarray,
        step_size: float = 1.0,
        dtype: str = "uint8",
        batch_triangles: int = 200,
    ):
        """
        mesh_loader:       pre‐loaded MeshLoader (with uv_pixels, triangle_vertices, triangle_normals)
        image_folder:      folder containing sorted PNG layers (“00.png”, “01.png”, …)
        output_zarr:       folder path in which to create a Zarr group containing:
                             - 'labels'  (Z_dim,Y_dim,X_dim), dtype=uint8
                             - 'mask'    (Z_dim,Y_dim,X_dim), dtype=uint8
        dims:              (Z_dim, Y_dim, X_dim) of the FULL scroll volume
        mask_2d:           a (rows, cols) NumPy array loaded from `<basename>_mask.png`
                           → nonzero = "keep", zero = ignore
        step_size:         distance (in world units) between consecutive layers
        dtype:             'uint8' (we assume labels in [0..255])
        batch_triangles:   how many triangles to process in one PPM batch
        """
        self.mesh = mesh_loader
        self.image_folder = image_folder
        self.output_zarr = output_zarr
        self.step_size = step_size
        self.dtype = dtype
        self.batch_triangles = batch_triangles

        # 1) Read all PNG layers
        pattern = os.path.join(self.image_folder, "*.png")
        self.layer_paths = sorted(glob.glob(pattern))
        if not self.layer_paths:
            raise FileNotFoundError(f"No PNG files found in {self.image_folder}")
        self.num_layers = len(self.layer_paths)
        logger.info(f"Found {self.num_layers} image layers.")

        # 2) Compute middle‐layer index
        self.middle_idx = self.num_layers // 2

        # 3) Device selection (for PPM)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            logger.info("Using CUDA for PPM.")
        else:
            logger.info("CUDA not available; using CPU for PPM.")

        # 4) Build PPM for the middle layer
        self._compute_middle_layer_ppm()

        # 5) Initialize the full‐volume Zarr group
        self.Z_dim, self.Y_dim, self.X_dim = dims
        self._initialize_zarr_group()

        # 6) Store the 2D mask array (rows × cols)
        self.mask_2d = mask_2d
        # Validate dimensions
        if self.mask_2d.ndim != 2:
            raise ValueError("mask_2d must be a single‐channel 2D array.")
        mask_rows, mask_cols = self.mask_2d.shape
        # Triangles’ UV came from an image of size (image_rows × image_cols)
        if (mask_rows != self.mesh.image_rows) or (mask_cols != self.mesh.image_cols):
            raise ValueError(
                f"Mask shape {(mask_rows,mask_cols)} does not match image dims "
                f"({self.mesh.image_rows},{self.mesh.image_cols})"
            )

    def _compute_middle_layer_ppm(self):
        """
        Build the Per‐Pixel Map *only* for the middle layer’s projection.
        After this, we obtain:
          - self.pixel_coords  :  (N_total, 2)  = (col, row) of each “hit‐pixel” on the image
          - self.surface_points:  (N_total, 3)  world‐coordinates of each hit on the middle slice
          - self.surface_normals: (N_total, 3)  the interpolated normal at each hit
        """
        logger.info("Computing PPM for middle layer …")
        tri_uv_np    = self.mesh.uv_pixels             # (T, 3, 2) numpy
        tri_verts_np = self.mesh.triangle_vertices     # (T, 3, 3) numpy
        tri_norms_np = self.mesh.triangle_normals      # (T, 3, 3) numpy

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
            min_uv_b = torch.floor(tri_uv_b.min(dim=1)[0])  # (B, 2)

            # 3) form a local (w×h) grid around each min_uv_b → (B, w*h, 2)
            grid_b = ppm.make_grid(min_uv_b, w, h)  # (B, w*h, 2)

            # 4) run barycentric test → bary (B,w*h,3), inside (B,w*h)
            bary, inside = ppm.point_in_triangle(grid_b, tri_uv_b)  # (B,w*h,3), (B,w*h)

            # 5) find all (triangle_idx, pixel_idx) where inside=True
            tri_idx, pix_idx = torch.nonzero(inside, as_tuple=True)  # each is (N_in,)

            if tri_idx.numel() == 0:
                # No pixels in this batch → skip
                del tri_uv_b, tri_verts_b, tri_norms_b, grid_b, bary, inside
                torch.cuda.empty_cache()
                continue

            # 6) gather pixel coordinates in (col, row)
            sel_pixels = grid_b[tri_idx, pix_idx]  # (N_in, 2) on device

            # 7) gather corresponding 3D vertices & normals
            chosen_verts = tri_verts_b[tri_idx]  # (N_in, 3, 3)
            chosen_norms = tri_norms_b[tri_idx]  # (N_in, 3, 3)
            chosen_bary  = bary[tri_idx, pix_idx, :]  # (N_in, 3)

            # 8) compute world coordinates and normals
            world_pts = torch.einsum("ij,ijk->ik", chosen_bary, chosen_verts)  # (N_in, 3)
            world_n   = torch.einsum("ij,ijk->ik", chosen_bary, chosen_norms)  # (N_in, 3)
            world_n   = normalize(world_n, p=2, dim=1)

            # 9) move back to CPU & numpy
            sel_pixels_np = sel_pixels.cpu().numpy().astype(np.int32)   # (N_in, 2)
            world_pts_np  = world_pts.cpu().numpy().astype(np.float32)  # (N_in, 3)
            world_ns_np   = world_n.cpu().numpy().astype(np.float32)    # (N_in, 3)

            all_pixels.append(sel_pixels_np)
            all_points.append(world_pts_np)
            all_norms.append(world_ns_np)

            # Free GPU memory
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

    def _initialize_zarr_group(self):
        """
        We assume origin=(0,0,0)
        and full scroll dims = (Z_dim, Y_dim, X_dim).  We create a Zarr group
        at self.output_zarr with two datasets:
            - 'labels' shape=(Z_dim, Y_dim, X_dim), dtype=uint8, fill_value=0
            - 'mask'   shape=(Z_dim, Y_dim, X_dim), dtype=uint8, fill_value=0
        """
        os.makedirs(self.output_zarr, exist_ok=True)
        grp = zarr.open_group(self.output_zarr, mode="a")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

        if "labels" not in grp:
            grp.create_dataset(
                "labels",
                shape=(self.Z_dim, self.Y_dim, self.X_dim),
                dtype=self.dtype,
                chunks=(128,128,128),
                compressor=compressor,
                fill_value=0
            )
        if "mask" not in grp:
            grp.create_dataset(
                "mask",
                shape=(self.Z_dim, self.Y_dim, self.X_dim),
                dtype="uint8",
                chunks=(128,128,128),
                compressor=compressor,
                fill_value=0
            )

        # Keep references for later writes
        self.z_labels = grp["labels"]
        self.z_mask   = grp["mask"]

        logger.info(f"Created Zarr group at '{self.output_zarr}' with datasets 'labels' and 'mask'.")

    def run(self):
        """
        Two‐pass, z‐bucket approach with “catch‐and‐grow” and parallelized chunk updates.

        PASS #1: Same as before—bucket hits by cz_idx, resizing on overflow.

        PASS #2: Uses a ThreadPoolExecutor to process each cz_idx in parallel.
        """
        import concurrent.futures
        # (No threading.Lock needed anymore.)

        # ───────────────────────────────────────────────────────────────────────
        # 1) Precompute middle‐layer geometry & 2D pixel coords
        # ───────────────────────────────────────────────────────────────────────
        pc        = self.pixel_coords[:, [1, 0]]  # (N_pts, 2) → (row, col)
        pts_mid   = self.surface_points           # (N_pts, 3)
        norms_mid = self.surface_normals          # (N_pts, 3)
        N_pts     = pts_mid.shape[0]
        L         = self.num_layers

        # ───────────────────────────────────────────────────────────────────────
        # 2) Determine chunk layout and initial bucket capacity
        # ───────────────────────────────────────────────────────────────────────
        cz, cy, cx = self.z_labels.chunks  # e.g. (128, 128, 128)
        n_chunks_z = (self.Z_dim + cz - 1) // cz
        n_chunks_y = (self.Y_dim + cy - 1) // cy
        n_chunks_x = (self.X_dim + cx - 1) // cx

        # Estimate total hits M_est = N_pts * L
        M_est = int(N_pts * L)
        # Initial per‐bucket capacity (10% headroom)
        base_capacity = max(int(np.ceil((M_est / max(n_chunks_z, 1)) * 1.1)), 1)

        # Track capacity per cz
        per_bucket_capacity = [base_capacity] * n_chunks_z

        # ───────────────────────────────────────────────────────────────────────
        # 3) Allocate one memmap “bucket” per cz_idx using base_capacity
        # ───────────────────────────────────────────────────────────────────────
        bucket_dtype = np.dtype([
            ("cy", np.int32),
            ("cx", np.int32),
            ("lz", np.int32),
            ("ly", np.int32),
            ("lx", np.int32),
            ("val", np.uint8),
        ])

        per_bucket_dir = tempfile.mkdtemp(prefix="bucket_z_")
        os.makedirs(per_bucket_dir, exist_ok=True)

        per_bucket_paths  = []
        per_bucket_counts = np.zeros((n_chunks_z,), dtype=np.int64)

        for cz_i in range(n_chunks_z):
            path = os.path.join(per_bucket_dir, f"bucket_z_{cz_i:03d}.dat")
            mp = np.memmap(path, dtype=bucket_dtype, mode="w+", shape=(base_capacity,))
            del mp
            per_bucket_paths.append(path)

        # ───────────────────────────────────────────────────────────────────────
        # 4) PASS #1: For each layer k, compute hits and append to z‐bucket,
        #            resizing if necessary
        # ───────────────────────────────────────────────────────────────────────
        for k, layer_path in enumerate(tqdm(self.layer_paths, desc="PASS #1: Bucketing hits")):
            img = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image: {layer_path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            offset  = (k - self.middle_idx) * self.step_size
            coords_k = pts_mid + norms_mid * offset
            rounded = np.floor(coords_k + 0.5).astype(np.int32)
            x_idx   = rounded[:, 0]
            y_idx   = rounded[:, 1]
            z_idx   = rounded[:, 2]

            valid = (
                (x_idx >= 0) & (x_idx < self.X_dim) &
                (y_idx >= 0) & (y_idx < self.Y_dim) &
                (z_idx >= 0) & (z_idx < self.Z_dim)
            )
            if not np.any(valid):
                continue

            x_v = x_idx[valid]
            y_v = y_idx[valid]
            z_v = z_idx[valid]

            py = pc[valid, 0]
            px = pc[valid, 1]
            mask_vals = self.mask_2d[py, px]
            keep_mask = (mask_vals != 0)
            if not np.any(keep_mask):
                continue

            x_v = x_v[keep_mask]
            y_v = y_v[keep_mask]
            z_v = z_v[keep_mask]
            py  = py[keep_mask]
            px  = px[keep_mask]
            vals = img[py, px]

            cz_idx = z_v // cz
            cy_idx = y_v // cy
            cx_idx = x_v // cx

            lz = z_v - (cz_idx * cz)
            ly = y_v - (cy_idx * cy)
            lx = x_v - (cx_idx * cx)

            unique_cz, inverse_cz = np.unique(cz_idx, return_inverse=True)
            for ui, this_cz in enumerate(unique_cz):
                mask_u = (inverse_cz == ui)
                n_hits = np.count_nonzero(mask_u)
                start  = int(per_bucket_counts[this_cz])
                end    = start + n_hits

                old_capacity = per_bucket_capacity[this_cz]
                if end > old_capacity:
                    # Read existing
                    bucket_path = per_bucket_paths[this_cz]
                    old_map = np.memmap(bucket_path, dtype=bucket_dtype, mode="r", shape=(old_capacity,))
                    existing_count = start
                    tmp = np.zeros((existing_count,), dtype=bucket_dtype)
                    tmp[:] = old_map[:existing_count]
                    del old_map
                    os.remove(bucket_path)

                    # Double capacity until it fits
                    new_capacity = old_capacity * 2
                    while new_capacity < end:
                        new_capacity *= 2

                    mp_big = np.memmap(bucket_path, dtype=bucket_dtype, mode="w+", shape=(new_capacity,))
                    mp_big[:existing_count] = tmp
                    del tmp
                    del mp_big

                    per_bucket_capacity[this_cz] = new_capacity
                    old_capacity = new_capacity

                bucket_path = per_bucket_paths[this_cz]
                bp_map = np.memmap(bucket_path, dtype=bucket_dtype, mode="r+", shape=(old_capacity,))
                idxs = np.nonzero(mask_u)[0]
                bp_map["cy"][start:end]  = cy_idx[idxs]
                bp_map["cx"][start:end]  = cx_idx[idxs]
                bp_map["lz"][start:end]  = lz[idxs]
                bp_map["ly"][start:end]  = ly[idxs]
                bp_map["lx"][start:end]  = lx[idxs]
                bp_map["val"][start:end] = vals[idxs]
                per_bucket_counts[this_cz] = end
                del bp_map

        # ───────────────────────────────────────────────────────────────────────
        # 5) PASS #2: Parallelized chunk updates with ThreadPoolExecutor
        # ───────────────────────────────────────────────────────────────────────

        def process_zbucket(cz_i):
            count = int(per_bucket_counts[cz_i])
            if count == 0:
                try:
                    os.remove(per_bucket_paths[cz_i])
                except:
                    pass
                return

            z0 = cz_i * cz
            z1 = min(z0 + cz, self.Z_dim)

            bucket_path = per_bucket_paths[cz_i]
            cap = per_bucket_capacity[cz_i]
            bp_map = np.memmap(bucket_path, dtype=bucket_dtype, mode="r", shape=(cap,))
            hits_array = bp_map[:count]

            cy_vals = hits_array["cy"]
            cx_vals = hits_array["cx"]
            twoD = np.stack([cy_vals, cx_vals], axis=1)
            uniq_2d, inv_2d = np.unique(twoD, axis=0, return_inverse=True)

            for gi in range(uniq_2d.shape[0]):
                cy_i2, cx_i2 = int(uniq_2d[gi, 0]), int(uniq_2d[gi, 1])
                mask_g = (inv_2d == gi)
                if not np.any(mask_g):
                    continue

                y0 = cy_i2 * cy
                y1 = min(y0 + cy, self.Y_dim)
                x0 = cx_i2 * cx
                x1 = min(x0 + cx, self.X_dim)

                # Read the chunk once
                labels_chunk = self.z_labels[z0:z1, y0:y1, x0:x1][:]
                mask_chunk   = self.z_mask  [z0:z1, y0:y1, x0:x1][:]

                sel = np.nonzero(mask_g)[0]
                lz_vals = hits_array["lz"][sel]
                ly_vals = hits_array["ly"][sel]
                lx_vals = hits_array["lx"][sel]
                v_vals  = hits_array["val"][sel]

                existing = labels_chunk[lz_vals, ly_vals, lx_vals]
                labels_chunk[lz_vals, ly_vals, lx_vals] = np.maximum(existing, v_vals)
                mask_chunk  [lz_vals, ly_vals, lx_vals] = 1

                # Write back the updated chunk
                self.z_labels[z0:z1, y0:y1, x0:x1] = labels_chunk
                self.z_mask  [z0:z1, y0:y1, x0:x1] = mask_chunk

            del bp_map
            try:
                os.remove(bucket_path)
            except:
                pass

        # Only submit cz indices that actually have hits
        nonzero_cz = [i for i in range(n_chunks_z) if per_bucket_counts[i] > 0]

        max_workers = min(32, (os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_zbucket, cz_i) for cz_i in nonzero_cz]
            for f in concurrent.futures.as_completed(futures):
                # Propagate exceptions, if any
                _ = f.result()

        # ───────────────────────────────────────────────────────────────────────
        # 6) Cleanup bucket directory
        # ───────────────────────────────────────────────────────────────────────
        try:
            os.rmdir(per_bucket_dir)
        except:
            pass

        logger.info("Finished writing all chunks into 'labels' and 'mask'.")





#################################################################
# Main entry point
#################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Inverse PPM → two‐dataset Zarr (['labels','mask']) over a full‐scroll volume."
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
        help="Output folder in which to create a Zarr group with 'labels' and 'mask'."
    )
    parser.add_argument(
        "--dims", type=int, nargs=3, required=True,
        metavar=("Z_dim", "Y_dim", "X_dim"),
        help="Three ints: full‐scroll dimensions: Z_dim Y_dim X_dim (e.g. 14376 7888 8096)."
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
        "--batch_triangles", type=int, default=4000,
        help="Number of triangles per PPM batch. Smaller → less RAM, more time; larger → faster if you have more memory."
    )
    args = parser.parse_args()

    # 1) Derive and load the 2D mask from "<basename>_mask.png"
    obj_base, _ = os.path.splitext(args.obj)
    mask_path = obj_base + "_mask.png"
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Expected mask file not found: {mask_path}")

    # Load the mask as a single‐channel array
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        raise RuntimeError(f"Failed to read mask image: {mask_path}")
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_2d = mask_img.astype(np.uint8)

    # 2) Read "00.png" to extract (cols, rows), for MeshLoader’s UV scaling
    example = os.path.join(args.images, "00.png")
    if not os.path.isfile(example):
        raise FileNotFoundError(f"Expected '00.png' not found in {args.images}")
    with Image.open(example) as ex:
        ex_cols, ex_rows = ex.size  # PIL gives (width, height)

    logger.info(f"Extracted image dimensions from '00.png': rows={ex_rows}, cols={ex_cols}")
    logger.info(f"Full‐scroll dims (Z,Y,X) = {tuple(args.dims)}")

    # 3) Load + subdivide the mesh
    mesh_loader = MeshLoader(
        obj_path=args.obj,
        image_rows=ex_rows,
        image_cols=ex_cols,
        max_side_triangle=args.max_side_triangle
    )

    # 4) Build the InverseMapperFull (computes middle‐slice PPM + creates Zarr group),
    #    passing in the 2D mask array.
    mapper = InverseMapperFull(
        mesh_loader=mesh_loader,
        image_folder=args.images,
        output_zarr=args.output,
        dims=tuple(args.dims),
        mask_2d=mask_2d,                # ← newly added
        step_size=args.step,
        dtype="uint8",
        batch_triangles=args.batch_triangles
    )

    # 5) Run: fills in 'labels' and 'mask' (only where mask_2d != 0)
    mapper.run()


if __name__ == "__main__":
    main()
