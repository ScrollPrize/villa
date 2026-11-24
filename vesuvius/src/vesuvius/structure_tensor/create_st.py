#create_st.py
import argparse
import heapq
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from vc_reader.reader import ZarrWriter, create_group, open_zarr

from vesuvius.models.run.inference import Inferer
from vc_reader.torch import structure_tensor
from .vf_format import zarr_codec

def _ceildiv(a: int, b: int) -> int:
    """Ceiling division for positives."""
    return -(-a // b)

def _quantize_dir_u8(v: np.ndarray) -> np.ndarray:
    """Quantize directional component in [-1,1] to uint8 [0,255]."""
    v = np.clip(v, -1.0, 1.0)
    return np.rint((v * 0.5 + 0.5) * 255.0).astype(np.uint8)

def _quantize_unit_u8(a: np.ndarray) -> np.ndarray:
    """Quantize scalar in [0,1] to uint8 [0,255]."""
    a = np.clip(a, 0.0, 1.0)
    return np.rint(a * 255.0).astype(np.uint8)

def _dtype_unit_scale(np_dtype) -> float:
    """Return a sensible scale for dtype: integer→max, float/bool→1.0."""
    import numpy as _np
    if _np.issubdtype(np_dtype, _np.integer):
        return float(_np.iinfo(np_dtype).max)   # e.g., 255, 65535
    return 1.0  # float/bool: treat as already normalized

class StructureTensorInferer(Inferer, nn.Module):
    """
    Inherits all of Inferer's I/O, patching, zarr & scheduling machinery,
    but replaces the nnU-Net inference with a 6‐channel 3D structure tensor.
    """
    def __init__(self,
                 *args,
                 sigma: float = 1.0,
                 smooth_components: bool = False,
                 volume: int = None,  # Add volume attribute
                 step_size: float = 1.0,
                 **kwargs):
        # --- Initialize Module first so register_buffer exists ---
        nn.Module.__init__(self)

        self.step_size = step_size
        # --- Remove any incoming args the base Inferer doesn't know about ---
        # (guards against CLI flags/wrapper options like OME export)
        for _k in (
            'normalization_scheme',
            'ome_out',           # from wrappers
            'keep_eigen',        # if present
            'confidence_metric', # if present
            'ome_downsample',    # if present
            'ome_scale'          # if present
        ):
            kwargs.pop(_k, None)

        # --- Now initialize Inferer, forcing normalization_scheme='none' ---
        Inferer.__init__(self, *args, normalization_scheme='none', **kwargs)

        self.num_classes = 6
        self.do_tta = False
        self.sigma = sigma
        self.smooth_components = smooth_components
        self.volume = volume  # Initialize volume attribute

        # --- Auto-infer patch_size from the input Zarr's chunking if none given ---
        if self.patch_size is None:
            store = open_zarr(self.input)
            chunks = store.chunks  # e.g. (1, pZ, pY, pX) or (pZ, pY, pX)
            if len(chunks) == 4:
                # drop the channel‐chunk
                self.patch_size = tuple(chunks[1:])
            elif len(chunks) == 3:
                self.patch_size = tuple(chunks)
            else:
                raise ValueError(
                    f"Cannot infer patch_size from input chunks={chunks}; "
                    "please supply --patch-size Z,Y,X"
                )
            if self.verbose:
                print(f"Inferred patch_size {self.patch_size} from input Zarr chunking")

        # Pavel derivatives have radius four; each Gaussian uses int(3*sigma).
        self._pad = int(3 * self.sigma) if self.sigma > 0 else 0
        halo = 4 + self._pad * (2 if self.smooth_components else 1)
        self._total_pad = (halo, halo, halo)
        
    def _load_model(self):
        """
        No model to load—just ensure num_classes is set.
        """
        self.num_classes = 6
        # patch_size logic from base class still applies (it will fall back
        # to user-specified patch_size or the model default, but here model
        # default is never used).
        return None

    def _create_output_stores(self):
        """
        Override to create a zarr group hierarchy with structure_tensor as a subgroup.
        """
        if self.num_classes is None or self.patch_size is None:
            raise RuntimeError("Cannot create output stores: model/patch info missing.")
        if not self.patch_start_coords_list:
            raise RuntimeError("Cannot create output stores: patch coordinates not available.")

        # Get the original volume shape
        if hasattr(self.dataset, 'input_shape'):
            if len(self.dataset.input_shape) == 4:  # has channel dimension
                original_volume_shape = list(self.dataset.input_shape[1:])
            else:  # no channel dimension
                original_volume_shape = list(self.dataset.input_shape)
        else:
            raise RuntimeError("Cannot determine original volume shape from dataset")

        # Check if we're in multi-GPU mode by seeing if output_dir ends with .zarr
        # and num_parts > 1, which indicates we should open existing shared store
        if self.num_parts > 1 and self.output_dir.endswith('.zarr'):
            # Multi-GPU mode: open existing shared store
            main_store_path = self.output_dir
            print(f"Opening existing shared store at: {main_store_path}")
            
            # Access the structure_tensor array
            self.output_store = ZarrWriter.open(f"{main_store_path}/structure_tensor", skip_empty=True)
            
        else:
            # Single-GPU mode: create new store with group hierarchy
            # Ensure output_dir ends with .zarr
            if not self.output_dir.endswith('.zarr'):
                main_store_path = self.output_dir + '.zarr'
            else:
                main_store_path = self.output_dir
            
            # Shape is (6 channels, Z, Y, X) for the full volume
            output_shape = (self.num_classes, *original_volume_shape)
            
            # Use the same chunking as patch size for efficient writing
            output_chunks = (self.num_classes, *self.patch_size)
            
            print(f"Creating output store at: {main_store_path}")
            print(f"Full volume shape: {output_shape}")
            print(f"Chunk shape: {output_chunks}")
            
            # Create the root group
            root_store = create_group(main_store_path, overwrite=True)
            
            # Create the structure_tensor array within the group
            self.output_store = root_store.create_array(
                'structure_tensor',
                shape=output_shape,
                chunks=output_chunks,
                dtype=np.float32,
                skip_empty=True,
                **zarr_codec(self.compressor_name, self.compression_level),
            )
            
            # Store metadata in the root group
            try:
                root_store.attrs['patch_size'] = list(self.patch_size)
                root_store.attrs['overlap'] = self.overlap
                root_store.attrs['part_id'] = self.part_id
                root_store.attrs['num_parts'] = self.num_parts
                root_store.attrs['original_volume_shape'] = original_volume_shape
                root_store.attrs['sigma'] = self.sigma
                root_store.attrs['smooth_components'] = self.smooth_components
            except Exception as e:
                print(f"Warning: Failed to write custom attributes: {e}")
            
            # Store the main path for later reference
            self.main_store_path = main_store_path
        
        # Set coords_store_path to None since we're not creating it
        self.coords_store_path = None
        
        if self.verbose: 
            print(f"Created output store structure_tensor group in: {main_store_path}")
        
        return self.output_store
    
    def compute_structure_tensor(self, x: torch.Tensor, sigma=None):
        sigma_val = float(self.sigma if sigma is None else sigma)
        target_device = torch.device(self.device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")
        return structure_tensor(x.to(device=target_device, dtype=torch.float32), sigma=sigma_val,
                                rho=sigma_val if self.smooth_components else 0.0)

    def _run_inference(self):
        """
        Skip model loading entirely, just build dataset, stores, then process.
        """
        if self.verbose: print("Preparing dataset & output stores for structure‐tensor...")
        # load_model is a no‐op now
        self.model = self._load_model()
        # dataset + dataloader
        self._create_dataset_and_loader()
        # zarr stores for logits & coords
        self._create_output_stores()
        # compute & write structure tensor
        self._process_batches()

    def infer(self):
        """
        Override to return just the output path (not a tuple with coords path).
        """
        try:
            self._run_inference()
            # Return the main store path (root group)
            if self.num_parts > 1 and self.output_dir.endswith('.zarr'):
                # Multi-GPU mode: return the shared store path
                main_output_path = self.output_dir
            else:
                # Single-GPU mode: return the main store path
                if hasattr(self, 'main_store_path'):
                    main_output_path = self.main_store_path
                else:
                    # Fallback to ensure .zarr extension
                    if not self.output_dir.endswith('.zarr'):
                        main_output_path = self.output_dir + '.zarr'
                    else:
                        main_output_path = self.output_dir
            return main_output_path
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_batches(self):
        """
        Iterate over patches using DataLoader, compute J over a padded region,
        trim to the original patch, and write into the full-volume zarr.
        """
        total = self.num_total_patches
        store = self.output_store
        processed_count = 0

        # Open the raw input volume once so we can read arbitrary slabs
        input_src = open_zarr(self.input)
        # Full volume dims (Z, Y, X)
        if input_src.ndim == 4:
            _, vol_Z, vol_Y, vol_X = input_src.shape
        else:
            vol_Z, vol_Y, vol_X = input_src.shape
        # Amount of padding on each side (pz, py, px), computed in __init__
        pz, py, px = self._total_pad

        with tqdm(total=total, desc="Struct-Tensor") as pbar:
            for batch_data in self.dataloader:
                # Handle different batch data formats
                if isinstance(batch_data, dict):
                    data_batch    = batch_data['data']
                    pos_batch     = batch_data.get('pos', [])
                    indices_batch = batch_data.get('index', [])
                else:
                    data_batch    = batch_data
                    pos_batch     = []
                    indices_batch = []

                batch_size = data_batch.shape[0]

                for i in range(batch_size):
                    # Determine the unpadded patch start
                    if pos_batch and i < len(pos_batch):
                        z0, y0, x0 = pos_batch[i]
                    elif indices_batch and i < len(indices_batch):
                        idx = indices_batch[i]
                        z0, y0, x0 = self.patch_start_coords_list[idx]
                    else:
                        z0, y0, x0 = self.patch_start_coords_list[processed_count + i]

                    # Compute end coords of the original patch
                    z1 = z0 + self.patch_size[0]
                    y1 = y0 + self.patch_size[1]
                    x1 = x0 + self.patch_size[2]

                    # Expand by padding, clamped to volume bounds
                    za, zb = max(z0 - pz, 0), min(z1 + pz, vol_Z)
                    ya, yb = max(y0 - py, 0), min(y1 + py, vol_Y)
                    xa, xb = max(x0 - px, 0), min(x1 + px, vol_X)

                    # --- robust channel handling: ensure [1,1,Z,Y,X] for conv3d ---
                    if input_src.ndim == 3:
                        raw = input_src[za:zb, ya:yb, xa:xb].astype('float32')
                    else:
                        raw = input_src[0, za:zb, ya:yb, xa:xb].astype('float32')

                    x = torch.from_numpy(raw).to(self.device).unsqueeze(0).unsqueeze(0)  # [1,1,Z,Y,X]

                    # Normalize to [0,1] by dtype max when NOT using a fiber mask
                    if self.volume is None:
                        scale = _dtype_unit_scale(input_src.dtype)
                        # guard against weird dtypes; no-op for floats/bools (scale=1)
                        x = x.float() / max(scale, 1e-8)

                    # Apply fiber-volume mask if needed
                    if self.volume is not None:
                        x = (x == float(self.volume)).float()

                    # Compute over padded patch
                    with torch.no_grad():
                        Jp = self.compute_structure_tensor(x, sigma=self.sigma)
                        # Jp shape: [1, 6, Zp, Yp, Xp]

                    # Border-aware trim to exactly the original patch extents
                    tz0 = z0 - za; tz1 = tz0 + (z1 - z0)
                    ty0 = y0 - ya; ty1 = ty0 + (y1 - y0)
                    tx0 = x0 - xa; tx1 = tx0 + (x1 - x0)
                    J = Jp[:, :, tz0:tz1, ty0:ty1, tx0:tx1]

                    # Convert to numpy and drop the leading batch dim
                    out_np = J.cpu().numpy().astype(np.float32).squeeze(0)

                    # Sanity check
                    if out_np.shape != (self.num_classes, *self.patch_size):
                        raise RuntimeError(
                            f"Trimmed output has shape {out_np.shape}, "
                            f"expected {(self.num_classes, *self.patch_size)}"
                        )

                    # Write the central patch into the full-volume zarr
                    store[:, z0:z1, y0:y1, x0:x1] = out_np

                    pbar.update(1)

                processed_count += batch_size
                self.current_patch_write_index = processed_count

        torch.cuda.empty_cache()

        if self.verbose:
            print(f"Written {self.current_patch_write_index}/{total} patches.")
    


class ChunkDataset(Dataset):
    """Dataset of spatial chunk bounds for structure‐tensor eigen decomposition."""
    def __init__(self, input_path, chunks, device):
        self.input_path = input_path
        self.chunks = chunks
        self.device = device

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # read block
        z0, z1, y0, y1, x0, x1 = self.chunks[idx]
        src = open_zarr(self.input_path)
        block_np = src[:, z0:z1, y0:y1, x0:x1].astype('float32')
        block = torch.from_numpy(block_np).to(self.device)  # [6, dz, dy, dx]
        if not torch.isfinite(block).all():
            bad = (~torch.isfinite(block)).sum().item()
            raise RuntimeError(
                f"Non-finite values detected in chunk {(z0, z1, y0, y1, x0, x1)}: {bad} entries"
            )
        return idx, (z0, z1, y0, y1, x0, x1), block


_EIGH_BATCH_SIZE = 65536
_SQRT3 = 1.7320508075688772


def _is_gpu_solver_error(err: BaseException) -> bool:
    """Detect NVIDIA cuSOLVER and AMD ROCm hipSOLVER/rocSOLVER runtime failures."""
    msg = str(err).lower()
    return any(token in msg for token in ("cusolver", "hipsolver", "rocsolver", "rocm"))


# solve the eigenvalue problem and sanitize the output
def _eigh_and_sanitize(M: torch.Tensor):
    # 1) enforce symmetry (numerically more stable? M is already symmetrical)
    M = 0.5 * (M + M.transpose(-1, -2))

    # --- sanitize INPUT before eigh to avoid NaN/Inf failures ---
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        w, v = torch.linalg.eigh(M)
    except RuntimeError as err:
        # Work around sporadic CUDA/ROCm GPU solver failures by retrying on CPU
        if not _is_gpu_solver_error(err):
            raise
        M_cpu = M.cpu()
        w_cpu, v_cpu = torch.linalg.eigh(M_cpu)
        w = w_cpu.to(M.device)
        v = v_cpu.to(M.device)
    # sanitize once
    w = torch.nan_to_num(w.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return w, v


def _eigh_3x3_analytical_f64(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    x5: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve batched 3x3 symmetric eigensystems in float64 using Cardano + row cross-products.

    Inputs are finite 1D tensors of length K representing the upper triangle
    ``[Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]``. Near-degenerate or poorly conditioned
    voxels delegate to ``_eigh_and_sanitize`` (which includes CUDA/ROCm CPU fallback).
    Returns ``(eigvals_3xK, eigvecs_9xK)`` in float32 with ascending eigenvalues.
    """
    a = x0.to(torch.float64)
    d = x1.to(torch.float64)
    e = x2.to(torch.float64)
    b = x3.to(torch.float64)
    f = x4.to(torch.float64)
    c = x5.to(torch.float64)

    scale = torch.maximum(
        torch.maximum(a.abs(), b.abs()),
        torch.maximum(c.abs(), torch.maximum(d.abs(), torch.maximum(e.abs(), f.abs()))),
    ).clamp_min(1e-30)
    inv_s = 1.0 / scale
    a = a * inv_s
    d = d * inv_s
    e = e * inv_s
    b = b * inv_s
    f = f * inv_s
    c = c * inv_s

    p1 = d * d + e * e + f * f
    q = (a + b + c) * (1.0 / 3.0)
    aq = a - q
    bq = b - q
    cq = c - q
    p = torch.sqrt(((aq * aq + bq * bq + cq * cq) + 2.0 * p1) * (1.0 / 6.0))
    inv_p = torch.where(p > 0.0, 1.0 / torch.where(p > 0.0, p, torch.ones_like(p)), torch.zeros_like(p))

    b00 = aq * inv_p
    b11 = bq * inv_p
    b22 = cq * inv_p
    b01 = d * inv_p
    b02 = e * inv_p
    b12 = f * inv_p
    det_b = (
        b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02)
    )
    phi = torch.acos(torch.clamp(0.5 * det_b, -1.0, 1.0)) * (1.0 / 3.0)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sqrt(torch.clamp(1.0 - cos_phi * cos_phi, min=0.0))

    # Ascending eigenvalues: l0 <= l1 <= l2
    l2 = q + 2.0 * p * cos_phi
    l0 = q + p * (-cos_phi - _SQRT3 * sin_phi)
    l1 = 3.0 * q - l0 - l2

    def _null_vec(lam: torch.Tensor):
        a0 = a - lam
        b0 = b - lam
        c0 = c - lam
        u0x = d * f - e * b0
        u0y = e * d - a0 * f
        u0z = a0 * b0 - d * d
        n0 = u0x * u0x + u0y * u0y + u0z * u0z

        u1x = d * c0 - e * f
        u1y = e * e - a0 * c0
        u1z = a0 * f - d * e
        n1 = u1x * u1x + u1y * u1y + u1z * u1z

        u2x = b0 * c0 - f * f
        u2y = f * e - d * c0
        u2z = d * f - b0 * e
        n2 = u2x * u2x + u2y * u2y + u2z * u2z

        use1 = n1 > n0
        vx = torch.where(use1, u1x, u0x)
        vy = torch.where(use1, u1y, u0y)
        vz = torch.where(use1, u1z, u0z)
        nm = torch.where(use1, n1, n0)

        use2 = n2 > nm
        vx = torch.where(use2, u2x, vx)
        vy = torch.where(use2, u2y, vy)
        vz = torch.where(use2, u2z, vz)
        nm = torch.where(use2, n2, nm)
        return vx, vy, vz, nm

    v2x, v2y, v2z, n2 = _null_vec(l2)
    v0x, v0y, v0z, n0 = _null_vec(l0)

    degen = (p <= 1e-9) | ((l2 - l1) <= 1e-6) | ((l1 - l0) <= 1e-6) | (n2 <= 1e-22) | (n0 <= 1e-22)
    inv_n2 = torch.rsqrt(torch.where(degen, torch.ones_like(n2), n2))
    v2x = v2x * inv_n2
    v2y = v2y * inv_n2
    v2z = v2z * inv_n2

    inv_n0 = torch.rsqrt(torch.where(degen, torch.ones_like(n0), n0))
    v0x = v0x * inv_n0
    v0y = v0y * inv_n0
    v0z = v0z * inv_n0
    # One Gram-Schmidt orthogonalization step ensures machine-precision orthogonality.
    dot02 = v0x * v2x + v0y * v2y + v0z * v2z
    v0x = v0x - dot02 * v2x
    v0y = v0y - dot02 * v2y
    v0z = v0z - dot02 * v2z
    n0_gs = v0x * v0x + v0y * v0y + v0z * v0z
    degen = degen | (n0_gs <= 0.5)
    inv_n0_gs = torch.rsqrt(torch.where(degen, torch.ones_like(n0_gs), n0_gs))
    v0x = v0x * inv_n0_gs
    v0y = v0y * inv_n0_gs
    v0z = v0z * inv_n0_gs

    v1x = v2y * v0z - v2z * v0y
    v1y = v2z * v0x - v2x * v0z
    v1z = v2x * v0y - v2y * v0x

    w3 = torch.stack((l0 * scale, l1 * scale, l2 * scale), dim=0).float()
    v9 = torch.stack((v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z), dim=0).float()

    n_degen = int(degen.sum().item())
    if n_degen > 0:
        idx = torch.nonzero(degen, as_tuple=False).squeeze(1)
        Md = torch.empty((n_degen, 3, 3), dtype=torch.float64, device=x0.device)
        Md[:, 0, 0] = x0[idx]
        Md[:, 0, 1] = x1[idx]
        Md[:, 0, 2] = x2[idx]
        Md[:, 1, 0] = Md[:, 0, 1]
        Md[:, 1, 1] = x3[idx]
        Md[:, 1, 2] = x4[idx]
        Md[:, 2, 0] = Md[:, 0, 2]
        Md[:, 2, 1] = Md[:, 1, 2]
        Md[:, 2, 2] = x5[idx]
        wd, vd = _eigh_and_sanitize(Md)
        w3[:, idx] = wd.transpose(0, 1)
        v9[:, idx] = vd.permute(0, 2, 1).reshape(n_degen, 9).transpose(0, 1)

    w3 = torch.nan_to_num(w3, nan=0.0, posinf=0.0, neginf=0.0)
    v9 = torch.nan_to_num(v9, nan=0.0, posinf=0.0, neginf=0.0)
    return w3, v9


_COHERENCE_CELL_SIZE = 16


def _align_vector_field_coherent_(v: torch.Tensor, cell_size: int = _COHERENCE_CELL_SIZE) -> None:
    """In-place sign alignment of a (3, dz, dy, dx) unit vector field for local coherence.

    Pools local 3x3 outer-product scatter matrices onto a coarse grid (cell_size^3 voxels),
    orients the coarse principal axes via a maximum-coherence spanning tree seeded with a
    deterministic hemisphere sign, trilinearly upsamples the guide field one channel at a
    time (bounded workspace), and aligns each voxel's sign to the local guide. Zero voxels
    stay exact +0.0.
    """
    _, dz, dy, dx = v.shape
    gz = max(1, (dz + cell_size - 1) // cell_size)
    gy = max(1, (dy + cell_size - 1) // cell_size)
    gx = max(1, (dx + cell_size - 1) // cell_size)

    if gz * gy * gx == 1:
        vf = v.reshape(3, -1)
        C = torch.empty((3, 3), dtype=torch.float64)
        for i in range(3):
            for j in range(i, 3):
                val = float(torch.dot(vf[i], vf[j]).item())
                C[i, j] = val
                C[j, i] = val
        if float(C.abs().sum().item()) == 0.0:
            return
        _, evecs = torch.linalg.eigh(C)
        ref = evecs[:, 2]
        if float((ref[0] + 0.37139068 * ref[1] + 0.13130643 * ref[2]).item()) < 0.0:
            ref = -ref
        ref = ref.to(device=v.device, dtype=v.dtype)
        dot = ref[0] * v[0] + ref[1] * v[1] + ref[2] * v[2]
    else:
        pooled = torch.empty((6, gz, gy, gx), dtype=torch.float32, device=v.device)
        for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            prod = (v[i] * v[j]).unsqueeze(0).unsqueeze(0)
            pooled[k] = F.adaptive_avg_pool3d(prod, (gz, gy, gx))[0, 0]
        pf = pooled.reshape(6, -1).cpu().double()
        n_cells = pf.shape[1]
        M = torch.empty((n_cells, 3, 3), dtype=torch.float64)
        M[:, 0, 0] = pf[0]
        M[:, 0, 1] = pf[1]
        M[:, 0, 2] = pf[2]
        M[:, 1, 0] = pf[1]
        M[:, 1, 1] = pf[3]
        M[:, 1, 2] = pf[4]
        M[:, 2, 0] = pf[2]
        M[:, 2, 1] = pf[4]
        M[:, 2, 2] = pf[5]
        evals, evecs = torch.linalg.eigh(M)
        top = evecs[:, :, 2].numpy().reshape(gz, gy, gx, 3)
        strength = (evals[:, 2] - evals[:, 1]).numpy().reshape(gz, gy, gx)
        energy = evals.sum(dim=1).numpy().reshape(gz, gy, gx)
        top[energy <= 0.0] = 0.0

        visited = np.zeros((gz, gy, gx), dtype=bool)
        nbrs = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        while True:
            rem = np.where(~visited & (energy > 0.0), strength, -1.0)
            if float(rem.max()) < 0.0:
                break
            sz, sy, sx = np.unravel_index(int(np.argmax(rem)), (gz, gy, gx))
            r0 = top[sz, sy, sx]
            if float(r0[0] + 0.37139068 * r0[1] + 0.13130643 * r0[2]) < 0.0:
                top[sz, sy, sx] = -r0
            visited[sz, sy, sx] = True
            pq: list[tuple[float, int, int, int, int, int, int]] = []
            for dz0, dy0, dx0 in nbrs:
                nz, ny, nx = sz + dz0, sy + dy0, sx + dx0
                if 0 <= nz < gz and 0 <= ny < gy and 0 <= nx < gx and not visited[nz, ny, nx] and energy[nz, ny, nx] > 0.0:
                    w = abs(float(np.dot(top[sz, sy, sx], top[nz, ny, nx])))
                    heapq.heappush(pq, (-w, nz, ny, nx, sz, sy, sx))
            while pq:
                _, cz, cy, cx, pz, py, px = heapq.heappop(pq)
                if visited[cz, cy, cx]:
                    continue
                visited[cz, cy, cx] = True
                if float(np.dot(top[cz, cy, cx], top[pz, py, px])) < 0.0:
                    top[cz, cy, cx] = -top[cz, cy, cx]
                for dz0, dy0, dx0 in nbrs:
                    nz, ny, nx = cz + dz0, cy + dy0, cx + dx0
                    if 0 <= nz < gz and 0 <= ny < gy and 0 <= nx < gx and not visited[nz, ny, nx] and energy[nz, ny, nx] > 0.0:
                        w = abs(float(np.dot(top[cz, cy, cx], top[nz, ny, nx])))
                        heapq.heappush(pq, (-w, nz, ny, nx, cz, cy, cx))

        guide = torch.from_numpy(top.transpose(3, 0, 1, 2).copy()).to(device=v.device, dtype=v.dtype).unsqueeze(0)
        dot = torch.zeros((dz, dy, dx), dtype=v.dtype, device=v.device)
        for c in range(3):
            gc = F.interpolate(guide[:, c : c + 1], size=(dz, dy, dx), mode="trilinear", align_corners=False)[0, 0]
            dot.addcmul_(v[c], gc)

    tie = (dot == 0.0) & ((v[0] + 0.37139068 * v[1] + 0.13130643 * v[2]) < 0.0)
    flip = (dot < 0.0) | tie
    if bool(flip.any().item()):
        for c in range(3):
            v[c] = torch.where(flip, -v[c], v[c])


def _compute_eigenvectors_impl(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, dz, dy, dx = block.shape
    N = dz * dy * dx

    x = torch.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).reshape(6, N)
    nz_mask = (x[0] != 0) | (x[1] != 0) | (x[2] != 0) | (x[3] != 0) | (x[4] != 0) | (x[5] != 0)
    nz_idx = torch.nonzero(nz_mask, as_tuple=False).squeeze(1)
    K = nz_idx.numel()

    eigvals_flat = torch.zeros((3, N), dtype=torch.float32, device=block.device)
    eigvecs_flat = torch.zeros((9, N), dtype=torch.float32, device=block.device)
    if K == 0:
        return eigvals_flat.view(3, dz, dy, dx), eigvecs_flat.view(9, dz, dy, dx)

    for start in range(0, K, _EIGH_BATCH_SIZE):
        bi = nz_idx[start : start + _EIGH_BATCH_SIZE]
        wi, vi = _eigh_3x3_analytical_f64(
            x[0, bi], x[1, bi], x[2, bi], x[3, bi], x[4, bi], x[5, bi]
        )
        eigvals_flat[:, bi] = wi
        eigvecs_flat[:, bi] = vi

    eigvecs_3d = eigvecs_flat.view(9, dz, dy, dx)
    v0 = eigvecs_3d[0:3]
    v2 = eigvecs_3d[6:9]
    _align_vector_field_coherent_(v2)
    _align_vector_field_coherent_(v0)
    # Recompute v1 = v2 x v0 so all three eigenvector fields are locally coherent and right-handed.
    eigvecs_3d[3] = v2[1] * v0[2] - v2[2] * v0[1]
    eigvecs_3d[4] = v2[2] * v0[0] - v2[0] * v0[2]
    eigvecs_3d[5] = v2[0] * v0[1] - v2[1] * v0[0]

    return eigvals_flat.view(3, dz, dy, dx), eigvecs_3d


def _compute_eigenvectors(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_eigenvectors_impl(block)


def _orient_eigenvectors_right_handed(eigvecs: torch.Tensor) -> torch.Tensor:
    """Enforce right-handedness (det >= 0) and align mean v0[0] toward +X on-device."""
    v = eigvecs.reshape(3, 3, *eigvecs.shape[1:]).clone()
    v0, v1, v2 = v[0], v[1], v[2]
    det = (
        v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
        - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
        + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
    )
    mask = det < 0
    if mask.any():
        v[2] = torch.where(mask.unsqueeze(0), -v[2], v[2])
    s = torch.sign(v[0, 0].mean())
    if s == 0:
        s = torch.tensor(1.0, dtype=v.dtype, device=v.device)
    v[0] *= s
    v[2] *= s
    return v

def _finalize_structure_tensor_torch(
    zarr_path,
    chunk_size,
    num_workers,
    codec,
    verbose,
    swap_eigenvectors: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    *,
    # NEW OME-style output options
    ome_out: bool = True,
    ome_downsample: int = 1,
    ome_scale: str = "0",
    confidence_metric: str = "fa",
    keep_eigen: bool = False,
):
    """
    Compute eigenvectors/eigenvalues from structure tensor. By default, write
    OME-style outputs:
      - first_component/{z,y,x}/<scale>  (uint8 in [-1,1] → [0,255])
      - second_component/{z,y,x}/<scale>
      - normal/{z,y,x}/<scale>
      - confidence/<scale>               (uint8; FA in [0,1] → [0,255])
    Optionally keep raw eigenvectors/eigenvalues when keep_eigen=True.
    Args:
        zarr_path: Path to the zarr file containing structure_tensor group
        chunk_size: Chunk size for processing
        num_workers: Number of workers for data loading
        codec: ``ZarrWriter.create`` codec arguments (see ``zarr_codec``)
        verbose: Enable verbose output
        swap_eigenvectors: Whether to swap eigenvectors 0 and 1
        device: torch device to use ('cpu', 'cuda:0', etc.)
    """
    # Open the root group
    root_store = create_group(zarr_path)
    
    # Access the structure tensor array
    src = open_zarr(f"{zarr_path}/structure_tensor")
    C, Z, Y, X = src.shape
    assert C == 6, f"Expect 6 channels, got {C}"

    # chunk dims
    if chunk_size is None:
        # src.chunks == (6, cz, cy, cx) 
        cz, cy, cx = src.chunks[1:]
    else:
        cz, cy, cx = chunk_size
    if verbose:
        print(f"[Eigen] using chunks (dz,dy,dx)=({cz},{cy},{cx})")

    # ---- OME-style outputs (downsampled, uint8) ----
    ds_factor = max(1, int(ome_downsample))
    Zds, Yds, Xds = _ceildiv(Z, ds_factor), _ceildiv(Y, ds_factor), _ceildiv(X, ds_factor)
    out_chunks_ds = (max(1, cz // ds_factor), max(1, cy // ds_factor), max(1, cx // ds_factor))

    def _ensure_vec_target(root, name: str):
        g = root.create_group(name)
        # (Re)create datasets per axis at this scale
        return tuple(_create_scale(g.create_group(axis)) for axis in ("z", "y", "x"))

    def _create_scale(group):
        return group.create_array(
            ome_scale, shape=(Zds, Yds, Xds), chunks=out_chunks_ds, dtype=np.uint8,
            overwrite=True, skip_empty=True, **codec
        )

    if ome_out:
        fz, fy, fx = _ensure_vec_target(root_store, "first_component")
        sz_, sy_, sx_ = _ensure_vec_target(root_store, "second_component")
        nz, ny, nx = _ensure_vec_target(root_store, "normal")
        # confidence dataset
        conf_ds = _create_scale(root_store.create_group("confidence"))

    # ---- Optional: keep full-precision eigen* arrays (float32) ----
    if keep_eigen:
        out_chunks = (1, cz, cy, cx)
        eigenvectors_arr = root_store.create_array(
            'eigenvectors',
            shape=(9, Z, Y, X),
            chunks=out_chunks,
            dtype=np.float32,
            overwrite=True,
            skip_empty=True,
            **codec,
        )
        eigenvalues_arr = root_store.create_array(
            'eigenvalues',
            shape=(3, Z, Y, X),
            chunks=out_chunks,
            dtype=np.float32,
            overwrite=True,
            skip_empty=True,
            **codec,
        )
    
    # build chunk list
    def gen_bounds():
        for z0 in range(0, Z, cz):
            for y0 in range(0, Y, cy):
                for x0 in range(0, X, cx):
                    yield (z0, min(z0+cz,Z),
                           y0, min(y0+cy,Y),
                           x0, min(x0+cx,X))
    chunks = list(gen_bounds())
    if verbose:
        print(f"[Eigen] {len(chunks)} chunks to solve the eigenvalue problem on")

    # Dataset & DataLoader
    # Resolve device (backward compatible default)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Modified ChunkDataset to work with zarr groups
    class GroupChunkDataset(Dataset):
        def __init__(self, structure_tensor_arr, chunks, device):
            self.src = structure_tensor_arr
            self.chunks = chunks
            self.device = device

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, idx):
            z0, z1, y0, y1, x0, x1 = self.chunks[idx]
            block_np = self.src[:, z0:z1, y0:y1, x0:x1].astype('float32')
            block = torch.from_numpy(block_np).to(self.device)
            if not torch.isfinite(block).all():
                bad = (~torch.isfinite(block)).sum().item()
                raise RuntimeError(
                    f"Non-finite values detected in chunk {(z0, z1, y0, y1, x0, x1)}: {bad} entries"
                )
            return idx, (z0, z1, y0, y1, x0, x1), block
    
    gds = GroupChunkDataset(src, chunks, device)
    loader = DataLoader(
        gds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda batch: batch[0]
    )

    # Process each chunk
    for idx, bounds, block in tqdm(loader, desc="[Eigen] Chunks"):
        z0, z1, y0, y1, x0, x1 = bounds

        with torch.no_grad():
            try:
                eigvals_block_gpu, eigvecs_block_gpu = _compute_eigenvectors(block)
            except RuntimeError as err:
                if block.device.type == "cuda" and _is_gpu_solver_error(err):
                    # Fall back to CPU eigen solve for this chunk and move results back to GPU
                    eigvals_cpu, eigvecs_cpu = _compute_eigenvectors_impl(block.cpu())
                    eigvals_block_gpu = eigvals_cpu.to(block.device)
                    eigvecs_block_gpu = eigvecs_cpu.to(block.device)
                else:
                    raise

            if swap_eigenvectors:
                v_swap = eigvecs_block_gpu.reshape(3, 3, *eigvecs_block_gpu.shape[1:])
                v_swap = v_swap[[1, 0, 2]]
                eigvecs_block_gpu = v_swap.reshape(9, *eigvecs_block_gpu.shape[1:])
                eigvals_block_gpu = eigvals_block_gpu[[1, 0, 2]]

            v_corrected = _orient_eigenvectors_right_handed(eigvecs_block_gpu)
            v_np = v_corrected.cpu().numpy()     # [3,3,dz,dy,dx]
            w_np = eigvals_block_gpu.cpu().numpy()  # [3,dz,dy,dx]

        del block, eigvals_block_gpu, eigvecs_block_gpu, v_corrected
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---------- Write OME-style outputs (downsampled + quantized) ----------
        if ome_out:
            # Compute aligned downsample slices per axis
            def _axis_ds(a0, a1, s):
                start = (s - (a0 % s)) % s
                sl = slice(start, a1 - a0, s)
                o0 = (a0 + start) // s
                o1 = (a1 - 1) // s + 1
                return sl, o0, o1
            slz, oz0, oz1 = _axis_ds(z0, z1, ds_factor)
            sly, oy0, oy1 = _axis_ds(y0, y1, ds_factor)
            slx, ox0, ox1 = _axis_ds(x0, x1, ds_factor)

            # FIRST (vector 0)
            fz[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[0, 0, slz, sly, slx])
            fy[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[0, 1, slz, sly, slx])
            fx[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[0, 2, slz, sly, slx])
            # SECOND (vector 1)
            sz_[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[1, 0, slz, sly, slx])
            sy_[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[1, 1, slz, sly, slx])
            sx_[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[1, 2, slz, sly, slx])
            # NORMAL (vector 2)
            nz[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[2, 0, slz, sly, slx])
            ny[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[2, 1, slz, sly, slx])
            nx[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_dir_u8(v_np[2, 2, slz, sly, slx])

            # Confidence (default: Fractional Anisotropy) — computed on the downsampled slice only
            l1 = w_np[0, slz, sly, slx]
            l2 = w_np[1, slz, sly, slx]
            l3 = w_np[2, slz, sly, slx]
            eps = 1e-12
            if confidence_metric == "linearity":
                conf = (l3 - l2) / (np.abs(l3) + eps)
            elif confidence_metric == "planarity":
                conf = (l2 - l1) / (np.abs(l3) + eps)
            elif confidence_metric == "max_lp":
                cl = (l3 - l2) / (np.abs(l3) + eps)
                cp = (l2 - l1) / (np.abs(l3) + eps)
                conf = np.maximum(cl, cp)
            else:  # "fa"
                lbar = (l1 + l2 + l3) / 3.0
                num = (l1 - lbar) ** 2 + (l2 - lbar) ** 2 + (l3 - lbar) ** 2
                den = (l1 ** 2 + l2 ** 2 + l3 ** 2) + eps
                conf = np.sqrt(1.5) * np.sqrt(num) / np.sqrt(den)
            conf_ds[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_unit_u8(conf)

        # ---------- Optional: keep raw eigen* ----------
        if keep_eigen:
            eigvecs_block_full = v_np.reshape(9, *v_np.shape[2:])
            eigenvectors_arr[:, z0:z1, y0:y1, x0:x1] = eigvecs_block_full
            eigenvalues_arr[:,  z0:z1, y0:y1, x0:x1] = w_np

    if verbose and ome_out:
        print(f"[OME] wrote first_component/ second_component/ normal/ and confidence/ at scale='{ome_scale}', downsample={ds_factor}")
    if verbose and keep_eigen:
        print(f"[Eigen] eigenvectors → {zarr_path}/eigenvectors")
        print(f"[Eigen] eigenvalues  → {zarr_path}/eigenvalues")


def main():
    parser = argparse.ArgumentParser(description='Compute 3D structure tensor or eigenvalues/eigenvectors')
    
    # Basic I/O arguments
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Path to store output results')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='structure-tensor',
                        choices=['structure-tensor', 'eigenanalysis'],
                        help='Mode of operation: compute structure tensor or perform eigenanalysis')
    
    # Structure tensor computation arguments
    parser.add_argument('--structure-tensor', action='store_true', dest='structure_tensor',
                        help='Compute 6-channel 3D structure tensor (sets mode to structure-tensor)')
    parser.add_argument('--structure-tensor-only', action='store_true',
                        help='Compute only the structure tensor, skip eigenanalysis')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian σ for structure-tensor smoothing')
    parser.add_argument('--smooth-components', action='store_true',
                        help='After computing Jxx…Jzz, apply a second Gaussian smoothing to each channel')
    parser.add_argument('--volume', type=int, default=None,
                        help='Volume ID for fiber-volume masking')
    
    # Patch processing arguments
    parser.add_argument('--patch_size', type=str, default=None, 
                        help='Override patch size, comma-separated (e.g., "192,192,192")')
    parser.add_argument('--overlap', type=float, default=0.0, 
                        help='Overlap between patches (0-1), default 0.0 for structure tensor')
    parser.add_argument('--step_size', type=float, default=None,
                        help='Step‐size factor for sliding window (0 < step_size ≤ 1). If unset, will be inferred as 1.0 − overlap.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num_parts', type=int, default=1, 
                        help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, 
                        help='Part ID to process (0-indexed)')
    
    # Eigenanalysis arguments
    parser.add_argument('--eigen-input', type=str, default=None,
                        help='Input path for eigenanalysis (6-channel structure tensor zarr)')
    parser.add_argument('--eigen-output', type=str, default=None,
                        help='Output path for eigenvectors')
    parser.add_argument('--chunk-size', type=str, default=None,
                        help='Chunk size for eigenanalysis, comma-separated (e.g., "64,64,64")')
    parser.add_argument('--swap-eigenvectors', action='store_true',
                        help='Swap eigenvectors 0 and 1')
    parser.add_argument('--delete-intermediate', action='store_true',
                        help='Delete intermediate structure tensor after eigenanalysis')
    
    # OME-style output options
    parser.add_argument('--ome-out', action='store_true', default=True,
                        help='Write OME-style outputs (first/second/normal + confidence).')
    parser.add_argument('--ome-downsample', type=int, default=1,
                        help='Pick-every-N downsample when writing OME outputs.')
    parser.add_argument('--ome-scale', type=str, default='0',
                        help='Name of the scale node to write under (e.g., "0").')
    parser.add_argument('--confidence-metric', type=str, default='fa',
                        choices=['fa','linearity','planarity','max_lp'],
                        help='Scalar confidence metric to export.')
    parser.add_argument('--keep-eigen', action='store_true', default=False,
                        help='Also keep raw eigenvectors/eigenvalues arrays.')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                        choices=['zstd', 'lz4', 'zlib', 'none'],
                        help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=3,
                        help='Compression level (1-9)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Handle mode logic
    if args.structure_tensor:
        args.mode = 'structure-tensor'
    
    # Validate required args conditionally by mode
    if args.mode == 'structure-tensor':
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir are required for structure-tensor mode")
            return 2

    # Parse patch size if provided
    patch_size = None
    if args.patch_size:
        try:
            patch_size = tuple(map(int, args.patch_size.split(',')))
            print(f"Using user-specified patch size: {patch_size}")
        except Exception as e:
            print(f"Error parsing patch_size: {e}")
            print("Using default patch size.")
    
    # Parse chunk size for eigenanalysis
    chunk_size = None
    if args.chunk_size:
        try:
            chunk_size = tuple(map(int, args.chunk_size.split(',')))
        except Exception as e:
            print(f"Error parsing chunk_size: {e}")
    
    codec = zarr_codec(args.zarr_compressor, args.zarr_compression_level)
    
    if args.mode == 'structure-tensor':
        # Run structure tensor computation
        print("\n--- Initializing Structure Tensor Inferer ---")
        # Decide on step_size: if unset, default to (1 – overlap)
        if args.step_size is None:
            inferred_step = 1.0 - args.overlap
        else:
            inferred_step = args.step_size

        inferer = StructureTensorInferer(
            model_path='dummy',  # Not used for structure tensor
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sigma=args.sigma,
            smooth_components=args.smooth_components,
            volume=args.volume,
            num_parts=args.num_parts,
            part_id=args.part_id,
            overlap=args.overlap,
            step_size=inferred_step,
            batch_size=args.batch_size,
            patch_size=patch_size,
            device=args.device,
            ome_out=args.ome_out,
            ome_downsample=args.ome_downsample,
            ome_scale=args.ome_scale,
            confidence_metric=args.confidence_metric,
            keep_eigen=args.keep_eigen,
            verbose=args.verbose,
            compressor_name=args.zarr_compressor,
            compression_level=args.zarr_compression_level,
            num_dataloader_workers=args.num_workers
        )
        
        try:
            print("\n--- Starting Structure Tensor Computation ---")
            result = inferer.infer()
            
            # Handle either single return value or tuple
            if isinstance(result, tuple):
                logits_path = result[0]
            else:
                logits_path = result
            
            if not logits_path:
                # infer() already printed the traceback; do not let the
                # runner report "Completed Successfully" on an empty store.
                print(f"\n--- Structure Tensor Computation Failed ---")
                return 1

            print(f"\n--- Structure Tensor Computation Finished ---")
            print(f"Structure tensor saved to: {logits_path}/structure_tensor")
                
            # Run eigenanalysis automatically unless --structure-tensor-only is specified
            if not args.structure_tensor_only:
                print("\n--- Running Eigenanalysis ---")
                _finalize_structure_tensor_torch(
                    zarr_path=logits_path,
                    chunk_size=chunk_size,
                    num_workers=args.num_workers,
                    codec=codec,
                    verbose=args.verbose,
                    swap_eigenvectors=args.swap_eigenvectors,
                    device=args.device
                )
                print("\n--- All computations completed successfully ---")
                print(f"Final output contains:")
                print(f"  - Structure tensor: {logits_path}/structure_tensor")
                if args.ome_out:
                    print(f"  - first_component/, second_component/, normal/, confidence/ (scale '{args.ome_scale}', ds={args.ome_downsample})")
                if args.keep_eigen:
                    print(f"  - Eigenvectors: {logits_path}/eigenvectors")
                    print(f"  - Eigenvalues: {logits_path}/eigenvalues")
                
        except Exception as e:
            print(f"\n--- Structure Tensor Computation Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    elif args.mode == 'eigenanalysis':
        # Run eigenanalysis only
        if not args.eigen_input:
            print("Error: --eigen-input must be provided for eigenanalysis mode")
            return 1
            
        print("\n--- Running Eigenanalysis ---")
        print(f"Input zarr: {args.eigen_input}")
        try:
            _finalize_structure_tensor_torch(
                zarr_path=args.eigen_input,
                chunk_size=chunk_size,
                num_workers=args.num_workers,
                codec=codec,
                verbose=args.verbose,
                swap_eigenvectors=args.swap_eigenvectors,
                device=args.device,
                ome_out=args.ome_out,
                ome_downsample=args.ome_downsample,
                ome_scale=args.ome_scale,
                confidence_metric=args.confidence_metric,
                keep_eigen=args.keep_eigen,
            )
            
            print("\n--- Eigenanalysis Completed Successfully ---")
            print(f"Results saved to:")
            if args.ome_out:
                print(f"  - first_component/, second_component/, normal/, confidence/ (scale '{args.ome_scale}', ds={args.ome_downsample})")
            if args.keep_eigen:
                print(f"  - Eigenvectors: {args.eigen_input}/eigenvectors")
                print(f"  - Eigenvalues: {args.eigen_input}/eigenvalues")
            
        except Exception as e:
            print(f"\n--- Eigenanalysis Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
