#create_st.py
import argparse
import sys
import warnings
import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Optional, Union
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import zarr
import numcodecs

from vesuvius.models.run.inference import Inferer
from vesuvius.data.utils import open_zarr
from vesuvius.image_proc.geometry.structure_tensor import (
    StructureTensorComputer,
    _get_gaussian_kernel_3d,
    _get_pavel_kernels_3d,
)
from numcodecs import Blosc
from ._compile_utils import _maybe_compile_function

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
            store = open_zarr(
                path=self.input,
                mode='r',
                storage_options={'anon': False}
                    if str(self.input).startswith('s3://') else None
            )
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

        self._st_computer = StructureTensorComputer(
            sigma=self.sigma,
            component_sigma=None,
            smooth_components=self.smooth_components,
            device=self.device,
        )

        dev = torch.device(self.device)
        dtype = torch.float32

        if self.sigma > 0:
            _, radius = _get_gaussian_kernel_3d(dev, dtype, self.sigma)
            self._pad = radius
        else:
            self._pad = 0

        kz, ky, kx = _get_pavel_kernels_3d(dev, dtype)
        pad_pz = kz.shape[2] // 2
        pad_py = ky.shape[3] // 2
        pad_px = kx.shape[4] // 2

        extra = self._pad if (self.sigma > 0 and self.smooth_components) else 0
        self._total_pad = (
            self._pad + pad_pz + extra,
            self._pad + pad_py + extra,
            self._pad + pad_px + extra,
        )
        
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
            
            # Open the root group
            root_store = zarr.open_group(
                main_store_path,
                mode='r+',
                storage_options={'anon': False} if main_store_path.startswith('s3://') else None
            )
            
            # Access the structure_tensor subgroup
            self.output_store = root_store['structure_tensor']
            
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
            
            compressor = self._get_zarr_compressor()
            
            print(f"Creating output store at: {main_store_path}")
            print(f"Full volume shape: {output_shape}")
            print(f"Chunk shape: {output_chunks}")
            
            # Create the root group
            root_store = zarr.open_group(
                main_store_path,
                mode='w',
                storage_options={'anon': False} if main_store_path.startswith('s3://') else None
            )
            
            # Create the structure_tensor array within the group
            self.output_store = root_store.create_dataset(
                'structure_tensor',
                shape=output_shape,
                chunks=output_chunks,
                dtype=np.float32,
                compressor=compressor,
                write_empty_chunks=False
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
        self._st_computer.device = target_device
        component_sigma = sigma_val if self.smooth_components else None
        return self._st_computer.compute(
            x,
            sigma=sigma_val,
            component_sigma=component_sigma,
            device=target_device,
            smooth_components=self.smooth_components,
            spatial_dims=3,
        )

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
        numcodecs.blosc.use_threads = True

        total = self.num_total_patches
        store = self.output_store
        processed_count = 0

        # Open the raw input volume once so we can read arbitrary slabs
        input_src = open_zarr(
            path=self.input,
            mode='r',
            storage_options={'anon': False} if str(self.input).startswith('s3://') else None
        )
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
        src = open_zarr(
            path=self.input_path, mode='r',
            storage_options={'anon': False} if self.input_path.startswith('s3://') else None
        )
        block_np = src[:, z0:z1, y0:y1, x0:x1].astype('float32')
        block = torch.from_numpy(block_np).to(self.device)  # [6, dz, dy, dx]
        if not torch.isfinite(block).all():
            bad = (~torch.isfinite(block)).sum().item()
            raise RuntimeError(
                f"Non-finite values detected in chunk {(z0, z1, y0, y1, x0, x1)}: {bad} entries"
            )
        return idx, (z0, z1, y0, y1, x0, x1), block


# solve the eigenvalue problem and sanitize the output
def _eigh_and_sanitize(M: torch.Tensor):
    # 1) enforce symmetry (numerically more stable? M is already symmetrical)
    M = 0.5 * (M + M.transpose(-1, -2))

    # --- sanitize INPUT before eigh to avoid NaN/Inf failures ---
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        w, v = torch.linalg.eigh(M)
    except RuntimeError as err:
        # Work around sporadic CUDA solver failures by retrying on CPU
        if "cusolver" not in str(err).lower():
            raise
        M_cpu = M.cpu()
        w_cpu, v_cpu = torch.linalg.eigh(M_cpu)
        w = w_cpu.to(M.device)
        v = v_cpu.to(M.device)
    # sanitize once
    w = torch.nan_to_num(w.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return w, v


def _compute_eigenvectors_impl(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, dz, dy, dx = block.shape
    N = dz * dy * dx

    block = torch.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    # build + sanitize
    x = block.view(6, N)
    M = torch.empty((N,3,3), dtype=torch.float64, device=block.device)
    M[:, 0, 0] = x[0]; M[:, 0, 1] = x[1]; M[:, 0, 2] = x[2]
    M[:, 1, 0] = x[1]; M[:, 1, 1] = x[3]; M[:, 1, 2] = x[4]
    M[:, 2, 0] = x[2]; M[:, 2, 1] = x[4]; M[:, 2, 2] = x[5]
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    zero_mask = M.abs().sum(dim=(1,2)) == 0

    batch_size = 1048576
    # eigen-decomp (either whole or in chunks)
    if batch_size is None or N <= batch_size:
        w, v = _eigh_and_sanitize(M)
    else:
        ws = []; vs = []
        for chunk in M.split(batch_size, dim=0):
            wi, vi = _eigh_and_sanitize(chunk)
            ws.append(wi); vs.append(vi)
        w = torch.cat(ws, 0)
        v = torch.cat(vs, 0)

    # zero out truly‐empty voxels without branching
    # zero_mask: [N], w: [N,3], v: [N,3,3]
    mask_w = zero_mask.unsqueeze(-1)             # [N,1]
    w = w.masked_fill(mask_w, 0.0)               # [N,3]
    mask_v = mask_w.unsqueeze(-1)                # [N,1,1]
    v = v.masked_fill(mask_v, 0.0)               # [N,3,3]

    # reshape back
    eigvals = w.transpose(0,1).view(3, dz, dy, dx)
    eigvecs = (
        v
        .permute(0,2,1)
        .reshape(N,9)
        .transpose(0,1)
        .view(9, dz, dy, dx)
    )
    return eigvals, eigvecs


_compute_eigenvectors_fn = _maybe_compile_function(
    _compute_eigenvectors_impl,
    compile_kwargs={"mode": "max-autotune-no-cudagraphs", "fullgraph": True},
)


def _compute_eigenvectors(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global _compute_eigenvectors_fn
    try:
        return _compute_eigenvectors_fn(block)
    except RuntimeError as exc:
        message = str(exc)
        if "Compiler: cl is not found" in message or "torch._inductor.exc" in message:
            warnings.warn(
                "Falling back to eager eigenvector computation because torch.compile "
                "failed (likely missing required compiler).",
                RuntimeWarning,
            )
            _compute_eigenvectors_fn = _compute_eigenvectors_impl
            return _compute_eigenvectors_fn(block)
        raise

def _finalize_structure_tensor_torch(
    zarr_path,
    chunk_size,
    num_workers,
    compressor,
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
        compressor: Zarr compressor to use
        verbose: Enable verbose output
        swap_eigenvectors: Whether to swap eigenvectors 0 and 1
        device: torch device to use ('cpu', 'cuda:0', etc.)
    """
    # Open the root group
    root_store = zarr.open_group(
        zarr_path,
        mode='r+',
        storage_options={'anon': False} if zarr_path.startswith('s3://') else None
    )
    
    # Access the structure tensor array
    src = root_store['structure_tensor']
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
        g = root.require_group(name)
        gz = g.require_group("z")
        gy = g.require_group("y")
        gx = g.require_group("x")
        # (Re)create datasets per axis at this scale
        for axg in (gz, gy, gx):
            if ome_scale in axg:
                del axg[ome_scale]
            axg.create_dataset(
                ome_scale, shape=(Zds, Yds, Xds), chunks=out_chunks_ds,
                dtype=np.uint8, compressor=compressor, write_empty_chunks=False
            )
        return gz[ome_scale], gy[ome_scale], gx[ome_scale]

    if ome_out:
        fz, fy, fx = _ensure_vec_target(root_store, "first_component")
        sz_, sy_, sx_ = _ensure_vec_target(root_store, "second_component")
        nz, ny, nx = _ensure_vec_target(root_store, "normal")
        # confidence dataset
        conf_group = root_store.require_group("confidence")
        if ome_scale in conf_group:
            del conf_group[ome_scale]
        conf_ds = conf_group.create_dataset(
            ome_scale, shape=(Zds, Yds, Xds), chunks=out_chunks_ds,
            dtype=np.uint8, compressor=compressor, write_empty_chunks=False
        )

    # ---- Optional: keep full-precision eigen* arrays (float32) ----
    if keep_eigen:
        out_chunks = (1, cz, cy, cx)
        eigenvectors_arr = root_store.create_dataset(
            'eigenvectors',
            shape=(9, Z, Y, X),
            chunks=out_chunks,
            compressor=compressor,
            dtype=np.float32,
            write_empty_chunks=False,
            overwrite=True
        )
        eigenvalues_arr = root_store.create_dataset(
            'eigenvalues',
            shape=(3, Z, Y, X),
            chunks=out_chunks,
            compressor=compressor,
            dtype=np.float32,
            write_empty_chunks=False,
            overwrite=True
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
        
        if not torch.isfinite(block).all():
            finite_mask = torch.isfinite(block)
            bad = (~finite_mask).sum().item()
            finite_vals = block[finite_mask]
            finite_min = finite_vals.min().item() if finite_vals.numel() else float('nan')
            finite_max = finite_vals.max().item() if finite_vals.numel() else float('nan')
            raise RuntimeError(
                f"Non-finite values before eigen solve for chunk {bounds}: {bad} bad entries, "
                f"finite range [{finite_min}, {finite_max}]"
            )

        with torch.no_grad():
            try:
                eigvals_block_gpu, eigvecs_block_gpu = _compute_eigenvectors(block)
            except RuntimeError as err:
                err_msg = str(err).lower()
                if (
                    block.device.type == "cuda"
                    and "cusolver" in err_msg
                ):
                    # Fall back to CPU eigen solve for this chunk and move results back to GPU
                    eigvals_cpu, eigvecs_cpu = _compute_eigenvectors_impl(block.cpu())
                    eigvals_block_gpu = eigvals_cpu.to(block.device)
                    eigvecs_block_gpu = eigvecs_cpu.to(block.device)
                else:
                    raise
        eigvals_block = eigvals_block_gpu.cpu().numpy()
        eigvecs_block = eigvecs_block_gpu.cpu().numpy()

        del block, eigvals_block_gpu, eigvecs_block_gpu
        torch.cuda.empty_cache()

        if swap_eigenvectors:
            # reshape eigenvectors into [3 eigenvectors, 3 components, dz,dy,dx]
            v = eigvecs_block.reshape(3, 3, *eigvecs_block.shape[1:])
            w = eigvals_block

            # swap eigenvector #0 <-> #1 and their eigenvalues
            v[[0, 1], :, ...] = v[[1, 0], :, ...]
            w[[0, 1],    ...] = w[[1, 0],    ...]

            # flatten back
            eigvecs_block = v.reshape(9, *eigvecs_block.shape[1:])
            eigvals_block = w

        # impose handedness
        v = torch.from_numpy(eigvecs_block.reshape(3, 3, *eigvecs_block.shape[1:])).to(device)
        V_flat = v.reshape(3,3,-1).permute(2,0,1)    # [N,3,3]
        det = torch.linalg.det(V_flat)               # [N]
        mask = det < 0
        # flip the *entire* 3rd eigenvector (row index = 2)
        if mask.any():
            V_flat[mask, 2, :] *= -1
        # back to original layout
        v_corrected = V_flat.permute(1,2,0).reshape(3,3,*eigvecs_block.shape[1:])

        # Orient eigenvectors: keep right-handedness but align first vector toward +X on average.
        # Compute sign from the first component of v0 averaged over the block
        s = torch.sign(v_corrected[0, 0, ...].mean())
        s = torch.tensor(1.0, device=v_corrected.device) if s == 0 else s
        # Flip v0 and v2 together to preserve determinant > 0
        v_corrected[0, ...] *= s
        v_corrected[2, ...] *= s

        # numpy views
        v_np = v_corrected.cpu().numpy()     # [3,3,dz,dy,dx]
        w_np = eigvals_block                 # [3,dz,dy,dx]

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

            # Confidence (default: Fractional Anisotropy)
            l1, l2, l3 = w_np[0], w_np[1], w_np[2]
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
            conf_ds[oz0:oz1, oy0:oy1, ox0:ox1] = _quantize_unit_u8(conf[slz, sly, slx])

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
    
    # Get compressor
    if args.zarr_compressor.lower() == 'zstd':
        compressor = Blosc(cname='zstd', clevel=args.zarr_compression_level, shuffle=Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'lz4':
        compressor = Blosc(cname='lz4', clevel=args.zarr_compression_level, shuffle=Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'zlib':
        compressor = Blosc(cname='zlib', clevel=args.zarr_compression_level, shuffle=Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'none':
        compressor = None
    else:
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
    
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
            
            if logits_path:
                print(f"\n--- Structure Tensor Computation Finished ---")
                print(f"Structure tensor saved to: {logits_path}/structure_tensor")
                
                # Run eigenanalysis automatically unless --structure-tensor-only is specified
                if not args.structure_tensor_only:
                    print("\n--- Running Eigenanalysis ---")
                    _finalize_structure_tensor_torch(
                        zarr_path=logits_path,
                        chunk_size=chunk_size,
                        num_workers=args.num_workers,
                        compressor=compressor,
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
                compressor=compressor,
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
