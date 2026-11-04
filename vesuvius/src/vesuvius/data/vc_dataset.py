import os
from typing import Optional, Tuple, Union
import numpy as np
# import zarr # No longer needed directly in VCDataset
import torch
from torch.utils.data import Dataset
import re
from pathlib import Path

# Assuming these are in the correct relative paths
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window
import vc_reader.transforms as A
from vc_reader.reader import ZarrGroup, open_node
from vesuvius.data.zarr_chunk_index import build_chunk_occupancy, compute_patch_non_empty_mask

def _normalization_pipeline(scheme, global_mean, global_std, intensity_props):
    """The transform of a normalization scheme; None for ``none``."""
    if scheme == 'none':
        return None
    if scheme == 'instance_zscore':
        transform = A.Normalize('zscore', nonzero=False, eps=1e-8)
    elif scheme == 'global_zscore':
        if global_mean is None or global_std is None:
            raise ValueError("global_mean and global_std must be provided when normalization_scheme is 'global_zscore'")
        transform = A.Normalize('fixed', center=float(global_mean), scale=max(float(global_std), 1e-8))
    elif scheme == 'instance_minmax':
        transform = A.Normalize('minmax')
    elif scheme == 'percentile_minmax':
        transform = A.Normalize('percentile', lower=1.0, upper=99.0)
    elif scheme == 'ct':
        required = ['mean', 'std', 'percentile_00_5', 'percentile_99_5']
        if not intensity_props or not all(key in intensity_props for key in required):
            raise ValueError(f"CT normalization requires intensity_props with keys: {required}")
        transform = A.Normalize(
            'fixed', center=float(intensity_props['mean']), scale=max(float(intensity_props['std']), 1e-8),
            bounds=(float(intensity_props['percentile_00_5']), float(intensity_props['percentile_99_5'])))
    else:
        raise ValueError(f"Invalid normalization_scheme: {scheme!r}")
    return A.Pipeline([transform], backend='cpu')


class VCDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            patch_size: Tuple[int, int, int],
            num_input_channels: Optional[int] = None, # Can often be inferred from the array
            step_size: float = 0.5,
            verbose: bool = False,
            mode: str = 'infer', # Currently only 'infer' logic is fully implemented
            num_parts: int = 1,
            part_id: int = 0,
            bbox: Optional[Tuple[int, int, int, int, int, int]] = None,
            targets = None,  # Added targets parameter with None default
            normalization_scheme: str = 'instance_zscore', # Default to instance z-score
            global_mean: Optional[float] = None,
            global_std: Optional[float] = None,
            intensity_props: Optional[dict] = None,
            return_as_type: str = 'np.float16', # Default float type for model input
            skip_empty_patches: bool = True,  # Whether to skip empty (homogeneous) patches
            anon: bool = False,  # Use anonymous (unsigned) requests for S3 input paths
            cache: bool = False,
            cache_size_mb: int = 256,
            io_threads: Optional[int] = None,  # Reader threads; 1 inside DataLoader workers
            ):
        """
        Dataset for nnUNet inference over a local or remote Zarr array or OME-Zarr group (level 0).

        Args:
            input_path: Path or URL of the input Zarr. ``vc_reader.catalog`` finds those of published volumes.
            targets: Output targets configuration (currently unused in inference).
            patch_size: Patch size for extraction (tuple of 3 ints - Z, Y, X).
            num_input_channels: Expected number of input channels (optional, can be inferred).
            step_size: Step size for sliding window as a fraction of patch size (0.5 = 50% overlap).
            verbose: Enable detailed output messages.
            mode: 'infer' (default). 'train' mode logic is not implemented here.
            num_parts: Number of parts to split the dataset into along Z-axis.
            part_id: Which part of the split dataset to use (0-indexed).
            bbox: Optional region of interest as (z_min, z_max, y_min, y_max, x_min, x_max)
                  in *global* voxel coordinates, half-open ([min, max)). When given, the
                  sliding-window grid covers only this region, but patch coordinates stay
                  in the global volume frame so downstream blending/finalization remain
                  aligned. Bounds are clamped to the volume; an ROI smaller than the patch
                  size along an axis is grown to the patch size (shifted back inside the
                  volume when possible).

            normalization_scheme: Normalization method ('none', 'instance_zscore',
                                  'global_zscore', 'instance_minmax').
            global_mean: Global mean for 'global_zscore' scheme.
            global_std: Global standard deviation for 'global_zscore' scheme.
            return_as_type: Target NumPy dtype string *before* tensor conversion
                             (e.g., 'np.float16', 'np.float32'). Default is 'np.float32'.
            anon: Use anonymous requests for input S3 reads.
            cache: Keep an in-memory LRU cache of fetched chunks.
                Overlapping patches then reuse chunks instead of fetching them again.
                Requires zarr 3.
            cache_size_mb: Size bound of that cache in megabytes (default 256).
        """
        self.input_path = input_path
        self.targets = targets
        self.patch_size = patch_size
        self.step_size = step_size
        # self.load_all = False # Always false now
        self.verbose = verbose
        self.mode = mode
        self.return_as_tensor = True # Dataset __getitem__ always returns tensors
        self.skip_empty_patches = skip_empty_patches
        self.anon = anon
        self.empty_patches_skipped = 0  # Counter for skipped patches
        self.non_empty_mask = None  # Per-patch bool mask; populated below for infer mode with zarr input

        # Data partitioning parameters
        if num_parts < 1:
            raise ValueError(f"num_parts must be >= 1, got {num_parts}")
        if not (0 <= part_id < num_parts):
            raise ValueError(f"part_id must be between 0 and {num_parts-1}, got {part_id}")
        self.num_parts = num_parts
        self.part_id = part_id

        if bbox is not None:
            if len(bbox) != 6:
                raise ValueError(
                    f"bbox must be (z_min, z_max, y_min, y_max, x_min, x_max), got {bbox}"
                )
            for axis, (lo, hi) in enumerate(zip(bbox[0::2], bbox[1::2])):
                if lo is not None and lo < 0:
                    raise ValueError(f"bbox axis {axis} min {lo} is negative")
                if lo is not None and hi is not None and hi <= lo:
                    raise ValueError(
                        f"bbox axis {axis} range [{lo}, {hi}) is empty; "
                        f"bounds are half-open and must satisfy min < max"
                    )
        self.bbox = tuple(int(v) if v is not None else None for v in bbox) if bbox is not None else None

        if self.mode != 'infer':
            print(f"Warning: VCDataset mode is '{self.mode}'. Only 'infer' mode logic (sliding window) is fully implemented.")

        use_path = str(input_path)
        if not use_path.startswith(('http://', 'https://', 's3://')):
            use_path = str(Path(use_path).resolve())
            if not os.path.exists(use_path):
                raise FileNotFoundError(f"Zarr path does not exist: {use_path}")
        self._normalize = _normalization_pipeline(normalization_scheme, global_mean, global_std, intensity_props)

        node = open_node(
            use_path, auth=None if self.anon else "auto", threads=io_threads,
            memory_cache=int(cache_size_mb) << 20 if cache else None)
        self._array_url = use_path.rstrip('/')
        if isinstance(node, ZarrGroup):
            node = node["0"]
            self._array_url += '/0'
        self.reader = node

        self.input_shape = tuple(int(n) for n in self.reader.shape)  # Z, Y, X or C, Z, Y, X
        self.input_dtype = self.reader.dtype
        target_dtype = return_as_type.replace('np.', '')
        self._output_np_dtype = np.dtype(target_dtype)
        self.output_dtype = getattr(torch, target_dtype) if self.return_as_tensor else getattr(np, target_dtype)

        # Infer num_input_channels if not provided
        if num_input_channels is None:
            self.num_input_channels = self.input_shape[0] if len(self.input_shape) == 4 else 1
        else:
            self.num_input_channels = num_input_channels
            if len(self.input_shape) == 4 and self.input_shape[0] != self.num_input_channels:
                print(f"Warning: Provided num_input_channels ({self.num_input_channels}) does not match "
                      f"first dimension of the array shape ({self.input_shape[0]}). Using provided value.")
            elif len(self.input_shape) == 3 and self.num_input_channels != 1:
                print(f"Warning: Provided num_input_channels ({self.num_input_channels}) != 1 for a 3D array. "
                      f"Will add channel dimension in getitem. Using provided value.")

        # --- Position Calculation (for infer mode) ---
        self.all_positions = []
        if mode == 'infer':
            if not isinstance(self.patch_size, (list, tuple)) or len(self.patch_size) != 3:
                 raise ValueError(f"patch_size must be a tuple/list of 3 integers (Z, Y, X), got {self.patch_size}")

            pZ, pY, pX = self.patch_size
            # Get the 3D spatial dimensions (Z, Y, X)
            if len(self.input_shape) == 3: # Z, Y, X
                image_size = self.input_shape
            elif len(self.input_shape) == 4: # C, Z, Y, X
                image_size = self.input_shape[1:]
            else:
                raise ValueError(f"Unsupported input shape dimension: {self.input_shape}")

            roi = None
            if self.bbox is not None:
                roi = self._clamp_bbox_to_image(self.bbox, image_size, self.patch_size)

            if roi is not None:
                if self.verbose:
                    print(f"\nUsing bbox-restricted sliding window (global coords): "
                          f"z=[{roi[0][0]}, {roi[0][1]}), y=[{roi[1][0]}, {roi[1][1]}), "
                          f"x=[{roi[2][0]}, {roi[2][1]})")
                # Build the grid the whole volume would have used, then keep the
                # patches that touch the ROI. Anchoring the grid to the ROI origin
                # instead would place patches the full-volume run never places -
                # a 256-long ROI yields stride 64 where the volume yields ~96 - so
                # every voxel gets a different set of overlapping patches and the
                # blended result no longer matches the corresponding sub-region of
                # a full run. Selecting from the global grid keeps them identical,
                # which is the whole point of being able to redo one region.
                (z0, z1), (y0, y1), (x0, x1) = roi

                def _overlapping(extent, patch, lo, hi):
                    return [s for s in compute_steps_for_sliding_window(extent, patch, self.step_size)
                            if s < hi and s + patch > lo]

                z_positions = _overlapping(image_size[0], pZ, z0, z1)
                y_positions = _overlapping(image_size[1], pY, y0, y1)
                x_positions = _overlapping(image_size[2], pX, x0, x1)
            else:
                # Full-volume sliding window
                if self.verbose:
                    print("\nUsing full volume sliding window")

                # Generate all potential coordinates
                z_positions = compute_steps_for_sliding_window(image_size[0], pZ, self.step_size)
                y_positions = compute_steps_for_sliding_window(image_size[1], pY, self.step_size)
                x_positions = compute_steps_for_sliding_window(image_size[2], pX, self.step_size)

            if self.verbose:
                print(f"  Image Size (Z, Y, X): {image_size}")
                print(f"  Patch Size (Z, Y, X): {self.patch_size}")
                print(f"  Step Size Factor: {self.step_size}")
                print(f"  Skip Empty Patches: {self.skip_empty_patches}")
                print(f"  Num Positions (Z, Y, X): ({len(z_positions)}, {len(y_positions)}, {len(x_positions)})")
                total_patches = len(z_positions) * len(y_positions) * len(x_positions)
                print(f"  Total potential patches: {total_patches}")

            # Combine positions
            for z in z_positions:
                for y in y_positions:
                    for x in x_positions:
                        self.all_positions.append((z, y, x))

            # Optional coverage diagnostics in verbose mode
            if self.verbose and self.all_positions:
                # Coverage is judged against the tiled extent: the ROI when a bbox is
                # active, the whole volume otherwise.
                (tz0, tz1), (ty0, ty1), (tx0, tx1) = roi if roi is not None else (
                    (0, image_size[0]), (0, image_size[1]), (0, image_size[2]))
                max_start_z = max(pos[0] for pos in self.all_positions)
                max_start_y = max(pos[1] for pos in self.all_positions)
                max_start_x = max(pos[2] for pos in self.all_positions)
                cov_z = (max_start_z + pZ == tz1) if (tz1 - tz0) >= pZ else (max_start_z == tz0)
                cov_y = (max_start_y + pY == ty1) if (ty1 - ty0) >= pY else (max_start_y == ty0)
                cov_x = (max_start_x + pX == tx1) if (tx1 - tx0) >= pX else (max_start_x == tx0)
                print("\nTiling coverage check:")
                print(f"  Z axis: start in [{tz0}..{max_start_z}], patch {pZ}, extent [{tz0}, {tz1}) -> covers end: {cov_z}")
                print(f"  Y axis: start in [{ty0}..{max_start_y}], patch {pY}, extent [{ty0}, {ty1}) -> covers end: {cov_y}")
                print(f"  X axis: start in [{tx0}..{max_start_x}], patch {pX}, extent [{tx0}, {tx1}) -> covers end: {cov_x}")

            # Apply Z-axis partitioning if num_parts > 1
            if self.num_parts > 1:
                # Partition the Z span the selected patches actually occupy. With a
                # bbox active, splitting the full volume range would leave most parts
                # empty; splitting the ROI range instead would silently drop the
                # patches that start before it, since the grid comes from the whole
                # volume and the first patch covering the ROI usually begins outside.
                if self.all_positions:
                    occupied_z = sorted({pos[0] for pos in self.all_positions})
                    part_z0, part_z1 = occupied_z[0], occupied_z[-1] + 1
                else:
                    part_z0, part_z1 = (0, image_size[0])
                z_per_part = (part_z1 - part_z0) / self.num_parts
                z_start = part_z0 + int(z_per_part * self.part_id)
                z_end = part_z0 + int(z_per_part * (self.part_id + 1)) if self.part_id < self.num_parts - 1 else part_z1

                if self.verbose:
                    print(f"\nApplying Z-axis partitioning:")
                    print(f"  Num Parts: {self.num_parts}, Part ID: {self.part_id}")
                    print(f"  Z Range for this part: [{z_start}, {z_end})")

                # Filter positions based on the patch *starting* coordinate
                # A patch belongs to the partition its starting Z coordinate falls into.
                # This is simpler than checking overlap and ensures non-overlapping assignment.
                original_count = len(self.all_positions)
                self.all_positions = [pos for pos in self.all_positions if z_start <= pos[0] < z_end]
                filtered_count = len(self.all_positions)

                if self.verbose:
                    print(f"  Filtered positions from {original_count} to {filtered_count}")
                    if filtered_count > 0:
                         print(f"  Part {self.part_id} Z-range of positions: [{self.all_positions[0][0]} - {self.all_positions[-1][0]}]")
                    else:
                         print(f"  Warning: No patch starting positions found in the Z-range [{z_start}, {z_end}) for part {self.part_id}.")

            # Build per-patch non-empty mask from the input zarr's chunk occupancy.
            # This lets us drop patches that fall entirely in empty regions of a sparse
            # input without ever touching the underlying data; fallback to None leaves
            # today's lazy min/max-based detection untouched.
            self.non_empty_mask = self._build_non_empty_mask_from_chunks(use_path)


    @staticmethod
    def _clamp_bbox_to_image(bbox, image_size, patch_size):
        """Clamp a global-coordinate bbox to the volume and grow it to patch size.

        Returns ((z0, z1), (y0, y1), (x0, x1)), half-open. Raises if the bbox lies
        entirely outside the volume. An ROI shorter than the patch along an axis is
        grown to the patch size, shifted back inside the volume when it would
        overhang (matching the sliding window's guarantee that starts stay >= 0).
        """
        roi = []
        for axis in range(3):
            lo, hi = bbox[2 * axis], bbox[2 * axis + 1]
            dim = image_size[axis]
            patch = patch_size[axis]
            lo = 0 if lo is None else lo
            hi = dim if hi is None else hi
            lo_c, hi_c = min(lo, dim), min(hi, dim)
            if hi_c <= lo_c:
                raise ValueError(
                    f"bbox axis {axis} range [{lo}, {hi}) lies outside the volume "
                    f"(size {dim} along this axis)"
                )
            if hi_c - lo_c < patch:
                hi_c = min(lo_c + patch, dim)
                lo_c = max(0, hi_c - patch)
            roi.append((lo_c, hi_c))
        return tuple(roi)

    def _read_patch(self, *region):
        """A normalized region as a tensor of the output dtype. Statistics are those of the patch
        before padding, per channel."""
        data = self.reader[region]
        if self._normalize is not None:
            has_channels = data.ndim > 3
            image = data.astype(np.float32, copy=False)
            image = self._normalize({'image': image if has_channels else image[np.newaxis]}, seed=0)['image']
            data = image if has_channels else image[0]
        return torch.from_numpy(np.ascontiguousarray(data.astype(self._output_np_dtype)))

    def _build_non_empty_mask_from_chunks(self, use_path):
        if not self.skip_empty_patches or not self.all_positions:
            return None
        if use_path is None:
            return None

        array_url = self._array_url
        array_chunks = tuple(self.reader.chunks)
        array_shape = tuple(self.reader.shape)

        occupancy = build_chunk_occupancy(
            array_url,
            chunks=array_chunks,
            shape=array_shape,
            verbose=self.verbose,
            anon=self.anon,
        )
        if occupancy is None:
            return None

        # Drop channel axis for 4D shapes to get spatial chunk sizes.
        if len(array_chunks) == 4:
            chunks_spatial = array_chunks[1:]
        else:
            chunks_spatial = array_chunks

        mask = compute_patch_non_empty_mask(
            occupancy,
            self.all_positions,
            tuple(self.patch_size),
            chunks_spatial,
        )

        if self.verbose:
            n = int(len(mask))
            k = int(mask.sum())
            print(f"  Pre-filtered {n - k} of {n} patches as empty via chunk index ({k} remain)")

        return mask


    def active_view(self):
        """Return the dataset view the DataLoader should iterate.

        If the input zarr's chunk occupancy let us pre-filter empty patches,
        this returns a `torch.utils.data.Subset` over the non-empty indices
        (so `len(...)` reflects only patches that will actually run through
        the model, and tqdm totals/ETAs become honest). Otherwise returns
        `self`. `Subset[i]` forwards to `self[original_idx]`, so the patch
        index carried through `__getitem__` and used as the zarr write index
        is unchanged.
        """
        mask = getattr(self, 'non_empty_mask', None)
        if mask is None or len(mask) != len(self.all_positions):
            return self
        indices = np.flatnonzero(mask).tolist()
        return torch.utils.data.Subset(self, indices)


    def set_distributed(self, rank: int, world_size: int):
        """
        Configures the dataset for distributed processing by assigning a subset of patches to this rank.
        """
        if world_size <= 1 or not (0 <= rank < world_size):
            if self.verbose: print("Distribution not required or invalid rank/world_size.")
            return

        total_positions = len(self.all_positions)
        if total_positions == 0:
             if self.verbose: print(f"Rank {rank}: No positions to distribute.")
             return

        # Determine the subset of indices for this rank
        num_per_rank = total_positions // world_size
        remainder = total_positions % world_size
        start_idx = rank * num_per_rank + min(rank, remainder)
        end_idx = start_idx + num_per_rank + (1 if rank < remainder else 0)

        # Slice the positions
        assigned_positions = self.all_positions[start_idx:end_idx]
        num_assigned = len(assigned_positions)

        if self.verbose:
            partition_info = f"(from Z-partition {self.part_id}/{self.num_parts})" if self.num_parts > 1 else ""
            print(f"\nDistributing data for Rank {rank}/{world_size} {partition_info}:")
            print(f"  Total positions in current partition: {total_positions}")
            print(f"  Assigning indices [{start_idx} to {end_idx -1}] ({num_assigned} positions) to Rank {rank}")
            if num_assigned > 0:
                z_values = [pos[0] for pos in assigned_positions]
                min_z, max_z = min(z_values), max(z_values)
                print(f"  Rank {rank} Z-range of positions: [{min_z} - {max_z}]")
            else:
                 print(f"  Warning: Rank {rank} has been assigned 0 positions.")


        self.all_positions = assigned_positions


    def __len__(self):
        return len(self.all_positions)
        
    def get_empty_patches_report(self):
        """Return a report of empty patches that were skipped"""
        return {
            "total_skipped": self.empty_patches_skipped,
            "total_positions": len(self.all_positions),
            "skip_ratio": self.empty_patches_skipped / (len(self.all_positions) + self.empty_patches_skipped) if self.empty_patches_skipped > 0 else 0
        }

    def __getitem__(self, idx):
        if idx >= len(self.all_positions):
             raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.all_positions)}")

        z, y, x = self.all_positions[idx]
        pZ, pY, pX = self.patch_size

        # Define the slices for fetching data
        # Use spatial dimensions (Z, Y, X) correctly based on input_shape length
        spatial_dims_indices = slice(1, None) if len(self.input_shape) == 4 else slice(None)
        image_shape_zyx = self.input_shape[spatial_dims_indices]

        z_slice = slice(z, min(z + pZ, image_shape_zyx[0]))
        y_slice = slice(y, min(y + pY, image_shape_zyx[1]))
        x_slice = slice(x, min(x + pX, image_shape_zyx[2]))

        try:
            # Fetch the (potentially smaller) patch, normalized and converted
            if len(self.input_shape) == 3: # Input is Z, Y, X
                # For 3D input (Z, Y, X), we always create a single-channel tensor
                # The model expects a single channel regardless of how many output classes it produces
                
                # (Z, Y, X) tensor
                extracted_tensor = self._read_patch(z_slice, y_slice, x_slice)
                
                # Fast check for empty patch - skip if all zeros or all values are the same
                # This avoids processing empty patches which won't contain any information
                # Check if tensor is empty (all zeros or same value) - but don't skip entirely
                is_empty = False
                if self.skip_empty_patches and extracted_tensor.numel() > 0:
                    # We use a fast min/max comparison rather than var() which is more expensive
                    min_val = extracted_tensor.min().item()
                    max_val = extracted_tensor.max().item()
                    if min_val == max_val:
                        # All values are the same - this is likely empty space
                        self.empty_patches_skipped += 1
                        if self.verbose and self.empty_patches_skipped % 100 == 0:
                            print(f"Detected {self.empty_patches_skipped} empty patches so far")
                        is_empty = True
                
                # We need to add channel dim and pad
                fetched_z, fetched_y, fetched_x = extracted_tensor.shape
                
                # Initialize the full-size patch tensor with zeros (for padding)
                # Use the dtype of the patch
                # ALWAYS use a single channel (1) here since we're coming from Z,Y,X data
                patch_tensor = torch.zeros((1, pZ, pY, pX), dtype=extracted_tensor.dtype)
                
                # Copy fetched data into the patch tensor
                patch_tensor[0, :fetched_z, :fetched_y, :fetched_x] = extracted_tensor

            elif len(self.input_shape) == 4: # Input is C, Z, Y, X
                # (C, Z, Y, X) tensor
                
                # For multiclass models, we need to ensure we're using exactly the right number of input channels
                available_channels = self.input_shape[0]
                
                # Verify that we have enough channels in our data
                if available_channels < self.num_input_channels:
                    raise ValueError(f"Model expects {self.num_input_channels} input channels, but data only has {available_channels}")
                
                # Extract exactly the number of channels the model needs
                if self.verbose:
                    print(f"4D input: Extracting {self.num_input_channels} channels from data with {available_channels} channels")
                
                # Take exactly what the model expects
                extracted_tensor = self._read_patch(slice(0, self.num_input_channels), z_slice, y_slice, x_slice)
                
                # Check for empty patch - flag if all values are the same
                is_empty = False
                if self.skip_empty_patches and extracted_tensor.numel() > 0:
                    # Check if tensor is empty (all same value) across all channels
                    is_empty = True
                    
                    # Check just the first few channels for efficiency
                    check_channels = min(3, extracted_tensor.shape[0])
                    for c in range(check_channels):
                        channel_tensor = extracted_tensor[c]
                        if channel_tensor.numel() > 0:
                            min_val = channel_tensor.min().item()
                            max_val = channel_tensor.max().item()
                            if min_val != max_val:
                                # Found variation in this channel, not empty
                                is_empty = False
                                break
                    
                    if is_empty:
                        # All checked channels have constant values - likely empty space
                        self.empty_patches_skipped += 1
                        if self.verbose and self.empty_patches_skipped % 100 == 0:
                            print(f"Detected {self.empty_patches_skipped} empty patches so far")
                
                # Get dimensions for padding
                fetched_c, fetched_z, fetched_y, fetched_x = extracted_tensor.shape
                
                # Initialize the full-size patch tensor with exact channel count
                patch_tensor = torch.zeros((self.num_input_channels, pZ, pY, pX), dtype=extracted_tensor.dtype)
                
                # Copy fetched data
                patch_tensor[:, :fetched_z, :fetched_y, :fetched_x] = extracted_tensor
            else:
                 # Should have been caught in init, but safety check
                 raise RuntimeError(f"Unsupported volume shape encountered in getitem: {self.input_shape}")

        except Exception as e:
             print(f"ERROR fetching/padding patch at ZYX=({z},{y},{x}), index={idx}")
             print(f"  Slices: Z={z_slice}, Y={y_slice}, X={x_slice}")
             print(f"  Array shape: {self.input_shape}")
             # Re-raise the error with more context
             raise RuntimeError(f"Failed to get item {idx} (pos {z},{y},{x}): {e}") from e


        # Data is already a tensor of the correct type
        # assert patch_tensor.dtype == self.output_dtype, f"Expected dtype {self.output_dtype} but got {patch_tensor.dtype}"


        position_tuple = (int(z), int(y), int(x))

        return {
            "data": patch_tensor, # Key required by nnUNet inference
            "pos": position_tuple, # Pass position for potential stitching later
            "index": idx, # Pass original index
            "is_empty": is_empty # Flag to indicate if patch is empty (to skip model inference)
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that processes empty patches without filtering them.
        """
        # Extract items by key (including the is_empty flag)
        data = torch.stack([item["data"] for item in batch])
        pos = [item["pos"] for item in batch]
        indices = [item["index"] for item in batch]
        is_empty = [item.get("is_empty", False) for item in batch]
        
        return {
            "data": data,
            "pos": pos,
            "index": indices,
            "is_empty": is_empty
        }
