import os
from typing import Optional, Tuple, Union
import numpy as np
# import zarr # No longer needed directly in VCDataset
# import tifffile # No longer needed directly in VCDataset
import torch
from torch.utils.data import Dataset
import re
from pathlib import Path

# Assuming these are in the correct relative paths
from utils.models.helpers import compute_steps_for_sliding_window
from data.volume import Volume
# Import utility functions directly from vesuvius package
from utils import list_files, is_aws_ec2_instance
# Import get_max_value from data.utils to avoid import errors
from data.utils import get_max_value, open_zarr


class VCDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            patch_size: Tuple[int, int, int],
            num_input_channels: Optional[int] = None, # Can often be inferred from Volume
            input_format: str = 'zarr', # Less critical now, type inferred from input_path/params
            step_size: float = 0.5,
            verbose: bool = False,
            mode: str = 'infer', # Currently only 'infer' logic is fully implemented
            num_parts: int = 1,
            part_id: int = 0,
            targets = None,  # Added targets parameter with None default
            # --- Volume Class Pass-through Parameters ---
            scroll_id: Optional[Union[int, str]] = None,
            energy: Optional[int] = None,
            resolution: Optional[float] = None,
            segment_id: Optional[int] = None,
            normalization_scheme: str = 'instance_zscore', # Default to instance z-score
            global_mean: Optional[float] = None,
            global_std: Optional[float] = None,
            return_as_type: str = 'np.float16', # Default float type for model input
            # return_as_tensor: bool = True, # Forcing True below
            domain: Optional[str] = None,
            skip_empty_patches: bool = True,  # Whether to skip empty (homogeneous) patches
            ):
        """
        Dataset for nnUNet inference using the Volume class for data access and preprocessing.

        Handles local/remote Zarr, scrolls, and segments uniformly via Volume.

        Args:
            input_path: Path to input data (local/remote Zarr, scroll ID, segment ID).
            targets: Output targets configuration (currently unused in inference).
            patch_size: Patch size for extraction (tuple of 3 ints - Z, Y, X).
            num_input_channels: Expected number of input channels (optional, can be inferred).
            input_format: Format hint ('zarr', 'volume'). Type is mainly inferred now.
            step_size: Step size for sliding window as a fraction of patch size (0.5 = 50% overlap).
            load_all: Ignored. Volume class handles data loading.
            verbose: Enable detailed output messages.
            mode: 'infer' (default). 'train' mode logic is not implemented here.
            num_parts: Number of parts to split the dataset into along Z-axis.
            part_id: Which part of the split dataset to use (0-indexed).

            scroll_id: Scroll ID for Volume (if input_path isn't a specific scroll/segment).
            energy: Energy value for Volume.
            resolution: Resolution value for Volume.
            segment_id: Segment ID for Volume (if input_path isn't a specific scroll/segment).
            normalization_scheme: Normalization method for Volume ('none', 'instance_zscore',
                                  'global_zscore', 'instance_minmax').
            global_mean: Global mean for 'global_zscore' scheme.
            global_std: Global standard deviation for 'global_zscore' scheme.
            return_as_type: Target NumPy dtype string for Volume output *before* tensor conversion
                             (e.g., 'np.float16', 'np.float32'). Default is 'np.float32'.
            domain: Data source domain for Volume ('dl.ash2txt', 'local'). Auto-detected if None.
        """
        self.input_path = input_path
        self.input_format = input_format # Keep for informational purposes
        self.targets = targets
        self.patch_size = patch_size
        self.step_size = step_size
        # self.load_all = False # Always false now
        self.verbose = verbose
        self.mode = mode
        self.return_as_tensor = True # Dataset __getitem__ always returns tensors
        self.skip_empty_patches = skip_empty_patches
        self.empty_patches_skipped = 0  # Counter for skipped patches

        # Data partitioning parameters
        if num_parts < 1:
            raise ValueError(f"num_parts must be >= 1, got {num_parts}")
        if not (0 <= part_id < num_parts):
            raise ValueError(f"part_id must be between 0 and {num_parts-1}, got {part_id}")
        self.num_parts = num_parts
        self.part_id = part_id

        if self.mode != 'infer':
            print(f"Warning: VCDataset mode is '{self.mode}'. Only 'infer' mode logic (sliding window) is fully implemented.")

        # --- Determine Volume Type and Path ---
        type_value: Union[str, int]
        use_path: Optional[str] = None

        if isinstance(input_path, str):
            input_lower = input_path.lower()
            # Check for Scroll ID format (e.g., "scroll1", "scroll1b")
            if re.match(r'scroll[0-9]+[a-z]*$', input_lower):
                 type_value = input_path # Pass original case potentially
                 if scroll_id is None: scroll_id = type_value # Set scroll_id if not provided
                 if self.verbose: print(f"Interpreting input_path '{input_path}' as Volume type 'scroll'.")
            # Check for Segment ID format (numeric string)
            elif input_path.isdigit():
                 type_value = input_path # Keep as string for Volume constructor flexibility
                 if segment_id is None: segment_id = int(input_path) # Set segment_id if not provided
                 if self.verbose: print(f"Interpreting input_path '{input_path}' as Volume type 'segment'.")
            # Check for explicit 'volume' format hint (less common now)
            elif input_format == 'volume':
                 # This case requires scroll_id or segment_id to be provided explicitly
                 if segment_id is not None:
                      type_value = str(segment_id)
                      if self.verbose: print(f"Using explicit segment_id {segment_id} as Volume type.")
                 elif scroll_id is not None:
                      type_value = f"scroll{scroll_id}" # Construct type
                      if self.verbose: print(f"Using explicit scroll_id {scroll_id} to set Volume type.")
                 else:
                      raise ValueError("input_format='volume' requires scroll_id or segment_id to be provided.")
            # Otherwise, assume it's a path (Zarr or potentially other formats Volume might support)
            else:
                 type_value = "zarr" # Default type when it looks like a path
                 use_path = input_path
                 if self.verbose: print(f"Interpreting input_path '{input_path}' as a path for Volume type 'zarr'.")
                 # Automatically set domain to local for file paths unless overridden
                 if domain is None and not use_path.startswith(('http://', 'https://', 's3://')):
                     domain = "local"
                     if self.verbose: print("Auto-setting domain to 'local' for non-HTTP/S3 path.")

        elif isinstance(input_path, int): # Handle integer scroll_id passed as input_path
             type_value = f"scroll{input_path}"
             if scroll_id is None: scroll_id = input_path
             if self.verbose: print(f"Interpreting integer input_path {input_path} as scroll ID.")
        else:
             raise TypeError(f"Unsupported input_path type: {type(input_path)}")


        # --- Initialize Volume Class ---
        try:
            if self.verbose:
                print("\n--- Initializing Volume ---")
                print(f"  Type: {type_value}")
                print(f"  Path: {use_path}")
                print(f"  Scroll ID: {scroll_id}")
                print(f"  Segment ID: {segment_id}")
                print(f"  Energy: {energy}")
                print(f"  Resolution: {resolution}")
                print(f"  Domain: {domain}")
                print(f"  Normalization: {normalization_scheme}")
                if normalization_scheme == 'global_zscore':
                     print(f"    Global Mean: {global_mean}, Global Std: {global_std}")
                print(f"  Return Type (NumPy): {return_as_type}")
                print(f"  Return As Tensor: {self.return_as_tensor}") # Use internal dataset flag
                print(f"  Input Channels: {num_input_channels}")
                print(f"  Targets: {targets}")
                print("---------------------------\n")

            # Validate Zarr path if provided - skip validation for remote paths
            if type_value == "zarr" and use_path is not None and not use_path.startswith(('http://', 'https://', 's3://')):
                p = Path(use_path)
                if not p.is_absolute():
                     abs_p = p.resolve()
                     if self.verbose: print(f"  Converting relative Zarr path '{use_path}' to absolute '{abs_p}'")
                     use_path = str(abs_p)

                if not os.path.exists(use_path):
                    raise FileNotFoundError(f"Zarr path does not exist: {use_path}")
                if not os.path.isdir(use_path):
                     if not use_path.endswith('.zip'): # Basic check
                         print(f"  Warning: Zarr path '{use_path}' exists but is not a directory.")
                # Check for key Zarr files (optional, Volume handles errors)
                # if not os.path.exists(os.path.join(use_path, '.zarray')) and not os.path.exists(os.path.join(use_path, '.zgroup')):
                #     print(f"  Warning: Path {use_path} might not be a Zarr store (missing .zarray/.zgroup).")
            elif type_value == "zarr" and use_path is not None and use_path.startswith('s3://'):
                if self.verbose: print(f"  Using S3 path: {use_path}")

            self.volume = Volume(
                type=type_value,
                scroll_id=scroll_id,
                energy=energy,
                resolution=resolution,
                segment_id=segment_id,
                # normalize=False, # Removed, use scheme
                normalization_scheme=normalization_scheme,
                global_mean=global_mean,
                global_std=global_std,
                return_as_type=return_as_type,
                return_as_tensor=self.return_as_tensor, # Ensure Volume returns tensors directly
                verbose=verbose,
                domain=domain,
                path=use_path
            )

            # Get shape and dtype from the primary resolution level (0)
            self.input_shape = self.volume.shape(0) # Z, Y, X or C, Z, Y, X
            self.input_dtype = self.volume.dtype # Original dtype before Volume's processing
            self.output_dtype = getattr(torch, return_as_type.replace('np.', '')) if self.return_as_tensor else getattr(np, return_as_type.replace('np.', ''))

            if self.verbose:
                print(f"Volume initialized successfully.")
                print(f"  Input Shape (from Volume level 0): {self.input_shape}")
                print(f"  Original Dtype (from Volume): {self.input_dtype}")
                print(f"  Output Dtype (expected from Volume): {self.output_dtype}")

            # Infer num_input_channels if not provided
            if num_input_channels is None:
                if len(self.input_shape) == 4: # Assume C, Z, Y, X
                    self.num_input_channels = self.input_shape[0]
                    if self.verbose: print(f"  Inferred num_input_channels: {self.num_input_channels}")
                else: # Assume Z, Y, X
                    self.num_input_channels = 1
                    if self.verbose: print(f"  Assuming single channel input (num_input_channels=1).")
            else:
                self.num_input_channels = num_input_channels
                 # Add a check
                if len(self.input_shape) == 4 and self.input_shape[0] != self.num_input_channels:
                     print(f"Warning: Provided num_input_channels ({self.num_input_channels}) does not match "
                           f"first dimension of Volume shape ({self.input_shape[0]}). Using provided value.")
                elif len(self.input_shape) == 3 and self.num_input_channels != 1:
                     print(f"Warning: Provided num_input_channels ({self.num_input_channels}) != 1 for 3D Volume shape. "
                           f"Will add channel dimension in getitem. Using provided value.")

        except Exception as e:
            print(f"ERROR: Failed to initialize Volume within VCDataset.")
            # Log details that might be helpful
            print(f"  Attempted Volume params: type={type_value}, path={use_path}, scroll={scroll_id}, seg={segment_id}, ...")
            raise ValueError(f"Error initializing Volume: {e}") from e


        # --- Sliding Window Position Calculation (for infer mode) ---
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
                raise ValueError(f"Unsupported input shape dimension from Volume: {self.input_shape}")

            # Generate all potential coordinates
            z_positions = compute_steps_for_sliding_window(image_size[0], pZ, self.step_size)
            y_positions = compute_steps_for_sliding_window(image_size[1], pY, self.step_size)
            x_positions = compute_steps_for_sliding_window(image_size[2], pX, self.step_size)

            if self.verbose:
                print(f"\nCalculating sliding window positions:")
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

            # Apply Z-axis partitioning if num_parts > 1
            if self.num_parts > 1:
                max_z = image_size[0]
                z_per_part = max_z / self.num_parts
                z_start = int(z_per_part * self.part_id)
                z_end = int(z_per_part * (self.part_id + 1)) if self.part_id < self.num_parts - 1 else max_z

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

        # Define the slices for fetching data from Volume
        # Use spatial dimensions (Z, Y, X) correctly based on input_shape length
        spatial_dims_indices = slice(1, None) if len(self.input_shape) == 4 else slice(None)
        image_shape_zyx = self.input_shape[spatial_dims_indices]

        z_slice = slice(z, min(z + pZ, image_shape_zyx[0]))
        y_slice = slice(y, min(y + pY, image_shape_zyx[1]))
        x_slice = slice(x, min(x + pX, image_shape_zyx[2]))

        try:
            # Fetch the (potentially smaller) patch from Volume
            # Volume's __getitem__ handles normalization, type conversion, and tensor conversion
            if len(self.input_shape) == 3: # Input is Z, Y, X
                # For 3D input (Z, Y, X), we always create a single-channel tensor
                # The model expects a single channel regardless of how many output classes it produces
                
                # Volume returns (Z, Y, X) tensor
                extracted_tensor = self.volume[z_slice, y_slice, x_slice]
                
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
                # Use the dtype returned by Volume
                # ALWAYS use a single channel (1) here since we're coming from Z,Y,X data
                patch_tensor = torch.zeros((1, pZ, pY, pX), dtype=extracted_tensor.dtype)
                
                # Copy fetched data into the patch tensor
                patch_tensor[0, :fetched_z, :fetched_y, :fetched_x] = extracted_tensor

            elif len(self.input_shape) == 4: # Input is C, Z, Y, X
                # Volume returns (C, Z, Y, X) tensor
                
                # For multiclass models, we need to ensure we're using exactly the right number of input channels
                available_channels = self.input_shape[0]
                
                # Verify that we have enough channels in our data
                if available_channels < self.num_input_channels:
                    raise ValueError(f"Model expects {self.num_input_channels} input channels, but data only has {available_channels}")
                
                # Extract exactly the number of channels the model needs
                if self.verbose:
                    print(f"4D input: Extracting {self.num_input_channels} channels from data with {available_channels} channels")
                
                # Take exactly what the model expects
                extracted_tensor = self.volume[:self.num_input_channels, z_slice, y_slice, x_slice]
                
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
             print(f"  Volume shape: {self.input_shape}")
             # Re-raise the error with more context
             raise RuntimeError(f"Failed to get item {idx} (pos {z},{y},{x}): {e}") from e


        # Data should already be a tensor of the correct type due to Volume settings
        # assert isinstance(patch_tensor, torch.Tensor), "Volume did not return a tensor as expected"
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
