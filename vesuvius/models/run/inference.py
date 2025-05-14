import torch
import tensorstore as ts
import numpy as np
import asyncio
import math
import tempfile
import shutil
import os
import json
import multiprocessing
# Set multiprocessing start method to 'spawn' for TensorStore compatibility
# 'fork' is not allowed since tensorstore uses internal threading
multiprocessing.set_start_method('spawn', force=True)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.models.load_nnunet_model import load_model_for_inference
from data.vc_dataset import VCDataset

class Inferer():
    def __init__(self,
                 model_path: str = None,
                 input_dir: str = None,
                 output_dir: str = None,
                 input_format: str = 'zarr',
                 tta_type: str = 'mirroring', # 'mirroring' or 'rotation'
                 # tta_combinations: int = 3,
                 # tta_rotation_weights: [list, tuple] = (1, 1, 1),
                 do_tta: bool = True,
                 num_parts: int = 1,
                 part_id: int = 0,
                 overlap: float = 0.5,
                 batch_size: int = 1,
                 patch_size: [list, tuple] = None,
                 save_softmax: bool = False,
                 cache_pool: float = 1e10,
                 normalization_scheme: str = 'instance_zscore',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_dataloader_workers: int = 4,
                 verbose: bool = False,
                 skip_empty_patches: bool = True,  # Skip empty/homogeneous patches
                 # Additional parameters for Volume class
                 scroll_id: [str, int] = None,
                 segment_id: [str, int] = None,
                 energy: int = None,
                 resolution: float = None,
                 # Hugging Face parameters
                 hf_token: str = None
                 ):

        self.model_path = model_path
        self.input = input_dir
        self.do_tta = do_tta
        self.tta_type = tta_type
        # self.tta_combinations = tta_combinations
        # self.tta_rotation_weights = tta_rotation_weights
        self.num_parts = num_parts
        self.part_id = part_id
        self.overlap = overlap
        self.batch_size = batch_size
        self.patch_size = tuple(patch_size) if patch_size is not None else None  # Can be None, will derive from model
        self.save_softmax = save_softmax
        self.cache_pool = cache_pool
        self.verbose = verbose
        self.normalization_scheme = normalization_scheme
        self.input_format = input_format
        self.device = torch.device(device)
        self.num_dataloader_workers = num_dataloader_workers
        self.skip_empty_patches = skip_empty_patches
        # Volume-specific parameters
        self.scroll_id = scroll_id
        self.segment_id = segment_id
        self.energy = energy
        self.resolution = resolution
        # Hugging Face parameters
        self.hf_token = hf_token
        # These will be set after model loading if not provided
        self.model_patch_size = None
        self.num_classes = None

        # --- Validation ---
        if not self.input or self.model_path is None:
            raise ValueError("Input directory and model path must be provided.")
        if self.num_parts > 1:
            if self.part_id < 0 or self.part_id >= self.num_parts:
                raise ValueError(f"Invalid part_id {self.part_id} for num_parts {self.num_parts}.")
        if self.overlap < 0 or self.overlap > 1:
            raise ValueError(f"Invalid overlap value {self.overlap}. Must be between 0 and 1.")
        if self.tta_type not in ['mirroring', 'rotation']:
             raise ValueError(f"Invalid tta_type '{self.tta_type}'. Must be 'mirroring' or 'rotation'.")
        # Defer patch size validation until after model loading if not explicitly provided
        if self.patch_size is not None and self.tta_type == 'rotation':
            if len(self.patch_size) != 3:
                raise ValueError(f"Rotation TTA requires 3D patch size, got {self.patch_size}.")
            # Relaxing square patch requirement, but should be aware torch.rot90 behavior
            # if self.patch_size[0] != self.patch_size[1] or self.patch_size[0] != self.patch_size[2]:
            #     print(f"Warning: Rotation TTA might behave unexpectedly with non-square patches {self.patch_size} depending on torch.rot90 implementation.")


        # --- Output Setup ---
        self._temp_dir_obj = None
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            raise ValueError("Output directory must be provided.")

        # --- Placeholders ---
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.output_store = None
        self.num_classes = None
        self.num_total_patches = None
        self.current_patch_write_index = 0


    def _load_model(self):
        # Load model onto the specified device
        # Check if model_path is a Hugging Face model path (starts with "hf://")
        if isinstance(self.model_path, str) and self.model_path.startswith("hf://"):
            # Extract the repository ID from the path
            hf_model_path = self.model_path.replace("hf://", "")
            if self.verbose:
                print(f"Loading model from Hugging Face repo: {hf_model_path}")
            model_info = load_model_for_inference(
                model_folder=None,
                hf_model_path=hf_model_path,
                hf_token=self.hf_token if hasattr(self, 'hf_token') else None,
                device_str=str(self.device),
                verbose=self.verbose
            )
        else:
            # Load from local path
            if self.verbose:
                print(f"Loading model from local path: {self.model_path}")
            model_info = load_model_for_inference(
                model_folder=self.model_path,
                device_str=str(self.device),
                verbose=self.verbose
            )
        
        # model loader returns a dict, network is the actual model
        model = model_info['network']
        model.eval()
        
        # Get patch size and number of classes from model_info
        self.model_patch_size = tuple(model_info.get('patch_size', (192, 192, 192)))
        self.num_classes = model_info.get('num_seg_heads', None)
        
        # use models patch size if one wasn't specified
        if self.patch_size is None:
            self.patch_size = self.model_patch_size
            if self.verbose:
                print(f"Using model's patch size: {self.patch_size}")
        else:
            if self.verbose and self.patch_size != self.model_patch_size:
                print(f"Warning: Using user-provided patch size {self.patch_size} instead of model's default: {self.model_patch_size}")
        
        # Validate patch size for rotation TTA if needed
        if self.patch_size is not None and self.tta_type == 'rotation':
            if len(self.patch_size) != 3:
                raise ValueError(f"Rotation TTA requires 3D patch size, got {self.patch_size}.")
        
        # Confirm num_classes if it couldn't be determined from model_info
        if self.num_classes is None:
            if self.verbose:
                print("Number of classes not found in model_info, performing dummy inference...")
            
            # Determine input channels from model_info if possible
            input_channels = model_info.get('num_input_channels', 1)
            dummy_input_shape = (1, input_channels, *self.patch_size)
            dummy_input = torch.randn(dummy_input_shape, device=self.device)
            
            try:
                with torch.no_grad():
                    dummy_output = model(dummy_input)
                self.num_classes = dummy_output.shape[1]  # N, C, D, H, W
                if self.verbose:
                    print(f"Inferred number of output classes via dummy inference: {self.num_classes}")
            except Exception as e:
                print(f"Warning: Could not automatically determine number of classes via dummy inference: {e}")
                print("Ensure your model is loaded correctly and check the expected input shape.")
                # Default to binary segmentation as fallback
                self.num_classes = 2
                print(f"Using default num_classes: {self.num_classes}")

        return model

    def _create_dataset_and_loader(self):
        # Use step_size instead of overlap (step_size is [0-1] representing stride as fraction of patch size)
        # step_size of 0.5 means 50% overlap
        self.dataset = VCDataset(
            input_path=self.input,
            patch_size=self.patch_size,
            step_size=self.overlap,
            num_parts=self.num_parts,
            part_id=self.part_id,
            cache_pool=self.cache_pool,
            normalization_scheme=self.normalization_scheme,
            input_format=self.input_format,
            verbose=self.verbose,
            mode='infer',
            # Pass skip_empty_patches flag
            skip_empty_patches=self.skip_empty_patches,
            # Pass Volume-specific parameters
            scroll_id=self.scroll_id,
            segment_id=self.segment_id,
            energy=self.energy,
            resolution=self.resolution
        )

        # Retrieve the calculated patch coordinates from the dataset instance
        # Look for 'all_positions' instead of 'patch_start_coords'
        expected_attr_name = 'all_positions'
        if not hasattr(self.dataset, expected_attr_name) or getattr(self.dataset, expected_attr_name) is None:
            raise AttributeError(f"The VCDataset instance must calculate and provide an "
                                 f"'{expected_attr_name}' attribute (list of coordinate tuples).")

        # Assign from 'all_positions'
        self.patch_start_coords_list = getattr(self.dataset, expected_attr_name)
        # ------------------------

        # Now use the length of the coordinates list to define the total patches
        self.num_total_patches = len(self.patch_start_coords_list)

        # Optional check: Make sure dataset __len__ matches coordinate list length
        if len(self.dataset) != self.num_total_patches:
            print(f"Warning: Dataset __len__ ({len(self.dataset)}) mismatch with "
                  f"{expected_attr_name} length ({self.num_total_patches}). Using {expected_attr_name} list length.")

        if self.num_total_patches == 0:
            raise RuntimeError(
                f"Dataset for part {self.part_id}/{self.num_parts} is empty (based on calculated coordinates in '{expected_attr_name}'). Check input data and partitioning.")

        if self.verbose:
            print(f"Total patches to process for part {self.part_id}: {self.num_total_patches}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            pin_memory=True if self.device != torch.device('cpu') else False,
            collate_fn=VCDataset.collate_fn  # Use the custom collate function that skips empty patches
        )
        return self.dataset, self.dataloader
        
    def _write_zattrs(self, zarr_path, attributes):
        """Helper method to write custom attributes to a .zattrs file in a Zarr store."""
        zattrs_path = os.path.join(zarr_path, '.zattrs')
        
        # Read existing .zattrs if it exists
        existing_data = {}
        if os.path.exists(zattrs_path):
            try:
                with open(zattrs_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"Warning: Could not parse existing .zattrs at {zattrs_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error reading {zattrs_path}: {e}")
        
        # Merge existing data with new attributes
        # For simplicity, we use top-level merging
        merged_data = {**existing_data, **attributes}
        
        # Write back the merged data
        try:
            with open(zattrs_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error writing to {zattrs_path}: {e}")
            return False

    async def _create_output_stores(self):
        """Creates the main output Zarr and the coordinate Zarr."""
        if self.num_classes is None or self.patch_size is None or self.num_total_patches is None:
            raise RuntimeError("Cannot create output stores: model/patch info missing.")
        if not self.patch_start_coords_list:
            raise RuntimeError("Cannot create output stores: patch coordinates not available.")

        # --- 1. Main Output Store ---
        output_shape = (self.num_total_patches, self.num_classes, *self.patch_size)
        output_chunks = (1, self.num_classes, *self.patch_size)
        main_store_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")  # Give unique name per part
        main_store_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': main_store_path},
            'metadata': {
                'dtype': '<f2',
                'shape': output_shape,
                'chunks': output_chunks,
                'compressor': {'id': 'blosc'},
            },
            'create': True, 'delete_existing': True
        }
        
        # Store custom attributes in a separate zattrs file instead of in the metadata
        if self.verbose: print(f"Will store custom attributes in .zattrs file after store creation")
        if self.verbose: print(f"Creating main output Zarr: {main_store_path}")

        # --- 2. Coordinate Store ---
        self.coords_store_path = os.path.join(self.output_dir, f"coordinates_part_{self.part_id}.zarr")
        coord_shape = (self.num_total_patches, len(self.patch_size))  # (N, 3) for 3D
        coord_chunks = (min(self.num_total_patches, 4096), len(self.patch_size))  # Chunk along patches dim
        coord_store_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': self.coords_store_path},
            'metadata': {
                'dtype': '<i4',  # Using 32-bit integer instead of int64
                'shape': coord_shape,
                'chunks': coord_chunks,
                'compressor': {'id': 'blosc'},
            },
            'create': True, 'delete_existing': True
        }
        if self.verbose: print(f"Creating coordinate Zarr: {self.coords_store_path}")

        cache_context = ts.Context({'cache_pool': {'total_bytes_limit': int(self.cache_pool)}})
        self.output_store = await ts.open(main_store_spec, context=cache_context)
        coords_store = await ts.open(coord_store_spec, context=cache_context)

        # --- Write custom attributes to zattrs files ---
        try:
            # Get original volume shape from dataset.input_shape
            original_volume_shape = None
            if hasattr(self.dataset, 'input_shape'):
                # Dataset input_shape could be (C, Z, Y, X) or (Z, Y, X)
                if len(self.dataset.input_shape) == 4:  # has channel dimension
                    original_volume_shape = list(self.dataset.input_shape[1:])  # Skip channel dimension
                else:  # no channel dimension
                    original_volume_shape = list(self.dataset.input_shape)
                if self.verbose:
                    print(f"Derived original volume shape from dataset.input_shape: {original_volume_shape}")
            
            if not original_volume_shape:
                print("Warning: Could not determine original volume shape from dataset")
            
            # Write custom attributes to main store .zattrs
            custom_attrs = {
                'patch_size': list(self.patch_size),
                'overlap': self.overlap,
                'part_id': self.part_id,
                'num_parts': self.num_parts,
            }
            
            # Add original volume shape if available (required by merge_outputs.py)
            if original_volume_shape:
                custom_attrs['original_volume_shape'] = original_volume_shape
            
            self._write_zattrs(main_store_path, custom_attrs)
            
            # Write custom attributes to coords store .zattrs 
            coords_attrs = {
                'part_id': self.part_id,
                'num_parts': self.num_parts,
            }
            self._write_zattrs(self.coords_store_path, coords_attrs)
            
            if self.verbose: print("Custom attributes written to .zattrs files")
        except Exception as e:
            print(f"Warning: Failed to write custom attributes: {e}")

        # --- Write Coordinates (all at once) ---
        coords_np = np.array(self.patch_start_coords_list, dtype=np.int32)  # Using int32 instead of int64
        if coords_np.shape != coord_shape:
            raise ValueError(f"Shape mismatch for coordinates. Expected {coord_shape}, got {coords_np.shape}")
        if self.verbose: print(f"Writing {self.num_total_patches} coordinates...")
        await coords_store.write(coords_np)
        # We don't need to keep the coords_store open after writing
        if self.verbose: print("Coordinates written successfully.")

        if self.verbose: print("Output stores opened successfully.")

    async def _process_batches(self):
        """Processes batches, performs inference (with inline TTA), and writes results asynchronously."""
        write_futures = []  # Track all write futures to await later
        self.current_patch_write_index = 0
        
        # Create a semaphore to limit concurrent write operations
        max_concurrent_writes = 16  # Adjust based on system capabilities
        semaphore = asyncio.Semaphore(max_concurrent_writes)
        
        # Always show progress bar, regardless of verbose mode
        pbar = tqdm(total=self.num_total_patches, desc=f"Inferring Part {self.part_id}")
        
        async def limited_write(slice_idx, data):
            """Write a patch with concurrency limiting"""
            async with semaphore:  # This limits concurrent writes
                future = self.output_store[slice_idx].write(data)  # Start the write
                return await future  # Wait for this specific write to complete

        for batch_data in self.dataloader:
            # Check if batch is empty (all patches were skipped)
            if isinstance(batch_data, dict) and batch_data.get('empty_batch', False):
                if self.verbose:
                    print("Skipping empty batch (all patches were homogeneous/empty)")
                continue  # Skip this batch entirely

            # Adapt data loading based on dataset output
            if isinstance(batch_data, (list, tuple)):
                input_batch = batch_data[0].to(self.device)  # Assuming first element is image
            elif isinstance(batch_data, dict):
                input_batch = batch_data['data'].to(self.device)  # Assuming key 'data'
            else:
                input_batch = batch_data.to(self.device)  # Assuming it's the tensor itself

            # Extra safety check: ensure we have a valid batch with data
            if input_batch is None or input_batch.shape[0] == 0:
                if self.verbose:
                    print("Skipping batch with no valid data")
                continue  # Skip this batch

            current_batch_size = input_batch.shape[0]

            with torch.no_grad(), torch.amp.autocast('cuda'):
                if self.do_tta:
                    # --- TTA ---
                    outputs_batch_tta = []  # Store list of outputs for each TTA for the batch

                    if self.tta_type == 'mirroring':
                        # Apply model to original and mirrored versions
                        m0 = self.model(input_batch)
                        m1 = self.model(torch.flip(input_batch, dims=[-1]))
                        m2 = self.model(torch.flip(input_batch, dims=[-2]))
                        m3 = self.model(torch.flip(input_batch, dims=[-3]))
                        m4 = self.model(torch.flip(input_batch, dims=[-1, -2]))
                        m5 = self.model(torch.flip(input_batch, dims=[-1, -3]))
                        m6 = self.model(torch.flip(input_batch, dims=[-2, -3]))
                        m7 = self.model(torch.flip(input_batch, dims=[-1, -2, -3]))

                        # Reverse the flips on the outputs before averaging
                        # Shape of each mi is (B, C, Z, Y, X)
                        outputs_batch_tta = [
                            m0,
                            torch.flip(m1, dims=[-1]),
                            torch.flip(m2, dims=[-2]),
                            torch.flip(m3, dims=[-3]),
                            torch.flip(m4, dims=[-1, -2]),
                            torch.flip(m5, dims=[-1, -3]),
                            torch.flip(m6, dims=[-2, -3]),
                            torch.flip(m7, dims=[-1, -2, -3])
                        ]

                    elif self.tta_type == 'rotation':
                        # Apply model to original and rotated versions (XY plane)
                        r0 = self.model(input_batch)
                        r1 = self.model(torch.rot90(input_batch, k=1, dims=(-2, -1)))  # 90 deg
                        r2 = self.model(torch.rot90(input_batch, k=2, dims=(-2, -1)))  # 180 deg
                        r3 = self.model(torch.rot90(input_batch, k=3, dims=(-2, -1)))  # 270 deg

                        # Rotate outputs back before averaging
                        # Shape of each ri is (B, C, Z, Y, X)
                        outputs_batch_tta = [
                            r0,
                            torch.rot90(r1, k=-1, dims=(-2, -1)),  # -90 deg
                            torch.rot90(r2, k=-2, dims=(-2, -1)),  # -180 deg
                            torch.rot90(r3, k=-3, dims=(-2, -1))   # -270 deg
                        ]

                    # --- Merge TTA results for the batch ---
                    # Stack along a new dimension (e.g., dim 0) -> (num_tta, B, C, Z, Y, X)
                    stacked_outputs = torch.stack(outputs_batch_tta, dim=0)
                    # Calculate the mean across the TTA dimension (dim 0)
                    output_batch = torch.mean(stacked_outputs, dim=0)  # Result shape: (B, C, Z, Y, X)

                else:
                    # --- No TTA ---
                    output_batch = self.model(input_batch)  # B, C, Z, Y, X
                    pbar.refresh()

            # Move output to CPU and convert to NumPy for TensorStore writing
            output_np = output_batch.cpu().numpy().astype(np.float16)  # B, C, Z, Y, X
            current_batch_size = output_np.shape[0]

            # Get patch indices from the batch data (needed when skipping empty patches)
            patch_indices = batch_data.get('index', [i for i in range(current_batch_size)])
            
            # Process each patch in the batch
            for i in range(current_batch_size):
                patch_data = output_np[i]  # Shape: (C, Z, Y, X)
                
                # Use the original dataset index as the write index to maintain correspondence with coordinates
                write_index = patch_indices[i]
                
                # Schedule the limited write operation (will be executed concurrently)
                # This creates a task that we can gather later
                write_future = asyncio.create_task(
                    limited_write(write_index, patch_data)
                )
                write_futures.append(write_future)
                
                self.current_patch_write_index += 1
                pbar.update(1)
                
            # Periodically clean up completed futures to avoid unbounded memory growth
            if len(write_futures) > max_concurrent_writes * 3:
                done, write_futures = await asyncio.wait(
                    write_futures, 
                    return_when=asyncio.FIRST_COMPLETED,
                )
                write_futures = list(write_futures)  # Convert back from set
                
                if self.verbose and done:
                    print(f"Cleaned up {len(done)} completed writes, {len(write_futures)} pending")

        # Wait for all remaining writes to complete
        if write_futures:
            if self.verbose:
                print(f"Waiting for {len(write_futures)} remaining writes to complete...")
            await asyncio.gather(*write_futures)
            
        pbar.close()

        if self.verbose:
            print(f"Finished writing {self.current_patch_write_index} non-empty patches.")
        
        # With skip_empty_patches, we expect fewer patches to be processed
        if not self.skip_empty_patches and self.current_patch_write_index != self.num_total_patches:
             print(f"Warning: Expected {self.num_total_patches} patches, but wrote {self.current_patch_write_index}.")

    async def _run_inference_async(self):
        """Asynchronous orchestration function."""
        if self.verbose: print("Loading model...")
        self.model = self._load_model()

        if self.verbose: print("Creating dataset and dataloader...")
        self._create_dataset_and_loader() # This now gets coordinates

        if self.num_total_patches > 0:
            if self.verbose: print("Creating output stores (logits and coordinates)...")
            await self._create_output_stores() # Create both stores

            if self.verbose: print("Starting inference and writing logits...")
            await self._process_batches() # Process and write only logits
        else:
             print(f"Skipping processing for part {self.part_id} as no patches were found.")

        if self.verbose: print("Inference complete.")


    def infer(self):
        """Public method to start the inference process."""
        try:

            context = ts.Context({
                'cache_pool': {
                    'total_bytes_limit': int(self.cache_pool)
                }
            })

            asyncio.run(self._run_inference_async())
            main_output_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
            return main_output_path, self.coords_store_path
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback


def main():
    """Entry point for the vesuvius.predict command line tool."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Run nnUNet inference on Zarr data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the nnUNet model folder')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to store output predictions')
    parser.add_argument('--input_format', type=str, default='zarr', help='Input format (zarr, volume)')
    parser.add_argument('--tta_type', type=str, default='mirroring', choices=['mirroring', 'rotation'], 
                      help='TTA type (mirroring or rotation)')
    parser.add_argument('--disable_tta', action='store_true', help='Disable test time augmentation')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, help='Part ID to process (0-indexed)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between patches (0-1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--patch_size', type=str, default=None, 
                      help='Optional: Override patch size, comma-separated (e.g., "192,192,192"). If not provided, uses the model\'s default patch size.')
    parser.add_argument('--save_softmax', action='store_true', help='Save softmax outputs')
    parser.add_argument('--cache_pool', type=float, default=1e10, help='TensorStore cache pool size in bytes')
    parser.add_argument('--normalization', type=str, default='instance_zscore', 
                      help='Normalization scheme (instance_zscore, global_zscore, instance_minmax, none)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--skip-empty-patches', dest='skip_empty_patches', action='store_true', 
                      help='Skip patches that are empty (all values the same). Default: True')
    parser.add_argument('--no-skip-empty-patches', dest='skip_empty_patches', action='store_false',
                      help='Process all patches, even if they appear empty')
    parser.set_defaults(skip_empty_patches=True)
    
    # Add arguments for the updated Volume class
    parser.add_argument('--scroll_id', type=str, default=None, help='Scroll ID to use (if input_format is volume)')
    parser.add_argument('--segment_id', type=str, default=None, help='Segment ID to use (if input_format is volume)')
    parser.add_argument('--energy', type=int, default=None, help='Energy level to use (if input_format is volume)')
    parser.add_argument('--resolution', type=float, default=None, help='Resolution to use (if input_format is volume)')
    
    # Add arguments for Hugging Face model loading
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for accessing private repositories')
    
    args = parser.parse_args()
    
    # Parse optional patch size if provided
    patch_size = None
    if args.patch_size:
        try:
            patch_size = tuple(map(int, args.patch_size.split(',')))
            print(f"Using user-specified patch size: {patch_size}")
        except Exception as e:
            print(f"Error parsing patch_size: {e}")
            print("Expected format: comma-separated integers, e.g. '192,192,192'")
            print("Using model's default patch size instead.")
    
    # Convert scroll_id and segment_id if needed
    scroll_id = args.scroll_id
    segment_id = args.segment_id
    
    if scroll_id is not None and scroll_id.isdigit():
        scroll_id = int(scroll_id)
    
    if segment_id is not None and segment_id.isdigit():
        segment_id = int(segment_id)
    
    print("\n--- Initializing Inferer ---")
    inferer = Inferer(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_format=args.input_format,
        tta_type=args.tta_type,
        do_tta=not args.disable_tta,
        num_parts=args.num_parts,
        part_id=args.part_id,
        overlap=args.overlap,
        batch_size=args.batch_size,
        patch_size=patch_size,  # Will use model's patch size if None
        save_softmax=args.save_softmax,
        cache_pool=args.cache_pool,
        normalization_scheme=args.normalization,
        device=args.device,
        verbose=args.verbose,
        skip_empty_patches=args.skip_empty_patches,  # Skip empty patches flag
        # Pass Volume-specific parameters to VCDataset
        scroll_id=scroll_id,
        segment_id=segment_id,
        energy=args.energy,
        resolution=args.resolution,
        # Pass Hugging Face parameters
        hf_token=args.hf_token
    )

    try:
        print("\n--- Starting Inference ---")
        logits_path, coords_path = inferer.infer()

        if logits_path and coords_path and os.path.exists(logits_path) and os.path.exists(coords_path):
            print(f"\n--- Inference Finished ---")
            print(f"Output logits saved to: {logits_path}")

            print("\n--- Inspecting Output Store ---")
            try:
                 # Open the store directly
                 output_store = ts.open({
                     'driver': 'zarr', 
                     'kvstore': {'driver': 'file', 'path': logits_path}
                 }).result()
                 print(f"Output shape: {output_store.shape}")
                 print(f"Output dtype: {output_store.dtype}")
                 print(f"Output chunks: {output_store.chunk_layout.read_chunk.shape}")
            except Exception as inspect_e:
                print(f"Could not inspect output Zarr: {inspect_e}")
                
            # Print empty patches report if skip_empty_patches was enabled
            if inferer.skip_empty_patches and hasattr(inferer.dataset, 'get_empty_patches_report'):
                report = inferer.dataset.get_empty_patches_report()
                print("\n--- Empty Patches Report ---")
                print(f"  Empty Patches Skipped: {report['total_skipped']}")
                print(f"  Total Available Positions: {report['total_positions']}")
                if report['total_skipped'] > 0:
                    print(f"  Skip Ratio: {report['skip_ratio']:.2%}")
                    print(f"  Effective Speedup: {1/(1-report['skip_ratio']):.2f}x")

            print("\n--- Inspecting Coordinate Store ---")
            try:
                coords_store = ts.open({'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': coords_path}}).result()
                print(f"Coords shape: {coords_store.shape}")
                print(f"Coords dtype: {coords_store.dtype}")
                first_few_coords = coords_store[0:5].read().result()
                print(f"First few coordinates:\n{first_few_coords}")
            except Exception as inspect_e:
                print(f"Could not inspect coordinate Zarr: {inspect_e}")
            return 0
        else:
             print("\n--- Inference finished, but output path seems invalid or wasn't created. ---")
             return 1

    except Exception as main_e:
        print(f"\n--- Inference Failed ---")
        print(f"Error: {main_e}")
        import traceback
        traceback.print_exc()
        return 1

# --- Command line usage ---
if __name__ == '__main__':
    import sys
    sys.exit(main())