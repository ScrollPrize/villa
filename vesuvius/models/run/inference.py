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
import gc
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
                 num_dataloader_workers: int = 2,
                 num_writer_workers: int = 16,  # Number of parallel writer workers
                 verbose: bool = False,
                 skip_empty_patches: bool = False,  # Skip empty/homogeneous patches
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
        self.num_writer_workers = num_writer_workers  # Store number of writer workers
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

        # Check model patch size if using rotation TTA
        if self.do_tta and self.tta_type == 'rotation':
            if len(self.model_patch_size) != 3:
                raise ValueError(f"Rotation TTA requires 3D model patch size, got {self.model_patch_size}.")
            if len(set(self.model_patch_size)) > 1:
                raise ValueError(f"Rotation TTA requires equal dimensions in all axes, model patch size is {self.model_patch_size}. "
                                f"Please use a model with equal patch dimensions or disable rotation TTA.")

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
            # Check if all dimensions have the same size (required for rotation TTA)
            if len(set(self.patch_size)) > 1:
                raise ValueError(f"Rotation TTA requires equal dimensions in all axes, got {self.patch_size}. " 
                                 f"Please use a model with equal patch dimensions or disable rotation TTA.")

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
            collate_fn=VCDataset.collate_fn,  # Use the custom collate function that skips empty patches
            prefetch_factor=2,  # Prefetch 2 batches per worker for better I/O overlap
            drop_last=False  # Make sure to process all data
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

        # Optimize chunking strategy - use larger chunks for the patches dimension
        # to improve throughput for multi-patch batches
        batch_size_for_chunks = min(self.batch_size * 2, 16)  # Use larger chunks but don't go too large
        output_chunks = (max(1, batch_size_for_chunks), self.num_classes, *self.patch_size)

        main_store_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")  # Give unique name per part
        main_store_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': main_store_path},
            'metadata': {
                'dtype': '<f2',  # float16
                'shape': output_shape,
                'chunks': output_chunks,
                'compressor': {
                    'id': 'blosc',
                    'cname': 'lz4',  # Use faster LZ4 compression
                    'clevel': 5,     # Balanced compression level
                    'shuffle': 1     # Enable shuffle filter
                },
                'order': 'C',  # Row-major order (better for our write pattern)
                'fill_value': 0,    # Initialize with zeros
            },
            'create': True,
            'delete_existing': True
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

        # Create a context with optimized cache and thread pool settings
        cache_context = ts.Context({
            'cache_pool': {'total_bytes_limit': int(self.cache_pool)},
            'file_io_concurrency': {'limit': 32},  # Increase file I/O concurrency
            'data_copy_concurrency': {'limit': 16}  # Increase data copy concurrency
        })
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
        """Processes batches using a producer-consumer pattern with dedicated writer workers."""
        import queue
        import threading
        import time

        # Create a shared queue for patches with controlled size
        max_queue_size = 64  # Maximum number of items in the queue (reduced to 128 patches)
        max_pending_threshold = max_queue_size * 0.8  # Threshold to pause inference (80% capacity = ~102)
        resume_threshold = max_queue_size * 0.5  # Resume inference when below 50% capacity (64)

        patch_queue = queue.Queue(maxsize=max_queue_size)
        stop_event = threading.Event()
        write_count = 0
        pending_count = 0  # Track number of pending writes - patches "in flight"
        max_pending_ever = 0  # Track maximum pending count ever reached
        lock = threading.Lock()

        # Use atomic counter to track queue size more accurately
        def get_queue_size():
            """Get accurate queue size atomically"""
            with lock:
                return pending_count

        # Track progress
        self.current_patch_write_index = 0
        pbar = tqdm(total=self.num_total_patches, desc=f"Inferring Part {self.part_id}")
        pbar.set_postfix({"pending": 0})

        # Define worker function to consume from queue and write patches
        def writer_worker():
            nonlocal write_count, pending_count, max_pending_ever  # Declare nonlocal at function start
            worker_count = 0
            while not stop_event.is_set() or not patch_queue.empty():
                try:
                    # Get item with timeout to allow checking stop_event periodically
                    item = patch_queue.get(timeout=0.5)
                    if item is None:  # Sentinel value
                        patch_queue.task_done()
                        continue

                    index, patch_data = item

                    try:
                        # Write patch to TensorStore (blocking operation, but in worker thread)
                        self.output_store[index].write(patch_data).result()
                        
                        # Explicitly delete the patch data after writing to free memory
                        del patch_data
                        
                        # Update counters - use minimal lock time
                        with lock:
                            write_count += 1
                            pending_count -= 1  # Decrement pending count after successful write
                            current_pending = pending_count  # Capture value while under lock

                        # Update progress bar outside of lock
                        pbar.set_postfix({"pending": current_pending}, refresh=True)
                        pbar.update(1)

                        # Periodically run garbage collection in the worker thread
                        worker_count += 1
                        if worker_count % 10 == 0:  # Every 10 patches
                            gc.collect()  # Clean up any lingering references
                            
                    except Exception as e:
                        # Handle patch writing errors
                        print(f"Error writing patch at index {index}: {e}")
                        # Still need to decrement pending count and mark task done
                        with lock:
                            pending_count -= 1
                            current_pending = pending_count
                        pbar.set_postfix({"pending": current_pending}, refresh=True)
                        
                    finally:
                        # Always mark the task as done, even if there was an error
                        patch_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    # Handle queue-level errors
                    print(f"Critical error in writer worker: {e}")
                    with lock:
                        pending_count -= 1  # Make sure we decrement on error
                        current_pending = pending_count
                    pbar.set_postfix({"pending": current_pending}, refresh=True)
                    try:
                        patch_queue.task_done()  # Try to mark task done
                    except:
                        pass  # Ignore if this fails

            if self.verbose:
                print(f"Writer worker completed after processing {worker_count} patches")

        # Create and start writer worker threads
        writer_threads = []
        for i in range(self.num_writer_workers):
            thread = threading.Thread(target=writer_worker, daemon=True)
            thread.start()
            writer_threads.append(thread)
            if self.verbose:
                print(f"Started writer worker {i+1}")

        # Process batches and add to queue (producer part)
        try:
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

                # Run inference with mixed precision
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    if self.do_tta:
                        # --- TTA ---
                        if self.tta_type == 'mirroring':
                            # Process TTA sequentially to save memory
                            # First, get the base prediction
                            output_batch = self.model(input_batch)
                            
                            # Process each TTA variant one at a time, accumulating the result
                            # Flip 1: X-axis
                            flipped_input = torch.flip(input_batch, dims=[-1])
                            m1 = self.model(flipped_input)
                            output_batch += torch.flip(m1, dims=[-1])
                            del m1, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 2: Y-axis
                            flipped_input = torch.flip(input_batch, dims=[-2])
                            m2 = self.model(flipped_input)
                            output_batch += torch.flip(m2, dims=[-2])
                            del m2, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 3: Z-axis
                            flipped_input = torch.flip(input_batch, dims=[-3])
                            m3 = self.model(flipped_input)
                            output_batch += torch.flip(m3, dims=[-3])
                            del m3, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 4: X,Y axes
                            flipped_input = torch.flip(input_batch, dims=[-1, -2])
                            m4 = self.model(flipped_input)
                            output_batch += torch.flip(m4, dims=[-1, -2])
                            del m4, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 5: X,Z axes
                            flipped_input = torch.flip(input_batch, dims=[-1, -3])
                            m5 = self.model(flipped_input)
                            output_batch += torch.flip(m5, dims=[-1, -3])
                            del m5, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 6: Y,Z axes
                            flipped_input = torch.flip(input_batch, dims=[-2, -3])
                            m6 = self.model(flipped_input)
                            output_batch += torch.flip(m6, dims=[-2, -3])
                            del m6, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Flip 7: X,Y,Z axes
                            flipped_input = torch.flip(input_batch, dims=[-1, -2, -3])
                            m7 = self.model(flipped_input)
                            output_batch += torch.flip(m7, dims=[-1, -2, -3])
                            del m7, flipped_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Calculate the mean by dividing by 8 (total number of TTA variants)
                            output_batch /= 8

                        elif self.tta_type == 'rotation':
                            # Process TTA sequentially for rotations too
                            # Base prediction (0 degrees)
                            output_batch = self.model(input_batch)
                            
                            # 90 degrees
                            rotated_input = torch.rot90(input_batch, k=1, dims=(-2, -1))
                            r1 = self.model(rotated_input)
                            output_batch += torch.rot90(r1, k=-1, dims=(-2, -1))
                            del r1, rotated_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # 180 degrees
                            rotated_input = torch.rot90(input_batch, k=2, dims=(-2, -1))
                            r2 = self.model(rotated_input)
                            output_batch += torch.rot90(r2, k=-2, dims=(-2, -1))
                            del r2, rotated_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # 270 degrees
                            rotated_input = torch.rot90(input_batch, k=3, dims=(-2, -1))
                            r3 = self.model(rotated_input)
                            output_batch += torch.rot90(r3, k=-3, dims=(-2, -1))
                            del r3, rotated_input
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Calculate the mean by dividing by 4 (total number of rotation variants)
                            output_batch /= 4

                    else:
                        # --- No TTA ---
                        output_batch = self.model(input_batch)  # B, C, Z, Y, X

                # Get patch indices
                patch_indices = batch_data.get('index', list(range(output_batch.shape[0])))
                batch_size = output_batch.shape[0]
                
                # First check if we need to pause inference before allocating more memory
                need_to_pause = False
                with lock:
                    if pending_count >= max_pending_threshold:
                        need_to_pause = True
                        current_pending = pending_count

                # If needed, pause until queue drains sufficiently - outside lock
                if need_to_pause:
                    if self.verbose:
                        print(f"Pausing inference, {current_pending} items pending in queue...")

                    # Force garbage collection to help clear memory
                    gc.collect()
                    torch.cuda.empty_cache()

                    paused = True
                    while paused:
                        time.sleep(0.5)  # Check every half second
                        with lock:
                            if pending_count < resume_threshold:
                                paused = False
                                current_pending = pending_count

                    if self.verbose:
                        print(f"Resuming inference, queue now at {current_pending} items")

                # Force garbage collection more aggressively
                gc.collect()
                torch.cuda.empty_cache()

                # Process each item in the batch individually to avoid holding all in memory
                for i in range(batch_size):
                    # Extract single item, move to CPU and convert to NumPy - one at a time
                    # This reduces peak memory usage by processing one patch at a time
                    single_output = output_batch[i:i+1]  # Keep batch dimension for now
                    
                    # Move to CPU and convert to float16 to save memory
                    patch_data = single_output.cpu().numpy().astype(np.float16).squeeze(0)  # Remove batch dim
                    # Explicitly delete the tensor after CPU conversion
                    del single_output
                    
                    # Get the write index
                    write_index = patch_indices[i]
                    
                    # Update our tracking counter before adding to queue
                    with lock:
                        # Pre-increment the pending count to reflect what we're about to queue
                        pending_count += 1
                        # Track max pending ever reached
                        max_pending_ever = max(max_pending_ever, pending_count)
                        current_pending = pending_count  # Capture for display
                    
                    # Add to queue - this operation is threadsafe
                    patch_queue.put((write_index, patch_data), block=True)
                
                # Clear the main output batch to free memory
                del output_batch
                # Delete input batch to prevent memory leak
                del input_batch
                
                # More aggressive memory cleanup
                gc.collect()
                torch.cuda.empty_cache()

                # Update global tracking counter after successful queueing
                self.current_patch_write_index += batch_size

                # Update progress bar outside of lock - show only pending count
                pbar.set_postfix({"pending": current_pending}, refresh=True)

            # Signal workers to finish - once all batches are processed
            if self.verbose:
                print("All batches processed, waiting for writer workers to complete...")

            # Wait for all queued items to be processed
            patch_queue.join()

        finally:
            # Signal all workers to stop
            stop_event.set()

            # Add sentinel values to ensure workers exit
            for _ in range(self.num_writer_workers):
                try:
                    patch_queue.put(None, block=False)
                except queue.Full:
                    pass

            # Wait for all writer threads to finish
            for thread in writer_threads:
                thread.join(timeout=5.0)  # Give threads time to finish

            pbar.close()

        # Final statistics
        if self.verbose:
            print(f"Finished writing {write_count} non-empty patches.")
            print(f"Maximum pending writes reached: {max_pending_ever} (useful for tuning)")
            print(f"Queue config: max={max_queue_size}, pause_at={max_pending_threshold:.0f}, resume_at={resume_threshold:.0f}")

        # With skip_empty_patches, we expect fewer patches to be processed
        if not self.skip_empty_patches and write_count != self.num_total_patches:
            print(f"Warning: Expected {self.num_total_patches} patches, but wrote {write_count}.")

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

            # Create optimized context for inference
            context = ts.Context({
                'cache_pool': {'total_bytes_limit': int(self.cache_pool)},
                'file_io_concurrency': {'limit': 32},  # Increase file I/O concurrency
                'data_copy_concurrency': {'limit': 16}  # Increase data copy concurrency
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
    parser.add_argument('--num_dataloader_workers', type=int, default=4,
                      help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_writer_workers', type=int, default=6,
                      help='Number of parallel writer threads for I/O (default: 6)')
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
        num_dataloader_workers=args.num_dataloader_workers,
        num_writer_workers=args.num_writer_workers,
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