import torch
import numpy as np
import zarr
import os
import json
import multiprocessing
import threading
import fsspec
import numcodecs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from threading import Thread
# fork causes issues on windows and w/ tensorstore , force to spawn
multiprocessing.set_start_method('spawn', force=True)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.models.load_nnunet_model import load_model_for_inference
from data.vc_dataset import VCDataset
from data.utils import open_zarr


def generate_hann_map(
        patch_size: tuple,
        verbose: bool = False
) -> np.ndarray:
    """
    Generates a 3-D Hann (raised-cosine) importance map of shape
    (1, pZ, pY, pX).  With 50 % patch overlap the window forms an exact
    partition of unity, so a single global normalisation constant
    suffices during accumulation.

    Parameters
    ----------
    patch_size : (int, int, int)
        The (Z, Y, X) size of the patch to be windowed.
    verbose : bool, optional
        If True, print basic stats.

    Returns
    -------
    np.ndarray
        Importance map with dtype float32, values ∈ [0, 1].
        Shape: (1, pZ, pY, pX) — ready for broadcasting.
    """
    pZ, pY, pX = patch_size

    # --- 1-D Hann helper ---------------------------------------------------
    def hann_1d(L: int) -> np.ndarray:
        if L == 1:                               # degenerate edge case
            return np.ones(1, dtype=np.float32)
        n = np.arange(L, dtype=np.float32)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (L - 1)))

    wz = hann_1d(pZ)[:, None, None]             # (Z,1,1)
    wy = hann_1d(pY)[None, :, None]             # (1,Y,1)
    wx = hann_1d(pX)[None, None, :]             # (1,1,X)

    # --- separable outer-product ------------------------------------------
    hann_map = (wz * wy * wx).astype(np.float32)   # (Z,Y,X)

    # --- normalise so max = 1 ------------------------------------------------
    hann_map /= hann_map.max()  # already 1, but safes against all-zero

    hann_map = hann_map.reshape(1, pZ, pY, pX)     # (1,Z,Y,X)

    if verbose:
        print(f"Hann map: shape {hann_map.shape}, "
              f"min {hann_map.min():.4f}, max {hann_map.max():.4f}")

    return hann_map


class Inferer():
    def __init__(self,
                 model_path: str = None,
                 input_dir: str = None,
                 output_dir: str = None,
                 input_format: str = 'zarr',
                 tta_type: str = 'rotation', # 'mirroring' or 'rotation'
                 # tta_combinations: int = 3,
                 # tta_rotation_weights: [list, tuple] = (1, 1, 1),
                 do_tta: bool = True,
                 num_parts: int = 1,
                 part_id: int = 0,
                 overlap: float = 0.5,
                 batch_size: int = 1,
                 patch_size: [list, tuple] = None,
                 save_softmax: bool = False,
                 normalization_scheme: str = 'instance_zscore',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_dataloader_workers: int = 4,
                 verbose: bool = False,
                 skip_empty_patches: bool = True,  # Skip empty/homogeneous patches
                 # parmas to get passed to Volume 
                 scroll_id: [str, int] = None,
                 segment_id: [str, int] = None,
                 energy: int = None,
                 resolution: float = None,
                 compressor_name: str = 'zstd',
                 compression_level: int = 1,
                 hf_token: str = None
                 ):
        print(f"Initializing Inferer with output_dir: '{output_dir}'")
        if output_dir and not output_dir.strip():
            raise ValueError("output_dir cannot be an empty string")

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
        self.verbose = verbose
        self.normalization_scheme = normalization_scheme
        self.input_format = input_format
        self.device = torch.device(device)
        self.num_dataloader_workers = num_dataloader_workers
        self.skip_empty_patches = skip_empty_patches
        self.scroll_id = scroll_id
        self.segment_id = segment_id
        self.energy = energy
        self.resolution = resolution
        self.compressor_name = compressor_name
        self.compression_level = compression_level
        self.hf_token = hf_token
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

        # --- Output Setup ---
        self._temp_dir_obj = None
        if output_dir:
            self.output_dir = output_dir
            
            # For S3 paths, use fsspec.filesystem.makedirs
            if self.output_dir.startswith('s3://'):
                fs = fsspec.filesystem('s3', anon=False)
                fs.makedirs(self.output_dir, exist_ok=True)
                print(f"Created S3 output directory: {self.output_dir}")
            else:
                # For local paths, use os.makedirs
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
        self.hann_map = None
        self.accumulator = None
        self.write_queue = None
        self.accumulation_worker = None


    def _load_model(self):
        # check if model_path is a Hugging Face model path (starts with "hf://")
        if isinstance(self.model_path, str) and self.model_path.startswith("hf://"):
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
        
        # Apply torch.compile for optimization if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            if self.verbose:
                print("Applying torch.compile optimization...")
            try:
                model = torch.compile(model)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")
        
        # patch size and number of classes from model_info
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

    def _create_hann_map(self):
        """Generate Hann weighting map for blending"""
        self.hann_map = generate_hann_map(
            self.patch_size, 
            verbose=self.verbose
        )[0]  # Remove batch dimension

    def _create_dataset_and_loader(self):
        # Use step_size instead of overlap (step_size is [0-1] representing stride as fraction of patch size)
        # step_size of 0.5 means 50% overlap
        self.dataset = VCDataset(
            input_path=self.input,
            patch_size=self.patch_size,
            step_size=self.overlap,
            num_parts=self.num_parts,
            part_id=self.part_id,
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

        expected_attr_name = 'all_positions'
        if not hasattr(self.dataset, expected_attr_name) or getattr(self.dataset, expected_attr_name) is None:
            raise AttributeError(f"The VCDataset instance must calculate and provide an "
                                 f"'{expected_attr_name}' attribute (list of coordinate tuples).")

        self.patch_start_coords_list = getattr(self.dataset, expected_attr_name)
        self.num_total_patches = len(self.patch_start_coords_list)

        # ensure dataset __len__ matches coordinate list length
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
            collate_fn=VCDataset.collate_fn  # we use custom collate fn here to tag patches that contain only zeros 
                                             # so we don't run them through the model 
        )
        return self.dataset, self.dataloader

    def _create_accumulator(self):
        """Create accumulator for direct accumulation instead of patch storage"""
        from models.run.partitioned_accumulator import PartitionedAccumulator
        
        # Create accumulator for this partition only
        self.accumulator = PartitionedAccumulator(
            dataset=self.dataset,
            num_classes=self.num_classes,
            hann_map=self.hann_map,
            output_path=os.path.join(self.output_dir, f"accumulator_part_{self.part_id}"),
            patch_size=self.patch_size
        )
        
    def _get_zarr_compressor(self):
        if self.compressor_name.lower() == 'zstd':
            return zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'lz4':
            return zarr.Blosc(cname='lz4', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'zlib':
            return zarr.Blosc(cname='zlib', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'none':
            return None
        else:
            return zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)


    def _start_accumulation_worker(self):
        """Start the background worker thread for async accumulation."""
        self.write_queue = Queue(maxsize=100)  # Limit queue size to control memory
        
        def accumulation_worker():
            while True:
                item = self.write_queue.get()
                if item is None:  # Sentinel value to stop
                    break
                patch_data, global_coords = item
                self.accumulator.accumulate_patch(patch_data, global_coords)
                self.write_queue.task_done()
        
        self.accumulation_worker = Thread(target=accumulation_worker)
        self.accumulation_worker.daemon = True
        self.accumulation_worker.start()
    
    def _stop_accumulation_worker(self):
        """Stop the accumulation worker and wait for pending writes."""
        if self.write_queue is not None:
            # Wait for all pending writes
            self.write_queue.join()
            # Send sentinel to stop worker
            self.write_queue.put(None)
            if self.accumulation_worker is not None:
                self.accumulation_worker.join()
    
    def _process_batches(self):
        self.current_patch_write_index = 0
        
        # Start the accumulation worker
        self._start_accumulation_worker()
        
        with tqdm(total=self.num_total_patches, desc=f"Inferring Part {self.part_id}") as pbar:
            for batch_data in self.dataloader:
                if isinstance(batch_data, (list, tuple)):
                    input_batch = batch_data[0].to(self.device)
                    is_empty_flags = [False] * input_batch.shape[0]
                elif isinstance(batch_data, dict):
                    input_batch = batch_data['data'].to(self.device)
                    is_empty_flags = batch_data.get('is_empty', [False] * input_batch.shape[0])
                else:
                    input_batch = batch_data.to(self.device)
                    is_empty_flags = [False] * input_batch.shape[0]
                
                # Skip invalid batches
                if input_batch is None or input_batch.shape[0] == 0:
                    if self.verbose:
                        print("Skipping batch with no valid data")
                    continue
                
                batch_size = input_batch.shape[0]
                output_shape = (batch_size, self.num_classes, *self.patch_size)
                output_batch = torch.zeros(output_shape, device=self.device, dtype=input_batch.dtype)
                
                # Find non-empty patches that need model inference
                non_empty_indices = [i for i, is_empty in enumerate(is_empty_flags) if not is_empty]
                
                # Only perform inference if there are non-empty patches
                if non_empty_indices:
                    non_empty_input = input_batch[non_empty_indices]
                    
                    # Perform inference with or without TTA
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        if self.do_tta:
                            # --- TTA with batched inference ---
                            if self.tta_type == 'mirroring':
                                # Batch all mirrored variants together
                                batch_variants = torch.cat([
                                    non_empty_input,
                                    torch.flip(non_empty_input, dims=[-1]),
                                    torch.flip(non_empty_input, dims=[-2]),
                                    torch.flip(non_empty_input, dims=[-3]),
                                    torch.flip(non_empty_input, dims=[-1, -2]),
                                    torch.flip(non_empty_input, dims=[-1, -3]),
                                    torch.flip(non_empty_input, dims=[-2, -3]),
                                    torch.flip(non_empty_input, dims=[-1, -2, -3])
                                ], dim=0)
                                
                                # Single model call for all variants
                                all_outputs = self.model(batch_variants)
                                
                                # Split outputs and reverse flips
                                batch_size = non_empty_input.shape[0]
                                outputs_batch_tta = [
                                    all_outputs[0*batch_size:1*batch_size],
                                    torch.flip(all_outputs[1*batch_size:2*batch_size], dims=[-1]),
                                    torch.flip(all_outputs[2*batch_size:3*batch_size], dims=[-2]),
                                    torch.flip(all_outputs[3*batch_size:4*batch_size], dims=[-3]),
                                    torch.flip(all_outputs[4*batch_size:5*batch_size], dims=[-1, -2]),
                                    torch.flip(all_outputs[5*batch_size:6*batch_size], dims=[-1, -3]),
                                    torch.flip(all_outputs[6*batch_size:7*batch_size], dims=[-2, -3]),
                                    torch.flip(all_outputs[7*batch_size:8*batch_size], dims=[-1, -2, -3])
                                ]

                            elif self.tta_type == 'rotation':
                                # Batch all rotation variants together
                                batch_variants = torch.cat([
                                    non_empty_input,
                                    torch.transpose(non_empty_input, -3, -1),  # X-up
                                    torch.transpose(non_empty_input, -3, -2)   # Z-up
                                ], dim=0)
                                
                                # Single model call for all variants
                                all_outputs = self.model(batch_variants)
                                
                                # Split and transform outputs back
                                batch_size = non_empty_input.shape[0]
                                outputs_batch_tta = [
                                    all_outputs[0*batch_size:1*batch_size],  # Original
                                    torch.transpose(all_outputs[1*batch_size:2*batch_size], -3, -1),  # X-up back
                                    torch.transpose(all_outputs[2*batch_size:3*batch_size], -3, -2)   # Z-up back
                                ]

                            # --- Merge TTA results for the batch ---
                            stacked_outputs = torch.stack(outputs_batch_tta, dim=0)
                            non_empty_output = torch.mean(stacked_outputs, dim=0)

                        else:
                            # --- No TTA ---
                            non_empty_output = self.model(non_empty_input) 
                    
                    # Place non-empty patch outputs in the correct positions in output_batch
                    for idx, original_idx in enumerate(non_empty_indices):
                        output_batch[original_idx] = non_empty_output[idx]
                
                else:
                    if self.verbose:
                        print("Batch contains only empty patches, skipping model inference")
                
                output_np = output_batch.cpu().numpy().astype(np.float32)
                current_batch_size = output_np.shape[0]
                
                patch_indices = batch_data.get('index', list(range(current_batch_size)))
                
                # Queue patches for async accumulation
                for i in range(current_batch_size):
                    patch_data = output_np[i]  # Shape: (C, Z, Y, X)
                    patch_index = patch_indices[i] if i < len(patch_indices) else i
                    
                    # Get global coordinates for this patch
                    global_coords = self.patch_start_coords_list[patch_index]
                    
                    # Queue for async accumulation
                    self.write_queue.put((patch_data.copy(), global_coords))
                    
                    pbar.update(1)
                    self.current_patch_write_index += 1
        
        # Stop the accumulation worker and wait for all writes to complete
        self._stop_accumulation_worker()
        
        if self.verbose:
            print(f"Finished processing {self.current_patch_write_index} patches with async accumulation.")
        
        # Verify completion and report
        if self.current_patch_write_index != self.num_total_patches:
            print(f"Warning: Expected {self.num_total_patches} patches, but processed {self.current_patch_write_index}.")

    def _run_inference(self):
        if self.verbose: print("Loading model...")
        self.model = self._load_model()

        if self.verbose: print("Creating Hann map...")
        self._create_hann_map()

        if self.verbose: print("Creating dataset and dataloader...")
        self._create_dataset_and_loader()

        if self.num_total_patches > 0:
            if self.verbose: print("Creating accumulator...")
            self._create_accumulator()

            if self.verbose: print("Starting inference with direct accumulation...")
            self._process_batches()
        else:
            print(f"Skipping processing for part {self.part_id} as no patches were found.")

        if self.verbose: print("Inference complete.")

    def infer(self):
        try:
            self._run_inference()
            accumulator_path = os.path.join(self.output_dir, f"accumulator_part_{self.part_id}")
            return accumulator_path
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc() 


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Run nnUNet inference on Zarr data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the nnUNet model folder')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to store output predictions')
    parser.add_argument('--input_format', type=str, default='zarr', help='Input format (zarr, volume)')
    parser.add_argument('--tta_type', type=str, default='rotation', choices=['mirroring', 'rotation'], 
                      help='TTA type (mirroring or rotation). Default: rotation')
    parser.add_argument('--disable_tta', action='store_true', help='Disable test time augmentation')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, help='Part ID to process (0-indexed)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between patches (0-1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--patch_size', type=str, default=None, 
                      help='Optional: Override patch size, comma-separated (e.g., "192,192,192"). If not provided, uses the model\'s default patch size.')
    parser.add_argument('--save_softmax', action='store_true', help='Save softmax outputs')
    parser.add_argument('--normalization', type=str, default='instance_zscore', 
                      help='Normalization scheme (instance_zscore, global_zscore, instance_minmax, none)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--skip-empty-patches', dest='skip_empty_patches', action='store_true', 
                      help='Skip patches that are empty (all values the same). Default: True')
    parser.add_argument('--no-skip-empty-patches', dest='skip_empty_patches', action='store_false',
                      help='Process all patches, even if they appear empty')
    parser.set_defaults(skip_empty_patches=True)
    
    # Add arguments for Zarr compression
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                      choices=['zstd', 'lz4', 'zlib', 'none'],
                      help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=1,
                      help='Compression level (1-9, higher = better compression but slower)')
    
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
        normalization_scheme=args.normalization,
        device=args.device,
        verbose=args.verbose,
        skip_empty_patches=args.skip_empty_patches,  # Skip empty patches flag
        # Pass Volume-specific parameters to VCDataset
        scroll_id=scroll_id,
        segment_id=segment_id,
        energy=args.energy,
        resolution=args.resolution,
        # Pass Zarr compression settings
        compressor_name=args.zarr_compressor,
        compression_level=args.zarr_compression_level,
        # Pass Hugging Face parameters
        hf_token=args.hf_token
    )

    try:
        print("\n--- Starting Inference ---")
        accumulator_path = inferer.infer()

        if accumulator_path:
            # Check if accumulator exists
            accumulator_exists = False
            
            try:
                logits_path = os.path.join(accumulator_path, 'logits')
                if accumulator_path.startswith('s3://'):
                    fs = fsspec.filesystem('s3', anon=False)
                    # Check if .zarray file exists within the zarr directory
                    accumulator_exists = fs.exists(os.path.join(logits_path, '.zarray'))
                else:
                    accumulator_exists = os.path.exists(logits_path)
            except Exception as e:
                print(f"Error checking if accumulator exists: {e}")
                # Continue anyway
                accumulator_exists = True
            
            if accumulator_exists:
                print(f"\n--- Inference Finished ---")
                print(f"Accumulator saved to: {accumulator_path}")

                print("\n--- Inspecting Accumulator ---")
                try:
                    # Open the logits zarr store using our helper function
                    logits_store = open_zarr(
                        path=logits_path,
                        mode='r',
                        storage_options={'anon': False} if logits_path.startswith('s3://') else None
                    )
                    print(f"Logits shape: {logits_store.shape}")
                    print(f"Logits dtype: {logits_store.dtype}")
                    print(f"Logits chunks: {logits_store.chunks}")
                    
                    # Print normalization factor from metadata
                    if hasattr(logits_store, 'attrs') and 'normalization_factor' in logits_store.attrs:
                        norm_factor = logits_store.attrs['normalization_factor']
                        print(f"Normalization factor: {norm_factor:.6f}")
                    
                except Exception as inspect_e:
                    print(f"Could not inspect accumulator: {inspect_e}")
                
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
                    coords_store = open_zarr(
                        path=coords_path,
                        mode='r',
                        storage_options={'anon': False} if coords_path.startswith('s3://') else None
                    )
                    print(f"Coords shape: {coords_store.shape}")
                    print(f"Coords dtype: {coords_store.dtype}")
                    first_few_coords = coords_store[0:5]
                    print(f"First few coordinates:\n{first_few_coords}")
                except Exception as inspect_e:
                    print(f"Could not inspect coordinate Zarr: {inspect_e}")
                return 0
            else:
                print(f"\n--- Inference finished, but output paths don't seem to exist ---")
                print(f"Logits path: {logits_path} (exists: {logits_exists})")
                print(f"Coordinates path: {coords_path} (exists: {coords_exists})")
                return 1
        else:
            print("\n--- Inference finished, but output paths are None ---")
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
