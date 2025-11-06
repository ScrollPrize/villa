import torch
import numpy as np
import zarr
import os
import json
import shutil
import multiprocessing
import threading
import fsspec
import numcodecs
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# fork causes issues on windows , force to spawn
multiprocessing.set_start_method('spawn', force=True)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from vesuvius.utils.models.load_nnunet_model import load_model_for_inference
from vesuvius.data.vc_dataset import VCDataset
from vesuvius.data.utils import open_zarr
from pathlib import Path
from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.run.blending import generate_gaussian_map
from vesuvius.models.run.tta import infer_with_tta
from vesuvius.data.volume import Volume
from datetime import datetime
from vesuvius.models.datasets.intensity_properties import load_intensity_props_formatted

# Optional import for TIFF IO
try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

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
                 tiff_activation: str = None,
                 normalization_scheme: str = 'instance_zscore',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_dataloader_workers: int = 4,
                 verbose: bool = False,
                 skip_empty_patches: bool = True,  # Skip empty/homogeneous patches
                 # params to get passed to Volume 
                 scroll_id: [str, int] = None,
                 segment_id: [str, int] = None,
                 energy: int = None,
                 resolution: float = None,
                 compressor_name: str = 'zstd',
                 compression_level: int = 1,
                 hf_token: str = None,
                 # optional fallback when .pth lacks embedded model_config
                 config_yaml: str = None,
                 slicewise_axes=None,
                 output_mode: str = 'binary'
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
        # TIFF activation control: 'softmax' | 'argmax' | 'none'; None => derive from save_softmax
        if tiff_activation is not None and tiff_activation not in ('softmax', 'argmax', 'none'):
            raise ValueError(f"Invalid tiff_activation '{tiff_activation}'. Must be one of: softmax, argmax, none")
        self.tiff_activation = tiff_activation
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
        self.config_yaml = config_yaml
        valid_modes = {'binary', 'multiclass', 'surface_frame'}
        if output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{output_mode}'. Must be one of {sorted(valid_modes)}.")
        if output_mode == 'surface_frame':
            if save_softmax:
                raise ValueError("save_softmax is not supported when output_mode is 'surface_frame'.")
            if tiff_activation not in (None, 'none'):
                raise ValueError("tiff_activation must be None or 'none' for surface_frame outputs.")
        self.output_mode = output_mode
        self.model_patch_size = None
        self.num_classes = None
        self.intensity_props = None

        # Configure slicewise axes for 2D volume inference
        allowed_axes = {'z', 'y', 'x'}
        if slicewise_axes is None:
            resolved_axes = ['z']
        elif isinstance(slicewise_axes, str):
            axes_norm = slicewise_axes.lower().strip()
            if axes_norm == 'all':
                resolved_axes = ['z', 'y', 'x']
            else:
                resolved_axes = [axes_norm]
        else:
            resolved_axes = list(slicewise_axes)

        normalized_axes = []
        for axis in resolved_axes:
            axis_lower = str(axis).lower().strip()
            if axis_lower == 'all':
                normalized_axes = ['z', 'y', 'x']
                break
            if axis_lower not in allowed_axes:
                raise ValueError(f"Invalid slicewise axis '{axis}'. Must be one of {sorted(allowed_axes)} or 'all'.")
            if axis_lower not in normalized_axes:
                normalized_axes.append(axis_lower)

        if not normalized_axes:
            normalized_axes = ['z']

        self.slicewise_axes = tuple(normalized_axes)

        # Internal: detect if input is a TIFF file or folder of TIFFs
        self._tiff_inputs = []
        if self.input:
            in_path = Path(self.input)
            if in_path.is_file() and in_path.suffix.lower() in {'.tif', '.tiff'}:
                self._tiff_inputs = [in_path]
                self.input_format = 'tiff'
            elif in_path.is_dir():
                tifs = sorted([p for p in in_path.iterdir() if p.suffix.lower() in {'.tif', '.tiff'}])
                if tifs:
                    self._tiff_inputs = tifs
                    self.input_format = 'tiff'
        
        # Store normalization info from model checkpoint
        self.model_normalization_scheme = None
        self.model_intensity_properties = None
        
        # Multi-task model info
        self.is_multi_task = False
        self.target_info = None  # Will store target names and channel counts

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
            
            # Check if this is a train.py model from HuggingFace
            if isinstance(model_info, dict) and model_info.get('is_train_py', False):
                checkpoint_path = Path(model_info['checkpoint_path'])
                if self.verbose:
                    print(f"Loading train.py checkpoint from HuggingFace: {checkpoint_path}")
                model_info = self._load_train_py_model(checkpoint_path)
        else:
            # Check if this is a train.py checkpoint (single .pth file)
            model_path = Path(self.model_path)
            is_train_py_checkpoint = model_path.is_file() and model_path.suffix == '.pth'
            
            if is_train_py_checkpoint:
                # Load train.py checkpoint
                if self.verbose:
                    print(f"Loading train.py checkpoint from: {self.model_path}")
                model_info = self._load_train_py_model(model_path)
            else:
                # Load from local path using nnUNet loader
                if self.verbose:
                    print(f"Loading nnUNet model from local path: {self.model_path}")
                model_info = load_model_for_inference(
                    model_folder=self.model_path,
                    device_str=str(self.device),
                    verbose=self.verbose
                )
        
        # model loader returns a dict, network is the actual model
        model = model_info['network']
        model.eval()
        
        # patch size and number of classes from model_info
        self.model_patch_size = tuple(model_info.get('patch_size', (192, 192, 192)))
        self.num_classes = model_info.get('num_seg_heads', None)
        
        # Check if this is a multi-task model from targets info
        if 'targets' in model_info and model_info['targets']:
            self.is_multi_task = True
            self.target_info = {}
            self.num_classes = 0
            for target_name, target_config in model_info['targets'].items():
                target_channels = target_config.get('out_channels', 1)
                self.target_info[target_name] = {
                    'out_channels': target_channels,
                    'start_channel': self.num_classes,
                    'end_channel': self.num_classes + target_channels
                }
                self.num_classes += target_channels
            if self.verbose:
                print(f"Detected multi-task model with targets: {list(model_info['targets'].keys())}")
                print(f"Total output channels: {self.num_classes}")
        
        # use models patch size if one wasn't specified
        if self.patch_size is None:
            self.patch_size = self.model_patch_size
            if self.verbose:
                print(f"Using model's patch size: {self.patch_size}")
        else:
            if self.verbose and self.patch_size != self.model_patch_size:
                print(f"Warning: Using user-provided patch size {self.patch_size} instead of model's default: {self.model_patch_size}")
        
        # Determine 2D vs 3D model
        self.is_2d_model = len(self.patch_size) == 2

        # Validate patch size for rotation TTA if needed (supports 2D and 3D)
        if self.patch_size is not None and self.tta_type == 'rotation':
            if len(self.patch_size) not in (2, 3):
                raise ValueError(f"Rotation TTA requires 2D or 3D patch size, got {self.patch_size}.")
        
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
                    if isinstance(dummy_output, dict):
                        # Multi-task model returning dict
                        self.is_multi_task = True
                        self.target_info = {}
                        self.num_classes = 0
                        for target_name, target_output in dummy_output.items():
                            target_channels = target_output.shape[1]
                            self.target_info[target_name] = {
                                'out_channels': target_channels,
                                'start_channel': self.num_classes,
                                'end_channel': self.num_classes + target_channels
                            }
                            self.num_classes += target_channels
                        if self.verbose:
                            print(f"Inferred multi-task model with total output channels: {self.num_classes}")
                            print(f"Target channel mapping: {self.target_info}")
                    else:
                        # Single task model
                        self.num_classes = dummy_output.shape[1]  # N, C, D, H, W
                        if self.verbose:
                            print(f"Inferred number of output classes via dummy inference: {self.num_classes}")
            except Exception as e:
                raise RuntimeError(f"Warning: Could not automatically determine number of classes via dummy inference: {e}. \nEnsure your model is loaded correctly and check the expected input shape")
            
        if self.output_mode == 'surface_frame':
            if self.is_multi_task:
                raise ValueError("surface_frame mode expects a single-target model (found multi-task outputs).")
            if self.num_classes != 9:
                raise ValueError(f"surface_frame mode expects exactly 9 output channels, but model provides {self.num_classes}.")
            if self.verbose:
                print("Validated surface-frame model with 9 output channels.")

        return model
    
    def _load_train_py_model(self, checkpoint_path):
        """Load a model checkpoint from train.py format."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model configuration
        model_config = checkpoint_data.get('model_config', {})
        if not model_config:
            raise ValueError("No model configuration found in checkpoint")
        
        # Extract normalization info if present
        if 'normalization_scheme' in checkpoint_data:
            self.model_normalization_scheme = checkpoint_data['normalization_scheme']
            if self.verbose:
                print(f"Found normalization scheme in checkpoint: {self.model_normalization_scheme}")
        
        if 'intensity_properties' in checkpoint_data:
            self.model_intensity_properties = checkpoint_data['intensity_properties']
            if self.verbose:
                print("Found intensity properties in checkpoint:")
                for key, value in self.model_intensity_properties.items():
                    print(f"  {key}: {value:.4f}")
        
        # Create minimal config manager for NetworkFromConfig
        class MinimalConfigManager:
            def __init__(self, model_config):
                self.model_config = model_config
                self.targets = model_config.get('targets', {})
                self.train_patch_size = model_config.get('train_patch_size', model_config.get('patch_size', (128, 128, 128)))
                self.train_batch_size = model_config.get('train_batch_size', model_config.get('batch_size', 2))
                self.in_channels = model_config.get('in_channels', 1)
                self.autoconfigure = model_config.get('autoconfigure', False)
                self.model_name = model_config.get('model_name', 'Model')
                
                # Set spacing based on patch size dimensions
                self.spacing = [1] * len(self.train_patch_size)
        
        mgr = MinimalConfigManager(model_config)
        
        # Build model using NetworkFromConfig
        model = NetworkFromConfig(mgr)
        model = model.to(self.device)
        
        # Load weights
        model_state_dict = checkpoint_data.get('model', checkpoint_data)
        
        # Check if this is a compiled model state dict
        is_compiled = any("_orig_mod." in key for key in model_state_dict.keys())
        
        # Compile model if needed
        if self.device.type == 'cuda' and is_compiled:
            if self.verbose:
                print("Compiling model to match checkpoint format")
            model = torch.compile(model)
        
        # Load state dict
        model.load_state_dict(model_state_dict, strict=True)
        if self.verbose:
            print("Model weights loaded successfully")
        
        # Handle multi-target models
        if len(mgr.targets) > 1:
            if self.verbose:
                print(f"Multi-target model detected with targets: {list(mgr.targets.keys())}")
            
            # Set multi-task flag
            self.is_multi_task = True
            
            # Calculate total output channels and store target info
            self.target_info = {}
            num_classes = 0
            for target_name, target_config in mgr.targets.items():
                target_channels = target_config.get('out_channels', 1)
                self.target_info[target_name] = {
                    'out_channels': target_channels,
                    'start_channel': num_classes,
                    'end_channel': num_classes + target_channels
                }
                num_classes += target_channels
            
            if self.verbose:
                print(f"Total output channels across all targets: {num_classes}")
                print(f"Target channel mapping: {self.target_info}")
        else:
            # Single target model
            target_name = list(mgr.targets.keys())[0] if mgr.targets else 'output'
            num_classes = mgr.targets.get(target_name, {}).get('out_channels', 1)
        
        # Create model_info dict compatible with the rest of the code
        model_info = {
            'network': model,
            'patch_size': mgr.train_patch_size,
            'num_input_channels': mgr.in_channels,
            'num_seg_heads': num_classes,
            'model_config': model_config,
            'targets': mgr.targets
        }
        
        return model_info

    def _resolve_normalization(self):
        """Resolve normalization scheme and parameters consistently for all paths.
        Returns a tuple: (normalization_scheme, global_mean, global_std, intensity_props).
        """
        normalization_scheme = self.model_normalization_scheme or self.normalization_scheme
        # Map legacy 'zscore' from train.py checkpoints to explicit schemes
        if self.model_normalization_scheme and normalization_scheme == 'zscore':
            if self.model_intensity_properties and 'mean' in self.model_intensity_properties and 'std' in self.model_intensity_properties:
                normalization_scheme = 'global_zscore'
                if self.verbose:
                    print("Mapped 'zscore' to 'global_zscore' (intensity properties available)")
            else:
                normalization_scheme = 'instance_zscore'
                if self.verbose:
                    print("Mapped 'zscore' to 'instance_zscore' (no intensity properties)")

        global_mean = None
        global_std = None
        intensity_props = None

        # Allow explicit nnU-Net-style intensity props JSON (ct normalization)
        if hasattr(self, 'intensity_props_json') and self.intensity_props_json:
            try:
                props = load_intensity_props_formatted(Path(self.intensity_props_json), channel=0)
                if props:
                    intensity_props = props
                    normalization_scheme = 'ct'
                    if self.verbose:
                        print(f"Loaded intensity properties from JSON for CT normalization: {self.intensity_props_json}")
            except Exception as e:
                print(f"Warning: Failed to load intensity properties from {self.intensity_props_json}: {e}")

        if normalization_scheme == 'global_zscore' and self.model_intensity_properties:
            global_mean = self.model_intensity_properties.get('mean')
            global_std = self.model_intensity_properties.get('std')
            if self.verbose and global_mean is not None and global_std is not None:
                print(f"Using global normalization from checkpoint: mean={global_mean:.4f}, std={global_std:.4f}")

        # Only use CT normalization if it was actually configured
        # Do NOT auto-switch a model trained with zscore to CT at inference time.
        if normalization_scheme == 'ct':
            # If user/model requested CT, source intensity props from checkpoint when available
            if intensity_props is None and self.model_intensity_properties and all(k in self.model_intensity_properties for k in ['percentile_00_5', 'percentile_99_5', 'mean', 'std']):
                intensity_props = {
                    'mean': self.model_intensity_properties['mean'],
                    'std': self.model_intensity_properties['std'],
                    'percentile_00_5': self.model_intensity_properties['percentile_00_5'],
                    'percentile_99_5': self.model_intensity_properties['percentile_99_5']
                }
                if self.verbose:
                    print("Using CT normalization from checkpoint intensity properties")
        return normalization_scheme, global_mean, global_std, intensity_props

    def _create_dataset_and_loader(self):
        # Use step_size instead of overlap (step_size is [0-1] representing stride as fraction of patch size)
        # step_size of 0.5 means 50% overlap
        
        # Resolve normalization scheme and parameters consistently
        normalization_scheme, global_mean, global_std, intensity_props = self._resolve_normalization()
        
        self.dataset = VCDataset(
            input_path=self.input,
            patch_size=self.patch_size,
            step_size=self.overlap,
            num_parts=self.num_parts,
            part_id=self.part_id,
            normalization_scheme=normalization_scheme,
            global_mean=global_mean,
            global_std=global_std,
            intensity_props=intensity_props,
            return_as_type='np.float32',
            input_format=self.input_format,
            verbose=self.verbose,
            mode='infer',
            skip_empty_patches=self.skip_empty_patches,
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
    
    def _concat_multi_task_outputs(self, outputs_dict):
        """Concatenate multi-task model outputs into a single tensor.
        
        Args:
            outputs_dict: Dictionary of target_name -> tensor outputs from multi-task model
            
        Returns:
            Concatenated tensor with all target outputs along the channel dimension
        """
        if not isinstance(outputs_dict, dict):
            return outputs_dict
            
        # Sort targets by their start_channel to preserve the correct channel order
        # This ensures outputs are concatenated in the same order they were allocated during model loading
        sorted_targets = sorted(self.target_info.items(), key=lambda x: x[1]['start_channel'])
        
        # Collect outputs in the correct channel order
        output_tensors = []
        for target_name, target_info in sorted_targets:
            if target_name in outputs_dict:
                output_tensors.append(outputs_dict[target_name])
            else:
                raise ValueError(f"Target '{target_name}' not found in model outputs")
        
        # Concatenate along channel dimension (dim=1)
        concatenated = torch.cat(output_tensors, dim=1)
        
        return concatenated
        
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

    def _create_output_stores(self):
        if self.num_classes is None or self.patch_size is None or self.num_total_patches is None:
            raise RuntimeError("Cannot create output stores: model/patch info missing.")
        if not self.patch_start_coords_list:
            raise RuntimeError("Cannot create output stores: patch coordinates not available.")

        compressor = self._get_zarr_compressor()
        # For 2D models, use a pseudo-3D patch size with pZ=1 to reuse the same zarr layout
        store_patch_size = self.patch_size if len(self.patch_size) == 3 else (1, *self.patch_size)
        output_shape = (self.num_total_patches, self.num_classes, *store_patch_size)
        output_chunks = (1, self.num_classes, *store_patch_size)
        main_store_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
        
        print(f"Creating output store at: {main_store_path}")
        
        self.output_store = open_zarr(
            path=main_store_path, 
            mode='w',  
            storage_options={'anon': False} if main_store_path.startswith('s3://') else None,
            verbose=self.verbose,
            shape=output_shape,
            chunks=output_chunks,
            dtype=np.float16,  
            compressor=compressor,
            write_empty_chunks=False  # we skip empty chunks here so we don't write all zero patches to the array but keep
                                      # the proper indices for later re-zarring 
        )
        
        print(f"Created zarr array at {main_store_path} with shape {self.output_store.shape}")
        
        self.coords_store_path = os.path.join(self.output_dir, f"coordinates_part_{self.part_id}.zarr")
        coord_len = len(store_patch_size)
        coord_shape = (self.num_total_patches, coord_len)
        coord_chunks = (min(self.num_total_patches, 4096), coord_len)
        
        print(f"Creating coordinates store at: {self.coords_store_path}")
        
        coords_store = open_zarr(
            path=self.coords_store_path,
            mode='w',
            storage_options={'anon': False} if self.coords_store_path.startswith('s3://') else None,
            verbose=self.verbose,
            shape=coord_shape,
            chunks=coord_chunks,
            dtype=np.int32,
            compressor=compressor,
            write_empty_chunks=False  
        )
        
        print(f"Created coordinates zarr array at {self.coords_store_path} with shape {coords_store.shape}")
        
        try:
            original_volume_shape = None
            # Prefer dataset input_shape if available
            if hasattr(self, 'dataset') and self.dataset is not None and hasattr(self.dataset, 'input_shape'):
                if len(self.dataset.input_shape) == 4:  # has channel dimension
                    original_volume_shape = list(self.dataset.input_shape[1:])
                else:  # no channel dimension
                    original_volume_shape = list(self.dataset.input_shape)
                if self.verbose:
                    print(f"Derived original volume shape from dataset.input_shape: {original_volume_shape}")
            # Fallback: explicit attribute set by custom paths (e.g., 2D slices over volume)
            if original_volume_shape is None and hasattr(self, 'original_volume_shape') and self.original_volume_shape is not None:
                original_volume_shape = list(self.original_volume_shape)
                if self.verbose:
                    print(f"Using original_volume_shape from Inferer: {original_volume_shape}")
            
            # store some metadata we might later want 
            self.output_store.attrs['patch_size'] = list(store_patch_size)
            self.output_store.attrs['overlap'] = self.overlap
            self.output_store.attrs['part_id'] = self.part_id
            self.output_store.attrs['num_parts'] = self.num_parts
            self.output_store.attrs['processing_mode'] = self.output_mode
            
            # Store multi-task metadata if applicable
            if self.is_multi_task and self.target_info:
                self.output_store.attrs['is_multi_task'] = True
                self.output_store.attrs['target_info'] = self.target_info
                if self.verbose:
                    print(f"Stored multi-task metadata in output zarr")
            
            if original_volume_shape:
                self.output_store.attrs['original_volume_shape'] = original_volume_shape
            # Add inference args metadata for downstream consumers
            try:
                inference_args = {
                    'model_path': str(self.model_path),
                    'model_name': type(self.model).__name__ if self.model is not None else None,
                    'input_dir': str(self.input),
                    'input_format': str(self.input_format),
                    'output_dir': str(self.output_dir),
                    'tta_type': str(self.tta_type),
                    'do_tta': bool(self.do_tta),
                    'overlap': float(self.overlap),
                    'batch_size': int(self.batch_size),
                    'patch_size': list(self.patch_size),
                    'normalization_scheme': str(self.model_normalization_scheme or self.normalization_scheme),
                    'device': str(self.device),
                    'skip_empty_patches': bool(self.skip_empty_patches),
                    'scroll_id': self.scroll_id,
                    'segment_id': self.segment_id,
                    'energy': self.energy,
                    'resolution': self.resolution,
                    'compressor_name': str(self.compressor_name),
                    'compression_level': int(self.compression_level),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'output_mode': self.output_mode
                }
                self.output_store.attrs['inference_args'] = inference_args
            except Exception as me:
                if self.verbose:
                    print(f"Warning: failed to attach inference_args metadata: {me}")
            
            coords_store.attrs['part_id'] = self.part_id
            coords_store.attrs['num_parts'] = self.num_parts
            
        except Exception as e:
            print(f"Warning: Failed to write custom attributes: {e}")

        coords_np = np.array(self.patch_start_coords_list, dtype=np.int32)
        coords_store[:] = coords_np
        
        if self.verbose: 
            print(f"Created output stores: {main_store_path} and {self.coords_store_path}")
        
        return self.output_store

    def _process_batches(self):
        numcodecs.blosc.use_threads = False
        
        self.current_patch_write_index = 0
        max_workers = min(16, os.cpu_count() or 4)
        
        zarr_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
        
        if not zarr_path:
            error_msg = f"Error: Empty zarr_path generated from output_dir='{self.output_dir}'"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Verify we have a valid output store from _create_output_stores()
        if self.output_store is None:
            raise RuntimeError(f"Error: output_store is None. Make sure _create_output_stores() was called successfully.")
            
        if self.verbose:
            print(f"Using existing output store: {zarr_path}")
            print(f"Output store shape: {self.output_store.shape}")
        
        # reference to the output store that will be shared by all threads
        output_store = self.output_store
        
        def write_patch(write_index, patch_data):
            try:
                output_store[write_index] = patch_data
                return write_index
            except Exception as e:
                raise RuntimeError(f"Failed to write patch at index {write_index}: {str(e)}")
            
        with tqdm(total=self.num_total_patches, desc=f"Inferring Part {self.part_id}") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for batch_data in self.dataloader:
                    if isinstance(batch_data, dict):
                        input_batch = batch_data['data'].to(self.device)
                        is_empty_flags = batch_data.get('is_empty', [False] * input_batch.shape[0])
                    elif isinstance(batch_data, (list, tuple)):
                        input_batch = batch_data[0].to(self.device)
                        is_empty_flags = [False] * input_batch.shape[0]
                    else:
                        input_batch = batch_data.to(self.device)
                        is_empty_flags = [False] * input_batch.shape[0]
                    
                    # the case that the batch is empty is valid, e.g. when the input volume is smaller than the patch size
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
                        
                        with torch.no_grad(), torch.autocast(device_type=self.device.type):
                            if self.do_tta:
                                non_empty_output = infer_with_tta(
                                    self.model,
                                    non_empty_input,
                                    self.tta_type,
                                    is_multi_task=self.is_multi_task,
                                    concat_multi_task_outputs=self._concat_multi_task_outputs
                                )
                            else:
                                non_empty_output = self.model(non_empty_input)
                                if self.is_multi_task:
                                    non_empty_output = self._concat_multi_task_outputs(non_empty_output)
                                # Convert dict outputs to concatenated tensors if multi-task
                                if self.is_multi_task:
                                    non_empty_output = self._concat_multi_task_outputs(non_empty_output) 
                        
                        # Place non-empty patch outputs in the correct positions in output_batch
                        for idx, original_idx in enumerate(non_empty_indices):
                            output_batch[original_idx] = non_empty_output[idx]
                    
                    else:
                        if self.verbose:
                            print("Batch contains only empty patches, skipping model inference")
                    
                    output_np = output_batch.cpu().numpy().astype(np.float16)
                    current_batch_size = output_np.shape[0]
                    
                    patch_indices = batch_data.get('index', list(range(current_batch_size)))
                    
                    for i in range(current_batch_size):
                        patch_data = output_np[i]  # Shape: (C, Z, Y, X)
                        write_index = patch_indices[i] if i < len(patch_indices) else i
                        future = executor.submit(write_patch, write_index, patch_data)
                        futures.append(future)
                        
                    completed = [f for f in futures if f.done()]
                    for future in completed:
                        try:
                            result = future.result() 
                            if result is not None:  # Only update if write was successful
                                pbar.update(1)
                                self.current_patch_write_index += 1
                        except Exception as e:
                            print(f"Error processing future result: {e}")
                    
                    futures = [f for f in futures if not f.done()]
                
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None: 
                            pbar.update(1)
                            self.current_patch_write_index += 1
                    except Exception as e:
                        print(f"Error processing future result: {e}")
        
        if self.verbose:
            print(f"Finished writing {self.current_patch_write_index} patches.")
        
        if self.current_patch_write_index != self.num_total_patches:
            print(f"Warning: Expected {self.num_total_patches} patches, but wrote {self.current_patch_write_index}.")

    # ------------- TIFF SUPPORT -------------
    def _resolve_normalization_for_tiff(self):
        # Backwards-compatible alias; keep name used elsewhere
        return self._resolve_normalization()

    def _normalize_numpy(self, arr: np.ndarray, normalization_scheme: str, global_mean: float = None, global_std: float = None, intensity_props: dict = None) -> np.ndarray:
        # arr is 2D (Y,X) or 3D (Z,Y,X); work in float32
        a = arr.astype(np.float32, copy=False)
        if normalization_scheme == 'none':
            return a
        if normalization_scheme == 'instance_zscore':
            mean = float(a.mean())
            std = float(a.std())
            return (a - mean) / max(std, 1e-8)
        if normalization_scheme == 'global_zscore':
            if global_mean is None or global_std is None:
                # Fallback to instance if globals not present
                mean = float(a.mean()); std = float(a.std())
                return (a - mean) / max(std, 1e-8)
            return (a - float(global_mean)) / max(float(global_std), 1e-8)
        if normalization_scheme == 'instance_minmax':
            mn = float(a.min()); mx = float(a.max())
            denom = max(mx - mn, 1e-8)
            return (a - mn) / denom
        if normalization_scheme == 'ct':
            if not intensity_props or not all(k in intensity_props for k in ('percentile_00_5', 'percentile_99_5', 'mean', 'std')):
                # Fallback: no-op if props missing
                return a
            lb = float(intensity_props['percentile_00_5'])
            ub = float(intensity_props['percentile_99_5'])
            mu = float(intensity_props['mean'])
            sd = float(intensity_props['std'])
            a = np.clip(a, lb, ub)
            return (a - mu) / max(sd, 1e-8)
        # Default: no-op
        return a

    def _compute_patch_positions(self, img_shape: tuple) -> list:
        # For 3D models: img_shape is (Z,Y,X)
        # For 2D models: img_shape is (Y,X)
        from vesuvius.utils.models.helpers import compute_steps_for_sliding_window
        if len(self.patch_size) == 3:
            pZ, pY, pX = self.patch_size
            Z, Y, X = img_shape
            z_positions = [0] if Z < pZ else compute_steps_for_sliding_window(Z, pZ, self.overlap)
            y_positions = [0] if Y < pY else compute_steps_for_sliding_window(Y, pY, self.overlap)
            x_positions = [0] if X < pX else compute_steps_for_sliding_window(X, pX, self.overlap)
            return [(z, y, x) for z in z_positions for y in y_positions for x in x_positions]
        else:
            pY, pX = self.patch_size
            Y, X = img_shape
            y_positions = [0] if Y < pY else compute_steps_for_sliding_window(Y, pY, self.overlap)
            x_positions = [0] if X < pX else compute_steps_for_sliding_window(X, pX, self.overlap)
            return [(y, x) for y in y_positions for x in x_positions]

    def _compute_plane_positions(self, plane_shape, plane_patch_size):
        """Compute sliding-window start positions for a 2D plane."""
        from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

        dim0, dim1 = plane_shape
        p0, p1 = plane_patch_size

        if p0 <= 0 or p1 <= 0:
            raise ValueError(f"Invalid plane patch size: {plane_patch_size}")

        pos0 = [0] if dim0 <= p0 else compute_steps_for_sliding_window(dim0, p0, self.overlap)
        pos1 = [0] if dim1 <= p1 else compute_steps_for_sliding_window(dim1, p1, self.overlap)

        if not pos0:
            pos0 = [0]
        if not pos1:
            pos1 = [0]

        return [(int(a), int(b)) for a in pos0 for b in pos1]

    def _create_volume_accumulators(self, volume_shape, chunk_hint=None):
        """Create accumulation stores for slicewise 2D inference."""
        Z, Y, X = volume_shape
        compressor = self._get_zarr_compressor()

        if chunk_hint is None:
            chunk_z = max(1, min(32, Z))
            chunk_y = max(1, min(32, Y))
            chunk_x = max(1, min(32, X))
        else:
            chunk_z, chunk_y, chunk_x = chunk_hint

        logits_chunks = (max(1, min(self.num_classes or 1, 2)), chunk_z, chunk_y, chunk_x)
        weights_chunks = (chunk_z, chunk_y, chunk_x)

        logits_sum_path = os.path.join(self.output_dir, f"logits_sum_part_{self.part_id}.zarr")
        weights_path = os.path.join(self.output_dir, f"weights_part_{self.part_id}.zarr")

        logits_store = open_zarr(
            path=logits_sum_path,
            mode='w',
            storage_options={'anon': False} if logits_sum_path.startswith('s3://') else None,
            verbose=self.verbose,
            shape=(self.num_classes, Z, Y, X),
            chunks=logits_chunks,
            dtype=np.float32,
            compressor=compressor,
            write_empty_chunks=False
        )

        weights_store = open_zarr(
            path=weights_path,
            mode='w',
            storage_options={'anon': False} if weights_path.startswith('s3://') else None,
            verbose=self.verbose,
            shape=(Z, Y, X),
            chunks=weights_chunks,
            dtype=np.float32,
            compressor=compressor,
            write_empty_chunks=False
        )

        return logits_store, logits_sum_path, weights_store, weights_path, (chunk_z, chunk_y, chunk_x)

    def _finalize_volume_logits(self, logits_store, weights_store, final_path, volume_shape, chunk_sizes):
        """Normalize accumulated logits by weights and write final output store."""
        Z, Y, X = volume_shape
        chunk_z, chunk_y, chunk_x = chunk_sizes
        compressor = self._get_zarr_compressor()

        final_chunks = (max(1, min(self.num_classes or 1, 2)), chunk_z, chunk_y, chunk_x)

        final_store = open_zarr(
            path=final_path,
            mode='w',
            storage_options={'anon': False} if final_path.startswith('s3://') else None,
            verbose=self.verbose,
            shape=(self.num_classes, Z, Y, X),
            chunks=final_chunks,
            dtype=np.float16,
            compressor=compressor,
            write_empty_chunks=False
        )

        for z0 in range(0, Z, chunk_z):
            z1 = min(z0 + chunk_z, Z)
            for y0 in range(0, Y, chunk_y):
                y1 = min(y0 + chunk_y, Y)
                for x0 in range(0, X, chunk_x):
                    x1 = min(x0 + chunk_x, X)
                    logits_chunk = logits_store.oindex[
                        (slice(None), slice(z0, z1), slice(y0, y1), slice(x0, x1))
                    ]
                    weights_chunk = weights_store.oindex[
                        (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                    ]

                    if not np.any(weights_chunk > 0):
                        final_store.oindex[
                            (slice(None), slice(z0, z1), slice(y0, y1), slice(x0, x1))
                        ] = np.zeros_like(logits_chunk, dtype=np.float16)
                        continue

                    safe_weights = np.where(weights_chunk > 0, weights_chunk, 1.0)
                    logits_chunk = logits_chunk / safe_weights[None, ...]
                    zero_mask = weights_chunk <= 0
                    if np.any(zero_mask):
                        logits_chunk[:, zero_mask] = 0.0

                    final_store.oindex[
                        (slice(None), slice(z0, z1), slice(y0, y1), slice(x0, x1))
                    ] = logits_chunk.astype(np.float16)

        return final_store

    def _cleanup_path(self, path):
        if not path:
            return
        try:
            if path.startswith('s3://'):
                fs = fsspec.filesystem('s3', anon=False)
                if fs.exists(path):
                    fs.rm(path, recursive=True)
            else:
                if os.path.exists(path):
                    shutil.rmtree(path)
        except Exception as exc:
            if self.verbose:
                print(f"Warning: failed to remove temporary path {path}: {exc}")

    def _accumulate_axis_slices(self, *, volume, axis, slice_positions, slice_count,
                                 patch_hw, gaussian_map, logits_store, weights_store,
                                 channels, volume_shape, pbar):
        patch_h, patch_w = patch_hw
        Z, Y, X = volume_shape

        def _run_model(patch_array):
            tensor = torch.from_numpy(patch_array).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                if self.do_tta:
                    logits = infer_with_tta(
                        self.model,
                        tensor,
                        self.tta_type,
                        is_multi_task=self.is_multi_task,
                        concat_multi_task_outputs=self._concat_multi_task_outputs
                    )
                else:
                    logits = self.model(tensor)
                    if self.is_multi_task:
                        logits = self._concat_multi_task_outputs(logits)
            return logits.detach().float().cpu().numpy()[0]

        for slice_idx in range(slice_count):
            for pos0, pos1 in slice_positions:
                if axis == 'z':
                    y0, x0 = pos0, pos1
                    y1 = min(y0 + patch_h, Y)
                    x1 = min(x0 + patch_w, X)

                    if channels > 1:
                        sub = volume[(slice(None), slice(slice_idx, slice_idx + 1), slice(y0, y1), slice(x0, x1))]
                        sub = np.asarray(sub)[:, 0, :, :]
                    else:
                        sub = volume[(slice(slice_idx, slice_idx + 1), slice(y0, y1), slice(x0, x1))]
                        sub = np.asarray(sub)[0, :, :]

                    valid_h = y1 - y0
                    valid_w = x1 - x0

                    if self.skip_empty_patches and (sub.size == 0 or float(np.min(sub)) == float(np.max(sub))):
                        continue

                    patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                    if channels > 1:
                        patch[:, :valid_h, :valid_w] = sub
                    else:
                        patch[0, :valid_h, :valid_w] = sub

                    logits_np = _run_model(patch)
                    logits_slice = logits_np[:, :valid_h, :valid_w]
                    weight_slice = gaussian_map[:valid_h, :valid_w]

                    logits_region = logits_store.oindex[(slice(None), slice_idx, slice(y0, y1), slice(x0, x1))]
                    logits_region += logits_slice * weight_slice[None, ...]
                    logits_store.oindex[(slice(None), slice_idx, slice(y0, y1), slice(x0, x1))] = logits_region

                    weight_region = weights_store.oindex[(slice(slice_idx, slice_idx + 1), slice(y0, y1), slice(x0, x1))][0]
                    weight_region += weight_slice
                    weights_store.oindex[(slice(slice_idx, slice_idx + 1), slice(y0, y1), slice(x0, x1))] = weight_region[np.newaxis, ...]

                elif axis == 'y':
                    z0, x0 = pos0, pos1
                    z1 = min(z0 + patch_h, Z)
                    x1 = min(x0 + patch_w, X)
                    y = slice_idx

                    if channels > 1:
                        sub = volume[(slice(None), slice(z0, z1), slice(y, y + 1), slice(x0, x1))]
                        sub = np.asarray(sub)[:, :, 0, :]
                    else:
                        sub = volume[(slice(z0, z1), slice(y, y + 1), slice(x0, x1))]
                        sub = np.asarray(sub)[:, 0, :]

                    valid_h = z1 - z0
                    valid_w = x1 - x0

                    if self.skip_empty_patches and (sub.size == 0 or float(np.min(sub)) == float(np.max(sub))):
                        continue

                    patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                    if channels > 1:
                        patch[:, :valid_h, :valid_w] = sub
                    else:
                        patch[0, :valid_h, :valid_w] = sub

                    logits_np = _run_model(patch)
                    logits_slice = logits_np[:, :valid_h, :valid_w]
                    weight_slice = gaussian_map[:valid_h, :valid_w]

                    logits_region = logits_store.oindex[(slice(None), slice(z0, z1), y, slice(x0, x1))]
                    logits_region += logits_slice * weight_slice[None, ...]
                    logits_store.oindex[(slice(None), slice(z0, z1), y, slice(x0, x1))] = logits_region

                    weight_region = weights_store.oindex[(slice(z0, z1), y, slice(x0, x1))]
                    weight_region += weight_slice
                    weights_store.oindex[(slice(z0, z1), y, slice(x0, x1))] = weight_region

                else:  # axis == 'x'
                    z0, y0 = pos0, pos1
                    z1 = min(z0 + patch_h, Z)
                    y1 = min(y0 + patch_w, Y)
                    x = slice_idx

                    if channels > 1:
                        sub = volume[(slice(None), slice(z0, z1), slice(y0, y1), slice(x, x + 1))]
                        sub = np.asarray(sub)[:, :, :, 0]
                    else:
                        sub = volume[(slice(z0, z1), slice(y0, y1), slice(x, x + 1))]
                        sub = np.asarray(sub)[:, :, 0]

                    valid_h = z1 - z0
                    valid_w = y1 - y0

                    if self.skip_empty_patches and (sub.size == 0 or float(np.min(sub)) == float(np.max(sub))):
                        continue

                    patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                    if channels > 1:
                        patch[:, :valid_h, :valid_w] = sub
                    else:
                        patch[0, :valid_h, :valid_w] = sub

                    logits_np = _run_model(patch)
                    logits_slice = logits_np[:, :valid_h, :valid_w]
                    weight_slice = gaussian_map[:valid_h, :valid_w]

                    logits_region = logits_store.oindex[(slice(None), slice(z0, z1), slice(y0, y1), x)]
                    logits_region += logits_slice * weight_slice[None, ...]
                    logits_store.oindex[(slice(None), slice(z0, z1), slice(y0, y1), x)] = logits_region

                    weight_region = weights_store.oindex[(slice(z0, z1), slice(y0, y1), x)]
                    weight_region += weight_slice
                    weights_store.oindex[(slice(z0, z1), slice(y0, y1), x)] = weight_region

                pbar.update(1)

    def _infer_numpy_volume_slicewise(self, volume_np, patch_hw, normalization_scheme,
                                      global_mean, global_std, intensity_props):
        axes_to_process = list(self.slicewise_axes) if getattr(self, 'slicewise_axes', None) else ['z']
        arr = np.asarray(volume_np)

        if arr.ndim == 3:
            Z, Y, X = arr.shape
            volume_data = arr.astype(np.float32)[np.newaxis, ...]
            channels = 1
        elif arr.ndim == 4 and arr.shape[0] <= 4:
            # Assume channel-first (C, Z, Y, X)
            channels, Z, Y, X = arr.shape
            volume_data = arr.astype(np.float32)
        elif arr.ndim == 4 and arr.shape[-1] <= 4:
            # Assume channel-last (Z, Y, X, C)
            Z, Y, X, channels = arr.shape
            volume_data = np.moveaxis(arr.astype(np.float32), -1, 0)
        else:
            raise ValueError("Unsupported TIFF shape for 2D slicewise inference. Expected (Z,Y,X) or (C,Z,Y,X).")

        patch_h, patch_w = patch_hw
        gaussian_map = generate_gaussian_map((1, patch_h, patch_w), sigma_scale=8.0, dtype=np.float32)[0][0]
        if np.any(gaussian_map == 0):
            min_nonzero = np.min(gaussian_map[gaussian_map > 0])
            gaussian_map = np.where(gaussian_map == 0, min_nonzero, gaussian_map)

        axis_configs = {}
        if 'z' in axes_to_process:
            coords_z = self._compute_plane_positions((Y, X), patch_hw)
            axis_configs['z'] = (coords_z, Z)
        if 'y' in axes_to_process:
            coords_y = self._compute_plane_positions((Z, X), patch_hw)
            axis_configs['y'] = (coords_y, Y)
        if 'x' in axes_to_process:
            coords_x = self._compute_plane_positions((Z, Y), patch_hw)
            axis_configs['x'] = (coords_x, X)

        total_positions = 0
        for axis in axes_to_process:
            coords, count = axis_configs.get(axis, (None, None))
            if coords is not None:
                total_positions += len(coords) * count

        if total_positions == 0:
            raise RuntimeError("No slice positions generated for the requested slicewise planes.")

        logits_acc = np.zeros((self.num_classes, Z, Y, X), dtype=np.float32)
        weights_acc = np.zeros((Z, Y, X), dtype=np.float32)

        def normalize_slice(slice_arr):
            if normalization_scheme in ('none', None):
                return slice_arr.astype(np.float32, copy=False)
            if slice_arr.ndim == 2:
                return self._normalize_numpy(slice_arr.astype(np.float32, copy=False),
                                             normalization_scheme, global_mean, global_std, intensity_props)
            normalized = np.zeros_like(slice_arr, dtype=np.float32)
            for ci in range(slice_arr.shape[0]):
                normalized[ci] = self._normalize_numpy(slice_arr[ci].astype(np.float32, copy=False),
                                                       normalization_scheme, global_mean, global_std, intensity_props)
            return normalized

        def run_model(patch_array):
            tensor = torch.from_numpy(patch_array).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                if self.do_tta:
                    logits = infer_with_tta(
                        self.model,
                        tensor,
                        self.tta_type,
                        is_multi_task=self.is_multi_task,
                        concat_multi_task_outputs=self._concat_multi_task_outputs
                    )
                else:
                    logits = self.model(tensor)
                    if self.is_multi_task:
                        logits = self._concat_multi_task_outputs(logits)
            return logits.detach().float().cpu().numpy()[0]

        for axis in axes_to_process:
            coords, slice_count = axis_configs.get(axis, (None, None))
            if not coords:
                continue
            for slice_idx in range(slice_count):
                for pos0, pos1 in coords:
                    if axis == 'z':
                        y0, x0 = pos0, pos1
                        y1 = min(y0 + patch_h, Y)
                        x1 = min(x0 + patch_w, X)
                        if channels > 1:
                            sub = volume_data[:, slice_idx, y0:y1, x0:x1]
                        else:
                            sub = volume_data[0, slice_idx, y0:y1, x0:x1]
                        valid_h = y1 - y0
                        valid_w = x1 - x0
                        if self.skip_empty_patches:
                            if sub.size == 0:
                                continue
                            smin = float(np.min(sub))
                            smax = float(np.max(sub))
                            if smin == smax:
                                continue
                        sub_norm = normalize_slice(sub)
                        patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                        if channels > 1:
                            patch[:, :valid_h, :valid_w] = sub_norm
                        else:
                            patch[0, :valid_h, :valid_w] = sub_norm
                        logits_np = run_model(patch)
                        logits_slice = logits_np[:, :valid_h, :valid_w]
                        weight_slice = gaussian_map[:valid_h, :valid_w]
                        logits_acc[:, slice_idx, y0:y1, x0:x1] += logits_slice * weight_slice[None, ...]
                        weights_acc[slice_idx, y0:y1, x0:x1] += weight_slice

                    elif axis == 'y':
                        z0, x0 = pos0, pos1
                        z1 = min(z0 + patch_h, Z)
                        x1 = min(x0 + patch_w, X)
                        if channels > 1:
                            sub = volume_data[:, z0:z1, slice_idx, x0:x1]
                        else:
                            sub = volume_data[0, z0:z1, slice_idx, x0:x1]
                        valid_h = z1 - z0
                        valid_w = x1 - x0
                        if self.skip_empty_patches:
                            if sub.size == 0:
                                continue
                            smin = float(np.min(sub))
                            smax = float(np.max(sub))
                            if smin == smax:
                                continue
                        sub_norm = normalize_slice(sub)
                        patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                        if channels > 1:
                            patch[:, :valid_h, :valid_w] = sub_norm
                        else:
                            patch[0, :valid_h, :valid_w] = sub_norm
                        logits_np = run_model(patch)
                        logits_slice = logits_np[:, :valid_h, :valid_w]
                        weight_slice = gaussian_map[:valid_h, :valid_w]
                        logits_acc[:, z0:z1, slice_idx, x0:x1] += logits_slice * weight_slice[None, ...]
                        weights_acc[z0:z1, slice_idx, x0:x1] += weight_slice

                    else:  # axis == 'x'
                        z0, y0 = pos0, pos1
                        z1 = min(z0 + patch_h, Z)
                        y1 = min(y0 + patch_w, Y)
                        if channels > 1:
                            sub = volume_data[:, z0:z1, y0:y1, slice_idx]
                        else:
                            sub = volume_data[0, z0:z1, y0:y1, slice_idx]
                        valid_h = z1 - z0
                        valid_w = y1 - y0
                        if self.skip_empty_patches:
                            if sub.size == 0:
                                continue
                            smin = float(np.min(sub))
                            smax = float(np.max(sub))
                            if smin == smax:
                                continue
                        sub_norm = normalize_slice(sub)
                        patch = np.zeros((channels, patch_h, patch_w), dtype=np.float32)
                        if channels > 1:
                            patch[:, :valid_h, :valid_w] = sub_norm
                        else:
                            patch[0, :valid_h, :valid_w] = sub_norm
                        logits_np = run_model(patch)
                        logits_slice = logits_np[:, :valid_h, :valid_w]
                        weight_slice = gaussian_map[:valid_h, :valid_w]
                        logits_acc[:, z0:z1, y0:y1, slice_idx] += logits_slice * weight_slice[None, ...]
                        weights_acc[z0:z1, y0:y1, slice_idx] += weight_slice

        safe_weights = np.where(weights_acc > 0, weights_acc, 1.0)
        logits_acc = logits_acc / safe_weights[np.newaxis, ...]
        zero_mask = weights_acc <= 0
        if np.any(zero_mask):
            logits_acc[:, zero_mask] = 0.0

        return logits_acc

    def _infer_2d_tiff_volume(self, volume_np: np.ndarray, out_path: Path):
        if self.patch_size is None or self.num_classes is None:
            raise RuntimeError("Model/patch metadata missing before TIFF inference.")
        normalization_scheme, gmean, gstd, iprops = self._resolve_normalization_for_tiff()
        if len(self.patch_size) == 2:
            patch_hw = self.patch_size
        elif len(self.patch_size) == 3 and self.patch_size[0] == 1:
            patch_hw = self.patch_size[1:]
        else:
            raise ValueError(f"2D model requires 2D patch_size, got {self.patch_size}")

        logits_acc_np = self._infer_numpy_volume_slicewise(
            volume_np=volume_np,
            patch_hw=patch_hw,
            normalization_scheme=normalization_scheme,
            global_mean=gmean,
            global_std=gstd,
            intensity_props=iprops
        )

        C, Z, Y, X = logits_acc_np.shape
        mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')

        if mode == 'softmax':
            m = logits_acc_np.max(axis=0, keepdims=True)
            e = np.exp(logits_acc_np - m)
            s = e.sum(axis=0, keepdims=True) + 1e-8
            out_arr = (e / s).astype(np.float32)
            if Z == 1:
                out_arr = out_arr[:, 0, :, :]
            out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
        elif mode == 'argmax':
            out_arr = np.argmax(logits_acc_np, axis=0)
            if Z == 1:
                out_arr = out_arr[0, :, :]
            if int(self.num_classes) == 2:
                out_arr = (out_arr.astype(np.uint8) * 255).astype(np.uint8)
            else:
                out_arr = out_arr.astype(np.uint8)
        else:
            out_arr = logits_acc_np.astype(np.float32)
            if Z == 1:
                out_arr = out_arr[:, 0, :, :]

        if tifffile is None:
            raise RuntimeError("tifffile is required for TIFF output but is not installed")

        tifffile.imwrite(str(out_path), out_arr, compression='zlib')
        return str(out_path)


    def _infer_single_tiff_in_memory(self, img: np.ndarray, out_path: Path):
        # Ensure patch_size and num_classes are set (model loaded)
        if self.patch_size is None or self.num_classes is None:
            raise RuntimeError("Model/patch metadata missing before TIFF inference.")

        normalization_scheme, gmean, gstd, iprops = self._resolve_normalization_for_tiff()

        if self.is_2d_model:
            if img.ndim != 2:
                raise ValueError("2D model expects 2D TIFF input (H, W)")
            Y, X = img.shape
            pY, pX = self.patch_size
            # If patch covers the whole image (equal or larger), skip blending/normalization
            if (pY >= Y and pX >= X):
                # Prepare single padded patch
                sub = self._normalize_numpy(img, normalization_scheme, gmean, gstd, iprops)
                patch = np.zeros((pY, pX), dtype=np.float32)
                patch[:sub.shape[0], :sub.shape[1]] = sub
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.autocast(device_type=self.device.type):
                    if self.do_tta:
                        logits = infer_with_tta(
                            self.model, t, self.tta_type,
                            is_multi_task=self.is_multi_task,
                            concat_multi_task_outputs=self._concat_multi_task_outputs
                        )
                    else:
                        logits = self.model(t)
                        if self.is_multi_task:
                            logits = self._concat_multi_task_outputs(logits)
                out_np = logits.detach().float().cpu().numpy()[0]  # (C,pY,pX)
                logits_acc = out_np[:, :Y, :X]  # crop to image size, no weighting

                # Convert logits to desired output and save as TIFF
                if tifffile is None:
                    raise RuntimeError("tifffile is required for TIFF output but is not installed")
                mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')
                if mode == 'softmax':
                    m = logits_acc.max(axis=0, keepdims=True)
                    e = np.exp(logits_acc - m)
                    s = e.sum(axis=0, keepdims=True) + 1e-8
                    out_arr = (e / s).astype(np.float32)  # (C,Y,X)
                    out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
                elif mode == 'argmax':
                    out_arr = np.argmax(logits_acc, axis=0)  # (Y,X)
                    if int(self.num_classes) == 2:
                        out_arr = (out_arr.astype(np.uint8) * 255).astype(np.uint8)
                    else:
                        out_arr = out_arr.astype(np.uint8)
                else:
                    out_arr = logits_acc.astype(np.float32)
                tifffile.imwrite(str(out_path), out_arr, compression='zlib')
                return str(out_path)

            # Regular blending path (accumulate on device like nnU-Net)
            g_np = generate_gaussian_map((1, pY, pX), sigma_scale=8.0, dtype=np.float32)[0][0]  # (pY, pX)
            gaussian_map_2d = torch.from_numpy(g_np).to(self.device, dtype=torch.float16)
            # ensure strictly positive weights to avoid division issues
            if torch.any(gaussian_map_2d == 0):
                min_nonzero = torch.min(gaussian_map_2d[gaussian_map_2d > 0])
                gaussian_map_2d = torch.where(gaussian_map_2d == 0, min_nonzero, gaussian_map_2d)

            logits_acc = torch.zeros((self.num_classes, Y, X), dtype=torch.float16, device=self.device)
            weights_acc = torch.zeros((Y, X), dtype=torch.float16, device=self.device)
            coords = self._compute_patch_positions((Y, X))
        else:
            # Promote 2D to 3D with Z=1
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            Z, Y, X = img.shape  # (Z,Y,X)
            pZ, pY, pX = self.patch_size
            # If patch covers the whole volume (equal or larger), skip blending/normalization
            if (pZ >= Z and pY >= Y and pX >= X):
                sub = self._normalize_numpy(img, normalization_scheme, gmean, gstd, iprops)
                patch = np.zeros((pZ, pY, pX), dtype=np.float32)
                patch[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.autocast(device_type=self.device.type):
                    if self.do_tta:
                        logits = infer_with_tta(
                            self.model, t, self.tta_type,
                            is_multi_task=self.is_multi_task,
                            concat_multi_task_outputs=self._concat_multi_task_outputs
                        )
                    else:
                        logits = self.model(t)
                        if self.is_multi_task:
                            logits = self._concat_multi_task_outputs(logits)
                out_np = logits.detach().float().cpu().numpy()[0]  # (C,pZ,pY,pX)
                logits_acc = out_np[:, :Z, :Y, :X]  # crop to volume size, no weighting

                # Convert logits to desired output and save as TIFF
                if tifffile is None:
                    raise RuntimeError("tifffile is required for TIFF output but is not installed")
                mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')
                if mode == 'softmax':
                    m = logits_acc.max(axis=0, keepdims=True)
                    e = np.exp(logits_acc - m)
                    s = e.sum(axis=0, keepdims=True) + 1e-8
                    out_arr = (e / s).astype(np.float32)  # (C,Z,Y,X)
                    if Z == 1:
                        out_arr = out_arr[:, 0, :, :]
                    out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
                elif mode == 'argmax':
                    out_arr = np.argmax(logits_acc, axis=0)  # (Z,Y,X)
                    if Z == 1:
                        out_arr = out_arr[0, :, :]
                    if int(self.num_classes) == 2:
                        out_arr = (out_arr.astype(np.uint8) * 255).astype(np.uint8)
                    else:
                        out_arr = out_arr.astype(np.uint8)
                else:
                    out_arr = logits_acc.astype(np.float32)
                    if Z == 1:
                        out_arr = out_arr[:, 0, :, :]
                tifffile.imwrite(str(out_path), out_arr, compression='zlib')
                return str(out_path)

            # Regular blending path (accumulate on device like nnU-Net)
            g_np = generate_gaussian_map(self.patch_size, sigma_scale=8.0, dtype=np.float32)[0]  # (pZ,pY,pX)
            gaussian_map = torch.from_numpy(g_np).to(self.device, dtype=torch.float16)
            if torch.any(gaussian_map == 0):
                min_nonzero = torch.min(gaussian_map[gaussian_map > 0])
                gaussian_map = torch.where(gaussian_map == 0, min_nonzero, gaussian_map)

            logits_acc = torch.zeros((self.num_classes, Z, Y, X), dtype=torch.float16, device=self.device)
            weights_acc = torch.zeros((Z, Y, X), dtype=torch.float16, device=self.device)
            coords = self._compute_patch_positions((Z, Y, X))

        # Simple batching for speed
        batch_buf = []
        batch_coords = []
        bs = max(1, int(self.batch_size))

        def run_batch(tensor_batch: torch.Tensor, coord_list: list):
            nonlocal logits_acc, weights_acc
            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                if self.do_tta:
                    batch_logits = infer_with_tta(
                        self.model,
                        tensor_batch,
                        self.tta_type,
                        is_multi_task=self.is_multi_task,
                        concat_multi_task_outputs=self._concat_multi_task_outputs
                    )
                else:
                    batch_logits = self.model(tensor_batch)
                    if self.is_multi_task:
                        batch_logits = self._concat_multi_task_outputs(batch_logits)

            # batch_logits: (B, C, pZ, pY, pX) for 3D, (B, C, pY, pX) for 2D (on device)
            if self.is_2d_model:
                for i, (y, x) in enumerate(coord_list):
                    p = batch_logits[i].to(dtype=torch.float16)  # (C,pY,pX) on device
                    iy0 = y; ix0 = x
                    iy1 = min(y + pY, Y); ix1 = min(x + pX, X)
                    py0 = max(0, - (y - 0)); px0 = max(0, - (x - 0))
                    py1 = py0 + (iy1 - iy0); px1 = px0 + (ix1 - ix0)
                    logits_slice = p[:, py0:py1, px0:px1]
                    w_slice = gaussian_map_2d[py0:py1, px0:px1]
                    logits_acc[:, iy0:iy1, ix0:ix1] += logits_slice * w_slice.unsqueeze(0)
                    weights_acc[iy0:iy1, ix0:ix1] += w_slice
            else:
                for i, (z, y, x) in enumerate(coord_list):
                    p = batch_logits[i].to(dtype=torch.float16)  # (C,pZ,pY,pX)
                    iz0 = z; iy0 = y; ix0 = x
                    iz1 = min(z + pZ, Z); iy1 = min(y + pY, Y); ix1 = min(x + pX, X)
                    pz0 = max(0, - (z - 0)); py0 = max(0, - (y - 0)); px0 = max(0, - (x - 0))
                    pz1 = pz0 + (iz1 - iz0); py1 = py0 + (iy1 - iy0); px1 = px0 + (ix1 - ix0)
                    logits_slice = p[:, pz0:pz1, py0:py1, px0:px1]
                    w_slice = gaussian_map[pz0:pz1, py0:py1, px0:px1]
                    logits_acc[:, iz0:iz1, iy0:iy1, ix0:ix1] += logits_slice * w_slice.unsqueeze(0)
                    weights_acc[iz0:iz1, iy0:iy1, ix0:ix1] += w_slice

        # Iterate positions
        # For TIFF in-memory inference we never skip patches based on "emptiness".
        # Skipping leads to uncovered regions (weights remain 0), which shows up
        # as blank holes in the blended logits/probabilities/labels. Always
        # evaluating all patches guarantees full coverage.
        for coord in tqdm(coords, desc=f"Inferring {out_path.name}"):
            # Build padded patch tensor (C=1)
            if self.is_2d_model:
                y, x = coord
                y1, x1 = min(y + pY, Y), min(x + pX, X)
                sub = img[y:y1, x:x1]
                # Normalize per-patch for consistency with dataset/large-TIFF path
                sub = self._normalize_numpy(sub, normalization_scheme, gmean, gstd, iprops)
            else:
                z, y, x = coord
                z1, y1, x1 = min(z + pZ, Z), min(y + pY, Y), min(x + pX, X)
                sub = img[z:z1, y:y1, x:x1]
                # Normalize per-patch for consistency
                sub = self._normalize_numpy(sub, normalization_scheme, gmean, gstd, iprops)
            # Do NOT skip "empty" patches here; ensure weights cover the full image.
            # Pad to patch size
            if self.is_2d_model:
                patch = np.zeros((pY, pX), dtype=np.float32)
                patch[:sub.shape[0], :sub.shape[1]] = sub
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,pY,pX)
            else:
                patch = np.zeros((pZ, pY, pX), dtype=np.float32)
                patch[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,pZ,pY,pX)
            batch_buf.append(t)
            batch_coords.append(coord)
            if len(batch_buf) >= bs:
                tensor_batch = torch.cat(batch_buf, dim=0)
                run_batch(tensor_batch, batch_coords)
                batch_buf.clear(); batch_coords.clear()
        # Final flush
        if batch_buf:
            tensor_batch = torch.cat(batch_buf, dim=0)
            run_batch(tensor_batch, batch_coords)
            batch_buf.clear(); batch_coords.clear()

        # Normalize by weights (only when using blending) on device
        eps = 1e-8
        # broadcast weights to channel dimension
        if self.is_2d_model:
            logits_acc = logits_acc / (weights_acc.clamp_min(eps).unsqueeze(0))
            logits_acc_np = logits_acc.to(dtype=torch.float32).cpu().numpy()
        else:
            logits_acc = logits_acc / (weights_acc.clamp_min(eps).unsqueeze(0))
            logits_acc_np = logits_acc.to(dtype=torch.float32).cpu().numpy()

        # Convert logits to desired output and save as TIFF
        if tifffile is None:
            raise RuntimeError("tifffile is required for TIFF output but is not installed")

        # Choose activation for TIFF output: softmax | argmax | none
        mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')
        if mode == 'softmax':
            # Softmax over channel dimension
            # logits_acc: (C,Y,X) for 2D; (C,Z,Y,X) for 3D
            if self.is_2d_model:
                m = logits_acc_np.max(axis=0, keepdims=True)
                e = np.exp(logits_acc_np - m)
                s = e.sum(axis=0, keepdims=True) + 1e-8
                out_arr = (e / s).astype(np.float32)  # (C,Y,X)
            else:
                m = logits_acc_np.max(axis=0, keepdims=True)
                e = np.exp(logits_acc_np - m)
                s = e.sum(axis=0, keepdims=True) + 1e-8
                out_arr = (e / s).astype(np.float32)  # (C,Z,Y,X)
                if Z == 1:
                    out_arr = out_arr[:, 0, :, :]
            # Convert softmax probabilities to uint8 [0, 255]
            out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
        elif mode == 'argmax':
            # Argmax over channels to produce label map
            if self.is_2d_model:
                out_arr = np.argmax(logits_acc_np, axis=0)  # (Y,X)
            else:
                out_arr = np.argmax(logits_acc_np, axis=0)  # (Z,Y,X)
                if Z == 1:
                    out_arr = out_arr[0, :, :]
            # Convert labels to uint8; for binary classes map {0,1}->{0,255}
            if int(self.num_classes) == 2:
                out_arr = (out_arr.astype(np.uint8) * 255).astype(np.uint8)
            else:
                out_arr = out_arr.astype(np.uint8)
        else:
            # mode == 'none': write raw logits as float32
            out_arr = logits_acc_np.astype(np.float32)
            if not self.is_2d_model and Z == 1:
                # Collapse singleton Z for 3D TIFFs that were actually 2D
                out_arr = out_arr[:, 0, :, :]

        tifffile.imwrite(str(out_path), out_arr, compression='zlib')
        return str(out_path)

    def _infer_single_tiff_large(self, img: np.ndarray, base_stem: str):
        # Compute coordinates and write per-patch logits/coords using existing writers
        if self.is_2d_model:
            if img.ndim != 2:
                raise ValueError("2D model expects 2D TIFF input (H, W)")
            Y, X = img.shape
            coords2d = self._compute_patch_positions((Y, X))
            coords = [(0, y, x) for (y, x) in coords2d]
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            Z, Y, X = img.shape
            coords = self._compute_patch_positions((Z, Y, X))
        self.patch_start_coords_list = coords
        self.num_total_patches = len(coords)
        # Create zarr output stores
        self._create_output_stores()

        normalization_scheme, gmean, gstd, iprops = self._resolve_normalization_for_tiff()
        if self.is_2d_model:
            pY, pX = self.patch_size
            pZ = 1
        else:
            pZ, pY, pX = self.patch_size

        write_index = 0
        with tqdm(total=self.num_total_patches, desc=f"Tiled TIFF {base_stem}") as pbar:
            for coord in coords:
                if self.is_2d_model:
                    _, y, x = coord
                    y1, x1 = min(y + pY, Y), min(x + pX, X)
                    sub = img[y:y1, x:x1]
                    sub = self._normalize_numpy(sub, normalization_scheme, gmean, gstd, iprops)
                    patch = np.zeros((pY, pX), dtype=np.float32)
                    patch[:sub.shape[0], :sub.shape[1]] = sub
                    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,pY,pX)
                else:
                    z, y, x = coord
                    z1, y1, x1 = min(z + pZ, Z), min(y + pY, Y), min(x + pX, X)
                    sub = img[z:z1, y:y1, x:x1]
                    sub = self._normalize_numpy(sub, normalization_scheme, gmean, gstd, iprops)
                    patch = np.zeros((pZ, pY, pX), dtype=np.float32)
                    patch[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.autocast(device_type=self.device.type):
                    if self.do_tta:
                        logits = infer_with_tta(
                            self.model,
                            t,
                            self.tta_type,
                            is_multi_task=self.is_multi_task,
                            concat_multi_task_outputs=self._concat_multi_task_outputs
                        )
                    else:
                        logits = self.model(t)
                        if self.is_multi_task:
                            logits = self._concat_multi_task_outputs(logits)
                # logits: (1,C,pZ,pY,pX) for 3D or (1,C,pY,pX) for 2D
                out_np = logits.detach().float().cpu().numpy()[0]
                if self.is_2d_model:
                    out_np = out_np[:, None, :, :]  # (C,1,pY,pX)
                self.output_store[write_index] = out_np
                write_index += 1
                pbar.update(1)

        main_output_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
        return main_output_path, self.coords_store_path

    def _infer_2d_over_slices_volume(self):
        # Prepare normalization
        normalization_scheme, gmean, gstd, iprops = self._resolve_normalization_for_tiff()

        # Initialize Volume
        vol_kwargs = dict(
            normalization_scheme=normalization_scheme,
            global_mean=gmean,
            global_std=gstd,
            intensity_props=iprops,
            return_as_type='np.float32',
            return_as_tensor=False,
            verbose=self.verbose
        )
        if self.input_format == 'zarr':
            volume = Volume(type='zarr', path=self.input, **vol_kwargs)
        elif self.input_format == 'volume':
            if self.segment_id is not None:
                volume = Volume(type='segment', segment_id=int(self.segment_id), scroll_id=self.scroll_id, energy=self.energy, resolution=self.resolution, **vol_kwargs)
            elif self.scroll_id is not None:
                volume = Volume(type='scroll', scroll_id=self.scroll_id, energy=self.energy, resolution=self.resolution, **vol_kwargs)
            else:
                raise ValueError("input_format='volume' requires scroll_id or segment_id for 2D model inference.")
        else:
            raise ValueError(f"Unsupported input_format '{self.input_format}' for 2D slice inference.")

        shape = volume.shape(0)
        if len(shape) == 4:
            channels, Z, Y, X = shape
        elif len(shape) == 3:
            channels, (Z, Y, X) = 1, shape
        else:
            raise ValueError(f"Unsupported volume shape for 2D inference: {shape}")

        if len(self.patch_size) == 2:
            patch_hw = tuple(int(v) for v in self.patch_size)
        elif len(self.patch_size) == 3 and self.patch_size[0] == 1:
            patch_hw = tuple(int(v) for v in self.patch_size[1:])
        else:
            raise ValueError(f"2D model requires 2D patch_size, got {self.patch_size}")

        patch_h, patch_w = patch_hw
        gaussian_map = generate_gaussian_map((1, patch_h, patch_w), sigma_scale=8.0, dtype=np.float32)[0][0]

        axes_to_process = list(self.slicewise_axes) if getattr(self, 'slicewise_axes', None) else ['z']
        axis_configs = {}
        if 'z' in axes_to_process:
            coords_z = self._compute_plane_positions((Y, X), patch_hw)
            axis_configs['z'] = (coords_z, Z)
        if 'y' in axes_to_process:
            coords_y = self._compute_plane_positions((Z, X), patch_hw)
            axis_configs['y'] = (coords_y, Y)
        if 'x' in axes_to_process:
            coords_x = self._compute_plane_positions((Z, Y), patch_hw)
            axis_configs['x'] = (coords_x, X)

        total_positions = 0
        for axis in axes_to_process:
            coords, count = axis_configs.get(axis, (None, None))
            if coords is not None:
                total_positions += len(coords) * count

        if total_positions == 0:
            raise RuntimeError("No slice positions generated for the requested slicewise planes.")

        volume_shape = (Z, Y, X)
        (logits_store,
         logits_sum_path,
         weights_store,
         weights_path,
         chunk_sizes) = self._create_volume_accumulators(volume_shape)

        self.patch_start_coords_list = []
        self.num_total_patches = total_positions
        self.original_volume_shape = volume_shape
        self.coords_store_path = None

        axis_plan = []
        for axis in axes_to_process:
            coords, count = axis_configs.get(axis, (None, None))
            if coords:
                axis_plan.append((axis, coords, count))

        desc = "2D slices over volume (multi-axis)"
        with tqdm(total=total_positions, desc=desc) as pbar:
            for axis_name, positions, slice_count in axis_plan:
                if not positions:
                    continue
                self._accumulate_axis_slices(
                    volume=volume,
                    axis=axis_name,
                    slice_positions=positions,
                    slice_count=slice_count,
                    patch_hw=patch_hw,
                    gaussian_map=gaussian_map,
                    logits_store=logits_store,
                    weights_store=weights_store,
                    channels=channels,
                    volume_shape=volume_shape,
                    pbar=pbar
                )

        final_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
        final_store = self._finalize_volume_logits(
            logits_store=logits_store,
            weights_store=weights_store,
            final_path=final_path,
            volume_shape=volume_shape,
            chunk_sizes=chunk_sizes
        )

        final_store.attrs['original_volume_shape'] = list(map(int, volume_shape))
        final_store.attrs['multi_axis_slicewise'] = True
        final_store.attrs['slicewise_axes'] = list(axes_to_process)
        final_store.attrs['patch_size_2d'] = list(map(int, patch_hw))
        final_store.attrs['overlap'] = float(self.overlap)
        final_store.attrs['part_id'] = self.part_id
        final_store.attrs['num_parts'] = self.num_parts
        final_store.attrs['processing_mode'] = self.output_mode

        if self.is_multi_task and self.target_info:
            final_store.attrs['is_multi_task'] = True
            final_store.attrs['target_info'] = self.target_info

        try:
            inference_args = {
                'model_path': str(self.model_path),
                'model_name': type(self.model).__name__ if self.model is not None else None,
                'input_dir': str(self.input),
                'input_format': str(self.input_format),
                'output_dir': str(self.output_dir),
                'tta_type': str(self.tta_type),
                'do_tta': bool(self.do_tta),
                'overlap': float(self.overlap),
                'batch_size': int(self.batch_size),
                'patch_size': list(self.patch_size),
                'normalization_scheme': str(self.model_normalization_scheme or self.normalization_scheme),
                'device': str(self.device),
                'skip_empty_patches': bool(self.skip_empty_patches),
                'scroll_id': self.scroll_id,
                'segment_id': self.segment_id,
                'energy': self.energy,
                'resolution': self.resolution,
                'compressor_name': str(self.compressor_name),
                'compression_level': int(self.compression_level),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'storage_mode': 'slicewise_multi_axis',
                'slicewise_axes': list(axes_to_process),
                'output_mode': self.output_mode
            }
            final_store.attrs['inference_args'] = inference_args
        except Exception as exc:
            if self.verbose:
                print(f"Warning: failed to attach inference metadata: {exc}")

        self.output_store = final_store

        del logits_store
        del weights_store

        self._cleanup_path(logits_sum_path)
        self._cleanup_path(weights_path)

        return final_path, None

    def _run_inference(self):
        if self.verbose: print("Loading model...")
        self.model = self._load_model()

        # TIFF path: handle separately if input is TIFF(s)
        if self._tiff_inputs:
            if tifffile is None:
                raise RuntimeError("tifffile is required to read TIFF inputs but is not installed")

            saved_outputs = []
            # Decide thresholds
            max_3d = (1024, 1024, 1024)
            max_2d = (10000, 10000)
            for tif in self._tiff_inputs:
                img = tifffile.imread(str(tif))
                if img.ndim == 2:
                    small_enough = (img.shape[0] <= max_2d[0] and img.shape[1] <= max_2d[1])
                elif img.ndim == 3:
                    small_enough = (img.shape[0] <= max_3d[0] and img.shape[1] <= max_3d[1] and img.shape[2] <= max_3d[2])
                else:
                    raise ValueError(f"Unsupported TIFF dimensionality: {img.shape}")

                if self.is_2d_model and img.ndim == 3:
                    mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')
                    out_suffix = "logits" if mode == 'none' else mode
                    out_name = f"{tif.stem}_{out_suffix}.tif"
                    out_path = Path(self.output_dir) / out_name
                    saved = self._infer_2d_tiff_volume(img, out_path)
                    saved_outputs.append(saved)
                    continue

                if small_enough:
                    # In-memory full-image blending, save final per-image output
                    # Name output according to activation mode
                    mode = self.tiff_activation if self.tiff_activation is not None else ('softmax' if self.save_softmax else 'argmax')
                    out_suffix = "logits" if mode == 'none' else mode
                    out_name = f"{tif.stem}_{out_suffix}.tif"
                    out_path = Path(self.output_dir) / out_name
                    saved = self._infer_single_tiff_in_memory(img, out_path)
                    saved_outputs.append(saved)
                else:
                    # Fallback to tiled logits/coords workflow
                    main_path, coords_path = self._infer_single_tiff_large(img, tif.stem)
                    saved_outputs.append(main_path)

            # Return a marker tuple that main() can recognize as TIFF mode
            return ("tiff", saved_outputs)
        else:
            if self.is_2d_model:
                # Run 2D model over each slice of the volume (zarr/volume)
                return self._infer_2d_over_slices_volume()
            if self.verbose: print("Creating dataset and dataloader...")
            self._create_dataset_and_loader()

            if self.num_total_patches > 0:
                if self.verbose: print("Creating output stores...")
                self._create_output_stores()

                if self.verbose: print("Starting inference and writing logits...")
                self._process_batches()
            else:
                print(f"Skipping processing for part {self.part_id} as no patches were found.")

        if self.verbose: print("Inference complete.")

    def infer(self):
        try:
            out = self._run_inference()
            # For TIFF mode, _run_inference returns ("tiff", [outputs...]); propagate it
            if isinstance(out, tuple) and len(out) == 2 and out[0] == 'tiff':
                return out
            main_output_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
            return main_output_path, self.coords_store_path
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
    parser.add_argument('--input_format', type=str, default='zarr', help='Input format (zarr, volume, tiff)')
    parser.add_argument('--tta_type', type=str, default='rotation', choices=['mirroring', 'rotation'], 
                      help='TTA type (mirroring or rotation). Default: rotation')
    parser.add_argument('--disable_tta', action='store_true', help='Disable test time augmentation')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, help='Part ID to process (0-indexed)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between patches (0-1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--patch_size', type=str, default=None, 
                      help='Optional: Override patch size, comma-separated (e.g., "192,192,192"). If not provided, uses the model\'s default patch size.')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass', 'surface_frame'], default='binary',
                      help='Output mode hint for downstream processing. Use "surface_frame" for 9-channel tangent-frame predictions.')
    parser.add_argument('--save_softmax', action='store_true', help='Save softmax outputs (deprecated; use --tif-activation softmax)')
    # Preferred flag for controlling TIFF activation
    parser.add_argument('--tif-activation', dest='tiff_activation', type=str, default=None,
                        choices=['softmax', 'argmax', 'none'],
                        help='Activation for TIFF outputs: softmax, argmax, or none (raw logits)')
    parser.add_argument('--slicewise-planes', type=str, default='z',
                        help="Comma-separated list of planes for 2D volume inference: choose from 'z','y','x' or 'all' for all axes")
    parser.add_argument('--normalization', type=str, default='instance_zscore',
                      help='Normalization scheme (instance_zscore, global_zscore, instance_minmax, ct, none)')
    parser.add_argument('--intensity-properties-json', type=str, default=None,
                      help='Path to nnU-Net style intensity properties JSON (dataset_fingerprint.json or intensity_props.json) for CT normalization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--config-yaml', dest='config_yaml', type=str, default=None,
                        help='Path to training YAML config to use if checkpoint lacks embedded model_config')
    parser.add_argument('--skip-empty-patches', dest='skip_empty_patches', action='store_true',
                      help='Skip patches that are empty (all values the same). Default: True')
    parser.add_argument('--no-skip-empty-patches', dest='skip_empty_patches', action='store_false',
                      help='Process all patches, even if they appear empty')
    parser.set_defaults(skip_empty_patches=True)
    
    # Add arguments for Zarr compression
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                      choices=['zstd', 'lz4', 'zlib', 'none'],
                      help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=3,
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

    slicewise_arg = (args.slicewise_planes or '').strip().lower()
    if not slicewise_arg:
        slicewise_axes = None
    elif slicewise_arg == 'all':
        slicewise_axes = ['z', 'y', 'x']
    else:
        slicewise_axes = [axis.strip() for axis in slicewise_arg.split(',') if axis.strip()]
        if not slicewise_axes:
            slicewise_axes = ['z']
        invalid_axes = [axis for axis in slicewise_axes if axis not in {'z', 'y', 'x'}]
        if invalid_axes:
            print(f"Invalid slicewise plane(s): {invalid_axes}. Use any of 'z','y','x', or 'all'.")
            return 1

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
        tiff_activation=args.tiff_activation,
        normalization_scheme=args.normalization,
        device=args.device,
        verbose=args.verbose,
        skip_empty_patches=args.skip_empty_patches,  # Skip empty patches flag
        # Intensity properties JSON for CT normalization
        # Store on the instance to be picked up in _resolve_normalization
        # (set after object creation below)
        # Pass Volume-specific parameters to VCDataset
        scroll_id=scroll_id,
        segment_id=segment_id,
        energy=args.energy,
        resolution=args.resolution,
        # Pass Zarr compression settings
        compressor_name=args.zarr_compressor,
        compression_level=args.zarr_compression_level,
        # Pass Hugging Face parameters
        hf_token=args.hf_token,
        # Fallback config for train.py checkpoints without embedded model_config
        config_yaml=args.config_yaml,
        slicewise_axes=slicewise_axes,
        output_mode=args.mode
    )

    try:
        # Stash intensity properties JSON path directly on inferer for resolution
        inferer.intensity_props_json = args.intensity_properties_json
        print("\n--- Starting Inference ---")
        result = inferer.infer()

        # Handle TIFF mode specially
        if isinstance(result, tuple) and len(result) == 2 and result[0] == 'tiff':
            outputs = result[1]
            print("\n--- TIFF Inference Finished ---")
            if outputs:
                print("Saved outputs:")
                for p in outputs:
                    print(f"  - {p}")
            return 0

        logits_path, coords_path = result
        if logits_path:
            logits_exists = False
            try:
                if logits_path.startswith('s3://'):
                    fs = fsspec.filesystem('s3', anon=False)
                    logits_exists = fs.exists(os.path.join(logits_path, '.zarray'))
                else:
                    logits_exists = os.path.exists(logits_path)
            except Exception as e:
                print(f"Warning: Could not verify if logits path exists: {e}")
                logits_exists = True

            if not logits_exists:
                print(f"\n--- Inference finished, but logits path doesn't seem to exist ---")
                print(f"Logits path: {logits_path} (exists: {logits_exists})")
                if coords_path:
                    print(f"Coordinates path: {coords_path}")
                return 1

            print(f"\n--- Inference Finished ---")
            print(f"Output logits saved to: {logits_path}")

            print("\n--- Inspecting Output Store ---")
            try:
                output_store = open_zarr(
                    path=logits_path,
                    mode='r',
                    storage_options={'anon': False} if logits_path.startswith('s3://') else None
                )
                print(f"Output shape: {output_store.shape}")
                print(f"Output dtype: {output_store.dtype}")
                print(f"Output chunks: {output_store.chunks}")
            except Exception as inspect_e:
                print(f"Could not inspect output Zarr: {inspect_e}")

            if (
                inferer.skip_empty_patches and
                getattr(inferer, 'dataset', None) is not None and
                hasattr(inferer.dataset, 'get_empty_patches_report')
            ):
                report = inferer.dataset.get_empty_patches_report()
                print("\n--- Empty Patches Report ---")
                print(f"  Empty Patches Skipped: {report['total_skipped']}")
                print(f"  Total Available Positions: {report['total_positions']}")
                if report['total_positions']:
                    skip_ratio = report.get('skip_ratio')
                    if skip_ratio is None:
                        skip_ratio = report['total_skipped'] / max(report['total_positions'], 1)
                    print(f"  Skip Ratio: {skip_ratio:.2%}")
                    if skip_ratio < 1.0:
                        print(f"  Effective Speedup: {1/(1-skip_ratio):.2f}x")

            if coords_path:
                coords_exists = False
                try:
                    if coords_path.startswith('s3://'):
                        fs = fsspec.filesystem('s3', anon=False)
                        coords_exists = fs.exists(os.path.join(coords_path, '.zarray'))
                    else:
                        coords_exists = os.path.exists(coords_path)
                except Exception as e:
                    print(f"Warning: Could not verify coordinate path: {e}")
                    coords_exists = True

                if coords_exists:
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
                else:
                    print(f"Coordinate path {coords_path} not found (expected for direct slicewise mode)")
            else:
                print("Coordinate store not produced (multi-axis slicewise mode writes final logits directly).")

            return 0

        print("\n--- Inference finished, but logits path is None ---")
        if coords_path:
            print(f"Coordinates path: {coords_path}")
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
