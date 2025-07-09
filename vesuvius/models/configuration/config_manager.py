from PIL import Image
import numpy as np
from pathlib import Path
from copy import deepcopy
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from utils.utils import determine_dimensionality


Image.MAX_IMAGE_PIXELS = None

class ConfigManager:
    def __init__(self, verbose):
        self._config_path = None
        self.data = None # note that config manager DOES NOT hold data, 
                         # it just holds the path to the data, currently an annoying holdover from old napari trainer
        self.verbose = verbose
        self.selected_loss_function = None

    def load_config(self, config_path):
        config_path = Path(config_path)
        self._config_path = config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.tr_info = config.get("tr_setup", {})
        self.tr_configs = config.get("tr_config", {})
        self.model_config = config.get("model_config", {}) 
        self.dataset_config = config.get("dataset_config", {})

        # Load targets from dataset_config or model_config if available
        self.targets = self.dataset_config.get("targets", {})
        if not self.targets and "targets" in self.model_config:
            self.targets = self.model_config.get("targets", {})

        # Load inference parameters directly
        infer_config = config.get("inference_config", {})
        self.infer_checkpoint_path = infer_config.get("checkpoint_path", None)
        self.infer_patch_size = infer_config.get("patch_size", None)
        self.infer_batch_size = infer_config.get("batch_size", None)
        self.infer_output_targets = infer_config.get("output_targets", ['all'])
        self.infer_overlap = infer_config.get("overlap", 0.50)
        self.load_strict = infer_config.get("load_strict", True)
        self.infer_num_dataloader_workers = infer_config.get("num_dataloader_workers", None)

        self.auxiliary_tasks = config.get("auxiliary_tasks", {})
        self._init_attributes()
    
        if self.auxiliary_tasks and self.targets:
            self._apply_auxiliary_tasks()

        return config

    def _init_attributes(self):


        self.model_name = self.tr_info.get("model_name", "Model")
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.compute_loss_on_labeled_only = bool(self.tr_info.get("compute_loss_on_labeled_only", False))

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
        if not self.ckpt_out_base.exists():
            self.ckpt_out_base.mkdir(parents=True)
        ckpt_path = self.tr_info.get("checkpoint_path", None)
        self.checkpoint_path = Path(ckpt_path) if ckpt_path else None
        self.load_weights_only = bool(self.tr_info.get("load_weights_only", False))

        ### Training config ### 
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.in_channels = 1
        self.train_batch_size = int(self.tr_configs.get("batch_size", 2))
        self.gradient_accumulation = int(self.tr_configs.get("gradient_accumulation", 1))
        self.max_steps_per_epoch = int(self.tr_configs.get("max_steps_per_epoch", 200))
        self.max_val_steps_per_epoch = int(self.tr_configs.get("max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(self.tr_configs.get("num_dataloader_workers", 4))
        self.max_epoch = int(self.tr_configs.get("max_epoch", 1000))
        self.optimizer = self.tr_configs.get("optimizer", "SGD")
        self.initial_lr = float(self.tr_configs.get("initial_lr", 0.0001))
        self.weight_decay = float(self.tr_configs.get("weight_decay", 0.0001))
        
        ### Dataset config ###
        self.min_labeled_ratio = float(self.dataset_config.get("min_labeled_ratio", 0.10))
        self.min_bbox_percent = float(self.dataset_config.get("min_bbox_percent", 0.95))

        # Skip patch validation -- consider all possible patch positions as valid
        self.skip_patch_validation = bool(self.dataset_config.get("skip_patch_validation", False))
        
        # Skip finding the minimum bounding box which would contain all the labels.
        # its a bit of a waste of computation when considering the downsampled zarr patches are quite fast to check
        self.skip_bounding_box = bool(self.dataset_config.get("skip_bounding_box", True))
        self.cache_valid_patches = bool(self.dataset_config.get("cache_valid_patches", True))
        
        # this horrific name is so you can set specific loss functions for specific label volumes,
        # say for example one volume doesn't have the same labels as the others.
        self.volume_task_loss_config = self.dataset_config.get("volume_task_loss_config", {})
        if self.volume_task_loss_config and self.verbose:
            print(f"Volume-task loss configuration loaded: {self.volume_task_loss_config}")
        
        # Label threshold - values below this will be set to 0, values at or above will be set to target_value
        self.label_threshold = self.dataset_config.get("label_threshold", None)
        if self.label_threshold is not None:
            self.label_threshold = float(self.label_threshold)
        
        # Spatial transformations control
        self.no_spatial = bool(self.dataset_config.get("no_spatial", False))

        # Normalization configuration
        self.normalization_scheme = self.dataset_config.get("normalization_scheme", "zscore")
        self.intensity_properties = self.dataset_config.get("intensity_properties", {})
        self.use_mask_for_norm = bool(self.dataset_config.get("use_mask_for_norm", False))                

        # model config

        # TODO: add support for timm encoders , will need a bit of refactoring as we'll
        # need to figure out the channels/feature map sizes to pass to the decoder
        # self.use_timm = self.model_config.get("use_timm", False)
        # self.timm_encoder_class = self.model_config.get("timm_encoder_class", None)

        # Determine dims for ops based on patch size
        dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
        self.model_config["conv_op"] = dim_props["conv_op"]
        self.model_config["pool_op"] = dim_props["pool_op"]
        self.model_config["norm_op"] = dim_props["norm_op"]
        self.model_config["dropout_op"] = dim_props["dropout_op"]
        self.spacing = dim_props["spacing"]
        self.op_dims = dim_props["op_dims"]

        # channel configuration
        self.in_channels = self.model_config.get("in_channels", 1)
        self.out_channels = ()
        for target_name, task_info in self.targets.items():
            # Look for either 'out_channels' or 'channels' in the task info
            if 'out_channels' in task_info:
                channels = task_info['out_channels']
            else:
                if self.verbose:
                    print(f"Target {target_name} has no channel specification - will attempt to auto-detect from data")
                channels = None  # Placeholder, will be set during auto-detection

            if channels is not None:
                self.out_channels += (channels,)

        # Inference attributes should already be set by _set_inference_attributes
        # If they weren't set (e.g., no inference_config in YAML), set defaults here
        if not hasattr(self, 'infer_checkpoint_path'):
            self.infer_checkpoint_path = None
        if not hasattr(self, 'infer_patch_size'):
            self.infer_patch_size = tuple(self.train_patch_size)
        if not hasattr(self, 'infer_batch_size'):
            self.infer_batch_size = int(self.train_batch_size)
        if not hasattr(self, 'infer_output_targets'):
            self.infer_output_targets = ['all']
        if not hasattr(self, 'infer_overlap'):
            self.infer_overlap = 0.50
        if not hasattr(self, 'load_strict'):
            self.load_strict = True
        if not hasattr(self, 'infer_num_dataloader_workers'):
            self.infer_num_dataloader_workers = int(self.train_num_dataloader_workers)

    def set_targets_and_data(self, targets_dict, data_dict):
        """
        Generic method to set targets and data from any source (napari, TIF, zarr, etc.)
        this is necessary primarily because the target dict has to be created/set , and the desired 
        loss functions have to be set for each target. it's a bit convoluted but i couldnt think of a simpler way 

        Parameters
        ----------
        targets_dict : dict
            Dictionary with target names as keys and target configuration as values
            Example: {"ink": {"out_channels": 1, "loss_fn": "BCEWithLogitsLoss", "activation": "sigmoid"}}
        data_dict : dict
            Dictionary with target names as keys and list of volume data as values
            Example: {"ink": [{"data": {...}, "out_channels": 1, "name": "image1_ink"}]}
        """
        self.targets = deepcopy(targets_dict)

        # Apply current loss function to all targets if not already set
        for target_name in self.targets:
            if "loss_fn" not in self.targets[target_name]:
                self.targets[target_name]["loss_fn"] = self.selected_loss_function

        # Apply auxiliary tasks to targets
        self._apply_auxiliary_tasks()

        # Only set out_channels if all targets have it defined, otherwise it will be auto-detected later
        if all('out_channels' in task_info for task_info in self.targets.values()):
            self.out_channels = tuple(task_info["out_channels"] for task_info in self.targets.values())
        else:
            self.out_channels = None  # Will be set during auto-detection


        if self.verbose:
            print(f"Set targets: {list(self.targets.keys())}")
            print(f"Output channels: {self.out_channels}")

        return data_dict

    def save_config(self):
        tr_setup = deepcopy(self.tr_info)
        tr_config = deepcopy(self.tr_configs)
        model_config = deepcopy(self.model_config)
        dataset_config = deepcopy(self.dataset_config)

        # Create inference_config from individual attributes
        inference_config = {
            "checkpoint_path": self.infer_checkpoint_path,
            "patch_size": list(self.infer_patch_size),
            "batch_size": self.infer_batch_size,
            "output_targets": self.infer_output_targets,
            "overlap": self.infer_overlap,
            "load_strict": self.load_strict,
            "num_dataloader_workers": self.infer_num_dataloader_workers
        }

        if hasattr(self, 'targets') and self.targets:
            dataset_config["targets"] = deepcopy(self.targets)

            model_config["targets"] = deepcopy(self.targets)

            if self.verbose:
                print(f"Saving targets to config: {self.targets}")

        combined_config = {
            "tr_setup": tr_setup,
            "tr_config": tr_config,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "inference_config": inference_config,
        }

        model_ckpt_dir = Path(self.ckpt_out_base) / self.model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)
        config_filename = f"{self.model_name}_config.yaml"
        config_path = model_ckpt_dir / config_filename

        with config_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {config_path}")

    def update_config(self, patch_size=None, min_labeled_ratio=None, max_epochs=None, loss_function=None, 
                     binarize_labels=None, target_value=None, skip_patch_validation=None,
                     normalization_scheme=None, intensity_properties=None, label_threshold=None,
                     skip_bounding_box=None):
        if patch_size is not None:
            if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
                self.train_patch_size = tuple(patch_size)
                self.tr_configs["patch_size"] = list(patch_size)

                dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
                self.model_config["conv_op"] = dim_props["conv_op"]
                self.model_config["pool_op"] = dim_props["pool_op"]
                self.model_config["norm_op"] = dim_props["norm_op"]
                self.model_config["dropout_op"] = dim_props["dropout_op"]
                self.spacing = dim_props["spacing"]
                self.op_dims = dim_props["op_dims"]

                if self.verbose:
                    print(f"Updated patch size: {self.train_patch_size}")

        if min_labeled_ratio is not None:
            self.min_labeled_ratio = float(min_labeled_ratio)
            self.dataset_config["min_labeled_ratio"] = self.min_labeled_ratio
            if self.verbose:
                print(f"Updated min labeled ratio: {self.min_labeled_ratio:.2f}")

        if binarize_labels is not None:
            self.binarize_labels = bool(binarize_labels)
            self.dataset_config["binarize_labels"] = self.binarize_labels
            if self.verbose:
                print(f"Updated binarize_labels: {self.binarize_labels}")

        if target_value is not None:
            self.target_value = target_value
            self.dataset_config["target_value"] = self.target_value
            if self.verbose:
                print(f"Updated target_value: {self.target_value}")

        if skip_patch_validation is not None:
            self.skip_patch_validation = bool(skip_patch_validation)
            self.dataset_config["skip_patch_validation"] = self.skip_patch_validation
            if self.verbose:
                print(f"Updated skip_patch_validation: {self.skip_patch_validation}")

        if loss_function is not None:
            self.selected_loss_function = loss_function
            if hasattr(self, 'targets') and self.targets:
                for target_name in self.targets:
                    self.targets[target_name]["loss_fn"] = self.selected_loss_function
                if self.verbose:
                    print(f"Applied loss function '{self.selected_loss_function}' to all targets")
            elif self.verbose:
                print(f"Set loss function: {self.selected_loss_function}")

        if normalization_scheme is not None:
            self.normalization_scheme = normalization_scheme
            self.dataset_config["normalization_scheme"] = self.normalization_scheme
            if self.verbose:
                print(f"Updated normalization scheme: {self.normalization_scheme}")

        if intensity_properties is not None:
            self.intensity_properties = intensity_properties
            self.dataset_config["intensity_properties"] = self.intensity_properties
            if self.verbose:
                print(f"Updated intensity properties: {self.intensity_properties}")

        if label_threshold is not None:
            self.label_threshold = float(label_threshold) if label_threshold is not None else None
            self.dataset_config["label_threshold"] = self.label_threshold
            if self.verbose:
                print(f"Updated label_threshold: {self.label_threshold}")

        if skip_bounding_box is not None:
            self.skip_bounding_box = bool(skip_bounding_box)
            self.dataset_config["skip_bounding_box"] = self.skip_bounding_box
            if self.verbose:
                print(f"Updated skip_bounding_box: {self.skip_bounding_box}")

    def _apply_auxiliary_tasks(self):
        """
        Apply auxiliary tasks by adding them to the targets dictionary.
        """
        if not self.auxiliary_tasks:
            return
            
        for aux_task_name, aux_config in self.auxiliary_tasks.items():
            task_type = aux_config["type"]
            
            if task_type == "distance_transform":
                source_target = aux_config["source_target"]
            
                if source_target not in self.targets:
                    raise ValueError(f"Source target '{source_target}' for auxiliary task '{aux_task_name}' not found in targets")
                
                aux_target_name = f"{aux_task_name}"  # dt = distance transform
                target_config = {
                    "out_channels": aux_config.get("out_channels", 1),
                    "activation": "none", 
                    "auxiliary_task": True,
                    "task_type": "distance_transform",
                    "source_target": source_target,
                    "weight": aux_config.get("loss_weight", 1.0)
                }
                

                target_config["losses"] = aux_config["losses"]
                self.targets[aux_target_name] = target_config
                
                if self.verbose:
                    print(f"Added distance transform auxiliary task '{aux_target_name}' from source '{source_target}'")
                    
            elif task_type == "surface_normals":
                source_target = aux_config["source_target"]
                
                if source_target not in self.targets:
                    raise ValueError(f"Source target '{source_target}' for auxiliary task '{aux_task_name}' not found in targets")
                
                aux_target_name = f"{aux_task_name}"
                target_config = {
                    "out_channels": aux_config.get("out_channels", None), 
                    "activation": "none", 
                    "auxiliary_task": True,
                    "task_type": "surface_normals",
                    "source_target": source_target,
                    "weight": aux_config.get("loss_weight", 1.0),  
                    "use_source_mask": True  
                }
                

                target_config["losses"] = aux_config["losses"]
                self.targets[aux_target_name] = target_config
                
                if self.verbose:
                    print(f"Added surface normals auxiliary task '{aux_target_name}' from source '{source_target}'")
                    
        if self.verbose and self.auxiliary_tasks:
            print(f"Applied {len(self.auxiliary_tasks)} auxiliary tasks to targets")

    def auto_detect_channels(self, dataset):
        """
        Automatically detect the number of output channels for each target from the dataset.
        
        Parameters
        ----------
        dataset : BaseDataset
            The dataset to inspect for channel information
        """
        if not dataset or len(dataset) == 0:
            print("Warning: Empty dataset, cannot auto-detect channels")
            return
            
        # Get a sample batch to inspect
        sample = dataset[0]
        
        # Update targets with detected channels
        targets_updated = False
        for target_name in self.targets:
            if 'out_channels' not in self.targets[target_name] or self.targets[target_name].get('out_channels') is None:
                if target_name in sample:
                    # Get the label tensor for this target
                    label_tensor = sample[target_name]
                    
                    # Determine number of channels based on label data
                    # For binary labels (0/1), we always use 2 channels minimum
                    unique_values = torch.unique(label_tensor)
                    num_unique = len(unique_values)
                    
                    if num_unique <= 2:
                        # Binary case - always use 2 channels (background/foreground)
                        detected_channels = 2
                    else:
                        # Multi-class case - use max value + 1
                        detected_channels = int(torch.max(label_tensor).item()) + 1
                        # Ensure at least 2 channels
                        detected_channels = max(detected_channels, 2)
                    
                    self.targets[target_name]['out_channels'] = detected_channels
                    targets_updated = True
                    
                    if self.verbose:
                        print(f"Auto-detected {detected_channels} channels for target '{target_name}'")
                    

        if targets_updated:
            self.out_channels = tuple(
                self.targets[t_name].get('out_channels', 2) 
                for t_name in self.targets
            )
            if self.verbose:
                print(f"Updated output channels: {self.out_channels}")
    
    def _print_summary(self):
        print("____________________________________________")
        print("Training Setup (tr_info):")
        for k, v in self.tr_info.items():
            print(f"  {k}: {v}")

        print("\nTraining Config (tr_configs):")
        for k, v in self.tr_configs.items():
            print(f"  {k}: {v}")

        print("\nDataset Config (dataset_config):")
        for k, v in self.dataset_config.items():
            print(f"  {k}: {v}")

        print("\nInference Config:")
        print(f"  checkpoint_path: {self.infer_checkpoint_path}")
        print(f"  patch_size: {self.infer_patch_size}")
        print(f"  batch_size: {self.infer_batch_size}")
        print(f"  output_targets: {self.infer_output_targets}")
        print(f"  overlap: {self.infer_overlap}")
        print(f"  load_strict: {self.load_strict}")
        print(f"  num_dataloader_workers: {self.infer_num_dataloader_workers}")
        print("____________________________________________")
