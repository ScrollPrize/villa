from PIL import Image
import numpy as np
from pathlib import Path
from copy import deepcopy
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from vesuvius.utils.utils import determine_dimensionality
from vesuvius.models.training.auxiliary_tasks import create_auxiliary_task
import os


Image.MAX_IMAGE_PIXELS = None

class ConfigManager:
    def __init__(self, verbose):
        self._config_path = None
        self.data = None # note that config manager DOES NOT hold data, 
                         # it just holds the path to the data, currently an annoying holdover from old napari trainer
        self.verbose = verbose
        self.selected_loss_function = "nnUNet_DC_and_CE_loss"

    def load_config(self, config_path):
        config_path = Path(config_path)
        self._config_path = config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.tr_info = config.get("tr_setup", {})
        self.tr_configs = config.get("tr_config", {})
        self.model_config = config.get("model_config", {}) 
        self.dataset_config = config.get("dataset_config", {})

        data_path_cfg = self.dataset_config.get("data_path")
        if data_path_cfg is None:
            self.data_path = config_path.parent.resolve()
        else:
            candidate = Path(data_path_cfg)
            if not candidate.is_absolute():
                candidate = (config_path.parent / candidate).resolve()
            self.data_path = candidate

        # Load targets from dataset_config or model_config if available
        raw_targets = self.dataset_config.get("targets") or {}
        if not raw_targets and "targets" in self.model_config:
            raw_targets = self.model_config.get("targets") or {}

        if raw_targets:
            self.validate_target_names(raw_targets.keys())

        self.targets = {}
        for target_name, target_info in raw_targets.items():
            info_dict = dict(target_info or {})
            if 'out_channels' not in info_dict and 'channels' not in info_dict:
                info_dict['out_channels'] = 2
            self.targets[target_name] = info_dict

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

        # Load mean teacher / semi-supervised trainer config
        # These parameters are used by TrainMeanTeacher and TrainUncertaintyAwareMeanTeacher
        mean_teacher_config = config.get("mean_teacher_config", {})
        for key, value in mean_teacher_config.items():
            setattr(self, key, value)

        # Load LeJEPA config - flatten into attributes
        # These parameters are used by TrainLeJEPA
        lejepa_config = config.get("lejepa_config", {})
        for key, value in lejepa_config.items():
            # Handle 'lambda' specially since it's a Python reserved word
            # and trainer expects 'lejepa_lambda' attribute name
            attr_name = "lejepa_lambda" if key == "lambda" else key
            setattr(self, attr_name, value)

        self._init_attributes()

        if self.auxiliary_tasks and self.targets:
            self._apply_auxiliary_tasks()

        return config

    def _explicit_volumes_require_unlabeled(self) -> bool:
        volumes_cfg = self.dataset_config.get("volumes")
        if not volumes_cfg:
            return False

        if isinstance(volumes_cfg, dict):
            volume_specs = list(volumes_cfg.values())
        elif isinstance(volumes_cfg, (list, tuple)):
            volume_specs = list(volumes_cfg)
        else:
            raise ValueError("dataset_config.volumes must be a list or mapping")

        target_names = list((self.targets or {}).keys())

        for spec in volume_specs:
            if spec is None:
                return True
            if isinstance(spec, (str, Path)):
                # Treat as image-only entry; labels are missing.
                return True
            if not isinstance(spec, dict):
                raise ValueError("Each volume entry must be a mapping or path string")

            labels = spec.get("labels")
            if labels is None:
                labels = spec.get("label_paths")
            if labels is None or labels == {}:
                return True
            if not isinstance(labels, dict):
                raise ValueError("Volume labels must be a mapping of target -> path")

            if not target_names:
                # If targets are unknown, assume labels are incomplete.
                return True

            for target in target_names:
                value = labels.get(target)
                if value in (None, ""):
                    return True

        return False

    def _init_attributes(self):


        self.model_name = self.tr_info.get("model_name", "Model")
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.90))
        self.compute_loss_on_labeled_only = bool(self.tr_info.get("compute_loss_on_labeled_only", False))
        self.wandb_project = self.tr_info.get("wandb_project", None)
        self.wandb_entity = self.tr_info.get("wandb_entity", None)

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
        if not self.ckpt_out_base.exists():
            self.ckpt_out_base.mkdir(parents=True)
        ckpt_path = self.tr_info.get("checkpoint_path", None)
        self.checkpoint_path = Path(ckpt_path) if ckpt_path else None
        self.load_weights_only = bool(self.tr_info.get("load_weights_only", False))

        ### Training config ### 
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        # Preserve the user-specified patch size before any slice-mode adjustments
        self._original_train_patch_size = tuple(self.train_patch_size)
        self.in_channels = 1
        self.train_batch_size = int(self.tr_configs.get("batch_size", 2))
        # Enable nnUNet-style deep supervision (disabled by default)
        self.enable_deep_supervision = bool(self.tr_configs.get("enable_deep_supervision", True))
        self.gradient_accumulation = int(self.tr_configs.get("gradient_accumulation", 1))
        self.max_steps_per_epoch = int(self.tr_configs.get("max_steps_per_epoch", 250))
        self.max_val_steps_per_epoch = int(self.tr_configs.get("max_val_steps_per_epoch", 50))
        self.train_num_dataloader_workers = int(self.tr_configs.get("num_dataloader_workers", 8))
        self.max_epoch = int(self.tr_configs.get("max_epoch", 5000))
        self.optimizer = self.tr_configs.get("optimizer", "SGD")
        self.initial_lr = float(self.tr_configs.get("initial_lr", 0.01))
        self.weight_decay = float(self.tr_configs.get("weight_decay", 0.00003))
        
        ### Dataset config ###
        self.min_labeled_ratio = float(self.dataset_config.get("min_labeled_ratio", 0.10))
        self.min_bbox_percent = float(self.dataset_config.get("min_bbox_percent", 0.95))

        mesh_cfg = self.dataset_config.get("meshes", {}) or {}
        allow_unlabeled_cfg = self.dataset_config.get("allow_unlabeled_data")
        if allow_unlabeled_cfg is None:
            allow_unlabeled_cfg = self.dataset_config.get("allow_unlabeled")
        if allow_unlabeled_cfg is None and self._explicit_volumes_require_unlabeled():
            allow_unlabeled_cfg = True
        if allow_unlabeled_cfg is None and mesh_cfg.get("enabled"):
            allow_unlabeled_cfg = True
        self.allow_unlabeled_data = bool(allow_unlabeled_cfg)
        if allow_unlabeled_cfg is not None:
            self.dataset_config["allow_unlabeled_data"] = bool(allow_unlabeled_cfg)

        # Skip patch validation -- consider all possible patch positions as valid
        self.skip_patch_validation = bool(self.dataset_config.get("skip_patch_validation", False))
        
        # Skip finding the minimum bounding box which would contain all the labels.
        # its a bit of a waste of computation when considering the downsampled zarr patches are quite fast to check
        self.skip_bounding_box = bool(self.dataset_config.get("skip_bounding_box", True))
        self.cache_valid_patches = bool(self.dataset_config.get("cache_valid_patches", True))

        # BG-only patch sampling configuration
        self.bg_sampling_enabled = bool(self.dataset_config.get("bg_sampling_enabled", False))
        self.bg_to_fg_ratio = float(self.dataset_config.get("bg_to_fg_ratio", 0.5))

        # Validate BG sampling requirements: must have ignore_label configured
        if self.bg_sampling_enabled:
            has_ignore_label = any(
                target_info.get("ignore_label") is not None or
                target_info.get("ignore_index") is not None or
                target_info.get("ignore_value") is not None
                for target_info in self.targets.values()
            )
            if not has_ignore_label:
                raise ValueError(
                    "bg_sampling_enabled requires at least one target with ignore_label configured. "
                    "Without ignore_label, unlabeled areas default to 0 making BG detection unreliable."
                )

        # Unlabeled foreground detection for semi-supervised learning
        self.unlabeled_foreground_enabled = bool(self.dataset_config.get("unlabeled_foreground_enabled", False))
        self.unlabeled_foreground_threshold = float(self.dataset_config.get("unlabeled_foreground_threshold", 0.05))
        self.unlabeled_foreground_bbox_threshold = float(self.dataset_config.get("unlabeled_foreground_bbox_threshold", 0.15))
        # List of volume IDs to scan for unlabeled foreground (opt-in)
        self.unlabeled_foreground_volumes = self.dataset_config.get("unlabeled_foreground_volumes", None)

        rotation_axes_cfg = self.dataset_config.get("rotation_axes", None)
        axis_name_to_index = {
            'z': 0,
            'depth': 0,
            'd': 0,
            'y': 1,
            'height': 1,
            'h': 1,
            'x': 2,
            'width': 2,
            'w': 2,
        }
        index_to_axis_name = {0: 'z', 1: 'y', 2: 'x'}
        if rotation_axes_cfg is None:
            self.allowed_rotation_axes = None
        else:
            if isinstance(rotation_axes_cfg, (str, int)):
                rotation_axes_iterable = [rotation_axes_cfg]
            else:
                rotation_axes_iterable = rotation_axes_cfg

            normalized_axes = []
            seen_axes = set()
            for token in rotation_axes_iterable:
                if isinstance(token, int):
                    idx = token
                else:
                    token_str = str(token).strip().lower()
                    if token_str in {'all', 'xyz', 'zyx'}:
                        normalized_axes = [0, 1, 2]
                        seen_axes = {0, 1, 2}
                        break
                    if token_str in axis_name_to_index:
                        idx = axis_name_to_index[token_str]
                    else:
                        try:
                            idx = int(token_str)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid axis '{token}' in dataset_config.rotation_axes; "
                                "use x,y,z (width,height,depth) or indices 0-2."
                            ) from exc
                if idx < 0 or idx > 2:
                    raise ValueError(
                        f"Axis index '{idx}' in dataset_config.rotation_axes is out of range; valid indices are 0, 1, 2."
                    )
                if idx not in seen_axes:
                    normalized_axes.append(idx)
                    seen_axes.add(idx)

            self.allowed_rotation_axes = tuple(normalized_axes)
            # Store canonical names for downstream logging
            self.dataset_config['rotation_axes'] = [index_to_axis_name[idx] for idx in normalized_axes]

        # Chunk-slicing worker configuration
        self.valid_patch_find_resolution = int(self.dataset_config.get("valid_patch_find_resolution", 1))
        self.num_workers = int(self.dataset_config.get("num_workers", 8))

        # Worker configuration for image→Zarr pipeline
        # Parallel workers for initial image/Zarr checks
        self.image_check_workers = int(self.dataset_config.get(
            "image_check_workers", max(1, os.cpu_count() // 4)
        ))
        # Parallel workers for actual image→Zarr writes
        self.image_to_zarr_workers = int(self.dataset_config.get(
            "image_to_zarr_workers", max(1, os.cpu_count() // 4)
        ))

        # this horrific name is so you can set specific loss functions for specific label volumes,
        # say for example one volume doesn't have the same labels as the others.
        self.volume_task_loss_config = self.dataset_config.get("volume_task_loss_config", {})
        if self.volume_task_loss_config and self.verbose:
            print(f"Volume-task loss configuration loaded: {self.volume_task_loss_config}")


        # Slice sampling configuration: train 2D slices from 3D inputs
        self.slice_sampling_config = self.dataset_config.get("slice_sampling", {}) or {}
        self.slice_sampling_enabled = bool(self.slice_sampling_config.get("enabled", False))
        self.slice_sample_planes = []
        self.slice_plane_weights = {}
        self.slice_plane_patch_sizes = {}
        self.slice_primary_plane = None

        self.slice_random_rotation_planes = {}
        self.slice_random_tilt_planes = {}
        self.slice_label_interpolation = {}
        self.slice_save_plane_masks = False
        self.slice_plane_mask_mode = 'plane'
        self.slice_label_interpolation = {}

        if self.slice_sampling_enabled:
            planes_cfg = self.slice_sampling_config.get("sample_planes", ["z"])
            sample_rates_cfg = self.slice_sampling_config.get("sample_rates", None)
            planes, weights = self._normalize_slice_sampling_planes(planes_cfg, sample_rates_cfg)

            if not planes:
                planes = ["z"]
                weights = {"z": 1.0}

            custom_sizes = self.slice_sampling_config.get("plane_patch_sizes", {})
            patch_sizes = self._resolve_slice_plane_patch_sizes(
                planes=planes,
                base_patch_size=self._original_train_patch_size,
                custom_sizes=custom_sizes
            )

            self.slice_sample_planes = planes
            self.slice_plane_weights = weights
            self.slice_plane_patch_sizes = patch_sizes
            self.slice_primary_plane = planes[0]

            rotation_cfg = self.slice_sampling_config.get("random_rotations")
            if rotation_cfg:
                self.slice_random_rotation_planes = self._normalize_slice_random_rotations(rotation_cfg, planes)

            tilt_cfg = self.slice_sampling_config.get("random_tilts")
            if tilt_cfg:
                self.slice_random_tilt_planes = self._normalize_slice_random_tilts(tilt_cfg, planes)

            interp_cfg = self.slice_sampling_config.get("label_interpolation")
            if interp_cfg:
                self.slice_label_interpolation = self._normalize_slice_label_interpolation(interp_cfg)

            self.slice_save_plane_masks = bool(self.slice_sampling_config.get("save_plane_masks", False))

            mask_mode = str(self.slice_sampling_config.get("plane_mask_mode", "plane")).lower()
            if mask_mode not in {"volume", "plane"}:
                if self.verbose:
                    print(f"Unrecognized slice plane mask mode '{mask_mode}', defaulting to 'plane'.")
                mask_mode = 'plane'
            self.slice_plane_mask_mode = mask_mode

            interp_cfg = self.slice_sampling_config.get("label_interpolation")
            if interp_cfg:
                self.slice_label_interpolation = self._normalize_slice_label_interpolation(interp_cfg)

            # Force downstream components to operate in 2D using the primary plane's patch size
            primary_size = patch_sizes.get(self.slice_primary_plane)
            if primary_size is None or len(primary_size) != 2:
                raise ValueError(
                    "Slice sampling requires a 2D patch size for the primary plane; "
                    f"got {primary_size} for plane '{self.slice_primary_plane}'."
                )

            self.train_patch_size = tuple(int(v) for v in primary_size)

            # Ensure dataset detection treats volumes as 2D slices even if stored as 3D
            self.force_2d = True
            self.force_3d = False

            if self.verbose:
                plane_desc = ", ".join(
                    f"{axis}: size={patch_sizes[axis]}, weight={weights[axis]:.3f}" for axis in planes
                )
                print(f"Slice sampling enabled across planes [{plane_desc}]")


        # Spatial transformations control
        self.no_spatial = bool(self.dataset_config.get("no_spatial", False))
        # Control where augmentations run; default to CPU (in Dataset workers)
        self.augment_on_device = bool(self.tr_configs.get("augment_on_device", False))
        # Enable per-augmentation timing (disabled by default)
        self.profile_augmentations = bool(self.tr_configs.get("profile_augmentations", False))

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
        self.model_config["norm_op"] = dim_props["norm_op"]
        self.spacing = dim_props["spacing"]
        self.op_dims = dim_props["op_dims"]

        # channel configuration
        self.in_channels = self.model_config.get("in_channels", 1)
        self.out_channels = ()
        for target_name, task_info in self.targets.items():
            # Determine channels, with special handling for derived tasks
            if 'out_channels' in task_info:
                channels = task_info['out_channels']
            elif 'channels' in task_info:
                channels = task_info['channels']
            else:
                task_type = str(task_info.get('task_type', '')).lower()
                if task_type == 'nearest_component':
                    # vector (op_dims) + 1 scalar distance
                    ch = (self.op_dims if hasattr(self, 'op_dims') and self.op_dims in (2, 3) else len(self.train_patch_size)) + 1
                    channels = ch
                    task_info['out_channels'] = ch
                else:
                    channels = 2  # Default to 2
                    task_info['out_channels'] = 2

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

    def _normalize_slice_sampling_planes(self, planes_cfg, rates_cfg):
        """Normalize plane selection and weights for slice sampling."""

        def normalize_axes(raw_axes):
            normalized = []
            seen = set()
            for axis in raw_axes:
                axis_norm = str(axis).lower()
                if axis_norm not in {"x", "y", "z"}:
                    raise ValueError(f"Unsupported slice sampling plane '{axis}'. Use any of: x, y, z")
                if axis_norm not in seen:
                    normalized.append(axis_norm)
                    seen.add(axis_norm)
            return normalized

        def to_positive_float(value, context_label):
            if value is None:
                return 1.0
            try:
                val = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Slice sampling {context_label} must be numeric; got {value!r}")
            if val <= 0:
                raise ValueError(f"Slice sampling {context_label} must be > 0; got {val}")
            return val

        # Determine plane order
        if isinstance(planes_cfg, dict):
            plane_list = normalize_axes(planes_cfg.keys())
            weights = {axis: to_positive_float(planes_cfg.get(axis), f"sample_planes[{axis}]") for axis in plane_list}
            return plane_list, weights

        if isinstance(planes_cfg, str):
            raw_planes = [part.strip() for part in planes_cfg.split(',') if part.strip()]
            plane_list = normalize_axes(raw_planes)
        elif isinstance(planes_cfg, (list, tuple, set)):
            plane_list = normalize_axes(planes_cfg)
        else:
            raise ValueError(
                "slice_sampling.sample_planes must be a string, list, or mapping of plane->weight"
            )

        if not plane_list:
            return [], {}

        # Determine weights
        weights = {}
        if isinstance(rates_cfg, dict):
            for axis in plane_list:
                weights[axis] = to_positive_float(rates_cfg.get(axis), f"sample_rates[{axis}]")
        elif isinstance(rates_cfg, (list, tuple)):
            if len(rates_cfg) == 0:
                weights = {axis: 1.0 for axis in plane_list}
            else:
                last_val = rates_cfg[-1]
                for idx, axis in enumerate(plane_list):
                    value = rates_cfg[idx] if idx < len(rates_cfg) else last_val
                    weights[axis] = to_positive_float(value, f"sample_rates[{axis}]")
        elif isinstance(rates_cfg, str):
            parts = [part.strip() for part in rates_cfg.split(',') if part.strip()]
            if parts:
                last_val = parts[-1]
                for idx, axis in enumerate(plane_list):
                    value = parts[idx] if idx < len(parts) else last_val
                    weights[axis] = to_positive_float(value, f"sample_rates[{axis}]")
        elif rates_cfg is not None:
            scalar = to_positive_float(rates_cfg, "sample_rates")
            weights = {axis: scalar for axis in plane_list}

        if not weights:
            weights = {axis: 1.0 for axis in plane_list}

        # Ensure all planes have weights (fill missing with 1.0)
        for axis in plane_list:
            weights.setdefault(axis, 1.0)

        return plane_list, weights

    def _resolve_slice_plane_patch_sizes(self, planes, base_patch_size, custom_sizes):
        """Compute per-plane 2D patch sizes for slice sampling."""

        def to_size_tuple(value, axis):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    h = int(value[0])
                    w = int(value[1])
                except (TypeError, ValueError):
                    raise ValueError(f"slice_sampling.plane_patch_sizes[{axis}] must contain integers")
                if h <= 0 or w <= 0:
                    raise ValueError(f"slice_sampling.plane_patch_sizes[{axis}] must be > 0; got {(h, w)}")
                return (h, w)
            raise ValueError(
                "slice_sampling.plane_patch_sizes entries must be two-element sequences of positive integers"
            )

        resolved = {}

        if custom_sizes and not isinstance(custom_sizes, dict):
            raise ValueError("slice_sampling.plane_patch_sizes must be a mapping of plane->[dim0, dim1]")

        if isinstance(custom_sizes, dict):
            for axis, size in custom_sizes.items():
                axis_norm = str(axis).lower()
                if axis_norm not in {"x", "y", "z"}:
                    raise ValueError(f"Unsupported plane '{axis}' in plane_patch_sizes")
                resolved[axis_norm] = to_size_tuple(size, axis_norm)

        base = tuple(int(v) for v in base_patch_size) if base_patch_size else ()
        if len(base) not in (2, 3):
            raise ValueError(
                f"Patch size for slice sampling must have 2 or 3 elements; got {base_patch_size}"
            )

        if len(base) == 3:
            dz, dy, dx = base
        else:  # len(base) == 2
            dy, dx = base
            dz = dy  # fallback depth so y/x planes still have a defined size

        defaults = {
            'z': (dy, dx),
            'y': (dz, dx),
            'x': (dz, dy)
        }

        for axis in planes:
            if axis in resolved:
                continue
            if len(base) == 2:
                resolved[axis] = (dy, dx)
            else:
                resolved[axis] = defaults[axis]

        return resolved

    def _normalize_slice_random_rotations(self, rotation_cfg, planes):
        """Normalize random rotation settings for slice sampling planes."""

        def _positive_float(val, label):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Slice sampling {label} must be numeric; got {val!r}")
            if fval <= 0:
                raise ValueError(f"Slice sampling {label} must be > 0; got {fval}")
            return fval

        def _probability(val, label):
            if val is None:
                return 1.0
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Slice sampling {label} must be numeric; got {val!r}")
            if fval < 0:
                raise ValueError(f"Slice sampling {label} must be >= 0; got {fval}")
            if fval > 1.0:
                if fval > 100.0:
                    raise ValueError(
                        f"Slice sampling {label} must be <= 100 when expressed as percent; got {fval}"
                    )
                fval = fval / 100.0
            if fval > 1.0:
                raise ValueError(
                    f"Slice sampling {label} must be between 0 and 1 after conversion; got {fval}"
                )
            return fval

        def _normalize_entry(value, axis):
            if isinstance(value, dict):
                max_deg = value.get('max_degrees', value.get('max_degree', value.get('degrees', None)))
                if max_deg is None:
                    raise ValueError(
                        f"slice_sampling.random_rotations[{axis}] must provide 'max_degrees' when using a mapping"
                    )
                probability = value.get('probability', value.get('prob', value.get('pct', value.get('percentage'))))
                return {
                    'max_degrees': _positive_float(max_deg, f"random_rotations[{axis}].max_degrees"),
                    'probability': _probability(probability, f"random_rotations[{axis}].probability")
                }
            return {
                'max_degrees': _positive_float(value, f"random_rotations[{axis}]") ,
                'probability': 1.0
            }

        normalized = {}

        if isinstance(rotation_cfg, bool):
            if rotation_cfg:
                for axis in planes:
                    normalized[axis] = {
                        'max_degrees': 360.0,
                        'probability': 1.0
                    }
            return normalized

        if isinstance(rotation_cfg, (list, tuple, set)):
            for axis in rotation_cfg:
                axis_norm = str(axis).lower()
                if axis_norm not in planes:
                    continue
                normalized[axis_norm] = {
                    'max_degrees': 360.0,
                    'probability': 1.0
                }
            return normalized

        if isinstance(rotation_cfg, dict):
            for axis, value in rotation_cfg.items():
                axis_norm = str(axis).lower()
                if axis_norm not in planes:
                    continue
                normalized[axis_norm] = _normalize_entry(value, axis_norm)
            return normalized

        raise ValueError(
            "slice_sampling.random_rotations must be a boolean, list, or mapping of plane->degrees"
        )

    def _normalize_slice_random_tilts(self, tilt_cfg, planes):
        """Normalize random tilt settings (per-axis) for slice sampling planes."""

        def to_positive_float(value, context_label):
            try:
                val = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Slice sampling {context_label} must be numeric; got {value!r}")
            if val <= 0:
                raise ValueError(f"Slice sampling {context_label} must be > 0; got {val}")
            return val

        def to_probability(value, context_label):
            if value is None:
                return 1.0
            try:
                prob = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Slice sampling {context_label} must be numeric; got {value!r}")
            if prob < 0:
                raise ValueError(f"Slice sampling {context_label} must be >= 0; got {prob}")
            if prob > 1.0:
                if prob > 100.0:
                    raise ValueError(
                        f"Slice sampling {context_label} must be <= 100 when expressed as percent; got {prob}"
                    )
                prob = prob / 100.0
            if prob > 1.0:
                raise ValueError(
                    f"Slice sampling {context_label} must be between 0 and 1 after conversion; got {prob}"
                )
            return prob

        if not isinstance(tilt_cfg, dict):
            raise ValueError("slice_sampling.random_tilts must be a mapping of plane->axis tilt settings")

        normalized = {}
        for plane, axis_cfg in tilt_cfg.items():
            plane_norm = str(plane).lower()
            if plane_norm not in planes:
                continue

            probability = 1.0
            per_axis = {}

            if isinstance(axis_cfg, (int, float)):
                per_axis['x'] = to_positive_float(axis_cfg, f"random_tilts[{plane_norm}]")
            elif isinstance(axis_cfg, dict):
                probability_value = axis_cfg.get('probability', axis_cfg.get('prob', axis_cfg.get('pct', axis_cfg.get('percentage'))))
                probability = to_probability(probability_value, f"random_tilts[{plane_norm}].probability")

                for axis_key, value in axis_cfg.items():
                    axis_lower = str(axis_key).lower()
                    if axis_lower in {'probability', 'prob', 'pct', 'percentage'}:
                        continue
                    if axis_lower not in {'x', 'y', 'z'}:
                        raise ValueError(
                            f"slice_sampling.random_tilts[{plane}][{axis_key}] must be one of: x, y, z"
                        )
                    per_axis[axis_lower] = to_positive_float(
                        value,
                        f"random_tilts[{plane_norm}][{axis_lower}]"
                    )
            else:
                raise ValueError(
                    f"slice_sampling.random_tilts[{plane}] must be a number or mapping of axis->degrees"
                )

            if per_axis:
                normalized[plane_norm] = {
                    'axes': per_axis,
                    'probability': probability
                }

        return normalized

    def _normalize_slice_label_interpolation(self, interp_cfg):
        normalized = {}
        if not isinstance(interp_cfg, dict):
            return normalized

        for target, policy in interp_cfg.items():
            if isinstance(policy, str):
                normalized[target] = {'__all__': policy.lower() == 'linear'}
            elif isinstance(policy, bool):
                normalized[target] = {'__all__': policy}
            elif isinstance(policy, dict):
                target_policy = {}
                for plane, mode in policy.items():
                    target_policy[str(plane).lower()] = str(mode).lower() == 'linear'
                if target_policy:
                    normalized[target] = target_policy
        return normalized

    def _normalize_slice_label_interpolation(self, interp_cfg):
        normalized = {}
        if not isinstance(interp_cfg, dict):
            return normalized

        for target, policy in interp_cfg.items():
            if isinstance(policy, str):
                normalized[target] = {'__all__': policy.lower() == 'linear'}
            elif isinstance(policy, dict):
                target_policy = {}
                for plane, mode in policy.items():
                    target_policy[str(plane).lower()] = str(mode).lower() == 'linear'
                if target_policy:
                    normalized[target] = target_policy
        return normalized

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
        self.validate_target_names(targets_dict.keys())

        self.targets = deepcopy(targets_dict)

        # Ensure all targets have out_channels, default to 2
        for target_name, target_info in self.targets.items():
            if 'out_channels' not in target_info and 'channels' not in target_info:
                target_info['out_channels'] = 2

        # Apply current loss function to all targets if not already set
        for target_name in self.targets:
            if "losses" not in self.targets[target_name]:
                self.targets[target_name]["losses"] = [{
                    "name": self.selected_loss_function,
                    "weight": 1.0,
                    "kwargs": {}
                }]

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

    def convert_to_dict(self):
        tr_setup = deepcopy(self.tr_info)
        tr_config = deepcopy(self.tr_configs)
        model_config = deepcopy(self.model_config)
        dataset_config = deepcopy(self.dataset_config)

        # Create inference_config from individual attributes
        inference_config = {
            "checkpoint_path": self.infer_checkpoint_path,
            "patch_size": list(self.infer_patch_size) if self.infer_patch_size else None,
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

        return combined_config

    def save_config(self):

        combined_config = self.convert_to_dict()
        
        model_ckpt_dir = Path(self.ckpt_out_base) / self.model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)
        config_filename = f"{self.model_name}_config.yaml"
        config_path = model_ckpt_dir / config_filename

        with config_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {config_path}")

    def update_config(self, patch_size=None, min_labeled_ratio=None, max_epochs=None, loss_function=None, 
                     skip_patch_validation=None,
                     normalization_scheme=None, intensity_properties=None,
                     min_bbox_percent=None,
                     skip_bounding_box=None):
        if patch_size is not None:
            if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
                self.train_patch_size = tuple(patch_size)
                self.tr_configs["patch_size"] = list(patch_size)

                dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
                self.model_config["conv_op"] = dim_props["conv_op"]
                self.model_config["norm_op"] = dim_props["norm_op"]
                self.spacing = dim_props["spacing"]
                self.op_dims = dim_props["op_dims"]

                if self.verbose:
                    print(f"Updated patch size: {self.train_patch_size}")

        if min_labeled_ratio is not None:
            self.min_labeled_ratio = float(min_labeled_ratio)
            self.dataset_config["min_labeled_ratio"] = self.min_labeled_ratio
            if self.verbose:
                print(f"Updated min labeled ratio: {self.min_labeled_ratio:.2f}")

        # Support setting max epochs from UI (note: internal key is 'max_epoch')
        if max_epochs is not None:
            try:
                self.max_epoch = int(max_epochs)
            except Exception:
                # Fallback if a non-int slips through
                self.max_epoch = int(float(max_epochs))
            self.tr_configs["max_epoch"] = self.max_epoch
            if self.verbose:
                print(f"Updated max epochs: {self.max_epoch}")

        if skip_patch_validation is not None:
            self.skip_patch_validation = bool(skip_patch_validation)
            self.dataset_config["skip_patch_validation"] = self.skip_patch_validation
            if self.verbose:
                print(f"Updated skip_patch_validation: {self.skip_patch_validation}")

        if min_bbox_percent is not None:
            self.min_bbox_percent = float(min_bbox_percent)
            self.dataset_config["min_bbox_percent"] = self.min_bbox_percent
            if self.verbose:
                print(f"Updated min bbox percent: {self.min_bbox_percent:.2f}")

        if loss_function is not None:
            self.selected_loss_function = loss_function
            if hasattr(self, 'targets') and self.targets:
                for target_name in self.targets:
                    self.targets[target_name]["losses"] = [{
                        "name": self.selected_loss_function,
                        "weight": 1.0,
                        "kwargs": {}
                    }]
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
            source_target = aux_config["source_target"]

            if source_target not in self.targets:
                raise ValueError(f"Source target '{source_target}' for auxiliary task '{aux_task_name}' not found in targets")

            # Use factory to create auxiliary task configuration
            target_config = create_auxiliary_task(task_type, aux_task_name, aux_config, source_target)
            self.targets[aux_task_name] = target_config

            if self.verbose:
                print(f"Added {task_type} auxiliary task '{aux_task_name}' from source '{source_target}'")
                    
        if self.verbose and self.auxiliary_tasks:
            print(f"Applied {len(self.auxiliary_tasks)} auxiliary tasks to targets")

    def validate_target_names(self, target_names):
        """
        Validate that target names don't conflict with reserved names.

        Parameters
        ----------
        target_names : iterable
            Collection of target names to validate

        Raises
        ------
        ValueError
            If any target name matches a reserved name
        """
        reserved_names = {'mask', 'is_unlabeled', 'plane_mask'}
        for name in target_names:
            if name in reserved_names:
                raise ValueError(
                    f"Target name '{name}' is reserved and cannot be used. "
                    f"Reserved names: {', '.join(sorted(reserved_names))}. "
                    f"Please choose a different name for your target."
                )

    def auto_detect_channels(self, dataset=None, sample=None):
        """
        Automatically detect the number of output channels for each target from the dataset.

        Parameters
        ----------
        dataset : BaseDataset
            The dataset to inspect for channel information
        sample : dict, optional
            Precomputed dataset sample. When provided, the dataset argument is optional.
        """
        if sample is None:
            if dataset is None or len(dataset) == 0:
                print("Warning: Empty dataset, cannot auto-detect channels")
                return
            sample = dataset[0]

        # Update targets with detected channels
        targets_updated = False
        for target_name in self.targets:
            if 'out_channels' not in self.targets[target_name] or self.targets[target_name].get('out_channels') is None:
                if target_name in sample:
                    # Get the label tensor for this target
                    label_tensor = sample[target_name]
                    
                    # Determine number of channels based on label data
                    # Regression/continuous aux targets: use channel dimension directly
                    if label_tensor.dtype.is_floating_point or (label_tensor.ndim >= 3 and label_tensor.shape[0] > 1):
                        detected_channels = int(label_tensor.shape[0])
                    else:
                        # For discrete labels, infer from unique values
                        unique_values = torch.unique(label_tensor)
                        num_unique = len(unique_values)
                        if num_unique <= 2:
                            detected_channels = 2
                        else:
                            # Multi-class case - use max value + 1
                            detected_channels = int(torch.max(label_tensor).item()) + 1
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
