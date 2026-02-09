"""
NetworkFromConfig: Adaptive Multi-Task U-Net Architecture

This module implements a flexible, configuration-driven U-Net architecture that supports:

ADAPTIVE CHANNEL BEHAVIOR:
- Input channels: Automatically detects and adapts to input channel count from ConfigManager
- Output channels: Adapts per task based on configuration or input channels
  * If task specifies 'out_channels' or 'channels': uses that value
  * If not specified: defaults to matching input channels (adaptive behavior)
  * Mixed configurations supported (some tasks adaptive, others fixed)

ARCHITECTURE FEATURES:
- Shared encoder with task-specific decoders
- Auto-configuration of network dimensions based on patch size and spacing
- Supports 2D/3D operations automatically based on patch dimensionality
- Configurable activation functions per task (sigmoid, softmax, none)
- Features: stochastic depth, squeeze-excitation, various block types

USAGE EXAMPLES:
1. Standard creation (uses ConfigManager settings):
   network = NetworkFromConfig(config_manager)

2. Override input channels:
   network = NetworkFromConfig.create_with_input_channels(config_manager, input_channels=3)

3. Configuration example for adaptive 3-channel I/O:
   config_manager.model_config["in_channels"] = 3
   targets = {
       "adaptive_task": {"activation": "sigmoid"},  # Will output 3 channels
       "fixed_task": {"out_channels": 1, "activation": "sigmoid"}  # Will output 1 channel
   }

RUNTIME VALIDATION:
- Checks input tensor channels against expected channels in forward pass
- Issues warnings for mismatched channel counts
- Continues processing but may produce unexpected results

The network automatically configures pooling, convolution, and normalization operations
based on the dimensionality of the input patch size (2D vs 3D).

this is inspired by the nnUNet architecture.
https://github.com/MIC-DKFZ/nnUNet
"""

import torch.nn as nn
from ..utilities.utils import get_pool_and_conv_props, get_n_blocks_per_stage
from .encoder import Encoder
from .decoder import Decoder
from .activations import SwiGLUBlock, GLUBlock
from .primus_wrapper import PrimusEncoder, PrimusDecoder

def get_activation_module(activation_str: str):
    if activation_str is None:
        activation_str = "none"
    act_str = activation_str.lower()
    if act_str == "none":
        return None
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "softmax":
        # Use channel dimension by default
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation type: {activation_str}")

class NetworkFromConfig(nn.Module):
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr
        self.targets = mgr.targets
        self.patch_size = mgr.train_patch_size
        self.batch_size = mgr.train_batch_size
        # Get input channels from manager if available, otherwise default to 1
        self.in_channels = getattr(mgr, 'in_channels', 1)
        self.autoconfigure = mgr.autoconfigure

        if hasattr(mgr, 'model_config') and mgr.model_config:
            model_config = mgr.model_config
        else:
            print("model_config is empty; using default configuration")
            model_config = {}

        self.save_config = False
        
        self.architecture_type = model_config.get("architecture_type", "unet")
        # Determine if deep supervision is requested
        ds_enabled = bool(getattr(mgr, 'enable_deep_supervision', False))

        # Primus decoders do not emit multi-scale logits; block DS to avoid silent misconfiguration
        if self.architecture_type.lower().startswith("primus"):
            if ds_enabled:
                raise ValueError(
                    "Deep supervision is enabled but the selected architecture 'primus' does not "
                    "support multi-scale logits. Please disable deep supervision or switch to the 'unet' architecture "
                    "with separate decoders for the supervised tasks."
                )
            self._init_primus(mgr, model_config)
            return

        # --------------------------------------------------------------------
        # Common nontrainable parameters (ops, activation, etc.)
        # --------------------------------------------------------------------
        self.conv_op = model_config.get("conv_op", "nn.Conv3d")
        self.conv_op_kwargs = model_config.get("conv_op_kwargs", {"bias": False})
        self.dropout_op = model_config.get("dropout_op", None)
        self.dropout_op_kwargs = model_config.get("dropout_op_kwargs", None)
        self.norm_op = model_config.get("norm_op", "nn.InstanceNorm3d")
        self.norm_op_kwargs = model_config.get("norm_op_kwargs", {"affine": True, "eps": 1e-5})
        self.conv_bias = model_config.get("conv_bias", True)
        self.nonlin = model_config.get("nonlin", "nn.LeakyReLU")
        self.nonlin_kwargs = model_config.get("nonlin_kwargs", {"inplace": True})

        self.op_dims = getattr(mgr, 'op_dims', None)
        if self.op_dims is None:
            if len(self.patch_size) == 2:
                self.op_dims = 2
                print(f"Using 2D operations based on patch_size {self.patch_size}")
            elif len(self.patch_size) == 3:
                self.op_dims = 3
                print(f"Using 3D operations based on patch_size {self.patch_size}")
            else:
                raise ValueError(f"Patch size must have either 2 or 3 dimensions! Got {len(self.patch_size)}D: {self.patch_size}")
        else:
            print(f"Using dimensionality ({self.op_dims}D) from ConfigManager")

        # Convert string operation types to actual PyTorch classes
        if isinstance(self.conv_op, str):
            if self.op_dims == 2:
                self.conv_op = nn.Conv2d
                print("Using 2D convolutions (nn.Conv2d)")
            else:
                self.conv_op = nn.Conv3d
                print("Using 3D convolutions (nn.Conv3d)")

        if isinstance(self.norm_op, str):
            if self.op_dims == 2:
                self.norm_op = nn.InstanceNorm2d
                print("Using 2D normalization (nn.InstanceNorm2d)")
            else:
                self.norm_op = nn.InstanceNorm3d
                print("Using 3D normalization (nn.InstanceNorm3d)")

        if isinstance(self.dropout_op, str):
            if self.op_dims == 2:
                self.dropout_op = nn.Dropout2d
                print("Using 2D dropout (nn.Dropout2d)")
            else:
                self.dropout_op = nn.Dropout3d
                print("Using 3D dropout (nn.Dropout3d)")
        elif self.dropout_op is None:
            pass

        if self.nonlin in ["nn.LeakyReLU", "LeakyReLU"]:
            self.nonlin = nn.LeakyReLU
            if "negative_slope" not in self.nonlin_kwargs:
                self.nonlin_kwargs["negative_slope"] = 0.01  # PyTorch default
        elif self.nonlin in ["nn.ReLU", "ReLU"]:
            self.nonlin = nn.ReLU
            self.nonlin_kwargs = {"inplace": True}
        elif self.nonlin in ["SwiGLU", "swiglu"]:
            self.nonlin = SwiGLUBlock
            self.nonlin_kwargs = {}  # SwiGLUBlock doesn't use standard kwargs
            print("Using SwiGLU activation - this will increase memory usage due to channel expansion")
        elif self.nonlin in ["GLU", "glu"]:
            self.nonlin = GLUBlock
            self.nonlin_kwargs = {}  # GLUBlock doesn't use standard kwargs
            print("Using GLU activation - this will increase memory usage due to channel expansion")

        # --------------------------------------------------------------------
        # Architecture parameters.
        # --------------------------------------------------------------------
        # Check if we have features_per_stage specified in model_config
        manual_features = model_config.get("features_per_stage", None)
        
        if self.autoconfigure or manual_features is not None:
            if manual_features is not None:
                print("--- Partial autoconfiguration: using provided features_per_stage ---")
                self.features_per_stage = manual_features
                self.num_stages = len(self.features_per_stage)
                print(f"Using provided features_per_stage: {self.features_per_stage}")
                print(f"Detected {self.num_stages} stages from features_per_stage")
            else:
                print("--- Full autoconfiguration from config ---")
            
            self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
            self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
            self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")

            num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, final_patch_size, must_div = \
                get_pool_and_conv_props(
                    spacing=mgr.spacing,
                    patch_size=self.patch_size,
                    min_feature_map_size=4,
                    max_numpool=999999
                )
            # Convert tuples from get_pool_and_conv_props to mutable lists so we can
            # trim/extend and selectively override with user-provided settings.
            pool_op_kernel_sizes = [list(k) for k in pool_op_kernel_sizes]
            conv_kernel_sizes = [list(k) for k in conv_kernel_sizes]

            self.num_pool_per_axis = num_pool_per_axis
            self.must_be_divisible_by = must_div
            original_patch_size = self.patch_size
            self.patch_size = final_patch_size
            print(f"Patch size adjusted from {original_patch_size} to {final_patch_size} to ensure divisibility by pooling factors {must_div}")

            # If features_per_stage was manually specified, adjust the auto-configured values
            if manual_features is not None:
                # Trim or extend the auto-configured lists to match the number of stages
                if len(pool_op_kernel_sizes) > self.num_stages:
                    pool_op_kernel_sizes = pool_op_kernel_sizes[:self.num_stages]
                    conv_kernel_sizes = conv_kernel_sizes[:self.num_stages]
                elif len(pool_op_kernel_sizes) < self.num_stages:
                    # Extend with reasonable defaults
                    while len(pool_op_kernel_sizes) < self.num_stages:
                        pool_op_kernel_sizes.append(pool_op_kernel_sizes[-1])
                        conv_kernel_sizes.append([3] * len(mgr.spacing))
            else:
                # Full auto-configuration
                self.num_stages = len(pool_op_kernel_sizes)
                base_features = 32
                max_features = 320
                features = []
                for i in range(self.num_stages):
                    feats = base_features * (2 ** i)
                    features.append(min(feats, max_features))
                self.features_per_stage = features
            
            manual_n_blocks = model_config.get("n_blocks_per_stage", None)
            if manual_n_blocks is None:
                self.n_blocks_per_stage = get_n_blocks_per_stage(self.num_stages)
            else:
                if isinstance(manual_n_blocks, int):
                    manual_n_blocks = [manual_n_blocks] * self.num_stages
                elif isinstance(manual_n_blocks, tuple):
                    manual_n_blocks = list(manual_n_blocks)

                if len(manual_n_blocks) != self.num_stages:
                    raise ValueError(
                        f"n_blocks_per_stage must have {self.num_stages} entries to match features_per_stage. "
                        f"Got {len(manual_n_blocks)} entries: {manual_n_blocks}"
                    )
                self.n_blocks_per_stage = manual_n_blocks
                print(f"Using provided n_blocks_per_stage: {self.n_blocks_per_stage}")
            # Respect user-provided architecture details in autoconfigure mode and
            # only fill missing parts from auto-derived defaults.
            self.n_conv_per_stage_decoder = model_config.get(
                "n_conv_per_stage_decoder",
                [1] * (self.num_stages - 1)
            )
            self.strides = model_config.get("strides", pool_op_kernel_sizes)
            self.kernel_sizes = model_config.get("kernel_sizes", conv_kernel_sizes)
            self.pool_op_kernel_sizes = model_config.get("pool_op_kernel_sizes", pool_op_kernel_sizes)

            # Validate stage-wise list lengths in autoconfigure mode.
            if len(self.kernel_sizes) != self.num_stages:
                raise ValueError(
                    f"kernel_sizes must have {self.num_stages} entries, got {len(self.kernel_sizes)}."
                )
            if len(self.strides) != self.num_stages:
                raise ValueError(
                    f"strides must have {self.num_stages} entries, got {len(self.strides)}."
                )
            if len(self.pool_op_kernel_sizes) != self.num_stages:
                raise ValueError(
                    f"pool_op_kernel_sizes must have {self.num_stages} entries, got {len(self.pool_op_kernel_sizes)}."
                )
            if len(self.n_conv_per_stage_decoder) != (self.num_stages - 1):
                raise ValueError(
                    f"n_conv_per_stage_decoder must have {self.num_stages - 1} entries, "
                    f"got {len(self.n_conv_per_stage_decoder)}."
                )

            # Check dimensionality for user-provided or auto-filled stage settings.
            for i in range(len(self.kernel_sizes)):
                if len(self.kernel_sizes[i]) != self.op_dims:
                    raise ValueError(
                        f"Kernel size at stage {i} has {len(self.kernel_sizes[i])} dimensions "
                        f"but patch size indicates {self.op_dims}D operations. "
                        f"Kernel: {self.kernel_sizes[i]}, Expected dimensions: {self.op_dims}"
                    )
            for i in range(len(self.strides)):
                if len(self.strides[i]) != self.op_dims:
                    raise ValueError(
                        f"Stride at stage {i} has {len(self.strides[i])} dimensions "
                        f"but patch size indicates {self.op_dims}D operations. "
                        f"Stride: {self.strides[i]}, Expected dimensions: {self.op_dims}"
                    )
            for i in range(len(self.pool_op_kernel_sizes)):
                if len(self.pool_op_kernel_sizes[i]) != self.op_dims:
                    raise ValueError(
                        f"Pool kernel size at stage {i} has {len(self.pool_op_kernel_sizes[i])} dimensions "
                        f"but patch size indicates {self.op_dims}D operations. "
                        f"Pool kernel: {self.pool_op_kernel_sizes[i]}, Expected dimensions: {self.op_dims}"
                    )
        else:
            print("--- Configuring network from config file ---")
            self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
            self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
            self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")
            self.features_per_stage = model_config.get("features_per_stage", [32, 64, 128, 256, 320, 320, 320])
            
            # If features_per_stage is provided, derive num_stages from it
            if "features_per_stage" in model_config:
                self.num_stages = len(self.features_per_stage)
                print(f"Derived num_stages={self.num_stages} from features_per_stage")
            else:
                self.num_stages = model_config.get("n_stages", 7)
            
            # Auto-configure n_blocks_per_stage if not provided
            if "n_blocks_per_stage" not in model_config:
                self.n_blocks_per_stage = get_n_blocks_per_stage(self.num_stages)
                print(f"Auto-configured n_blocks_per_stage: {self.n_blocks_per_stage}")
            else:
                self.n_blocks_per_stage = model_config.get("n_blocks_per_stage")
                
            self.num_pool_per_axis = model_config.get("num_pool_per_axis", None)
            self.must_be_divisible_by = model_config.get("must_be_divisible_by", None)

            # Set default kernel sizes and pool kernel sizes based on dimensionality
            default_kernel = [[3, 3]] * self.num_stages if self.op_dims == 2 else [[3, 3, 3]] * self.num_stages
            default_pool = [[1, 1]] * self.num_stages if self.op_dims == 2 else [[1, 1, 1]] * self.num_stages
            default_strides = [[1, 1]] * self.num_stages if self.op_dims == 2 else [[1, 1, 1]] * self.num_stages

            print(f"Using {'2D' if self.op_dims == 2 else '3D'} kernel defaults: {default_kernel[0]}")
            print(f"Using {'2D' if self.op_dims == 2 else '3D'} pool defaults: {default_pool[0]}")

            self.kernel_sizes = model_config.get("kernel_sizes", default_kernel)
            self.pool_op_kernel_sizes = model_config.get("pool_op_kernel_sizes", default_pool)
            self.n_conv_per_stage_decoder = model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1))
            self.strides = model_config.get("strides", default_strides)

            # Check for dimensionality mismatches 
            for i in range(len(self.kernel_sizes)):
                if len(self.kernel_sizes[i]) != self.op_dims:
                    raise ValueError(f"Kernel size at stage {i} has {len(self.kernel_sizes[i])} dimensions "
                                   f"but patch size indicates {self.op_dims}D operations. "
                                   f"Kernel: {self.kernel_sizes[i]}, Expected dimensions: {self.op_dims}")

            for i in range(len(self.strides)):
                if len(self.strides[i]) != self.op_dims:
                    raise ValueError(f"Stride at stage {i} has {len(self.strides[i])} dimensions "
                                   f"but patch size indicates {self.op_dims}D operations. "
                                   f"Stride: {self.strides[i]}, Expected dimensions: {self.op_dims}")

            for i in range(len(self.pool_op_kernel_sizes)):
                if len(self.pool_op_kernel_sizes[i]) != self.op_dims:
                    raise ValueError(f"Pool kernel size at stage {i} has {len(self.pool_op_kernel_sizes[i])} dimensions "
                                   f"but patch size indicates {self.op_dims}D operations. "
                                   f"Pool kernel: {self.pool_op_kernel_sizes[i]}, Expected dimensions: {self.op_dims}")

        # Derive stem channels from first feature map if not provided.
        self.stem_n_channels = self.features_per_stage[0]

        # --------------------------------------------------------------------
        # Build network.
        # --------------------------------------------------------------------
        self.shared_encoder = Encoder(
            input_channels=self.in_channels,
            basic_block=self.basic_encoder_block,
            n_stages=self.num_stages,
            features_per_stage=self.features_per_stage,
            n_blocks_per_stage=self.n_blocks_per_stage,
            bottleneck_block=self.bottleneck_block,
            conv_op=self.conv_op,
            kernel_sizes=self.kernel_sizes,
            conv_bias=self.conv_bias,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            strides=self.strides,
            return_skips=True,
            do_stem=model_config.get("do_stem", True),
            stem_channels=model_config.get("stem_channels", self.stem_n_channels),
            bottleneck_channels=model_config.get("bottleneck_channels", None),
            stochastic_depth_p=model_config.get("stochastic_depth_p", 0.0),
            squeeze_excitation=model_config.get("squeeze_excitation", False),
            squeeze_excitation_reduction_ratio=model_config.get("squeeze_excitation_reduction_ratio", 1.0/16.0),
            squeeze_excitation_type=model_config.get("squeeze_excitation_type", "channel"),
            squeeze_excitation_add_maxpool=model_config.get("squeeze_excitation_add_maxpool", False),
            pool_type=model_config.get("pool_type", "conv")
        )
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        # Decide decoder sharing strategy
        # If deep supervision is enabled, prefer separate decoders so tasks can emit multi-scale logits
        separate_decoders_default = model_config.get("separate_decoders", ds_enabled)

        # Determine which tasks use separate decoders vs shared head
        tasks_using_separate = set()
        tasks_using_shared = set()

        # First, normalize out_channels for each task and decide strategy
        for target_name, target_info in self.targets.items():
            if 'out_channels' in target_info:
                out_channels = target_info['out_channels']
            elif 'channels' in target_info:
                out_channels = target_info['channels']
            else:
                out_channels = self.in_channels
                print(f"No channel specification found for task '{target_name}', defaulting to {out_channels} channels (matching input)")
            target_info["out_channels"] = out_channels

            # Determine per-task override for decoder sharing
            use_separate = target_info.get("separate_decoder", separate_decoders_default)
            if use_separate:
                tasks_using_separate.add(target_name)
            else:
                tasks_using_shared.add(target_name)

        # If DS is enabled, force all tasks to use separate decoders (shared path is features-only and can't DS)
        if ds_enabled and len(tasks_using_shared) > 0:
            print("Deep supervision enabled: switching shared-decoder tasks to separate decoders for DS support:",
                  ", ".join(sorted(tasks_using_shared)))
            tasks_using_separate.update(tasks_using_shared)
            tasks_using_shared.clear()

        # If at least one task uses shared, build a single shared decoder trunk (features-only)
        if len(tasks_using_shared) > 0:
            self.shared_decoder = Decoder(
                encoder=self.shared_encoder,
                basic_block=model_config.get("basic_decoder_block", "ConvBlock"),
                num_classes=None,  # features-only mode
                n_conv_per_stage=model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
                deep_supervision=False
            )
            # Heads map from decoder feature channels at highest resolution to task outputs
            head_in_ch = self.shared_encoder.output_channels[0]
            for target_name in sorted(tasks_using_shared):
                out_ch = self.targets[target_name]["out_channels"]
                self.task_heads[target_name] = self.conv_op(head_in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
                activation_str = self.targets[target_name].get("activation", "none")
                self.task_activations[target_name] = get_activation_module(activation_str)
                print(f"Task '{target_name}' configured with shared decoder + head ({out_ch} channels)")

        # Build separate decoders for tasks that requested them
        for target_name in sorted(tasks_using_separate):
            out_channels = self.targets[target_name]["out_channels"]
            activation_str = self.targets[target_name].get("activation", "none")
            self.task_decoders[target_name] = Decoder(
                encoder=self.shared_encoder,
                basic_block=model_config.get("basic_decoder_block", "ConvBlock"),
                num_classes=out_channels,
                n_conv_per_stage=model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
                deep_supervision=False
            )
            self.task_activations[target_name] = get_activation_module(activation_str)
            print(f"Task '{target_name}' configured with separate decoder ({out_channels} channels)")

        # --------------------------------------------------------------------
        # Build final configuration snapshot.
        # --------------------------------------------------------------------

        self.final_config = {
            "model_name": self.mgr.model_name,
            "basic_encoder_block": self.basic_encoder_block,
            "basic_decoder_block": model_config.get("basic_decoder_block", "ConvBlock"),
            "bottleneck_block": self.bottleneck_block,
            "features_per_stage": self.features_per_stage,
            "num_stages": self.num_stages,
            "n_blocks_per_stage": self.n_blocks_per_stage,
            "n_conv_per_stage_decoder": model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
            "kernel_sizes": self.kernel_sizes,
            "pool_op_kernel_sizes": self.pool_op_kernel_sizes,
            "conv_op": self.conv_op.__name__ if hasattr(self.conv_op, "__name__") else self.conv_op,
            "conv_bias": self.conv_bias,
            "norm_op": self.norm_op.__name__ if hasattr(self.norm_op, "__name__") else self.norm_op,
            "norm_op_kwargs": self.norm_op_kwargs,
            "dropout_op": self.dropout_op.__name__ if hasattr(self.dropout_op, "__name__") else self.dropout_op,
            "dropout_op_kwargs": self.dropout_op_kwargs,
            "nonlin": self.nonlin.__name__ if hasattr(self.nonlin, "__name__") else self.nonlin,
            "nonlin_kwargs": self.nonlin_kwargs,
            "strides": self.strides,
            "return_skips": model_config.get("return_skips", True),
            "do_stem": model_config.get("do_stem", True),
            "stem_channels": model_config.get("stem_channels", self.stem_n_channels),
            "bottleneck_channels": model_config.get("bottleneck_channels", None),
            "stochastic_depth_p": model_config.get("stochastic_depth_p", 0.0),
            "squeeze_excitation": model_config.get("squeeze_excitation", False),
            "squeeze_excitation_reduction_ratio": model_config.get("squeeze_excitation_reduction_ratio", 1.0/16.0),
            "squeeze_excitation_type": model_config.get("squeeze_excitation_type", "channel"),
            "squeeze_excitation_add_maxpool": model_config.get("squeeze_excitation_add_maxpool", False),
            "pool_type": model_config.get("pool_type", "conv"),
            "op_dims": self.op_dims,
            "patch_size": self.patch_size,
            "batch_size": self.batch_size,
            "in_channels": self.in_channels,
            "autoconfigure": self.autoconfigure,
            "targets": self.targets,
            "separate_decoders": len(tasks_using_separate) > 0,
            "num_pool_per_axis": getattr(self, 'num_pool_per_axis', None),
            "must_be_divisible_by": getattr(self, 'must_be_divisible_by', None)
        }

        print("NetworkFromConfig initialized with final configuration:")
        for k, v in self.final_config.items():
            print(f"  {k}: {v}")
    
    def _init_primus(self, mgr, model_config):
        """
        Initialize Primus transformer architecture.
        """
        print(f"--- Initializing Primus architecture ---")
        
        # Extract Primus variant (S, B, M, L) from architecture_type
        arch_type = self.architecture_type.lower()
        if arch_type == "primus_s":
            config_name = "S"
        elif arch_type == "primus_b":
            config_name = "B"
        elif arch_type == "primus_m":
            config_name = "M"
        elif arch_type == "primus_l":
            config_name = "L"
        else:
            # Try to extract from the string (e.g., "Primus-B")
            parts = arch_type.split("-")
            if len(parts) > 1:
                config_name = parts[1].upper()
            else:
                config_name = "M"  # Default to M
        
        print(f"Using Primus-{config_name} configuration")
        
        patch_embed_size = model_config.get("patch_embed_size", (8, 8, 8))
        if isinstance(patch_embed_size, int):
            patch_embed_size = (patch_embed_size,) * 3
        
        # Ensure input shape is specified
        input_shape = model_config.get("input_shape", self.patch_size)
        if input_shape is None:
            raise ValueError("input_shape must be specified for Primus architecture")
        
        # Get Primus-specific parameters
        primus_kwargs = {
            "drop_path_rate": model_config.get("drop_path_rate", 0.0),
            "patch_drop_rate": model_config.get("patch_drop_rate", 0.0),
            "proj_drop_rate": model_config.get("proj_drop_rate", 0.0),
            "attn_drop_rate": model_config.get("attn_drop_rate", 0.0),
            "num_register_tokens": model_config.get("num_register_tokens", 0),
            "use_rot_pos_emb": model_config.get("use_rot_pos_emb", True),
            "use_abs_pos_embed": model_config.get("use_abs_pos_embed", True),
            "pos_emb_type": model_config.get("pos_emb_type", "rope"),
            "mlp_ratio": model_config.get("mlp_ratio", 4 * 2 / 3),
            "init_values": model_config.get("init_values", 0.1 if config_name != "S" else 0.1),
            "scale_attn_inner": model_config.get("scale_attn_inner", True),
        }
        
        # Get decoder normalization and activation settings
        decoder_norm_str = model_config.get("decoder_norm", "LayerNormNd")
        decoder_act_str = model_config.get("decoder_act", "GELU")
        print(f"Using decoder normalization: {decoder_norm_str}")
        print(f"Using decoder activation: {decoder_act_str}")
        
        # Initialize shared Primus encoder
        self.shared_encoder = PrimusEncoder(
            input_channels=self.in_channels,
            config_name=config_name,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            **primus_kwargs
        )
        
        # Initialize decoders/heads based on sharing strategy
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        separate_decoders_default = model_config.get("separate_decoders", False)
        decoder_head_channels = model_config.get("decoder_head_channels", 32)

        tasks_using_shared, tasks_using_separate = set(), set()

        # Decide per-task channels and strategy
        for target_name, target_info in self.targets.items():
            if 'out_channels' in target_info:
                out_channels = target_info['out_channels']
            elif 'channels' in target_info:
                out_channels = target_info['channels']
            else:
                out_channels = self.in_channels
                print(f"No channel specification found for task '{target_name}', defaulting to {out_channels} channels")
            target_info["out_channels"] = out_channels

            use_separate = target_info.get("separate_decoder", separate_decoders_default)
            if use_separate:
                tasks_using_separate.add(target_name)
            else:
                tasks_using_shared.add(target_name)

        # Shared Primus decoder trunk
        if len(tasks_using_shared) > 0:
            self.shared_decoder = PrimusDecoder(
                encoder=self.shared_encoder,
                num_classes=decoder_head_channels,
                norm=decoder_norm_str,
                activation=decoder_act_str,
            )
            for target_name in sorted(tasks_using_shared):
                out_ch = self.targets[target_name]["out_channels"]
                head_conv = nn.Conv2d if self.shared_encoder.ndim == 2 else nn.Conv3d
                self.task_heads[target_name] = head_conv(
                    decoder_head_channels, out_ch, kernel_size=1, stride=1, padding=0, bias=True
                )
                activation_str = self.targets[target_name].get("activation", "none")
                self.task_activations[target_name] = get_activation_module(activation_str)
                print(f"Primus task '{target_name}' configured with shared decoder + head ({out_ch} channels)")

        # Separate Primus decoders per task
        for target_name in sorted(tasks_using_separate):
            out_channels = self.targets[target_name]["out_channels"]
            activation_str = self.targets[target_name].get("activation", "none")
            self.task_decoders[target_name] = PrimusDecoder(
                encoder=self.shared_encoder,
                num_classes=out_channels,
                norm=decoder_norm_str,
                activation=decoder_act_str,
            )
            self.task_activations[target_name] = get_activation_module(activation_str)
            print(f"Primus task '{target_name}' configured with separate decoder ({out_channels} channels)")
        
        # Store configuration for reference
        self.final_config = {
            "model_name": self.mgr.model_name,
            "architecture_type": self.architecture_type,
            "primus_variant": config_name,
            "patch_embed_size": patch_embed_size,
            "input_shape": input_shape,
            "in_channels": self.in_channels,
            "targets": self.targets,
            "decoder_norm": decoder_norm_str,
            "decoder_act": decoder_act_str,
            "separate_decoders": len(tasks_using_separate) > 0,
            "decoder_head_channels": decoder_head_channels,
            **primus_kwargs
        }
        
        print("Primus network initialized with configuration:")
        for k, v in self.final_config.items():
            print(f"  {k}: {v}")

    @classmethod
    def create_with_input_channels(cls, mgr, input_channels):
        """
        Create a NetworkFromConfig instance with a specific number of input channels.
        This will override the manager's in_channels setting.
        """
        # Temporarily set the input channels on the manager
        original_in_channels = getattr(mgr, 'in_channels', 1)
        mgr.in_channels = input_channels

        # Create the network
        network = cls(mgr)

        # Restore original value
        mgr.in_channels = original_in_channels

        print(f"Created network with {input_channels} input channels")
        return network

    def check_input_channels(self, x):
        """
        Check if the input tensor has the expected number of channels.
        Issue a warning if there's a mismatch.
        """
        input_channels = x.shape[1]  # Assuming NCHW or NCHWD format
        if input_channels != self.in_channels:
            print(f"Warning: Input has {input_channels} channels but network was configured for {self.in_channels} channels.")
            print(f"The encoder may not work properly. Consider reconfiguring the network with the correct input channels.")
            return False
        return True

    def forward(self, x, return_mae_mask=False):
        # Check input channels and warn if mismatch
        self.check_input_channels(x)

        # Get features from encoder (works for both U-Net and Primus)
        # For MAE training with Primus, we need to get the mask
        if return_mae_mask:
            # MAE training requires mask from the encoder
            if not isinstance(self.shared_encoder, PrimusEncoder):
                raise RuntimeError(
                    "MAE training (return_mae_mask=True) is only supported with Primus architecture. "
                    f"Current encoder type: {type(self.shared_encoder).__name__}"
                )
            
            # Get features with mask from Primus encoder
            encoder_output = self.shared_encoder(x, ret_mask=True)
            
            if not isinstance(encoder_output, tuple) or len(encoder_output) != 2:
                raise RuntimeError(
                    "Primus encoder did not return expected (features, mask) tuple "
                    "for MAE training. This is likely a bug in PrimusEncoder."
                )
            
            features, restoration_mask = encoder_output
            
            if restoration_mask is None:
                raise RuntimeError(
                    "Primus encoder returned None for restoration_mask. "
                    "Ensure patch_drop_rate is set > 0 in model config for MAE training."
                )
        else:
            # Standard forward pass
            features = self.shared_encoder(x)
            restoration_mask = None
        
        results = {}
        shared_features = None

        # Handle tasks with separate decoders first
        ds_enabled = bool(getattr(self.mgr, 'enable_deep_supervision', False))
        for task_name, decoder in self.task_decoders.items():
            logits = decoder(features)
            # If deep supervision is disabled, collapse to highest-res output for convenience.
            # If enabled, keep the list so training can supervise all scales.
            if isinstance(logits, (list, tuple)) and len(logits) > 0 and not ds_enabled:
                logits = logits[0]
            activation_fn = self.task_activations[task_name] if task_name in self.task_activations else None
            if activation_fn is not None and not self.training:
                if isinstance(logits, (list, tuple)):
                    logits = type(logits)(activation_fn(l) for l in logits)
                else:
                    logits = activation_fn(logits)
            results[task_name] = logits

        # Handle tasks that use shared decoder + heads
        if hasattr(self, 'task_heads') and len(self.task_heads) > 0:
            if shared_features is None:
                shared_features = self.shared_decoder(features)
            for task_name, head in self.task_heads.items():
                logits = head(shared_features)
                activation_fn = self.task_activations[task_name] if task_name in self.task_activations else None
                if activation_fn is not None and not self.training:
                    if isinstance(logits, (list, tuple)):
                        logits = type(logits)(activation_fn(l) for l in logits)
                    else:
                        logits = activation_fn(logits)
                results[task_name] = logits
        
        # Return MAE mask if requested (for MAE training)
        if return_mae_mask:
            return results, restoration_mask
        return results
