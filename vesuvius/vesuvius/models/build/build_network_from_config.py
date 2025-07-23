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
from ..utils import get_pool_and_conv_props, get_n_blocks_per_stage
from .encoder import Encoder
from .decoder import Decoder
from .activations import SwiGLUBlock, GLUBlock

def get_activation_module(activation_str: str):
    act_str = activation_str.lower()
    if act_str == "none":
        return None
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "softmax":
        return nn.Sigmoid()
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
        
        # Check if we're in MAE pretraining mode
        self.mae_mode = mgr.model_config.get('mae_mode', False)
        if self.mae_mode:
            print("Initializing network in MAE pretraining mode")

        if hasattr(mgr, 'model_config') and mgr.model_config:
            model_config = mgr.model_config
        else:
            print("model_config is empty; using default configuration")
            model_config = {}

        self.save_config = False

        # --------------------------------------------------------------------
        # Common nontrainable parameters (ops, activation, etc.)
        # --------------------------------------------------------------------
        self.conv_op = model_config.get("conv_op", "nn.Conv3d")
        self.conv_op_kwargs = model_config.get("conv_op_kwargs", {"bias": False})
        self.pool_op = model_config.get("pool_op", "nn.AvgPool3d")
        self.dropout_op = model_config.get("dropout_op", "nn.Dropout3d")
        self.dropout_op_kwargs = model_config.get("dropout_op_kwargs", {"p": 0.0})
        self.norm_op = model_config.get("norm_op", "nn.InstanceNorm3d")
        self.norm_op_kwargs = model_config.get("norm_op_kwargs", {"affine": False, "eps": 1e-5})
        self.conv_bias = model_config.get("conv_bias", False)
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

        if isinstance(self.pool_op, str):
            if self.op_dims == 2:
                self.pool_op = nn.AvgPool2d
                print("Using 2D pooling (nn.AvgPool2d)")
            else:
                self.pool_op = nn.AvgPool3d
                print("Using 3D pooling (nn.AvgPool3d)")

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

        if self.nonlin in ["nn.LeakyReLU", "LeakyReLU"]:
            self.nonlin = nn.LeakyReLU
            self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
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
            
            self.n_blocks_per_stage = get_n_blocks_per_stage(self.num_stages)
            self.n_conv_per_stage_decoder = [1] * (self.num_stages - 1)
            self.strides = pool_op_kernel_sizes
            self.kernel_sizes = conv_kernel_sizes
            self.pool_op_kernel_sizes = pool_op_kernel_sizes
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
            squeeze_excitation_reduction_ratio=model_config.get("squeeze_excitation_reduction_ratio", 1.0/16.0)
        )
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        
        if self.mae_mode:
            # In MAE mode, create a lightweight reconstruction head
            self._build_mae_reconstruction_head()
        else:
            # Standard supervised mode - create task-specific decoders
            for target_name, target_info in self.targets.items():
                # Determine output channels - use task-specific channels if specified, otherwise match input channels
                if 'out_channels' in target_info:
                    out_channels = target_info['out_channels']
                elif 'channels' in target_info:
                    out_channels = target_info['channels']
                else:
                    # Default to matching input channels for adaptive behavior
                    out_channels = self.in_channels
                    print(f"No channel specification found for task '{target_name}', defaulting to {out_channels} channels (matching input)")

                # Update target_info with the determined channels
                target_info["out_channels"] = out_channels

                activation_str = target_info.get("activation", "sigmoid")
                self.task_decoders[target_name] = Decoder(
                    encoder=self.shared_encoder,
                    basic_block=model_config.get("basic_decoder_block", "ConvBlock"),
                    num_classes=out_channels,
                    n_conv_per_stage=model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
                    deep_supervision=False
                )
                self.task_activations[target_name] = get_activation_module(activation_str)
                print(f"Task '{target_name}' configured with {out_channels} output channels")

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
            "pool_op": self.pool_op.__name__ if hasattr(self.pool_op, "__name__") else self.pool_op,
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
            "op_dims": self.op_dims,
            "patch_size": self.patch_size,
            "batch_size": self.batch_size,
            "in_channels": self.in_channels,
            "autoconfigure": self.autoconfigure,
            "targets": self.targets,
            # Include autoconfiguration results if available
            "num_pool_per_axis": getattr(self, 'num_pool_per_axis', None),
            "must_be_divisible_by": getattr(self, 'must_be_divisible_by', None)
        }

        print("NetworkFromConfig initialized with final configuration:")
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

    def forward(self, x):
        # Check input channels and warn if mismatch
        self.check_input_channels(x)

        if self.mae_mode:
            return self.forward_mae(x)
        else:
            skips = self.shared_encoder(x)
            results = {}
            for task_name, decoder in self.task_decoders.items():
                logits = decoder(skips)
                activation_fn = self.task_activations[task_name]
                if activation_fn is not None and not self.training:
                    logits = activation_fn(logits)
                results[task_name] = logits
            return results
    
    def _build_mae_reconstruction_head(self):
        """Build a lightweight reconstruction head for MAE pretraining.
        
        Uses a simplified decoder with fewer stages for efficiency.
        """
        model_config = self.mgr.model_config
        
        # The decoder needs configuration for all encoder stages - 1
        n_encoder_stages = len(self.shared_encoder.output_channels)
        
        # Option to use fewer convolutions per stage for MAE (lighter decoder)
        mae_n_conv_per_stage = model_config.get("mae_n_conv_per_stage", 1)
        if isinstance(mae_n_conv_per_stage, int):
            # If it's an int, apply to all decoder stages
            n_conv_per_stage = [mae_n_conv_per_stage] * (n_encoder_stages - 1)
        else:
            # If it's a list, make sure it has the right length
            n_conv_per_stage = mae_n_conv_per_stage
            if len(n_conv_per_stage) < n_encoder_stages - 1:
                # Pad with 1s if not enough values provided
                n_conv_per_stage = n_conv_per_stage + [1] * (n_encoder_stages - 1 - len(n_conv_per_stage))
        
        # Create reconstruction decoder
        self.reconstruction_decoder = Decoder(
            encoder=self.shared_encoder,
            basic_block=model_config.get("mae_decoder_block", "ConvBlock"),
            num_classes=self.in_channels,  # Reconstruct input channels
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=False
        )
        
        # Option 2 (Alternative): Simple convolutional head
        # This would be used if mae_use_simple_head is True in config
        if model_config.get("mae_use_simple_head", False):
            # Get the number of channels from the encoder's bottleneck
            bottleneck_channels = self.shared_encoder.stages[-1].output_channels
            
            # Simple conv head
            if self.op_dims == 2:
                self.reconstruction_head = nn.Sequential(
                    nn.Conv2d(bottleneck_channels, 256, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, self.in_channels, kernel_size=1)
                )
            else:  # 3D
                self.reconstruction_head = nn.Sequential(
                    nn.Conv3d(bottleneck_channels, 256, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(256, 128, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(128, self.in_channels, kernel_size=1)
                )
        
        print(f"MAE reconstruction decoder created with {n_encoder_stages - 1} stages")
    
    def forward_mae(self, x):
        """Forward pass for MAE pretraining.
        
        Args:
            x: Masked input tensor
            
        Returns:
            Dictionary with 'reconstruction' key containing the reconstructed image
        """
        # Encode the masked input
        skips = self.shared_encoder(x)
        
        # Reconstruct using the decoder
        if hasattr(self, 'reconstruction_head') and self.mgr.model_config.get("mae_use_simple_head", False):
            # Use simple head - only use bottleneck features
            reconstruction = self.reconstruction_head(skips[-1])
            # Upsample to original size if needed
            if reconstruction.shape[2:] != x.shape[2:]:
                reconstruction = F.interpolate(reconstruction, size=x.shape[2:], 
                                             mode='trilinear' if self.op_dims == 3 else 'bilinear',
                                             align_corners=False)
        else:
            # Use decoder-based reconstruction
            reconstruction = self.reconstruction_decoder(skips)
        
        return {"reconstruction": reconstruction}
