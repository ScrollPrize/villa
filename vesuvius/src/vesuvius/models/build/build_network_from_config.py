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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities.utils import get_pool_and_conv_props, get_n_blocks_per_stage, pad_shape
from .encoder import Encoder
from .decoder import Decoder
from .activations import SwiGLUBlock, GLUBlock
from .guidance import TokenBook3D
from .primus_wrapper import PrimusEncoder, PrimusDecoder
from .pretrained_backbones.dinov2 import build_dinov2_backbone, build_dinov2_decoder


class LearnedMLPZProjection(nn.Module):
    """Project [B, C, Z, H, W] logits to [B, C, H, W] with an MLP over Z."""

    def __init__(self, depth: int, hidden: int, dropout: float):
        super().__init__()
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError(f"z-projection mlp depth must be > 0, got {self.depth}")
        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError(f"z-projection mlp hidden must be > 0, got {hidden}")
        dropout = float(dropout)
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"z-projection mlp dropout must be in [0, 1], got {dropout}")

        self.mlp = nn.Sequential(
            nn.Linear(self.depth, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, logits_3d: torch.Tensor) -> torch.Tensor:
        if logits_3d.ndim != 5:
            raise ValueError(
                f"LearnedMLPZProjection expects [B, C, Z, H, W], got shape {tuple(logits_3d.shape)}"
            )
        depth = int(logits_3d.shape[2])
        if depth != self.depth:
            raise ValueError(
                f"z-projection depth mismatch: expected Z={self.depth}, got Z={depth}"
            )
        x = logits_3d.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, Z]
        return self.mlp(x).squeeze(-1)  # [B, C, H, W]

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
    def _read_projection_value(self, raw_cfg, target_info, model_config, key, default):
        if isinstance(raw_cfg, dict) and key in raw_cfg:
            return raw_cfg[key]
        if key in target_info:
            return target_info[key]
        return model_config.get(key, default)

    def _resolve_target_projection(self, target_name, target_info, model_config):
        raw_cfg = target_info.get("z_projection", None)
        if raw_cfg is not None and not isinstance(raw_cfg, dict):
            raise TypeError(
                f"Target '{target_name}' z_projection must be a dict when provided, "
                f"got {type(raw_cfg).__name__}"
            )

        mode_default = model_config.get("z_projection_mode", "none")
        if isinstance(raw_cfg, dict):
            mode = raw_cfg.get("mode", mode_default)
        else:
            mode = target_info.get("z_projection_mode", mode_default)
        mode = str(mode).strip().lower()
        if mode in {"", "none", "off", "false", "0"}:
            return None
        if mode not in {"max", "mean", "logsumexp", "learned_mlp"}:
            raise ValueError(
                f"Unknown z_projection mode for target '{target_name}': {mode!r}. "
                "Expected one of: max, mean, logsumexp, learned_mlp, none"
            )
        if self.op_dims != 3:
            raise ValueError(
                f"z_projection is only supported for 3D outputs. "
                f"Target '{target_name}' requested mode={mode!r} with op_dims={self.op_dims}"
            )

        spec = {"mode": mode}
        if mode == "logsumexp":
            tau = float(self._read_projection_value(
                raw_cfg, target_info, model_config, "z_projection_lse_tau", 1.0
            ))
            if tau <= 0:
                raise ValueError(
                    f"Target '{target_name}' z_projection_lse_tau must be > 0, got {tau}"
                )
            spec["lse_tau"] = tau
        elif mode == "learned_mlp":
            default_depth = int(self.patch_size[0]) if len(self.patch_size) == 3 else None
            depth = self._read_projection_value(
                raw_cfg, target_info, model_config, "z_projection_mlp_depth", default_depth
            )
            if depth is None:
                raise ValueError(
                    f"Target '{target_name}' learned_mlp z-projection requires z_projection_mlp_depth"
                )
            hidden = int(self._read_projection_value(
                raw_cfg, target_info, model_config, "z_projection_mlp_hidden", 64
            ))
            dropout = float(self._read_projection_value(
                raw_cfg, target_info, model_config, "z_projection_mlp_dropout", 0.0
            ))
            spec.update({
                "mlp_depth": int(depth),
                "mlp_hidden": hidden,
                "mlp_dropout": dropout,
            })
        return spec

    def _register_target_projection(self, target_name, target_info, model_config):
        spec = self._resolve_target_projection(target_name, target_info, model_config)
        if spec is None:
            return
        self.task_z_projection_cfg[target_name] = spec
        if spec["mode"] == "learned_mlp":
            self.task_z_projection_heads[target_name] = LearnedMLPZProjection(
                depth=spec["mlp_depth"],
                hidden=spec["mlp_hidden"],
                dropout=spec["mlp_dropout"],
            )
        print(f"Task '{target_name}' configured with z-projection mode '{spec['mode']}'")

    def _apply_z_projection_tensor(self, task_name, tensor, spec):
        if tensor.ndim != 5:
            return tensor
        mode = spec["mode"]
        if mode == "max":
            return torch.amax(tensor, dim=2)
        if mode == "mean":
            return torch.mean(tensor, dim=2)
        if mode == "logsumexp":
            tau = float(spec.get("lse_tau", 1.0))
            return tau * torch.logsumexp(tensor / tau, dim=2)
        if mode == "learned_mlp":
            if task_name not in self.task_z_projection_heads:
                raise RuntimeError(
                    f"Target '{task_name}' requested learned_mlp z-projection but no head was initialized"
                )
            return self.task_z_projection_heads[task_name](tensor)
        raise ValueError(f"Unknown z-projection mode {mode!r} for target '{task_name}'")

    def _apply_z_projection(self, task_name, logits):
        spec = self.task_z_projection_cfg.get(task_name)
        if spec is None:
            return logits
        if isinstance(logits, (list, tuple)):
            return type(logits)(self._apply_z_projection_tensor(task_name, l, spec) for l in logits)
        return self._apply_z_projection_tensor(task_name, logits, spec)

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

        self.task_z_projection_cfg = {}
        self.task_z_projection_heads = nn.ModuleDict()

        self.save_config = False
        
        self.architecture_type = model_config.get("architecture_type", "unet")
        self.pretrained_backbone = model_config.get("pretrained_backbone")
        self.guide_backbone_name = model_config.get("guide_backbone")
        self.guide_backbone_config_path = model_config.get("guide_backbone_config_path")
        self.guide_fusion_stage = str(model_config.get("guide_fusion_stage", "input")).strip().lower()
        self.guide_enabled = bool(self.guide_backbone_name)
        self.guide_backbone = None
        self.guide_tokenbook = None
        self.guide_patch_grid = None
        self.guide_freeze = bool(model_config.get("guide_freeze", True))
        guide_compile_policy = str(model_config.get("guide_compile_policy", "off")).strip().lower()
        if guide_compile_policy not in {"off", "backbone_only", "tokenbook_only"}:
            raise ValueError(
                "guide_compile_policy must be one of "
                "{'off', 'backbone_only', 'tokenbook_only'}"
            )
        self.guide_compile_policy = guide_compile_policy
        # Determine if deep supervision is requested
        ds_enabled = bool(getattr(mgr, 'enable_deep_supervision', False))

        if self.guide_enabled:
            if self.op_dims != 3:
                raise ValueError("guide_backbone is only supported for 3D models in v1")
            if self.pretrained_backbone:
                raise ValueError("guide_backbone cannot be combined with pretrained_backbone in v1")
            if self.architecture_type.lower().startswith("primus"):
                raise ValueError("guide_backbone is not supported with primus architectures in v1")
            if self.guide_fusion_stage not in {"input", "input_gating"}:
                raise ValueError(
                    "Unsupported guide_fusion_stage for v1. Only input-space gating is implemented."
                )

        if self.pretrained_backbone:
            if ds_enabled:
                print(
                    "Warning: Deep supervision is enabled but the selected pretrained backbone path does not "
                    "support multi-scale logits. Disabling deep supervision for this run."
                )
                setattr(mgr, "enable_deep_supervision", False)
            self._init_pretrained_backbone(mgr, model_config)
            return

        # Primus decoders do not emit multi-scale logits; block DS to avoid silent misconfiguration
        if self.architecture_type.lower().startswith("primus"):
            if ds_enabled:
                print(
                    "Warning: Deep supervision is enabled but the selected architecture 'primus' does not "
                    "support multi-scale logits. Disabling deep supervision for this run."
                )
                setattr(mgr, "enable_deep_supervision", False)
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

        # Convert string operation types to actual PyTorch classes
        if isinstance(self.conv_op, str):
            if self.op_dims == 2:
                self.conv_op = nn.Conv2d
                print("Using 2D convolutions (nn.Conv2d)")
            else:
                self.conv_op = nn.Conv3d
                print("Using 3D convolutions (nn.Conv3d)")

        if isinstance(self.norm_op, str):
            norm_key = self.norm_op.strip().lower().replace(" ", "")
            if norm_key.startswith("torch.nn."):
                norm_key = norm_key[len("torch.nn."):]
            if norm_key.startswith("nn."):
                norm_key = norm_key[len("nn."):]

            is_batch = norm_key in {"batch", "batchnorm", "batchnorm2d", "batchnorm3d"}
            is_instance = norm_key in {"instance", "instancenorm", "instancenorm2d", "instancenorm3d"}

            if not (is_batch or is_instance):
                raise ValueError(
                    f"Unknown norm_op string: {self.norm_op!r}. "
                    "Expected batch/BatchNorm* or instance/InstanceNorm*."
                )

            if self.op_dims == 2:
                if is_batch:
                    self.norm_op = nn.BatchNorm2d
                    print("Using 2D normalization (nn.BatchNorm2d)")
                else:
                    self.norm_op = nn.InstanceNorm2d
                    print("Using 2D normalization (nn.InstanceNorm2d)")
            else:
                if is_batch:
                    self.norm_op = nn.BatchNorm3d
                    print("Using 3D normalization (nn.BatchNorm3d)")
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
        # Check if we have stage-wise architecture settings specified in model_config
        manual_features = model_config.get("features_per_stage", None)
        manual_kernel_sizes = model_config.get("kernel_sizes", None)
        manual_strides = model_config.get("strides", None)
        manual_pool_op_kernel_sizes = model_config.get("pool_op_kernel_sizes", None)

        manual_stage_specs = {
            "features_per_stage": manual_features,
            "kernel_sizes": manual_kernel_sizes,
            "strides": manual_strides,
            "pool_op_kernel_sizes": manual_pool_op_kernel_sizes,
        }
        explicit_stage_count = None
        for spec_name, spec_value in manual_stage_specs.items():
            if spec_value is None:
                continue
            spec_len = len(spec_value)
            if explicit_stage_count is None:
                explicit_stage_count = spec_len
            elif spec_len != explicit_stage_count:
                raise ValueError(
                    "Provided stage-wise settings must agree on stage count. "
                    f"Expected {explicit_stage_count} entries but {spec_name} has {spec_len}."
                )
        
        if self.autoconfigure or manual_features is not None:
            if manual_features is not None:
                print("--- Partial autoconfiguration: using provided stage-wise settings ---")
                self.features_per_stage = manual_features
                self.num_stages = len(self.features_per_stage)
                print(f"Using provided features_per_stage: {self.features_per_stage}")
                print(f"Detected {self.num_stages} stages from features_per_stage")
            elif explicit_stage_count is not None:
                print("--- Partial autoconfiguration: using provided stage layout ---")
                self.num_stages = explicit_stage_count
                print(f"Detected {self.num_stages} stages from provided stage-wise settings")
            else:
                print("--- Full autoconfiguration from config ---")
            
            self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
            self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
            self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")

            auto_num_pool_per_axis, auto_pool_op_kernel_sizes, auto_conv_kernel_sizes, auto_final_patch_size, auto_must_div = \
                get_pool_and_conv_props(
                    spacing=mgr.spacing,
                    patch_size=self.patch_size,
                    min_feature_map_size=4,
                    max_numpool=999999
                )
            auto_pool_op_kernel_sizes = [list(k) for k in auto_pool_op_kernel_sizes]
            auto_conv_kernel_sizes = [list(k) for k in auto_conv_kernel_sizes]

            user_controls_pooling = (
                manual_strides is not None or manual_pool_op_kernel_sizes is not None
            )

            if user_controls_pooling:
                effective_strides = manual_strides if manual_strides is not None else manual_pool_op_kernel_sizes
                effective_pool_op_kernel_sizes = (
                    manual_pool_op_kernel_sizes if manual_pool_op_kernel_sizes is not None else effective_strides
                )
                if len(effective_strides) != len(effective_pool_op_kernel_sizes):
                    raise ValueError(
                        "strides and pool_op_kernel_sizes must have the same number of stages "
                        f"when both are provided. Got {len(effective_strides)} and {len(effective_pool_op_kernel_sizes)}."
                    )
                if explicit_stage_count is None:
                    self.num_stages = len(effective_strides)

                must_div = [1] * self.op_dims
                num_pool_per_axis = [0] * self.op_dims
                for stage_idx, stage_stride in enumerate(effective_strides):
                    if len(stage_stride) != self.op_dims:
                        raise ValueError(
                            f"Stride at stage {stage_idx} has {len(stage_stride)} dimensions "
                            f"but patch size indicates {self.op_dims}D operations. "
                            f"Stride: {stage_stride}, Expected dimensions: {self.op_dims}"
                        )
                    for axis, stride_value in enumerate(stage_stride):
                        stride_int = int(stride_value)
                        if stride_int < 1:
                            raise ValueError(
                                f"Stride values must be >= 1. Found {stride_value} at "
                                f"stage {stage_idx}, axis {axis}."
                            )
                        must_div[axis] *= stride_int
                        stride_remainder = stride_int
                        while stride_remainder > 1 and stride_remainder % 2 == 0:
                            num_pool_per_axis[axis] += 1
                            stride_remainder //= 2

                pool_op_kernel_sizes = [list(k) for k in effective_pool_op_kernel_sizes]
                conv_kernel_sizes = list(auto_conv_kernel_sizes)
                final_patch_size = tuple(int(v) for v in pad_shape(self.patch_size, must_div))
            else:
                pool_op_kernel_sizes = auto_pool_op_kernel_sizes
                conv_kernel_sizes = auto_conv_kernel_sizes
                num_pool_per_axis = auto_num_pool_per_axis
                must_div = auto_must_div
                final_patch_size = auto_final_patch_size
                if explicit_stage_count is None:
                    self.num_stages = len(pool_op_kernel_sizes)

            self.num_pool_per_axis = num_pool_per_axis
            self.must_be_divisible_by = must_div
            original_patch_size = self.patch_size
            self.patch_size = final_patch_size
            print(
                f"Patch size adjusted from {original_patch_size} to {final_patch_size} "
                f"to ensure divisibility by pooling factors {must_div}"
            )

            if len(conv_kernel_sizes) > self.num_stages:
                conv_kernel_sizes = conv_kernel_sizes[:self.num_stages]
            elif len(conv_kernel_sizes) < self.num_stages:
                while len(conv_kernel_sizes) < self.num_stages:
                    conv_kernel_sizes.append([3] * len(mgr.spacing))

            if len(pool_op_kernel_sizes) > self.num_stages:
                pool_op_kernel_sizes = pool_op_kernel_sizes[:self.num_stages]
            elif len(pool_op_kernel_sizes) < self.num_stages:
                while len(pool_op_kernel_sizes) < self.num_stages:
                    pool_op_kernel_sizes.append(pool_op_kernel_sizes[-1])

            if manual_features is None:
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
            self.strides = manual_strides if manual_strides is not None else pool_op_kernel_sizes
            self.kernel_sizes = manual_kernel_sizes if manual_kernel_sizes is not None else conv_kernel_sizes
            self.pool_op_kernel_sizes = (
                manual_pool_op_kernel_sizes
                if manual_pool_op_kernel_sizes is not None
                else pool_op_kernel_sizes
            )

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
            self._register_target_projection(target_name, target_info, model_config)

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

        self._init_input_guidance(model_config)

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
            "target_z_projection": self.task_z_projection_cfg,
            "separate_decoders": len(tasks_using_separate) > 0,
            "num_pool_per_axis": getattr(self, 'num_pool_per_axis', None),
            "must_be_divisible_by": getattr(self, 'must_be_divisible_by', None),
            "guide_backbone": self.guide_backbone_name,
            "guide_backbone_config_path": self.guide_backbone_config_path,
            "guide_freeze": self.guide_freeze,
            "guide_patch_grid": self.guide_patch_grid,
            "guide_tokenbook_tokens": getattr(self, "guide_tokenbook_tokens", None),
            "guide_compile_policy": self.guide_compile_policy if self.guide_enabled else "off",
            "guide_fusion_stage": "input" if self.guide_enabled else None,
        }

        print("NetworkFromConfig initialized with final configuration:")
        for k, v in self.final_config.items():
            print(f"  {k}: {v}")

    def _init_input_guidance(self, model_config):
        if not self.guide_enabled:
            return

        input_shape = tuple(model_config.get("input_shape", self.patch_size))
        self.guide_backbone = build_dinov2_backbone(
            self.guide_backbone_name,
            input_channels=self.in_channels,
            input_shape=input_shape,
            config_path=self.guide_backbone_config_path,
        )
        if getattr(self.guide_backbone, "ndim", None) != 3:
            raise ValueError("guide_backbone must resolve to a 3D Dinovol backbone")

        patch_embed_size = tuple(int(v) for v in self.guide_backbone.patch_embed_size)
        if any(size % patch != 0 for size, patch in zip(input_shape, patch_embed_size)):
            raise ValueError(
                f"Configured input_shape {input_shape} must be divisible by guide patch size {patch_embed_size}"
            )

        self.guide_patch_grid = tuple(size // patch for size, patch in zip(input_shape, patch_embed_size))
        default_token_count = int(math.prod(self.guide_patch_grid))
        configured_token_count = model_config.get("guide_tokenbook_tokens")
        if configured_token_count is None:
            token_count = default_token_count
        else:
            token_count = int(configured_token_count)
            if token_count <= 0:
                raise ValueError(f"guide_tokenbook_tokens must be > 0, got {token_count}")
        self.guide_tokenbook_tokens = token_count
        self.guide_tokenbook = TokenBook3D(
            n_tokens=token_count,
            embed_dim=int(self.guide_backbone.embed_dim),
            dropout=float(model_config.get("guide_tokenbook_dropout", 0.0)),
            ema_decay=model_config.get("guide_tokenbook_ema_decay"),
            use_ema=bool(model_config.get("guide_tokenbook_use_ema", False)),
        )
        self.guide_tokenbook_sample_rate = float(model_config.get("guide_tokenbook_sample_rate", 1.0))

        if self.guide_freeze:
            for parameter in self.guide_backbone.parameters():
                parameter.requires_grad_(False)
            self.guide_backbone.eval()

    def _sample_guide_token_mask(self, guide_features):
        if self.guide_tokenbook_sample_rate >= 1.0:
            return None

        batch_size = guide_features.shape[0]
        token_count = int(guide_features.shape[2] * guide_features.shape[3] * guide_features.shape[4])
        mask = torch.rand(batch_size, token_count, device=guide_features.device) < self.guide_tokenbook_sample_rate
        if mask.sum(dim=1).min().item() == 0:
            random_indices = torch.randint(0, token_count, (batch_size,), device=guide_features.device)
            mask[torch.arange(batch_size, device=guide_features.device), random_indices] = True
        return mask

    @staticmethod
    def _compile_module_in_place(module: nn.Module) -> nn.Module:
        compile_method = getattr(module, "compile", None)
        if callable(compile_method):
            compile_method()
            return module
        return torch.compile(module)

    def _compile_guidance_submodules(self, *, device_type: str) -> list[str]:
        if not self.guide_enabled or device_type != "cuda":
            return []

        compiled_modules: list[str] = []
        if self.guide_compile_policy == "backbone_only" and self.guide_backbone is not None:
            self.guide_backbone = self._compile_module_in_place(self.guide_backbone)
            compiled_modules.append("guide_backbone")
        elif self.guide_compile_policy == "tokenbook_only" and self.guide_tokenbook is not None:
            self.guide_tokenbook = self._compile_module_in_place(self.guide_tokenbook)
            compiled_modules.append("guide_tokenbook")
        return compiled_modules

    @torch.compiler.disable(reason="guided backbone compile failure")
    def _apply_input_guidance(self, x):
        if not self.guide_enabled:
            return x, {}

        if self.guide_freeze:
            with torch.inference_mode():
                frozen_features = self.guide_backbone(x)[0]
            guide_features = frozen_features.clone()
        else:
            guide_features = self.guide_backbone(x)[0]

        token_mask = self._sample_guide_token_mask(guide_features)
        guide_mask = self.guide_tokenbook(guide_features, token_mask=token_mask)
        guide_for_input = F.interpolate(
            guide_mask,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return x * guide_for_input, {"guide_mask": guide_mask}
    
    def _init_pretrained_backbone(self, mgr, model_config):
        print(f"--- Initializing pretrained backbone '{self.pretrained_backbone}' ---")
        input_shape = tuple(model_config.get("input_shape", self.patch_size))
        decoder_type = model_config.get("pretrained_decoder_type", "primus_patch_decode")
        config_path = model_config.get("pretrained_backbone_config_path")

        self.shared_encoder = build_dinov2_backbone(
            self.pretrained_backbone,
            input_channels=self.in_channels,
            input_shape=input_shape,
            config_path=config_path,
        )
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        separate_decoders_default = model_config.get("separate_decoders", True)
        decoder_head_channels = model_config.get("decoder_head_channels", 32)
        tasks_using_shared, tasks_using_separate = set(), set()

        for target_name, target_info in self.targets.items():
            if 'out_channels' in target_info:
                out_channels = target_info['out_channels']
            elif 'channels' in target_info:
                out_channels = target_info['channels']
            else:
                out_channels = self.in_channels
                print(f"No channel specification found for task '{target_name}', defaulting to {out_channels} channels")
            target_info["out_channels"] = out_channels
            self._register_target_projection(target_name, target_info, model_config)

            use_separate = target_info.get("separate_decoder", separate_decoders_default)
            if use_separate:
                tasks_using_separate.add(target_name)
            else:
                tasks_using_shared.add(target_name)

        if len(tasks_using_shared) > 0:
            self.shared_decoder = build_dinov2_decoder(decoder_type, self.shared_encoder, decoder_head_channels)
            head_conv = nn.Conv2d if self.shared_encoder.ndim == 2 else nn.Conv3d
            for target_name in sorted(tasks_using_shared):
                out_ch = self.targets[target_name]["out_channels"]
                self.task_heads[target_name] = head_conv(
                    decoder_head_channels, out_ch, kernel_size=1, stride=1, padding=0, bias=True
                )
                activation_str = self.targets[target_name].get("activation", "none")
                self.task_activations[target_name] = get_activation_module(activation_str)
                print(
                    f"Pretrained task '{target_name}' configured with shared {decoder_type} decoder + head ({out_ch} channels)"
                )

        for target_name in sorted(tasks_using_separate):
            out_channels = self.targets[target_name]["out_channels"]
            activation_str = self.targets[target_name].get("activation", "none")
            self.task_decoders[target_name] = build_dinov2_decoder(
                decoder_type,
                self.shared_encoder,
                out_channels,
            )
            self.task_activations[target_name] = get_activation_module(activation_str)
            print(f"Pretrained task '{target_name}' configured with separate {decoder_type} decoder ({out_channels} channels)")

        self.final_config = {
            "model_name": self.mgr.model_name,
            "architecture_type": self.architecture_type,
            "pretrained_backbone": self.pretrained_backbone,
            "pretrained_decoder_type": decoder_type,
            "input_shape": input_shape,
            "in_channels": self.in_channels,
            "targets": self.targets,
            "target_z_projection": self.task_z_projection_cfg,
            "separate_decoders": len(tasks_using_separate) > 0,
            "decoder_head_channels": decoder_head_channels,
        }

        print("Pretrained backbone network initialized with configuration:")
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
            patch_embed_size = (patch_embed_size,) * len(self.patch_size)
        
        # Ensure input shape is specified
        input_shape = model_config.get("input_shape", self.patch_size)
        if input_shape is None:
            raise ValueError("input_shape must be specified for Primus architecture")
        
        # Get Primus-specific parameters
        primus_kwargs = {
            # Align unspecified Primus configs with the MIC-DKFZ nnUNet Primus trainers.
            "drop_path_rate": model_config.get("drop_path_rate", 0.2),
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

        # Default Primus to per-task decoders to mirror the direct Primus head layout.
        separate_decoders_default = model_config.get("separate_decoders", True)
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
            self._register_target_projection(target_name, target_info, model_config)

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
            "target_z_projection": self.task_z_projection_cfg,
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.guide_enabled and self.guide_freeze and self.guide_backbone is not None:
            self.guide_backbone.eval()
        return self

    def forward(self, x, return_mae_mask=False, return_aux=False):
        # Check input channels and warn if mismatch
        self.check_input_channels(x)
        aux_outputs = {}

        if self.guide_enabled:
            x, aux_outputs = self._apply_input_guidance(x)

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
            logits = self._apply_z_projection(task_name, logits)
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
                logits = self._apply_z_projection(task_name, logits)
                activation_fn = self.task_activations[task_name] if task_name in self.task_activations else None
                if activation_fn is not None and not self.training:
                    if isinstance(logits, (list, tuple)):
                        logits = type(logits)(activation_fn(l) for l in logits)
                    else:
                        logits = activation_fn(logits)
                results[task_name] = logits
        if getattr(self, 'return_shared_features', False) and shared_features is not None:
            results['shared_features'] = shared_features
        
        # Return MAE mask if requested (for MAE training)
        if return_mae_mask:
            if return_aux:
                return results, restoration_mask, aux_outputs
            return results, restoration_mask
        if return_aux:
            return results, aux_outputs
        return results
