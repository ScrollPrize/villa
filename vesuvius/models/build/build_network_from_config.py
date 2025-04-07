import torch.nn as nn
from .utils import get_pool_and_conv_props, get_n_blocks_per_stage
from .blocks.encoder import Encoder
from .blocks.decoder import Decoder

def get_activation_module(activation_str: str):
    act_str = activation_str.lower()
    if act_str == "none":
        return None
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation type: {activation_str}")

class NetworkFromConfig(nn.Module):
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr
        # Set targets from the config (if they exist)
        self.targets = mgr.targets  # Defined in dataset_config.targets
        self.patch_size = mgr.train_patch_size
        self.batch_size = mgr.train_batch_size
        self.in_channels = mgr.in_channels
        self.vram_target = mgr.vram_max
        self.autoconfigure = mgr.autoconfigure

        # Choose the config dictionary (model_config has precedence)
        if mgr.model_config:
            model_config = mgr.model_config
        else:
            print("model_config is empty; using inference_config instead")
            model_config = mgr.inference_config

        self.save_config = True

        # Determine if we are loading an nnUNet checkpoint.
        self.load_from_nnunet = mgr.inference_config.get("load_from_nnunet", False)
        # Optionally, a flag to drop the background channel (if you only want the foreground)
        self.drop_background = mgr.inference_config.get("drop_background", False)

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

        # Determine op_dims from patch_size
        if len(self.patch_size) == 2:
            self.op_dims = 2
        elif len(self.patch_size) == 3:
            self.op_dims = 3
        else:
            raise ValueError("Patch size must have either 2 or 3 dimensions!")
        if isinstance(self.conv_op, str):
            self.conv_op = nn.Conv2d if self.op_dims == 2 else nn.Conv3d
        if isinstance(self.pool_op, str):
            self.pool_op = nn.AvgPool2d if self.op_dims == 2 else nn.AvgPool3d
        if isinstance(self.norm_op, str):
            self.norm_op = nn.InstanceNorm2d if self.op_dims == 2 else nn.InstanceNorm3d
        if isinstance(self.dropout_op, str):
            self.dropout_op = nn.Dropout2d if self.op_dims == 2 else nn.Dropout3d
        if self.nonlin in ["nn.LeakyReLU", "LeakyReLU"]:
            self.nonlin = nn.LeakyReLU
            self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        elif self.nonlin in ["nn.ReLU", "ReLU"]:
            self.nonlin = nn.ReLU
            self.nonlin_kwargs = {"inplace": True}

        # --------------------------------------------------------------------
        # Architecture parameters.
        # --------------------------------------------------------------------
        if self.autoconfigure:
            print("--- Autoconfiguring network from config ---")
            self.basic_encoder_block = "BasicBlockD"
            self.basic_decoder_block = "ConvBlock"
            self.bottleneck_block = "BasicBlockD"
            num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, final_patch_size, must_div = \
                get_pool_and_conv_props(
                    spacing=mgr.spacing,
                    patch_size=self.patch_size,
                    min_feature_map_size=4,
                    max_numpool=999999
                )
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
        else:
            print("--- Configuring network from config file ---")
            self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
            self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
            self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")
            self.features_per_stage = model_config.get("features_per_stage",
                                                        mgr.inference_config.get("features_per_stage",
                                                                                   [32, 64, 128, 256, 320, 320, 320]))
            self.num_stages = model_config.get("n_stages", 7)
            self.n_blocks_per_stage = model_config.get("n_blocks_per_stage", [1, 3, 4, 6, 6, 6, 6])
            self.kernel_sizes = model_config.get("kernel_sizes", [[3, 3, 3]] * self.num_stages)
            self.pool_op_kernel_sizes = model_config.get("pool_op_kernel_sizes", [[1, 1, 1]] * self.num_stages)
            self.n_conv_per_stage_decoder = model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1))
            self.strides = model_config.get("strides", self.pool_op_kernel_sizes)

        # Derive stem channels from first feature map if not provided.
        self.stem_n_channels = self.features_per_stage[0]

        # --------------------------------------------------------------------
        # Build network.
        # If loading an nnUNet checkpoint, import and instantiate the original ResidualEncoderUNet.
        # Otherwise, use your original multi-target mode.
        # --------------------------------------------------------------------
        if self.load_from_nnunet:
            print("Building model in nnUNet mode using ResidualEncoderUNet.")
            # In nnUNet mode, assume a single target.
            self.target_name = list(mgr.targets.keys())[0] if mgr.targets else "default_target"
            self.out_channels = 2  # Force 2 channels (background + foreground)
            # Overwrite targets so that final_config can reference them.
            self.targets = {self.target_name: {"out_channels": self.out_channels, "activation": "softmax"}}
            print(f"nnUNet mode: Using 2 output channels (background + foreground) with softmax activation")
            # Import the ResidualEncoderUNet and valid blocks.
            from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
            from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
            self.model = ResidualEncoderUNet(
                input_channels=self.in_channels,
                n_stages=model_config.get("n_stages", 7),
                features_per_stage=model_config.get("features_per_stage",
                                  mgr.inference_config.get("features_per_stage", [32, 64, 128, 256, 320, 320, 320])),
                conv_op=self.conv_op,
                kernel_sizes=model_config.get("kernel_sizes", [[3,3,3]] * model_config.get("n_stages", 7)),
                strides=model_config.get("strides", self.strides),
                n_blocks_per_stage=model_config.get("n_blocks_per_stage", [1, 3, 4, 6, 6, 6, 6]),
                num_classes=self.out_channels,
                n_conv_per_stage_decoder=model_config.get("n_conv_per_stage_decoder", [1]*(model_config.get("n_stages",7)-1)),
                conv_bias=self.conv_bias,
                norm_op=self.norm_op,
                norm_op_kwargs=self.norm_op_kwargs,
                dropout_op=self.dropout_op,
                dropout_op_kwargs=self.dropout_op_kwargs,
                nonlin=self.nonlin,
                nonlin_kwargs=self.nonlin_kwargs,
                deep_supervision=True,
                block=model_config.get("block", BasicBlockD),
                bottleneck_channels=model_config.get("bottleneck_channels", None),
                stem_channels=model_config.get("stem_channels", self.features_per_stage[0])
            )
        else:
            # Original multi-target mode.
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
            for target_name, target_info in self.targets.items():
                # Ensure that binary segmentation tasks have 2 output channels (background + foreground)
                out_channels = target_info.get("out_channels", 2)
                is_segmentation = target_info.get("is_segmentation", True)
                
                # For binary segmentation tasks, enforce 2 output channels to match nnUNet convention
                if out_channels == 1 and is_segmentation:
                    out_channels = 2
                    print(f"Enforcing 2 output channels for task '{target_name}' to match nnUNet convention (background + foreground)")
                    # Update the target info for consistency
                    self.targets[target_name]["out_channels"] = out_channels
                
                # Set appropriate activation
                if "activation" not in target_info:
                    if out_channels > 1:
                        # Multi-class: use softmax
                        self.targets[target_name]["activation"] = "softmax"
                    else:
                        # Single output channel (e.g., regression): use sigmoid
                        self.targets[target_name]["activation"] = "sigmoid"
                
                activation_str = self.targets[target_name].get("activation", "none")
                
                # Create the decoder
                self.task_decoders[target_name] = Decoder(
                    encoder=self.shared_encoder,
                    basic_block=model_config.get("basic_decoder_block", "ConvBlock"),
                    num_classes=out_channels,
                    n_conv_per_stage=model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
                    deep_supervision=False
                )
                self.task_activations[target_name] = get_activation_module(activation_str)
                
                print(f"Created decoder for task '{target_name}' with {out_channels} output channels and '{activation_str}' activation")

        # --------------------------------------------------------------------
        # Build final configuration snapshot.
        # --------------------------------------------------------------------
        self.final_config = {
            "model_name": self.mgr.tr_info.get("model_name", "Model"),
            "use_timm_encoder": model_config.get("use_timm_encoder", False),
            "basic_encoder_block": self.basic_encoder_block,
            "basic_decoder_block": model_config.get("basic_decoder_block", "ConvBlock"),
            "bottleneck_block": self.bottleneck_block,
            "features_per_stage": self.features_per_stage,
            "num_stages": self.num_stages,
            "n_blocks_per_stage": self.n_blocks_per_stage,
            "n_conv_per_stage_decoder": (model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1))
                                         if not self.load_from_nnunet else [1] * (self.num_stages - 1)),
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
            "return_skips": (model_config.get("return_skips", True)
                             if not self.load_from_nnunet else True),
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
            "vram_target": self.vram_target,
            "autoconfigure": self.autoconfigure,
            "targets": self.targets  # In nnUNet mode, this is now set to a single target.
        }
        if self.save_config:
            mgr.model_config["final_config"] = self.final_config

        print("NetworkFromConfig initialized with final configuration:")
        for k, v in self.final_config.items():
            print(f"  {k}: {v}")

    def forward(self, x):
        if self.load_from_nnunet:
            # Use the imported ResidualEncoderUNet model.
            out = self.model(x)
            if self.drop_background:
                # Drop background channel (assume channel 0 is background).
                out = out[:, 1:]
            return {self.target_name: out}
        else:
            skips = self.shared_encoder(x)
            results = {}
            for task_name, decoder in self.task_decoders.items():
                logits = decoder(skips)
                
                # During training, don't apply activation - most loss functions (CrossEntropyLoss, BCEWithLogitsLoss)
                # expect raw logits without activation applied
                # During inference, apply activation for final prediction if specified
                activation_fn = self.task_activations[task_name]
                if activation_fn is not None and not self.training:
                    logits = activation_fn(logits)
                
                results[task_name] = logits
            return results

    def load_state_dict(self, state_dict, strict=True):
        if self.load_from_nnunet:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Remove known prefixes
                if new_key.startswith("module."):
                    new_key = new_key[len("module."):]
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]

                # Remap the module index from 1 to 2.
                # (Adjust this replacement if your situation is more complex.)
                new_key = new_key.replace("all_modules.1", "all_modules.2")
                new_state_dict[new_key] = value
            return self.model.load_state_dict(new_state_dict, strict=strict)
        else:
            return super().load_state_dict(state_dict, strict=strict)