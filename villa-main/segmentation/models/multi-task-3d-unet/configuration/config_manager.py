import json
import yaml
from pathlib import Path
import torch.nn as nn

class ConfigManager:
    def __init__(self, config_file):
        self._config_path = Path(config_file)
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Load the various config sections.
        self.tr_info = config.get("tr_setup", {})
        self.tr_configs = config.get("tr_config", {})
        self.model_config = config.get("model_config", {})  # used for training (if needed)
        self.dataset_config = config.get("dataset_config", {})
        self.inference_config = config.get("inference_config", {})

        # Example training parameters (you may already have these)
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.in_channels = 1  # or however you determine this

        # *** Option 1: Load nnUNet settings for training if desired ***
        if self.model_config.get("load_from_nnunet", False):
            self._load_nnunet_config(source="model_config", patch_size=self.train_patch_size)

        # *** Option 2: Load nnUNet settings for inference if flagged in inference_config ***
        if self.inference_config.get("load_from_nnunet", False):
            # You can choose which patch size to use (for inference you might have a different one)
            infer_patch_size = tuple(self.inference_config.get("patch_size", self.train_patch_size))
            self._load_nnunet_config(source="inference_config", patch_size=infer_patch_size)

        self.model_name = self.tr_info.get("model_name", "Model")
        self.vram_max = float(self.tr_info.get("vram_max", 16))
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.dilate_label = bool(self.tr_info.get("dilate_label", False))

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
        if not self.ckpt_out_base.exists():
            self.ckpt_out_base.mkdir(parents=True)
        ckpt_path = self.tr_info.get("checkpoint_path", None)
        self.checkpoint_path = Path(ckpt_path) if ckpt_path else None
        self.load_weights_only = bool(self.tr_info.get("load_weights_only", False))
        self.tensorboard_log_dir = self.tr_info.get("tensorboard_log_dir", "./tensorboard_logs/")

        # Training config
        self.optimizer = self.tr_configs.get("optimizer", "AdamW")
        self.initial_lr = float(self.tr_configs.get("initial_lr", 1e-3))
        self.weight_decay = float(self.tr_configs.get("weight_decay", 0))
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.train_batch_size = int(self.tr_configs.get("batch_size", 2))
        self.gradient_accumulation = int(self.tr_configs.get("gradient_accumulation", 1))
        self.max_steps_per_epoch = int(self.tr_configs.get("max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(self.tr_configs.get("max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(self.tr_configs.get("num_dataloader_workers", 4))
        self.max_epoch = int(self.tr_configs.get("max_epoch", 1000))

        # Dataset config
        self.min_labeled_ratio = float(self.dataset_config.get("min_labeled_ratio", 0.10))
        self.min_bbox_percent = float(self.dataset_config.get("min_bbox_percent", 0.95))
        self.use_cache = bool(self.dataset_config.get("use_cache", True))
        self.cache_folder = Path(self.dataset_config.get("cache_folder", "patch_cache"))
        self.targets = self.dataset_config.get("targets", {})

        # model config
        self.use_timm = self.model_config.get("use_timm", False)
        self.timm_encoder_class = self.model_config.get("timm_encoder_class", None)

        # wk configuration
        self.is_wk = self.dataset_config.get("is_wk", False)
        self.is_wk_zarr_link = self.dataset_config.get("is_wk_zarr_link", False)
        self.wk_url = self.dataset_config.get("wk_url", None)
        self.wk_token = self.dataset_config.get("wk_token", None)
        if self.is_wk and (self.wk_url is None or self.wk_token is None):
            raise ValueError("If is_wk is True, both wk_url and wk_token must be specified")

        # channel configuration
        self.in_channels = 1
        self.spacing= [1, 1, 1]
        self.out_channels = ()
        for target_name, task_info in self.targets.items():
            if 'out_channels' not in task_info:
                raise ValueError(f"Target {target_name} is missing out_channels specification")
            self.out_channels += (task_info['out_channels'],)

        # Inference config
        self.infer_checkpoint_path = self.inference_config.get("checkpoint_path", None)
        self.infer_patch_size = tuple(self.inference_config.get("patch_size", self.train_patch_size))
        self.infer_batch_size = int(self.inference_config.get("batch_size", self.train_batch_size))
        self.infer_input_path = self.inference_config.get("input_path", None)
        self.infer_input_format = str(getattr(self.inference_config, "input_format", "zarr"))
        self.infer_output_path = self.inference_config.get("output_path", "./outputs")
        self.infer_output_format = str(getattr(self.inference_config, "output_format", "zarr"))
        self.infer_output_format = self.inference_config.get("output_format", "zarr")
        self.infer_type = self.inference_config.get("type", "np.uint8")
        self.infer_load_all = bool(getattr(self.inference_config, "load_all", False))
        self.infer_output_targets = self.inference_config.get("output_targets", ['all'])
        self.infer_overlap = float(self.inference_config.get("overlap", 0.25))
        self.load_strict = bool(getattr(self.inference_config, "load_strict", True))
        self.infer_num_dataloader_workers = int(
            getattr(self.inference_config, "num_dataloader_workers", self.train_num_dataloader_workers))

        if self.model_config.get("load_from_nnunet", False):
            nnunet_json_path = self.model_config.get("nnunet_json_path", None)
            if nnunet_json_path is None:
                raise ValueError("load_from_nnunet is True but no nnunet_json_path provided in model_config.")
            with open(nnunet_json_path, 'r') as f:
                nnunet_config = json.load(f)

            # Decide which configuration to use based on patch dimensions.
            # For example, if patch size is 2D (length==2) choose the "2d" config; otherwise, use "3d_fullres".
            if len(self.train_patch_size) == 2:
                res_key = "2d"
            else:
                res_key = "3d_fullres"
            if res_key not in nnunet_config["configurations"]:
                raise ValueError(f"{res_key} configuration not found in the nnUNet JSON file.")
            nnunet_arch = nnunet_config["configurations"][res_key]["architecture"]["arch_kwargs"]

            # Now update self.model_config with keys expected by your network builder.
            self.model_config["n_stages"] = nnunet_arch["n_stages"]
            self.model_config["features_per_stage"] = nnunet_arch["features_per_stage"]
            self.model_config["kernel_sizes"] = nnunet_arch["kernel_sizes"]
            self.model_config["strides"] = nnunet_arch["strides"]
            self.model_config["n_blocks_per_stage"] = nnunet_arch["n_blocks_per_stage"]
            self.model_config["n_conv_per_stage_decoder"] = nnunet_arch["n_conv_per_stage_decoder"]
            self.model_config["conv_bias"] = nnunet_arch["conv_bias"]

            # For operations specified by string, convert them to actual objects.
            # Note: the nnUNet JSON provides strings such as "torch.nn.modules.conv.Conv3d".
            conv_op_str = nnunet_arch.get("conv_op")
            norm_op_str = nnunet_arch.get("norm_op")
            nonlin_str = nnunet_arch.get("nonlin")
            if conv_op_str is not None:
                # Convert, for example, "torch.nn.modules.conv.Conv3d" to nn.Conv3d
                self.model_config["conv_op"] = getattr(nn, conv_op_str.split('.')[-1])
            if norm_op_str is not None:
                self.model_config["norm_op"] = getattr(nn, norm_op_str.split('.')[-1])
            if nonlin_str is not None:
                self.model_config["nonlin"] = getattr(nn, nonlin_str.split('.')[-1])

            # The kwargs can be transferred directly
            self.model_config["norm_op_kwargs"] = nnunet_arch.get("norm_op_kwargs", {})
            self.model_config["nonlin_kwargs"] = nnunet_arch.get("nonlin_kwargs", {})

            # For dropout, you may want to do something similar (here we simply pass the values):
            self.model_config["dropout_op"] = nn.Dropout3d if nnunet_arch.get(
                "dropout_op") is None else None  # adjust as needed
            self.model_config["dropout_op_kwargs"] = nnunet_arch.get("dropout_op_kwargs", {})

            print("Loaded nnUNet configuration from JSON:")
            print(self.model_config)



        self._print_summary()

    def _load_nnunet_config(self, source, patch_size):
        """
        Load nnUNet configuration from the JSON file indicated in the config section (either
        "model_config" or "inference_config") and update that section with the extracted architecture parameters.
        """
        # Choose the config dictionary based on the source
        if source == "model_config":
            cfg = self.model_config
        elif source == "inference_config":
            cfg = self.inference_config
        else:
            raise ValueError("Unknown source for nnUNet config loading.")

        nnunet_json_path = cfg.get("nnunet_json_path", None)
        if nnunet_json_path is None:
            raise ValueError(f"load_from_nnunet is true in {source} but no nnunet_json_path was provided.")

        # Read the nnUNet JSON file
        with open(nnunet_json_path, 'r') as f:
            nnunet_config = json.load(f)

        # Decide which configuration to use (for example, if patch size is 2D or 3D)
        if len(patch_size) == 2:
            res_key = "2d"
        else:
            res_key = "3d_fullres"
        if res_key not in nnunet_config["configurations"]:
            raise ValueError(f"{res_key} configuration not found in the nnUNet JSON file.")

        nnunet_arch = nnunet_config["configurations"][res_key]["architecture"]["arch_kwargs"]

        # Update the configuration dictionary with keys your network builder expects.
        cfg["n_stages"] = nnunet_arch["n_stages"]
        cfg["features_per_stage"] = nnunet_arch["features_per_stage"]
        cfg["kernel_sizes"] = nnunet_arch["kernel_sizes"]
        cfg["strides"] = nnunet_arch["strides"]
        cfg["n_blocks_per_stage"] = nnunet_arch["n_blocks_per_stage"]
        cfg["n_conv_per_stage_decoder"] = nnunet_arch["n_conv_per_stage_decoder"]
        cfg["conv_bias"] = nnunet_arch["conv_bias"]

        # For keys that refer to classes (provided as strings), convert them.
        conv_op_str = nnunet_arch.get("conv_op")
        norm_op_str = nnunet_arch.get("norm_op")
        nonlin_str = nnunet_arch.get("nonlin")
        if conv_op_str is not None:
            # e.g. "torch.nn.modules.conv.Conv3d" --> nn.Conv3d
            cfg["conv_op"] = getattr(nn, conv_op_str.split('.')[-1])
        if norm_op_str is not None:
            cfg["norm_op"] = getattr(nn, norm_op_str.split('.')[-1])
        if nonlin_str is not None:
            cfg["nonlin"] = getattr(nn, nonlin_str.split('.')[-1])

        # Optionally copy over additional kwargs if present
        cfg["norm_op_kwargs"] = nnunet_arch.get("norm_op_kwargs", {})
        cfg["nonlin_kwargs"] = nnunet_arch.get("nonlin_kwargs", {})

        # You might need to adjust dropout settings, etc.
        # (Customize here as needed.)
        print(f"Loaded nnUNet configuration from JSON into {source} section:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

    def save_config(self):
        """
        Dump the current config (including any updates in model_config["final_config"])
        to a YAML file in the same directory as the original config, but
        with "_final" appended before the extension.

        E.g. "my_config.yaml" -> "my_config_final.yaml"
        """
        # Reconstruct the full config dictionary
        combined_config = {
            "tr_setup": self.tr_info,
            "tr_config": self.tr_configs,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "inference_config": self.inference_config,
        }

        # Figure out original path parts
        original_stem = self._config_path.stem  # e.g. "my_config"
        original_ext = self._config_path.suffix  # e.g. ".yaml"
        original_parent = self._config_path.parent

        # Create the new filename with "_final" inserted
        final_filename = f"{original_stem}_final{original_ext}"

        # Full path to the new file
        final_path = original_parent / final_filename

        # Write out the YAML
        with final_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {final_path}")

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

        print("\nInference Config (inference_config):")
        for k, v in self.inference_config.items():
            print(f"  {k}: {v}")
        print("____________________________________________")
