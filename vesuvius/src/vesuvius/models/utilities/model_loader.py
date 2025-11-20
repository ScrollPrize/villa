import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False

__all__ = [
    "load_for_inference",
    "resume_training_from_checkpoint",
    "load_checkpoint",
]


DeviceLike = Union[str, torch.device]


def _load_checkpoint_file(checkpoint_path: Union[str, Path], device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary and no state_dict could be found.")

    candidates = [
        "model",
        "model_state_dict",
        "state_dict",
        "network",
        "network_weights",
        "net",
        "student",
    ]
    for key in candidates:
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]

    if all(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError("Could not locate a valid state_dict inside checkpoint.")


def _canonicalize_state_dict(state_dict: Dict[str, Any], strip_common_prefix: bool = True) -> Dict[str, Any]:
    def _strip_prefixes(key: str) -> str:
        prefixes = ("_orig_mod.", "module.")
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p) :]
                    changed = True
        return key

    cleaned = {_strip_prefixes(k): v for k, v in state_dict.items()}

    if strip_common_prefix and cleaned:
        keys = list(cleaned.keys())
        first = keys[0]
        if "." in first:
            prefix = first.split(".", 1)[0]
            share = sum(1 for k in keys if k.startswith(prefix + ".")) / len(keys)
            if share > 0.9:
                cleaned = {k.split(".", 1)[1]: v for k, v in cleaned.items()}

    return cleaned


def _infer_targets_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    inferred = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("task_heads.") and k.endswith(".weight"):
            try:
                tgt_name = k.split(".")[1]
            except Exception:
                continue
            try:
                out_ch = int(v.shape[0])
            except Exception:
                out_ch = 1
            inferred[tgt_name] = {"out_channels": out_ch, "activation": "none"}
    return inferred


def _load_nnunet_export(
    model_folder: Path,
    fold: Union[int, str],
    checkpoint_name: str,
    device: torch.device,
    patch_size_override: Optional[Tuple[int, ...]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    try:
        from batchgenerators.utilities.file_and_folder_operations import load_json, join
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from torch._dynamo import OptimizedModule
    except ImportError as e:  # pragma: no cover
        raise ImportError("nnunetv2 is required to load nnU-Net export directories.") from e

    model_path = model_folder
    if model_path.name.startswith("fold_"):
        model_path = model_path.parent

    dataset_json_path = join(str(model_path), "dataset.json")
    plans_json_path = join(str(model_path), "plans.json")

    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"dataset.json not found at: {dataset_json_path}")
    if not os.path.exists(plans_json_path):
        raise FileNotFoundError(f"plans.json not found at: {plans_json_path}")

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_json_path)
    plans_manager = PlansManager(plans)

    fold_str = str(fold)
    if model_folder.name.startswith("fold_"):
        checkpoint_file = join(str(model_folder), checkpoint_name)
    else:
        fold_dir = fold_str if fold_str.startswith("fold_") else f"fold_{fold_str}"
        checkpoint_file = join(str(model_folder), fold_dir, checkpoint_name)

    if not os.path.exists(checkpoint_file) and checkpoint_name == "checkpoint_final.pth":
        alt = "checkpoint_best.pth"
        checkpoint_file = (
            join(str(model_folder), alt)
            if model_folder.name.startswith("fold_")
            else join(str(model_folder), f"fold_{fold}", alt)
        )
        if os.path.exists(checkpoint_file):
            checkpoint_name = alt

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    try:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    configuration_manager = plans_manager.get_configuration(configuration_name)

    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)

    # Normalize nnU-Net arch kwargs (notably conv_op may come as a dotted string)
    def _normalize_arch_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        import torch.nn as nn

        fixed = dict(kwargs)
        if "conv_op" in fixed:
            co = fixed["conv_op"]
            mapping = {
                "Conv3d": nn.Conv3d,
                "torch.nn.modules.conv.Conv3d": nn.Conv3d,
                "nn.Conv3d": nn.Conv3d,
                "Conv2d": nn.Conv2d,
                "torch.nn.modules.conv.Conv2d": nn.Conv2d,
                "nn.Conv2d": nn.Conv2d,
                "Conv1d": nn.Conv1d,
                "torch.nn.modules.conv.Conv1d": nn.Conv1d,
                "nn.Conv1d": nn.Conv1d,
            }
            if isinstance(co, str) and co in mapping:
                fixed["conv_op"] = mapping[co]
            elif hasattr(co, "__name__") and co.__name__ in mapping:
                fixed["conv_op"] = mapping[co.__name__]
        if "norm_op" in fixed:
            no = fixed["norm_op"]
            norm_map = {
                "InstanceNorm3d": nn.InstanceNorm3d,
                "torch.nn.modules.instancenorm.InstanceNorm3d": nn.InstanceNorm3d,
                "nn.InstanceNorm3d": nn.InstanceNorm3d,
                "InstanceNorm2d": nn.InstanceNorm2d,
                "torch.nn.modules.instancenorm.InstanceNorm2d": nn.InstanceNorm2d,
                "nn.InstanceNorm2d": nn.InstanceNorm2d,
                "BatchNorm3d": nn.BatchNorm3d,
                "torch.nn.modules.batchnorm.BatchNorm3d": nn.BatchNorm3d,
                "nn.BatchNorm3d": nn.BatchNorm3d,
                "BatchNorm2d": nn.BatchNorm2d,
                "torch.nn.modules.batchnorm.BatchNorm2d": nn.BatchNorm2d,
                "nn.BatchNorm2d": nn.BatchNorm2d,
            }
            if isinstance(no, str) and no in norm_map:
                fixed["norm_op"] = norm_map[no]
            elif hasattr(no, "__name__") and no.__name__ in norm_map:
                fixed["norm_op"] = norm_map[no.__name__]
        if "dropout_op" in fixed:
            do = fixed["dropout_op"]
            drop_map = {
                "Dropout3d": nn.Dropout3d,
                "torch.nn.modules.dropout.Dropout3d": nn.Dropout3d,
                "nn.Dropout3d": nn.Dropout3d,
                "Dropout2d": nn.Dropout2d,
                "torch.nn.modules.dropout.Dropout2d": nn.Dropout2d,
                "nn.Dropout2d": nn.Dropout2d,
            }
            if isinstance(do, str) and do in drop_map:
                fixed["dropout_op"] = drop_map[do]
            elif hasattr(do, "__name__") and do.__name__ in drop_map:
                fixed["dropout_op"] = drop_map[do.__name__]
        if "nonlin" in fixed:
            nl = fixed["nonlin"]
            nonlin_map = {
                "LeakyReLU": nn.LeakyReLU,
                "torch.nn.modules.activation.LeakyReLU": nn.LeakyReLU,
                "torch.nn.LeakyReLU": nn.LeakyReLU,
                "nn.LeakyReLU": nn.LeakyReLU,
                "ReLU": nn.ReLU,
                "torch.nn.modules.activation.ReLU": nn.ReLU,
                "torch.nn.ReLU": nn.ReLU,
                "nn.ReLU": nn.ReLU,
            }
            if isinstance(nl, str) and nl in nonlin_map:
                fixed["nonlin"] = nonlin_map[nl]
            elif hasattr(nl, "__name__") and nl.__name__ in nonlin_map:
                fixed["nonlin"] = nonlin_map[nl.__name__]
        return fixed

    arch_kwargs = _normalize_arch_kwargs(configuration_manager.network_arch_init_kwargs)

    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )

    network = None
    if trainer_class is not None:
        try:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                arch_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            )
        except Exception:
            network = None

    if network is None:
        network_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "network_architecture"),
            configuration_manager.network_arch_class_name,
            current_module="nnunetv2.network_architecture",
        )
        if network_class is None:
            # Fallback: try importing by dotted path (e.g., dynamic_network_architectures.architectures.unet.ResidualEncoderUNet)
            arch_cls = configuration_manager.network_arch_class_name
            if "." in arch_cls:
                try:
                    import importlib

                    module_path, cls_name = arch_cls.rsplit(".", 1)
                    mod = importlib.import_module(module_path)
                    network_class = getattr(mod, cls_name)
                except Exception:
                    network_class = None
        if network_class is None:
            raise RuntimeError(
                f"Network architecture class {configuration_manager.network_arch_class_name} not found in nnunetv2 or importable by dotted path."
            )
        network = network_class(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=False,
            **arch_kwargs,
        )

    network = network.to(device)

    network_state_dict = checkpoint["network_weights"]
    if not isinstance(network, OptimizedModule):
        network.load_state_dict(network_state_dict)
    else:
        network._orig_mod.load_state_dict(network_state_dict)

    network.eval()

    should_compile = os.environ.get("nnUNet_compile", "true").lower() in ("true", "1", "t")
    if should_compile and not isinstance(network, OptimizedModule):
        try:
            network = torch.compile(network)
        except Exception:
            pass

    patch_size = (
        tuple(patch_size_override)
        if patch_size_override is not None
        else tuple(configuration_manager.patch_size)
    )

    return {
        "network": network,
        "plans_manager": plans_manager,
        "configuration_manager": configuration_manager,
        "dataset_json": dataset_json,
        "label_manager": label_manager,
        "trainer_name": trainer_name,
        "num_input_channels": num_input_channels,
        "num_seg_heads": label_manager.num_segmentation_heads,
        "patch_size": patch_size,
        "allowed_mirroring_axes": checkpoint.get("inference_allowed_mirroring_axes"),
        "source": "nnunet_export",
    }


def _load_train_checkpoint_for_inference(
    checkpoint_path: Path, device: torch.device, verbose: bool = False
) -> Dict[str, Any]:
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    raw_state = _extract_state_dict(checkpoint)
    model_state = _canonicalize_state_dict(raw_state, strip_common_prefix=True)

    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    from vesuvius.models.configuration.config_manager import ConfigManager
    from vesuvius.utils.utils import determine_dimensionality

    mgr = ConfigManager(verbose=verbose)
    mgr.tr_info = getattr(mgr, "tr_info", {})
    mgr.tr_configs = getattr(mgr, "tr_configs", {})
    mgr.model_config = getattr(mgr, "model_config", {})
    mgr.dataset_config = getattr(mgr, "dataset_config", {})

    model_config = checkpoint.get("model_config", {})

    if "patch_size" in model_config:
        mgr.tr_configs["patch_size"] = list(model_config["patch_size"])
    if "targets" in model_config:
        mgr.targets = model_config["targets"]
        mgr.dataset_config["targets"] = model_config["targets"]
    else:
        mgr.targets = {}

    mgr._init_attributes()

    if "patch_size" in model_config:
        mgr.train_patch_size = tuple(model_config["patch_size"])
        mgr.tr_configs["patch_size"] = list(model_config["patch_size"])
        dim_props = determine_dimensionality(mgr.train_patch_size, getattr(mgr, "verbose", False))
        mgr.model_config["conv_op"] = dim_props["conv_op"]
        mgr.model_config["pool_op"] = dim_props["pool_op"]
        mgr.model_config["norm_op"] = dim_props["norm_op"]
        mgr.model_config["dropout_op"] = dim_props["dropout_op"]
        mgr.spacing = dim_props["spacing"]
        mgr.op_dims = dim_props["op_dims"]

    if "targets" in model_config:
        mgr.targets = model_config["targets"]
    if "in_channels" in model_config:
        mgr.in_channels = model_config["in_channels"]
    if "autoconfigure" in model_config:
        mgr.autoconfigure = model_config["autoconfigure"]

    mgr.model_config = model_config or {}
    for key, value in model_config.items():
        if not hasattr(mgr, key):
            setattr(mgr, key, value)

    if "dataset_config" in checkpoint:
        dataset_config = checkpoint["dataset_config"]
        if "normalization_scheme" in dataset_config:
            mgr.normalization_scheme = dataset_config["normalization_scheme"]
        if "intensity_properties" in dataset_config:
            mgr.intensity_properties = dataset_config["intensity_properties"]
        mgr.dataset_config.update(dataset_config)

    try:
        mgr.enable_deep_supervision = False
    except Exception:
        pass

    if not getattr(mgr, "targets", None):
        inferred = _infer_targets_from_state_dict(model_state)
        if inferred:
            mgr.targets = inferred

    try:
        has_shared = any(str(k).startswith("shared_decoder.") for k in model_state.keys())
        has_task = any(str(k).startswith("task_decoders.") for k in model_state.keys())
        if "separate_decoders" not in mgr.model_config:
            if has_shared and not has_task:
                mgr.model_config["separate_decoders"] = False
            elif has_task and not has_shared:
                mgr.model_config["separate_decoders"] = True
    except Exception:
        pass

    model = NetworkFromConfig(mgr)

    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        exp_keys = list(model.state_dict().keys())[:10]
        got_keys = list(model_state.keys())[:10]
        msg = (
            f"Failed to load checkpoint weights into NetworkFromConfig.\n"
            f"- Expected (example keys): {exp_keys}\n"
            f"- Got (example keys): {got_keys}\n"
            f"Original error: {e}"
        )
        raise RuntimeError(msg)

    model = model.to(device)
    model.eval()

    patch_size = (
        tuple(mgr.train_patch_size)
        if getattr(mgr, "train_patch_size", None) is not None
        else tuple(model_config.get("patch_size", (192, 192, 192)))
    )
    targets = getattr(mgr, "targets", {}) or {}
    num_seg_heads = 0
    for tgt_cfg in targets.values():
        num_seg_heads += int(tgt_cfg.get("out_channels", 1))
    if num_seg_heads == 0:
        num_seg_heads = next(iter(model_state.values())).shape[0] if model_state else 1

    return {
        "network": model,
        "patch_size": patch_size,
        "num_input_channels": getattr(mgr, "in_channels", model_config.get("in_channels", 1)),
        "num_seg_heads": num_seg_heads,
        "targets": targets or None,
        "normalization_scheme": getattr(mgr, "normalization_scheme", None),
        "intensity_properties": getattr(mgr, "intensity_properties", None),
        "model_config": model_config,
        "source": "train_checkpoint",
    }


def load_for_inference(
    model_path: Optional[Union[str, Path]] = None,
    *,
    hf_model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    fold: Union[int, str] = 0,
    checkpoint_name: str = "checkpoint_final.pth",
    patch_size_override: Optional[Tuple[int, ...]] = None,
    device: DeviceLike = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    target_device = torch.device(device)

    if hf_model_path is not None:
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required to load models from Hugging Face.")
        temp_dir = tempfile.mkdtemp(prefix="vesuvius_hf_model_")
        try:
            download_path = snapshot_download(repo_id=hf_model_path, local_dir=temp_dir, token=hf_token)
            train_checkpoints = [
                f for f in os.listdir(download_path) if f.startswith("Model_epoch") and f.endswith(".pth")
            ]
            if train_checkpoints:
                ckpt_path = Path(download_path) / train_checkpoints[0]
                info = _load_train_checkpoint_for_inference(ckpt_path, device=target_device, verbose=verbose)
                info["temp_dir"] = temp_dir
                return info

            has_checkpoint = os.path.exists(os.path.join(download_path, checkpoint_name))
            has_plans = os.path.exists(os.path.join(download_path, "plans.json"))
            has_dataset = os.path.exists(os.path.join(download_path, "dataset.json"))
            if has_checkpoint and has_plans and has_dataset:
                fold_dir = os.path.join(download_path, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                shutil.copy(os.path.join(download_path, checkpoint_name), os.path.join(fold_dir, checkpoint_name))

            info = load_for_inference(
                Path(download_path),
                hf_model_path=None,
                fold=fold,
                checkpoint_name=checkpoint_name,
                patch_size_override=patch_size_override,
                device=target_device,
                verbose=verbose,
            )
            info["temp_dir"] = temp_dir
            return info
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    if model_path is None:
        raise ValueError("model_path or hf_model_path must be provided.")

    model_path = Path(model_path)

    if model_path.is_file():
        if model_path.suffix != ".pth":
            raise FileNotFoundError(f"Expected a .pth checkpoint file, got: {model_path}")
        parent = model_path.parent
        nnunet_root = parent.parent
        if (nnunet_root / "plans.json").exists() and (nnunet_root / "dataset.json").exists():
            fold_name = parent.name if parent.name.startswith("fold_") else fold
            info = _load_nnunet_export(
                model_folder=nnunet_root,
                fold=fold_name,
                checkpoint_name=model_path.name,
                device=target_device,
                patch_size_override=patch_size_override,
                verbose=verbose,
            )
        else:
            info = _load_train_checkpoint_for_inference(model_path, device=target_device, verbose=verbose)
    elif model_path.is_dir():
        has_dataset = (model_path / "dataset.json").exists()
        has_plans = (model_path / "plans.json").exists()
        if has_dataset and has_plans:
            info = _load_nnunet_export(
                model_folder=model_path,
                fold=fold,
                checkpoint_name=checkpoint_name,
                device=target_device,
                patch_size_override=patch_size_override,
                verbose=verbose,
            )
        else:
            pth_candidates = [p for p in model_path.iterdir() if p.is_file() and p.suffix == ".pth"]
            if pth_candidates:
                info = _load_train_checkpoint_for_inference(pth_candidates[0], device=target_device, verbose=verbose)
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found. Expected an nnU-Net export folder or a .pth file in {model_path}."
                )
    else:
        raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")

    if patch_size_override is not None:
        info["patch_size"] = tuple(patch_size_override)

    if verbose:
        num_classes = info.get("num_seg_heads", 1)
        if num_classes > 2:
            print(f"Detected multiclass model with {num_classes} classes")
        elif num_classes == 2:
            print("Detected binary segmentation model")
        else:
            print("Detected single-channel model")

    return info


def resume_training_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    mgr: Any,
    device: DeviceLike,
    load_weights_only: bool = False,
    verbose: bool = False,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"No valid checkpoint found at {checkpoint_path}")
        return model, optimizer, scheduler, 0, False

    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = _load_checkpoint_file(checkpoint_path, torch.device(device))

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        if hasattr(mgr, "targets") and "targets" in checkpoint["model_config"]:
            mgr.targets = checkpoint["model_config"]["targets"]
            if verbose:
                print(f"Updated targets from checkpoint: {mgr.targets}")

    if isinstance(checkpoint, dict) and "normalization_scheme" in checkpoint:
        mgr.normalization_scheme = checkpoint["normalization_scheme"]
        if hasattr(mgr, "dataset_config"):
            mgr.dataset_config["normalization_scheme"] = checkpoint["normalization_scheme"]

    if isinstance(checkpoint, dict) and "intensity_properties" in checkpoint:
        mgr.intensity_properties = checkpoint["intensity_properties"]
        if hasattr(mgr, "dataset_config"):
            mgr.dataset_config["intensity_properties"] = checkpoint["intensity_properties"]

    def _rebuild_model_from_checkpoint_config():
        nonlocal model, optimizer
        from vesuvius.models.build.build_network_from_config import NetworkFromConfig
        from vesuvius.models.training.optimizers import create_optimizer

        class ConfigWrapper:
            def __init__(self, config_dict, base_mgr):
                self.__dict__.update(config_dict)
                for attr_name in dir(base_mgr):
                    if not attr_name.startswith("__") and not hasattr(self, attr_name):
                        setattr(self, attr_name, getattr(base_mgr, attr_name))

        config_dict = checkpoint.get("model_config", {})
        config_wrapper = ConfigWrapper(config_dict, mgr)
        try:
            sep_dec = bool(config_dict.get("separate_decoders", False))
            if not sep_dec:
                setattr(config_wrapper, "enable_deep_supervision", False)
        except Exception:
            pass

        model = NetworkFromConfig(config_wrapper)
        model = model.to(device)

        optimizer_config = {
            "name": mgr.optimizer,
            "learning_rate": mgr.initial_lr,
            "weight_decay": mgr.weight_decay,
        }
        optimizer = create_optimizer(optimizer_config, model)
        try:
            mgr.model_config = config_dict
        except Exception:
            pass

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        force_rebuild = bool(getattr(mgr, "rebuild_from_checkpoint_config", False))
        if not load_weights_only or force_rebuild:
            _rebuild_model_from_checkpoint_config()

    model_state = _extract_state_dict(checkpoint)
    model_state = _canonicalize_state_dict(model_state, strip_common_prefix=True)

    try:
        target_keys = list(model.state_dict().keys())
        ckpt_keys = list(model_state.keys())
        target_has_module = all(k.startswith("module.") for k in target_keys) if target_keys else False
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys) if ckpt_keys else False
        if target_has_module and not ckpt_has_module:
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        elif not target_has_module and ckpt_has_module:
            model_state = {k[7:] if k.startswith("module.") else k: v for k, v in model_state.items()}
    except Exception:
        pass

    target_state = model.state_dict()
    allow_partial = bool(load_weights_only)
    if allow_partial:
        model_state = {k: v for k, v in model_state.items() if k in target_state and target_state[k].shape == v.shape}

    strict = getattr(mgr, "load_strict", not allow_partial)
    strict_loaded = False

    try:
        incompatible = model.load_state_dict(model_state, strict=strict)
        strict_loaded = True
        try:
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if (missing or unexpected) and not load_weights_only and isinstance(checkpoint, dict) and "model_config" in checkpoint:
                _rebuild_model_from_checkpoint_config()
                target_state = model.state_dict()
                model.load_state_dict(model_state, strict=True)
        except Exception:
            pass
    except RuntimeError:
        if strict and not allow_partial and isinstance(checkpoint, dict) and "model_config" in checkpoint:
            _rebuild_model_from_checkpoint_config()
            try:
                model.load_state_dict(model_state, strict=True)
                strict_loaded = True
            except RuntimeError:
                strict_loaded = False

        if not strict_loaded:
            target_state = model.state_dict()
            model_state_partial = {k: v for k, v in model_state.items() if k in target_state and target_state[k].shape == v.shape}
            model.load_state_dict(model_state_partial, strict=False)
            strict_loaded = False
            load_weights_only = True

    start_epoch = 0

    if (
        not load_weights_only
        and strict_loaded
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
        and "scheduler" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
    else:
        from vesuvius.models.training.lr_schedulers import get_scheduler

        scheduler_type = getattr(mgr, "scheduler", "poly")
        scheduler_kwargs = getattr(mgr, "scheduler_kwargs", {})

        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=mgr.initial_lr,
            max_steps=mgr.max_epoch,
            **scheduler_kwargs,
        )

    return model, optimizer, scheduler, start_epoch, True


# Backwards-compatible alias for training code
def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    mgr: Any,
    device: DeviceLike,
    load_weights_only: bool = False,
):
    return resume_training_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        mgr=mgr,
        device=device,
        load_weights_only=load_weights_only,
    )
