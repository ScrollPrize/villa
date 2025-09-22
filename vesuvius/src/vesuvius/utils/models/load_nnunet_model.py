import os
import torch
from typing import Union, List, Tuple, Dict, Any, Optional
from batchgenerators.utilities.file_and_folder_operations import load_json, join
import tempfile
import shutil

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import nnunetv2
import torch.nn as nn
from torch._dynamo import OptimizedModule

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

__all__ = ['load_model', 'initialize_network', 'load_model_for_inference', 'load_model_from_checkpoint']

def initialize_network(architecture_class_name: str,
                      arch_init_kwargs: dict,
                      arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                      num_input_channels: int,
                      num_output_channels: int,
                      enable_deep_supervision: bool = False) -> nn.Module:
    """
    Args:
        architecture_class_name: Class name of the network architecture
        arch_init_kwargs: Keyword arguments for initializing the architecture
        arch_init_kwargs_req_import: Names of modules that need to be imported for kwargs
        num_input_channels: Number of input channels
        num_output_channels: Number of output channels (segmentation classes)
        enable_deep_supervision: Whether to enable deep supervision
        
    Returns:
        The initialized network
    """

    for i in arch_init_kwargs_req_import:
        if i != "":
            exec(f"import {i}")
            
    network_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "network_architecture"),
        architecture_class_name,
        current_module="nnunetv2.network_architecture"
    )
    
    if network_class is None:
        raise RuntimeError(f"Network architecture class {architecture_class_name} not found in nnunetv2.network_architecture.")
    
    network = network_class(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **arch_init_kwargs
    )
    
    return network

def load_model(model_folder: str, fold: Union[int, str] = 0, checkpoint_name: str = 'checkpoint_final.pth', 
            device='cuda', custom_plans_json=None, custom_dataset_json=None, verbose: bool = False, rank: int = 0):
    """
    Load a trained nnUNet model from a model folder.
    
    Args:
        model_folder: Path to the model folder containing plans.json, dataset.json and fold_X folders
        fold: Which fold to load (default: 0, can also be 'all')
        checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
        device: Device to load the model on ('cuda' or 'cpu')
        custom_plans_json: Optional custom plans.json to use instead of the one in model_folder
        custom_dataset_json: Optional custom dataset.json to use instead of the one in model_folder
        verbose: Enable detailed output messages during loading (default: False)
        rank: Distributed rank of the process (default: 0, used to suppress output from non-rank-0 processes)
        
    Returns:
        model_info: Dictionary with model information and parameters
    """
    
    should_print = rank == 0
    if should_print:
        print(f"Starting load_model for {model_folder}, fold={fold}, device={device}")
    model_path = model_folder
    if os.path.basename(model_folder).startswith('fold_'):
        # We're inside a fold directory, move up one level
        model_path = os.path.dirname(model_folder)
    
    # Check for dataset.json and plans.json
    dataset_json_path = join(model_path, 'dataset.json')
    plans_json_path = join(model_path, 'plans.json')
    
    if custom_dataset_json is None and not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"dataset.json not found at: {dataset_json_path}")
        
    if custom_plans_json is None and not os.path.exists(plans_json_path):
        raise FileNotFoundError(f"plans.json not found at: {plans_json_path}")
    

    dataset_json = custom_dataset_json if custom_dataset_json is not None else load_json(dataset_json_path)
    plans = custom_plans_json if custom_plans_json is not None else load_json(plans_json_path)
    plans_manager = PlansManager(plans)
    
    if os.path.basename(model_folder).startswith('fold_'):
        checkpoint_file = join(model_folder, checkpoint_name)
    else:
        checkpoint_file = join(model_folder, f'fold_{fold}', checkpoint_name)

    if not os.path.exists(checkpoint_file) and checkpoint_name == 'checkpoint_final.pth':
        alt_checkpoint_name = 'checkpoint_best.pth'
        if os.path.basename(model_folder).startswith('fold_'):
            checkpoint_file = join(model_folder, alt_checkpoint_name)
        else:
            checkpoint_file = join(model_folder, f'fold_{fold}', alt_checkpoint_name)

        if os.path.exists(checkpoint_file):
            if should_print:
                print(f"WARNING: '{checkpoint_name}' not found; using '{alt_checkpoint_name}' instead.")
            checkpoint_name = alt_checkpoint_name

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    if should_print:
        print(f"Loading checkpoint: {checkpoint_file}")
    
    try:
        # Try with weights_only=False first (required for PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    
    # Get configuration
    configuration_manager = plans_manager.get_configuration(configuration_name)
    
    # Determine input channels and number of output classes
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    # Build the network architecture (without deep supervision for inference)
    # Try trainer class first, fallback to direct initialization
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                               trainer_name, 'nnunetv2.training.nnUNetTrainer')
    
    network = None
    if trainer_class is not None:
        try:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False
            )
        except Exception as e:
            if verbose and should_print:
                print(f"Error using trainer's build_network_architecture: {e}, falling back to direct initialization.")
    
    # Fallback to direct network initialization if trainer approach failed
    if network is None:
        if verbose and should_print:
            print(f"Using direct network initialization (trainer class: {trainer_name}).")
        network = initialize_network(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False
        )
    
    device = torch.device(device)
    network = network.to(device)
    
    network_state_dict = checkpoint['network_weights']
    if not isinstance(network, OptimizedModule):
        network.load_state_dict(network_state_dict)
    else:
        network._orig_mod.load_state_dict(network_state_dict)
    
    network.eval()
    
    # Compile by default unless explicitly disabled
    should_compile = os.environ.get('nnUNet_compile', 'true').lower() in ('true', '1', 't')
    if should_compile and not isinstance(network, OptimizedModule):
        if should_print:
            print('Using torch.compile for potential performance improvement')
        try:
            network = torch.compile(network)
        except Exception as e:
            if should_print:
                print(f"Warning: Could not compile model: {e}")
    
    model_info = {
        'network': network,
        'plans_manager': plans_manager,
        'configuration_manager': configuration_manager,
        'dataset_json': dataset_json,
        'label_manager': label_manager,
        'trainer_name': trainer_name,
        'num_input_channels': num_input_channels,
        'num_seg_heads': label_manager.num_segmentation_heads,
        'patch_size': configuration_manager.patch_size,
        'allowed_mirroring_axes': checkpoint.get('inference_allowed_mirroring_axes'),
    }
    
    return model_info


def load_model_for_inference(
    model_folder: str = None,
    hf_model_path: str = None,
    hf_token: str = None,
    fold: Union[int, str] = 0,
    checkpoint_name: str = 'checkpoint_final.pth',
    patch_size: Optional[Tuple[int, int, int]] = None,
    device_str: str = 'cuda',
    verbose: bool = False,
    rank: int = 0
) -> Dict[str, Any]:
    """
    Load a trained nnUNet model for inference from local folder or Hugging Face.
    
    Args:
        model_folder: Path to the nnUNet model folder (for local loading)
        hf_model_path: Hugging Face repository ID (e.g., 'username/model-name') for HF loading
        hf_token: Hugging Face token for private repositories
        fold: Which fold to load (default: 0)
        checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
        patch_size: Optional override for the patch size
        device_str: Device to run inference on ('cuda' or 'cpu')
        verbose: Enable detailed output messages during loading
        rank: Process rank for distributed processing (default: 0)
        
    Returns:
        model_info: Dictionary with model information and parameters
    """
    should_print = rank == 0
    local_verbose = verbose and should_print
    
    # Load from Hugging Face or local folder
    if hf_model_path is not None:
        if not HF_AVAILABLE:
            raise ImportError(
                "The huggingface_hub package is required to load models from Hugging Face. "
                "Please install it with: pip install huggingface_hub"
            )
        
        if should_print:
            print(f"Loading model from Hugging Face: {hf_model_path}, fold {fold}")
        
        # Download from Hugging Face and load
        # Create persistent temp directory for the download
        temp_dir = tempfile.mkdtemp(prefix='vesuvius_hf_model_')
        
        try:
            download_path = snapshot_download(
                repo_id=hf_model_path,
                local_dir=temp_dir,
                token=hf_token
            )

            # Check for train.py checkpoint files (Model_epoch*.pth pattern)
            train_py_checkpoints = [f for f in os.listdir(download_path) 
                                   if f.startswith('Model_epoch') and f.endswith('.pth')]
            
            if train_py_checkpoints:
                # This is a train.py model checkpoint
                if should_print:
                    print(f"Detected train.py checkpoint: {train_py_checkpoints[0]}")
                
                # Use the first train.py checkpoint found
                checkpoint_path = os.path.join(download_path, train_py_checkpoints[0])
                
                # Load using the dedicated function
                model, mgr = load_model_from_checkpoint(checkpoint_path, device=device_str)
                
                # Build model info dict compatible with inference
                model_info = {
                    'network': model,
                    'patch_size': mgr.train_patch_size if hasattr(mgr, 'train_patch_size') else (192, 192, 192),
                    'num_seg_heads': len(mgr.targets) if mgr.targets else 1,
                    'normalization_scheme': mgr.normalization_scheme if hasattr(mgr, 'normalization_scheme') else None,
                    'intensity_properties': mgr.intensity_properties if hasattr(mgr, 'intensity_properties') else None,
                    'targets': mgr.targets if hasattr(mgr, 'targets') and mgr.targets else None,
                    'temp_dir': temp_dir  # Keep reference to temp dir for cleanup
                }
                
                return model_info
            
            # Check if this is a flat repository structure (no fold directories)
            has_checkpoint = os.path.exists(os.path.join(download_path, checkpoint_name))
            has_plans = os.path.exists(os.path.join(download_path, 'plans.json'))
            has_dataset = os.path.exists(os.path.join(download_path, 'dataset.json'))

            if has_checkpoint and has_plans and has_dataset:
                # Create a temporary fold directory to match the expected structure
                fold_dir = os.path.join(download_path, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(download_path, checkpoint_name),
                    os.path.join(fold_dir, checkpoint_name)
                )
                
            model_info = load_model(
                model_folder=download_path,
                fold=fold,
                checkpoint_name=checkpoint_name,
                device=device_str,
                verbose=local_verbose,
                rank=rank
            )
            model_info['temp_dir'] = temp_dir  # Keep reference for cleanup
            return model_info
            
        except Exception as e:
            # Clean up temp dir on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    else:
        if should_print:
            print(f"Loading model from {model_folder}, fold {fold}")

        # Pre-validate the provided local path and provide a clearer error
        if not os.path.exists(model_folder):
            raise FileNotFoundError(
                f"Model checkpoint not found: path does not exist -> {model_folder}. "
                f"Provide a valid nnU-Net export directory (with dataset.json and plans.json) or a .pth checkpoint file."
            )

        # If a folder was provided, check whether it looks like an nnU-Net export folder
        if os.path.isdir(model_folder):
            has_dataset = os.path.exists(os.path.join(model_folder, 'dataset.json'))
            has_plans = os.path.exists(os.path.join(model_folder, 'plans.json'))
            has_pth_top = any(
                fname.endswith('.pth') for fname in os.listdir(model_folder)
                if os.path.isfile(os.path.join(model_folder, fname))
            )

            # If it is neither an nnU-Net export nor contains any .pth checkpoints at the top level,
            # emit a clearer, actionable error
            if not (has_dataset and has_plans) and not has_pth_top:
                raise FileNotFoundError(
                    "Model checkpoint not found. Expected either: "
                    "(a) an nnU-Net model directory containing 'dataset.json' and 'plans.json', "
                    "or (b) a .pth checkpoint file (pass the file path, not just the folder). "
                    f"Path checked: {model_folder}"
                )
        elif os.path.isfile(model_folder):
            # File provided but not a .pth
            if not model_folder.endswith('.pth'):
                raise FileNotFoundError(
                    f"Model checkpoint not found: expected a .pth file, got '{model_folder}'. "
                    "Provide a .pth checkpoint file or an nnU-Net export directory."
                )

        model_info = load_model(
            model_folder=model_folder,
            fold=fold,
            checkpoint_name=checkpoint_name,
            device=device_str,
            verbose=local_verbose,
            rank=rank
        )

    # Override patch size if specified
    if patch_size is not None:
        model_info['patch_size'] = patch_size
    else:
        # Ensure patch_size is a tuple
        model_info['patch_size'] = tuple(model_info['patch_size'])

    # Report model type
    num_classes = model_info.get('num_seg_heads', 1)
    if should_print:
        if num_classes > 2:
            print(f"Detected multiclass model with {num_classes} classes")
        elif num_classes == 2:
            print(f"Detected binary segmentation model")
        else:
            print(f"Detected single-channel model")
    
    return model_info

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from train.py checkpoint using vesuvius utilities"""
    
    # First load the checkpoint to extract configuration
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Import required modules
    from vesuvius.models.configuration.config_manager import ConfigManager
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    
    # Create a ConfigManager instance
    mgr = ConfigManager(verbose=True)
    
    # Initialize the required dictionaries that _init_attributes expects
    mgr.tr_info = {}
    mgr.tr_configs = {}
    mgr.model_config = {}
    mgr.dataset_config = {}
    
    # Extract critical info BEFORE _init_attributes to ensure proper initialization
    if 'model_config' in checkpoint:
        # Get patch size first as it determines 2D/3D operations
        if 'patch_size' in checkpoint['model_config']:
            patch_size = checkpoint['model_config']['patch_size']
            mgr.tr_configs['patch_size'] = list(patch_size)
            print(f"Pre-loading patch_size from checkpoint: {patch_size}")
    
    # Extract targets early if available, as _init_attributes needs it
    if 'model_config' in checkpoint and 'targets' in checkpoint['model_config']:
        mgr.targets = checkpoint['model_config']['targets']
        mgr.dataset_config['targets'] = checkpoint['model_config']['targets']
    else:
        mgr.targets = {}
    
    # Initialize ConfigManager attributes with defaults first
    mgr._init_attributes()
    
    # Extract critical parameters from checkpoint
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        
        # Critical parameters that must be set before building network
        if 'patch_size' in model_config:
            mgr.train_patch_size = tuple(model_config['patch_size'])
            mgr.tr_configs["patch_size"] = list(model_config['patch_size'])
            print(f"Loaded patch_size from checkpoint: {mgr.train_patch_size}")
            
            # Re-determine dimensionality based on loaded patch size
            from vesuvius.utils.utils import determine_dimensionality
            dim_props = determine_dimensionality(mgr.train_patch_size, mgr.verbose)
            mgr.model_config["conv_op"] = dim_props["conv_op"]
            mgr.model_config["pool_op"] = dim_props["pool_op"]
            mgr.model_config["norm_op"] = dim_props["norm_op"]
            mgr.model_config["dropout_op"] = dim_props["dropout_op"]
            mgr.spacing = dim_props["spacing"]
            mgr.op_dims = dim_props["op_dims"]
        
        if 'targets' in model_config:
            mgr.targets = model_config['targets']
            print(f"Loaded targets from checkpoint: {list(mgr.targets.keys())}")
        
        if 'in_channels' in model_config:
            mgr.in_channels = model_config['in_channels']
            print(f"Loaded in_channels from checkpoint: {mgr.in_channels}")
            
        if 'autoconfigure' in model_config:
            mgr.autoconfigure = model_config['autoconfigure']
            print(f"Loaded autoconfigure from checkpoint: {mgr.autoconfigure}")
        
        # Set the entire model_config on mgr
        mgr.model_config = model_config
        
        # Also set individual attributes for any that might be accessed directly
        for key, value in model_config.items():
            if not hasattr(mgr, key):
                setattr(mgr, key, value)
    
    # Load dataset configuration if available (contains normalization info)
    if 'dataset_config' in checkpoint:
        dataset_config = checkpoint['dataset_config']
        
        if 'normalization_scheme' in dataset_config:
            mgr.normalization_scheme = dataset_config['normalization_scheme']
            print(f"Loaded normalization_scheme from checkpoint: {mgr.normalization_scheme}")
            
        if 'intensity_properties' in dataset_config:
            mgr.intensity_properties = dataset_config['intensity_properties']
            print(f"Loaded intensity_properties from checkpoint")
            
        # Also update dataset_config on mgr
        mgr.dataset_config.update(dataset_config)
    
    # Heuristics for missing/ambiguous config when loading train.py checkpoints
    # 1) Disable deep supervision for inference to avoid forcing separate decoders by default
    try:
        mgr.enable_deep_supervision = False
    except Exception:
        pass

    # 2) If targets are missing, try to infer from head keys in the state dict
    if not getattr(mgr, 'targets', None):
        inferred_targets = {}
        for k, v in model_state.items():
            if isinstance(k, str) and k.startswith('task_heads.') and k.endswith('.weight'):
                try:
                    tgt_name = k.split('.')[1]
                except Exception:
                    continue
                # v expected shape (out_ch, in_ch, ...)
                try:
                    out_ch = int(v.shape[0])
                except Exception:
                    out_ch = 1
                inferred_targets[tgt_name] = {'out_channels': out_ch, 'activation': 'none'}
        if inferred_targets:
            mgr.targets = inferred_targets
            if verbose and should_print:
                print(f"Inferred targets from checkpoint heads: {list(inferred_targets.keys())}")

    # 3) If decoder sharing strategy is ambiguous, infer from keys
    try:
        has_shared = any(str(k).startswith('shared_decoder.') for k in model_state.keys())
        has_task = any(str(k).startswith('task_decoders.') for k in model_state.keys())
        if 'separate_decoders' not in mgr.model_config:
            if has_shared and not has_task:
                mgr.model_config['separate_decoders'] = False
            elif has_task and not has_shared:
                mgr.model_config['separate_decoders'] = True
            # if both present, leave as-is (custom hybrid)
            if verbose and should_print:
                print(f"Decoder strategy inferred: separate_decoders={mgr.model_config.get('separate_decoders', False)}")
    except Exception:
        pass

    # Build the model using the (possibly adjusted) config
    model = NetworkFromConfig(mgr)
    
    # For inference, we'll load the model weights directly instead of using load_checkpoint
    # to avoid optimizer state issues
    print("Loading model weights from checkpoint...")
    # Try to robustly locate the state dict in the checkpoint
    model_state = checkpoint.get('model')
    if not isinstance(model_state, dict):
        # common alternatives
        for cand_key in ['model_state_dict', 'state_dict', 'network', 'network_weights', 'net', 'student']:
            if isinstance(checkpoint.get(cand_key), dict):
                model_state = checkpoint[cand_key]
                if verbose and should_print:
                    print(f"Using state dict from checkpoint['{cand_key}']")
                break
    if not isinstance(model_state, dict):
        raise RuntimeError("Could not locate a valid model state_dict in checkpoint. Expected one of: 'model', 'model_state_dict', 'state_dict', 'network', 'network_weights', 'net', 'student'.")
    
    # Check if this is a compiled model (has _orig_mod prefix)
    is_compiled = any(key.startswith("_orig_mod.") for key in model_state.keys())
    
    if is_compiled:
        print("Detected compiled model checkpoint, handling state dict conversion...")
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in model_state.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model_state = new_state_dict

    # Strip DistributedDataParallel 'module.' prefix if present
    if any(key.startswith('module.') for key in model_state.keys()):
        if verbose and should_print:
            print("Stripping 'module.' prefix from state dict keys")
        model_state = {key.replace('module.', '', 1): val for key, val in model_state.items()}

    # If there is a common leading prefix across most keys (e.g., 'model.'), strip it once
    keys = list(model_state.keys())
    if len(keys) > 0 and '.' in keys[0]:
        first_prefix = keys[0].split('.', 1)[0]
        # check if most keys share this prefix
        share = sum(1 for k in keys if k.startswith(first_prefix + '.')) / len(keys)
        if share > 0.9:
            if verbose and should_print:
                print(f"Stripping common prefix '{first_prefix}.' from state dict keys")
            model_state = {k.split('.', 1)[1]: v for k, v in model_state.items()}

    try:
        model.load_state_dict(model_state)
    except RuntimeError as e:
        # Provide a more helpful error with hints on likely causes
        exp_keys = list(model.state_dict().keys())[:10]
        got_keys = list(model_state.keys())[:10]
        msg = (
            f"Failed to load checkpoint weights into NetworkFromConfig.\n"
            f"This typically indicates a mismatch between the checkpoint architecture and the rebuilt network.\n"
            f"- Expected (example keys): {exp_keys}\n"
            f"- Got (example keys): {got_keys}\n"
            f"Hints: If your checkpoint lacks 'model_config' (training config), we cannot reconstruct the exact\n"
            f"architecture. If this is the case, load from an nnU-Net export directory, or provide a YAML\n"
            f"config that matches the training run and use a loader that consumes it. Original error: {e}"
        )
        raise RuntimeError(msg)
    
    # Move model to device
    model = model.to(device)
    
    # Set model to eval mode
    model.eval()
    
    return model, mgr

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load a trained nnUNet model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the model folder')
    parser.add_argument('--fold', type=str, default='0', help='Fold to load (default: 0)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', 
                      help='Checkpoint file name (default: checkpoint_final.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on (default: cuda)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load the model
    model_info = load_model(
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        device=args.device,
        verbose=args.verbose
    )
    
    # Print basic model information
    network = model_info['network']
    print("Model loaded successfully!")
    print(f"Trainer: {model_info['trainer_name']}")
    print(f"Model type: {type(network).__name__}")
    print(f"Input channels: {model_info['num_input_channels']}")
    print(f"Output segmentation heads: {model_info['num_seg_heads']}")
    print(f"Expected patch size: {model_info['patch_size']}")
