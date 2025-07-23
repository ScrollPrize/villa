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

__all__ = ['load_model', 'initialize_network', 'load_model_for_inference']

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
        with tempfile.TemporaryDirectory() as temp_dir:
            download_path = snapshot_download(
                repo_id=hf_model_path,
                local_dir=temp_dir,
                token=hf_token
            )

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
    else:
        if should_print:
            print(f"Loading model from {model_folder}, fold {fold}")
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
