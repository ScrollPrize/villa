import os
import torch
import time
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

# Define the module's public interface
__all__ = ['load_model', 'initialize_network', 'load_model_from_hf', 'load_model_for_inference']

def initialize_network(architecture_class_name: str,
                      arch_init_kwargs: dict,
                      arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                      num_input_channels: int,
                      num_output_channels: int,
                      enable_deep_supervision: bool = False) -> nn.Module:
    """
    Initialize a network architecture based on its class name and parameters.
    This is our own implementation that doesn't depend on the nnUNetTrainer class.
    
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
    # Import required modules for kwargs if needed
    for i in arch_init_kwargs_req_import:
        if i != "":
            exec(f"import {i}")
            
    # Import the network architecture class
    network_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "network_architecture"),
        architecture_class_name,
        current_module="nnunetv2.network_architecture"
    )
    
    if network_class is None:
        raise RuntimeError(f"Network architecture class {architecture_class_name} not found in nnunetv2.network_architecture.")
    
    # Initialize the network with the appropriate parameters
    network = network_class(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **arch_init_kwargs
    )
    
    return network

def load_model(model_folder: str, fold: Union[int, str] = 0, checkpoint_name: str = 'checkpoint_final.pth', 
            device='cuda', custom_plans_json=None, custom_dataset_json=None, verbose: bool = False, rank: int = 0):

    # Only print from rank 0 by default
    if rank == 0:
        print(f"Starting load_model for {model_folder}, fold={fold}, device={device}")
    import time
    start_time = time.time()
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
        network: The loaded model
        parameters: The model parameters
    """
    # Load dataset and plans - check if we're in a fold directory
    model_path = model_folder
    if os.path.basename(model_folder).startswith('fold_'):
        # We're inside a fold directory, move up one level
        model_path = os.path.dirname(model_folder)
    
    # Check for dataset.json and plans.json
    dataset_json_path = join(model_path, 'dataset.json')
    plans_json_path = join(model_path, 'plans.json')
    
    if custom_dataset_json is None and not os.path.exists(dataset_json_path):
        error_msg = f"ERROR: dataset.json not found at: {dataset_json_path}\n"
        error_msg += f"\nThis file is required for nnUNet model loading.\n"
        if os.path.isdir(model_path):
            error_msg += f"Contents of model directory ({model_path}):\n"
            error_msg += f"  {', '.join(os.listdir(model_path))}\n"
        raise FileNotFoundError(error_msg)
        
    if custom_plans_json is None and not os.path.exists(plans_json_path):
        error_msg = f"ERROR: plans.json not found at: {plans_json_path}\n"
        error_msg += f"\nThis file is required for nnUNet model loading.\n"
        if os.path.isdir(model_path):
            error_msg += f"Contents of model directory ({model_path}):\n"
            error_msg += f"  {', '.join(os.listdir(model_path))}\n"
        raise FileNotFoundError(error_msg)
    
    # Load the JSON files
    try:
        dataset_json = custom_dataset_json if custom_dataset_json is not None else load_json(dataset_json_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset.json: {str(e)}")
        
    try:
        plans = custom_plans_json if custom_plans_json is not None else load_json(plans_json_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load plans.json: {str(e)}")
        
    try:
        plans_manager = PlansManager(plans)
    except Exception as e:
        raise RuntimeError(f"Failed to create PlansManager: {str(e)}")
    
    # Load checkpoint - handle if we're already in a fold directory or not
    if os.path.basename(model_folder).startswith('fold_'):
        checkpoint_file = join(model_folder, checkpoint_name)
    else:
        checkpoint_file = join(model_folder, f'fold_{fold}', checkpoint_name)

    # --- fallback: if final checkpoint not found, try 'checkpoint_best.pth' ---
    if not os.path.exists(checkpoint_file) and checkpoint_name == 'checkpoint_final.pth':
        alt_checkpoint_name = 'checkpoint_best.pth'
        if os.path.basename(model_folder).startswith('fold_'):
            alt_checkpoint_file = join(model_folder, alt_checkpoint_name)
        else:
            alt_checkpoint_file = join(model_folder, f'fold_{fold}', alt_checkpoint_name)

        if os.path.exists(alt_checkpoint_file):
            if rank == 0:
                print(f"WARNING: '{checkpoint_name}' not found; using '{alt_checkpoint_name}' instead.")
            checkpoint_file = alt_checkpoint_file
            checkpoint_name = alt_checkpoint_name
    # ---------------------------------------------------------------------

    # Check if the (possibly updated) checkpoint file exists
    if not os.path.exists(checkpoint_file):
        # List available folds and checkpoints to help the user
        available_folds = []
        if os.path.isdir(model_folder):
            for item in os.listdir(model_folder):
                if item.startswith('fold_') and os.path.isdir(join(model_folder, item)):
                    available_folds.append(item)
        error_msg = f"ERROR: Checkpoint file not found: {checkpoint_file}\n"
        if available_folds:
            error_msg += "\nAvailable folds in this model folder:\n"
            for fold_dir in available_folds:
                fold_path = join(model_folder, fold_dir)
                checkpoints = [f for f in os.listdir(fold_path) if f.endswith('.pth')]
                if checkpoints:
                    error_msg += f"  - {fold_dir}: {', '.join(checkpoints)}\n"
                else:
                    error_msg += f"  - {fold_dir}: No checkpoint files found\n"
        else:
            error_msg += f"\nThe model folder does not contain any 'fold_X' subdirectories.\n"
            if os.path.isdir(model_folder):
                error_msg += f"Contents of {model_folder}:\n"
                error_msg += f"  {', '.join(os.listdir(model_folder))}\n"
            else:
                error_msg += f"The model folder does not exist or is not accessible: {model_folder}\n"
        error_msg += "\nPlease check:\n"
        error_msg += "1. The model_folder path is correct\n"
        error_msg += "2. The fold number is correct\n"
        error_msg += "3. The checkpoint_name is correct\n"
        raise FileNotFoundError(error_msg)
        
    if rank == 0:  # Only print from rank 0
        print(f"Loading checkpoint: {checkpoint_file}")
    try:
        if rank == 0:
            print(f"Attempting to load checkpoint from: {checkpoint_file}")
        try:
            # Try with weights_only=False first (required for PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
            if rank == 0:
                print("Loaded checkpoint with weights_only=False")
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            if rank == 0:
                print("Falling back to loading without weights_only parameter")
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint (rank {rank}): {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    
    # Get configuration
    configuration_manager = plans_manager.get_configuration(configuration_name)
    
    # Determine input channels and number of output classes
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    # Find the trainer class
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                               trainer_name, 'nnunetv2.training.nnUNetTrainer')
    
    # Build the network architecture (without deep supervision for inference)
    if trainer_class is None:
        if verbose or rank == 0:
            print(f"Unable to locate trainer class {trainer_name}. Using direct network initialization.")
        
        # try to build the network with basically a copy of build_network_architecture, without relying on trainer class
        network = initialize_network(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False
            )
    else:
        try:
            # Use the trainer class's build_network_architecture method
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False
            )
        except Exception as e:
            if verbose or rank == 0:
                print(f"Error using trainer's build_network_architecture: {e}")
                print("Falling back to direct network initialization.")
            
            # Fallback to our custom initialization
            network = initialize_network(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False
            )
    
    # Move to the specified device
    device = torch.device(device)
    network = network.to(device)
    
    # Load the state dict
    network_state_dict = checkpoint['network_weights']
    if not isinstance(network, OptimizedModule):
        network.load_state_dict(network_state_dict)
    else:
        network._orig_mod.load_state_dict(network_state_dict)
    
    # Set to evaluation mode
    network.eval()
    
    # Compile by default unless explicitly disabled
    should_compile = True
    if 'nnUNet_compile' in os.environ.keys():
        should_compile = os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
    
    if should_compile and not isinstance(network, OptimizedModule):
        if rank == 0:
            print('Using torch.compile for potential performance improvement')
        try:
            network = torch.compile(network)
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not compile model: {e}")
                print("Continuing with uncompiled model")
    
    # Get allowed mirroring axes from checkpoint if available
    inference_allowed_mirroring_axes = None
    if 'inference_allowed_mirroring_axes' in checkpoint.keys():
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
    
    # Return useful information for inference
    model_info = {
        'network': network,
        'checkpoint': checkpoint,
        'plans_manager': plans_manager,
        'configuration_manager': configuration_manager,
        'dataset_json': dataset_json,
        'label_manager': label_manager,
        'trainer_name': trainer_name,
        'num_input_channels': num_input_channels,
        'num_seg_heads': label_manager.num_segmentation_heads,
        'patch_size': configuration_manager.patch_size,
        'allowed_mirroring_axes': inference_allowed_mirroring_axes,
        'verbose': verbose
    }
    
    return model_info


def load_model_from_hf(repo_id: str, fold: Union[int, str] = 0, checkpoint_name: str = 'checkpoint_final.pth', 
                     device='cuda', verbose: bool = False, rank: int = 0,
                     token: str = None):
    """
    Load a trained nnUNet model from a Hugging Face model repository.
    
    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/model-name')
        fold: Which fold to load (default: 0, can also be 'all')
        checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
        device: Device to load the model on ('cuda' or 'cpu')
        verbose: Enable detailed output messages during loading (default: False)
        rank: Distributed rank of the process (default: 0)
        token: Optional Hugging Face token for private repositories
        
    Returns:
        model_info: Dictionary with model information and parameters
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "The huggingface_hub package is required to load models from Hugging Face. "
            "Please install it with: pip install huggingface_hub"
        )

    if rank == 0:
        print(f"Loading model from Hugging Face repository: {repo_id}")
        
    # Create a temporary directory to download the model
    with tempfile.TemporaryDirectory() as temp_dir:
        if rank == 0 and verbose:
            print(f"Downloading model to temporary directory: {temp_dir}")

        # Download the model repository
        try:
            download_path = snapshot_download(
                repo_id=repo_id,
                local_dir=temp_dir,
                token=token
            )

            if rank == 0 and verbose:
                print(f"Model downloaded to: {download_path}")

            # Check if this is a flat repository structure (no fold directories)
            has_checkpoint = os.path.exists(os.path.join(download_path, checkpoint_name))
            has_plans = os.path.exists(os.path.join(download_path, 'plans.json'))
            has_dataset = os.path.exists(os.path.join(download_path, 'dataset.json'))

            if has_checkpoint and has_plans and has_dataset:
                if rank == 0 and verbose:
                    print("Detected flat repository structure with model files in root directory")
                # Create a temporary fold directory to match the expected structure
                fold_dir = os.path.join(download_path, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)

                # Copy checkpoint to the fold directory
                shutil.copy(
                    os.path.join(download_path, checkpoint_name),
                    os.path.join(fold_dir, checkpoint_name)
                )

                if rank == 0 and verbose:
                    print(f"Copied {checkpoint_name} to {fold_dir}")
                
            # Load the model from the downloaded directory
            model_info = load_model(
                model_folder=download_path,
                fold=fold,
                checkpoint_name=checkpoint_name,
                device=device,
                verbose=verbose,
                rank=rank
            )

            return model_info

        except Exception as e:
            raise RuntimeError(f"Failed to download or load model from Hugging Face: {str(e)}")



def load_model_for_inference(
    model_folder: str = None,
    hf_model_path: str = None,
    hf_token: str = None,
    fold: Union[int, str] = 0,
    checkpoint_name: str = 'checkpoint_final.pth',
    patch_size: Optional[Tuple[int, int, int]] = None,
    device_str: str = 'cuda',
    use_mirroring: bool = True,
    verbose: bool = False,
    rank: int = 0
) -> Dict[str, Any]:
    """
    Load a trained nnUNet model for inference.
    
    Args:
        model_folder: Path to the nnUNet model folder
        hf_model_path: Path to the Hugging Face model
        hf_token: Hugging Face token for private repositories
        fold: Which fold to load (default: 0, can also be 'all')
        checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
        patch_size: Optional override for the patch size
        device_str: Device to run inference on ('cuda' or 'cpu')
        use_mirroring: Enable test time augmentation via mirroring (default: True)
        verbose: Enable detailed output messages during loading
        rank: Process rank for distributed processing (default: 0)
        
    Returns:
        model_info: Dictionary with model information and parameters
    """
    try:
        # Set verbose to False for non-rank-0 processes to avoid duplicate messages
        local_verbose = verbose and rank == 0
        
        # Determine whether to load from local folder or Hugging Face
        if hf_model_path is not None:
            # Load from Hugging Face
            if rank == 0:
                print(f"Loading model from Hugging Face: {hf_model_path}, fold {fold}")
                
            if local_verbose:
                print(f"Test time augmentation (mirroring): {'enabled' if use_mirroring else 'disabled'}")
            
            model_info = load_model_from_hf(
                repo_id=hf_model_path,
                fold=fold,
                checkpoint_name=checkpoint_name,
                device=device_str,
                verbose=local_verbose,
                rank=rank,
                token=hf_token
            )
        else:
            # Load from local folder
            if rank == 0:
                print(f"Loading model from {model_folder}, fold {fold}")

            if local_verbose:
                print(f"Test time augmentation (mirroring): {'enabled' if use_mirroring else 'disabled'}")

            model_info = load_model(
                model_folder=model_folder,
                fold=fold,
                checkpoint_name=checkpoint_name,
                device=device_str,
                verbose=local_verbose,
                rank=rank
            )

        # Use the model's patch size if none was specified
        if patch_size is None:
            # Always convert to tuple for consistency (model_info often has it as a list)
            patch_size = tuple(model_info['patch_size'])
            if verbose and rank == 0:
                print(f"Using model's patch size: {patch_size}")
            model_info['patch_size'] = patch_size
        else:
            # Override the patch size in model_info
            model_info['patch_size'] = patch_size

        # Report multiclass vs binary if rank 0
        num_classes = model_info.get('num_seg_heads', 1)  # Default to 1 if not specified
        if rank == 0:
            if num_classes > 2:
                print(f"Detected multiclass model with {num_classes} classes from model_info")
            elif num_classes == 2:
                print(f"Detected binary segmentation model from model_info")
            elif num_classes == 1:
                print(f"Detected single-channel model from model_info")
        
        return model_info
    except Exception as e:
        print(f"Error loading model (rank {rank}): {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load a trained nnUNet model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the model folder')
    parser.add_argument('--fold', type=str, default='0', help='Fold to load (default: 0, can also be "all")')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', 
                      help='Checkpoint file name (default: checkpoint_final.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on (default: cuda)')
    parser.add_argument('--custom_plans', type=str, default=None, 
                      help='Path to custom plans.json file (optional)')
    parser.add_argument('--custom_dataset', type=str, default=None, 
                      help='Path to custom dataset.json file (optional)')
    parser.add_argument('--test_inference', action='store_true', 
                      help='Run a test inference with random input')
    parser.add_argument('--disable_tta', action='store_true',
                      help='Disable test time augmentation (mirroring) for faster inference')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable detailed output messages during loading and inference')
    
    args = parser.parse_args()
    
    # Load custom JSON files if specified
    custom_plans_json = load_json(args.custom_plans) if args.custom_plans else None
    custom_dataset_json = load_json(args.custom_dataset) if args.custom_dataset else None
    
    # Load the model
    model_info = load_model(
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        device=args.device,
        custom_plans_json=custom_plans_json,
        custom_dataset_json=custom_dataset_json,
        verbose=args.verbose
    )
    
    # Print model information
    network = model_info['network']
    checkpoint = model_info['checkpoint']
    print("\nModel loaded successfully!")
    print(f"Trainer: {model_info['trainer_name']}")
    print(f"Configuration: {checkpoint['init_args']['configuration']}")
    print(f"Model type: {type(network).__name__}")
    print(f"Model is on device: {next(network.parameters()).device}")
    print(f"Input channels: {model_info['num_input_channels']}")
    print(f"Output segmentation heads: {model_info['num_seg_heads']}")
    print(f"Expected patch size: {model_info['patch_size']}")
    
    # Show TTA status if verbose
    if args.verbose:
        use_mirroring = model_info.get('use_mirroring', True)
        mirroring_axes = model_info.get('allowed_mirroring_axes', None)
        if use_mirroring and mirroring_axes is not None:
            print(f"Test time augmentation: Enabled with mirroring axes {mirroring_axes}")
        else:
            print(f"Test time augmentation: {'Disabled by user' if not use_mirroring else 'Not available (no mirroring axes in checkpoint)'}")
    
    # Run a test inference if requested
    if args.test_inference:
        print("\nRunning test inference with random input...")
        patch_size = model_info['patch_size']
        input_channels = model_info['num_input_channels']
        
        # Create a random input batch with the right dimensions
        if len(patch_size) == 2:  # 2D model
            dummy_input = torch.randn(1, input_channels, *patch_size, device=args.device)
        else:  # 3D model
            dummy_input = torch.randn(1, input_channels, *patch_size, device=args.device)
        
        # Run inference
        with torch.no_grad():
            try:
                output = network(dummy_input)
                print(f"Test inference successful!")
                print(f"Input shape: {dummy_input.shape}")
                print(f"Output shape: {output.shape}")
                del dummy_input, output
                torch.cuda.empty_cache()  # Clean up GPU memory
            except Exception as e:
                print(f"Test inference failed with error: {e}")
