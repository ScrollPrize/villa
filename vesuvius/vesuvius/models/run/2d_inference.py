import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
from PIL import Image

from monai.inferers import SlidingWindowInferer
from torch.utils.data import Dataset, DataLoader


from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.utilities.load_checkpoint import load_checkpoint
from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.normalization import get_normalization


class InferenceDataset(Dataset):
    def __init__(self, image_paths, patch_size, normalization_scheme=None, intensity_properties=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.normalization_scheme = normalization_scheme
        self.intensity_properties = intensity_properties
        # Initialize normalizer using the same system as training
        self.normalizer = None
        if normalization_scheme is not None:
            self.normalizer = get_normalization(normalization_scheme, intensity_properties)
        
    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # Load image based on extension
        if path.suffix.lower() in ['.tif', '.tiff']:
            image = tifffile.imread(path)
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image = np.array(Image.open(path).convert('L'))  # Convert to grayscale
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Handle different image formats based on what model expects
        is_3d_model = len(self.patch_size) == 3
        
        if is_3d_model:
            # Model expects 3D input
            if image.ndim == 2:
                raise ValueError(f"Model expects 3D input but got 2D image. Please provide 3D TIFF files.")
            elif image.ndim == 3:
                # Already 3D, just ensure it matches expected depth
                expected_depth = self.patch_size[0]
                if image.shape[0] != expected_depth:
                    raise ValueError(f"Model expects depth {expected_depth} but image has depth {image.shape[0]}")
        else:
            # Model expects 2D input  
            if image.ndim == 3:
                raise ValueError(f"Model expects 2D input but got 3D image with shape {image.shape}")
            # image.ndim == 2, which is what we want
        
        # Apply normalization using the same system as training
        if self.normalizer is not None:
            image = self.normalizer.run(image)
        else:
            # Default normalization to [0, 1] if no normalizer specified
            if image.max() > 1:
                image = image / 255.0
        
        # Add channel dimension
        if is_3d_model:
            # For 3D: (C, D, H, W)
            image = image[np.newaxis, :, :, :]
        else:
            # For 2D: (C, H, W)
            image = image[np.newaxis, :, :]
        
        # Convert to torch tensor
        image = torch.from_numpy(image)
        
        return {
            'image': image,
            'path': str(path),
            'original_shape': image.shape
        }


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from train.py checkpoint using vesuvius utilities"""
    
    # First load the checkpoint to extract configuration
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    # Build the model using the config
    model = NetworkFromConfig(mgr)
    
    # For inference, we'll load the model weights directly instead of using load_checkpoint
    # to avoid optimizer state issues
    print("Loading model weights from checkpoint...")
    model_state = checkpoint['model']
    model.load_state_dict(model_state)
    
    # Move model to device
    model = model.to(device)
    
    # Set model to eval mode
    model.eval()
    
    return model, mgr


def inference_on_folder(
    model,
    mgr,
    image_folder,
    output_folder,
    overlap_ratio=0.5,
    batch_size=1,
    device='cuda'
):
    """Run inference on a folder of images"""
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_folder).glob(f'*{ext}'))
        image_paths.extend(Path(image_folder).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Get patch size from model config
    if hasattr(mgr, 'train_patch_size'):
        patch_size = mgr.train_patch_size
    else:
        raise ValueError("Could not determine patch size from model configuration")
    
    print(f"Using patch size: {patch_size}")
    
    # Create dataset and dataloader
    dataset = InferenceDataset(
        image_paths, 
        patch_size=patch_size,
        normalization_scheme=getattr(mgr, 'normalization_scheme', 'zscore'),
        intensity_properties=getattr(mgr, 'intensity_properties', None)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Setup sliding window inferer
    roi_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
    overlap = overlap_ratio
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=4,
        overlap=overlap,
        mode='gaussian',
        progress=True
    )
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            images = batch['image'].to(device)
            paths = batch['path']
            
            # Run sliding window inference
            outputs = inferer(images, model)
            
            # Handle multi-task outputs
            if isinstance(outputs, dict):
                # If model has multiple outputs, use the first one or a specific task
                task_names = list(outputs.keys())
                if 'ink' in outputs:
                    outputs = outputs['ink']
                else:
                    outputs = outputs[task_names[0]]
                    print(f"Using output from task: {task_names[0]}")
            
            # Process each output
            for i, (output, path) in enumerate(zip(outputs, paths)):
                # Remove batch dimension if present
                if output.ndim == 4 and len(patch_size) == 3:  # 3D model output: B, C, D, H, W
                    output = output[0]  # Remove batch
                elif output.ndim == 3 and len(patch_size) == 2:  # 2D model output: B, C, H, W
                    output = output[0]  # Remove batch
                
                # Handle channel dimension
                if output.ndim == 4:  # C, D, H, W (3D)
                    output = output[0]  # Take first channel
                elif output.ndim == 3:  # C, H, W (2D)
                    output = output[0]  # Take first channel
                
                # Now output should be either 3D (D, H, W) or 2D (H, W)
                
                # Convert to numpy
                output_np = output.cpu().numpy()
                
                # Apply sigmoid if needed (for binary segmentation)
                if output_np.min() < 0 or output_np.max() > 1:
                    output_np = 1 / (1 + np.exp(-output_np))
                
                # Save as TIFF
                input_path = Path(path)
                output_path = Path(output_folder) / f"{input_path.stem}_prediction.tif"
                tifffile.imwrite(output_path, output_np.astype(np.float32))
                
                # Also save as normalized PNG for visualization
                output_norm = (output_np - output_np.min()) / (output_np.max() - output_np.min() + 1e-8)
                output_png = (output_norm * 255).astype(np.uint8)
                png_path = Path(output_folder) / f"{input_path.stem}_prediction.png"
                Image.fromarray(output_png).save(png_path)


def main():
    parser = argparse.ArgumentParser(description='2D Inference with trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file from train.py)')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing input images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for predictions')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio for sliding window inference')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            print("CUDA not available, using MPS (Apple Silicon)")
            args.device = 'mps'
        else:
            print("Neither CUDA nor MPS available, using CPU")
            args.device = 'cpu'
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, mgr = load_model_from_checkpoint(args.checkpoint, device=args.device)
    
    # Print model info
    if hasattr(mgr, 'targets'):
        print(f"Model has {len(mgr.targets)} output task(s): {list(mgr.targets.keys())}")
    if hasattr(mgr, 'normalization_scheme'):
        print(f"Using normalization scheme: {mgr.normalization_scheme}")
    else:
        print("Using default normalization scheme: zscore")
    
    # Run inference
    print(f"Running inference on images in {args.input_folder}")
    inference_on_folder(
        model=model,
        mgr=mgr,
        image_folder=args.input_folder,
        output_folder=args.output_folder,
        overlap_ratio=args.overlap,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"Inference complete. Results saved to {args.output_folder}")


if __name__ == "__main__":
    main()