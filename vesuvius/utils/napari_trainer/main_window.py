import napari
from magicgui import magicgui, widgets
import napari.viewer
import scipy.ndimage

from .inference_widget import inference_widget
from models.configuration.config_manager import ConfigManager
from PIL import Image
import numpy as np
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy
import napari.layers
import json
import yaml
from pathlib import Path
import torch.nn as nn


Image.MAX_IMAGE_PIXELS = None

_config_manager = None

@magicgui(filenames={"label": "select config file", "filter": "*.yaml"},
          auto_call=True)
def filespicker(filenames: Sequence[Path] = str(Path(__file__).parent.parent.parent / 'models' / 'configuration' / 'single_task_config.yaml')) -> Sequence[Path]:
    print("selected config : ", filenames)
    if filenames and _config_manager is not None:
        # Load the first selected file into the config manager
        _config_manager.load_config(filenames[0])
        print(f"Config loaded from {filenames[0]}")
    return filenames

@magicgui(
    call_button='run training',
    patch_size_z={'widget_type': 'SpinBox', 'label': 'Patch Size Z', 'min': 0, 'max': 4096, 'value': 0},
    patch_size_x={'widget_type': 'SpinBox', 'label': 'Patch Size X', 'min': 0, 'max': 4096, 'value': 128},
    patch_size_y={'widget_type': 'SpinBox', 'label': 'Patch Size Y', 'min': 0, 'max': 4096, 'value': 128},
    min_labeled_percentage={'widget_type': 'SpinBox', 'label': 'Min Labeled Percentage', 'min': 0.0, 'max': 100.0, 'step': 1.0, 'value': 10.0},
    max_epochs={'widget_type': 'SpinBox', 'label': 'Max Epochs', 'min': 1, 'max': 1000, 'value': 5},
    loss_function={'widget_type': 'ComboBox', 'choices': ["BCELoss", "BCEWithLogitsLoss", "MSELoss", 
                                                         "L1Loss", "SoftDiceLoss", "BCEDiceLoss", 
                                                         "CrossEntropyLoss", "DiceLoss", "CEDiceLoss"], 'value': "CEDiceLoss"}
)
def run_training(patch_size_z: int = 128, patch_size_x: int = 128, patch_size_y: int = 128,
                min_labeled_percentage: float = 10.0,
                max_epochs: int = 5,
                loss_function: str = "BCEDiceLoss"):
    if _config_manager is None:
        print("Error: No configuration loaded. Please load a config file first.")
        return
    
    print("Starting training process...")
    print("Using images and labels from current viewer")
    
    # Set data format before updating other configurations
    _config_manager.data_format = 'napari'
    
    # Handle 2D case when patch_size_z is 0
    if patch_size_z == 0:
        new_patch_size = [patch_size_y, patch_size_x]  # 2D: [height, width]
    else:
        new_patch_size = [patch_size_z, patch_size_y, patch_size_x]  # 3D: [depth, height, width]
    
    min_labeled_ratio = min_labeled_percentage / 100.0
    
    # Update configuration with new parameters
    _config_manager.update_config(
        patch_size=new_patch_size,
        min_labeled_ratio=min_labeled_ratio,
        max_epochs=max_epochs
    )
    
    # Now that data_format is set to 'napari', the dataset will handle target detection
    # The NapariDataset will extract images from the viewer and configure targets automatically
    
    # Check if viewer has valid image-label pairs
    viewer = napari.current_viewer()
    if viewer is None:
        print("Error: No active napari viewer found.")
        return
    
    # Quick validation of layer naming
    all_layers = list(viewer.layers)
    image_layers = [l for l in all_layers if isinstance(l, napari.layers.Image)]
    label_layers = [l for l in all_layers if isinstance(l, napari.layers.Labels)]
    
    if not image_layers:
        print("Error: No image layers found in the viewer.")
        return
    
    if not label_layers:
        print("Error: No label layers found in the viewer.")
        return
    
    # Check for proper naming convention
    valid_pairs = False
    for img_layer in image_layers:
        for lbl_layer in label_layers:
            if lbl_layer.name.startswith(f"{img_layer.name}_") and not lbl_layer.name.endswith("_mask"):
                valid_pairs = True
                break
        if valid_pairs:
            break
    
    if not valid_pairs:
        print("Error: No valid image-label pairs found.")
        print("Label layers should be named as: {image_name}_{target_name}")
        print("For example: if image is 'image1', label should be 'image1_ink'")
        return
    
    try:
        from models.run.train import BaseTrainer
        
        # Create trainer - it will initialize the dataset which will detect targets
        trainer = BaseTrainer(mgr=_config_manager, verbose=True)
        
        # Now apply the selected loss function to all detected targets
        if hasattr(_config_manager, 'targets') and _config_manager.targets:
            print(f"\nApplying {loss_function} to all detected targets...")
            for target_name in _config_manager.targets:
                _config_manager.targets[target_name]['loss_fn'] = loss_function
                print(f"  - Set loss function for target '{target_name}' to {loss_function}")
        
        print("\nStarting training...")
        trainer.train()
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    viewer = napari.Viewer()
    global _config_manager
    _config_manager = ConfigManager(verbose=True)
    # Use an absolute path based on the location of this script
    default_config_path = Path(__file__).parent.parent.parent / 'models' / 'configuration' / 'single_task_config.yaml'
    _config_manager.load_config(default_config_path)
    print(f"Default config loaded from {default_config_path}")

    file_picker_widget = filespicker
    viewer.window.add_dock_widget(file_picker_widget, area='right', name="config file")
    viewer.window.add_dock_widget(run_training, area='right', name="training")
    viewer.window.add_dock_widget(inference_widget, area='right', name="inference")

    napari.run()

if __name__ == "__main__":
    main()
