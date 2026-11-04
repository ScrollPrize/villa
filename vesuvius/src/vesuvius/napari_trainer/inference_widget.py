import napari
from magicgui import magicgui
from pathlib import Path
import os
import torch
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
from enum import Enum

from .inference import run_inference

# Function to get available layers for inference
def get_data_layer_choices(viewer):
    """Get available image layers as choices for inference"""
    return [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]

# Define activation type options
class ActivationType(Enum):
    NONE = "none"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid" 
    ARGMAX = "argmax"

@magicgui(
    call_button="Run Inference",
    layout="vertical",
    model_path={"label": "Model Checkpoint", "widget_type": "FileEdit", "filter": "PyTorch Checkpoint (*.pth)"},
    layer={"label": "Layer for inference", "choices": get_data_layer_choices},
    activation_type={"label": "Activation Type", "choices": [ActivationType.NONE, ActivationType.SOFTMAX, ActivationType.SIGMOID, ActivationType.ARGMAX]},
    batch_size={"label": "Batch Size", "widget_type": "SpinBox", "min": 1, "max": 256, "step": 1, "value": 2},
    overlap={"label": "Overlap", "widget_type": "FloatSpinBox", "min": 0.0, "max": 0.75, "step": 0.05, "value": 0.25},
    sigma_scale={"label": "Gaussian sigma = patch / ", "widget_type": "FloatSpinBox", "min": 1.0, "max": 64.0, "step": 0.5, "value": 8.0},
)
def inference_widget(
    viewer: napari.Viewer,
    model_path: str,
    layer: napari.layers.Layer,
    activation_type: ActivationType = ActivationType.NONE,
    batch_size: int = 2,
    overlap: float = 0.25,
    sigma_scale: float = 8.0,
) -> Optional[napari.layers.Image]:
    if not model_path or not os.path.exists(model_path):
        print("Please select a valid model checkpoint file")
        return None
    
    if layer is None:
        print("Please select a layer for inference")
        return None
    
    try:
        results = run_inference(
            viewer=viewer,
            layer=layer,
            checkpoint_path=model_path,
            overlap=overlap,
            batch_size=batch_size,
            sigma_scale=sigma_scale,
            activation_type=activation_type.value
        )
        
        return None
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"inference error: {str(e)}")
        return None

if __name__ == "__main__":

    viewer = napari.Viewer()
    if len(viewer.layers) == 0:
        sample = np.random.rand(512, 512)
        viewer.add_image(sample, name="Sample")
    widget = inference_widget()
    
    viewer.window.add_dock_widget(
        widget, 
        name="inference"
    )
    napari.run()
