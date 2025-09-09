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

# Define mode options
class BlendMode(Enum):
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"

# Define padding mode options
class PaddingMode(Enum):
    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"

@magicgui(
    call_button="Run Inference",
    layout="vertical",
    model_path={"label": "Model Checkpoint", "widget_type": "FileEdit", "filter": "PyTorch Checkpoint (*.pth)"},
    layer={"label": "Layer for inference", "choices": get_data_layer_choices},
    activation_type={"label": "Activation Type", "choices": [ActivationType.NONE, ActivationType.SOFTMAX, ActivationType.SIGMOID, ActivationType.ARGMAX]},
    batch_size={"label": "Batch Size", "widget_type": "SpinBox", "min": 1, "max": 256, "step": 1, "value": 2},
    overlap={"label": "Overlap", "widget_type": "FloatSpinBox", "min": 0.0, "max": 0.75, "step": 0.05, "value": 0.25},
    mode={"label": "Blend Mode", "choices": [BlendMode.CONSTANT, BlendMode.GAUSSIAN]},
    sigma_scale={"label": "Sigma Scale (gaussian)", "widget_type": "FloatSpinBox", "min": 0.01, "max": 1.0, "step": 0.025, "value": 0.125},
    padding_mode={"label": "Padding Mode", "choices": [PaddingMode.CONSTANT, PaddingMode.REFLECT, PaddingMode.REPLICATE, PaddingMode.CIRCULAR]},
)
def inference_widget(
    viewer: napari.Viewer,
    model_path: str,
    layer: napari.layers.Layer,
    activation_type: ActivationType = ActivationType.NONE,
    batch_size: int = 2,
    overlap: float = 0.25,
    mode: BlendMode = BlendMode.CONSTANT,
    sigma_scale: float = 0.125,
    padding_mode: PaddingMode = PaddingMode.CONSTANT,
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
            sw_batch_size=batch_size,
            mode=mode.value,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode.value,
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
