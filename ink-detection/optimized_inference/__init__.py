"""
Optimized Inference Package for Ink Detection

A production-ready inference system for TimeSformer-based ink detection
"""

from .inference_timesformer import (
    RegressionPLModel,
    InferenceConfig,
    run_inference,
    preprocess_layers,
)

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__description__ = "Production inference system for ink detection"

__all__ = [
    "RegressionPLModel",
    "InferenceConfig", 
    "run_inference",
    "preprocess_layers",
    "load_model",
]