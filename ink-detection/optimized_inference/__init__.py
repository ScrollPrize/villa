"""
Optimized Inference Package for Ink Detection

A production-ready inference system for TimeSformer-based ink detection
on ancient scroll layers with FastAPI REST API.
"""

from .inference_timesformer import (
    RegressionPLModel,
    InferenceConfig,
    run_inference,
    convert_to_human_readable,
    preprocess_layers,
)

from .config import (
    ModelConfig,
    setup_config,
    load_model,
    warmup_model,
    get_global_model,
    initialize_global_model,
    cleanup_global_model,
)

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__description__ = "Production inference system for ink detection"

__all__ = [
    "RegressionPLModel",
    "InferenceConfig", 
    "run_inference",
    "convert_to_human_readable",
    "preprocess_layers",
    "ModelConfig",
    "setup_config",
    "load_model",
    "warmup_model",
    "get_global_model",
    "initialize_global_model",
    "cleanup_global_model",
]