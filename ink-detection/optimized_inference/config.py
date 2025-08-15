"""
Configuration and model management for ink detection inference.
"""
import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import random
import torch
from torch.nn import DataParallel

from inference_timesformer import RegressionPLModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for model and inference settings"""
    
    def __init__(self):
        # Model settings
        self.model_path = os.getenv(
            'MODEL_PATH', 
            'outputs/vesuvius/pretraining_all/vesuvius-models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.model_compile = os.getenv('MODEL_COMPILE', 'true').lower() == 'true'
        
        # S3 settings
        self.s3_endpoint = os.getenv('S3_ENDPOINT')
        self.s3_access_key = os.getenv('S3_ACCESS_KEY')
        self.s3_secret_key = os.getenv('S3_SECRET_KEY')
        self.s3_bucket = os.getenv('S3_BUCKET')
        
        # Cache settings
        self.cache_dir = Path(os.getenv('CACHE_DIR', '/tmp/inference_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Known reverse segments (for backwards compatibility)
        self.reverse_segments = {
            '20230701020044', 'verso', '20230901184804', '20230901234823',
            '20230531193658', '20231007101615', '20231005123333', '20231011144857',
            '20230522215721', '20230919113918', '20230625171244', '20231022170900',
            '20231012173610', '20231016151000'
        }

# Global config instance
config = ModelConfig()

def setup_config(seed=42, cudnn_deterministic=True):
    """Set up reproducible configuration"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def load_model(model_path: Optional[str] = None) -> RegressionPLModel:
    """
    Load and initialize the TimeSformer model.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Loaded and initialized model
    """
    try:
        if model_path is None:
            model_path = config.model_path
        
        logger.info(f"Loading model from: {model_path}")
        
        # Try to load with PyTorch Lightning first
        try:
            model = RegressionPLModel.load_from_checkpoint(model_path, strict=False)
            logger.info("Model loaded with PyTorch Lightning")
        except Exception as e:
            logger.warning(f"PyTorch Lightning loading failed: {e}, trying manual loading")
            # Fallback to manual loading
            model = RegressionPLModel(pred_shape=(1, 1))
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Model loaded manually")
        
        # Compile model if enabled and using PyTorch 2.0+
        if config.model_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Setup multi-GPU if available
        if config.num_gpus > 1:
            model = DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {config.num_gpus} GPUs")
        
        # Move to device
        model.to(config.device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {config.device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def warmup_model(model: RegressionPLModel, device: torch.device):
    """
    Warm up the model with a dummy forward pass.
    
    Args:
        model: Loaded model
        device: Target device
    """
    try:
        logger.info("Warming up model...")
        
        # Create dummy input matching expected shape (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 26, 64, 64, device=device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        logger.info("Model warmup completed")
        
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        raise

# Global model instance (will be initialized on startup)
_global_model = None

def get_global_model() -> RegressionPLModel:
    """Get the global model instance"""
    global _global_model
    if _global_model is None:
        raise RuntimeError("Model not initialized. Call initialize_global_model() first.")
    return _global_model

def initialize_global_model():
    """Initialize the global model instance"""
    global _global_model
    try:
        setup_config()
        _global_model = load_model()
        warmup_model(_global_model, config.device)
        logger.info("Global model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global model: {e}")
        raise

def cleanup_global_model():
    """Clean up the global model instance"""
    global _global_model
    if _global_model is not None:
        try:
            if isinstance(_global_model, DataParallel):
                _global_model = _global_model.module
            _global_model.cpu()
            del _global_model
            _global_model = None
            torch.cuda.empty_cache()
            logger.info("Global model cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model: {e}")