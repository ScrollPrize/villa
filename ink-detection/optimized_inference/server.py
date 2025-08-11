"""
FastAPI server for ink detection inference.
Provides REST API for running TimeSformer inference on scroll layers.
"""
import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from .config import config, get_global_model, initialize_global_model, cleanup_global_model
from .inference_timesformer import run_inference, convert_to_human_readable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Ink Detection Inference API",
    description="TimeSformer-based ink detection inference service for ancient scrolls",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class InferenceRequest(BaseModel):
    """Request model for inference"""
    scroll_name: str = Field(..., description="Name of the scroll")
    segment_name: str = Field(..., description="Name of the segment")
    s3_path_prefix: str = Field(..., description="S3 path prefix to layers (e.g., 'scrolls/scroll1/segment1/')")
    layer_start: int = Field(default=0, ge=0, le=65, description="Starting layer index")
    layer_end: int = Field(default=26, ge=1, le=65, description="Ending layer index")
    fragment_mask_path: Optional[str] = Field(None, description="Optional S3 path to fragment mask")
    output_format: str = Field(default="uint8", description="Output format: 'uint8' or 'float32'")
    
    @validator('layer_end')
    def validate_layer_end(cls, v, values):
        if 'layer_start' in values and v <= values['layer_start']:
            raise ValueError('layer_end must be greater than layer_start')
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        if v not in ['uint8', 'float32']:
            raise ValueError('output_format must be either "uint8" or "float32"')
        return v

class InferenceResponse(BaseModel):
    """Response model for inference"""
    status: str
    scroll_name: str
    segment_name: str
    prediction_shape: List[int]
    processing_time_seconds: float
    output_format: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: float

# S3 client initialization
def get_s3_client():
    """Initialize S3 client with configuration"""
    try:
        if config.s3_access_key and config.s3_secret_key:
            return boto3.client(
                's3',
                endpoint_url=config.s3_endpoint,
                aws_access_key_id=config.s3_access_key,
                aws_secret_access_key=config.s3_secret_key,
                region_name='us-east-1'  # Default region
            )
        else:
            # Use default credentials (IAM role, environment variables, etc.)
            return boto3.client('s3', endpoint_url=config.s3_endpoint)
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise

async def download_layer_from_s3(s3_client, s3_path: str, local_path: Path) -> bool:
    """
    Download a single layer file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        s3_path: S3 object key
        local_path: Local file path to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.download_file(config.s3_bucket, s3_path, str(local_path))
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.warning(f"Layer not found: {s3_path}")
        else:
            logger.error(f"Error downloading {s3_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {s3_path}: {e}")
        return False

async def load_layers_from_s3(s3_path_prefix: str, 
                             layer_start: int, 
                             layer_end: int) -> np.ndarray:
    """
    Load layers from S3 and stack them into a numpy array.
    
    Args:
        s3_path_prefix: S3 prefix path to layers
        layer_start: Starting layer index
        layer_end: Ending layer index
        
    Returns:
        Stacked layers as numpy array (H, W, C)
    """
    try:
        s3_client = get_s3_client()
        layers = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download layers
            for i in range(layer_start, layer_end):
                # Try different file extensions
                for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                    s3_key = f"{s3_path_prefix.rstrip('/')}/{i:02d}{ext}"
                    local_file = temp_path / f"{i:02d}{ext}"
                    
                    if await download_layer_from_s3(s3_client, s3_key, local_file):
                        # Load image
                        image = cv2.imread(str(local_file), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            layers.append(image)
                            logger.debug(f"Loaded layer {i} from {s3_key}")
                            break
                else:
                    logger.warning(f"Could not find layer {i} in any format")
                    # Use zeros as placeholder
                    if layers:
                        h, w = layers[0].shape
                        layers.append(np.zeros((h, w), dtype=np.uint8))
                    else:
                        # Default size if no layers loaded yet
                        layers.append(np.zeros((512, 512), dtype=np.uint8))
        
        if not layers:
            raise ValueError(f"No layers could be loaded from {s3_path_prefix}")
        
        # Ensure all layers have the same dimensions
        target_shape = layers[0].shape
        normalized_layers = []
        
        for layer in layers:
            if layer.shape != target_shape:
                layer = cv2.resize(layer, (target_shape[1], target_shape[0]))
            normalized_layers.append(layer)
        
        # Stack layers
        stacked_layers = np.stack(normalized_layers, axis=2)
        logger.info(f"Loaded {len(layers)} layers with shape {stacked_layers.shape}")
        
        return stacked_layers
        
    except Exception as e:
        logger.error(f"Error loading layers from S3: {e}")
        raise

async def load_mask_from_s3(mask_path: str) -> Optional[np.ndarray]:
    """
    Load fragment mask from S3.
    
    Args:
        mask_path: S3 path to mask file
        
    Returns:
        Mask as numpy array or None if not found
    """
    try:
        s3_client = get_s3_client()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "mask.png"
            
            if await download_layer_from_s3(s3_client, mask_path, temp_file):
                mask = cv2.imread(str(temp_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    logger.info(f"Loaded mask from {mask_path}")
                    return mask
        
        logger.warning(f"Could not load mask from {mask_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading mask from S3: {e}")
        return None

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_global_model()
        model_loaded = model is not None
        device = str(config.device)
    except:
        model_loaded = False
        device = "unknown"
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=device,
        timestamp=time.time()
    )

@app.post("/predict", response_model=InferenceResponse)
async def run_prediction(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Run ink detection inference on scroll layers.
    
    Args:
        request: Inference request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Inference results and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting inference for {request.scroll_name}/{request.segment_name}")
        
        # Get model
        model = get_global_model()
        
        # Check if segment should be reversed
        is_reverse_segment = any(
            segment_id in request.segment_name 
            for segment_id in config.reverse_segments
        )
        
        # Load layers from S3
        layers = await load_layers_from_s3(
            request.s3_path_prefix,
            request.layer_start,
            request.layer_end
        )
        
        # Load mask if provided
        fragment_mask = None
        if request.fragment_mask_path:
            fragment_mask = await load_mask_from_s3(request.fragment_mask_path)
        
        # Run inference
        prediction_mask = run_inference(
            layers=layers,
            model=model,
            device=config.device,
            fragment_mask=fragment_mask,
            is_reverse_segment=is_reverse_segment
        )
        
        # Convert to human-readable format
        output_mask = convert_to_human_readable(prediction_mask, request.output_format)
        
        processing_time = time.time() - start_time
        
        # Save prediction to cache (optional)
        cache_path = config.cache_dir / f"{request.scroll_name}_{request.segment_name}_{int(start_time)}.npy"
        background_tasks.add_task(np.save, cache_path, output_mask)
        
        logger.info(f"Inference completed in {processing_time:.2f} seconds")
        
        return InferenceResponse(
            status="success",
            scroll_name=request.scroll_name,
            segment_name=request.segment_name,
            prediction_shape=list(output_mask.shape),
            processing_time_seconds=processing_time,
            output_format=request.output_format,
            metadata={
                "layer_range": f"{request.layer_start}-{request.layer_end}",
                "is_reverse_segment": is_reverse_segment,
                "has_fragment_mask": fragment_mask is not None,
                "cache_path": str(cache_path),
                "device": str(config.device)
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Inference failed: {e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time_seconds": processing_time,
                "scroll_name": request.scroll_name,
                "segment_name": request.segment_name
            }
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model = get_global_model()
        
        return {
            "model_loaded": True,
            "device": str(config.device),
            "num_gpus": config.num_gpus,
            "model_compiled": config.model_compile,
            "model_path": config.model_path
        }
    except Exception as e:
        return {
            "model_loaded": False,
            "error": str(e)
        }

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        logger.info("Initializing model on startup...")
        initialize_global_model()
        logger.info("Server startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up model on shutdown"""
    try:
        logger.info("Cleaning up model on shutdown...")
        cleanup_global_model()
        logger.info("Server shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker due to GPU model loading
        log_level="info",
        access_log=True
    )
