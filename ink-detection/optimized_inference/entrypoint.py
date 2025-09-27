import os
import io
import sys
import json
import time
import math
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import uuid
import boto3
import numpy as np
from torch.nn import DataParallel
from botocore.config import Config
import cv2
import torch
import concurrent.futures
from huggingface_hub import snapshot_download

from inference_timesformer import (
    RegressionPLModel,
    run_inference,
    CFG,
)

from torch.nn import DataParallel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("optimized_inference.entrypoint")


@dataclass
class Inputs:
    model_key: str
    s3_path: str
    start_layer: int
    end_layer: int
    force_reverse: bool = False

def parse_env() -> Inputs:
    try:
        model_key = os.environ["MODEL"].strip()
        s3_path = os.environ["S3_PATH"].strip()
        start_layer = int(os.environ["START_LAYER"].strip())
        end_layer = int(os.environ["END_LAYER"].strip())
        force_reverse = os.getenv("FORCE_REVERSE", "false").lower() == "true"
        if start_layer > end_layer:
            raise ValueError("START_LAYER must be <= END_LAYER")
        return Inputs(
            model_key=model_key,
            s3_path=s3_path,
            start_layer=start_layer,
            end_layer=end_layer,
            force_reverse=force_reverse,
        )
    except KeyError as e:
        raise RuntimeError(f"Missing required env var: {e.args[0]}") from e


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[len("s3://") :]
    parts = path.split("/", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def list_layers_objects(
    s3_client, bucket: str, prefix: str, start_layer: int, end_layer: int
) -> List[Tuple[str, str]]:
    # Return list of (key, basename) for .tif/.tiff/.png/.jpeg/.jpg files inside any "layers/" folder under prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: List[Tuple[str, str]] = []
    SUPPORTED_IMAGE_FORMATS = {'.tif', '.tiff', '.png', '.jpeg', '.jpg'}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            
            # Check if it's a layer file with supported format
            if "/layers/" not in key.lower():
                continue
                
            base = os.path.basename(key)
            name, ext = os.path.splitext(base)
            
            # Check if the file extension is supported
            if ext.lower() not in SUPPORTED_IMAGE_FORMATS:
                continue
                
            try:
                # Tolerate leading zeros, e.g., 01, 02, ...
                layer_idx = int(name)
            except ValueError:
                continue
                
            # Check if layer index is within range (exclusive end)
            if start_layer <= layer_idx < end_layer:
                keys.append((key, base))
    if not keys:
        raise RuntimeError(
            f"No layers found within range [{start_layer}, {end_layer}) under s3://{bucket}/{prefix}"
        )
    # Sort by numeric layer index
    keys.sort(key=lambda kv: int(os.path.splitext(kv[1])[0]))
    return keys


def download_layers(
    s3_client, bucket: str, objects: List[Tuple[str, str]], out_dir: str
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    session = boto3.Session()
    config = Config(
        max_pool_connections=20,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )

    def _download_one(args):
        idx, key, base, bucket, out_dir = args
        out_path = os.path.join(out_dir, base)
        # Each thread needs its own s3_client, so we pass it in the outer scope
        session.client("s3", config=config).download_file(bucket, key, out_path)
        logger.info(f"Finished downloading layer {idx}: {out_path}")
        return out_path

    paths: List[str] = []
    # Prepare arguments for each download
    download_args = [(idx, key, base, bucket, out_dir) for idx, (key, base) in enumerate(objects)]

    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(objects))) as executor:
        # Map returns results in order of the input
        results = list(executor.map(_download_one, download_args))
        paths.extend(results)

    return paths


def load_layers_to_numpy(layer_paths: List[str]) -> np.ndarray:
    if not layer_paths:
        raise ValueError("No layer paths provided")
    
    # Load all images first to ensure consistent processing
    images = []
    for i, path in enumerate(layer_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        images.append(img)
    
    # Ensure all images have the same shape
    h, w = images[0].shape
    for i, img in enumerate(images):
        if img.shape != (h, w):
            raise RuntimeError(
                f"Layer size mismatch: {layer_paths[i]} has {img.shape}, expected {(h, w)}"
            )
    
    # Stack layers using the same method as local script
    # This creates (H, W, C) format like the working version
    stacked_layers = np.stack(images, axis=2)
    
    # Ensure proper dtype - match the local script behavior
    # Don't convert to float32 here, let the inference function handle it
    return stacked_layers


def download_model_weights(model_name: str, dest_dir: str, s3_client) -> str:
    """
    Resolve model weights by checking S3 registry first, then fall back to Hugging Face.

    Search order preference for files: .ckpt, .safetensors, .bin, .pt
    """
    os.makedirs(dest_dir, exist_ok=True)

    registry_bucket = "scrollprize-models-registry"
    registry_prefix = f"ink-detection/{model_name.strip().rstrip('/')}/"

    def _prefer(weights: List[str]) -> Optional[str]:
        if not weights:
            return None
        order = [".ckpt", ".safetensors", ".bin", ".pt"]
        for ext in order:
            matches = [w for w in weights if w.lower().endswith(ext)]
            if matches:
                return sorted(matches)[0]
        return sorted(weights)[0]

    # 1) Try S3 registry first
    logger.info(
        f"Attempting to locate weights in S3 registry s3://{registry_bucket}/{registry_prefix}"
    )
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        found_keys: List[str] = []
        for page in paginator.paginate(Bucket=registry_bucket, Prefix=registry_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                lower = key.lower()
                if lower.endswith(".ckpt") or lower.endswith(".safetensors") or lower.endswith(".bin") or lower.endswith(".pt"):
                    found_keys.append(key)

        chosen_key = _prefer(found_keys)
        if chosen_key:
            logger.info(f"Found weights in S3 registry: s3://{registry_bucket}/{chosen_key}")
            local_path = os.path.join(dest_dir, os.path.basename(chosen_key))
            s3_client.download_file(registry_bucket, chosen_key, local_path)
            logger.info(f"Downloaded weights from S3 to: {local_path}")
            return local_path
        else:
            logger.info("No suitable weights found in S3 registry. Falling back to Hugging Face.")
    except Exception as e:
        logger.warning(f"S3 registry lookup failed ({e}). Falling back to Hugging Face.")

    # 2) Fall back to Hugging Face
    logger.info(f"Downloading model from Hugging Face: {model_name}")
    local_dir = snapshot_download(repo_id=model_name, local_dir=dest_dir, local_dir_use_symlinks=False)
    candidates = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".ckpt") or lf.endswith(".safetensors") or lf.endswith(".bin") or lf.endswith(".pt"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise RuntimeError("No model weight files (.ckpt/.safetensors/.bin/.pt) found in downloaded repo")
    chosen = _prefer(candidates)
    logger.info(f"Using model weights: {chosen}")
    return chosen


def load_model(model_path: str, device: torch.device) -> RegressionPLModel:
    """
    Load and initialize the TimeSformer model.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Loaded and initialized model
    """
    try:
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
        
        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")
        
        # Move to device
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def save_and_upload_prediction(
    s3_client, bucket: str, prefix: str, prediction: np.ndarray, model_key: str, start_layer: int, end_layer: int
) -> str:
    # Convert to uint8 PNG and upload to s3://bucket/prefix/predictions/prediction_START_END.png
    out_key = os.path.join(prefix.rstrip("/"), "predictions", f"prediction_{model_key}_{start_layer:02d}_{end_layer:02d}.png")
    # Ensure parent prefix virtually exists
    _, tmp_path = os.path.split(out_key)
    os.makedirs("/tmp/outputs", exist_ok=True)
    local_path = os.path.join("/tmp/outputs", tmp_path)
    prediction_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(local_path, prediction_uint8)
    s3_client.upload_file(local_path, bucket, out_key)
    return f"s3://{bucket}/{out_key}"


def main() -> None:

    task_id = uuid.uuid4()
    logger.info(f"Task ID generated: {task_id}")

    logger.info("Parsing environment variables for input configuration...")
    inputs = parse_env()
    logger.info(
        f"Starting optimized inference with task_id={task_id}, model={inputs.model_key}, s3_path={inputs.s3_path}, "
        f"layers=[{inputs.start_layer}, {inputs.end_layer}]"
    )

    # Prepare I/O directories
    work_dir = "/workspace"
    input_dir = os.path.join(work_dir, "input", "layers")
    models_dir = os.path.join(work_dir, "models")
    logger.info(f"Ensuring clean input directory at {os.path.join(work_dir, 'input')}")
    ensure_clean_dir(os.path.join(work_dir, "input"))
    logger.info(f"Ensuring models directory exists at {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    # S3 setup
    logger.info("Setting up S3 client...")
    s3_client = boto3.client("s3")
    logger.info(f"Parsing S3 URI: {inputs.s3_path}")
    bucket, prefix = parse_s3_uri(inputs.s3_path)
    logger.info(f"Listing layer objects in S3 bucket '{bucket}' with prefix '{prefix}' for layers {inputs.start_layer} to {inputs.end_layer}")
    layer_objects = list_layers_objects(
        s3_client, bucket, prefix, inputs.start_layer, inputs.end_layer
    )
    logger.info(f"Found {len(layer_objects)} layer objects to download")
    logger.info(f"Downloading layer files to {input_dir} ...")
    layer_paths = download_layers(s3_client, bucket, layer_objects, input_dir)
    logger.info(f"Downloaded {len(layer_paths)} layer files")

    # Stream tiles directly from the downloaded files inside the inference module.
    num_layers = len(layer_paths)
    logger.info(f"Prepared {num_layers} layer files for streaming")
    logger.info(f"Model expects {CFG.in_chans} channels, got {num_layers} layers") 

    if CFG.in_chans != num_layers:
        raise ValueError(f"Channel mismatch: model expects {CFG.in_chans}, got {num_layers}")
        
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Resolve and download model weights (S3-first, then HF fallback)
    logger.info(f"Resolving model for key: {inputs.model_key}")
    logger.info(f"Looking for weights in S3 registry, else HF repo: {inputs.model_key}")
    weight_path = download_model_weights(inputs.model_key, models_dir, s3_client)
    logger.info(f"Loading model from weights at: {weight_path}")
    model = load_model(weight_path, device)

    # -------- Performance toggles ------------------------------------------------
    # TF32 on Ampere+ gives fast GEMMs with tiny accuracy impact for this task.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # torch.compile defaults
    COMPILE = os.getenv("COMPILE", "1") == "1" and hasattr(torch, "compile")
    COMPILE_MODE = os.getenv("COMPILE_MODE", "reduce-overhead")  # <- default changed
    if COMPILE:
        # Persist Inductor cache across runs (huge win after the first run)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("./inductor_cache"))
        # If not doing max tuning, disable heavy autotuning to avoid OOM spam & overhead
        if COMPILE_MODE != "max-autotune":
            os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
            os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
            os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", "0")
        # Optional: CUDA graphs (static shapes); enable if you don’t hit driver bugs
        if os.getenv("CUDAGRAPHS", "0") == "1":
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "1")
        # Compile
        target = model.module if isinstance(model, DataParallel) else model
        model_compiled = torch.compile(target, mode=COMPILE_MODE, fullgraph=True, dynamic=False)
        if isinstance(model, DataParallel):
            model.module = model_compiled
        else:
            model = model_compiled
        logger.info(f"Enabled torch.compile (mode={COMPILE_MODE})")
        # Tiny warmup to trigger compilation before the big loop (hides first-iter cost)
        try:
            dummy = torch.zeros((1, 1, CFG.in_chans, CFG.size, CFG.size), device=device)
            with torch.inference_mode():
                with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=True):
                    _ = model(dummy)
            del dummy
        except Exception as e:
            logger.warning(f"Warmup after compile failed (continuing un-warmed): {e}")

    # Determine reverse option similar to local test
    segment_name = os.path.basename(prefix.rstrip("/")) if prefix else bucket
    if inputs.force_reverse:
        is_reverse_segment = True
        logger.info("Force reverse enabled via env")
    else:
        is_reverse_segment = False

    # Run inference
    logger.info("Running inference from streamed layer files...")
    start_infer_time = time.time()
    prediction = run_inference(layer_paths, model, device, is_reverse_segment=is_reverse_segment)
    logger.info(f"Inference completed in {time.time() - start_infer_time:.2f} seconds")

    # Upload result
    logger.info("Saving and uploading prediction mask to S3...")
    result_uri = save_and_upload_prediction(
        s3_client, bucket, prefix, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
    )
    logger.info(f"Writing result S3 URI to /tmp/result_s3_url.txt: {result_uri}")
    with open("/tmp/result_s3_url.txt", "w", encoding="utf-8") as f:
        f.write(result_uri)
    logger.info(f"Uploaded result to {result_uri}")


if __name__ == "__main__":
    main()