import os
import io
import sys

# Allow huge images before anything imports cv2 (and before importing inference_timesformer)
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")

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
from botocore.config import Config
import cv2
import concurrent.futures
import zarr
from huggingface_hub import snapshot_download

# WebKnossos imports
try:
    from webknossos.dataset import Dataset
    from webknossos.dataset.layer import Layer
    from webknossos.geometry.mag import Mag
except:
    pass

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
    wk_inference: bool = False
    wk_dataset_id: str = ""
    step: str = "inference"  # "prepare", "inference", or "reduce"
    num_parts: int = 1
    part_id: int = 0
    zarr_output_dir: str = "/tmp/partitions"
    surface_volume_zarr: str = ""  # Path to pre-created surface volume zarr
    chunk_size: int = 1024  # Chunk size for zarr array creation (SURFACE_VOLUME_CHUNK_SIZE)
    use_zarr_compression: bool = False  # Enable/disable zarr compression
    model_type: str = "timesformer"  # "timesformer" or "resnet3d"
    encoder: str = "resnet50"  # For resnet3d: "resnet34", "resnet50", "resnet101"

def parse_env() -> Inputs:
    try:
        model_key = os.environ["MODEL"].strip()
        s3_path = os.getenv("S3_PATH", "").strip()
        start_layer = int(os.environ["START_LAYER"].strip())
        end_layer = int(os.environ["END_LAYER"].strip())
        force_reverse = os.getenv("FORCE_REVERSE", "false").lower() == "true"
        wk_dataset_id = os.getenv("WK_DATASET_ID", "").strip()

        # Map/reduce parameters
        step = os.getenv("STEP", "inference").strip().lower()
        num_parts = int(os.getenv("NUM_PARTS", "1"))
        part_id = int(os.getenv("PART_ID", "0"))
        zarr_output_dir = os.getenv("ZARR_OUTPUT_DIR", "/tmp/partitions").strip()
        surface_volume_zarr = os.getenv("SURFACE_VOLUME_ZARR", "").strip()
        chunk_size = int(os.getenv("SURFACE_VOLUME_CHUNK_SIZE", "1024"))
        use_zarr_compression = os.getenv("USE_ZARR_COMPRESSION", "false").lower() == "true"

        # Model type and encoder parameters
        model_type = os.getenv("MODEL_TYPE", "timesformer").strip().lower()
        encoder = os.getenv("ENCODER", "resnet50").strip().lower()

        # Validate model_type
        if model_type not in ("timesformer", "resnet3d"):
            raise ValueError(f"MODEL_TYPE must be 'timesformer' or 'resnet3d', got '{model_type}'")

        # Validate step parameter
        if step not in ("prepare", "inference", "reduce"):
            raise ValueError(f"STEP must be 'prepare', 'inference', or 'reduce', got '{step}'")

        # Validate partitioning parameters
        if step == "inference" and num_parts > 1:
            if part_id < 0 or part_id >= num_parts:
                raise ValueError(f"PART_ID must be in range [0, {num_parts}), got {part_id}")

        if step == "reduce" and num_parts <= 1:
            raise ValueError("STEP=reduce requires NUM_PARTS > 1")

        # Validate required parameters per step
        if step == "prepare":
            # Prepare step requires s3_path or wk_dataset_id
            if not s3_path and not wk_dataset_id:
                raise ValueError("STEP=prepare requires either S3_PATH or WK_DATASET_ID")
        elif step == "inference":
            # Inference step requires surface_volume_zarr
            if not surface_volume_zarr:
                raise ValueError("STEP=inference requires SURFACE_VOLUME_ZARR (run STEP=prepare first)")
            # For non-partitioned inference, also need s3_path for result upload
            if num_parts == 1 and not s3_path and not wk_dataset_id:
                raise ValueError("STEP=inference with NUM_PARTS=1 requires S3_PATH or WK_DATASET_ID for result upload")
        # reduce step doesn't require these

        wk_inference = bool(wk_dataset_id)

        if start_layer > end_layer:
            raise ValueError("START_LAYER must be <= END_LAYER")

        return Inputs(
            model_key=model_key,
            s3_path=s3_path,
            start_layer=start_layer,
            end_layer=end_layer,
            force_reverse=force_reverse,
            wk_inference=wk_inference,
            wk_dataset_id=wk_dataset_id,
            step=step,
            num_parts=num_parts,
            part_id=part_id,
            zarr_output_dir=zarr_output_dir,
            surface_volume_zarr=surface_volume_zarr,
            chunk_size=chunk_size,
            use_zarr_compression=use_zarr_compression,
            model_type=model_type,
            encoder=encoder,
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
    # OR directly under prefix for surface-volumes (e.g., paths ending in .tifs/)
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: List[Tuple[str, str]] = []
    SUPPORTED_IMAGE_FORMATS = {'.tif', '.tiff', '.png', '.jpeg', '.jpg'}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Check if it's a layer file with supported format
            # Support both: paths with "/layers/" subdirectory AND surface-volumes with .tifs/ directory
            if "/layers/" not in key.lower() and not prefix.endswith(".tifs/"):
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

            # Check if layer index is within range (exclusive end) -> [start_layer, end_layer)
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


def get_wk_dataset_metadata(wk_dataset_id: str) -> str:
    """
    Fetch metadata from WebKnossos dataset and extract s3_path.
    
    Args:
        wk_dataset_id: WebKnossos dataset ID
        
    Returns:
        s3_path extracted from dataset metadata
        
    Raises:
        RuntimeError: If s3_path is not found in metadata
    """
    try:
        logger.info(f"Fetching metadata for WebKnossos dataset: {wk_dataset_id}")
        dataset = Dataset.open_remote(wk_dataset_id)
        metadata = dataset.metadata

        if "s3_path" not in metadata:
            raise RuntimeError(f"s3_path not found in metadata for dataset {wk_dataset_id}")

        s3_path = metadata["s3_path"]
        logger.info(f"Found s3_path in dataset metadata: {s3_path}")
        return s3_path

    except Exception as e:
        logger.error(f"Failed to fetch metadata from WebKnossos dataset {wk_dataset_id}: {e}")
        raise RuntimeError(f"Failed to fetch WebKnossos dataset metadata: {e}") from e


def upload_to_webknossos(wk_dataset_id: str, prediction: np.ndarray, model_key: str, start_layer: int, end_layer: int) -> str:
    """
    Upload prediction results to WebKnossos dataset as a new layer.
    
    Args:
        wk_dataset_id: WebKnossos dataset ID
        prediction: Prediction array to upload
        model_key: Model identifier for layer naming
        start_layer: Start layer index
        end_layer: End layer index
        
    Returns:
        Layer name of uploaded prediction
    """
    try:
        logger.info(f"Uploading prediction to WebKnossos dataset: {wk_dataset_id}")

        # Open the remote dataset
        dataset = Dataset.open_remote(wk_dataset_id)

        # Create layer name
        layer_name = f"ink_prediction_{model_key}_{start_layer:02d}_{end_layer:02d}"
        logger.info(f"Creating layer: {layer_name}")

        # Convert prediction to uint8
        prediction_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

        # Add layer to dataset
        # The prediction is 2D, so we need to add a third dimension for WebKnossos
        prediction_3d = prediction_uint8[:, :, np.newaxis]

        layer = dataset.add_layer(
            layer_name=layer_name,
            category="segmentation",
            dtype_per_channel="uint8",
            num_channels=1,
            data_format="wkw"
        )

        # Write the prediction data
        with layer.open_mag(Mag(1)) as mag:
            mag.write(prediction_3d, offset=(0, 0, 0))

        logger.info(f"Successfully uploaded prediction as layer: {layer_name}")
        return layer_name

    except Exception as e:
        logger.error(f"Failed to upload to WebKnossos: {e}")
        raise RuntimeError(f"Failed to upload prediction to WebKnossos: {e}") from e


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


def run_prepare_step(inputs: Inputs) -> None:
    """Execute the prepare step to create surface volume zarr from S3 layers."""
    # Import torch-free processing utilities
    from processing import create_surface_volume_zarr, path_exists

    logger.info("Starting prepare step: creating surface volume zarr")

    # Handle WebKnossos workflow
    if inputs.wk_inference:
        logger.info(f"WebKnossos inference mode: fetching metadata for dataset {inputs.wk_dataset_id}")
        inputs.s3_path = get_wk_dataset_metadata(inputs.wk_dataset_id)
        logger.info(f"Retrieved s3_path from WebKnossos metadata: {inputs.s3_path}")

    # S3 setup
    logger.info("Setting up S3 client...")
    s3_client = boto3.client("s3")
    logger.info(f"Parsing S3 URI: {inputs.s3_path}")
    bucket, prefix = parse_s3_uri(inputs.s3_path)

    logger.info(f"Listing layer objects in S3 bucket '{bucket}' with prefix '{prefix}' for layers [{inputs.start_layer}, {inputs.end_layer})")
    layer_objects = list_layers_objects(
        s3_client, bucket, prefix, inputs.start_layer, inputs.end_layer
    )
    logger.info(f"Found {len(layer_objects)} layer objects to download")

    # Download layers to temporary directory
    work_dir = "/workspace"
    input_dir = os.path.join(work_dir, "input", "layers")
    logger.info(f"Ensuring clean input directory at {os.path.join(work_dir, 'input')}")
    ensure_clean_dir(os.path.join(work_dir, "input"))

    logger.info(f"Downloading layer files to {input_dir} ...")
    layer_paths = download_layers(s3_client, bucket, layer_objects, input_dir)
    logger.info(f"Downloaded {len(layer_paths)} layer files")

    if inputs.surface_volume_zarr:
        output_path = inputs.surface_volume_zarr
    else:
        output_path = f"/tmp/surface_volume_{inputs.start_layer:02d}_{inputs.end_layer:02d}.zarr"

    logger.info(f"Creating surface volume zarr at {output_path} with chunk_size={inputs.chunk_size}, compression={inputs.use_zarr_compression}")
    created_zarr_path = create_surface_volume_zarr(
        layer_paths,
        output_path,
        chunk_size=inputs.chunk_size,
        use_compression=inputs.use_zarr_compression
    )

    # Write output path to file for next step
    output_file = "/tmp/surface_volume_zarr_path.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(created_zarr_path)

    logger.info(f"Prepare step completed successfully. Surface volume zarr created at: {created_zarr_path}")
    logger.info(f"Zarr path written to: {output_file}")


def run_inference_step(inputs: Inputs) -> None:
    """Execute the inference step (either standard or partitioned mode)."""
    # Import torch and related dependencies only when doing inference
    import torch
    from torch.nn import DataParallel
    from inference import run_inference, CFG
    from processing import path_exists

    # Import model-specific module based on model_type
    if inputs.model_type == "timesformer":
        from model_timesformer import load_model, TimeSformerConfig
        CFG.in_chans = TimeSformerConfig.in_chans
        logger.info(f"Using TimeSformer model with {CFG.in_chans} input channels")
    elif inputs.model_type == "resnet3d":
        from model_resnet3d import load_model, ResNet3DConfig
        CFG.in_chans = ResNet3DConfig.in_chans
        logger.info(f"Using ResNet3D model ({inputs.encoder}) with {CFG.in_chans} input channels")
    else:
        raise ValueError(f"Unknown model_type: {inputs.model_type}")

    task_id = uuid.uuid4()
    logger.info(f"Task ID generated: {task_id}")

    # Verify surface volume zarr exists (supports both local paths and S3 URLs)
    if not path_exists(inputs.surface_volume_zarr):
        raise RuntimeError(f"Surface volume zarr not found at: {inputs.surface_volume_zarr}")

    logger.info(f"Using surface volume zarr: {inputs.surface_volume_zarr}")

    # Configure CFG with partition parameters
    CFG.num_parts = inputs.num_parts
    CFG.part_id = inputs.part_id
    CFG.zarr_output_dir = inputs.zarr_output_dir

    logger.info(
        f"Inference step: num_parts={inputs.num_parts}, part_id={inputs.part_id}, "
        f"zarr_output_dir={inputs.zarr_output_dir}"
    )

    # For uploading results, we need s3_path
    # Handle WebKnossos workflow if needed
    if inputs.wk_inference:
        logger.info(f"WebKnossos inference mode: fetching metadata for dataset {inputs.wk_dataset_id}")
        inputs.s3_path = get_wk_dataset_metadata(inputs.wk_dataset_id)
        logger.info(f"Retrieved s3_path from WebKnossos metadata: {inputs.s3_path}")

    logger.info(
        f"Starting optimized inference with task_id={task_id}, model={inputs.model_key}, "
        f"surface_volume_zarr={inputs.surface_volume_zarr}, "
        f"layers=[{inputs.start_layer}, {inputs.end_layer}], wk_inference={inputs.wk_inference}"
    )

    # Prepare models directory
    work_dir = "/workspace"
    models_dir = os.path.join(work_dir, "models")
    logger.info(f"Ensuring models directory exists at {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    # S3 setup (for model download and result upload)
    logger.info("Setting up S3 client...")
    s3_client = boto3.client("s3")

    # Parse s3_path for result upload
    if inputs.s3_path:
        logger.info(f"Parsing S3 URI for result upload: {inputs.s3_path}")
        bucket, prefix = parse_s3_uri(inputs.s3_path)
    else:
        bucket, prefix = None, None

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Resolve and download model weights (S3-first, then HF fallback)
    logger.info(f"Resolving model for key: {inputs.model_key}")
    logger.info(f"Looking for weights in S3 registry, else HF repo: {inputs.model_key}")
    weight_path = download_model_weights(inputs.model_key, models_dir, s3_client)
    logger.info(f"Loading model from weights at: {weight_path}")

    # Load model with appropriate parameters
    if inputs.model_type == "resnet3d":
        model = load_model(weight_path, device, encoder=inputs.encoder)
    else:
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
        # Access the underlying model from the wrapper
        target = model.model.module if isinstance(model.model, DataParallel) else model.model
        model_compiled = torch.compile(target, mode=COMPILE_MODE, fullgraph=True, dynamic=False)
        if isinstance(model.model, DataParallel):
            model.model.module = model_compiled
        else:
            model.model = model_compiled
        logger.info(f"Enabled torch.compile (mode={COMPILE_MODE})")
        # Tiny warmup to trigger compilation before the big loop (hides first-iter cost)
        try:
            dummy = torch.zeros((1, 1, CFG.in_chans, CFG.size, CFG.size), device=device)
            with torch.inference_mode():
                with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=True):
                    _ = model.forward(dummy)
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

    # Run inference from zarr
    # Note: When using a predefined surface volume zarr (not created by the prepare step),
    # it may contain more layers than requested. We use start_layer and end_layer to crop
    # to the desired range within the zarr.
    logger.info(f"Running inference from surface volume zarr with layer range [{inputs.start_layer}, {inputs.end_layer})...")
    start_infer_time = time.time()
    result = run_inference(
        inputs.surface_volume_zarr,
        model,
        device,
        is_reverse_segment=is_reverse_segment,
        start_z=inputs.start_layer,
        end_z=inputs.end_layer
    )
    logger.info(f"Inference completed in {time.time() - start_infer_time:.2f} seconds")

    # Handle partitioned vs standard mode
    if inputs.num_parts > 1:
        # Partitioned mode: zarr files written, just log and exit
        logger.info(f"Partition {inputs.part_id} completed. Zarr arrays written to {inputs.zarr_output_dir}")
        logger.info("Partition inference step complete. Run reduce step after all partitions finish.")
        return

    # Standard mode (num_parts == 1): upload result
    prediction = result  # result is np.ndarray in standard mode
    if inputs.wk_inference:
        # For WebKnossos inference, upload to both S3 and WebKnossos
        logger.info("Saving and uploading prediction mask to S3...")
        result_uri = save_and_upload_prediction(
            s3_client, bucket, prefix, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
        )
        logger.info(f"Uploaded result to S3: {result_uri}")

        # Upload to WebKnossos as new layer
        logger.info("Uploading prediction to WebKnossos dataset...")
        wk_layer_name = upload_to_webknossos(
            inputs.wk_dataset_id, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
        )
        logger.info(f"Uploaded prediction to WebKnossos layer: {wk_layer_name}")

        # Write both results
        logger.info(f"Writing result S3 URI to /tmp/result_s3_url.txt: {result_uri}")
        with open("/tmp/result_s3_url.txt", "w", encoding="utf-8") as f:
            f.write(result_uri)

        logger.info(f"Writing WebKnossos layer name to /tmp/result_wk_layer.txt: {wk_layer_name}")
        with open("/tmp/result_wk_layer.txt", "w", encoding="utf-8") as f:
            f.write(wk_layer_name)

        logger.info(f"Inference completed successfully - S3: {result_uri}, WebKnossos layer: {wk_layer_name}")
    else:
        # Standard S3-only workflow
        logger.info("Saving and uploading prediction mask to S3...")
        result_uri = save_and_upload_prediction(
            s3_client, bucket, prefix, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
        )
        logger.info(f"Writing result S3 URI to /tmp/result_s3_url.txt: {result_uri}")
        with open("/tmp/result_s3_url.txt", "w", encoding="utf-8") as f:
            f.write(result_uri)
        logger.info(f"Uploaded result to {result_uri}")


def run_reduce_step(inputs: Inputs) -> None:
    """Execute the reduce step to blend all partitions."""
    # Import torch-free processing utilities
    from processing import reduce_partitions, write_tiled_tiff

    logger.info(f"Starting reduce step: blending {inputs.num_parts} partitions")

    # S3 setup (need it for uploading final result)
    s3_client = boto3.client("s3")

    # Handle WebKnossos workflow to get s3_path
    if inputs.wk_inference:
        logger.info(f"WebKnossos inference mode: fetching metadata for dataset {inputs.wk_dataset_id}")
        inputs.s3_path = get_wk_dataset_metadata(inputs.wk_dataset_id)
        logger.info(f"Retrieved s3_path from WebKnossos metadata: {inputs.s3_path}")

    bucket, prefix = parse_s3_uri(inputs.s3_path)

    # We need to determine the prediction shape from one of the partition zarr files
    mask_pred_path = os.path.join(inputs.zarr_output_dir, "mask_pred_part_000.zarr")
    if not os.path.exists(mask_pred_path):
        raise RuntimeError(f"Partition 0 not found at {mask_pred_path}. Ensure all inference partitions completed.")

    logger.info(f"Reading prediction shape from {mask_pred_path}")
    z = zarr.open(mask_pred_path, mode='r')
    pred_shape = z.shape
    logger.info(f"Prediction shape: {pred_shape}")

    # Run reduce/blend
    logger.info(f"Reducing {inputs.num_parts} partitions from {inputs.zarr_output_dir}")
    start_reduce_time = time.time()
    prediction = reduce_partitions(inputs.zarr_output_dir, inputs.num_parts, pred_shape)
    logger.info(f"Reduce completed in {time.time() - start_reduce_time:.2f} seconds")

    # Write to local tiled TIFF
    local_tiff_path = f"/tmp/prediction_{inputs.model_key}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif"
    write_tiled_tiff(prediction, local_tiff_path)

    # Upload TIFF to S3
    logger.info("Uploading tiled TIFF to S3...")
    out_key = os.path.join(prefix.rstrip("/"), "predictions", f"prediction_{inputs.model_key}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif")
    s3_client.upload_file(local_tiff_path, bucket, out_key)
    result_uri = f"s3://{bucket}/{out_key}"
    logger.info(f"Uploaded result to S3: {result_uri}")

    # Write result URI
    with open("/tmp/result_s3_url.txt", "w", encoding="utf-8") as f:
        f.write(result_uri)

    # If WebKnossos mode, also upload to WebKnossos
    if inputs.wk_inference:
        logger.info("Uploading prediction to WebKnossos dataset...")
        wk_layer_name = upload_to_webknossos(
            inputs.wk_dataset_id, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
        )
        logger.info(f"Uploaded prediction to WebKnossos layer: {wk_layer_name}")

        with open("/tmp/result_wk_layer.txt", "w", encoding="utf-8") as f:
            f.write(wk_layer_name)

        logger.info(f"Reduce completed successfully - S3: {result_uri}, WebKnossos layer: {wk_layer_name}")
    else:
        logger.info(f"Reduce completed successfully - S3: {result_uri}")


def main() -> None:
    """Main entrypoint with step dispatch."""
    logger.info("Parsing environment variables for input configuration...")
    inputs = parse_env()

    logger.info(
        f"Starting optimized inference: step={inputs.step}, model={inputs.model_key}, "
        f"s3_path={inputs.s3_path}, layers=[{inputs.start_layer}, {inputs.end_layer}]"
    )

    # Dispatch to appropriate step
    if inputs.step == "prepare":
        run_prepare_step(inputs)
    elif inputs.step == "inference":
        run_inference_step(inputs)
    elif inputs.step == "reduce":
        run_reduce_step(inputs)
    else:
        raise ValueError(f"Unknown step: {inputs.step}")


if __name__ == "__main__":
    main()
