import json
import socket
import os
import uuid
import click
import torch
import numpy as np
import random
import edt
from pathlib import Path

from vesuvius.neural_tracing.models import make_model, load_checkpoint
from vesuvius.neural_tracing.infer import Inference, EdtInference

# Debug output - writes data on first request only
_debug_written = False
DEBUG_OUTPUT_DIR = "/tmp/neural_trace_debug"


def write_debug_data(subdir: str, **data):
    """Write debug tensors/arrays to /tmp/neural_trace_debug/{subdir}/"""
    out_path = Path(DEBUG_OUTPUT_DIR) / subdir
    out_path.mkdir(parents=True, exist_ok=True)
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            np.save(out_path / f"{name}.npy", value.detach().cpu().numpy())
        elif isinstance(value, np.ndarray):
            np.save(out_path / f"{name}.npy", value)
        elif isinstance(value, (dict, list)):
            with open(out_path / f"{name}.json", "w") as f:
                json.dump(value, f, indent=2, default=str)
    print(f"[DEBUG] Wrote debug data to {out_path}")


MAX_OUTPUT_FILES = 50


def cleanup_old_outputs(output_dir: Path, max_files: int = MAX_OUTPUT_FILES):
    """Remove oldest zarr outputs if count exceeds max_files."""
    import shutil
    if not output_dir.exists():
        return
    # Get all zarr directories sorted by modification time (oldest first)
    zarr_dirs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.suffix == '.zarr'],
        key=lambda p: p.stat().st_mtime
    )
    # Remove oldest files until we're under the limit
    while len(zarr_dirs) >= max_files:
        oldest = zarr_dirs.pop(0)
        shutil.rmtree(oldest)


def save_array(arr: np.ndarray, output_dir: Path, prefix: str) -> str:
    """Save numpy array to zarr, return path."""
    import zarr
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old outputs before saving new one
    cleanup_old_outputs(output_dir)

    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.zarr"
    path = output_dir / filename
    zarr.save_array(
        path,
        arr.astype(np.float32),
        chunks=arr.shape,  # Single chunk - arrays are small (64^3 or 96^3)
    )
    return str(path)


def load_zarr_array(path: str) -> torch.Tensor:
    """Load array from zarr file."""
    import zarr

    print(f"[DEBUG] load_zarr_array called with path: '{path}'", flush=True)

    if not path:
        raise ValueError(f"Empty path provided to load_zarr_array")
    if not os.path.exists(path):
        # List parent directory contents for debugging
        parent = os.path.dirname(path)
        if os.path.exists(parent):
            contents = os.listdir(parent)
            print(f"[DEBUG] Parent dir {parent} contains: {contents}", flush=True)
        raise FileNotFoundError(f"Zarr path does not exist: {path}")

    # List contents of the zarr directory
    if os.path.isdir(path):
        contents = os.listdir(path)
        print(f"[DEBUG] Zarr dir {path} contains: {contents}", flush=True)

    # Check if this is a flat array (.zarray at root) or a group (attributes.json at root)
    zarray_path = os.path.join(path, '.zarray')
    if os.path.exists(zarray_path):
        # Flat array structure - open directly
        arr = zarr.open_array(path, mode='r')[:]
    else:
        # Group structure (z5 creates this) - look for array in subdirectory
        # z5 creates: path/attributes.json, path/0/.zarray, path/0/chunks...
        subdir = os.path.join(path, '0')
        if os.path.exists(subdir) and os.path.exists(os.path.join(subdir, '.zarray')):
            arr = zarr.open_array(subdir, mode='r')[:]
        else:
            # Fallback to generic open
            store = zarr.open(path, mode='r')
            if isinstance(store, zarr.Array):
                arr = store[:]
            elif hasattr(store, 'keys'):
                keys = list(store.keys())
                if keys:
                    arr = store[keys[0]][:]
                else:
                    raise ValueError(f"Zarr group at {path} has no arrays")
            else:
                arr = np.asarray(store)

    print(f"[DEBUG] Loaded array shape: {arr.shape}", flush=True)
    return torch.from_numpy(arr.astype(np.float32))


def detect_model_type(config: dict) -> str:
    """Detect model type from checkpoint config."""
    targets = config.get('targets', {})
    if 'dt' in targets:
        return 'edt'
    return 'uv'


def build_mask_from_points(points_zyx, center_zyx, crop_size, dilation_radius):
    """Rasterize points to dilated conditioning mask (all coords in model space ZYX).

    Args:
        points_zyx: List of torch.Tensor or np.ndarray, each shape [3] in ZYX model coords
        center_zyx: torch.Tensor or np.ndarray, shape [3] in ZYX model coords
        crop_size: int or list/tuple of 3 ints, size of the output volume (Z, Y, X)
        dilation_radius: float, radius to dilate points by (in model voxels)

    Returns:
        torch.Tensor: Binary mask of shape crop_size
    """
    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    center_np = to_numpy(center_zyx)
    points_np = [to_numpy(p) for p in points_zyx]

    # Handle both scalar and list crop_size
    if isinstance(crop_size, (list, tuple)):
        crop_size_zyx = np.array(crop_size)
    else:
        crop_size_zyx = np.array([crop_size, crop_size, crop_size])

    min_corner_zyx = center_np - crop_size_zyx // 2
    mask = np.zeros(tuple(crop_size_zyx), dtype=np.float32)

    for p in points_np:
        local = np.round(p - min_corner_zyx).astype(int)
        if (local >= 0).all() and (local < crop_size_zyx).all():
            mask[local[0], local[1], local[2]] = 1.0

    # Dilate to match training - compute distance from non-mask points
    dist = edt.edt(1 - mask, parallel=1)
    dilated_mask = (dist <= dilation_radius).astype(np.float32)

    return torch.from_numpy(dilated_mask)


@click.command()
@click.option('--checkpoint_path', type=click.Path(exists=True), required=True, help='Path to checkpoint file')
@click.option('--volume_zarr', type=click.Path(exists=True), required=True, help='Path to ome-zarr folder')
@click.option('--volume_scale', type=int, required=True, help='OME scale to use')
@click.option('--socket_path', type=click.Path(), required=True, help='Path to Unix domain socket')
@click.option('--no-cache', is_flag=True, help='Disable crop cache')
@click.option('--output_dir', type=click.Path(), default='/tmp/trace_service_output',
              help='Directory for output .npy files')
def serve(checkpoint_path, volume_zarr, volume_scale, socket_path, no_cache, output_dir):

    model, config = load_checkpoint(checkpoint_path)
    model_type = detect_model_type(config)
    output_dir = Path(output_dir)

    if no_cache:
        config['use_crop_cache'] = False

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    if model_type == 'edt':
        inference = None
        edt_inference = EdtInference(model, config, volume_zarr, volume_scale)
        print(f"Loaded EDT model")
    else:
        inference = Inference(model, config, volume_zarr, volume_scale)
        edt_inference = None
        print(f"Loaded UV heatmap model")

    socket_path = Path(socket_path)
    if socket_path.exists():
        os.unlink(socket_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(socket_path))
    sock.listen(1)

    print(f"neural-tracing service listening on {socket_path}")

    try:
        while True:
            conn, _ = sock.accept()
            try:
                handle_connection(conn, lambda request: process_request(
                    request, inference, edt_inference, volume_scale, model_type, output_dir
                ))
            except Exception as e:
                print(f"error handling connection: {e}")
            finally:
                conn.close()
    finally:
        sock.close()
        if socket_path.exists():
            os.unlink(socket_path)


def handle_connection(conn, request_fn):
    """Handle a single connection, processing JSON line requests."""

    with conn:
        fp = conn.makefile('rwb')
        for line in fp:
            line = line.strip()
            try:
                request = json.loads(line)
                response = request_fn(request)
                conn.sendall((json.dumps(response) + '\n').encode('utf-8'))
            except json.JSONDecodeError as e:
                print('error: invalid json:', e)
                error_response = {'error': f'Invalid JSON: {e}'}
                conn.sendall((json.dumps(error_response) + '\n').encode('utf-8'))
            except Exception as e:
                print('error:', e)
                error_response = {'error': f'Processing error: {e}'}
                conn.sendall((json.dumps(error_response) + '\n').encode('utf-8'))


def process_edt_batch_request(request, edt_inference, volume_scale, output_dir):
    """Process a batched EDT inference request."""
    global _debug_written
    import time
    t0 = time.time()

    if edt_inference is None:
        return {'error': 'EDT model not loaded. Load an EDT checkpoint to use edt_batch'}

    if 'center_xyz_batch' not in request:
        return {'error': 'Missing required field: center_xyz_batch'}

    centers_xyz = request['center_xyz_batch']
    batch_size = len(centers_xyz)
    print(f"[EDT_BATCH] Received batch request with {batch_size} samples", flush=True)
    scale = 2 ** volume_scale
    crop_size = edt_inference.config['crop_size']
    dilation_radius = edt_inference.config.get('dilation_radius', 1.0)

    # Convert centers to model-space ZYX
    centers_zyx = [
        torch.tensor([c[2], c[1], c[0]]) / scale
        for c in centers_xyz
    ]

    # Build conditioning masks for each sample
    conditioning_masks = []

    if 'conditioning_points_xyz_batch' in request:
        # Build masks from points
        points_batch = request['conditioning_points_xyz_batch']
        if len(points_batch) != batch_size:
            return {'error': f'conditioning_points_xyz_batch length {len(points_batch)} != center_xyz_batch length {batch_size}'}

        for center_zyx, points_xyz in zip(centers_zyx, points_batch):
            try:
                points_zyx = [
                    torch.tensor([p[2], p[1], p[0]]) / scale
                    for p in points_xyz
                ]
                mask = build_mask_from_points(points_zyx, center_zyx, crop_size, dilation_radius)
                conditioning_masks.append(mask)
            except Exception as e:
                print(f"[EDT_BATCH] Mask building failed: {e}", flush=True)
                conditioning_masks.append(None)

    elif 'conditioning_mask_paths' in request:
        # Load masks from files
        mask_paths = request['conditioning_mask_paths']
        if len(mask_paths) != batch_size:
            return {'error': f'conditioning_mask_paths length {len(mask_paths)} != center_xyz_batch length {batch_size}'}

        for path in mask_paths:
            try:
                mask = load_zarr_array(path)
                # Apply dilation to match training
                dist = edt.edt(1 - mask.numpy(), parallel=1)
                mask = torch.from_numpy((dist <= dilation_radius).astype(np.float32))
                conditioning_masks.append(mask)
            except Exception as e:
                print(f"[EDT_BATCH] Mask loading failed for {path}: {e}", flush=True)
                conditioning_masks.append(None)
    else:
        return {'error': 'edt_batch requires conditioning_points_xyz_batch or conditioning_mask_paths'}

    t1 = time.time()
    print(f"[EDT_BATCH] Mask building took {t1-t0:.3f}s", flush=True)

    # Run batched inference
    with torch.inference_mode():
        batch_results = edt_inference.get_distance_transform_batch(centers_zyx, conditioning_masks)

    t2 = time.time()
    print(f"[EDT_BATCH] Inference took {t2-t1:.3f}s for {batch_size} samples", flush=True)

    # Debug output for first sample of first batch
    if not _debug_written and batch_results:
        dt_first, min_corner_first, success, error_msg = batch_results[0]
        if success:
            # Re-run single inference with debug to get intermediate tensors
            _, _, debug_data = edt_inference.get_distance_transform_at(
                centers_zyx[0], conditioning_masks[0], return_debug=True
            )
            write_debug_data("edt",
                conditioning_mask=conditioning_masks[0],
                distance_transform=dt_first,
                metadata={
                    "center_xyz": centers_xyz[0],
                    "crop_size": crop_size,
                    "min_corner_zyx": min_corner_first.tolist(),
                    "volume_scale": volume_scale,
                },
                **debug_data
            )
        else:
            # Write whatever we have on failure for debugging
            mask_to_save = conditioning_masks[0] if conditioning_masks[0] is not None else None
            write_debug_data("edt_failed",
                metadata={
                    "center_xyz": centers_xyz[0],
                    "crop_size": crop_size if isinstance(crop_size, list) else [crop_size, crop_size, crop_size],
                    "volume_scale": volume_scale,
                    "error": error_msg,
                    "mask_shape": list(mask_to_save.shape) if mask_to_save is not None else None,
                },
                **({"conditioning_mask": mask_to_save} if mask_to_save is not None else {}),
            )
        _debug_written = True

    # Build response
    response = {'batch_results': [], 'scale_factor': scale, 'crop_size': crop_size}

    for i, (dt, min_corner_zyx, success, error_msg) in enumerate(batch_results):
        if not success:
            response['batch_results'].append({'error': error_msg})
            continue

        min_corner_xyz = (min_corner_zyx.flip(0) * scale).tolist()
        dt_path = save_array(dt.numpy(), output_dir, f'dt_batch_{i}')

        response['batch_results'].append({
            'distance_transform': {
                'path': dt_path,
                'shape': list(dt.shape),
                'min_corner_xyz': min_corner_xyz,
                'scale_factor': scale,
                'crop_size': crop_size
            }
        })

    t3 = time.time()
    print(f"[EDT_BATCH] Zarr saving took {t3-t2:.3f}s, total {t3-t0:.3f}s", flush=True)

    return response


def process_request(request, inference, edt_inference, volume_scale, model_type, output_dir):
    """Process a single inference request and return results."""
    global _debug_written

    request_type = request.get('request_type', model_type)

    # Handle batch EDT requests
    if request_type == 'edt_batch':
        return process_edt_batch_request(request, edt_inference, volume_scale, output_dir)

    if 'center_xyz' not in request:
        return {'error': 'Missing required field: center_xyz'}
    center_xyz = request['center_xyz']
    output_types = request.get('output_types',
                               ['distance_transform'] if request_type == 'edt' else ['uv_candidates'])

    def xyz_to_scaled_zyx(xyzs):
        for xyz in xyzs:
            if xyz is None:
                continue
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError(f'Coordinate must be a list of 3 numbers, got {xyz}')
        return [
            torch.tensor(xyz).flip(0) / (2 ** volume_scale) if xyz is not None else None
            for xyz in xyzs
        ]

    def zyxs_to_scaled_xyzs(zyxss):
        return [(zyxs.flip(1) * (2 ** volume_scale)).tolist() for zyxs in zyxss]

    response = {'center_xyz': center_xyz}

    with torch.inference_mode():
        if request_type == 'edt':
            # Convert single request to batch format and route through batch handler
            batch_request = {
                'request_type': 'edt_batch',
                'center_xyz_batch': [request['center_xyz']],
            }
            if 'conditioning_points_xyz' in request:
                batch_request['conditioning_points_xyz_batch'] = [request['conditioning_points_xyz']]
            elif 'conditioning_mask_path' in request:
                batch_request['conditioning_mask_paths'] = [request['conditioning_mask_path']]

            batch_response = process_edt_batch_request(batch_request, edt_inference, volume_scale, output_dir)

            # Convert batch response back to single format
            if 'error' in batch_response:
                return batch_response
            result = batch_response['batch_results'][0]
            if 'error' in result:
                return result
            return {
                'center_xyz': request['center_xyz'],
                'distance_transform': result['distance_transform']
            }
        else:
            if inference is None:
                return {'error': 'UV model not loaded. Load a UV checkpoint to use request_type=uv'}

            prev_u_xyz = request['prev_u_xyz']
            prev_v_xyz = request['prev_v_xyz']
            prev_diag_xyz = request['prev_diag_xyz']

            center_zyx = xyz_to_scaled_zyx(center_xyz)
            prev_u = xyz_to_scaled_zyx(prev_u_xyz)
            prev_v = xyz_to_scaled_zyx(prev_v_xyz)
            prev_diag = xyz_to_scaled_zyx(prev_diag_xyz)

            if not _debug_written:
                heatmaps, min_corner_zyxs, debug_data = inference.get_heatmaps_at(
                    center_zyx, prev_u, prev_v, prev_diag, return_debug=True
                )
                write_debug_data("uv",
                    model_output=heatmaps,
                    metadata={
                        "center_xyz": center_xyz,
                        "prev_u_xyz": prev_u_xyz,
                        "prev_v_xyz": prev_v_xyz,
                        "prev_diag_xyz": prev_diag_xyz,
                        "crop_size": inference.config['crop_size'],
                        "min_corner_zyxs": min_corner_zyxs.tolist() if hasattr(min_corner_zyxs, 'tolist') else [m.tolist() for m in min_corner_zyxs],
                        "volume_scale": volume_scale,
                    },
                    **debug_data
                )
                _debug_written = True
            else:
                heatmaps, min_corner_zyxs = inference.get_heatmaps_at(center_zyx, prev_u, prev_v, prev_diag)

            u_coordinates = [inference.get_blob_coordinates(heatmap[0, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]
            v_coordinates = [inference.get_blob_coordinates(heatmap[1, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]

            response['u_candidates'] = zyxs_to_scaled_xyzs(u_coordinates)
            response['v_candidates'] = zyxs_to_scaled_xyzs(v_coordinates)

    return response


if __name__ == '__main__':
    serve()
