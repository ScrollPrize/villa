import json
import socket
import os
import click
import torch
import numpy as np
import random
from pathlib import Path

from vesuvius.neural_tracing.models import make_model, load_checkpoint
from vesuvius.neural_tracing.infer import Inference
from vesuvius.neural_tracing.heatmap_utils import expected_heatmap_centroid


@click.command()
@click.option('--checkpoint_path', type=click.Path(exists=True), required=True, help='Path to checkpoint file')
@click.option('--volume_zarr', type=click.Path(exists=True), required=True, help='Path to ome-zarr folder')
@click.option('--volume_scale', type=int, required=True, help='OME scale to use')
@click.option('--socket_path', type=click.Path(), required=True, help='Path to Unix domain socket')
@click.option('--no-cache', is_flag=True, help='Disable crop cache')

def serve(checkpoint_path, volume_zarr, volume_scale, socket_path, no_cache):

    model, config = load_checkpoint(checkpoint_path)

    if no_cache:
        config['use_crop_cache'] = False

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    inference = Inference(model, config, volume_zarr, volume_scale)

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
                handle_connection(conn, lambda request: process_request(request, inference, volume_scale))
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


def xyz_to_scaled_zyx(xyzs, volume_scale):
    return [
        xyz.flip(0) / (2 ** volume_scale) if xyz is not None else None
        for xyz in xyzs
    ]


def zyxs_to_scaled_xyzs(zyxs, volume_scale):
    return zyxs.flip(-1) * (2 ** volume_scale)


def process_request(request, inference, volume_scale):
    """Process a single inference request and return results."""

    def to_tensors(xyzs):
        for xyz in xyzs:
            if xyz is None:
                continue
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError(f'Coordinate must be a list of 3 numbers, got {xyz}')
        return [
            torch.tensor(xyz, dtype=torch.float32) if xyz is not None else None
            for xyz in xyzs
        ]

    if 'center_xyz' not in request:
        return {'error': 'Missing required field: center_xyz'}
    center_xyz = to_tensors(request['center_xyz'])
    if any(xyz is None for xyz in center_xyz):
        return {'error': 'All elements of center_xyz must be non-null'}

    prev_u_xyz = to_tensors(request['prev_u_xyz'])
    prev_v_xyz = to_tensors(request['prev_v_xyz'])
    prev_diag_xyz = to_tensors(request['prev_diag_xyz'])

    return_jacobian = request.get('return_jacobian', False)

    print(f'handling request with batch size = {len(center_xyz)}, center_xyz = {center_xyz}, prev_u_xyz = {prev_u_xyz}, prev_v_xyz = {prev_v_xyz}, prev_diag_xyz = {prev_diag_xyz}, return_jacobian = {return_jacobian}')

    if not return_jacobian:
        response = process_without_grad(inference, center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz, volume_scale)
    else:
        response = process_with_grad(inference, center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz, volume_scale)

    response['center_xyz'] = torch.stack(center_xyz).tolist()
    print(f'response: {response}')

    return response


def process_without_grad(inference, center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz, volume_scale):

    center_zyx = xyz_to_scaled_zyx(center_xyz, volume_scale)
    prev_u_zyx = xyz_to_scaled_zyx(prev_u_xyz, volume_scale)
    prev_v_zyx = xyz_to_scaled_zyx(prev_v_xyz, volume_scale)
    prev_diag_zyx = xyz_to_scaled_zyx(prev_diag_xyz, volume_scale)

    with torch.inference_mode():
        heatmaps, min_corner_zyxs = inference.get_heatmaps_at(center_zyx, prev_u_zyx, prev_v_zyx, prev_diag_zyx)
        u_coordinates = [inference.get_blob_coordinates(heatmap[0, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]
        v_coordinates = [inference.get_blob_coordinates(heatmap[1, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]

    return {
        'u_candidates': [zyxs_to_scaled_xyzs(zyx, volume_scale).tolist() for zyx in u_coordinates],
        'v_candidates': [zyxs_to_scaled_xyzs(zyx, volume_scale).tolist() for zyx in v_coordinates],
    }


def process_with_grad(inference, center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz, volume_scale):

    # FIXME: nasty hack! jacobian doesn't work with compiled model here, for some reason
    if hasattr(inference.model, '_orig_mod'):
        compiled_model = inference.model
        inference.model = inference.model._orig_mod
    else:
        compiled_model = None

    def expected_coords_fn(center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz):
        [center_zyx] = xyz_to_scaled_zyx([center_xyz], volume_scale)
        [prev_u_zyx] = xyz_to_scaled_zyx([prev_u_xyz], volume_scale)
        [prev_v_zyx] = xyz_to_scaled_zyx([prev_v_xyz], volume_scale)
        [prev_diag_zyx] = xyz_to_scaled_zyx([prev_diag_xyz], volume_scale)
        heatmaps, min_corner_zyxs = inference.get_heatmaps_at(center_zyx, prev_u_zyx, prev_v_zyx, prev_diag_zyx)
        if heatmaps.shape[1] != 1:
            raise NotImplementedError('trace_service does not support return_jacobian for step_count > 1')
        expected_coords_zyx = expected_heatmap_centroid(
            heatmaps.squeeze(1),  # remove the singleton step dimension
            apply_sigmoid=False,
        ) + min_corner_zyxs.to(heatmaps.device)
        return zyxs_to_scaled_xyzs(expected_coords_zyx, volume_scale)

    uv_candidates = [
        expected_coords_fn(center_xyz[idx], prev_u_xyz[idx], prev_v_xyz[idx], prev_diag_xyz[idx])
        for idx in range(len(center_xyz))
    ]  # [batch], u/v, xyz
    u_candidates = [candidates[0, None, :].tolist() for candidates in uv_candidates]  # the unsqueeze here is because the result should include a step dim
    v_candidates = [candidates[1, None, :].tolist() for candidates in uv_candidates]

    def jac_expected_coords_for_idx(idx):
        if prev_u_xyz[idx] is not None:
            if prev_v_xyz[idx] is not None and prev_diag_xyz[idx] is not None:
                return torch.autograd.functional.jacobian(expected_coords_fn, (center_xyz[idx], prev_u_xyz[idx], prev_v_xyz[idx], prev_diag_xyz[idx]), vectorize=True)
            elif prev_v_xyz[idx] is not None and prev_diag_xyz[idx] is None:
                jac = torch.autograd.functional.jacobian(lambda c, pu, pv: expected_coords_fn(c, pu, pv, None), (center_xyz[idx], prev_u_xyz[idx], prev_v_xyz[idx]), vectorize=True)
                return jac[0], jac[1], None
            elif prev_v_xyz[idx] is None and prev_diag_xyz[idx] is not None:
                jac = torch.autograd.functional.jacobian(lambda c, pu, pd: expected_coords_fn(c, pu, None, pd), (center_xyz[idx], prev_u_xyz[idx], prev_diag_xyz[idx]), vectorize=True)
                return jac[0], None, jac[1]
            else:
                jac = torch.autograd.functional.jacobian(lambda c, pu: expected_coords_fn(c, pu, None, None), (center_xyz[idx], prev_u_xyz[idx]), vectorize=True)
                return jac[0], None, None
        else:
            # TODO!
            raise NotImplementedError('trace_service does not support return_jacobian with prev_u unspecified')

    uv_jacobians = [jac_expected_coords_for_idx(idx) for idx in range(len(center_xyz))]  # [batch], [wrt-center/-prev*], u/v, xyz (out), xyz (in)
    u_jacobians = [
        [jac[0].tolist() if jac is not None else None for jac in jacs]
        for jacs in uv_jacobians
    ]
    v_jacobians = [
        [jac[1].tolist() if jac is not None else None for jac in jacs]
        for jacs in uv_jacobians
    ]

    if compiled_model is not None:
        inference.model = compiled_model

    return {
        'u_candidates': u_candidates,
        'v_candidates': v_candidates,
        'u_jacobians': u_jacobians,
        'v_jacobians': v_jacobians,
    }


if __name__ == '__main__':
    serve()
