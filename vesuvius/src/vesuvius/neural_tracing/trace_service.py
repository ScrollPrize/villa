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

    print(f"neural-tracing service listening on {socket_path}", flush=True)

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


def process_request(request, inference, volume_scale):
    """Process a single inference request and return results."""

    if 'center_xyz' not in request:
        return {'error': 'Missing required field: center_xyz'}
    center_xyz = request['center_xyz']

    prev_u_xyz = request['prev_u_xyz']
    prev_v_xyz = request['prev_v_xyz']
    prev_diag_xyz = request['prev_diag_xyz']

    print(f'handling request with batch size = {len(center_xyz)}, center_xyz = {center_xyz}, prev_u_xyz = {prev_u_xyz}, prev_v_xyz = {prev_v_xyz}, prev_diag_xyz = {prev_diag_xyz}')

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

    with torch.inference_mode():

        center_zyx = xyz_to_scaled_zyx(center_xyz)
        prev_u = xyz_to_scaled_zyx(prev_u_xyz)
        prev_v = xyz_to_scaled_zyx(prev_v_xyz)
        prev_diag = xyz_to_scaled_zyx(prev_diag_xyz)

        heatmaps, min_corner_zyxs = inference.get_heatmaps_at(center_zyx, prev_u, prev_v, prev_diag)

        u_coordinates = [inference.get_blob_coordinates(heatmap[0, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]
        v_coordinates = [inference.get_blob_coordinates(heatmap[1, 0], min_corner_zyx) for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)]

        response = {
            'center_xyz': center_xyz,
            'u_candidates': zyxs_to_scaled_xyzs(u_coordinates),
            'v_candidates': zyxs_to_scaled_xyzs(v_coordinates),
        }

    print(f'response: {response}')

    return response


if __name__ == '__main__':
    serve()
