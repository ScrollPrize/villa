import json
import os
import random
import socket
import threading
from pathlib import Path

import click
import numpy as np
import torch

from vesuvius.neural_tracing.infer import Inference
from vesuvius.neural_tracing.models import load_checkpoint

HEATMAP_REQUEST_TYPE = "heatmap_next_points"
DENSE_REQUEST_TYPE = "dense_displacement_grow"


@click.command()
@click.option(
    "--checkpoint_path",
    type=str,
    required=False,
    default=None,
    help=(
        "Optional heatmap checkpoint path or checkpoint sentinel. "
        "Dense requests do not require this."
    ),
)
@click.option(
    "--volume_zarr",
    type=click.Path(exists=True),
    required=True,
    help="Path to ome-zarr folder (used as default volume_path for dense requests).",
)
@click.option("--volume_scale", type=int, required=True, help="OME scale to use")
@click.option("--socket_path", type=click.Path(), required=True, help="Path to Unix domain socket")
@click.option("--no-cache", is_flag=True, help="Disable crop cache")
def serve(checkpoint_path, volume_zarr, volume_scale, socket_path, no_cache):
    state = {
        "checkpoint_path": checkpoint_path,
        "volume_zarr": volume_zarr,
        "volume_scale": int(volume_scale),
        "no_cache": bool(no_cache),
        "inference": None,
        "inference_lock": threading.Lock(),
        "dense_lock": threading.Lock(),
    }

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
                handle_connection(conn, lambda request: process_request(request, state))
            except Exception as exc:
                print(f"error handling connection: {exc}")
            finally:
                conn.close()
    finally:
        sock.close()
        if socket_path.exists():
            os.unlink(socket_path)


def _ensure_heatmap_inference(state):
    inference = state.get("inference")
    if inference is not None:
        return inference

    with state["inference_lock"]:
        inference = state.get("inference")
        if inference is not None:
            return inference

        checkpoint_path = state.get("checkpoint_path")
        if not checkpoint_path:
            raise RuntimeError(
                "Heatmap request received but no --checkpoint_path was provided when starting trace_service."
            )

        model, config = load_checkpoint(checkpoint_path)
        if state.get("no_cache"):
            config["use_crop_cache"] = False

        seed = int(config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inference = Inference(model, config, state["volume_zarr"], state["volume_scale"])
        state["inference"] = inference
        return inference


def _normalize_key(key):
    return str(key).replace("-", "_")


def _merge_dense_args(dst, src):
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        dst[_normalize_key(key)] = value


def _build_dense_args(request, state):
    from vesuvius.neural_tracing.inference.infer_global_extrap import normalize_dense_args

    dense_args = {}

    _merge_dense_args(dense_args, request.get("dense_args"))
    _merge_dense_args(dense_args, request.get("args"))
    _merge_dense_args(dense_args, request.get("overrides"))
    _merge_dense_args(dense_args, request)

    dense_args = normalize_dense_args(dense_args)
    dense_args.setdefault("iterations", 1)
    dense_args.setdefault("volume_path", state.get("volume_zarr"))
    dense_args.setdefault("volume_scale", state.get("volume_scale"))

    if dense_args.get("tifxyz_path") is None:
        return None, "Missing required field for dense displacement: tifxyz_path"
    if dense_args.get("grow_direction") is None:
        return None, "Missing required field for dense displacement: grow_direction"
    if dense_args.get("volume_path") is None:
        return None, "Missing required field for dense displacement: volume_path"
    return dense_args, None


def _process_dense_request(request, state):
    dense_args, err = _build_dense_args(request, state)
    if err is not None:
        return {"error": err}

    from vesuvius.neural_tracing.inference.infer_global_extrap import run_global_extrap

    try:
        with state["dense_lock"]:
            output_path = run_global_extrap(dense_args)
    except Exception as exc:
        return {"error": f"Dense displacement failed: {exc}"}

    return {
        "ok": True,
        "output_tifxyz_path": output_path,
        "grow_direction": str(dense_args.get("grow_direction")),
        "iterations": int(dense_args.get("iterations", 1)),
        "resolved": {
            "volume_path": dense_args.get("volume_path"),
            "volume_scale": int(dense_args.get("volume_scale")),
            "checkpoint_path": dense_args.get("checkpoint_path"),
            "config_path": dense_args.get("config_path"),
            "tifxyz_path": dense_args.get("tifxyz_path"),
            "tifxyz_out_dir": dense_args.get("tifxyz_out_dir"),
            "edge_input_rowscols": dense_args.get("edge_input_rowscols"),
        },
    }


def _process_heatmap_request(request, state):
    if "center_xyz" not in request:
        return {"error": "Missing required field: center_xyz"}

    center_xyz = request["center_xyz"]
    prev_u_xyz = request["prev_u_xyz"]
    prev_v_xyz = request["prev_v_xyz"]
    prev_diag_xyz = request["prev_diag_xyz"]

    print(
        "handling request with batch size = "
        f"{len(center_xyz)}, center_xyz = {center_xyz}, prev_u_xyz = {prev_u_xyz}, "
        f"prev_v_xyz = {prev_v_xyz}, prev_diag_xyz = {prev_diag_xyz}"
    )

    inference = _ensure_heatmap_inference(state)
    volume_scale = int(state["volume_scale"])

    def xyz_to_scaled_zyx(xyzs):
        for xyz in xyzs:
            if xyz is None:
                continue
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError(f"Coordinate must be a list of 3 numbers, got {xyz}")
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

        u_coordinates = [
            inference.get_blob_coordinates(heatmap[0, 0], min_corner_zyx)
            for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)
        ]
        v_coordinates = [
            inference.get_blob_coordinates(heatmap[1, 0], min_corner_zyx)
            for heatmap, min_corner_zyx in zip(heatmaps, min_corner_zyxs)
        ]

        response = {
            "center_xyz": center_xyz,
            "u_candidates": zyxs_to_scaled_xyzs(u_coordinates),
            "v_candidates": zyxs_to_scaled_xyzs(v_coordinates),
        }

    print(f"response: {response}")
    return response


def handle_connection(conn, request_fn):
    """Handle a single connection, processing JSON line requests."""

    with conn:
        fp = conn.makefile("rwb")
        for line in fp:
            line = line.strip()
            try:
                request = json.loads(line)
                response = request_fn(request)
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except json.JSONDecodeError as exc:
                print("error: invalid json:", exc)
                error_response = {"error": f"Invalid JSON: {exc}"}
                conn.sendall((json.dumps(error_response) + "\n").encode("utf-8"))
            except Exception as exc:
                print("error:", exc)
                error_response = {"error": f"Processing error: {exc}"}
                conn.sendall((json.dumps(error_response) + "\n").encode("utf-8"))


def process_request(request, state):
    """Process a single inference request and return results."""

    request_type = request.get("request_type", HEATMAP_REQUEST_TYPE)
    if request_type == DENSE_REQUEST_TYPE:
        return _process_dense_request(request, state)
    if request_type == HEATMAP_REQUEST_TYPE:
        return _process_heatmap_request(request, state)
    return {
        "error": (
            f"Unknown request_type '{request_type}'. "
            f"Supported: '{HEATMAP_REQUEST_TYPE}', '{DENSE_REQUEST_TYPE}'."
        )
    }


if __name__ == "__main__":
    serve()
