import json
import os
import random
import socket
import threading
import time
from pathlib import Path

import click
import numpy as np
import torch

from vesuvius.neural_tracing.infer import Inference
from vesuvius.neural_tracing.nets.models import load_checkpoint

HEATMAP_REQUEST_TYPE = "heatmap_next_points"
DENSE_REQUEST_TYPE = "dense_displacement_grow"
COPY_REQUEST_TYPE = "displacement_copy_grow"


def _print_json_log(label, payload):
    try:
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception as exc:  # pragma: no cover - defensive logging fallback
        body = f"<unserializable payload: {exc}> {payload!r}"
    print(f"[trace_service] {label}: {body}", flush=True)


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
    type=str,
    required=True,
    help="Path or URL to ome-zarr folder (used as default volume_path for dense requests).",
)
@click.option("--volume_scale", type=int, required=True, help="OME scale to use")
@click.option("--socket_path", type=click.Path(), required=True, help="Path to Unix domain socket")
@click.option(
    "--parent_pid",
    type=int,
    required=False,
    default=None,
    help="Optional parent PID watchdog. Service exits if this PID is no longer its parent.",
)
@click.option("--no-cache", is_flag=True, help="Disable crop cache")
def serve(checkpoint_path, volume_zarr, volume_scale, socket_path, parent_pid, no_cache):
    state = {
        "checkpoint_path": checkpoint_path,
        "volume_zarr": volume_zarr,
        "volume_scale": int(volume_scale),
        "no_cache": bool(no_cache),
        "inference": None,
        "inference_lock": threading.Lock(),
        "dense_lock": threading.Lock(),
    }

    if parent_pid is not None and int(parent_pid) > 1:
        parent_pid = int(parent_pid)

        def parent_watchdog():
            while True:
                if os.getppid() != parent_pid:
                    print(
                        f"parent process {parent_pid} is gone (current ppid={os.getppid()}); exiting.",
                        flush=True,
                    )
                    os._exit(0)
                time.sleep(1.0)

        threading.Thread(target=parent_watchdog, daemon=True).start()

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



def _build_copy_args(request, state):
    from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import normalize_copy_args

    copy_args = {}

    _merge_dense_args(copy_args, request.get("copy_args"))
    _merge_dense_args(copy_args, request.get("args"))
    _merge_dense_args(copy_args, request.get("overrides"))
    _merge_dense_args(copy_args, request)

    copy_args = normalize_copy_args(copy_args)
    copy_args.setdefault("volume_path", state.get("volume_zarr"))
    copy_args.setdefault("volume_scale", state.get("volume_scale"))
    copy_args.setdefault("checkpoint_path", state.get("checkpoint_path"))

    if copy_args.get("tifxyz_path") is None:
        return None, "Missing required field for displacement copy: tifxyz_path"
    if copy_args.get("volume_path") is None:
        return None, "Missing required field for displacement copy: volume_path"
    if copy_args.get("checkpoint_path") is None:
        return None, "Missing required field for displacement copy: checkpoint_path"
    return copy_args, None



def _copy_outputs_for_service(outputs):
    if not isinstance(outputs, dict):
        return None, None, "Displacement copy returned invalid outputs payload.", {}

    front_path = outputs.get("front")
    back_path = outputs.get("back")
    if isinstance(front_path, str) and front_path and isinstance(back_path, str) and back_path:
        return front_path, back_path, None, {"mode": "single"}

    outputs_by_iteration = outputs.get("outputs_by_iteration")
    if not isinstance(outputs_by_iteration, dict):
        return None, None, "Displacement copy output missing 'front'/'back' paths.", {}

    numeric_items = []
    other_items = []
    for key, value in outputs_by_iteration.items():
        try:
            numeric_items.append((int(key), str(key), value))
        except (TypeError, ValueError):
            other_items.append((str(key), value))
    numeric_items.sort(key=lambda item: item[0], reverse=True)

    iteration_items = [(key, value) for _, key, value in numeric_items]
    iteration_items.extend(other_items)

    selected = {}
    selected_iterations = {}
    for role in ("front", "back"):
        for iteration_key, iteration_payload in iteration_items:
            if not isinstance(iteration_payload, dict):
                continue
            iteration_outputs = iteration_payload.get("outputs")
            if not isinstance(iteration_outputs, dict):
                continue
            path = iteration_outputs.get(role)
            if isinstance(path, str) and path:
                selected[role] = path
                selected_iterations[role] = iteration_key
                break

    missing = [role for role in ("front", "back") if role not in selected]
    if missing:
        return (
            None,
            None,
            "Displacement copy iterative output missing latest path(s): " + ", ".join(missing),
            {
                "mode": "iterative",
                "iterations_requested": outputs.get("iterations_requested"),
                "iterations_completed": outputs.get("iterations_completed"),
                "iter_direction": outputs.get("iter_direction"),
                "selected_iterations": selected_iterations,
            },
        )

    return (
        selected["front"],
        selected["back"],
        None,
        {
            "mode": "iterative",
            "iterations_requested": outputs.get("iterations_requested"),
            "iterations_completed": outputs.get("iterations_completed"),
            "iter_direction": outputs.get("iter_direction"),
            "selected_iterations": selected_iterations,
        },
    )



def _process_copy_request(request, state):
    copy_args, err = _build_copy_args(request, state)
    if err is not None:
        _print_json_log(
            "copy request rejected",
            {
                "request_type": COPY_REQUEST_TYPE,
                "error": err,
                "request_keys": sorted(str(k) for k in request.keys())
                if isinstance(request, dict)
                else [],
            },
        )
        return {"error": err}

    _print_json_log(
        "copy request args",
        {"request_type": COPY_REQUEST_TYPE, "run_args": copy_args},
    )

    from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import run_copy_displacement

    try:
        with state["dense_lock"]:
            outputs = run_copy_displacement(copy_args)
    except Exception as exc:
        return {"error": f"Displacement copy failed: {exc}"}

    front_path, back_path, output_err, output_meta = _copy_outputs_for_service(outputs)
    if output_err is not None:
        return {"error": output_err, "copy_output": output_meta}

    return {
        "ok": True,
        "output_tifxyz_paths": {
            "front": front_path,
            "back": back_path,
        },
        "copy_output": output_meta,
        "resolved": {
            "volume_path": copy_args.get("volume_path"),
            "volume_scale": int(copy_args.get("volume_scale")),
            "checkpoint_path": copy_args.get("checkpoint_path"),
            "tifxyz_path": copy_args.get("tifxyz_path"),
            "out_dir": copy_args.get("out_dir"),
            "output_prefix": copy_args.get("output_prefix"),
        },
    }


def _build_dense_args(request, state):
    dense_args = {}
    _merge_dense_args(dense_args, request.get("dense_args"))
    _merge_dense_args(dense_args, request.get("args"))
    _merge_dense_args(dense_args, request.get("overrides"))
    _merge_dense_args(dense_args, request)

    if "iterations" in dense_args and "num_iterations" not in dense_args:
        dense_args["num_iterations"] = dense_args.get("iterations")
    if "volume_zarr" in dense_args and "volume_path" not in dense_args:
        dense_args["volume_path"] = dense_args.get("volume_zarr")
    if "dense_checkpoint_path" in dense_args and "checkpoint_path" not in dense_args:
        dense_args["checkpoint_path"] = dense_args.get("dense_checkpoint_path")

    dense_args.setdefault("volume_path", state.get("volume_zarr"))
    dense_args.setdefault("volume_scale", state.get("volume_scale"))
    dense_args.setdefault("checkpoint_path", state.get("checkpoint_path"))

    tifxyz_path = dense_args.get("tifxyz_path")
    if tifxyz_path is not None and dense_args.get("output_dir") is None:
        input_path = Path(str(tifxyz_path)).resolve()
        direction = str(dense_args.get("grow_direction", "grow"))
        suffix = f"dense_{direction}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}"
        dense_args["output_dir"] = str(input_path.parent / f"{input_path.name}_{suffix}")

    if dense_args.get("tifxyz_path") is None:
        return None, "Missing required field for dense displacement: tifxyz_path"
    if dense_args.get("volume_path") is None:
        return None, "Missing required field for dense displacement: volume_path"
    if dense_args.get("checkpoint_path") is None:
        return None, "Missing required field for dense displacement: checkpoint_path"
    if dense_args.get("output_dir") is None:
        return None, "Missing required field for dense displacement: output_dir"
    return dense_args, None


def _append_dense_cli_arg(argv, flag, value):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(v) for v in value)
        return
    argv.extend([flag, str(value)])


def _dense_args_to_argv(dense_args):
    arg_to_cli = {
        "tifxyz_path": "--tifxyz-path",
        "checkpoint_path": "--checkpoint-path",
        "volume_path": "--volume-path",
        "volume_scale": "--volume-scale",
        "volume_cache_dir": "--volume-cache-dir",
        "volume_cache_retry_seconds": "--volume-cache-retry-seconds",
        "output_dir": "--output-dir",
        "grow_direction": "--grow-direction",
        "num_iterations": "--num-iterations",
        "tifxyz_steps": "--tifxyz-steps",
        "tifxyz_voxel_step": "--tifxyz-voxel-step",
        "tifxyz_voxel_size_um": "--tifxyz-voxel-size-um",
        "batch_size": "--batch-size",
        "bbox_overlap": "--bbox-overlap",
        "device": "--device",
        "distributed_backend": "--distributed-backend",
    }
    argv = []
    for key, flag in arg_to_cli.items():
        if key in dense_args:
            _append_dense_cli_arg(argv, flag, dense_args.get(key))
    if "tta" in dense_args and bool(dense_args.get("tta")) is False:
        argv.append("--no-tta")
    if "compile_model" in dense_args and bool(dense_args.get("compile_model")) is False:
        argv.append("--no-compile")
    if "save_each_iteration" in dense_args and bool(dense_args.get("save_each_iteration")) is False:
        argv.append("--final-only")
    if "show_napari" in dense_args and bool(dense_args.get("show_napari")):
        argv.append("--show-napari")
    return argv


def _process_dense_request(request, state):
    dense_args, err = _build_dense_args(request, state)
    if err is not None:
        _print_json_log(
            "dense request rejected",
            {
                "request_type": DENSE_REQUEST_TYPE,
                "error": err,
                "request_keys": sorted(str(k) for k in request.keys())
                if isinstance(request, dict)
                else [],
            },
        )
        return {"error": err}

    _print_json_log(
        "dense request args",
        {"request_type": DENSE_REQUEST_TYPE, "run_args": dense_args},
    )

    from vesuvius.neural_tracing.inference.infer_streamline import _parse_args, run_from_args

    try:
        args = _parse_args(_dense_args_to_argv(dense_args))
        with state["dense_lock"]:
            summary = run_from_args(args)
    except SystemExit as exc:
        return {"error": f"Invalid dense displacement args: {exc}"}
    except Exception as exc:
        return {"error": f"Dense displacement failed: {exc}"}

    if not isinstance(summary, dict):
        return {"error": "Dense displacement returned invalid summary payload."}
    output_path = summary.get("final_output_tifxyz_path")
    if not isinstance(output_path, str) or not output_path:
        return {"error": "Dense displacement output missing final_output_tifxyz_path."}

    return {
        "ok": True,
        "output_tifxyz_path": output_path,
        "resolved": {
            "volume_path": dense_args.get("volume_path"),
            "volume_scale": int(dense_args.get("volume_scale")),
            "checkpoint_path": dense_args.get("checkpoint_path"),
            "tifxyz_path": dense_args.get("tifxyz_path"),
            "output_dir": dense_args.get("output_dir"),
            "grow_direction": dense_args.get("grow_direction"),
            "num_iterations": dense_args.get("num_iterations"),
            "bbox_overlap": dense_args.get("bbox_overlap"),
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
    if request_type == COPY_REQUEST_TYPE:
        return _process_copy_request(request, state)
    if request_type == DENSE_REQUEST_TYPE:
        return _process_dense_request(request, state)
    if request_type == HEATMAP_REQUEST_TYPE:
        return _process_heatmap_request(request, state)
    return {
        "error": (
            f"Unknown request_type '{request_type}'. "
            f"Supported: '{HEATMAP_REQUEST_TYPE}', '{DENSE_REQUEST_TYPE}', '{COPY_REQUEST_TYPE}'."
        )
    }


if __name__ == "__main__":
    serve()
