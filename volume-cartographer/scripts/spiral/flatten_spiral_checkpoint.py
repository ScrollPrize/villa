#!/usr/bin/env python3
"""Export and Lasagna-flatten a fitted Spiral checkpoint.

The normal invocation needs only the checkpoint and final TIFXYZ directory:

    python flatten_spiral_checkpoint.py checkpoint_fitted.ckpt output.tifxyz

The script reconstructs a combined Spiral surface, starts a private Lasagna
fit service on an ephemeral localhost port, runs flatten_fast_nofilter.json,
publishes the resulting TIFXYZ atomically, and always stops the service.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from checkpoint_io import load_checkpoint_cpu
from sample_spiral import get_spiral_yxs
from tifxyz import save_combined_tifxyz
from transforms import SpiralAndTransform
from umbilicus import json_umbilicus_z_to_yx


MODEL_CONFIG_KEYS = (
    "num_flow_integration_steps",
    "flow_integration_solver",
    "num_flow_timesteps",
    "num_flow_stages",
    "flow_bounds_z_margin",
    "flow_bounds_radius",
    "flow_voxel_resolution",
    "flow_field_type",
    "gap_expander_logit_resolution",
    "gap_expander_num_windings",
    "gap_expander_lr_scale",
    "linear_z_resolution",
    "initial_dr_per_winding",
)


def _checkpoint_config(checkpoint: dict) -> dict:
    """Return the model config with current ``model_*`` aliases populated."""
    raw = checkpoint.get("cfg")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint has no Spiral 'cfg' dictionary")
    config = dict(raw)
    for suffix in MODEL_CONFIG_KEYS:
        current = f"model_{suffix}"
        if current not in config and suffix in config:
            config[current] = config[suffix]
    missing = [f"model_{key}" for key in MODEL_CONFIG_KEYS
               if f"model_{key}" not in config]
    if missing:
        raise ValueError(
            "checkpoint is missing model configuration: "
            + ", ".join(missing))
    return config


def _value(config: dict, name: str, default=None):
    if name in config:
        return config[name]
    current = f"output_{name}"
    return config.get(current, default)


def _resolve_umbilicus(checkpoint_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser()]
    else:
        candidates = []
        dataset_env = os.environ.get("SPIRAL_DATASET")
        if dataset_env:
            candidates.append(Path(dataset_env).expanduser() / "umbilicus.json")
        for parent in (checkpoint_path.parent, *checkpoint_path.parents):
            candidates.append(parent / "umbilicus.json")
        # Common local layouts used by the legacy CLI and current s1 runs.
        candidates.extend([
            Path.home() / "Desktop/spiral_dataset/to_hf/umbilicus.json",
            Path("/ephemeral/paul/spiral/dataset/umbilicus.json"),
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate umbilicus.json; pass --umbilicus PATH or set "
        "SPIRAL_DATASET to the dataset root")


def _resolve_lasagna(explicit: Path | None) -> tuple[Path, Path]:
    candidates = [explicit.expanduser()] if explicit is not None else []
    configured = os.environ.get("LASAGNA_SERVICE_PATH")
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.append(
            configured_path.parent
            if configured_path.name == "fit_service.py"
            else configured_path)
    here = Path(__file__).resolve()
    candidates.extend([
        here.parents[3] / "lasagna",
        Path.home() / "villa/lasagna",
    ])
    for root in candidates:
        service = root / "fit_service.py"
        config = root / "configs/flatten_fast_nofilter.json"
        if service.is_file() and config.is_file():
            return service.resolve(), config.resolve()
    raise FileNotFoundError(
        "could not find Lasagna's fit_service.py and "
        "configs/flatten_fast_nofilter.json; pass --lasagna-dir PATH")


def _build_model(
    checkpoint: dict,
    config: dict,
    umbilicus_path: Path,
    device: torch.device,
) -> SpiralAndTransform:
    try:
        z_begin = int(checkpoint["z_begin"])
        z_end = int(checkpoint["z_end"])
        state = checkpoint["spiral_and_transform"]
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing {exc.args[0]!r}") from exc
    if z_begin >= z_end or not isinstance(state, dict):
        raise ValueError("checkpoint has an invalid model z range or state")

    z_values = np.arange(z_begin, z_end)
    centre = json_umbilicus_z_to_yx(
        str(umbilicus_path), coordinate_scale=1.0)
    umbilicus_zyx = torch.from_numpy(np.concatenate([
        z_values[:, None], centre(z_values),
    ], axis=-1).astype(np.float32)).to(device)
    radius = int(config["model_flow_bounds_radius"])
    margin = int(config["model_flow_bounds_z_margin"])
    flow_min = torch.tensor(
        [z_begin - margin, -radius, -radius],
        dtype=torch.int64, device=device)
    flow_max = torch.tensor(
        [z_end + margin, radius, radius],
        dtype=torch.int64, device=device)
    model = SpiralAndTransform(
        flow_integration_steps=int(
            config["model_num_flow_integration_steps"]),
        flow_integration_solver=str(
            config["model_flow_integration_solver"]),
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        umbilicus_zyx=umbilicus_zyx,
        config=config,
        spiral_outward_sense=str(
            checkpoint.get("spiral_outward_sense") or "CW"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def _export_source_surface(
    checkpoint: dict,
    config: dict,
    umbilicus_path: Path,
    destination: Path,
    *,
    device: torch.device,
    voxel_size_um: float,
    chunk_size: int,
) -> Path:
    model = _build_model(checkpoint, config, umbilicus_path, device)
    transform = model.get_slice_to_spiral_transform()
    dr_per_winding = model.get_dr_per_winding()
    z_begin = int(checkpoint["z_begin"])
    z_end = int(checkpoint["z_end"])
    step = int(_value(config, "step_size", 20))
    if step <= 0:
        raise ValueError("checkpoint output_step_size must be positive")
    first = int(checkpoint.get(
        "preview_first_winding",
        _value(config, "first_winding", 10)))
    last_raw = config.get("shell_outer_winding_idx")
    if last_raw is None:
        last_raw = int(config["model_gap_expander_num_windings"]) - 1
    last = int(last_raw)
    if last < first:
        raise ValueError(
            f"invalid output winding range [{first}, {last}]")

    print(
        f"[spiral] reconstructing windings {first}..{last}, "
        f"z [{z_begin}, {z_end}), step {step}", flush=True)
    yxs_by_winding = get_spiral_yxs(
        last + 1, dr_per_winding, step,
        group_by_winding=True, device=str(device))
    margin = int(config["model_flow_bounds_z_margin"])
    z0 = z_begin - margin
    z_values = torch.arange(
        z0, z_end + margin, step, dtype=torch.float32, device=device)
    winding_grids: dict[int, np.ndarray] = {}
    for winding in range(first, last + 1):
        yxs = yxs_by_winding[winding]
        spiral = torch.cat([
            z_values[:, None, None].expand(-1, yxs.shape[0], 1),
            yxs[None, :, :].expand(z_values.shape[0], -1, 2),
        ], dim=-1)
        flat = spiral.reshape(-1, 3)
        pieces = [
            transform.inv(flat[start:start + chunk_size]).cpu()
            for start in range(0, flat.shape[0], chunk_size)
        ]
        scroll = torch.cat(pieces).reshape_as(spiral)
        outside = (
            (scroll[..., 0] < z_begin) | (scroll[..., 0] >= z_end))
        scroll[outside] = -1.0
        winding_grids[winding] = scroll.numpy().astype(
            np.float32, copy=False)
        print(
            f"[spiral] reconstructed winding {winding} "
            f"({winding - first + 1}/{last - first + 1})",
            flush=True)

    surface_id = "spiral-checkpoint.tifxyz"
    manifest = save_combined_tifxyz(
        winding_grids,
        destination,
        surface_id,
        step,
        voxel_size_um,
        source="flatten_spiral_checkpoint.py",
        first_winding=first,
        cleanup_erosion_cells=3,
    )
    del transform, model, dr_per_winding, winding_grids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return Path(manifest["surface_path"]).resolve()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return f"md5:{digest.hexdigest()}"


def _store_surface(surface: Path, object_store: Path) -> dict:
    lines = []
    for path in sorted(item for item in surface.rglob("*") if item.is_file()):
        lines.append(
            f"{path.relative_to(surface).as_posix()}\t{_md5_file(path)}\n")
    manifest_hash = hashlib.md5(
        "".join(lines).encode(), usedforsecurity=False).hexdigest()
    ref = {
        "type": "tifxyz_segment",
        "name": surface.name,
        "hash": f"md5:{manifest_hash}",
    }
    destination = (
        object_store / ref["type"] / manifest_hash
        / quote(ref["name"], safe=""))
    destination.mkdir(parents=True)
    (destination / "segment").symlink_to(surface, target_is_directory=True)
    (destination / "object.json").write_text(
        json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return ref


def _service_json(port: int, path: str, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="GET" if data is None else "POST",
        headers={
            "X-Fit-Service-API-Version": "2",
            "Content-Type": "application/json",
            "X-VC3D-Source": "Spiral checkpoint flattener",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode())
        except Exception:
            payload = {}
        raise RuntimeError(
            str(payload.get("error") or f"Lasagna HTTP {exc.code}")) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _stop_service(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()


def _flatten(
    surface: Path,
    output: Path,
    service_path: Path,
    flatten_config_path: Path,
    work: Path,
    *,
    voxel_size_um: float,
) -> None:
    object_store = work / "objects"
    surface_ref = _store_surface(surface, object_store)
    config = json.loads(flatten_config_path.read_text(encoding="utf-8"))
    config["external_surfaces"] = [surface_ref]
    config["voxel_size_um"] = float(voxel_size_um)
    results = work / "results"
    results.mkdir()
    model_output = work / "flatten-model.pt"
    service = None
    ready = threading.Event()
    port_holder: dict[str, int] = {}
    try:
        service = subprocess.Popen(
            [
                sys.executable, str(service_path),
                "--port", "0",
                "--allow-no-data-dir",
                "--object-store-dir", str(object_store),
            ],
            cwd=str(service_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )

        def relay_output():
            pattern = re.compile(r"listening on http://[^:]+:(\d+)")
            assert service is not None and service.stdout is not None
            for line in service.stdout:
                text = line.rstrip()
                if text:
                    print(f"[lasagna] {text}", flush=True)
                match = pattern.search(text)
                if match:
                    port_holder["port"] = int(match.group(1))
                    ready.set()

        relay = threading.Thread(target=relay_output, daemon=True)
        relay.start()
        if not ready.wait(60):
            code = service.poll()
            raise RuntimeError(
                "Lasagna service failed to start"
                + (f" (exit {code})" if code is not None else ""))
        port = port_holder["port"]
        output_name = "flattened.tifxyz"
        request = {
            "config": config,
            "job_spec": {
                "config": config,
                "linked_surfaces": [surface_ref],
            },
            "single_segment": True,
            "config_name": flatten_config_path.name,
            "output_name": output_name,
            "output_dir": str(results),
            "model_output": str(model_output),
            "source": "flatten_spiral_checkpoint.py",
        }
        accepted = _service_json(port, "/optimize", request, timeout=60)
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise RuntimeError("Lasagna service returned no job id")
        while True:
            status = _service_json(port, f"/jobs/{job_id}", timeout=30)
            state = str(status.get("state") or "")
            stage = str(
                status.get("stage_name") or status.get("stage") or state)
            step = int(status.get("step") or 0)
            total = int(status.get("total_steps") or 0)
            suffix = f" {step}/{total}" if total else ""
            print(f"[lasagna] {stage}{suffix}", flush=True)
            if state == "finished":
                break
            if state in {"error", "cancelled"}:
                raise RuntimeError(
                    str(status.get("error") or f"Lasagna job {state}"))
            if service.poll() is not None:
                raise RuntimeError(
                    f"Lasagna service exited with code {service.returncode}")
            time.sleep(1.0)

        produced = results / output_name
        if not (produced / "meta.json").is_file():
            raise RuntimeError(
                "Lasagna reported success but produced no TIFXYZ")
        os.replace(produced, output)
    finally:
        _stop_service(service)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a fitted Spiral checkpoint and flatten it through a "
            "temporary Lasagna service."))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--umbilicus", type=Path,
        help="umbilicus.json (auto-detected when omitted)")
    parser.add_argument(
        "--lasagna-dir", type=Path,
        help="Lasagna repository containing fit_service.py (auto-detected)")
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device used to reconstruct the Spiral surface (default: cuda)")
    parser.add_argument(
        "--voxel-size-um", type=float, default=9.6,
        help="source voxel size written to TIFXYZ metadata (default: 9.6)")
    parser.add_argument(
        "--chunk-size", type=int, default=65536,
        help="points transformed per reconstruction batch (default: 65536)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.voxel_size_um <= 0:
        raise ValueError("--voxel-size-um must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu if intentional")

    umbilicus = _resolve_umbilicus(checkpoint_path, args.umbilicus)
    service, flatten_config = _resolve_lasagna(args.lasagna_dir)
    print(f"[spiral] loading {checkpoint_path}", flush=True)
    checkpoint = load_checkpoint_cpu(str(checkpoint_path))
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    config = _checkpoint_config(checkpoint)
    print(f"[spiral] using umbilicus {umbilicus}", flush=True)
    print(f"[lasagna] using config {flatten_config}", flush=True)

    work_path = tempfile.mkdtemp(
        prefix=f".{output.name}.flatten-", dir=output.parent)
    work = Path(work_path)
    try:
        surface = _export_source_surface(
            checkpoint,
            config,
            umbilicus,
            work / "source",
            device=device,
            voxel_size_um=float(args.voxel_size_um),
            chunk_size=int(args.chunk_size),
        )
        del checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _flatten(
            surface,
            output,
            service,
            flatten_config,
            work,
            voxel_size_um=float(args.voxel_size_um),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
