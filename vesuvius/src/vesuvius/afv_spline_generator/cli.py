"""Generate an Automated Fiber Volume (.afv) from a zone of a CT volume.

A fiber model predicts vertical and horizontal fiber probabilities over the
zone, smooth polylines are fitted to them (see ``splines``) and written in
native L0 coordinates, ready to open in VC3D.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .afv import Fiber, write_afv


DEFAULT_MODEL = "Qualzz20/afv_fiber_9um"
DEFAULT_THRESHOLD = 60.0
# Voxel sizes of the scans DEFAULT_MODEL was trained on, in micrometres.
DEFAULT_MODEL_VOXEL_SIZES = (8.64, 9.362)


class Reporter:
    """Progress as text on stderr, or as JSON lines on stdout for VC3D.

    JSON events have an ``event`` of ``progress`` (with ``stage``, ``message``
    and the completed ``fraction`` of the run), ``warning``, ``done`` or
    ``error``.
    """

    STAGES = {"read": (0.0, 0.04), "model": (0.04, 0.08), "predict": (0.08, 0.85), "splines": (0.85, 0.97), "write": (0.97, 1.0)}

    def __init__(self, json_lines: bool):
        self.json_lines = json_lines
        self.stage = None
        self.last = 0.0

    def emit(self, event: str, **fields: Any) -> None:
        if self.json_lines:
            print(json.dumps({"event": event, **fields}), flush=True)
        elif event == "progress":
            print(f"[{fields['fraction']:6.1%}] {fields['message']}", file=sys.stderr, flush=True)
        elif event == "done":
            print(f"Wrote {fields['fibers']} fibers to {fields['output']}", flush=True)
        else:
            print(f"{event}: {fields['message']}", file=sys.stderr, flush=True)

    def __call__(self, stage: str, message: str, done: int = 0, total: int = 0) -> None:
        now = time.monotonic()
        if stage == self.stage and done < total and now - self.last < 0.5:
            return
        self.stage, self.last = stage, now
        start, end = self.STAGES[stage]
        fraction = start + (end - start) * (done / total if total else 0.0)
        self.emit("progress", stage=stage, message=message, fraction=round(fraction, 4))


def open_volume(location: str, level: int):
    import zarr

    from vesuvius.data.utils import open_zarr

    store = open_zarr(location)
    if isinstance(store, zarr.Group):
        if str(level) not in store:
            raise ValueError(f"The volume has no array {level}")
        store = store[str(level)]
    elif level != 0:
        raise ValueError("--level selects an array of an OME-Zarr group")
    if store.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {store.shape}")
    return store


def zone_slices(shape_zyx: Sequence[int], origin_xyz: Sequence[int], size_xyz: Sequence[int]) -> tuple[slice, ...]:
    zone = []
    for axis, extent, start, size in zip("zyx", shape_zyx, reversed(origin_xyz), reversed(size_xyz)):
        if start < 0 or size < 1 or start + size > extent:
            raise ValueError(f"The zone [{start}, {start + size}) exceeds the volume along {axis} (size {extent})")
        zone.append(slice(start, start + size))
    return tuple(zone)


def select_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model(model: str) -> tuple[Path, dict[str, str]]:
    """An nnU-Net export folder, or a Hugging Face repository fetched into the Hugging Face cache."""
    path = Path(model).expanduser()
    if path.exists():
        return path, {"model": str(path.resolve())}
    # Hugging Face repository IDs are "name" or "owner/name".
    if model.startswith((".", "/", "~")) or model.count("/") > 1 or "\\" in model or ":" in model:
        raise FileNotFoundError(f"Model folder not found: {model}")
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(repo_id=model))
    # Snapshot folders are named after the repository commit.
    return folder, {"model": model, "revision": folder.name}


def load_network(folder: Path, device) -> tuple[Any, tuple[int, int, int]]:
    from .predict import FIBER_LABELS

    # Compiling takes longer than predicting a zone; set nnUNet_compile=true to compile anyway.
    os.environ.setdefault("nnUNet_compile", "false")
    from vesuvius.utils.models.load_nnunet_model import load_model

    # The loader prints to stdout, which carries the JSON progress events.
    with contextlib.redirect_stdout(sys.stderr):
        info = load_model(str(folder), device=str(device))
    if info["dataset_json"].get("labels") != FIBER_LABELS:
        raise ValueError(f"Not a fiber model: expected the classes {FIBER_LABELS}")
    if list(info["configuration_manager"].normalization_schemes) != ["ZScoreNormalization"]:
        raise ValueError("Only z-score normalized models are supported")
    if list(info["plans_manager"].transpose_forward) != [0, 1, 2]:
        raise ValueError("Only models without axis transposition are supported")
    return info["network"], tuple(int(p) for p in info["patch_size"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesuvius.afv_spline_generator",
        description="Predict fibers in a zone of a CT volume and write them as an Automated Fiber Volume (.afv) for VC3D.",
    )
    parser.add_argument("--volume", required=True, help="Zarr or OME-Zarr volume: local path, http(s):// or s3:// URL")
    parser.add_argument("--level", type=int, default=0, help="Array of an OME-Zarr group read as the volume (default: 0)")
    parser.add_argument("--origin", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"), help="First voxel of the zone")
    parser.add_argument("--size", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"), help="Size of the zone in voxels")
    parser.add_argument("--output", type=Path, required=True, help="New .afv file; existing files are never replaced")
    parser.add_argument(
        "--coordinate-space",
        required=True,
        help="VC3D coordinate identity of the volume (its vc-open-data-coordinate-space tag); VC3D opens the file only on that volume",
    )
    parser.add_argument(
        "--native-scale",
        type=int,
        default=1,
        help="Native L0 voxels per voxel of the volume, 2^level of the coordinate identity (default: 1)",
    )
    parser.add_argument("--voxel-size", type=float, help="Native voxel size in micrometres, for lengths in millimetres")
    parser.add_argument("--source-path", help="Native source volume recorded with the fibers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"nnU-Net fiber model: folder or Hugging Face repository (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable test-time mirroring: about 8x faster, with visibly worse predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum fiber probability in percent (default: {DEFAULT_THRESHOLD:g})",
    )
    parser.add_argument("--device", default="auto", help="auto (default), cuda, cuda:N, mps or cpu")
    parser.add_argument("--progress", choices=("text", "json"), default="text", help="Progress as text on stderr, or JSON lines on stdout")
    return parser


def generate(args: argparse.Namespace, report: Reporter) -> dict[str, Any]:
    import torch

    from .predict import predict_fibers
    from .splines import extract_splines, threshold_to_u8

    output = args.output.expanduser().absolute()
    if output.suffix.lower() != ".afv":
        raise ValueError("The output must be a .afv file")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"The folder {output.parent} does not exist")
    if args.native_scale < 1:
        raise ValueError("--native-scale must be at least 1")
    threshold = threshold_to_u8(args.threshold)
    if args.voxel_size is not None and not args.voxel_size > 0:
        raise ValueError("--voxel-size must be positive")
    if args.model == DEFAULT_MODEL and args.voxel_size:
        effective = args.voxel_size * args.native_scale
        if not 8.0 <= effective <= 10.0:
            report.emit(
                "warning",
                message=f"This model was trained on {' and '.join(map(str, DEFAULT_MODEL_VOXEL_SIZES))} µm scans; this volume has {effective:g} µm voxels.",
            )

    report("read", "Opening the volume")
    volume = open_volume(args.volume, args.level)
    zone = zone_slices(volume.shape, args.origin, args.size)
    report("read", "Reading {} × {} × {} voxels".format(*args.size))
    ct = np.asarray(volume[zone])

    device = select_device(args.device)
    report("model", f"Loading {args.model} on {device}")
    if device.type == "cpu":
        report.emit("warning", message="No GPU found: predicting on the CPU is very slow.")
    folder, provenance = resolve_model(args.model)
    network, patch = load_network(folder, device)
    mirror = not args.no_mirror
    probabilities = predict_fibers(
        network,
        ct,
        patch,
        device=device,
        mirror=mirror,
        progress=lambda done, total: report("predict", f"Predicting fibers · window {done} of {total}", done, total),
    )
    del ct, network
    if device.type == "cuda":
        torch.cuda.empty_cache()

    names = {"V": ("vertical", 0), "H": ("horizontal", 1)}

    def spline_progress(family: str, done: int, total: int) -> None:
        name, index = names[family]
        report("splines", f"Fitting {name} fibers · {done} of {total} pieces", index * total + done, 2 * total)

    traces = extract_splines(probabilities, threshold, spline_progress)
    del probabilities

    origin = np.asarray(args.origin, dtype=np.float64)
    voxel_mm = args.voxel_size / 1000 if args.voxel_size else None

    def fibers() -> Iterator[Fiber]:
        for index, (family, line) in enumerate(traces, 1):
            points = (line[:, ::-1] + origin) * args.native_scale
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            annotation = {
                "type": "vc3d_fiber",
                "version": 1,
                "sequence": index,
                "hv_classification": {"manual_tag": family},
                "control_points": [points[0].tolist(), points[-1].tolist()],
                "length_voxels": length,
            }
            if voxel_mm:
                annotation["length_mm"] = length * voxel_mm
            yield Fiber(name=f"fiber_{index}", family=family, points=points, annotation=annotation)

    frame = {
        "vc_open_data_coordinate_space": args.coordinate_space,
        "vc_open_data_source_coordinate_level": 0,
        "vc_open_data_source_coordinate_scale_factor": 1,
    }
    root = {"type": "vc3d_fiber_collection", "version": 1, "scale": 1, **frame, "coordinate_order": "XYZ", "coordinate_units": "voxel"}
    if args.source_path:
        root["vc_open_data_source_path"] = args.source_path
    if args.voxel_size:
        root["vc_open_data_source_original_resolution"] = args.voxel_size
        root["voxel_size_mm"] = voxel_mm
    if args.native_scale == 1:
        # Only known exactly when the volume is the native volume.
        root["coordinate_base_shape_zyx"] = list(volume.shape)
    generator = {
        "name": "vesuvius.afv_spline_generator",
        **provenance,
        "mirror": mirror,
        "threshold_percent": args.threshold,
        "volume": args.volume,
        "level": args.level,
        "origin_xyz": list(args.origin),
        "size_xyz": list(args.size),
        "native_scale": args.native_scale,
    }
    report("write", f"Writing {len(traces)} fibers")
    written = write_afv(output, fibers(), frame=frame, root=root, metadata={"generator": generator})
    return {"output": str(output), "fibers": written["fiber_count"], "points": written["point_count"], "uuid": written["uuid"]}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = Reporter(args.progress == "json")
    # Cancelling from VC3D terminates the process: unwind so that no
    # temporary file is left behind.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        result = generate(args, report)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        report.emit("error", message=str(error) or type(error).__name__)
        return 1
    report.emit("done", fraction=1.0, **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
