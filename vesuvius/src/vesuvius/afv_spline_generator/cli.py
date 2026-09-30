"""Generate an Automated Fiber Volume (.afv) from a zone of a CT volume.

The zone is processed in blocks. In each block a fiber model predicts vertical
and horizontal fiber probabilities and smooth polylines are fitted to them
(see ``splines``). The blocks' polylines are then stitched into long fibers
(see ``extend``) and written in native L0 coordinates, ready to open in VC3D.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .afv import Fiber, write_afv


DEFAULT_MODEL = "Qualzz20/afv_fiber_9um"
DEFAULT_THRESHOLD = 60.0
# Voxel sizes of the scans DEFAULT_MODEL was trained on, in micrometres.
DEFAULT_MODEL_VOXEL_SIZES = (8.64, 9.362)
DEFAULT_BLOCK_SIZE = 512
# Each block is predicted with this much context on every side, so the
# splines of neighbouring blocks overlap and can be joined.
BLOCK_MARGIN = 32
GAP_MODEL = Path(__file__).parent / "extend" / "gap_model.pt"


class Reporter:
    """Progress as text on stderr, or as JSON lines on stdout for VC3D.

    JSON events have an ``event`` of ``progress`` (with ``stage``, ``message``
    and the completed ``fraction`` of the run), ``plan`` (the ``blocks``, each
    an ``origin`` and ``size`` in volume voxels), ``block`` (the ``index`` of
    a block and its new ``state``), ``preview`` (the ``path`` of a .afv with
    the fibers stitched so far), ``warning``, ``done`` or ``error``.
    """

    STAGES = {"read": (0.0, 0.01), "model": (0.01, 0.04), "blocks": (0.04, 0.85), "stitch": (0.85, 0.97), "write": (0.97, 1.0)}

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
        elif event in ("warning", "error"):
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


def zone_blocks(origin_xyz: Sequence[int], size_xyz: Sequence[int], block: int) -> list[tuple[list[int], list[int]]]:
    """The zone cut into blocks of ``block`` voxels, the last ones shorter, x varying fastest."""
    if block < 1:
        raise ValueError("--block-size must be at least 1")
    axes = [[(o + start, min(block, s - start)) for start in range(0, s, block)] for o, s in zip(origin_xyz, size_xyz)]
    return [([x, y, z], [sx, sy, sz]) for z, sz in axes[2] for y, sy in axes[1] for x, sx in axes[0]]


class RegionReader:
    """Reads small regions of a volume through a cache of its storage chunks.

    Chunks leaving the memory cache are kept in ``spill`` when it is given, so
    that each chunk of a remote volume is downloaded once.
    """

    def __init__(self, volume, budget: int = 64 * 1024**2, spill: Path | None = None):
        self.volume = volume
        self.spill = spill
        self.chunks = np.asarray(volume.chunks, dtype=int)
        self.shape = np.asarray(volume.shape, dtype=int)
        self.cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self.budget = budget
        self.bytes = 0

    def _chunk(self, key: tuple[int, ...]) -> np.ndarray:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        stored = self.spill / ("_".join(map(str, key)) + ".npy") if self.spill else None
        if stored and stored.exists():
            block = np.load(stored)
        else:
            start = np.asarray(key) * self.chunks
            block = np.asarray(self.volume[tuple(slice(int(a), int(b)) for a, b in zip(start, np.minimum(start + self.chunks, self.shape)))])
            if stored:
                np.save(stored, block)
        self.cache[key] = block
        self.bytes += block.nbytes
        while self.bytes > self.budget and len(self.cache) > 1:
            self.bytes -= self.cache.popitem(last=False)[1].nbytes
        return block

    def __call__(self, low_xyz: Sequence[int], high_xyz: Sequence[int]) -> np.ndarray:
        low, high = np.asarray(low_xyz, dtype=int)[::-1], np.asarray(high_xyz, dtype=int)[::-1]
        out = np.empty(tuple(high - low), dtype=self.volume.dtype)
        for key in np.ndindex(tuple((high - 1) // self.chunks - low // self.chunks + 1)):
            key = tuple(int(k) for k in np.asarray(key) + low // self.chunks)
            start = np.asarray(key) * self.chunks
            a, b = np.maximum(low, start), np.minimum(high, start + self.chunks)
            out[tuple(slice(int(x), int(y)) for x, y in zip(a - low, b - low))] = self._chunk(key)[
                tuple(slice(int(x), int(y)) for x, y in zip(a - start, b - start))
            ]
        return out


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
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Test-time mirroring: better predictions, about 8x slower (default: off)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum fiber probability in percent (default: {DEFAULT_THRESHOLD:g})",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"The zone is processed in cubes of this many voxels per side (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument("--preview-dir", type=Path, help="Folder receiving a .afv of the fibers stitched so far after each block")
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
    zone_slices(volume.shape, args.origin, args.size)
    blocks = zone_blocks(args.origin, args.size, args.block_size)
    shape_xyz = np.asarray(volume.shape[::-1], dtype=int)
    report.emit("plan", blocks=[{"origin": origin, "size": size} for origin, size in blocks], block_size=args.block_size)
    if args.preview_dir is not None and not args.preview_dir.is_dir():
        raise FileNotFoundError(f"The folder {args.preview_dir} does not exist")

    device = select_device(args.device)
    report("model", f"Loading {args.model} on {device}")
    if device.type == "cpu":
        report.emit("warning", message="No GPU found: predicting on the CPU is very slow.")
    folder, provenance = resolve_model(args.model)
    network, patch = load_network(folder, device)
    mirror = args.mirror

    from .extend.ct_support import CTSupport
    from .extend.gap_model import Predictor, geometry_features
    from .extend.stitch import Stitcher

    gap_model = Predictor(GAP_MODEL)
    voxel_microns = args.voxel_size * args.native_scale if args.voxel_size else float(gap_model.metadata["voxelMicrons"])
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
        root["voxel_size_mm"] = args.voxel_size / 1000
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
        "block_size": args.block_size,
        "block_margin": BLOCK_MARGIN,
        "gap_model": gap_model.metadata.get("modelName", GAP_MODEL.name),
    }

    def write(path: Path, traces: list[dict[str, Any]]) -> dict[str, Any]:
        return write_afv(path, afv_fibers(traces, args.native_scale, args.voxel_size), frame=frame, root=root, metadata={"generator": generator})

    names = {"V": "vertical", "H": "horizontal"}
    count = len(blocks)
    with tempfile.TemporaryDirectory(prefix="afv-stitch-") as scratch:
        stitcher = Stitcher(scratch)
        try:
            for index, (origin, size) in enumerate(blocks):
                label = f"Block {index + 1} of {count}"

                def step(fraction: float, message: str) -> None:
                    report("blocks", f"{label} · {message}", int(1000 * (index + fraction)), 1000 * count)

                low = np.maximum(np.asarray(origin) - BLOCK_MARGIN, 0)
                high = np.minimum(np.asarray(origin) + np.asarray(size) + BLOCK_MARGIN, shape_xyz)
                report.emit("block", index=index, state="reading")
                step(0.0, "reading the CT")
                ct = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(low[::-1], high[::-1]))])
                report.emit("block", index=index, state="predicting")
                probabilities = predict_fibers(
                    network,
                    ct,
                    patch,
                    device=device,
                    mirror=mirror,
                    progress=lambda done, total: step(0.05 + 0.8 * done / total, f"predicting fibers · window {done} of {total}"),
                )
                del ct
                report.emit("block", index=index, state="splines")

                def spline_progress(family: str, done: int, total: int) -> None:
                    part = (0 if family == "V" else 1) + (done / total if total else 1)
                    step(0.85 + 0.05 * part, f"fitting {names[family]} fibers · {done} of {total} pieces")

                traces = extract_splines(probabilities, threshold, spline_progress)
                del probabilities
                report.emit("block", index=index, state="stitching")
                step(0.95, f"joining {len(traces)} fibers to the neighbouring blocks")
                stitcher.add_block(f"block-{index}", low.tolist(), (high - low).tolist(), traces,
                                   (np.asarray(origin), np.asarray(origin) + np.asarray(size)))
                report.emit("block", index=index, state="stitched")
                if args.preview_dir is not None:
                    preview = args.preview_dir.expanduser().absolute() / f"preview-{index + 1:04d}.afv"
                    chains = list(stitcher.chains())
                    write(preview, chains)
                    report.emit("preview", path=str(preview), fibers=len(chains))
            del network
            if device.type == "cuda":
                torch.cuda.empty_cache()

            (Path(scratch) / "ct").mkdir()
            ct_support = CTSupport(RegionReader(volume, spill=Path(scratch) / "ct"), shape_xyz)

            def score(proposals: list[dict[str, Any]], row: dict[str, Any]) -> np.ndarray:
                features = np.stack(
                    [geometry_features(p["a"], p["b"], p["context"], family=row["family"], voxel_microns=voxel_microns) for p in proposals]
                )
                return gap_model.score(features)

            def measure(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
                evidence = ct_support.features(a, b)
                return dict(state="available" if evidence["valid"] else "unavailable", status=evidence["status"])

            owner = {}
            for cid in stitcher.chain_ids():
                curve = stitcher.catalog.get_curve(cid, include_provenance=False)
                middle = np.asarray(curve["points"][len(curve["points"]) // 2])
                owner[cid] = next(
                    (i for i, (o, sz) in enumerate(blocks) if np.all(middle >= o) and np.all(middle < np.asarray(o) + sz)),
                    int(np.argmin([np.linalg.norm(middle - (np.asarray(o) + np.asarray(sz) / 2)) for o, sz in blocks])),
                )
            remaining = np.bincount(list(owner.values()), minlength=count)
            for index in range(count):
                report.emit("block", index=index, state="extending" if remaining[index] else "done")
            order = stitcher.chain_ids()

            def expand_progress(done: int, total: int) -> None:
                if done:
                    block = owner[order[done - 1]]
                    remaining[block] -= 1
                    if remaining[block] == 0:
                        report.emit("block", index=int(block), state="done")
                report("stitch", f"Extending fibers across gaps · {done} of {total}", done, total)

            traces = stitcher.expand(score, measure, expand_progress)
        finally:
            stitcher.close()

    report("write", f"Writing {len(traces)} fibers")
    written = write(output, traces)
    return {"output": str(output), "fibers": written["fiber_count"], "points": written["point_count"], "uuid": written["uuid"]}


def afv_fibers(traces: list[dict[str, Any]], native_scale: int, voxel_size: float | None) -> Iterator[Fiber]:
    families = {0: "V", 1: "H"}
    for index, trace in enumerate(traces, 1):
        points = np.asarray(trace["points"], dtype=np.float64) * native_scale
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        family = families[trace["family"]]
        annotation = {
            "type": "vc3d_fiber",
            "version": 1,
            "sequence": index,
            "hv_classification": {"manual_tag": family},
            "control_points": [points[0].tolist(), points[-1].tolist()],
            "length_voxels": length,
        }
        if voxel_size:
            annotation["length_mm"] = length * voxel_size / 1000
        if trace.get("gaps"):
            # Point index ranges bridging a gap between two observed pieces.
            annotation["inferred_gap_point_ranges"] = trace["gaps"]
        yield Fiber(name=f"fiber_{index}", family=family, points=points, annotation=annotation)


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
