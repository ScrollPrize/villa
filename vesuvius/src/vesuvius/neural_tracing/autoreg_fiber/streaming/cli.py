"""``vesuvius.trace_fiber`` — streaming-inference CLI for the autoregressive
fiber tracer.

Example
-------
::

    uv run vesuvius.trace_fiber \\
        --config  $CONFIG \\
        --ckpt    $CKPT \\
        --prompt  $PROMPT_NPZ \\
        --volume  PHercParis4 \\
        --out     /tmp/trace_out \\
        --direction both \\
        --max-steps 500 \\
        --upload-to-wk

The CLI never accepts hardcoded checkpoint paths or tokens — every secret/
local path flows through arguments or the project's existing
``webknossos-api-token.txt`` discovery (or the ``WK_TOKEN`` env var).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_fiber.config import load_autoreg_fiber_config
from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import open_streaming_volume
from vesuvius.neural_tracing.autoreg_fiber.streaming.tracer import (
    BidirectionalResult,
    FiberTracer,
    TraceResult,
)
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader
from vesuvius.neural_tracing.autoreg_fiber.streaming.wk_io import (
    DEFAULT_WK_SERVER_URL,
    build_annotation,
    build_skeleton,
    load_prompt_npz,
    save_annotation,
    upload_annotation,
)
from vesuvius.neural_tracing.autoreg_fiber.train import (
    load_autoreg_fiber_model_from_checkpoint,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesuvius.trace_fiber",
        description="Stream-trace one fiber across a remote zarr volume with the autoregressive tracer.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the training JSON config.")
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Path to the trained checkpoint (.pth). Loaded with weights_only=False.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        type=Path,
        help="Path to a fiber-cache .npz (output of build_fiber_cache.py or streaming.prompts.from_wk_url).",
    )
    parser.add_argument(
        "--volume",
        required=True,
        help="Key into config['volumes'] selecting which remote zarr to fetch crops from.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory for trace.npz/nml/zip.")
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
        help="Tracing direction. 'both' runs forward and backward and concatenates.",
    )
    parser.add_argument("--max-steps", type=int, default=5000, help="Per-direction step budget.")
    parser.add_argument(
        "--stop-prob-threshold",
        type=float,
        default=0.5,
        help="Greedy stop trigger on the head's stop probability. Set <=0 to disable.",
    )
    parser.add_argument("--min-steps", type=int, default=8, help="Minimum steps before stop_prob can fire.")
    parser.add_argument(
        "--cache-chunks",
        type=int,
        default=32,
        help="Number of native S3 chunks to keep in the LRU.",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=2,
        help="Thread-pool size for async chunk prefetch. 0 disables prefetch.",
    )
    parser.add_argument(
        "--reanchor-margin",
        type=int,
        default=24,
        help="How close (voxels) to a face the leading point may get before re-anchoring.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Autocast dtype for streaming forward. Matches the training run's mixed_precision.",
    )
    parser.add_argument(
        "--upload-to-wk",
        action="store_true",
        help="Upload the trace as a new WK annotation in addition to writing locally.",
    )
    parser.add_argument(
        "--wk-server",
        default=DEFAULT_WK_SERVER_URL,
        help="WebKnossos server URL for uploads.",
    )
    parser.add_argument(
        "--wk-dataset",
        default=None,
        help=(
            "Target WK dataset name for uploads. Defaults to PHercParis4-69da9fa9010000c20022c400 "
            "(only meaningful with --upload-to-wk)."
        ),
    )
    parser.add_argument(
        "--wk-voxel-size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Voxel size (xyz) recorded on the uploaded skeleton. Defaults to (1,1,1).",
    )
    parser.add_argument(
        "--annotation-name",
        default="autoreg_fiber_prediction",
        help="Name for the saved/uploaded annotation.",
    )
    parser.add_argument(
        "--tree-name",
        default="autoreg_fiber_prediction",
        help="Name for the predicted skeleton tree.",
    )
    return parser


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


def _resolve_volume_spec(config: dict, volume_key: str) -> dict:
    volumes = config.get("volumes") or {}
    if not isinstance(volumes, dict) or volume_key not in volumes:
        raise SystemExit(
            f"--volume={volume_key!r} not found in config.volumes; available: {sorted(volumes)}"
        )
    spec = dict(volumes[volume_key])
    if "volume_zarr_url" not in spec:
        raise SystemExit(f"config.volumes[{volume_key!r}] is missing 'volume_zarr_url'")
    return spec


def _trace_dict(result: TraceResult | BidirectionalResult) -> dict[str, Any]:
    if isinstance(result, BidirectionalResult):
        return {
            "polyline_world_zyx": result.polyline_world_zyx,
            "forward_polyline_world_zyx": result.forward.polyline_world_zyx,
            "backward_polyline_world_zyx": result.backward.polyline_world_zyx,
            "forward_stop_reason": result.forward.stop_reason,
            "backward_stop_reason": result.backward.stop_reason,
            "forward_steps": result.forward.steps,
            "backward_steps": result.backward.steps,
            "forward_reanchors": result.forward.reanchors,
            "backward_reanchors": result.backward.reanchors,
        }
    return {
        "polyline_world_zyx": result.polyline_world_zyx,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "reanchors": result.reanchors,
    }


def _write_npz(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trace.npz"
    save_kwargs: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            save_kwargs[key] = value
    np.savez_compressed(path, **save_kwargs)
    summary_path = out_dir / "trace_summary.json"
    summary = {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    config = load_autoreg_fiber_config(args.config)
    volume_spec = _resolve_volume_spec(config, args.volume)

    model = load_autoreg_fiber_model_from_checkpoint(
        args.ckpt,
        map_location=args.device,
    ).to(args.device)
    model.eval()

    cache = open_streaming_volume(
        url=str(volume_spec["volume_zarr_url"]),
        storage_options=volume_spec.get("storage_options"),
        maxsize=int(args.cache_chunks),
        num_prefetch_workers=int(args.prefetch_workers),
    )
    reader = WindowedVolumeReader(
        cache,
        crop_size=tuple(int(v) for v in config["input_shape"]),
        reanchor_margin=int(args.reanchor_margin),
    )
    tracer = FiberTracer(
        model,
        reader,
        max_steps=int(args.max_steps),
        stop_prob_threshold=None if float(args.stop_prob_threshold) <= 0.0 else float(args.stop_prob_threshold),
        min_steps=int(args.min_steps),
        dtype=_resolve_dtype(args.dtype),
    )

    prompt = load_prompt_npz(args.prompt)
    fiber_zyx = prompt.points_zyx

    if args.direction == "forward":
        result: TraceResult | BidirectionalResult = tracer.trace_one_direction(fiber_zyx)
    elif args.direction == "backward":
        result = tracer.trace_one_direction(fiber_zyx[::-1].copy())
    else:
        result = tracer.trace_bidirectional(fiber_zyx)

    payload = _trace_dict(result)
    polyline = payload["polyline_world_zyx"]
    payload["chunk_cache_stats"] = cache.stats.as_dict()
    npz_path = _write_npz(args.out, payload)
    print(f"Wrote {npz_path}", file=sys.stderr)

    dataset_name = args.wk_dataset or "PHercParis4-69da9fa9010000c20022c400"
    skeleton = build_skeleton(
        polyline,
        dataset_name=dataset_name,
        voxel_size=tuple(float(v) for v in args.wk_voxel_size),
        tree_name=str(args.tree_name),
    )
    annotation = build_annotation(skeleton, name=str(args.annotation_name))
    saved = save_annotation(annotation, args.out, basename="trace")
    print(f"Wrote {saved['nml']}", file=sys.stderr)
    print(f"Wrote {saved['zip']}", file=sys.stderr)

    if args.upload_to_wk:
        url = upload_annotation(annotation, server_url=str(args.wk_server))
        print(f"Uploaded annotation: {url}")

    cache.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
