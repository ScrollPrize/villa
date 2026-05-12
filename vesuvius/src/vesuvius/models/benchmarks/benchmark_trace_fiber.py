"""Benchmark for the streaming autoregressive fiber tracer.

Reports
-------
* Steady-state per-step latency for the **cached** path
  (``init_kv_cache`` + ``step_from_encoded_cached``).
* Steady-state per-step latency for the **naive** path
  (``forward_from_encoded`` over a growing prefix, mirroring what the
  existing ``infer.py`` does).
* End-to-end seconds for a fixed-length trace under both paths.
* :class:`ChunkLRUCache` stats (hits, prefetch_hits, misses) over the
  trace.

The benchmark can run on either:

* A tiny CPU model + on-disk synthetic zarr (the default) — no GPU or S3.
  Useful for CI gating that the cached path stays ≥ ~3x faster than the
  naive path on the same workload.
* A user-supplied config + checkpoint via ``--config`` / ``--ckpt`` /
  ``--prompt`` / ``--volume`` (forwarded to the streaming CLI components).
  Useful for end-to-end timings on the real PHercParis4 zarr.

Writes a JSON record under ``_data/benchmarks/`` (or ``--out``) so trends
are tracked across runs. The summary is also printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from vesuvius.neural_tracing.autoreg_fiber.config import load_autoreg_fiber_config
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.infer import infer_autoreg_fiber
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel
from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import ChunkLRUCache
from vesuvius.neural_tracing.autoreg_fiber.streaming.tracer import FiberTracer
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader
from vesuvius.neural_tracing.autoreg_fiber.streaming.wk_io import load_prompt_npz
from vesuvius.neural_tracing.autoreg_fiber.train import load_autoreg_fiber_model_from_checkpoint


def _tiny_synthetic_config() -> dict:
    return {
        "dinov2_backbone": None,
        "crop_size": [16, 16, 16],
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "prompt_length": 2,
        "target_length": 16,
        "point_stride": 1,
        "decoder_dim": 24,
        "decoder_depth": 2,
        "decoder_num_heads": 2,
        "decoder_dropout": 0.0,
        "max_fiber_position_embeddings": 64,
        "coarse_prediction_mode": "axis_factorized",
        "cross_attention_every_n_blocks": 1,
        "distance_aware_coarse_targets_enabled": False,
        "position_refine_start_step": 0,
        "position_refine_weight": 0.01,
        "xyz_soft_loss_weight": 0.1,
        "segment_vector_loss_weight": 0.01,
        "optimizer": {"name": "adamw", "learning_rate": 2e-3, "weight_decay": 0.0},
        "batch_size": 1,
        "num_workers": 0,
        "val_num_workers": 0,
        "val_fraction": 0.0,
        "num_steps": 1,
        "log_frequency": 1,
        "ckpt_frequency": 50,
        "save_final_checkpoint": False,
        "out_dir": "/tmp/ar_bench",
    }


def _make_synthetic_zarr(path: Path, shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> None:
    arr = zarr.open(str(path), mode="w", shape=shape, chunks=chunks, dtype="float32")
    rng = np.random.default_rng(0)
    arr[:] = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)


def _make_synthetic_prompt(out_dir: Path) -> Path:
    points_zyx = np.stack(
        [np.linspace(8.0, 12.0, 4), np.full(4, 16.0), np.full(4, 16.0)], axis=-1
    ).astype(np.float32)
    fiber = FiberPath(
        annotation_id="bench-ann",
        tree_id="bench-tree",
        target_volume="BenchVol",
        marker="fibers_s1a",
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32, copy=False),
        points_zyx=points_zyx,
        transform_checksum="identity",
        densify_step=1.0,
    )
    return write_fiber_cache(fiber, out_dir)


def _build_synthetic_setup(tmp_dir: Path) -> tuple[AutoregFiberModel, ChunkLRUCache, WindowedVolumeReader, Path]:
    cfg = _tiny_synthetic_config()
    zarr_path = tmp_dir / "bench.zarr"
    _make_synthetic_zarr(zarr_path, shape=(64, 32, 32), chunks=(16, 16, 16))
    prompt_path = _make_synthetic_prompt(tmp_dir)
    torch.manual_seed(0)
    model = AutoregFiberModel(cfg).eval()
    src = zarr.open(str(zarr_path), mode="r")
    cache = ChunkLRUCache(src, maxsize=16, num_prefetch_workers=2)
    reader = WindowedVolumeReader(cache, crop_size=tuple(cfg["input_shape"]), reanchor_margin=2)
    return model, cache, reader, prompt_path


def _time_cached(model: AutoregFiberModel, reader: WindowedVolumeReader, prompt: np.ndarray, *, max_steps: int) -> dict[str, Any]:
    tracer = FiberTracer(
        model,
        reader,
        max_steps=max_steps,
        stop_prob_threshold=None,
        min_steps=1,
        dtype=torch.float32,
    )
    # Warmup
    tracer.trace_one_direction(prompt, prefetch=False)
    t0 = time.perf_counter()
    result = tracer.trace_one_direction(prompt, prefetch=True)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": float(elapsed),
        "steps": int(result.steps),
        "steps_per_second": float(result.steps) / max(elapsed, 1e-9),
        "reanchors": int(result.reanchors),
    }


def _time_naive(model: AutoregFiberModel, *, prompt_world_zyx: np.ndarray, max_steps: int, cache: ChunkLRUCache, reader: WindowedVolumeReader) -> dict[str, Any]:
    """Replicate the existing infer.py O(T^2) loop *without* re-anchoring.

    This isolates the per-step model cost. The "naive" path that includes
    re-anchoring is identical to the cached path in I/O cost (same chunk
    cache) but multiplies the per-step model cost by ~T, so this is the
    cleanest apples-to-apples timing.
    """

    reader.anchor_on(prompt_world_zyx[-tracer_prompt_length(model) :].mean(axis=0))
    volume = torch.from_numpy(reader.fetch_crop()).unsqueeze(0).unsqueeze(0)
    prompt_tokens = _build_prompt_tokens_for_naive(model, reader, prompt_world_zyx)
    batch = {
        "volume": volume,
        "prompt_tokens": prompt_tokens,
        "prompt_anchor_xyz": prompt_tokens["xyz"][:, -1, :],
        "prompt_anchor_valid": prompt_tokens["valid_mask"][:, -1],
    }
    # Warmup
    infer_autoreg_fiber(model, batch, max_steps=4, stop_probability_threshold=None, min_steps=1, greedy=True)
    t0 = time.perf_counter()
    result = infer_autoreg_fiber(
        model,
        batch,
        max_steps=max_steps,
        stop_probability_threshold=None,
        min_steps=1,
        greedy=True,
    )
    elapsed = time.perf_counter() - t0
    steps = int(result["predicted_fiber_local_zyx"].shape[0])
    return {
        "elapsed_s": float(elapsed),
        "steps": steps,
        "steps_per_second": steps / max(elapsed, 1e-9),
    }


def tracer_prompt_length(model: AutoregFiberModel) -> int:
    return int(model.config["prompt_length"])


def _build_prompt_tokens_for_naive(model: AutoregFiberModel, reader: WindowedVolumeReader, prompt_world_zyx: np.ndarray) -> dict[str, torch.Tensor]:
    from vesuvius.neural_tracing.autoreg_mesh.serialization import quantize_local_xyz

    p_len = tracer_prompt_length(model)
    prompt_local = prompt_world_zyx[-p_len:] - reader.min_corner.astype(np.float32)
    coarse_ids, offset_bins, valid_mask = quantize_local_xyz(
        prompt_local.astype(np.float32),
        volume_shape=tuple(model.input_shape),
        patch_size=tuple(model.patch_size),
        offset_num_bins=tuple(model.offset_num_bins),
    )
    return {
        "coarse_ids": torch.from_numpy(coarse_ids).long().unsqueeze(0),
        "offset_bins": torch.from_numpy(offset_bins).long().unsqueeze(0),
        "xyz": torch.from_numpy(prompt_local.astype(np.float32)).unsqueeze(0),
        "positions": torch.arange(p_len).long().unsqueeze(0),
        "valid_mask": torch.from_numpy(valid_mask).bool().unsqueeze(0),
        "mask": torch.ones((1, p_len), dtype=torch.bool),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark cached vs naive autoreg fiber inference.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--volume", default=None)
    parser.add_argument("--out", type=Path, default=Path("_data/benchmarks/trace_fiber.json"))
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    using_user_inputs = all(p is not None for p in (args.config, args.ckpt, args.prompt, args.volume))

    if using_user_inputs:
        config = load_autoreg_fiber_config(args.config)
        volumes = config.get("volumes") or {}
        if args.volume not in volumes:
            raise SystemExit(f"--volume {args.volume!r} not found in config; have {sorted(volumes)}")
        spec = volumes[args.volume]
        model = load_autoreg_fiber_model_from_checkpoint(args.ckpt, map_location=args.device).to(args.device)
        zarr_url = str(spec["volume_zarr_url"])
        storage_options = spec.get("storage_options") or {}
        src = zarr.open(zarr_url, mode="r", storage_options=storage_options)
        if hasattr(src, "keys") and "0" in src:
            src = src["0"]
        cache = ChunkLRUCache(src, maxsize=32, num_prefetch_workers=2)
        reader = WindowedVolumeReader(cache, crop_size=tuple(config["input_shape"]), reanchor_margin=24)
        prompt_path = args.prompt
        prompt = load_prompt_npz(prompt_path).points_zyx
    else:
        # Synthetic fallback for CI / smoke runs.
        scratch = Path("/tmp/ar_fiber_bench")
        scratch.mkdir(parents=True, exist_ok=True)
        model, cache, reader, prompt_path = _build_synthetic_setup(scratch)
        prompt = load_prompt_npz(prompt_path).points_zyx

    model.eval()

    cached = _time_cached(model, reader, prompt, max_steps=int(args.max_steps))
    # The naive timing reuses the same model / reader / cache so the per-step
    # comparison is direct. Reset the reader anchor before naive timing.
    naive = _time_naive(model, prompt_world_zyx=prompt, max_steps=int(args.max_steps), cache=cache, reader=reader)
    speedup = cached["steps_per_second"] / max(naive["steps_per_second"], 1e-9)

    report = {
        "config": str(args.config) if using_user_inputs else "synthetic",
        "ckpt": str(args.ckpt) if using_user_inputs else "synthetic",
        "volume": str(args.volume) if using_user_inputs else "synthetic",
        "max_steps": int(args.max_steps),
        "device": str(args.device),
        "cached": cached,
        "naive": naive,
        "speedup_cached_vs_naive": float(speedup),
        "chunk_cache_stats": cache.stats.as_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cached: {cached['steps_per_second']:8.2f} steps/s ({cached['steps']} steps, {cached['reanchors']} reanchors)")
    print(f"Naive:  {naive['steps_per_second']:8.2f} steps/s ({naive['steps']} steps)")
    print(f"Speedup: {speedup:.2f}x  (cached / naive)")
    print(f"Chunk cache: {cache.stats.as_dict()}")
    print(f"Wrote {args.out}")
    cache.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
