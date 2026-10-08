"""Isolated grouped-Conv3d benchmark for Fiber3D value augmentation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DLoader,
    _BlurKernels3D,
    _apply_blur_kernels_3d,
    _build_blur_kernels_3d,
    _directional_kernel_groups,
    load_config,
)


def _summary(milliseconds: list[float], batch_size: int) -> dict[str, float | int]:
    values = np.asarray(milliseconds, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "count": int(values.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_per_second": float(batch_size * 1000.0 / mean_ms),
    }


def _measure_cuda(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    result: torch.Tensor | None = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        measurements.append(float(start.elapsed_time(end)))
    if result is None:
        raise RuntimeError("benchmark operation did not run")
    return measurements


def _distribution_summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def _kernel_distribution(reference_path: Path, sample_count: int, batch_size: int) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    payload = reference.get("config", reference)
    if not isinstance(payload, dict):
        raise ValueError("reference must be a training config or contain a config object")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        loader = FiberTrace3DLoader(load_config(handle.name))

    isotropic_length_counts: Counter[int] = Counter()
    directional_shape_counts: Counter[tuple[int, int, int]] = Counter()
    tap_counts: list[int] = []
    dense_counts: list[int] = []
    directional_kernels = []
    directional = 0
    for sample_index in range(sample_count):
        record, _record_index, cp_index = loader.descriptor_for_sample_index(sample_index)
        params = loader._sample_augment_params(record, cp_index, sample_index)
        kernels = _build_blur_kernels_3d(params)
        isotropic_length_counts[1 if kernels.isotropic_1d is None else int(kernels.isotropic_1d.size)] += 1
        kernel = kernels.directional_zyx
        directional_kernels.append(kernel)
        if kernel is not None:
            directional += 1
            shape = tuple(int(v) for v in kernel.dense_zyx.shape)
            directional_shape_counts[shape] += 1
            tap_counts.append(int(np.count_nonzero(kernel.dense_zyx)))
            dense_counts.append(int(np.prod(shape)))

    common_shape_counts: Counter[tuple[int, int, int]] = Counter()
    actual_dense_total = 0
    padded_dense_total = 0
    nonzero_total = 0
    group_size_counts: Counter[int] = Counter()
    for start in range(0, len(directional_kernels), batch_size):
        batch_kernels = tuple(directional_kernels[start : start + batch_size])
        if len(batch_kernels) != batch_size:
            break
        for group in _directional_kernel_groups(batch_kernels):
            shapes = [
                tuple(int(v) for v in batch_kernels[index].dense_zyx.shape)
                for index in group
                if batch_kernels[index] is not None
            ]
            common = tuple(max(shape[axis] for shape in shapes) for axis in range(3))
            common_shape_counts[common] += 1
            group_size_counts[len(group)] += 1
            actual_dense_total += sum(int(np.prod(shape)) for shape in shapes)
            padded_dense_total += len(group) * int(np.prod(common))
            nonzero_total += sum(
                int(np.count_nonzero(batch_kernels[index].dense_zyx))
                for index in group
                if batch_kernels[index] is not None
            )

    def encode(counter: Counter[tuple[int, int, int]]) -> dict[str, int]:
        return {
            "x".join(str(value) for value in shape): count
            for shape, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        }

    return {
        "reference": str(reference_path.resolve()),
        "samples": sample_count,
        "batch_size": batch_size,
        "directional_samples": directional,
        "directional_fraction": directional / sample_count,
        "isotropic_length_counts": {str(key): value for key, value in sorted(isotropic_length_counts.items())},
        "directional_shape_counts": encode(directional_shape_counts),
        "nonzero_taps": _distribution_summary(tap_counts),
        "dense_bounding_voxels": _distribution_summary(dense_counts),
        "batch_common_shape_counts": encode(common_shape_counts),
        "directional_group_size_counts": {
            str(key): value for key, value in sorted(group_size_counts.items())
        },
        "batch_dense_padding_inflation": padded_dense_total / actual_dense_total,
        "batch_dense_to_nonzero_inflation": padded_dense_total / nonzero_total,
    }


def _strategy_benchmark(
    reference_path: Path,
    *,
    patch_size: int,
    batch_size: int,
    batches: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    payload = reference.get("config", reference)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        loader = FiberTrace3DLoader(load_config(handle.name))
    sampled = []
    for sample_index in range(batch_size * batches):
        record, _record_index, cp_index = loader.descriptor_for_sample_index(sample_index)
        sampled.append(
            _build_blur_kernels_3d(
                loader._sample_augment_params(record, cp_index, sample_index)
            )
        )
    kernel_batches = [
        tuple(sampled[start : start + batch_size])
        for start in range(0, len(sampled), batch_size)
    ]
    volume = torch.rand(
        (batch_size, 1, patch_size, patch_size, patch_size),
        dtype=dtype,
        device=device,
    )

    def measure(kernel_sets: list[tuple[_BlurKernels3D, ...]], ratio: float) -> dict[str, object]:
        def operation() -> torch.Tensor:
            result = volume
            for kernels in kernel_sets:
                result = _apply_blur_kernels_3d(
                    volume,
                    kernels,
                    max_directional_padding_ratio=ratio,
                )
            return result

        times = _measure_cuda(operation, warmup=warmup, iterations=iterations)
        summary = _summary(times, batch_size * batches)
        summary["mean_ms_per_batch"] = float(summary["mean_ms"]) / batches
        return summary

    isotropic_only = [
        tuple(_BlurKernels3D(item.isotropic_1d, None) for item in kernel_batch)
        for kernel_batch in kernel_batches
    ]
    return {
        "batches": batches,
        "batch_size": batch_size,
        "isotropic_only": measure(isotropic_only, 1.0),
        "exact_shape_groups": measure(kernel_batches, 1.0),
        "greedy_1_25": measure(kernel_batches, 1.25),
        "one_directional_group": measure(kernel_batches, float("inf")),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the grouped Conv3d benchmark")
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.backends.cudnn.benchmark = True
    rows: list[dict[str, object]] = []
    with torch.inference_mode(), torch.cuda.device(device):
        for kernel_size in args.kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("kernel sizes must be positive odd integers")
            radius = kernel_size // 2
            for batch_size in args.batch_sizes:
                torch.cuda.empty_cache()
                volume = torch.rand(
                    (1, batch_size, args.patch_size, args.patch_size, args.patch_size),
                    dtype=dtype,
                    device=device,
                )
                weights = torch.rand(
                    (batch_size, 1, kernel_size, kernel_size, kernel_size),
                    dtype=dtype,
                    device=device,
                )
                weights /= weights.sum(dim=(2, 3, 4), keepdim=True)
                padding = (radius, radius, radius, radius, radius, radius)
                padded = F.pad(volume, padding, mode="replicate")

                pad_times = _measure_cuda(
                    lambda: F.pad(volume, padding, mode="replicate"),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                conv_times = _measure_cuda(
                    lambda: F.conv3d(padded, weights, groups=batch_size),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                combined_times = _measure_cuda(
                    lambda: F.conv3d(
                        F.pad(volume, padding, mode="replicate"),
                        weights,
                        groups=batch_size,
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                rows.append(
                    {
                        "batch_size": batch_size,
                        "kernel_size": kernel_size,
                        "kernel_voxels": kernel_size**3,
                        "padding": _summary(pad_times, batch_size),
                        "convolution": _summary(conv_times, batch_size),
                        "combined": _summary(combined_times, batch_size),
                    }
                )
                print(
                    f"batch={batch_size:2d} kernel={kernel_size:2d} "
                    f"conv={statistics.median(conv_times):8.3f}ms "
                    f"combined={statistics.median(combined_times):8.3f}ms "
                    f"throughput={batch_size * 1000.0 / statistics.mean(combined_times):8.1f} samples/s",
                    flush=True,
                )
    result: dict[str, object] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "patch_shape_zyx": [args.patch_size] * 3,
        "batch_sizes": args.batch_sizes,
        "kernel_sizes": args.kernel_sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "rows": rows,
    }
    if args.reference is not None and args.kernel_samples > 0:
        result["kernel_distribution"] = _kernel_distribution(
            args.reference,
            args.kernel_samples,
            max(args.batch_sizes),
        )
    if args.reference is not None and args.strategy_batches > 0:
        result["actual_strategy_benchmark"] = _strategy_benchmark(
            args.reference,
            patch_size=args.patch_size,
            batch_size=max(args.batch_sizes),
            batches=args.strategy_batches,
            warmup=args.strategy_warmup,
            iterations=args.strategy_iterations,
            device=device,
            dtype=dtype,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--kernel-samples", type=int, default=0)
    parser.add_argument("--strategy-batches", type=int, default=0)
    parser.add_argument("--strategy-warmup", type=int, default=1)
    parser.add_argument("--strategy-iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result = run_benchmark(args)
    result["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
