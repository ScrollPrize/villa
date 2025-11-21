"""
Benchmark harness for neural tracing dataset methods (e.g. HeatmapDatasetV2.make_heatmaps).

Usage examples:

    # Compare the default implementation to two alternates on 2 samples
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.make_heatmaps_alts.SeparableConv1d_fft \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.make_heatmaps_alts.Depthwise3dConv_fft \
        --method make_heatmaps --num-samples 16

    # Benchmark instance-method alternates for perturbation sampling
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.perturbation_alts.CachedPatchPoints \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.perturbation_alts.CachedPatchPointsVectorizedDistance \
        --method _get_perturbed_zyx_from_patch --num-samples 16


    # Benchmark end-to-end per-sample iterator timings (no method capture/replay)
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.dataset.HeatmapDatasetV2 \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.HeatMapDatasetv3.HeatmapDatasetV3 \
        --benchmark-iterator --num-samples 8

    # Benchmark patch-point sampling alternates
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.patch_points_alts.BBoxFilteredPatchPoints \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.patch_points_alts.PrecomputedSamplingPatchPoints \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.patch_points_alts.BBoxPrecomputedPatchPoints \
        --method _get_patch_points_in_crop --num-samples 16

    # Benchmark quad-in-crop alternates
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.quads_in_crop_alts.BBoxFilteredQuads \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.quads_in_crop_alts.CachedBBoxFilteredQuads \
        --method _get_quads_in_crop --num-samples 16

    # Benchmark patch-point caching alternates
    python -m vesuvius.neural_tracing.benchmarking.benchmark_methods \
        --config src/vesuvius/neural_tracing/config.json \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.cached_patch_points_alts.LastCallMemoizedPatchPoints \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.cached_patch_points_alts.VolumeFilteredPatchPoints \
        --dataset-class vesuvius.neural_tracing.benchmarking.datasets.cached_patch_points_alts.VolumeBBoxFilteredPatchPoints \
        --method _get_cached_patch_points --num-samples 16

The first class is treated as the reference implementation. Inputs are captured
from real dataset iterations and replayed against each implementation. Timing
stats are reported for every class; outputs are checked against the reference
unless disabled with --no-output-check.
"""

import argparse
import importlib
import inspect
import json
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# Ensure the repository root (and its src/neural_tracing directory) are on sys.path when running directly
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
NT_ROOT = SRC_ROOT / "vesuvius" / "neural_tracing"
for path in (REPO_ROOT, SRC_ROOT, NT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from vesuvius.neural_tracing.dataset import load_datasets


@dataclass
class CallRecord:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    elapsed: float
    output: Any
    torch_rng_state: Optional[torch.Tensor] = None
    python_rng_state: Optional[tuple] = None
    numpy_rng_state: Optional[tuple] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _is_classmethod(cls, name: str) -> bool:
    descriptor = inspect.getattr_static(cls, name)
    return isinstance(descriptor, classmethod)


def _is_staticmethod(cls, name: str) -> bool:
    descriptor = inspect.getattr_static(cls, name)
    return isinstance(descriptor, staticmethod)


def _snapshot_rng_state() -> Tuple[tuple, tuple, torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )


def _restore_rng_state(py_state: Optional[tuple], np_state: Optional[tuple], torch_state: Optional[torch.Tensor]) -> None:
    if py_state is not None:
        random.setstate(py_state)
    if np_state is not None:
        np.random.set_state(np_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state)


def detach_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    if isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    return obj


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def flatten_tensors(obj: Any, prefix: str = "output") -> List[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        return [(prefix, obj)]
    tensors: List[Tuple[str, torch.Tensor]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            tensors.extend(flatten_tensors(val, f"{prefix}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for idx, val in enumerate(obj):
            tensors.extend(flatten_tensors(val, f"{prefix}[{idx}]"))
    return tensors


def compare_outputs(reference: Any, candidate: Any, atol: float) -> Tuple[bool, float, List[str]]:
    ref_tensors = flatten_tensors(reference)
    cand_tensors = flatten_tensors(candidate)

    messages: List[str] = []
    max_abs_diff = 0.0

    if len(ref_tensors) != len(cand_tensors):
        messages.append(
            f"Tensor structure mismatch: ref has {len(ref_tensors)} leaves, candidate has {len(cand_tensors)}."
        )
        return False, max_abs_diff, messages

    for (r_path, r_tensor), (c_path, c_tensor) in zip(ref_tensors, cand_tensors):
        if r_path != c_path:
            messages.append(f"Path mismatch: {r_path} != {c_path}")
            continue
        if r_tensor.shape != c_tensor.shape:
            messages.append(f"Shape mismatch at {r_path}: {tuple(r_tensor.shape)} vs {tuple(c_tensor.shape)}")
            continue
        if r_tensor.numel() == 0:
            diff = 0.0
        else:
            diff = (r_tensor.to(torch.float32) - c_tensor.to(torch.float32)).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        if diff > atol:
            messages.append(f"Max abs diff {diff:.3e} exceeds tolerance at {r_path}")

    matched = len(messages) == 0
    return matched, max_abs_diff, messages


def summarize_times(times: Sequence[float]) -> str:
    if not times:
        return "no calls recorded"
    times_ms = [t * 1000 for t in times]
    return (
        f"mean {np.mean(times_ms):.2f} ms | median {np.median(times_ms):.2f} ms | "
        f"min {np.min(times_ms):.2f} ms | max {np.max(times_ms):.2f} ms"
    )


def capture_inputs(
    dataset_cls, method_name: str, config: dict, patches: Iterable[Any], num_samples: int, store_outputs: bool
) -> List[CallRecord]:
    records: List[CallRecord] = []
    dataset = dataset_cls(config, patches)

    # Patch the target method to record its invocations while iterating the dataset
    descriptor = inspect.getattr_static(dataset_cls, method_name)
    if isinstance(descriptor, classmethod):
        raw_method = descriptor.__func__

        def wrapper(cls, *args, **kwargs):
            py_state, np_state, torch_state = _snapshot_rng_state()
            start = time.perf_counter()
            result = raw_method(cls, *args, **kwargs)
            elapsed = time.perf_counter() - start
            records.append(
                CallRecord(
                    args=detach_to_cpu(args),
                    kwargs=detach_to_cpu(kwargs),
                    elapsed=elapsed,
                    output=detach_to_cpu(result) if store_outputs else None,
                    torch_rng_state=torch_state,
                    python_rng_state=py_state,
                    numpy_rng_state=np_state,
                )
            )
            return result

        patched = classmethod(wrapper)
    elif isinstance(descriptor, staticmethod):
        raw_method = descriptor.__func__

        def wrapper(*args, **kwargs):
            py_state, np_state, torch_state = _snapshot_rng_state()
            start = time.perf_counter()
            result = raw_method(*args, **kwargs)
            elapsed = time.perf_counter() - start
            records.append(
                CallRecord(
                    args=detach_to_cpu(args),
                    kwargs=detach_to_cpu(kwargs),
                    elapsed=elapsed,
                    output=detach_to_cpu(result) if store_outputs else None,
                    torch_rng_state=torch_state,
                    python_rng_state=py_state,
                    numpy_rng_state=np_state,
                )
            )
            return result

        patched = staticmethod(wrapper)
    elif inspect.isfunction(descriptor):
        raw_method = descriptor

        def wrapper(self, *args, **kwargs):
            py_state, np_state, torch_state = _snapshot_rng_state()
            start = time.perf_counter()
            result = raw_method(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            records.append(
                CallRecord(
                    args=detach_to_cpu(args),
                    kwargs=detach_to_cpu(kwargs),
                    elapsed=elapsed,
                    output=detach_to_cpu(result) if store_outputs else None,
                    torch_rng_state=torch_state,
                    python_rng_state=py_state,
                    numpy_rng_state=np_state,
                )
            )
            return result

        patched = wrapper
    else:
        raise ValueError(
            f"Method {method_name} on {dataset_cls.__name__} must be @classmethod, @staticmethod, or a plain method for profiling."
        )

    original = getattr(dataset_cls, method_name)
    setattr(dataset_cls, method_name, patched)

    try:
        iterator = iter(dataset)
        with torch.no_grad():
            for _ in range(num_samples):
                next(iterator)
    finally:
        setattr(dataset_cls, method_name, original)

    return records


def replay_method(
    dataset_cls,
    method_name: str,
    records: List[CallRecord],
    device: torch.device,
    sync_cuda: bool,
    config: dict,
    patches: Iterable[Any],
) -> Tuple[List[float], List[Any]]:
    descriptor = inspect.getattr_static(dataset_cls, method_name)
    is_class = isinstance(descriptor, classmethod)
    is_static = isinstance(descriptor, staticmethod)

    dataset_instance = None
    if is_class or is_static:
        method = getattr(dataset_cls, method_name)
    else:
        dataset_instance = dataset_cls(config, patches)
        method = getattr(dataset_instance, method_name)

    times: List[float] = []
    outputs: List[Any] = []

    with torch.no_grad():
        for record in records:
            _restore_rng_state(record.python_rng_state, record.numpy_rng_state, record.torch_rng_state)

            args = record.args
            kwargs = record.kwargs

            if is_class or is_static:
                args = move_to_device(args, device)
                kwargs = move_to_device(kwargs, device)
            start = time.perf_counter()
            result = method(*args, **kwargs)
            if device.type == "cuda" and sync_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            outputs.append(detach_to_cpu(result))

    return times, outputs


def load_class(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dataset class methods against a reference implementation.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to neural_tracing config JSON (same structure used for training).",
    )
    parser.add_argument(
        "--dataset-class",
        dest="dataset_classes",
        action="append",
        default=["vesuvius.neural_tracing.dataset.HeatmapDatasetV2"],
        help="Dotted path to dataset class. The first entry is the reference; repeat flag to compare multiple classes.",
    )
    parser.add_argument(
        "--benchmark-iterator",
        action="store_true",
        help="Benchmark end-to-end per-sample iteration time instead of a specific method (skips capture/replay).",
    )
    parser.add_argument(
        "--method",
        default="make_heatmaps",
        help="Name of the dataset method to profile (class, static, or instance).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to draw from the reference dataset while capturing method calls (or full iterations when --benchmark-iterator is set).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for replay benchmarking (e.g. cpu, cuda, cuda:0). Capture always runs on CPU.",
    )
    parser.add_argument(
        "--no-output-check",
        action="store_false",
        dest="compare_outputs",
        help="Skip output comparisons; only report timings (avoids storing large tensors).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Absolute tolerance for output comparisons.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Unused in harness; kept for parity with training scripts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for Python/NumPy/PyTorch before capture to make samples repeatable.",
    )
    parser.add_argument(
        "--sync-cuda",
        action="store_true",
        default=False,
        help="Call torch.cuda.synchronize() when timing CUDA runs (more accurate, slightly slower).",
    )
    return parser.parse_args()


def benchmark_iterator(
    dataset_classes,
    config: dict,
    patches: Iterable[Any],
    num_samples: int,
    device: torch.device,
    sync_cuda: bool,
    seed: int,
) -> None:
    for cls in dataset_classes:
        print("\n===", cls.__name__, "===")
        set_seed(seed)
        dataset = cls(config, patches)
        iterator = iter(dataset)

        times: List[float] = []
        with torch.no_grad():
            for _ in range(num_samples):
                start = time.perf_counter()
                next(iterator)
                if device.type == "cuda" and sync_cuda:
                    torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
        print(f"Iterator timings: {summarize_times(times)}")


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    # Build patch set once from config. The harness always uses train patches.
    train_patches, _ = load_datasets(config)

    dataset_classes = [load_class(name) for name in args.dataset_classes]
    if args.benchmark_iterator:
        device = torch.device(args.device)
        print(f"Iterating {args.num_samples} sample(s) per class...")
        benchmark_iterator(dataset_classes, config, train_patches, args.num_samples, device, args.sync_cuda, args.seed)
        return

    reference_cls = dataset_classes[0]

    print(f"Capturing inputs by iterating {args.num_samples} sample(s) of {reference_cls.__name__}.{args.method}...")
    records = capture_inputs(reference_cls, args.method, config, train_patches, args.num_samples, args.compare_outputs)
    call_count = len(records)
    if call_count == 0:
        print(f"No calls to {args.method} were captured; is the method used during iteration?")
        return

    print(f"Captured {call_count} calls. Reference per-call time during capture: {summarize_times([r.elapsed for r in records])}")

    # Optionally cache reference outputs from capture; otherwise recompute during replay.
    reference_outputs = [r.output for r in records] if args.compare_outputs else [None] * call_count

    device = torch.device(args.device)
    for cls in dataset_classes:
        print("\n===", cls.__name__, "===")
        times, outputs = replay_method(cls, args.method, records, device, args.sync_cuda, config, train_patches)
        print(f"Replay timings on {device}: {summarize_times(times)}")

        if not args.compare_outputs:
            continue

        all_match = True
        worst_diff = 0.0
        total_mismatches = 0
        for ref_out, cand_out in zip(reference_outputs, outputs):
            matched, max_diff, messages = compare_outputs(ref_out, cand_out, args.tolerance)
            worst_diff = max(worst_diff, max_diff)
            if not matched:
                total_mismatches += 1
                all_match = False
                print("  Output mismatch:")
                for msg in messages:
                    print(f"    - {msg}")
        if all_match:
            print(f"Outputs match reference within atol={args.tolerance} (max abs diff {worst_diff:.3e}).")
        else:
            print(
                f"Outputs differ in {total_mismatches}/{call_count} calls (max abs diff across mismatches {worst_diff:.3e})."
            )


if __name__ == "__main__":
    main()
