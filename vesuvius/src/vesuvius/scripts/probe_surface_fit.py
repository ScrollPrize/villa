from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


FIXED_TARGETS = {
    "surface": {
        "out_channels": 2,
        "activation": "none",
        "ignore_label": 2,
    }
}

FIXED_MODEL_CONFIG = {
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "n_stages": 6,
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "separate_decoders": True,
}


def _flatten_tensors(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten_tensors(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_tensors(child)


def _build_manager(patch_size: tuple[int, int, int], batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        targets=FIXED_TARGETS,
        model_name="surface-fit-probe",
        train_patch_size=patch_size,
        train_batch_size=batch_size,
        in_channels=1,
        spacing=(1.0, 1.0, 1.0),
        autoconfigure=False,
        model_config=FIXED_MODEL_CONFIG,
        enable_deep_supervision=True,
        op_dims=3,
    )


def _run_single_trial(batch_size: int, patch_size: tuple[int, int, int]) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for probe_surface_fit")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = None
    optimizer = None
    inputs = None
    loss = None
    outputs = None
    try:
        mgr = _build_manager(patch_size, batch_size)
        model = NetworkFromConfig(mgr).to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
        inputs = torch.randn((batch_size, 1, *patch_size), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = None
            for tensor in _flatten_tensors(outputs):
                term = tensor.float().mean()
                loss = term if loss is None else loss + term
            if loss is None:
                raise RuntimeError("Network produced no tensors during fit probe")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        payload = {
            "success": True,
            "batch_size": int(batch_size),
            "patch_size": list(patch_size),
            "peak_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "total_memory": int(torch.cuda.get_device_properties(device).total_memory),
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except RuntimeError as exc:
        message = str(exc)
        payload = {
            "success": False,
            "batch_size": int(batch_size),
            "patch_size": list(patch_size),
            "error": message,
        }
        print(json.dumps(payload, sort_keys=True))
        if "out of memory" in message.lower():
            return 3
        return 4
    finally:
        del outputs
        del loss
        del inputs
        del optimizer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _invoke_trial(batch_size: int, patch_size: tuple[int, int, int]) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--mode",
        "single-trial",
        "--batch-size",
        str(batch_size),
        "--patch-size",
        ",".join(str(v) for v in patch_size),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = None
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise RuntimeError(
            "Trial produced no JSON payload.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    payload["returncode"] = proc.returncode
    return payload


def _candidate_edges(min_edge: int, max_edge: int, step: int) -> list[int]:
    if min_edge > max_edge:
        raise ValueError("min_edge must be <= max_edge")
    if step <= 0:
        raise ValueError("step must be > 0")
    return list(range(min_edge, max_edge + 1, step))


def _binary_search_max_success(candidates: list[int], evaluator) -> tuple[int, list[dict]]:
    attempts: list[dict] = []
    lo = 0
    hi = len(candidates) - 1
    best_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = candidates[mid]
        payload = evaluator(candidate)
        attempts.append(payload)
        if payload.get("success"):
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best_idx is None:
        raise RuntimeError("No successful fit found in the provided search range")
    return candidates[best_idx], attempts


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_result(path: Path, key: str, payload: dict) -> None:
    existing = _load_existing(path)
    existing[key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe H100 fit limits for surface training.")
    parser.add_argument("--mode", required=True, choices=["iso-bs2", "ps128-maxbs", "single-trial"])
    parser.add_argument("--min-edge", type=int, default=128)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--min-batch", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--patch-size", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "single-trial":
        if args.patch_size is None or args.batch_size is None:
            raise SystemExit("--mode single-trial requires --patch-size and --batch-size")
        patch_size = tuple(int(v) for v in args.patch_size.split(","))
        if len(patch_size) != 3:
            raise SystemExit("--patch-size must have exactly 3 comma-separated integers")
        raise SystemExit(_run_single_trial(args.batch_size, patch_size))

    if args.out is None:
        raise SystemExit("--out is required for aggregate probe modes")

    if args.mode == "iso-bs2":
        candidates = _candidate_edges(args.min_edge, args.max_edge, args.step)

        def evaluator(edge: int) -> dict:
            payload = _invoke_trial(2, (edge, edge, edge))
            payload["candidate_edge"] = edge
            return payload

        max_edge, attempts = _binary_search_max_success(candidates, evaluator)
        result = {
            "batch_size": 2,
            "max_edge": max_edge,
            "patch_size": [max_edge, max_edge, max_edge],
            "attempts": attempts,
        }
        _write_result(args.out, "iso_bs2", result)
        print(json.dumps(result, sort_keys=True))
        return

    candidates = list(range(args.min_batch, args.max_batch + 1))

    def evaluator(batch_size: int) -> dict:
        payload = _invoke_trial(batch_size, (128, 128, 128))
        payload["candidate_batch_size"] = batch_size
        return payload

    max_batch, attempts = _binary_search_max_success(candidates, evaluator)
    result = {
        "batch_size": max_batch,
        "max_batch": max_batch,
        "patch_size": [128, 128, 128],
        "attempts": attempts,
    }
    _write_result(args.out, "ps128_maxbs", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
