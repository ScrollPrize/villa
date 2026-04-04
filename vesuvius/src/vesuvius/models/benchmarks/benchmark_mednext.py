from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


def _parse_patch_sizes(values: list[str]) -> list[tuple[int, int, int]]:
    patch_sizes = []
    for value in values:
        dims = tuple(int(v) for v in value.split(","))
        if len(dims) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got {value!r}")
        patch_sizes.append(dims)
    return patch_sizes


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _make_unet_mgr(patch_size: tuple[int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        targets={"surface": {"out_channels": 2, "activation": "none"}},
        train_patch_size=patch_size,
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name="mednext_benchmark_unet",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "features_per_stage": [32, 64, 128, 256, 320],
            "n_stages": 5,
            "n_blocks_per_stage": [1, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [1, 1, 1, 1],
            "kernel_sizes": [[3, 3, 3]] * 5,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": "ConvBlock",
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
        },
    )


def _make_mednext_mgr(patch_size: tuple[int, int, int], model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        targets={"surface": {"out_channels": 2, "activation": "none"}},
        train_patch_size=patch_size,
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name=f"mednext_benchmark_{model_id.lower()}",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "architecture_type": "mednext_v1",
            "mednext_model_id": model_id,
            "mednext_kernel_size": 3,
        },
    )


def _build_model(variant: str, patch_size: tuple[int, int, int], model_id: str, device: torch.device) -> torch.nn.Module:
    if variant == "unet":
        mgr = _make_unet_mgr(patch_size)
    elif variant == "mednext_v1":
        mgr = _make_mednext_mgr(patch_size, model_id)
    else:
        raise ValueError(f"Unknown benchmark variant {variant!r}")
    return NetworkFromConfig(mgr).to(device)


def _timed_call(fn, *, device: torch.device):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def _run_variant(
    variant: str,
    *,
    patch_size: tuple[int, int, int],
    model_id: str,
    device: torch.device,
    iterations: int,
) -> dict[str, float | str | None]:
    model = _build_model(variant, patch_size, model_id, device)
    inputs = torch.randn(1, 1, *patch_size, device=device)
    labels = torch.randint(0, 2, (1, *patch_size), device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        _, startup_ms = _timed_call(lambda: model(inputs), device=device)
    except Exception as exc:  # pragma: no cover
        return {"startup_forward_ms": None, "steady_state_forward_ms_mean": None, "steady_state_train_step_ms_mean": None, "max_memory_mb": None, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}

    forward_times: list[float] = []
    train_times: list[float] = []
    train_step_error: str | None = None

    model.train()
    for _ in range(iterations):
        def _forward_only():
            with _autocast(device):
                return model(inputs)

        _, forward_ms = _timed_call(_forward_only, device=device)
        forward_times.append(forward_ms)

        def _train_step():
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(inputs)["surface"]
                loss = F.cross_entropy(logits.float(), labels)
            loss.backward()
            optimizer.step()

        try:
            _, train_ms = _timed_call(_train_step, device=device)
            train_times.append(train_ms)
        except torch.OutOfMemoryError as exc:  # pragma: no cover
            train_step_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            break

    result = {
        "startup_forward_ms": startup_ms,
        "steady_state_forward_ms_mean": sum(forward_times) / len(forward_times) if forward_times else None,
        "steady_state_train_step_ms_mean": sum(train_times) / len(train_times) if train_times else None,
        "max_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
    }
    if train_step_error is not None:
        result["train_step_error"] = train_step_error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UNet vs MedNeXt whole-model performance")
    parser.add_argument("--patch-size", action="append", default=["128,128,128", "192,192,192"], help="3 comma-separated ints; can be passed multiple times")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mednext-model-id", default="B")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    patch_sizes = _parse_patch_sizes(args.patch_size)

    results = {}
    for patch_size in patch_sizes:
        patch_key = "x".join(str(v) for v in patch_size)
        results[patch_key] = {
            "unet": _run_variant("unet", patch_size=patch_size, model_id=args.mednext_model_id, device=device, iterations=args.iterations),
            "mednext_v1": _run_variant("mednext_v1", patch_size=patch_size, model_id=args.mednext_model_id, device=device, iterations=args.iterations),
        }

    payload = {
        "device": str(device),
        "mednext_model_id": args.mednext_model_id,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
