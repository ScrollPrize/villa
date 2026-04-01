from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


def _parse_patch_size(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {text!r}")
    return values


def _make_mgr(
    *,
    patch_size: tuple[int, int, int],
    guide_checkpoint: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        targets={"ink": {"out_channels": 2, "activation": "none"}},
        train_patch_size=patch_size,
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_benchmark",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "features_per_stage": [16, 32, 64, 128, 192],
            "n_stages": 5,
            "n_blocks_per_stage": [1, 1, 1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1, 1, 1],
            "kernel_sizes": [[3, 3, 3]] * 5,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": "ConvBlock",
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
            "input_shape": list(patch_size),
            **(
                {
                    "guide_backbone": str(guide_checkpoint),
                    "guide_freeze": True,
                    "guide_tokenbook_sample_rate": 1.0,
                }
                if guide_checkpoint is not None
                else {}
            ),
        },
    )


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_model(
    model: NetworkFromConfig,
    *,
    device: torch.device,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-4,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, return_aux=True) if model.guide_enabled else model(inputs)
        if model.guide_enabled:
            logits_dict, aux = outputs
        else:
            logits_dict, aux = outputs, {}
        logits = logits_dict["ink"]
        loss = F.cross_entropy(logits, labels)
        if "guide_mask" in aux:
            guide_target = (labels > 0).float().unsqueeze(1)
            guide_target = F.interpolate(guide_target, size=aux["guide_mask"].shape[2:], mode="nearest")
            loss = loss + 0.5 * F.binary_cross_entropy(
                aux["guide_mask"].clamp(1e-6, 1.0 - 1e-6),
                guide_target,
            )
        loss.backward()
        optimizer.step()
        _sync_if_needed(device)

    forward_times = []
    train_step_times = []
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)

        start = time.perf_counter()
        outputs = model(inputs, return_aux=True) if model.guide_enabled else model(inputs)
        _sync_if_needed(device)
        forward_times.append((time.perf_counter() - start) * 1000.0)

        if model.guide_enabled:
            logits_dict, aux = outputs
        else:
            logits_dict, aux = outputs, {}
        logits = logits_dict["ink"]
        loss = F.cross_entropy(logits, labels)
        if "guide_mask" in aux:
            guide_target = (labels > 0).float().unsqueeze(1)
            guide_target = F.interpolate(guide_target, size=aux["guide_mask"].shape[2:], mode="nearest")
            loss = loss + 0.5 * F.binary_cross_entropy(
                aux["guide_mask"].clamp(1e-6, 1.0 - 1e-6),
                guide_target,
            )

        start = time.perf_counter()
        loss.backward()
        optimizer.step()
        _sync_if_needed(device)
        train_step_times.append((time.perf_counter() - start) * 1000.0)

    result = {
        "forward_ms_mean": sum(forward_times) / len(forward_times),
        "train_step_ms_mean": sum(train_step_times) / len(train_step_times),
    }
    if device.type == "cuda":
        result["max_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        result["max_memory_mb"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs guided volumetric Dinovol gating.")
    parser.add_argument(
        "--guide-checkpoint",
        type=str,
        default="/home/giorgio/Projects/dino-vesuvius/dino-checkpoints/checkpoint_step_342500.pt",
        help="Local volumetric DINO checkpoint path.",
    )
    parser.add_argument("--patch-size", type=str, default="64,64,64")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    patch_size = _parse_patch_size(args.patch_size)
    device = torch.device(args.device)
    guide_checkpoint = Path(args.guide_checkpoint)
    if not guide_checkpoint.exists():
        raise FileNotFoundError(f"Guide checkpoint not found: {guide_checkpoint}")

    baseline = NetworkFromConfig(_make_mgr(patch_size=patch_size, guide_checkpoint=None))
    guided = NetworkFromConfig(_make_mgr(patch_size=patch_size, guide_checkpoint=str(guide_checkpoint)))

    torch.manual_seed(0)
    inputs = torch.randn(1, 1, *patch_size, device=device)
    labels = torch.randint(0, 2, (1, *patch_size), device=device)

    baseline_result = _time_model(
        baseline,
        device=device,
        inputs=inputs,
        labels=labels,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    guided_result = _time_model(
        guided,
        device=device,
        inputs=inputs,
        labels=labels,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    summary = {
        "device": str(device),
        "patch_size": patch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "baseline": baseline_result,
        "guided": guided_result,
        "guided_forward_overhead_pct": (
            100.0 * (guided_result["forward_ms_mean"] - baseline_result["forward_ms_mean"])
            / baseline_result["forward_ms_mean"]
        ),
        "guided_train_step_overhead_pct": (
            100.0 * (guided_result["train_step_ms_mean"] - baseline_result["train_step_ms_mean"])
            / baseline_result["train_step_ms_mean"]
        ),
    }

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
