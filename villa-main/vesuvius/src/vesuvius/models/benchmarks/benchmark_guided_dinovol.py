from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import nullcontext
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


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    non_blocking = device.type == "cuda"
    return tensor.to(device=device, non_blocking=non_blocking)


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _make_mgr(
    *,
    patch_size: tuple[int, int, int],
    guide_checkpoint: str | None,
    guide_tokenbook_tokens: int | None = None,
    guide_fusion_stage: str | None = None,
    guide_tokenbook_prototype_weighting: str = "mean",
    guide_tokenbook_weight_mlp_hidden: int | None = None,
) -> SimpleNamespace:
    benchmark_features_per_stage = [16, 32, 64, 128, 192]
    guided_config = {}
    if guide_checkpoint is not None:
        guided_config = {
            "guide_backbone": str(guide_checkpoint),
            "guide_freeze": True,
        }
        if guide_fusion_stage is not None:
            guided_config["guide_fusion_stage"] = str(guide_fusion_stage)
        if guide_fusion_stage != "feature_skip_concat":
            guided_config["guide_tokenbook_sample_rate"] = 1.0
            guided_config["guide_tokenbook_prototype_weighting"] = str(guide_tokenbook_prototype_weighting)
            if guide_tokenbook_tokens is not None:
                guided_config["guide_tokenbook_tokens"] = int(guide_tokenbook_tokens)
            if guide_tokenbook_weight_mlp_hidden is not None:
                guided_config["guide_tokenbook_weight_mlp_hidden"] = int(guide_tokenbook_weight_mlp_hidden)
        else:
            guided_config["guide_skip_concat_projector_channels"] = [max(1, int(ch) // 4) for ch in benchmark_features_per_stage]

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
            "features_per_stage": benchmark_features_per_stage,
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
            **guided_config,
        },
    )


def _label_name(tokenbook_tokens: int | None) -> str:
    return "full" if tokenbook_tokens is None else f"k{int(tokenbook_tokens)}"


def _build_model(
    *,
    patch_size: tuple[int, int, int],
    device: torch.device,
    guide_checkpoint: str | None,
    guide_tokenbook_tokens: int | None,
    guide_fusion_stage: str | None,
    guide_tokenbook_prototype_weighting: str,
    guide_tokenbook_weight_mlp_hidden: int | None,
    compile_model: bool,
    channels_last_3d: bool,
):
    mgr = _make_mgr(
        patch_size=patch_size,
        guide_checkpoint=guide_checkpoint,
        guide_tokenbook_tokens=guide_tokenbook_tokens,
        guide_fusion_stage=guide_fusion_stage,
        guide_tokenbook_prototype_weighting=guide_tokenbook_prototype_weighting,
        guide_tokenbook_weight_mlp_hidden=guide_tokenbook_weight_mlp_hidden,
    )
    model = NetworkFromConfig(mgr).to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    if compile_model:
        model = torch.compile(model, mode="default")
    return model


def _make_cpu_batch(
    patch_size: tuple[int, int, int],
    *,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(1, 1, *patch_size)
    labels = torch.randint(0, 2, (1, *patch_size))
    if pin_memory:
        image = image.pin_memory()
        labels = labels.pin_memory()
    return image, labels


def _compute_loss(logits_dict: dict[str, torch.Tensor], aux: dict[str, torch.Tensor], labels: torch.Tensor, *, guide_fusion_stage: str | None) -> torch.Tensor:
    logits = logits_dict["ink"].float()
    loss = F.cross_entropy(logits, labels)
    if "guide_mask" in aux and guide_fusion_stage in {"input", "input_gating"}:
        guide_target = (labels > 0).float().unsqueeze(1)
        guide_target = F.interpolate(guide_target, size=aux["guide_mask"].shape[2:], mode="nearest")
        loss = loss + 0.5 * F.binary_cross_entropy(
            aux["guide_mask"].float().clamp(1e-6, 1.0 - 1e-6),
            guide_target,
        )
    return loss


def _timed_call(fn, *, device: torch.device):
    _sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    _sync_if_needed(device)
    return result, (time.perf_counter() - start) * 1000.0


def _run_variant_benchmark(
    model,
    *,
    device: torch.device,
    cpu_inputs: torch.Tensor,
    cpu_labels: torch.Tensor,
    iterations: int,
) -> dict[str, float | None]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _first_forward():
        inputs = _to_device(cpu_inputs, device)
        with _autocast_context(device):
            outputs = model(inputs)
        logits = outputs["ink"].to(dtype=torch.float16)
        _ = logits.cpu()
        return outputs

    try:
        _result, startup_ms = _timed_call(_first_forward, device=device)
    except Exception as exc:  # pragma: no cover - benchmark fallback path
        return {
            "startup_forward_ms": None,
            "steady_state_forward_ms_mean": None,
            "steady_state_train_step_ms_mean": None,
            "max_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else None
            ),
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }

    forward_times = []
    train_step_times = []
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-4)
    model.train()
    for _ in range(iterations):
        inputs = _to_device(cpu_inputs, device)
        labels = _to_device(cpu_labels, device)

        optimizer.zero_grad(set_to_none=True)
        def _forward_only():
            with _autocast_context(device):
                return model(inputs)
        _, forward_ms = _timed_call(
            _forward_only,
            device=device,
        )
        forward_times.append(forward_ms)

        def _train_step():
            with _autocast_context(device):
                outputs = model(inputs, return_aux=True) if getattr(model, "guide_enabled", False) else model(inputs)
            if getattr(model, "guide_enabled", False):
                logits_dict, aux = outputs
            else:
                logits_dict, aux = outputs, {}
            loss = _compute_loss(
                logits_dict,
                aux,
                labels,
                guide_fusion_stage=getattr(model, "guide_fusion_stage", None),
            )
            loss.backward()
            optimizer.step()

        try:
            _, train_ms = _timed_call(_train_step, device=device)
            train_step_times.append(train_ms)
        except Exception as exc:  # pragma: no cover - benchmark fallback path
            train_step_times = []
            train_step_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            break
    else:
        train_step_error = None

    model.eval()
    result = {
        "startup_forward_ms": startup_ms,
        "steady_state_forward_ms_mean": sum(forward_times) / len(forward_times),
        "steady_state_train_step_ms_mean": (
            sum(train_step_times) / len(train_step_times) if train_step_times else None
        ),
        "max_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        ),
    }
    if train_step_error is not None:
        result["train_step_error"] = train_step_error
    return result


def _profile_guided_stages(
    model: NetworkFromConfig,
    *,
    device: torch.device,
    cpu_inputs: torch.Tensor,
    iterations: int,
) -> dict[str, float]:
    if not model.guide_enabled:
        return {}

    timings = defaultdict(list)
    model.eval()

    for _ in range(iterations):
        inputs, h2d_ms = _timed_call(lambda: _to_device(cpu_inputs, device), device=device)
        timings["host_to_device_ms"].append(h2d_ms)

        def _guide_backbone():
            with torch.no_grad():
                with _autocast_context(device):
                    return model.guide_backbone(inputs)[0]

        guide_features, guide_backbone_ms = _timed_call(_guide_backbone, device=device)
        timings["guide_backbone_ms"].append(guide_backbone_ms)

        token_mask = model._sample_guide_token_mask(guide_features)
        if model.guide_fusion_stage in {"input", "input_gating"}:
            guide_mask, tokenbook_ms = _timed_call(
                lambda: model.guide_tokenbook(guide_features, token_mask=token_mask),
                device=device,
            )
            timings["tokenbook_ms"].append(tokenbook_ms)

            def _upsample():
                with _autocast_context(device):
                    return F.interpolate(
                        guide_mask,
                        size=inputs.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )

            guide_for_input, upsample_ms = _timed_call(_upsample, device=device)
            timings["guide_upsample_ms"].append(upsample_ms)

            def _segmentation():
                with _autocast_context(device):
                    guided_input = inputs * guide_for_input
                    features = model.shared_encoder(guided_input)
                    logits = model.task_decoders["ink"](features)
                    return logits

        elif model.guide_fusion_stage == "feature_encoder":
            stage_shapes = model._get_encoder_stage_shapes(inputs.shape[2:])

            def _feature_gates():
                with _autocast_context(device):
                    feature_gates = {}
                    for stage_idx, stage_key in enumerate(model.guide_stage_keys or []):
                        stage_guide = model.guide_stage_tokenbooks[stage_key](
                            guide_features,
                            token_mask=token_mask,
                        )
                        target_shape = stage_shapes[stage_idx]
                        if stage_guide.shape[2:] != target_shape:
                            stage_guide = F.interpolate(
                                stage_guide,
                                size=target_shape,
                                mode="trilinear",
                                align_corners=False,
                            )
                        feature_gates[stage_key] = stage_guide
                    return feature_gates

            feature_gates, tokenbook_ms = _timed_call(_feature_gates, device=device)
            timings["encoder_guide_heads_ms"].append(tokenbook_ms)

            def _segmentation():
                with _autocast_context(device):
                    features = model.shared_encoder(inputs, feature_gates=feature_gates)
                    logits = model.task_decoders["ink"](features)
                    return logits

        elif model.guide_fusion_stage == "feature_skip_concat":
            stage_shapes = model._get_encoder_stage_shapes(inputs.shape[2:])

            def _projected_features():
                with _autocast_context(device):
                    projected_features = []
                    for stage_idx, stage_key in enumerate(model.guide_stage_keys or []):
                        projected = model.guide_skip_projectors[stage_key](guide_features)
                        if projected.shape[2:] != stage_shapes[stage_idx]:
                            projected = F.interpolate(
                                projected,
                                size=stage_shapes[stage_idx],
                                mode="trilinear",
                                align_corners=False,
                            )
                        projected_features.append(projected)
                    return projected_features

            projected_features, projector_ms = _timed_call(_projected_features, device=device)
            timings["skip_projectors_ms"].append(projector_ms)

            def _segmentation():
                with _autocast_context(device):
                    features = model.shared_encoder(inputs)
                    features = [
                        torch.cat([stage_feature, projected_feature], dim=1)
                        for stage_feature, projected_feature in zip(features, projected_features)
                    ]
                    logits = model.task_decoders["ink"](features)
                    return logits

        elif model.guide_fusion_stage == "direct_segmentation":
            def _guide_mask():
                with _autocast_context(device):
                    return model.guide_tokenbook(guide_features, token_mask=token_mask)

            guide_mask, tokenbook_ms = _timed_call(_guide_mask, device=device)
            timings["tokenbook_ms"].append(tokenbook_ms)

            def _segmentation():
                with _autocast_context(device):
                    guide_probability = F.interpolate(
                        guide_mask,
                        size=inputs.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    ).clamp_(1e-6, 1.0 - 1e-6)
                    fg_logit = torch.logit(guide_probability)
                    bg_logit = torch.zeros_like(fg_logit)
                    return torch.cat([bg_logit, fg_logit], dim=1)

        else:  # pragma: no cover - defensive path for future guide modes
            raise ValueError(f"Unsupported guided benchmark mode: {model.guide_fusion_stage!r}")

        logits, segmentation_ms = _timed_call(_segmentation, device=device)
        timings["segmentation_trunk_ms"].append(segmentation_ms)

        def _device_to_host():
            return logits.to(dtype=torch.float16).cpu()

        _host_logits, d2h_ms = _timed_call(_device_to_host, device=device)
        timings["device_to_host_ms"].append(d2h_ms)

    return {name: sum(values) / len(values) for name, values in timings.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline and guided volumetric Dinovol modes.")
    parser.add_argument(
        "--guide-checkpoint",
        type=str,
        default="/home/giorgio/Projects/dino-vesuvius/dino-checkpoints/checkpoint_step_342500.pt",
        help="Local volumetric DINO checkpoint path.",
    )
    parser.add_argument(
        "--patch-size",
        type=str,
        action="append",
        default=None,
        help="3D patch size as z,y,x. Repeat to benchmark multiple sizes.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Reserved for compatibility; startup is measured explicitly.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guide-tokenbook-tokens", type=str, default="full", help="'full' or integer prototype count.")
    parser.add_argument(
        "--guide-tokenbook-prototype-weighting",
        type=str,
        default="mean",
        choices=["mean", "token_mlp"],
        help="Prototype aggregation mode for guided variants.",
    )
    parser.add_argument(
        "--guide-tokenbook-weight-mlp-hidden",
        type=int,
        default=None,
        help="Hidden width for token-conditioned prototype weighting MLP.",
    )
    parser.add_argument("--skip-compile-variants", action="store_true")
    parser.add_argument("--skip-stage-breakdown", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    patch_sizes = args.patch_size or ["64,64,64"]
    patch_sizes = [_parse_patch_size(text) for text in patch_sizes]

    if args.guide_tokenbook_tokens.strip().lower() == "full":
        guide_tokenbook_tokens = None
    else:
        guide_tokenbook_tokens = int(args.guide_tokenbook_tokens)
        if guide_tokenbook_tokens <= 0:
            raise ValueError("--guide-tokenbook-tokens must be positive or 'full'")

    guide_checkpoint = Path(args.guide_checkpoint)
    if not guide_checkpoint.exists():
        raise FileNotFoundError(f"Guide checkpoint not found: {guide_checkpoint}")

    summary: dict[str, object] = {
        "device": str(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "guide_tokenbook_tokens": _label_name(guide_tokenbook_tokens),
        "guide_tokenbook_prototype_weighting": args.guide_tokenbook_prototype_weighting,
        "guide_tokenbook_weight_mlp_hidden": args.guide_tokenbook_weight_mlp_hidden,
        "results": {},
    }

    pin_memory = device.type == "cuda"
    for patch_size in patch_sizes:
        patch_label = "x".join(str(v) for v in patch_size)
        cpu_inputs, cpu_labels = _make_cpu_batch(patch_size, pin_memory=pin_memory)
        guided_eager_model = _build_model(
            patch_size=patch_size,
            device=device,
            guide_checkpoint=str(guide_checkpoint),
            guide_tokenbook_tokens=guide_tokenbook_tokens,
            guide_fusion_stage="input",
            guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
            guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
            compile_model=False,
            channels_last_3d=False,
        )
        feature_encoder_eager_model = _build_model(
            patch_size=patch_size,
            device=device,
            guide_checkpoint=str(guide_checkpoint),
            guide_tokenbook_tokens=guide_tokenbook_tokens,
            guide_fusion_stage="feature_encoder",
            guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
            guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
            compile_model=False,
            channels_last_3d=False,
        )
        feature_skip_concat_eager_model = _build_model(
            patch_size=patch_size,
            device=device,
            guide_checkpoint=str(guide_checkpoint),
            guide_tokenbook_tokens=guide_tokenbook_tokens,
            guide_fusion_stage="feature_skip_concat",
            guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
            guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
            compile_model=False,
            channels_last_3d=False,
        )
        direct_segmentation_eager_model = _build_model(
            patch_size=patch_size,
            device=device,
            guide_checkpoint=str(guide_checkpoint),
            guide_tokenbook_tokens=guide_tokenbook_tokens,
            guide_fusion_stage="direct_segmentation",
            guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
            guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
            compile_model=False,
            channels_last_3d=False,
        )

        patch_summary: dict[str, object] = {}
        variants = {
            "baseline_eager": _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=None,
                guide_tokenbook_tokens=None,
                guide_fusion_stage=None,
                guide_tokenbook_prototype_weighting="mean",
                guide_tokenbook_weight_mlp_hidden=None,
                compile_model=False,
                channels_last_3d=False,
            ),
            "guided_input_eager": guided_eager_model,
            "guided_feature_encoder_eager": feature_encoder_eager_model,
            "guided_feature_skip_concat_eager": feature_skip_concat_eager_model,
            "guided_direct_segmentation_eager": direct_segmentation_eager_model,
        }
        if not args.skip_compile_variants:
            variants["guided_input_compile"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="input",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=False,
            )
            variants["guided_input_compile_channels_last_3d"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="input",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=True,
            )
            variants["guided_feature_encoder_compile"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="feature_encoder",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=False,
            )
            variants["guided_feature_encoder_compile_channels_last_3d"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="feature_encoder",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=True,
            )
            variants["guided_feature_skip_concat_compile"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="feature_skip_concat",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=False,
            )
            variants["guided_feature_skip_concat_compile_channels_last_3d"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="feature_skip_concat",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=True,
            )
            variants["guided_direct_segmentation_compile"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="direct_segmentation",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=False,
            )
            variants["guided_direct_segmentation_compile_channels_last_3d"] = _build_model(
                patch_size=patch_size,
                device=device,
                guide_checkpoint=str(guide_checkpoint),
                guide_tokenbook_tokens=guide_tokenbook_tokens,
                guide_fusion_stage="direct_segmentation",
                guide_tokenbook_prototype_weighting=args.guide_tokenbook_prototype_weighting,
                guide_tokenbook_weight_mlp_hidden=args.guide_tokenbook_weight_mlp_hidden,
                compile_model=True,
                channels_last_3d=True,
            )

        if not args.skip_stage_breakdown:
            patch_summary["guided_input_eager_stage_breakdown_ms"] = _profile_guided_stages(
                guided_eager_model,
                device=device,
                cpu_inputs=cpu_inputs.detach().clone(),
                iterations=args.iterations,
            )
            patch_summary["guided_feature_encoder_eager_stage_breakdown_ms"] = _profile_guided_stages(
                feature_encoder_eager_model,
                device=device,
                cpu_inputs=cpu_inputs.detach().clone(),
                iterations=args.iterations,
            )
            patch_summary["guided_feature_skip_concat_eager_stage_breakdown_ms"] = _profile_guided_stages(
                feature_skip_concat_eager_model,
                device=device,
                cpu_inputs=cpu_inputs.detach().clone(),
                iterations=args.iterations,
            )
            patch_summary["guided_direct_segmentation_eager_stage_breakdown_ms"] = _profile_guided_stages(
                direct_segmentation_eager_model,
                device=device,
                cpu_inputs=cpu_inputs.detach().clone(),
                iterations=args.iterations,
            )

        for variant_name, model in variants.items():
            variant_inputs = cpu_inputs
            if variant_name.endswith("channels_last_3d"):
                variant_inputs = cpu_inputs.contiguous(memory_format=torch.channels_last_3d)
            patch_summary[variant_name] = _run_variant_benchmark(
                model,
                device=device,
                cpu_inputs=variant_inputs,
                cpu_labels=cpu_labels,
                iterations=args.iterations,
            )
        summary["results"][patch_label] = patch_summary

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
