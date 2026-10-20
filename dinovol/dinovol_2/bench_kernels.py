"""Isolated CUDA benchmarks using synthetic inputs or saved real-volume features."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from dinovol_2.loss.ibot_patch_loss import iBOTPatchLoss
from dinovol_2.ops.point_embeddings import sample_grid_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("interpolation", "ibot"))
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--prototypes", type=int, default=131072)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--features", help="Fixture written by dinovol_2.validate_real_kernels")
    args = parser.parse_args()
    if args.rows < 1 or args.warmup < 1 or args.iterations < 1:
        parser.error("rows, warmup, and iterations must be positive")
    torch.manual_seed(5)
    dtype = getattr(torch, args.dtype)
    fixture = torch.load(args.features, map_location="cpu", weights_only=True) if args.features else None
    if args.operation == "interpolation":
        grid = torch.randn(4, 24, 24, 24, 480, device="cuda", dtype=dtype)
        coordinates = torch.rand(4, 13824, 3, device="cuda") * 192 - 0.5
        source_rows = torch.tensor([2, 3, 0, 1], device="cuda")
        indices = torch.randperm(4 * 13824, device="cuda")[:args.rows]
        if fixture is not None:
            grid = fixture["grid"].to(device="cuda", dtype=dtype)
            coordinates = fixture["coordinates"].cuda()
            source_rows, indices = fixture["source_rows"].cuda(), fixture["indices"].cuda()
            args.rows = indices.numel()

        def run():
            if args.backend == "torch":
                return sample_grid_points(grid[source_rows], coordinates, (192,) * 3).to(dtype).flatten(0, 1)[indices]
            from dinovol_2.ops.point_embeddings import sample_selected_grid_points
            return sample_selected_grid_points(grid, coordinates, (192,) * 3, indices, source_rows)
    else:
        features = F.normalize(torch.randn(args.rows, args.dim, device="cuda"), dim=-1).to(dtype)
        teacher_features = F.normalize(torch.randn_like(features).float(), dim=-1).to(dtype)
        weight = F.normalize(torch.randn(args.prototypes, args.dim, device="cuda"), dim=-1).to(dtype)
        teacher_weight = F.normalize(torch.randn_like(weight).float(), dim=-1).to(dtype)
        if fixture is not None:
            features, teacher_features, weight, teacher_weight = [
                fixture[name].to(device="cuda", dtype=dtype)
                for name in ("student_features", "teacher_features", "student_weight", "teacher_weight")
            ]
            args.rows, args.dim = features.shape
            args.prototypes = weight.shape[0]
        features.requires_grad_()
        weight.requires_grad_()
        options = {"masked_loss_chunk_size": args.chunk_size}
        if args.backend != "torch":
            options["kernel_backend"] = args.backend
        loss_fn = iBOTPatchLoss(args.prototypes, **options).cuda()
        if args.compile:
            loss_fn.compile_kernels(backend="inductor", mode="max-autotune-no-cudagraphs")
        count = torch.tensor([args.rows], device="cuda")
        masks = torch.empty(4, 13824, device="cuda", dtype=torch.bool)
        weights = torch.full((args.rows,), 4 / args.rows, device="cuda")
        if fixture is not None:
            weights = fixture["masks_weight"].cuda()

        def run():
            features.grad = weight.grad = None
            # Match pretrain: autocast forward, backward outside autocast.
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                scales = loss_fn.sinkhorn_knopp_prototype_scales(
                    teacher_features, teacher_weight, 0.07, count
                )
                loss = loss_fn.forward_masked_from_features(
                    features, weight, teacher_features, teacher_weight, scales, 0.07, masks, weights
                )
            loss.backward()
            return loss

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    elapsed = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
    print(json.dumps({
        **vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "mean_ms": statistics.mean(elapsed), "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed), "max_ms": max(elapsed),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "extra_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
    }, indent=2), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            run()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
