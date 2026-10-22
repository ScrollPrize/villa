"""Train a Vesuvius surface decoder on a frozen dinovol checkpoint."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch._functorch import config as functorch_config

from .data import SurfaceData
from .model import SurfaceModel, load_encoder
from .vesuvius import medial_loss


def save_preview(path, sample, probability, metadata):
    image = sample["image"][0, 0].float().cpu().numpy()
    label = sample["label"][0, 0].cpu().numpy()
    prob = probability[0].float().cpu().numpy()
    low, high = np.percentile(image, [1, 99])
    image = np.uint8(np.clip((image - low) / max(high - low, 1e-8), 0, 1) * 255)
    rows = []
    for axis in range(3):
        gray, target, p = [np.take(v, v.shape[axis] // 2, axis=axis) for v in (image, label, prob)]
        gray = np.repeat(gray[..., None], 3, axis=-1)
        truth = gray.copy()
        truth[target == 1] = (0, 255, 0)
        truth[target == 2] = (180, 0, 180)
        heat = np.zeros_like(gray)
        heat[..., 0] = np.uint8(p * 255)
        heat[..., 2] = np.uint8((1 - p) * 255)
        # Continuous softmax confidence controls the cyan overlay opacity.
        # Keep predictions visible even where the ground truth is ignored.
        alpha = 0.75 * np.clip(p, 0, 1)[..., None]
        overlay = np.uint8(np.rint(gray * (1 - alpha) + np.array([0, 255, 255]) * alpha))
        row = Image.fromarray(np.concatenate([gray, truth, heat, overlay], axis=1))
        # A common row width allows anisotropic crop sizes.
        row = row.resize((768, max(1, round(row.height * 768 / row.width))))
        rows.append(row)
    canvas = Image.new("RGB", (768, 48 + sum(r.height for r in rows)))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"{metadata['id']}  origin ZYX={metadata['origin_zyx']}  planes: Z / Y / X", fill="white")
    draw.text((5, 22), "CT | target (green; ignore purple) | P(surface) (red=1) | soft P(surface) overlay (cyan)", fill="white")
    y = 48
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def autocast(args):
    if args.precision == "float32":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.precision]
    return torch.autocast(device_type=torch.device(args.device).type, dtype=dtype)


def compute_loss(criterion, logits, sample):
    if not (sample["label"] != 2).any():
        return logits.sum() * 0
    return criterion(logits.float(), sample["label"].long(), sample["label_skel"].long())


@torch.no_grad()
def validate(model, criterion, data, args, step):
    model.eval()
    totals = np.zeros(5, dtype=np.float64)
    loss_total, count = 0.0, 0
    locations = []
    previewed_cases = set()
    preview_records = []
    for sample, metadata in data.validation(seed=args.seed + 1000003,
                                           crops_per_case=args.val_crops_per_case,
                                           case_limit=args.val_case_limit):
        with autocast(args):
            logits = model(sample["image"])
        loss = compute_loss(criterion, logits, sample)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite validation loss at {metadata}")
        probability = logits.float().softmax(1)[:, 1]
        valid = sample["label"][:, 0] != 2
        prediction = (probability >= 0.5) & valid
        truth = (sample["label"][:, 0] == 1) & valid
        medial = (sample["label_skel"][:, 0] > 0) & valid
        totals += [int((prediction & truth).sum()), int(prediction.sum()), int(truth.sum()),
                   int((prediction & medial).sum()), int(medial.sum())]
        loss_total += float(loss)
        should_preview = (metadata["id"] not in previewed_cases if args.preview_all_cases
                          else count < args.preview_count)
        if should_preview:
            if args.preview_all_cases:
                path = args.output / "previews" / "all_cases" / f"step_{step:07d}_{metadata['id']}.png"
            else:
                path = args.output / "previews" / f"step_{step:07d}_{count:03d}_{metadata['id']}.png"
            save_preview(path, sample, probability, metadata)
            previewed_cases.add(metadata["id"])
            preview_records.append({**metadata, "image": str(path.relative_to(args.output))})
        locations.append(metadata)
        count += 1
    intersection, predicted, foreground, recalled, medial_count = totals
    metrics = {"kind": "validation", "step": step, "loss": loss_total / count,
               "foreground_dice": 2 * intersection / (predicted + foreground) if predicted + foreground else 1.0,
               "medial_recall": recalled / medial_count if medial_count else None,
               "crops": count, "foreground_voxels": int(foreground),
               "medial_voxels": int(medial_count)}
    (args.output / "validation_crops.json").write_text(json.dumps(locations, indent=2) + "\n")
    (args.output / "preview_manifest.json").write_text(json.dumps(preview_records, indent=2) + "\n")
    model.train()
    return metrics


def log(args, record):
    record = {**record, "time": time.time()}
    with (args.output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record), flush=True)


def save_checkpoint(path, *, model, optimizer, scaler, rng, step, best, signature, metadata):
    state = {"format": "dinovol-surface-decoder/1", "step": step, "best_dice": best,
             "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "numpy_rng": rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "signature": signature, "encoder": metadata}
    temporary = path.with_suffix(".pt.partial")
    torch.save(state, temporary)
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="JSON defaults; command-line flags override them")
    p.add_argument("--checkpoint", type=Path, default=Path("/mnt/raid_nvme/dinovol_ckpts/checkpoint_step_640000.pt"))
    p.add_argument("--branch", choices=["teacher", "student"], default="teacher")
    p.add_argument("--data", type=Path, default=Path("evals/kaggle_surfaces/data"))
    p.add_argument("--output", type=Path, default=Path("evals/kaggle_surfaces/runs/frozen_640000"))
    p.add_argument("--vesuvius-source", type=Path, help="Vesuvius src directory; omit for installed package")
    p.add_argument("--device", default="cuda:0", help="CUDA for training; cpu is available for smoke tests")
    p.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--patch-size", type=int, nargs=3, default=[192, 192, 192])
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Compile encoder and decoder forwards with TorchInductor (default: enabled)")
    p.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=12.0)
    p.add_argument("--normalization", choices=["robust", "zscore", "minmax"], default="robust")
    p.add_argument("--min-valid-fraction", type=float, default=0.01)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-crops-per-case", type=int, default=2)
    p.add_argument("--val-case-limit", type=int, help="Smoke tests only; default evaluates every validation case")
    p.add_argument("--preview-count", type=int, default=6)
    p.add_argument("--preview-all-cases", action="store_true",
                   help="Save a soft-probability panel from the first evaluated crop of every validation case")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-only", action="store_true", help="Requires --resume; writes validation metrics and PNGs")
    return p


def main(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text())
        unknown = defaults.keys() - {a.dest for a in p._actions}
        if unknown:
            p.error(f"Unknown config keys: {sorted(unknown)}")
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    for name in ("checkpoint", "data", "output", "vesuvius_source", "resume"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    positive = ("blocks", "batch_size", "steps", "val_every", "val_crops_per_case", "log_every", "cpu_threads")
    if any(getattr(args, k) <= 0 for k in positive) or min(args.patch_size) <= 0:
        p.error("Sizes, steps and intervals must be positive")
    if not 0 < args.min_valid_fraction <= 1 or args.preview_count < 0:
        p.error("min-valid-fraction must be in (0,1]; preview-count must be nonnegative")
    if args.val_case_limit is not None and args.val_case_limit <= 0:
        p.error("val-case-limit must be positive")
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.clip_grad)) or \
            args.lr <= 0 or args.weight_decay < 0 or args.clip_grad <= 0:
        p.error("lr and clip-grad must be finite and positive; weight-decay must be finite and nonnegative")
    if args.eval_only and not args.resume:
        p.error("--eval-only requires --resume")
    device = torch.device(args.device)
    if device.type == "cuda":
        from vc_reader import cuda
        if not torch.cuda.is_available() or not cuda.is_available():
            raise RuntimeError("Training requires a working CUDA device and CUDA-enabled vc_reader")
        torch.cuda.set_device(device)
    elif device.type != "cpu" or args.precision != "float32":
        p.error("CPU smoke tests require --device cpu --precision float32")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output / "metrics.jsonl").exists():
        raise FileExistsError("Run output already exists; choose another --output or use --resume")
    encoder, metadata = load_encoder(args.checkpoint, args.branch)
    metadata["checkpoint"] = str(args.checkpoint)
    model = SurfaceModel(encoder, blocks=args.blocks, vesuvius_source=args.vesuvius_source).to(device)
    model.train()
    criterion = medial_loss(args.vesuvius_source).to(device)
    parameters = [v for v in model.decoder.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "float16")
    manifest_hash = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    signature = {k: getattr(args, k) for k in ("branch", "patch_size", "blocks", "normalization", "seed",
                                               "batch_size", "lr", "weight_decay", "precision", "min_valid_fraction")}
    signature.update(checkpoint=str(args.checkpoint), checkpoint_bytes=args.checkpoint.stat().st_size,
                     checkpoint_mtime_ns=args.checkpoint.stat().st_mtime_ns, manifest_sha256=manifest_hash)
    start, best = 0, -math.inf
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("format") != "dinovol-surface-decoder/1" or state["signature"] != signature:
            raise ValueError("Resume configuration, encoder checkpoint, or dataset differs from saved run")
        model.decoder.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, best = state["step"], state["best_dice"]
        if not args.eval_only and args.steps <= start:
            p.error(f"--steps must exceed the resumed step {start}, or use --eval-only")
    if args.compile:
        # In-place Module.compile preserves parameter identities and state_dict
        # keys, including compatibility with existing eager decoder checkpoints.
        model.compile(backend="inductor", mode=args.compile_mode, dynamic=False)
        print(f"TorchInductor compilation enabled (mode={args.compile_mode}); "
              "first training/validation calls include compilation.", flush=True)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "run.json").write_text(json.dumps({"args": run_config, "encoder": metadata,
                                                    "signature": signature}, indent=2) + "\n")
    print(f"Frozen encoder: {sum(v.numel() for v in encoder.parameters()):,} parameters; "
          f"trainable decoder: {sum(v.numel() for v in parameters):,}", flush=True)
    data = SurfaceData(args.data, args.patch_size, device=device, normalization=args.normalization,
                       min_valid_fraction=args.min_valid_fraction)
    try:
        if args.eval_only:
            log(args, validate(model, criterion, data, args, start))
            return
        for step in range(start + 1, args.steps + 1):
            sample = data.training_batch(rng, args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            # AOTAutograd captures backward during forward; backward runs outside
            # autocast, just as in dinovol pretraining.
            backward_context = (functorch_config.patch(backward_pass_autocast="off")
                                if args.compile else nullcontext())
            with backward_context, autocast(args):
                logits = model(sample["image"])
            loss = compute_loss(criterion, logits, sample)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % args.log_every == 0:
                log(args, {"kind": "train", "step": step, "loss": float(loss.detach())})
            # Release training tensors before validation to bound peak GPU memory.
            del sample, logits, loss
            if step % args.val_every == 0 or step == args.steps:
                metrics = validate(model, criterion, data, args, step)
                log(args, metrics)
                improved = metrics["foreground_dice"] > best
                best = max(best, metrics["foreground_dice"])
                options = dict(model=model, optimizer=optimizer, scaler=scaler, rng=rng,
                               step=step, best=best, signature=signature, metadata=metadata)
                save_checkpoint(args.output / "last.pt", **options)
                if improved:
                    save_checkpoint(args.output / "best.pt", **options)
    finally:
        data.close()


if __name__ == "__main__":
    main()
