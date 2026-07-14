"""DDP 5-class self-distillation trainer (fiber/ink/papyrus on PHercParis4).

Pseudo-label = FiveClassLabelGenerator(image, raw_image) — fiber teacher
(ihoo3tpl ckpt) + ink teacher (S3 ckpt) + cuws watershed + per-instance PCA
orientation classifier + papyrus dark-tissue catchall.

Student is a fresh NetworkFromConfig UNet with a 5-channel softmax head.
Loss = CE(label_smoothing=0.1) + multiclass soft Dice (smoothing=0.1).
Augmentations (flips/rot90 + image-only gamma/contrast/noise) are applied
AFTER pseudo-label generation, identically on image and label.

Logs every step (loss / lr / class fractions / instance counts) and every
``val_every`` steps a matplotlib debug figure (categorical fixed palette)
written to BOTH wandb AND a local debug dir (``debug_dir`` in the config, or
``<tempdir>/fiber4c_debug`` by default).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import RandomFiberCropDataset, collate_random_crops  # noqa: E402
from model import build_fiber_unet  # noqa: E402
from label_generator import (  # noqa: E402
    FiveClassConfig,
    FiveClassLabelGenerator,
)
from visualization import (  # noqa: E402
    categorical_class_labels,
    make_debug_figure,
)


# ---------- helpers ---------------------------------------------------------

class EMAWrapper:
    """Exponential moving average of a model's state_dict, identical to fiber_dinoguided."""
    def __init__(self, model: torch.nn.Module, *, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.state_dict().items():
            if not torch.is_floating_point(p):
                self.shadow[n].copy_(p.detach())
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.shadow.items()}


# ---------- Augmentations (5-class label-friendly variants) -----------------

def apply_spatial_aug_label(image: torch.Tensor, label: torch.Tensor):
    """Spatial flip + 90deg rotation jointly on image (B,1,Z,Y,X) and label (B,1,Z,Y,X) int."""
    B = image.shape[0]
    out_img = image.clone()
    out_lbl = label.clone()
    for b in range(B):
        i = out_img[b:b + 1]
        l = out_lbl[b:b + 1]
        for axis in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                i = torch.flip(i, dims=[axis])
                l = torch.flip(l, dims=[axis])
        if torch.rand(1).item() < 0.75:
            planes = [(2, 3), (2, 4), (3, 4)]
            p = int(torch.randint(0, 3, (1,)).item())
            a, b_ = planes[p]
            k = int(torch.randint(1, 4, (1,)).item())
            i = torch.rot90(i, k=k, dims=(a, b_))
            l = torch.rot90(l, k=k, dims=(a, b_))
            # Rotating around the Z-Y or Z-X plane swaps vertical/horizontal classes.
            # Class 1 == vertical, class 2 == horizontal: when we rotate around an
            # axis that contains Z (planes (Z,Y) or (Z,X)) by 90 or 270 degrees,
            # what was vertical becomes horizontal and vice versa.
            if (a, b_) in ((2, 3), (2, 4)) and (k % 2 == 1):
                # Swap 1<->2 in the rotated label.
                m1 = l == 1
                m2 = l == 2
                l = torch.where(m1, torch.full_like(l, 2), l)
                l = torch.where(m2, torch.full_like(l, 1), l)
        out_img[b:b + 1] = i
        out_lbl[b:b + 1] = l
    return out_img, out_lbl


def apply_intensity_aug(image: torch.Tensor) -> torch.Tensor:
    """Image-only intensity perturbations (gamma, contrast, Gaussian noise)."""
    B = image.shape[0]
    out = image.clone()
    for b in range(B):
        x = out[b:b + 1]
        if torch.rand(1).item() < 0.5:
            gamma = float(0.8 + 0.4 * torch.rand(1).item())
            x = x.clamp(0.0, 1.0) ** gamma
        if torch.rand(1).item() < 0.5:
            c = float(0.85 + 0.3 * torch.rand(1).item())
            x = ((x - 0.5) * c + 0.5).clamp(0.0, 1.0)
        if torch.rand(1).item() < 0.3:
            sigma = float(0.005 + 0.025 * torch.rand(1).item())
            x = (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
        out[b:b + 1] = x
    return out


# ---------- Losses ----------------------------------------------------------

def multiclass_dice(softmax: torch.Tensor, target_onehot: torch.Tensor,
                   smoothing: float, eps: float = 1e-6,
                   include_bg: bool = False) -> torch.Tensor:
    """Mean soft Dice over classes. softmax/target_onehot: (B, C, Z, Y, X)."""
    if not include_bg:
        softmax = softmax[:, 1:]
        target_onehot = target_onehot[:, 1:]
    smoothed = target_onehot * (1.0 - smoothing) + 0.5 * smoothing
    dims = (0, 2, 3, 4)
    inter = (softmax * smoothed).sum(dim=dims)
    s = softmax.sum(dim=dims) + smoothed.sum(dim=dims)
    dice_per_class = 1.0 - (2.0 * inter + eps) / (s + eps)
    return dice_per_class.mean()


def per_class_dice(softmax: torch.Tensor, target_onehot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-class hard Dice (no smoothing) for logging. Returns shape (C,)."""
    pred = softmax.argmax(dim=1)
    C = target_onehot.shape[1]
    out = torch.zeros(C, device=softmax.device, dtype=torch.float32)
    for c in range(C):
        p = (pred == c).float()
        t = target_onehot[:, c]
        inter = (p * t).sum()
        s = p.sum() + t.sum()
        out[c] = (2.0 * inter + eps) / (s + eps)
    return out


# ---------- Local /tmp PNG sink (LRU 50) -----------------------------------

def _evict_lru(out_dir: Path, keep: int = 50) -> None:
    files = sorted(out_dir.glob("step_*.png"))
    if len(files) <= keep:
        return
    for p in files[: len(files) - keep]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ---------- Training loop ---------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wandb_offline", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DDP setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size > 1
    if ddp_enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    is_rank0 = (global_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        cfg.get("mixed_precision", "bf16")
    ]
    if is_rank0:
        print(f"[rank0] device={device} world={world_size} dtype={dtype}", flush=True)

    # ---- WandB ----
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if is_rank0:
        init_kwargs = dict(
            project=cfg["wandb_project"],
            entity=cfg["wandb_entity"],
            name=cfg.get("wandb_run_name"),
            tags=cfg.get("wandb_tags", []),
            config=cfg,
            dir=str(out_dir),
            reinit="finish_previous",
        )
        if cfg.get("wandb_run_id"):
            init_kwargs["id"] = cfg["wandb_run_id"]
            init_kwargs["resume"] = cfg.get("wandb_resume", "allow")
        wandb.init(**init_kwargs)
        print(f"[rank0] wandb run: {wandb.run.url}", flush=True)
        run_name = cfg.get("wandb_run_name") or wandb.run.name
    else:
        run_name = cfg.get("wandb_run_name") or "rank_nonzero"
    if ddp_enabled:
        dist.barrier()

    # ---- Local PNG sink (config-driven; defaults under the system temp dir) ----
    tmp_dir = Path(cfg.get("debug_dir") or (Path(tempfile.gettempdir()) / "fiber4c_debug"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Student UNet (trainable, 5-class head, fresh) ----
    target_name = cfg.get("target_name", "labels")
    out_channels = int(cfg["out_channels"])
    student = build_fiber_unet(
        crop_size=tuple(cfg["patch_size"]),
        target_name=target_name,
        out_channels=out_channels,
        activation=cfg.get("activation", "none"),
        in_channels=cfg.get("in_channels", 1),
    ).to(device)

    student_module = student
    if ddp_enabled:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)

    # ---- Frozen label generator (loaded on this rank's device) ----
    pseudo_cfg_in = cfg["pseudo"]
    five_cfg = FiveClassConfig(
        fiber_thr=float(pseudo_cfg_in.get("fiber_thr", 0.5)),
        ink_thr=float(pseudo_cfg_in.get("ink_thr", 0.5)),
        papyrus_raw_thr=int(pseudo_cfg_in.get("papyrus_raw_thr", 90)),
        dark_voxel_thr=int(pseudo_cfg_in.get("dark_voxel_thr", 90)),
        ws_image_mode=str(pseudo_cfg_in.get("ws_image_mode", "distance")),
        ws_h_merge=int(pseudo_cfg_in.get("ws_h_merge", 14000)),
        ws_min_voxels=int(pseudo_cfg_in.get("ws_min_voxels", 400)),
        pca_cos_threshold=float(pseudo_cfg_in.get("pca_cos_threshold", 0.819)),
    )
    label_gen = FiveClassLabelGenerator(
        fiber_teacher_ckpt=pseudo_cfg_in["fiber_teacher"],
        ink_teacher_ckpt=pseudo_cfg_in.get("ink_teacher"),
        config=five_cfg,
        device=device,
        dtype=dtype,
        crop_size=tuple(cfg["patch_size"]),
    )
    if is_rank0:
        print(f"[rank0] label_gen ready", flush=True)

    # ---- Optimizer + LR schedule ----
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.005))
    if opt_cfg.get("name", "SGD").lower() == "sgd":
        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.99)),
            weight_decay=float(opt_cfg.get("weight_decay", 3e-5)),
            nesterov=bool(opt_cfg.get("nesterov", True)),
        )
    else:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=lr,
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        )
    warmup_steps = int(cfg.get("warmup_steps", 1500))
    total_steps = int(cfg.get("num_iterations", 12000))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ---- EMA ----
    ema_cfg = cfg.get("ema", {"enabled": True, "decay": 0.9995, "start_step": 1000})
    ema = EMAWrapper(student_module, decay=float(ema_cfg.get("decay", 0.9995))) if ema_cfg.get("enabled", True) else None
    ema_start = int(ema_cfg.get("start_step", 1000))

    # ---- Dataloader ----
    ds_cfg = cfg["dataset"]
    rank_seed = int(cfg.get("seed", 27)) * 100 + global_rank
    train_dataset = RandomFiberCropDataset(
        volume_url=ds_cfg["volume_url"],
        crop_size=tuple(cfg["patch_size"]),
        storage_options=ds_cfg.get("storage_options", {"anon": True}),
        scale=int(ds_cfg.get("scale", 0)),
        min_nonempty_frac=float(ds_cfg.get("min_nonempty_frac", 0.10)),
        dark_threshold=int(ds_cfg.get("dark_threshold", 50)),
        seed=rank_seed,
    )
    nw = int(cfg.get("dataloader_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        num_workers=nw,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        persistent_workers=(nw > 0),
        collate_fn=collate_random_crops,
    )
    train_iter = iter(train_loader)

    # ---- Optional resume ----
    resume_path = cfg.get("resume_from_ckpt")
    start_step = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        student_module.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            except Exception as e:
                if is_rank0:
                    print(f"WARN: failed to load optimizer state: {e}", flush=True)
        if ckpt.get("ema") and ema is not None:
            shadow_src = ckpt["ema"].get("model_state", ckpt["ema"])
            ema.shadow = {k: v.to(device) for k, v in shadow_src.items()}
        start_step = int(ckpt.get("step", 0))
        if is_rank0:
            print(f"[rank0] resumed from {resume_path} step {start_step}", flush=True)

    # ---- Training ----
    val_every = int(cfg.get("val_every", 100))
    save_every = int(cfg.get("save_every", 1000))
    ce_smoothing = float(cfg.get("ce_label_smoothing", 0.1))
    dice_smoothing = float(cfg.get("dice_label_smoothing", 0.1))

    if is_rank0:
        print(f"[rank0] training start: total_steps={total_steps} warmup={warmup_steps} "
              f"per_rank_batch={cfg.get('batch_size', 1)}", flush=True)

    step = start_step
    t_start = time.time()
    student.train()

    while step < total_steps:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(step)

        batch = next(train_iter)
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        raw = batch["raw_image"].to(device=device, non_blocking=True)

        # 1) Pseudo-label on the CLEAN image (no augmentation yet).
        with torch.inference_mode():
            pseudo_uint8, debug = label_gen.generate(image, raw)
        pseudo = pseudo_uint8.to(dtype=torch.int64)         # (B, 1, Z, Y, X) int64 in {0..4}

        # 2) Augment AFTER label generation. Use joint spatial aug on (image, label),
        #    then image-only intensity aug.
        image_aug, pseudo_aug = apply_spatial_aug_label(image, pseudo)
        image_aug = apply_intensity_aug(image_aug.float()).to(dtype=torch.float32)

        # 3) Student forward (mixed-precision context).
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=dtype, enabled=device.type == "cuda"):
            out = student(image_aug)
            logits = out[target_name] if isinstance(out, dict) else out
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = logits.float()                                      # (B, 5, Z, Y, X)

        # 4) Loss. CE expects (B, C, ...) logits + (B, ...) target.
        target = pseudo_aug.squeeze(1)                                   # (B, Z, Y, X)
        ce = F.cross_entropy(logits, target, label_smoothing=ce_smoothing)
        target_onehot = F.one_hot(target, num_classes=out_channels).permute(0, 4, 1, 2, 3).float()
        softmax = torch.softmax(logits, dim=1)
        dice = multiclass_dice(softmax, target_onehot, smoothing=dice_smoothing, include_bg=False)
        loss = ce + dice

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        if ema is not None and step >= ema_start:
            ema.update(student_module)

        # 5) Logging.
        if ddp_enabled:
            for t in (loss, ce, dice):
                dist.all_reduce(t.detach(), op=dist.ReduceOp.AVG)

        class_counts = debug.class_counts.float().sum(dim=0)             # (5,) total voxels per class in this rank's batch
        total_voxels = class_counts.sum().clamp_min(1.0)
        if ddp_enabled:
            dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_voxels, op=dist.ReduceOp.SUM)
        class_fracs = (class_counts / total_voxels).detach().cpu().numpy().tolist()

        n_inst_mean = float(debug.n_instances.float().mean().item())
        n_vert_mean = float(debug.n_vert.float().mean().item())

        log_payload = {
            "loss": float(loss.item()),
            "loss_ce": float(ce.item()),
            "loss_dice": float(dice.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - t_start),
            "pseudo/frac_bg": class_fracs[0],
            "pseudo/frac_vert": class_fracs[1],
            "pseudo/frac_horiz": class_fracs[2],
            "pseudo/frac_ink": class_fracs[3],
            "pseudo/n_instances_mean": n_inst_mean,
            "pseudo/n_vert_mean": n_vert_mean,
        }

        # Debug image every val_every (rank 0 only).
        if step % val_every == 0 and is_rank0:
            image_np = image[0, 0].detach().float().cpu().numpy()
            raw_np = raw[0, 0].detach().cpu().numpy().astype(np.uint8)
            label_np = pseudo_uint8[0, 0].detach().cpu().numpy().astype(np.int32)
            fiber_prob_np = debug.fiber_prob[0, 0].detach().float().cpu().numpy()
            ink_prob_np = debug.ink_prob[0, 0].detach().float().cpu().numpy()
            inst_np = debug.instance_map[0, 0].detach().cpu().numpy().astype(np.int64)

            # Re-forward the student on the CLEAN image for spatial alignment.
            with torch.inference_mode(), torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=dtype, enabled=device.type == "cuda",
            ):
                vis_out = student(image)
                vis_logits = vis_out[target_name] if isinstance(vis_out, dict) else vis_out
                if isinstance(vis_logits, (list, tuple)):
                    vis_logits = vis_logits[0]
                vis_logits = vis_logits.float()
                vis_pred = vis_logits.argmax(dim=1)
            pred_np = vis_pred[0].detach().cpu().numpy().astype(np.int32)

            # per-class Dice (hard) on the clean prediction.
            student.train()
            pseudo_oh = F.one_hot(pseudo.squeeze(1), num_classes=out_channels).permute(0, 4, 1, 2, 3).float()
            pc = per_class_dice(torch.softmax(vis_logits, dim=1), pseudo_oh).detach().cpu().numpy().tolist()
            for c, name in categorical_class_labels().items():
                log_payload[f"metrics/dice_{c}_{name.replace(' ', '_')}"] = float(pc[c])
            log_payload["metrics/dice_fg_mean"] = float(np.mean(pc[1:]))

            fig = make_debug_figure(
                image_zyx=image_np,
                raw_zyx=raw_np,
                label_zyx=label_np,
                fiber_prob_zyx=fiber_prob_np,
                ink_prob_zyx=ink_prob_np,
                instance_map_zyx=inst_np,
                student_pred_zyx=pred_np,
                title_suffix=f"step {step}",
            )
            local_png = tmp_dir / f"{run_name}_step{step:06d}.png"
            try:
                fig.savefig(local_png, dpi=110, bbox_inches="tight")
            except Exception as e:
                print(f"[rank0] WARN: failed to save {local_png}: {e}", flush=True)
            log_payload["debug_figure"] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            _evict_lru(tmp_dir, keep=50)

            # Also log a native wandb categorical mask (interactive legend).
            try:
                z_mid = label_np.shape[0] // 2
                base = image_np[z_mid]
                base = (base - base.min()) / max(1e-6, base.max() - base.min())
                log_payload["debug_mask"] = wandb.Image(
                    base,
                    masks={
                        "pseudo": {
                            "mask_data": label_np[z_mid].astype(np.uint8),
                            "class_labels": categorical_class_labels(),
                        },
                        "student": {
                            "mask_data": pred_np[z_mid].astype(np.uint8),
                            "class_labels": categorical_class_labels(),
                        },
                    },
                )
            except Exception as e:
                print(f"[rank0] WARN: wandb mask image failed: {e}", flush=True)

        if is_rank0:
            wandb.log(log_payload, step=step)
            if step % 50 == 0:
                msg = (f"step {step:>6} | loss {log_payload['loss']:.4f} "
                       f"(ce {log_payload['loss_ce']:.4f} dice {log_payload['loss_dice']:.4f}) "
                       f"lr {log_payload['lr']:.4g} | "
                       f"fracs bg={class_fracs[0]:.3f} v={class_fracs[1]:.3f} "
                       f"h={class_fracs[2]:.3f} i={class_fracs[3]:.3f} | "
                       f"n_inst {n_inst_mean:.1f}")
                print(msg, flush=True)

        # 6) Checkpoint.
        if step > 0 and step % save_every == 0 and is_rank0:
            ckpt_path = out_dir / f"ckpt_{step:06d}.pth"
            payload = {
                "model": student_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg,
            }
            if ema is not None:
                payload["ema"] = {"model_state": ema.state_dict()}
            torch.save(payload, str(ckpt_path))
            print(f"[rank0] saved {ckpt_path}", flush=True)

        step += 1

    if is_rank0:
        wandb.finish()
        print("[rank0] training complete.", flush=True)
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
