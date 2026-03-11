from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from vesuvius.models.datasets.ssl_zarr_dataset import SSLZarrDataset

from .collate import build_dino_ibot_collate_fn
from .loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from .model import DinoVitStudentTeacher

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return float(value[0]), float(value[1])


def _cosine_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    ratio = step / float(total_steps - 1)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * ratio))

class DinoIBOTPretrainer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model_config = dict(self.config["model"])
        self.device = torch.device(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = bool(self.config.get("use_amp", self.device.type == "cuda"))

        self.model = DinoVitStudentTeacher(self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=float(self.config.get("lr", 1e-4)),
            betas=tuple(self.config.get("betas", (0.9, 0.999))),
            weight_decay=float(self.config.get("weight_decay", 0.04)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.device.type == "cuda")

        dino_out_dim = int(self.model_config.get("dino_out_dim", 65536))
        ibot_out_dim = int(self.model_config.get("ibot_out_dim", dino_out_dim))
        self.dino_loss = DINOLoss(dino_out_dim).to(self.device)
        self.ibot_patch_loss = iBOTPatchLoss(ibot_out_dim).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        self.dino_loss_weight = float(self.config.get("dino_loss_weight", 1.0))
        self.ibot_loss_weight = float(self.config.get("ibot_loss_weight", 1.0))
        self.koleo_loss_weight = float(self.config.get("koleo_loss_weight", 0.0))
        self.centering = str(self.config.get("centering", "centering"))

        self.epochs = int(self.config.get("epochs", 1))
        self.steps_per_epoch = int(self.config.get("steps_per_epoch", 100))
        self.num_iterations = int(self.config.get("num_iterations", self.epochs * self.steps_per_epoch))
        self.warmup_steps = int(self.config.get("warmup_steps", 0))
        self.total_steps = self.num_iterations
        self.base_lr = float(self.config.get("lr", 1e-4))
        self.min_lr = float(self.config.get("min_lr", 1e-6))
        self.clip_grad = float(self.config.get("clip_grad", 3.0))

        self.teacher_temp = float(self.config.get("teacher_temp", 0.07))
        self.warmup_teacher_temp = float(self.config.get("warmup_teacher_temp", self.teacher_temp))
        self.warmup_teacher_temp_steps = int(self.config.get("warmup_teacher_temp_steps", 0))
        self.momentum_teacher = float(self.config.get("momentum_teacher", 0.996))
        self.final_momentum_teacher = float(self.config.get("final_momentum_teacher", 1.0))

        self.log_every = int(self.config.get("log_every", 20))
        self.save_every = int(self.config.get("save_every", 0))
        self.output_dir = Path(self.config.get("output_dir", "./dinov2_pretrain_runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataloader(self) -> DataLoader:
        dataset = SSLZarrDataset(self.config["dataset"], do_augmentations=True)
        collate_fn = build_dino_ibot_collate_fn(
            {
                "global_crop_size": dataset.global_crop_size,
                "patch_size": self.model_config.get("patch_size", (8, 8, 8)),
                "mask_ratio_min_max": _as_float_pair(self.config.get("mask_ratio_min_max"), (0.1, 0.5)),
                "mask_sample_probability": float(self.config.get("mask_sample_probability", 0.5)),
                "dtype": torch.float32,
            }
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 2)),
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            collate_fn=collate_fn,
        )

    def _teacher_temp(self, step: int) -> float:
        if step < self.warmup_teacher_temp_steps:
            return _cosine_schedule(
                step,
                max(self.warmup_teacher_temp_steps, 1),
                self.warmup_teacher_temp,
                self.teacher_temp,
            )
        return self.teacher_temp

    def _teacher_momentum(self, step: int) -> float:
        return _cosine_schedule(step, self.total_steps, self.momentum_teacher, self.final_momentum_teacher)

    def _set_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr = self.base_lr * float(step + 1) / float(self.warmup_steps)
        else:
            tail_step = max(step - self.warmup_steps, 0)
            tail_total = max(self.total_steps - self.warmup_steps, 1)
            lr = _cosine_schedule(tail_step, tail_total, self.base_lr, self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _center_teacher_cls(self, teacher_cls: torch.Tensor, teacher_temp: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.centering == "sinkhorn_knopp":
            teacher_targets = self.dino_loss.sinkhorn_knopp_teacher(teacher_cls, teacher_temp)
        else:
            teacher_targets = self.dino_loss.softmax_center_teacher(teacher_cls, teacher_temp)
            self.dino_loss.update_center(teacher_cls)
        return teacher_targets.chunk(2)

    def _center_teacher_patch(self, teacher_patch: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        if teacher_patch.numel() == 0:
            return teacher_patch
        if self.centering == "sinkhorn_knopp":
            n_masked = torch.tensor([teacher_patch.shape[0]], device=teacher_patch.device, dtype=torch.long)
            return self.ibot_patch_loss.sinkhorn_knopp_teacher(teacher_patch, teacher_temp, n_masked)
        teacher_patch_batched = teacher_patch.unsqueeze(0)
        targets = self.ibot_patch_loss.softmax_center_teacher(teacher_patch_batched, teacher_temp).squeeze(0)
        self.ibot_patch_loss.update_center(teacher_patch_batched)
        return targets

    def train_step(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        self.model.train()
        lr = self._set_lr(step)
        teacher_temp = self._teacher_temp(step)

        global_crops = batch["collated_global_crops"].to(self.device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(self.device, non_blocking=True)
        masks = batch["collated_masks"].to(self.device, non_blocking=True)
        mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
        n_local_views = int(batch["n_local_views"])
        n_masked = int(batch["n_masked_patches"].item())


        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            teacher_outputs = self.model._forward_branch(self.model.teacher, global_crops, masks=None)
            teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(teacher_outputs["cls_projections"], teacher_temp)
            teacher_patch = self.model.project_masked_patch_tokens(
                self.model.teacher,
                teacher_outputs["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            teacher_patch_targets = self._center_teacher_patch(teacher_patch, teacher_temp)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            student_global = self.model._forward_branch(self.model.student, global_crops, masks=masks)
            global_cls_0, global_cls_1 = student_global["cls_projections"].chunk(2)

            total_terms = 2 + (2 * n_local_views if n_local_views else 0)
            dino_global_loss = (
                self.dino_loss([global_cls_0], [teacher_cls_1]) +
                self.dino_loss([global_cls_1], [teacher_cls_0])
            ) / total_terms

            if n_local_views:
                student_local = self.model._forward_branch(self.model.student, local_crops, masks=None)
                local_cls_chunks = list(student_local["cls_projections"].chunk(n_local_views))
                dino_local_loss = self.dino_loss(local_cls_chunks, [teacher_cls_0, teacher_cls_1]) / total_terms
            else:
                dino_local_loss = global_crops.new_zeros(())

            if self.ibot_loss_weight > 0.0 and n_masked > 0:
                student_patch = self.model.project_masked_patch_tokens(
                    self.model.student,
                    student_global["patch_tokens"],
                    mask_indices,
                    n_masked_patches=n_masked,
                )
                ibot_loss = self.ibot_patch_loss.forward_masked(
                    student_patch,
                    teacher_patch_targets,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked,
                    masks_weight=masks_weight,
                )
            else:
                ibot_loss = global_crops.new_zeros(())

            if self.koleo_loss_weight > 0.0:
                koleo_loss = sum(self.koleo_loss(chunk) for chunk in student_global["cls_tokens"].chunk(2))
            else:
                koleo_loss = global_crops.new_zeros(())

            loss = (
                self.dino_loss_weight * (dino_global_loss + dino_local_loss) +
                self.ibot_loss_weight * ibot_loss +
                self.koleo_loss_weight * koleo_loss
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.student.parameters(), self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.update_teacher(self._teacher_momentum(step))

        return {
            "loss": float(loss.detach()),
            "dino_global_loss": float(dino_global_loss.detach()),
            "dino_local_loss": float(dino_local_loss.detach()),
            "ibot_loss": float(ibot_loss.detach()),
            "koleo_loss": float(koleo_loss.detach()),
            "lr": lr,
            "teacher_temp": teacher_temp,
        }

    def save_checkpoint(self, step: int) -> Path:
        path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "config": self.config,
                "student": self.model.student.state_dict(),
                "teacher": self.model.teacher.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def fit(self) -> None:
        dataloader = self.build_dataloader()
        dataloader_iter: Iterator[Any] = iter(dataloader)
        for step in range(self.num_iterations):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            metrics = self.train_step(batch, step)
            if step % self.log_every == 0:
                print(f"step={step} loss={metrics['loss']:.4f} lr={metrics['lr']:.2e}")
            if self.save_every and step > 0 and step % self.save_every == 0:
                self.save_checkpoint(step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal standalone 3D DINO+iBOT pretrainer")
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    trainer = DinoIBOTPretrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
