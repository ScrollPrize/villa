from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

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
        self.max_iterations = int(self.config.get("max_iterations", self.config.get("num_iterations", 1000000)))
        self.total_steps = self.max_iterations
        self.base_lr = float(self.config.get("lr", 1e-4))
        self.min_lr = float(self.config.get("min_lr", 1e-6))
        self.base_weight_decay = float(self.config.get("weight_decay", 0.04))
        self.final_weight_decay = float(self.config.get("weight_decay_end", 0.4))
        self.betas = tuple(self.config.get("betas", (0.9, 0.999)))
        self.clip_grad = float(self.config.get("clip_grad", 3.0))

        warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        default_warmup_steps = 0
        if self.total_steps > 0 and warmup_ratio > 0.0:
            default_warmup_steps = max(1, round(self.total_steps * warmup_ratio))
        self.warmup_steps = int(self.config["warmup_steps"]) if "warmup_steps" in self.config else default_warmup_steps

        self.model = DinoVitStudentTeacher(self.model_config).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.device.type == "cuda")

        dino_out_dim = int(self.model_config.get("dino_out_dim", 65536))
        ibot_out_dim = int(self.model_config.get("ibot_out_dim", dino_out_dim))
        self.dino_loss = DINOLoss(dino_out_dim).to(self.device)
        self.ibot_patch_loss = iBOTPatchLoss(ibot_out_dim).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        self.dino_loss_weight = float(self.config.get("dino_loss_weight", 1.0))
        self.ibot_loss_weight = float(self.config.get("ibot_loss_weight", 1.0))
        self.koleo_loss_weight = float(self.config.get("koleo_loss_weight", 0.1))
        self.centering = str(self.config.get("centering", "centering"))

        self.teacher_temp = float(self.config.get("teacher_temp", 0.07))
        self.warmup_teacher_temp = float(self.config.get("warmup_teacher_temp", 0.04))
        self.warmup_teacher_temp_steps = int(
            self.config.get("warmup_teacher_temp_steps", round(self.total_steps * 0.3))
        )
        self.momentum_teacher = float(self.config.get("momentum_teacher", 0.992))
        self.final_momentum_teacher = float(self.config.get("final_momentum_teacher", 1.0))

        self.log_every = int(self.config.get("log_every", 20))
        self.val_every_n = int(self.config.get("val_every_n", 0))
        self.save_every_n = int(self.config.get("save_every_n", self.config.get("save_every", 0)))
        self.output_dir = Path(self.config.get("output_dir", "./dinov2_pretrain_runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_dir = self.output_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._warned_missing_val_dataset = False

    def _student_no_weight_decay_names(self) -> set[str]:
        no_decay_names: set[str] = set()
        for module_name, module in self.model.student.named_modules():
            if module is self.model.student or not hasattr(module, "no_weight_decay"):
                continue
            module_prefix = f"{module_name}." if module_name else ""
            no_decay_names.update(module_prefix + name for name in module.no_weight_decay())
        return no_decay_names

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay_names = self._student_no_weight_decay_names()
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for name, parameter in self.model.student.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "weight_decay": self.base_weight_decay,
                    "apply_weight_decay": True,
                }
            )
        if no_decay_params:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "apply_weight_decay": False,
                }
            )

        return torch.optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=self.betas,
        )

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

    def build_val_dataloader(self) -> DataLoader | None:
        val_dataset_config = self.config.get("val_dataset")
        if val_dataset_config is None:
            return None
        dataset = SSLZarrDataset(val_dataset_config, do_augmentations=True)
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

    def build_monitor_batch(self) -> dict[str, Any]:
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
        monitor_batch_size = int(self.config.get("monitor_batch_size", 2))
        samples = [dataset[i] for i in range(monitor_batch_size)]
        return collate_fn(samples)

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

    def _set_weight_decay(self, step: int) -> float:
        weight_decay = _cosine_schedule(step, self.total_steps, self.base_weight_decay, self.final_weight_decay)
        for group in self.optimizer.param_groups:
            if group.get("apply_weight_decay", True):
                group["weight_decay"] = weight_decay
        return weight_decay

    def _center_teacher_cls(
        self,
        teacher_cls: torch.Tensor,
        teacher_temp: float,
        *,
        update_centers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.centering == "sinkhorn_knopp":
            teacher_targets = self.dino_loss.sinkhorn_knopp_teacher(teacher_cls, teacher_temp)
        else:
            teacher_targets = self.dino_loss.softmax_center_teacher(teacher_cls, teacher_temp)
            if update_centers:
                self.dino_loss.update_center(teacher_cls)
        return teacher_targets.chunk(2)

    def _center_teacher_patch(
        self,
        teacher_patch: torch.Tensor,
        teacher_temp: float,
        *,
        update_centers: bool = True,
    ) -> torch.Tensor:
        if teacher_patch.numel() == 0:
            return teacher_patch
        if self.centering == "sinkhorn_knopp":
            n_masked = torch.tensor([teacher_patch.shape[0]], device=teacher_patch.device, dtype=torch.long)
            return self.ibot_patch_loss.sinkhorn_knopp_teacher(teacher_patch, teacher_temp, n_masked)
        teacher_patch_batched = teacher_patch.unsqueeze(0)
        targets = self.ibot_patch_loss.softmax_center_teacher(teacher_patch_batched, teacher_temp).squeeze(0)
        if update_centers:
            self.ibot_patch_loss.update_center(teacher_patch_batched)
        return targets

    def train_step(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        self.model.train()
        lr = self._set_lr(step)
        weight_decay = self._set_weight_decay(step)
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
            "weight_decay": weight_decay,
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

    @staticmethod
    def _normalize_image(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.uint8)
        scaled = (array - min_value) / (max_value - min_value)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _center_slice(volume: torch.Tensor) -> np.ndarray:
        array = volume.detach().cpu().float()
        if array.ndim == 4:
            depth = array.shape[1] // 2
            return array[0, depth].numpy()
        if array.ndim == 3:
            return array[0].numpy()
        raise ValueError(f"unexpected tensor shape for visualization: {tuple(array.shape)}")

    def _patch_norm_slice(self, patch_tokens: torch.Tensor, sample_index: int, target_hw: tuple[int, int]) -> np.ndarray:
        patch_size = self.model_config.get("patch_size", (8, 8, 8))
        global_crop = self.config["dataset"].get("global_crop_size", self.config["dataset"].get("crop_size"))
        if isinstance(global_crop, int):
            global_crop = (global_crop, global_crop, global_crop)
        feature_shape = tuple(int(size) // int(patch) for size, patch in zip(global_crop, patch_size))
        feature_map = patch_tokens[sample_index].reshape(*feature_shape, patch_tokens.shape[-1]).norm(dim=-1)
        depth = feature_map.shape[0] // 2
        heatmap = feature_map[depth][None, None].float()
        resized = F.interpolate(heatmap, size=target_hw, mode="bilinear", align_corners=False)
        return resized[0, 0].detach().cpu().numpy()

    def save_monitor_image(self, monitor_batch: Mapping[str, Any], step: int, metrics: Mapping[str, float]) -> Path:
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            global_crops = monitor_batch["collated_global_crops"].to(self.device, non_blocking=True)
            student_outputs = self.model._forward_branch(
                self.model.student,
                global_crops,
                masks=None,
                project_patch_tokens=False,
            )

        global_views = monitor_batch["collated_global_crops"]
        n_global_views = int(monitor_batch["n_global_views"])
        batch_size = int(monitor_batch["batch_size"])

        rows: list[np.ndarray] = []
        for sample_index in range(min(batch_size, 2)):
            panels: list[np.ndarray] = []
            for view_index in range(n_global_views):
                tensor_index = view_index * batch_size + sample_index
                center_slice = self._center_slice(global_views[tensor_index])
                heatmap = self._patch_norm_slice(
                    student_outputs["patch_tokens"],
                    tensor_index,
                    target_hw=center_slice.shape,
                )
                input_rgb = np.stack([self._normalize_image(center_slice)] * 3, axis=-1)
                heatmap_rgb = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
                heatmap_rgb[..., 0] = self._normalize_image(heatmap)
                heatmap_rgb[..., 1] = self._normalize_image(center_slice)
                panels.extend([input_rgb, heatmap_rgb])
            rows.append(np.concatenate(panels, axis=1))

        canvas = np.concatenate(rows, axis=0) if rows else np.zeros((256, 256, 3), dtype=np.uint8)
        image_path = self.monitor_dir / f"monitor_step_{step:06d}.jpg"
        Image.fromarray(canvas, mode="RGB").save(image_path, quality=90)
        print(
            f"step={step} monitor_image={image_path.name} "
            f"loss={metrics['loss']:.4f} glob={metrics['dino_global_loss']:.4f} "
            f"loc={metrics['dino_local_loss']:.4f} ibot={metrics['ibot_loss']:.4f} "
            f"koleo={metrics['koleo_loss']:.4f}"
        )
        return image_path

    def validate(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        self.model.eval()
        teacher_temp = self._teacher_temp(step)

        global_crops = batch["collated_global_crops"].to(self.device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(self.device, non_blocking=True)
        masks = batch["collated_masks"].to(self.device, non_blocking=True)
        mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
        n_local_views = int(batch["n_local_views"])
        n_masked = int(batch["n_masked_patches"].item())

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            teacher_outputs = self.model._forward_branch(self.model.teacher, global_crops, masks=None)
            teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(
                teacher_outputs["cls_projections"],
                teacher_temp,
                update_centers=False,
            )
            teacher_patch = self.model.project_masked_patch_tokens(
                self.model.teacher,
                teacher_outputs["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            teacher_patch_targets = self._center_teacher_patch(
                teacher_patch,
                teacher_temp,
                update_centers=False,
            )

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

        return {
            "loss": float(loss.detach()),
            "dino_global_loss": float(dino_global_loss.detach()),
            "dino_local_loss": float(dino_local_loss.detach()),
            "ibot_loss": float(ibot_loss.detach()),
            "koleo_loss": float(koleo_loss.detach()),
            "teacher_temp": teacher_temp,
        }

    def fit(self) -> None:
        dataloader = self.build_dataloader()
        dataloader_iter: Iterator[Any] = iter(dataloader)
        val_dataloader = self.build_val_dataloader()
        val_dataloader_iter: Iterator[Any] | None = iter(val_dataloader) if val_dataloader is not None else None
        monitor_batch = self.build_monitor_batch() if self.val_every_n else None
        with tqdm(total=self.max_iterations, desc="training", unit="iter") as progress:
            for step in range(self.max_iterations):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                metrics = self.train_step(batch, step)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    glob_loss=f"{metrics['dino_global_loss']:.4f}",
                    loc_loss=f"{metrics['dino_local_loss']:.4f}",
                    ibot_loss=f"{metrics['ibot_loss']:.4f}",
                    koleo_loss=f"{metrics['koleo_loss']:.4f}",
                )
                progress.update(1)

                if step % self.log_every == 0:
                    print(f"step={step} loss={metrics['loss']:.4f} lr={metrics['lr']:.2e}")
                if self.val_every_n and step > 0 and step % self.val_every_n == 0:
                    if val_dataloader_iter is None:
                        if not self._warned_missing_val_dataset:
                            print("val_every_n is set but no val_dataset is configured; skipping validation.")
                            self._warned_missing_val_dataset = True
                    else:
                        try:
                            val_batch = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            val_batch = next(val_dataloader_iter)
                        val_metrics = self.validate(val_batch, step)
                        print(f"step={step} val_loss={val_metrics['loss']:.4f}")
                    if monitor_batch is not None:
                        self.save_monitor_image(monitor_batch, step, metrics)
                if self.save_every_n and step > 0 and step % self.save_every_n == 0:
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
