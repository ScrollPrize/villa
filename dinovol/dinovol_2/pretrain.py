from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset

from dinovol_2.ops.collate import build_dino_ibot_collate_fn
from dinovol_2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from dinovol_2.model.model import DinoVitStudentTeacher


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return float(value[0]), float(value[1])


def dino_loss_term_count(n_local_views: int, n_global_views: int = 2) -> int:
    if n_local_views < 0:
        raise ValueError(f"n_local_views must be non-negative, got {n_local_views}")
    if n_global_views <= 0:
        raise ValueError(f"n_global_views must be positive, got {n_global_views}")
    
    n_local_terms = n_local_views * n_global_views
    n_global_terms = (n_global_views - 1) * n_global_views
    return n_global_terms + n_local_terms


class CosineScheduler:
    def __init__(
            self,
            *,
            base_value: float,
            final_value: float,
            total_iters: int,
            warmup_iters: int = 0,
            start_warmup_value: float = 0.0,
            freeze_iters: int = 0,
    ) -> None:
        self.final_value = float(final_value)
        self.total_iters = int(total_iters)
        
        if self.total_iters <= 0:
            self.schedule = np.zeros((0,), dtype=np.float64)
            return
        
        freeze_iters = max(0, min(int(freeze_iters), self.total_iters))
        warmup_iters = max(0, min(int(warmup_iters), self.total_iters - freeze_iters))
        
        freeze_schedule = np.zeros((freeze_iters,), dtype=np.float64)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, dtype=np.float64)
        
        cosine_iters = self.total_iters - warmup_iters - freeze_iters
        if cosine_iters > 0:
            iters = np.arange(cosine_iters, dtype=np.float64)
            cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
                        1.0 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_schedule = np.zeros((0,), dtype=np.float64)
        
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule))
        if len(self.schedule) != self.total_iters:
            raise AssertionError("invalid scheduler length")
    
    def __getitem__(self, step: int) -> float:
        if step >= self.total_iters:
            return self.final_value
        return float(self.schedule[step])


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
        self.layer_decay = float(self.config.get("layer_decay", self.config.get("layerwise_decay", 1.0)))
        self.patch_embed_lr_mult = float(self.config.get("patch_embed_lr_mult", 1.0))
        
        warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        default_warmup_steps = 0
        if self.total_steps > 0 and warmup_ratio > 0.0:
            default_warmup_steps = max(1, round(self.total_steps * warmup_ratio))
        self.warmup_steps = int(self.config["warmup_steps"]) if "warmup_steps" in self.config else default_warmup_steps
        self.freeze_last_layer_steps = self._resolve_step_count(
            "freeze_last_layer_steps",
            "freeze_last_layer_epochs",
            default=0,
        )
        
        self.model = DinoVitStudentTeacher(self.model_config).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.device.type == "cuda")
        
        dino_out_dim = int(self.model_config.get("dino_out_dim", 65536))
        if "ibot_out_dim" in self.model_config and int(self.model_config["ibot_out_dim"]) != dino_out_dim:
            raise ValueError("Shared DINO/iBOT head requires ibot_out_dim to match dino_out_dim.")
        self.dino_loss = DINOLoss(dino_out_dim).to(self.device)
        masked_loss_chunk_size = self.config.get("ibot_masked_loss_chunk_size")
        if masked_loss_chunk_size is not None:
            masked_loss_chunk_size = int(masked_loss_chunk_size)
        self.ibot_patch_loss = iBOTPatchLoss(
            dino_out_dim,
            masked_loss_chunk_size=masked_loss_chunk_size,
        ).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)
        
        self.dino_loss_weight = float(self.config.get("dino_loss_weight", 1.0))
        self.ibot_loss_weight = float(self.config.get("ibot_loss_weight", 1.0))
        self.koleo_loss_weight = float(self.config.get("koleo_loss_weight", 0.1))
        self.do_dino = self.dino_loss_weight > 0.0
        self.do_ibot = self.ibot_loss_weight > 0.0
        self.do_koleo = self.koleo_loss_weight > 0.0 and self.do_dino
        self.centering = str(self.config.get("centering", "centering"))
        
        self.teacher_temp = float(self.config.get("teacher_temp", 0.07))
        self.warmup_teacher_temp = float(self.config.get("warmup_teacher_temp", 0.04))
        self.warmup_teacher_temp_steps = int(
            self.config.get("warmup_teacher_temp_steps", round(self.total_steps * 0.3))
        )
        self.momentum_teacher = float(self.config.get("momentum_teacher", 0.992))
        self.final_momentum_teacher = float(self.config.get("final_momentum_teacher", 1.0))
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule,
        ) = self._build_schedulers()
        
        self.log_every = int(self.config.get("log_every", 20))
        self.val_every_n = int(self.config.get("val_every_n", 0))
        self.save_every_n = int(self.config.get("save_every_n", self.config.get("save_every", 0)))
        self.monitor_batch_size = int(self.config.get("monitor_batch_size", 2))
        self.monitor_pool_size = max(1, int(self.config.get("monitor_pool_size", 5000)))
        self.monitor_seed = int(self.config.get("monitor_seed", 0))
        self.output_dir = Path(self.config.get("output_dir", "./dinov2_pretrain_runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_dir = self.output_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._warned_missing_val_dataset = False
        self.resume = bool(self.config.get("resume", False))
        self.auto_resume = bool(self.config.get("auto_resume", self.resume or bool(self.config.get("resume_from"))))
        self._monitor_dataset: SSLZarrDataset | None = None
        self._monitor_collate_fn: Any | None = None
        self._monitor_seed_pool = tuple(self.monitor_seed + offset for offset in range(self.monitor_pool_size))
        self._monitor_selection_rng = random.Random(self.monitor_seed)
    
    def _resolve_step_count(self, steps_key: str, epochs_key: str, *, default: int) -> int:
        if steps_key in self.config:
            return int(self.config[steps_key])
        if epochs_key in self.config:
            epoch_length = self.config.get("official_epoch_length", self.config.get("epoch_length"))
            if epoch_length is None:
                raise ValueError(f"{epochs_key} requires official_epoch_length or epoch_length in the config.")
            return int(self.config[epochs_key]) * int(epoch_length)
        return int(default)
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = self.model.get_params_groups(
            lr_decay_rate=self.layer_decay,
            patch_embed_lr_mult=self.patch_embed_lr_mult,
        )
        return torch.optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=self.betas,
        )
    
    def _build_schedulers(self) -> tuple[
        CosineScheduler, CosineScheduler, CosineScheduler, CosineScheduler, CosineScheduler]:
        lr_schedule = CosineScheduler(
            base_value=self.base_lr,
            final_value=self.min_lr,
            total_iters=self.total_steps,
            warmup_iters=self.warmup_steps,
            start_warmup_value=0.0,
        )
        wd_schedule = CosineScheduler(
            base_value=self.base_weight_decay,
            final_value=self.final_weight_decay,
            total_iters=self.total_steps,
        )
        momentum_schedule = CosineScheduler(
            base_value=self.momentum_teacher,
            final_value=self.final_momentum_teacher,
            total_iters=self.total_steps,
        )
        teacher_temp_schedule = CosineScheduler(
            base_value=self.teacher_temp,
            final_value=self.teacher_temp,
            total_iters=self.total_steps,
            warmup_iters=self.warmup_teacher_temp_steps,
            start_warmup_value=self.warmup_teacher_temp,
        )
        last_layer_lr_schedule = CosineScheduler(
            base_value=self.base_lr,
            final_value=self.min_lr,
            total_iters=self.total_steps,
            warmup_iters=self.warmup_steps,
            start_warmup_value=0.0,
        )
        last_layer_lr_schedule.schedule[: self.freeze_last_layer_steps] = 0.0
        return (
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            teacher_temp_schedule,
            last_layer_lr_schedule,
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
    
    def _get_monitor_source(self) -> tuple[SSLZarrDataset, Any]:
        if self._monitor_dataset is None or self._monitor_collate_fn is None:
            monitor_dataset_config = self.config.get("monitor_dataset")
            if monitor_dataset_config is None:
                monitor_dataset_config = self.config.get("val_dataset", self.config["dataset"])
            dataset = SSLZarrDataset(monitor_dataset_config, do_augmentations=True)
            collate_fn = build_dino_ibot_collate_fn(
                {
                    "global_crop_size": dataset.global_crop_size,
                    "patch_size": self.model_config.get("patch_size", (8, 8, 8)),
                    "mask_ratio_min_max": _as_float_pair(self.config.get("mask_ratio_min_max"), (0.1, 0.5)),
                    "mask_sample_probability": float(self.config.get("mask_sample_probability", 0.5)),
                    "dtype": torch.float32,
                }
            )
            self._monitor_dataset = dataset
            self._monitor_collate_fn = collate_fn
        return self._monitor_dataset, self._monitor_collate_fn
    
    def build_monitor_batch(self, seed: int | None = None) -> dict[str, Any]:
        dataset, collate_fn = self._get_monitor_source()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            samples = [dataset[i] for i in range(self.monitor_batch_size)]
            return collate_fn(samples)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    
    def _next_monitor_seed(self) -> int:
        return self._monitor_seed_pool[self._monitor_selection_rng.randrange(len(self._monitor_seed_pool))]
    
    def sample_monitor_batch(self) -> dict[str, Any]:
        return self.build_monitor_batch(seed=self._next_monitor_seed())
    
    def _teacher_temp(self, step: int) -> float:
        return self.teacher_temp_schedule[step]
    
    def _apply_optim_scheduler(self, step: int) -> tuple[float, float, float, float]:
        lr = self.lr_schedule[step]
        weight_decay = self.wd_schedule[step]
        teacher_temp = self.teacher_temp_schedule[step]
        last_layer_lr = self.last_layer_lr_schedule[step]
        for group in self.optimizer.param_groups:
            group["weight_decay"] = weight_decay * group.get("wd_multiplier", 1.0)
            group["lr"] = (last_layer_lr if group.get("is_last_layer", False) else lr) * group.get("lr_multiplier", 1.0)
        return lr, weight_decay, self.momentum_schedule[step], teacher_temp
    
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
    
    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
        }
        if tensor.numel() == 0:
            stats["finite"] = True
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["std"] = None
            return stats
        
        if tensor.dtype == torch.bool:
            stats["finite"] = True
            stats["true_count"] = int(tensor.sum().item())
            stats["false_count"] = int((~tensor).sum().item())
            return stats
        
        finite = torch.isfinite(tensor)
        stats["finite"] = bool(finite.all().item())
        values = tensor.detach().float()
        stats["min"] = float(values.min().item())
        stats["max"] = float(values.max().item())
        stats["mean"] = float(values.mean().item())
        stats["std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        return stats
    
    def verify_batch_pipeline(self, batch: Mapping[str, Any], step: int = 0) -> dict[str, Any]:
        teacher_temp = self._teacher_temp(step)
        global_crops = batch["collated_global_crops"].to(self.device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(self.device, non_blocking=True)
        masks = batch["collated_masks"].to(self.device, non_blocking=True)
        mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
        n_global_views = int(batch["n_global_views"])
        n_local_views = int(batch["n_local_views"])
        batch_size = int(batch["batch_size"])
        n_masked = int(batch["n_masked_patches"].item())
        
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            teacher_outputs = self.model._forward_branch(
                self.model.teacher,
                global_crops,
                masks=None,
                project_cls_tokens=False,
            )
            teacher_cls_projections, teacher_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.teacher,
                teacher_outputs["cls_tokens"],
                teacher_outputs["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            if self.do_dino:
                teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(
                    teacher_cls_projections,
                    teacher_temp,
                    update_centers=False,
                )
            else:
                teacher_cls_0 = teacher_cls_1 = None
            if self.do_ibot:
                teacher_patch_targets = self._center_teacher_patch(
                    teacher_patch,
                    teacher_temp,
                    update_centers=False,
                )
            else:
                teacher_patch_targets = None
            
            student_global = self.model._forward_branch(
                self.model.student,
                global_crops,
                masks=masks,
                project_cls_tokens=False,
            )
            student_global_cls, student_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.student,
                student_global["cls_tokens"],
                student_global["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            global_cls_0, global_cls_1 = student_global_cls.chunk(n_global_views)
            
            if n_local_views:
                student_local = self.model._forward_branch(self.model.student, local_crops, masks=None)
                local_cls_chunks = list(student_local["cls_projections"].chunk(n_local_views))
            else:
                student_local = None
                local_cls_chunks = []
            
            total_terms = dino_loss_term_count(n_local_views, n_global_views=n_global_views)
            if self.do_dino:
                dino_global_loss = (
                                           self.dino_loss([global_cls_0], [teacher_cls_1]) +
                                           self.dino_loss([global_cls_1], [teacher_cls_0])
                                   ) / total_terms
                if n_local_views:
                    dino_local_loss = self.dino_loss(local_cls_chunks, [teacher_cls_0, teacher_cls_1]) / total_terms
                else:
                    dino_local_loss = global_crops.new_zeros(())
            else:
                dino_global_loss = global_crops.new_zeros(())
                dino_local_loss = global_crops.new_zeros(())
            
            if self.do_ibot and n_masked > 0:
                ibot_loss = self.ibot_patch_loss.forward_masked(
                    student_patch,
                    teacher_patch_targets,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked,
                    masks_weight=masks_weight,
                )
            else:
                ibot_loss = global_crops.new_zeros(())
            
            if self.do_koleo:
                koleo_loss = sum(self.koleo_loss(chunk) for chunk in student_global["cls_tokens"].chunk(n_global_views))
            else:
                koleo_loss = global_crops.new_zeros(())
            
            loss = (
                    self.dino_loss_weight * (dino_global_loss + dino_local_loss) +
                    self.ibot_loss_weight * ibot_loss +
                    self.koleo_loss_weight * koleo_loss
            )
        
        backbone = self.model.student.backbone
        expected_global_shape = tuple(int(dim) for dim in global_crops.shape[2:])
        expected_local_shape = tuple(int(dim) for dim in local_crops.shape[2:]) if n_local_views else None
        global_config_shape = tuple(int(dim) for dim in backbone.global_crops_size)
        local_config_shape = tuple(int(dim) for dim in backbone.local_crops_size)
        
        checks = {
            "global_rows_match_views": global_crops.shape[0] == batch_size * n_global_views,
            "local_rows_match_views": local_crops.shape[0] == batch_size * n_local_views,
            "mask_rows_match_global_rows": masks.shape[0] == global_crops.shape[0],
            "mask_width_matches_patch_tokens": masks.shape[1] == student_global["patch_tokens"].shape[1],
            "mask_indices_match_masked_count": mask_indices.numel() == n_masked,
            "mask_weights_match_masked_count": masks_weight.numel() == n_masked,
            "global_input_matches_model_config": expected_global_shape == global_config_shape,
            "local_input_matches_model_config": expected_local_shape is None or expected_local_shape == local_config_shape,
            "teacher_patch_count_matches_masked_count": teacher_patch.shape[0] == n_masked,
            "student_patch_count_matches_masked_count": student_patch.shape[0] == n_masked,
            "global_cls_chunks_match_batch": global_cls_0.shape[0] == batch_size and global_cls_1.shape[
                0] == batch_size,
            "loss_is_finite": bool(torch.isfinite(loss).item()),
        }
        
        teacher_cls_row_sums = None
        teacher_patch_row_sums = None
        if teacher_cls_0 is not None and teacher_cls_1 is not None:
            cls_row_sums = torch.cat((teacher_cls_0.sum(dim=-1), teacher_cls_1.sum(dim=-1)))
            teacher_cls_row_sums = {
                "min": float(cls_row_sums.min().item()),
                "max": float(cls_row_sums.max().item()),
                "mean": float(cls_row_sums.mean().item()),
            }
            checks["teacher_cls_targets_sum_to_one"] = bool(torch.allclose(
                cls_row_sums,
                torch.ones_like(cls_row_sums),
                atol=1e-4,
                rtol=1e-4,
            ))
        if teacher_patch_targets is not None and teacher_patch_targets.numel() > 0:
            patch_row_sums = teacher_patch_targets.sum(dim=-1)
            teacher_patch_row_sums = {
                "min": float(patch_row_sums.min().item()),
                "max": float(patch_row_sums.max().item()),
                "mean": float(patch_row_sums.mean().item()),
            }
            checks["teacher_patch_targets_sum_to_one"] = bool(torch.allclose(
                patch_row_sums,
                torch.ones_like(patch_row_sums),
                atol=1e-4,
                rtol=1e-4,
            ))
        else:
            checks["teacher_patch_targets_sum_to_one"] = True
        
        checks["all_passed"] = all(checks.values())
        
        report: dict[str, Any] = {
            "step": int(step),
            "teacher_temp": float(teacher_temp),
            "batch": {
                "n_global_views": n_global_views,
                "n_local_views": n_local_views,
                "batch_size": batch_size,
                "n_masked_patches": n_masked,
                "global_crops": self._tensor_stats(global_crops),
                "local_crops": self._tensor_stats(local_crops),
                "masks": self._tensor_stats(masks),
                "mask_indices": self._tensor_stats(mask_indices),
                "masks_weight": self._tensor_stats(masks_weight),
            },
            "teacher": {
                "cls_tokens": self._tensor_stats(teacher_outputs["cls_tokens"]),
                "patch_tokens": self._tensor_stats(teacher_outputs["patch_tokens"]),
                "cls_projections": self._tensor_stats(teacher_cls_projections),
                "masked_patch_projections": self._tensor_stats(teacher_patch),
                "cls_target_row_sums": teacher_cls_row_sums,
                "patch_target_row_sums": teacher_patch_row_sums,
            },
            "student": {
                "global_cls_tokens": self._tensor_stats(student_global["cls_tokens"]),
                "global_patch_tokens": self._tensor_stats(student_global["patch_tokens"]),
                "global_cls_projections": self._tensor_stats(student_global_cls),
                "masked_patch_projections": self._tensor_stats(student_patch),
                "local_cls_projections": self._tensor_stats(
                    student_local["cls_projections"]) if student_local else None,
            },
            "losses": {
                "total": float(loss.detach().item()),
                "dino_global": float(dino_global_loss.detach().item()),
                "dino_local": float(dino_local_loss.detach().item()),
                "ibot": float(ibot_loss.detach().item()),
                "koleo": float(koleo_loss.detach().item()),
                "term_count": total_terms,
            },
            "checks": checks,
        }
        return report
    
    def verify_train_step(self, batch: Mapping[str, Any], step: int = 0) -> dict[str, Any]:
        forward_report = self.verify_batch_pipeline(batch, step=step)
        
        student_named_parameters = list(self.model.student.named_parameters())
        teacher_named_parameters = list(self.model.teacher.named_parameters())
        student_before = [parameter.detach().clone() for _, parameter in student_named_parameters]
        teacher_before = [parameter.detach().clone() for _, parameter in teacher_named_parameters]
        
        metrics = self.train_step(batch, step)
        
        student_after = [parameter.detach() for parameter in self.model.student.parameters()]
        teacher_after = [parameter.detach() for parameter in self.model.teacher.parameters()]
        
        student_delta_norm_sq = 0.0
        teacher_delta_norm_sq = 0.0
        student_teacher_gap_sq = 0.0
        student_grad_norm_sq = 0.0
        student_grad_params = 0
        teacher_grad_params = 0
        student_changed_params = 0
        teacher_changed_params = 0
        student_nonfinite_grad_names: list[str] = []
        teacher_nonfinite_grad_names: list[str] = []
        
        for (name, parameter), before, after in zip(student_named_parameters, student_before, student_after):
            delta = (after - before).float()
            delta_norm = float(torch.sum(delta * delta).item())
            student_delta_norm_sq += delta_norm
            if delta_norm > 0.0:
                student_changed_params += 1
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                student_grad_params += 1
                if torch.isfinite(grad).all():
                    student_grad_norm_sq += float(torch.sum(grad * grad).item())
                elif len(student_nonfinite_grad_names) < 10:
                    student_nonfinite_grad_names.append(name)
        
        for (name, parameter), before, after in zip(teacher_named_parameters, teacher_before, teacher_after):
            delta = (after - before).float()
            delta_norm = float(torch.sum(delta * delta).item())
            teacher_delta_norm_sq += delta_norm
            if delta_norm > 0.0:
                teacher_changed_params += 1
            if parameter.grad is not None:
                teacher_grad_params += 1
                if not torch.isfinite(parameter.grad.detach()).all() and len(teacher_nonfinite_grad_names) < 10:
                    teacher_nonfinite_grad_names.append(name)
        
        for student_parameter, teacher_parameter in zip(student_after, teacher_after):
            gap = (student_parameter - teacher_parameter).float()
            student_teacher_gap_sq += float(torch.sum(gap * gap).item())
        
        update_checks = {
            "loss_is_finite": all(np.isfinite(float(value)) for value in metrics.values()),
            "student_has_gradients": student_grad_params > 0,
            "student_gradients_are_finite": len(student_nonfinite_grad_names) == 0,
            "teacher_has_no_gradients": teacher_grad_params == 0,
            "student_parameters_changed": student_changed_params > 0,
            "teacher_parameters_changed_via_ema": teacher_changed_params > 0,
            "student_and_teacher_diverged_after_step": student_teacher_gap_sq > 0.0,
        }
        update_checks["all_passed"] = all(update_checks.values())
        
        return {
            "step": int(step),
            "forward": forward_report,
            "train_step": {
                "metrics": {key: float(value) for key, value in metrics.items()},
                "student_delta_l2": float(student_delta_norm_sq ** 0.5),
                "teacher_delta_l2": float(teacher_delta_norm_sq ** 0.5),
                "student_teacher_gap_l2": float(student_teacher_gap_sq ** 0.5),
                "student_grad_l2": float(student_grad_norm_sq ** 0.5),
                "student_grad_parameter_count": int(student_grad_params),
                "teacher_grad_parameter_count": int(teacher_grad_params),
                "student_nonfinite_grad_names": student_nonfinite_grad_names,
                "teacher_nonfinite_grad_names": teacher_nonfinite_grad_names,
                "student_changed_parameter_count": int(student_changed_params),
                "teacher_changed_parameter_count": int(teacher_changed_params),
                "checks": update_checks,
            },
        }
    
    def train_step(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        self.model.train()
        lr, weight_decay, teacher_momentum, teacher_temp = self._apply_optim_scheduler(step)
        
        global_crops = batch["collated_global_crops"].to(self.device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(self.device, non_blocking=True)
        masks = batch["collated_masks"].to(self.device, non_blocking=True)
        mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
        n_local_views = int(batch["n_local_views"])
        n_masked = int(batch["n_masked_patches"].item())
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            teacher_outputs = self.model._forward_branch(
                self.model.teacher,
                global_crops,
                masks=None,
                project_cls_tokens=False,
            )
            teacher_cls_projections, teacher_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.teacher,
                teacher_outputs["cls_tokens"],
                teacher_outputs["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            if self.do_dino:
                teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(teacher_cls_projections, teacher_temp)
            if self.do_ibot:
                teacher_patch_targets = self._center_teacher_patch(teacher_patch, teacher_temp)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            student_global = self.model._forward_branch(
                self.model.student,
                global_crops,
                masks=masks,
                project_cls_tokens=False,
            )
            student_global_cls, student_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.student,
                student_global["cls_tokens"],
                student_global["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            global_cls_0, global_cls_1 = student_global_cls.chunk(2)
            
            total_terms = dino_loss_term_count(n_local_views)
            if self.do_dino:
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
            else:
                dino_global_loss = global_crops.new_zeros(())
                dino_local_loss = global_crops.new_zeros(())
            
            if self.do_ibot and n_masked > 0:
                ibot_loss = self.ibot_patch_loss.forward_masked(
                    student_patch,
                    teacher_patch_targets,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked,
                    masks_weight=masks_weight,
                )
            else:
                ibot_loss = global_crops.new_zeros(())
            
            if self.do_koleo:
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
        self.model.update_teacher(teacher_momentum)
        
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
    
    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state
    
    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    
    def _optimizer_to_device(self) -> None:
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
    
    def save_checkpoint(self, step: int) -> Path:
        path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "config": self.config,
                "student": self.model.student.state_dict(),
                "teacher": self.model.teacher.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                "dino_loss": self.dino_loss.state_dict(),
                "ibot_patch_loss": self.ibot_patch_loss.state_dict(),
                "rng_state": self._capture_rng_state(),
            },
            path,
        )
        return path
    
    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.student.load_state_dict(checkpoint["student"])
        self.model.teacher.load_state_dict(checkpoint["teacher"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._optimizer_to_device()
        if checkpoint.get("scaler") is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler"])
        if "dino_loss" in checkpoint:
            self.dino_loss.load_state_dict(checkpoint["dino_loss"])
        if "ibot_patch_loss" in checkpoint:
            self.ibot_patch_loss.load_state_dict(checkpoint["ibot_patch_loss"])
        if "rng_state" in checkpoint:
            self._restore_rng_state(checkpoint["rng_state"])
        return int(checkpoint.get("step", -1))
    
    def _find_latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        return checkpoints[-1]
    
    def _resolve_resume_path(self) -> Path | None:
        resume_from = self.config.get("resume_from")
        if resume_from:
            return Path(resume_from)
        if self.auto_resume:
            return self._find_latest_checkpoint()
        return None
    
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
    
    def _patch_pca_slice(self, patch_tokens: torch.Tensor, sample_index: int, target_hw: tuple[int, int]) -> np.ndarray:
        patch_size = self.model_config.get("patch_size", (8, 8, 8))
        global_crop = self.config["dataset"].get("global_crop_size", self.config["dataset"].get("crop_size"))
        if isinstance(global_crop, int):
            global_crop = (global_crop, global_crop, global_crop)
        feature_shape = tuple(int(size) // int(patch) for size, patch in zip(global_crop, patch_size))
        feature_map = patch_tokens[sample_index].reshape(*feature_shape, patch_tokens.shape[-1])
        depth = feature_map.shape[0] // 2
        slice_features = feature_map[depth].float()
        h, w, c = slice_features.shape
        flat = slice_features.reshape(h * w, c)
        flat = flat - flat.mean(dim=0)
        _, _, V = torch.pca_lowrank(flat, q=3)
        projected = (flat @ V[:, :3]).reshape(h, w, 3).detach().cpu().numpy()
        result = np.stack([self._normalize_image(projected[..., i]) for i in range(3)], axis=-1)
        resized = F.interpolate(
            torch.from_numpy(result).permute(2, 0, 1).float()[None],
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized[0].permute(1, 2, 0).numpy().astype(np.uint8)
    
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
                pca_rgb = self._patch_pca_slice(
                    student_outputs["patch_tokens"],
                    tensor_index,
                    target_hw=center_slice.shape,
                )
                input_rgb = np.stack([self._normalize_image(center_slice)] * 3, axis=-1)
                panels.extend([input_rgb, pca_rgb])
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
            teacher_outputs = self.model._forward_branch(
                self.model.teacher,
                global_crops,
                masks=None,
                project_cls_tokens=False,
            )
            teacher_cls_projections, teacher_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.teacher,
                teacher_outputs["cls_tokens"],
                teacher_outputs["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(
                teacher_cls_projections,
                teacher_temp,
                update_centers=False,
            ) if self.do_dino else (None, None)
            teacher_patch_targets = self._center_teacher_patch(
                teacher_patch,
                teacher_temp,
                update_centers=False,
            ) if self.do_ibot else None
            
            student_global = self.model._forward_branch(
                self.model.student,
                global_crops,
                masks=masks,
                project_cls_tokens=False,
            )
            student_global_cls, student_patch = self.model.project_cls_and_masked_patch_tokens(
                self.model.student,
                student_global["cls_tokens"],
                student_global["patch_tokens"],
                mask_indices,
                n_masked_patches=n_masked,
            )
            global_cls_0, global_cls_1 = student_global_cls.chunk(2)
            
            total_terms = dino_loss_term_count(n_local_views)
            if self.do_dino:
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
            else:
                dino_global_loss = global_crops.new_zeros(())
                dino_local_loss = global_crops.new_zeros(())
            
            if self.do_ibot and n_masked > 0:
                ibot_loss = self.ibot_patch_loss.forward_masked(
                    student_patch,
                    teacher_patch_targets,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked,
                    masks_weight=masks_weight,
                )
            else:
                ibot_loss = global_crops.new_zeros(())
            
            if self.do_koleo:
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
        start_step = 0
        resume_path = self._resolve_resume_path()
        if resume_path is not None:
            start_step = self.load_checkpoint(resume_path) + 1
        
        dataloader = self.build_dataloader()
        dataloader_iter: Iterator[Any] = iter(dataloader)
        val_dataloader = self.build_val_dataloader()
        val_dataloader_iter: Iterator[Any] | None = iter(val_dataloader) if val_dataloader is not None else None
        with tqdm(total=self.max_iterations, initial=start_step, desc="training", unit="iter") as progress:
            for step in range(start_step, self.max_iterations):
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
                    self.save_monitor_image(self.sample_monitor_batch(), step, metrics)
                if self.save_every_n and step > 0 and step % self.save_every_n == 0:
                    self.save_checkpoint(step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal standalone 3D DINO+iBOT pretrainer")
    parser.add_argument("config", type=str)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from
        config["resume"] = True
    if args.no_resume:
        config["resume"] = False
        config["auto_resume"] = False
        config.pop("resume_from", None)
    
    trainer = DinoIBOTPretrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()