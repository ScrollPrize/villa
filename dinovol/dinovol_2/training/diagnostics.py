# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

"""DINO pipeline diagnostics used by dinovol_2.verify."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from dinovol_2.methods.dino import dino_loss_term_count

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
    gram_teacher_crops = batch.get("collated_gram_teacher_crops")
    if gram_teacher_crops is not None:
        gram_teacher_crops = gram_teacher_crops.to(self.device, non_blocking=True)
    masks = batch["collated_masks"].to(self.device, non_blocking=True)
    mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
    masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
    n_global_views = int(batch["n_global_views"])
    n_local_views = int(batch["n_local_views"])
    batch_size = int(batch["batch_size"])
    n_masked = int(batch["n_masked_patches"].item())

    self.model.eval()
    with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
        model_outputs = self._forward_model(
            global_crops=global_crops,
            local_crops=local_crops if n_local_views else None,
            masks=masks,
            mask_indices=mask_indices,
            n_masked=n_masked,
            return_teacher=self.do_dino or self.do_ibot,
        )
        teacher_branch = model_outputs.get("teacher")
        if teacher_branch is not None:
            teacher_outputs = teacher_branch["global"]
            teacher_cls_projections = teacher_branch["global_cls_projections"]
            teacher_patch = teacher_branch["global_masked_patch_projections"]
        else:
            teacher_outputs = None
            teacher_cls_projections = None
            teacher_patch = global_crops.new_zeros((0,))
        if self.do_dino and teacher_cls_projections is not None:
            teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(
                teacher_cls_projections,
                teacher_temp,
                update_centers=False,
            )
        else:
            teacher_cls_0 = teacher_cls_1 = None
        if self.do_ibot and teacher_branch is not None:
            teacher_patch_targets = self._center_teacher_patch(
                teacher_patch,
                teacher_temp,
                update_centers=False,
            )
        else:
            teacher_patch_targets = None

        student_branch = model_outputs["student"]
        student_global = student_branch["global"]
        student_global_cls = student_branch["global_cls_projections"]
        student_patch = student_branch["global_masked_patch_projections"]
        global_cls_0, global_cls_1 = student_global_cls.chunk(n_global_views)

        if n_local_views:
            student_local = student_branch["local"]
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
        gram_loss, gram_teacher_patch_tokens, resized_gram_teacher_patch_tokens = self._compute_gram_loss(
            student_patch_tokens=student_global["patch_tokens"],
            global_crops=global_crops,
            gram_teacher_crops=gram_teacher_crops,
        )
        point_result, point_embeddings = self._compute_point_loss(
            student_patch_tokens=student_global["patch_tokens"],
            batch=batch,
            global_crops=global_crops,
        )

        loss = (
            self.dino_loss_weight * (dino_global_loss + dino_local_loss) +
            self.ibot_loss_weight * ibot_loss +
            self.koleo_loss_weight * koleo_loss +
            self.gram_loss_weight * gram_loss +
            self.point_loss_weight * point_result.loss
        )

    backbone = self.model_module.student.backbone
    expected_global_shape = tuple(int(dim) for dim in global_crops.shape[2:])
    expected_local_shape = tuple(int(dim) for dim in local_crops.shape[2:]) if n_local_views else None
    try:
        global_feature_map_shape = self._feature_map_shape(backbone, expected_global_shape, view_kind="global")
        global_input_supported = True
    except ValueError:
        global_feature_map_shape = None
        global_input_supported = False
    try:
        local_feature_map_shape = (
            self._feature_map_shape(backbone, expected_local_shape, view_kind="local")
            if expected_local_shape is not None
            else None
        )
        local_input_supported = True
    except ValueError:
        local_feature_map_shape = None
        local_input_supported = False

    checks = {
        "global_rows_match_views": global_crops.shape[0] == batch_size * n_global_views,
        "local_rows_match_views": local_crops.shape[0] == batch_size * n_local_views,
        "mask_rows_match_global_rows": masks.shape[0] == global_crops.shape[0],
        "mask_width_matches_patch_tokens": masks.shape[1] == student_global["patch_tokens"].shape[1],
        "mask_indices_match_masked_count": mask_indices.numel() == n_masked,
        "mask_weights_match_masked_count": masks_weight.numel() == n_masked,
        "global_input_supported_by_model": global_input_supported,
        "local_input_supported_by_model": expected_local_shape is None or local_input_supported,
        "teacher_patch_count_matches_masked_count": teacher_patch.shape[0] == n_masked,
        "student_patch_count_matches_masked_count": student_patch.shape[0] == n_masked,
        "global_cls_chunks_match_batch": global_cls_0.shape[0] == batch_size and global_cls_1.shape[0] == batch_size,
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "point_loss_is_finite": bool(torch.isfinite(point_result.loss).item()),
    }
    if global_feature_map_shape is not None:
        checks["global_patch_grid_matches_resolved_shape"] = student_global["patch_tokens"].shape[1] == int(np.prod(global_feature_map_shape))
    else:
        checks["global_patch_grid_matches_resolved_shape"] = False
    if local_feature_map_shape is not None and student_local is not None:
        checks["local_patch_grid_matches_resolved_shape"] = student_local["patch_tokens"].shape[1] == int(np.prod(local_feature_map_shape))
    else:
        checks["local_patch_grid_matches_resolved_shape"] = student_local is None
    if self.do_gram:
        checks["gram_teacher_rows_match_views"] = (
            gram_teacher_crops is not None and gram_teacher_crops.shape[0] == batch_size * n_global_views
        )
        checks["gram_loss_is_finite"] = bool(torch.isfinite(gram_loss).item())
    else:
        checks["gram_teacher_rows_match_views"] = True
        checks["gram_loss_is_finite"] = True

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
            "global_crops": _tensor_stats(global_crops),
            "local_crops": _tensor_stats(local_crops),
            "gram_teacher_crops": _tensor_stats(gram_teacher_crops) if gram_teacher_crops is not None else None,
            "masks": _tensor_stats(masks),
            "mask_indices": _tensor_stats(mask_indices),
            "masks_weight": _tensor_stats(masks_weight),
            "point_coordinates": _tensor_stats(
                batch.get("collated_point_coordinates", torch.empty((0, 3)))
            ),
            "point_type_ids": _tensor_stats(
                batch.get("collated_point_type_ids", torch.empty((0,), dtype=torch.long))
            ),
            "point_rows": _tensor_stats(
                batch.get("collated_point_rows", torch.empty((0,), dtype=torch.long))
            ),
        },
        "teacher": {
            "cls_tokens": _tensor_stats(teacher_outputs["cls_tokens"]) if teacher_outputs is not None else None,
            "patch_tokens": _tensor_stats(teacher_outputs["patch_tokens"]) if teacher_outputs is not None else None,
            "cls_projections": _tensor_stats(teacher_cls_projections) if teacher_cls_projections is not None else None,
            "masked_patch_projections": _tensor_stats(teacher_patch),
            "cls_target_row_sums": teacher_cls_row_sums,
            "patch_target_row_sums": teacher_patch_row_sums,
        },
        "student": {
            "global_cls_tokens": _tensor_stats(student_global["cls_tokens"]),
            "global_patch_tokens": _tensor_stats(student_global["patch_tokens"]),
            "global_cls_projections": _tensor_stats(student_global_cls),
            "masked_patch_projections": _tensor_stats(student_patch),
            "local_cls_projections": _tensor_stats(student_local["cls_projections"]) if student_local else None,
            "local_patch_tokens": _tensor_stats(student_local["patch_tokens"]) if student_local else None,
        },
        "gram": {
            "enabled": self.do_gram,
            "loss_weight": float(self.gram_loss_weight),
            "raw_teacher_patch_tokens": _tensor_stats(gram_teacher_patch_tokens) if gram_teacher_patch_tokens is not None else None,
            "resized_teacher_patch_tokens": _tensor_stats(resized_gram_teacher_patch_tokens) if resized_gram_teacher_patch_tokens is not None else None,
        },
        "point_supervision": {
            "enabled": self.do_point_supervision,
            "loss_weight": float(self.point_loss_weight),
            "embeddings": _tensor_stats(point_embeddings),
            "point_count": point_result.point_count,
            "same_pair_count": point_result.same_pair_count,
            "different_pair_count": point_result.different_pair_count,
        },
        "losses": {
            "total": float(loss.detach().item()),
            "dino_global": float(dino_global_loss.detach().item()),
            "dino_local": float(dino_local_loss.detach().item()),
            "ibot": float(ibot_loss.detach().item()),
            "koleo": float(koleo_loss.detach().item()),
            "gram": float(gram_loss.detach().item()),
            "point": float(point_result.loss.detach().item()),
            "point_same_type": float(point_result.same_type.detach().item()),
            "point_different_type": float(point_result.different_type.detach().item()),
            "term_count": total_terms,
        },
        "checks": checks,
    }
    return report


def verify_train_step(trainer, batch: Mapping[str, Any], step: int = 0) -> dict[str, Any]:
    self = trainer.method
    forward_report = verify_batch_pipeline(self, batch, step=step)

    student_named_parameters = list(self.model_module.student.named_parameters())
    teacher_named_parameters = list(self.model_module.teacher.named_parameters())
    student_before = [parameter.detach().clone() for _, parameter in student_named_parameters]
    teacher_before = [parameter.detach().clone() for _, parameter in teacher_named_parameters]

    metrics = trainer.train_step(batch, step)

    student_after = [parameter.detach() for parameter in self.model_module.student.parameters()]
    teacher_after = [parameter.detach() for parameter in self.model_module.teacher.parameters()]

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


