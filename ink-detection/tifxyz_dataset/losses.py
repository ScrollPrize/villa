from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vesuvius.models.training.loss.nnunet_losses import LabelSmoothedDCAndBCELoss


_BETTI_MODULE = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _betti_build_candidates() -> List[Path]:
    root = _repo_root()
    return [
        root / "external" / "Betti-Matching-3D" / "build",
        root / "src" / "external" / "Betti-Matching-3D" / "build",
        root / "scratch" / "Betti-Matching-3D" / "build",
    ]


def _load_betti_module() -> object:
    candidate_build_dirs = _betti_build_candidates()
    betti_build_path = next((p for p in candidate_build_dirs if p.exists()), None)
    if betti_build_path is None:
        checked = "\n".join(f"  - {path}" for path in candidate_build_dirs)
        raise ImportError(
            "Betti-Matching-3D build directory not found. Checked:\n"
            f"{checked}\n"
            "Build it with:\n"
            "  python tifxyz_dataset/build_betti.py"
        )

    candidates: List[Path] = []
    for pattern in ("betti_matching*.so", "betti_matching*.pyd", "betti_matching*.dll"):
        candidates.extend(betti_build_path.rglob(pattern))
    if not candidates:
        raise ImportError(
            f"betti_matching extension not found under {betti_build_path}. "
            "Re-run:\n  python tifxyz_dataset/build_betti.py"
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("betti_matching", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load betti_matching from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_betti_module() -> object:
    global _BETTI_MODULE
    if _BETTI_MODULE is None:
        _BETTI_MODULE = _load_betti_module()
    return _BETTI_MODULE


def _to_numpy(fields: List[torch.Tensor]) -> List[np.ndarray]:
    return [np.ascontiguousarray(field.detach().cpu().numpy().astype(np.float64)) for field in fields]


def _tensor_values_at_coords(t: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    if coords.size == 0:
        return t.new_zeros((0,), dtype=t.dtype)
    idx = torch.as_tensor(coords, device=t.device, dtype=torch.long)
    return t[tuple(idx[:, d] for d in range(idx.shape[1]))]


def _stack_pairs(values_birth: torch.Tensor, values_death: torch.Tensor) -> torch.Tensor:
    if values_birth.numel() == 0:
        return values_birth.new_zeros((0, 2))
    return torch.stack([values_birth, values_death], dim=1)


def _loss_unmatched(pairs: torch.Tensor, push_to: str = "diagonal") -> torch.Tensor:
    if pairs.numel() == 0:
        return pairs.new_zeros(())
    if push_to == "diagonal":
        return ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    if push_to == "one_zero":
        return 2.0 * (((pairs[:, 0] - 1.0) ** 2) + (pairs[:, 1] ** 2)).sum()
    if push_to == "death_death":
        return 2.0 * ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    raise ValueError(f"Unsupported push_to mode: {push_to!r}")


def _concat_result_arrays(list_of_arrays, ndim: int) -> np.ndarray:
    flat: List[np.ndarray] = []
    if list_of_arrays is not None:
        for arr in list_of_arrays:
            if arr is None:
                continue
            if isinstance(arr, (list, tuple)):
                for inner in arr:
                    if isinstance(inner, np.ndarray) and inner.size > 0:
                        flat.append(inner)
            elif isinstance(arr, np.ndarray) and arr.size > 0:
                flat.append(arr)
    if not flat:
        return np.zeros((0, ndim), dtype=np.int64)
    return np.ascontiguousarray(np.concatenate(flat, axis=0))


def _compute_loss_from_result(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    result,
    *,
    include_unmatched_target: bool,
    push_to: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ndim = pred_field.ndim
    pred_birth_coords = _concat_result_arrays(result.input1_matched_birth_coordinates, ndim)
    pred_death_coords = _concat_result_arrays(result.input1_matched_death_coordinates, ndim)
    tgt_birth_coords = _concat_result_arrays(result.input2_matched_birth_coordinates, ndim)
    tgt_death_coords = _concat_result_arrays(result.input2_matched_death_coordinates, ndim)

    pred_unmatched_birth = _concat_result_arrays(result.input1_unmatched_birth_coordinates, ndim)
    pred_unmatched_death = _concat_result_arrays(result.input1_unmatched_death_coordinates, ndim)
    tgt_unmatched_birth = _concat_result_arrays(result.input2_unmatched_birth_coordinates, ndim)
    tgt_unmatched_death = _concat_result_arrays(result.input2_unmatched_death_coordinates, ndim)

    pred_birth_vals = _tensor_values_at_coords(pred_field, pred_birth_coords)
    pred_death_vals = _tensor_values_at_coords(pred_field, pred_death_coords)
    tgt_birth_vals = _tensor_values_at_coords(tgt_field, tgt_birth_coords)
    tgt_death_vals = _tensor_values_at_coords(tgt_field, tgt_death_coords)

    pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
    tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
    loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, pred_unmatched_birth)
    pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, pred_unmatched_death)
    pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
    loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred
    loss_unmatched_tgt = pred_field.new_zeros(())
    if include_unmatched_target and tgt_unmatched_birth.size > 0:
        tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_birth)
        tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_death)
        tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
        loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
        total = total + loss_unmatched_tgt

    aux = {
        "betti/matched": loss_matched.reshape(1).detach(),
        "betti/unmatched_pred": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["betti/unmatched_target"] = loss_unmatched_tgt.reshape(1).detach()
    return total.reshape(1), aux


def _filter_coords_by_mask(
    coords: np.ndarray,
    mask: Optional[torch.Tensor],
    threshold: float = 0.5,
) -> np.ndarray:
    if mask is None or coords.size == 0:
        return np.ones(coords.shape[0], dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask-aware projected Betti loss expects 2D masks, got shape {tuple(mask.shape)}")

    mask_np = mask.detach().cpu().numpy()
    height, width = mask_np.shape
    row = np.clip(coords[:, 0].astype(np.int64), 0, height - 1)
    col = np.clip(coords[:, 1].astype(np.int64), 0, width - 1)
    return mask_np[row, col] >= threshold


def _compute_loss_from_result_with_mask(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    result,
    mask: Optional[torch.Tensor],
    *,
    include_unmatched_target: bool,
    push_to: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ndim = pred_field.ndim
    if mask is not None and ndim != 2:
        raise ValueError("Mask-aware projected Betti loss only supports 2D fields")

    def _filtered_coords(list_of_arrays) -> np.ndarray:
        return _concat_result_arrays(list_of_arrays, ndim)

    pred_birth_coords = _filtered_coords(result.input1_matched_birth_coordinates)
    pred_death_coords = _filtered_coords(result.input1_matched_death_coordinates)
    tgt_birth_coords = _filtered_coords(result.input2_matched_birth_coordinates)
    tgt_death_coords = _filtered_coords(result.input2_matched_death_coordinates)

    matched_keep = _filter_coords_by_mask(pred_birth_coords, mask)
    matched_keep &= _filter_coords_by_mask(pred_death_coords, mask)
    matched_keep &= _filter_coords_by_mask(tgt_birth_coords, mask)
    matched_keep &= _filter_coords_by_mask(tgt_death_coords, mask)

    pred_birth_coords = pred_birth_coords[matched_keep]
    pred_death_coords = pred_death_coords[matched_keep]
    tgt_birth_coords = tgt_birth_coords[matched_keep]
    tgt_death_coords = tgt_death_coords[matched_keep]

    pred_birth_vals = _tensor_values_at_coords(pred_field, pred_birth_coords)
    pred_death_vals = _tensor_values_at_coords(pred_field, pred_death_coords)
    tgt_birth_vals = _tensor_values_at_coords(tgt_field, tgt_birth_coords)
    tgt_death_vals = _tensor_values_at_coords(tgt_field, tgt_death_coords)

    pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
    tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
    loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    pred_unmatched_birth = _filtered_coords(result.input1_unmatched_birth_coordinates)
    pred_unmatched_death = _filtered_coords(result.input1_unmatched_death_coordinates)
    pred_unmatched_keep = _filter_coords_by_mask(pred_unmatched_birth, mask)
    pred_unmatched_keep &= _filter_coords_by_mask(pred_unmatched_death, mask)
    pred_unmatched_birth = pred_unmatched_birth[pred_unmatched_keep]
    pred_unmatched_death = pred_unmatched_death[pred_unmatched_keep]

    pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, pred_unmatched_birth)
    pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, pred_unmatched_death)
    pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
    loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred
    loss_unmatched_tgt = pred_field.new_zeros(())

    if include_unmatched_target:
        tgt_unmatched_birth = _filtered_coords(result.input2_unmatched_birth_coordinates)
        tgt_unmatched_death = _filtered_coords(result.input2_unmatched_death_coordinates)
        tgt_unmatched_keep = _filter_coords_by_mask(tgt_unmatched_birth, mask)
        tgt_unmatched_keep &= _filter_coords_by_mask(tgt_unmatched_death, mask)
        tgt_unmatched_birth = tgt_unmatched_birth[tgt_unmatched_keep]
        tgt_unmatched_death = tgt_unmatched_death[tgt_unmatched_keep]

        tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_birth)
        tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_death)
        tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
        loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
        total = total + loss_unmatched_tgt

    aux = {
        "betti/matched": loss_matched.reshape(1).detach(),
        "betti/unmatched_pred": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["betti/unmatched_target"] = loss_unmatched_tgt.reshape(1).detach()
    return total.reshape(1), aux


def _aggregate_aux(aux_parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    aux_agg: Dict[str, torch.Tensor] = {}
    if not aux_parts:
        return aux_agg
    all_keys = set().union(*(part.keys() for part in aux_parts))
    for key in all_keys:
        values = [part[key] for part in aux_parts if key in part]
        aux_agg[key] = torch.mean(torch.cat(values))
    return aux_agg


def _extract_target_regions_and_valid_mask(
    input: torch.Tensor,
    target: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if target.ndim == input.ndim - 1:
        target = target.unsqueeze(1)

    if target.ndim != input.ndim:
        raise ValueError(f"Expected target ndim {input.ndim} or {input.ndim - 1}, got shape {tuple(target.shape)}")

    if target.shape[1] >= 2:
        valid_mask = 1.0 - target[:, -1:].float()
        target_regions = target[:, :1].float()
    else:
        target_regions = target.float()
        valid_mask = loss_mask.float() if loss_mask is not None else None

    return target_regions, valid_mask


def _foreground_probabilities(input: torch.Tensor) -> torch.Tensor:
    num_channels = int(input.shape[1])
    if num_channels == 2:
        return torch.softmax(input, dim=1)[:, 1:2]
    if num_channels == 1:
        if bool((input.min() >= 0).item()) and bool((input.max() <= 1).item()):
            return input
        return torch.sigmoid(input)
    raise ValueError(f"Boundary-style losses expect 1 or 2 output channels, got {num_channels}")


def _max_pool_spatial(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    spatial_dims = x.ndim - 2
    if spatial_dims == 2:
        return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    if spatial_dims == 3:
        return F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
    raise ValueError(f"Expected 2D or 3D tensors for boundary loss, got shape {tuple(x.shape)}")


def _min_pool_spatial(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return -_max_pool_spatial(-x, kernel_size)


def _erode_spatial(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return _min_pool_spatial(x, kernel_size)


def _boundary_map(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return torch.clamp(x - _erode_spatial(x, kernel_size), min=0.0, max=1.0)


class BettiMatchingLoss(nn.Module):
    def __init__(
        self,
        filtration: str = "superlevel",
        include_unmatched_target: bool = False,
        push_unmatched_to: str = "diagonal",
    ):
        super().__init__()
        if filtration not in {"superlevel", "sublevel", "bothlevel"}:
            raise ValueError(f"Unsupported filtration: {filtration!r}")
        if push_unmatched_to not in {"diagonal", "one_zero", "death_death"}:
            raise ValueError(f"Unsupported push target: {push_unmatched_to!r}")
        self.filtration = filtration
        self.include_unmatched_target = bool(include_unmatched_target)
        self.push_unmatched_to = push_unmatched_to

    def _prepare_fields(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_size = input.shape[0]
        num_channels = input.shape[1]

        if num_channels == 2:
            probs = torch.softmax(input, dim=1)
            pred_fg = probs[:, 1:2]
            tgt_fg = target[:, 1:2] if target.shape[1] == 2 else target
        else:
            pred_fg = torch.sigmoid(input) if not (input.min() >= 0 and input.max() <= 1) else input
            tgt_fg = target

        pred_fg = pred_fg.contiguous()
        tgt_fg = tgt_fg.contiguous()
        preds_fields = [pred_fg[b, 0].contiguous() for b in range(batch_size)]
        tgts_fields = [tgt_fg[b, 0].contiguous() for b in range(batch_size)]
        return preds_fields, tgts_fields

    def _compute_loss_batch(
        self,
        pred_fields: List[torch.Tensor],
        tgt_fields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bm = _get_betti_module()
        results = bm.compute_matching(
            _to_numpy(pred_fields),
            _to_numpy(tgt_fields),
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=self.include_unmatched_target,
        )

        losses: List[torch.Tensor] = []
        aux_parts: List[Dict[str, torch.Tensor]] = []
        for pred_field, tgt_field, result in zip(pred_fields, tgt_fields, results):
            loss_part, aux_part = _compute_loss_from_result(
                pred_field,
                tgt_field,
                result,
                include_unmatched_target=self.include_unmatched_target,
                push_to=self.push_unmatched_to,
            )
            losses.append(loss_part)
            aux_parts.append(aux_part)

        return torch.mean(torch.cat(losses)), _aggregate_aux(aux_parts)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.shape[0] == 0:
            zero = input.new_tensor(0.0)
            return zero, {}

        pred_fields, tgt_fields = self._prepare_fields(input, target)

        if self.filtration == "bothlevel":
            loss_super, aux_super = self._compute_loss_batch(
                [1.0 - field for field in pred_fields],
                [1.0 - field for field in tgt_fields],
            )
            loss_sub, aux_sub = self._compute_loss_batch(pred_fields, tgt_fields)
            loss = 0.5 * (loss_super + loss_sub)
            aux = {}
            for key in set(aux_super) | set(aux_sub):
                if key in aux_super and key in aux_sub:
                    aux[key] = 0.5 * (aux_super[key] + aux_sub[key])
                elif key in aux_super:
                    aux[key] = 0.5 * aux_super[key]
                else:
                    aux[key] = 0.5 * aux_sub[key]
            return loss, aux

        if self.filtration == "superlevel":
            pred_fields = [1.0 - field for field in pred_fields]
            tgt_fields = [1.0 - field for field in tgt_fields]

        return self._compute_loss_batch(pred_fields, tgt_fields)


class MaskedBettiMatchingLoss(BettiMatchingLoss):
    def _split_target_and_mask(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return _extract_target_regions_and_valid_mask(input, target, loss_mask=loss_mask)

    def _compute_loss_batch_masked(
        self,
        pred_fields: List[torch.Tensor],
        tgt_fields: List[torch.Tensor],
        valid_masks: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bm = _get_betti_module()
        results = bm.compute_matching(
            _to_numpy(pred_fields),
            _to_numpy(tgt_fields),
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=self.include_unmatched_target,
        )

        losses: List[torch.Tensor] = []
        aux_parts: List[Dict[str, torch.Tensor]] = []
        for pred_field, tgt_field, valid_mask, result in zip(pred_fields, tgt_fields, valid_masks, results):
            loss_part, aux_part = _compute_loss_from_result_with_mask(
                pred_field,
                tgt_field,
                result,
                valid_mask,
                include_unmatched_target=self.include_unmatched_target,
                push_to=self.push_unmatched_to,
            )
            losses.append(loss_part)
            aux_parts.append(aux_part)

        return torch.mean(torch.cat(losses)), _aggregate_aux(aux_parts)

    def forward(self, input: torch.Tensor, target: torch.Tensor, loss_mask: Optional[torch.Tensor] = None):
        if input.shape[0] == 0:
            zero = input.new_tensor(0.0)
            return zero, {}

        target_regions, valid_mask = self._split_target_and_mask(input, target, loss_mask=loss_mask)
        pred_fields, tgt_fields = self._prepare_fields(input, target_regions)
        valid_masks = None
        if valid_mask is not None:
            valid_masks = [valid_mask[b, 0].contiguous() for b in range(valid_mask.shape[0])]
        else:
            valid_masks = [None] * input.shape[0]

        if self.filtration == "bothlevel":
            loss_super, aux_super = self._compute_loss_batch_masked(
                [1.0 - field for field in pred_fields],
                [1.0 - field for field in tgt_fields],
                valid_masks,
            )
            loss_sub, aux_sub = self._compute_loss_batch_masked(pred_fields, tgt_fields, valid_masks)
            loss = 0.5 * (loss_super + loss_sub)
            aux = {}
            for key in set(aux_super) | set(aux_sub):
                if key in aux_super and key in aux_sub:
                    aux[key] = 0.5 * (aux_super[key] + aux_sub[key])
                elif key in aux_super:
                    aux[key] = 0.5 * aux_super[key]
                else:
                    aux[key] = 0.5 * aux_sub[key]
            return loss, aux

        if self.filtration == "superlevel":
            pred_fields = [1.0 - field for field in pred_fields]
            tgt_fields = [1.0 - field for field in tgt_fields]

        return self._compute_loss_batch_masked(pred_fields, tgt_fields, valid_masks)


class BoundaryLoss(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        weight_bce: float = 1.0,
        weight_dice: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1 or (kernel_size % 2) != 1:
            raise ValueError(f"BoundaryLoss kernel_size must be odd and >= 1, got {kernel_size}")
        if weight_bce < 0.0 or weight_dice < 0.0:
            raise ValueError("BoundaryLoss weights must be non-negative")
        if weight_bce == 0.0 and weight_dice == 0.0:
            raise ValueError("BoundaryLoss requires at least one non-zero component weight")
        if smooth <= 0.0:
            raise ValueError(f"BoundaryLoss smooth must be > 0, got {smooth}")
        if eps <= 0.0:
            raise ValueError(f"BoundaryLoss eps must be > 0, got {eps}")

        self.kernel_size = kernel_size
        self.weight_bce = float(weight_bce)
        self.weight_dice = float(weight_dice)
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if input.shape[0] == 0:
            zero = input.new_tensor(0.0)
            aux = {
                "boundary/bce": zero.reshape(1),
                "boundary/dice": zero.reshape(1),
            }
            return zero, aux

        target_regions, valid_mask = _extract_target_regions_and_valid_mask(input, target, loss_mask=loss_mask)
        pred_fg = _foreground_probabilities(input).float()
        tgt_fg = target_regions.float()

        pred_boundary = _boundary_map(pred_fg, self.kernel_size)
        tgt_boundary = _boundary_map(tgt_fg, self.kernel_size)

        boundary_valid_mask = None
        if valid_mask is not None:
            boundary_valid_mask = _erode_spatial(valid_mask.float(), self.kernel_size)
            if torch.count_nonzero(boundary_valid_mask) == 0:
                zero = input.new_tensor(0.0)
                aux = {
                    "boundary/bce": zero.reshape(1),
                    "boundary/dice": zero.reshape(1),
                }
                return zero, aux

        bce_loss = input.new_zeros(())
        if self.weight_bce > 0.0:
            pred_boundary_clamped = pred_boundary.clamp(min=self.eps, max=1.0 - self.eps)
            bce_map = F.binary_cross_entropy(pred_boundary_clamped, tgt_boundary, reduction="none")
            if boundary_valid_mask is None:
                bce_loss = bce_map.mean()
            else:
                denom = torch.clamp(boundary_valid_mask.sum(), min=self.eps)
                bce_loss = (bce_map * boundary_valid_mask).sum() / denom

        if boundary_valid_mask is None:
            pred_for_dice = pred_boundary
            tgt_for_dice = tgt_boundary
        else:
            pred_for_dice = pred_boundary * boundary_valid_mask
            tgt_for_dice = tgt_boundary * boundary_valid_mask

        dice_loss = input.new_zeros(())
        if self.weight_dice > 0.0:
            intersection = (pred_for_dice * tgt_for_dice).sum()
            denom = pred_for_dice.sum() + tgt_for_dice.sum()
            dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth))

        total = (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss)
        aux = {
            "boundary/bce": bce_loss.reshape(1).detach(),
            "boundary/dice": dice_loss.reshape(1).detach(),
        }
        return total.reshape(()), aux


class WeightedLossTerm(nn.Module):
    def __init__(self, name: str, weight: float, module: nn.Module, metric_name: Optional[str] = None):
        super().__init__()
        self.name = str(name)
        self.weight = float(weight)
        self.module = module
        self.metric_name = str(metric_name or name)


def _sanitize_metric_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    cleaned = cleaned.strip("_")
    return cleaned or "term"


def _split_loss_output(output) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(f"Expected (loss, aux_dict) tuple, got tuple of length {len(output)}")
        loss_value, aux = output
        if not isinstance(aux, dict):
            raise TypeError(f"Expected aux metrics dict, got {type(aux).__name__}")
        return loss_value.reshape(()), aux
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected loss tensor, got {type(output).__name__}")
    return output.reshape(()), {}


def _build_label_smoothed_dice_bce_term(term_cfg: dict, config: dict) -> nn.Module:
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs=dict(term_cfg.get("bce_kwargs", {})),
        soft_dice_kwargs={
            "label_smoothing": float(
                term_cfg.get(
                    "dice_label_smoothing",
                    config.get("dice_label_smoothing", 0.0),
                )
            ),
        },
        weight_dice=float(term_cfg.get("weight_dice", term_cfg.get("dice_weight", 1.0))),
        weight_ce=float(term_cfg.get("weight_ce", term_cfg.get("ce_weight", 1.0))),
        use_ignore_label=True,
        bce_label_smoothing=float(
            term_cfg.get(
                "bce_label_smoothing",
                config.get("bce_label_smoothing", 0.0),
            )
        ),
    )


def _build_masked_betti_matching_term(term_cfg: dict, _config: dict) -> nn.Module:
    return MaskedBettiMatchingLoss(
        filtration=str(term_cfg.get("filtration", "superlevel")),
        include_unmatched_target=bool(term_cfg.get("include_unmatched_target", False)),
        push_unmatched_to=str(term_cfg.get("push_unmatched_to", "diagonal")),
    )


def _build_boundary_term(term_cfg: dict, _config: dict) -> nn.Module:
    return BoundaryLoss(
        kernel_size=int(term_cfg.get("kernel_size", 3)),
        weight_bce=float(term_cfg.get("weight_bce", 1.0)),
        weight_dice=float(term_cfg.get("weight_dice", 1.0)),
        smooth=float(term_cfg.get("smooth", 1.0)),
        eps=float(term_cfg.get("eps", 1e-6)),
    )


LOSS_TERM_BUILDERS = {
    "LabelSmoothedDCAndBCELoss": _build_label_smoothed_dice_bce_term,
    "MaskedBettiMatchingLoss": _build_masked_betti_matching_term,
    "BettiMatchingLoss": _build_masked_betti_matching_term,
    "BoundaryLoss": _build_boundary_term,
}


class CompositeLoss(nn.Module):
    def __init__(self, terms: List[WeightedLossTerm]):
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss requires at least one term")
        self.terms = nn.ModuleList(terms)
        self.latest_metrics: Dict[str, float] = {}

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = net_output.new_zeros(())
        metrics: Dict[str, float] = {}
        used_metric_names = set()

        for index, term in enumerate(self.terms):
            raw_output = term.module(net_output, target)
            raw_loss, aux_metrics = _split_loss_output(raw_output)
            weighted_loss = raw_loss * term.weight
            total = total + weighted_loss

            metric_name = _sanitize_metric_name(term.metric_name)
            if metric_name in used_metric_names:
                metric_name = f"{metric_name}_{index}"
            used_metric_names.add(metric_name)

            metrics[f"loss_terms/{metric_name}_raw"] = float(raw_loss.detach().item())
            metrics[f"loss_terms/{metric_name}_weighted"] = float(weighted_loss.detach().item())

            for key, value in aux_metrics.items():
                aux_key = _sanitize_metric_name(key)
                metrics[f"loss_aux/{metric_name}/{aux_key}"] = float(value.detach().item())

        metrics["loss/total"] = float(total.detach().item())
        self.latest_metrics = metrics
        return total


ProjectedSegmentationLoss = CompositeLoss


def _default_terms(config: dict, loss_cfg: dict) -> List[dict]:
    return [
        {
            "name": "LabelSmoothedDCAndBCELoss",
            "metric_name": "base",
            "weight": 1.0,
            "weight_dice": float(loss_cfg.get("dice_weight", 0.25)),
            "weight_ce": float(loss_cfg.get("ce_weight", 1.0)),
            "dice_label_smoothing": float(
                loss_cfg.get("dice_label_smoothing", config.get("dice_label_smoothing", 0.0))
            ),
            "bce_label_smoothing": float(
                loss_cfg.get("bce_label_smoothing", config.get("bce_label_smoothing", 0.0))
            ),
        }
    ]


def _normalize_terms_config(config: dict) -> List[dict]:
    loss_cfg = dict(config.get("loss", {}) or {})
    raw_terms = loss_cfg.get("terms")
    if raw_terms is None:
        return _default_terms(config, loss_cfg)
    if not isinstance(raw_terms, list) or len(raw_terms) == 0:
        raise ValueError("loss.terms must be a non-empty list when provided")
    return [dict(term or {}) for term in raw_terms]


def create_loss_from_config(config: dict) -> CompositeLoss:
    terms_cfg = _normalize_terms_config(config)
    terms: List[WeightedLossTerm] = []

    for idx, term_cfg in enumerate(terms_cfg):
        name = term_cfg.get("name")
        if not name:
            raise ValueError(f"loss term at index {idx} is missing required key 'name'")
        weight = float(term_cfg.get("weight", 1.0))
        if weight == 0.0:
            continue
        builder = LOSS_TERM_BUILDERS.get(str(name))
        if builder is None:
            supported = ", ".join(sorted(LOSS_TERM_BUILDERS))
            raise ValueError(f"Unsupported loss term {name!r}. Supported terms: {supported}")

        module = builder(term_cfg, config)
        terms.append(
            WeightedLossTerm(
                name=str(name),
                weight=weight,
                module=module,
                metric_name=term_cfg.get("metric_name"),
            )
        )

    if not terms:
        raise ValueError("All configured loss terms have zero weight; at least one active term is required")

    return CompositeLoss(terms)
