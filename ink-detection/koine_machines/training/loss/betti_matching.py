from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


_BETTI_MODULE = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
            "  python common/build_betti.py"
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


def _sum_aux_parts(aux_parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    aux_sum: Dict[str, torch.Tensor] = {}
    if not aux_parts:
        return aux_sum
    all_keys = set().union(*(part.keys() for part in aux_parts))
    for key in all_keys:
        values = [part[key].reshape(1) for part in aux_parts if key in part]
        aux_sum[key] = torch.sum(torch.cat(values)).reshape(1)
    return aux_sum


def _iter_connected_component_masks(mask: torch.Tensor) -> List[Tuple[slice, slice, np.ndarray]]:
    if mask.ndim != 2:
        raise ValueError(f"Mask-aware projected Betti loss expects 2D masks, got shape {tuple(mask.shape)}")

    mask_np = mask.detach().cpu().numpy() >= 0.5
    if not np.any(mask_np):
        return []

    height, width = mask_np.shape
    visited = np.zeros_like(mask_np, dtype=bool)
    components: List[Tuple[slice, slice, np.ndarray]] = []

    for start_row, start_col in np.argwhere(mask_np):
        start_row = int(start_row)
        start_col = int(start_col)
        if visited[start_row, start_col]:
            continue

        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True
        coords: List[Tuple[int, int]] = []
        min_row = max_row = start_row
        min_col = max_col = start_col

        while queue:
            row, col = queue.pop()
            coords.append((row, col))
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)

            for row_offset, col_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + row_offset
                next_col = col + col_offset
                if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                    continue
                if visited[next_row, next_col] or not mask_np[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

        component_mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=bool)
        for row, col in coords:
            component_mask[row - min_row, col - min_col] = True

        components.append(
            (
                slice(min_row, max_row + 1),
                slice(min_col, max_col + 1),
                component_mask,
            )
        )

    return components


def _masked_component_fields(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if valid_mask is None:
        return [(pred_field, tgt_field, None)]

    if pred_field.ndim != 2 or tgt_field.ndim != 2:
        raise ValueError("Mask-aware projected Betti loss only supports 2D fields")

    component_specs = _iter_connected_component_masks(valid_mask)
    if not component_specs:
        return []

    components: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
    for row_slice, col_slice, component_mask_np in component_specs:
        pred_crop = pred_field[row_slice, col_slice].clone()
        tgt_crop = tgt_field[row_slice, col_slice].clone()
        component_mask = torch.from_numpy(component_mask_np).to(device=pred_field.device, dtype=torch.bool)

        if not bool(component_mask.all()):
            pred_crop = pred_crop.masked_fill(~component_mask, 1.0)
            tgt_crop = tgt_crop.masked_fill(~component_mask, 1.0)

        components.append((pred_crop.contiguous(), tgt_crop.contiguous(), component_mask.contiguous()))

    return components


def _compute_loss_from_result(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    result,
    mask: Optional[torch.Tensor] = None,
    *,
    include_unmatched_target: bool,
    push_to: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ndim = pred_field.ndim
    if mask is not None and ndim != 2:
        raise ValueError("Mask-aware projected Betti loss only supports 2D fields")

    pred_birth_coords = _concat_result_arrays(result.input1_matched_birth_coordinates, ndim)
    pred_death_coords = _concat_result_arrays(result.input1_matched_death_coordinates, ndim)
    tgt_birth_coords = _concat_result_arrays(result.input2_matched_birth_coordinates, ndim)
    tgt_death_coords = _concat_result_arrays(result.input2_matched_death_coordinates, ndim)

    pred_unmatched_birth = _concat_result_arrays(result.input1_unmatched_birth_coordinates, ndim)
    pred_unmatched_death = _concat_result_arrays(result.input1_unmatched_death_coordinates, ndim)
    tgt_unmatched_birth = _concat_result_arrays(result.input2_unmatched_birth_coordinates, ndim)
    tgt_unmatched_death = _concat_result_arrays(result.input2_unmatched_death_coordinates, ndim)

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

    def _split_target_and_mask(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return _extract_target_regions_and_valid_mask(input, target, loss_mask=loss_mask)

    def _compute_loss_batch(
        self,
        pred_fields: List[torch.Tensor],
        tgt_fields: List[torch.Tensor],
        valid_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bm = _get_betti_module()
        if valid_masks is None:
            valid_masks = [None] * len(pred_fields)

        sample_losses: List[torch.Tensor] = []
        sample_aux_parts: List[Dict[str, torch.Tensor]] = []

        for pred_field, tgt_field, valid_mask in zip(pred_fields, tgt_fields, valid_masks):
            component_fields = _masked_component_fields(pred_field, tgt_field, valid_mask)
            if not component_fields:
                sample_losses.append(pred_field.new_zeros((1,)))
                sample_aux_parts.append({})
                continue

            masked_pred_fields = [part[0] for part in component_fields]
            masked_tgt_fields = [part[1] for part in component_fields]
            component_masks = [part[2] for part in component_fields]
            results = bm.compute_matching(
                _to_numpy(masked_pred_fields),
                _to_numpy(masked_tgt_fields),
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=self.include_unmatched_target,
            )

            component_losses: List[torch.Tensor] = []
            component_aux_parts: List[Dict[str, torch.Tensor]] = []
            for masked_pred, masked_tgt, component_mask, result in zip(
                masked_pred_fields,
                masked_tgt_fields,
                component_masks,
                results,
            ):
                loss_part, aux_part = _compute_loss_from_result(
                    masked_pred,
                    masked_tgt,
                    result,
                    mask=component_mask,
                    include_unmatched_target=self.include_unmatched_target,
                    push_to=self.push_unmatched_to,
                )
                component_losses.append(loss_part)
                component_aux_parts.append(aux_part)

            sample_losses.append(torch.sum(torch.cat(component_losses)).reshape(1))
            sample_aux_parts.append(_sum_aux_parts(component_aux_parts))

        return torch.mean(torch.cat(sample_losses)), _aggregate_aux(sample_aux_parts)

    def forward(self, input: torch.Tensor, target: torch.Tensor, loss_mask: Optional[torch.Tensor] = None):
        if input.shape[0] == 0:
            zero = input.new_tensor(0.0)
            return zero, {}

        target_regions, valid_mask = self._split_target_and_mask(input, target, loss_mask=loss_mask)
        pred_fields, tgt_fields = self._prepare_fields(input, target_regions)
        valid_masks = [None] * input.shape[0]
        if valid_mask is not None:
            valid_masks = [valid_mask[b, 0].contiguous() for b in range(valid_mask.shape[0])]

        if self.filtration == "bothlevel":
            loss_super, aux_super = self._compute_loss_batch(
                [1.0 - field for field in pred_fields],
                [1.0 - field for field in tgt_fields],
                valid_masks,
            )
            loss_sub, aux_sub = self._compute_loss_batch(pred_fields, tgt_fields, valid_masks)
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

        return self._compute_loss_batch(pred_fields, tgt_fields, valid_masks)
