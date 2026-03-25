from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


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
        results = bm.compute_matching(
            _to_numpy(pred_fields),
            _to_numpy(tgt_fields),
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=self.include_unmatched_target,
        )

        losses: List[torch.Tensor] = []
        aux_parts: List[Dict[str, torch.Tensor]] = []
        if valid_masks is None:
            valid_masks = [None] * len(pred_fields)

        for pred_field, tgt_field, valid_mask, result in zip(pred_fields, tgt_fields, valid_masks, results):
            loss_part, aux_part = _compute_loss_from_result(
                pred_field,
                tgt_field,
                result,
                mask=valid_mask,
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
