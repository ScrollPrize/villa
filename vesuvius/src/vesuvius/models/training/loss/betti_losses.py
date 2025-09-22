"""
Betti Matching losses for topologically accurate segmentation.
from https://github.com/nstucki/Betti-matching
Uses the C++ implementation of Betti matching with Python bindings found here https://github.com/nstucki/Betti-Matching-3D

This implementation mirrors the example Betti matching loss in scratch/loss_function.py:
- Uses matched birth/death value differences (squared, with factor 2)
- Penalizes unmatched pairs by pushing to the diagonal (default) or other strategies
- Supports 'sublevel', 'superlevel', and 'bothlevel' filtrations

To install and use this loss, make sure you run the build_betti.py script in vesuvius/utils
"""

import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_betti_module() -> object:
    """Load the compiled betti_matching extension without mutating sys.path."""

    vesuvius_src_path = Path(__file__).resolve().parents[5]
    betti_build_path = vesuvius_src_path / "external" / "Betti-Matching-3D" / "build"

    if not betti_build_path.exists():
        raise ImportError(
            f"Betti-Matching-3D build not found at {betti_build_path}. "
            "Please run the build_betti.py script in vesuvius/utils/. "
            "You may need to force the system libstdc++ with:\n"
            "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        )

    candidates = []
    for pattern in ("betti_matching*.so", "betti_matching*.pyd", "betti_matching*.dll"):
        candidates.extend(betti_build_path.rglob(pattern))

    if not candidates:
        raise ImportError(
            f"betti_matching extension not found under {betti_build_path}. "
            "Please rerun build_betti.py and ensure the extension compiles."
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("betti_matching", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load betti_matching from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    bm = _load_betti_module()
except ImportError as e:
    raise ImportError(
        "Could not import betti_matching module. "
        "Please run the build_betti.py script in vesuvius/utils/.  "
        "This will clone and build Betti-Matching-3D automatically. "
        "You may need to force the system libstdc++ with:\n"
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
        f"Error: {e}"
    ) from e


def _tensor_values_at_coords(t: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    """Index values from tensor t at integer voxel coordinates coords (N, D)."""
    if coords.size == 0:
        return t.new_zeros((0,), dtype=t.dtype)
    idx = torch.as_tensor(coords, device=t.device, dtype=torch.long)
    return t[tuple(idx[:, d] for d in range(idx.shape[1]))]


def _stack_pairs(values_birth: torch.Tensor, values_death: torch.Tensor) -> torch.Tensor:
    if values_birth.numel() == 0:
        return values_birth.new_zeros((0, 2))
    return torch.stack([values_birth, values_death], dim=1)


def _loss_unmatched(pairs: torch.Tensor, push_to: str = "diagonal") -> torch.Tensor:
    """Compute unmatched loss given (N, 2) pairs of (birth, death) values.
    push_to:
      - 'diagonal': sum((birth - death)^2)
      - 'one_zero': 2 * sum((birth - 1)^2 + death^2)
      - 'death_death': 2 * sum((birth - death)^2)
    """
    if pairs.numel() == 0:
        return pairs.new_zeros(())
    if push_to == "diagonal":
        return ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    elif push_to == "one_zero":
        return 2.0 * (((pairs[:, 0] - 1.0) ** 2) + (pairs[:, 1] ** 2)).sum()
    elif push_to == "death_death":
        return 2.0 * ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    else:
        # default to diagonal if unknown
        return ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()


def _compute_loss_from_result(pred_field: torch.Tensor,
                              tgt_field: torch.Tensor,
                              res,
                              *,
                              include_unmatched_target: bool,
                              push_to: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mirror of scratch/_betti_matching_loss using our binding attribute names."""
    # Matched coordinates (concatenate across homology dimensions)
    def _concat(list_of_arrays):
        # Flatten potential nested lists of arrays across homology dimensions
        flat: list[np.ndarray] = []
        if list_of_arrays is not None:
            for a in list_of_arrays:
                if a is None:
                    continue
                if isinstance(a, (list, tuple)):
                    for b in a:
                        if isinstance(b, np.ndarray) and b.size > 0:
                            flat.append(b)
                elif isinstance(a, np.ndarray):
                    if a.size > 0:
                        flat.append(a)
        if len(flat) == 0:
            # infer ndim from fields
            ndim = pred_field.ndim
            return np.zeros((0, ndim), dtype=np.int64)
        return np.ascontiguousarray(np.concatenate(flat, axis=0))

    pred_birth_coords = _concat(res.input1_matched_birth_coordinates)
    pred_death_coords = _concat(res.input1_matched_death_coordinates)
    tgt_birth_coords = _concat(res.input2_matched_birth_coordinates)
    tgt_death_coords = _concat(res.input2_matched_death_coordinates)

    pred_unmatched_birth = _concat(res.input1_unmatched_birth_coordinates)
    pred_unmatched_death = _concat(res.input1_unmatched_death_coordinates)
    tgt_unmatched_birth = _concat(res.input2_unmatched_birth_coordinates)
    tgt_unmatched_death = _concat(res.input2_unmatched_death_coordinates)

    # Gather values from tensors at those coordinates
    pred_birth_vals = _tensor_values_at_coords(pred_field, pred_birth_coords)
    pred_death_vals = _tensor_values_at_coords(pred_field, pred_death_coords)
    tgt_birth_vals = _tensor_values_at_coords(tgt_field, tgt_birth_coords)
    tgt_death_vals = _tensor_values_at_coords(tgt_field, tgt_death_coords)

    pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, pred_unmatched_birth)
    pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, pred_unmatched_death)
    tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_birth)
    tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_death)

    # Matched loss: 2 * sum((birth_pred - birth_tgt)^2 + (death_pred - death_tgt)^2)
    pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
    tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
    loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    # Unmatched pairs (default push to diagonal)
    pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
    tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
    loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred

    loss_unmatched_tgt = pred_field.new_zeros(())
    if include_unmatched_target:
        tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
        loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
        total = total + loss_unmatched_tgt

    # Auxiliary stats (detached)
    aux = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["Betti matching loss (unmatched target)"] = loss_unmatched_tgt.reshape(1).detach()
    return total.reshape(1), aux


class BettiMatchingLoss(nn.Module):
    """
    Betti matching loss for topological accuracy. See https://github.com/nstucki/Betti-matching for details

    Filtration is implemented via preprocessing (inverting values for superlevel filtration).

    Parameters:
    - filtration: 'superlevel' | 'sublevel' | 'bothlevel'
    - include_unmatched_target: whether to penalize unmatched target pairs (mirrors scratch include_unmatched_target)
    - push_unmatched_to: 'diagonal' | 'one_zero' | 'death_death'
    """

    def __init__(self,
                 filtration: str = 'superlevel',
                 include_unmatched_target: bool = False,
                 push_unmatched_to: str = 'diagonal'):
        super().__init__()
        assert filtration in ('superlevel', 'sublevel', 'bothlevel'), "filtration must be one of: superlevel, sublevel, bothlevel"
        assert push_unmatched_to in ('diagonal', 'one_zero', 'death_death'), "push_unmatched_to must be one of: diagonal, one_zero, death_death"
        self.filtration = filtration
        self.include_unmatched_target = include_unmatched_target
        self.push_unmatched_to = push_unmatched_to

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute Betti matching loss matching the scratch example semantics.

        Accepts logits or probabilities for `input` with shape (B, C, ...). For C==2 uses softmax and the
        foreground channel; for C==1 applies sigmoid if outside [0,1]. Target may be 1-channel or one-hot with C==2.
        Returns either a scalar tensor or (loss, aux_dict) where aux_dict contains diagnostic components.
        """
        device = input.device
        batch_size = input.shape[0]
        num_channels = input.shape[1]

        if batch_size == 0:
            zero = input.new_tensor(0.0)
            return zero, {}

        # Select foreground channel and normalize to [0,1] if needed
        if num_channels == 2:
            probs = torch.softmax(input, dim=1)
            pred_fg = probs[:, 1:2]
            tgt_fg = target[:, 1:2] if target.shape[1] == 2 else target
        else:
            pred_fg = torch.sigmoid(input) if not (input.min() >= 0 and input.max() <= 1) else input
            tgt_fg = target

        pred_fg = pred_fg.contiguous()
        tgt_fg = tgt_fg.contiguous()

        # Build lists for batched calls to the C++ extension
        preds_fields = [pred_fg[b, 0].contiguous() for b in range(batch_size)]
        tgts_fields = [tgt_fg[b, 0].contiguous() for b in range(batch_size)]

        def _to_numpy(fields):
            return [np.ascontiguousarray(f.detach().cpu().numpy().astype(np.float64)) for f in fields]

        total_losses = []
        aux_parts = []

        if self.filtration == 'bothlevel':
            preds_sub_fields = preds_fields
            tgts_sub_fields = tgts_fields
            preds_super_fields = [1.0 - p for p in preds_fields]
            tgts_super_fields = [1.0 - t for t in tgts_fields]

            results_super = bm.compute_matching(
                _to_numpy(preds_super_fields),
                _to_numpy(tgts_super_fields),
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=self.include_unmatched_target,
            )
            results_sub = bm.compute_matching(
                _to_numpy(preds_sub_fields),
                _to_numpy(tgts_sub_fields),
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=self.include_unmatched_target,
            )

            for b in range(batch_size):
                loss_super, aux_super = _compute_loss_from_result(
                    preds_super_fields[b],
                    tgts_super_fields[b],
                    results_super[b],
                    include_unmatched_target=self.include_unmatched_target,
                    push_to=self.push_unmatched_to,
                )
                loss_sub, aux_sub = _compute_loss_from_result(
                    preds_sub_fields[b],
                    tgts_sub_fields[b],
                    results_sub[b],
                    include_unmatched_target=self.include_unmatched_target,
                    push_to=self.push_unmatched_to,
                )

                total_losses.append(0.5 * (loss_super + loss_sub))

                keys = set(aux_super.keys()) | set(aux_sub.keys())
                combined_aux: Dict[str, torch.Tensor] = {}
                for k in keys:
                    val_super = aux_super.get(k)
                    val_sub = aux_sub.get(k)
                    if val_super is not None and val_sub is not None:
                        combined_aux[k] = 0.5 * (val_super + val_sub)
                    elif val_super is not None:
                        combined_aux[k] = 0.5 * val_super
                    elif val_sub is not None:
                        combined_aux[k] = 0.5 * val_sub
                aux_parts.append(combined_aux)
        else:
            if self.filtration == 'superlevel':
                preds_proc_fields = [1.0 - p for p in preds_fields]
                tgts_proc_fields = [1.0 - t for t in tgts_fields]
            else:
                preds_proc_fields = preds_fields
                tgts_proc_fields = tgts_fields

            results = bm.compute_matching(
                _to_numpy(preds_proc_fields),
                _to_numpy(tgts_proc_fields),
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=self.include_unmatched_target,
            )

            for b in range(batch_size):
                loss_b, aux_b = _compute_loss_from_result(
                    preds_proc_fields[b],
                    tgts_proc_fields[b],
                    results[b],
                    include_unmatched_target=self.include_unmatched_target,
                    push_to=self.push_unmatched_to,
                )
                total_losses.append(loss_b)
                aux_parts.append(aux_b)

        loss = torch.mean(torch.cat(total_losses)) if len(total_losses) > 0 else torch.tensor(0.0, device=device)

        # Aggregate aux
        aux_agg: Dict[str, torch.Tensor] = {}
        if len(aux_parts) > 0:
            all_keys = set().union(*(d.keys() for d in aux_parts))
            for k in all_keys:
                values = [d[k] for d in aux_parts if k in d]
                aux_agg[k] = torch.mean(torch.cat(values))

        # Return tuple to make DeepSupervisionWrapper pick the scalar part automatically
        return loss, aux_agg
