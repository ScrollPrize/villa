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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

try:
    import sys
    from pathlib import Path

    # Look for the external Betti build in the Vesuvius installation
    # Path structure: .../vesuvius/src/vesuvius/models/training/loss/betti_losses.py
    # Need to go up to .../vesuvius/src/
    vesuvius_src_path = Path(__file__).parent.parent.parent.parent.parent  # Go up to vesuvius/src
    betti_build_path = vesuvius_src_path / "external" / "Betti-Matching-3D" / "build"

    if betti_build_path.exists():
        sys.path.insert(0, str(betti_build_path))
        import betti_matching as bm
    else:
        raise ImportError(
            f"Betti-Matching-3D build not found at {betti_build_path}. "
            f"Please run the build_betti.py script in vesuvius/utils/ "
            f"This will automatically clone and build Betti-Matching-3D. "
            f"You may need to force the system libstdc++ with:\n"
            f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        )

except ImportError as e:
    raise ImportError(
        f"Could not import betti_matching module. "
        f"Please run the build_betti.py script in vesuvius/utils/.  "
        f"This will clone and build Betti-Matching-3D automatically. "
        f"You may need to force the system libstdc++ with:\n"
        f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
        f"Error: {e}"
    )


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
                              res) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
    loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to="diagonal")
    loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to="diagonal")

    total = loss_matched + loss_unmatched_pred + loss_unmatched_tgt

    # Auxiliary stats (detached)
    aux = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_pred.reshape(1).detach(),
        "Betti matching loss (unmatched target)": loss_unmatched_tgt.reshape(1).detach(),
    }
    return total.reshape(1), aux


class BettiMatchingLoss(nn.Module):
    """
    Betti matching loss for topological accuracy. See https://github.com/nstucki/Betti-matching for details

    Filtration is implemented via preprocessing (inverting values for superlevel filtration).

    Parameters:
    - filtration: 'superlevel' | 'sublevel' | 'bothlevel'
    """

    def __init__(self, filtration: str = 'superlevel'):
        super().__init__()
        assert filtration in ('superlevel', 'sublevel', 'bothlevel'), "filtration must be one of: superlevel, sublevel, bothlevel"
        self.filtration = filtration

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute Betti matching loss matching the scratch example semantics.

        Accepts logits or probabilities for `input` with shape (B, C, ...). For C==2 uses softmax and the
        foreground channel; for C==1 applies sigmoid if outside [0,1]. Target may be 1-channel or one-hot with C==2.
        Returns either a scalar tensor or (loss, aux_dict) where aux_dict contains diagnostic components.
        """
        device = input.device
        batch_size = input.shape[0]
        num_channels = input.shape[1]
        is_3d = (input.ndim == 5)

        # Select foreground channel and normalize to [0,1] if needed
        if num_channels == 2:
            probs = torch.softmax(input, dim=1)
            pred_fg = probs[:, 1:2]
            tgt_fg = target[:, 1:2] if target.shape[1] == 2 else target
        else:
            pred_fg = torch.sigmoid(input) if not (input.min() >= 0 and input.max() <= 1) else input
            tgt_fg = target


        # Build lists for a single batched call to the C++ extension
        preds_fields = [pred_fg[b].squeeze(0) for b in range(batch_size)]
        tgts_fields = [tgt_fg[b].squeeze(0) for b in range(batch_size)]

        preds_np = [np.ascontiguousarray(p.detach().cpu().numpy().astype(np.float64)) for p in preds_fields]
        tgts_np = [np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float64)) for t in tgts_fields]

        # Binarize targets (training targets are masks)
        tgts_np = [(t > 0.5).astype(np.float64) for t in tgts_np]

        total_losses = []
        aux_parts = []

        if self.filtration == 'bothlevel':
            preds_super = [1.0 - p for p in preds_np]
            tgts_super = [1.0 - t for t in tgts_np]

            results_super = bm.compute_matching(
                preds_super, tgts_super,
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=True
            )
            results_sub = bm.compute_matching(
                preds_np, tgts_np,
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=True
            )

            for b in range(batch_size):
                loss_super, aux_super = _compute_loss_from_result(preds_fields[b], tgts_fields[b], results_super[b])
                loss_sub, aux_sub = _compute_loss_from_result(preds_fields[b], tgts_fields[b], results_sub[b])
                # Match scratch/Betti-matching-master semantics: sum superlevel + sublevel
                total_losses.append(loss_super + loss_sub)
                aux_parts.append({k: (aux_super[k] + aux_sub[k]) for k in aux_super.keys()})
        else:
            if self.filtration == 'superlevel':
                preds_in = [1.0 - p for p in preds_np]
                tgts_in = [1.0 - t for t in tgts_np]
            else:
                preds_in = preds_np
                tgts_in = tgts_np

            results = bm.compute_matching(
                preds_in, tgts_in,
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=True
            )

            for b in range(batch_size):
                loss_b, aux_b = _compute_loss_from_result(preds_fields[b], tgts_fields[b], results[b])
                total_losses.append(loss_b)
                aux_parts.append(aux_b)

        loss = torch.mean(torch.cat(total_losses)) if len(total_losses) > 0 else torch.tensor(0.0, device=device)

        # Aggregate aux
        aux_agg: Dict[str, torch.Tensor] = {}
        if len(aux_parts) > 0:
            keys = aux_parts[0].keys()
            for k in keys:
                aux_agg[k] = torch.mean(torch.cat([d[k] for d in aux_parts]))

        # Return tuple to make DeepSupervisionWrapper pick the scalar part automatically
        return loss, aux_agg
