from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StitchLossBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    boundary_dist_map: torch.Tensor | None = None


def _stitch_loss_terms(recipe) -> tuple[object, ...]:
    if isinstance(recipe, tuple):
        return recipe
    if isinstance(recipe, list):
        return tuple(recipe)
    if recipe is None:
        return ()
    return (recipe,)


def _requires_boundary_dist_map(recipe) -> bool:
    return any(bool(getattr(term, "requires_boundary_dist_map", False)) for term in _stitch_loss_terms(recipe))


def _merge_stitch_term_components(out, term_components):
    loss = term_components.get("loss")
    if loss is None:
        raise TypeError("stitched loss component terms must return a 'loss' entry")
    out["loss"] = out["loss"] + loss

    for key, value in term_components.items():
        if key == "loss":
            continue
        out[key] = value


def compute_stitched_loss_components(
    recipe,
    batch: StitchLossBatch,
):
    terms = _stitch_loss_terms(recipe)
    if len(terms) <= 0:
        raise TypeError("stitched loss components require at least one stitched loss term")
    if not isinstance(batch, StitchLossBatch):
        raise TypeError("stitched loss components require StitchLossBatch")

    zero = torch.zeros((), device=batch.logits.device, dtype=batch.logits.dtype)
    out = {
        "loss": zero.clone(),
        "covered_px": int(batch.valid_mask.sum().detach().item()),
    }

    for term in terms:
        compute = getattr(term, "compute", None)
        if not callable(compute):
            raise TypeError("stitched loss component terms must provide compute(...)")
        term_components = compute(batch)
        _merge_stitch_term_components(out, term_components)
    return out
