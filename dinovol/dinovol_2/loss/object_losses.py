"""Point and object losses between two global views (student vs. EMA teacher).

All terms compare a student view with the teacher's other view of the same source
window, at points whose coordinates are known in both views:

- point: each point's dense embedding should agree (cosine) with the teacher's at the
  same physical point in the other view. There are no negatives: unlabeled points
  nearby may lie on the same object, so nothing pushes points apart here.
- object: dense embeddings of points on the same pseudo-object are positives; points on
  objects that are provably different (stacked sheet layers, crossing fibers) are
  negatives. Pairs with an unknown relation are ignored.
- region: token features averaged over each object (DetCon-style), projected, and
  contrasted the same way at the object level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ObjectLossResult:
    point: torch.Tensor
    object: torch.Tensor
    region: torch.Tensor
    point_anchors: torch.Tensor
    object_anchors: torch.Tensor
    region_anchors: torch.Tensor


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A finite fill keeps rows without any key finite: masked-out rows must not turn
    # into NaN gradients (0 * NaN) when they are excluded afterwards.
    return torch.logsumexp(logits.masked_fill(~mask, -1e9), dim=-1)


def _mean_over(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum()
    loss = torch.where(mask, values, torch.zeros_like(values)).sum() / count.clamp(min=1)
    return loss, count


class ObjectLoss(nn.Module):
    def __init__(
        self,
        *,
        object_temperature: float = 0.1,
        region_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.object_temperature = float(object_temperature)
        self.region_temperature = float(region_temperature)

    def forward(
        self,
        *,
        student_dense: torch.Tensor,
        teacher_dense: torch.Tensor,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        student_valid: torch.Tensor,
        teacher_valid: torch.Tensor,
        objects: torch.Tensor,
        negatives: torch.Tensor,
        student_region_head: Callable[[torch.Tensor], torch.Tensor],
        teacher_region_head: Callable[[torch.Tensor], torch.Tensor],
        n_views: int = 2,
    ) -> ObjectLossResult:
        """Dense/token tensors are [views * batch, P, C] (view-major); objects [B, P];
        negatives [B, M + 1, M + 1] indexed by object id.

        Computed in float32 with autocast disabled: the similarities feed masked
        log-sum-exps whose fill value does not fit in float16.
        """
        with torch.autocast(device_type=objects.device.type, enabled=False):
            return self._forward(
                student_dense=student_dense, teacher_dense=teacher_dense, student_tokens=student_tokens,
                teacher_tokens=teacher_tokens, student_valid=student_valid, teacher_valid=teacher_valid,
                objects=objects, negatives=negatives, student_region_head=student_region_head,
                teacher_region_head=teacher_region_head, n_views=n_views,
            )

    def _forward(
        self,
        *,
        student_dense,
        teacher_dense,
        student_tokens,
        teacher_tokens,
        student_valid,
        teacher_valid,
        objects,
        negatives,
        student_region_head,
        teacher_region_head,
        n_views,
    ) -> ObjectLossResult:
        if n_views != 2:
            raise ValueError("object losses compare exactly two global views")
        batch, points = objects.shape
        split = lambda tensor: tensor.reshape(n_views, batch, *tensor.shape[1:])
        student_dense, teacher_dense = split(student_dense.float()), split(teacher_dense.float())
        student_tokens, teacher_tokens = split(student_tokens.float()), split(teacher_tokens.float())
        student_valid, teacher_valid = split(student_valid), split(teacher_valid)

        labeled = objects > 0
        same_object = labeled[:, :, None] & (objects[:, :, None] == objects[:, None, :])
        index = objects.clamp(min=0)
        different = torch.gather(negatives, 1, index[:, :, None].expand(-1, -1, negatives.shape[-1]))
        different = torch.gather(different, 2, index[:, None, :].expand(-1, points, -1))
        different &= labeled[:, :, None] & labeled[:, None, :]

        point_terms, point_masks, object_terms, object_masks = [], [], [], []
        for view in range(n_views):
            other = 1 - view
            # Point term: agreement with the teacher at the same physical point (pull only).
            point_terms.append(1.0 - (student_dense[view] * teacher_dense[other]).sum(-1))
            point_masks.append(student_valid[view] & teacher_valid[other])

            logits = student_dense[view] @ teacher_dense[other].transpose(1, 2) / self.object_temperature
            keys_valid = teacher_valid[other][:, None, :]

            positives = same_object & keys_valid
            negative_keys = different & keys_valid
            object_anchor = (
                student_valid[view] & labeled & positives.any(-1) & negative_keys.any(-1)
            )
            object_terms.append(
                _masked_logsumexp(logits, positives | negative_keys) - _masked_logsumexp(logits, positives)
            )
            object_masks.append(object_anchor)
        point_loss, point_anchors = _mean_over(torch.stack(point_terms), torch.stack(point_masks))
        object_loss, object_anchors = _mean_over(torch.stack(object_terms), torch.stack(object_masks))

        region_loss, region_anchors = self._region_loss(
            student_tokens, teacher_tokens, student_valid, teacher_valid, objects, negatives,
            student_region_head, teacher_region_head,
        )
        return ObjectLossResult(point_loss, object_loss, region_loss, point_anchors, object_anchors, region_anchors)

    def _region_loss(
        self,
        student_tokens,
        teacher_tokens,
        student_valid,
        teacher_valid,
        objects,
        negatives,
        student_region_head,
        teacher_region_head,
    ):
        n_objects = negatives.shape[-1]
        one_hot = F.one_hot(objects, n_objects).float()  # [B, P, M + 1]; column 0 = unlabeled

        def pool(tokens, valid):
            weights = one_hot[None] * valid[..., None].float()  # [V, B, P, M + 1]
            counts = weights.sum(2)
            pooled = torch.einsum("vbpm,vbpc->vbmc", weights, tokens) / counts.clamp(min=1.0)[..., None]
            return pooled, counts > 0

        student_pooled, student_present = pool(student_tokens, student_valid)
        with torch.no_grad():
            teacher_pooled, teacher_present = pool(teacher_tokens, teacher_valid)
            teacher_embeddings = F.normalize(teacher_region_head(teacher_pooled).float(), dim=-1)
        student_embeddings = F.normalize(student_region_head(student_pooled).float(), dim=-1)

        terms, masks = [], []
        real = torch.ones(n_objects, dtype=torch.bool, device=objects.device)
        real[0] = False
        for view in range(2):
            other = 1 - view
            logits = student_embeddings[view] @ teacher_embeddings[other].transpose(1, 2) / self.region_temperature
            negative_keys = negatives & teacher_present[other][:, None, :]
            positive = torch.diagonal(logits, dim1=1, dim2=2)
            anchors = student_present[view] & teacher_present[other] & negative_keys.any(-1) & real
            keys = negative_keys | torch.eye(n_objects, dtype=torch.bool, device=objects.device)[None]
            terms.append(_masked_logsumexp(logits, keys) - positive)
            masks.append(anchors)
        return _mean_over(torch.stack(terms), torch.stack(masks))
