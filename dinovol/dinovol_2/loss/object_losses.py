"""Point, object, region, and cross-batch ink losses between global views (student vs. EMA teacher).

All terms compare a student view with the teacher's other view:

- point: optional pull-only cosine agreement at the same physical point.
- object: margin-capped SupCon (L_out, mean over positives outside the log, excluding the
  self-point diagonal) on dense object embeddings of points on the same sheet or fiber vs.
  points on provably different objects in the same crop.
- ink: cross-view and cross-batch margin-capped SupCon on the separate `ink` readout,
  pulling confirmed ink points (+1) together across views and across batches while
  contrasting against confirmed non-ink surface points (-1) within the supervision mask.
- region: token features averaged over each object (DetCon-style), projected, and
  contrasted at the object level.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    ink: torch.Tensor | None = None
    ink_anchors: torch.Tensor | None = None


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
        object_temperature: float = 0.15,
        region_temperature: float = 0.15,
        ink_temperature: float = 0.15,
        pos_margin: float = 0.75,
        ink_queue_size: int = 512,
        ink_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.object_temperature = float(object_temperature)
        self.region_temperature = float(region_temperature)
        self.ink_temperature = float(ink_temperature)
        self.pos_margin = float(pos_margin)
        self.ink_queue_size = int(ink_queue_size)
        self.ink_dim = int(ink_dim) if ink_dim is not None else 0
        self.register_buffer("ink_pos_queue", torch.empty((0, self.ink_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("ink_neg_queue", torch.empty((0, self.ink_dim), dtype=torch.float32), persistent=False)

    @property
    def ink_pos_count(self) -> torch.Tensor:
        return torch.tensor(self.ink_pos_queue.shape[0], device=self.ink_pos_queue.device)

    @property
    def ink_neg_count(self) -> torch.Tensor:
        return torch.tensor(self.ink_neg_queue.shape[0], device=self.ink_neg_queue.device)

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
        student_ink: torch.Tensor | None = None,
        teacher_ink: torch.Tensor | None = None,
        ink: torch.Tensor | None = None,
        n_views: int = 2,
    ) -> ObjectLossResult:
        """Dense/token/ink tensors are [views * batch, P, C] (view-major); objects [B, P];
        negatives [B, M + 1, M + 1] indexed by object id; ink [B, P] in {-1, 0, +1}.

        Computed in float32 with autocast disabled: the similarities feed masked
        log-sum-exps whose fill value does not fit in float16.
        """
        with torch.autocast(device_type=objects.device.type, enabled=False):
            return self._forward(
                student_dense=student_dense,
                teacher_dense=teacher_dense,
                student_tokens=student_tokens,
                teacher_tokens=teacher_tokens,
                student_valid=student_valid,
                teacher_valid=teacher_valid,
                objects=objects,
                negatives=negatives,
                student_region_head=student_region_head,
                teacher_region_head=teacher_region_head,
                student_ink=student_ink,
                teacher_ink=teacher_ink,
                ink=ink,
                n_views=n_views,
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
        student_ink,
        teacher_ink,
        ink,
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
        not_self = ~torch.eye(points, dtype=torch.bool, device=objects.device)[None]
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

            sim = student_dense[view] @ teacher_dense[other].transpose(1, 2)
            logits = sim / self.object_temperature
            pos_logits = (sim.clamp(max=self.pos_margin) if self.pos_margin < 1.0 else sim) / self.object_temperature
            keys_valid = teacher_valid[other][:, None, :]

            # Exclude the self-point diagonal whenever another valid point on the same object exists.
            positives_non_self = same_object & keys_valid & not_self
            positives = torch.where(
                positives_non_self.any(-1, keepdim=True),
                positives_non_self,
                same_object & keys_valid,
            )
            negative_keys = different & keys_valid
            object_anchor = (
                student_valid[view] & labeled & positives.any(-1) & negative_keys.any(-1)
            )
            # SupCon L_out: mean over positives outside the log.
            neg_lse = _masked_logsumexp(logits, negative_keys)[..., None]
            per_pos = F.softplus(neg_lse - pos_logits)
            pos_count = positives.sum(-1).clamp(min=1)
            object_terms.append(torch.where(positives, per_pos, torch.zeros_like(per_pos)).sum(-1) / pos_count)
            object_masks.append(object_anchor)
        point_loss, point_anchors = _mean_over(torch.stack(point_terms), torch.stack(point_masks))
        object_loss, object_anchors = _mean_over(torch.stack(object_terms), torch.stack(object_masks))

        region_loss, region_anchors = self._region_loss(
            student_tokens, teacher_tokens, student_valid, teacher_valid, objects, negatives,
            student_region_head, teacher_region_head,
        )
        ink_loss, ink_anchors = self._ink_loss(
            student_ink, teacher_ink, student_valid, teacher_valid, ink, n_views=n_views
        )
        return ObjectLossResult(
            point_loss,
            object_loss,
            region_loss,
            point_anchors,
            object_anchors,
            region_anchors,
            ink=ink_loss,
            ink_anchors=ink_anchors,
        )

    def _ink_loss(
        self,
        student_ink: torch.Tensor | None,
        teacher_ink: torch.Tensor | None,
        student_valid: torch.Tensor,
        teacher_valid: torch.Tensor,
        ink: torch.Tensor | None,
        *,
        n_views: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = student_valid.device
        zero = torch.zeros((), dtype=torch.float32, device=device)
        zero_count = torch.zeros((), dtype=torch.long, device=device)
        if student_ink is None or teacher_ink is None:
            return zero, zero_count
        # Keep ink readout parameters in the autograd graph even on batches without ink labels.
        zero_touch = 0.0 * student_ink.float().sum()
        if ink is None or not bool((ink != 0).any()):
            return zero + zero_touch, zero_count

        batch, points = ink.shape
        s_ink = student_ink.float().reshape(n_views, batch, points, -1)
        t_ink = teacher_ink.float().reshape(n_views, batch, points, -1).detach()
        dim = s_ink.shape[-1]

        q_pos = (
            self.ink_pos_queue.to(device=device, dtype=torch.float32)
            if self.ink_pos_queue.numel() > 0 and self.ink_pos_queue.shape[-1] == dim
            else s_ink.new_empty((0, dim))
        )
        q_neg = (
            self.ink_neg_queue.to(device=device, dtype=torch.float32)
            if self.ink_neg_queue.numel() > 0 and self.ink_neg_queue.shape[-1] == dim
            else s_ink.new_empty((0, dim))
        )

        loss_terms: list[torch.Tensor] = []
        total_anchors = torch.zeros((), dtype=torch.long, device=device)

        for view in range(n_views):
            other = 1 - view
            s_flat = s_ink[view].reshape(batch * points, dim)
            t_flat = t_ink[other].reshape(batch * points, dim)
            s_val = student_valid[view].reshape(batch * points)
            t_val = teacher_valid[other].reshape(batch * points)
            ink_flat = ink.reshape(batch * points)

            s_pos_idx = torch.nonzero(s_val & (ink_flat == 1), as_tuple=False).squeeze(-1)
            s_neg_idx = torch.nonzero(s_val & (ink_flat == -1), as_tuple=False).squeeze(-1)
            t_pos_idx = torch.nonzero(t_val & (ink_flat == 1), as_tuple=False).squeeze(-1)
            t_neg_idx = torch.nonzero(t_val & (ink_flat == -1), as_tuple=False).squeeze(-1)

            pos_keys = torch.cat([t_flat[t_pos_idx], q_pos], dim=0) if t_pos_idx.numel() else q_pos
            neg_keys = torch.cat([t_flat[t_neg_idx], q_neg], dim=0) if t_neg_idx.numel() else q_neg

            # Require confirmed ink (+1) anchors in the view so non-ink-only crops do not
            # pull all embeddings toward a single cluster without opposing ink anchors.
            if s_pos_idx.numel() > 0 and pos_keys.shape[0] > 0 and neg_keys.shape[0] > 0:
                anchors = s_flat[s_pos_idx]
                sim_pos = anchors @ pos_keys.T
                if t_pos_idx.numel() > 0:
                    # Exclude exact same physical point in the other view when >1 positive keys exist.
                    same_pt = s_pos_idx[:, None] == t_pos_idx[None, :]
                    if pos_keys.shape[0] > t_pos_idx.numel():
                        pad = torch.zeros(
                            (s_pos_idx.numel(), pos_keys.shape[0] - t_pos_idx.numel()),
                            dtype=torch.bool,
                            device=device,
                        )
                        same_pt = torch.cat([same_pt, pad], dim=1)
                    pos_mask = ~same_pt
                    pos_mask = torch.where(pos_mask.any(-1, keepdim=True), pos_mask, torch.ones_like(pos_mask))
                else:
                    pos_mask = torch.ones_like(sim_pos, dtype=torch.bool)
                # Do not clamp ink positive similarities with pos_margin: ink is a semantic
                # class that should converge to high cosine agreement across views/batches,
                # and clamping above pos_margin zeroes out the attractive gradient.
                pos_logits = sim_pos / self.ink_temperature
                neg_logits = (anchors @ neg_keys.T) / self.ink_temperature
                neg_lme = torch.logsumexp(neg_logits, dim=-1, keepdim=True) - math.log(neg_keys.shape[0])
                per_pos = F.softplus(neg_lme - pos_logits)
                anchor_losses = (
                    torch.where(pos_mask, per_pos, torch.zeros_like(per_pos)).sum(-1)
                    / pos_mask.sum(-1).clamp(min=1)
                )
                pos_view_loss = anchor_losses.mean()
                total_anchors = total_anchors + anchor_losses.numel()

                if s_neg_idx.numel() > 0:
                    neg_anchors = s_flat[s_neg_idx]
                    same_logits = (neg_anchors @ neg_keys.T) / self.ink_temperature
                    opp_logits = (neg_anchors @ pos_keys.T) / self.ink_temperature
                    opp_lme = torch.logsumexp(opp_logits, dim=-1, keepdim=True) - math.log(pos_keys.shape[0])
                    neg_anchor_losses = F.softplus(opp_lme - same_logits).mean(-1)
                    neg_view_loss = neg_anchor_losses.mean()
                    loss_terms.append(0.5 * (pos_view_loss + neg_view_loss))
                    total_anchors = total_anchors + neg_anchor_losses.numel()
                else:
                    loss_terms.append(pos_view_loss)

        if self.ink_queue_size > 0 and torch.is_grad_enabled():
            with torch.no_grad():
                all_t = t_ink.reshape(-1, dim)
                all_val = teacher_valid.reshape(-1)
                all_ink = ink[None].expand(n_views, -1, -1).reshape(-1)
                new_pos = all_t[all_val & (all_ink == 1)]
                new_neg = all_t[all_val & (all_ink == -1)]
                if new_pos.shape[0] > 0:
                    if new_pos.shape[0] > 128:
                        perm = torch.randperm(new_pos.shape[0], device=device)[:128]
                        new_pos = new_pos[perm]
                    updated_pos = torch.cat([q_pos, new_pos], dim=0)[-self.ink_queue_size:]
                    self.ink_pos_queue = updated_pos.detach()
                if new_neg.shape[0] > 0:
                    if new_neg.shape[0] > 128:
                        perm = torch.randperm(new_neg.shape[0], device=device)[:128]
                        new_neg = new_neg[perm]
                    updated_neg = torch.cat([q_neg, new_neg], dim=0)[-self.ink_queue_size:]
                    self.ink_neg_queue = updated_neg.detach()

        if not loss_terms:
            return zero + zero_touch, zero_count
        return torch.stack(loss_terms).mean() + zero_touch, total_anchors

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
            sim = student_embeddings[view] @ teacher_embeddings[other].transpose(1, 2)
            logits = sim / self.region_temperature
            pos_sim = torch.diagonal(sim, dim1=1, dim2=2)
            positive = (pos_sim.clamp(max=self.pos_margin) if self.pos_margin < 1.0 else pos_sim) / self.region_temperature
            negative_keys = negatives & teacher_present[other][:, None, :]
            anchors = student_present[view] & teacher_present[other] & negative_keys.any(-1) & real
            neg_lse = _masked_logsumexp(logits, negative_keys)
            terms.append(F.softplus(neg_lse - positive))
            masks.append(anchors)
        return _mean_over(torch.stack(terms), torch.stack(masks))
