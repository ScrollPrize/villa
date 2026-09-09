"""Independent human 2D supervision and scroll-routed soft 3D supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vesuvius.ink_detection.training.dynamic_labels import load_frozen_ink_model
from vesuvius.ink_detection.data.multiteacher import SCROLLS


def masked_sample_mean(values, mask):
    mask = mask.to(dtype=values.dtype).expand_as(values)
    axes = tuple(range(1, values.ndim))
    return (values * mask).sum(axes) / mask.sum(axes).clamp_min(1)


def joint_loss(outputs, batch, teacher_targets, teacher_weights, *, human_2d_weight=1.0,
               gradient_diagnostics=False):
    if not 0 < human_2d_weight < float("inf"):
        raise ValueError("human_2d_weight must be finite and positive")
    logits = outputs["ink"].float()
    labels = batch["labels_2d"].float()
    mask = batch["mask_2d"].float()
    bce_2d = masked_sample_mean(
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none"), mask)
    p = logits.sigmoid()
    axes = tuple(range(1, p.ndim))
    dice = 1 - (2 * (p * labels * mask).sum(axes) + 1e-5) / (
        ((p + labels) * mask).sum(axes) + 1e-5)
    dice = dice * (mask.sum(axes) > 0)
    bce_3d = masked_sample_mean(F.binary_cross_entropy_with_logits(
        outputs["ink_3d_logits"].float(), teacher_targets.float(), reduction="none"
    ), batch["valid_3d"])
    human = (bce_2d + 0.25 * dice).mean()
    distill = (bce_3d * teacher_weights).mean()
    metrics = {
        "loss/2d": human.detach(), "loss/2d_bce": bce_2d.mean().detach(),
        "loss/2d_weighted": (human_2d_weight*human).detach(),
        "loss/2d_dice": dice.mean().detach(),
        "loss/3d": bce_3d.mean().detach(), "loss/3d_weighted": distill.detach(),
    }
    if "teacher_id" in batch:
        for teacher_id, name in ((0, "precise"), (1, "coarse")):
            selected = batch["teacher_id"] == teacher_id
            metrics[f"teacher/{name}_fraction"] = selected.float().mean().detach()
            metrics[f"loss/3d_{name}_unweighted_contribution"] = (bce_3d*selected).mean().detach()
            metrics[f"loss/3d_{name}_contribution"] = (bce_3d*teacher_weights*selected).mean().detach()
    if gradient_diagnostics:
        # Observe the competing signals at the shared 3D-logit interface. These
        # calls do not accumulate parameter .grad or replace the actual backward.
        shared_logits = outputs.get("projection_3d_logits", outputs["ink_3d_logits"])
        human_grad, = torch.autograd.grad(human_2d_weight*human, shared_logits,
                                         retain_graph=True)
        teacher_grad, = torch.autograd.grad(distill, shared_logits, retain_graph=True)
        human_grad, teacher_grad = human_grad.detach().float(), teacher_grad.detach().float()
        metrics["gradient_at_3d_logits/human_squared"] = human_grad.square().sum()
        metrics["gradient_at_3d_logits/teacher_squared"] = teacher_grad.square().sum()
        metrics["gradient_at_3d_logits/dot"] = (human_grad*teacher_grad).sum()
    return human_2d_weight*human + distill, metrics


def finish_loss_metrics(values):
    """Compute teacher-conditional means after global/microbatch averaging."""
    for name in ("precise", "coarse"):
        fraction = values.get(f"teacher/{name}_fraction", 0.)
        if fraction:
            values[f"loss/3d_{name}"] = values[f"loss/3d_{name}_unweighted_contribution"]/fraction
            values[f"loss/3d_{name}_weighted"] = values[f"loss/3d_{name}_contribution"]/fraction
    for name in ("precise", "coarse"):
        for group in ("positive", "negative"):
            prefix = f"ink_retention/{name}/{group}"
            count = values.pop(prefix+"_count", 0.)
            total = values.pop(prefix+"_sum", 0.)
            if count:
                values[prefix+"_student_probability"] = total/count
    count = values.pop("attention/column_count", 0.)
    for name in ("effective_depth", "maximum_weight"):
        total = values.pop(f"attention/{name}_sum", 0.)
        if count:
            values[f"attention/{name}"] = total/count
    prefix = "gradient_at_3d_logits/"
    if prefix+"human_squared" in values:
        human = values.pop(prefix+"human_squared")**.5
        teacher = values.pop(prefix+"teacher_squared")**.5
        dot = values.pop(prefix+"dot")
        values[prefix+"human_rms_l2"] = human
        values[prefix+"teacher_rms_l2"] = teacher
        values[prefix+"human_to_teacher_ratio"] = human/max(teacher, 1e-12)
        values[prefix+"cosine"] = dot/max(human*teacher, 1e-12)
    return values


@torch.no_grad()
def suppression_statistics(outputs, batch, targets):
    """Sum/count diagnostics, combined across ranks and microbatches before division."""
    probability = outputs["ink_3d_logits"].detach().float().sigmoid()
    valid = batch["valid_3d"].bool()
    values = {}
    for teacher_id, name in ((0, "precise"), (1, "coarse")):
        selected = valid & (batch["teacher_id"] == teacher_id).view(-1, 1, 1, 1, 1)
        for group, mask in (("positive", targets >= .8), ("negative", targets <= .1)):
            mask = selected & mask
            prefix = f"ink_retention/{name}/{group}"
            values[prefix+"_sum"] = (probability*mask).sum()
            values[prefix+"_count"] = mask.sum().float()
    weights = outputs["depth_weights"].detach().float()
    columns = valid.any(2)
    entropy = -(weights*weights.clamp_min(1e-12).log()).sum(2)
    values["attention/effective_depth_sum"] = (entropy.exp()*columns).sum()
    values["attention/maximum_weight_sum"] = (weights.amax(2)*columns).sum()
    values["attention/column_count"] = columns.sum().float()
    return values


class FrozenInkTeachers:
    """Teacher ID 0 is Paris 4, ID 1 is the coarse multi-scroll checkpoint."""

    def __init__(self, config: dict, device):
        self.models = [load_frozen_ink_model(config[key]["checkpoint"],
                                           device=device, dtype=torch.float32)
                       for key in ("paris4", "coarse")]
        self.weights = torch.tensor([config[key]["weight"]
                                     for key in ("paris4", "coarse")], device=device)
        self.threshold = float(config["coarse"].get("background_threshold", 50))
        thresholds = config["coarse"].get("background_thresholds")
        self.threshold_by_scroll = None
        if thresholds is not None:
            required = set(SCROLLS) - {"phercparis4"}
            if not required <= thresholds.keys():
                raise ValueError(f"Missing calibrated thresholds: {required - thresholds.keys()}")
            if any(not 0 <= float(v) < 255 for v in thresholds.values()):
                raise ValueError("Raw uint8 background thresholds must be in [0,255)")
            self.threshold_by_scroll = torch.tensor(
                [float(thresholds.get(scroll, 0)) for scroll in SCROLLS], device=device)

    def dark_mask(self, batch):
        if self.threshold_by_scroll is None:
            threshold = self.threshold
        else:
            threshold = self.threshold_by_scroll[batch["scroll_id"]].view(-1, 1, 1, 1, 1)
        return ((batch["raw"] <= threshold)
                & (batch["teacher_id"] == 1).view(-1, 1, 1, 1, 1))

    @torch.no_grad()
    def generate(self, batch):
        ids = batch["teacher_id"]
        if not ((ids == 0) | (ids == 1)).all():
            raise ValueError("Unrecognized teacher ID")
        raw_probability = torch.empty_like(batch["teacher_image"], dtype=torch.float32)
        for teacher_id, teacher in enumerate(self.models):
            selected = ids == teacher_id
            if not selected.any():
                continue
            image = batch["teacher_image"][selected]
            prediction = (teacher.forward_3d(image) if hasattr(teacher, "forward_3d")
                          else teacher(image)["ink"])
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            raw_probability[selected] = prediction.float().sigmoid()
        targets = raw_probability.clone()
        targets.masked_fill_(self.dark_mask(batch), 0)
        return targets, self.weights[ids], raw_probability
