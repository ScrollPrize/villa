"""Distributed per-scroll metrics and aligned, fixed-scale 3D/2D previews."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from vesuvius.ink_detection.training.metrics import Confusion
from vesuvius.ink_detection.types import MetricBatch


class ScrollMetrics:
    """Exact confusion/BCE and a 4096-bin approximation to PR area."""

    bins = 4096

    def __init__(self, records, device):
        self.scrolls = sorted({r["scroll"] for r in records})
        self.record_scroll = [self.scrolls.index(r["scroll"]) for r in records]
        self.state = torch.zeros((len(self.scrolls), 10 + 2*self.bins),
                                 dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, output, batch, target):
        for i, record_id in enumerate(batch["record_id"].tolist()):
            row = self.state[self.record_scroll[record_id]]
            logits = output["ink"][i].float()
            y, m = batch["labels_2d"][i], batch["mask_2d"][i].bool()
            counts = Confusion().compute_batch(MetricBatch(logits=logits, targets=y, valid_mask=m))
            row[:4] += torch.stack((counts.tp, counts.fp, counts.fn, counts.tn))
            row[4] += F.binary_cross_entropy_with_logits(logits[m], y[m], reduction="sum")
            row[5] += m.sum()
            valid = batch["valid_3d"][i].bool()
            pred = output["ink_3d_logits"][i].float()
            row[6] += F.binary_cross_entropy_with_logits(pred[valid], target[i][valid], reduction="sum")
            row[7] += valid.sum()
            row[8] += (pred.sigmoid()[valid] - target[i][valid]).square().sum()
            row[9] += 1
            b = (logits.sigmoid()[m] * (self.bins - 1)).long()
            positive = y[m] > 0.5
            row[10:10+self.bins] += torch.bincount(b[positive], minlength=self.bins)
            row[10+self.bins:] += torch.bincount(b[~positive], minlength=self.bins)

    def compute(self, accelerator):
        state = accelerator.reduce(self.state, reduction="sum").cpu().numpy()
        result, per_scroll = {}, []
        for scroll, row in zip(self.scrolls, state):
            tp, fp, fn, tn, bce, n, bce3, n3, mse3, patches = row[:10]
            if patches == 0:
                continue
            positives, negatives = tp + fn, tn + fp
            recall = tp / positives if positives else float("nan")
            specificity = tn / negatives if negatives else float("nan")
            pos = row[10:10+self.bins][::-1].cumsum()
            neg = row[10+self.bins:][::-1].cumsum()
            r = pos / max(positives, 1)
            p = pos / np.maximum(pos + neg, 1)
            area = float(np.sum(np.diff(np.r_[0., r]) * p)) if positives else float("nan")
            metrics = {"bce": bce/max(n, 1), "dice": 2*tp/max(2*tp+fp+fn, 1),
                       "balanced_accuracy": (recall+specificity)/2,
                       "precision": tp/max(tp+fp, 1), "recall": recall,
                       "pr_auc": area, "teacher_bce_3d": bce3/max(n3, 1),
                       "teacher_mse_3d": mse3/max(n3, 1), "valid_pixels": n,
                       "positive_pixels": positives, "negative_pixels": negatives,
                       "patches": patches}
            result.update({f"val/{scroll}/{k}": float(v) for k, v in metrics.items()})
            per_scroll.append(metrics)
        for key in ("bce", "dice", "balanced_accuracy", "precision", "recall", "pr_auc",
                    "teacher_bce_3d", "teacher_mse_3d"):
            values = [m[key] for m in per_scroll if np.isfinite(m[key])]
            result[f"val/macro/{key}"] = float(np.mean(values)) if values else float("nan")
        return result


def save_preview(path, batch, output, targets, raw_targets, records, sample=0):
    """One readable montage: CT, raw/gated teacher, and student at equal depths."""
    def array(t):
        return t[sample, 0].detach().float().cpu().numpy()
    raw = array(batch["raw"])
    teacher, target = array(raw_targets), array(targets)
    student = array(output["ink_3d_logits"].sigmoid())
    weights = array(output["depth_weights"])
    slices = (0, 8, 16, 24, 32, 40, 48, 56, 63)
    tile, left, header, row_height = 128, 115, 55, 150
    canvas = Image.new("RGB", (left + tile*len(slices), header + row_height*6), "#202020")
    draw = ImageDraw.Draw(canvas)
    record = records[int(batch["record_id"][sample])]
    yx = batch["yx"][sample].tolist()
    caption = f"{record['scroll']} / {record['segment']}  yx={yx}  teacher={record['teacher_id']}"
    if 'reverse_depth' in batch:
        caption += f"  reversed={bool(batch['reverse_depth'][sample])}"
    draw.text((8, 5), caption, fill="white")
    draw.text((8, 23), "CT: 0..255; probabilities/masks: 0..1; depth weights displayed x8 (fixed scale).", fill="white")
    def paste(a, col, row, title):
        a = np.nan_to_num(a, nan=0, posinf=1, neginf=0)
        pic = Image.fromarray(np.uint8(np.clip(a, 0, 1)*255)).resize((tile, tile))
        x, y = left+col*tile, header+row*row_height
        canvas.paste(pic.convert("RGB"), (x, y))
        draw.text((x+2, y+tile+2), title, fill="white")
    for row, (name, volume) in enumerate((("CT", raw/255), ("Teacher raw", teacher),
                                          ("3D target", target), ("Student 3D", student),
                                          ("Depth weight x8", weights*8))):
        draw.text((5, header+row*row_height+50), name, fill="white")
        for col, z in enumerate(slices):
            paste(volume[z], col, row, f"z={z} off={z-32:+d}")
    extras = ((array(batch["labels_2d"]), "Human 2D label"),
              (array(batch["mask_2d"]), "2D supervision"),
              (array(output["ink"].sigmoid()), "Student 2D"),
              ((weights*np.arange(64)[:, None, None]).sum(0)/63, "Selected depth"),
              (target[:, 128], "Target XZ"), (student[:, 128], "Student XZ"),
              (target[:, :, 128], "Target YZ"), (student[:, :, 128], "Student YZ"),
              (array(batch["valid_3d"])[32], "Valid 3D support"))
    for col, (a, label) in enumerate(extras):
        paste(a, col, 5, label)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return caption
