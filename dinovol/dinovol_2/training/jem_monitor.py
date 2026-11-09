"""JEM monitor image: one source's oriented views, masks, correspondence and feature PCA.

Each row is one student view of the first source and its set's teacher. Columns:
student input, student features, teacher input, teacher features. Inputs are one
axial slice of the final (oriented) view: red tint = masked token (zero-filled
input), dimmed = token without a teacher correspondence (student) or not targeted
by this student (teacher). Feature panels show the final normalized patch tokens
(student with its masks, teacher unmasked) projected onto the top three principal
components of the source's teacher tokens as RGB; one basis and one color scale for
every panel, so equal colors mean equal features. The student slice is the one with
the most overlapping tokens and the teacher slice the one with the most targeted
tokens. Orientations are independent, so matching structures appear
rotated/reflected between student and teacher.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dinovol_2.dataset.jem_views import zyx


def _gray(image: torch.Tensor, valid: np.ndarray | None = None) -> np.ndarray:
    """Slice scaled to [0, 1] by the 1st/99th percentiles of its valid (unmasked) voxels."""
    values = image.float().cpu().numpy()
    sample = values[valid] if valid is not None and valid.any() else values
    lo, hi = np.percentile(sample, [1, 99])
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


def _cells(cells: np.ndarray, patch) -> np.ndarray:
    """Token-cell array [h, w, ...] repeated to voxels [h * patch_y, w * patch_x, ...]."""
    return np.repeat(np.repeat(cells, patch[0], axis=0), patch[1], axis=1)


def _input_panel(gray, masked, kept, patch):
    """Gray slice with masked cells tinted red and cells outside `kept` dimmed."""
    rgb = np.repeat(gray[..., None], 3, -1)
    rgb = rgb * np.where(_cells(kept, patch), 1.0, 0.4)[..., None]
    red = _cells(masked, patch)[..., None]
    return rgb * (1 - 0.45 * red) + 0.45 * red * np.array([1.0, 0.15, 0.15])


def _busiest_slice(cells: np.ndarray) -> int:
    """Index along the first axis with the most set cells; ties go to the slice nearest the middle."""
    counts = cells.reshape(cells.shape[0], -1).sum(-1)
    best = np.flatnonzero(counts == counts.max())
    return int(best[np.abs(best - (cells.shape[0] - 1) / 2).argmin()])


def _patch_tokens(backbone, crops, masks, *, view_kind, device, dtype):
    """Final normalized patch tokens [B, N, C] in FP32 (masked cells already zero-filled)."""
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        features = backbone.forward_features(crops.to(device, dtype), masks, view_kind=view_kind, mask_voxels=False)
    return features["x_norm_patchtokens"].float()


@torch.no_grad()
def save_jem_monitor_image(method, batch, step, metrics, *, directory) -> Path | None:
    if not batch.get("jem"):
        return None
    patch = zyx(batch["patch"], "patch")
    plane = patch[1:]  # token cell edges within a z slice (y, x)
    B = int(batch["batch_size"])
    teacher_grid = tuple(batch["teacher_grid"])
    device, dtype = method.device, method.input_dtype
    module = method.model_module
    student_backbone, teacher_backbone = module.student.backbone, module.teacher.backbone
    was_training = student_backbone.training
    student_backbone.eval()  # no drop-path in the monitored student features
    try:
        n_sets = int(batch["n_sets"])
        teacher_rows = [s * B for s in range(n_sets)]  # source 0 of every set
        teacher_tokens = _patch_tokens(teacher_backbone, batch["teacher_crops"][teacher_rows], None,
                                       view_kind="global", device=device, dtype=dtype)
        views = []
        for kind, shown in (("global", 2), ("local", 3)):
            rows = [slot * B for slot in range(min(int(batch[f"n_{kind}_views"]), shown))]
            masks = batch[f"{kind}_masks"][rows].to(device)
            tokens = _patch_tokens(student_backbone, batch[f"{kind}_crops"][rows], masks,
                                   view_kind=kind, device=device, dtype=dtype)
            views += [(kind, slot, row, tokens[slot]) for slot, row in enumerate(rows)]
    finally:
        student_backbone.train(was_training)

    # Top-3 principal components of this source's teacher tokens; per-component color range
    # from their 1st/99th percentiles.
    reference = teacher_tokens.flatten(0, 1)
    mean = reference.mean(0)
    _, singular, vh = torch.linalg.svd(reference - mean, full_matrices=False)
    basis = vh[:3].T
    explained = float((singular[:3].square().sum() / singular.square().sum().clamp_min(1e-12)))
    projected = (reference - mean) @ basis
    lo, hi = torch.quantile(projected, 0.01, dim=0), torch.quantile(projected, 0.99, dim=0)

    def pca_rgb(tokens, grid, z):
        rgb = ((tokens - mean) @ basis - lo) / (hi - lo).clamp_min(1e-6)
        return rgb.clamp(0, 1).view(*grid, 3)[z].cpu().numpy()

    panels = []
    for kind, slot, row, tokens in views:
        grid = tuple(batch[f"{kind}_grid"])
        teacher_row = int(batch[f"{kind}_teacher_rows"][row])
        targets = batch[f"{kind}_targets"][row].view(grid).cpu().numpy()
        masks = batch[f"{kind}_masks"][row].view(grid).cpu().numpy()
        overlap = targets >= 0
        z = _busiest_slice(overlap)
        hit = np.zeros(int(np.prod(teacher_grid)), dtype=bool)
        hit[targets[overlap]] = True
        hit = hit.reshape(teacher_grid)
        tz = _busiest_slice(hit)
        student = batch[f"{kind}_crops"][row, 0, z * patch[0] + patch[0] // 2]
        teacher = batch["teacher_crops"][teacher_row, 0, tz * patch[0] + patch[0] // 2]
        s_input = _input_panel(_gray(student, ~_cells(masks[z], plane)), masks[z], overlap[z], plane)
        t_input = _input_panel(_gray(teacher), np.zeros_like(hit[tz]), hit[tz], plane)
        t_tokens = teacher_tokens[teacher_rows.index(teacher_row)]
        panels.append((f"{kind} slot {slot}  (student z {z}/{grid[0]}, teacher z {tz}/{teacher_grid[0]})",
                       [s_input, _cells(pca_rgb(tokens, grid, z), plane),
                        t_input, _cells(pca_rgb(t_tokens, teacher_grid, tz), plane)]))

    size, gap, header = 192, 8, 44
    columns = ("student", "student PCA", "teacher", "teacher PCA")
    canvas = Image.new("RGB", (len(columns) * (size + gap) + gap, header + len(panels) * (size + 18) + gap), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"step {step}  loss {metrics.get('loss', float('nan')):.4f}  "
                      f"PCA: top 3 of teacher tokens ({100 * explained:.0f}% variance)", fill="white")
    draw.text((5, 17), "red = masked token, dimmed = no correspondence", fill="white")
    for column, name in enumerate(columns):
        draw.text((gap + column * (size + gap), 31), name, fill="white")
    for index, (title, images) in enumerate(panels):
        y = header + index * (size + 18)
        draw.text((gap, y), title, fill="white")
        for column, image in enumerate(images):
            tile = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
            canvas.paste(tile, (gap + column * (size + gap), y + 14))
    path = Path(directory) / f"jem_monitor_step_{step:06d}.png"
    canvas.save(path)
    return path
