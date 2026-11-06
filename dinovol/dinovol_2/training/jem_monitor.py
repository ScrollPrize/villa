"""JEM monitor image: one source's oriented views, student masks and correspondence support.

Each row is one student view of the first source next to its set's teacher, as the
middle slice along the first axis of each final (oriented) view. Student overlays:
red = masked token, green = token with a valid teacher correspondence (O).
Teacher overlay: blue = teacher tokens targeted by this student. Orientations are
independent, so matching structures appear rotated/reflected between panels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _gray(image: torch.Tensor) -> np.ndarray:
    values = image.float().cpu().numpy()
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay(gray, cells, color, patch, alpha=0.35):
    rgb = np.repeat(gray[..., None], 3, -1)
    mask = np.kron(cells.astype(np.float32), np.ones((patch, patch), dtype=np.float32))[: gray.shape[0], : gray.shape[1]]
    return rgb * (1 - alpha * mask[..., None]) + alpha * mask[..., None] * np.asarray(color, dtype=np.float32)


@torch.no_grad()
def save_jem_monitor_image(method, batch, step, metrics, *, directory) -> Path | None:
    if not batch.get("jem"):
        return None
    patch = int(batch["patch"])
    B = int(batch["batch_size"])
    teacher_grid = tuple(batch["teacher_grid"])
    panels = []
    for kind in ("global", "local"):
        grid = tuple(batch[f"{kind}_grid"])
        slots = int(batch[f"n_{kind}_views"])
        for slot in range(min(slots, 2 if kind == "global" else 3)):
            row = slot * B
            teacher_row = int(batch[f"{kind}_teacher_rows"][row])
            student = batch[f"{kind}_crops"][row, 0]
            teacher = batch["teacher_crops"][teacher_row, 0]
            z = grid[0] // 2
            s_slice = _gray(student[z * patch + patch // 2])
            masks = batch[f"{kind}_masks"][row].view(grid).cpu().numpy()[z]
            targets = batch[f"{kind}_targets"][row].view(grid).cpu().numpy()
            s_rgb = _overlay(s_slice, masks, (1, 0, 0), patch)
            s_rgb = _overlay(s_rgb.mean(-1), targets[z] >= 0, (0, 1, 0), patch, alpha=0.2) * 0.5 + s_rgb * 0.5
            hit = np.zeros(int(np.prod(teacher_grid)), dtype=bool)
            hit[targets[targets >= 0]] = True
            hit = hit.reshape(teacher_grid)
            tz = teacher_grid[0] // 2
            t_rgb = _overlay(_gray(teacher[tz * patch + patch // 2]), hit[tz], (0, 0, 1), patch)
            panels.append((f"{kind} slot {slot}", s_rgb, t_rgb))
    size = 256
    canvas = Image.new("RGB", (2 * size + 30, len(panels) * (size + 20) + 30), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"step {step}  loss {metrics.get('loss', float('nan')):.4f}", fill="white")
    for index, (title, student, teacher) in enumerate(panels):
        y = 25 + index * (size + 20)
        draw.text((5, y), title, fill="white")
        for column, image in enumerate((student, teacher)):
            tile = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
            canvas.paste(tile, (10 + column * (size + 10), y + 12))
    path = Path(directory) / f"jem_monitor_step_{step:06d}.png"
    canvas.save(path)
    return path
