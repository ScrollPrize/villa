"""Fixed-palette categorical visualization for 5-class fiber labels.

The wandb default mask renderer assigns colors itself, so cross-image
consistency is not guaranteed. Tuning decisions therefore come from a
matplotlib figure where we control the palette explicitly via direct
RGB lookup (no colormap quirks when only 2-3 classes are present in
a given crop).

Class indices:
    0 background (always rendered as pure black)
    1 vertical fiber
    2 horizontal / angular fiber
    3 papyrus
    4 ink
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


CLASS_NAMES: dict[int, str] = {
    0: "bg",
    1: "vert fiber",
    2: "horiz fiber",
    3: "ink",
}
NUM_CLASSES = 4

# Hand-picked palette: black for class 0, then three distinct, colorblind-aware
# hues. Vert is warm (red-orange), horiz is cool (azure) so the warm/cool axis
# gives an immediate read of orientation. Ink is green (matches the internal
# sean_ink convention).
PALETTE_RGB = np.array(
    [
        [0.000, 0.000, 0.000],  # 0 bg          black
        [0.902, 0.298, 0.051],  # 1 vert        orange-red
        [0.051, 0.553, 0.949],  # 2 horiz       azure blue
        [0.196, 0.847, 0.302],  # 3 ink         green
    ],
    dtype=np.float32,
)
LISTED_CMAP = ListedColormap(PALETTE_RGB.tolist(), name="fiber4c")


def label_to_rgb(label_zyx: np.ndarray) -> np.ndarray:
    """Map an integer-label image to RGB via direct lookup. Out-of-range clipped to valid range."""
    return PALETTE_RGB[np.clip(label_zyx, 0, NUM_CLASSES - 1)]


def overlay_label_on_image(
    image_zyx: np.ndarray,
    label_zyx: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend the categorical label color over a grayscale image; bg pixels stay grayscale."""
    base = np.repeat(image_zyx[..., None], 3, axis=-1).astype(np.float32)
    lo, hi = float(base.min()), float(base.max())
    base = (base - lo) / max(1e-6, hi - lo)
    fg = label_to_rgb(label_zyx)
    keep = (label_zyx > 0).astype(np.float32)[..., None]
    return base * (1.0 - keep * alpha) + fg * (keep * alpha)


def _legend_patches() -> list[Patch]:
    return [
        Patch(facecolor=PALETTE_RGB[i].tolist(), edgecolor="white", label=f"{i}: {CLASS_NAMES[i]}")
        for i in range(NUM_CLASSES)
    ]


def _instance_color_map(instance_map_zyx: np.ndarray) -> np.ndarray:
    """Map instance ids to distinct colors (HSV cycle) for the watershed panel.

    Background (id 0) stays black. Other ids get colors from an HSV cycle so
    adjacent ids don't collapse into the same hue.
    """
    ids = instance_map_zyx
    out = np.zeros((*ids.shape, 3), dtype=np.float32)
    mask = ids > 0
    if not mask.any():
        return out
    uniq = np.unique(ids[mask])
    # Map id -> uniform-spaced hue
    n = uniq.size
    hsv = np.zeros((n, 3), dtype=np.float32)
    hsv[:, 0] = np.linspace(0.0, 1.0, n, endpoint=False)
    hsv[:, 1] = 0.78
    hsv[:, 2] = 0.95
    # Vectorized hsv -> rgb
    h6 = hsv[:, 0] * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    s = hsv[:, 1]
    v = hsv[:, 2]
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    rgb = np.zeros_like(hsv)
    rgb[i == 0] = np.stack([v, t, p], -1)[i == 0]
    rgb[i == 1] = np.stack([q, v, p], -1)[i == 1]
    rgb[i == 2] = np.stack([p, v, t], -1)[i == 2]
    rgb[i == 3] = np.stack([p, q, v], -1)[i == 3]
    rgb[i == 4] = np.stack([t, p, v], -1)[i == 4]
    rgb[i == 5] = np.stack([v, p, q], -1)[i == 5]
    id_to_rgb = dict(zip(uniq.tolist(), rgb.tolist()))
    palette_lut = np.zeros((int(uniq.max()) + 2, 3), dtype=np.float32)
    for k, c in id_to_rgb.items():
        palette_lut[int(k)] = c
    out = palette_lut[np.clip(ids, 0, palette_lut.shape[0] - 1)]
    return out


def make_debug_figure(
    *,
    image_zyx: np.ndarray,           # (Z, Y, X) fp32 in [0, 1]
    raw_zyx: np.ndarray | None,      # (Z, Y, X) uint8, original (unused in 4-class but kept for compat)
    label_zyx: np.ndarray,           # (Z, Y, X) int in {0..3}, pseudo-label
    fiber_prob_zyx: np.ndarray,      # (Z, Y, X) fp32 in [0, 1]
    ink_prob_zyx: np.ndarray,        # (Z, Y, X) fp32 in [0, 1]
    instance_map_zyx: np.ndarray,    # (Z, Y, X) int instance ids (0 = bg)
    student_pred_zyx: np.ndarray | None = None,  # optional argmax pred (Z, Y, X) in {0..3}
    z_index: int | None = None,
    title_suffix: str = "",
) -> "plt.Figure":
    """A 2x4 grid; row 0 = pseudo-label & image; row 1 = teacher probs & student."""
    z = image_zyx.shape[0] // 2 if z_index is None else int(z_index)

    # Constrained-layout handles spacing between rows automatically, so panel
    # titles in row 1 don't collide with row 0 imshow boxes (the bug the user
    # called out: "title of the first row is not visible"). Extra vertical
    # height gives suptitle + legend their own breathing room.
    fig, axes = plt.subplots(
        2, 4,
        figsize=(4 * 4, 4 * 2 + 1.8),
        constrained_layout=True,
    )

    panels_top: list[tuple[str, np.ndarray, str | None]] = [
        ("Image", image_zyx[z], "gray"),
        ("Pseudo-label overlay", overlay_label_on_image(image_zyx[z], label_zyx[z]), None),
        ("Pseudo-label (categorical)", label_to_rgb(label_zyx[z]), None),
        (
            "Student argmax" if student_pred_zyx is not None else "(student n/a)",
            label_to_rgb(student_pred_zyx[z]) if student_pred_zyx is not None else np.zeros_like(image_zyx[z]),
            None if student_pred_zyx is not None else "gray",
        ),
    ]
    panels_bot: list[tuple[str, np.ndarray, str | None]] = [
        ("Fiber teacher prob", fiber_prob_zyx[z], "magma"),
        ("Ink teacher prob", ink_prob_zyx[z], "magma"),
        ("Watershed instances", _instance_color_map(instance_map_zyx[z]), None),
        ("Student-vs-pseudo (diff)",
         (
             label_to_rgb(np.where(student_pred_zyx[z] == label_zyx[z], 0, label_zyx[z]).astype(np.int32))
             if student_pred_zyx is not None else np.zeros_like(image_zyx[z])
         ),
         None if student_pred_zyx is not None else "gray"),
    ]

    for col in range(4):
        for row, panels in enumerate((panels_top, panels_bot)):
            ax = axes[row, col]
            title, arr, cmap = panels[col]
            if cmap is None:
                ax.imshow(arr)
            else:
                if title in ("Fiber teacher prob", "Ink teacher prob"):
                    ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(arr, cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

    fig.legend(
        handles=_legend_patches(),
        ncols=NUM_CLASSES,
        loc="outside lower center",
        fontsize=11,
        frameon=False,
    )
    fig.suptitle(
        f"4-class pseudo-label / debug — z={z}{(' — ' + title_suffix) if title_suffix else ''}",
        fontsize=13,
    )
    return fig


def categorical_class_labels() -> dict[int, str]:
    return dict(CLASS_NAMES)
