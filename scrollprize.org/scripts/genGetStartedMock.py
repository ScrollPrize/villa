# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
Generate SYNTHETIC stand-in assets for the /get_started interactive demos.

Run:  uv run scripts/genGetStartedMock.py

Everything this script writes into static/get-started/ is a placeholder that
matches the data contract the demo components code against. To swap in real
data, replace the image files and regenerate/adjust manifest.json — the
components read only the manifest and the files it names.

=============================================================================
DATA CONTRACT — ink demo (static/get-started/ink/)
=============================================================================
manifest.json:
  width, height     image dimensions (all three images share them)
  papyrus           grayscale surface render the user draws on (webp).
                    REAL DATA: a crackle-rich surface-volume patch where
                    letters are faintly visible to the naked eye.
  label             lossless binary ink mask, white = ink (png).
                    REAL DATA: the GT ink label for the same patch.
  prediction        precomputed model inference, white = ink (webp).
                    REAL DATA: the real model's probability map, same patch.
  brushRadius       stroke radius in IMAGE pixels (fixed-size brush)
  revealRadius      radius (image px) around a correct stroke inside which
                    the prediction is revealed
  letters           bounding boxes of individual letters [{x,y,w,h}, ...]
                    (used for the "model generalizes to a letter you didn't
                    label" moment). Computed here from the label's connected
                    components; for real data either supply them or let the
                    generator recompute from the real label.

=============================================================================
DATA CONTRACT — segmentation demo (static/get-started/seg/)
=============================================================================
manifest.json:
  dotRadius         dot marker radius in slice-image px
  tolerancePx       distance (slice px) from the GT surface polyline within
                    which a dot counts as "on the surface"; quality decays
                    beyond it up to maxDistPx (-> worst offset render)
  maxDistPx         distance at which quality reaches 0
  slices[3]         one entry per selectable slice plane (top/middle/bottom)
    id, label       "top" | "middle" | "bottom"
    image           cross-section image the user places dots on (webp)
    width, height   its dimensions
    surface         GT polyline of the TARGET WRAP in slice px [[x,y],...],
                    ordered by arc length.
                    REAL DATA: the tifxyz surface intersected with this
                    z-slice, projected to slice coords.
    vBand           [y0,y1) rows of the render image this slice controls
                    (the tifxyz v-range covered by this z-slice)
  render:
    width, height   render image dimensions (all offsets share them)
    offsets         normal offsets in voxels, ascending, offsets[0] == 0
    images          one render per offset; images[0] is the on-surface
                    (correct) render, later ones step further along the
                    surface normal and look progressively wrong.
                    REAL DATA: vc_render_tifxyz outputs at those
                    --slice-step offsets (max over a few slices per offset).

The runtime rule the SegDemo component implements:
  for each user dot -> nearest polyline point gives (arcFraction t, dist d)
  quality q = clamp(1 - max(0, d - tolerancePx)/(maxDistPx - tolerancePx))
  q selects the offset image (q=1 -> offsets[0], q=0 -> last offset) in the
  render column band around x = t * render.width, rows = slice.vBand.
  Regions with no nearby dot stay dark ("not yet segmented").
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RNG = np.random.default_rng(79)  # 79 AD :)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "get-started"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


# ---------------------------------------------------------------------------
# shared texture helpers
# ---------------------------------------------------------------------------
def smooth_noise(shape, scale, rng=RNG):
    """Low-frequency noise in [0,1] by upsampling a small random grid."""
    h, w = shape
    small = rng.random((max(2, h // scale), max(2, w // scale)))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC
    )
    return np.asarray(img, dtype=np.float32) / 255.0


def papyrus_texture(h, w, base=118.0, rng=RNG):
    """Grayscale papyrus: vertical fiber striations + low-freq mottling."""
    # vertical fibers: per-column 1D noise, smoothed
    cols = rng.random(w)
    kernel = np.ones(5) / 5
    cols = np.convolve(cols, kernel, mode="same")
    fibers_v = np.tile(cols, (h, 1)) * 26.0
    # faint horizontal fibers
    rows = np.convolve(rng.random(h), kernel, mode="same")
    fibers_h = np.tile(rows[:, None], (1, w)) * 10.0
    mottle = (smooth_noise((h, w), 48, rng) - 0.5) * 34.0
    grain = rng.normal(0, 3.2, (h, w))
    tex = base + fibers_v + fibers_h + mottle + grain
    return np.clip(tex, 0, 255).astype(np.float32)


def letter_mask(h, w, text_rows, font_size):
    """Render Greek text rows -> boolean mask + per-letter bboxes."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, font_size)
    boxes = []
    for text, (cx, cy) in text_rows:
        total_w = draw.textlength(text, font=font)
        x = cx - total_w / 2
        for ch in text:
            ch_w = draw.textlength(ch, font=font)
            if not ch.isspace():
                draw.text((x, cy), ch, fill=255, font=font)
                bbox = draw.textbbox((x, cy), ch, font=font)
                boxes.append(
                    {
                        "x": int(bbox[0]) - 4,
                        "y": int(bbox[1]) - 4,
                        "w": int(bbox[2] - bbox[0]) + 8,
                        "h": int(bbox[3] - bbox[1]) + 8,
                    }
                )
            x += ch_w
    mask = np.asarray(img, dtype=np.float32) / 255.0
    return mask, boxes


def save_webp(arr, path, quality=82):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        path, "WEBP", quality=quality
    )


# ---------------------------------------------------------------------------
# ink demo
# ---------------------------------------------------------------------------
def gen_ink():
    out = OUT / "ink"
    out.mkdir(parents=True, exist_ok=True)
    W, H = 1024, 640

    mask, boxes = letter_mask(
        H,
        W,
        [
            ("ΑΡΕΤΗ", (W / 2, 96)),
            ("ΗΔΟΝΗ", (W / 2, 330)),
        ],
        font_size=190,
    )
    soft = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.2)
        ),
        dtype=np.float32,
    ) / 255.0

    # papyrus the user draws on: letters appear only as subtle crackle
    tex = papyrus_texture(H, W)
    crackle = (smooth_noise((H, W), 3) > 0.62).astype(np.float32)
    crackle *= (smooth_noise((H, W), 9) > 0.45).astype(np.float32)
    papyrus = tex + soft * crackle * 34.0 + soft * 6.0
    save_webp(papyrus, out / "papyrus.webp")

    # GT label: clean binary mask (lossless)
    Image.fromarray(((mask > 0.5) * 255).astype(np.uint8)).save(
        out / "label.png", "PNG", optimize=True
    )

    # "model prediction": blurred label + noise + imperfections
    pred = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(2.5)
        ),
        dtype=np.float32,
    )
    dropout = smooth_noise((H, W), 14) > 0.22          # eat some regions
    pred = pred * (0.55 + 0.45 * dropout)
    pred += (smooth_noise((H, W), 6) - 0.5) * 26.0     # background wisps
    save_webp(pred, out / "prediction.webp", quality=80)

    manifest = {
        "version": 1,
        "width": W,
        "height": H,
        "papyrus": "papyrus.webp",
        "label": "label.png",
        "prediction": "prediction.webp",
        "brushRadius": 13,
        "revealRadius": 46,
        "letters": boxes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ink: {W}x{H}, {len(boxes)} letters -> {out}")


# ---------------------------------------------------------------------------
# segmentation demo
# ---------------------------------------------------------------------------
def spiral_slice(size, phase, wobble_seed):
    """Cross-section: an Archimedean spiral of papyrus wraps on dark bg.

    Returns (image float array, target-wrap polyline [[x,y],...]).
    """
    s = size
    img = np.full((s, s), 16.0, dtype=np.float32)
    img += (smooth_noise((s, s), 40, RNG) - 0.5) * 10.0

    cx = cy = s / 2
    a, b = 26.0, 11.5
    rng = np.random.default_rng(wobble_seed)
    wobble_amp = rng.uniform(2.0, 4.0)
    wobble_freq = rng.uniform(2.0, 3.5)

    canvas = Image.fromarray(np.zeros((s, s), dtype=np.uint8))
    draw = ImageDraw.Draw(canvas)
    thetas = np.linspace(0, 6.2 * math.pi, 2400)
    pts = []
    for t in thetas:
        r = a + b * t / (2 * math.pi) * 2 * math.pi / (2 * math.pi)
        r = a + b * t
        r = a + 11.5 * t  # px per radian -> ~72px per winding
        r += wobble_amp * math.sin(wobble_freq * t + phase)
        pts.append((cx + r * math.cos(t + phase), cy + r * math.sin(t + phase)))
    draw.line(pts, fill=255, width=7)
    sheet = np.asarray(canvas, dtype=np.float32) / 255.0

    # texture the sheet like papyrus
    tex = papyrus_texture(s, s, base=120.0)
    img = img * (1 - sheet) + tex * sheet
    img += sheet * (smooth_noise((s, s), 12) - 0.5) * 22.0

    # target wrap: one full winding, mid-spiral
    target = []
    for t in np.linspace(2.6 * math.pi, 4.6 * math.pi, 120):
        r = a + 11.5 * t + wobble_amp * math.sin(wobble_freq * t + phase)
        target.append(
            [
                round(cx + r * math.cos(t + phase), 1),
                round(cy + r * math.sin(t + phase), 1),
            ]
        )
    return np.clip(img, 0, 255), target


def gen_seg():
    out = OUT / "seg"
    out.mkdir(parents=True, exist_ok=True)
    S = 640                 # slice size
    RW, RH = 800, 480       # render size
    offsets = [0, 6, 14, 28]

    # --- three cross-section slices ---
    slices = []
    for i, sid in enumerate(["top", "middle", "bottom"]):
        img, target = spiral_slice(S, phase=0.5 + 0.35 * i, wobble_seed=100 + i)
        name = f"slice_{sid}.webp"
        save_webp(img, out / name)
        slices.append(
            {
                "id": sid,
                "label": sid.capitalize(),
                "image": name,
                "width": S,
                "height": S,
                "surface": target,
                "vBand": [i * RH // 3, (i + 1) * RH // 3],
            }
        )

    # --- render stack: offset 0 = crisp letter, later = weird ---
    mask, _ = letter_mask(RH, RW, [("Ω", (RW / 2, 40))], font_size=340)
    soft = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.5)
        ),
        dtype=np.float32,
    ) / 255.0
    base = papyrus_texture(RH, RW, base=128.0)
    render0 = base - soft * 88.0  # carbon ink: darker than papyrus

    images = []
    for k, off in enumerate(offsets):
        if off == 0:
            arr = render0
        else:
            # stepping off-surface: smear (max over shifted copies), blur,
            # fade the letter, drift brightness — "weird papyrus"
            im = Image.fromarray(np.clip(render0, 0, 255).astype(np.uint8))
            stack = []
            for dy in range(-off // 2, off // 2 + 1, max(1, off // 6)):
                stack.append(
                    np.asarray(
                        im.transform(
                            im.size,
                            Image.AFFINE,
                            (1, 0.04 * (k), dy * 0.6, 0.02 * k, 1, dy),
                        ),
                        dtype=np.float32,
                    )
                )
            arr = np.maximum.reduce(stack)
            arr = np.asarray(
                Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(0.8 * k)
                ),
                dtype=np.float32,
            )
            fade = 1.0 - 0.85 * (k / (len(offsets) - 1))
            arr = base * (1 - fade) + arr * fade
            arr += (smooth_noise((RH, RW), 10) - 0.5) * (14.0 * k)
        name = f"render_o{off}.webp"
        save_webp(arr, out / name)
        images.append(name)

    manifest = {
        "version": 1,
        "dotRadius": 9,
        "tolerancePx": 12,
        "maxDistPx": 60,
        "slices": slices,
        "render": {"width": RW, "height": RH, "offsets": offsets, "images": images},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"seg: 3 slices {S}px, render {RW}x{RH}, offsets {offsets} -> {out}")


if __name__ == "__main__":
    gen_ink()
    gen_seg()
