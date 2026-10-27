"""Quick looks at a volume without opening a viewer: a slice, a sweep, a projection or an orbit.

Reads only the chunks it draws, so it works on a remote OME-Zarr as well as a local one.
Examples::

    vesuvius.preview https://.../volume.zarr -o slice.png                 # middle z slice
    vesuvius.preview vol.zarr --axis y --at 0.3 -o y.png                  # a y slice 30% in
    vesuvius.preview vol.zarr --views zyx -o three.png                    # z, y and x side by side
    vesuvius.preview vol.zarr --crop 5000:5128,3900:4156,4000:4256 --sweep 32 -o sweep.gif
    vesuvius.preview vol.zarr --crop ... --mip -o mip.png                 # max over the crop's depth
    vesuvius.preview vol.zarr --crop ... --orbit 36 -o orbit.gif          # projection turning about z
    vesuvius.preview vol.zarr --overlay mask.zarr -o overlay.png         # mask drawn over the volume
    vesuvius.preview a.zarr b.zarr --crop ... --sweep 16 -o compare.gif  # several inputs side by side

An input is a local path or URL of an OME-Zarr group or array, or ``scroll:ID`` / ``segment:ID``
for the packaged catalog. Coordinates (``--crop``, ``--at`` in voxels) are full-resolution voxels;
``--level`` picks the pyramid level that is read and drawn.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AXES = "zyx"
OVERLAY_RGB = (255, 64, 64)


@dataclass
class Source:
    """One input: its pyramid levels and a display name."""

    name: str
    levels: List  # array-like objects, level 0 first

    def level(self, n: int):
        if not 0 <= n < len(self.levels):
            raise SystemExit(f"{self.name} has levels 0 to {len(self.levels) - 1}, not {n}")
        return self.levels[n]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.levels[0].shape)


def _storage_options(path: str):
    return {"anon": True} if path.startswith("s3://") else None


def resolve(spec: str) -> str:
    """A catalog entry (``scroll:1``, ``segment:ID``) becomes its URL; anything else is kept."""
    kind, _, ident = spec.partition(":")
    if kind not in ("scroll", "segment") or not ident:
        return spec
    from vesuvius import Volume

    if kind == "scroll":
        return Volume(type="scroll", scroll_id=ident).url
    return Volume(type="segment", segment_id=int(ident)).url


def open_source(spec: str) -> Source:
    """Open an OME-Zarr group, a zarr array, or a catalog entry (``scroll:1``, ``segment:ID``)."""
    from vesuvius.data.utils import open_zarr

    spec = resolve(spec)
    store = open_zarr(spec, mode="r", storage_options=_storage_options(spec))
    if hasattr(store, "shape"):
        return Source(name=_short_name(spec), levels=[store])
    levels = []
    multiscales = store.attrs.get("multiscales") if hasattr(store, "attrs") else None
    paths = [d["path"] for d in multiscales[0]["datasets"]] if multiscales else None
    if not paths:
        paths = sorted((k for k in store.array_keys() if k.isdigit()), key=int)
    for p in paths:
        levels.append(store[p])
    if not levels:
        raise SystemExit(f"{spec} holds no arrays to draw")
    return Source(name=_short_name(spec), levels=levels)


def _short_name(spec: str) -> str:
    return Path(spec.rstrip("/")).name or spec


def parse_crop(text: Optional[str], shape: Sequence[int]) -> List[Tuple[int, int]]:
    """``Z0:Z1,Y0:Y1,X0:X1`` in full-resolution voxels; empty parts mean the whole axis."""
    spatial = shape[-3:]
    if not text:
        return [(0, int(n)) for n in spatial]
    parts = text.split(",")
    if len(parts) != 3:
        raise SystemExit(f"--crop needs three ranges Z0:Z1,Y0:Y1,X0:X1, got {text!r}")
    out = []
    for part, n in zip(parts, spatial):
        lo, _, hi = part.partition(":")
        lo = int(lo) if lo else 0
        hi = int(hi) if hi else int(n)
        if not 0 <= lo < hi <= n:
            raise SystemExit(f"--crop range {part!r} is outside 0:{n}")
        out.append((lo, hi))
    return out


def pick_level(source: Source, crop, max_size: int, axis: Optional[int] = None) -> int:
    """The finest level at which the drawn sides of the crop are at most max_size pixels.

    With an axis, only the two sides in the drawn plane count.
    """
    side = max(hi - lo for i, (lo, hi) in enumerate(crop) if i != axis)
    for n in range(len(source.levels)):
        factor = source.shape[-1] / source.levels[n].shape[-1]
        if side / factor <= max_size:
            return n
    return len(source.levels) - 1


def _scaled(crop, source: Source, level: int):
    full = source.shape[-3:]
    here = source.level(level).shape[-3:]
    out = []
    for (lo, hi), f, h in zip(crop, full, here):
        s = f / h
        a = int(lo / s)
        b = max(a + 1, int(math.ceil(hi / s)))
        out.append((a, min(b, h)))
    return out, [f / h for f, h in zip(full, here)]


def read_block(arr, box) -> np.ndarray:
    index = tuple(slice(lo, hi) for lo, hi in box)
    if arr.ndim > 3:
        index = (0,) * (arr.ndim - 3) + index
    return np.asarray(arr[index])


def contrast(images: Sequence[np.ndarray], value_range=None) -> Tuple[float, float]:
    if value_range:
        return float(value_range[0]), float(value_range[1])
    sample = np.concatenate([np.asarray(im, dtype=np.float32).ravel()[::7] for im in images])
    sample = sample[np.isfinite(sample)]
    nonzero = sample[sample != 0]
    if nonzero.size > sample.size // 10:
        sample = nonzero  # masked volumes: leave the zero background out of the stretch
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, [0.5, 99.5])
    return float(lo), float(hi if hi > lo else lo + 1)


def to_u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (np.asarray(img, dtype=np.float32) - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def blend(gray: np.ndarray, mask: Optional[np.ndarray], alpha: float) -> Image.Image:
    if mask is None:
        return Image.fromarray(gray)  # stays 8-bit gray, so GIFs stay small
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    on = mask > 0
    rgb[on] = rgb[on] * (1 - alpha) + np.array(OVERLAY_RGB, dtype=np.float32) * alpha
    return Image.fromarray(rgb.astype(np.uint8))


def orbit_projection(block: np.ndarray, angle: float, how: str = "max") -> np.ndarray:
    """Projection (max or mean) of a (z, y, x) block seen from an angle about the z axis.

    Rays are sampled every half voxel, across and along, so a feature one voxel wide is
    never stepped over at an oblique angle; pairs of rays are then pooled to one output
    column per voxel.
    """
    nz, ny, nx = block.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    n = 2 * int(math.ceil(math.hypot(ny, nx)))
    coords = (np.arange(n) - (n - 1) / 2) * 0.5
    uu, tt = np.meshgrid(coords, coords, indexing="ij")
    xs = np.rint(cx + uu * math.cos(angle) - tt * math.sin(angle)).astype(np.int64)
    ys = np.rint(cy + uu * math.sin(angle) + tt * math.cos(angle)).astype(np.int64)
    inside = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    fine = np.zeros((nz, n), dtype=np.float32)
    for i in range(n):
        keep = inside[i]
        if keep.any():
            ray = block[:, ys[i, keep], xs[i, keep]]
            # mean over the full ray length, so the edges of the block fade like a radiograph
            fine[:, i] = ray.max(axis=1) if how == "max" else ray.sum(axis=1, dtype=np.float64) / n
    pairs = fine.reshape(nz, n // 2, 2)
    return pairs.max(axis=2) if how == "max" else pairs.mean(axis=2)


def _label(img: Image.Image, text: str) -> Image.Image:
    if not text:
        return img
    bar = 22
    room = max(4, (img.width - 12) // 6)
    if len(text) > room:
        text = "..." + text[-(room - 3):]
    canvas = Image.new(img.mode, (img.width, img.height + bar), 16 if img.mode == "L" else (16, 16, 16))
    canvas.paste(img, (0, bar))
    ImageDraw.Draw(canvas).text((6, 4), text, fill=235 if img.mode == "L" else (235, 235, 235),
                                font=ImageFont.load_default())
    return canvas


def side_by_side(images: Sequence[Image.Image], gap: int = 6) -> Image.Image:
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    mode = "L" if all(im.mode == "L" for im in images) else "RGB"
    canvas = Image.new(mode, (width, height), 16 if mode == "L" else (16, 16, 16))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _resize_to(img: Image.Image, scale: Sequence[float], max_size: int) -> Image.Image:
    """Undo anisotropic pyramid steps and keep the result under max_size."""
    w = img.width * scale[1]
    h = img.height * scale[0]
    f = min(1.0, max_size / max(w, h)) if max(w, h) > max_size else 1.0
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    return img if size == img.size else img.resize(size, Image.NEAREST)


def build_frames(args, sources: List[Source], masks: List[Optional[Source]]):
    """Return a list of frames (one for a PNG), each a list of (panel image, label) per input."""
    frames: List[List[Tuple[np.ndarray, Optional[np.ndarray], str, Tuple[float, float]]]] = []
    plans = []  # one per (input, view): what to read and how to draw it
    for src, msk in zip(sources, masks):
        crop = parse_crop(args.crop, src.shape)
        for axis in args.views:
            a = AXES.index(axis)
            if args.level is not None:
                level = args.level
            else:
                level = pick_level(src, crop, args.max_size, None if args.orbit else a)
            box, factors = _scaled(crop, src, level)
            plans.append((src, msk, crop, level, box, factors, axis, a))

    n_frames = args.sweep or args.orbit or 1
    blocks = {}
    for f in range(n_frames):
        panels = []
        for k, (src, msk, crop, level, box, factors, axis, a) in enumerate(plans):
            arr = src.level(level)
            marr = msk.level(level) if msk is not None else None
            tag = f"{src.name}  level {level}"
            if args.orbit or args.mip:
                if k not in blocks:  # read the block once, not once per frame
                    blocks[k] = (read_block(arr, box),
                                 read_block(marr, box) if marr is not None else None)
                block, mblock = blocks[k]
            if args.orbit:
                angle = 2 * math.pi * f / args.orbit
                img = orbit_projection(block, angle, args.project)
                mimg = orbit_projection(mblock, angle) if mblock is not None else None
                scale = (factors[0], factors[2])
                label = f"{tag}  orbit {360 * f // args.orbit} deg"
            elif args.mip:
                img = block.max(axis=a) if args.project == "max" else block.mean(axis=a)
                mimg = mblock.max(axis=a) if mblock is not None else None
                scale = tuple(fac for i, fac in enumerate(factors) if i != a)
                label = f"{tag}  {args.project} over {axis} {crop[a][0]}:{crop[a][1]}"
            else:
                lo, hi = box[a]
                if args.sweep:
                    idx = lo + (hi - lo - 1) * f // max(1, args.sweep - 1)
                else:
                    idx = _index_for(args.at, crop[a], factors[a], arr.shape[-3:][a])
                one = list(box)
                one[a] = (idx, idx + 1)
                img = np.squeeze(read_block(arr, one), axis=a)
                mimg = np.squeeze(read_block(marr, one), axis=a) if marr is not None else None
                scale = tuple(fac for i, fac in enumerate(factors) if i != a)
                label = f"{tag}  {axis}={int(idx * factors[a])}"
            panels.append((img, mimg, label, scale))
        frames.append(panels)
    return frames


def _index_for(at: Optional[str], rng, factor: float, n: int) -> int:
    lo, hi = rng
    if at is None:
        full = (lo + hi - 1) / 2
    elif "." in at:
        full = lo + float(at) * (hi - lo - 1)
    else:
        full = int(at)
        if not lo <= full < hi:
            raise SystemExit(f"--at {at} is outside {lo}:{hi}")
    return min(n - 1, int(full / factor))


def make_images(args, sources: List[Source], masks: Optional[List[Optional[Source]]] = None) -> List[Image.Image]:
    """One image per frame: every input and view side by side."""
    masks = masks or [None] * len(sources)
    frames = build_frames(args, sources, masks)
    lo_hi = contrast([p[0] for p in frames[len(frames) // 2]], args.range)
    images = []
    for panels in frames:
        tiles = []
        for img, mimg, label, scale in panels:
            tile = blend(to_u8(img, *lo_hi), mimg, args.alpha)
            tile = _resize_to(tile, scale, args.max_size)
            tiles.append(_label(tile, "" if args.no_labels else label))
        images.append(side_by_side(tiles))
    return images


def options(**overrides) -> argparse.Namespace:
    """The command line defaults, for calling the preview from Python."""
    args = build_parser().parse_args(["-", "-o", "-"])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown preview option {key!r}")
        setattr(args, key, value)
    return _normalize(args)


def _normalize(args: argparse.Namespace) -> argparse.Namespace:
    args.views = "z" if args.orbit else (args.views or args.axis)
    if any(v not in AXES for v in args.views):
        raise SystemExit(f"--views takes letters from zyx, got {args.views!r}")
    return args


def render(args) -> Path:
    sources = [open_source(s) for s in args.inputs]
    masks: List[Optional[Source]] = [None] * len(sources)
    if args.overlay:
        overlay = open_source(args.overlay)
        masks = [overlay if s.shape[-3:] == overlay.shape[-3:] else None for s in sources]
        if not any(m is not None for m in masks):
            raise SystemExit(f"--overlay {args.overlay} has shape {overlay.shape}, unlike every input")
    images = make_images(args, sources, masks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        if len(images) == 1:
            images[0].save(out)
        else:
            images[0].save(out, save_all=True, append_images=images[1:],
                           duration=int(1000 / args.fps), loop=0, optimize=True)
    else:
        if len(images) > 1:
            raise SystemExit("--sweep and --orbit make several frames; write them to a .gif")
        images[0].save(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vesuvius.preview",
        description="Draw a slice, sweep, projection or orbit of a volume to a PNG or GIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    p.add_argument("inputs", nargs="+", help="OME-Zarr or zarr path/URL, or scroll:ID / segment:ID")
    p.add_argument("-o", "--output", required=True, help="output .png or .gif")
    p.add_argument("--axis", default="z", choices=list(AXES), help="slice axis (default z)")
    p.add_argument("--views", help="several axes side by side, such as zyx (overrides --axis)")
    p.add_argument("--at", help="slice position: a voxel index, or a fraction such as 0.3 (default: middle)")
    p.add_argument("--crop", help="Z0:Z1,Y0:Y1,X0:X1 in full-resolution voxels")
    p.add_argument("--level", type=int, help="pyramid level to read (default: finest that fits --max-size)")
    p.add_argument("--max-size", type=int, default=1024, help="largest side of each panel, in pixels")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sweep", type=int, metavar="N", help="GIF of N slices across the crop")
    mode.add_argument("--mip", action="store_true", help="project the crop along the axis (see --project)")
    mode.add_argument("--orbit", type=int, metavar="N", help="GIF of N projections turning about z")
    p.add_argument("--project", choices=("max", "mean"), default="max",
                   help="projection for --mip and --orbit: max, or mean for an X-ray-like view")
    p.add_argument("--overlay", help="mask or prediction of the same shape, drawn in red where > 0")
    p.add_argument("--alpha", type=float, default=0.45, help="overlay opacity (default 0.45)")
    p.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"), help="fixed contrast range")
    p.add_argument("--fps", type=float, default=8, help="GIF frames per second (default 8)")
    p.add_argument("--no-labels", action="store_true", help="leave out the caption bar")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _normalize(build_parser().parse_args(argv))
    out = render(args)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
