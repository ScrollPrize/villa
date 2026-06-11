#!/usr/bin/env python
"""Generate presentation (render) textures: small interior alpha holes infilled with
locally-matched papyrus (compositing dropout, not physical lacunae);
large genuine lacunae keep their cutout. Applies identically to the plain composite and
the black-ink overlay so the reveal endpoints stay aligned.

Outputs: outputs/render_textures/<stem>_plain.png and <stem>_ink.png (+ preview pairs in
reports/infill_previews/ for the requested meshes). Originals are never modified.

Usage: uv run python scripts/make_render_textures.py [--only stem ...] [--previews stem ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from scrollkit.ink.infill import build_render_texture  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def wrap_rows() -> list[dict]:
    audit = json.loads((ROOT / "reports/audit.json").read_text())
    rows = []
    for m in audit["meshes"]:
        if m["group"] not in ("A", "B"):
            continue
        tex = next((t for t in m["textures"] if t.get("exists")), None)
        src_dir = (ROOT / m["path"]).parent
        rows.append({
            "stem": m["stem"], "group": m["group"],
            "plain": src_dir / tex["name"],
            "ink": ROOT / "outputs/overlays" / ("march" if m["group"] == "A" else "may")
                   / f"{m['stem']}_inkoverlay.png",
        })
    return rows


def process(path: Path, out: Path) -> dict | None:
    if not path.exists():
        return None
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    filled, info = build_render_texture(rgba)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(filled).save(out, compress_level=6)
    return {k: v for k, v in info.items() if k != "fill_mask"}


def preview(before: Path, after: Path, dst: Path) -> None:
    a = Image.open(before).convert("RGBA")
    b = Image.open(after).convert("RGBA")
    w = 900
    bg = (16, 17, 20, 255)
    tiles = []
    for im in (a, b):
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        canvas = Image.new("RGBA", im.size, bg)
        canvas.alpha_composite(im)
        tiles.append(canvas.convert("RGB"))
    sheet = Image.new("RGB", (2 * w + 8, tiles[0].height), (40, 40, 40))
    sheet.paste(tiles[0], (0, 0))
    sheet.paste(tiles[1], (w + 8, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88)


def one_mesh(r: dict, previews: list[str]) -> tuple[str, dict]:
    out_dir = ROOT / "outputs/render_textures"
    rec = {}
    for kind in ("plain", "ink"):
        dst = out_dir / f"{r['stem']}_{kind}.png"
        info = process(Path(r[kind]), dst)
        rec[kind] = info
        if r["stem"] in previews and kind == "plain" and info:
            preview(Path(r[kind]), dst, ROOT / "reports/infill_previews" / f"{r['stem']}_plain.jpg")
    return r["stem"], rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--previews", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = wrap_rows()
    if args.only:
        rows = [r for r in rows if r["stem"] in set(args.only)]
    out_dir = ROOT / "outputs/render_textures"
    stats = {}
    t0 = time.time()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_mesh, {**r, "plain": str(r["plain"]), "ink": str(r["ink"])},
                          list(args.previews)): r["stem"] for r in rows}
        for fut in as_completed(futs):
            stem, rec = fut.result()
            stats[stem] = rec
            p = rec.get("plain") or {}
            print(f"[{stem}] filled {p.get('n_filled_components')} comps "
                  f"({(p.get('filled_frac_of_papyrus') or 0) * 100:.1f}%), "
                  f"kept {p.get('n_kept_lacunae')} lacunae", flush=True)
    if (out_dir / "infill_stats.json").exists():
        old = json.loads((out_dir / "infill_stats.json").read_text())
        old.update(stats)
        stats = old
    (out_dir / "infill_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"done {len(rows)} meshes in {time.time() - t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
