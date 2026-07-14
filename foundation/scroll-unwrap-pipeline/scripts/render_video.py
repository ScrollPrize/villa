#!/usr/bin/env python
"""Render one M4 cinematic video (one mesh x variant x framing).

Usage:
  uv run python scripts/render_video.py <anim_dir> <group> --variant {plain,ink,reveal}
      --framing {h,v} [--res {4k,1080p}] [--keep-rawcache]

  --res 4k (default): 3840x2160 / 2160x3840 master piped raw-RGB24 into ffmpeg,
      then a 1080p derivative DOWNSCALED from the master (lanczos). Production.
  --res 1080p: native 1080p render under outputs/anim/<stem>/smoke/ — pipeline
      smoke only, never a deliverable (deliverable 1080p files are derivatives).

One persistent SceneRenderer per video (frames morph via update_points — actors
are never rebuilt). GPU work is serial: the chirality probe (camera plan) closes
before the main renderer opens. While piping the PLAIN variant the raw frames
are also written to a rawcache so 'reveal' streams with zero GPU renders; the
reveal run deletes the rawcache when done (--keep-rawcache to override).

Camera plans are cached per framing (outputs/anim/<stem>/camera_plan_<framing>.json)
so plain/ink/reveal share bit-identical cameras (the reveal crossfade is an exact
image-space blend of static renders).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from scrollkit.render.scene import SceneRenderer  # noqa: E402  (pins EGL env first)
from scrollkit.render import cinematics  # noqa: E402
from scrollkit.render.encode import FFmpegWriter, RawCache, ffprobe_check, make_derivative  # noqa: E402

MIN_FREE_GB = 80
MORPH_KEY_TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def find_texture(stem: str, group: str) -> Path:
    # presentation texture (small dropout holes infilled) preferred for RENDERING;
    # falls back to the raw composite. Metrics/IO always use originals.
    rt = ROOT / "outputs/render_textures" / f"{stem}_plain.png"
    if rt.is_file():
        return rt
    base = ROOT / ("textured_plys/textured_ply_march" if group == "A" else "textured_plys/textured_ply_may")
    cands = sorted(base.glob(f"{stem}*.png"))
    if not cands:
        raise FileNotFoundError(f"composite texture for {stem} in {base}")
    return cands[0]


def find_ink_overlay(stem: str, group: str) -> Path:
    rt = ROOT / "outputs/render_textures" / f"{stem}_ink.png"
    if rt.is_file():
        return rt
    p = ROOT / "outputs/overlays" / ("march" if group == "A" else "may") / f"{stem}_inkoverlay.png"
    if not p.is_file():
        raise FileNotFoundError(f"ink overlay missing: {p}")
    return p


def texture_stats(texture_path) -> dict:
    """Midtone/p95 of the plain composite within its alpha mask (0-1), for the
    rig's exposure adaptation. Always computed from the PLAIN composite so the
    plain and ink variants are lit identically (exact reveal crossfade)."""
    with Image.open(texture_path) as im:
        im = im.convert("RGBA")
        im.thumbnail((1024, 1024))
        a = np.asarray(im, dtype=np.float32)
    L = a[..., :3].mean(-1) / 255.0
    fg = L[a[..., 3] >= 128]
    return {"p50": float(np.median(fg)), "p95": float(np.percentile(fg, 95))}


def check_disk(where: str) -> None:
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"disk guard ({where}): only {free_gb:.1f} GB free (< {MIN_FREE_GB} GB)")


def save_keyframe(img: np.ndarray, full_path: Path, report_path: Path | None) -> None:
    full_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(img)
    im.save(full_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        small = im.copy()
        small.thumbnail((1280, 1280), Image.LANCZOS)
        small.save(report_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("anim_dir")
    ap.add_argument("group", choices=["A", "B"])
    ap.add_argument("--variant", required=True, choices=["plain", "ink", "reveal"])
    ap.add_argument("--framing", required=True, choices=["h", "v"])
    ap.add_argument("--res", default="4k", choices=["4k", "1080p"])
    ap.add_argument("--keep-rawcache", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    check_disk("start")
    cfg = yaml.safe_load((ROOT / "configs/global.yaml").read_text())
    rcfg, ecfg, vcfg = cfg["render"], cfg["encode"], cfg["variants"]
    tex_orientation = json.loads((ROOT / "reports/audit.json").read_text())["tex_orientation"][args.group]

    adir = Path(args.anim_dir).resolve()
    stem = adir.name
    data = np.load(adir / "frames.npz")
    frames, faces, uv = data["frames"], data["faces"], data["uv"]
    F = frames.shape[0]

    smoke = args.res == "1080p"
    fkey = "horizontal" if args.framing == "h" else "vertical"
    fdim = rcfg["framings"][fkey]
    if smoke:
        W, H = (1920, 1080) if args.framing == "h" else (1080, 1920)
        crf, level = ecfg["crf_1080"], "4.2"
        out_dir = adir / "smoke"
        master = out_dir / f"{stem}_{args.variant}_{args.framing}_1080p_native.mp4"
        kf_dir = out_dir / f"keyframes_{args.variant}_{args.framing}"
        report_kf_dir = None
        cache_path = out_dir / f"rawcache_{args.framing}_smoke.raw"
        inkflat_path = out_dir / f"inkflat_{args.framing}_smoke.npy"
    else:
        W, H = int(fdim["width"]), int(fdim["height"])
        crf, level = ecfg["crf_4k"], "5.1"
        out_dir = adir / "video"
        master = out_dir / f"{stem}_{args.variant}_{args.framing}_4k.mp4"
        kf_dir = adir / "keyframes_m4" / f"{args.variant}_{args.framing}"
        report_kf_dir = ROOT / "reports/m4_keyframes" / f"{stem}_{args.variant}_{args.framing}"
        cache_path = adir / f"rawcache_{args.framing}.raw"
        inkflat_path = adir / f"inkflat_{args.framing}.npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    aspect = W / H

    tex_plain = find_texture(stem, args.group)

    # ---- camera plan (cached per framing; KEYED to the trajectory so a re-animated
    # mesh can never reuse a camera computed for different geometry) ------------------
    fz = (adir / "frames.npz").stat()
    frames_sig = f"{fz.st_mtime_ns}:{fz.st_size}"
    plan_path = adir / f"camera_plan_{args.framing}.json"
    up_path = adir / "up_choice.json"
    plan = None
    if plan_path.is_file():
        cand = json.loads(plan_path.read_text())
        if cand.get("frames_sig") == frames_sig:
            plan = cand
        else:
            plan_path.unlink()
            if up_path.is_file() and json.loads(up_path.read_text()).get("frames_sig") != frames_sig:
                up_path.unlink()
    if plan is None:
        up_choice = None
        if up_path.is_file():
            uc = json.loads(up_path.read_text())
            up_choice = uc if uc.get("frames_sig") == frames_sig else None
        plan = cinematics.plan_camera(
            frames, faces, uv, aspect, args.group,
            texture_path=tex_plain, tex_orientation=tex_orientation, up_choice=up_choice,
            margin_frac=float(rcfg["margin_frac"]), push_in_frac=float(rcfg["push_in_frac"]),
        )
        plan["frames_sig"] = frames_sig
        plan["up_choice"]["frames_sig"] = frames_sig
        plan_path.write_text(json.dumps(plan, indent=1))
        if not up_path.is_file():
            up_path.write_text(json.dumps(plan["up_choice"], indent=1))
        print(f"[plan] {args.framing}: up_choice={plan['up_choice']['best']} "
              f"scores={ {k: round(v, 4) for k, v in plan['up_choice']['scores'].items()} } "
              f"min_margin={plan['min_margin']:.4f}")
    if abs(plan["aspect"] - aspect) > 1e-9 or len(plan["per_frame_scale"]) != F:
        raise RuntimeError(f"cached plan {plan_path} does not match this run (aspect/frames)")
    scales = np.asarray(plan["per_frame_scale"], dtype=np.float64)

    kf_idx = [int(round(t * (F - 1))) for t in MORPH_KEY_TS]
    mask = cinematics.vignette_mask(W, H)
    hold_in, fade, hold_out = (int(vcfg["reveal_hold_in"]), int(vcfg["reveal_fade"]),
                               int(vcfg["reveal_hold_out"]))
    zoom_hold = int(vcfg.get("reveal_zoom_hold", 0))
    zoom_factor = float(vcfg.get("reveal_zoom_factor", 1.05))
    total_frames = (F if args.variant != "reveal"
                    else F + hold_in + fade + hold_out + zoom_hold)
    wm = cinematics.watermark_overlay(W, H)

    run_info = {
        "stem": stem, "group": args.group, "variant": args.variant, "framing": args.framing,
        "res": args.res, "size": [W, H], "frames": total_frames, "crf": crf, "level": level,
        "camera_plan": str(plan_path.relative_to(ROOT)),
        "up_choice": plan["up_choice"]["best"], "min_margin": plan["min_margin"],
    }

    # ---- render or stream ------------------------------------------------------------
    t_render0 = time.time()
    if args.variant in ("plain", "ink"):
        if args.variant == "plain":
            tex_front, tex_back = tex_plain, None
        else:
            # physical ink: the overlay goes on the INNER face only; the outer face of the
            # winding stays plain papyrus. SceneRenderer's texture_path binds to the
            # +winding-normal side, so map via the geometric inner side.
            ink_tex = find_ink_overlay(stem, args.group)
            inner_sign = (plan.get("up_choice") or {}).get("inner_side", {}).get("inner_sign")
            if inner_sign is None:
                inner_sign = cinematics.inner_side_sign(frames[0], faces, uv)["inner_sign"]
            if inner_sign > 0:
                tex_front, tex_back = ink_tex, tex_plain
            else:
                tex_front, tex_back = tex_plain, ink_tex
        run_info["texture"] = str(tex_front)
        run_info["texture_back"] = str(tex_back) if tex_back else None
        sr = SceneRenderer(frames[0], faces, uv, tex_front, tex_orientation,
                           size=(W, H), background=cinematics.BACKGROUND_HEX, lighting="studio",
                           back_texture_path=tex_back)
        try:
            sr.set_camera({k: plan[k] for k in ("position", "focal_point", "up", "parallel",
                                                "parallel_scale")})
            rig = cinematics.studio_lights(
                sr, {"camera": plan, "bounds_lo": plan["bounds_lo"], "bounds_hi": plan["bounds_hi"],
                     "texture_stats": texture_stats(tex_plain)})
            run_info["rig"] = rig
            cache = RawCache(cache_path, W, H) if args.variant == "plain" else None
            with FFmpegWriter(master, W, H, crf=crf, level=level, preset=ecfg["preset"]) as wr:
                for f in range(F):
                    sr.update_points(frames[f])
                    sr.plotter.camera.parallel_scale = float(scales[f])
                    img = sr.screenshot()
                    if img.shape != (H, W, 3):
                        raise RuntimeError(f"render size {img.shape} != ({H},{W},3)")
                    img = cinematics.apply_vignette(img, mask)
                    img = cinematics.apply_watermark(img, wm)
                    wr.write(img)
                    if cache is not None:
                        cache.append(img)
                    if args.variant == "ink" and f == F - 1:
                        np.save(inkflat_path, img)
                    if f in kf_idx:
                        t_lbl = MORPH_KEY_TS[kf_idx.index(f)]
                        name = f"kf_t{t_lbl:.2f}.png"
                        save_keyframe(img, kf_dir / name,
                                      report_kf_dir / name if report_kf_dir else None)
            if cache is not None:
                cache.finalize({"variant": "plain", "framing": args.framing,
                                "camera_plan": plan_path.name})
        finally:
            sr.close()
    else:  # reveal: stream the plain rawcache + image-space crossfade tail (no GPU)
        if not cache_path.is_file():
            raise RuntimeError(f"reveal needs the plain rawcache — run --variant plain first ({cache_path})")
        cache = RawCache.open(cache_path)
        if (cache.width, cache.height, cache.n_frames) != (W, H, F):
            raise RuntimeError(f"rawcache {cache_path} is {cache.width}x{cache.height}x{cache.n_frames}, "
                               f"expected {W}x{H}x{F}")
        # (the tight ink still is rendered fresh below — no inkflat dependency)
        # Reveal tail with a REAL camera push-in onto the text (design rule: emphasize
        # the inner-face reveal). Geometry is frozen flat, so the move is rendered fresh
        # per frame at full sharpness; the crossfade then blends two stills at the TIGHT
        # framing (still exact — camera frozen during the fade).
        zoom_n = max(0, hold_in + fade + hold_out - 10 - 45 - 10)  # budget-neutral split
        hold_in2, fade2, hold_out2 = 10, 45, 10
        if zoom_n == 0:  # fall back to the configured split if budget too small
            hold_in2, fade2, hold_out2, zoom_n = hold_in, fade, hold_out, 0

        flatV = frames[-1]
        tight = cinematics.tight_flat_camera(flatV, faces, uv, plan, margin_frac=0.10)
        reveal_mid_global = F + zoom_n + hold_in2 + fade2 // 2
        with FFmpegWriter(master, W, H, crf=crf, level=level, preset=ecfg["preset"]) as wr:
            for f, img in enumerate(cache.iter_frames()):
                wr.write(img)
                if f in kf_idx:
                    t_lbl = MORPH_KEY_TS[kf_idx.index(f)]
                    name = f"kf_t{t_lbl:.2f}.png"
                    save_keyframe(img, kf_dir / name,
                                  report_kf_dir / name if report_kf_dir else None)
            # camera move + stills on static geometry (two-sided: plain outer face)
            ink_tex2 = find_ink_overlay(stem, args.group)
            inner_sign2 = (plan.get("up_choice") or {}).get("inner_side", {}).get("inner_sign")
            if inner_sign2 is None:
                inner_sign2 = cinematics.inner_side_sign(frames[0], faces, uv)["inner_sign"]
            ft2, bt2 = (ink_tex2, tex_plain) if inner_sign2 > 0 else (tex_plain, ink_tex2)
            sr2 = SceneRenderer(flatV, faces, uv, tex_plain, tex_orientation, size=(W, H),
                                background=cinematics.BACKGROUND_HEX, lighting="studio")
            sr_ink = None
            try:
                cinematics.studio_lights(
                    sr2, {"camera": plan, "bounds_lo": plan["bounds_lo"],
                          "bounds_hi": plan["bounds_hi"], "texture_stats": texture_stats(tex_plain)})
                base_cam = {k: plan[k] for k in ("position", "focal_point", "up", "parallel")}
                base_cam["parallel_scale"] = float(scales[-1])
                for i in range(zoom_n):  # quintic-eased push-in, plain texture
                    s = (i + 1) / zoom_n
                    e = s * s * s * (s * (6 * s - 15) + 10)
                    cam = cinematics.lerp_camera(base_cam, tight, e)
                    sr2.set_camera(cam)
                    img = cinematics.apply_vignette(sr2.screenshot(), mask)
                    wr.write(cinematics.apply_watermark(img, wm))
                sr2.set_camera(tight)
                plain_tight = cinematics.apply_vignette(sr2.screenshot(), mask)
                sr_ink = SceneRenderer(flatV, faces, uv, ft2, tex_orientation, size=(W, H),
                                       background=cinematics.BACKGROUND_HEX, lighting="studio",
                                       back_texture_path=bt2)
                cinematics.studio_lights(
                    sr_ink, {"camera": plan, "bounds_lo": plan["bounds_lo"],
                             "bounds_hi": plan["bounds_hi"], "texture_stats": texture_stats(tex_plain)})
                sr_ink.set_camera(tight)
                ink_tight = cinematics.apply_vignette(sr_ink.screenshot(), mask)
            finally:
                sr2.close()
                if sr_ink is not None:
                    sr_ink.close()
            pf = plain_tight.astype(np.float32)
            inf_ = ink_tight.astype(np.float32)
            for i in range(hold_in2):
                wr.write(cinematics.apply_watermark(plain_tight, wm))
            for i in range(fade2):  # cosine ease, exact image-space blend (camera frozen)
                a = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (fade2 + 1)))
                img = np.clip(np.rint(pf * (1.0 - a) + inf_ * a), 0, 255).astype(np.uint8)
                img = cinematics.apply_watermark(img, wm)
                wr.write(img)
                if F + zoom_n + hold_in2 + i == reveal_mid_global:
                    save_keyframe(img, kf_dir / "kf_reveal_mid.png",
                                  report_kf_dir / "kf_reveal_mid.png" if report_kf_dir else None)
            for i in range(hold_out2):
                wr.write(cinematics.apply_watermark(ink_tight, wm))
            # post-reveal dwell: the ink stays on screen with a slow eased push-in
            # (image-space zoom of the tight ink still; the watermark is composited
            # AFTER the zoom so it stays pinned to the frame corner).
            from PIL import Image as _Image
            ink_im = _Image.fromarray(ink_tight)
            for i in range(zoom_hold):
                s = (i + 1) / max(zoom_hold, 1)
                e = s * s * s * (s * (6 * s - 15) + 10)
                z = 1.0 + (zoom_factor - 1.0) * e
                cw, ch = int(round(W / z)), int(round(H / z))
                x0, y0 = (W - cw) // 2, (H - ch) // 2
                img = np.asarray(ink_im.crop((x0, y0, x0 + cw, y0 + ch))
                                 .resize((W, H), _Image.LANCZOS))
                img = cinematics.apply_watermark(img, wm)
                wr.write(img)
                if i == zoom_hold - 1:
                    save_keyframe(img, kf_dir / "kf_reveal_end.png",
                                  report_kf_dir / "kf_reveal_end.png" if report_kf_dir else None)
    run_info["render_encode_s"] = round(time.time() - t_render0, 1)

    # ---- self-checks + derivative ------------------------------------------------------
    st = ffprobe_check(master, width=W, height=H, frames=total_frames)
    run_info["master"] = {"path": str(master.relative_to(ROOT)), "size_mb": round(st["file_size"] / 2**20, 1)}
    print(f"[ok] master {master.name}: {st['width']}x{st['height']} {st['nb_frames']}f "
          f"{st['pix_fmt']} faststart={st['faststart']} {run_info['master']['size_mb']} MB")

    if not smoke:
        dw, dh = (1920, 1080) if args.framing == "h" else (1080, 1920)
        deriv = out_dir / f"{stem}_{args.variant}_{args.framing}_1080p.mp4"
        t0 = time.time()
        make_derivative(master, deriv, dw, dh, crf=ecfg["crf_1080"], level="4.2",
                        preset=ecfg["preset"])
        std = ffprobe_check(deriv, width=dw, height=dh, frames=total_frames)
        run_info["derivative"] = {"path": str(deriv.relative_to(ROOT)),
                                  "size_mb": round(std["file_size"] / 2**20, 1),
                                  "encode_s": round(time.time() - t0, 1)}
        print(f"[ok] derivative {deriv.name}: {std['width']}x{std['height']} {std['nb_frames']}f "
              f"faststart={std['faststart']} {run_info['derivative']['size_mb']} MB")

    # ---- rawcache cleanup (reveal is the last consumer) -------------------------------
    if args.variant == "reveal" and not args.keep_rawcache:
        RawCache.open(cache_path).delete()
        inkflat_path.unlink(missing_ok=True)
        run_info["rawcache_deleted"] = True
        print(f"[clean] deleted {cache_path.name} + {inkflat_path.name}")

    run_info["total_s"] = round(time.time() - t_start, 1)
    run_path = master.with_suffix(".run.json")
    run_path.write_text(json.dumps(run_info, indent=1))
    print(f"[done] {stem} {args.variant}/{args.framing}/{args.res} in {run_info['total_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
