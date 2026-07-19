#!/usr/bin/env python3
"""'Scanning' tutorial thumb — stylized but physical. Rev 6.

A very slender, heavily weathered charcoal scroll — lumpy deformed
cross-section, large flakes, small pits, cracks, ragged silhouette — stands
on a finely brushed reflective metal table in a near-black room. The scroll
spins slowly (half a turn per loop) while the camera orbits it in a wide
one-way arc: the table texture, light sheens, shadow, and x-ray source all
rotate together on screen, so scroll and tabletop stay fixed relative to
each other apart from the scroll's own spin.

Usage: scan_thumb.py            -> one prototype frame (scan-proto.png)
       scan_thumb.py loop       -> 74 frames in scanloop/
"""
import math
import sys
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SCRATCH = "/tmp/claude-1000/-home-djosey/cef33c32-39cb-4c5c-9dc3-337094ac1570/scratchpad"

# ---- palette ----------------------------------------------------------------
BG_TOP = (26, 27, 31)
FLOOR = (96, 100, 107)        # stainless steel, cool grey
RED = (198, 40, 34)
CHAR = (40, 38, 36)
RIM = (30, 40, 62)
SPEC_KEY = (205, 196, 182)
SPEC_RIM = (110, 140, 205)
XRAY = (170, 205, 255)
S = 600
HORIZON = int(S * 0.55)

rng = np.random.default_rng(7)

# ---- scroll geometry: slender, lumpy, heavily weathered -----------------------
VOX = 0.13                  # fine charcoal columns
RVOX = 0.30                 # chunky reconstruction voxels (red)
R_MAX = 3.4                  # grid bound (leave room for lumps)
HEIGHT = 23.0
PITCH = 0.58
ROUND = 2.0

# lumpy cross-section deformation (applies to the whole roll)
_ph = rng.uniform(0, 2 * math.pi, 3)
def r_mult(ang):
    return (1.0 + 0.10 * math.sin(2 * ang + _ph[0])
                + 0.06 * math.sin(3 * ang + _ph[1])
                + 0.035 * math.sin(5 * ang + _ph[2]))

def build_columns():
    r0, turns, thick = 0.5, 3.4, 0.17
    theta = np.linspace(0, turns * 2 * math.pi, 12000)
    r = r0 + PITCH * theta / (2 * math.pi)
    sx, sy = r * np.cos(theta), r * np.sin(theta)
    r_out = r[-1]
    n = int(2 * R_MAX / VOX)
    cols = {}
    for x, y, th in zip(sx, sy, theta):
        i, j = int((x + R_MAX) / VOX), int((y + R_MAX) / VOX)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if not (0 <= ii < n and 0 <= jj < n):
                    continue
                cx = (ii + 0.5) * VOX - R_MAX
                cy = (jj + 0.5) * VOX - R_MAX
                if math.hypot(cx - x, cy - y) < thick:
                    cols.setdefault((ii, jj), (cx, cy, int(th / (2 * math.pi))))
    max_wind = max(c[2] for c in cols.values())
    large = [(rng.uniform(0, 2 * math.pi), rng.uniform(4, HEIGHT - 3),
              rng.uniform(0.30, 0.75), rng.uniform(1.2, 3.2)) for _ in range(10)]
    deep = [(rng.uniform(0, 2 * math.pi), rng.uniform(6, HEIGHT - 5),
             rng.uniform(0.25, 0.45), rng.uniform(1.0, 2.2)) for _ in range(3)]
    pits = [(rng.uniform(0, 2 * math.pi), rng.uniform(2.5, HEIGHT - 1.5),
             rng.uniform(0.08, 0.28), rng.uniform(0.25, 1.1)) for _ in range(44)]
    def cut(segs, ang, blobs):
        for (fa, fz, fra, frz) in blobs:
            d = min(abs(ang - fa), 2 * math.pi - abs(ang - fa))
            if d < fra:
                w = math.sqrt(1 - (d / fra) ** 2)
                lo = fz - frz * w - abs(rng.normal(0, 0.30))
                hi = fz + frz * w + abs(rng.normal(0, 0.30))
                new = []
                for z0, z1 in segs:
                    if hi <= z0 or lo >= z1:
                        new.append([z0, z1]); continue
                    if lo > z0: new.append([z0, lo])
                    if hi < z1: new.append([hi, z1])
                segs = new
        return segs
    out = {}
    for key, (cx, cy, wind) in cols.items():
        ang = math.atan2(cy, cx) % (2 * math.pi)
        rad = math.hypot(cx, cy)
        drop = ROUND * (rad / r_out) ** 2.4
        top = HEIGHT - drop - abs(rng.normal(0, 0.45))
        bot = drop * 0.22
        segs = [[bot, max(bot + 2.0, top)]]
        if wind >= max_wind:
            segs = cut(segs, ang, large)
            segs = cut(segs, ang, pits)
        if wind >= max_wind - 1:
            segs = cut(segs, ang, deep)
        segs = [s for s in segs if s[1] - s[0] > VOX]
        if not segs:
            continue
        # lumpy deformation + per-column bump
        m = r_mult(ang) * (1.0 + max(-0.06, min(0.06, rng.normal(0, 0.03))))
        out[key] = (cx * m, cy * m, ang, segs)
    return out, r_out

COLS, R_OUT = build_columns()
COL_NOISE = {k: rng.normal(0, 6.5) for k in COLS}
COL_TEX = {k: rng.normal(0, 0.075, 28) for k in COLS}          # vertical mottle
COL_W = {k: 1.0 + max(-0.25, min(0.38, rng.normal(0, 0.16))) for k in COLS}  # ragged widths
COL_ANG = {k: rng.normal(0, 0.22) for k in COLS}
COL_AJIT = {k: rng.normal(0, 0.34, 28) for k in COLS}  # per-chunk normal scatter       # per-column normal perturbation
COL_SPEC = {k: rng.uniform(0.12, 1.0) for k in COLS}   # matte<->glossy facet mix

# ---- camera (farther, more tele) ----------------------------------------------
CAM_DIST = 95.0
CENTER_Z = HEIGHT * 0.48
_elev = math.radians(26)
_scale = 20.0
_voff = 0.0

def project_pt(x, y, z, spin):
    c, s = math.cos(spin), math.sin(spin)
    xr = x * c - y * s
    yr = x * s + y * c
    zr = z - CENTER_Z
    ce, se = math.cos(_elev), math.sin(_elev)
    yc = yr * ce - zr * se
    zc = yr * se + zr * ce
    persp = CAM_DIST / (CAM_DIST + yc)
    return (S / 2 + xr * _scale * persp, S * 0.52 + _voff - zc * _scale * persp, yc, persp)

def clamp255(v):
    return 255 if v > 255 else (0 if v < 0 else int(v))

# ---- world-space floor texture (rotated per frame as the camera orbits) -------
def build_floor_tex():
    """Tabletop from the ORIGINAL top-scanning thumb: the left half of the
    scratch strip, illumination-flattened and stretched as ONE piece over the
    whole floor region (near-uniform ~4x upscale), then sharpened to deblur.
    No tiling, no mirroring: one continuous grain."""
    frame = Image.open(f"{SCRATCH}/scanorig-first.png").convert("RGB")
    strip = frame.crop((0, 236, 150, 298))          # left half of the strip
    a = np.asarray(strip).astype(np.float32)
    lum = np.asarray(strip.convert("L").filter(ImageFilter.GaussianBlur(45)),
                     dtype=np.float32)[..., None] + 1e-3
    flat = np.clip(a / lum * lum.mean(), 0, 255)

    fh = S - HORIZON
    img = Image.fromarray(flat.astype(np.uint8)).resize((S, fh), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    arr *= 88.0 / max(arr.mean(), 1e-6)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=180, threshold=1))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=1))
    return img

FLOOR_TEX = build_floor_tex()
_SCENE_CACHE = {}

KEY_AZ = math.radians(35)
RIM_AZ = math.radians(215)
FILL_AZ = math.radians(270)

def charcoal_color(lk, lr, jit, sk, sr):
    f = 0.36 + 0.55 * lk + jit
    return (clamp255(CHAR[0] * f + RIM[0] * lr + SPEC_KEY[0] * sk + SPEC_RIM[0] * sr),
            clamp255(CHAR[1] * f + RIM[1] * lr + SPEC_KEY[1] * sk + SPEC_RIM[1] * sr),
            clamp255(CHAR[2] * f + RIM[2] * lr + SPEC_KEY[2] * sk + SPEC_RIM[2] * sr))

def shade(base, f):
    return tuple(clamp255(ch * f) for ch in base)

def compose_scene(sway):
    """Empty space: near-black with a faint radial glow behind the scroll."""
    img = Image.new("RGB", (S, S), BG_TOP)
    glow = Image.new("L", (S, S), 0)
    dg = ImageDraw.Draw(glow)
    dg.ellipse([S * 0.10, S * 0.06, S * 0.90, S * 0.94], fill=46)
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    img = Image.composite(Image.new("RGB", (S, S), (48, 50, 57)), img, glow)
    return img

def render_frame(spin, scan_h, out, t=0.0):
    global _elev, _scale, _voff
    # camera: wide one-way orbit; scroll/table/lights all rotate together
    sway = 0.0   # static camera/table: only the scroll spins
    spin_s = spin + sway                    # scroll's screen rotation
    _elev = math.radians(26)
    _scale = 20.0 * (1 + 0.025 * t)
    _voff = 0.0
    if "img" not in _SCENE_CACHE:
        _SCENE_CACHE["img"] = compose_scene(0.0)
    img = _SCENE_CACHE["img"].copy()

    u0, v0, *_ = project_pt(0, 0, 0, spin_s)
    rw = R_OUT * _scale

    # ---- scroll layer ----
    obj = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(obj, "RGBA")
    items = []
    half_w = VOX * _scale * 0.5
    key_az = KEY_AZ + sway
    rim_az = RIM_AZ + sway
    fill_az = FILL_AZ                          # fill stays with the camera
    for key, (cx, cy, ang, segs) in COLS.items():
        a_scr = ang + COL_ANG[key] + spin_s        # perturbed shading normal
        gl = COL_SPEC[key]
        lk = max(0.0, math.cos(a_scr - key_az)) + 0.42 * max(0.0, math.cos(a_scr - fill_az))
        lr = 0.55 * max(0.0, math.cos(a_scr - rim_az)) ** 5
        sk = 0.20 * gl * max(0.0, math.cos(a_scr - key_az)) ** 12
        sr = 0.08 * gl * max(0.0, math.cos(a_scr - rim_az)) ** 14
        jit = COL_NOISE[key] / 255.0
        tex = COL_TEX[key]
        wf = COL_W[key]
        ajit = COL_AJIT[key]
        for z0, z1 in segs:
            if scan_h <= z0:
                items.append(("strip", cx, cy, z0, z1, a_scr, gl, jit, tex, wf, ajit))
                items.append(("cap", cx, cy, z1, lk, lr, jit, sk, sr))
            elif scan_h >= z1:
                for kz in np.arange(z0, z1 - 1e-6, RVOX):
                    items.append(("cube", cx, cy, kz + RVOX / 2, lk))
            else:
                for kz in np.arange(z0, scan_h - 1e-6, RVOX):
                    items.append(("cube", cx, cy, kz + RVOX / 2, lk))
                items.append(("strip", cx, cy, scan_h, z1, a_scr, gl, jit, tex, wf, ajit))
                items.append(("cap", cx, cy, z1, lk, lr, jit, sk, sr))

    def sort_key(it):
        z = (it[3] + it[4]) / 2 if it[0] == "strip" else it[3]
        return -project_pt(it[1], it[2], z, spin_s)[2]
    items.sort(key=sort_key)

    front_h = min(scan_h, HEIGHT)
    for it in items:
        if it[0] == "strip":
            _, cx, cy, z0, z1, a_scr, gl, jit, tex, wf, ajit = it
            zs = np.linspace(z0, z1, max(2, int((z1 - z0) / 0.9) + 1))
            for a, b in zip(zs[:-1], zs[1:]):
                ua, va, _, pa = project_pt(cx, cy, a, spin_s)
                ub, vb, _, pb = project_pt(cx, cy, b, spin_s)
                ha = half_w * pa * 1.14 * wf
                ti = int(a / HEIGHT * (len(tex) - 1))
                tj = tex[ti]
                ac = a_scr + ajit[ti]                     # per-chunk normal
                lk = max(0.0, math.cos(ac - key_az)) + 0.42 * max(0.0, math.cos(ac - fill_az))
                lr = 0.55 * max(0.0, math.cos(ac - rim_az)) ** 5
                sk = 0.20 * gl * max(0.0, math.cos(ac - key_az)) ** 12
                sr = 0.08 * gl * max(0.0, math.cos(ac - rim_az)) ** 14
                mult = 0.55 if tj < -0.10 else 1.0        # crack shadows
                col = charcoal_color(lk * mult, lr, jit + tj, sk * mult, sr)
                dr.polygon([(ub - ha, vb), (ub + ha, vb), (ua + ha, va + ha), (ua - ha, va + ha)],
                           fill=col)
                if a == z0:      # rounded lower terminus (flake edge, no 90deg)
                    dr.ellipse([ua - ha, va, ua + ha, va + 2 * ha], fill=col)
                if b == z1:      # rounded upper terminus
                    hb = half_w * pb * 1.14 * wf
                    dr.ellipse([ub - hb, vb - hb, ub + hb, vb + hb], fill=col)
        elif it[0] == "cap":
            _, cx, cy, z, lk, lr, jit, sk, sr = it
            u_, v_, _, p = project_pt(cx, cy, z, spin_s)
            h = half_w * p * 1.16
            c = charcoal_color(lk, lr, jit, sk * 0.5, sr * 0.5)
            c = tuple(clamp255(ch * 1.5) for ch in c)
            dr.polygon([(u_ - h, v_ - h * 0.52), (u_ + h, v_ - h * 0.52),
                        (u_ + h, v_ + h * 0.52), (u_ - h, v_ + h * 0.52)], fill=c)
        else:
            _, cx, cy, z, lk = it
            u_, v_, _, p = project_pt(cx, cy, z, spin_s)
            h = (RVOX * _scale * 0.5) * p
            em = 1.18 if abs(z - front_h) < 0.9 else 1.0
            top = [(u_ - h, v_ - h * 1.55), (u_ + h, v_ - h * 1.55), (u_ + h, v_ - h * 0.55), (u_ - h, v_ - h * 0.55)]
            left = [(u_ - h, v_ - h * 0.55), (u_, v_ - h * 0.05), (u_, v_ + h * 1.05), (u_ - h, v_ + h * 0.5)]
            right = [(u_, v_ - h * 0.05), (u_ + h, v_ - h * 0.55), (u_ + h, v_ + h * 0.5), (u_, v_ + h * 1.05)]
            dr.polygon(top, fill=shade(RED, 1.22 * em))
            dr.polygon(left, fill=shade(RED, (0.76 + lk * 0.10) * em))
            dr.polygon(right, fill=shade(RED, (0.94 + lk * 0.10) * em))

    img.paste(obj, (0, 0), obj)

    # ---- x-ray beam: source fixed in the room -> orbits on screen with sway --
    fu, fv, _, _ = project_pt(0, 0, front_h, spin_s)
    bx, by = S * 0.58, S * 0.62               # base offset from the scan front
    c, s_ = math.cos(sway), math.sin(sway)
    sx = fu + bx * c + 40 * s_                # rotate the offset with the orbit
    sy = fv + by
    bw = rw + 22
    beam = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    db = ImageDraw.Draw(beam)
    db.polygon([(sx - 12, sy + 12), (sx + 16, sy - 14),
                (fu + bw, fv + 2), (fu - bw + 6, fv + 11)], fill=XRAY + (22,))
    for tt in np.linspace(-1, 1, 9):
        db.line([sx, sy, fu + tt * (bw - 6), fv + 7 - abs(tt) * 4], fill=XRAY + (52,), width=2)
    beam = beam.filter(ImageFilter.GaussianBlur(1.8))
    img.paste(beam, (0, 0), beam)
    dr = ImageDraw.Draw(img, "RGBA")
    dr.ellipse([fu - (rw + 14), fv - (rw + 14) * 0.15, fu + (rw + 14), fv + (rw + 14) * 0.15],
               outline=XRAY + (90,), width=3)

    img = img.resize((300, 300), Image.LANCZOS)
    img.save(out)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        N = 75
        for f in range(N):
            t = f / (N - 1)
            spin = t * math.pi
            scan = t * (HEIGHT + 2.0)
            render_frame(spin, scan, f"{SCRATCH}/scanloop/f{f:03d}.png", t=t)
            print(f"\r{f+1}/{N}", end="", flush=True)
        print()
    else:
        render_frame(math.radians(35), HEIGHT * 0.55, f"{SCRATCH}/scan-proto.png", t=0.35)
        print("wrote scan-proto.png")
