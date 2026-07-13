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
                    cols.setdefault((ii, jj), (cx, cy, int(th / (2 * math.pi)), th))
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
    for key, (cx, cy, wind, th) in cols.items():
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
        koff = PITCH / (2 * math.pi)
        s_arc = r0 * th + koff * th * th / 2      # arc length along the spiral
        off = math.hypot(cx, cy) - (r0 + koff * th)   # radial offset in the sheet
        out[key] = (cx * m, cy * m, ang, segs, s_arc, off)
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
_scale = 14.4
_voff = 0.0

_cz = CENTER_Z

def project_pt(x, y, z, spin):
    c, s = math.cos(spin), math.sin(spin)
    xr = x * c - y * s
    yr = x * s + y * c
    zr = z - _cz
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

SPIN_END = math.pi              # scanning gif ends here
SCALE_HOLD = 20.0 * 1.025       # scanning gif ends here too
N_FRAMES = 150                  # 6.25 s @24fps = ink-gif cycle (strip sync)
SPIN_RATE = 2 * math.pi           # same angular velocity as gif 1 (pi/75 per frame)

MESH = (98, 148, 255)

def clamp01(x):
    return max(0.0, min(1.0, x))

def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)

# --- mesh vertex grid over the spiral sheet (ideal surface: no flake holes) ---
KOFF = PITCH / (2 * math.pi)
TH_MAX = 3.4 * 2 * math.pi
N_TH, N_Z = 58, 48
MESH_TH = np.linspace(0.15, TH_MAX - 0.05, N_TH)
MESH_Z = np.linspace(0.4, HEIGHT - 1.2, N_Z)
_th_step = MESH_TH[1] - MESH_TH[0]
_z_step = MESH_Z[1] - MESH_Z[0]
# static irregularity: jittered vertices, randomly flipped quad diagonals
VJIT = {(i, j): (rng.normal(0, 0.30) * _th_step, rng.normal(0, 0.26) * _z_step)
        for i in range(N_TH) for j in range(N_Z)}
DIAG = {(i, j): rng.random() < 0.5 for i in range(N_TH) for j in range(N_Z)}

def mesh_pt(i, j):
    dth, dz = VJIT[(i, j)]
    th = min(max(MESH_TH[i] + dth, 0.05), TH_MAX)
    z = min(max(MESH_Z[j] + dz, 0.3), HEIGHT - 0.7)
    return th, z


# crumpled sheet: smooth radial displacement field (fixed phases so the
# unwrap and flatten thumbs share the exact same surface)
def crumple(th, z):
    return (0.11 * math.sin(1.7 * th + 0.60 * z + 1.3)
            + 0.08 * math.sin(3.3 * th - 1.10 * z + 4.1)
            + 0.05 * math.sin(6.1 * th + 2.20 * z + 2.6)
            + 0.06 * math.sin(0.9 * th + 1.60 * z + 5.5))

def mesh_vertex(th, z, spin):
    ang = th % (2 * math.pi)
    rad = (0.5 + KOFF * th) * r_mult(ang) + 0.10 + crumple(th, z)
    return project_pt(rad * math.cos(th), rad * math.sin(th), z, spin)

def render_frame(t, out, spin_frac=None):
    global _elev, _scale, _voff, _cz
    _elev = math.radians(26)
    zt = smoothstep(spin_frac if spin_frac is not None else t)
    _scale = SCALE_HOLD * (1 + 0.62 * zt)          # 20.5 -> ~33
    _cz = CENTER_Z + (15.2 - CENTER_Z) * zt        # look-at drifts to the top half
    _voff = 0.0
    spin_s = SPIN_END + SPIN_RATE * (spin_frac if spin_frac is not None else t)
    img = compose_scene(0.0)
    dr = ImageDraw.Draw(img, "RGBA")

    # timeline: hold solid -> fade to translucent; mesh grows bottom-up
    fade = smoothstep((t - 0.07) / 0.14)
    ghost = 1.0 - 0.68 * fade                      # layer opacity 1.0 -> 0.32
    grow = smoothstep((t - 0.13) / 0.58)
    front_th = grow * (TH_MAX + 0.4)
    dis = smoothstep((t - 0.74) / 0.22)
    cutoff = (HEIGHT + 1.5) * (1.0 - dis) - 0.5    # voxels above this are gone

    # --- translucent red voxels ---
    items = []
    for key, (cx, cy, ang, segs, s_arc, off) in COLS.items():
        a_scr = ang + COL_ANG[key] + spin_s
        lk = max(0.0, math.cos(a_scr - KEY_AZ)) + 0.42 * max(0.0, math.cos(a_scr - FILL_AZ))
        for z0, z1 in segs:
            for kz in np.arange(z0, z1 - 1e-6, RVOX):
                items.append((cx, cy, kz + RVOX / 2, lk))
    items.sort(key=lambda it: -project_pt(it[0], it[1], it[2], spin_s)[2])
    obj = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    do = ImageDraw.Draw(obj)
    for x, y, z, lk in items:
        fac = max(0.0, min(1.0, (cutoff - z) / 1.4))
        if fac <= 0.0:
            continue
        av = (int(255 * fac),) if fac < 1.0 else ()
        u_, v_, _, p = project_pt(x, y, z, spin_s)
        h = (RVOX * _scale * 0.5) * p
        top = [(u_ - h, v_ - h * 1.55), (u_ + h, v_ - h * 1.55), (u_ + h, v_ - h * 0.55), (u_ - h, v_ - h * 0.55)]
        left = [(u_ - h, v_ - h * 0.55), (u_, v_ - h * 0.05), (u_, v_ + h * 1.05), (u_ - h, v_ + h * 0.5)]
        right = [(u_, v_ - h * 0.05), (u_ + h, v_ - h * 0.55), (u_ + h, v_ + h * 0.5), (u_, v_ + h * 1.05)]
        do.polygon(top, fill=shade(RED, 1.22) + av)
        do.polygon(left, fill=shade(RED, 0.76 + lk * 0.10) + av)
        do.polygon(right, fill=shade(RED, 0.94 + lk * 0.10) + av)
    # fade the whole voxel layer uniformly (no per-face accumulation)
    if ghost < 1.0:
        obj.putalpha(obj.getchannel("A").point(lambda v: int(v * ghost)))
    img.paste(obj, (0, 0), obj)

    # --- triangular mesh growing from the bottom, mapping the spiral ---
    if grow > 0:
        V, VT, VF = {}, {}, {}
        for i in range(N_TH):
            for j in range(N_Z):
                th, z = mesh_pt(i, j)
                if th <= front_th:
                    V[(i, j)] = mesh_vertex(th, z, spin_s)
                    VT[(i, j)] = th
                    VF[(i, j)] = math.sin(th + spin_s)   # <0 = faces the camera
        edges = set()
        for i in range(N_TH - 1):
            for j in range(N_Z - 1):
                edges.add(((i, j), (i + 1, j)))
                edges.add(((i, j), (i, j + 1)))
                if DIAG[(i, j)]:
                    edges.add(((i, j), (i + 1, j + 1)))
                else:
                    edges.add(((i + 1, j), (i, j + 1)))
        # boundary rows/cols
        for i in range(N_TH - 1):
            edges.add(((i, N_Z - 1), (i + 1, N_Z - 1)))
        for j in range(N_Z - 1):
            edges.add(((N_TH - 1, j), (N_TH - 1, j + 1)))
        edge_px = []
        for a_, b_ in edges:
            if a_ in V and b_ in V:
                u1, v1, d1, _ = V[a_]
                u2, v2, d2, _ = V[b_]
                edge_px.append((u1, v1, u2, v2, (d1 + d2) / 2,
                                max(VT[a_], VT[b_]), (VF[a_] + VF[b_]) / 2))
        edge_px.sort(key=lambda e: -e[4])
        for u1, v1, u2, v2, d, ez, fr in edge_px:
            near_front = abs(ez - front_th) < 0.55 and grow < 1.0
            a = 148 if fr < 0 else 24              # angular front-ness, not depth
            if near_front:
                a = min(255, a + 55)
            dr.line([u1, v1, u2, v2], fill=MESH + (a,), width=1)

    img = img.resize((300, 300), Image.LANCZOS)
    img.save(out)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        N = N_FRAMES
        for f in range(N):
            phase = min(1.0, f / 124.0)          # action ends ~1s before the loop
            spin_frac = f / (N - 1)              # spin never stops
            render_frame(phase, f"{SCRATCH}/unwraploop/f{f:03d}.png", spin_frac)
            print(f"\r{f+1}/{N}", end="", flush=True)
        print()
    else:
        render_frame(float(sys.argv[1]) if len(sys.argv) > 1 else 0.55,
                     f"{SCRATCH}/unwrap-proto.png")
        print("wrote unwrap-proto.png")
