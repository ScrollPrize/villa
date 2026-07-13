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

# ---- Flattening: picks up exactly where the unwrapping thumb ends ---------
SPIN_START = 3 * math.pi                      # unwrap's final spin angle
SPIN_STEP = math.pi / 75                      # per-frame velocity (gif-1 rate)
SCALE_ROLL = 20.5 * 1.62        # unwrap v9 ends zoomed on the top half
SCALE_FLAT = 17.0
CZ_ROLL = 15.2
CZ_FLAT = 11.5
PSI = math.radians(-25)         # final sheet yaw: LEFT side recedes
TILT = math.radians(36)         # final pitch: must exceed the 26deg camera elevation
ZP = HEIGHT * 0.45              # tilt pivot height

MESH = (98, 148, 255)

def clamp01(x):
    return max(0.0, min(1.0, x))

def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)

KOFF = PITCH / (2 * math.pi)
TH_MAX = 3.4 * 2 * math.pi
N_TH, N_Z = 58, 48
MESH_TH = np.linspace(0.15, TH_MAX - 0.05, N_TH)
MESH_Z = np.linspace(0.4, HEIGHT - 1.2, N_Z)
_th_step = MESH_TH[1] - MESH_TH[0]
_z_step = MESH_Z[1] - MESH_Z[0]
VJIT = {(i, j): (rng.normal(0, 0.30) * _th_step, rng.normal(0, 0.26) * _z_step)
        for i in range(N_TH) for j in range(N_Z)}
DIAG = {(i, j): rng.random() < 0.5 for i in range(N_TH) for j in range(N_Z)}

def arc_len(th):
    return 0.5 * th + KOFF * th * th / 2

S_ARC_MAX = arc_len(TH_MAX)


# crumpled sheet: smooth radial displacement field (fixed phases so the
# unwrap and flatten thumbs share the exact same surface)
def crumple(th, z):
    return (0.11 * math.sin(1.7 * th + 0.60 * z + 1.3)
            + 0.08 * math.sin(3.3 * th - 1.10 * z + 4.1)
            + 0.05 * math.sin(6.1 * th + 2.20 * z + 2.6)
            + 0.06 * math.sin(0.9 * th + 1.60 * z + 5.5))

def mesh_pt(i, j):
    dth, dz = VJIT[(i, j)]
    th = min(max(MESH_TH[i] + dth, 0.05), TH_MAX)
    z = min(max(MESH_Z[j] + dz, 0.3), HEIGHT - 0.7)
    return th, z

# ---- towel-unroll kinematics ------------------------------------------------
# The roll travels along the sheet plane, spinning, laying the outer portion
# flat behind it. Laid points stay put; the contact is tangent-continuous.
PSI_D = (math.cos(PSI), math.sin(PSI))          # sheet direction (world xy)
PSI_N = (math.sin(PSI), -math.cos(PSI))         # plane normal (toward camera)
A0 = PSI + math.pi / 2                          # azimuth of -n (contact dir)
S_TOTAL = S_ARC_MAX

def th_of_arc(s):
    # invert s = 0.5*th + KOFF*th^2/2
    return (-0.5 + math.sqrt(0.25 + 2 * KOFF * max(0.0, s))) / KOFF

CHI0 = (SPIN_START + TH_MAX - A0) % (2 * math.pi)

_VOXSHEET = None

def _voxsheet_cols():
    # Solid columns clipped to the mesh boundary (inherits its wavy jitter).
    global _VOXSHEET
    if _VOXSHEET is None:
        top = sorted((arc_len(mesh_pt(i, N_Z - 1)[0]) - S_TOTAL / 2,
                      mesh_pt(i, N_Z - 1)[1]) for i in range(N_TH))
        bot = sorted((arc_len(mesh_pt(i, 0)[0]) - S_TOTAL / 2,
                      mesh_pt(i, 0)[1]) for i in range(N_TH))
        left = sorted((mesh_pt(0, j)[1],
                       arc_len(mesh_pt(0, j)[0]) - S_TOTAL / 2) for j in range(N_Z))
        right = sorted((mesh_pt(N_TH - 1, j)[1],
                        arc_len(mesh_pt(N_TH - 1, j)[0]) - S_TOTAL / 2) for j in range(N_Z))
        tx, tz = zip(*top); bx, bz = zip(*bot)
        lz, lx = zip(*left); rz, rx = zip(*right)
        r2 = np.random.default_rng(11)
        cols = []
        for xi in np.arange(-S_TOTAL / 2, S_TOTAL / 2 + RVOX, RVOX * 0.92):
            zt = float(np.interp(xi, tx, tz))
            zb = float(np.interp(xi, bx, bz))
            cubes_z = []
            for kz in np.arange(zb, zt - 1e-6, RVOX):
                zc = kz + RVOX / 2
                if np.interp(zc, lz, lx) - 0.15 <= xi <= np.interp(zc, rz, rx) + 0.15:
                    cubes_z.append(kz)
            if cubes_z:
                cols.append((xi, 0.9 + r2.normal(0, 6.5) / 255.0, cubes_z))
        _VOXSHEET = cols
    return _VOXSHEET

def tilt_pt(x, y, z, tau):
    xi_p = x * PSI_D[0] + y * PSI_D[1]
    nu = x * PSI_N[0] + y * PSI_N[1]
    ze = z - ZP
    nu2 = nu * math.cos(tau) - ze * math.sin(tau)
    ze2 = nu * math.sin(tau) + ze * math.cos(tau)
    return (xi_p * PSI_D[0] + nu2 * PSI_N[0],
            xi_p * PSI_D[1] + nu2 * PSI_N[1],
            ZP + ze2)

def render_frame(u, chi, out, vox=0.0):
    """u: unrolled fraction 0..1; chi: extra roll rotation (continuity/hold)."""
    global _elev, _scale, _voff, _cz
    e = smoothstep(u)
    _elev = math.radians(26)
    _scale = SCALE_ROLL * (1 - e) + SCALE_FLAT * e
    _cz = CZ_ROLL * (1 - e) + CZ_FLAT * e
    _voff = 0.0
    img = compose_scene(0.0)
    dr = ImageDraw.Draw(img, "RGBA")

    laid = u * S_TOTAL
    s_c = S_TOTAL - laid                         # arc position of the contact
    th_c = th_of_arc(s_c)
    R_c = 0.5 + KOFF * th_c                      # roll radius at the contact
    xi_c = s_c - S_TOTAL / 2                     # contact position on the plane
    C = (xi_c * PSI_D[0] + R_c * PSI_N[0],       # roll center (world xy)
         xi_c * PSI_D[1] + R_c * PSI_N[1])
    # global drift: roll starts centered on screen (unwrap pose), sheet ends centered
    O = (-C[0] * (1 - e), -C[1] * (1 - e))
    cch, sch = math.cos(chi), math.sin(chi)

    V, VF = {}, {}
    for i in range(N_TH):
        for j in range(N_Z):
            th, z = mesh_pt(i, j)
            s = arc_len(th)
            if s >= s_c:                          # laid flat
                xi = s - S_TOTAL / 2
                x = xi * PSI_D[0] + O[0]
                y = xi * PSI_D[1] + O[1]
                fr = -1.0
            else:                                 # still on the roll
                ang = th % (2 * math.pi)
                r_v = (0.5 + KOFF * th) * r_mult(ang) + 0.10 + crumple(th, z)
                dth = th_c - th
                # u-hat = -n*cos - d*sin  (tangent-continuous at the contact)
                ux = -PSI_N[0] * math.cos(dth) - PSI_D[0] * math.sin(dth)
                uy = -PSI_N[1] * math.cos(dth) - PSI_D[1] * math.sin(dth)
                # extra rotation chi about the roll center (hold-phase spin)
                rx, ry = r_v * ux, r_v * uy
                rx, ry = rx * cch - ry * sch, rx * sch + ry * cch
                x, y = C[0] + rx + O[0], C[1] + ry + O[1]
                fr = math.sin(math.atan2(ry, rx))  # crude front-ness of the roll
            # rigid pitch about the d-axis: top edge back, bottom forward
            if e > 0:
                x, y, z = tilt_pt(x, y, z, TILT * e)
            V[(i, j)] = project_pt(x, y, z, 0.0)
            VF[(i, j)] = fr
    edges = set()
    for i in range(N_TH - 1):
        for j in range(N_Z - 1):
            edges.add(((i, j), (i + 1, j)))
            edges.add(((i, j), (i, j + 1)))
            if DIAG[(i, j)]:
                edges.add(((i, j), (i + 1, j + 1)))
            else:
                edges.add(((i + 1, j), (i, j + 1)))
    for i in range(N_TH - 1):
        edges.add(((i, N_Z - 1), (i + 1, N_Z - 1)))
    for j in range(N_Z - 1):
        edges.add(((N_TH - 1, j), (N_TH - 1, j + 1)))
    edge_px = []
    for a_, b_ in edges:
        u1, v1, d1, _ = V[a_]
        u2, v2, d2, _ = V[b_]
        edge_px.append((u1, v1, u2, v2, (d1 + d2) / 2, (VF[a_] + VF[b_]) / 2))
    edge_px.sort(key=lambda e_: -e_[4])
    for u1, v1, u2, v2, d, fr in edge_px:
        a = 148 if fr < 0 else 24
        dr.line([u1, v1, u2, v2], fill=MESH + (a,), width=1)

    if vox > 0:
        front = -S_TOTAL / 2 + vox * (S_TOTAL + 1.5)
        cubes = []
        for xi, lk, zlist in _voxsheet_cols():
            if xi > front:
                continue
            em = 1.18 if (front - xi) < 0.9 and vox < 1.0 else 1.0
            wx = xi * PSI_D[0]
            wy = xi * PSI_D[1]
            for kz in zlist:
                x, y, z = tilt_pt(wx, wy, kz + RVOX / 2, TILT)
                cubes.append((x, y, z, lk, em))
        cubes.sort(key=lambda cb: -project_pt(cb[0], cb[1], cb[2], 0.0)[2])
        for x, y, z, lk, em in cubes:
            u_, v_, _, p = project_pt(x, y, z, 0.0)
            h = (RVOX * _scale * 0.5) * p
            top = [(u_ - h, v_ - h * 1.55), (u_ + h, v_ - h * 1.55), (u_ + h, v_ - h * 0.55), (u_ - h, v_ - h * 0.55)]
            left = [(u_ - h, v_ - h * 0.55), (u_, v_ - h * 0.05), (u_, v_ + h * 1.05), (u_ - h, v_ + h * 0.5)]
            right = [(u_, v_ - h * 0.05), (u_ + h, v_ - h * 0.55), (u_ + h, v_ + h * 0.5), (u_, v_ + h * 1.05)]
            dr.polygon(top, fill=shade(RED, 1.22 * em))
            dr.polygon(left, fill=shade(RED, (0.76 + lk * 0.10) * em))
            dr.polygon(right, fill=shade(RED, (0.94 + lk * 0.10) * em))

    img = img.resize((300, 300), Image.LANCZOS)
    img.save(out)

if __name__ == "__main__":
    N = 150                     # 6.25 s @24fps = ink-gif cycle (strip sync)
    HOLD0 = 10
    UNROLL_END = 89
    VOX_START, VOX_END = 95, 134
    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        for f in range(N):
            if f < HOLD0:
                u = 0.0
                chi = CHI0 + SPIN_STEP * f
            elif f < UNROLL_END:
                u = (f - HOLD0) / (UNROLL_END - HOLD0 - 1)
                chi = (CHI0 + SPIN_STEP * HOLD0) * (1 - smoothstep(u))
            else:
                u, chi = 1.0, 0.0
            vox = 0.0 if f < VOX_START else min(1.0, (f - VOX_START) / (VOX_END - VOX_START))
            render_frame(u, chi, f"{SCRATCH}/flatloop/f{f:03d}.png", vox)
            print(f"\r{f+1}/{N}", end="", flush=True)
        print()
    else:
        u = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        chi = CHI0 * (1 - smoothstep(u))
        render_frame(u, chi, f"{SCRATCH}/flat-proto.png")
        print("wrote flat-proto.png")
