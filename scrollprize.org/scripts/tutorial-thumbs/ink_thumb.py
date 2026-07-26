#!/usr/bin/env python3
"""'Ink Detection' tutorial thumb — stylized but physical. Rev 12.

Picks up exactly where the flattening thumb ends: the tilted flat sheet,
solidly red-voxelized, at the SAME framing (no zoom-out — the camera
never pulls back before it zooms in). As the scan starts, the red sheet
LIFTS straight up off its plane and hovers overhead: a TINY glowing
inference kernel (a fraction of a letter — it finds ink *signal*;
signals form letters, letters form the text) sweeps the INTACT hovering
sheet in raster rows — nothing is consumed — its reading projected DOWN
across the gap, a ray landing on the EXACT plane the sheet lay on,
where a synced write head prints the ink prediction line by line (below
the raster front: nothing but background). The decoded text is written
right where the sheet was — same footprint, same level, never an
extension off to the side, and the sheets never overlap at any zoom.
The camera dwells a long time on the projection, then a single, slow,
DIRECT zoom dives into the prediction, easing into a letter-by-letter
cruise — the raster front riding just below the letter tops. The final
move is one last zoom: straight out to the COMPLETED reconstruction —
the text finishes filling in as the zoom lands, the rig untilts to a
flat, dead-on view, and the red input rises away and fades like a
retracting scanner — leaving the finished prediction (a REAL PHerc.
Paris 4 ink prediction, recolored to the strip palette) held as a
padded full image for the last second.

The write strip is exactly the kernel height at all times.

This thumb runs 300 frames = 12.5 s (2x the strip cycle: the strip still
re-syncs every 12.5 s). Shares flatten_thumb's exact geometry (same seeds,
same rng consumption order) so frame 0 is pixel-identical to flatten's
final frame.

Usage: ink_thumb.py             -> one prototype frame (ink-proto.png)
       ink_thumb.py loop        -> 300 frames in inkloop/
"""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SCRATCH = "/tmp/claude-1000/-home-err-src-villa-explore-villa-restyle/6641ab8c-874b-4ec5-a12a-9c19330c58f9/scratchpad"

# ---- palette ----------------------------------------------------------------
BG_TOP = (26, 27, 31)
RED = (198, 40, 34)
CHAR = (40, 38, 36)
RIM = (30, 40, 62)
SPEC_KEY = (205, 196, 182)
XRAY = (170, 205, 255)
S = 600

rng = np.random.default_rng(7)

# ---- geometry: copied verbatim from flatten_thumb.py so the rng stream and
# ---- therefore the sheet surface/boundary match exactly ----------------------
VOX = 0.13
RVOX = 0.30
R_MAX = 3.4
HEIGHT = 23.0
PITCH = 0.58
ROUND = 2.0

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
        m = r_mult(ang) * (1.0 + max(-0.06, min(0.06, rng.normal(0, 0.03))))
        koff = PITCH / (2 * math.pi)
        s_arc = r0 * th + koff * th * th / 2
        off = math.hypot(cx, cy) - (r0 + koff * th)
        out[key] = (cx * m, cy * m, ang, segs, s_arc, off)
    return out, r_out

COLS, R_OUT = build_columns()
COL_NOISE = {k: rng.normal(0, 6.5) for k in COLS}
COL_TEX = {k: rng.normal(0, 0.075, 28) for k in COLS}
COL_W = {k: 1.0 + max(-0.25, min(0.38, rng.normal(0, 0.16))) for k in COLS}
COL_ANG = {k: rng.normal(0, 0.22) for k in COLS}
COL_AJIT = {k: rng.normal(0, 0.34, 28) for k in COLS}
COL_SPEC = {k: rng.uniform(0.12, 1.0) for k in COLS}

# ---- camera: look-at-target projection; starts at flatten_thumb's final pose
# (target (0,0,CZ_FLAT) + scale SCALE_FLAT reproduces flatten's fixed camera
# exactly; set_pose() animates target/scale/elevation per frame)
CAM_DIST = 95.0
SCALE_FLAT = 17.0
CZ_FLAT = 11.5
_elev = math.radians(26)
_scale = SCALE_FLAT
_TARGET = (0.0, 0.0, CZ_FLAT)

def project_pt(x, y, z, spin):
    c, s = math.cos(spin), math.sin(spin)
    xr = x * c - y * s
    yr = x * s + y * c
    dx, dy, dz = xr - _TARGET[0], yr - _TARGET[1], z - _TARGET[2]
    ce, se = math.cos(_elev), math.sin(_elev)
    yc = dy * ce - dz * se
    zc = dy * se + dz * ce
    persp = CAM_DIST / (CAM_DIST + yc)
    return (S / 2 + dx * _scale * persp, S * 0.52 - zc * _scale * persp, yc, persp)

def clamp255(v):
    return 255 if v > 255 else (0 if v < 0 else int(v))

def shade(base, f):
    return tuple(clamp255(ch * f) for ch in base)

def compose_scene():
    img = Image.new("RGB", (S, S), BG_TOP)
    glow = Image.new("L", (S, S), 0)
    dg = ImageDraw.Draw(glow)
    dg.ellipse([S * 0.10, S * 0.06, S * 0.90, S * 0.94], fill=46)
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    img = Image.composite(Image.new("RGB", (S, S), (48, 50, 57)), img, glow)
    return img

def clamp01(x):
    return max(0.0, min(1.0, x))

def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)

PSI = math.radians(-25)
TILT = math.radians(36)
ZP = HEIGHT * 0.45
MESH = (98, 148, 255)

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
S_TOTAL = S_ARC_MAX

def mesh_pt(i, j):
    dth, dz = VJIT[(i, j)]
    th = min(max(MESH_TH[i] + dth, 0.05), TH_MAX)
    z = min(max(MESH_Z[j] + dz, 0.3), HEIGHT - 0.7)
    return th, z

PSI_D = (math.cos(PSI), math.sin(PSI))
PSI_N = (math.sin(PSI), -math.cos(PSI))

_VOXSHEET = None

def _voxsheet_cols():
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

# ---- ink texture: carbonized papyrus + REAL ink prediction -------------------
# PHerc. Paris 4 (2 µm), TimeSformer-style patch inference, Hann-blended tiles.
# The jpg is stored upside down; flipped upright here. Full picture is used —
# its aspect (1.48) matches the sheet texture region (1.47) almost exactly.
PRED = ("/media/err/xdext4xd/ink/preds/"
        "outputs_PHercParis4_2um_paris4_r152_2umr1_256_ep29_r152_s256_fwd/"
        "w05_4424_tile256_stride128_layers1_63_hann_fwd.jpg")
TEX_W, TEX_H = 1408, 954
XI0, XI1 = -S_TOTAL / 2 - 0.3, S_TOTAL / 2 + 0.3
Z_TOP, Z_BOT = HEIGHT - 0.9, 0.2

# output sheet: the prediction rasters onto the EXACT plane the red sheet
# started on — same footprint, same level. The red input sheet LIFTS
# straight up (a pure z rise in rig space) and the kernel's reading is
# projected DOWN across the gap: the decoded text is written right where
# the sheet lay, not off to the side (it must not read as an extension
# of the prediction). At the very end the input rises further and fades,
# like a scanner retracting.
D_XI = 0.0
D_NU = 0.0
R_LIFT = 18.0                   # the hover: high enough that the red slab
                                # does not sit in front of the freshly
                                # painted rows — with the dwell pull-back
                                # BOTH sheets show almost completely:
                                # kernel riding the slab, the gap, and the
                                # prediction being written below
F_LIFT = 45                     # ...reached here (eased from frame 0)

def red_z(f):
    """The input sheet's lift at frame f, in WORLD z (applied after the
    tilt): it hovers straight up. Together with the flat dwell tilt the
    stack reads as a right prism — the projection drops perpendicularly
    onto the output plane. Three phases: rise to the hover (0 at frame 0
    — frame 0 stays pixel-identical to flatten's final frame), rise
    FURTHER as the dive starts so the slab clears the frame before the
    cruise (no overlap at any zoom), and the final retracting rise as it
    is dismissed."""
    return (R_LIFT * smoothstep(f / F_LIFT)
            + 12.0 * smoothstep((f - 108) / 37)
            + 16.0 * smoothstep((f - 238) / 34))

def plane_pt(xi, z, dxi=0.0, nu=0.0, dz=0.0):
    """Project a point of the sheet-plane family (offset dxi in-plane,
    nu along the normal, dz straight up in WORLD space) at the current
    pose."""
    x = (xi + dxi) * PSI_D[0] + nu * PSI_N[0]
    y = (xi + dxi) * PSI_D[1] + nu * PSI_N[1]
    x, y, zz = tilt_pt(x, y, z, TAU)
    return project_pt(x, y, zz + dz, 0.0)

def tex_uv(xi, z):
    return ((xi - XI0) / (XI1 - XI0) * TEX_W,
            (Z_TOP - z) / (Z_TOP - Z_BOT) * TEX_H)

def build_ink_tex():
    r3 = np.random.default_rng(23)
    # carbonized sheet: charcoal mottle + horizontal papyrus fibers + key light
    lo = r3.normal(0, 1, (TEX_H // 16, TEX_W // 16))
    lo = np.asarray(Image.fromarray(((lo - lo.min()) / (np.ptp(lo) + 1e-6) * 255
                                     ).astype(np.uint8)).resize((TEX_W, TEX_H),
                                     Image.BILINEAR), np.float32) / 255 - 0.5
    fib = r3.normal(0, 1, (TEX_H, TEX_W)).astype(np.float32)
    fib = np.asarray(Image.fromarray(np.clip(fib * 60 + 128, 0, 255).astype(np.uint8)
                                     ).filter(ImageFilter.GaussianBlur(1)).resize(
                                     (TEX_W, TEX_H)), np.float32)
    fib = (np.roll(fib, 1, 1) + fib + np.roll(fib, -1, 1)) / 3 / 255 - 0.5
    yy, xx = np.mgrid[0:TEX_H, 0:TEX_W]
    key = 1.0 + 0.18 * (1 - xx / TEX_W) * (1 - yy / TEX_H) - 0.10 * (yy / TEX_H)
    f = (1.0 + 0.22 * lo + 0.16 * fib) * key
    base = np.clip(np.array([[[52, 49, 45]]], np.float32) * f[..., None], 0, 255)

    # real prediction as the ink layer, recolored to the strip's pale-ink tone
    Image.MAX_IMAGE_PIXELS = None
    pred = (Image.open(PRED).convert("L")
            .transpose(Image.FLIP_TOP_BOTTOM)
            .resize((TEX_W, TEX_H), Image.LANCZOS))
    v = np.asarray(pred, np.float32) / 255
    a = np.clip((v - 0.16) / (0.70 - 0.16), 0, 1) ** 0.85
    ink = np.array([[[218, 208, 188]]], np.float32)
    rgb = base * (1 - 0.94 * a[..., None]) + ink * 0.94 * a[..., None]
    return Image.fromarray(rgb.astype(np.uint8))

def build_sheet_mask():
    m = Image.new("L", (TEX_W, TEX_H), 0)
    poly = []
    for i in range(N_TH):
        th, z = mesh_pt(i, N_Z - 1)
        poly.append(tex_uv(arc_len(th) - S_TOTAL / 2, z))
    for j in range(N_Z - 1, -1, -1):
        th, z = mesh_pt(N_TH - 1, j)
        poly.append(tex_uv(arc_len(th) - S_TOTAL / 2, z))
    for i in range(N_TH - 1, -1, -1):
        th, z = mesh_pt(i, 0)
        poly.append(tex_uv(arc_len(th) - S_TOTAL / 2, z))
    for j in range(N_Z):
        th, z = mesh_pt(0, j)
        poly.append(tex_uv(arc_len(th) - S_TOTAL / 2, z))
    ImageDraw.Draw(m).polygon(poly, fill=255)
    return m.filter(ImageFilter.GaussianBlur(1.5))

INK_TEX = None
SHEET_MASK = None

# homography: texture px -> screen for a plane of the sheet family (offset
# dxi in-plane, nu along the normal), as PIL PERSPECTIVE coeffs (screen -> tex)
def _perspective_coeffs(dxi=0.0, nu=0.0):
    src = []   # screen
    dst = []   # texture
    for xi, z in ((XI0, Z_TOP), (XI1, Z_TOP), (XI1, Z_BOT), (XI0, Z_BOT)):
        u, v, _, _ = plane_pt(xi, z, dxi, nu)
        src.append((u, v))
        dst.append(tex_uv(xi, z))
    A, B = [], []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        B += [dx, dy]
    return np.linalg.solve(np.array(A), np.array(B))

# ---- raster scan kinematics ---------------------------------------------------
# The kernel is TINY (KH < a letter height): it detects ink signal, signals
# form letters, letters form the text. It rasters boustrophedon rows over
# the intact red sheet from the top edge downward; its reading is beamed
# across the gap to the output sheet, where the synced write head prints
# the matching strip of prediction. The write strip is exactly the kernel
# height at all times.
N = 300
KH = 0.22                       # kernel side (world units); letters are ~1.2
CUBE = KH
Z_START = Z_TOP + KH            # scan starts one kernel-height above the
                                # top edge (the first pass skims the torn
                                # boundary): no rows are ever revealed
                                # "for free" above the kernel — and it
                                # shifts every row up half a letter-notch,
                                # placing the cruise front exactly where
                                # the letter tops just keep emerging
NR = math.ceil((Z_START - Z_BOT) / KH)
EW = S_TOTAL / 2 + 0.8
ROWPATH = 2 * EW
PATH_TOTAL = NR * ROWPATH

# ---- camera profile (kernel-independent; the kernel speed plan needs it) -----
POSE0 = (math.radians(36), math.radians(-25), math.radians(26))
POSE_D = (math.radians(72), math.radians(-5), math.radians(20))
                                # dwell pose: the rig rotates ON from
                                # frame 0 (not just up!) to a much flatter
                                # tilt — the two sheets read as stacked
                                # shelves and the world-z lift is close to
                                # their normal: a right prism, the beam
                                # dropping perpendicularly
POSE1 = (math.radians(11), math.radians(-7), math.radians(11))
POSE_FLAT = (0.0, 0.0, 0.0)     # the very end: no tilt at all — the
                                # prediction faces the camera dead-on,
                                # a flat 2D image
TAU = POSE0[0]
C0 = (0.0, 0.0, CZ_FLAT)        # flatten's look-at
SC_MAX = 136.0                  # really close: double the previous zoom
SC_DWELL = 12.0                 # dwell scale: a gentle pull-back during
                                # the opening pan so the WHOLE projection
                                # fits — both sheets almost complete, the
                                # painted rows never hidden behind the
                                # hovering slab
SC_END = 15.0                   # final scale: the flat prediction, padded
F_PAN = 40                      # opening PAN RIGHT (no zoom!): flatten's
                                # framing slides over to the projection —
                                # red sheet left, output screen right —
                                # as the scan starts
F_DIVE = 110                    # long dwell on the projection first; only
                                # then does the (single, direct) zoom-in
F_ZOOM = 175                    # start — slow — completing here
F_B = 165                       # rendezvous: the (slowed) kernel meets the
                                # (zoomed) camera about 2/3 down the sheet,
                                # with the lines above already read
F_FREEZE = 215                  # the follow point freezes here: from then
                                # on the camera only zooms, zero shake
F_TILT0 = 215                   # one LAST zoom: straight out to the
F_TILT1 = 278                   # completed reconstruction — untilting to
                                # the flat dead-on view as it lands
F_DONE_TGT = 276                # ...and the text fully completes right as
                                # that last zoom settles

def scale_at(f):
    # pan with a gentle pull-back to the dwell scale, long projection
    # dwell, one direct slow zoom-in, then one last zoom straight out to
    # the flat completed reconstruction
    s0 = SCALE_FLAT + (SC_DWELL - SCALE_FLAT) * smoothstep(f / F_PAN)
    s_in = s0 + (SC_MAX - s0) * smoothstep((f - F_DIVE) / (F_ZOOM - F_DIVE))
    w_e = smoothstep((f - F_TILT0) / (F_TILT1 - F_TILT0))
    return s_in * (1 - w_e) + SC_END * w_e

# ---- kernel speed plan ---------------------------------------------------------
# Planned in SCREEN px/frame, then divided by the zoom: while the camera
# dives in, the kernel keeps a constant apparent speed, so it never *looks*
# like it is braking — the zoom absorbs the slow-down (special relativity,
# sort of). Fast over the bare torn-margin rows, cruise across the letter
# row while zoomed, exponential growth once the camera pulls back. Two
# one-dimensional solves pin the plan: the opening speed lands the kernel
# on the letter row exactly at F_B, the growth rate reads the whole sheet
# by F_DONE_TGT.
A_CRUISE = 11.0                 # apparent cruise speed (px/frame @600):
                                # each letter dwells ~7 frames at full zoom
S_MEET = 30 * ROWPATH + 22.0    # path position at F_B: 22 units into row 30
                                # (past the red sheet's right edge — the
                                # zoom lands right-of-center on the
                                # prediction, at the start of a column) —
                                # two rows ABOVE the line's core: the
                                # raster front sits just below the letter
                                # tops, so only a hint of each letter
                                # shows as it is written; an even row so
                                # the cruise runs LTR

KICK = 15.0                     # when the camera lets go of the kernel it
                                # sprints: a smooth x15 ramp over 10 frames,
                                # then exponential growth (rate solved)
A0 = 18.0                       # opening apparent speed (px/frame @600);
F_PEAK = F_DIVE                 # the kernel accelerates UNIFORMLY (linear
                                # in screen speed) from A0 to the solved
                                # peak exactly as the dive starts...
F_SETTLE = 150                  # ...then eases down to the cruise speed
                                # BEFORE the zoom lands: through the
                                # landing the apparent speed is already
                                # constant — no hitch, no slowdown

def _plan():
    def build(a_hi, g):
        v = []
        base = A_CRUISE / scale_at(F_FREEZE)
        for f in range(N + 1):
            if f <= F_PEAK:
                a = A0 + (a_hi - A0) * (f / F_PEAK)
                v.append(a / scale_at(f))
            elif f <= F_SETTLE:
                a = A_CRUISE + (a_hi - A_CRUISE) * (1 - smoothstep((f - F_PEAK) / (F_SETTLE - F_PEAK)))
                v.append(a / scale_at(f))
            elif f <= F_FREEZE:
                v.append(A_CRUISE / scale_at(f))
            else:
                v.append(base * KICK ** smoothstep((f - F_FREEZE) / 10)
                         * g ** max(0, f - F_FREEZE - 10))
        s = [0.0]
        for f in range(N):
            s.append(min(s[-1] + v[f], PATH_TOTAL))
        return v, s
    lo, hi = A_CRUISE, 6000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        _, s = build(mid, 1.0)
        lo, hi = (mid, hi) if s[F_B] < S_MEET else (lo, mid)
    a_hi = (lo + hi) / 2
    lo, hi = 1.0, 1.3
    for _ in range(60):
        mid = (lo + hi) / 2
        _, s = build(a_hi, mid)
        lo, hi = (mid, hi) if s[F_DONE_TGT] < PATH_TOTAL else (lo, mid)
    g = (lo + hi) / 2
    v, s = build(a_hi, g)
    done = next(f for f in range(N + 1) if s[f] >= PATH_TOTAL - 1e-6)
    return v, s, done

V_TAB, S_TAB, F_DONE = _plan()

def _pos_at(s):
    r = min(NR - 1, int(s / ROWPATH))
    prog = s - r * ROWPATH
    d = 1 if r % 2 == 0 else -1
    kxi = -EW + prog if d > 0 else EW - prog
    return kxi, Z_START - (r + 0.5) * KH, r, d

def kernel_state(f):
    """-> (kxi, kz, row, dir, done)"""
    s = S_TAB[max(0, min(int(f), N))]
    kxi, kz, r, d = _pos_at(s)
    return kxi, kz, r, d, s >= PATH_TOTAL - 1e-9

def kernel_pos_cont(t):
    """Continuous-time kernel position (for the camera: no 1-frame steps)."""
    t = max(0.0, min(t, float(N)))
    i = int(t)
    fr = t - i
    s = S_TAB[i] * (1 - fr) + S_TAB[min(i + 1, N)] * fr
    return _pos_at(s)

def row_of(z):
    if z >= Z_START:
        return 0
    return min(NR - 1, int((Z_START - z) / KH))

# ---- camera choreography --------------------------------------------------------
# One steady move, no lateral shake: pan right (constant scale) from
# flatten's framing to the projection — red sheet left, output screen
# right — dwell there watching the beam print the first rows, then glide
# to the precomputed spot on the PREDICTION sheet where the write head
# will be when the zoom lands, follow it along the letter row (the
# cruise is one long row: the follow is a straight line, nothing whips),
# then the follow point freezes and one last zoom pulls straight out to
# the completed reconstruction — untilting to the flat, dead-on, padded
# full image as it lands. Frame 0 is exactly flatten's final camera.
ST_XI = 2.0                     # projection-stage look-at (in-plane /
ST_NU = 0.0                     # normal offsets, sheet height, and world
ST_Z = 11.5                     # lift): frames the full stack — kernel on
ST_DZ = 9.0                     # the hovering sheet, the perpendicular
                                # down-beam, and the first written lines

def set_pose(f):
    global TAU, PSI_D, PSI_N, _elev, _scale, _TARGET
    # pose in three stages: rotate to the flat DWELL tilt as the sheet
    # lifts (the stack is watched as a right prism), swing to the working
    # angle through the dive, HOLD it through the cruise, then — only
    # over the last zoom — untilt completely to the flat, dead-on view
    e1 = smoothstep(f / 60)
    e2 = smoothstep((f - F_DIVE) / (F_B - F_DIVE))
    w_e = smoothstep((f - F_TILT0) / (F_TILT1 - F_TILT0))
    tau, psi, el = (a + (d - a) * e1 + (b - d) * e2 + (c - b) * w_e
                    for a, d, b, c in zip(POSE0, POSE_D, POSE1, POSE_FLAT))
    TAU = tau
    PSI_D = (math.cos(psi), math.sin(psi))
    PSI_N = (math.sin(psi), -math.cos(psi))
    _elev = el
    # follow time: pinned at the rendezvous -> live (eased in) -> frozen
    if f <= F_B:
        fB = float(F_B)
    elif f <= F_B + 16:
        fB = F_B + (f - F_B) * smoothstep((f - F_B) / 16)
    else:
        fB = float(f)
    fB = min(fB, float(F_FREEZE))
    kxi, kz, r, d = kernel_pos_cont(fB)
    txi = kxi - d * 1.2
    # rendezvous/follow point: the WRITE HEAD on the prediction sheet
    P = tilt_pt((txi + D_XI) * PSI_D[0] + D_NU * PSI_N[0],
                (txi + D_XI) * PSI_D[1] + D_NU * PSI_N[1], kz + 0.2, TAU)
    # opening: pan right from flatten's framing to the projection stage
    ST = tilt_pt(ST_XI * PSI_D[0] + ST_NU * PSI_N[0],
                 ST_XI * PSI_D[1] + ST_NU * PSI_N[1], ST_Z, TAU)
    ST = (ST[0], ST[1], ST[2] + ST_DZ)
    w_s = smoothstep(f / F_PAN)
    tgt = tuple(a + (b - a) * w_s for a, b in zip(C0, ST))
    w_t = smoothstep((f - F_DIVE) / (F_B - F_DIVE))
    tgt = tuple(a + (b - a) * w_t for a, b in zip(tgt, P))
    # ...and at the very end settle on the flat prediction alone
    PC = tilt_pt(D_XI * PSI_D[0] + D_NU * PSI_N[0],
                 D_XI * PSI_D[1] + D_NU * PSI_N[1], (Z_TOP + Z_BOT) / 2, TAU)
    _scale = scale_at(f)
    _TARGET = tuple(a + (b - a) * w_e for a, b in zip(tgt, PC))

def draw_cube(dr, cube_xi, cube_z, fade=1.0, lift=0.0):
    a = CUBE
    corners = {}
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                xi = cube_xi + (s1 - 0.5) * a
                nu = s2 * a * 0.9
                z = cube_z + (s3 - 0.5) * a
                x = xi * PSI_D[0] + nu * PSI_N[0]
                y = xi * PSI_D[1] + nu * PSI_N[1]
                x, y, z = tilt_pt(x, y, z, TAU)
                u, v, dep, _ = project_pt(x, y, z + lift, 0.0)
                corners[(s1, s2, s3)] = (u, v, dep)
    faces = [
        [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
        [(0,0,0),(0,1,0),(0,1,1),(0,0,1)], [(1,0,0),(1,1,0),(1,1,1),(1,0,1)],
        [(0,0,0),(1,0,0),(1,1,0),(0,1,0)], [(0,0,1),(1,0,1),(1,1,1),(0,1,1)],
    ]
    faces.sort(key=lambda f_: -sum(corners[c][2] for c in f_))
    cy = (130, 225, 250)
    for f_ in faces:
        pts = [corners[c][:2] for c in f_]
        dr.polygon(pts, fill=cy + (int(70 * fade),))
    edges = set()
    for f_ in faces:
        for k in range(4):
            e = tuple(sorted((f_[k], f_[(k + 1) % 4])))
            edges.add(e)
    for a_, b_ in edges:
        u1, v1, d1 = corners[a_]
        u2, v2, d2 = corners[b_]
        al = int((255 if (d1 + d2) / 2 < 0 else 190) * fade)
        dr.line([u1, v1, u2, v2], fill=cy + (al,), width=2)

def draw_kernel(img, f, kxi, kz, krow, kd, al):
    """Kernel with a soft glow halo and a motion trail along its row —
    it must stand out and stay visible at every zoom level."""
    rz = red_z(f)
    def kpt(xi, z):
        return plane_pt(xi, z, 0.0, 0.45 * CUBE, rz)
    u, v, _, p = kpt(kxi, kz)
    cpx = CUBE * _scale * p
    # trail start: previous position, or the row entry edge after a wrap
    pxi, _, prow, _, _ = kernel_state(max(0, f - 1))
    sxi = pxi if prow == krow else (-EW if kd > 0 else EW)
    su, sv, _, _ = kpt(sxi, kz)
    dx, dy = u - su, v - sv
    L = math.hypot(dx, dy)
    if L > 120.0:
        su, sv = u - dx * 120.0 / L, v - dy * 120.0 / L
        L = 120.0
    glow = Image.new("L", (S, S), 0)
    dg = ImageDraw.Draw(glow)
    rg = max(cpx * 1.6, 12.0)
    if L > 1.5:
        dg.line([su, sv, u, v], fill=int(85 * al), width=max(int(rg * 0.9), 5))
    dg.ellipse([u - rg, v - rg, u + rg, v + rg], fill=int(150 * al))
    glow = glow.filter(ImageFilter.GaussianBlur(6))
    img = Image.composite(Image.new("RGB", (S, S), (140, 235, 255)), img, glow)
    dr = ImageDraw.Draw(img, "RGBA")
    if L > 1.5:
        dr.line([su, sv, u, v], fill=(150, 235, 255, int(140 * al)),
                width=max(2, int(cpx * 0.4)))
    draw_cube(dr, kxi, kz, al, rz)
    return img

def draw_write_head(img, kxi, kz, al):
    """The output cursor: a small bright square riding the raster front of
    the prediction sheet, in sync with the kernel — it prints what the
    kernel reads."""
    u, v, _, p = plane_pt(kxi, kz, D_XI, D_NU)
    rg = max(CUBE * _scale * p * 1.1, 8.0)
    glow = Image.new("L", (S, S), 0)
    ImageDraw.Draw(glow).ellipse([u - rg, v - rg, u + rg, v + rg],
                                 fill=int(110 * al))
    glow = glow.filter(ImageFilter.GaussianBlur(4))
    img = Image.composite(Image.new("RGB", (S, S), (140, 235, 255)), img, glow)
    dr = ImageDraw.Draw(img, "RGBA")
    pts = [plane_pt(kxi + sx * CUBE / 2, kz + sz * CUBE / 2, D_XI, D_NU)[:2]
           for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    dr.polygon(pts, fill=(130, 225, 250, int(60 * al)),
               outline=(150, 235, 255, int(230 * al)), width=2)
    return img

def draw_beam(img, f, kxi, kz, al):
    """The projection: the kernel's reading beamed DOWN across the gap onto
    the output sheet — a soft tapered ray from the kernel to the write
    head, right below it."""
    ku, kv, _, kp = plane_pt(kxi, kz, 0.0, 0.45 * CUBE, red_z(f))
    hu, hv, _, hp = plane_pt(kxi, kz, D_XI, D_NU)
    glow = Image.new("L", (S, S), 0)
    ImageDraw.Draw(glow).line([ku, kv, hu, hv], fill=int(55 * al), width=7)
    glow = glow.filter(ImageFilter.GaussianBlur(4))
    img = Image.composite(Image.new("RGB", (S, S), (140, 235, 255)), img, glow)
    dr = ImageDraw.Draw(img, "RGBA")
    dxu, dyv = hu - ku, hv - kv
    L = math.hypot(dxu, dyv)
    if L > 1.0:
        nx, ny = -dyv / L, dxu / L
        wk = max(CUBE * _scale * kp * 0.45, 1.5)
        wh = max(CUBE * _scale * hp * 0.55, 2.5)
        dr.polygon([(ku + nx * wk, kv + ny * wk), (hu + nx * wh, hv + ny * wh),
                    (hu - nx * wh, hv - ny * wh), (ku - nx * wk, kv - ny * wk)],
                   fill=(140, 235, 255, int(44 * al)))
        dr.line([ku, kv, hu, hv], fill=(150, 235, 255, int(105 * al)), width=2)
    return img

def render_frame(f, out):
    global INK_TEX, SHEET_MASK
    if INK_TEX is None:
        INK_TEX = build_ink_tex()
        SHEET_MASK = build_sheet_mask()
    set_pose(f)
    kxi, kz, krow, kd, kdone = kernel_state(f)
    img = compose_scene()
    dr = ImageDraw.Draw(img, "RGBA")

    # ghost-mesh edge set shared by both sheets
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

    def draw_mesh(dr_, dxi, nu, alpha, dz=0.0):
        V = {}
        for i in range(N_TH):
            for j in range(N_Z):
                th, z = mesh_pt(i, j)
                V[(i, j)] = plane_pt(arc_len(th) - S_TOTAL / 2, z, dxi, nu, dz)
        for a_, b_ in edges:
            dr_.line([V[a_][0], V[a_][1], V[b_][0], V[b_][1]],
                     fill=MESH + (alpha,), width=1)

    # output sheet (right, behind, a step lower): a faint ghost mesh marks
    # the empty stage — fading in with the pull-out so frame 0 stays
    # pixel-identical to flatten's final frame — then the prediction
    # texture rasters onto it; alpha = torn-sheet boundary * raster mask
    # (completed rows above + a partial current row, exactly kernel-
    # height). Below the raster front there is nothing but background.
    mesh_al = int(52 * smoothstep(f / 12))
    if mesh_al > 0:
        draw_mesh(dr, D_XI, D_NU, mesh_al)
    rev = Image.new("L", (TEX_W, TEX_H), 0)
    dv = ImageDraw.Draw(rev)
    if kdone:
        dv.rectangle([0, 0, TEX_W, TEX_H], fill=255)
    else:
        v0 = tex_uv(0, Z_START - krow * KH)[1]
        v1 = tex_uv(0, Z_START - (krow + 1) * KH)[1]
        if krow > 0:
            dv.rectangle([0, 0, TEX_W, v0], fill=255)
        trail = kxi - kd * CUBE / 2
        u_t = min(max(tex_uv(trail, 0)[0], 0), TEX_W)
        if kd > 0 and u_t > 0:
            dv.rectangle([0, v0, u_t, v1], fill=255)
        elif kd < 0 and u_t < TEX_W:
            dv.rectangle([u_t, v0, TEX_W, v1], fill=255)
    rev = rev.filter(ImageFilter.GaussianBlur(2))
    alpha = np.minimum(np.asarray(SHEET_MASK, np.uint16),
                       np.asarray(rev, np.uint16)).astype(np.uint8)
    tex = INK_TEX.copy()
    tex.putalpha(Image.fromarray(alpha))
    coeffs = _perspective_coeffs(D_XI, D_NU)
    warped = tex.transform((S, S), Image.PERSPECTIVE, tuple(coeffs),
                           Image.BILINEAR)
    img = Image.alpha_composite(img.convert("RGBA"), warped).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")

    # input sheet (lifted overhead): ghost mesh + red voxels, intact for
    # the ENTIRE scan (the kernel reads them, nothing is consumed); the
    # scan front just glows warmer. Over the last zoom the input is
    # DISMISSED: it rises further away while fading, like a scanner
    # retracting — the flat final view is the reconstruction alone
    red_al = 1.0 - smoothstep((f - 240) / 32)
    rz = red_z(f)
    if red_al > 0.003:
        draw_mesh(dr, 0.0, 0.0, int(148 * red_al), rz)
    cubes = []
    for xi, lk, zlist in (_voxsheet_cols() if red_al > 0.003 else []):
        for vz in zlist:
            zc = vz + RVOX / 2
            em = 1.0
            if not kdone and row_of(zc) == krow and abs(xi - (kxi - kd * CUBE / 2)) < 0.9:
                em = 1.18
            x = xi * PSI_D[0]
            y = xi * PSI_D[1]
            x, y, z = tilt_pt(x, y, zc, TAU)
            cubes.append((x, y, z + rz, lk, em))
    cubes.sort(key=lambda cb: -project_pt(cb[0], cb[1], cb[2], 0.0)[2])
    v_al = int(255 * red_al)
    for x, y, z, lk, em in cubes:
        u_, v_, _, p = project_pt(x, y, z, 0.0)
        h = (RVOX * _scale * 0.5) * p
        top = [(u_ - h, v_ - h * 1.55), (u_ + h, v_ - h * 1.55), (u_ + h, v_ - h * 0.55), (u_ - h, v_ - h * 0.55)]
        left = [(u_ - h, v_ - h * 0.55), (u_, v_ - h * 0.05), (u_, v_ + h * 1.05), (u_ - h, v_ + h * 0.5)]
        right = [(u_, v_ - h * 0.05), (u_ + h, v_ - h * 0.55), (u_ + h, v_ + h * 0.5), (u_, v_ + h * 1.05)]
        dr.polygon(top, fill=shade(RED, 1.22 * em) + (v_al,))
        dr.polygon(left, fill=shade(RED, (0.76 + lk * 0.10) * em) + (v_al,))
        dr.polygon(right, fill=shade(RED, (0.94 + lk * 0.10) * em) + (v_al,))

    # the projection ray + both cursors work in view for the ENTIRE scan
    # (glow + trail keep them legible even shrunk to sparks); they only
    # dim out over the last few frames as the final rows land
    if not kdone:
        # the cursors dim out before the red sheet starts its slide-off
        # (the kernel rides the sheet — it must not be seen detaching)
        k_al = (smoothstep(f / 8) * clamp01((F_DONE - f) / 6)
                * (1 - smoothstep((f - 236) / 16)))
        if k_al > 0.01:
            img = draw_beam(img, f, kxi, kz, k_al)
            img = draw_kernel(img, f, kxi, kz, krow, kd, k_al)
            img = draw_write_head(img, kxi, kz, k_al)

    img = img.resize((300, 300), Image.LANCZOS)
    img.save(out)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        for f in range(N):
            render_frame(f, f"{SCRATCH}/inkloop/f{f:03d}.png")
            print(f"\r{f+1}/{N}", end="", flush=True)
        print()
    else:
        f = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        render_frame(f, f"{SCRATCH}/ink-proto.png")
        print("wrote ink-proto.png")
