#!/usr/bin/env python3
"""Spiral-fit (fit_spiral) GPU-memory model and tables, see ../POOL_OFFLOAD.md.

peak_GiB(H, shell, r) = pool(H) + tracks(H) + flow_r * (H + 320)/1000 + N_GAP * gap(H, shell) + act_r * H/1000
  H      fit height in L0 slices (z_end - z_begin);  shell = shell_outer_winding_idx;  cap = shell + 14 (gap-expander capacity, arm142.sh)
  pool   lasagna normals brick pool, uint8, resident on the card (scroll-specific; measured or from the store's brick count x 64 KiB)
  tracks device track store (cached points x 20 B x 1.3 for per-track arrays; 343 full height measured 3.66 GiB)
  flow_r flow-field family per 1000 slices incl. the 2 x 160-slice z margin: param + Adam m, v + gradient accumulator + LR-upsampled copy
         (measured: r=16 0.555 GiB, r=32 0.0698 GiB; r=8 = 8 x r16, extrapolated); flow radius = model_flow_bounds_radius 3200 vox (default)
  gap    one copy of the gap-expander logit lattice: 4 B x (H+320)/24 x sum_k round(2 pi (k+0.5) 16/24 + 0.5) over k < cap  (~2.09 cap^2)
  N_GAP  live copies WITH the periodic garbage-collection fix (measured 5-7 at z 16,300, shell 250; 11-17 at z 1,500); WITHOUT it the
         copies accumulate ~2 per iteration until Python's cyclic GC runs (up to ~200 measured), which is what OOMed every long fit
  act_r  transient per-step tensors (peak - live census) per 1000 slices (sample counts scale with H): measured 0.213 GiB (r=32, full 343)
All coefficients are disclosed parameters (D33) with their measurement in ../POOL_OFFLOAD.md.  usage: rb_mem_table.py [--json OUT]"""
import json, math, sys

FLOW = {8: 8 * 0.555, 16: 0.555, 32: 0.0698}
ACT = {8: 0.229, 16: 0.229, 32: 0.213}          # GiB per 1000 slices; measured on the full-height 343 probes (peak - live census): flow16 0.229, flow32 0.213
N_GAP = 7
HEADROOM = 1.5                                  # GiB: CUDA context + allocator fragmentation (nvidia-smi peak - torch max = 0.9 GiB measured)
CARDS = [16, 24, 32, 40, 80, 96, 144]


def gap_copy_gib(H, shell):
    cap = shell + 14
    w = sum(int(2 * math.pi * (k + 0.5) * 16 / 24 + 0.5) for k in range(1, cap))
    return 4 * (H + 320) / 24 * w / 2 ** 30


def peak(H, shell, r, pool, tracks, n_gap=N_GAP):
    return pool + tracks + FLOW[r] * (H + 320) / 1000 + n_gap * gap_copy_gib(H, shell) + ACT[r] * H / 1000


def usable(card):
    return card * 0.99 - HEADROOM


def max_h(card, shell, r, pool_per_k, tracks_per_k):
    lo, hi = 0, 60000
    while hi - lo > 50:
        m = (lo + hi) // 2
        if peak(m, shell, r, pool_per_k * m / 1000, tracks_per_k * m / 1000) <= usable(card): lo = m
        else: hi = m
    return lo


# per-scroll inputs: full height H (L0 slices of the upstream surface prediction), shell (used / box default 200 = placeholder), pool GiB for the
# full height (142 respool store if present, else the box's packed brick count for z 4500-17500 scaled by H/13000), cached track points for
# z 4500-17500 (box full-fit log) scaled to H
GiB = 2 ** 30
SCROLLS = {
    # name: (H, shell, shell_source, pool_full_GiB, pool_source, cached_points_13000, it_per_s_box48)
    "PHerc0175A": (12748, 200, "box default (not measured)", 9.79, "142 respool (measured file size)", 108880491, 3.9),
    "PHerc0175B": (15897, 200, "box default (not measured)", 16.69, "142 respool (measured file size)", 128981872, 6.0),
    "PHerc0191": (18977, 146, "fits", 11.01, "142 respool", 117483137, 8.1),
    "PHerc0257": (18872, 214, "fits", 7.70, "142 respool", 77441982, 7.2),
    "PHerc0268": (14833, 408, "fits", 21.43, "142 respool", 159525449, 4.6),
    "PHerc0306B": (15898, 200, "box default (not measured)", 162412 * 65536 / GiB * 15898 / 13000, "box packed x H/13000", 110838913, 5.3),
    "PHerc0343": (17998, 250, "fits", 10.81, "142 respool (9.58 loaded for z 1200-17500)", None, None),
    "PHerc0358": (14744, 252, "fits", 7.09, "142 respool", 70785547, 5.6),
    "PHerc0483A": (15898, 200, "box default (not measured)", 10.13, "142 respool (measured file size)", 90273138, 6.0),
    "PHerc0483B": (11880, 200, "box default (not measured)", 8.16, "142 respool (measured file size)", 72811322, 7.2),
    "PHerc0490A": (11698, 250, "fits (sense unknown)", 0.29, "142 respool (measured file size)", None, None),
    "PHerc0490B": (10648, 250, "fits (sense unknown)", 0.21, "142 respool (measured file size)", 47716502, 7.7),
    "PHerc0800": (24298, 275, "fits", 19.80, "142 respool", 155951965, 5.0),
    "PHerc0813": (16993, 197, "fits", 8.15, "142 respool", 98716648, 6.6),
    "PHerc0826": (16920, 217, "fits", 6.32, "142 respool", 64004220, 5.3),
    "PHerc0846A": (14019, 250, "fits (sense unknown)", 0.18, "142 respool (measured file size)", None, None),
    "PHerc0846B": (13926, None, "none (no fit yet)", 7.53, "142 respool (measured file size)", None, None),
    "PHerc1203": (18977, 187, "queue", 7.75, "142 respool (measured file size)", None, None),
    "PHerc1218": (23247, 195, "queue (box used 200)", 7.98, "142 respool (measured file size)", 72842567, 6.5),
    "PHerc1447": (24297, None, "none (no fit yet)", None, "no store measured", None, None),
    "PHerc1545": (20961, 204, "queue (box used 200)", 7.09, "142 respool (measured file size)", 75676636, 6.0),
}
DEFAULT_POOL_PER_K = 9.58 / 16.3            # 343 measured, GiB per 1000 slices (used where no store was measured; announced)
DEFAULT_TRACK_PER_K = 3.66 / 16.3
DEFAULT_SHELL = 250


def scroll_rows():
    rows = []
    for s, (H, shell, ssrc, pool, psrc, pts, ips) in SCROLLS.items():
        notes = []
        if shell is None: shell = DEFAULT_SHELL; notes.append(f"shell {DEFAULT_SHELL} assumed")
        if pool is None: pool = DEFAULT_POOL_PER_K * H / 1000; notes.append("pool from 343 density")
        if s == "PHerc0343": tracks = 3.66 * H / 16300
        elif pts: tracks = pts * H / 13000 * 20 * 1.3 / GiB
        else: tracks = DEFAULT_TRACK_PER_K * H / 1000; notes.append("tracks from 343 density")
        g16, g32, g8 = (peak(H, shell, r, pool, tracks) for r in (16, 32, 8))
        ok = lambda g: next((c for c in CARDS if usable(c) >= g), None)
        rows.append(dict(scroll=s, height=H, shell=shell, shell_source=ssrc, pool_gib=round(pool, 2), pool_source=psrc, tracks_gib=round(tracks, 2),
                         min_gib_flow16=round(g16 + HEADROOM, 1), min_gib_flow32=round(g32 + HEADROOM, 1), min_gib_flow8=round(g8 + HEADROOM, 1),
                         card_flow16=ok(g16), card_flow32=ok(g32), notes="; ".join(notes)))
    return rows


def maxz_rows(shells=(100, 120, 150, 200, 250, 300, 400)):
    out = []
    for r in (16, 32):
        for sh in shells:
            out.append(dict(flow=r, shell=sh, **{f"{c}GB": max_h(c, sh, r, DEFAULT_POOL_PER_K, DEFAULT_TRACK_PER_K) for c in CARDS}))
    return out


if __name__ == "__main__":
    res = dict(coefficients=dict(flow=FLOW, act=ACT, n_gap=N_GAP, headroom=HEADROOM, pool_per_k_343=DEFAULT_POOL_PER_K, tracks_per_k_343=DEFAULT_TRACK_PER_K),
               scrolls=scroll_rows(), maxz=maxz_rows())
    if "--json" in sys.argv: json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    print("| scroll | height (slices) | shell | pool GiB | tracks GiB | min GiB, 16-vox flow | min GiB, 32-vox flow | smallest card, 16 | smallest card, 32 | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["scrolls"]:
        print(f"| {r['scroll']} | {r['height']:,} | {r['shell']} ({r['shell_source']}) | {r['pool_gib']} | {r['tracks_gib']} | {r['min_gib_flow16']} | {r['min_gib_flow32']} | {r['card_flow16'] or '>144'} GB | {r['card_flow32'] or '>144'} GB | {r['notes']} |")
    print()
    print("| flow grid | shell | " + " | ".join(f"{c} GB" for c in CARDS) + " |"); print("|---|---|" + "---|" * len(CARDS))
    for r in res["maxz"]:
        print(f"| {r['flow']} vox | {r['shell']} | " + " | ".join((f"{r[f'{c}GB']:,}" if r[f'{c}GB'] < 59000 else "> 60,000") for c in CARDS) + " |")
