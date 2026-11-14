#!/usr/bin/env python3
"""VRAM per Route B spiral fit, per scroll, per flow-voxel grid, with GC and GC + pool offload (user 2026-11-13).
Model = rb_mem_table.py (this folder) (calibrated on PHerc0343 full height: 20.5 / 28.8 GiB predicted vs 20.1 / 28.2 measured, 32- / 16-voxel grid).
Measured diamonds = measured.json (+ proof.json when the 16-voxel proof runs have a sampler CSV).
usage: make_vram_chart.py OUTDIR   -> vram_chart.png, vram_table.csv, vram_table.md"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rb_mem_table as M
PUBLIC = True; ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]; OUT = ARGS[0] if ARGS else HERE
CARD = 31.73            # V100-SXM2-32GB, nvidia-smi total GiB
CARD3090 = 23.65
rows = [r for r in M.scroll_rows() ]; meas = json.load(open(os.path.join(HERE, "measured.json")))["points"]
if os.path.exists(os.path.join(HERE, "proof.json")): meas += json.load(open(os.path.join(HERE, "proof.json")))["points"]

def parts(r, g):
    H, sh, pool, tr = r["height"], r["shell"], r["pool_gib"], r["tracks_gib"]
    gap = M.N_GAP * M.gap_copy_gib(H, sh); flow = M.FLOW[g] * (H + 320) / 1000; act = M.ACT[g] * H / 1000
    base = M.peak(H, sh, g, pool, tr, n_gap=0)
    it = max(0.0, ((CARD - 0.9 - base) / M.gap_copy_gib(H, sh) - 7) / 2)        # 7 live copies + ~2 leaked per iteration (measured up to about 200 alive at 150-300 its)
    return dict(tracks=tr, flow=flow, act=act, gap=gap, pool=pool, head=M.HEADROOM, nogc_it=it)

order = sorted(rows, key=lambda r: -sum(parts(r, 16)[k] for k in ("tracks", "flow", "act", "gap", "pool")))
C = dict(tracks="#264653", flow="#2a9d8f", act="#8ab17d", gap="#e9c46a", head="#c9c4bb", pool="#e76f51", leak="#b5179e")
BG, FG, GR = "#faf7f2", "#232323", "#8a857c"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.edgecolor": GR, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG})
fig = plt.figure(figsize=(21, 18), dpi=150, facecolor=BG)
gs = fig.add_gridspec(1, 2, left=0.158, right=0.99, top=0.80, bottom=0.145, wspace=0.10)
fig.text(0.015, 0.968, "How much GPU memory does one whole-height Route B spiral fit need?", fontsize=26, fontweight="bold", va="top")
fig.text(0.015, 0.935, "Per scroll and per flow-voxel grid, as predicted by the calibrated memory model: what the Python garbage-collection hook saves, and what moving the lasagna brick pool\n"
         "to pinned host RAM saves on top. Dashed red line = the 31.7 GiB of one V100-32; dotted = a 23.6 GiB RTX 3090. Diamonds = nvidia-smi measurements on real fits.", fontsize=13.5, va="top", color="#4a463f", linespacing=1.5)
res_rows = []
for gi, g in enumerate((32, 16)):
    ax = fig.add_subplot(gs[0, gi]); ax.set_facecolor(BG); n = len(order)
    for yi, r in enumerate(order):
        p = parts(r, g); y = n - 1 - yi; x = 0.0; h = 0.62
        for k in ("tracks", "flow", "act", "gap", "head"):
            ax.barh(y, p[k], left=x, height=h, color=C[k], edgecolor=BG, linewidth=0.6, zorder=3); x += p[k]
        off_total = x
        ax.barh(y, p["pool"], left=x, height=h, color=C["pool"], edgecolor=BG, linewidth=0.6, zorder=3, hatch="////", alpha=0.95); x += p["pool"]
        gc_total = x
        if p["nogc_it"] < 1e9:                                                  # leak: unbounded growth past the GC total, until OOM
            ax.annotate("", xy=(gc_total + 6.6, y), xytext=(gc_total + 3.0, y), arrowprops=dict(arrowstyle="-|>", color=C["leak"], lw=1.6, ls=(0, (2, 1.5))), zorder=4)
        fits_gc, fits_off = gc_total <= CARD, off_total <= CARD
        ax.text(61.5, y, ("✓" if fits_gc else "✗"), color=("#2d7d46" if fits_gc else "#c1121f"), fontsize=15, fontweight="bold", va="center", ha="center", clip_on=False)
        ax.text(67.5, y, ("✓" if fits_off else "✗"), color=("#2d7d46" if fits_off else "#c1121f"), fontsize=15, fontweight="bold", va="center", ha="center", clip_on=False)
        ax.text(off_total + 0.35, y + 0.0, f"{off_total:.1f}", fontsize=9.2, va="center", ha="left", color="#ffffff" if False else FG, zorder=5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.85)) if p["pool"] > 3.2 else None
        ax.text(gc_total + 0.35, y, f"{gc_total:.1f}", fontsize=9.2, va="center", ha="left", color=C["pool"], zorder=5, fontweight="bold") if p["pool"] > 3.2 else None
        for m in meas:
            if m["scroll"] == r["scroll"] and m["grid"] == g:
                shift = -0.19 if m["setting"] == "gc" else 0.19
                ax.plot([m["steady"]], [y + shift * 0.0 - 0.34], marker="D", ms=8.5, mfc="#ffffff", mec=FG, mew=1.4, zorder=6, ls="none")
                ax.text(m["steady"] - 0.5, y - 0.34, "GC" if m["setting"] == "gc" else "offload", fontsize=7.8, ha="right", va="center", color=FG, zorder=7)
                if m["max"] > m["steady"] + 0.2: ax.plot([m["max"]], [y - 0.34], marker="D", ms=8.5, mfc=FG, mec="#ffffff", mew=1.0, zorder=6, ls="none"); ax.plot([m["steady"], m["max"]], [y - 0.34] * 2, color=FG, lw=1.1, zorder=5)
        res_rows.append(dict(scroll=r["scroll"], grid=g, height=r["height"], shell=r["shell"], pool=p["pool"], tracks=p["tracks"], flow=p["flow"], act=p["act"], gap7=p["gap"],
                             nogc_oom_iter=p["nogc_it"], gc_need=gc_total, off_need=off_total, saved_by_gc="unbounded leak", saved_by_offload=p["pool"], fits_gc=fits_gc, fits_off=fits_off))
    ax.axvline(CARD, color="#c1121f", lw=2.0, ls=(0, (6, 3)), zorder=2); ax.axvline(CARD3090, color="#c1121f", lw=1.2, ls=(0, (1, 2.5)), zorder=2, alpha=0.7)
    ax.text(CARD + 0.3, n - 0.45, "V100-32\n31.7 GiB", color="#c1121f", fontsize=10.5, va="bottom", ha="left", fontweight="bold", linespacing=1.1)
    ax.text(CARD3090 - 0.3, n - 0.45, "RTX 3090\n23.6 GiB", color="#c1121f", fontsize=10, va="bottom", ha="right", alpha=0.8, linespacing=1.1)
    ax.set_xlim(0, 70.5); ax.set_ylim(-0.7, n - 0.1); ax.set_yticks(range(n)); ax.invert_yaxis() if False else None
    ax.set_yticklabels([f"{r['scroll'].replace('PHerc', 'PHerc ')}   {r['height'] / 1000:.1f}k slices · shell {r['shell']}" for r in order][::-1] if gi == 0 else [], fontsize=11)
    ax.tick_params(axis="y", length=0); ax.set_xticks(range(0, 60, 10)); ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.grid(True, color="#e3ded4", lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_xlabel("GPU memory (GiB), whole-height fit", fontsize=12)
    ax.set_title(f"{g}-voxel flow grid" + ("   (the coarse grid: smaller, less accurate)" if g == 32 else "   (the fine grid: better fit, much more memory)"), fontsize=16, fontweight="bold", loc="left", pad=48)
    n32 = sum(1 for q in res_rows if q["grid"] == g and q["fits_gc"]); n_off = sum(1 for q in res_rows if q["grid"] == g and q["fits_off"])
    ax.text(0, n + 1.9, f"fits one V100-32:  with GC {n32} of {n}   ·   with GC + offload {n_off} of {n}", fontsize=12.5, color="#4a463f", va="bottom", clip_on=False, transform=ax.transData)
    ax.text(61.5, n - 0.1 + 0.0, "GC", fontsize=10.5, ha="center", va="bottom", fontweight="bold", clip_on=False); ax.text(67.5, n - 0.1, "GC+\noffload", fontsize=9.5, ha="center", va="bottom", fontweight="bold", clip_on=False, linespacing=0.95)
leg = [Patch(fc=C["tracks"], label="track store"), Patch(fc=C["flow"], label="flow field + Adam"), Patch(fc=C["act"], label="per-step temporaries"), Patch(fc=C["gap"], label="7 live gap-logit copies (with GC)"),
       Patch(fc=C["head"], label="CUDA context + fragmentation (1.5 GiB)"), Patch(fc=C["pool"], hatch="////", ec=BG, label="lasagna brick pool = what OFFLOAD moves to host RAM"),
       Line2D([], [], color=C["leak"], lw=1.8, ls=(0, (2, 1.5)), marker=">", ms=7, label="without the GC hook: copies leak every step until OOM"),
       Line2D([], [], marker="D", mfc="#fff", mec=FG, ls="none", ms=8.5, label="measured, optimiser phase (median)"), Line2D([], [], marker="D", mfc=FG, mec="#fff", ls="none", ms=8.5, label="measured, whole-run max (incl. mesh export)")]
fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.012, 0.045), ncol=3, fontsize=11.2, frameon=False, columnspacing=2.2, handlelength=2.2)
fig.text(0.012, 0.008, "Bar end without hatch = need with GC + offload; hatched end = need with GC only (pool resident on the card). Model peak = torch-allocated + 1.5 GiB headroom.\n"
         "Model: pool_offload/rb_mem_table.py, calibrated on PHerc0343 (±2 %). Measured: V100-SXM2-32GB, 2 s nvidia-smi samples (measured.json).",
         fontsize=9.6, ha="left", va="bottom", color="#6b665d", linespacing=1.5)
os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "vram_chart_public.png" if PUBLIC else "vram_chart.png"), facecolor=BG); print("wrote", os.path.join(OUT, "vram_chart_public.png" if PUBLIC else "vram_chart.png"))
with open(os.path.join(OUT, "vram_table_public.csv" if PUBLIC else "vram_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res_rows[0].keys())); w.writeheader(); [w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()}) for q in res_rows]
by = {(q["scroll"], q["grid"]): q for q in res_rows}
with open(os.path.join(OUT, "vram_table_public.md" if PUBLIC else "vram_table.md"), "w") as f:
    f.write("| scroll | slices | shell | pool GiB | 32-vox: no GC (OOM at iteration, model) | 32-vox GC GiB | 32-vox GC+offload GiB | 16-vox: no GC (OOM at iteration, model) | 16-vox GC GiB | 16-vox GC+offload GiB | fits V100-32 (32-vox: GC / GC+off; 16-vox: GC / GC+off) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    mk = lambda b: "✓" if b else "✗"
    for r in sorted(rows, key=lambda r: r["scroll"]):
        a, b = by[(r["scroll"], 32)], by[(r["scroll"], 16)]
        f.write(f"| {r['scroll']} | {r['height']:,} | {r['shell']} | {a['pool']:.1f} | {'never starts' if a['nogc_oom_iter'] < 1 else f'~{a['nogc_oom_iter']:.0f}'} | {a['gc_need']:.1f} | {a['off_need']:.1f} | {'never starts' if b['nogc_oom_iter'] < 1 else f'~{b['nogc_oom_iter']:.0f}'} | {b['gc_need']:.1f} | {b['off_need']:.1f} | {mk(a['fits_gc'])} / {mk(a['fits_off'])}; {mk(b['fits_gc'])} / {mk(b['fits_off'])} |\n")
print("wrote table")
