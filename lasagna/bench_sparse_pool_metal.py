"""Per-call time of the pooled sparse sampler: Metal kernels vs torch ops on MPS vs torch ops on CPU.

    python bench_sparse_pool_metal.py [--sizes 2500,20000,100000,1000000] [--slots 3000,20000]
        [--pattern surface|random] [--rounds 3] [--out results.json]

Every (variant, size, slots) measurement runs in a fresh process; the driver alternates the
variants' order from round to round and reports the median over rounds of each process's p50.
Inside a process each function gets warm-up calls, then two timings:
- p50_ms: one call between two torch.mps.synchronize() calls (time per call including the wait for
  the GPU; the latency a call adds when its result is needed at once);
- queued_ms: 10 calls queued, then one synchronize, divided by 10 (what a call costs inside a step
  that syncs once, as the fit's forward and backward do). missing_chunks syncs inside (its caller needs
  the rows), and the CPU has no queue, so for those queued_ms equals p50_ms.

Variants: each one times what PooledChunkGroupCache runs on that backend (the label is what runs;
a child asserts it).
- metal: sparse_pool_metal on MPS, with LASAGNA_METAL=1 so that any fallback raises. missing_chunks
  includes the result's .cpu(), which the cache's prefetch does next.
- mps_ops: the cache's torch-ops backend on MPS: the samplers on MPS tensors; missing_chunks on the
  host (the points' .cpu(), then sparse_pool_ops on the CPU with the host copy of the table).
- cpu_ops: sparse_pool_ops on CPU tensors, torch.set_num_threads(--cpu-threads, default 4).

Data (synthetic, seeded): the PHerc0826 cos level's chunk table (265, 128, 128), C = 1,
offset 0, inv_scale 0.5 (spacing 2); `slots` chunks loaded, as a compact block packed in shuffled
slot order; N sample points of which 80 % lie in loaded chunks.
- surface: a gently tilted, wavy sheet sampled on a (1, H, W) grid, as a fit's mesh is; neighbouring
  samples are neighbours in space. The sheet runs 25 % past the loaded block along x.
- random: points uniform in the loaded block (80 %) and in the unloaded chunks around it (20 %).
missing_chunks gets the same points (origin 0, spacing 2).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_SHAPE = (265, 128, 128)
SPACING = 2.0
FUNCTIONS = ("sample_u8", "sample_f32", "sample_grad", "missing_chunks")
VARIANTS = ("metal", "mps_ops", "cpu_ops")


def _build(n, slots, pattern, seed=0):
    """CPU tensors: pool, table, grid (1, H, W, 3) full-res xyz, grad_output, and the loaded fraction."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    cZ, cY, cX = TABLE_SHAPE
    b = math.ceil(slots ** (1.0 / 3.0))
    bz = math.ceil(slots / (b * b))
    z0, y0, x0 = (cZ - bz) // 2, (cY - b) // 2, (cX - b) // 2
    block = [(z0 + i // (b * b), y0 + (i // b) % b, x0 + i % b) for i in range(slots)]
    table = np.full(TABLE_SHAPE, -1, np.int32)
    for slot, i in enumerate(rng.permutation(slots)):
        table[block[i]] = slot
    # Random bytes, 64 distinct bricks tiled over the slots (cheap to build; the addresses still differ).
    bricks = rng.integers(0, 256, (64, 1, 34, 34, 34), dtype=np.uint8)
    pool = torch.from_numpy(np.ascontiguousarray(np.resize(bricks, (slots, 1, 34, 34, 34))))
    H = max(1, int(math.sqrt(n)))
    W = max(1, n // H)
    lo = np.array([x0, y0, z0], np.float64) * 32.0  # level voxels (x, y, z)
    full_z = bz if slots % (b * b) == 0 else bz - 1  # complete layers of the block
    if pattern == "surface":
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        uu, vv = np.meshgrid(u, v)
        x = lo[0] + uu * b * 32.0 * 1.25  # 80 % of the sheet over the block
        y = lo[1] + vv * b * 32.0
        z = lo[2] + max(full_z, 1) * 16.0 + 0.1 * (x - lo[0]) + 3.0 * np.sin(y / 17.0)
        local = np.stack([x, y, z], -1).reshape(-1, 3)
    else:
        m = H * W
        inside = rng.random(m) < 0.8
        size = np.array([b, b, max(full_z, 1)], np.float64) * 32.0
        local = lo + rng.random((m, 3)) * size
        out = lo - 32.0 + rng.random((m, 3)) * (size + 64.0)
        shell = ~inside
        local[shell] = out[shell]
    grid = torch.from_numpy((local * SPACING).astype(np.float32).reshape(1, H, W, 3))
    go = torch.from_numpy(rng.standard_normal((1, 1, H, W)).astype(np.float32))
    p = (grid.reshape(-1, 3).double() / SPACING).numpy()
    ci = np.floor(p / 32.0).astype(np.int64)
    ok = (ci >= 0).all(1) & (ci[:, 0] < cX) & (ci[:, 1] < cY) & (ci[:, 2] < cZ)
    loaded = np.zeros(len(p), bool)
    loaded[ok] = table[ci[ok, 2], ci[ok, 1], ci[ok, 0]] >= 0
    return pool, torch.from_numpy(table), grid, go, float(loaded.mean())


def _child(variant, n, slots, pattern, target_s, cpu_threads):
    import torch

    sys.path.insert(0, HERE)
    if variant == "metal":
        os.environ["LASAGNA_METAL"] = "1"  # a fallback raises instead of timing the torch ops
        import sparse_pool_metal as module
        module.compile()
    else:
        import sparse_pool_ops as module
    if variant != "metal":
        torch.set_num_threads(cpu_threads)  # mps_ops' prefetch runs on the host too
    device = torch.device("cpu" if variant == "cpu_ops" else "mps")
    pool, table, grid, go, loaded = _build(n, slots, pattern)
    table_host = table
    pool, table, grid, go = (t.to(device) for t in (pool, table, grid, go))  # (the host pool is freed)
    offset = torch.zeros(3, device=device)
    inv_scale = torch.full((3,), 1.0 / SPACING, device=device)
    origin = torch.zeros(3, device=device)
    spacing = torch.full((3,), SPACING, device=device)
    calls = {
        "sample_u8": lambda: module.sample_u8(pool, table, grid, offset, inv_scale),
        "sample_f32": lambda: module.sample_f32(pool, table, grid, offset, inv_scale),
        "sample_grad": lambda: module.sample_grad(pool, table, grid, offset, inv_scale, go),
        "missing_chunks": lambda: module.missing_chunks(grid, table, origin, spacing).cpu(),
    }
    if variant == "mps_ops":  # sparse_pool_cache: torch-ops prefetch runs on the host
        origin_host, spacing_host = origin.cpu(), spacing.cpu()
        calls["missing_chunks"] = lambda: module.missing_chunks(grid.detach().cpu(), table_host, origin_host,
                                                                spacing_host)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    result = {"variant": variant, "n": int(grid.shape[1] * grid.shape[2]), "slots": slots, "pattern": pattern,
              "loaded_frac": round(loaded, 4), "module": module.__name__, "device": device.type,
              "contraction": getattr(module, "CONTRACTION", None), "torch": torch.__version__,
              "threads": torch.get_num_threads() if variant != "metal" else None, "p50_ms": {}, "queued_ms": {},
              "reps": {}}
    for name in FUNCTIONS:
        fn = calls[name]
        for _ in range(3):
            fn()
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        first = time.perf_counter() - t0
        reps = int(min(60, max(5, target_s / max(first, 1e-6))))
        times = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
        result["p50_ms"][name] = statistics.median(times)
        result["reps"][name] = reps
        if device.type == "mps" and name != "missing_chunks":
            queued = []
            for _ in range(max(3, reps // 10)):
                sync()
                t0 = time.perf_counter()
                for _ in range(10):
                    fn()
                sync()
                queued.append((time.perf_counter() - t0) * 1e3 / 10)
            result["queued_ms"][name] = statistics.median(queued)
        else:
            result["queued_ms"][name] = result["p50_ms"][name]
    print("RESULT " + json.dumps(result), flush=True)


def _driver(args):
    sizes = [int(float(v)) for v in args.sizes.split(",")]
    slot_counts = [int(v) for v in args.slots.split(",")]
    rows = []
    for r in range(args.rounds):
        order = VARIANTS[r % 3:] + VARIANTS[:r % 3]  # alternate which variant goes first
        for slots in slot_counts:
            for n in sizes:
                for variant in order:
                    cmd = [sys.executable, os.path.abspath(__file__), "--child", variant, str(n), str(slots),
                           "--pattern", args.pattern, "--target-s", str(args.target_s),
                           "--cpu-threads", str(args.cpu_threads)]
                    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads))
                    env.pop("LASAGNA_METAL", None)
                    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
                    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                    if out.returncode != 0 or not line:
                        print(f"child failed: {' '.join(cmd)}\n{out.stderr[-2000:]}", file=sys.stderr)
                        sys.exit(1)
                    row = json.loads(line[0][7:])
                    row["round"] = r
                    rows.append(row)
                    print(f"round {r} slots {slots} n {row['n']} {variant}: "
                          + " ".join(f"{k}={v:.3f}" for k, v in row["p50_ms"].items()), flush=True)
    summary = []
    for metric, title in (("p50_ms", "one call between two syncs"), ("queued_ms", "10 calls queued per sync")):
        table = {}
        for row in rows:
            for name, ms in row[metric].items():
                table.setdefault((row["slots"], row["n"], name, row["variant"]), []).append(ms)
        print(f"\npattern={args.pattern}, rounds={args.rounds}: ms per call, {title} "
              "(median over rounds of each process's p50)")
        print("| slots | N | function | metal | mps_ops | cpu_ops | metal vs mps_ops | metal vs cpu_ops |")
        print("|---|---|---|---|---|---|---|---|")
        for slots in slot_counts:
            ns = sorted({row["n"] for row in rows if row["slots"] == slots})
            for n in ns:
                for name in FUNCTIONS:
                    med = {v: statistics.median(table[(slots, n, name, v)]) for v in VARIANTS}
                    summary.append({"metric": metric, "slots": slots, "n": n, "function": name,
                                    **{f"{v}_ms": med[v] for v in VARIANTS}})
                    print(f"| {slots} | {n} | {name} | {med['metal']:.3f} | {med['mps_ops']:.3f} | "
                          f"{med['cpu_ops']:.3f} | {med['mps_ops'] / med['metal']:.1f}x | "
                          f"{med['cpu_ops'] / med['metal']:.1f}x |")
    if args.out:
        meta = {"machine": platform.machine(), "macos": platform.mac_ver()[0], "python": platform.python_version(),
                "pattern": args.pattern, "rounds": args.rounds, "cpu_threads": args.cpu_threads, **_code_identity()}
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows, "summary": summary}, f, indent=1)


def _code_identity():
    """The git commit of the checkout this file is in, and whether lasagna's tracked files differ from it (both
    None outside a git checkout), so that results name the code they timed."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                                check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."], cwd=here,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(changes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=3, metavar=("VARIANT", "N", "SLOTS"))
    ap.add_argument("--sizes", default="2500,20000,100000,1000000")
    ap.add_argument("--slots", default="3000,20000")
    ap.add_argument("--pattern", choices=("surface", "random"), default="surface")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--target-s", type=float, default=1.0, help="timed seconds per function per process (5-60 calls)")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.child:
        variant, n, slots = args.child
        if variant not in VARIANTS:
            ap.error(f"variant must be one of {VARIANTS}")
        _child(variant, int(n), int(slots), args.pattern, args.target_s, args.cpu_threads)
    else:
        _driver(args)


if __name__ == "__main__":
    main()
