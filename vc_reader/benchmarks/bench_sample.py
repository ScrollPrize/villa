"""Point reads and sampling: read_points, sample(order=0) and sample(order=1), warm.

Measures the CPU cost of the point paths on local data that is already in the
page cache (one untimed repetition first), through memory maps and through the
memory cache. Point sets, all inside one stored 1024^3 region:

* ``random``: points uniform over the region (512 chunks, no locality).
* ``patch``: a surface-like sampling grid, 16 layers of 512 x 512 points on a
  tilted plane with non-integer spacing (an ink-detection patch).
* ``local``: points uniform in a 96^3 box (a few chunks, cache-resident).

Every scenario's output is hashed. ``--compare`` checks the hashes against an
earlier results file, so a speedup can be shown to return exactly the same
values as the build it is compared with.

    python benchmarks/bench_sample.py
    python benchmarks/bench_sample.py --reps 7 --compare results/sample-YYYYMMDD-HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from _common import chunk_files, digest, environment, evict_files, filesystem_type

from vc_reader.reader import ZarrReader

DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
REGION = ((9000, 3600, 3600), 1024)  # start (z, y, x), edge: stored throughout at level 2


def patch_geometry(start, edge, scale):
    """Shared physical plane geometry for explicit-point and affine benchmarks."""
    s = np.asarray(start, np.float64)
    u = np.array([0.1, 0.2, 0.97]); u /= np.linalg.norm(u)
    w = np.array([0.95, 0.3, 0.0]); w -= u * (u @ w); w /= np.linalg.norm(w)
    v = np.cross(w, u)
    side = int(512 * scale ** 0.5)
    origin = s + edge / 2 - 0.73 * side / 2 * (u + v)
    return side, origin, u, v, w


def point_sets(start, edge, scale: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    s = np.asarray(start, np.float64)
    n = int(1_000_000 * scale)
    random = s + rng.uniform(0, edge - 1, size=(n, 3))
    # 16 x 512 x 512 grid on a tilted plane: in-plane axes u, v and a normal w.
    side, origin, u, v, w = patch_geometry(start, edge, scale)
    a, b, c = np.meshgrid(np.arange(16), np.arange(side), np.arange(side), indexing="ij")
    patch = origin + 0.73 * (b[..., None] * v + c[..., None] * u) + 0.5 * a[..., None] * w
    local = s + 300 + rng.uniform(0, 95, size=(n, 3))
    return {"random": random, "patch": patch.reshape(-1, 3), "local": local}


OPS = {
    "read_points": lambda r, p: r.read_points(p),
    "nearest": lambda r, p: r.sample(p, order=0),
    "linear": lambda r, p: r.sample(p, order=1),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of points")
    ap.add_argument("--only", default="", help="comma-separated scenario prefixes, e.g. random-linear")
    ap.add_argument("--compare", help="results JSON whose output hashes must match")
    ap.add_argument("--label", default="", help="free-form note stored with the results")
    ap.add_argument("--cache-state", choices=["warm", "cold", "both"], default="warm")
    ap.add_argument("--readers", default="mmap,memory", help="comma-separated mmap, slab, memory")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--start", type=int, nargs=3, default=REGION[0], metavar=("Z", "Y", "X"))
    ap.add_argument("--edge", type=int, default=REGION[1])
    ap.add_argument("--output", type=Path, help="write results to this file")
    args = ap.parse_args(argv)
    if args.reps < 1 or args.scale <= 0:
        ap.error("reps and scale must be positive")
    reader_names = args.readers.split(",")
    if not set(reader_names) <= {"mmap", "slab", "memory"}:
        ap.error("readers must be mmap, slab or memory")
    if args.cache_state != "warm" and filesystem_type(Path(args.array)) in ("tmpfs", "ramfs"):
        ap.error("cold reads require a disk-backed array")

    sets = point_sets(args.start, args.edge, args.scale)
    for name, pts in sets.items():
        sets[name] = np.ascontiguousarray(pts)
    states = ["cold", "warm"] if args.cache_state == "both" else [args.cache_state]
    only = [o for o in args.only.split(",") if o]
    rows = []
    for set_name, pts in sets.items():
        ipts = np.rint(pts).astype(np.int64)
        for op_name, op in OPS.items():
            # Evict the enclosing box, including the upper interpolation neighbours.
            # All readers are closed first: live mappings can keep pages resident.
            p = ipts if op_name == "read_points" else pts
            lo = np.floor(p.min(axis=0)).astype(np.int64)
            hi = np.ceil(p.max(axis=0)).astype(np.int64) + 1
            files = chunk_files(args.array, [(tuple(lo), tuple(hi - lo))])
            for reader_name, state in ((r, s) for r in reader_names for s in states):
                scenario = f"{set_name}-{op_name}-{reader_name}" + ("-cold" if state == "cold" else "")
                if only and not any(scenario.startswith(o) for o in only):
                    continue
                def open_reader():
                    return ZarrReader.open(args.array, threads=args.threads,
                                           read_mode="slab" if reader_name == "slab" else "mmap",
                                           memory_cache=(4 << 30) if reader_name == "memory" else 0)

                times = []
                # Also prime the output pool for cold reads; close before eviction.
                r = open_reader()
                try:
                    out = op(r, p)
                    h = digest(out)
                    del out
                    for _ in range(args.reps):
                        if state == "cold":
                            r.close()
                            evict_files(files)
                            r = open_reader()
                        t = time.perf_counter()
                        out = op(r, p)
                        times.append(time.perf_counter() - t)
                        if digest(out) != h:
                            raise SystemExit(f"{scenario}: output changed between repetitions")
                        del out
                finally:
                    r.close()
                med = statistics.median(times)
                rows.append(dict(scenario=scenario, points=len(p), reps=args.reps, seconds_min=min(times),
                                 seconds_median=med, seconds_max=max(times), mpts_s_median=len(p) / med / 1e6,
                                 seconds_mean=statistics.mean(times), seconds=times, digest=h))
                print(f"  {scenario:<28} {len(p):>9} pts  median {med * 1e3:8.1f} ms "
                      f"({min(times) * 1e3:.1f}-{max(times) * 1e3:.1f})  {len(p) / med / 1e6:7.1f} Mpts/s",
                      flush=True)

    ok = True
    if args.compare:
        previous = json.loads(Path(args.compare).read_text())
        if previous["array"] != args.array or previous["region"] != [list(args.start), args.edge]:
            raise SystemExit("comparison uses a different array or region")
        before = {r["scenario"]: r for r in previous["rows"]}
        print(f"\n| scenario | points | before ms | after ms | speedup | same output |\n|---|---|---|---|---|---|")
        for r in rows:
            b = before.get(r["scenario"])
            if b is None:
                continue
            same = b["digest"] == r["digest"] and b["points"] == r["points"]
            ok &= same
            print(f"| {r['scenario']} | {r['points']} | {b['seconds_median'] * 1e3:.1f} | "
                  f"{r['seconds_median'] * 1e3:.1f} | {b['seconds_median'] / r['seconds_median']:.2f}x | "
                  f"{'yes' if same else 'NO'} |")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = args.output or out_dir / f"sample-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(dict(command=" ".join([sys.executable, *sys.argv]), environment=environment(),
                                    array=args.array, region=(args.start, args.edge), threads=args.threads,
                                    label=args.label, rows=rows), indent=1))
    print(f"\nwrote {path}")
    if not ok:
        raise SystemExit("output differs from the compared results")


if __name__ == "__main__":
    main()
