"""Full-volume surface probabilities from the frozen-encoder surface decoder.

Tiles lie on one global lattice: tile i on an axis starts at ``i * stride - (tile - stride) // 2``,
and the stride equals the output chunk size, so every output chunk is the centre of one tile and
the neighbouring tiles overlap it by ``(tile - stride) / 2`` voxels on each side. Overlaps blend
with complementary linear ramps, and the weight sum of the tiles that ran is divided out. Only tiles
whose central chunk holds nonzero CT run. Each tile is normalized on its own (robust, as in
training) and averaged over flip test-time augmentation.

The volume streams through z rows of tiles. A three-slot ring of chunk-deep float32 accumulators
(host memory by default) holds the chunks a row touches; after row j the chunk j - 1 receives no further contributions and
is written as ``uint8 = round(255 * p)`` with lossless 3D vc-delta3d compression. Voxels whose CT
value is zero (outside the masked scroll) stay zero; all-zero chunks are omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from vc_reader.reader import open_zarr

from dinovol_2.dataset.cuda_pipeline import normalize_batch
from dinovol_2.eval.embed_volume import ZarrGroupWriter, atomic_json, output_lock

FORMAT = "dinovol-surface-probability/1"
ARRAY = "0"
# Flip axes of a (D, H, W) tile; --tta N uses the first N.
FLIPS = ((), (2,), (1,), (0,), (1, 2), (0, 2), (0, 1), (0, 1, 2))


def ramp(tile: int, stride: int) -> np.ndarray:
    """1D tile weight: linear ramps over each overlap, complementary between neighbouring tiles."""
    overlap = tile - stride
    if not 0 <= overlap <= stride:
        raise ValueError("stride must lie in [tile / 2, tile]")
    w = np.ones(tile, np.float64)
    if overlap:
        r = (np.arange(overlap) + 0.5) / overlap
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


def axis_starts(lo: int, hi: int, tile: int, stride: int) -> list[int]:
    """Lattice tile starts whose extent intersects [lo, hi)."""
    origin = -((tile - stride) // 2)
    first = math.floor((lo - tile - origin) / stride) + 1
    last = math.ceil((hi - origin) / stride) - 1
    return [origin + i * stride for i in range(first, last + 1)]


class SurfacePredictor:
    """Surface probability of one uint8 (D, H, W) CUDA tile: robust normalization, flip TTA, softmax."""

    def __init__(self, model, tta: int = 8, dtype=torch.bfloat16):
        self.model, self.flips, self.dtype = model, FLIPS[:tta], dtype

    @torch.no_grad()
    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        image = normalize_batch(tile[None, None].contiguous(), "robust", device=tile.device)
        total = torch.zeros(tile.shape, device=tile.device, dtype=torch.float32)
        for axes in self.flips:
            dims = [2 + a for a in axes]
            x = image.flip(dims) if dims else image
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(x)
            p = logits.float().softmax(1)[:, 1:2]
            total += (p.flip(dims) if dims else p)[0, 0]
        return total / len(self.flips)


def bounding_box(volume: str, level: int, occupancy_level: int, shape, chunk: int):
    """Nonzero Y/X extent (union over z) from a coarse level, padded one coarse voxel, chunk-aligned."""
    coarse = np.asarray(open_zarr(volume, level=occupancy_level, threads=16)[:]) > 0
    factor = 2 ** (occupancy_level - level)
    ys, xs = np.nonzero(coarse.any(0))
    out = []
    for v, n in ((ys, shape[1]), (xs, shape[2])):
        lo = max(0, (int(v.min()) - 1) * factor) // chunk * chunk
        hi = min(n, -(-min(n, (int(v.max()) + 2) * factor) // chunk) * chunk)
        out.append((lo, hi))
    return out


def stream(read, predict, *, shape, bbox, tile, stride, write, start_chunk=0, end_chunk=None, device="cuda",
           accumulator_device=None, log=print):
    """Blend lattice tiles into chunk slabs.

    read(z0, z1) -> uint8 (z1 - z0, Y, X) CUDA array over the bbox with zero outside the volume.
    predict(tile) -> float32 probability. write(chunk_index, uint8 CPU slab of that chunk's depth).
    Only chunks [start_chunk, end_chunk) are written; earlier ones were already committed.

    A tile runs only when its core (the stride-sized chunk at its centre) holds nonzero CT. Cores
    partition the volume, so every nonzero voxel is still predicted; tiles that would only add
    overlap context to empty space are skipped. The blend weight sum of the tiles that actually ran
    is kept per row as a 2D Y/X map and combined with the separable z ramp, so it is exact.
    accumulator_device (default: device) holds the float32 chunk ring, e.g. "cpu" to save GPU memory.
    """
    Z = shape[0]
    (y0, y1), (x0, x1) = bbox
    w = ramp(tile, stride)
    margin = (tile - stride) // 2
    ys, xs = axis_starts(y0, y1, tile, stride), axis_starts(x0, x1, tile, stride)
    wz = torch.tensor(w, device=device, dtype=torch.float32)
    wyx = torch.tensor(w[:, None] * w[None, :], device=device, dtype=torch.float32)
    chunks = -(-Z // stride)
    end_chunk = chunks if end_chunk is None else min(end_chunk, chunks)
    H, W = y1 - y0, x1 - x0
    acc = device if accumulator_device is None else accumulator_device
    ring = torch.zeros((3, stride, H, W), device=acc, dtype=torch.float32)
    row_weights = {}  # row -> (H, W) sum of Y/X weights of its tiles that ran
    origin = -margin

    def rect(start, lo, hi):
        return max(lo, start), min(hi, start + tile)

    # Row j covers chunks j - 1 .. j + 1 and is the last contribution to chunk j - 1;
    # row -1 can only overlap chunk 0, whose voxels are cores of row 0, so it never runs.
    for row in range(start_chunk - 1, end_chunk + 1):
        zs = origin + row * stride
        t0 = time.time()
        done = 0
        weights = row_weights[row] = torch.zeros((H, W), device=device, dtype=torch.float32)
        core_z = (max(0, zs + margin), min(Z, zs + margin + stride))
        if core_z[0] < core_z[1]:
            block = read(zs, zs + tile)
            core_block = block[core_z[0] - zs:core_z[1] - zs]
            for ty in ys:
                cy = rect(ty + margin - y0, 0, H)
                cy = (cy[0], min(cy[1], ty + margin + stride - y0))
                if cy[0] >= cy[1]:
                    continue
                for tx in xs:
                    cx = rect(tx + margin - x0, 0, W)
                    cx = (cx[0], min(cx[1], tx + margin + stride - x0))
                    if cx[0] >= cx[1] or not bool((core_block[:, cy[0]:cy[1], cx[0]:cx[1]] > 0).any()):
                        continue
                    # Tile rectangle in bbox coordinates, clipped to the bbox (outside it the CT is zero).
                    a, b = rect(ty - y0, 0, H)
                    c, d = rect(tx - x0, 0, W)
                    pad = (c - (tx - x0), (tx + tile - x0) - d, a - (ty - y0), (ty + tile - y0) - b)
                    ct = block[:, a:b, c:d]
                    full = F.pad(ct, pad) if any(pad) else ct
                    p = predict(full) * (full > 0)
                    tw = wyx[pad[2]:tile - pad[3], pad[0]:tile - pad[1]]
                    p = (p[:, pad[2]:tile - pad[3], pad[0]:tile - pad[1]] * (wz[:, None, None] * tw)).to(acc)
                    weights[a:b, c:d] += tw
                    # Rows of this tile split over the (up to three) chunks it touches.
                    for chunk in range(math.floor(zs / stride), math.floor((zs + tile - 1) / stride) + 1):
                        if chunk < 0 or chunk >= chunks:
                            continue
                        lo, hi = max(zs, chunk * stride), min(zs + tile, (chunk + 1) * stride)
                        ring[chunk % 3, lo - chunk * stride:hi - chunk * stride, a:b, c:d] += p[lo - zs:hi - zs]
                    done += 1
        finished = row - 1
        if 0 <= finished < chunks:
            z_lo, z_hi = finished * stride, min(Z, (finished + 1) * stride)
            slot = ring[finished % 3, :z_hi - z_lo]
            if start_chunk <= finished < end_chunk:
                out = torch.empty(slot.shape, device=device, dtype=torch.uint8)
                step = 16  # bounded temporary for the weight sum
                for s0 in range(0, z_hi - z_lo, step):
                    s1 = min(z_hi - z_lo, s0 + step)
                    norm = torch.zeros((s1 - s0, H, W), device=device, dtype=torch.float32)
                    for r in (finished - 1, finished, finished + 1):
                        if r not in row_weights:
                            continue
                        rz = z_lo + s0 - (origin + r * stride)
                        k0, k1 = max(0, rz), min(tile, rz + s1 - s0)
                        if k0 < k1:
                            norm[k0 - rz:k1 - rz] += wz[k0:k1, None, None] * row_weights[r]
                    part = slot[s0:s1].to(device)
                    prob = torch.where(norm > 0, part / norm.clamp(min=1e-12), torch.zeros_like(norm))
                    out[s0:s1] = (prob.clamp(0, 1) * 255).round().to(torch.uint8)
                write(finished, out.cpu().numpy())
            ring[finished % 3].zero_()
            row_weights.pop(finished - 1, None)
            log(f"chunk {finished + 1}/{chunks}: z [{z_lo}, {z_hi}) row tiles {done} ({time.time() - t0:.1f}s)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--volume", required=True, help="OME-Zarr CT group (local path or vc_reader cache path)")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--occupancy-level", type=int, default=5, help="coarse level used for the Y/X bounding box")
    p.add_argument("--checkpoint", type=Path, default=Path("/mnt/raid_nvme/dinovol_ckpts/checkpoint_step_640000.pt"))
    p.add_argument("--branch", choices=["teacher", "student"], default="teacher")
    p.add_argument("--decoder", type=Path, default=Path("evals/kaggle_surfaces/runs/frozen_640000/best.pt"))
    p.add_argument("--vesuvius-source", type=Path, default=Path("../vesuvius/src"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile", type=int, default=192)
    p.add_argument("--stride", type=int, default=128, help="lattice stride = output chunk size")
    p.add_argument("--tta", type=int, default=8, choices=range(1, 9), metavar="1..8", help="number of flip TTA views")
    p.add_argument("--z-chunks", type=int, nargs=2, default=None, metavar=("FIRST", "END"),
                   help="only write output chunks [FIRST, END) along z (testing)")
    p.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--writer-workers", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--accumulator-device", default="cpu",
                   help="device of the float32 three-chunk blend ring (cpu saves ~3 chunk slabs of GPU memory)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true")
    g.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .model import SurfaceModel, load_encoder

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    reader = open_zarr(args.volume, level=args.level, threads=args.threads)
    shape = tuple(int(n) for n in reader.shape)
    stride, tile = args.stride, args.tile
    bbox = bounding_box(args.volume, args.level, args.occupancy_level, shape, stride)
    chunks = -(-shape[0] // stride)
    first, end = args.z_chunks or (0, chunks)
    config = {"volume": str(args.volume), "level": args.level, "shape": list(shape),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "branch": args.branch, "decoder": str(args.decoder.resolve()), "decoder_sha256": sha256(args.decoder),
              "tile": tile, "stride": stride, "tta_flips": [list(f) for f in FLIPS[:args.tta]],
              "normalization": "robust (per tile)", "precision": args.precision, "compile": args.compile,
              "bbox_yx": [list(b) for b in bbox], "z_chunks": [first, end], "torch_version": torch.__version__}
    attrs = {"format": FORMAT, **config, "values": "uint8 = round(255 * softmax surface probability); "
             "zero where the CT is zero", "completed_chunks": 0,
             "multiscales": [{"version": "0.4", "name": "surface probability",
                              "axes": [{"name": a, "type": "space"} for a in "zyx"],
                              "datasets": [{"path": ARRAY, "coordinateTransformations": [
                                  {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]}
    with output_lock(args.output):
        if args.resume and (args.output / ".zattrs").exists():
            saved = json.loads((args.output / ".zattrs").read_text())
            if saved.get("format") != FORMAT or any(saved.get(k) != v for k, v in config.items()):
                raise ValueError("resume configuration differs from the saved output")
            attrs = saved
            writer = ZarrGroupWriter(args.output, attrs=None, workers=args.writer_workers)
            start = first + attrs["completed_chunks"]
            # Remove chunks of an unfinished slab before rewriting it.
            writer.clear_z_range((start * stride, min(shape[0], end * stride)), array_z_start=0, steps={ARRAY: 1})
        else:
            writer = ZarrGroupWriter(args.output, attrs=attrs, overwrite=args.overwrite, workers=args.writer_workers)
            writer.create(ARRAY, shape, (stride,) * 3, "uint8", codec="vc-delta3d", channel_axis=None)
            start = first
        print(f"volume {shape}, bbox y {bbox[0]} x {bbox[1]}, chunks [{start}, {end}) of {chunks}, "
              f"tta {args.tta}", flush=True)
        if start >= end:
            writer.close()
            return

        encoder, _ = load_encoder(args.checkpoint, args.branch)
        model = SurfaceModel(encoder, blocks=2, vesuvius_source=args.vesuvius_source)
        state = torch.load(args.decoder, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(state["decoder"], strict=True)
        model = model.to(device).eval()
        if args.compile:
            model.compile(backend="inductor", mode="default", dynamic=False)
        predict = SurfacePredictor(model, args.tta, torch.bfloat16 if args.precision == "bfloat16" else torch.float32)

        (y0, y1), (x0, x1) = bbox

        def read(z0, z1):
            out = torch.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=torch.uint8, device=device)
            a, b = max(0, z0), min(shape[0], z1)
            if a < b:
                out[a - z0:b - z0] = torch.from_numpy(np.asarray(reader.read(np.s_[a:b, y0:y1, x0:x1]))).to(device)
            return out

        def write(chunk, slab):
            writer.write(ARRAY, (chunk * stride, y0, x0), slab)
            writer.flush()
            attrs["completed_chunks"] = chunk + 1 - first
            atomic_json(args.output / ".zattrs", attrs)

        try:
            stream(read, predict, shape=shape, bbox=bbox, tile=tile, stride=stride, write=write,
                   start_chunk=start, end_chunk=end, device=device, accumulator_device=args.accumulator_device,
                   log=lambda m: print(m, flush=True))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
