"""Render dinovol embeddings on a tifxyz surface, embedding only the lattice cells the surface needs.

The surface is rendered as a mesh intersection: vc_reader bilinearly interpolates the tifxyz grid at
every output pixel (``--render-step`` UV pixels apart; by default one pixel per embedding-level voxel)
and samples each volume there, trilinearly, on CUDA. Embeddings are computed with the same globally
consistent tiling as ``embed_volume`` -- only in a shell of voxels around the surface -- written to a
sparse zarr, rendered, and deleted unless ``--keep-embeddings`` is given.

  python -m dinovol_2.eval.render_surface_embeddings --volume s3://.../volume.zarr --level 2 \\
      --cache-dir /ephemeral/zarr_cache --offline --tifxyz SEGMENT.tifxyz \\
      --checkpoint checkpoint_step_640000.pt --out OUT --gpus 5,6,7 [--labels INK_LABEL_DIR]

Writes OUT/<name>_{ct,tokens_pca,dense_pca,ink_pca}.tif (uint8; PCA panels map the top three
principal components of the unit embeddings to RGB), OUT/<name>_inklabels.tif when labels are given
(upsampled to the render), and OUT/<name>_render.json.

Ink focus: with ink labels (or ``--ink-prototypes`` from other segments), reference embeddings are the
mean unit embedding of labelled ink pixels and of non-ink pixels -- only those inside ``supervision``
labelled zero; unlabelled pixels are unknown, never background. OUT/<name>_inkfocus_{ink,tokens}.tif
shows cos(e, ink) - cos(e, non-ink) per pixel (cos(e, ink) without supervision; uint8 over its
1-99th percentiles) and OUT/<name>_ink_prototypes.npz keeps the references. References taken from the
segment's own labels make that image a visualisation, not a held-out test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from dinovol_2.dataset.volume_reader import open_volume
from dinovol_2.eval.embed_volume import (
    AxisLattice, RegionEmbedder, ZarrGroupWriter, global_normalization, sample_volume_histogram,
    statistics_from_histogram, plan_slabs,
)
from dinovol_2.eval.napari_visualizer import load_backbone_from_checkpoint

OUTPUTS = ("tokens_unit", "dense", "ink")
FOCUS = ("ink", "tokens_unit")  # embeddings used for the ink-focus image
STORAGE = {"tokens_unit": "float32", "dense": "unit8", "ink": "unit8"}
SHELL_DILATION = 2   # voxels around the surface's voxels: covers trilinear support at any render point
SLAB_CELLS = 4       # lattice cells of z per embedding slab (one unit of GPU work)


# ---------------------------------------------------------------------------
# Surface geometry


def load_tifxyz(path: Path):
    """(H, W, 3) zyx coordinates in the tifxyz's own voxel frame, validity, meta."""
    x, y, z = (tifffile.imread(path / f"{a}.tif").astype(np.float32) for a in "xyz")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x != -1) & (y != -1) & (z != -1)
    mask = path / "mask.tif"
    if mask.exists():
        valid &= tifffile.imread(mask) > 0
    return np.stack([z, y, x], -1), valid, json.loads((path / "meta.json").read_text())


@torch.inference_mode()
def surface_shell(grid: np.ndarray, valid: np.ndarray, factor: float, device, spacing: float = 0.5,
                  rows_per_chunk: int = 32) -> np.ndarray:
    """Unique (z, y, x) voxels (embedding level) the bilinear surface passes through, int32 (N, 3).

    Every valid quad is sampled at <= ``spacing`` voxels so no crossed voxel is skipped.
    """
    g = torch.from_numpy(grid).to(device) * factor
    v = torch.from_numpy(valid).to(device)
    quad = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
    edge = torch.maximum((g[:-1, 1:] - g[:-1, :-1]).norm(dim=-1), (g[1:, :-1] - g[:-1, :-1]).norm(dim=-1))
    n = int(math.ceil(float(edge[quad].max()) / spacing)) + 1 if quad.any() else 2
    n = min(n, 64)
    t = torch.linspace(0, 1, n, device=device)
    a, b = torch.meshgrid(t, t, indexing="ij")
    keys = []
    for r0 in range(0, quad.shape[0], rows_per_chunk):
        q = quad[r0:r0 + rows_per_chunk]
        rr, cc = q.nonzero(as_tuple=True)
        if not len(rr):
            continue
        rr = rr + r0
        p00, p01, p10, p11 = g[rr, cc], g[rr, cc + 1], g[rr + 1, cc], g[rr + 1, cc + 1]
        for i in range(0, len(rr), 1 << 14):
            s = slice(i, i + (1 << 14))
            pts = ((1 - a)[None, ..., None] * (1 - b)[None, ..., None] * p00[s, None, None]
                   + (1 - a)[None, ..., None] * b[None, ..., None] * p01[s, None, None]
                   + a[None, ..., None] * (1 - b)[None, ..., None] * p10[s, None, None]
                   + a[None, ..., None] * b[None, ..., None] * p11[s, None, None]).reshape(-1, 3)
            vox = pts.floor().long()
            vox = vox[(vox >= 0).all(1)]
            keys.append(torch.unique((vox[:, 0] << 42) | (vox[:, 1] << 21) | vox[:, 2]))
        if len(keys) > 64:  # merge as we go to bound memory
            keys = [torch.unique(torch.cat(keys))]
    keys = torch.unique(torch.cat(keys)) if keys else torch.zeros(0, dtype=torch.long, device=device)
    out = torch.stack([keys >> 42, (keys >> 21) & ((1 << 21) - 1), keys & ((1 << 21) - 1)], 1)
    return out.int().cpu().numpy()


class Shell:
    """Surface voxels bucketed by (y, x) lattice cell for fast column queries."""

    def __init__(self, voxels: np.ndarray, cell: int):
        self.cell = cell
        by, bx = voxels[:, 1] // cell, voxels[:, 2] // cell
        order = np.lexsort((voxels[:, 0], bx, by))
        self.voxels = voxels[order]
        keys = by[order].astype(np.int64) * (1 << 20) + bx[order]
        self.keys, self.starts = np.unique(keys, return_index=True)
        self.ends = np.append(self.starts[1:], len(keys))

    def save(self, path):
        np.savez(path, voxels=self.voxels, cell=self.cell)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        shell = cls.__new__(cls)
        shell.cell = int(data["cell"])
        shell.voxels = data["voxels"]
        by, bx = shell.voxels[:, 1] // shell.cell, shell.voxels[:, 2] // shell.cell
        keys = by.astype(np.int64) * (1 << 20) + bx
        shell.keys, shell.starts = np.unique(keys, return_index=True)
        shell.ends = np.append(shell.starts[1:], len(keys))
        return shell

    def query(self, lo, hi):
        """Voxels inside [lo, hi) (zyx)."""
        parts = []
        for cy in range(int(lo[1]) // self.cell, (int(hi[1]) - 1) // self.cell + 1):
            for cx in range(int(lo[2]) // self.cell, (int(hi[2]) - 1) // self.cell + 1):
                i = np.searchsorted(self.keys, cy * (1 << 20) + cx)
                if i < len(self.keys) and self.keys[i] == cy * (1 << 20) + cx:
                    v = self.voxels[self.starts[i]:self.ends[i]]
                    z0, z1 = np.searchsorted(v[:, 0], [lo[0], hi[0]])
                    parts.append(v[z0:z1])
        if not parts:
            return np.zeros((0, 3), np.int32)
        v = np.concatenate(parts)
        return v[np.all((v >= lo) & (v < hi), axis=1)]

    def need_fn(self, device, dilation=SHELL_DILATION):
        def need(z0, z1, y0, x0, size, margin):
            m = margin + dilation
            lo = np.array([z0 - m, y0 - m, x0 - m]); hi = np.array([z1 + m, y0 + size + m, x0 + size + m])
            v = self.query(lo, hi)
            if not len(v):
                return None
            mask = torch.zeros(tuple(hi - lo), dtype=torch.float16, device=device)
            idx = torch.from_numpy((v - lo).astype(np.int64)).to(device)
            mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
            if dilation:
                mask = F.max_pool3d(mask[None], 2 * dilation + 1, stride=1, padding=dilation)[0]
            mask = mask[dilation:-dilation, dilation:-dilation, dilation:-dilation] if dilation else mask
            return mask > 0 if bool(mask.any()) else None
        return need


# ---------------------------------------------------------------------------
# Embedding (one worker per GPU)


def _model_and_lattice(args, device):
    loaded = load_backbone_from_checkpoint(args.checkpoint, device=device)
    tile = int(loaded.global_input_size[0]); patch = int(loaded.patch_size[0])
    overlap = max(patch, tile // 6 // patch * patch)
    return loaded, [AxisLattice(tile, overlap, overlap)] * 3


def _open(args):
    s3 = {"anon": True} if str(args.volume).startswith("s3://") else None
    return open_volume(args.volume, level=args.level, s3_storage_options=s3, cache_dir=args.cache_dir,
                       offline=args.offline, cache_writes=not (args.offline or args.no_cache_writes),
                       connections=32, threads=16)


def worker(args) -> None:
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = False
    work = Path(args.work)
    plan = json.loads((work / "plan.json").read_text())
    reader = _open(args)
    loaded, lattice = _model_and_lattice(args, device)
    transforms = global_normalization(loaded.normalization_scheme, plan["statistics"], loaded.intensity_properties)
    embedder = RegionEmbedder(reader, loaded, transforms=transforms, outputs=OUTPUTS, dense_stride=1,
                              lattice=lattice, dtypes=STORAGE, strip_device=args.strip_device)
    shell = Shell.load(work / "shell.npz")
    need = shell.need_fn(device)
    sink = ZarrGroupWriter(work / "embeddings.zarr", attrs=None)
    totals = {}
    for i, (z0, z1) in enumerate(plan["slabs"]):
        if i % args.worker_count != args.worker_rank:
            continue
        t0 = time.time()
        stats = embedder.run(sink, z_range=(z0, z1), need=need, array_z_range=tuple(plan["array_z"]),
                             read="column", progress=False)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"[rank {args.worker_rank}] slab z {z0}-{z1}: {stats} {time.time() - t0:.0f}s", flush=True)
    sink.close()
    (work / f"worker{args.worker_rank}.json").write_text(json.dumps(totals))


def embed(args, work: Path, shell: Shell, statistics, reader, lattice) -> dict:
    z_lo, z_hi = int(shell.voxels[:, 0].min()) - SHELL_DILATION, int(shell.voxels[:, 0].max()) + SHELL_DILATION + 1
    stride = lattice[0].stride
    slab = SLAB_CELLS * stride
    start = max(0, z_lo) // stride * stride
    slabs = plan_slabs((start, min(z_hi, reader.shape[0])), slab)
    plan = {"slabs": slabs, "array_z": [0, int(reader.shape[0])], "statistics": statistics}
    (work / "plan.json").write_text(json.dumps(plan))
    # Arrays span the volume (z aligned to cells) so slabs write disjoint chunks.
    loaded = load_backbone_from_checkpoint(args.checkpoint, device=torch.device("cuda", 0))
    embedder = RegionEmbedder(reader, loaded, transforms=[], outputs=OUTPUTS, dense_stride=1, lattice=lattice,
                              dtypes=STORAGE)
    sink = ZarrGroupWriter(work / "embeddings.zarr", attrs={"checkpoint": str(args.checkpoint),
                                                            "volume": str(args.volume), "level": args.level,
                                                            "normalization": statistics,
                                                            "lattice": lattice[0].__dict__}, overwrite=True)
    embedder.create_arrays(sink, (0, int(reader.shape[0])), z_cell=True)
    sink.close()
    del loaded, embedder
    torch.cuda.empty_cache()
    gpus = args.gpus.split(",")
    procs = []
    for rank, gpu in enumerate(gpus):
        cmd = [sys.executable, "-m", "dinovol_2.eval.render_surface_embeddings", "--worker-rank", str(rank),
               "--worker-count", str(len(gpus)), "--work", str(work), *sys.argv[1:]]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"embedding workers failed: {codes}")
    totals = {}
    for rank in range(len(gpus)):
        for k, v in json.loads((work / f"worker{rank}.json").read_text()).items():
            totals[k] = totals.get(k, 0) + v
    return {"slabs": len(slabs), **totals}


# ---------------------------------------------------------------------------
# Rendering


def _sources(args, work: Path, um: float, tif_level: int, has_mask: bool):
    from vc_reader import Affine, Source, Storage, Surface, VolumeRegistration
    s = 2.0 ** args.level
    spacing = (um * s,) * 3
    reg = lambda affine: VolumeRegistration("sample", "volume", affine=affine, volume_spacing_um=(um,) * 3)
    # The CT is read like the embedding input: through --cache-dir, and remotely for uncached chunks.
    ct = Source("ct", Storage(f"{str(args.volume).rstrip('/')}/{args.level}", "zarr"), role="image",
                axes=("z", "y", "x"), registration=reg(Affine.scale(s)), normalize=False)
    emb = work / "embeddings.zarr"
    targets = {name: Source(name, Storage(str(emb / name), "zarr"), role="image", axes=("z", "y", "x", "c"),
                            registration=reg(Affine.scale(s)), normalize=False) for name in ("dense", "ink")}
    # Token j covers voxels [8j, 8j + 8) at the embedding level: its value sits at voxel 8j + 3.5.
    t = 8 * s
    targets["tokens_unit"] = Source("tokens_unit", Storage(str(emb / "tokens_unit"), "zarr"), role="image",
                                    axes=("z", "y", "x", "c"), normalize=False,
                                    registration=reg(Affine.from_rows([[t, 0, 0, 3.5 * s], [0, t, 0, 3.5 * s],
                                                                       [0, 0, t, 3.5 * s]])))
    surface = Surface("surface", Storage(str(args.tifxyz), "tifxyz"), "sample", "volume",
                      component_order=("x", "y", "z"), xyz_to_volume=Affine.scale(2.0 ** tif_level),
                      spacing_um=(um * 2.0 ** tif_level,) * 3, tifxyz_mask=has_mask)
    return ct, targets, surface


def render(args, work: Path, out: Path, name: str, meta, um, tif_level, step, grid_shape, has_mask) -> dict:
    from vc_reader.sampling import open_sample
    ct, targets, surface = _sources(args, work, um, tif_level, has_mask)
    sx, sy = meta["scale"]
    H, W = int(math.ceil(grid_shape[0] / sy / step)), int(math.ceil(grid_shape[1] / sx / step))
    device = torch.device("cuda", 0)
    scales = {"dense": 1 / 127, "ink": 1 / 127, "tokens_unit": 1.0}
    tile = args.render_tile

    def tiles(stride):
        for r in range(0, H, tile):
            for c in range(0, W, tile):
                yield r, c, min(tile, H - r), min(tile, W - c), stride

    def draw(reader, r, c, h, w, k):
        # Output pixel (r, c) at render step k sits at UV (r*k, c*k) of the full-resolution grid.
        hh, ww = -(-h // (k // step)), -(-w // (k // step))
        res = reader.render_surface(surface, (r * step, c * step), (hh, ww), offsets_um=(0,), step=k)
        out_ = {"ct": torch.from_dlpack(res["image"].data)[0, 0], "valid": torch.from_dlpack(res["image"].valid)[0, 0]}
        for n in OUTPUTS:
            v = torch.from_dlpack(res[n].data)[:, 0].permute(1, 2, 0) * scales[n]
            ok = torch.from_dlpack(res[n].valid)[0, 0] & (v.abs().sum(-1) > 0)
            out_[n] = (F.normalize(v, dim=-1), ok)
        return out_

    # Labels live on the tifxyz grid: output pixel (r, c) reads grid (r*step*sy, c*step*sx).
    rows = np.minimum((np.arange(H) * step * sy).round().astype(int), grid_shape[0] - 1)
    cols = np.minimum((np.arange(W) * step * sx).round().astype(int), grid_shape[1] - 1)
    labels = {}
    if args.labels:
        for label in ("inklabels", "supervision"):
            src = Path(args.labels) / f"{label}.tif"
            if src.exists():
                labels[label] = tifffile.imread(src) > 0
    focus_samples = {n: {"ink": [], "background": []} for n in FOCUS}
    remote = dict(cache_dir=args.cache_dir, offline=args.offline,
                  cache_writes=not (args.offline or args.no_cache_writes)) if args.cache_dir else {}
    with open_sample(ct, targets, backend="cuda", device=0, surface_output_bytes=16 << 30,
                     sampling_cache_bytes=8 << 30, sampling_work_bytes=2 << 30, memory_cache=8 << 30,
                     **remote) as reader:
        # Pass 1: PCA bases (and labelled ink references) from a coarse render (every 8th output pixel).
        coarse = 8 * step
        samples = {n: [] for n in OUTPUTS}
        for r in range(0, H, tile * 8):
            for c in range(0, W, tile * 8):
                h, w = min(tile * 8, H - r), min(tile * 8, W - c)
                d = draw(reader, r, c, h, w, coarse)
                for n in OUTPUTS:
                    v, ok = d[n]
                    samples[n].append(v[ok])
                if "inklabels" in labels:
                    rr, cc = np.ix_(rows[r:r + h:8], cols[c:c + w:8])
                    ink = torch.from_numpy(labels["inklabels"][rr, cc]).to(device)
                    for n in FOCUS:
                        v, ok = d[n]
                        focus_samples[n]["ink"].append(v[ok & ink])
                        # Non-ink only where supervised and labelled zero: unlabelled is unknown, not background.
                        if "supervision" in labels:
                            sup = torch.from_numpy(labels["supervision"][rr, cc]).to(device)
                            focus_samples[n]["background"].append(v[ok & sup & ~ink])
        bases = {}
        for n in OUTPUTS:
            v = torch.cat(samples[n])
            v = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(0))[:200_000].to(v.device)]
            mean = v.mean(0)
            comps = torch.linalg.svd(v - mean, full_matrices=False)[2][:3]
            proj = (v - mean) @ comps.T
            bases[n] = (mean, comps, torch.quantile(proj, 0.01, dim=0), torch.quantile(proj, 0.99, dim=0))
        prototypes = {}
        if "inklabels" in labels:
            for n in FOCUS:
                ink = torch.cat(focus_samples[n]["ink"])
                bg = torch.cat(focus_samples[n]["background"]) if focus_samples[n]["background"] else ink[:0]
                if len(ink):
                    prototypes[n] = {"ink": F.normalize(ink.mean(0), dim=0), "count_ink": len(ink),
                                     "count_background": len(bg)}
                    if len(bg):
                        prototypes[n]["background"] = F.normalize(bg.mean(0), dim=0)
        elif args.ink_prototypes:
            saved = np.load(args.ink_prototypes)
            for n in FOCUS:
                if f"{n}_ink" in saved:
                    prototypes[n] = {"ink": torch.from_numpy(saved[f"{n}_ink"]).to(device)}
                    if f"{n}_background" in saved:
                        prototypes[n]["background"] = torch.from_numpy(saved[f"{n}_background"]).to(device)

        def ink_score(v, proto):
            # Without a supervised non-ink reference, similarity to ink alone.
            return v @ proto["ink"] - (v @ proto["background"] if "background" in proto else 0)

        focus = {}
        for n, proto in prototypes.items():
            v = torch.cat(samples[n])
            score = ink_score(v, proto)
            focus[n] = (proto, torch.quantile(score[torch.randperm(len(score), device=score.device)[:1_000_000]],
                                              torch.tensor([0.01, 0.99], device=score.device)))
        # Pass 2: full resolution.
        images = {n: np.zeros((H, W, 3), np.uint8) for n in OUTPUTS}
        images["ct"] = np.zeros((H, W), np.uint8)
        for n in focus:
            images[f"inkfocus_{n}"] = np.zeros((H, W), np.uint8)
        coverage = 0
        for r, c, h, w, _ in tiles(step):
            d = draw(reader, r, c, h, w, step)
            ok = d["valid"]
            images["ct"][r:r + h, c:c + w] = (d["ct"].clamp(0, 255).round() * ok).to(torch.uint8).cpu().numpy()
            for n in OUTPUTS:
                v, okn = d[n]
                mean, comps, lo, hi = bases[n]
                rgb = (((v - mean) @ comps.T - lo) / (hi - lo)).clamp(0, 1) * 255
                images[n][r:r + h, c:c + w] = (rgb.round() * okn[..., None]).to(torch.uint8).cpu().numpy()
            for n, (proto, (lo_s, hi_s)) in focus.items():
                v, okn = d[n]
                score = ((ink_score(v, proto) - lo_s) / (hi_s - lo_s)).clamp(0, 1)
                # 0 is reserved for pixels without a valid embedding.
                images[f"inkfocus_{n}"][r:r + h, c:c + w] = ((score * 254 + 1).round() * okn).to(torch.uint8).cpu().numpy()
            coverage += int(ok.sum())
    paths = {}
    for n, img in images.items():
        key = (n.replace("tokens_unit", "tokens") if n == "ct" or n.startswith("inkfocus_")
               else f"{'tokens' if n == 'tokens_unit' else n}_pca")
        paths[key] = out / f"{name}_{key}.tif"
        tifffile.imwrite(paths[key], img, bigtiff=True, compression="zlib", tile=(512, 512),
                         photometric="rgb" if img.ndim == 3 else "minisblack")
    for label, mask in labels.items():
        paths[label] = out / f"{name}_{label}.tif"
        tifffile.imwrite(paths[label], (mask[rows][:, cols] * 255).astype(np.uint8), bigtiff=True,
                         compression="zlib", tile=(512, 512))
    focus_info = {}
    if focus:
        arrays = {}
        for n, (proto, (lo_s, hi_s)) in focus.items():
            arrays[f"{n}_ink"] = proto["ink"].cpu().numpy()
            if "background" in proto:
                arrays[f"{n}_background"] = proto["background"].cpu().numpy()
            focus_info[n] = {"score": "cos(e, ink) - cos(e, background)" if "background" in proto else "cos(e, ink)",
                             "background": "supervised pixels labelled zero" if "background" in proto else None,
                             "uint8_range": [float(lo_s), float(hi_s)],
                             "references": "own labels" if "inklabels" in labels else str(args.ink_prototypes),
                             **{k: int(v) for k, v in proto.items() if k.startswith("count")}}
        paths["ink_prototypes"] = out / f"{name}_ink_prototypes.npz"
        np.savez(paths["ink_prototypes"], **arrays)
    return {"shape": [H, W], "render_step_uv": step, "valid_pixels": coverage, "ink_focus": focus_info,
            "files": {k: str(v) for k, v in paths.items()}}


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--volume", required=True, help="CT ome-zarr (s3:// URL with --cache-dir, or a path)")
    p.add_argument("--level", type=int, required=True, help="volume level to embed (the checkpoint's training scale)")
    p.add_argument("--tifxyz", required=True, type=Path)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", default=None, help="output file prefix (default: the tifxyz directory name)")
    p.add_argument("--labels", default=None, help="ink label directory (inklabels.tif/supervision.tif): upsampled to "
                   "the render and used for the ink-focus references")
    p.add_argument("--ink-prototypes", default=None, help="ink/background references (.npz from another render) "
                   "for segments without labels")
    p.add_argument("--tifxyz-level", type=int, default=None,
                   help="level of the tifxyz coordinates (default: meta.json download_coordinates, else --level)")
    p.add_argument("--voxel-um", type=float, default=None, help="level-0 voxel size (default: parsed from the volume name)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--offline", action="store_true", help="read only cached chunks; never write the cache")
    p.add_argument("--no-cache-writes", action="store_true",
                   help="read uncached chunks remotely without adding them to --cache-dir")
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids; one embedding worker per GPU")
    p.add_argument("--render-step", type=int, default=None, help="UV pixels between output pixels (default: 1 per voxel)")
    p.add_argument("--render-tile", type=int, default=1024)
    p.add_argument("--strip-device", default="cuda", help="where pending overlap strips live (cuda or cpu)")
    p.add_argument("--normalization-stats", default=None, help="statistics JSON or an embedding group's .zattrs")
    p.add_argument("--work", default=None, help="scratch directory for the embeddings (default: OUT/work_<name>)")
    p.add_argument("--keep-embeddings", action="store_true")
    p.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.worker_rank is not None:
        return worker(args)
    import re
    t0 = time.time()
    name = args.name or args.tifxyz.name.removesuffix(".tifxyz")
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out / f"work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    args.work = str(work)
    grid, valid, meta = load_tifxyz(args.tifxyz)
    tif_level = args.tifxyz_level
    if tif_level is None:
        tif_level = meta.get("download_coordinates", {}).get("target_level", args.level)
    um = args.voxel_um or float(re.search(r"([\d.]+)um", str(args.volume)).group(1))
    factor = 2.0 ** (tif_level - args.level)
    device = torch.device("cuda", 0)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus.split(",")[0])
    reader = _open(args)
    loaded, lattice = _model_and_lattice(args, device)
    scheme = loaded.normalization_scheme
    del loaded
    torch.cuda.empty_cache()
    # Volume-wide statistics from the cached chunks (deterministic for a given cache).
    statistics = None
    if scheme not in ("none", "ct"):
        if args.normalization_stats:
            saved = json.loads(Path(args.normalization_stats).read_text())
            statistics = saved.get("normalization", saved)
            statistics = statistics.get("statistics", statistics)
        else:
            histogram, sampled = sample_volume_histogram(reader, skip_missing=True)
            statistics = {**statistics_from_histogram(histogram), "sampled_chunks": sampled, "chunk_stride": 8,
                          "selection": "nonzero voxels of every 8th chunk along each axis"}
    print(f"{name}: grid {grid.shape[:2]}, tifxyz level {tif_level}, embedding level {args.level}, "
          f"normalization {json.dumps(statistics)}", flush=True)
    voxels = surface_shell(grid, valid, factor, device)
    inside = np.all((voxels >= 0) & (voxels < np.array(reader.shape)), axis=1)
    shell = Shell(voxels[inside], lattice[1].stride)
    shell.save(work / "shell.npz")
    print(f"{name}: {len(shell.voxels)} surface voxels", flush=True)
    t1 = time.time()
    embed_stats = embed(args, work, shell, statistics, reader, lattice)
    t2 = time.time()
    print(f"{name}: embedded {embed_stats} in {t2 - t1:.0f}s", flush=True)
    spacing = np.median(np.linalg.norm(np.diff(grid, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]]) * factor
    step = args.render_step or max(1, int(round(1 / meta["scale"][0] / spacing)))
    info = render(args, work, out, name, meta, um, tif_level, step, grid.shape[:2],
                  (args.tifxyz / "mask.tif").exists())
    t3 = time.time()
    emb_bytes = sum(f.stat().st_size for f in (work / "embeddings.zarr").rglob("*") if f.is_file())
    record = {"name": name, "tifxyz": str(args.tifxyz), "volume": str(args.volume), "level": args.level,
              "tifxyz_level": tif_level, "checkpoint": str(args.checkpoint), "normalization": statistics,
              "lattice": lattice[0].__dict__, "surface_voxels": int(len(shell.voxels)), "embedding": embed_stats,
              "embedding_bytes": emb_bytes, "render": info,
              "seconds": {"shell": round(t1 - t0), "embed": round(t2 - t1), "render": round(t3 - t2)}}
    (out / f"{name}_render.json").write_text(json.dumps(record, indent=2))
    if not args.keep_embeddings:
        shutil.rmtree(work)
    print(json.dumps(record["seconds"]), "done", flush=True)


if __name__ == "__main__":
    main()
