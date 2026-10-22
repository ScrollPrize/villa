"""Stream saved ZYXC embeddings to a separate, PCA-colored OME-Zarr pyramid."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from vc_reader.reader import open_zarr

from .embed_volume import ZarrGroupWriter, atomic_json, output_lock
from .napari_visualizer import compute_patch_embedding_pca_basis


def stored_chunks(path: Path) -> list[tuple[int, int, int]]:
    """Index stored ZYXC chunks, without visiting the much larger logical grid."""
    result = []
    for z in os.scandir(path):
        if not z.name.isdecimal() or not z.is_dir():
            continue
        for y in os.scandir(z.path):
            if not y.name.isdecimal() or not y.is_dir():
                continue
            for x in os.scandir(y.path):
                if x.name.isdecimal() and x.is_dir() and (Path(x.path) / "0").is_file():
                    result.append((int(z.name), int(y.name), int(x.name)))
    return sorted(result)


def fit_colors(reader, indices, chunks, *, sample_chunks, sample_voxels, seed, value_scale):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), min(sample_chunks, len(indices)), replace=False)
    per_chunk = max(1, -(-sample_voxels // len(chosen)))
    samples = []
    # Rejection sampling uniformly samples voxels, rather than overweighting sparse chunks.
    for offset in range(0, len(chosen), 16):
        regions = [(tuple(i * c for i, c in zip(indices[j], chunks)) + (0,), chunks)
                   for j in chosen[offset:offset + 16]]
        for block in reader.read_regions(regions):
            flat = block.reshape(-1, chunks[-1])
            sample = flat[rng.choice(len(flat), min(per_chunk, len(flat)), replace=False)]
            samples.append(sample[np.any(sample != 0, axis=1)])
    samples = np.concatenate(samples).astype(np.float32) * np.float32(value_scale)
    if len(samples) < 4:
        raise ValueError("too few nonzero sampled voxels; increase --sample-chunks/--sample-voxels")
    mean, components = compute_patch_embedding_pca_basis(samples[:, None, None, :])
    # Give the largest loading of each component a positive sign for repeatability.
    for column in components.T:
        if column[np.abs(column).argmax()] < 0:
            column *= -1
    projected = (samples - mean) @ components
    lo, hi = np.percentile(projected, [1, 99], axis=0)
    variance = samples.var(axis=0, dtype=np.float64).sum()
    explained = projected.var(axis=0, dtype=np.float64) / variance if variance > 0 else np.zeros(3)
    return dict(mean=mean.tolist(), components=components.tolist(), low=lo.tolist(), high=hi.tolist(),
                value_scale=value_scale, sample_count=len(samples), seed=seed,
                sample_chunks=len(chosen), requested_sample_voxels=sample_voxels,
                percentiles=[1, 99], explained_variance_ratio=explained.tolist())


class ColorProjector:
    def __init__(self, colors, device):
        self.device = torch.device(device)
        mean = np.array(colors["mean"], dtype=np.float32)
        components = np.array(colors["components"], dtype=np.float32)
        lo, hi = (np.array(colors[k], dtype=np.float32) for k in ("low", "high"))
        span = hi - lo
        gain = np.divide(255., span, out=np.zeros_like(span), where=span > 1e-8)
        self.weight = torch.tensor(components * gain * colors["value_scale"], device=self.device)
        self.bias = torch.tensor((-mean @ components - lo) * gain, device=self.device)

    @torch.inference_mode()
    def __call__(self, blocks):
        raw = torch.from_numpy(np.stack(blocks)).to(self.device)
        rgb = raw.float() @ self.weight + self.bias
        rgb = rgb.round().clamp_(0, 255).to(torch.uint8)
        rgb.masked_fill_(~raw.ne(0).any(dim=-1, keepdim=True), 0)
        return rgb.cpu().numpy()


@torch.inference_mode()
def downsample_colors(blocks, device):
    """2x box means; sums of eight uint8 values are exact in float32."""
    if torch.device(device).type == "cuda":
        values = torch.from_numpy(np.stack(blocks)).to(device=device, dtype=torch.float32)
        reduced = torch.nn.functional.avg_pool3d(values, kernel_size=2, stride=2)
        return reduced.round().to(torch.uint8).cpu().numpy()
    _, z, y, x = blocks[0].shape
    return [np.rint(block.reshape(3, z // 2, 2, y // 2, 2, x // 2, 2)
                    .mean(axis=(2, 4, 6), dtype=np.float32)).astype(np.uint8) for block in blocks]


def ome_metadata(source, array, source_attrs, shape, chunks, colors, levels, fingerprint):
    spec = source_attrs.get("outputs", {}).get(array, {})
    step = spec.get("step_zyx", [1, 1, 1])
    origin = spec.get("origin_zyx", [0, 0, 0])
    datasets = []
    for level in range(levels):
        factor = 2 ** level
        datasets.append(dict(path=str(level), coordinateTransformations=[
            dict(type="scale", scale=[1, *[s * factor for s in step]]),
            dict(type="translation", translation=[0, *[o + s * (factor - 1) / 2
                                                        for o, s in zip(origin, step)]])]))
    return {
        "multiscales": [dict(version="0.4", name=f"{array} PCA RGB", axes=[
            dict(name="c", type="channel"), *[dict(name=a, type="space") for a in "zyx"]],
            datasets=datasets, type="mean", metadata=dict(description="2x box mean, zero padded at array edges"))],
        "omero": dict(id=1, name=f"{array} PCA RGB", version="0.4",
                      channels=[dict(label=name, color=color, active=True,
                                     window=dict(start=0, end=255, min=0, max=255))
                                for name, color in zip(("Red", "Green", "Blue"), ("FF0000", "00FF00", "0000FF"))],
                      rdefs=dict(model="color", defaultT=0, defaultZ=shape[0] // 2)),
        "embedding_rgb": dict(version=1, source=str(source), array=array, source_fingerprint=fingerprint,
                              shape_zyxc=list(shape), chunks_zyxc=list(chunks), pca=colors,
                              coordinates="voxel coordinates at source embedding volume level; no physical unit inferred",
                              source_volume=source_attrs.get("volume"), source_level=source_attrs.get("level"),
                              description="Three global principal components mapped to RGB; zero vectors are black. Colors are not semantic labels.",
                              completed_chunks={}, complete=False),
    }


def convert(source: Path, output: Path, *, array="dense", device="cpu", threads=16,
            writer_workers=16, batch_chunks=16, sample_chunks=1024, sample_voxels=500_000,
            seed=0, levels=6, resume=False):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the source group")
    if min(threads, writer_workers, batch_chunks, sample_chunks, sample_voxels, levels) < 1:
        raise ValueError("thread, batch, sample and level counts must be positive")
    root = source / array
    meta = json.loads((root / ".zarray").read_text())
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    if (len(shape) != 4 or shape[-1] < 3 or chunks[-1] != shape[-1]
            or meta.get("dimension_separator") != "/" or meta.get("fill_value") != 0):
        raise ValueError("requires zero-filled ZYXC input with '/' separator and all channels per chunk")
    if Path(str(root) + "_scale").exists():
        raise ValueError("per-vector scale arrays are not supported; use dense or another fixed-scale array")
    source_attrs = json.loads((source / ".zattrs").read_text())
    if source_attrs.get("complete") is False:
        raise ValueError("finish embedding inference before exporting its colors")
    array_attrs = json.loads((root / ".zattrs").read_text()) if (root / ".zattrs").exists() else {}
    indices = stored_chunks(root)
    if not indices:
        raise ValueError("source has no stored embedding chunks")
    fingerprint = hashlib.sha256(json.dumps([meta, array_attrs, source_attrs], sort_keys=True).encode()
                                 + np.array(indices, dtype="<i8").tobytes()).hexdigest()
    reader = open_zarr(str(root), threads=threads, memory_cache=0)
    print(f"Source: {len(indices):,} stored chunks; shape {shape}", flush=True)
    with output_lock(output):
        if resume and (output / ".zattrs").exists():
            attrs = json.loads((output / ".zattrs").read_text())
            state = attrs["embedding_rgb"]
            if (state["source_fingerprint"] != fingerprint or state["source"] != str(source)
                    or state["array"] != array or len(attrs["multiscales"][0]["datasets"]) != levels):
                raise ValueError("resume source or pyramid levels differ")
            if state["complete"]:
                print(f"Already complete: {output}", flush=True)
                return attrs
            writer = ZarrGroupWriter(output, attrs=None, workers=writer_workers)
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"{output} exists; use --resume")
            print("Fitting global PCA from spatially distributed nonzero samples...", flush=True)
            colors = fit_colors(reader, indices, chunks, sample_chunks=sample_chunks,
                                sample_voxels=sample_voxels, seed=seed,
                                value_scale=float(array_attrs.get("value_scale", 1.)))
            attrs = ome_metadata(source, array, source_attrs, shape, chunks, colors, levels, fingerprint)
            writer = ZarrGroupWriter(output, attrs=attrs, workers=writer_workers)
        state = attrs["embedding_rgb"]
        projector = ColorProjector(state["pca"], device)
        print(f"PCA: {state['pca']['sample_count']:,} voxels; explained variance "
              f"{sum(state['pca']['explained_variance_ratio']):.1%}", flush=True)
        started = time.monotonic()
        try:
            for level in range(levels):
                name = str(level)
                spatial_shape = tuple(-(-s // (2 ** level)) for s in shape[:3])
                out_shape, out_chunks = (3, *spatial_shape), (3, *chunks[:3])
                if name not in writer.arrays:
                    writer.create(name, out_shape, out_chunks, "uint8", codec="vc-delta3d",
                                  channel_axis="first", attrs={"_ARRAY_DIMENSIONS": list("czyx")})
                elif writer.arrays[name] != (out_shape, out_chunks, np.dtype("uint8")):
                    raise ValueError(f"resume schema differs at level {level}")
                if level:
                    indices = sorted({tuple(i // 2 for i in index) for index in indices})
                    reader = open_zarr(str(output / str(level - 1)), threads=threads, memory_cache=0)
                done = state["completed_chunks"].get(name, 0)
                if done == len(indices):
                    continue
                print(f"Level {level}: {done:,}/{len(indices):,} chunks", flush=True)
                level_start, initial, last_report = time.monotonic(), done, time.monotonic()

                def read_batch(batch):
                    if level == 0:
                        regions = [(tuple(i * c for i, c in zip(index, chunks)) + (0,), chunks)
                                   for index in batch]
                    else:
                        regions = [((0, *[i * c * 2 for i, c in zip(index, chunks)]),
                                    (3, *[c * 2 for c in chunks[:3]])) for index in batch]
                    return reader.read_regions(regions)

                with ThreadPoolExecutor(1) as prefetch:
                    pending = prefetch.submit(read_batch, indices[done:done + batch_chunks])
                    while done < len(indices):
                        batch = indices[done:done + batch_chunks]
                        blocks = pending.result()
                        next_done = done + len(batch)
                        if next_done < len(indices):
                            pending = prefetch.submit(read_batch, indices[next_done:next_done + batch_chunks])
                        if level == 0:
                            colors = projector(blocks)
                            outputs = [np.moveaxis(rgb, -1, 0) for rgb in colors]
                        else:
                            outputs = downsample_colors(blocks, device)
                        for index, block in zip(batch, outputs):
                            start = tuple(i * c for i, c in zip(index, chunks))
                            size = tuple(min(c, n - s) for c, n, s in zip(chunks, spatial_shape, start))
                            writer.write(name, (0, *start), block[(slice(None), *[slice(0, s) for s in size])])
                        done = next_done
                        now = time.monotonic()
                        if now - last_report > 20 or done == len(indices):
                            writer.flush()
                            state["completed_chunks"][name] = done
                            atomic_json(output / ".zattrs", attrs)
                            speed = (done - initial) / max(now - level_start, 1e-6)
                            print(f"Level {level}: {done:,}/{len(indices):,} ({done / len(indices):.1%}), "
                                  f"{speed:.1f} chunks/s, ETA {(len(indices) - done) / max(speed, 1e-6) / 60:.1f} min", flush=True)
                            last_report = time.monotonic()
            writer.flush()
            state["complete"] = True
            state["elapsed_seconds_last_run"] = time.monotonic() - started
            atomic_json(output / ".zattrs", attrs)
            print(f"Complete: {output}", flush=True)
        finally:
            writer.close()
    return attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array", default="dense")
    parser.add_argument("--device", default="cpu", help="cpu or cuda for PCA projection and pyramid averaging")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--writer-workers", type=int, default=16)
    parser.add_argument("--batch-chunks", type=int, default=16)
    parser.add_argument("--sample-chunks", type=int, default=1024)
    parser.add_argument("--sample-voxels", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    convert(args.input, args.output, array=args.array, device=args.device, threads=args.threads,
            writer_workers=args.writer_workers, batch_chunks=args.batch_chunks,
            sample_chunks=args.sample_chunks, sample_voxels=args.sample_voxels,
            seed=args.seed, levels=args.levels, resume=args.resume)


if __name__ == "__main__":
    main()
