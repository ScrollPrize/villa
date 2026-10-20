"""Benchmark CUDA data batches using fully cached local source windows.

Run --prepare once to inventory the partial cache, then measure in a fresh process.
Unavailable occupancy masks and label sources are excluded explicitly.
"""
from __future__ import annotations

from dinovol_2.training.config import as_legacy_config
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset, SourcePlan
from dinovol_2.dataset.cuda_pipeline import CudaDataLoader
from dinovol_2.ops.collate import build_dino_ibot_collate_fn


def seed(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)


def prepare(args):
    config = as_legacy_config(json.loads(args.config.read_text()))
    dc = config["dataset"]
    selected, inventory, starts = [], [], {}
    window = np.asarray(dc["source_sampling_size"])
    for original in dc["datasets"]:
        relative = original["volume_path"].split("vesuvius-challenge-open-data/", 1)[1]
        path = args.cache_root / relative
        level = path / str(original["volume_scale"])
        metadata = level / ".zarray"
        if not metadata.exists():
            continue
        meta = json.loads(metadata.read_text())
        chunks = np.asarray(meta["chunks"])
        assert meta["dimension_separator"] == "/" and len(chunks) == 3
        grid_shape = (np.asarray(meta["shape"]) + chunks - 1) // chunks
        present = np.zeros(grid_shape, dtype=bool)
        count = 0
        for z in level.iterdir():
            if not z.name.isdigit() or not z.is_dir():
                continue
            for y in z.iterdir():
                if not y.name.isdigit() or not y.is_dir():
                    continue
                for x in y.iterdir():
                    if x.name.isdigit() and x.is_file():
                        present[int(z.name), int(y.name), int(x.name)] = True
                        count += 1
        # A 192 window fits in two 128 chunks for offsets 0..64 inclusive.
        extent = (window + chunks - 1) // chunks
        complete = ndimage.minimum_filter(present, size=tuple(extent), origin=tuple(-(extent // 2)), mode="constant", cval=0)
        anchors = np.argwhere(complete) * chunks
        anchors = anchors[np.all(anchors + extent * chunks <= np.asarray(meta["shape"]), axis=1)]
        row = dict(sample=original["sample_id"], path=str(path), level=original["volume_scale"],
                   present_chunks=count, complete_neighborhoods=len(anchors), shape=meta["shape"],
                   chunks=meta["chunks"], max_offset=(extent * chunks - window).tolist())
        print("INVENTORY", json.dumps(row), flush=True)
        inventory.append(row)
        if not len(anchors):
            continue
        selected.append(dict(sample_id=original["sample_id"], volume_path=str(path),
                             volume_scale=original["volume_scale"], pixel_size_um=original["pixel_size_um"]))
        starts[str(path)] = (anchors.astype(np.int32), extent * chunks - window)
    if not selected:
        raise RuntimeError("No complete cached source windows")
    dc.update(datasets=selected, occupancy_level=None, object_mask={"enabled": False},
              vol_trim_pct=1.0, epoch_length=1000000,
              cross_view_token_patch_size=config["model"]["patch_size"])
    dc.setdefault("object_points", {}).update(use_metadata_meshes=False, mesh_cache_dir=None)
    dataset = CachedDataset(dc, do_augmentations=True)
    dataset.cached_starts = starts
    dataset.benchmark_seed = args.seed
    dataset.source_rng = None
    weights = [len(starts[v.path][0]) for v in dataset.volumes]
    for volume, weight in zip(dataset.volumes, weights):
        volume.weight = weight / sum(weights)
    dataset.close()
    with (args.output / "dataset.pkl").open("wb") as stream:
        pickle.dump(dataset, stream)
    (args.output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))


class CachedDataset(SSLZarrDataset):
    def _draw_source_plan(self, nonzero_threshold):
        if self.source_rng is None:
            worker = torch.utils.data.get_worker_info()
            self.source_rng = np.random.default_rng(self.benchmark_seed + (worker.id if worker else 0))
        rng = self.source_rng
        volume = self.volumes[int(rng.choice(len(self.volumes), p=[v.weight for v in self.volumes]))]
        anchors, max_offset = self.cached_starts[volume.path]
        start = anchors[int(rng.integers(len(anchors)))] + rng.integers(max_offset + 1)
        return SourcePlan(volume, self._get_volume_array(volume), tuple(map(int, start)), None)


def benchmark(args):
    seed(args.seed)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    config = json.loads((args.output / "resolved_config.json").read_text())
    with (args.output / "dataset.pkl").open("rb") as stream:
        dataset = pickle.load(stream)
    dataset.benchmark_seed, dataset.source_rng = args.seed, None
    workers = config["num_workers"] if args.workers is None else args.workers
    dataset.epoch_length = (args.warmup + args.batches + int(args.profile)) * config["batch_size"]
    collate = build_dino_ibot_collate_fn(dict(
        global_crop_size=config["dataset"]["global_crop_size"], patch_size=config["model"]["patch_size"],
        mask_ratio_min_max=config.get("mask_ratio_min_max", (0.1, 0.5)),
        mask_sample_probability=config.get("mask_sample_probability", 0.5), dtype=torch.float32))
    loader = CudaDataLoader(dataset, batch_size=config["batch_size"], num_workers=workers,
        collate_fn=collate, pin_memory=True,
        **(dict(prefetch_factor=config.get("prefetch_factor", 2), persistent_workers=True) if workers else {}))
    report = dict(command=[sys.executable, *sys.argv], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, batch_size=config["batch_size"], workers=workers,
        warmup=args.warmup, batches=args.batches, seed=args.seed,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        protocol="GPU-complete data batches; no model; startup/warmup excluded; local uncompressed partial cache; "
                 "same complete-neighborhood sampler, nonzero rejection, read-ahead, crop policy, object-point and iBOT collation; "
                 "no unavailable occupancy masks or labels; independently seeded augmentation distributions",
        prefetch_factor=config["prefetch_factor"], read_ahead=config["dataset"]["read_ahead"],
        image_residency="uint8 host crops -> CUDA normalization/resampling/augmentation/collation",
        times_seconds=[])
    output = args.output / f"{args.tag}.json"
    begin = time.perf_counter()
    iterator = iter(loader)
    try:
        for index in range(args.warmup + args.batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if index == args.warmup:
                measure_start = t0
            batch = next(iterator)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if index == 0:
                assert batch["collated_global_crops"].is_cuda
                assert batch["collated_local_crops"].is_cuda
                for key, value in batch.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        assert torch.isfinite(value).all(), key
                report["output_shapes"] = {k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}
                report["first_batch_seconds"] = time.perf_counter() - begin
            if index >= args.warmup:
                report["times_seconds"].append(elapsed)
            print("WARMUP" if index < args.warmup else "MEASURE", index, f"{elapsed:.4f}s", flush=True)
            del batch
        measured_wall_seconds = time.perf_counter() - measure_start
        times = report["times_seconds"]
        report.update(mean_seconds=statistics.mean(times), p50_seconds=float(np.percentile(times,50)),
                      p95_seconds=float(np.percentile(times,95)), min_seconds=min(times), max_seconds=max(times),
                      measured_wall_seconds=measured_wall_seconds,
                      iterations_per_second=len(times)/measured_wall_seconds,
                      samples_per_second=config["batch_size"]*len(times)/measured_wall_seconds,
                      peak_torch_cuda_bytes=torch.cuda.max_memory_allocated())
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                batch = next(iterator)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output / f"{args.tag}_trace.json"))
            (args.output / f"{args.tag}_profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
            del batch
        try:
            next(iterator)
            raise AssertionError("Unexpected extra benchmark batch")
        except StopIteration:
            pass
        report["status"] = "ok"
        print("RESULT", json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}), flush=True)
    except BaseException as exc:
        report.update(status="error", error=repr(exc))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        if workers:
            loader._iterator._shutdown_workers()
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dinovol_2/configs/pretrain_8_9um_heads32768.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data"))
    parser.add_argument("--output", type=Path, default=Path("perf_results/cuda_pipeline"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="run1")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
