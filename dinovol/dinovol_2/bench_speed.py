from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/ephemeral/sean/dinovol_perf/code")

import numpy as np
import torch

from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.pretrain import DinoIBOTPretrainer


def bench_gpu(config_path: str, batch_sizes: list[int]) -> None:
    torch._dynamo.config.recompile_limit = 32
    base_cfg = json.load(open(config_path))
    base_cfg["compile"]["enabled"] = True
    base_cfg["compile"]["mode"] = "default"
    base_cfg["dataset"]["object_mask"]["enabled"] = False
    base_cfg["dataset"]["occupancy_level"] = None
    base_cfg["dataset"]["object_points"]["use_metadata_meshes"] = False
    base_cfg["dataset"]["object_points"]["mesh_cache_dir"] = None
    base_cfg["dataset"]["datasets"] = base_cfg["dataset"]["datasets"][:1]
    base_cfg["num_workers"] = 0
    base_cfg["grad_accumulation_steps"] = 1
    n_global = int(base_cfg["dataset"].get("num_global_crops", 2))
    n_local = int(base_cfg["dataset"].get("num_local_crops", 2))

    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = json.loads(json.dumps(base_cfg))
        cfg["batch_size"] = bs
        try:
            trainer = DinoIBOTPretrainer(cfg)
            B = bs
            batch = {
                "collated_global_crops": torch.randn(B * n_global, 1, 128, 128, 128),
                "collated_local_crops": torch.randn(B * n_local, 1, 64, 64, 64),
                "collated_masks": torch.zeros(B * n_global, 4096, dtype=torch.bool),
                "mask_indices_list": torch.arange(B * n_global * 1024, dtype=torch.long),
                "masks_weight": torch.ones(B * n_global * 1024, dtype=torch.float32) / 1024.0,
                "n_global_views": n_global,
                "n_local_views": n_local,
                "n_masked_patches": torch.tensor([B * n_global * 1024], dtype=torch.long),
                "global_token_targets": torch.rand(B * 2, 4096, 3) * 127.0,
                "object_point_coordinates": torch.rand(B * 2, 384, 3) * 127.0,
                "object_point_valid": torch.ones(B * 2, 384, dtype=torch.bool),
                "object_point_source": torch.rand(B, 384, 3) * 191.0,
                "object_point_objects": torch.randint(0, 13, (B, 384), dtype=torch.long),
                "object_point_kinds": torch.randint(0, 3, (B, 384), dtype=torch.int8),
                "object_negatives": torch.ones(B, 13, 13, dtype=torch.bool),
                "object_point_ink": torch.randint(-1, 2, (B, 384), dtype=torch.int8),
            }
            batch["collated_masks"][:, :1024] = True
            for step in range(3):
                trainer.train_step(batch, 160001 + step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(10):
                trainer.train_step(batch, 160010 + step)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10.0
            mem = torch.cuda.max_memory_allocated() / (1024**3)
            res = torch.cuda.max_memory_reserved() / (1024**3)
            print(
                f"COMPILED BS={bs} grad_accum=1 (no ckpt): {dt*1000:.1f} ms/it, "
                f"alloc={mem:.2f} GiB, reserved={res:.2f} GiB",
                flush=True,
            )
        except Exception as exc:
            exc.__traceback__ = None
            print(f"COMPILED BS={bs} failed: {exc}", flush=True)
        finally:
            trainer = None
            batch = None
            gc.collect()
            torch.cuda.empty_cache()


def bench_dataset_breakdown(config_path: str) -> None:
    from dinovol_2.dataset.object_labels import crop_mesh_objects

    cfg = json.load(open(config_path))
    ds_cfg = cfg["dataset"]
    ds_cfg["object_mask"]["enabled"] = False
    ds_cfg["occupancy_level"] = None
    ds = SSLZarrDataset(ds_cfg)
    for vol in ds.volumes:
        if vol.meshes:
            name = Path(vol.path).parent.parent.name
            starts = ds._sample_mesh_centered_start(vol) or ds._sample_source_start(vol.usable_bbox)
            lo = np.asarray(starts, dtype=np.float32) + 20.0
            hi = lo + 128.0
            t0 = time.perf_counter()
            sheets, negs = crop_mesh_objects(vol.meshes, lo, hi)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"mesh crop {name} ({len(vol.meshes)} meshes): {dt:.2f}ms -> {len(sheets)} sheets", flush=True)
    for i in range(6):
        t0 = time.perf_counter()
        plan, source_crop = ds._next_source_crop(0.30)
        t1 = time.perf_counter()
        vol = plan.volume
        crop_regions = [
            ds._sample_global_crop_region(source_crop, vol, None, None)
            for _ in ds.global_transforms
        ]
        obj_pts = ds._sample_object_points(vol, plan.starts, crop_regions)
        t2 = time.perf_counter()
        print(
            f"sample {i}: vol={Path(vol.path).parent.parent.name} meshes={bool(vol.meshes)} "
            f"surf={bool(vol.surface_path)} fib={bool(vol.fiber_paths)} "
            f"src_read={(t1-t0)*1000:.1f}ms obj_pts={(t2-t1)*1000:.1f}ms",
            flush=True,
        )


if __name__ == "__main__":
    from pathlib import Path
    import sys

    cfg_path = "/ephemeral/sean/dinovol_perf/runs/pretrain_8_9um_objects_meshes_ink_160k/config.json"
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("data", "both"):
        bench_dataset_breakdown(cfg_path)
    if mode in ("gpu", "both"):
        bs_list = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [10]
        bench_gpu(cfg_path, bs_list)
