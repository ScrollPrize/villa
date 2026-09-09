"""Backbone-only inference on flattened Zarr volumes, with exclusive output tiles.

Every tile uses the same global overlapping patch lattice, including contributions
from outside its boundary. Workers never share output storage chunks. Probability
blending happens in float32 on the GPU; only the final result is quantized.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
import zarr

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.inference.infer import compute_importance_map_2d
from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_model_state
from vesuvius.ink_detection.models.model import make_model


def axis_starts(length, patch, stride):
    if not 0 < stride <= patch or length <= 0:
        raise ValueError("Invalid patch lattice")
    last = max(0, length - patch)
    return sorted(set([*range(0, last + 1, stride), last]))


def depth_support(shape, patch, depth_mode="sliding"):
    """Return the evaluated half-open depth interval in the original volume."""
    if depth_mode == "sliding":
        return 0, shape[0]
    if depth_mode == "centered":
        start = max(0, shape[0] // 2 - patch[0] // 2)
        return start, min(shape[0], start + patch[0])
    raise ValueError(f"Unknown depth mode: {depth_mode}")


def evaluated_depth_support(shape, patch, depth_mode='sliding', margin=0, source_depth_reversed=False):
    start, stop = depth_support(shape, patch, depth_mode)
    if margin:
        if depth_mode != 'centered' or not 0 <= margin < (stop-start)//2:
            raise ValueError('Depth margins require a nonempty centered prediction window')
        start, stop = start+margin, stop-margin
    return (shape[0]-stop, shape[0]-start) if source_depth_reversed else (start, stop)


class ReversedDepthVolume:
    """Lazy full-segment preprocessing, before any crop or patch lattice."""
    def __init__(self, source):
        self.source, self.shape = source, source.shape

    def __getitem__(self, slices):
        z, y, x = slices
        start, stop, step = z.indices(self.shape[0])
        if step != 1:
            raise ValueError('Depth reads must use unit steps')
        return np.asarray(self.source[self.shape[0]-stop:self.shape[0]-start, y, x])[::-1].copy()


class Canonical3DOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.forward_3d(image)


def tile_origins(shape, patch, stride, bounds, depth_mode="sliding"):
    y0, y1, x0, x1 = bounds
    axes = [axis_starts(n, p, s) for n, p, s in zip(shape, patch, stride, strict=True)]
    supported_start, _ = depth_support(shape, patch, depth_mode)
    if depth_mode == "centered":
        axes[0] = [supported_start]
    axes[1] = [y for y in axes[1] if y < y1 and y + patch[1] > y0]
    axes[2] = [x for x in axes[2] if x < x1 and x + patch[2] > x0]
    return list(itertools.product(*axes))


def overlap_slices(origin, patch, shape, bounds):
    y0, y1, x0, x1 = bounds
    lo = (0, y0, x0)
    hi = (shape[0], y1, x1)
    start = tuple(max(a, b) for a, b in zip(origin, lo, strict=True))
    stop = tuple(min(a + p, b) for a, p, b in zip(origin, patch, hi, strict=True))
    target = tuple(slice(a - b, c - b) for a, b, c in zip(start, lo, stop, strict=True))
    source = tuple(slice(a - b, c - b) for a, b, c in zip(start, origin, stop, strict=True))
    return target, source


def blend_window(patch):
    xy = compute_importance_map_2d(patch_size=tuple(patch[1:]), mode="hann")
    z = torch.hann_window(patch[0], periodic=False).clamp_min(0.001)
    return z[:, None, None] * xy[None]


def load_predictor_model(plan, device):
    payload = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if payload.get("config", {}).get("projection", {}).get("kind") != "canonical_logits":
        raise ValueError("Require a canonical student checkpoint")
    update = payload.get("optimizer_step", payload.get("teacher_source", {}).get("optimizer_update"))
    if update != plan["expected_update"]:
        raise ValueError("Checkpoint update differs from the inference plan")
    if "ema_model" not in payload:
        raise ValueError("EMA weights are required for released inference")
    config = config_from_checkpoint(payload, source=plan["checkpoint"])
    model = make_model(config)
    load_model_state(model, payload["ema_model"])
    model.requires_grad_(False).eval().to(device)
    return Canonical3DOnly(model), NormalizationConfig.from_value(payload["config"]["image_normalization"])


class SurfacePredictor:
    def __init__(self, model, normalization, plan, device):
        self.model, self.normalization, self.plan, self.device = model, normalization, plan, device
        self.patch = tuple(plan["patch_zyx"])
        self.stride = tuple(plan["stride_zyx"])
        self.depth_mode = plan.get("depth_mode", "sliding")
        self.prediction_mode = plan.get("prediction_mode", "ink3d")
        if self.prediction_mode != "ink3d":
            raise ValueError("Only 3D ink probabilities are supported")
        self.window = blend_window(self.patch).to(device)
        self.depth_margin = int(plan.get('valid_depth_margin', 0))
        if self.depth_margin:
            evaluated_depth_support(self.patch, self.patch, self.depth_mode, self.depth_margin)
            self.window[:self.depth_margin] = 0
            self.window[-self.depth_margin:] = 0
        self.column_window = self.window[self.patch[0] // 2]
        self.normalizers = ThreadPoolExecutor(max_workers=plan.get("normalization_workers", 4))
        self.prefetcher = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.prefetcher.shutdown()
        self.normalizers.shutdown()

    def _normalize_batch(self, raw, offset, origins):
        def normalize(origin):
            start = tuple(a - b for a, b in zip(origin, offset, strict=True))
            crop = raw[tuple(slice(a, a + p) for a, p in zip(start, self.patch, strict=True))]
            if not crop.any():
                return None
            # Pad only beyond actual volume edges, never at an output tile boundary.
            if crop.shape != self.patch:
                crop = np.pad(crop, [(0, p - n) for p, n in zip(self.patch, crop.shape, strict=True)])
            support = np.any(crop != 0, axis=0)
            return normalize_image(crop, self.normalization), support
        values = list(self.normalizers.map(normalize, origins))
        indices = [i for i, value in enumerate(values) if value is not None]
        if not indices:
            return indices, None, None
        images = torch.from_numpy(np.stack([values[i][0] for i in indices])[:, None])
        supports = torch.from_numpy(np.stack([values[i][1] for i in indices]))
        if self.device.type == "cuda":
            images, supports = images.pin_memory(), supports.pin_memory()
        return indices, images, supports

    @torch.inference_mode()
    def predict_tile(self, volume, bounds):
        started = time.perf_counter()
        shape = tuple(volume.shape)
        origins = tile_origins(shape, self.patch, self.stride, bounds, self.depth_mode)
        za, zb = depth_support(shape, self.patch, self.depth_mode)
        ya, xa = min(o[1] for o in origins), min(o[2] for o in origins)
        yb = min(shape[1], max(o[1] for o in origins) + self.patch[1])
        xb = min(shape[2], max(o[2] for o in origins) + self.patch[2])
        raw = np.asarray(volume[za:zb, ya:yb, xa:xb])
        read_seconds = time.perf_counter() - started
        y0, y1, x0, x1 = bounds
        output_shape = (shape[0], y1 - y0, x1 - x0)
        if not raw.any():
            return np.zeros(output_shape, np.uint8), {"patches": 0, "skipped_patches": len(origins),
                    "read_seconds": read_seconds, "seconds": time.perf_counter() - started}
        numerator = torch.zeros(output_shape, device=self.device)
        denominator = torch.zeros_like(numerator)
        slices = [overlap_slices(o, self.patch, shape, bounds) for o in origins]
        for target, source in slices:
            denominator[target].add_(self.window[source])
        valid_start, valid_stop = evaluated_depth_support(shape, self.patch, self.depth_mode, self.depth_margin)
        if not (denominator[valid_start:valid_stop] > 0).all():
            raise ValueError("Patch lattice leaves evaluated voxels uncovered")
        # Outer depths of a centered pass are unevaluated zero placeholders.
        # Output metadata must expose [za, zb) as the valid depth interval.
        denominator.masked_fill_(denominator == 0, 1)
        batch_size = self.plan["batch_size"]
        batches = [origins[i:i + batch_size] for i in range(0, len(origins), batch_size)]
        pending = self.prefetcher.submit(self._normalize_batch, raw, (za, ya, xa), batches[0])
        predicted = 0
        for batch_index, batch_origins in enumerate(batches):
            indices, images, supports = pending.result()
            if batch_index + 1 < len(batches):
                pending = self.prefetcher.submit(self._normalize_batch, raw, (za, ya, xa), batches[batch_index + 1])
            if images is None:
                continue
            images = images.to(self.device, non_blocking=True)
            if self.plan.get("channels_last", False):
                images = images.contiguous(memory_format=torch.channels_last_3d)
            supports = supports.to(self.device, non_blocking=True)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            with context:
                logits = self.model(images)
            if tuple(logits.shape) != (len(indices), 1, *self.patch):
                raise ValueError(f"Unexpected 3D output shape: {logits.shape}")
            probabilities = logits[:, 0].float()
            if self.prediction_mode == "ink3d":
                probabilities = probabilities.sigmoid()
            probabilities.mul_(supports[:, None])
            probabilities.mul_(self.window)
            for i, local in enumerate(indices):
                target, source = slices[batch_index * batch_size + local]
                numerator[target].add_(probabilities[i][source])
            predicted += len(indices)
        probabilities = numerator.div_(denominator)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite blended probabilities or uncovered voxels")
        result = probabilities.clamp_(0, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()
        return result, {"patches": predicted, "skipped_patches": len(origins) - predicted,
                "read_seconds": read_seconds, "seconds": time.perf_counter() - started}


def connect_queue(root):
    connection = sqlite3.connect(str(Path(root) / "jobs.sqlite"), timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def worker(plan_path, gpu):
    import os
    import numcodecs
    plan = json.loads(Path(plan_path).read_text())
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    numcodecs.blosc.set_nthreads(1)
    if int(zarr.__version__.split(".")[0]) >= 3:
        zarr.config.set({"async.concurrency": 8, "threading.max_workers": 4})
    device = torch.device("cuda:0")
    model, normalization = load_predictor_model(plan, device)
    if plan.get("channels_last", False):
        model.to(memory_format=torch.channels_last_3d)
    predictor = SurfacePredictor(model, normalization, plan, device)
    connection = connect_queue(plan["root"])
    datasets = {d["id"]: d for d in plan["datasets"]}
    arrays = {}
    try:
        while True:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT id,dataset,y0,y1,x0,x1 FROM jobs WHERE status='pending' "
                                     "AND dataset IN (SELECT id FROM datasets WHERE ready=1) ORDER BY id LIMIT 1").fetchone()
            if job is None:
                remaining = connection.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]
                connection.commit()
                if not remaining:
                    break
                time.sleep(3)
                continue
            job_id, name, *bounds = job
            connection.execute("UPDATE jobs SET status='running',owner=?,started=? WHERE id=?",
                               (os.getpid(), time.time(), job_id))
            connection.commit()
            try:
                if name not in arrays:
                    source = zarr.open_array(str(Path(datasets[name]["input"]) / "0"), mode="r")
                    if datasets[name].get('reverse_depth', False):
                        source = ReversedDepthVolume(source)
                    arrays[name] = (source, zarr.open_array(str(Path(datasets[name]["output"]) / "0"), mode="r+"))
                source, output = arrays[name]
                result, metrics = predictor.predict_tile(source, bounds)
                if datasets[name].get('reverse_depth', False):
                    result = result[::-1].copy()
                t = time.perf_counter()
                y0, y1, x0, x1 = bounds
                output[:, y0:y1, x0:x1] = result
                metrics.update(write_seconds=time.perf_counter() - t, gpu=gpu,
                               voxels=int(result.size), nonzero_voxels=int(np.count_nonzero(result)),
                               value_sum=int(result.sum(dtype=np.uint64)), maximum=int(result.max()))
                connection.execute("UPDATE jobs SET status='done',finished=?,metrics=? WHERE id=?",
                                   (time.time(), json.dumps(metrics), job_id))
                connection.commit()
                print(json.dumps({"job": job_id, "dataset": name, **metrics}), flush=True)
            except BaseException as error:
                connection.execute("UPDATE jobs SET status='failed',metrics=? WHERE id=?",
                                   (json.dumps({"error": repr(error)}), job_id))
                connection.commit()
                raise
    finally:
        predictor.close()
        connection.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    worker(args.plan, args.gpu)
