"""Process JEM view sets on CPU or CUDA and build the batch the JEM method consumes.

Per view (independently seeded): RicianNoise (full recipe) -> normalization -> scanner
effects -> exact orientation (Transpose/Flip voxel reindexing) -> intensity recipe
without Defocus, GaussianBlur and LowResolution -> (students) zero-filled masked cells.
No resampling: crops keep the selected level's native voxel grid.

Batch rows (B sources, S=2 sets, L locals per set):
    teacher row  s*B + b         global row s*B + b (slot s)
    local row    (s*L + j)*B + b (slot s*L + j)
Correspondence tensors index the teacher row of the same set, in final oriented
token order of both views: targets[r, i] = flat teacher token or -1.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import torch

from dinovol_2.dataset.cuda_pipeline import BatchProcessor, normalization_transforms, spatial_transforms
from dinovol_2.dataset.jem_views import zyx

JEM_EXCLUDED_OPERATIONS = ("Defocus", "GaussianBlur", "LowResolution")

DEFAULT_MASKING = {
    "global": {"mode": "multiblock", "probability": 0.70, "fraction": 0.55, "min_block_tokens": 4,
               "max_block_fraction": 0.5, "min_aspect": 0.3, "attempts": 10, "cyclic_shift": True},
    "local": {"mode": "random", "probability": 0.70, "fraction": 0.40},
}


def resolve_masking(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    config = dict(config or {})
    unknown = set(config) - {"global", "local"}
    if unknown:
        raise ValueError(f"Unknown JEM masking crop types: {sorted(unknown)}")
    resolved = {}
    for kind, defaults in DEFAULT_MASKING.items():
        values = {**defaults, **dict(config.get(kind) or {})}
        if values["mode"] not in ("multiblock", "random"):
            raise ValueError(f"JEM {kind} mask mode must be 'multiblock' or 'random'")
        if not 0.0 <= float(values["probability"]) <= 1.0 or not 0.0 <= float(values["fraction"]) <= 1.0:
            raise ValueError(f"JEM {kind} mask probability and fraction must lie in [0, 1]")
        resolved[kind] = values
    return resolved


def patch_masks(spec_grid, seeds, settings) -> dict[str, torch.Tensor]:
    """Native seeded patch masks (host generation; identical for either training backend)."""
    from vc_reader import views
    n_tokens = math.prod(spec_grid)
    kwargs = {"mode": settings["mode"], "probability": float(settings["probability"]),
              "fraction": float(settings["fraction"])}
    if settings["mode"] == "multiblock":
        kwargs.update(min_block_tokens=int(settings["min_block_tokens"]),
                      max_block_tokens=max(1, int(math.floor(float(settings["max_block_fraction"]) * n_tokens))),
                      min_aspect=float(settings["min_aspect"]), attempts=int(settings["attempts"]),
                      cyclic_shift=bool(settings["cyclic_shift"]))
    result = views.patch_masks(tuple(spec_grid), seeds.numpy().astype("uint64"), **kwargs)
    return {"masks": torch.as_tensor(result["masks"]).bool().reshape(len(seeds), -1),
            "applied": torch.as_tensor(result["applied"]).bool(),
            "counts": torch.as_tensor(result["counts"]).long()}


class JemBatchProcessor(BatchProcessor):
    def __init__(self, config, *, device, patch: int, masking=None, lattice: bool = False,
                 storage: str = "device", dtype: torch.dtype = torch.float32, group_size: int | None = None):
        super().__init__(config, None, device=device)
        self.patch = zyx(patch, "patch")
        self.masking = resolve_masking(masking)
        self.lattice = bool(lattice)
        if storage not in ("device", "host"):
            raise ValueError("JEM batch storage must be 'device' or 'host'")
        self.storage = storage
        self.dtype = dtype
        # Sources processed together; host storage defaults to small groups to bound device memory.
        self.group_size = int(group_size) if group_size is not None else (4 if storage == "host" else 1 << 30)

    def recipe(self, target) -> dict[str, Any]:
        """Resolved per-view operations, for the run recipe and checkpoints."""
        scanner = self.config.get("augmentation", "full") == "full"
        return {
            "prepare": (["RicianNoise"] if scanner else []) + ["Normalize"] + (
                ["Smear", "SliceIllumination"] if scanner else []),
            "orientation": ["Transpose", "Flip"],
            "intensity": [name for name, _ in self.intensity_recipe(target) if name not in JEM_EXCLUDED_OPERATIONS],
            "excluded": list(JEM_EXCLUDED_OPERATIONS),
            "masking": self.masking,
        }

    def _process(self, views):
        """Process CropViews; returns tensors in input order."""
        outputs = [None] * len(views)
        groups = defaultdict(list)
        raws = {}
        for index, view in enumerate(views):
            raw = view.raw
            if tuple(raw.shape) != tuple(view.target):
                # Scaled student crop: resize to the target shape, voxel centers aligned, in FP32.
                raws[index] = self.resize(raw, tuple(view.target))
            groups[(tuple(view.target), view.augment, index in raws)].append(index)
        oriented = defaultdict(list)
        for (shape, augment, _), indices in groups.items():
            scanner = augment and self.config.get("augmentation", "full") == "full"
            pipeline = self.pipeline(("jem-prepare", shape, scanner), lambda: [
                *([self.A.RicianNoise(std=(1, 4), p=.10)] if scanner else []),
                *normalization_transforms(self.config.get("normalization_scheme", "robust"),
                                          self.config.get("intensity_properties")),
                *(self.scanner(shape) if scanner else [])])
            raw = torch.stack([raws[i] if i in raws else views[i].raw for i in indices]).unsqueeze(1)
            images = self.run(pipeline, raw, [views[i].seed for i in indices])
            for i, image in zip(indices, images):
                oriented[(shape, views[i].spatial, augment)].append((i, image))
        intensity = defaultdict(list)
        for (_, operations, augment), entries in oriented.items():
            images = torch.stack([image for _, image in entries])
            if operations:
                pipeline = self.pipeline(("spatial", operations), lambda: spatial_transforms(operations))
                images = self.run(pipeline, images, [views[i].seed for i, _ in entries])
            for (i, _), image in zip(entries, images):
                intensity[(tuple(image.shape[1:]), augment)].append((i, image))
        for (shape, augment), entries in intensity.items():
            images = torch.stack([image for _, image in entries])
            if augment:
                pipeline = self.pipeline(("jem-intensity", shape), lambda: self.intensity(
                    shape, exclude=JEM_EXCLUDED_OPERATIONS))
                images = self.run(pipeline, images, [views[i].seed + 1 for i, _ in entries])
            for (i, _), image in zip(entries, images):
                outputs[i] = image
        return outputs

    def resize(self, raw: torch.Tensor, shape) -> torch.Tensor:
        """Trilinear resize of one crop to `shape` with voxel centers aligned (vc_reader Affine), FP32."""
        source = tuple(raw.shape)
        pipeline = self.pipeline(("jem-resize", source, shape), lambda: [self.A.Affine(
            shape=list(shape), border=True, scale_factors=[t / r for t, r in zip(shape, source)])])
        image = raw.to(self.device if self.backend == "cuda" else "cpu").float()[None, None]
        return self.run(pipeline, image, [0])[0, 0]

    def _empty(self, rows, size):
        pin = self.storage == "host" and self.device.type == "cuda"
        device = "cpu" if self.storage == "host" else self.device
        return torch.empty((rows, 1, *size), dtype=self.dtype, device=device, pin_memory=pin)

    def __call__(self, samples):
        from vc_reader import views as native
        B = len(samples)
        plans = [sample["jem_plan"] for sample in samples]
        n_sets = len(samples[0]["teacher_views"])
        L = len(samples[0]["local_views"]) // n_sets
        patch = self.patch
        slots = {"teacher": n_sets, "global": n_sets, "local": n_sets * L}
        field = {"teacher": "teacher_views", "global": "global_views", "local": "local_views"}
        # Orientations keep every crop's shape (cubes, or z-preserving with equal y/x), so the
        # token grid is the same before and after orientation.
        size = {kind: tuple(int(v) for v in samples[0][field[kind]][0].target) for kind in slots}
        grid = {kind: tuple(s // p for s, p in zip(size[kind], patch)) for kind in slots}
        device = self.device

        masks = {}
        for kind in ("global", "local"):
            seeds = torch.stack([plans[b][f"{kind}_mask_seeds"][v] for v in range(slots[kind]) for b in range(B)])
            masks[kind] = patch_masks(grid[kind], seeds, self.masking[kind])
            masks[kind]["device"] = masks[kind]["masks"].to(device)

        # Process a group of sources at a time and move its crops to their storage, so device
        # memory is bounded by the group rather than the whole batch. Rows: slot * B + b.
        outputs = {kind: self._empty(slots[kind] * B, size[kind]) for kind in slots}
        for start in range(0, B, self.group_size):
            group = range(start, min(start + self.group_size, B))
            rows = {kind: [v * B + b for v in range(slots[kind]) for b in group] for kind in slots}
            views = [samples[b][field[kind]][v] for kind in slots for v in range(slots[kind]) for b in group]
            processed = iter(self._process(views))
            for kind in slots:
                images = torch.stack([next(processed) for _ in rows[kind]])
                if kind != "teacher":
                    cells = masks[kind]["device"][rows[kind]].view(len(images), *grid[kind])
                    images = native.mask_patch_cells(images, cells, patch)
                outputs[kind][torch.tensor(rows[kind])] = images.to(outputs[kind].device, self.dtype)

        teacher_grid = grid["teacher"]
        batch: dict[str, Any] = {"jem": True, "batch_size": B, "n_sets": n_sets,
                                 "n_global_views": n_sets, "n_local_views": n_sets * L, "patch": patch}
        support = torch.stack([plan["support"] for plan in plans]).to(device)
        sources = torch.arange(B)
        for kind, crops, slots in (("global", outputs["global"], n_sets), ("local", outputs["local"], n_sets * L)):
            # Row r = slot * B + b; the teacher of slot v is set v // per_set.
            per_set = slots // n_sets
            slot = torch.arange(slots).repeat_interleave(B)
            source = sources.repeat(slots)
            teacher_rows = (slot // per_set) * B + source
            mask_tensor = masks[kind]["device"]
            student_origins = torch.stack([plans[b][f"{kind}_origins"][v] for v in range(slots) for b in range(B)])
            student_orient = torch.stack([plans[b][f"{kind}_orientations"][v] for v in range(slots) for b in range(B)])
            teacher_set = slot // per_set
            teacher_origins = torch.stack([plans[int(b)]["teacher_origins"][int(s)]
                                           for s, b in zip(teacher_set, source)])
            teacher_orient = torch.stack([plans[int(b)]["teacher_orientations"][int(s)]
                                          for s, b in zip(teacher_set, source)])
            scaled = f"{kind}_boxes" in plans[0]
            student_boxes = (torch.stack([plans[b][f"{kind}_boxes"][v] for v in range(slots) for b in range(B)])
                             if scaled else None)
            corr = native.correspondence(
                support, source, student_origins, grid[kind], student_orient,
                teacher_origins, teacher_grid, teacher_orient, lattice=self.lattice,
                student_boxes=student_boxes, patch=patch if scaled else None)
            batch[f"{kind}_crops"] = crops
            batch[f"{kind}_masks"] = mask_tensor
            batch[f"{kind}_mask_applied"] = masks[kind]["applied"]
            batch[f"{kind}_mask_counts"] = masks[kind]["counts"]
            batch[f"{kind}_targets"] = torch.as_tensor(corr["teacher_index"]).long()
            # Host copy of the overlapping-token count per row: the step sizes its compacted
            # overlap tensors from it without waiting for the GPU (prefetch runs this ahead).
            batch[f"{kind}_overlap_count"] = (batch[f"{kind}_targets"] >= 0).sum(-1).cpu()
            batch[f"{kind}_valid"] = torch.as_tensor(corr["student_valid"]).bool()
            if self.lattice:
                batch[f"{kind}_lattice"] = torch.as_tensor(corr["lattice"]).long()
            batch[f"{kind}_teacher_rows"] = teacher_rows
            batch[f"{kind}_slots"] = slot
            batch[f"{kind}_sources"] = source
            batch[f"{kind}_grid"] = grid[kind]
            batch[f"{kind}_orientations"] = student_orient
            batch[f"{kind}_overlap"] = torch.stack([plans[b][f"{kind}_overlap"][v]
                                                    for v in range(slots) for b in range(B)])
        batch["teacher_crops"] = outputs["teacher"]
        batch["teacher_grid"] = teacher_grid
        batch["teacher_orientations"] = torch.stack([plans[b]["teacher_orientations"][s]
                                                     for s in range(n_sets) for b in range(B)])
        batch["spacing"] = torch.stack([plan["spacing"] for plan in plans])
        batch["plan_stats"] = {
            key: torch.stack([plan[key].reshape(-1).double() for plan in plans])
            for key in ("global_attempts", "local_attempts", "teacher_attempts", "rejections", "source_resamples")
        }
        batch["plans"] = plans
        return batch
