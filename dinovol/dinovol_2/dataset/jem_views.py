"""JEM view sets: patch-lattice crop placement, exact orientations and per-view seeds.

One independent source sample yields two view sets. Each set has one teacher
global, one student global and `locals_per_set` student locals, all on the
selected pyramid level's native voxel grid (no resizing, zoom or aspect change).
The placement window is a nominal source region; crops are planned on its patch
lattice by vc_reader (`views.plan_view_sets`), which accepts a student only when
it shares at least `min_overlap_fraction` of its tokens (and `min_overlap_count`)
with its teacher on valid source support. Only then are orientations drawn: each
of the 12 crops independently gets one of the 48 signed axis permutations. The
union box of the planned crops is read once and every crop is cut from it.

Validity: a voxel is valid when it lies inside the stored array (fill or padding
is invalid; CT air is valid). Windows are drawn inside the usable bounds, so the
support is complete unless a window reaches past the array edge.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Mapping

import numpy as np
import torch

from dinovol_2.dataset.cuda_pipeline import CropView

N_SETS = 2


def _cube(value, name):
    if isinstance(value, int):
        return value
    values = [int(v) for v in value]
    if len(values) != 3 or len(set(values)) != 1:
        raise ValueError(f"JEM {name} must be a cube (one size or three equal sizes), got {value}")
    return values[0]


@dataclass(frozen=True)
class JemViewSpec:
    patch: int = 8
    source_size: int = 512
    teacher_size: int = 256
    global_size: int = 256
    local_size: int = 144
    locals_per_set: int = 4
    min_overlap_fraction: float = 0.10
    min_overlap_count: int = 2
    max_student_attempts: int = 64
    max_teacher_attempts: int = 16
    max_source_attempts: int = 64

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, patch_size) -> "JemViewSpec":
        config = dict(config)
        unknown = set(config) - set(cls.__dataclass_fields__) - {"patch"}
        if unknown:
            raise ValueError(f"Unknown JEM view settings: {sorted(unknown)}")
        patch = _cube(patch_size, "patch_size")
        if "patch" in config and int(config["patch"]) != patch:
            raise ValueError(f"JEM views patch {config['patch']} must equal the backbone patch size {patch}")
        values = {key: config[key] for key in cls.__dataclass_fields__ if key in config}
        for key in ("source_size", "teacher_size", "global_size", "local_size"):
            if key in values:
                values[key] = _cube(values[key], key)
        spec = cls(**{**values, "patch": patch})
        spec.validate()
        return spec

    def validate(self) -> None:
        for name in ("source_size", "teacher_size", "global_size", "local_size"):
            size = getattr(self, name)
            if size <= 0 or size % self.patch:
                raise ValueError(f"JEM {name}={size} must be a positive multiple of the patch size {self.patch}")
        if max(self.teacher_size, self.global_size, self.local_size) > self.source_size:
            raise ValueError("JEM crops must fit inside the source placement window")
        if not 0.0 < self.min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be in (0, 1]")
        if self.min_overlap_count < 2:
            raise ValueError("min_overlap_count must be at least 2 (structural pairs need two tokens)")
        if self.locals_per_set < 1:
            raise ValueError("locals_per_set must be positive")
        for name in ("max_student_attempts", "max_teacher_attempts", "max_source_attempts"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        # Smallest possible overlap of a contained/edge placement is checked by the planner;
        # reject configurations whose overlap floor can never be met.
        need = math.ceil(self.min_overlap_fraction * self.tokens(self.global_size))
        if need > self.tokens(min(self.global_size, self.teacher_size)):
            raise ValueError("min_overlap_fraction cannot be met by the global crop sizes")

    def tokens(self, size: int) -> int:
        return (size // self.patch) ** 3

    def grid(self, size: int) -> tuple[int, int, int]:
        return (size // self.patch,) * 3

    @property
    def student_shapes(self):
        return [self.grid(self.global_size)] + [self.grid(self.local_size)] * self.locals_per_set

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def orientation_operations(orientation: int) -> tuple:
    """vc_reader spatial operations for an orientation id: transpose by perm, then flip output axes."""
    from vc_reader import views
    perm, flips = views.orientation(int(orientation))
    operations = []
    if tuple(perm) != (0, 1, 2):
        operations.append(("transpose", "".join("zyx"[axis] for axis in perm)))
    axes = "".join("zyx"[k] for k, flip in enumerate(flips) if flip)
    if axes:
        operations.append(("flip", axes))
    return tuple(operations)


def _seeds(count: int) -> np.ndarray:
    return np.random.randint(0, 2**63 - 1, size=count, dtype=np.int64).astype(np.uint64)


def window_support(spec: JemViewSpec, starts, array_shape) -> np.ndarray:
    """Token support of the placement window: tokens whose voxels all lie inside the array."""
    grid = np.ones((spec.source_size // spec.patch,) * 3, dtype=np.uint8)
    for axis, (start, extent) in enumerate(zip(starts, array_shape)):
        first = np.arange(grid.shape[axis]) * spec.patch + int(start)
        inside = (first >= 0) & (first + spec.patch <= int(extent))
        shape = [1, 1, 1]
        shape[axis] = -1
        grid &= inside.reshape(shape).astype(np.uint8)
    return grid


def sample_view_sets(dataset, spec: JemViewSpec, *, augment: bool) -> dict[str, Any]:
    """Draw a source, plan both view sets, read their union box once and cut 12 crops."""
    from vc_reader import views

    nonzero_threshold = float(dataset.config.get("nonzero_threshold", 0.30))
    source_resamples = 0
    planner_failures = 0
    for _ in range(spec.max_source_attempts):
        plan = dataset._draw_source_plan(nonzero_threshold)
        if plan.axes is not None:
            raise ValueError("JEM needs axis-aligned source windows (source_rotation='none')")
        support = window_support(spec, plan.starts, plan.array.shape)
        planned = views.plan_view_sets(
            support, spec.grid(spec.teacher_size), spec.student_shapes, n_sets=N_SETS,
            min_overlap_fraction=spec.min_overlap_fraction, min_overlap_count=spec.min_overlap_count,
            max_student_attempts=spec.max_student_attempts, max_teacher_attempts=spec.max_teacher_attempts,
            seed=int(_seeds(1)[0]),
        )
        if not bool(planned["ok"]):
            planner_failures += 1
            source_resamples += 1
            continue
        teacher_origins = np.asarray(planned["teacher_origins"], dtype=np.int64)
        student_origins = np.asarray(planned["student_origins"], dtype=np.int64)
        sizes = [spec.teacher_size] + [spec.global_size] + [spec.local_size] * spec.locals_per_set
        boxes = []
        for s in range(N_SETS):
            boxes.append((teacher_origins[s] * spec.patch, spec.teacher_size))
            for index in range(1 + spec.locals_per_set):
                boxes.append((student_origins[s, index] * spec.patch, sizes[1 + index]))
        lo = np.min([origin for origin, _ in boxes], axis=0)
        hi = np.max([origin + size for origin, size in boxes], axis=0)
        starts = np.asarray(plan.starts, dtype=np.int64)
        region = dataset._read_region(plan.array, starts + lo, starts + hi)
        if region.size == 0:
            source_resamples += 1
            continue
        if not dataset.object_mask_enabled and np.count_nonzero(region) / region.size < nonzero_threshold:
            source_resamples += 1
            continue
        break
    else:
        raise RuntimeError(
            f"JEM could not place view sets after {spec.max_source_attempts} source draws "
            f"({planner_failures} planner failures); relax the overlap settings or check the volumes")

    def crop(origin, size):
        offset = origin - lo
        values = region[offset[0]:offset[0] + size, offset[1]:offset[1] + size, offset[2]:offset[2] + size]
        return torch.from_numpy(np.ascontiguousarray(values))

    n_views = N_SETS * (2 + spec.locals_per_set)
    orientations = np.asarray(views.draw_orientations(_seeds(n_views)), dtype=np.int64)
    intensity_seeds = _seeds(n_views)
    mask_seeds = _seeds(N_SETS * (1 + spec.locals_per_set))
    teacher_views, global_views, local_views = [], [], []
    teacher_orientation, global_orientation, local_orientation = [], [], []
    view = 0
    for s in range(N_SETS):
        for role, origin, size in [("teacher", teacher_origins[s] * spec.patch, spec.teacher_size),
                                   ("global", student_origins[s, 0] * spec.patch, spec.global_size)] + [
                ("local", student_origins[s, 1 + j] * spec.patch, spec.local_size)
                for j in range(spec.locals_per_set)]:
            raw = crop(origin, size)
            cropped = CropView(raw, (size,) * 3, orientation_operations(orientations[view]),
                               int(intensity_seeds[view] % (2**63 - 1)), augment)
            {"teacher": teacher_views, "global": global_views, "local": local_views}[role].append(cropped)
            {"teacher": teacher_orientation, "global": global_orientation,
             "local": local_orientation}[role].append(int(orientations[view]))
            view += 1

    volume = plan.volume
    spacing = dataset.volume_spacing(volume)
    student_overlap = np.asarray(planned["student_overlap"], dtype=np.int64)
    student_attempts = np.asarray(planned["student_attempts"], dtype=np.int64)
    local_slice = slice(1, 1 + spec.locals_per_set)
    return {
        "teacher_views": teacher_views,
        "global_views": global_views,
        "local_views": local_views,
        "jem_plan": {
            "support": torch.from_numpy(support),
            "teacher_origins": torch.from_numpy(teacher_origins),
            "global_origins": torch.from_numpy(student_origins[:, 0]),
            "local_origins": torch.from_numpy(student_origins[:, local_slice].reshape(-1, 3)),
            "teacher_orientations": torch.tensor(teacher_orientation),
            "global_orientations": torch.tensor(global_orientation),
            "local_orientations": torch.tensor(local_orientation),
            "global_mask_seeds": torch.from_numpy(mask_seeds.reshape(N_SETS, -1)[:, 0].astype(np.int64)),
            "local_mask_seeds": torch.from_numpy(mask_seeds.reshape(N_SETS, -1)[:, 1:].reshape(-1).astype(np.int64)),
            "global_overlap": torch.from_numpy(student_overlap[:, 0]),
            "local_overlap": torch.from_numpy(student_overlap[:, local_slice].reshape(-1)),
            "global_attempts": torch.from_numpy(student_attempts[:, 0]),
            "local_attempts": torch.from_numpy(student_attempts[:, local_slice].reshape(-1)),
            "teacher_attempts": torch.from_numpy(np.asarray(planned["teacher_attempts"], dtype=np.int64)),
            "rejections": torch.from_numpy(np.asarray(planned["rejections"], dtype=np.int64)),
            "source_resamples": torch.tensor(source_resamples),
            "spacing": torch.tensor(spacing, dtype=torch.float64),
            "window_start": torch.tensor(starts),
            "union_box": torch.from_numpy(np.stack([lo, hi])),
            "volume_index": torch.tensor(dataset.volumes.index(volume)),
            "level": torch.tensor(int(volume.scale)),
        },
    }
