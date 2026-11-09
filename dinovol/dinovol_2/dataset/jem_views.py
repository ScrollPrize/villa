"""JEM view sets: patch-lattice crop placement, exact orientations and per-view seeds.

One independent source sample yields two view sets. Each set has one teacher
global, one student global and `locals_per_set` student locals, all on the
selected pyramid level's native voxel grid. By default nothing is resized; with
`student_scale` [lo, hi] != [1, 1] every student crop instead covers a box of
round(s * size) voxels per axis (s log-uniform in [lo, hi], one per student view,
isotropic) at a random voxel offset and is resized to its size (voxel centers
aligned), so student and teacher tokens no longer cut the same voxels; each
student token is matched to the teacher token holding its center
(`views.correspondence` with student boxes). Teachers are never resized.
The placement window is a nominal source region; crops are planned on its patch
lattice by vc_reader (`views.plan_view_sets`), which accepts a student only when
it shares at least `min_overlap_fraction` of its tokens (and `min_overlap_count`)
with its teacher on valid source support. Only then are orientations drawn: each
of the 12 crops independently gets one of the 48 signed axis permutations
(`orientations: "all"`, cubic patches and crops only) or one of the 16 that keep
the z axis in place (`"z_preserving"`: flips and the y/x transpose; needed when
patches or crops differ between z and y/x, which must then have equal y and x
sizes so every orientation keeps the crop shape). Sizes and the patch are one
integer (a cube) or three ZYX integers. The union box of the planned crops is
read once and every crop is cut from it.

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


ORIENTATION_MODES = {"all": 48, "z_preserving": 16}  # orientation ids [0, n): z-preserving ids come first


def zyx(value, name="size") -> tuple[int, int, int]:
    """One integer (a cube) or three ZYX integers, as a ZYX tuple."""
    if isinstance(value, (int, np.integer)):
        return (int(value),) * 3
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"JEM {name} must be one integer or three ZYX integers, got {value}")
    return values


@dataclass(frozen=True)
class JemViewSpec:
    patch: tuple[int, int, int] = (8, 8, 8)
    source_size: tuple[int, int, int] = (512, 512, 512)
    teacher_size: tuple[int, int, int] = (256, 256, 256)
    global_size: tuple[int, int, int] = (256, 256, 256)
    local_size: tuple[int, int, int] = (144, 144, 144)
    orientations: str = "all"
    locals_per_set: int = 4
    min_overlap_fraction: float = 0.10
    min_overlap_count: int = 2
    student_scale: tuple[float, float] = (1.0, 1.0)
    max_student_attempts: int = 64
    max_teacher_attempts: int = 16
    max_source_attempts: int = 64

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, patch_size) -> "JemViewSpec":
        config = dict(config)
        unknown = set(config) - set(cls.__dataclass_fields__) - {"patch"}
        if unknown:
            raise ValueError(f"Unknown JEM view settings: {sorted(unknown)}")
        patch = zyx(patch_size, "patch_size")
        if "patch" in config and zyx(config["patch"], "patch") != patch:
            raise ValueError(f"JEM views patch {config['patch']} must equal the backbone patch size {patch}")
        values = {key: config[key] for key in cls.__dataclass_fields__ if key in config}
        for key in ("source_size", "teacher_size", "global_size", "local_size"):
            if key in values:
                values[key] = zyx(values[key], key)
        if "student_scale" in values:
            scale = tuple(float(v) for v in values["student_scale"])
            if len(scale) != 2:
                raise ValueError(f"JEM student_scale must be [low, high], got {values['student_scale']}")
            values["student_scale"] = scale
        spec = cls(**{**values, "patch": patch})
        spec.validate()
        return spec

    def validate(self) -> None:
        if any(p <= 0 for p in self.patch):
            raise ValueError(f"JEM patch {self.patch} must be positive")
        for name in ("source_size", "teacher_size", "global_size", "local_size"):
            size = getattr(self, name)
            if any(s <= 0 or s % p for s, p in zip(size, self.patch)):
                raise ValueError(f"JEM {name}={size} must be a positive multiple of the patch size {self.patch}")
        if any(max(t, g, l) > s for t, g, l, s in zip(self.teacher_size, self.global_size, self.local_size,
                                                        self.source_size)):
            raise ValueError("JEM crops must fit inside the source placement window")
        if self.orientations not in ORIENTATION_MODES:
            raise ValueError(f"JEM orientations must be one of {sorted(ORIENTATION_MODES)}, got {self.orientations!r}")
        shaped = (self.patch, self.teacher_size, self.global_size, self.local_size)
        if self.orientations == "all" and any(len(set(value)) != 1 for value in shaped):
            raise ValueError("JEM orientations 'all' transposes every axis, so the patch and crops must be cubes; "
                             "use 'z_preserving' for anisotropic patches or crops")
        if any(value[1] != value[2] for value in shaped):
            raise ValueError("JEM patch and crop sizes need equal y and x (the y/x transpose keeps the shape)")
        lo, hi = self.student_scale
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
            raise ValueError(f"JEM student_scale {self.student_scale} must satisfy 0 < low <= high")
        for size in (self.global_size, self.local_size):
            if any(f > s for f, s in zip(self.footprint_tokens(self.extent(size, hi)), self.grid(self.source_size))):
                raise ValueError(f"JEM student crops of {size} at scale {hi} must fit the source placement window")
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
        if need > self.tokens(tuple(map(min, self.global_size, self.teacher_size))):
            raise ValueError("min_overlap_fraction cannot be met by the global crop sizes")

    def tokens(self, size) -> int:
        return math.prod(self.grid(size))

    def grid(self, size) -> tuple[int, int, int]:
        return tuple(s // p for s, p in zip(size, self.patch))

    @property
    def scaled(self) -> bool:
        return tuple(self.student_scale) != (1.0, 1.0)

    @staticmethod
    def extent(size, scale: float) -> tuple[int, int, int]:
        """Voxels a student crop of `size` covers at `scale`."""
        return tuple(max(1, round(scale * s)) for s in size)

    def footprint_tokens(self, extent) -> tuple[int, int, int]:
        """Tokens of the patch-lattice box the planner places for a crop covering `extent` voxels."""
        return tuple(-(-e // p) for e, p in zip(extent, self.patch))

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
    grid = np.ones(spec.grid(spec.source_size), dtype=np.uint8)
    for axis, (start, extent) in enumerate(zip(starts, array_shape)):
        patch = spec.patch[axis]
        first = np.arange(grid.shape[axis]) * patch + int(start)
        inside = (first >= 0) & (first + patch <= int(extent))
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
        sizes = [spec.global_size] + [spec.local_size] * spec.locals_per_set
        patch = np.asarray(spec.patch, dtype=np.int64)
        if spec.scaled:
            # One isotropic scale per student view; the planner places each view's covering
            # token box, one set at a time (shapes differ per set).
            lo, hi = spec.student_scale
            scales = np.exp(np.random.uniform(math.log(lo), math.log(hi), size=(N_SETS, len(sizes))))
            extents = np.asarray([[spec.extent(size, s) for size, s in zip(sizes, row)] for row in scales],
                                 dtype=np.int64)
            parts = [views.plan_view_sets(
                support, spec.grid(spec.teacher_size), [spec.footprint_tokens(e) for e in extents[s]], n_sets=1,
                min_overlap_fraction=spec.min_overlap_fraction, min_overlap_count=spec.min_overlap_count,
                max_student_attempts=spec.max_student_attempts, max_teacher_attempts=spec.max_teacher_attempts,
                seed=int(_seeds(1)[0])) for s in range(N_SETS)]
            planned = {key: np.concatenate([np.asarray(part[key]) for part in parts])
                       for key in parts[0] if key != "ok"}
            planned["ok"] = all(bool(part["ok"]) for part in parts)
        else:
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
        if spec.scaled:
            # Voxel box of each student: a uniform offset of its extent inside the planned token box.
            slack = (extents + patch - 1) // patch * patch - extents
            student_boxes = np.concatenate(
                [student_origins * patch + np.random.randint(0, slack + 1), extents], axis=-1)
        else:
            student_boxes = np.concatenate(
                [student_origins * patch, np.broadcast_to(np.asarray(sizes), student_origins.shape)], axis=-1)
        boxes = []
        for s in range(N_SETS):
            boxes.append((teacher_origins[s] * patch, np.asarray(spec.teacher_size)))
            for index in range(1 + spec.locals_per_set):
                boxes.append((student_boxes[s, index, :3], student_boxes[s, index, 3:]))
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
        values = region[offset[0]:offset[0] + size[0], offset[1]:offset[1] + size[1], offset[2]:offset[2] + size[2]]
        return torch.from_numpy(np.ascontiguousarray(values))

    n_views = N_SETS * (2 + spec.locals_per_set)
    # Uniform over [0, 48); 48 is a multiple of 16, so the remainder is uniform over the
    # 16 z-preserving ids (permutations (0,1,2) and (0,2,1), every flip).
    orientations = np.asarray(views.draw_orientations(_seeds(n_views)), dtype=np.int64)
    orientations %= ORIENTATION_MODES[spec.orientations]
    intensity_seeds = _seeds(n_views)
    mask_seeds = _seeds(N_SETS * (1 + spec.locals_per_set))
    teacher_views, global_views, local_views = [], [], []
    teacher_orientation, global_orientation, local_orientation = [], [], []
    view = 0
    for s in range(N_SETS):
        for role, origin, extent, size in [
                ("teacher", teacher_origins[s] * patch, spec.teacher_size, spec.teacher_size),
                ("global", student_boxes[s, 0, :3], student_boxes[s, 0, 3:], spec.global_size)] + [
                ("local", student_boxes[s, 1 + j, :3], student_boxes[s, 1 + j, 3:], spec.local_size)
                for j in range(spec.locals_per_set)]:
            raw = crop(origin, extent)  # resized to `size` by the batch processor when they differ
            cropped = CropView(raw, tuple(int(v) for v in size), orientation_operations(orientations[view]),
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
            **({"global_boxes": torch.from_numpy(np.ascontiguousarray(student_boxes[:, 0])),
                "local_boxes": torch.from_numpy(np.ascontiguousarray(student_boxes[:, local_slice].reshape(-1, 6)))}
               if spec.scaled else {}),
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
