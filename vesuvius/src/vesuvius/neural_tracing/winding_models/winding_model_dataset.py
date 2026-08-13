from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from skimage.restoration import unwrap_phase
from vc.grid_raycast import GridRaycaster

import vesuvius.tifxyz as tifxyz  # noqa: PLR0402
from vesuvius.neural_tracing.winding_models.volume_slab_extractor import (
    SlabFrame,
    VolumeSlabExtractor,
)
from vesuvius.neural_tracing.winding_models.winding_targets import (
    render_column_targets_batched,
)

_WINDING_NAME = re.compile(r"(?:^|-)w(?P<index>\d+)(?:_|$)")

_SEGMENT_CACHE_VERSION = 2

_MALFORMED_CHUNK_ERROR = "decoded chunk byte size does not match full chunk shape"

# Surface rasterization constants, all in slab sample units. Quads are
# subdivided to at most _RASTER_STEP between samples, each sample is assigned
# to a supervised column when it falls within _COLUMN_WINDOW of the column
# center on both transverse axes (step < 2 * window guarantees every covered
# column receives samples), and samples chain-cluster into one crossing while
# consecutive ray-axis gaps stay below _CLUSTER_GAP.
_RASTER_STEP = 0.9
_COLUMN_WINDOW = 0.5
_CLUSTER_GAP = 1.0
_MAX_SUBDIVISION = 64
_BLOCK_QUADS = 8


@dataclass
class Segment:
    winding_idx: int | None
    xyz: np.ndarray
    raycaster: GridRaycaster | None
    valid_quads: np.ndarray
    sample_cells: np.ndarray
    vertex_turns: np.ndarray | None
    block_min: np.ndarray
    block_max: np.ndarray


@dataclass
class PatchRef:
    """Lazy handle for a crossings-only patch; geometry loads on first use."""

    path: Path
    weight: float
    index: int = -1


_PATCH_PACK_VERSION = 1


@dataclass
class PatchPack:
    """All patches of one dataset consolidated into a few mmapped arrays.

    Loading a patch becomes a set of zero-copy views instead of three tif
    reads; the OS page cache shares the touched pages across dataloader
    workers and across runs. Erosion and the inner-quad crop are baked in
    at build time.
    """

    xyz: np.ndarray  # (total_vertices, 3) float32, mmapped
    quads: np.ndarray  # (total_quads,) bool, mmapped
    block_min: np.ndarray  # (total_blocks, 3) float32, mmapped
    block_max: np.ndarray  # (total_blocks, 3) float32, mmapped
    shapes: np.ndarray  # (N, 2) vertex grid shape
    vertex_offsets: np.ndarray  # (N + 1,)
    quad_offsets: np.ndarray  # (N + 1,)
    block_shapes: np.ndarray  # (N, 2)
    block_offsets: np.ndarray  # (N + 1,)
    quad_counts: np.ndarray  # (N,) valid quads per patch
    bbox_min: np.ndarray  # (N, 3) post-erosion world AABB
    bbox_max: np.ndarray  # (N, 3)
    names: list[str]

    def segment(self, index: int) -> Segment:
        height, width = (int(value) for value in self.shapes[index])
        v0, v1 = self.vertex_offsets[index], self.vertex_offsets[index + 1]
        q0, q1 = self.quad_offsets[index], self.quad_offsets[index + 1]
        b0, b1 = self.block_offsets[index], self.block_offsets[index + 1]
        blocks_r, blocks_c = (int(value) for value in self.block_shapes[index])
        return Segment(
            winding_idx=None,
            xyz=self.xyz[v0:v1].reshape(height, width, 3),
            raycaster=None,
            valid_quads=self.quads[q0:q1].reshape(height - 1, width - 1),
            sample_cells=None,
            vertex_turns=None,
            block_min=self.block_min[b0:b1].reshape(blocks_r, blocks_c, 3),
            block_max=self.block_max[b0:b1].reshape(blocks_r, blocks_c, 3),
        )


@dataclass
class VolumeDataset:
    volume_path: Path
    segments: list[Segment]
    segment_to_volume_xyz: np.ndarray
    sampling_weight: float = 1.0
    crossings_only: bool = False
    patches: list[PatchRef] | None = None
    patch_bbox_min: np.ndarray | None = None
    patch_bbox_max: np.ndarray | None = None
    erode_cells: int = 0
    inner_fraction: float | None = None
    pack: PatchPack | None = None


def _segment_block_bounds(
    xyz: np.ndarray, valid_quads: np.ndarray, block: int = _BLOCK_QUADS
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative world AABBs for each block x block tile of valid quads.

    Only vertices participating in at least one valid quad count; empty
    blocks get inverted (inf, -inf) bounds and never match an overlap test.
    """
    rows, cols = xyz.shape[:2]
    vertex_used = np.zeros((rows, cols), dtype=bool)
    vertex_used[:-1, :-1] |= valid_quads
    vertex_used[1:, :-1] |= valid_quads
    vertex_used[:-1, 1:] |= valid_quads
    vertex_used[1:, 1:] |= valid_quads

    blocks_r = -(-rows // block)
    blocks_c = -(-cols // block)
    lo = np.full((blocks_r * block, blocks_c * block, 3), np.inf, dtype=np.float32)
    hi = np.full_like(lo, -np.inf)
    lo[:rows, :cols] = np.where(vertex_used[..., None], xyz, np.inf)
    hi[:rows, :cols] = np.where(vertex_used[..., None], xyz, -np.inf)
    lo = lo.reshape(blocks_r, block, blocks_c, block, 3).min(axis=(1, 3))
    hi = hi.reshape(blocks_r, block, blocks_c, block, 3).max(axis=(1, 3))

    # A quad tile [iB, (i+1)B) x [jB, (j+1)B) touches vertices up to row
    # (i+1)B and column (j+1)B, so combine vertex tiles {i, i+1} x {j, j+1}.
    lo_pad = np.full((blocks_r + 1, blocks_c + 1, 3), np.inf, dtype=np.float32)
    hi_pad = np.full_like(lo_pad, -np.inf)
    lo_pad[:blocks_r, :blocks_c] = lo
    hi_pad[:blocks_r, :blocks_c] = hi
    block_min = np.minimum.reduce(
        [lo_pad[:-1, :-1], lo_pad[1:, :-1], lo_pad[:-1, 1:], lo_pad[1:, 1:]]
    )
    block_max = np.maximum.reduce(
        [hi_pad[:-1, :-1], hi_pad[1:, :-1], hi_pad[:-1, 1:], hi_pad[1:, 1:]]
    )
    quad_blocks_r = -(-(rows - 1) // block)
    quad_blocks_c = -(-(cols - 1) // block)
    return (
        block_min[:quad_blocks_r, :quad_blocks_c],
        block_max[:quad_blocks_r, :quad_blocks_c],
    )


def _bilinear(
    v00: np.ndarray, v10: np.ndarray, v01: np.ndarray, v11: np.ndarray,
    u: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Bilinear patch samples; u follows the row axis, v the column axis."""
    return (
        v00[:, None] * ((1.0 - u) * (1.0 - v))[None, :, None]
        + v10[:, None] * (u * (1.0 - v))[None, :, None]
        + v01[:, None] * ((1.0 - u) * v)[None, :, None]
        + v11[:, None] * (u * v)[None, :, None]
    )


class WindingModelDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict):
        self.num_samples = int(cfg.get("num_samples", 500_000))
        self.inner_fraction = float(cfg.get("inner_fraction", 0.7))
        self.min_winding_gap = int(cfg.get("min_winding_gap", 4))
        self.max_sample_attempts = int(cfg.get("max_sample_attempts", 64))
        self.ray_skew_degrees = float(cfg.get("ray_skew_degrees", 3.0))
        self.ray_origin_offset = float(cfg.get("ray_origin_offset", 1.0))
        self.hit_merge_tolerance = float(cfg.get("hit_merge_tolerance", 1e-3))
        self.crossing_merge_distance = float(cfg.get("crossing_merge_distance", 3.0))
        self.crossing_sigma_wv = float(cfg.get("crossing_sigma_wv", 1.0))
        self.transverse_size = int(cfg.get("transverse_size", 96))
        self.ray_length = int(cfg.get("ray_length", 384))
        self.spacing = float(cfg.get("spacing", 1.0))
        self.column_stride = int(cfg.get("column_stride", 4))
        self.min_supervised_columns = int(cfg.get("min_supervised_columns", 1))
        # Fraction of samples that carry a second overlapping slab for the
        # multiview consistency loss; the pair's ray direction stays within
        # this cone of the primary's so both phases ascend the same winding
        # direction.
        self.multiview_fraction = float(cfg.get("multiview_fraction", 0.0))
        if not 0.0 <= self.multiview_fraction <= 1.0:
            raise ValueError("multiview_fraction must be in [0, 1]")
        self.multiview_cone_degrees = float(cfg.get("multiview_cone_degrees", 20.0))
        if not 0.0 < self.multiview_cone_degrees < 90.0:
            raise ValueError("multiview_cone_degrees must be in (0, 90)")
        if self.ray_length < 2:
            raise ValueError("ray_length must be at least 2")
        self._validate_transverse_size(self.transverse_size)
        cache_dir = cfg.get(
            "segment_cache_dir", "~/.cache/vesuvius/winding_model_segments"
        )
        self.segment_cache_dir = Path(cache_dir).expanduser() if cache_dir else None

        self.patch_cache_segments = int(cfg.get("patch_cache_segments", 1024))
        self._patch_cache: OrderedDict[Path, Segment | None] = OrderedDict()

        self.volumes = []
        sample_fractions: dict[int, float] = {}
        for volume_cfg in cfg["datasets"]:
            scale = int(volume_cfg["volume_scale"])
            crossings_only = bool(volume_cfg.get("crossings_only", False))
            erode_cells = int(volume_cfg.get("erode_cells", 0))
            inner_fraction = volume_cfg.get("inner_fraction")
            if inner_fraction is not None:
                inner_fraction = float(inner_fraction)
            segments = []
            patch_refs: list[PatchRef] = []
            patch_bbox_min: list = []
            patch_bbox_max: list = []
            pack = None
            if crossings_only:
                # Position-only patches: consolidate every patch into one
                # mmapped pack (built once, cached); geometry then loads as
                # zero-copy views and pages shared across workers.
                segments_root = Path(volume_cfg["segments_path"])
                pack = self._load_or_build_patch_pack(
                    segments_root,
                    erode_cells=erode_cells,
                    inner_fraction=inner_fraction,
                )
                for index, name in enumerate(pack.names):
                    patch_refs.append(
                        PatchRef(
                            segments_root / name,
                            float(pack.quad_counts[index]),
                            index,
                        )
                    )
                patch_bbox_min = pack.bbox_min
                patch_bbox_max = pack.bbox_max
            else:
                for info in tifxyz.list_tifxyz(volume_cfg["segments_path"]):
                    segment = self._load_segment(
                        info.path,
                        erode_cells=erode_cells,
                        inner_fraction=inner_fraction,
                    )
                    if not len(segment.sample_cells):
                        print(
                            "winding dataset: skipping segment with no valid "
                            f"inner quads: {info.path}"
                        )
                        continue
                    segments.append(segment)
            if segments or patch_refs:
                configured_path = Path(volume_cfg["volume_path"])
                volume_path = VolumeSlabExtractor.scaled_volume_path(
                    configured_path, scale
                )
                segment_to_volume_xyz = (
                    VolumeSlabExtractor.load_segment_to_volume_transform(
                        configured_path,
                        scale,
                        segment_downscale=int(volume_cfg.get("segment_downscale", 1)),
                        use_registration=(
                            volume_cfg.get("segment_volume_id") is not None
                            and volume_cfg.get("volume_id") is not None
                            and volume_cfg["segment_volume_id"]
                            != volume_cfg["volume_id"]
                        ),
                    )
                )
                self.volumes.append(
                    VolumeDataset(
                        volume_path,
                        segments,
                        segment_to_volume_xyz,
                        sampling_weight=float(volume_cfg.get("sampling_weight", 1.0)),
                        crossings_only=crossings_only,
                        patches=patch_refs or None,
                        patch_bbox_min=(
                            np.asarray(patch_bbox_min, dtype=np.float64)
                            if len(patch_refs)
                            else None
                        ),
                        patch_bbox_max=(
                            np.asarray(patch_bbox_max, dtype=np.float64)
                            if len(patch_refs)
                            else None
                        ),
                        erode_cells=erode_cells,
                        inner_fraction=inner_fraction,
                        pack=pack,
                    )
                )
                if "sample_fraction" in volume_cfg:
                    sample_fractions[len(self.volumes) - 1] = float(
                        volume_cfg["sample_fraction"]
                    )

        if not self.volumes:
            raise ValueError("No tifxyz segments were found")

        # Sample segments proportional to their valid-quad count so every
        # candidate ray origin is equally likely regardless of how the surface
        # area happens to be split into segments; uniform segment choice would
        # oversample small segments by orders of magnitude. float64 because
        # quad counts exceed float32's integer range. Lazy patches stand in
        # with their metadata surface area, proportional to the same thing.
        self._segment_lookup = []
        weight_list = []
        for volume_idx, volume in enumerate(self.volumes):
            if volume.crossings_only:
                for patch in volume.patches:
                    self._segment_lookup.append((volume_idx, patch))
                    weight_list.append(volume.sampling_weight * patch.weight)
            else:
                for segment in volume.segments:
                    self._segment_lookup.append((volume_idx, segment))
                    weight_list.append(
                        volume.sampling_weight * len(segment.sample_cells)
                    )
        weights = torch.tensor(weight_list, dtype=torch.float64)

        # sample_fraction pins a dataset's share of ray-origin draws
        # regardless of its area (84k small patches would otherwise swamp
        # the fully labeled segments); within the dataset, draws stay
        # area-proportional.
        if sample_fractions:
            total_fraction = sum(sample_fractions.values())
            if not 0.0 < total_fraction < 1.0:
                raise ValueError("sample_fraction values must sum into (0, 1)")
            volume_of = torch.tensor(
                [volume_idx for volume_idx, _ in self._segment_lookup]
            )
            free = torch.tensor(
                [
                    volume_idx not in sample_fractions
                    for volume_idx, _ in self._segment_lookup
                ]
            )
            free_total = float(weights[free].sum())
            if free_total <= 0:
                raise ValueError(
                    "sample_fraction requires at least one dataset without it"
                )
            for volume_idx, fraction in sample_fractions.items():
                mask = volume_of == volume_idx
                block = float(weights[mask].sum())
                if block > 0:
                    weights[mask] *= (
                        fraction / (1.0 - total_fraction) * free_total / block
                    )

        if weights.sum() <= 0:
            raise ValueError("All segment sampling weights are zero")
        self._segment_cdf = torch.cumsum(weights, dim=0)

        self.slab_extractor = VolumeSlabExtractor(
            [volume.volume_path for volume in self.volumes],
            transverse_size=self.transverse_size,
            ray_length=self.ray_length,
            spacing=self.spacing,
            sampling=str(cfg.get("sampling", "trilinear")),
            tile_size=int(cfg.get("tile_size", 32)),
            cache_bytes=int(cfg.get("volume_cache_bytes", 0)),
            io_threads=int(cfg.get("volume_io_threads", 0)),
            segment_to_volume_xyz=[
                volume.segment_to_volume_xyz for volume in self.volumes
            ],
        )

    def _validate_transverse_size(self, transverse_size: int) -> None:
        # The model halves the transverse axes three times and the columns
        # sit at multiples of the column stride, so the transverse size must
        # tile exactly.
        if transverse_size < 2 * self.column_stride:
            raise ValueError("transverse_size is too small for the column stride")
        if transverse_size % (2 * self.column_stride):
            raise ValueError(
                "transverse_size must be a multiple of twice the column stride"
            )

    @property
    def ray_extent(self) -> float:
        """Physical ray extent represented by the sampled slab length."""
        return (self.ray_length - 1) * self.spacing

    @property
    def columns_per_axis(self) -> int:
        return self.transverse_size // self.column_stride

    @property
    def max_crossings(self) -> int:
        """Fixed width of the per-column crossing lists.

        Kept crossings are separated by more than the crossing merge
        distance along a ray of ray_length samples, which bounds their
        count; padding every sample to that bound (rather than the
        batch's widest column) makes batches shape-static, so a
        dispatching dataloader can concatenate independently collated
        batches.
        """
        merge_samples = self.crossing_merge_distance / self.spacing
        return int(self.ray_length / max(merge_samples, 1e-6)) + 2

    def set_slab_dimensions(self, *, ray_length: int, transverse_size: int) -> None:
        """Set the output dimensions used for subsequent samples."""
        ray_length = int(ray_length)
        transverse_size = int(transverse_size)
        if ray_length < 2:
            raise ValueError("ray_length must be at least 2")
        self._validate_transverse_size(transverse_size)
        self.ray_length = ray_length
        self.transverse_size = transverse_size
        self.slab_extractor.ray_length = ray_length
        self.slab_extractor.transverse_size = transverse_size

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _winding_index(path: Path) -> int | None:
        match = _WINDING_NAME.search(path.name)
        return int(match["index"]) if match else None

    def _load_segment(
        self,
        path: Path,
        *,
        erode_cells: int = 0,
        inner_fraction: float | None = None,
        build_raycaster: bool = True,
        need_turns: bool = True,
    ) -> Segment:
        surface = tifxyz.read_tifxyz(path)
        xyz = np.ascontiguousarray(
            np.stack([surface._x, surface._y, surface._z], axis=-1),
            dtype=np.float32,
        )
        winding_idx = self._winding_index(path)

        # Trim vertices near the border of the valid region, where annotation
        # errors concentrate — the same erosion fit_spiral applies to patches
        # before consuming them.
        valid_vertex = surface.valid_vertex_mask
        if erode_cells > 0:
            valid_vertex = scipy.ndimage.binary_erosion(
                valid_vertex, iterations=erode_cells, border_value=0
            )
            xyz[~valid_vertex] = -1.0
            quad_mask = (
                valid_vertex[:-1, :-1]
                & valid_vertex[1:, :-1]
                & valid_vertex[:-1, 1:]
                & valid_vertex[1:, 1:]
            )
        else:
            quad_mask = surface.valid_quad_mask

        # Winding turns are only needed for segments without an explicit
        # winding index; unwrapping the phase of a large grid is the one
        # derivation expensive enough to cache.
        vertex_turns = None
        if need_turns and winding_idx is None:
            cache_path = self._segment_cache_path(path, erode_cells)
            arrays = self._read_segment_cache(cache_path)
            if arrays is None or "vertex_turns" not in arrays:
                arrays = {"vertex_turns": self._vertex_turns(xyz, valid_vertex)}
                self._write_segment_cache(cache_path, arrays)
            vertex_turns = arrays["vertex_turns"]

        valid_quads = np.ascontiguousarray(
            self._inner_quads(quad_mask, inner_fraction)
        )
        block_min, block_max = _segment_block_bounds(xyz, valid_quads)
        return Segment(
            winding_idx=winding_idx,
            xyz=xyz,
            raycaster=GridRaycaster(xyz, valid_quads) if build_raycaster else None,
            valid_quads=valid_quads,
            sample_cells=np.argwhere(valid_quads),
            vertex_turns=vertex_turns,
            block_min=block_min,
            block_max=block_max,
        )

    def _load_patch(
        self, volume: VolumeDataset, ref: PatchRef, *, for_seed: bool = False
    ) -> Segment | None:
        """Materialize a crossings-only patch.

        With a pack this is a set of zero-copy views into the mmapped
        arrays. Only the one patch per sample that seeds the ray needs a
        raycaster and sample cells (``for_seed``); rasterizing neighbours
        into the slab reads the surface grid alone. Patches the erosion
        left without valid quads resolve to None.
        """
        if volume.pack is not None:
            if volume.pack.quad_counts[ref.index] == 0:
                return None
            segment = volume.pack.segment(ref.index)
        else:
            # Fallback without a cache dir: load from the tifs, LRU-cached
            # per worker.
            if ref.path in self._patch_cache:
                self._patch_cache.move_to_end(ref.path)
                segment = self._patch_cache[ref.path]
            else:
                segment = self._load_segment(
                    ref.path,
                    erode_cells=volume.erode_cells,
                    inner_fraction=volume.inner_fraction,
                    build_raycaster=False,
                    need_turns=False,
                )
                if not len(segment.sample_cells):
                    segment = None
                self._patch_cache[ref.path] = segment
                while len(self._patch_cache) > self.patch_cache_segments:
                    self._patch_cache.popitem(last=False)
            if segment is None:
                return None
        if for_seed:
            xyz = np.ascontiguousarray(segment.xyz)
            quads = np.ascontiguousarray(segment.valid_quads)
            segment = replace(
                segment,
                xyz=xyz,
                valid_quads=quads,
                raycaster=GridRaycaster(xyz, quads),
                sample_cells=np.argwhere(quads),
            )
        return segment

    def _patch_pack_dir(
        self,
        segments_path: Path,
        patch_dirs: list[Path],
        erode_cells: int,
        inner_fraction: float | None,
    ) -> Path | None:
        """Cache location keyed on the patch set and derivation options."""
        if self.segment_cache_dir is None:
            return None
        fingerprint = [
            (path.name, stat.st_mtime_ns, stat.st_size)
            for path in patch_dirs
            for stat in ((path / "meta.json").stat(),)
        ]
        key = json.dumps(
            [
                _PATCH_PACK_VERSION,
                str(segments_path.resolve()),
                erode_cells,
                inner_fraction,
                self.inner_fraction,
                fingerprint,
            ]
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.segment_cache_dir / f"patch-pack-{digest}"

    def _load_or_build_patch_pack(
        self,
        segments_path: Path,
        *,
        erode_cells: int,
        inner_fraction: float | None,
    ) -> PatchPack:
        patch_dirs = sorted(
            entry
            for entry in segments_path.iterdir()
            if (entry / "meta.json").is_file()
        )
        if not patch_dirs:
            raise ValueError(f"No tifxyz patches were found in {segments_path}")
        pack_dir = self._patch_pack_dir(
            segments_path, patch_dirs, erode_cells, inner_fraction
        )
        if pack_dir is not None and pack_dir.is_dir():
            return self._read_patch_pack(pack_dir)

        print(
            f"winding dataset: packing {len(patch_dirs)} patches from "
            f"{segments_path} (one-time)"
        )

        def load_one(path: Path) -> Segment:
            return self._load_segment(
                path,
                erode_cells=erode_cells,
                inner_fraction=inner_fraction,
                build_raycaster=False,
                need_turns=False,
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            loaded = list(executor.map(load_one, patch_dirs))

        shapes, block_shapes, quad_counts = [], [], []
        bbox_min, bbox_max = [], []
        for segment in loaded:
            shapes.append(segment.xyz.shape[:2])
            block_shapes.append(segment.block_min.shape[:2])
            quad_counts.append(int(segment.valid_quads.sum()))
            finite = np.isfinite(segment.block_min).all(axis=-1)
            if finite.any():
                bbox_min.append(segment.block_min[finite].min(axis=0))
                bbox_max.append(segment.block_max[finite].max(axis=0))
            else:
                bbox_min.append(np.zeros(3, dtype=np.float32))
                bbox_max.append(np.full(3, -1.0, dtype=np.float32))

        def offsets(sizes: list[int]) -> np.ndarray:
            return np.r_[0, np.cumsum(np.asarray(sizes, dtype=np.int64))]

        arrays = {
            "xyz": np.concatenate(
                [segment.xyz.reshape(-1, 3) for segment in loaded]
            ),
            "quads": np.concatenate(
                [segment.valid_quads.reshape(-1) for segment in loaded]
            ),
            "block_min": np.concatenate(
                [segment.block_min.reshape(-1, 3) for segment in loaded]
            ),
            "block_max": np.concatenate(
                [segment.block_max.reshape(-1, 3) for segment in loaded]
            ),
        }
        index = {
            "shapes": np.asarray(shapes, dtype=np.int32),
            "vertex_offsets": offsets([h * w for h, w in shapes]),
            "quad_offsets": offsets([(h - 1) * (w - 1) for h, w in shapes]),
            "block_shapes": np.asarray(block_shapes, dtype=np.int32),
            "block_offsets": offsets([r * c for r, c in block_shapes]),
            "quad_counts": np.asarray(quad_counts, dtype=np.int64),
            "bbox_min": np.asarray(bbox_min, dtype=np.float64),
            "bbox_max": np.asarray(bbox_max, dtype=np.float64),
        }
        names = [path.name for path in patch_dirs]

        if pack_dir is None:
            return PatchPack(**arrays, **index, names=names)

        tmp_dir = pack_dir.with_name(f"{pack_dir.name}.{os.getpid()}.tmp")
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for name, array in arrays.items():
                np.save(tmp_dir / f"{name}.npy", array)
            np.savez(tmp_dir / "index.npz", **index)
            (tmp_dir / "names.json").write_text(json.dumps(names))
            tmp_dir.replace(pack_dir)
        finally:
            if tmp_dir.is_dir():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return self._read_patch_pack(pack_dir)

    @staticmethod
    def _read_patch_pack(pack_dir: Path) -> PatchPack:
        with np.load(pack_dir / "index.npz") as data:
            index = {name: data[name] for name in data.files}
        # The quad mask and block bounds sit on the candidate-quad hot path
        # (hundreds of patches per drawn sample), where memmap slicing
        # overhead exceeds the cost of holding them resident; fork()ed
        # dataloader workers share the parent's copy. Only the vertex grid
        # (~2.5 GB) stays memory-mapped.
        return PatchPack(
            xyz=np.load(pack_dir / "xyz.npy", mmap_mode="r"),
            quads=np.load(pack_dir / "quads.npy"),
            block_min=np.load(pack_dir / "block_min.npy"),
            block_max=np.load(pack_dir / "block_max.npy"),
            names=json.loads((pack_dir / "names.json").read_text()),
            **index,
        )

    def _inner_quads(
        self, valid_quads: np.ndarray, fraction: float | None = None
    ) -> np.ndarray:
        """Restrict the valid-quad mask to the configured inner UV region."""
        fraction = self.inner_fraction if fraction is None else fraction
        valid_quads = valid_quads.copy()
        if fraction >= 1.0:
            return valid_quads
        margin = (1.0 - fraction) / 2.0
        height, width = valid_quads.shape
        inner = np.zeros_like(valid_quads)
        inner[
            int(margin * height) : int((1.0 - margin) * height),
            int(margin * width) : int((1.0 - margin) * width),
        ] = True
        valid_quads &= inner
        return valid_quads

    def _segment_cache_path(self, path: Path, erode_cells: int = 0) -> Path | None:
        """Key the cache on the source files, derivation options, and version."""
        if self.segment_cache_dir is None:
            return None
        resolved = path.resolve()
        sources = sorted(resolved.iterdir()) if resolved.is_dir() else [resolved]
        stats = [
            (source.name, stat.st_mtime_ns, stat.st_size)
            for source in sources
            for stat in (source.stat(),)
        ]
        # Erosion changes the derived arrays; un-eroded segments keep the
        # legacy key so existing caches stay valid.
        parts = [_SEGMENT_CACHE_VERSION, str(resolved)]
        if erode_cells:
            parts.append(erode_cells)
        key = json.dumps(parts + [stats])
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.segment_cache_dir / f"{resolved.name}-{digest}.npz"

    @staticmethod
    def _read_segment_cache(cache_path: Path | None) -> dict[str, np.ndarray] | None:
        if cache_path is None or not cache_path.is_file():
            return None
        try:
            with np.load(cache_path) as data:
                return {name: data[name] for name in data.files}
        except (OSError, ValueError, zipfile.BadZipFile):
            return None

    @staticmethod
    def _write_segment_cache(
        cache_path: Path | None, arrays: dict[str, np.ndarray]
    ) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("wb") as tmp_file:
                np.savez(tmp_file, **arrays)
            tmp_path.replace(cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _vertex_turns(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
        center = xyz[valid].mean(axis=0)
        phase = np.arctan2(xyz[..., 1] - center[1], xyz[..., 0] - center[0])
        phase = unwrap_phase(np.ma.array(phase, mask=~valid), rng=0)
        return (phase.filled(np.nan) / (2.0 * math.pi)).astype(np.float32)

    @staticmethod
    def _randint(high: int) -> int:
        return int(torch.randint(high, ()).item())

    def _point_and_normal(
        self, segment: Segment
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not len(segment.sample_cells):
            return None
        row, col = segment.sample_cells[self._randint(len(segment.sample_cells))]
        quad = segment.xyz[row : row + 2, col : col + 2].astype(np.float64)
        point = (quad[1, 0] + quad[0, 1]) / 2
        normal = np.cross(quad[1, 0] - quad[0, 0], quad[0, 1] - quad[0, 0])
        normal += np.cross(quad[0, 1] - quad[1, 1], quad[1, 0] - quad[1, 1])
        norm = np.linalg.norm(normal)
        return None if not np.isfinite(norm) or norm < 1e-8 else (point, normal / norm)

    def _ray(
        self, segment: Segment
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        sampled = self._point_and_normal(segment)
        if sampled is None:
            return None
        point, direction = sampled

        axis = np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        tangent_a = np.cross(direction, axis)
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(direction, tangent_a)
        azimuth = 2.0 * math.pi * float(torch.rand(()))
        tangent = math.cos(azimuth) * tangent_a + math.sin(azimuth) * tangent_b
        angle = math.radians(self.ray_skew_degrees * float(torch.rand(())))
        direction = math.cos(angle) * direction + math.sin(angle) * tangent
        if torch.rand(()) < 0.5:
            direction = -direction

        offset = self.ray_origin_offset * (0.5 + 0.5 * float(torch.rand(())))
        return point, point - offset * direction, direction

    def _pair_direction(self, direction: np.ndarray) -> np.ndarray:
        """Second-view ray direction within the multiview cone.

        Never flipped: keeping the pair inside a cone well under 90 degrees
        of the primary guarantees both views ascend the same winding
        direction, so the consistency loss needs no sign registration.
        """
        axis = np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        tangent_a = np.cross(direction, axis)
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(direction, tangent_a)
        azimuth = 2.0 * math.pi * float(torch.rand(()))
        tangent = math.cos(azimuth) * tangent_a + math.sin(azimuth) * tangent_b
        angle = math.radians(self.multiview_cone_degrees * float(torch.rand(())))
        return math.cos(angle) * direction + math.sin(angle) * tangent

    def _multiview_pair(
        self, volume_idx: int, origin: np.ndarray, frame: SlabFrame
    ) -> dict[str, torch.Tensor]:
        """Second overlapping slab for the multiview consistency loss.

        Anchored at the primary ray's midpoint, cast within the multiview
        cone with a randomized position along the new ray. Needs no
        targets: the loss is label-free. Samples that draw no pair (or
        whose extraction hits a truncated chunk) return zero-filled fields
        with ``has_pair`` False so batch keys stay collate-uniform.
        """
        size = self.transverse_size
        empty = {
            "pair_image": torch.zeros(
                (size, size, self.ray_length), dtype=torch.float32
            ),
            "pair_valid": torch.zeros(
                (size, size, self.ray_length), dtype=torch.bool
            ),
            "pair_origin_zyx": torch.zeros(3, dtype=torch.float32),
            "pair_axis_a_zyx": torch.zeros(3, dtype=torch.float32),
            "pair_axis_b_zyx": torch.zeros(3, dtype=torch.float32),
            "pair_direction_zyx": torch.zeros(3, dtype=torch.float32),
            "has_pair": torch.tensor(False),
        }
        if float(torch.rand(())) >= self.multiview_fraction:
            return empty
        anchor = origin + 0.5 * self.ray_extent * frame.direction
        direction = self._pair_direction(frame.direction)
        offset = self.ray_extent * (0.35 + 0.3 * float(torch.rand(())))
        try:
            image, valid, pair_frame = self.slab_extractor.extract(
                volume_idx, direction, anchor - offset * direction
            )
        except RuntimeError as exc:
            if _MALFORMED_CHUNK_ERROR not in str(exc):
                raise
            return empty
        return {
            "pair_image": torch.from_numpy(np.ascontiguousarray(image)),
            "pair_valid": torch.from_numpy(np.ascontiguousarray(valid)).bool(),
            "pair_origin_zyx": torch.from_numpy(
                pair_frame.origin[::-1].copy()
            ).float(),
            "pair_axis_a_zyx": torch.from_numpy(
                pair_frame.axis_a[::-1].copy()
            ).float(),
            "pair_axis_b_zyx": torch.from_numpy(
                pair_frame.axis_b[::-1].copy()
            ).float(),
            "pair_direction_zyx": torch.from_numpy(
                pair_frame.direction[::-1].copy()
            ).float(),
            "has_pair": torch.tensor(True),
        }

    def _hits(
        self,
        segment: Segment,
        origin: np.ndarray,
        direction: np.ndarray,
        max_t: float = math.inf,
    ) -> list[tuple[float, np.ndarray, float | None]]:
        """Ordered ray crossings as (t, xyz, winding turns) tuples.

        The turns entry is None for segments with an explicit winding index.
        """
        ts, locations, rows, cols, triangles = segment.raycaster.hits(
            origin, direction, 0.0, max_t + self.hit_merge_tolerance
        )
        kept = []
        for i in range(len(ts)):
            if not kept or ts[i] - ts[kept[-1]] > self.hit_merge_tolerance:
                kept.append(i)
        turns = None
        if segment.vertex_turns is not None and kept:
            grid = segment.vertex_turns
            row, col, tri = rows[kept], cols[kept], triangles[kept]
            shared = grid[row + 1, col] + grid[row, col + 1]
            turns = (
                np.where(tri == 0, grid[row, col], grid[row + 1, col + 1]) + shared
            ) / 3.0
        return [
            (float(ts[i]), locations[i], None if turns is None else float(turns[k]))
            for k, i in enumerate(kept)
        ]

    def _multi_wrap_ray(
        self, segment: Segment
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]], float] | None:
        ray = self._ray(segment)
        if ray is None:
            return None
        _, origin, direction = ray
        hits = self._hits(segment, origin, direction, self.ray_extent)
        if len(hits) < 2:
            return None

        assert segment.vertex_turns is not None
        turns = np.asarray([turn for _, _, turn in hits], dtype=np.float64)
        if not np.isfinite(turns).all():
            return None
        indices = np.rint(turns - turns[0]).astype(int)

        ordered = [(hits[0][0], hits[0][1], 0)]
        sign = 0
        for hit, index in zip(hits[1:], indices[1:]):
            delta = int(index) - ordered[-1][2]
            if delta == 0:
                continue
            # Crossings this close are almost surely one wrap counted twice.
            if hit[0] - ordered[-1][0] <= self.crossing_merge_distance:
                continue
            if sign == 0:
                sign = 1 if delta > 0 else -1
            elif delta * sign < 0:
                break
            ordered.append((hit[0], hit[1], int(index)))

        if len(ordered) < 2:
            return None
        # The first hit's fractional turn anchors integer winding indices for
        # every column of the slab: vertex turns are globally consistent
        # across the segment, so all columns share one index origin.
        return origin, direction, ordered, float(turns[0])

    def _labelled_ray(
        self, volume: VolumeDataset, source: Segment
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]], list[Segment]] | None:
        assert source.winding_idx is not None
        targets = [
            segment
            for segment in volume.segments
            if segment.winding_idx is not None
            and abs(segment.winding_idx - source.winding_idx) >= self.min_winding_gap
        ]
        if not targets:
            return None

        target = targets[self._randint(len(targets))]
        ray = self._ray(source)
        if ray is None:
            return None
        point, origin, direction = ray

        target_hits = self._hits(target, origin, direction, self.ray_extent)
        if not target_hits:
            direction = -direction
            origin = point - self.ray_origin_offset * direction
            target_hits = self._hits(target, origin, direction, self.ray_extent)
        if not target_hits:
            return None

        low, high = sorted((source.winding_idx, target.winding_idx))
        bracket = [
            segment
            for segment in volume.segments
            if segment.winding_idx is not None
            and low <= segment.winding_idx <= high
        ]
        hits = []
        for segment in bracket:
            hits.extend(
                (t, xyz, segment.winding_idx)
                for t, xyz, _ in self._hits(
                    segment, origin, direction, target_hits[0][0]
                )
            )
        hits.sort(key=lambda hit: hit[0])

        sign = 1 if target.winding_idx > source.winding_idx else -1
        ordered = []
        for hit in hits:
            if ordered and hit[2] == ordered[-1][2]:
                continue
            # Crossings this close are almost surely one wrap counted twice.
            if ordered and hit[0] - ordered[-1][0] <= self.crossing_merge_distance:
                continue
            if ordered and (hit[2] - ordered[-1][2]) * sign < 0:
                return None
            ordered.append(hit)

        if (
            not ordered
            or ordered[0][2] != source.winding_idx
            or ordered[-1][2] != target.winding_idx
        ):
            return None
        return origin, direction, ordered, bracket

    def _randomly_position_crossings(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        hits: list[tuple[float, np.ndarray, int]],
    ) -> tuple[np.ndarray, list[tuple[float, np.ndarray, int]]]:
        """Place a crossing group anywhere it fits along the sampled ray."""
        crossing_span = hits[-1][0] - hits[0][0]
        available_start = self.ray_extent - crossing_span
        if available_start <= 0:
            return origin, hits

        new_first_t = available_start * float(torch.rand(()))
        t_shift = new_first_t - hits[0][0]
        shifted_origin = origin - t_shift * direction
        shifted_hits = [(t + t_shift, xyz, index) for t, xyz, index in hits]
        return shifted_origin, shifted_hits

    def _filter_crossings(
        self,
        volume_idx: int,
        hits: list[tuple[float, np.ndarray, int]],
        *,
        require_anchored: bool = True,
    ) -> list[tuple[float, np.ndarray, int]]:
        """Drop central-ray crossings that cannot serve as supervision targets.

        A crossing on zero-valued CT data sits outside the scanned material,
        so its position is not observable in the sampled slab. A crossing
        with no neighbour exactly one winding away (e.g. a lone index 18 after
        1, 2, 3) has no adjacent wrap anchoring its label; the zero-CT pass
        runs first because it can strand such crossings. Position-only labels
        carry no winding indices, so they skip the anchor requirement.
        """
        values = self.slab_extractor.sample_points(
            volume_idx, np.stack([xyz for _, xyz, _ in hits])
        )
        hits = [hit for hit, value in zip(hits, values) if value > 0]
        if not require_anchored:
            return hits
        return [
            hit
            for i, hit in enumerate(hits)
            if (i > 0 and abs(hit[2] - hits[i - 1][2]) == 1)
            or (i + 1 < len(hits) and abs(hits[i + 1][2] - hit[2]) == 1)
        ]

    def _candidate_quads(
        self, segment: Segment, slab_min: np.ndarray, slab_max: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Valid quad cells whose block AABB intersects the slab AABB."""
        overlap = (segment.block_min <= slab_max).all(-1) & (
            segment.block_max >= slab_min
        ).all(-1)
        rows, cols = [], []
        for block_r, block_c in np.argwhere(overlap):
            local = segment.valid_quads[
                block_r * _BLOCK_QUADS : (block_r + 1) * _BLOCK_QUADS,
                block_c * _BLOCK_QUADS : (block_c + 1) * _BLOCK_QUADS,
            ]
            local_r, local_c = np.nonzero(local)
            rows.append(local_r + block_r * _BLOCK_QUADS)
            cols.append(local_c + block_c * _BLOCK_QUADS)
        if not rows:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(rows), np.concatenate(cols)

    def _rasterize_segment(
        self,
        segment: Segment,
        frame: SlabFrame,
        wind_grid: np.ndarray | None,
        wind_value: float,
        slab_min: np.ndarray,
        slab_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the segment surface into slab coordinates.

        Returns fractional slab positions [N, 3] and per-sample winding
        values [N] (fractional turns from ``wind_grid`` or the constant
        ``wind_value``). Quads are bilinearly subdivided finely enough that
        every supervised column covered by the surface receives samples.
        """
        rows, cols = self._candidate_quads(segment, slab_min, slab_max)
        empty = np.zeros((0, 3)), np.zeros(0)
        if not len(rows):
            return empty
        xyz = segment.xyz
        corners = [
            frame.to_slab(xyz[rows, cols]),
            frame.to_slab(xyz[rows + 1, cols]),
            frame.to_slab(xyz[rows, cols + 1]),
            frame.to_slab(xyz[rows + 1, cols + 1]),
        ]

        size = self.transverse_size
        stacked = np.stack(corners)
        transverse_lo = stacked[..., :2].min(axis=0)
        transverse_hi = stacked[..., :2].max(axis=0)
        ray_lo = stacked[..., 2].min(axis=0)
        ray_hi = stacked[..., 2].max(axis=0)
        margin = _COLUMN_WINDOW
        keep = (
            (transverse_hi >= -margin).all(-1)
            & (transverse_lo <= size - 1 + margin).all(-1)
            & (ray_hi >= -0.5)
            & (ray_lo <= self.ray_length - 0.5)
        )
        if not keep.any():
            return empty
        rows, cols = rows[keep], cols[keep]
        corners = [corner[keep] for corner in corners]

        if wind_grid is None:
            winds = [np.full(len(rows), wind_value) for _ in range(4)]
        else:
            winds = [
                wind_grid[rows, cols].astype(np.float64),
                wind_grid[rows + 1, cols].astype(np.float64),
                wind_grid[rows, cols + 1].astype(np.float64),
                wind_grid[rows + 1, cols + 1].astype(np.float64),
            ]

        edges = np.maximum.reduce(
            [
                np.linalg.norm(corners[1][:, :2] - corners[0][:, :2], axis=-1),
                np.linalg.norm(corners[2][:, :2] - corners[0][:, :2], axis=-1),
                np.linalg.norm(corners[3][:, :2] - corners[1][:, :2], axis=-1),
                np.linalg.norm(corners[3][:, :2] - corners[2][:, :2], axis=-1),
            ]
        )
        subdivisions = np.clip(
            np.ceil(edges / _RASTER_STEP).astype(np.int64), 1, _MAX_SUBDIVISION
        )

        point_chunks, wind_chunks = [], []
        for level in np.unique(subdivisions):
            select = subdivisions == level
            steps = np.linspace(0.0, 1.0, level + 1)
            u_grid, v_grid = (
                grid.reshape(-1) for grid in np.meshgrid(steps, steps, indexing="ij")
            )
            samples = _bilinear(
                corners[0][select],
                corners[1][select],
                corners[2][select],
                corners[3][select],
                u_grid,
                v_grid,
            )
            sample_winds = _bilinear(
                winds[0][select, None],
                winds[1][select, None],
                winds[2][select, None],
                winds[3][select, None],
                u_grid,
                v_grid,
            )
            # Only samples within the column window of the supervised column
            # lattice can ever contribute (the exact filter _cluster_column_hits
            # applies); dropping the rest here keeps ~1/16 of the points, which
            # shrinks the concatenation and the clustering sort by the same
            # factor.
            flat = samples.reshape(-1, 3)
            flat_winds = sample_winds.reshape(-1)
            sample_keep = self._column_window_mask(flat)
            point_chunks.append(flat[sample_keep])
            wind_chunks.append(flat_winds[sample_keep])
        return np.concatenate(point_chunks), np.concatenate(wind_chunks)

    def _column_window_mask(self, points: np.ndarray) -> np.ndarray:
        """Samples close enough to a supervised column to be clustered.

        Must match _cluster_column_hits' keep condition exactly: the filter
        runs once here on the full point set and again there on the survivors
        (where it keeps everything), so any drift would change the targets.
        """
        stride = self.column_stride
        columns = self.columns_per_axis
        transverse = points[:, :2]
        nearest = np.rint(transverse / stride)
        offset = np.abs(transverse - nearest * stride)
        ray_position = points[:, 2]
        return (
            (offset <= _COLUMN_WINDOW).all(-1)
            & (nearest >= 0).all(-1)
            & (nearest <= columns - 1).all(-1)
            & (ray_position >= -0.5)
            & (ray_position <= self.ray_length - 0.5)
        )

    def _cluster_column_hits(
        self, points: np.ndarray, winds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign surface samples to supervised columns and cluster crossings.

        Returns (column_id, ray_position, winding_value) per crossing,
        ordered by column then ray position. Ray positions are fractional
        sample indices along the ray axis.
        """
        stride = self.column_stride
        columns = self.columns_per_axis
        transverse = points[:, :2]
        nearest = np.rint(transverse / stride)
        offset = np.abs(transverse - nearest * stride)
        ray_position = points[:, 2]
        keep = (
            (offset <= _COLUMN_WINDOW).all(-1)
            & (nearest >= 0).all(-1)
            & (nearest <= columns - 1).all(-1)
            & (ray_position >= -0.5)
            & (ray_position <= self.ray_length - 0.5)
        )
        if not keep.any():
            empty = np.zeros(0)
            return empty.astype(np.int64), empty, empty
        nearest = nearest[keep].astype(np.int64)
        ray_position = ray_position[keep]
        winds = winds[keep]
        column = nearest[:, 0] * columns + nearest[:, 1]

        order = np.lexsort((ray_position, column))
        column, ray_position, winds = (
            column[order], ray_position[order], winds[order]
        )
        boundary = np.ones(len(column), dtype=bool)
        boundary[1:] = (column[1:] != column[:-1]) | (
            np.diff(ray_position) > _CLUSTER_GAP
        )
        cluster = np.cumsum(boundary) - 1
        counts = np.bincount(cluster)
        return (
            column[boundary],
            np.bincount(cluster, ray_position) / counts,
            np.bincount(cluster, winds) / counts,
        )

    def _render_slab_targets(
        self,
        column_ids: np.ndarray,
        ray_positions: np.ndarray,
        winding_indices: np.ndarray,
        slab_image: np.ndarray,
        slab_valid: np.ndarray,
        *,
        crossings_only: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        """Filter each column's crossings and densify its targets.

        Columns keep crossings that sit on observable CT data, then walk
        them in ray order dropping duplicates (same index, or closer than
        the crossing merge distance) and truncating at the first winding
        reversal — the slab-global sign makes phase increase along the ray
        in every column, so a reversal marks geometry (a fold, a grazing
        sheet) whose labels are unreliable. Crossings without a neighbour
        exactly one winding away are dropped last; columns with fewer than
        two survivors are left fully unsupervised.

        With ``crossings_only`` the winding values carry no information:
        duplicates merge purely by distance, a single crossing suffices, and
        the rendered targets supervise the crossing head alone.
        """
        columns = self.columns_per_axis
        length = self.ray_length
        stride = self.column_stride

        boundaries = np.flatnonzero(
            np.r_[True, column_ids[1:] != column_ids[:-1]]
        )
        ends = np.r_[boundaries[1:], len(column_ids)] - 1
        if crossings_only:
            sign = 1.0
        else:
            # One winding sign per slab, voted by every column's endpoint
            # delta.
            sign = -1.0 if (winding_indices[ends] - winding_indices[boundaries]).sum() < 0 else 1.0

        dense_keys = (
            "phase_target",
            "phase_valid",
            "crossing_target",
            "crossing_valid",
            "density_target",
            "density_gap_wv",
        )
        dense = {
            key: np.zeros((columns * columns, length), dtype=np.float32)
            for key in dense_keys
        }
        kept_ts: list[np.ndarray] = [np.zeros(0)] * (columns * columns)
        kept_indices: list[np.ndarray] = [np.zeros(0, np.int64)] * (columns * columns)
        supervised_columns: list[int] = []
        merge_samples = self.crossing_merge_distance / self.spacing

        # Observability and index rounding are vectorized over all clusters
        # up front; the sequential dedup walk below then runs on plain
        # Python lists (per-column numpy calls dominated its cost).
        nearest = np.clip(np.rint(ray_positions).astype(np.int64), 0, length - 1)
        observable = (
            slab_image[
                (column_ids // columns) * stride,
                (column_ids % columns) * stride,
                nearest,
            ]
            > 0
        ) & slab_valid[
            (column_ids // columns) * stride,
            (column_ids % columns) * stride,
            nearest,
        ]
        rounded_indices = np.rint(sign * winding_indices).astype(np.int64)

        for start, end in zip(boundaries, ends + 1):
            if end - start < (1 if crossings_only else 2):
                continue
            column = int(column_ids[start])
            keep = observable[start:end]
            positions = ray_positions[start:end][keep].tolist()

            if crossings_only:
                kept_pos = []
                for position in positions:
                    if kept_pos and position - kept_pos[-1] <= merge_samples:
                        continue
                    kept_pos.append(position)
                if not kept_pos:
                    continue
                ts = np.asarray(kept_pos) * self.spacing
                idx = np.zeros(len(ts), dtype=np.int64)
            else:
                indices = rounded_indices[start:end][keep].tolist()
                kept_pos: list[float] = []
                kept_idx: list[int] = []
                for position, index in zip(positions, indices):
                    if kept_idx:
                        delta = index - kept_idx[-1]
                        if delta == 0 or position - kept_pos[-1] <= merge_samples:
                            continue
                        if delta < 0:
                            break
                    kept_pos.append(position)
                    kept_idx.append(index)

                anchored = [
                    i
                    for i in range(len(kept_idx))
                    if (i > 0 and kept_idx[i] - kept_idx[i - 1] == 1)
                    or (i + 1 < len(kept_idx) and kept_idx[i + 1] - kept_idx[i] == 1)
                ]
                if len(anchored) < 2:
                    continue
                ts = np.asarray([kept_pos[i] for i in anchored]) * self.spacing
                idx = np.asarray([kept_idx[i] for i in anchored], dtype=np.int64)
            kept_ts[column] = ts
            kept_indices[column] = idx
            supervised_columns.append(column)

        if len(supervised_columns) < self.min_supervised_columns:
            return None

        # One batched render over the supervised columns replaces the
        # per-column renderer calls, which dominated dataset CPU time.
        pad = max(len(kept_ts[column]) for column in supervised_columns)
        batch_t = np.zeros((len(supervised_columns), pad))
        batch_idx = np.zeros((len(supervised_columns), pad), dtype=np.int64)
        batch_counts = np.zeros(len(supervised_columns), dtype=np.int64)
        for slot, column in enumerate(supervised_columns):
            batch_counts[slot] = len(kept_ts[column])
            batch_t[slot, : batch_counts[slot]] = kept_ts[column]
            batch_idx[slot, : batch_counts[slot]] = kept_indices[column]
        rendered = render_column_targets_batched(
            batch_t,
            batch_idx,
            batch_counts,
            ray_length=length,
            spacing=self.spacing,
            crossing_sigma_wv=self.crossing_sigma_wv,
            crossings_only=crossings_only,
        )
        for key in dense_keys:
            dense[key][supervised_columns] = rendered[key]

        max_crossings = self.max_crossings
        crossing_t = np.full(
            (columns * columns, max_crossings), np.nan, dtype=np.float32
        )
        crossing_indices = np.zeros(
            (columns * columns, max_crossings), dtype=np.int64
        )
        num_crossings = np.zeros(columns * columns, dtype=np.int64)
        for column, (ts, idx) in enumerate(zip(kept_ts, kept_indices)):
            crossing_t[column, : len(ts)] = ts
            crossing_indices[column, : len(idx)] = idx
            num_crossings[column] = len(ts)

        grid = (columns, columns)
        sample = {
            key: torch.from_numpy(
                dense[key].reshape(*grid, length).astype(
                    bool if key.endswith("_valid") else np.float32
                )
            )
            for key in dense_keys
        }
        sample["crossing_t"] = torch.from_numpy(
            crossing_t.reshape(*grid, max_crossings)
        )
        sample["crossing_indices"] = torch.from_numpy(
            crossing_indices.reshape(*grid, max_crossings)
        )
        sample["num_crossings"] = torch.from_numpy(num_crossings.reshape(grid))
        return sample

    def _slab_bounds(self, frame: SlabFrame) -> tuple[np.ndarray, np.ndarray]:
        """Padded world AABB of the sampled slab."""
        size = self.transverse_size
        slab_corners = frame.to_world(
            np.array(
                [
                    (i, j, k)
                    for i in (0, size - 1)
                    for j in (0, size - 1)
                    for k in (0, self.ray_length - 1)
                ],
                dtype=np.float64,
            )
        )
        pad = self.spacing * (2.0 + _COLUMN_WINDOW)
        return slab_corners.min(axis=0) - pad, slab_corners.max(axis=0) + pad

    def _patches_in_bounds(
        self, volume: VolumeDataset, slab_min: np.ndarray, slab_max: np.ndarray
    ) -> list[PatchRef]:
        """Patch refs whose metadata bbox intersects the slab AABB."""
        overlap = (volume.patch_bbox_min <= slab_max).all(-1) & (
            volume.patch_bbox_max >= slab_min
        ).all(-1)
        return [volume.patches[i] for i in np.flatnonzero(overlap)]

    def _slab_targets(
        self,
        frame: SlabFrame,
        raster_specs: list[tuple[Segment, np.ndarray | None, float]],
        slab_image: np.ndarray,
        slab_valid: np.ndarray,
        *,
        crossings_only: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        slab_min, slab_max = self._slab_bounds(frame)

        point_chunks, wind_chunks = [], []
        for segment, wind_grid, wind_value in raster_specs:
            points, winds = self._rasterize_segment(
                segment, frame, wind_grid, wind_value, slab_min, slab_max
            )
            point_chunks.append(points)
            wind_chunks.append(winds)
        points = np.concatenate(point_chunks)
        if not len(points):
            return None
        column_ids, ray_positions, winding_values = self._cluster_column_hits(
            points, np.concatenate(wind_chunks)
        )
        if not len(column_ids):
            return None
        return self._render_slab_targets(
            column_ids,
            ray_positions,
            winding_values,
            slab_image,
            slab_valid,
            crossings_only=crossings_only,
        )

    def _sample_volume_and_segment(self) -> tuple[int, Segment]:
        pick = torch.rand((), dtype=torch.float64) * self._segment_cdf[-1]
        choice = int(torch.searchsorted(self._segment_cdf, pick, right=True))
        return self._segment_lookup[choice]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        for _ in range(self.max_sample_attempts):
            volume_idx, entry = self._sample_volume_and_segment()
            volume = self.volumes[volume_idx]
            crossings_only = volume.crossings_only
            if crossings_only:
                # Position-only patch: its own hit seeds the ray; every patch
                # intersecting the slab contributes crossing positives once
                # the slab frame is known.
                patch = self._load_patch(volume, entry, for_seed=True)
                if patch is None:
                    continue
                ray = self._ray(patch)
                if ray is None:
                    continue
                _, origin, direction = ray
                hits = [
                    (t, xyz, 0)
                    for t, xyz, _ in self._hits(
                        patch, origin, direction, self.ray_extent
                    )
                ]
                if not hits:
                    continue
                raster_specs = None
            elif entry.winding_idx is None:
                segment = entry
                ray = self._multi_wrap_ray(segment)
                if ray is None:
                    continue
                origin, direction, hits, turn_reference = ray
                raster_specs = [
                    (segment, segment.vertex_turns - turn_reference, 0.0)
                ]
            else:
                segment = entry
                ray = self._labelled_ray(volume, segment)
                if ray is None:
                    continue
                origin, direction, hits, bracket = ray
                raster_specs = [
                    (bracket_segment, None, float(bracket_segment.winding_idx))
                    for bracket_segment in bracket
                ]

            try:
                hits = self._filter_crossings(
                    volume_idx, hits, require_anchored=not crossings_only
                )
                if len(hits) < (1 if crossings_only else 2):
                    continue
                origin, hits = self._randomly_position_crossings(
                    origin, direction, hits
                )
                slab_image, slab_valid, frame = self.slab_extractor.extract(
                    volume_idx, direction, origin
                )
                if crossings_only:
                    slab_min, slab_max = self._slab_bounds(frame)
                    raster_specs = [
                        (neighbor, None, 0.0)
                        for ref in self._patches_in_bounds(
                            volume, slab_min, slab_max
                        )
                        for neighbor in (self._load_patch(volume, ref),)
                        if neighbor is not None
                    ]
                    if not raster_specs:
                        continue
                targets = self._slab_targets(
                    frame,
                    raster_specs,
                    slab_image,
                    slab_valid,
                    crossings_only=crossings_only,
                )
            except RuntimeError as exc:
                # A few locally stored Zarr chunks may be truncated. Discard
                # only samples that touch those chunks; unrelated runtime
                # failures must still terminate training.
                if _MALFORMED_CHUNK_ERROR not in str(exc):
                    raise
                continue
            if targets is None:
                continue
            sample = {
                "volume_idx": torch.tensor(volume_idx),
                "slab_image": torch.from_numpy(np.ascontiguousarray(slab_image)),
                "slab_valid": torch.from_numpy(
                    np.ascontiguousarray(slab_valid)
                ).bool(),
                "slab_origin_zyx": torch.from_numpy(
                    frame.origin[::-1].copy()
                ).float(),
                "slab_axis_a_zyx": torch.from_numpy(
                    frame.axis_a[::-1].copy()
                ).float(),
                "slab_axis_b_zyx": torch.from_numpy(
                    frame.axis_b[::-1].copy()
                ).float(),
                "ray_origin_zyx": torch.from_numpy(origin[::-1].copy()).float(),
                "ray_direction_zyx": torch.from_numpy(
                    frame.direction[::-1].copy()
                ).float(),
                "spacing": torch.tensor(self.spacing, dtype=torch.float32),
                "ray_extent": torch.tensor(self.ray_extent, dtype=torch.float32),
                "ray_length": torch.tensor(self.ray_length, dtype=torch.int64),
                "transverse_size": torch.tensor(
                    self.transverse_size, dtype=torch.int64
                ),
                "column_stride": torch.tensor(
                    self.column_stride, dtype=torch.int64
                ),
                **targets,
            }
            if self.multiview_fraction > 0.0:
                sample.update(self._multiview_pair(volume_idx, origin, frame))
            return sample

        raise RuntimeError("Could not find a slab with enough labeled columns")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect winding dataset samples")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="winding dataset configuration JSON",
    )
    args = parser.parse_args()
    from vesuvius.neural_tracing.winding_models.napari_helpers import (
        run_napari_inspector,
    )

    run_napari_inspector(args.config)


if __name__ == "__main__":
    main()
