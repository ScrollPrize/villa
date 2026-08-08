from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from skimage.restoration import unwrap_phase
from vc.grid_raycast import GridRaycaster

import vesuvius.tifxyz as tifxyz  # noqa: PLR0402
from vesuvius.neural_tracing.winding_models.volume_plane_extractor import (
    VolumePlaneExtractor,
)

_WINDING_NAME = re.compile(r"(?:^|-)w(?P<index>\d+)(?:_|$)")

_SEGMENT_CACHE_VERSION = 2


@dataclass
class Segment:
    winding_idx: int | None
    xyz: np.ndarray
    raycaster: GridRaycaster
    sample_cells: np.ndarray
    vertex_turns: np.ndarray | None


@dataclass
class VolumeDataset:
    volume_path: Path
    segments: list[Segment]
    segment_to_volume_xyz: np.ndarray


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
        legacy_shape = cfg.get("plane_shape", (256, 256))
        if isinstance(legacy_shape, int):
            legacy_shape = (legacy_shape, legacy_shape)
        if len(legacy_shape) != 2:
            raise ValueError("plane_shape must contain [height, width]")
        self.plane_height = int(cfg.get("plane_height", legacy_shape[0]))
        self.ray_length = int(cfg.get("ray_length", legacy_shape[1]))
        self.plane_spacing = float(cfg.get("plane_spacing", 1.0))
        self.num_planes = int(cfg.get("num_planes", 2))
        if self.plane_height < 2 or self.ray_length < 2:
            raise ValueError("plane_height and ray_length must be at least 2")
        if self.num_planes not in {2, 4}:
            raise ValueError("num_planes must be 2 or 4")
        cache_dir = cfg.get(
            "segment_cache_dir", "~/.cache/vesuvius/winding_model_segments"
        )
        self.segment_cache_dir = Path(cache_dir).expanduser() if cache_dir else None

        self.volumes = []
        for volume_cfg in cfg["datasets"]:
            scale = int(volume_cfg["volume_scale"])
            segments = [
                self._load_segment(info.path)
                for info in tifxyz.list_tifxyz(volume_cfg["segments_path"])
            ]
            if segments:
                configured_path = Path(volume_cfg["volume_path"])
                volume_path = VolumePlaneExtractor.scaled_volume_path(
                    configured_path, scale
                )
                segment_to_volume_xyz = (
                    VolumePlaneExtractor.load_segment_to_volume_transform(
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
                    VolumeDataset(volume_path, segments, segment_to_volume_xyz)
                )

        if not self.volumes:
            raise ValueError("No tifxyz segments were found")

        self.plane_extractor = VolumePlaneExtractor(
            [volume.volume_path for volume in self.volumes],
            shape=(self.plane_height, self.ray_length),
            spacing=self.plane_spacing,
            num_planes=self.num_planes,
            sampling=str(cfg.get("plane_sampling", "trilinear")),
            tile_size=int(cfg.get("plane_tile_size", 32)),
            cache_bytes=int(cfg.get("volume_cache_bytes", 0)),
            io_threads=int(cfg.get("volume_io_threads", 0)),
            segment_to_volume_xyz=[
                volume.segment_to_volume_xyz for volume in self.volumes
            ],
        )

    @property
    def ray_extent(self) -> float:
        """Physical ray extent represented by the sampled plane width."""
        return (self.ray_length - 1) * self.plane_spacing

    def set_plane_dimensions(self, *, ray_length: int, plane_height: int) -> None:
        """Set the output dimensions used for subsequent samples."""
        ray_length = int(ray_length)
        plane_height = int(plane_height)
        if ray_length < 2 or plane_height < 2:
            raise ValueError("ray_length and plane_height must be at least 2")
        self.ray_length = ray_length
        self.plane_height = plane_height
        self.plane_extractor.shape = (plane_height, ray_length)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _winding_index(path: Path) -> int | None:
        match = _WINDING_NAME.search(path.name)
        return int(match["index"]) if match else None

    def _load_segment(self, path: Path) -> Segment:
        surface = tifxyz.read_tifxyz(path)
        xyz = np.ascontiguousarray(
            np.stack([surface._x, surface._y, surface._z], axis=-1),
            dtype=np.float32,
        )
        winding_idx = self._winding_index(path)

        # Winding turns are only needed for segments without an explicit
        # winding index; unwrapping the phase of a large grid is the one
        # derivation expensive enough to cache.
        vertex_turns = None
        if winding_idx is None:
            cache_path = self._segment_cache_path(path)
            arrays = self._read_segment_cache(cache_path)
            if arrays is None or "vertex_turns" not in arrays:
                arrays = {
                    "vertex_turns": self._vertex_turns(
                        xyz, surface.valid_vertex_mask
                    )
                }
                self._write_segment_cache(cache_path, arrays)
            vertex_turns = arrays["vertex_turns"]

        valid_quads = self._inner_quads(surface.valid_quad_mask)
        return Segment(
            winding_idx=winding_idx,
            xyz=xyz,
            raycaster=GridRaycaster(xyz, np.ascontiguousarray(valid_quads)),
            sample_cells=np.argwhere(valid_quads),
            vertex_turns=vertex_turns,
        )

    def _inner_quads(self, valid_quads: np.ndarray) -> np.ndarray:
        """Restrict the valid-quad mask to the configured inner UV region."""
        valid_quads = valid_quads.copy()
        margin = (1.0 - self.inner_fraction) / 2.0
        height, width = valid_quads.shape
        inner = np.zeros_like(valid_quads)
        inner[
            int(margin * height) : int((1.0 - margin) * height),
            int(margin * width) : int((1.0 - margin) * width),
        ] = True
        valid_quads &= inner
        return valid_quads

    def _segment_cache_path(self, path: Path) -> Path | None:
        """Key the cache on the source files and version."""
        if self.segment_cache_dir is None:
            return None
        resolved = path.resolve()
        sources = sorted(resolved.iterdir()) if resolved.is_dir() else [resolved]
        stats = [
            (source.name, stat.st_mtime_ns, stat.st_size)
            for source in sources
            for stat in (source.stat(),)
        ]
        key = json.dumps([_SEGMENT_CACHE_VERSION, str(resolved), stats])
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
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]]] | None:
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

        return None if len(ordered) < 2 else (origin, direction, ordered)

    def _labelled_ray(
        self, volume: VolumeDataset, source: Segment
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]]] | None:
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
        hits = []
        for segment in volume.segments:
            if segment.winding_idx is None or not low <= segment.winding_idx <= high:
                continue
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
        return origin, direction, ordered

    def _sample_intersecting_planes(
        self,
        volume_idx: int,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        images, valid, geometry = self.plane_extractor.extract(
            volume_idx, direction, origin
        )
        return torch.from_numpy(images), torch.from_numpy(valid).bool(), geometry

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
        self, volume_idx: int, hits: list[tuple[float, np.ndarray, int]]
    ) -> list[tuple[float, np.ndarray, int]]:
        """Drop crossings that cannot serve as supervision targets.

        A crossing on zero-valued CT data sits outside the scanned material,
        so its position is not observable in the sampled planes. A crossing
        with no neighbour exactly one winding away (e.g. a lone index 18 after
        1, 2, 3) has no adjacent wrap anchoring its label; the zero-CT pass
        runs first because it can strand such crossings.
        """
        values = self.plane_extractor.sample_points(
            volume_idx, np.stack([xyz for _, xyz, _ in hits])
        )
        hits = [hit for hit, value in zip(hits, values) if value > 0]
        return [
            hit
            for i, hit in enumerate(hits)
            if (i > 0 and abs(hit[2] - hits[i - 1][2]) == 1)
            or (i + 1 < len(hits) and abs(hits[i + 1][2] - hit[2]) == 1)
        ]

    def _winding_valid_mask(
        self, hits: list[tuple[float, np.ndarray, int]]
    ) -> np.ndarray:
        """Mark ray samples whose winding labels are trustworthy.

        Consecutive crossings whose winding indices differ by more than one
        bracket a region that may contain unlabeled wraps; training should not
        penalize predictions there. Samples before the first crossing and
        after the last one are equally unconstrained.
        """
        sample_ts = np.arange(self.ray_length, dtype=np.float64) * self.plane_spacing
        valid = (sample_ts >= hits[0][0]) & (sample_ts <= hits[-1][0])
        for (t_near, _, near_idx), (t_far, _, far_idx) in itertools.pairwise(hits):
            if abs(far_idx - near_idx) > 1:
                valid &= (sample_ts <= t_near) | (sample_ts >= t_far)
        return valid

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        for _ in range(self.max_sample_attempts):
            volume_idx = self._randint(len(self.volumes))
            volume = self.volumes[volume_idx]
            segment = volume.segments[self._randint(len(volume.segments))]
            ray = (
                self._multi_wrap_ray(segment)
                if segment.winding_idx is None
                else self._labelled_ray(volume, segment)
            )
            if ray is None:
                continue

            origin, direction, hits = ray
            hits = self._filter_crossings(volume_idx, hits)
            if len(hits) < 2:
                continue
            origin, hits = self._randomly_position_crossings(
                origin, direction, hits
            )
            indices = torch.tensor([hit[2] for hit in hits], dtype=torch.int64)
            xyz = np.stack([hit[1] for hit in hits])
            plane_images, plane_valid, plane_geometry = (
                self._sample_intersecting_planes(volume_idx, origin, direction)
            )
            plane_origins, plane_x_steps, plane_y_steps = plane_geometry
            return {
                "volume_idx": torch.tensor(volume_idx),
                "plane_images": plane_images,
                "plane_valid": plane_valid,
                "plane_origins_zyx": torch.from_numpy(plane_origins[:, ::-1].copy()),
                "plane_x_steps_zyx": torch.from_numpy(plane_x_steps[:, ::-1].copy()),
                "plane_y_steps_zyx": torch.from_numpy(plane_y_steps[:, ::-1].copy()),
                "ray_origin_zyx": torch.from_numpy(origin[::-1].copy()).float(),
                "ray_direction_zyx": torch.from_numpy(direction[::-1].copy()).float(),
                "ray_extent": torch.tensor(self.ray_extent, dtype=torch.float32),
                "ray_length": torch.tensor(self.ray_length, dtype=torch.int64),
                "plane_height": torch.tensor(self.plane_height, dtype=torch.int64),
                "num_planes": torch.tensor(self.num_planes, dtype=torch.int64),
                "crossing_zyx": torch.from_numpy(xyz[:, ::-1].copy()).float(),
                "crossing_t": torch.tensor([hit[0] for hit in hits]).float(),
                "winding_indices": indices - indices[0],
                "winding_valid": torch.from_numpy(self._winding_valid_mask(hits)),
            }

        raise RuntimeError("Could not find a ray with at least two winding crossings")


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
