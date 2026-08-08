from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh
import numpy as np
import torch
import vesuvius.tifxyz as tifxyz
from skimage.restoration import unwrap_phase


_WINDING_NAME = re.compile(r"(?:^|-)w(?P<index>\d+)(?:_|$)")


def winding_index_from_path(path: Path) -> int | None:
    """Return the winding encoded in a segment name, if it has one."""
    match = _WINDING_NAME.search(path.name)
    return int(match["index"]) if match is not None else None


@dataclass(frozen=True)
class Segment:
    path: Path
    winding_idx: int | None
    surface: tifxyz.Tifxyz
    mesh: Any
    inner_valid_quads: np.ndarray
    face_winding_turns: np.ndarray | None


@dataclass
class VolumeDataset:
    volume_scale: int
    volume_path: Path
    segments: list[Segment] = field(default_factory=list)


class WindingModelDataset(torch.utils.data.Dataset):
    """Sample ordered surface crossings along rays through papyrus windings.

    This currently returns ray geometry and winding indices only. Volume
    sampling and model-specific padding/collation are intentionally left for a
    later stage.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.volumes: list[VolumeDataset] = []

        self.num_samples = int(cfg.get("num_samples", 500_000))
        self.inner_fraction = float(cfg.get("inner_fraction", 0.70))
        self.min_winding_gap = int(cfg.get("min_winding_gap", 4))
        self.max_sample_attempts = int(cfg.get("max_sample_attempts", 64))
        self.ray_skew_degrees = float(cfg.get("ray_skew_degrees", 3.0))
        self.ray_origin_offset = float(cfg.get("ray_origin_offset", 1.0))
        self.hit_merge_tolerance = float(cfg.get("hit_merge_tolerance", 1e-3))

        for dataset in cfg["datasets"]:
            volume = VolumeDataset(
                volume_scale=int(dataset["volume_scale"]),
                volume_path=Path(dataset["volume_path"]),
            )

            for segment_info in tifxyz.list_tifxyz(dataset["segments_path"]):
                volume.segments.append(
                    self._load_segment(
                        path=segment_info.path,
                        winding_idx=winding_index_from_path(segment_info.path),
                        volume_scale=volume.volume_scale,
                    )
                )

            if volume.segments:
                self.volumes.append(volume)

        if not self.volumes:
            raise ValueError("No tifxyz segments were found in the configured datasets")

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _randint(high: int) -> int:
        return int(torch.randint(high, size=()).item())

    @staticmethod
    def _rand() -> float:
        # torch's RNG is seeded separately for each PyTorch DataLoader worker.
        return float(torch.rand(()).item())

    def _load_segment(
        self, *, path: Path, winding_idx: int | None, volume_scale: int
    ) -> Segment:
        """Load one segment and build all geometry used by __getitem__."""
        surface = tifxyz.read_tifxyz(path)
        factor = float(2**volume_scale)
        if factor != 1.0:
            surface = surface.retarget(factor)

        valid_quads = surface.valid_quad_mask
        height, width = valid_quads.shape
        margin = (1.0 - self.inner_fraction) / 2.0
        row_start = int(math.floor(margin * height))
        row_end = int(math.ceil((1.0 - margin) * height))
        col_start = int(math.floor(margin * width))
        col_end = int(math.ceil((1.0 - margin) * width))

        inner_mask = np.zeros_like(valid_quads)
        inner_mask[row_start:row_end, col_start:col_end] = True
        inner_valid_quads = np.argwhere(valid_quads & inner_mask)

        rows, cols = np.nonzero(valid_quads)
        top_left = rows * (width + 1) + cols
        faces = np.concatenate(
            [
                np.stack([top_left, top_left + width + 1, top_left + 1], axis=1),
                np.stack(
                    [top_left + 1, top_left + width + 1, top_left + width + 2],
                    axis=1,
                ),
            ],
            axis=0,
        )

        vertices = np.stack(
            [surface._x, surface._y, surface._z], axis=-1
        ).reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            validate=False,
        )
        mesh.remove_unreferenced_vertices()

        face_winding_turns = None
        if winding_idx is None:
            vertex_turns = self._winding_turns(surface)
            face_winding_turns = np.concatenate(
                [
                    (
                        vertex_turns[rows, cols]
                        + vertex_turns[rows + 1, cols]
                        + vertex_turns[rows, cols + 1]
                    )
                    / 3.0,
                    (
                        vertex_turns[rows, cols + 1]
                        + vertex_turns[rows + 1, cols]
                        + vertex_turns[rows + 1, cols + 1]
                    )
                    / 3.0,
                ]
            ).astype(np.float32, copy=False)

        return Segment(
            path=path,
            winding_idx=winding_idx,
            surface=surface,
            mesh=mesh,
            inner_valid_quads=inner_valid_quads,
            face_winding_turns=face_winding_turns,
        )

    @staticmethod
    def _winding_turns(surface: tifxyz.Tifxyz) -> np.ndarray:
        """Unwrap each vertex's angle around the mesh centroid over its UV mesh.

        The sign and zero point are arbitrary; differences of one represent a
        complete winding around the centroid. Masked phase unwrapping uses both
        UV axes and therefore does not assume which axis follows the winding.
        """
        valid = surface.valid_vertex_mask
        centroid_x = float(surface._x[valid].mean())
        centroid_y = float(surface._y[valid].mean())
        wrapped = np.arctan2(
            surface._y - centroid_y,
            surface._x - centroid_x,
        )
        masked_phase = np.ma.array(wrapped, mask=~valid)
        unwrapped = unwrap_phase(masked_phase, rng=0)
        return (unwrapped.filled(np.nan) / (2.0 * math.pi)).astype(np.float32)

    def _sample_cell(
        self, segment: Segment
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(segment.inner_valid_quads) == 0:
            return None

        row, col = segment.inner_valid_quads[
            self._randint(len(segment.inner_valid_quads))
        ]
        xyz = np.stack(
            [
                segment.surface._x[row : row + 2, col : col + 2],
                segment.surface._y[row : row + 2, col : col + 2],
                segment.surface._z[row : row + 2, col : col + 2],
            ],
            axis=-1,
        ).astype(np.float64, copy=False)

        # The midpoint of the quad's shared triangle edge is guaranteed to lie
        # on the mesh. A four-corner mean need not lie on a non-planar quad.
        point = 0.5 * (xyz[1, 0] + xyz[0, 1])
        normal_a = np.cross(xyz[1, 0] - xyz[0, 0], xyz[0, 1] - xyz[0, 0])
        normal_b = np.cross(xyz[0, 1] - xyz[1, 1], xyz[1, 0] - xyz[1, 1])
        normal = normal_a + normal_b
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None
        return point, normal / norm

    def _augment_direction(self, normal: np.ndarray) -> np.ndarray:
        direction = normal.astype(np.float64, copy=True)
        if self.ray_skew_degrees > 0.0:
            tangent = np.asarray(
                [self._rand() - 0.5, self._rand() - 0.5, self._rand() - 0.5],
                dtype=np.float64,
            )
            tangent -= np.dot(tangent, direction) * direction
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-8:
                tangent /= tangent_norm
                angle = math.radians(self.ray_skew_degrees * self._rand())
                direction = math.cos(angle) * direction + math.sin(angle) * tangent

        direction /= np.linalg.norm(direction)
        return direction

    def _ray_hits(
        self,
        segment: Segment,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        max_t: float | None = None,
    ) -> list[tuple[float, np.ndarray, int]]:
        locations, _, triangle_indices = segment.mesh.ray.intersects_location(
            ray_origins=origin[None],
            ray_directions=direction[None],
            multiple_hits=True,
        )
        if len(locations) == 0:
            return []

        ts = (locations - origin) @ direction
        order = np.argsort(ts)
        hits: list[tuple[float, np.ndarray, int]] = []
        for hit_idx in order:
            t = float(ts[hit_idx])
            if t < 0.0 or (max_t is not None and t > max_t + self.hit_merge_tolerance):
                continue
            if hits and abs(t - hits[-1][0]) <= self.hit_merge_tolerance:
                continue
            hits.append(
                (
                    t,
                    locations[hit_idx].astype(np.float64, copy=False),
                    int(triangle_indices[hit_idx]),
                )
            )
        return hits

    def _make_ray(
        self, point: np.ndarray, normal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        direction = self._augment_direction(normal)
        if self._rand() < 0.5:
            direction = -direction
        offset = self.ray_origin_offset * (0.5 + 0.5 * self._rand())
        origin = point - offset * direction
        return origin, direction

    def _sample_multi_wrap(
        self, volume_idx: int, segment: Segment
    ) -> dict[str, torch.Tensor] | None:
        sampled_cell = self._sample_cell(segment)
        if sampled_cell is None:
            return None

        origin, direction = self._make_ray(*sampled_cell)
        ray_hits = self._ray_hits(segment, origin, direction)
        if len(ray_hits) < 2:
            return None

        assert segment.face_winding_turns is not None
        hit_turns = np.asarray(
            [segment.face_winding_turns[triangle_idx] for _, _, triangle_idx in ray_hits]
        )
        if not np.isfinite(hit_turns).all():
            return None
        winding_indices = np.rint(hit_turns - hit_turns[0]).astype(np.int64)

        # An inward-facing ray may eventually pass through the center and hit
        # the far side of the same rolled sheet. Keep only the first monotonic
        # stack of windings. Equal indices are duplicate crossings of one wrap;
        # a sign reversal means the ray has reached the opposite side.
        ordered_hits: list[tuple[float, np.ndarray, int]] = []
        winding_direction = 0
        previous_idx: int | None = None
        for (t, xyz, _), winding_idx_value in zip(ray_hits, winding_indices):
            winding_idx = int(winding_idx_value)
            if previous_idx is not None:
                delta = winding_idx - previous_idx
                if delta == 0:
                    continue
                if winding_direction == 0:
                    winding_direction = 1 if delta > 0 else -1
                elif delta * winding_direction < 0:
                    break
            ordered_hits.append((t, xyz, winding_idx))
            previous_idx = winding_idx

        if len(ordered_hits) < 2:
            return None

        crossing_xyz = np.stack([hit[1] for hit in ordered_hits])
        winding_indices = torch.tensor(
            [hit[2] for hit in ordered_hits], dtype=torch.int64
        )
        return {
            "volume_idx": torch.tensor(volume_idx, dtype=torch.int64),
            "ray_origin_zyx": torch.from_numpy(origin[::-1].copy()).float(),
            "ray_direction_zyx": torch.from_numpy(direction[::-1].copy()).float(),
            "crossing_zyx": torch.from_numpy(crossing_xyz[:, ::-1].copy()).float(),
            "crossing_t": torch.tensor(
                [hit[0] for hit in ordered_hits], dtype=torch.float32
            ),
            "winding_indices": winding_indices,
            "absolute_winding_indices": torch.full_like(winding_indices, -1),
            "has_absolute_winding_indices": torch.tensor(False),
        }

    def _sample_labelled_wrap(
        self, volume_idx: int, volume: VolumeDataset, source: Segment
    ) -> dict[str, torch.Tensor] | None:
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
        sampled_cell = self._sample_cell(source)
        if sampled_cell is None:
            return None

        origin, direction = self._make_ray(*sampled_cell)
        target_hits = self._ray_hits(target, origin, direction)
        if not target_hits:
            # Normal orientation is arbitrary, so try the same skewed line in
            # the other direction rather than requiring an inward/outward rule.
            direction = -direction
            point = sampled_cell[0]
            offset = self.ray_origin_offset * (0.5 + 0.5 * self._rand())
            origin = point - offset * direction
            target_hits = self._ray_hits(target, origin, direction)
        if not target_hits:
            return None

        max_t = target_hits[0][0]
        assert target.winding_idx is not None
        winding_min = min(source.winding_idx, target.winding_idx)
        winding_max = max(source.winding_idx, target.winding_idx)
        labelled_hits: list[tuple[float, np.ndarray, int]] = []
        for segment in volume.segments:
            if (
                segment.winding_idx is None
                or not winding_min <= segment.winding_idx <= winding_max
            ):
                continue
            for t, xyz, _ in self._ray_hits(
                segment, origin, direction, max_t=max_t
            ):
                labelled_hits.append((t, xyz, segment.winding_idx))

        labelled_hits.sort(key=lambda hit: hit[0])
        winding_direction = 1 if target.winding_idx > source.winding_idx else -1
        ordered_hits: list[tuple[float, np.ndarray, int]] = []
        for hit in labelled_hits:
            if ordered_hits and hit[2] == ordered_hits[-1][2]:
                continue
            if (
                ordered_hits
                and (hit[2] - ordered_hits[-1][2]) * winding_direction < 0
            ):
                return None
            ordered_hits.append(hit)

        if (
            len(ordered_hits) < 2
            or ordered_hits[0][2] != source.winding_idx
            or ordered_hits[-1][2] != target.winding_idx
        ):
            return None

        crossing_xyz = np.stack([hit[1] for hit in ordered_hits])
        absolute_indices = torch.tensor(
            [hit[2] for hit in ordered_hits], dtype=torch.int64
        )
        return {
            "volume_idx": torch.tensor(volume_idx, dtype=torch.int64),
            "ray_origin_zyx": torch.from_numpy(origin[::-1].copy()).float(),
            "ray_direction_zyx": torch.from_numpy(direction[::-1].copy()).float(),
            "crossing_zyx": torch.from_numpy(crossing_xyz[:, ::-1].copy()).float(),
            "crossing_t": torch.tensor(
                [hit[0] for hit in ordered_hits], dtype=torch.float32
            ),
            "winding_indices": absolute_indices - absolute_indices[0],
            "absolute_winding_indices": absolute_indices,
            "has_absolute_winding_indices": torch.tensor(True),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Samples are stochastic; DataLoader worker seeding controls
        # reproducibility rather than the nominal dataset index.
        del idx

        for _ in range(self.max_sample_attempts):
            volume_idx = self._randint(len(self.volumes))
            volume = self.volumes[volume_idx]
            segment = volume.segments[self._randint(len(volume.segments))]

            if segment.winding_idx is None:
                sample = self._sample_multi_wrap(volume_idx, segment)
            else:
                sample = self._sample_labelled_wrap(volume_idx, volume, segment)

            if sample is not None:
                return sample

        raise RuntimeError(
            "Could not construct a ray with at least two surface crossings after "
            f"{self.max_sample_attempts} attempts"
        )
