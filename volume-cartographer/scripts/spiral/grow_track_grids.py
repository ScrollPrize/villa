#!/usr/bin/env python3
"""Generate topology-safe TIFXYZ grids from exact track crossings.

The expensive track/crossing index is constructed once per ROI and cached.  A
Linux fork worker pool then grows independent surfaces from randomly selected,
spatially separated seed tracks without copying that index.

Example:

    python scripts/spiral/grow_track_grids.py \
      /path/to/tracks.dbm /tmp/track-grids \
      --center-xyz 3848 2775 8212 \
      --bbox-size-zyx 3000 1000 1000 \
      --length-bin long --count 32 --seed-spacing 256 --workers 8

Coordinates and distances are full-resolution voxels.  Crossings are exact
shared voxels; near misses are never treated as contacts.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import dbm
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import time
from typing import Callable, Iterable

import numpy as np
import tifffile
from tqdm import tqdm


CACHE_VERSION = 2
QUADRANTS = tuple(itertools.product((-1, 1), repeat=2))


@dataclasses.dataclass
class TrackGraph:
    tracks: list[np.ndarray]
    provenance: list[tuple[str, str, int]]
    cumulative: list[np.ndarray]
    lengths: np.ndarray
    representative_zyx: np.ndarray
    edges: list[tuple[int, int, float, float, int, int, np.ndarray]]
    incident: dict[int, list[tuple[int, float, int]]]
    pair_edges: dict[tuple[int, int], list[int]]
    roi_lo: np.ndarray | None
    roi_hi: np.ndarray | None
    z_range: tuple[int, int]


@dataclasses.dataclass
class GrowthConfig:
    min_side: float
    max_side: float
    max_recenters: int
    max_edge_tortuosity: float = 1.35
    max_edge_deviation: float = 0.20
    min_normal_dot: float = 0.0
    max_local_recenters: int = 0
    max_local_attempts: int = 0


@dataclasses.dataclass
class Surface:
    anchor: int
    partner: int
    center_edge: int
    base_chains: dict[tuple[int, int], list[dict]]
    grid: dict
    initial_quadrant_counts: dict[str, int]
    recenter_history: list[dict]
    growth_stats: dict = dataclasses.field(default_factory=dict)


def normalize_dbm_path(path: str | Path) -> str:
    text = str(path)
    if dbm.whichdb(text):
        return text
    if text.endswith(".db") and dbm.whichdb(text[:-3]):
        return text[:-3]
    raise FileNotFoundError(f"not a readable DBM logical path: {path}")


def db_signature(path: str) -> list[tuple[str, int, int]]:
    logical = Path(path)
    matches = sorted(logical.parent.glob(logical.name + "*"))
    return [(item.name, item.stat().st_size, item.stat().st_mtime_ns)
            for item in matches if item.is_file()]


def cumulative_length(track: np.ndarray) -> np.ndarray:
    if len(track) < 2:
        return np.zeros(len(track), dtype=np.float64)
    step = np.linalg.norm(np.diff(track.astype(np.float64), axis=0), axis=1)
    return np.r_[0.0, np.cumsum(step)]


def pack_points(points: np.ndarray) -> np.ndarray:
    if np.any(points < 0) or np.any(points >= (1 << 20)):
        raise ValueError("track coordinates must lie in [0, 2**20) for voxel packing")
    packed = points.astype(np.uint64, copy=False)
    return ((packed[:, 0] << np.uint64(40))
            | (packed[:, 1] << np.uint64(20)) | packed[:, 2])


def tangent(track: np.ndarray, raw_index: int, radius_voxels: float) -> np.ndarray | None:
    point = track[raw_index].astype(np.float64)
    left = raw_index
    while left > 0 and np.linalg.norm(track[left].astype(np.float64) - point) < radius_voxels:
        left -= 1
    right = raw_index
    while (right + 1 < len(track)
           and np.linalg.norm(track[right].astype(np.float64) - point) < radius_voxels):
        right += 1
    vector = track[right].astype(np.float64) - track[left].astype(np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def key_in_roi(key: bytes, lo: np.ndarray | None, hi: np.ndarray | None) -> bool:
    if lo is None or hi is None:
        return True
    try:
        prefix, value_text = key.decode().split(":", 1)
        value = int(value_text)
    except (UnicodeDecodeError, ValueError):
        return False
    axis = {"h": 0, "vy": 1, "vx": 2}.get(prefix)
    return axis is not None and lo[axis] - 4 <= value < hi[axis] + 4


def centered_roi(center_xyz: Iterable[float],
                 bbox_size_zyx: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    center_xyz_array = np.asarray(tuple(center_xyz), dtype=np.float64)
    size_zyx = np.asarray(tuple(bbox_size_zyx), dtype=np.float64)
    if center_xyz_array.shape != (3,) or size_zyx.shape != (3,):
        raise ValueError("center and bounding-box size must each contain three values")
    if not np.all(np.isfinite(center_xyz_array)) or not np.all(np.isfinite(size_zyx)):
        raise ValueError("center and bounding-box size must be finite")
    if np.any(size_zyx <= 0):
        raise ValueError("bounding-box edge lengths must be positive")
    center_zyx = center_xyz_array[[2, 1, 0]]
    half_edge_zyx = size_zyx * 0.5
    return center_zyx - half_edge_zyx, center_zyx + half_edge_zyx


def load_tracks(path: str, z_range: tuple[int, int],
                roi_lo: np.ndarray | None, roi_hi: np.ndarray | None,
                show_progress: bool) -> tuple[list[np.ndarray], list[tuple[str, str, int]]]:
    tracks: list[np.ndarray] = []
    provenance: list[tuple[str, str, int]] = []
    with dbm.open(path, "r") as database:
        keys = sorted(key for key in database.keys() if key_in_roi(key, roi_lo, roi_hi))
        iterator = tqdm(keys, desc="loading track keys", disable=not show_progress)
        for key in iterator:
            key_text = key.decode()
            prefix = key_text.split(":", 1)[0]
            entries = pickle.loads(database[key])
            for source_index, value in enumerate(entries):
                if not len(value):
                    continue
                track = np.asarray(value, dtype=np.int32)
                if track[:, 0].min() < z_range[0] or track[:, 0].max() >= z_range[1]:
                    continue
                if roi_lo is not None:
                    inside = np.all(track >= roi_lo, axis=1) & np.all(track < roi_hi, axis=1)
                    if not np.any(inside):
                        continue
                tracks.append(track)
                provenance.append((prefix, key_text, source_index))
    return tracks, provenance


def build_graph(path: str, z_range: tuple[int, int],
                roi_lo: np.ndarray | None, roi_hi: np.ndarray | None,
                angle_degrees: float, tangent_radius: float,
                show_progress: bool = True) -> TrackGraph:
    started = time.perf_counter()
    tracks, provenance = load_tracks(path, z_range, roi_lo, roi_hi, show_progress)
    if not tracks:
        raise RuntimeError("no tracks survived the requested ROI and z range")

    cumulative = [cumulative_length(track) for track in tracks]
    lengths = np.asarray([value[-1] if len(value) else 0.0 for value in cumulative])
    representative = np.stack([
        track[int(np.searchsorted(distance, distance[-1] * 0.5))]
        for track, distance in zip(tracks, cumulative)
    ]).astype(np.float32)

    point_parts = []
    track_parts = []
    local_parts = []
    for track_id, track in enumerate(tracks):
        if roi_lo is None:
            local = np.arange(len(track), dtype=np.int32)
        else:
            inside = np.all(track >= roi_lo, axis=1) & np.all(track < roi_hi, axis=1)
            local = np.flatnonzero(inside).astype(np.int32)
        if not len(local):
            continue
        point_parts.append(track[local])
        track_parts.append(np.full(len(local), track_id, dtype=np.int32))
        local_parts.append(local)
    if not point_parts:
        raise RuntimeError("no track points lie inside the crossing ROI")

    points = np.concatenate(point_parts)
    track_ids = np.concatenate(track_parts)
    local_indices = np.concatenate(local_parts)
    packed = pack_points(points)
    order = np.argsort(packed, kind="stable")
    packed = packed[order]
    points = points[order]
    track_ids = track_ids[order]
    local_indices = local_indices[order]
    boundaries = np.flatnonzero(packed[1:] != packed[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(packed)]

    angle_cutoff = math.cos(math.radians(angle_degrees))
    tangent_cache: dict[tuple[int, int], np.ndarray | None] = {}
    raw_events: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = (
        collections.defaultdict(list))
    iterator = zip(starts, stops)
    if show_progress:
        iterator = tqdm(iterator, total=len(starts), desc="finding exact crossings")
    for start, stop in iterator:
        if stop - start < 2:
            continue
        unique: dict[int, int] = {}
        for position in range(int(start), int(stop)):
            unique.setdefault(int(track_ids[position]), int(local_indices[position]))
        if len(unique) < 2:
            continue
        for first, second in itertools.combinations(unique, 2):
            first_index, second_index = unique[first], unique[second]
            first_key = (first, first_index)
            second_key = (second, second_index)
            if first_key not in tangent_cache:
                tangent_cache[first_key] = tangent(
                    tracks[first], first_index, tangent_radius)
            if second_key not in tangent_cache:
                tangent_cache[second_key] = tangent(
                    tracks[second], second_index, tangent_radius)
            first_tangent = tangent_cache[first_key]
            second_tangent = tangent_cache[second_key]
            if first_tangent is None or second_tangent is None:
                continue
            if abs(float(np.dot(first_tangent, second_tangent))) > angle_cutoff:
                continue
            if first > second:
                first, second = second, first
                first_index, second_index = second_index, first_index
            raw_events[(first, second)].append(
                (first_index, second_index, points[int(start)].copy()))

    compact = []
    for (first, second), events in raw_events.items():
        events.sort(key=lambda event: (event[0], event[1]))
        cluster = []
        for event in events:
            if (cluster and abs(event[0] - cluster[-1][0]) <= 4
                    and abs(event[1] - cluster[-1][1]) <= 4):
                cluster.append(event)
            else:
                if cluster:
                    compact.append((first, second, cluster[len(cluster) // 2]))
                cluster = [event]
        if cluster:
            compact.append((first, second, cluster[len(cluster) // 2]))

    edges = []
    incident: dict[int, list[tuple[int, float, int]]] = collections.defaultdict(list)
    pair_edges: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for first, second, (first_index, second_index, point) in compact:
        edge_id = len(edges)
        edge = (first, second,
                float(cumulative[first][first_index]),
                float(cumulative[second][second_index]),
                first_index, second_index, point)
        edges.append(edge)
        incident[first].append((edge_id, edge[2], second))
        incident[second].append((edge_id, edge[3], first))
        pair_edges[(first, second)].append(edge_id)

    # Space batch seeds by where they participate in this crossing graph, not by
    # a full-track midpoint which may lie well outside a small requested ROI.
    crossing_sum = np.zeros((len(tracks), 3), dtype=np.float64)
    crossing_count = np.zeros(len(tracks), dtype=np.int64)
    for first, second, *_, point in edges:
        crossing_sum[first] += point
        crossing_sum[second] += point
        crossing_count[first] += 1
        crossing_count[second] += 1
    has_crossing = crossing_count > 0
    representative[has_crossing] = (
        crossing_sum[has_crossing] / crossing_count[has_crossing, None])

    graph = TrackGraph(
        tracks=tracks,
        provenance=provenance,
        cumulative=cumulative,
        lengths=lengths,
        representative_zyx=representative,
        edges=edges,
        incident=dict(incident),
        pair_edges=dict(pair_edges),
        roi_lo=roi_lo,
        roi_hi=roi_hi,
        z_range=z_range,
    )
    if show_progress:
        print(f"indexed {len(tracks):,} tracks, {len(points):,} ROI points and "
              f"{len(edges):,} crossings in {time.perf_counter() - started:.1f}s")
    return graph


def cache_key(path: str, z_range: tuple[int, int], roi_lo: np.ndarray | None,
              roi_hi: np.ndarray | None, angle_degrees: float,
              tangent_radius: float) -> dict:
    return {
        "version": CACHE_VERSION,
        "db": str(Path(path).resolve()),
        "db_signature": db_signature(path),
        "z_range": list(z_range),
        "roi_lo": None if roi_lo is None else roi_lo.tolist(),
        "roi_hi": None if roi_hi is None else roi_hi.tolist(),
        "angle_degrees": angle_degrees,
        "tangent_radius": tangent_radius,
    }


def load_or_build_graph(path: str, cache_path: Path | None,
                        z_range: tuple[int, int], roi_lo: np.ndarray | None,
                        roi_hi: np.ndarray | None, angle_degrees: float,
                        tangent_radius: float) -> tuple[TrackGraph, bool, float]:
    signature = cache_key(path, z_range, roi_lo, roi_hi,
                          angle_degrees, tangent_radius)
    started = time.perf_counter()
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as stream:
                cached = pickle.load(stream)
            if cached.get("signature") == signature:
                payload = cached["graph"]
                graph = TrackGraph(**payload) if isinstance(payload, dict) else payload
                return graph, True, time.perf_counter() - started
        except (OSError, EOFError, AttributeError, pickle.UnpicklingError):
            pass

    graph = build_graph(path, z_range, roi_lo, roi_hi,
                        angle_degrees, tangent_radius)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + f".tmp-{os.getpid()}")
        payload = {field.name: getattr(graph, field.name)
                   for field in dataclasses.fields(graph)}
        with temporary.open("wb") as stream:
            pickle.dump({"signature": signature, "graph": payload}, stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    return graph, False, time.perf_counter() - started


class GrowthEngine:
    def __init__(self, graph: TrackGraph, config: GrowthConfig):
        self.graph = graph
        self.config = config
        self._candidate_cache: dict[tuple[int, int, int], list[dict]] = {}
        self._track_geometry_cache: dict[tuple[int, int, int], bool] = {}
        self._cell_normal_cache: dict[tuple[int, int, int, int], np.ndarray | None] = {}

    def edge_position(self, edge_id: int, track_id: int) -> float:
        edge = self.graph.edges[edge_id]
        return edge[2] if edge[0] == track_id else edge[3]

    def pair(self, first: int, second: int) -> tuple[int, int]:
        return (first, second) if first < second else (second, first)

    def track_family(self, track_id: int) -> str:
        # vx/vy are two extraction planes for the same vertical family.
        return "horizontal" if self.graph.provenance[track_id][0] == "h" else "vertical"

    def base_cross(self, anchor: int):
        return next(iter(self.base_crosses(anchor)), None)

    def base_crosses(self, anchor: int) -> list[tuple[float, float, int, int]]:
        partners: dict[int, list[int]] = collections.defaultdict(list)
        for edge_id, _, partner in self.graph.incident.get(anchor, ()):
            partners[partner].append(edge_id)
        choices = []
        for partner, edge_ids in partners.items():
            if self.track_family(partner) == self.track_family(anchor):
                continue
            best_edge = None
            best_clearance = -1.0
            for edge_id in edge_ids:
                anchor_position = self.edge_position(edge_id, anchor)
                partner_position = self.edge_position(edge_id, partner)
                clearance = min(
                    anchor_position, self.graph.lengths[anchor] - anchor_position,
                    partner_position, self.graph.lengths[partner] - partner_position)
                if clearance > best_clearance:
                    best_clearance, best_edge = clearance, edge_id
            if best_clearance >= self.config.min_side:
                choices.append((self.graph.lengths[partner], best_clearance,
                                partner, best_edge))
        return sorted(choices, reverse=True)

    def growth_candidates(self, anchor: int, partner: int,
                          center_edge: int) -> list[dict]:
        cache_key_value = (anchor, partner, center_edge)
        if cache_key_value in self._candidate_cache:
            return self._candidate_cache[cache_key_value]
        center_anchor = self.edge_position(center_edge, anchor)
        center_partner = self.edge_position(center_edge, partner)
        maximum = self.config.max_side
        anchor_arms = [
            (edge_id, position, other)
            for edge_id, position, other in self.graph.incident.get(anchor, ())
            if edge_id != center_edge and other not in (anchor, partner)
            and self.config.min_side <= abs(position - center_anchor) <= maximum
        ]
        partner_arms = [
            (edge_id, position, other)
            for edge_id, position, other in self.graph.incident.get(partner, ())
            if edge_id != center_edge and other not in (anchor, partner)
            and self.config.min_side <= abs(position - center_partner) <= maximum
        ]
        result = {}
        for anchor_edge, anchor_position, outer_row in anchor_arms:
            for partner_edge, partner_position, outer_column in partner_arms:
                if outer_row == outer_column:
                    continue
                if (self.track_family(outer_row) != self.track_family(partner)
                        or self.track_family(outer_column) != self.track_family(anchor)):
                    continue
                for corner_edge in self.graph.pair_edges.get(
                        self.pair(outer_row, outer_column), ()):
                    row_travel = abs(self.edge_position(corner_edge, outer_row)
                                     - self.edge_position(anchor_edge, outer_row))
                    column_travel = abs(self.edge_position(corner_edge, outer_column)
                                        - self.edge_position(partner_edge, outer_column))
                    if not (self.config.min_side <= row_travel <= maximum
                            and self.config.min_side <= column_travel <= maximum):
                        continue
                    key = tuple(sorted((center_edge, anchor_edge,
                                        corner_edge, partner_edge)))
                    result[key] = {
                        "center_tracks": (anchor, partner),
                        "quadrant": (
                            1 if anchor_position > center_anchor else -1,
                            1 if partner_position > center_partner else -1),
                        "distance_a": abs(anchor_position - center_anchor),
                        "distance_b": abs(partner_position - center_partner),
                        "max_leg": max(abs(anchor_position - center_anchor),
                                       abs(partner_position - center_partner),
                                       row_travel, column_travel),
                        "outer_tracks": (outer_row, outer_column),
                        "edges": (center_edge, anchor_edge,
                                  corner_edge, partner_edge),
                    }
        candidates = list(result.values())
        self._candidate_cache[cache_key_value] = candidates
        return candidates

    def complete_chain_options(self, candidates: list[dict], anchor: int,
                               partner: int) -> list[dict]:
        by_quadrant = {
            quadrant: [candidate for candidate in candidates
                       if candidate["quadrant"] == quadrant]
            for quadrant in QUADRANTS
        }
        sort_keys: tuple[Callable[[dict], tuple], ...] = (
            lambda value: (max(value["distance_a"], value["distance_b"]),
                           value["distance_a"] + value["distance_b"],
                           value["max_leg"]),
            lambda value: (value["distance_a"] + value["distance_b"],
                           value["max_leg"]),
            lambda value: (value["distance_a"], value["distance_b"],
                           value["max_leg"]),
            lambda value: (value["distance_b"], value["distance_a"],
                           value["max_leg"]),
        )
        options = {}
        for quadrant_order in itertools.permutations(QUADRANTS):
            for sort_key in sort_keys:
                rows = {partner}
                columns = {anchor}
                row_axis_edges = {partner: None}
                column_axis_edges = {anchor: None}
                chains = {quadrant: [] for quadrant in QUADRANTS}
                for quadrant in quadrant_order:
                    last_a = 0.0
                    last_b = 0.0
                    while True:
                        eligible = []
                        for candidate in by_quadrant[quadrant]:
                            if (candidate["distance_a"] < last_a + self.config.min_side
                                    or candidate["distance_b"] < last_b + self.config.min_side):
                                continue
                            row, column = candidate["outer_tracks"]
                            edge_a, edge_b = candidate["edges"][1], candidate["edges"][3]
                            if row in columns or column in rows:
                                continue
                            if row in row_axis_edges and row_axis_edges[row] not in (None, edge_a):
                                continue
                            if (column in column_axis_edges
                                    and column_axis_edges[column] not in (None, edge_b)):
                                continue
                            if row in rows and column in columns:
                                continue
                            if any(self.pair(row, old_column) not in self.graph.pair_edges
                                   for old_column in columns):
                                continue
                            if any(self.pair(old_row, column) not in self.graph.pair_edges
                                   for old_row in rows):
                                continue
                            eligible.append(candidate)
                        if not eligible:
                            break
                        candidate = min(eligible, key=sort_key)
                        row, column = candidate["outer_tracks"]
                        chains[quadrant].append(candidate)
                        rows.add(row)
                        columns.add(column)
                        row_axis_edges.setdefault(row, candidate["edges"][1])
                        column_axis_edges.setdefault(column, candidate["edges"][3])
                        last_a = candidate["distance_a"]
                        last_b = candidate["distance_b"]
                signature = tuple(
                    tuple(tuple(item["edges"]) for item in chains[quadrant])
                    for quadrant in QUADRANTS)
                counts = [len(chains[quadrant]) for quadrant in QUADRANTS]
                score = (sum(count > 0 for count in counts), min(counts),
                         (len(rows) - 1) * (len(columns) - 1), sum(counts))
                options[signature] = (score, chains)
        return [item[1] for item in sorted(options.values(), reverse=True,
                                           key=lambda item: item[0])]

    def assemble(self, anchor: int, partner: int, center_edge: int,
                 base_chains: dict, rows_override: Iterable[int] = (),
                 columns_override: Iterable[int] = (),
                 require_complete: bool = True) -> dict | None:
        row_positions = {partner: self.edge_position(center_edge, anchor)}
        column_positions = {anchor: self.edge_position(center_edge, partner)}
        for chain in base_chains.values():
            for candidate in chain:
                row, column = candidate["outer_tracks"]
                row_positions[row] = self.edge_position(candidate["edges"][1], anchor)
                column_positions[column] = self.edge_position(candidate["edges"][3], partner)
        for row in rows_override:
            if row in row_positions:
                continue
            candidates = self.graph.pair_edges.get(self.pair(anchor, row), ())
            if not candidates:
                return None
            edge_id = max(candidates, key=lambda item: min(
                self.edge_position(item, anchor),
                self.graph.lengths[anchor] - self.edge_position(item, anchor)))
            row_positions[row] = self.edge_position(edge_id, anchor)
        for column in columns_override:
            if column in column_positions:
                continue
            candidates = self.graph.pair_edges.get(self.pair(partner, column), ())
            if not candidates:
                return None
            edge_id = max(candidates, key=lambda item: min(
                self.edge_position(item, partner),
                self.graph.lengths[partner] - self.edge_position(item, partner)))
            column_positions[column] = self.edge_position(edge_id, partner)

        rows = sorted(row_positions, key=row_positions.__getitem__)
        columns = sorted(column_positions, key=column_positions.__getitem__)
        xyz = np.full((len(rows), len(columns), 3), -1.0, dtype=np.float32)
        edge_ids = np.full((len(rows), len(columns)), -1, dtype=np.int32)
        center_point = self.graph.edges[center_edge][6].astype(np.float64)

        row_axis_points = {}
        for row in rows:
            candidates = self.graph.pair_edges.get(self.pair(anchor, row), ())
            if candidates:
                edge_id = min(candidates, key=lambda item: abs(
                    self.edge_position(item, anchor) - row_positions[row]))
                row_axis_points[row] = self.graph.edges[edge_id][6].astype(np.float64)
        column_axis_points = {}
        for column in columns:
            candidates = self.graph.pair_edges.get(self.pair(partner, column), ())
            if candidates:
                edge_id = min(candidates, key=lambda item: abs(
                    self.edge_position(item, partner) - column_positions[column]))
                column_axis_points[column] = self.graph.edges[edge_id][6].astype(np.float64)

        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                candidates = self.graph.pair_edges.get(self.pair(row, column), ())
                if not candidates:
                    continue
                target = (row_axis_points.get(row, center_point)
                          + column_axis_points.get(column, center_point) - center_point)
                edge_id = min(candidates, key=lambda item: float(np.linalg.norm(
                    self.graph.edges[item][6] - target)))
                xyz[row_index, column_index] = (
                    self.graph.edges[edge_id][6].astype(np.float32)[[2, 1, 0]])
                edge_ids[row_index, column_index] = edge_id

        return self._finish_grid(xyz, edge_ids, rows, columns, require_complete)

    def _finish_grid(self, xyz: np.ndarray, edge_ids: np.ndarray,
                     rows: list[int], columns: list[int],
                     require_complete: bool) -> dict | None:
        valid = edge_ids >= 0
        if require_complete and not np.all(valid):
            return None

        row_arclength = np.full(edge_ids.shape, np.nan)
        column_arclength = np.full(edge_ids.shape, np.nan)
        for row_index, column_index in zip(*np.where(valid)):
            edge_id = int(edge_ids[row_index, column_index])
            row_arclength[row_index, column_index] = self.edge_position(
                edge_id, rows[row_index])
            column_arclength[row_index, column_index] = self.edge_position(
                edge_id, columns[column_index])

        def valid_sequence(values: np.ndarray) -> bool:
            values = values[np.isfinite(values)]
            if len(values) < 2:
                return True
            delta = np.diff(values)
            return (np.all(np.abs(delta) >= self.config.min_side)
                    and (np.all(delta > 0) or np.all(delta < 0)))

        if any(not valid_sequence(row_arclength[index]) for index in range(len(rows))):
            return None
        if any(not valid_sequence(column_arclength[:, index])
               for index in range(len(columns))):
            return None

        cell_counts = (valid[:-1, :-1].astype(np.uint8)
                       + valid[1:, :-1].astype(np.uint8)
                       + valid[:-1, 1:].astype(np.uint8)
                       + valid[1:, 1:].astype(np.uint8))
        if not require_complete:
            full_cells = cell_counts == 4
            components = self._cell_components(full_cells)
            if not components:
                return None
            component = max(components, key=len)
            kept_cells = np.zeros_like(full_cells)
            for row_index, column_index in component:
                kept_cells[row_index, column_index] = True
            if self._has_cell_hole(kept_cells):
                return None
            used = np.zeros_like(valid)
            row_indices, column_indices = np.where(kept_cells)
            used[row_indices, column_indices] = True
            used[row_indices + 1, column_indices] = True
            used[row_indices, column_indices + 1] = True
            used[row_indices + 1, column_indices + 1] = True
            xyz[~used] = -1
            edge_ids[~used] = -1
            valid = used
            cell_counts = (valid[:-1, :-1].astype(np.uint8)
                           + valid[1:, :-1].astype(np.uint8)
                           + valid[:-1, 1:].astype(np.uint8)
                           + valid[1:, 1:].astype(np.uint8))
        if not self._grid_geometry_valid(
                xyz, edge_ids, rows, columns, cell_counts == 4):
            return None
        return {
            "xyz": xyz,
            "edge_ids": edge_ids,
            "rows": list(rows),
            "columns": list(columns),
            "valid_vertices": int(valid.sum()),
            "valid_quads": int((cell_counts == 4).sum()),
            "three_corner_cells": int((cell_counts == 3).sum()),
        }

    def _track_edge_geometry_valid(self, track_id: int, first_edge: int,
                                   second_edge: int, first_xyz: np.ndarray,
                                   second_xyz: np.ndarray) -> bool:
        key = (track_id, min(first_edge, second_edge), max(first_edge, second_edge))
        cached = self._track_geometry_cache.get(key)
        if cached is not None:
            return cached
        valid = self._track_edge_geometry_valid_uncached(
            track_id, first_edge, second_edge, first_xyz, second_xyz)
        self._track_geometry_cache[key] = valid
        return valid

    def _track_edge_geometry_valid_uncached(
            self, track_id: int, first_edge: int, second_edge: int,
            first_xyz: np.ndarray, second_xyz: np.ndarray) -> bool:
        edge_a = self.graph.edges[first_edge]
        edge_b = self.graph.edges[second_edge]
        index_a = edge_a[4] if edge_a[0] == track_id else edge_a[5]
        index_b = edge_b[4] if edge_b[0] == track_id else edge_b[5]
        chord_vector = second_xyz.astype(np.float64) - first_xyz.astype(np.float64)
        chord = float(np.linalg.norm(chord_vector))
        if chord <= 1e-6:
            return False
        travel = abs(self.edge_position(second_edge, track_id)
                     - self.edge_position(first_edge, track_id))
        if travel / chord > self.config.max_edge_tortuosity:
            return False

        start, stop = sorted((index_a, index_b))
        points = self.graph.tracks[track_id][start:stop + 1].astype(np.float64)[:, [2, 1, 0]]
        if index_a > index_b:
            points = points[::-1]
        direction = chord_vector / chord
        relative = points - first_xyz.astype(np.float64)
        projection = relative @ direction
        perpendicular = np.linalg.norm(
            relative - projection[:, None] * direction, axis=1)
        if float(perpendicular.max(initial=0.0)) / chord > self.config.max_edge_deviation:
            return False
        # Permit voxel-scale jitter, but not genuine reversal along the chord.
        drawdown = np.maximum.accumulate(projection) - projection
        return float(drawdown.max(initial=0.0)) <= max(2.0, 0.05 * chord)

    def _cell_normal(self, xyz: np.ndarray, edge_ids: np.ndarray,
                     row: int, column: int) -> np.ndarray | None:
        key = (int(edge_ids[row, column]), int(edge_ids[row, column + 1]),
               int(edge_ids[row + 1, column]),
               int(edge_ids[row + 1, column + 1]))
        if key in self._cell_normal_cache:
            return self._cell_normal_cache[key]
        p00 = xyz[row, column].astype(np.float64)
        p01 = xyz[row, column + 1].astype(np.float64)
        p10 = xyz[row + 1, column].astype(np.float64)
        p11 = xyz[row + 1, column + 1].astype(np.float64)
        first_normal = np.cross(p00 - p10, p01 - p10)
        second_normal = np.cross(p01 - p10, p11 - p10)
        first_length = float(np.linalg.norm(first_normal))
        second_length = float(np.linalg.norm(second_normal))
        if first_length <= 1e-6 or second_length <= 1e-6:
            self._cell_normal_cache[key] = None
            return None
        first_normal /= first_length
        second_normal /= second_length
        if float(np.dot(first_normal, second_normal)) < self.config.min_normal_dot:
            self._cell_normal_cache[key] = None
            return None
        cell_normal = first_normal + second_normal
        cell_length = float(np.linalg.norm(cell_normal))
        if cell_length <= 1e-6:
            self._cell_normal_cache[key] = None
            return None
        result = cell_normal / cell_length
        self._cell_normal_cache[key] = result
        return result

    def _grid_geometry_valid(self, xyz: np.ndarray, edge_ids: np.ndarray,
                             rows: list[int], columns: list[int],
                             full_cells: np.ndarray) -> bool:
        checked_edges = set()
        cell_normals = {}
        for row, column in zip(*np.where(full_cells)):
            cell_normal = self._cell_normal(xyz, edge_ids, int(row), int(column))
            if cell_normal is None:
                return False
            cell_normals[(int(row), int(column))] = cell_normal

            grid_edges = (
                ("row", int(row), int(row), int(column),
                 int(row), int(column + 1)),
                ("row", int(row + 1), int(row + 1), int(column),
                 int(row + 1), int(column + 1)),
                ("column", int(column), int(row), int(column),
                 int(row + 1), int(column)),
                ("column", int(column + 1), int(row), int(column + 1),
                 int(row + 1), int(column + 1)),
            )
            for axis, track_index, row_a, column_a, row_b, column_b in grid_edges:
                key = (axis, track_index, row_a, column_a, row_b, column_b)
                if key in checked_edges:
                    continue
                checked_edges.add(key)
                track_id = rows[track_index] if axis == "row" else columns[track_index]
                if not self._track_edge_geometry_valid(
                        track_id,
                        int(edge_ids[row_a, column_a]),
                        int(edge_ids[row_b, column_b]),
                        xyz[row_a, column_a], xyz[row_b, column_b]):
                    return False

        for (row, column), normal in cell_normals.items():
            for neighbor in ((row + 1, column), (row, column + 1)):
                if (neighbor in cell_normals
                        and float(np.dot(normal, cell_normals[neighbor]))
                        < self.config.min_normal_dot):
                    return False
        return True

    @staticmethod
    def _cell_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        seen = np.zeros_like(mask)
        result = []
        for start_row, start_column in zip(*np.where(mask)):
            if seen[start_row, start_column]:
                continue
            stack = [(int(start_row), int(start_column))]
            seen[start_row, start_column] = True
            component = []
            while stack:
                row, column = stack.pop()
                component.append((row, column))
                for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row, next_column = row + delta_row, column + delta_column
                    if (0 <= next_row < mask.shape[0]
                            and 0 <= next_column < mask.shape[1]
                            and mask[next_row, next_column]
                            and not seen[next_row, next_column]):
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
            result.append(component)
        return result

    @staticmethod
    def _has_cell_hole(kept_cells: np.ndarray) -> bool:
        outside = np.pad(~kept_cells, 1, constant_values=True)
        reached = np.zeros_like(outside)
        stack = [(0, 0)]
        reached[0, 0] = True
        while stack:
            row, column = stack.pop()
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row, next_column = row + delta_row, column + delta_column
                if (0 <= next_row < outside.shape[0]
                        and 0 <= next_column < outside.shape[1]
                        and outside[next_row, next_column]
                        and not reached[next_row, next_column]):
                    reached[next_row, next_column] = True
                    stack.append((next_row, next_column))
        return bool(np.any(outside & ~reached))

    @staticmethod
    def _grid_points(grid: dict) -> set[tuple[int, int, int]]:
        valid = grid["xyz"][..., 0] >= 0
        return {tuple(point.astype(int)) for point in grid["xyz"][valid]}

    def _insertion_slots(self, indexed_positions: list[tuple[int, float]],
                         candidate_position: float, center_index: int,
                         axis_length: int) -> list[int]:
        """Return locally plausible insertion slots, nearest the center first."""
        indexed_positions.sort()
        if len(indexed_positions) < 2:
            slots = list(range(axis_length + 1))
        else:
            increasing = indexed_positions[-1][1] > indexed_positions[0][1]
            signed_candidate = candidate_position if increasing else -candidate_position
            signed_positions = [value if increasing else -value
                                for _, value in indexed_positions]
            insertion = int(np.searchsorted(signed_positions, signed_candidate))
            if insertion == 0:
                slots = list(range(indexed_positions[0][0] + 1))
            elif insertion == len(indexed_positions):
                slots = list(range(indexed_positions[-1][0] + 1,
                                   axis_length + 1))
            else:
                lower = indexed_positions[insertion - 1][0]
                upper = indexed_positions[insertion][0]
                slots = list(range(lower + 1, upper + 1))

        center_position = next(
            (value for index, value in indexed_positions if index == center_index), None)
        after_center = (center_position is not None
                        and candidate_position > center_position)
        if len(indexed_positions) >= 2 and not (
                indexed_positions[-1][1] > indexed_positions[0][1]):
            after_center = not after_center
        preferred = center_index + int(after_center)
        return sorted(slots, key=lambda slot: (abs(slot - preferred), slot))

    def _row_insertion_slots(self, grid: dict, center_column: int,
                             edge_id: int, center_row: int) -> list[int]:
        column_index = grid["columns"].index(center_column)
        positions = []
        for row_index in range(len(grid["rows"])):
            current_edge = int(grid["edge_ids"][row_index, column_index])
            if current_edge >= 0:
                positions.append((row_index,
                                  self.edge_position(current_edge, center_column)))
        return self._insertion_slots(
            positions, self.edge_position(edge_id, center_column),
            grid["rows"].index(center_row), len(grid["rows"]))

    def _column_insertion_slots(self, grid: dict, center_row: int,
                                edge_id: int, center_column: int) -> list[int]:
        row_index = grid["rows"].index(center_row)
        positions = []
        for column_index in range(len(grid["columns"])):
            current_edge = int(grid["edge_ids"][row_index, column_index])
            if current_edge >= 0:
                positions.append((column_index,
                                  self.edge_position(current_edge, center_row)))
        return self._insertion_slots(
            positions, self.edge_position(edge_id, center_row),
            grid["columns"].index(center_column), len(grid["columns"]))

    def _candidate_grid(self, surface: Surface, candidate: dict,
                        rows: list[int], columns: list[int]) -> dict | None:
        """Assemble one local expansion while keeping the accepted grid fixed."""
        old = surface.grid
        old_rows = set(old["rows"])
        old_columns = set(old["columns"])
        center_column, center_row = candidate["center_tracks"]
        outer_row, outer_column = candidate["outer_tracks"]
        center_edge, row_edge, corner_edge, column_edge = candidate["edges"]
        row_lookup = {track_id: index for index, track_id in enumerate(rows)}
        column_lookup = {track_id: index for index, track_id in enumerate(columns)}
        xyz = np.full((len(rows), len(columns), 3), -1.0, dtype=np.float32)
        edge_ids = np.full((len(rows), len(columns)), -1, dtype=np.int32)
        mapped_rows = np.asarray([row_lookup[row] for row in old["rows"]])
        mapped_columns = np.asarray([column_lookup[column]
                                     for column in old["columns"]])
        mapped = np.ix_(mapped_rows, mapped_columns)
        edge_ids[mapped] = old["edge_ids"]
        xyz[mapped] = old["xyz"]

        def place(row: int, column: int, edge_id: int) -> bool:
            edge = self.graph.edges[edge_id]
            if self.pair(row, column) != self.pair(edge[0], edge[1]):
                return False
            row_index = row_lookup[row]
            column_index = column_lookup[column]
            current = int(edge_ids[row_index, column_index])
            if current >= 0 and current != edge_id:
                return False
            edge_ids[row_index, column_index] = edge_id
            xyz[row_index, column_index] = (
                edge[6].astype(np.float32)[[2, 1, 0]])
            return True

        explicit = (
            (center_row, center_column, center_edge),
            (outer_row, center_column, row_edge),
            (outer_row, outer_column, corner_edge),
            (center_row, outer_column, column_edge),
        )
        if any(not place(*item) for item in explicit):
            return None

        center_xyz = self.graph.edges[center_edge][6].astype(np.float64)[[2, 1, 0]]
        row_delta = (self.graph.edges[row_edge][6].astype(np.float64)[[2, 1, 0]]
                     - center_xyz)
        column_delta = (
            self.graph.edges[column_edge][6].astype(np.float64)[[2, 1, 0]]
            - center_xyz)
        old_row_lookup = {track_id: index
                          for index, track_id in enumerate(old["rows"])}
        old_column_lookup = {track_id: index
                             for index, track_id in enumerate(old["columns"])}

        def old_point(row: int, column: int) -> np.ndarray | None:
            row_index = old_row_lookup.get(row)
            column_index = old_column_lookup.get(column)
            if row_index is None or column_index is None:
                return None
            if int(old["edge_ids"][row_index, column_index]) < 0:
                return None
            return old["xyz"][row_index, column_index].astype(np.float64)

        row_is_new = outer_row not in old_rows
        column_is_new = outer_column not in old_columns

        def place_nearest(row: int, column: int, target: np.ndarray) -> None:
            row_index = row_lookup[row]
            column_index = column_lookup[column]
            if edge_ids[row_index, column_index] >= 0:
                return
            choices = self.graph.pair_edges.get(self.pair(row, column), ())
            if not choices:
                return
            def squared_distance(item: int) -> float:
                delta = (self.graph.edges[item][6].astype(np.float64)[[2, 1, 0]]
                         - target)
                return float(delta @ delta)
            place(row, column, min(choices, key=squared_distance))

        if row_is_new:
            for column in old["columns"]:
                point = old_point(center_row, column)
                if point is not None:
                    place_nearest(outer_row, column, point + row_delta)
        if column_is_new:
            for row in old["rows"]:
                point = old_point(row, center_column)
                if point is not None:
                    place_nearest(row, outer_column, point + column_delta)

        return self._finish_grid(xyz, edge_ids, rows, columns,
                                 require_complete=False)

    def _legacy_candidate_trial(self, surface: Surface,
                                candidate: dict) -> dict | None:
        grid = surface.grid
        rows = set(grid["rows"])
        columns = set(grid["columns"])
        row, column = candidate["outer_tracks"]
        if row in columns or column in rows:
            return None
        if row in rows and column in columns:
            return None
        center_column, center_row = candidate["center_tracks"]
        if center_row not in rows or center_column not in columns:
            return None

        old_points = self._grid_points(grid)
        trial = self.assemble(
            surface.anchor, surface.partner, surface.center_edge,
            surface.base_chains, rows | {row}, columns | {column},
            require_complete=False)
        if trial is None or trial["valid_quads"] <= grid["valid_quads"]:
            return None
        if not old_points.issubset(self._grid_points(trial)):
            return None
        return trial

    def _local_candidate_trial(self, surface: Surface,
                               candidate: dict) -> dict | None:
        grid = surface.grid
        rows = set(grid["rows"])
        columns = set(grid["columns"])
        row, column = candidate["outer_tracks"]
        if row in columns or column in rows:
            return None
        if row in rows and column in columns:
            return None
        center_column, center_row = candidate["center_tracks"]
        if center_row not in rows or center_column not in columns:
            return None
        old_points = self._grid_points(grid)

        if row in rows:
            row_options = [list(grid["rows"])]
        else:
            row_options = []
            for slot in self._row_insertion_slots(
                    grid, center_column, candidate["edges"][1], center_row):
                ordered = list(grid["rows"])
                ordered.insert(slot, row)
                row_options.append(ordered)
        if column in columns:
            column_options = [list(grid["columns"])]
        else:
            column_options = []
            for slot in self._column_insertion_slots(
                    grid, center_row, candidate["edges"][3], center_column):
                ordered = list(grid["columns"])
                ordered.insert(slot, column)
                column_options.append(ordered)

        # The slot nearest the recentered crossing is the locally continuous
        # ordering. Trying every position in a sparse global grid makes frontier
        # exhaustion quadratic and mostly tests knowingly non-local layouts.
        arrangements = itertools.islice(
            itertools.product(row_options, column_options), 1)
        for ordered_rows, ordered_columns in arrangements:
            trial = self._candidate_grid(
                surface, candidate, ordered_rows, ordered_columns)
            if trial is None or trial["valid_quads"] <= grid["valid_quads"]:
                continue
            if old_points.issubset(self._grid_points(trial)):
                return trial
        return None

    def _candidate_trial(self, surface: Surface, candidate: dict) -> dict | None:
        """Try seed-axis assembly first, then locally ordered frontier growth."""
        trial = self._legacy_candidate_trial(surface, candidate)
        return trial if trial is not None else self._local_candidate_trial(
            surface, candidate)

    def initial_surface(self, anchor: int) -> tuple[Surface | None, str | None]:
        bases = self.base_crosses(anchor)
        if not bases:
            return None, "no transverse partner with sufficient clearance"
        for _, _, partner, center_edge in bases:
            candidates = self.growth_candidates(anchor, partner, center_edge)
            best = None
            for chains in self.complete_chain_options(candidates, anchor, partner):
                grid = self.assemble(anchor, partner, center_edge, chains)
                if grid is None or grid["valid_quads"] == 0:
                    continue
                counts = [len(chains[quadrant]) for quadrant in QUADRANTS]
                score = (sum(count > 0 for count in counts), min(counts),
                         grid["valid_quads"], sum(counts))
                if best is None or score > best[0]:
                    best = (score, chains, grid)
            if best is not None:
                _, chains, grid = best
                return Surface(
                    anchor=anchor,
                    partner=partner,
                    center_edge=center_edge,
                    base_chains=chains,
                    grid=grid,
                    initial_quadrant_counts={
                        str(q): len(chains[q]) for q in QUADRANTS},
                    recenter_history=[],
                ), None
        return None, "no complete ordered initial grid"

    def _grow_phase(self, surface: Surface, rng: random.Random,
                    trial_candidate: Callable[[Surface, dict], dict | None],
                    max_recenters: int, phase: str,
                    max_center_attempts: int | None = None,
                    candidate_limit: int | None = None) -> int:
        used_centers = set()
        attempted_centers = 0

        for _ in range(max_recenters):
            centers = []
            for row_index, row in enumerate(surface.grid["rows"]):
                for column_index, column in enumerate(surface.grid["columns"]):
                    edge_id = int(surface.grid["edge_ids"][row_index, column_index])
                    center = (column, row, edge_id)
                    if edge_id >= 0 and center not in used_centers:
                        centers.append(center)
            rng.shuffle(centers)
            selected_center = None
            for center in centers:
                if (max_center_attempts is not None
                        and attempted_centers >= max_center_attempts):
                    return attempted_centers
                attempted_centers += 1
                raw = self.growth_candidates(*center)
                if candidate_limit is not None:
                    rows = set(surface.grid["rows"])
                    columns = set(surface.grid["columns"])
                    raw = [candidate for candidate in raw
                           if candidate["outer_tracks"][0] not in columns
                           and candidate["outer_tracks"][1] not in rows
                           and not (candidate["outer_tracks"][0] in rows
                                   and candidate["outer_tracks"][1] in columns)]
                    raw = sorted(raw, key=lambda candidate: (
                        candidate["max_leg"],
                        candidate["distance_a"] + candidate["distance_b"]))[
                            :candidate_limit]
                if any(trial_candidate(surface, candidate) is not None
                       for candidate in raw):
                    selected_center = center
                    break
            if selected_center is None:
                break

            used_centers.add(selected_center)
            raw = self.growth_candidates(*selected_center)
            counts = {str(q): 0 for q in QUADRANTS}
            added_rows = 0
            added_columns = 0
            for quadrant in QUADRANTS:
                last_a = 0.0
                last_b = 0.0
                ordered = sorted(
                    (candidate for candidate in raw
                     if candidate["quadrant"] == quadrant),
                    key=lambda candidate: (
                        max(candidate["distance_a"], candidate["distance_b"]),
                        candidate["distance_a"] + candidate["distance_b"]))
                while True:
                    accepted = None
                    for candidate in ordered:
                        if (candidate["distance_a"]
                                < last_a + self.config.min_side
                                or candidate["distance_b"]
                                < last_b + self.config.min_side):
                            continue
                        trial = trial_candidate(surface, candidate)
                        if trial is not None:
                            accepted = (candidate, trial)
                            break
                    if accepted is None:
                        break
                    candidate, trial = accepted
                    old_rows = len(surface.grid["rows"])
                    old_columns = len(surface.grid["columns"])
                    surface.grid = trial
                    added_rows += len(trial["rows"]) - old_rows
                    added_columns += len(trial["columns"]) - old_columns
                    counts[str(quadrant)] += 1
                    last_a = candidate["distance_a"]
                    last_b = candidate["distance_b"]

            surface.recenter_history.append({
                "phase": phase,
                "center_column": self.graph.provenance[selected_center[0]],
                "center_row": self.graph.provenance[selected_center[1]],
                "center_xyz": self.graph.edges[selected_center[2]][6]
                    .astype(int)[[2, 1, 0]].tolist(),
                "quadrant_counts": counts,
                "added_rows": added_rows,
                "added_columns": added_columns,
                "grid_rows": len(surface.grid["rows"]),
                "grid_columns": len(surface.grid["columns"]),
                "grid_quads": surface.grid["valid_quads"],
            })
        return attempted_centers

    def grow(self, anchor: int, random_seed: int) -> tuple[Surface | None, str | None]:
        surface, failure = self.initial_surface(anchor)
        if surface is None:
            return None, failure
        rng = random.Random(random_seed)
        seed_attempts = self._grow_phase(
            surface, rng, self._legacy_candidate_trial,
            self.config.max_recenters, "seed_axes")
        local_attempts = self._grow_phase(
            surface, rng, self._local_candidate_trial,
            self.config.max_local_recenters, "local_frontier",
            max_center_attempts=self.config.max_local_attempts,
            candidate_limit=16)
        surface.growth_stats = {
            "seed_axis_center_attempts": seed_attempts,
            "local_frontier_center_attempts": local_attempts,
        }
        return surface, None


def length_bin_bounds(lengths: np.ndarray,
                      explicit_edges: tuple[float, float] | None) -> tuple[float, float]:
    if explicit_edges is not None:
        low, high = explicit_edges
    else:
        low, high = np.quantile(lengths, [1 / 3, 2 / 3]).tolist()
    if not 0 <= low <= high:
        raise ValueError("length bin edges must satisfy 0 <= SHORT_MAX <= MEDIUM_MAX")
    return float(low), float(high)


def tracks_in_bin(graph: TrackGraph, name: str, bounds: tuple[float, float],
                  eligible: Iterable[int]) -> list[int]:
    low, high = bounds
    if name == "short":
        mask = graph.lengths <= low
    elif name == "medium":
        mask = (graph.lengths > low) & (graph.lengths <= high)
    else:
        mask = graph.lengths > high
    return [int(track_id) for track_id in eligible if mask[track_id]]


def spaced_random_tracks(graph: TrackGraph, candidates: list[int], count: int,
                         spacing: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    candidates = candidates.copy()
    rng.shuffle(candidates)
    if spacing <= 0:
        return candidates[:count]
    cell_size = spacing
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = collections.defaultdict(list)
    selected = []
    for track_id in candidates:
        point = graph.representative_zyx[track_id]
        cell = tuple(np.floor(point / cell_size).astype(int))
        okay = True
        for offset in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            if any(np.linalg.norm(point - other) < spacing for other in buckets.get(neighbor, ())):
                okay = False
                break
        if not okay:
            continue
        selected.append(track_id)
        buckets[cell].append(point)
        if len(selected) >= count:
            break
    return selected


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in value)


def write_surface(output_root: Path, graph: TrackGraph, surface: Surface,
                  anchor: int, task_seed: int, bin_name: str,
                  bin_bounds: tuple[float, float], config: GrowthConfig,
                  elapsed: float) -> Path:
    prefix, key, source_index = graph.provenance[anchor]
    identity = (f"{key}:{source_index}:{task_seed}:{config.min_side}:"
                f"{config.max_side}:{config.max_recenters}:"
                f"{config.max_edge_tortuosity}:{config.max_edge_deviation}:"
                f"{config.min_normal_dot}:{config.max_local_recenters}:"
                f"{config.max_local_attempts}")
    digest = hashlib.sha1(identity.encode()).hexdigest()[:10]
    name = f"{bin_name}_{safe_name(prefix)}_{digest}.tifxyz"
    destination = output_root / name
    if destination.exists():
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=output_root))
    try:
        xyz = np.pad(surface.grid["xyz"], ((1, 1), (1, 1), (0, 0)),
                     constant_values=-1).astype(np.float32, copy=False)
        for channel, channel_name in enumerate(("x", "y", "z")):
            tifffile.imwrite(temporary / f"{channel_name}.tif", xyz[..., channel],
                             dtype=np.float32, compression=None)
        valid_points = xyz[xyz[..., 0] >= 0]
        metadata = {
            "format": "tifxyz",
            "type": "seg",
            "uuid": name,
            "scale": [1.0, 1.0],
            "bbox": [valid_points.min(axis=0).astype(float).tolist(),
                     valid_points.max(axis=0).astype(float).tolist()],
            "source": "exact-voxel recursive track-grid growth",
            "seed_track": graph.provenance[anchor],
            "seed_track_arclength": float(graph.lengths[anchor]),
            "random_seed": task_seed,
            "length_bin": bin_name,
            "length_bin_edges": list(bin_bounds),
            "minimum_side_arclength": config.min_side,
            "maximum_side_arclength": (
                None if math.isinf(config.max_side) else config.max_side),
            "maximum_edge_tortuosity": config.max_edge_tortuosity,
            "maximum_edge_deviation_fraction": config.max_edge_deviation,
            "minimum_normal_dot": config.min_normal_dot,
            "maximum_local_recenters": config.max_local_recenters,
            "maximum_local_center_attempts": config.max_local_attempts,
            "contact_tolerance_voxels": 0,
            "anchor": graph.provenance[surface.anchor],
            "crossing_partner": graph.provenance[surface.partner],
            "initial_quadrant_counts": surface.initial_quadrant_counts,
            "recenter_count": len(surface.recenter_history),
            "recenter_history": surface.recenter_history,
            "growth_stats": surface.growth_stats,
            "grid_shape": list(xyz.shape[:2]),
            "invalid_border_padding": 1,
            "invalid_value": -1,
            "valid_vertices": surface.grid["valid_vertices"],
            "valid_quads": surface.grid["valid_quads"],
            "three_corner_boundary_cells": surface.grid["three_corner_cells"],
            "row_tracks": [graph.provenance[item] for item in surface.grid["rows"]],
            "column_tracks": [graph.provenance[item] for item in surface.grid["columns"]],
            "growth_seconds": elapsed,
        }
        (temporary / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


_WORKER_GRAPH: TrackGraph | None = None
_WORKER_CONFIG: GrowthConfig | None = None
_WORKER_OUTPUT: Path | None = None
_WORKER_BIN_NAME: str | None = None
_WORKER_BIN_BOUNDS: tuple[float, float] | None = None
_WORKER_ENGINE: GrowthEngine | None = None
_WORKER_MIN_QUADS = 1


def generate_task(task: tuple[int, int]) -> dict:
    global _WORKER_ENGINE
    anchor, task_seed = task
    assert _WORKER_GRAPH is not None
    assert _WORKER_CONFIG is not None
    assert _WORKER_OUTPUT is not None
    assert _WORKER_BIN_NAME is not None
    assert _WORKER_BIN_BOUNDS is not None
    started = time.perf_counter()
    try:
        if _WORKER_ENGINE is None:
            _WORKER_ENGINE = GrowthEngine(_WORKER_GRAPH, _WORKER_CONFIG)
        surface, failure = _WORKER_ENGINE.grow(anchor, task_seed)
        if surface is None:
            return {"ok": False, "anchor": anchor, "reason": failure}
        if surface.grid["valid_quads"] < _WORKER_MIN_QUADS:
            return {"ok": False, "anchor": anchor,
                    "reason": f"fewer than {_WORKER_MIN_QUADS} valid quads"}
        destination = write_surface(
            _WORKER_OUTPUT, _WORKER_GRAPH, surface, anchor, task_seed,
            _WORKER_BIN_NAME, _WORKER_BIN_BOUNDS, _WORKER_CONFIG,
            time.perf_counter() - started)
        return {
            "ok": True,
            "anchor": anchor,
            "path": str(destination),
            "quads": surface.grid["valid_quads"],
            "recenters": len(surface.recenter_history),
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:  # Worker failures must not kill the whole batch.
        return {"ok": False, "anchor": anchor,
                "reason": f"{type(error).__name__}: {error}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks_dbm", help="logical DBM path (with or without .db backing suffix)")
    parser.add_argument("output", type=Path, help="folder receiving *.tifxyz directories")
    parser.add_argument("--length-bin", choices=("short", "medium", "long"),
                        default="long")
    parser.add_argument("--bin-edges", type=float, nargs=2, metavar=("SHORT_MAX", "MEDIUM_MAX"),
                        help="fixed arclength bin edges; default is empirical tertiles")
    parser.add_argument("--count", type=int, default=16,
                        help="number of valid TIFXYZ outputs requested")
    parser.add_argument("--max-attempts", type=int,
                        help="maximum spaced seed tracks to try; default is 10 * COUNT")
    parser.add_argument("--seed-spacing", type=float, default=256.0,
                        help="minimum distance between seed crossing-centroid voxels")
    parser.add_argument("--seed", type=int, default=1945)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--min-side", type=float, default=20.0)
    parser.add_argument("--max-side", type=float, default=float("inf"))
    parser.add_argument("--max-recenters", type=int, default=1000)
    parser.add_argument("--max-local-recenters", type=int, default=8,
                        help="additional locally ordered frontier expansions after seed-axis growth")
    parser.add_argument("--max-local-attempts", type=int, default=64,
                        help="maximum recentered crossings examined during local frontier growth")
    parser.add_argument("--max-edge-tortuosity", type=float, default=1.35,
                        help="maximum track arclength / TIFXYZ edge chord ratio")
    parser.add_argument("--max-edge-deviation", type=float, default=0.20,
                        help="maximum track deviation from its chord, as a chord fraction")
    parser.add_argument("--min-normal-dot", type=float, default=0.0,
                        help="minimum dot product for triangles and neighboring quad normals")
    parser.add_argument("--min-quads", type=int, default=1,
                        help="reject and retry surfaces smaller than this")
    parser.add_argument("--z-range", type=int, nargs=2, default=(4000, 17000),
                        metavar=("Z_MIN", "Z_MAX"))
    parser.add_argument("--center-xyz", type=float, nargs=3,
                        metavar=("X", "Y", "Z"),
                        help="XYZ center of the crossing-index bounding box")
    parser.add_argument("--bbox-size-zyx", type=float, nargs=3,
                        metavar=("Z_SIZE", "Y_SIZE", "X_SIZE"),
                        help="full ZYX edge lengths centered on --center-xyz")
    parser.add_argument("--roi", type=float, nargs=6,
                        metavar=("Z0", "Y0", "X0", "Z1", "Y1", "X1"),
                        help="deprecated explicit ZYX bounds; use --center-xyz and --bbox-size-zyx")
    parser.add_argument("--angle-degrees", type=float, default=30.0,
                        help="minimum acute tangent angle for a crossing")
    parser.add_argument("--tangent-radius", type=float, default=12.0)
    parser.add_argument("--index-cache", type=Path,
                        help="pickle cache path; default OUTPUT/.track-grid-index.pkl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.min_side <= 0 or args.max_side < args.min_side:
        raise SystemExit("require 0 < --min-side <= --max-side")
    if (args.max_recenters < 0 or args.max_local_recenters < 0
            or args.max_local_attempts < 0):
        raise SystemExit("recenter limits must be nonnegative")
    if args.min_quads < 1:
        raise SystemExit("--min-quads must be positive")
    if args.max_edge_tortuosity < 1:
        raise SystemExit("--max-edge-tortuosity must be at least 1")
    if args.max_edge_deviation < 0:
        raise SystemExit("--max-edge-deviation must be nonnegative")
    if not -1 <= args.min_normal_dot <= 1:
        raise SystemExit("--min-normal-dot must lie in [-1, 1]")
    path = normalize_dbm_path(args.tracks_dbm)
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.index_cache or args.output / ".track-grid-index.pkl"
    has_center = args.center_xyz is not None
    has_size = args.bbox_size_zyx is not None
    if has_center != has_size:
        raise SystemExit("--center-xyz and --bbox-size-zyx must be provided together")
    if args.roi is not None and has_center:
        raise SystemExit("use either centered bbox options or --roi, not both")
    if has_center:
        try:
            roi_lo, roi_hi = centered_roi(args.center_xyz, args.bbox_size_zyx)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    elif args.roi is None:
        roi_lo = roi_hi = None
    else:
        roi_lo = np.asarray(args.roi[:3], dtype=np.float64)
        roi_hi = np.asarray(args.roi[3:], dtype=np.float64)
        if np.any(roi_hi <= roi_lo):
            raise SystemExit("--roi upper bounds must exceed lower bounds")

    graph, cache_hit, index_seconds = load_or_build_graph(
        path, cache_path, tuple(args.z_range), roi_lo, roi_hi,
        args.angle_degrees, args.tangent_radius)
    print(f"crossing index {'loaded from cache' if cache_hit else 'built'} in "
          f"{index_seconds:.2f}s")

    seed_viable = [track_id for track_id in graph.incident
                   if graph.lengths[track_id] >= 2 * args.min_side]
    if not seed_viable:
        raise SystemExit("no crossing tracks are long enough for the minimum side length")
    bounds = length_bin_bounds(graph.lengths[seed_viable],
                               None if args.bin_edges is None else tuple(args.bin_edges))
    candidates = tracks_in_bin(graph, args.length_bin, bounds, seed_viable)
    if not candidates:
        raise SystemExit(f"no crossing tracks in the {args.length_bin} length bin")
    max_attempts = args.max_attempts or args.count * 10
    if max_attempts < args.count:
        raise SystemExit("--max-attempts must be at least --count")
    selected = spaced_random_tracks(
        graph, candidates, max_attempts, args.seed_spacing, args.seed)
    if len(selected) < max_attempts:
        print(f"warning: spacing left {len(selected)} of {max_attempts} possible attempts")
    print(f"length bins: short <= {bounds[0]:.1f}, medium <= {bounds[1]:.1f}, "
          f"long > {bounds[1]:.1f}; {len(candidates):,} eligible {args.length_bin} tracks")

    global _WORKER_GRAPH, _WORKER_CONFIG, _WORKER_OUTPUT
    global _WORKER_BIN_NAME, _WORKER_BIN_BOUNDS, _WORKER_MIN_QUADS
    _WORKER_GRAPH = graph
    _WORKER_CONFIG = GrowthConfig(
        args.min_side, args.max_side, args.max_recenters,
        args.max_edge_tortuosity, args.max_edge_deviation,
        args.min_normal_dot, args.max_local_recenters,
        args.max_local_attempts)
    _WORKER_OUTPUT = args.output
    _WORKER_BIN_NAME = args.length_bin
    _WORKER_BIN_BOUNDS = bounds
    _WORKER_MIN_QUADS = args.min_quads
    tasks = [(track_id, args.seed + 1_000_003 * index)
             for index, track_id in enumerate(selected)]

    results = []
    started = time.perf_counter()
    cursor = 0
    successes = 0
    progress = tqdm(total=args.count, desc="growing TIFXYZs", unit="grid")
    if args.workers == 1:
        while successes < args.count and cursor < len(tasks):
            batch_size = min(args.count - successes, len(tasks) - cursor)
            for result in map(generate_task, tasks[cursor:cursor + batch_size]):
                results.append(result)
                if result["ok"]:
                    successes += 1
                    progress.update()
            cursor += batch_size
            progress.set_postfix(attempted=len(results), failed=len(results) - successes)
    else:
        if "fork" not in mp.get_all_start_methods():
            raise SystemExit("multiple workers require the fork multiprocessing start method")
        context = mp.get_context("fork")
        with context.Pool(min(args.workers, len(tasks))) as pool:
            while successes < args.count and cursor < len(tasks):
                batch_size = min(args.count - successes, len(tasks) - cursor)
                batch = tasks[cursor:cursor + batch_size]
                for result in pool.imap_unordered(generate_task, batch, chunksize=1):
                    results.append(result)
                    if result["ok"]:
                        successes += 1
                        progress.update()
                cursor += batch_size
                progress.set_postfix(attempted=len(results),
                                     failed=len(results) - successes)
    progress.close()

    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    elapsed = time.perf_counter() - started
    summary = {
        "requested": args.count,
        "attempted": len(results),
        "written": len(successes),
        "failed": len(failures),
        "workers": args.workers,
        "index_cache_hit": cache_hit,
        "index_seconds": index_seconds,
        "generation_seconds": elapsed,
        "length_bin": args.length_bin,
        "length_bin_edges": list(bounds),
        "seed_spacing": args.seed_spacing,
        "minimum_quads": args.min_quads,
        "center_xyz": args.center_xyz,
        "bbox_size_zyx": args.bbox_size_zyx,
        "roi_lo_zyx": None if roi_lo is None else roi_lo.tolist(),
        "roi_hi_zyx": None if roi_hi is None else roi_hi.tolist(),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(successes)}/{args.count} requested TIFXYZs after "
          f"{len(results)} attempts in {elapsed:.2f}s "
          f"with {args.workers} worker(s)")
    if failures:
        reasons = collections.Counter(result["reason"] for result in failures)
        print("failures:")
        for reason, count in reasons.most_common():
            print(f"  {count:4d}  {reason}")
    return 0 if len(successes) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
