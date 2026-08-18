#!/usr/bin/env python3
"""Graph-topological track-sheet growth.

Tracks grow as an alternating crossing lattice. Proposals require a broad
frontier cross-section, accepted spans require independent four-cycle
witnesses, and crossing-graph bridges are trimmed before rasterization.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import json
import math
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

from tifxyz_quality import DEFAULT_MAX_THICK_CELL_FRAC


TOP_ARCLENGTH_FRACTION = 0.30
VOXEL_SIZE_UM = 9.6
SQUARE_UM_PER_SQUARE_CM = 1.0e8
DEFAULT_UV_RADIUS = 600.0
DEFAULT_MAX_SIZE_CM2 = (
    4.0 * DEFAULT_UV_RADIUS**2 * VOXEL_SIZE_UM**2 / SQUARE_UM_PER_SQUARE_CM
)

# Internal algorithm tuning.  These are intentionally constants rather than
# CLI options; change them here when experimenting with the growth pipeline.
GRAPH_SPAN_WITNESSES = 1
GRAPH_WITNESS_MIN_OFFSET = 4.0
GRAPH_WITNESS_MAX_OFFSET = 150.0
GRAPH_WITNESS_SHEAR_TOL = 8.0
GRAPH_TRIM = "bridges"
GRAPH_MAX_SPAN = 200.0
GRAPH_BLIND_GAP = 0.0
GRAPH_ANCHOR_CORROBORATE = False
SPAN_VERIFY_TOL = 0.0

LATTICE_EVIDENCE_SPACING = 6.0
LATTICE_RAIL_AGREE_FRAC = 0.5
LATTICE_RAIL_MIN_VOTES = 3
LATTICE_PROPOSAL_AGREE_FRAC = 0.0
LATTICE_GAP_PUBLISH_TOL = 5.0
LATTICE_FILL_GAP = 60.0

ANCHOR_REACH = 40.0
RASTER_SUPPORT_RADIUS = 35.0
MIN_COMPONENT_VX2 = 8000.0
SLIM_FINE_RASTER = False
SLIM_ITERATIONS = 8


@dataclasses.dataclass(frozen=True, slots=True)
class Crossing:
    """One undirected exact crossing in canonical track-row order."""

    first: int
    second: int
    first_local: int
    second_local: int
    # Crossing sidecars already store the directed self-track arclength.
    # An incident() result therefore knows the position on the row that was
    # queried; the reciprocal side remains optional.
    first_position: float | None = None
    second_position: float | None = None

    @classmethod
    def canonical(
        cls,
        track: int,
        partner: int,
        local: int,
        partner_local: int,
        position: float | None = None,
    ) -> Crossing:
        if track < partner:
            return cls(
                track,
                partner,
                local,
                partner_local,
                first_position=position,
            )
        return cls(
            partner,
            track,
            partner_local,
            local,
            second_position=position,
        )

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.first, self.second, self.first_local, self.second_local

    def other(self, track: int) -> int:
        if track == self.first:
            return self.second
        if track == self.second:
            return self.first
        raise ValueError(f"track {track} is not incident to crossing {self.key}")

    def local_index(self, track: int) -> int:
        if track == self.first:
            return self.first_local
        if track == self.second:
            return self.second_local
        raise ValueError(f"track {track} is not incident to crossing {self.key}")

    def stored_position(self, track: int) -> float | None:
        if track == self.first:
            return self.first_position
        if track == self.second:
            return self.second_position
        raise ValueError(f"track {track} is not incident to crossing {self.key}")


class StoredNpz:
    """Memory-map NPY payloads stored without compression inside an NPZ."""

    def __init__(self, path: str | Path):
        candidate = Path(path)
        if candidate.is_dir():
            files = list(candidate.glob("*.npz"))
            if len(files) != 1:
                raise ValueError(
                    f"{candidate} must contain exactly one .npz file; "
                    f"found {len(files)}"
                )
            candidate = files[0]
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        self.path = candidate

    def array(self, name: str) -> np.memmap:
        member = f"{name}.npy"
        with zipfile.ZipFile(self.path) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError as error:
                raise ValueError(f"{self.path} does not contain {member}") from error
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    f"{member} is compressed; rebuild the crossing NPZ with "
                    "np.savez so its large arrays can be memory-mapped"
                )

        with self.path.open("rb") as stream:
            stream.seek(info.header_offset)
            local_header = stream.read(30)
            if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
                raise ValueError(f"invalid ZIP local header for {member}")
            filename_size, extra_size = struct.unpack_from("<HH", local_header, 26)
            npy_offset = info.header_offset + 30 + filename_size + extra_size
            stream.seek(npy_offset)
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError(f"invalid NPY header for {member}")
            major = stream.read(1)[0]
            stream.read(1)
            header_size_bytes = 2 if major == 1 else 4
            header_size = int.from_bytes(stream.read(header_size_bytes), "little")
            header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
            data_offset = stream.tell()

        if header["fortran_order"]:
            raise ValueError(f"Fortran-ordered {member} is unsupported")
        return np.memmap(
            self.path,
            mode="r",
            dtype=np.dtype(header["descr"]),
            offset=data_offset,
            shape=header["shape"],
            order="C",
        )

class PackedTracks:
    """Read-only, memory-mapped access to a ``.vctracks`` directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            metadata = json.loads((self.path / "metadata.json").read_text())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"not a packed .vctracks directory: {self.path}"
            ) from error
        self.track_count = int(metadata["track_count"])
        point_count = int(metadata["point_count"])
        self.coordinates = np.memmap(
            self.path / "coordinates.i32",
            mode="r",
            dtype="<i4",
            shape=(point_count, 3),
        )
        self.offsets = np.memmap(
            self.path / "offsets.i64",
            mode="r",
            dtype="<i8",
            shape=(self.track_count + 1,),
        )
        self.source_ids = np.memmap(
            self.path / "source_ids.u64",
            mode="r",
            dtype="<u8",
            shape=(self.track_count,),
        )
        self.family_codes = np.memmap(
            self.path / "family_codes.i8",
            mode="r",
            dtype="i1",
            shape=(self.track_count,),
        )
        self.arclengths = np.memmap(
            self.path / "arclengths.f64",
            mode="r",
            dtype="<f8",
            shape=(self.track_count,),
        )

    def validate_track(self, track: int) -> int:
        track = int(track)
        if not 0 <= track < self.track_count:
            raise IndexError(f"track row {track} is outside [0, {self.track_count})")
        return track

    def points(self, track: int) -> np.ndarray:
        track = self.validate_track(track)
        begin = int(self.offsets[track])
        end = int(self.offsets[track + 1])
        return self.coordinates[begin:end]

    @functools.lru_cache(maxsize=8192)
    def cumulative_length(self, track: int) -> np.ndarray:
        points = np.asarray(self.points(track), dtype=np.float64)
        if len(points) == 0:
            return np.empty(0, dtype=np.float64)
        if len(points) == 1:
            return np.zeros(1, dtype=np.float64)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(steps)))

    def crossing_position(self, crossing: Crossing, track: int) -> float:
        stored = crossing.stored_position(track)
        if stored is not None:
            return stored
        local = crossing.local_index(track)
        cumulative = self.cumulative_length(track)
        if not 0 <= local < len(cumulative):
            raise ValueError(f"crossing local index {local} is outside track {track}")
        return float(cumulative[local])


class CrossingCsr:
    """Lazy undirected crossing graph backed by directed CSR arrays."""

    def __init__(
        self,
        path: str | Path,
        tracks: PackedTracks,
        *,
        validate_source_ids: bool = True,
    ):
        npz = StoredNpz(path)
        source_ids = npz.array("source_ids")
        self.offsets = npz.array("offsets")
        self.partners = npz.array("partners")
        self.self_local = npz.array("self_local")
        self.partner_local = npz.array("partner_local")
        self.positions = npz.array("positions")
        if source_ids.shape != (tracks.track_count,):
            raise ValueError("crossing and track stores have different counts")
        if self.offsets.shape != (tracks.track_count + 1,):
            raise ValueError("crossing offsets have the wrong shape")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.partners):
            raise ValueError("crossing offsets do not span the record arrays")
        if validate_source_ids and not np.array_equal(
            source_ids, tracks.source_ids
        ):
            raise ValueError("crossing and track source IDs do not match")
        expected = self.partners.shape
        if (
            self.self_local.shape != expected
            or self.partner_local.shape != expected
            or self.positions.shape != expected
        ):
            raise ValueError("crossing record arrays have inconsistent shapes")
    @functools.lru_cache(maxsize=32768)
    def incident(self, track: int) -> tuple[Crossing, ...]:
        track = int(track)
        begin = int(self.offsets[track])
        end = int(self.offsets[track + 1])
        result = []
        seen = set()
        for record in range(begin, end):
            crossing = Crossing.canonical(
                track,
                int(self.partners[record]),
                int(self.self_local[record]),
                int(self.partner_local[record]),
                float(self.positions[record]),
            )
            if crossing.key not in seen:
                seen.add(crossing.key)
                result.append(crossing)
        return tuple(result)


def surface_area_vx2(grid_xyz: np.ndarray) -> float:
    """Return exact two-triangle area of all valid TIFXYZ quads."""
    grid_xyz = np.asarray(grid_xyz, dtype=np.float64)
    valid = grid_xyz[..., 0] >= 0
    quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    rows, columns = np.where(quads)
    if not len(rows):
        return 0.0
    p00 = grid_xyz[rows, columns]
    p01 = grid_xyz[rows, columns + 1]
    p10 = grid_xyz[rows + 1, columns]
    p11 = grid_xyz[rows + 1, columns + 1]
    first = 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00), axis=1)
    second = 0.5 * np.linalg.norm(np.cross(p01 - p11, p10 - p11), axis=1)
    return float(np.sum(first + second, dtype=np.float64))


def area_vx2_to_cm2(area_vx2: float) -> float:
    """Convert an area in square voxels to cm²."""
    return float(area_vx2) * VOXEL_SIZE_UM**2 / SQUARE_UM_PER_SQUARE_CM


def size_cm2_to_uv_radius(size_cm2: float) -> float:
    """Return the half-width of a square UV window with this flat area."""
    area_vx2 = float(size_cm2) * SQUARE_UM_PER_SQUARE_CM / VOXEL_SIZE_UM**2
    return 0.5 * math.sqrt(area_vx2)


def random_top_arclength_rows(
    tracks: PackedTracks,
    *,
    random_seed: int | None,
    excluded: set[int],
    top_fraction: float = TOP_ARCLENGTH_FRACTION,
) -> tuple[np.ndarray, float, int, int]:
    """Return a shuffled, exact top-fraction set without replacement."""
    top_fraction = float(top_fraction)
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    arclengths = np.asarray(tracks.arclengths)
    finite = np.isfinite(arclengths)
    finite_count = int(np.count_nonzero(finite))
    if not finite_count:
        raise ValueError("the packed store has no finite track arclengths")
    top_count = max(1, int(math.ceil(top_fraction * finite_count)))
    finite_lengths = np.array(arclengths[finite], dtype=np.float64, copy=True)
    partition = finite_count - top_count
    finite_lengths.partition(partition)
    cutoff = float(finite_lengths[partition])
    rng = np.random.default_rng(random_seed)
    above = np.flatnonzero(finite & (arclengths > cutoff))
    ties = np.flatnonzero(finite & (arclengths == cutoff))
    ties_needed = top_count - len(above)
    if ties_needed < len(ties):
        rng.shuffle(ties)
        ties = ties[:ties_needed]
    selected = np.concatenate((above, ties)) if len(above) else ties
    if len(selected) != top_count:
        raise AssertionError("top-arclength selection produced a wrong count")
    rng.shuffle(selected)
    selected = np.asarray(
        [row for row in selected if int(row) not in excluded],
        dtype=np.int64,
    )
    return selected, cutoff, finite_count, top_count


def previously_used_seed_rows(
    output_parent: str | Path,
    tracks_path: str | Path,
) -> set[int]:
    """Find completed seed rows for this packed store under one parent."""
    output_parent = Path(output_parent)
    if not output_parent.is_dir():
        return set()
    target = Path(tracks_path).resolve()
    used = set()
    for metadata_path in output_parent.glob("*/meta.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            recorded_path = metadata.get("tracks_path")
            seed = metadata.get("seed_track_row")
            if (
                recorded_path is not None
                and Path(recorded_path).resolve() == target
                and seed is not None
            ):
                used.add(int(seed))
        except OSError, TypeError, ValueError, json.JSONDecodeError:
            continue
    return used


@dataclasses.dataclass
class TrackUV:
    """Piecewise-linear UV parameterization of one accepted track."""

    family: int  # 0 horizontal (u is the parallel coordinate), 1 vertical
    # anchors: sorted (arc_s, u, v)
    anchors: list
    sign: float = 1.0  # used when fewer than two anchors fix orientation
    gauge: bool = False  # seed/bootstrap tracks trusted along full length
    verified_lo: float | None = None  # arc positions verified against sheet
    verified_hi: float | None = None
    mode: str = ""  # seed | bootstrap | multi | single (diagnostics)
    # arc intervals cleared for publishing/output: anchor neighborhoods and
    # windows explicitly verified against the sheet.  None = everything
    # inside the trusted bounds (gauge tracks).
    ok_spans: list | None = None

    def add_ok_span(self, lo: float, hi: float) -> None:
        if hi <= lo:
            return
        spans = self.ok_spans or []
        spans.append([lo, hi])
        spans.sort()
        merged = [spans[0]]
        for span_lo, span_hi in spans[1:]:
            if span_lo <= merged[-1][1] + 1.0:
                merged[-1][1] = max(merged[-1][1], span_hi)
            else:
                merged.append([span_lo, span_hi])
        self.ok_spans = merged

    def in_ok_span(self, s: float, slack: float = 1.0) -> bool:
        if self.gauge:
            return True
        for lo, hi in self.ok_spans or ():
            if lo - slack <= s <= hi + slack:
                return True
        return False

    def ok_gaps(self):
        """Interior arc intervals between anchors not yet cleared."""
        if self.gauge or not self.anchors:
            return []
        lo = self.anchors[0][0]
        hi = self.anchors[-1][0]
        gaps = []
        cursor = lo
        for span_lo, span_hi in self.ok_spans or ():
            if span_lo > cursor:
                gaps.append((cursor, min(span_lo, hi)))
            cursor = max(cursor, span_hi)
            if cursor >= hi:
                break
        if cursor < hi:
            gaps.append((cursor, hi))
        return [(a, b) for a, b in gaps if b - a > 1.0]

    def _sign(self) -> float:
        if len(self.anchors) < 2:
            return self.sign
        p = 0 if self.family == 0 else 1
        first = self.anchors[0]
        last = self.anchors[-1]
        d = last[1 + p] - first[1 + p]
        ds = last[0] - first[0]
        return 1.0 if d * ds >= 0 else -1.0

    def uv_at(self, s: float, extrap_limit: float = float("inf")):
        a = self.anchors
        if not a:
            return None
        # A track whose anchors span only a few voxels has no reliable
        # orientation; do not let it vouch far beyond its anchored span.
        span = a[-1][0] - a[0][0]
        if len(a) == 1:
            span = float("inf") if self.gauge else 0.0
        credit = 60.0
        if not self.gauge and span < 15.0 and self.verified_lo is None:
            credit = 20.0
        lo = a[0][0] if self.verified_lo is None else min(a[0][0], self.verified_lo)
        hi = a[-1][0] if self.verified_hi is None else max(a[-1][0], self.verified_hi)
        p = 0 if self.family == 0 else 1
        if s <= a[0][0]:
            gap = a[0][0] - s
            if gap > extrap_limit or (not self.gauge and s < lo - credit):
                return None
            uv = [a[0][1], a[0][2]]
            uv[p] -= self._sign() * gap
            return uv[0], uv[1]
        if s >= a[-1][0]:
            gap = s - a[-1][0]
            if gap > extrap_limit or (not self.gauge and s > hi + credit):
                return None
            uv = [a[-1][1], a[-1][2]]
            uv[p] += self._sign() * gap
            return uv[0], uv[1]
        # binary search bracket
        lo, hi = 0, len(a) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a[mid][0] <= s:
                lo = mid
            else:
                hi = mid
        s0, u0, v0 = a[lo]
        s1, u1, v1 = a[hi]
        t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        return u0 + t * (u1 - u0), v0 + t * (v1 - v0)

    def add_anchor(self, s: float, u: float, v: float) -> None:
        self.anchors.append((s, u, v))
        self.anchors.sort(key=lambda x: x[0])


def _catmull_rom_segments(
    controls: np.ndarray,
    knots: np.ndarray,
    segment_ids: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Evaluate samples from many non-uniform Catmull-Rom segments at once."""
    k = knots[segment_ids]
    a = (ts - k[:, 0]) / (k[:, 1] - k[:, 0])
    b = (ts - k[:, 1]) / (k[:, 2] - k[:, 1])
    c = (ts - k[:, 2]) / (k[:, 3] - k[:, 2])
    d = (ts - k[:, 0]) / (k[:, 2] - k[:, 0])
    e = (ts - k[:, 1]) / (k[:, 3] - k[:, 1])
    one_b = 1.0 - b
    weights = np.empty((len(ts), 4), dtype=np.float64)
    weights[:, 0] = one_b * (1.0 - d) * (1.0 - a)
    weights[:, 1] = (
        one_b * ((1.0 - d) * a + d * one_b)
        + b * (1.0 - e) * one_b
    )
    weights[:, 2] = one_b * d * b + b * ((1.0 - e) * b + e * (1.0 - c))
    weights[:, 3] = b * e * c
    return np.einsum(
        "ni,nij->nj",
        weights,
        controls[segment_ids],
        optimize=False,
    )


class GraphLatticeGrower:
    """Crossing-lattice growth with graph-topological verification."""

    def __init__(
        self,
        tracks: PackedTracks,
        crossings: CrossingCsr,
        *,
        uv_radius: float = DEFAULT_UV_RADIUS,
        pair_tol_abs: float = 6.0,
        pair_tol_frac: float = 0.25,
        transverse_frac: float = 0.6,
        extrap_limit: float = 200.0,
        min_obs: int = 2,
        min_track_arclength: float = 200.0,
        anchor_reach: float = ANCHOR_REACH,
        min_connect: int = 3,
        evidence_spacing: float = LATTICE_EVIDENCE_SPACING,
        rail_agree_frac: float = LATTICE_RAIL_AGREE_FRAC,
        rail_min_votes: int = LATTICE_RAIL_MIN_VOTES,
        proposal_agree_frac: float = LATTICE_PROPOSAL_AGREE_FRAC,
        span_witnesses: int = GRAPH_SPAN_WITNESSES,
        witness_min_offset: float = GRAPH_WITNESS_MIN_OFFSET,
        witness_max_offset: float = GRAPH_WITNESS_MAX_OFFSET,
        witness_shear_tol: float = GRAPH_WITNESS_SHEAR_TOL,
        graph_trim: str = GRAPH_TRIM,
        max_span: float = GRAPH_MAX_SPAN,
        blind_gap: float = GRAPH_BLIND_GAP,
        anchor_corroborate: bool = GRAPH_ANCHOR_CORROBORATE,
        growth_min_span: float = 80.0,
    ) -> None:
        self.tracks = tracks
        self.crossings = crossings
        self.uv_radius = float(uv_radius)
        self.pair_tol_abs = float(pair_tol_abs)
        self.pair_tol_frac = float(pair_tol_frac)
        self.transverse_frac = float(transverse_frac)
        self.extrap_limit = float(extrap_limit)
        self.min_obs = int(min_obs)
        self.min_track_arclength = float(min_track_arclength)
        self.anchor_reach = float(anchor_reach)
        self.accepted: dict[int, TrackUV] = {}
        self.diag = collections.Counter()
        self.min_connect = int(min_connect)
        self.evidence_spacing = float(evidence_spacing)
        self.rail_agree_frac = float(rail_agree_frac)
        self.rail_min_votes = int(rail_min_votes)
        self.proposal_agree_frac = float(proposal_agree_frac)
        self.discarded: set[int] = set()
        self.evidence: dict[int, list] = collections.defaultdict(list)
        self.span_witnesses = int(span_witnesses)
        self.witness_min_offset = float(witness_min_offset)
        self.witness_max_offset = float(witness_max_offset)
        self.witness_shear_tol = float(witness_shear_tol)
        self.graph_trim = str(graph_trim)
        self.max_span = float(max_span)
        self.blind_gap = float(blind_gap)
        self.anchor_corroborate = bool(anchor_corroborate)
        self.growth_min_span = float(growth_min_span)

    def _family(self, track: int) -> int:
        return int(self.tracks.family_codes[track])

    def _inside_uv_radius(self, uv) -> bool:
        return not (abs(uv[0]) > self.uv_radius or abs(uv[1]) > self.uv_radius)

    def _observations(self, cand: int):
        """UV observations of a candidate from accepted partner tracks."""
        obs = []
        for crossing in self.crossings.incident(cand):
            partner = crossing.other(cand)
            tuv = self.accepted.get(partner)
            if tuv is None:
                continue
            if self._family(partner) == self._family(cand):
                continue
            s_partner = self.tracks.crossing_position(crossing, partner)
            uv = tuv.uv_at(s_partner, self.extrap_limit)
            if uv is None:
                self.diag["obs_extrapolation_limit"] += 1
                continue
            if not self._inside_uv_radius(uv):
                self.diag["obs_outside_radius"] += 1
                continue
            s_cand = self.tracks.crossing_position(crossing, cand)
            obs.append((s_cand, uv[0], uv[1], partner, s_partner))
        return obs

    def _consistent_subset(self, cand: int, obs: list):
        """Greedy largest pairwise-consistent subset of observations."""
        n = len(obs)
        if n < self.min_obs:
            return []
        fam = self._family(cand)
        p = 0 if fam == 0 else 1
        q = 1 - p
        ok = np.zeros((n, n), dtype=bool)
        values = np.asarray([o[:3] for o in obs], dtype=np.float64)
        # Block the broadcast so the temporary float64 pair matrices remain
        # bounded even for unusually high-degree tracks.  ``ok`` itself was
        # already quadratic in the original implementation.
        block_rows = max(1, min(n, 1024 * 1024 // n))
        for begin in range(0, n, block_rows):
            end = min(n, begin + block_rows)
            left = values[begin:end]
            ds = np.abs(left[:, None, 0] - values[None, :, 0])
            du = left[:, None, 1] - values[None, :, 1]
            dv = left[:, None, 2] - values[None, :, 2]
            duv = np.hypot(du, dv)
            tol = np.maximum(self.pair_tol_abs, self.pair_tol_frac * ds)
            dtrans = (
                dv
                if q == 1
                else left[:, None, 1] - values[None, :, 1]
            )
            ok[begin:end] = (np.abs(duv - ds) <= tol) & (
                (ds <= 2.0)
                | (np.abs(dtrans) <= self.transverse_frac * ds)
            )
        np.fill_diagonal(ok, False)

        # Run the original deterministic greedy-clique search with packed
        # bitsets.  The former nested ``all(ok[j, k] for k in group_set)``
        # made this section cubic in Python; candidates with a few hundred
        # observations could spend seconds here.  Python integers perform
        # the same subset test a machine word at a time in C.
        row_bits = [
            int.from_bytes(
                np.packbits(row, bitorder="little").tobytes(),
                byteorder="little",
            )
            for row in ok
        ]
        best_bits = 0
        best_size = 0
        for i in range(n):
            if (row_bits[i] | (1 << i)).bit_count() <= best_size:
                continue
            candidates = row_bits[i]
            selected = 1 << i
            while candidates:
                bit = candidates & -candidates
                j = bit.bit_length() - 1
                if selected & ~row_bits[j] == 0:
                    selected |= bit
                candidates ^= bit
            selected_size = selected.bit_count()
            if selected_size > best_size:
                best_bits = selected
                best_size = selected_size
                if best_size == n:
                    break
        best = [i for i in range(n) if best_bits & (1 << i)]

        # require distinct partners for the pair to count as corroboration
        partners = {obs[i][3] for i in best}
        if len(best) < self.min_obs or len(partners) < 2:
            return []
        # global orientation: parallel coordinate monotone in s
        chosen = sorted((obs[i] for i in best), key=lambda o: o[0])
        signs = []
        for a, b in zip(chosen, chosen[1:]):
            ds = b[0] - a[0]
            dp = b[1 + p] - a[1 + p]
            if ds > 2.0 and abs(dp) > 2.0:
                signs.append(1.0 if dp > 0 else -1.0)
        if signs and (np.abs(np.sum(signs)) != len(signs)):
            self.diag["orientation_flip_reject"] += 1
            return []
        return chosen

    def _tangent(self, track: int, local: int) -> np.ndarray | None:
        pts = np.asarray(self.tracks.points(track), dtype=np.float64)
        lo = max(0, local - 2)
        hi = min(len(pts) - 1, local + 2)
        if hi <= lo:
            return None
        d = pts[hi] - pts[lo]
        n = np.linalg.norm(d)
        return d / n if n > 0 else None

    def _bootstrap_tracks(self, seed: int) -> None:
        """Gauge-accept opposite-family tracks crossing the seed.

        A single crossing cannot orient a track's arclength direction in UV,
        so the first generation gets its sign from 3D tangent agreement with
        one reference tangent (a single global gauge choice, like the seed's
        own u direction).
        """
        seed_uv = self.accepted[seed]
        reference = None
        for crossing in self.crossings.incident(seed):
            partner = crossing.other(seed)
            if partner in self.accepted:
                continue
            if self._family(partner) == self._family(seed):
                continue
            s_seed = self.tracks.crossing_position(crossing, seed)
            uv = seed_uv.uv_at(s_seed)
            if uv is None or not self._inside_uv_radius(uv):
                continue
            tangent = self._tangent(partner, crossing.local_index(partner))
            if tangent is None:
                continue
            if reference is None:
                reference = tangent
                sign = 1.0
            else:
                dot = float(np.dot(tangent, reference))
                if abs(dot) < 0.3:
                    self.diag["bootstrap_ambiguous_tangent"] += 1
                    continue
                sign = 1.0 if dot > 0 else -1.0
            s_partner = self.tracks.crossing_position(crossing, partner)
            tuv = TrackUV(
                self._family(partner),
                [(s_partner, uv[0], uv[1])],
                sign=sign,
                verified_lo=s_partner - 60.0,
                verified_hi=s_partner + 60.0,
                mode="bootstrap",
            )
            self.accepted[partner] = tuv
            self.diag["bootstrap_accepted"] += 1

    def refine_uv(
        self,
        transverse_weight: float = 0.25,
        stay_weight: float = 0.05,
    ) -> dict:
        """Global least-squares pass over the anchor graph.

        Anchors copied between tracks (observation + knitting share exact
        floats) become one node; consecutive anchors along a track demand
        an arclength step in the parallel coordinate and no change in the
        transverse one.  This removes accumulated dead-reckoning drift,
        which otherwise folds the interpolated surface in-plane."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr

        node_of: dict[tuple[float, float], int] = {}
        current: list[tuple[float, float]] = []

        def node_id(u: float, v: float) -> int:
            key = (u, v)
            found = node_of.get(key)
            if found is None:
                found = len(current)
                node_of[key] = found
                current.append(key)
            return found

        track_nodes = {}
        equations = []  # (n2, n1, coord, target, weight)
        for track, tuv in self.accepted.items():
            p = 0 if tuv.family == 0 else 1
            q = 1 - p
            ids = [(a[0], node_id(a[1], a[2])) for a in tuv.anchors]
            track_nodes[track] = ids
            for (s1, n1), (s2, n2) in zip(ids, ids[1:]):
                ds = s2 - s1
                if n1 == n2 or ds <= 0:
                    continue
                step = current[n2][p] - current[n1][p]
                sign = 1.0 if step >= 0 else -1.0
                equations.append((n2, n1, p, sign * ds, 1.0))
                equations.append((n2, n1, q, 0.0, transverse_weight))

        n_nodes = len(current)
        refined = np.asarray(current, dtype=np.float64)
        for coord in (0, 1):
            rows, cols, vals, targets = [], [], [], []
            equation = 0
            for n2, n1, c, target, weight in equations:
                if c != coord:
                    continue
                rows.extend((equation, equation))
                cols.extend((n2, n1))
                vals.extend((weight, -weight))
                targets.append(weight * target)
                equation += 1
            for node in range(n_nodes):
                rows.append(equation)
                cols.append(node)
                vals.append(stay_weight)
                targets.append(stay_weight * current[node][coord])
                equation += 1
            system = coo_matrix((vals, (rows, cols)), shape=(equation, n_nodes)).tocsr()
            refined[:, coord] = lsqr(system, np.asarray(targets))[0]

        shift = np.linalg.norm(refined - np.asarray(current, dtype=np.float64), axis=1)
        for track, tuv in self.accepted.items():
            tuv.anchors = [
                (s, refined[node][0], refined[node][1])
                for s, node in track_nodes[track]
            ]
        return {
            "nodes": n_nodes,
            "median_shift": float(np.median(shift)) if n_nodes else 0.0,
            "p95_shift": float(np.percentile(shift, 95)) if n_nodes else 0.0,
        }

    def _thin_spaced(self, obs: list) -> list:
        """Greedy arclength thinning: evidence crossings must sit at least
        evidence_spacing apart along the candidate, so a bundle of
        near-coincident parallel rails counts as one witness."""
        thinned = []
        for o in sorted(obs, key=lambda o: o[0]):
            if thinned and o[0] - thinned[-1][0] < self.evidence_spacing:
                continue
            thinned.append(o)
        return thinned

    def _candidates(self, fam: int) -> list:
        partners: dict[int, set] = collections.defaultdict(set)
        for rail, tuv in self.accepted.items():
            if self._family(rail) == fam:
                continue
            for crossing in self.crossings.incident(rail):
                cand = crossing.other(rail)
                if cand in self.accepted or cand in self.discarded:
                    continue
                if self._family(cand) != fam:
                    continue
                partners[cand].add(rail)
        return [
            cand
            for cand, rails in partners.items()
            if len(rails) >= self.min_connect
            and float(self.tracks.arclengths[cand]) >= self.min_track_arclength
        ]

    def _iterate(self, fam: int) -> tuple[int, int]:
        """Grow one family; return counts of accepted and discarded tracks."""
        provisional = []
        agree: collections.Counter = collections.Counter()
        total: collections.Counter = collections.Counter()
        for cand in self._candidates(fam):
            obs = self._observations(cand)
            thinned = self._thin_spaced(obs)
            chosen = self._consistent_subset(cand, thinned)
            chosen_partners = {o[3] for o in chosen}
            if len(chosen_partners) < self.min_connect:
                self.diag["lattice_underconnected"] += 1
                continue
            lo = min(o[0] for o in chosen)
            hi = max(o[0] for o in chosen)
            if self.proposal_agree_frac > 0.0:
                # symmetric check: a proposal that geometrically disagrees
                # with too many of the spaced rails it straddles is itself
                # suspect (wrap drift keeps in-plane consistency, so the
                # witness SET is the discriminative signal)
                straddled = {
                    o[3] for o in thinned if lo <= o[0] <= hi
                } | chosen_partners
                if len(chosen_partners) < self.proposal_agree_frac * len(straddled):
                    self.diag["lattice_disagree_reject"] += 1
                    continue
            rejection = self._growth_rejection(chosen)
            if rejection is not None:
                self.diag[rejection] += 1
                continue
            for o in thinned:
                partner = o[3]
                if partner in chosen_partners:
                    agree[partner] += 1
                    total[partner] += 1
                elif lo <= o[0] <= hi:
                    # straddled by the consistent set yet excluded from it
                    total[partner] += 1
            provisional.append((cand, chosen))
        # rail vote: a rail mostly excluded by this round's consistent
        # proposals is on the wrong surface — discard it
        removed = 0
        for rail, votes in total.items():
            tuv = self.accepted.get(rail)
            if tuv is None or tuv.gauge:
                continue
            if (
                votes >= self.rail_min_votes
                and agree[rail] < self.rail_agree_frac * votes
            ):
                del self.accepted[rail]
                self.discarded.add(rail)
                self.diag["rail_discarded"] += 1
                removed += 1
        committed = 0
        for cand, chosen in provisional:
            chosen = [o for o in chosen if o[3] in self.accepted]
            if len({o[3] for o in chosen}) < self.min_connect:
                self.diag["lattice_lost_support"] += 1
                continue
            rejection = self._growth_rejection(chosen)
            if rejection is not None:
                self.diag[rejection] += 1
                continue
            anchors = [(s, u, v) for s, u, v, _, _ in chosen]
            tuv = TrackUV(fam, anchors, mode="multi")
            self.accepted[cand] = tuv
            self.evidence[cand] = list(chosen)
            committed += 1
            self.diag["accepted"] += 1
            for s, u, v, partner, s_partner in chosen:
                partner_uv = self.accepted.get(partner)
                if partner_uv is None or partner_uv.gauge:
                    continue
                if any(abs(s_partner - a[0]) < 2.0 for a in partner_uv.anchors):
                    continue
                partner_uv.add_anchor(s_partner, u, v)
                self.evidence[partner].append((s_partner, u, v, cand, s))
                self.diag["knitted_anchors"] += 1
        return committed, removed

    def _grow_lattice(self, seed: int) -> dict[int, TrackUV]:
        seed = self.tracks.validate_track(seed)
        length = self.tracks.cumulative_length(seed)
        if len(length) < 2:
            raise ValueError("seed track too short")
        total = float(length[-1])
        fam = self._family(seed)
        if fam == 0:
            anchors = [(0.0, -total / 2.0, 0.0), (total, total / 2.0, 0.0)]
        else:
            anchors = [(0.0, 0.0, -total / 2.0), (total, 0.0, total / 2.0)]
        seed_uv = TrackUV(fam, anchors, gauge=True, mode="seed")
        self.accepted[seed] = seed_uv
        self._bootstrap(seed)
        t0 = time.time()
        iteration = 0
        cycles = 0
        while True:
            cycles += 1
            changed = 0
            # Seed-family proposals are judged first against bootstrap rails.
            for grow_fam in (fam, 1 - fam):
                iteration += 1
                accepted, discarded = self._iterate(grow_fam)
                changed += accepted + discarded
            if changed == 0:
                break
        self.diag["iterations"] = iteration
        self.diag["growth_cycles"] = cycles
        self.diag["growth_stop_exhausted"] = 1
        self.diag["grow_seconds"] = round(time.time() - t0, 1)
        return self.accepted

    def surface_points_arcs(self, rast_extrap: float = 100.0):
        """surface_points plus each point's arclength along its track."""
        uvs, xyzs, track_ids, arcs = [], [], [], []
        for track, tuv in self.accepted.items():
            pts = np.asarray(self.tracks.points(track), dtype=np.float64)
            cum = self.tracks.cumulative_length(track)
            for point, s in zip(pts, cum):
                s = float(s)
                if not tuv.in_ok_span(s):
                    continue
                uv = tuv.uv_at(s, rast_extrap)
                if uv is None:
                    continue
                if not self._inside_uv_radius(uv):
                    continue
                uvs.append(uv)
                xyzs.append(point[::-1])
                track_ids.append(track)
                arcs.append(s)
        return (
            np.asarray(uvs, dtype=np.float64),
            np.asarray(xyzs, dtype=np.float64),
            np.asarray(track_ids, dtype=np.int64),
            np.asarray(arcs, dtype=np.float64),
        )

    def _window_on_surface(
        self,
        tuv: TrackUV,
        pts_xyz: np.ndarray,
        cum: np.ndarray,
        w_lo: float,
        w_hi: float,
        tree,
        uv_kept: np.ndarray,
        xyz_kept: np.ndarray,
        *,
        tol: float,
        reach_uv: float,
        min_neighbors: int,
        exclude: np.ndarray | None = None,
    ) -> str:
        """Judge one unpublished arc window against a local quadric fit of
        the trim-surviving surface: 'pass', 'fail', or 'frontier' (not
        enough kept points nearby to judge)."""
        samples = []
        for i in range(len(cum)):
            s = float(cum[i])
            if not (w_lo <= s <= w_hi):
                continue
            uv = tuv.uv_at(s, 1.0e9)
            if uv is None:
                continue
            if not self._inside_uv_radius(uv):
                continue
            samples.append((uv[0], uv[1], *pts_xyz[i]))
        if len(samples) < 3:
            return "frontier"
        arr = np.asarray(samples)
        center = arr[:, :2].mean(axis=0)
        idx = tree.query_ball_point(center, reach_uv)
        if exclude is not None:
            idx = [i for i in idx if not exclude[i]]
        if len(idx) < min_neighbors:
            return "frontier"
        du = uv_kept[idx, 0] - center[0]
        dv = uv_kept[idx, 1] - center[1]
        design = np.stack([np.ones_like(du), du, dv, du * du, du * dv, dv * dv], axis=1)
        coef, *_ = np.linalg.lstsq(design, xyz_kept[idx], rcond=None)
        su = arr[:, 0] - center[0]
        sv = arr[:, 1] - center[1]
        sample_design = np.stack(
            [np.ones_like(su), su, sv, su * su, su * sv, sv * sv], axis=1
        )
        residual = np.linalg.norm(arr[:, 2:] - sample_design @ coef, axis=1)
        if (
            float(np.median(residual)) > tol
            or float((residual > 1.5 * tol).mean()) > 0.34
        ):
            return "fail"
        return "pass"

    def verify_published_windows(
        self,
        uv_kept: np.ndarray,
        xyz_kept: np.ndarray,
        track_ids_kept: np.ndarray,
        *,
        window: float = 40.0,
        tol: float = 5.0,
        reach_uv: float = 50.0,
        min_neighbors: int = 30,
    ) -> int:
        """Re-judge every published window against a local quadric fit of
        the trim-surviving surface (excluding the track's own points) and
        unpublish the windows that fail.  The lattice/graph publish rules
        are topological; this adds the geometric check that only 3D
        verification has been able to provide against wrap drift.
        Returns the number of windows unpublished."""
        from scipy.spatial import cKDTree

        if not len(uv_kept):
            return 0
        tree = cKDTree(uv_kept)
        removed = 0
        for track, tuv in self.accepted.items():
            if tuv.gauge or not tuv.ok_spans:
                continue
            cum = np.asarray(self.tracks.cumulative_length(track))
            if not len(cum):
                continue
            pts_xyz = np.asarray(self.tracks.points(track), dtype=np.float64)[:, ::-1]
            exclude = track_ids_kept == track
            spans = [tuple(s) for s in tuv.ok_spans]
            tuv.ok_spans = None
            for lo, hi in spans:
                cursor = lo
                while cursor < hi:
                    w_lo = cursor
                    w_hi = min(hi, cursor + window)
                    cursor = w_hi
                    verdict = self._window_on_surface(
                        tuv,
                        pts_xyz,
                        cum,
                        w_lo,
                        w_hi,
                        tree,
                        uv_kept,
                        xyz_kept,
                        tol=tol,
                        reach_uv=reach_uv,
                        min_neighbors=min_neighbors,
                        exclude=exclude,
                    )
                    if verdict == "fail":
                        removed += 1
                        self.diag["span_verify_unpublished"] += 1
                    else:
                        tuv.add_ok_span(w_lo, w_hi)
        return removed

    def publish_consensus_gaps(
        self,
        uv_kept: np.ndarray,
        xyz_kept: np.ndarray,
        *,
        window: float = 40.0,
        tol: float = 5.0,
        reach_uv: float = 50.0,
        min_neighbors: int = 30,
        max_extend: float = 120.0,
    ) -> int:
        """Second-chance publishing against the trimmed surface.

        Graph finalization only publishes corroborated spans, which leaves
        holes where anchors are sparse even though accepted tracks run
        straight through them.  This pass verifies each unpublished window
        and outward extension against a local quadric fit of the
        trim-surviving surface and publishes the windows that lie on it.
        Returns the number published."""
        from scipy.spatial import cKDTree

        if not len(uv_kept):
            return 0
        tree = cKDTree(uv_kept)
        published = 0
        for track, tuv in self.accepted.items():
            if tuv.gauge or not tuv.anchors:
                continue
            cum = np.asarray(self.tracks.cumulative_length(track))
            if not len(cum):
                continue
            pts_xyz = np.asarray(self.tracks.points(track), dtype=np.float64)[:, ::-1]
            total = float(cum[-1])

            def judge(w_lo, w_hi):
                return self._window_on_surface(
                    tuv,
                    pts_xyz,
                    cum,
                    w_lo,
                    w_hi,
                    tree,
                    uv_kept,
                    xyz_kept,
                    tol=tol,
                    reach_uv=reach_uv,
                    min_neighbors=min_neighbors,
                )

            # interior gaps: a failing window skips ahead, far side may pass
            for gap_lo, gap_hi in list(tuv.ok_gaps()):
                cursor = gap_lo
                while cursor < gap_hi - 1.0:
                    w_lo = cursor
                    w_hi = min(gap_hi, cursor + window)
                    cursor = w_hi
                    verdict = judge(w_lo, w_hi)
                    if verdict == "pass":
                        tuv.add_ok_span(w_lo, w_hi)
                        published += 1
                        self.diag["gap_window_published"] += 1
                    elif verdict == "fail":
                        self.diag["gap_window_reject"] += 1
            # outward extensions: sequential, stop at the first non-pass
            for direction in (-1, 1):
                if not tuv.ok_spans:
                    break
                edge = tuv.ok_spans[0][0] if direction < 0 else tuv.ok_spans[-1][1]
                walked = 0.0
                while walked < max_extend:
                    if direction < 0:
                        w_hi = edge
                        w_lo = max(0.0, edge - window)
                        if w_hi <= 0.0:
                            break
                    else:
                        w_lo = edge
                        w_hi = min(total, edge + window)
                        if w_lo >= total:
                            break
                    if judge(w_lo, w_hi) != "pass":
                        break
                    tuv.add_ok_span(w_lo, w_hi)
                    published += 1
                    walked += window
                    edge = w_lo if direction < 0 else w_hi
                    self.diag["end_window_published"] += 1
        return published

    def _monotone_param(self, tuv: TrackUV):
        """Anchor arrays (parallel increasing) for inverting the UV map:
        (parallel, arc_s, transverse), or None if unusable."""
        p = 0 if tuv.family == 0 else 1
        a = np.asarray(tuv.anchors, dtype=np.float64)
        par = a[:, 1 + p]
        sign = 1.0 if par[-1] >= par[0] else -1.0
        keep = [0]
        for i in range(1, len(par)):
            if sign * (par[i] - par[keep[-1]]) > 0.1:
                keep.append(i)
        if len(keep) < 2:
            return None
        k = np.asarray(keep)
        s_arr = a[k, 0]
        pu = a[k, 1 + p]
        qv = a[k, 2 - p]
        if sign < 0:
            s_arr, pu, qv = s_arr[::-1], pu[::-1], qv[::-1]
        return pu, s_arr, qv

    def fill_points(
        self,
        kept_arcs: dict,
        *,
        spacing: float = 2.0,
        fill_gap: float = 60.0,
        step: float | None = None,
    ):
        """Catmull-Rom densification across rails.

        For every parallel-coordinate sample, each rail of the family
        contributes one control point (transverse coordinate + 3D position,
        both from its anchor parameterization); consecutive controls closer
        than fill_gap are bridged with a non-uniform Catmull-Rom spline
        sampled at the output spacing.  kept_arcs maps track -> sorted
        arclengths of its trim-surviving points; controls with no kept
        support nearby are skipped so trimmed rail regions spawn no fill."""
        if step is None:
            step = spacing
        out_uv: list = []
        segment_controls: list = []
        segment_knots: list = []
        sample_segments: list = []
        sample_parameters: list = []
        for fam in (0, 1):
            rails = []
            for track, tuv in self.accepted.items():
                if tuv.gauge or tuv.family != fam or len(tuv.anchors) < 2:
                    continue
                mono = self._monotone_param(tuv)
                if mono is not None:
                    rails.append((track, tuv, mono))
            if len(rails) < 2:
                continue
            t_lo = max(-self.uv_radius, min(m[0][0] for _, _, m in rails))
            t_hi = min(self.uv_radius, max(m[0][-1] for _, _, m in rails))
            if t_hi <= t_lo:
                continue
            t_grid = np.arange(t_lo, t_hi + 1e-9, step)
            per_t: list = [[] for _ in t_grid]
            for track, tuv, (pu, s_arr, qv) in rails:
                kept = kept_arcs.get(int(track))
                if kept is None or not len(kept):
                    continue
                m = (t_grid >= pu[0]) & (t_grid <= pu[-1])
                idxs = np.flatnonzero(m)
                if not len(idxs):
                    continue
                ts = t_grid[idxs]
                s_t = np.interp(ts, pu, s_arr)
                q_t = np.interp(ts, pu, qv)
                pts = np.asarray(self.tracks.points(track), dtype=np.float64)
                cum = np.asarray(self.tracks.cumulative_length(track))
                x_t = np.interp(s_t, cum, pts[:, 2])
                y_t = np.interp(s_t, cum, pts[:, 1])
                z_t = np.interp(s_t, cum, pts[:, 0])
                pos = np.searchsorted(kept, s_t)
                left = kept[np.clip(pos - 1, 0, len(kept) - 1)]
                right = kept[np.clip(pos, 0, len(kept) - 1)]
                near = np.minimum(np.abs(s_t - left), np.abs(right - s_t)) <= 8.0
                for j, i_t in enumerate(idxs):
                    if not near[j] or abs(q_t[j]) > self.uv_radius:
                        continue
                    if not tuv.in_ok_span(float(s_t[j])):
                        continue
                    per_t[i_t].append((q_t[j], x_t[j], y_t[j], z_t[j]))
            for i_t, controls in enumerate(per_t):
                if len(controls) < 2:
                    continue
                controls.sort()
                deduped = [controls[0]]
                for c in controls[1:]:
                    if c[0] - deduped[-1][0] > 0.5:
                        deduped.append(c)
                arr = np.asarray(deduped)
                q = arr[:, 0]
                xyz = arr[:, 1:4]
                t_par = float(t_grid[i_t])
                for i in range(len(arr) - 1):
                    gap = q[i + 1] - q[i]
                    if gap <= spacing or gap > fill_gap:
                        continue
                    p1, p2 = xyz[i], xyz[i + 1]
                    p0 = xyz[i - 1] if i > 0 else 2 * p1 - p2
                    t0 = q[i - 1] if i > 0 else 2 * q[i] - q[i + 1]
                    p3 = xyz[i + 2] if i + 2 < len(arr) else 2 * p2 - p1
                    t3 = q[i + 2] if i + 2 < len(arr) else 2 * q[i + 1] - q[i]
                    qs = np.arange(q[i] + spacing, q[i + 1] - 1e-6, spacing)
                    qs = qs[np.abs(qs) <= self.uv_radius]
                    if not len(qs):
                        continue
                    segment = len(segment_controls)
                    segment_controls.append(np.asarray([p0, p1, p2, p3]))
                    segment_knots.append(np.asarray([t0, q[i], q[i + 1], t3]))
                    sample_segments.append(np.full(len(qs), segment, dtype=np.int64))
                    sample_parameters.append(qs)
                    if fam == 0:
                        out_uv.extend((t_par, qq) for qq in qs)
                    else:
                        out_uv.extend((qq, t_par) for qq in qs)
        if not segment_controls:
            return np.zeros((0, 2)), np.zeros((0, 3))
        sample_segment_ids = np.concatenate(sample_segments)
        sample_ts = np.concatenate(sample_parameters)
        out_xyz = _catmull_rom_segments(
            np.asarray(segment_controls),
            np.asarray(segment_knots),
            sample_segment_ids,
            sample_ts,
        )
        return (
            np.asarray(out_uv, dtype=np.float64),
            out_xyz,
        )

    def _growth_rejection(self, chosen: list) -> str | None:
        """Reject a frontier supported only by a narrow bundle of rails.

        Alternating lattice growth can otherwise percolate indefinitely as
        a thin ladder: every new track has ``min_connect`` partners, but all
        of those partners occupy the same small cross-section.  Requiring
        their consistent crossings to span a real width prevents that
        ladder from entering the accepted graph and spawning another round.
        The proposal is not discarded; a later round may accept it after
        support arrives through a wider route.
        """
        if self.growth_min_span <= 0.0:
            return None
        span = float(chosen[-1][0] - chosen[0][0])
        if span < self.growth_min_span:
            return "graph_growth_narrow_reject"
        return None

    def _bootstrap(self, seed: int) -> None:
        self._bootstrap_tracks(seed)
        # record evidence for bootstrap anchors (partner = the seed)
        for track, tuv in self.accepted.items():
            if tuv.gauge or tuv.mode != "bootstrap":
                continue
            s_anchor, u, v = tuv.anchors[0]
            for crossing in self.crossings.incident(track):
                if crossing.other(track) != seed:
                    continue
                s_t = self.tracks.crossing_position(crossing, track)
                if abs(s_t - s_anchor) < 1e-6:
                    s_seed = self.tracks.crossing_position(crossing, seed)
                    self.evidence[track].append((s_t, u, v, seed, s_seed))
                    break

    def grow(self, seed: int):
        self._grow_lattice(seed)
        self._finalize_graph(self.tracks.validate_track(seed))
        return self.accepted

    def _finalize_graph(self, seed: int) -> None:
        t0 = time.time()
        for _ in range(5):
            if self.graph_trim != "none":
                self._trim_bridges(seed)
            if not self._rebuild_anchors():
                break
        self._graph_spans()
        self.diag["graph_finalize_seconds"] = round(time.time() - t0, 1)

    def _trim_bridges(self, seed: int) -> None:
        """Keep only tracks with >=2 edge-disjoint paths to the seed."""
        import rustworkx as rx

        graph = rx.PyGraph()
        idx = {t: graph.add_node(t) for t in self.accepted}
        edge_set = set()
        for t in self.accepted:
            for e in self.evidence.get(t, ()):
                p = e[3]
                if p in self.accepted and p != t:
                    edge_set.add((min(t, p), max(t, p)))
        graph.add_edges_from([(idx[a], idx[b], None) for a, b in edge_set])
        for a, b in rx.bridges(graph):
            graph.remove_edge(a, b)
        keep = {graph[n] for n in rx.node_connected_component(graph, idx[seed])}
        for t in list(self.accepted):
            if t not in keep and t != seed:
                del self.accepted[t]
                self.discarded.add(t)
                self.diag["graph_trimmed"] += 1

    def _rebuild_anchors(self) -> int:
        """Anchors become the surviving evidence crossings; tracks left
        without a usable parameterization drop.  Returns drop count."""
        dropped = 0
        changed = True
        while changed:
            changed = False
            for t in list(self.accepted):
                tuv = self.accepted[t]
                if tuv.gauge:
                    continue
                evs = [e for e in self.evidence.get(t, ()) if e[3] in self.accepted]
                evs.sort(key=lambda e: e[0])
                dedup: list = []
                for e in evs:
                    if dedup and e[0] - dedup[-1][0] < 2.0:
                        continue
                    dedup.append(e)
                min_needed = 1 if tuv.mode == "bootstrap" else 2
                if len(dedup) < min_needed:
                    del self.accepted[t]
                    self.discarded.add(t)
                    self.diag["graph_no_evidence_drop"] += 1
                    dropped += 1
                    changed = True
                    continue
                self.evidence[t] = dedup
                tuv.anchors = [(e[0], e[1], e[2]) for e in dedup]
        return dropped

    def _rail_crossers(self) -> dict:
        """rail -> [(crosser, s_along_crosser, transverse_uv, s_along_rail)]
        for every raw crossing between the rail and an accepted track of
        the opposite family (the crosser pool witnesses quads)."""
        crossers: dict[int, list] = {}
        for rail, rail_uv in self.accepted.items():
            fam_r = self._family(rail)
            q_w = 1 if fam_r == 1 else 0
            entries = []
            for crossing in self.crossings.incident(rail):
                other = crossing.other(rail)
                if other == rail or other not in self.accepted:
                    continue
                if self._family(other) == fam_r:
                    continue
                s_r = self.tracks.crossing_position(crossing, rail)
                uv = rail_uv.uv_at(s_r, self.extrap_limit)
                if uv is None:
                    continue
                s_o = self.tracks.crossing_position(crossing, other)
                entries.append((other, s_o, uv[q_w], s_r))
            crossers[rail] = entries
        return crossers

    def _span_witnesses(self, track: int, e_i, e_j, crossers) -> list:
        """Tracks closing a quad T-R_i-W-R_j over span (e_i, e_j).

        Returns [(w, w_lo, w_hi)]: each witness with the arc interval of
        its own return path between the two rails."""
        r_i, r_j = e_i[3], e_j[3]
        if r_i == r_j:
            return []
        fam = self._family(track)
        q = 1 if fam == 0 else 0
        t_i = e_i[1 + q]
        t_j = e_j[1 + q]
        span = e_j[0] - e_i[0]
        tol_arc = max(self.pair_tol_abs, self.pair_tol_frac * span)
        by_w: dict[int, list] = {}
        for w, s_o, qval, _ in crossers.get(r_i, ()):
            if w == track:
                continue
            off = qval - t_i
            if self.witness_min_offset <= abs(off) <= self.witness_max_offset:
                by_w.setdefault(w, []).append((off, s_o))
        hits: list = []
        seen: set[int] = set()
        for w, s_o, qval, _ in crossers.get(r_j, ()):
            if w == track or w in seen:
                continue
            entries = by_w.get(w)
            if not entries:
                continue
            off_j = qval - t_j
            if not (self.witness_min_offset <= abs(off_j) <= self.witness_max_offset):
                continue
            for off_i, s_o1 in entries:
                if off_i * off_j <= 0:
                    continue
                if abs(off_i - off_j) > self.witness_shear_tol:
                    continue
                if abs(abs(s_o - s_o1) - span) > tol_arc:
                    continue
                seen.add(w)
                hits.append((w, min(s_o1, s_o), max(s_o1, s_o)))
                break
        return hits

    @staticmethod
    def _interval_covered(intervals, lo: float, hi: float, slack: float):
        lo, hi = lo + slack, hi - slack
        if hi <= lo:
            return True
        return any(a <= lo and hi <= b for a, b in intervals)

    def _graph_spans(self) -> None:
        """Publish anchor neighborhoods plus cycle-corroborated spans.

        Validity is a fixed point: a span's witnesses only count while
        their own return intervals are covered by THEIR valid spans and
        corroborated anchor neighborhoods.  Trust therefore percolates
        outward from the gauge seed through closed cycles, and bundles of
        tracks that drift together lose their mutual witnesses."""
        crossers = self._rail_crossers()
        tracks_spans: dict[int, list] = {}
        for t, tuv in self.accepted.items():
            if tuv.gauge:
                continue
            evs = self.evidence[t]
            spans = []
            for e_i, e_j in zip(evs, evs[1:]):
                span = e_j[0] - e_i[0]
                blind = span <= self.blind_gap
                too_long = span > self.max_span
                wit = (
                    self._span_witnesses(t, e_i, e_j, crossers)
                    if not (blind or too_long)
                    else []
                )
                if too_long:
                    self.diag["graph_span_too_long"] += 1
                spans.append(
                    [
                        e_i[0],
                        e_j[0],
                        wit,
                        blind,
                        too_long,
                        not too_long and (blind or len(wit) >= self.span_witnesses),
                    ]
                )
            tracks_spans[t] = spans

        def rebuild_covered() -> dict:
            covered = {}
            for t, tuv in self.accepted.items():
                if tuv.gauge:
                    covered[t] = [(-1e18, 1e18)]
                    continue
                probe = TrackUV(tuv.family, [])
                spans = tracks_spans[t]
                for lo, hi, _, _, _, ok in spans:
                    if ok:
                        probe.add_ok_span(lo, hi)
                evs = self.evidence[t]
                for i, e in enumerate(evs):
                    if self.anchor_corroborate:
                        partner_uv = self.accepted.get(e[3])
                        on_seed = partner_uv is not None and partner_uv.gauge
                        touching = (i > 0 and spans[i - 1][5]) or (
                            i < len(spans) and spans[i][5]
                        )
                        if not (on_seed or touching):
                            continue
                    probe.add_ok_span(
                        e[0] - self.anchor_reach, e[0] + self.anchor_reach
                    )
                covered[t] = [tuple(s) for s in probe.ok_spans or ()]
            return covered

        covered = rebuild_covered()
        for iteration in range(25):
            changed = False
            for t, spans in tracks_spans.items():
                for s in spans:
                    lo, hi, wit, blind, too_long, ok = s
                    if not ok or blind:
                        continue
                    live = sum(
                        1
                        for w, w_lo, w_hi in wit
                        if self._interval_covered(
                            covered.get(w, ()), w_lo, w_hi, self.anchor_reach
                        )
                    )
                    if live < self.span_witnesses:
                        s[5] = False
                        changed = True
            if not changed:
                break
            covered = rebuild_covered()
        self.diag["graph_fixpoint_iterations"] = iteration + 1

        for t, tuv in self.accepted.items():
            if tuv.gauge:
                continue
            spans = tracks_spans[t]
            evs = self.evidence[t]
            tuv.ok_spans = None
            for lo, hi, _, _, _, ok in spans:
                if ok:
                    tuv.add_ok_span(lo, hi)
                    self.diag["graph_span_published"] += 1
                else:
                    self.diag["graph_span_rejected"] += 1
            for i, e in enumerate(evs):
                if self.anchor_corroborate:
                    partner_uv = self.accepted.get(e[3])
                    on_seed = partner_uv is not None and partner_uv.gauge
                    touching = (i > 0 and spans[i - 1][5]) or (
                        i < len(spans) and spans[i][5]
                    )
                    if not (on_seed or touching):
                        self.diag["graph_anchor_uncorroborated"] += 1
                        continue
                tuv.add_ok_span(e[0] - self.anchor_reach, e[0] + self.anchor_reach)


def sheet_thickness_stats(
    uv: np.ndarray,
    xyz: np.ndarray,
    *,
    cell: float = 24.0,
    thick_tol: float = 7.0,
) -> dict:
    """GT-free wrap-purity signal: fraction of UV cells whose points spread
    more than thick_tol along the local normal (p85-p15).  A clean single
    sheet is a few voxels thick; a cell fed by two wraps is ~spacing thick.
    High values mean the patch mixes surfaces (e.g. a seed lying between
    wraps) and should not be trusted, however smooth the raster looks."""
    if len(uv) == 0:
        return {"thick_cell_frac": 0.0, "thickness_cells": 0}
    cu = np.floor(uv[:, 0] / cell).astype(np.int64)
    cv = np.floor(uv[:, 1] / cell).astype(np.int64)
    key = (cu - cu.min()) * 100000 + (cv - cv.min())
    order = np.argsort(key, kind="stable")
    boundaries = np.flatnonzero(np.diff(key[order])) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(key)]))
    cells = 0
    thick = 0
    for s0, e0 in zip(starts, ends):
        sel = order[s0:e0]
        if len(sel) < 8:
            continue
        cells += 1
        pts = xyz[sel]
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal_comp = centered @ vt[2]
        if np.percentile(normal_comp, 85) - np.percentile(normal_comp, 15) > thick_tol:
            thick += 1
    return {
        "thick_cell_frac": float(thick / max(1, cells)),
        "thickness_cells": int(cells),
    }


def _median_reference(
    uv: np.ndarray,
    xyz: np.ndarray,
    *,
    cell: float = 8.0,
    smooth: int = 3,
):
    """Per-cell componentwise median surface on a coarse UV grid."""
    from scipy import ndimage

    uv_min = uv.min(axis=0)
    iu = np.floor((uv[:, 0] - uv_min[0]) / cell).astype(np.int64)
    iv = np.floor((uv[:, 1] - uv_min[1]) / cell).astype(np.int64)
    nu = int(iu.max()) + 1
    nv = int(iv.max()) + 1
    flat = iv * nu + iu
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    boundaries = np.flatnonzero(np.diff(sorted_flat)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(flat)]))
    cells = sorted_flat[starts]
    ref = np.full((nv * nu, 3), np.nan)
    for axis in range(3):
        vals = xyz[:, axis]
        ax_order = np.lexsort((vals, flat))
        sorted_vals = vals[ax_order]
        mid = starts + (ends - starts) // 2
        ref[cells, axis] = sorted_vals[mid]
    ref = ref.reshape(nv, nu, 3)
    valid = ~np.isnan(ref[..., 0])
    weight = valid.astype(np.float64)
    smoothed = np.empty_like(ref)
    denominator = ndimage.uniform_filter(weight, smooth)
    for axis in range(3):
        filled = np.where(valid, ref[..., axis], 0.0)
        smoothed[..., axis] = np.where(
            denominator > 0,
            ndimage.uniform_filter(filled, smooth) / np.maximum(denominator, 1e-12),
            np.nan,
        )
    return smoothed, uv_min, cell


def _contiguous_track_groups(track_ids: np.ndarray) -> list[slice | np.ndarray]:
    """Return each track's point indices without repeated whole-array masks.

    ``surface_points_arcs`` emits one contiguous block per accepted track,
    which makes the common path just a scan for block boundaries.  Keep a
    stable-sort fallback so the trimming functions remain correct for
    callers that supply interleaved track IDs.
    """
    if len(track_ids) == 0:
        return []
    boundaries = np.flatnonzero(track_ids[1:] != track_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(track_ids)]))
    block_ids = track_ids[starts]
    if len(np.unique(block_ids)) == len(block_ids):
        return [slice(int(start), int(end)) for start, end in zip(starts, ends)]

    order = np.argsort(track_ids, kind="stable")
    sorted_ids = track_ids[order]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    return list(np.split(order, boundaries))


def trim_offsheet_points(
    uv: np.ndarray,
    xyz: np.ndarray,
    track_ids: np.ndarray,
    *,
    residual_tol: float = 6.0,
    iterations: int = 2,
    window: int = 9,
) -> np.ndarray:
    """Keep points whose along-track median residual to the consensus
    surface stays below tolerance.  Returns a boolean mask."""
    from scipy import ndimage

    track_groups = _contiguous_track_groups(track_ids)
    keep = np.ones(len(uv), dtype=bool)
    for _ in range(iterations):
        ref, uv_min, cell = _median_reference(uv[keep], xyz[keep])
        nv, nu = ref.shape[:2]
        iu = np.clip(((uv[:, 0] - uv_min[0]) / cell).astype(np.int64), 0, nu - 1)
        iv = np.clip(((uv[:, 1] - uv_min[1]) / cell).astype(np.int64), 0, nv - 1)
        predicted = ref[iv, iu]
        residual = np.linalg.norm(xyz - predicted, axis=1)
        residual[np.isnan(residual)] = 0.0  # no consensus -> no evidence
        # median-filter the residual along each track so isolated noise
        # survives but sustained off-sheet segments are removed
        smoothed = np.empty_like(residual)
        for group in track_groups:
            r = residual[group]
            if len(r) >= 3:
                r = ndimage.median_filter(r, size=min(window, len(r)))
            smoothed[group] = r
        keep = smoothed <= residual_tol
        if not keep.any():
            raise RuntimeError("all points trimmed; residual_tol too strict")
    return keep


def irls_trim(
    uv: np.ndarray,
    xyz: np.ndarray,
    *,
    cell: float = 4.0,
    sigma_cells: float = 2.0,
    iterations: int = 3,
    tau: float = 5.0,
    residual_tol: float = 7.0,
) -> np.ndarray:
    """Robust local-plane consensus trim.

    Fits xyz ~ plane(u, v) per grid cell with Gaussian-weighted least
    squares, iteratively downweighting 3D outliers.  Where two track
    populations overlap in UV at different 3D positions, the fit converges
    to the dominant one and the minority is dropped — a median reference
    splits the difference and keeps both, which folds the interpolation.
    Returns a keep mask."""
    from scipy import ndimage

    uv_min = uv.min(axis=0)
    iu = np.clip(((uv[:, 0] - uv_min[0]) / cell).astype(np.int64), 0, None)
    iv = np.clip(((uv[:, 1] - uv_min[1]) / cell).astype(np.int64), 0, None)
    nu = int(iu.max()) + 1
    nv = int(iv.max()) + 1
    flat = iv * nu + iu
    # coordinates relative to each point's own cell center keep the moment
    # magnitudes small
    cu = uv_min[0] + (iu + 0.5) * cell
    cv = uv_min[1] + (iv + 0.5) * cell
    du = uv[:, 0] - cu
    dv = uv[:, 1] - cv

    weights = np.ones(len(uv))
    residual = np.zeros(len(uv))
    for _ in range(iterations):
        channels = [
            np.ones_like(du),
            du,
            dv,
            du * du,
            du * dv,
            dv * dv,
        ]
        for c in range(3):
            channels.extend((xyz[:, c], xyz[:, c] * du, xyz[:, c] * dv))
        binned = np.zeros((len(channels), nv * nu))
        for k, q in enumerate(channels):
            np.add.at(binned[k], flat, weights * q)
        smoothed = np.stack(
            [
                ndimage.gaussian_filter(b.reshape(nv, nu), sigma_cells).ravel()
                for b in binned
            ]
        )
        m00, m10, m01, m20, m11, m02 = smoothed[:6]
        design = np.stack(
            [
                np.stack([m00, m10, m01], axis=-1),
                np.stack([m10, m20, m11], axis=-1),
                np.stack([m01, m11, m02], axis=-1),
            ],
            axis=-2,
        )
        det = np.linalg.det(design)
        solvable = (m00 > 1e-8) & (np.abs(det) > 1e-6 * np.maximum(m00, 1) ** 3)
        fallback = ~solvable & (m00 > 1e-8)
        # evaluate the local plane at each point's offset from cell center
        plane = np.zeros((len(uv), 3))
        for c in range(3):
            b0, b1, b2 = smoothed[6 + 3 * c : 9 + 3 * c]
            rhs = np.stack([b0, b1, b2], axis=-1)
            coefficients = np.zeros((nv * nu, 3))
            if solvable.any():
                coefficients[solvable] = np.linalg.solve(
                    design[solvable], rhs[solvable][..., None]
                )[..., 0]
            coefficients[fallback, 0] = b0[fallback] / m00[fallback]
            k = coefficients[flat]
            plane[:, c] = k[:, 0] + k[:, 1] * du + k[:, 2] * dv
        residual = np.linalg.norm(xyz - plane, axis=1)
        weights = 1.0 / (1.0 + (residual / tau) ** 2)
    return residual <= residual_tol


def z_consistency_trim(
    uv: np.ndarray,
    xyz: np.ndarray,
    track_ids: np.ndarray,
    *,
    cell: float = 8.0,
    sigma_cells: float = 4.0,
    tol: float = 12.0,
    window: int = 9,
) -> np.ndarray:
    """Drop points whose v placement disagrees with the local z field.

    Vertical tracks run roughly along z, so w = z - v must vary smoothly
    across the sheet; a band that slipped a wrap (or drifted in v) shows a
    step in w.  Two robust passes: the second recomputes the reference
    without the first pass's outliers."""
    from scipy import ndimage

    w = xyz[:, 2] - uv[:, 1]
    uv_min = uv.min(axis=0)
    iu = np.clip(((uv[:, 0] - uv_min[0]) / cell).astype(np.int64), 0, None)
    iv = np.clip(((uv[:, 1] - uv_min[1]) / cell).astype(np.int64), 0, None)
    nu = int(iu.max()) + 1
    nv = int(iv.max()) + 1
    flat = iv * nu + iu
    track_groups = _contiguous_track_groups(track_ids)
    keep = np.ones(len(uv), dtype=bool)
    residual = np.zeros(len(uv))
    for _ in range(2):
        weight_img = np.zeros(nv * nu)
        value_img = np.zeros(nv * nu)
        np.add.at(weight_img, flat[keep], 1.0)
        np.add.at(value_img, flat[keep], w[keep])
        weight_smooth = ndimage.gaussian_filter(
            weight_img.reshape(nv, nu), sigma_cells
        ).ravel()
        value_smooth = ndimage.gaussian_filter(
            value_img.reshape(nv, nu), sigma_cells
        ).ravel()
        reference = np.where(
            weight_smooth > 1e-6, value_smooth / np.maximum(weight_smooth, 1e-6), 0.0
        )
        residual = np.abs(w - reference[flat])
        smoothed = np.empty_like(residual)
        for group in track_groups:
            r = residual[group]
            if len(r) >= 3:
                r = ndimage.median_filter(r, size=min(window, len(r)))
            smoothed[group] = r
        keep = smoothed <= tol
    return keep


def masked_gaussian_smooth(grid: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth valid vertices without bleeding invalid values into the grid."""
    from scipy import ndimage

    valid = grid[..., 0] >= 0
    vmask = valid.astype(np.float64)
    weight = ndimage.gaussian_filter(vmask, sigma)
    out = grid.copy()
    for channel in range(3):
        filled = np.where(valid, grid[..., channel], 0.0)
        blurred = ndimage.gaussian_filter(filled, sigma)
        out[..., channel] = np.where(
            valid & (weight > 1e-6),
            blurred / np.maximum(weight, 1e-6),
            grid[..., channel],
        )
    return out


def mask_folded_cells(
    grid: np.ndarray,
    *,
    dilate: int = 2,
    normal_sigma: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Invalidate cells whose normal opposes the smoothed normal field —
    honest holes instead of folded geometry."""
    from scipy import ndimage

    x = grid[..., 0]
    valid = x >= 0
    du = grid[:, 1:] - grid[:, :-1]
    dv = grid[1:] - grid[:-1]
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    normal = np.cross(du[:-1], dv[:, :-1])
    size = np.linalg.norm(normal, axis=-1)
    good = cell_valid & (size > 0)
    normal[good] /= size[good][..., None]
    normal[~good] = 0.0
    smooth = np.stack(
        [ndimage.gaussian_filter(normal[..., c], normal_sigma) for c in range(3)],
        axis=-1,
    )
    ssize = np.linalg.norm(smooth, axis=-1)
    ok = good & (ssize > 1e-6)
    dot = np.einsum("ijk,ijk->ij", normal, smooth) / np.maximum(ssize, 1e-6)
    bad_cells = ok & (dot < 0.3)
    bad_vertices = np.zeros(valid.shape, dtype=bool)
    for dr in (0, 1):
        for dc in (0, 1):
            bad_vertices[
                dr : dr + bad_cells.shape[0], dc : dc + bad_cells.shape[1]
            ] |= bad_cells
    if dilate:
        bad_vertices = ndimage.binary_dilation(bad_vertices, iterations=dilate)
    masked = int((bad_vertices & valid).sum())
    grid = grid.copy()
    grid[bad_vertices] = -1.0
    return grid, {"fold_masked_vertices": masked}


def rasterize(
    uv: np.ndarray,
    xyz: np.ndarray,
    *,
    spacing: float = 2.0,
    support_radius: float = 35.0,
    smooth: bool = True,
    query_workers: int = -1,
) -> tuple[np.ndarray, dict]:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree

    input_points = len(uv)
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    nu = int(math.ceil((uv_max[0] - uv_min[0]) / spacing)) + 1
    nv = int(math.ceil((uv_max[1] - uv_min[1]) / spacing)) + 1
    u_axis = uv_min[0] + np.arange(nu) * spacing
    v_axis = uv_min[1] + np.arange(nv) * spacing
    gu, gv = np.meshgrid(u_axis, v_axis)
    flat = np.stack([gu.ravel(), gv.ravel()], axis=1)

    tree = cKDTree(uv)
    dist, _ = tree.query(flat, k=1, workers=query_workers)
    supported = dist <= support_radius

    interp = LinearNDInterpolator(uv, xyz)
    dense = np.full((flat.shape[0], 3), np.nan)
    dense[supported] = interp(flat[supported])
    valid = supported & ~np.isnan(dense[:, 0])
    grid = np.full((nv, nu, 3), -1.0, dtype=np.float64)
    grid.reshape(-1, 3)[valid] = dense[valid]

    # masked Gaussian smoothing: individual track points carry ~2-5 vx of
    # sheet-normal noise which crumples the raw triangulation
    from scipy import ndimage

    if not smooth:
        stats = {
            "grid_shape": [nv, nu],
            "valid_vertices": int(valid.sum()),
            "support_radius": support_radius,
            "spacing": spacing,
            "input_points": input_points,
        }
        return grid, stats
    vmask = valid.reshape(nv, nu).astype(np.float64)
    weight = ndimage.gaussian_filter(vmask, 2.0)
    for channel in range(3):
        filled = np.where(vmask > 0, grid[..., channel], 0.0)
        blurred = ndimage.gaussian_filter(filled, 2.0)
        grid[..., channel] = np.where(
            (vmask > 0) & (weight > 1e-6),
            blurred / np.maximum(weight, 1e-6),
            grid[..., channel],
        )
    stats = {
        "grid_shape": [nv, nu],
        "valid_vertices": int(valid.sum()),
        "support_radius": support_radius,
        "spacing": spacing,
        "input_points": input_points,
    }
    return grid, stats


def slim_reparameterize(
    grid: np.ndarray,
    *,
    spacing: float = 2.0,
    decimate: int = 4,
    iterations: int = 8,
    query_workers: int = -1,
    fine_raster: bool = False,
    fold_normal_sigma: float = 5.0,
    fold_dilate: int = 2,
) -> tuple[np.ndarray, dict]:
    """Re-parameterize the rasterized surface for isometry.

    The growth UV is dead-reckoned and can carry 30-40% local compression
    or shear, which warps the flattened rendering.  A SLIM pass (symmetric
    Dirichlet, free boundary) over a decimated mesh of the surface pulls
    the map to near-isometry; the dense grid is then resampled through it.
    """
    import igl

    sub = grid[::decimate, ::decimate]
    valid = sub[..., 0] >= 0
    if valid.sum() < 16:
        return grid, {"slim": "skipped: too few vertices"}
    vid = -np.ones(valid.shape, np.int64)
    vid[valid] = np.arange(int(valid.sum()))
    vertices = sub[valid].astype(np.float64)
    rows, cols = np.where(valid)
    uv0 = np.stack(
        [cols * spacing * decimate, rows * spacing * decimate], axis=1
    ).astype(np.float64)
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    cr, cc = np.where(cell_valid)
    tl = vid[cr, cc]
    tr = vid[cr, cc + 1]
    bl = vid[cr + 1, cc]
    br = vid[cr + 1, cc + 1]
    faces = np.concatenate(
        [np.stack([tl, tr, bl], axis=1), np.stack([tr, br, bl], axis=1)]
    ).astype(np.int32)

    def distortion(uvm):
        p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
        q0, q1, q2 = (uvm[faces[:, i]] for i in range(3))
        e3 = np.stack([p1 - p0, p2 - p0], axis=2)
        e2 = np.stack([q1 - q0, q2 - q0], axis=2)
        det = np.linalg.det(e2)
        ok = np.abs(det) > 1e-9
        jac = np.einsum("fik,fkj->fij", e3[ok], np.linalg.inv(e2[ok]))
        singular = np.linalg.svd(jac, compute_uv=False)
        energy = (singular**2 + 1.0 / np.maximum(singular, 1e-9) ** 2).sum(axis=1)
        return float(energy.mean()), float(np.percentile(energy, 95))

    before = distortion(uv0)
    data = igl.slim_precompute(
        np.asfortranarray(vertices),
        np.asfortranarray(faces),
        np.asfortranarray(uv0),
        igl.SYMMETRIC_DIRICHLET,
        np.zeros(0, np.int32),
        np.asfortranarray(np.zeros((0, 2))),
        0.0,
    )
    uv1 = np.asarray(igl.slim_solve(data, int(iterations)))
    after = distortion(uv1)

    if fine_raster:
        # Legacy/exact path: transfer the coarse SLIM displacement to every
        # fine vertex before triangulating the deformed surface.
        from scipy.interpolate import LinearNDInterpolator

        fine_valid = grid[..., 0] >= 0
        frows, fcols = np.where(fine_valid)
        fine_uv = np.stack([fcols * spacing, frows * spacing], axis=1).astype(
            np.float64
        )
        displacement = LinearNDInterpolator(uv0, uv1 - uv0)
        delta = displacement(fine_uv)
        ok = ~np.isnan(delta[:, 0])
        new_grid, raster_stats = rasterize(
            fine_uv[ok] + delta[ok],
            grid[fine_valid][ok],
            spacing=spacing,
            support_radius=3.0 * spacing,
            smooth=False,
            query_workers=query_workers,
        )
        reraster_source = "fine_displacement"
    else:
        # Rasterize the solved coarse mesh directly.  Interpolating its
        # displacement back to every fine vertex and triangulating those
        # vertices is wasted work before the final output reduction.
        new_grid, raster_stats = rasterize(
            uv1,
            vertices,
            spacing=spacing,
            support_radius=decimate * spacing,
            smooth=False,
            query_workers=query_workers,
        )
        reraster_source = "decimated_slim_mesh"
    new_grid, mask_stats = mask_folded_cells(
        new_grid,
        dilate=fold_dilate,
        normal_sigma=fold_normal_sigma,
    )
    return new_grid, {
        "slim_iterations": int(iterations),
        "mesh_vertices": int(len(vertices)),
        "sd_energy_mean_before": before[0],
        "sd_energy_p95_before": before[1],
        "sd_energy_mean_after": after[0],
        "sd_energy_p95_after": after[1],
        "reraster_source": reraster_source,
        "reraster_valid_vertices": raster_stats["valid_vertices"],
        **mask_stats,
    }


def clean_valid_mask(
    valid: np.ndarray,
    *,
    erode_px: int = 6,
    min_component_px: int = 2000,
) -> np.ndarray:
    """Trim border artifacts: erode the rim (ragged dead-reckoned edges and
    interpolation streaks live there), then drop small connected components
    (floating islands)."""
    from scipy import ndimage

    keep = valid.copy()
    if erode_px:
        # erode holes and border alike: hole rims are where the consensus
        # broke and carry wrap-transition geometry, so they must go too
        keep = ndimage.binary_erosion(keep, iterations=int(erode_px))
    labels, count = ndimage.label(keep)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        big = sizes >= max(min_component_px, 0.01 * sizes.max())
        keep = big[labels]
    return keep


def resample_grid(
    grid: np.ndarray,
    factor: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Bilinear resample onto a grid `factor`x coarser; a coarse vertex is
    valid only when its whole 2x2 fine support is."""
    nv, nu = grid.shape[:2]
    if valid is None:
        valid = grid[..., 0] >= 0
    rows = np.arange(0, nv - 1 + 1e-9, factor)
    cols = np.arange(0, nu - 1 + 1e-9, factor)
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    fr = (rows - r0)[:, None]
    fc = (cols - c0)[None, :]
    r1 = np.where(rows - r0 > 1e-9, np.minimum(r0 + 1, nv - 1), r0)
    c1 = np.where(cols - c0 > 1e-9, np.minimum(c0 + 1, nu - 1), c0)

    def cell(rr, cc):
        return grid[rr[:, None], cc[None, :]]

    out = (
        ((1 - fr) * (1 - fc))[..., None] * cell(r0, c0)
        + ((1 - fr) * fc)[..., None] * cell(r0, c1)
        + (fr * (1 - fc))[..., None] * cell(r1, c0)
        + (fr * fc)[..., None] * cell(r1, c1)
    )
    ok = (
        valid[r0[:, None], c0[None, :]]
        & valid[r0[:, None], c1[None, :]]
        & valid[r1[:, None], c0[None, :]]
        & valid[r1[:, None], c1[None, :]]
    )
    out[~ok] = -1.0
    return out


def finalize_coarse_grid(grid: np.ndarray) -> np.ndarray:
    """Post-resample tidy: erode one cell (boundary cells whose fine
    support straddled the cleaned rim), then keep only the largest
    connected component."""
    from scipy import ndimage

    valid = grid[..., 0] >= 0
    keep = ndimage.binary_erosion(valid, iterations=1)
    labels, count = ndimage.label(keep)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = labels == sizes.argmax()
    grid = grid.copy()
    grid[~keep] = -1.0
    return grid


def median_edge_vx(grid: np.ndarray) -> float:
    """Median 3D distance between adjacent valid vertices."""
    valid = grid[..., 0] >= 0
    edges = []
    for axis in (0, 1):
        a = grid[:-1] if axis == 0 else grid[:, :-1]
        b = grid[1:] if axis == 0 else grid[:, 1:]
        m = (valid[:-1] & valid[1:]) if axis == 0 else (valid[:, :-1] & valid[:, 1:])
        if m.any():
            edges.append(np.linalg.norm((a - b)[m], axis=-1))
    if not edges:
        return float("nan")
    return float(np.median(np.concatenate(edges)))


def write_tifxyz(
    out: Path,
    grid: np.ndarray,
    meta_extra: dict,
    scale: tuple = (1.0, 1.0),
) -> None:
    import tifffile

    out.mkdir(parents=True, exist_ok=False)
    nv, nu = grid.shape[:2]
    bordered = np.full((nv + 2, nu + 2, 3), -1.0, dtype=np.float32)
    bordered[1:-1, 1:-1] = grid.astype(np.float32)
    for channel, name in enumerate(("x", "y", "z")):
        tifffile.imwrite(
            out / f"{name}.tif",
            bordered[..., channel],
            dtype=np.float32,
            compression=None,
            photometric="minisblack",
        )
    valid = bordered[..., 0] >= 0
    points = bordered[valid]
    area_vx2 = surface_area_vx2(bordered)
    meta = {
        "format": "tifxyz",
        "type": "seg",
        "uuid": out.name,
        "scale": [float(scale[0]), float(scale[1])],
        "bbox": [
            points.min(axis=0).astype(float).tolist(),
            points.max(axis=0).astype(float).tolist(),
        ],
        "source": "band grower prototype (crossing-corroborated UV)",
        "area_vx2": area_vx2,
        "area_cm2": area_vx2_to_cm2(area_vx2),
        **meta_extra,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _collect_trimmed_surface(grower, args, *, min_points: int = 0):
    """Collect the currently published surface and apply all point trims."""
    uv, xyz, track_ids, arcs = grower.surface_points_arcs()
    if len(uv) < min_points:
        raise ValueError(f"only {len(uv)} surface points grown")
    keep = trim_offsheet_points(
        uv,
        xyz,
        track_ids,
        residual_tol=max(6.0, args.gate_tol),
    )
    keep &= z_consistency_trim(uv, xyz, track_ids)
    keep &= irls_trim(uv, xyz)
    return uv, xyz, track_ids, arcs, keep


def process_seed(seed, out, args, tracks, crossings) -> dict:
    """Grow one seed and write its tifxyz; returns a result record."""
    start = time.time()
    stage_start = start
    timings = {}

    def stage(name: str) -> None:
        nonlocal stage_start
        now = time.time()
        timings[name] = round(now - stage_start, 3)
        stage_start = now

    try:
        grower = GraphLatticeGrower(
            tracks,
            crossings,
            uv_radius=size_cm2_to_uv_radius(args.max_size),
            min_track_arclength=args.min_track_arclength,
            anchor_reach=ANCHOR_REACH,
            min_connect=args.min_connect,
            evidence_spacing=LATTICE_EVIDENCE_SPACING,
            rail_agree_frac=LATTICE_RAIL_AGREE_FRAC,
            rail_min_votes=LATTICE_RAIL_MIN_VOTES,
            proposal_agree_frac=LATTICE_PROPOSAL_AGREE_FRAC,
            span_witnesses=GRAPH_SPAN_WITNESSES,
            witness_min_offset=GRAPH_WITNESS_MIN_OFFSET,
            witness_max_offset=GRAPH_WITNESS_MAX_OFFSET,
            witness_shear_tol=GRAPH_WITNESS_SHEAR_TOL,
            graph_trim=GRAPH_TRIM,
            max_span=GRAPH_MAX_SPAN,
            blind_gap=GRAPH_BLIND_GAP,
            anchor_corroborate=GRAPH_ANCHOR_CORROBORATE,
            growth_min_span=args.growth_min_span,
        )
        grower.grow(seed)
        stage("grow")
        refine = grower.refine_uv()
        stage("refine_uv")
        uv, xyz, track_ids, arcs, keep = _collect_trimmed_surface(
            grower, args, min_points=64
        )
        if SPAN_VERIFY_TOL > 0 and keep.any():
            unpublished = grower.verify_published_windows(
                uv[keep],
                xyz[keep],
                track_ids[keep],
                tol=SPAN_VERIFY_TOL,
            )
            if unpublished:
                uv, xyz, track_ids, arcs, keep = _collect_trimmed_surface(grower, args)
        gap_published = 0
        if LATTICE_GAP_PUBLISH_TOL > 0 and keep.any():
            gap_published = grower.publish_consensus_gaps(
                uv[keep], xyz[keep], tol=LATTICE_GAP_PUBLISH_TOL
            )
            if gap_published:
                # re-collect (new spans publish new points) and re-trim
                uv, xyz, track_ids, arcs, keep = _collect_trimmed_surface(grower, args)
        stage("trim_and_publish")
        uv_out, xyz_out = uv[keep], xyz[keep]
        kept_arcs = {}
        for group in _contiguous_track_groups(track_ids):
            group_keep = keep[group]
            if not group_keep.any():
                continue
            group_tracks = track_ids[group]
            group_arcs = arcs[group]
            kept_arcs[int(group_tracks[0])] = np.sort(group_arcs[group_keep])
        fill_uv, fill_xyz = grower.fill_points(
            kept_arcs,
            spacing=args.resample_spacing,
            fill_gap=LATTICE_FILL_GAP,
        )
        fill_count = int(len(fill_uv))
        if fill_count:
            uv_out = np.concatenate([uv_out, fill_uv])
            xyz_out = np.concatenate([xyz_out, fill_xyz])
        stage("fill")
        thickness = sheet_thickness_stats(uv_out, xyz_out)
        thick_cell_frac = float(thickness["thick_cell_frac"])
        if (
            args.max_thick_cell_frac > 0
            and thick_cell_frac >= args.max_thick_cell_frac
        ):
            return {
                "ok": False,
                "discarded": True,
                "seed": seed,
                "reason": (
                    f"mixed-sheet cell fraction {thick_cell_frac:.2%} "
                    f">= --max-thick-cell-frac "
                    f"{args.max_thick_cell_frac:.2%}"
                ),
            }
        query_workers = -1 if args.workers <= 1 else 1
        grid, stats = rasterize(
            uv_out,
            xyz_out,
            spacing=args.resample_spacing,
            support_radius=RASTER_SUPPORT_RADIUS,
            smooth=False,
            query_workers=query_workers,
        )
        stage("rasterize")
        if fill_count:
            stats["fill_points"] = fill_count
        if gap_published:
            stats["gap_windows_published"] = gap_published
        # Keep filters at their historical physical scale (2 grid pixels at
        # 2-vx spacing for point smoothing; 5 pixels for the normal field).
        smooth_sigma = max(0.0, 4.0 / args.resample_spacing)
        fold_normal_sigma = max(0.5, 10.0 / args.resample_spacing)
        fold_dilate = max(0, int(round(4.0 / args.resample_spacing)))
        grid = masked_gaussian_smooth(grid, sigma=smooth_sigma)
        grid, mask_stats = mask_folded_cells(
            grid,
            dilate=fold_dilate,
            normal_sigma=fold_normal_sigma,
        )
        # SLIM historically solved an 8-vx mesh (2-vx raster, decimate 4).
        # At coarse working spacings, use every raster vertex.
        slim_decimate = max(1, int(round(8.0 / args.resample_spacing)))
        grid, slim_stats = slim_reparameterize(
            grid,
            spacing=args.resample_spacing,
            decimate=slim_decimate,
            iterations=SLIM_ITERATIONS,
            query_workers=query_workers,
            fine_raster=SLIM_FINE_RASTER,
            fold_normal_sigma=fold_normal_sigma,
            fold_dilate=fold_dilate,
        )
        stats["slim"] = slim_stats
        stage("smooth_and_slim")
        stats.update(mask_stats)
        stats["valid_vertices"] = int((grid[..., 0] >= 0).sum())
        stats["trimmed_fraction"] = float(1.0 - keep.mean())
        if args.reject_any_fold_fixes and (
            int(mask_stats["fold_masked_vertices"]) > 0
            or int(slim_stats["fold_masked_vertices"]) > 0
        ):
            return {
                "ok": False,
                "discarded": True,
                "seed": seed,
                "reason": (
                    "fold fixes applied "
                    f"(raster={int(mask_stats['fold_masked_vertices'])}, "
                    f"slim={int(slim_stats['fold_masked_vertices'])})"
                ),
            }
        # --min-valid-vertices historically counts a 2-vx working raster.
        # Scale the threshold so changing working resolution does not change
        # the minimum accepted physical patch area by spacing².
        min_valid_at_spacing = int(
            math.ceil(
                args.min_valid_vertices
                * (2.0 / args.resample_spacing) ** 2
            )
        )
        stats["min_valid_vertices_at_spacing"] = min_valid_at_spacing
        if args.random_count and stats["valid_vertices"] < min_valid_at_spacing:
            return {
                "ok": False,
                "discarded": True,
                "seed": seed,
                "valid": stats["valid_vertices"],
                "reason": (
                    f"{stats['valid_vertices']} valid vertices "
                    f"< scaled --min-valid-vertices {min_valid_at_spacing} "
                    f"at {args.resample_spacing:g}-vx spacing"
                ),
            }
        scale = (1.0, 1.0)
        if args.output_spacing > 0:
            fine_edge = median_edge_vx(grid)
            factor = args.output_spacing / fine_edge
            erode_px = max(1, int(round(args.border_erode_vx / fine_edge)))
            cleaned = clean_valid_mask(
                grid[..., 0] >= 0,
                erode_px=erode_px,
                min_component_px=int(MIN_COMPONENT_VX2 / fine_edge**2),
            )
            grid = grid.copy()
            grid[~cleaned] = -1.0
            grid = finalize_coarse_grid(resample_grid(grid, factor))
            # tifxyz scale = grid units per voxel (20vx spacing -> 0.05)
            scale = (1.0 / args.output_spacing, 1.0 / args.output_spacing)
            stats["resample"] = {
                "fine_edge_vx": fine_edge,
                "factor": factor,
                "border_erode_px": erode_px,
                "coarse_valid_vertices": int((grid[..., 0] >= 0).sum()),
            }
        stage("cleanup_and_resample")
        area_cm2 = area_vx2_to_cm2(surface_area_vx2(grid))
        stats["area_cm2"] = area_cm2
        if not args.min_size <= area_cm2 <= args.max_size:
            comparison = "< --min-size" if area_cm2 < args.min_size else "> --max-size"
            limit = args.min_size if area_cm2 < args.min_size else args.max_size
            return {
                "ok": False,
                "discarded": True,
                "seed": seed,
                "valid": stats["valid_vertices"],
                "area_cm2": area_cm2,
                "reason": f"{area_cm2:.6f} cm² {comparison} {limit:.6f} cm²",
            }
        write_tifxyz(
            out,
            grid,
            {
                "seed_track_row": seed,
                "tracks_path": str(tracks.path.resolve()),
                "mode": "graph",
                "accepted_tracks": len(grower.accepted),
                "surface_points": int(len(uv)),
                "min_size_cm2": args.min_size,
                "max_size_cm2": args.max_size,
                "max_thick_cell_frac": args.max_thick_cell_frac,
                "reject_any_fold_fixes": args.reject_any_fold_fixes,
                "sheet_gate_tol": args.gate_tol,
                "anchor_reach": ANCHOR_REACH,
                "lattice": {
                    "min_connect": args.min_connect,
                    "evidence_spacing": LATTICE_EVIDENCE_SPACING,
                    "rail_agree_frac": LATTICE_RAIL_AGREE_FRAC,
                    "rail_min_votes": LATTICE_RAIL_MIN_VOTES,
                    "proposal_agree_frac": LATTICE_PROPOSAL_AGREE_FRAC,
                    "fill_gap": LATTICE_FILL_GAP,
                    "gap_publish_tol": LATTICE_GAP_PUBLISH_TOL,
                    "discarded_rails": len(grower.discarded),
                },
                "graph": {
                    "span_witnesses": GRAPH_SPAN_WITNESSES,
                    "witness_min_offset": GRAPH_WITNESS_MIN_OFFSET,
                    "witness_max_offset": GRAPH_WITNESS_MAX_OFFSET,
                    "witness_shear_tol": GRAPH_WITNESS_SHEAR_TOL,
                    "graph_trim": GRAPH_TRIM,
                    "max_span": GRAPH_MAX_SPAN,
                    "blind_gap": GRAPH_BLIND_GAP,
                    "growth_min_span": args.growth_min_span,
                },
                "quality": thickness,
                "uv_refine": refine,
                "raster": stats,
                "timings_seconds": timings,
                "diagnostics": dict(grower.diag),
            },
            scale=scale,
        )
        stage("write")
        return {
            "ok": True,
            "seed": seed,
            "out": str(out),
            "tracks": len(grower.accepted),
            "valid": stats["valid_vertices"],
            "area_cm2": area_cm2,
            "secs": time.time() - start,
        }
    except Exception as error:
        return {
            "ok": False,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        }


_WORKER: tuple | None = None
_THREADPOOL_LIMITER = None


def _initialize_worker(tracks_path: str, crossings_path: str, args) -> None:
    global _THREADPOOL_LIMITER, _WORKER
    # Each process handles an independent patch.  Letting every NumPy/SciPy
    # operation also fan out over all cores badly oversubscribes machines
    # when --workers is greater than one.
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    tracks = PackedTracks(tracks_path)
    crossings = CrossingCsr(crossings_path, tracks, validate_source_ids=False)
    _WORKER = (tracks, crossings, args)


def _worker_task(task: tuple) -> dict:
    seed, out_text = task
    assert _WORKER is not None
    tracks, crossings, args = _WORKER
    return process_seed(int(seed), Path(out_text), args, tracks, crossings)


def main(argv=None) -> int:
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path, help="packed .vctracks directory")
    parser.add_argument("crossings", type=Path, help="uncompressed crossings .npz")
    parser.add_argument(
        "output", type=Path, help="parent directory for .tifxyz outputs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="explicit seed track rows to grow (one output per seed)",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        help=(
            "grow this many patches from random unused seeds drawn from the "
            "configured longest-track percentage (alternative to --seeds)"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="RNG seed for --random-count selection (default: 0)",
    )
    parser.add_argument(
        "--random-top-percent",
        type=float,
        default=100.0 * TOP_ARCLENGTH_FRACTION,
        help="draw random seeds from the longest this percent of tracks",
    )
    parser.add_argument(
        "--min-valid-vertices",
        type=int,
        default=20000,
        help=(
            "random mode: outputs smaller than this are discarded and do "
            "not count toward --random-count (default: 20000)"
        ),
    )
    parser.add_argument(
        "--growth-min-span",
        type=float,
        default=80.0,
        help=(
            "graph: minimum arclength span (vx) covered by the consistent "
            "rails supporting a growth proposal; narrow rail bundles are "
            "rejected before they can propagate tendrils (0 disables)"
        ),
    )
    parser.add_argument(
        "--min-connect",
        type=int,
        default=3,
        help=(
            "lattice: distinct spaced opposite-family rails that must "
            "consistently cross a proposal for acceptance"
        ),
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.0,
        metavar="CM2",
        help="reject final surfaces smaller than this measured area in cm²",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=DEFAULT_MAX_SIZE_CM2,
        metavar="CM2",
        help=(
            "maximum surface size in cm²; also sets the equivalent square "
            "growth window (default preserves the former 600-vx radius)"
        ),
    )
    parser.add_argument(
        "--max-thick-cell-frac",
        type=float,
        default=DEFAULT_MAX_THICK_CELL_FRAC,
        help=(
            "discard patches with at least this fraction of locally "
            "multi-layer UV cells; 0 disables (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reject-any-fold-fixes",
        action="store_true",
        help=(
            "discard a patch if either rasterization stage masks one or "
            "more folded vertices"
        ),
    )
    parser.add_argument("--gate-tol", type=float, default=9.0)
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=5.0,
        help=(
            "working raster spacing in voxels before the final output "
            "reduction (default: 5)"
        ),
    )
    parser.add_argument(
        "--min-track-arclength",
        type=float,
        default=40.0,
        help="skip candidate tracks shorter than this (voxels)",
    )
    parser.add_argument(
        "--output-spacing",
        type=float,
        default=20.0,
        help=(
            "3D vertex spacing (vx) of the written grid: the fine raster "
            "is border-cleaned and bilinearly resampled to this spacing, "
            "and meta scale is set to match; 0 writes the fine grid as-is"
        ),
    )
    parser.add_argument(
        "--border-erode-vx",
        type=float,
        default=40.0,
        help=(
            "erode this many voxels off the valid border before the "
            "output resample (ragged rims and interpolation streaks)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="grow this many seeds concurrently (default: 1, in-process)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.seeds) == bool(args.random_count):
        parser.error("pass exactly one of --seeds or --random-count")
    if not math.isfinite(args.random_top_percent) or not (
        0.0 < args.random_top_percent <= 100.0
    ):
        parser.error("--random-top-percent must lie in (0, 100]")
    if not math.isfinite(args.min_size) or args.min_size < 0.0:
        parser.error("--min-size must be finite and non-negative")
    if not math.isfinite(args.max_size) or args.max_size <= 0.0:
        parser.error("--max-size must be finite and positive")
    if args.min_size > args.max_size:
        parser.error("--min-size must not exceed --max-size")
    if (
        not math.isfinite(args.max_thick_cell_frac)
        or not 0.0 <= args.max_thick_cell_frac <= 1.0
    ):
        parser.error("--max-thick-cell-frac must be a finite number in [0, 1]")
    if args.growth_min_span < 0:
        parser.error("--growth-min-span must be non-negative")
    if not math.isfinite(args.resample_spacing) or args.resample_spacing <= 0:
        parser.error("--resample-spacing must be finite and positive")
    tracks = PackedTracks(args.tracks)
    crossings = CrossingCsr(args.crossings, tracks, validate_source_ids=False)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        candidates = list(args.seeds)
        target = len(candidates)
    else:
        already_used = previously_used_seed_rows(args.output, tracks.path)
        selected, cutoff, _, _ = random_top_arclength_rows(
            tracks,
            random_seed=args.random_seed,
            excluded=already_used,
            top_fraction=args.random_top_percent / 100.0,
        )
        candidates = [int(row) for row in selected]
        target = int(args.random_count)
        print(
            f"random mode: {target} patches from {len(candidates)} unused "
            f"top-{args.random_top_percent:g}% arclength seeds "
            f"(cutoff {cutoff:.0f} vx, "
            f"{len(already_used)} previously used)"
        )

    failures = 0
    successes = 0
    cursor = 0

    def next_task():
        """Advance to the next seed that actually needs growing; handles
        skip-existing/overwrite in the parent so workers stay simple."""
        nonlocal cursor, successes
        while cursor < len(candidates):
            seed = candidates[cursor]
            cursor += 1
            existing = sorted(args.output.glob(f"band-seed{seed}-*.tifxyz")) + [
                p for p in (args.output / f"band-seed{seed}.tifxyz",) if p.exists()
            ]
            if existing:
                if not args.overwrite:
                    print(f"skip existing {existing[0]}")
                    if args.seeds:
                        successes += 1
                    continue
                for p in existing:
                    shutil.rmtree(p)
            now = time.time()
            stamp = (
                time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
                + f"-{int(now * 1000) % 1000:03d}"
            )
            out = args.output / f"band-seed{seed}-{stamp}.tifxyz"
            return seed, str(out)
        return None

    def report(result) -> None:
        nonlocal successes, failures
        if result.get("discarded"):
            if args.seeds:
                failures += 1
            print(f"seed {result['seed']} discarded: {result['reason']}")
            return
        if result["ok"]:
            successes += 1
            print(
                f"[{successes}/{target}] wrote {result['out']} "
                f"tracks={result['tracks']} "
                f"valid={result['valid']} "
                f"area={result['area_cm2']:.6f}cm² "
                f"secs={result['secs']:.1f}"
            )
        else:
            failures += 1
            print(f"seed {result['seed']} failed: {result['error']}")

    if args.workers <= 1:
        while successes < target:
            task = next_task()
            if task is None:
                break
            seed, out_text = task
            report(process_seed(seed, Path(out_text), args, tracks, crossings))
    else:
        import multiprocessing as mp
        from concurrent.futures import (
            FIRST_COMPLETED,
            ProcessPoolExecutor,
            wait,
        )

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(args.tracks), str(args.crossings), args),
        ) as executor:
            pending = {}
            while successes < target:
                while len(pending) < args.workers and successes + len(pending) < target:
                    task = next_task()
                    if task is None:
                        break
                    pending[executor.submit(_worker_task, task)] = task
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    report(future.result())

    if successes < target:
        print(f"only {successes} of {target} requested patches were written")
        return 2
    return 2 if (args.seeds and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
