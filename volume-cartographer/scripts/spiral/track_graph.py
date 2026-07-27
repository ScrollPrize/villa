from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow
import vc.track_store as track_store
import vc.compression.vcz1_numcodecs  # Registers the VCZ1 codec with numcodecs.
import zarr
import fsspec
import napari
import dask.array as da
from huggingface_hub import sync_bucket as _sync_bucket
from tifxyz import save_tifxyz

family_names = {
  -1: "unknown",
  0: "hz",
  1: "vt",
}

@dataclass(slots=True, repr=False)
class TrackCollection:
    coordinates: np.ndarray
    offsets: np.ndarray
    source_ids: np.ndarray
    family_codes: np.ndarray
    z_bounds: np.ndarray
    arclengths: np.ndarray
    tortuosities: np.ndarray
    crossing_graph: csr_array
    crossing_positions: np.ndarray


    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, row):
        return Track(self, int(row))

    def __repr__(self):
        return f"TrackCollection({len(self):,} tracks)"


@dataclass(frozen=True, slots=True)
class Track:
    """Lightweight handle that materializes track data only when accessed."""

    collection: TrackCollection = field(repr=False, compare=False)
    row: int

    @property
    def source_id(self):
        return int(self.collection.source_ids[self.row])

    @property
    def family(self):
        return int(self.collection.family_codes[self.row])

    @property
    def family_name(self):
        return family_names[self.family]

    @property
    def arclength(self):
        return float(self.collection.arclengths[self.row])

    @property
    def tortuosity(self):
        return float(self.collection.tortuosities[self.row])

    @property
    def z_bounds(self):
        return tuple(map(int, self.collection.z_bounds[self.row]))

    @property
    def points_zyx(self):
        begin = int(self.collection.offsets[self.row])
        end = int(self.collection.offsets[self.row + 1])
        return self.collection.coordinates[begin:end]

    @property
    def neighbors(self):
        indptr = self.collection.crossing_graph.indptr
        begin = int(indptr[self.row])
        end = int(indptr[self.row + 1])
        return self.collection.crossing_graph.indices[begin:end]
    
    @property
    def crossing_positions(self):
        indptr = self.collection.crossing_graph.indptr
        begin = int(indptr[self.row])
        end = int(indptr[self.row + 1])
        return self.collection.crossing_positions[begin:end]

    def spaced_crossing_tracks(
        self,
        min_arclength=0.0,
        min_spacing=0.0,
        min_shared_z_extent=0.0,
    ):
        """Return crossing tracks ordered from this track's first endpoint.

        ``min_arclength`` filters the crossing tracks, while ``min_spacing``
        is the minimum arclength along this track between selected crossings.
        ``min_shared_z_extent`` requires every returned track to cover a
        common z interval of at least that size.
        The result contains as many tracks as possible and, among that number,
        chooses crossings that are as close as possible to uniform spacing.
        Results are ordered by increasing arclength along this track, starting
        at the endpoint represented by ``points_zyx[0]``.
        All three arguments use the same coordinate units as ``arclength``.
        """
        min_arclength = float(min_arclength)
        min_spacing = float(min_spacing)
        min_shared_z_extent = float(min_shared_z_extent)
        if not np.isfinite(min_arclength) or min_arclength < 0:
            raise ValueError("min_arclength must be finite and non-negative")
        if not np.isfinite(min_spacing) or min_spacing < 0:
            raise ValueError("min_spacing must be finite and non-negative")
        if (
            not np.isfinite(min_shared_z_extent)
            or min_shared_z_extent < 0
        ):
            raise ValueError(
                "min_shared_z_extent must be finite and non-negative"
            )

        neighbor_rows = np.asarray(self.neighbors, dtype=np.int64)
        positions = np.asarray(self.crossing_positions, dtype=np.float64)
        keep = (
            np.isfinite(positions)
            & (self.collection.arclengths[neighbor_rows] >= min_arclength)
        )
        neighbor_rows = neighbor_rows[keep]
        positions = positions[keep]
        if not neighbor_rows.size:
            return []

        if min_shared_z_extent > 0:
            z_bounds = np.asarray(
                self.collection.z_bounds[neighbor_rows],
                dtype=np.float64,
            )
            z_min = np.min(z_bounds, axis=1)
            z_max = np.max(z_bounds, axis=1)

            # A track can contain a shared interval [start, start + extent]
            # exactly when start lies in [z_min, z_max - extent]. Find the
            # start covered by the greatest number of eligible tracks.
            interval_starts = z_min
            interval_ends = z_max - min_shared_z_extent
            feasible = interval_starts <= interval_ends
            neighbor_rows = neighbor_rows[feasible]
            positions = positions[feasible]
            interval_starts = interval_starts[feasible]
            interval_ends = interval_ends[feasible]
            if not neighbor_rows.size:
                return []

            candidate_starts = np.unique(interval_starts)
            sorted_starts = np.sort(interval_starts)
            sorted_ends = np.sort(interval_ends)
            coverage_counts = (
                np.searchsorted(sorted_starts, candidate_starts, side="right")
                - np.searchsorted(sorted_ends, candidate_starts, side="left")
            )
            shared_start = candidate_starts[np.argmax(coverage_counts)]
            shares_interval = (
                (interval_starts <= shared_start)
                & (interval_ends >= shared_start)
            )
            neighbor_rows = neighbor_rows[shares_interval]
            positions = positions[shares_interval]

        position_order = np.argsort(positions, kind="stable")
        neighbor_rows = neighbor_rows[position_order]
        positions = positions[position_order]

        if min_spacing == 0 or neighbor_rows.size == 1:
            return [self.collection[int(row)] for row in neighbor_rows]

        # Greedy interval scheduling gives the largest feasible selection size.
        maximum_count = 0
        last_position = -np.inf
        for position in positions:
            if position - last_position >= min_spacing:
                maximum_count += 1
                last_position = position

        if maximum_count == 1:
            middle = 0.5 * (positions[0] + positions[-1])
            chosen = np.array([np.argmin(np.abs(positions - middle))])
        elif maximum_count == neighbor_rows.size:
            chosen = np.arange(neighbor_rows.size)
        else:
            # Dynamic programming chooses exactly maximum_count records while
            # minimizing squared distance from uniformly distributed targets.
            targets = np.linspace(
                positions[0], positions[-1], maximum_count,
                dtype=np.float64,
            )
            count = neighbor_rows.size
            previous_cost = (positions - targets[0]) ** 2
            back_pointers = []
            for target in targets[1:]:
                current_cost = np.full(count, np.inf)
                current_back = np.full(count, -1, dtype=np.int32)

                # For each crossing, find the final previous crossing that is
                # far enough away, then look up the cheapest DP state in that
                # prefix.
                final_predecessors = np.searchsorted(
                    positions,
                    positions - min_spacing,
                    side="right",
                ) - 1
                prefix_costs = np.minimum.accumulate(previous_cost)
                new_best = np.empty(count, dtype=bool)
                new_best[0] = np.isfinite(previous_cost[0])
                new_best[1:] = previous_cost[1:] < prefix_costs[:-1]
                prefix_indices = np.maximum.accumulate(
                    np.where(new_best, np.arange(count), -1)
                )

                valid = final_predecessors >= 0
                valid_indices = np.flatnonzero(valid)
                predecessor_limits = final_predecessors[valid]
                finite = np.isfinite(prefix_costs[predecessor_limits])
                valid_indices = valid_indices[finite]
                predecessor_limits = predecessor_limits[finite]
                current_cost[valid_indices] = (
                    prefix_costs[predecessor_limits]
                    + (positions[valid_indices] - target) ** 2
                )
                current_back[valid_indices] = prefix_indices[
                    predecessor_limits
                ]
                previous_cost = current_cost
                back_pointers.append(current_back)

            chosen = np.empty(maximum_count, dtype=np.int64)
            chosen[-1] = int(np.argmin(previous_cost))
            for level in range(maximum_count - 2, -1, -1):
                chosen[level] = back_pointers[level][chosen[level + 1]]

        return [
            self.collection[int(row)]
            for row in neighbor_rows[chosen]
        ]

    def connectivity_summary(self):
        neighbors = self.neighbors
        unique_neighbors = np.unique(neighbors)
        families, counts = np.unique(
            self.collection.family_codes[unique_neighbors],
            return_counts=True,
        )
        return {
            "crossings": int(neighbors.size),
            "unique_neighbors": int(unique_neighbors.size),
            "neighbors_by_family": {
                family_names[int(family)]: int(count)
                for family, count in zip(families, counts)
            },
        }

def scale_track(points_zyx, source_scale, target_scale):
    """Map (z, y, x) coordinates between Zarr pyramid scales."""
    scale_ratio = 2.0 ** (target_scale - source_scale)
    return np.asarray(points_zyx, dtype=np.float64) / scale_ratio

def crop_tracks_to_shared_z_extent(tracks):
    """Return track point arrays cropped to their common inclusive z extent."""
    point_sets = [np.asarray(track.points_zyx) for track in tracks]
    if not point_sets:
        return None, []

    z_min = max(int(points[:, 0].min()) for points in point_sets)
    z_max = min(int(points[:, 0].max()) for points in point_sets)
    if z_min > z_max:
        raise ValueError("crossing tracks do not have a shared z extent")

    cropped_point_sets = [
        points[(points[:, 0] >= z_min) & (points[:, 0] <= z_max)]
        for points in point_sets
    ]
    return (z_min, z_max), cropped_point_sets

def sample_tracks_at_shared_z_levels(tracks, spacing=20.0):
    """Sample every track at common z levels, ordered highest z first.

    Returns ``(z_levels, points_zyx)`` where ``points_zyx`` has shape
    ``(number_of_z_levels, number_of_tracks, 3)``. Tracks are treated as
    single-valued functions of z. Multiple points at one z are averaged before
    linear interpolation.
    """
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")

    point_sets = [
        np.asarray(
            track.points_zyx if hasattr(track, "points_zyx") else track,
            dtype=np.float64,
        )
        for track in tracks
    ]
    if not point_sets:
        return np.empty(0, dtype=np.float64), np.empty((0, 0, 3))
    if any(points.ndim != 2 or points.shape[1] != 3 or not len(points)
           for points in point_sets):
        raise ValueError("every track must be a non-empty (n, 3) ZYX array")

    z_min = max(points[:, 0].min() for points in point_sets)
    z_max = min(points[:, 0].max() for points in point_sets)
    if z_min > z_max:
        raise ValueError("tracks do not have a shared z extent")

    level_count = int(np.floor((z_max - z_min) / spacing)) + 1
    z_levels = z_max - spacing * np.arange(level_count, dtype=np.float64)
    sampled = np.empty((level_count, len(point_sets), 3), dtype=np.float64)
    sampled[..., 0] = z_levels[:, None]

    for track_index, points in enumerate(point_sets):
        z_values, inverse = np.unique(points[:, 0], return_inverse=True)
        yx_sums = np.zeros((len(z_values), 2), dtype=np.float64)
        np.add.at(yx_sums, inverse, points[:, 1:])
        counts = np.bincount(inverse)
        yx_values = yx_sums / counts[:, None]
        sampled[:, track_index, 1] = np.interp(
            z_levels, z_values, yx_values[:, 0]
        )
        sampled[:, track_index, 2] = np.interp(
            z_levels, z_values, yx_values[:, 1]
        )

    return z_levels, sampled

def catmull_rom_interpolate(points, samples_per_segment):
    """Interpolate ordered control points with a vectorized Catmull–Rom spline.

    ``points`` has shape ``(..., control_points, dimensions)``. The leading
    dimensions are evaluated in parallel. ``samples_per_segment`` may be one
    integer or one integer per adjacent pair of control points. Every control
    point occurs exactly in the result.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 2 or points.shape[-2] < 2:
        raise ValueError("points must contain at least two control points")

    segment_count = points.shape[-2] - 1
    subdivisions = np.asarray(samples_per_segment)
    if subdivisions.ndim == 0:
        subdivisions = np.full(segment_count, subdivisions)
    if subdivisions.shape != (segment_count,):
        raise ValueError(
            "samples_per_segment must be scalar or have one value per segment"
        )
    if not np.issubdtype(subdivisions.dtype, np.integer):
        if np.any(subdivisions != np.floor(subdivisions)):
            raise ValueError("samples_per_segment values must be integers")
        subdivisions = subdivisions.astype(np.int64)
    else:
        subdivisions = subdivisions.astype(np.int64, copy=False)
    if np.any(subdivisions < 1):
        raise ValueError("samples_per_segment values must be positive")

    padded = np.concatenate(
        (points[..., :1, :], points, points[..., -1:, :]),
        axis=-2,
    )
    interpolated_segments = []
    for index, sample_count in enumerate(subdivisions):
        p0 = padded[..., index, :]
        p1 = padded[..., index + 1, :]
        p2 = padded[..., index + 2, :]
        p3 = padded[..., index + 3, :]
        t = (
            np.arange(sample_count, dtype=np.float64) / sample_count
        ).reshape((1,) * (p0.ndim - 1) + (sample_count, 1))
        t2 = t * t
        t3 = t2 * t
        segment = 0.5 * (
            2.0 * p1[..., None, :]
            + (-p0 + p2)[..., None, :] * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[..., None, :] * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[..., None, :] * t3
        )
        interpolated_segments.append(segment)

    interpolated_segments.append(points[..., -1:, :])
    return np.concatenate(interpolated_segments, axis=-2)

def tracks_to_2d_grid(tracks, row_spacing=20.0, column_spacing=20.0):
    """Create a dense ``(rows, columns, 3)`` ZYX grid from ordered tracks.

    Rows descend from the highest shared z level by ``row_spacing``. Each row
    is a Catmull–Rom connector through the tracks in their supplied order.
    Track-to-track segments use a shared subdivision count across all rows,
    chosen from their median length to approximate ``column_spacing`` while
    preserving every sampled track point as a grid vertex.
    """
    column_spacing = float(column_spacing)
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        raise ValueError("column_spacing must be finite and positive")

    _, control_grid = sample_tracks_at_shared_z_levels(
        tracks, spacing=row_spacing
    )
    if control_grid.shape[1] < 2:
        raise ValueError("at least two tracks are required to create a grid")

    segment_lengths = np.linalg.norm(
        np.diff(control_grid, axis=1),
        axis=2,
    )
    reference_lengths = np.median(segment_lengths, axis=0)
    subdivisions = np.maximum(
        1,
        np.rint(reference_lengths / column_spacing).astype(np.int64),
    )
    return catmull_rom_interpolate(control_grid, subdivisions)

def write_2d_grid_as_tifxyz(
    grid_zyx,
    output_directory,
    *,
    spacing=20.0,
    voxel_size_um,
    source="ordered crossing-track Catmull-Rom grid",
    surface_uuid=None,
):
    """Write a ZYX grid beneath ``output_directory`` using its UUID as its ID."""
    grid_zyx = np.asarray(grid_zyx)
    if grid_zyx.ndim != 3 or grid_zyx.shape[-1] != 3:
        raise ValueError("grid_zyx must have shape (rows, columns, 3)")
    if not np.all(np.isfinite(grid_zyx)):
        raise ValueError("grid_zyx must contain only finite coordinates")

    if surface_uuid is None:
        surface_uuid = datetime.now().strftime("%y%m%d%H%S")
    else:
        surface_uuid = str(surface_uuid)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_tifxyz(
        grid_zyx,
        output_directory,
        surface_uuid,
        step_size=float(spacing),
        voxel_size_um=float(voxel_size_um),
        source=source,
    )
    return output_directory / surface_uuid

def maybe_fetch_tracks(tracks_path= Path("/home/sean/Desktop/spiral_dataset/to_hf/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks")):
    
    _track_basename = "2um_ds2_ps256_surf_v2.dbm"
    _plan = _sync_bucket(
        "hf://buckets/scrollprize/datasets/spiral/PHercParis4/tracks",
        str(tracks_path.parent),
        include=[
            f"{_track_basename}.vctracks/*",
            f"{_track_basename}.crossings.npz",
        ],
        ignore_times=True,
        token=False,
    )
    print("Hugging Face sync:", _plan.summary())
    
def load_tracks(local_tracks_path):
    info = track_store.inspect(str(local_tracks_path))
    
    crossing_cache_path = local_tracks_path.with_suffix(".crossings.npz")
    with np.load(crossing_cache_path, allow_pickle=False) as crossing_cache:
        crossing_indptr = crossing_cache["offsets"].astype(np.int32)
        crossing_indices = crossing_cache["partners"]
        crossing_positions = crossing_cache["positions"]
    
    crossing_capacities = np.ones(crossing_indices.size, dtype=np.uint8)
    crossing_graph = csr_array(
        (crossing_capacities, crossing_indices, crossing_indptr),
        shape=(info["track_count"], info["track_count"]),
        copy=False,
    )
    
    print(
        f"{crossing_graph.shape[0]:,} tracks, "
        f"{crossing_graph.nnz:,} directed adjacency entries"
    )
    
    coordinates = np.memmap(
        local_tracks_path / "coordinates.i32",
        mode="r",
        dtype="<i4",
        shape=(info["point_count"], 3),
    )
    
    offsets = np.memmap(
        local_tracks_path / "offsets.i64",
        mode="r",
        dtype="<i8",
        shape=(info["track_count"] + 1,),
    )
    
    source_ids = np.memmap(
        local_tracks_path / "source_ids.u64",
        mode="r",
        dtype="<u8",
        shape=(info["track_count"],),
    )
    
    track_types = np.memmap(
        local_tracks_path / "family_codes.i8",
        mode="r",
        dtype="i1",
        shape=(info["track_count"],),
    )
    
    z_bounds = np.memmap(
        local_tracks_path / "z_bounds.i32",
        mode="r",
        dtype="<i4",
        shape=(info["track_count"], 2),
    )
    
    arclengths = np.memmap(
        local_tracks_path / "arclengths.f64",
        mode="r",
        dtype="<f8",
        shape=(info["track_count"],),
    )
    
    tortuosities = np.memmap(
        local_tracks_path / "tortuosities.f64",
        mode="r",
        dtype="<f8",
        shape=(info["track_count"],),
    )
    
    return TrackCollection(
        coordinates=coordinates,
        offsets=offsets,
        source_ids=source_ids,
        family_codes=track_types,
        z_bounds=z_bounds,
        arclengths=arclengths,
        tortuosities=tortuosities,
        crossing_graph=crossing_graph,
        crossing_positions=crossing_positions,
    )


if __name__ == "__main__":
    local_tracks_path = Path(
        "/home/sean/Desktop/spiral_dataset/to_hf/tracks/"
        "2um_ds2_ps256_surf_v2.dbm.vctracks"
    )
    ct_vol_path = (
        "/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr"
    )
    # To use the public CT instead of the local Zarr above, comment out the
    # local path and uncomment this assignment:
    # ct_vol_path = (
    #     "s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
    #     "20260411134726-2.400um-0.2m-78keV-masked.zarr"
    # )
    crossing_min_arclength = 500.0
    crossing_min_shared_z_extent = 500.0
    crossing_min_spacing = 20.0
    grid_spacing = 20.0
    track_voxel_size_um = 4.0
    tifxyz_output_directory = Path(
        "/home/sean/Documents/volpkgs/s1_ds2.volpkg/traces"
    )
    using_s3_ct = ct_vol_path.startswith("s3://")

    tracks = load_tracks(local_tracks_path)
    longest_track = tracks[int(np.nanargmax(tracks.arclengths))]
    crossing_tracks = longest_track.spaced_crossing_tracks(
        min_arclength=crossing_min_arclength,
        min_spacing=crossing_min_spacing,
        min_shared_z_extent=crossing_min_shared_z_extent,
    )
    shared_crossing_z_extent, crossing_points_zyx = (
        crop_tracks_to_shared_z_extent(crossing_tracks)
    )
    interpolated_grid_zyx = tracks_to_2d_grid(
        crossing_points_zyx,
        row_spacing=grid_spacing,
        column_spacing=grid_spacing,
    )
    tifxyz_output_path = write_2d_grid_as_tifxyz(
        interpolated_grid_zyx,
        tifxyz_output_directory,
        spacing=grid_spacing,
        voxel_size_um=track_voxel_size_um,
    )

    if using_s3_ct:
        track_scale = 2
        ct_level = 4
        full_points_zyx = scale_track(
            longest_track.points_zyx,
            source_scale=track_scale,
            target_scale=ct_level,
        )
        interpolated_points_zyx = scale_track(
            interpolated_grid_zyx.reshape(-1, 3),
            source_scale=track_scale,
            target_scale=ct_level,
        )
        ct_store = fsspec.get_mapper(ct_vol_path, anon=True)
        ct_group = zarr.open_group(store=ct_store, mode="r")
        ct_data = ct_group[str(ct_level)]

        xy_padding = 25
        y_min = max(
            0,
            int(np.floor(full_points_zyx[:, 1].min())) - xy_padding,
        )
        y_max = min(
            ct_data.shape[1],
            int(np.ceil(full_points_zyx[:, 1].max())) + xy_padding + 1,
        )
        x_min = max(
            0,
            int(np.floor(full_points_zyx[:, 2].min())) - xy_padding,
        )
        x_max = min(
            ct_data.shape[2],
            int(np.ceil(full_points_zyx[:, 2].max())) + xy_padding + 1,
        )
        ct_image = da.from_zarr(ct_data)[
            :,
            y_min:y_max,
            x_min:x_max,
        ]
        ct_image_name = f"CT level {ct_level}"
        ct_image_kwargs = {"translate": (0, y_min, x_min)}
        ct_description = (
            f"CT level={ct_level}, "
            f"crop=y[{y_min}:{y_max}], x[{x_min}:{x_max}]"
        )
    else:
        full_points_zyx = np.asarray(longest_track.points_zyx)
        interpolated_points_zyx = interpolated_grid_zyx.reshape(-1, 3)
        ct_group = zarr.open_group(store=ct_vol_path, mode="r")
        datasets = ct_group.attrs["multiscales"][0]["datasets"]
        ct_image = [
            da.from_zarr(ct_group[dataset["path"]])
            for dataset in datasets
        ]
        ct_image_name = "CT multiscale"
        ct_image_kwargs = {"multiscale": True}
        ct_description = f"CT multiscale ({len(ct_image)} levels)"
    points_zyx = full_points_zyx[::10]

    viewer = napari.Viewer(ndisplay=2)
    viewer.add_image(
        ct_image,
        name=ct_image_name,
        colormap="gray",
        **ct_image_kwargs,
    )
    viewer.add_points(
        points_zyx,
        name=f"longest track {longest_track.row}",
        face_color="yellow",
        size=4,
        out_of_slice_display=True,
    )
    viewer.add_points(
        interpolated_points_zyx,
        name="Catmull-Rom interpolated grid",
        face_color="magenta",
        size=3,
        out_of_slice_display=True,
    )

    z_slice = float(np.median(points_zyx[:, 0]))
    viewer.dims.set_point(0, z_slice)
    viewer.reset_view()

    print(
        f"Displaying track {longest_track.row}: "
        f"source_id={longest_track.source_id}, "
        f"arclength={longest_track.arclength:.3f}, "
        f"tortuosity={longest_track.tortuosity:.3f}, "
        f"selected_crossings={len(crossing_tracks)}, "
        f"shared_crossing_z_extent={shared_crossing_z_extent}, "
        f"grid_shape={interpolated_grid_zyx.shape}, "
        f"tifxyz={tifxyz_output_path}, "
        f"{ct_description}, z={z_slice:.2f}"
    )
    napari.run()
