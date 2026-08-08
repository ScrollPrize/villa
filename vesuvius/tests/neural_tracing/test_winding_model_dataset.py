from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.winding_models import winding_model_dataset
from vesuvius.neural_tracing.winding_models.volume_plane_extractor import (
    VolumePlaneExtractor,
)
from vesuvius.neural_tracing.winding_models.winding_model_dataset import (
    WindingModelDataset,
)


def _extractor() -> VolumePlaneExtractor:
    return VolumePlaneExtractor([], shape=(5, 7), spacing=2.0)


def _fake_surface_dataset(monkeypatch) -> WindingModelDataset:
    row, col = np.mgrid[:11, :11]

    class FakeSurface:
        _x = col.astype(np.float32)
        _y = row.astype(np.float32)
        _z = np.zeros_like(_x)
        valid_quad_mask = np.ones((10, 10), dtype=bool)
        valid_vertex_mask = np.ones((11, 11), dtype=bool)

    monkeypatch.setattr(
        winding_model_dataset.tifxyz,
        "read_tifxyz",
        lambda _path: FakeSurface(),
    )
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.inner_fraction = 0.7
    dataset.segment_cache_dir = None
    return dataset


def test_segment_raycaster_is_trimmed_to_inner_uv_region(monkeypatch) -> None:
    dataset = _fake_surface_dataset(monkeypatch)
    dataset.hit_merge_tolerance = 1e-3

    segment = dataset._load_segment(Path("PHerc0139-w01_fake.tifxyz"))

    expected_cells = np.argwhere(
        np.pad(np.ones((7, 7), dtype=bool), ((1, 2), (1, 2)))
    )
    np.testing.assert_array_equal(segment.sample_cells, expected_cells)
    assert segment.raycaster.valid_quad_count == len(expected_cells)

    def crossings(x: float, y: float) -> list:
        return dataset._hits(
            segment, np.asarray([x, y, -1.0]), np.asarray([0.0, 0.0, 1.0])
        )

    inner = crossings(4.5, 4.5)
    assert len(inner) == 1
    t, xyz, turn = inner[0]
    assert t == 1.0
    np.testing.assert_allclose(xyz, [4.5, 4.5, 0.0])
    assert turn is None
    assert crossings(0.5, 0.5) == []
    assert crossings(8.5, 8.5) == []


def test_hits_merge_duplicate_crossings_within_tolerance(monkeypatch) -> None:
    dataset = _fake_surface_dataset(monkeypatch)
    dataset.hit_merge_tolerance = 1e-3

    segment = dataset._load_segment(Path("unlabeled_fake.tifxyz"))

    # A ray through a shared vertex intersects every adjacent triangle; the
    # duplicates must merge into a single crossing with finite winding turns.
    hits = dataset._hits(
        segment, np.asarray([5.0, 5.0, -1.0]), np.asarray([0.0, 0.0, 1.0])
    )
    assert len(hits) == 1
    t, xyz, turn = hits[0]
    assert t == 1.0
    np.testing.assert_allclose(xyz, [5.0, 5.0, 0.0])
    assert np.isfinite(turn)


def test_cached_segment_skips_derivation_and_matches_fresh_load(
    monkeypatch, tmp_path
) -> None:
    dataset = _fake_surface_dataset(monkeypatch)
    dataset.segment_cache_dir = tmp_path / "cache"
    segment_path = tmp_path / "unlabeled_fake.tifxyz"
    segment_path.mkdir()

    fresh = dataset._load_segment(segment_path)
    assert list(dataset.segment_cache_dir.glob("*.npz"))

    def fail(*_args, **_kwargs):
        raise AssertionError("derivation must not run on a cache hit")

    monkeypatch.setattr(WindingModelDataset, "_vertex_turns", fail)
    cached = dataset._load_segment(segment_path)

    np.testing.assert_array_equal(cached.xyz, fresh.xyz)
    np.testing.assert_array_equal(cached.sample_cells, fresh.sample_cells)
    np.testing.assert_allclose(cached.vertex_turns, fresh.vertex_turns)


def test_segment_cache_key_tracks_source_file_changes(monkeypatch, tmp_path) -> None:
    dataset = _fake_surface_dataset(monkeypatch)
    dataset.segment_cache_dir = tmp_path / "cache"
    segment_path = tmp_path / "PHerc0139-w01_fake.tifxyz"
    segment_path.mkdir()
    (segment_path / "x.tif").write_bytes(b"1")

    original = dataset._segment_cache_path(segment_path)
    (segment_path / "x.tif").write_bytes(b"22")

    assert dataset._segment_cache_path(segment_path) != original
    labeled = dataset._load_segment(segment_path)
    assert labeled.vertex_turns is None


def test_intersecting_planes_start_at_origin_and_share_the_ray_axis() -> None:
    extractor = _extractor()
    direction = np.asarray([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    ray_origin = np.asarray([10.0, 20.0, 30.0])

    origins, x_steps, y_steps = extractor.intersecting_geometry(direction, ray_origin)

    starts = origins + 2.0 * y_steps
    np.testing.assert_allclose(starts, np.stack((ray_origin, ray_origin)), atol=2e-6)
    np.testing.assert_allclose(x_steps[0], 2.0 * direction, atol=2e-6)
    np.testing.assert_allclose(x_steps[1], 2.0 * direction, atol=2e-6)
    np.testing.assert_allclose(y_steps @ direction, 0.0, atol=2e-6)
    np.testing.assert_allclose(np.dot(y_steps[0], y_steps[1]), 0.0, atol=2e-6)
    normals = np.cross(x_steps, y_steps)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    np.testing.assert_allclose(np.dot(normals[0], normals[1]), 0.0, atol=2e-6)
    ends = origins + 6.0 * x_steps + 2.0 * y_steps
    expected_end = ray_origin + 12.0 * direction
    np.testing.assert_allclose(ends, np.stack((expected_end, expected_end)), atol=2e-6)


def test_four_intersecting_planes_append_both_diagonal_crossings() -> None:
    extractor = VolumePlaneExtractor(
        [], shape=(5, 7), spacing=2.0, num_planes=4
    )
    direction = np.asarray([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    ray_origin = np.asarray([10.0, 20.0, 30.0])

    origins, x_steps, y_steps = extractor.intersecting_geometry(
        direction, ray_origin
    )

    assert origins.shape == x_steps.shape == y_steps.shape == (4, 3)
    starts = origins + 2.0 * y_steps
    np.testing.assert_allclose(starts, np.broadcast_to(ray_origin, (4, 3)), atol=2e-6)
    np.testing.assert_allclose(
        x_steps, np.broadcast_to(2.0 * direction, (4, 3)), atol=2e-6
    )
    np.testing.assert_allclose(np.linalg.norm(y_steps, axis=1), 2.0, atol=2e-6)
    np.testing.assert_allclose(y_steps @ direction, 0.0, atol=2e-6)
    np.testing.assert_allclose(
        y_steps[2], (y_steps[0] + y_steps[1]) / np.sqrt(2.0), atol=2e-6
    )
    np.testing.assert_allclose(
        y_steps[3], (y_steps[0] - y_steps[1]) / np.sqrt(2.0), atol=2e-6
    )


def test_plane_count_must_be_two_or_four() -> None:
    for num_planes in (1, 3, 5):
        with np.testing.assert_raises_regex(ValueError, "num_planes must be 2 or 4"):
            VolumePlaneExtractor([], num_planes=num_planes)


def test_longer_ray_grows_width_without_changing_isotropic_spacing() -> None:
    extractor = VolumePlaneExtractor([], shape=(5, 11), spacing=1.5)
    direction = np.asarray([0.0, 0.0, 1.0])
    ray_origin = np.asarray([4.0, 5.0, 6.0])

    origins, x_steps, y_steps = extractor.intersecting_geometry(direction, ray_origin)

    np.testing.assert_allclose(np.linalg.norm(x_steps, axis=1), 1.5)
    np.testing.assert_allclose(np.linalg.norm(y_steps, axis=1), 1.5)
    centerline_end = origins[0] + 10.0 * x_steps[0] + 2.0 * y_steps[0]
    np.testing.assert_allclose(centerline_end, ray_origin + 15.0 * direction)


def test_crossings_are_randomly_positioned_without_losing_ct_alignment(
    monkeypatch,
) -> None:
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.ray_length = 11
    dataset.plane_spacing = 1.0
    origin = np.asarray([0.0, 4.0, 5.0])
    direction = np.asarray([1.0, 0.0, 0.0])
    hits = [
        (2.0, np.asarray([2.0, 4.0, 5.0]), 0),
        (5.0, np.asarray([5.0, 4.0, 5.0]), 1),
    ]
    monkeypatch.setattr(torch, "rand", lambda _shape: torch.tensor(0.25))

    shifted_origin, shifted_hits = dataset._randomly_position_crossings(
        origin, direction, hits
    )

    # The crossings span 3 units, leaving 7 possible units for the first hit.
    assert shifted_hits[0][0] == 1.75
    assert shifted_hits[-1][0] == 4.75
    for t, xyz, _ in shifted_hits:
        np.testing.assert_allclose(shifted_origin + t * direction, xyz)


def test_winding_valid_mask_excludes_non_consecutive_crossing_gaps() -> None:
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.ray_length = 13
    dataset.plane_spacing = 0.5
    point = np.zeros(3)
    hits = [
        (1.0, point, 0),
        (2.0, point, 1),  # consecutive: stays valid
        (4.0, point, 3),  # skips winding 2: invalidate (2.0, 4.0)
        (5.5, point, 4),
    ]

    valid = dataset._winding_valid_mask(hits)

    # Samples sit at t = 0.0, 0.5, ..., 6.0; 2.5, 3.0, and 3.5 fall strictly
    # inside the non-consecutive gap, and 0.0, 0.5, and 6.0 fall outside the
    # sampled crossings.
    expected = np.ones(13, dtype=bool)
    expected[:2] = False
    expected[5:8] = False
    expected[12] = False
    np.testing.assert_array_equal(valid, expected)


def test_winding_valid_mask_keeps_samples_at_crossings() -> None:
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.ray_length = 5
    dataset.plane_spacing = 1.0
    point = np.zeros(3)

    valid = dataset._winding_valid_mask([(1.0, point, 0), (3.0, point, -2)])

    np.testing.assert_array_equal(valid, [False, True, False, True, False])


def _dataset_with_point_values(values: np.ndarray) -> WindingModelDataset:
    dataset = WindingModelDataset.__new__(WindingModelDataset)

    class FakeExtractor:
        def sample_points(self, _volume_idx, points_xyz):
            assert len(points_xyz) == len(values)
            return np.asarray(values)

    dataset.plane_extractor = FakeExtractor()
    return dataset


def test_filter_crossings_drops_windings_with_no_adjacent_neighbour() -> None:
    point = np.zeros(3)
    hits = [
        (1.0, point, 1),
        (3.0, point, 2),
        (5.0, point, 3),
        (17.0, point, 18),
    ]
    dataset = _dataset_with_point_values(np.full(4, 200, dtype=np.uint8))

    kept = dataset._filter_crossings(0, hits)

    assert [index for _, _, index in kept] == [1, 2, 3]


def test_filter_crossings_drops_zero_ct_and_the_crossings_it_strands() -> None:
    point = np.zeros(3)
    hits = [
        (1.0, point, 1),
        (3.0, point, 2),
        (5.0, point, 3),
    ]
    # The middle crossing sits on empty CT; without it the outer two are no
    # longer adjacent to anything and must go too.
    dataset = _dataset_with_point_values(np.asarray([200, 0, 200], dtype=np.uint8))

    assert dataset._filter_crossings(0, hits) == []


def test_filter_crossings_keeps_adjacent_runs_on_nonzero_ct() -> None:
    point = np.zeros(3)
    hits = [
        (1.0, point, -2),
        (3.0, point, -1),
        (9.0, point, 3),
        (11.0, point, 4),
    ]
    dataset = _dataset_with_point_values(np.asarray([50, 50, 0, 50], dtype=np.uint8))

    kept = dataset._filter_crossings(0, hits)

    assert [index for _, _, index in kept] == [-2, -1]


def test_sample_points_maps_through_segment_to_volume_transform() -> None:
    transform = np.eye(4)
    transform[:3, :3] *= 2.0
    transform[:3, 3] = [1.0, 2.0, 3.0]
    extractor = VolumePlaneExtractor(
        [Path("unused")],
        sampling="nearest",
        tile_size=16,
        segment_to_volume_xyz=[transform],
    )
    calls = []

    class FakeVolume:
        def sample_coords(self, coords, valid, **kwargs):
            calls.append((coords, valid, kwargs))
            return (
                np.arange(coords.shape[1], dtype=np.uint8).reshape(1, -1, 1),
                np.ones(coords.shape[:2] + (1,), dtype=np.uint8),
                {},
            )

    extractor._volume = lambda volume_idx: FakeVolume()
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    values = extractor.sample_points(0, points)

    assert len(calls) == 1
    coords, valid, kwargs = calls[0]
    assert coords.shape == (1, 2, 3)
    assert coords.dtype == np.float32
    np.testing.assert_allclose(coords[0], [[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    assert valid.all()
    assert kwargs == {"sampling": "nearest", "tile_size": 16}
    np.testing.assert_array_equal(values, [0, 1])


def _bare_segment(
    winding_idx: int | None = None, vertex_turns: np.ndarray | None = None
) -> winding_model_dataset.Segment:
    return winding_model_dataset.Segment(
        winding_idx=winding_idx,
        xyz=np.zeros((2, 2, 3), dtype=np.float32),
        raycaster=None,
        sample_cells=np.zeros((1, 2), dtype=np.int64),
        vertex_turns=vertex_turns,
    )


def test_multi_wrap_ray_drops_crossings_within_merge_distance() -> None:
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.ray_length = 21
    dataset.plane_spacing = 1.0
    dataset.crossing_merge_distance = 3.0
    point = np.zeros(3)
    segment = _bare_segment(vertex_turns=np.zeros((2, 2), dtype=np.float32))
    dataset._ray = lambda _segment: (point, point, np.asarray([1.0, 0.0, 0.0]))
    # The hit at t=6.5 sits 1.5 units from its neighbour: a duplicated wrap.
    dataset._hits = lambda _segment, _origin, _direction, _max_t: [
        (1.0, point, 0.0),
        (5.0, point, 1.0),
        (6.5, point, 2.0),
        (10.0, point, 2.0),
    ]

    _, _, ordered = dataset._multi_wrap_ray(segment)

    assert [(t, index) for t, _, index in ordered] == [(1.0, 0), (5.0, 1), (10.0, 2)]


def test_labelled_ray_drops_crossings_within_merge_distance() -> None:
    dataset = WindingModelDataset.__new__(WindingModelDataset)
    dataset.ray_length = 21
    dataset.plane_spacing = 1.0
    dataset.ray_origin_offset = 1.0
    dataset.min_winding_gap = 4
    dataset.crossing_merge_distance = 3.0
    point = np.zeros(3)
    segments = [_bare_segment(winding_idx=index) for index in range(5)]
    volume = winding_model_dataset.VolumeDataset(
        volume_path=Path("unused"),
        segments=segments,
        segment_to_volume_xyz=np.eye(4),
    )
    # Winding 2's crossing at t=7.5 duplicates winding 1's at t=6.0.
    hits_by_winding = {0: 1.0, 1: 6.0, 2: 7.5, 3: 12.0, 4: 16.0}
    dataset._ray = lambda _segment: (point, point, np.asarray([1.0, 0.0, 0.0]))
    dataset._hits = lambda segment, _origin, _direction, max_t: [
        (t, point, 0)
        for t in [hits_by_winding[segment.winding_idx]]
        if t <= max_t
    ]

    _, _, ordered = dataset._labelled_ray(volume, segments[0])

    assert [(t, index) for t, _, index in ordered] == [
        (1.0, 0),
        (6.0, 1),
        (12.0, 3),
        (16.0, 4),
    ]


def test_sample_intersecting_planes_uses_one_fused_native_call() -> None:
    extractor = VolumePlaneExtractor(
        [Path("unused")],
        shape=(5, 7),
        spacing=2.0,
        sampling="nearest",
        tile_size=16,
    )
    calls = []

    class FakeVolume:
        def sample_planes(self, *args, **kwargs):
            calls.append((args, kwargs))
            return (
                np.full((2, 5, 7), 13, dtype=np.uint8),
                np.ones((2, 5, 7), dtype=np.uint8),
                {},
            )

    extractor._volume = lambda volume_idx: FakeVolume()
    direction = np.asarray([0.0, 0.0, 1.0])
    ray_origin = np.asarray([4.0, 5.0, 6.0])

    images, valid, _ = extractor.extract(0, direction, ray_origin)

    assert len(calls) == 1
    assert calls[0][0][3] == (5, 7)
    assert calls[0][1] == {"sampling": "nearest", "tile_size": 16}
    assert images.dtype == np.uint8
    assert images.shape == (2, 5, 7)
    assert valid.dtype == np.uint8
    assert valid.all()


def test_scaled_volume_path_selects_array_from_multiscale_group(tmp_path) -> None:
    scale_array = tmp_path / "2"
    scale_array.mkdir()
    (scale_array / ".zarray").touch()

    assert VolumePlaneExtractor.scaled_volume_path(tmp_path, 2) == scale_array
    assert VolumePlaneExtractor.scaled_volume_path(tmp_path, 3) == tmp_path


def test_load_segment_to_volume_transform_scales_translation(tmp_path) -> None:
    (tmp_path / "transform.json").write_text(
        json.dumps(
            {
                "transformation_matrix": [
                    [2.0, 0.0, 0.0, 8.0],
                    [0.0, 2.0, 0.0, 12.0],
                    [0.0, 0.0, 2.0, 16.0],
                ]
            }
        )
    )

    transform = VolumePlaneExtractor.load_segment_to_volume_transform(
        tmp_path, 2, segment_downscale=4
    )

    fixed_scaled = np.asarray([10.0, 20.0, 30.0, 1.0])
    np.testing.assert_allclose(
        transform @ fixed_scaled,
        np.asarray([4.0, 8.5, 13.0, 1.0]),
    )


def test_same_volume_transform_uses_declared_downscales(tmp_path) -> None:
    same_scale = VolumePlaneExtractor.load_segment_to_volume_transform(
        tmp_path, 2, segment_downscale=4, use_registration=False
    )
    base_segments = VolumePlaneExtractor.load_segment_to_volume_transform(
        tmp_path, 2, segment_downscale=1, use_registration=False
    )

    np.testing.assert_array_equal(same_scale, np.eye(4))
    np.testing.assert_array_equal(
        np.diag(base_segments), np.asarray([0.25, 0.25, 0.25, 1.0])
    )
