from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

spiral_graph = pytest.importorskip("spiral_graph")
export_component_module = pytest.importorskip("spiral_graph.export_component")


def theta_provider(zyx):
    points = np.asarray(zyx, dtype=np.float32)
    return np.mod(points[:, 2] * 0.1, 2 * np.pi).astype(np.float32)


theta_provider.geometric_theta = theta_provider


class KeyedTheta:
    def __init__(self, key):
        self.cache_key = key

    def __call__(self, zyx):
        return theta_provider(zyx)

    def geometric_theta(self, zyx):
        return theta_provider(zyx)


class BoundedTheta(KeyedTheta):
    z_begin = 0
    z_end = 20

    def __call__(self, zyx):
        points = np.asarray(zyx, dtype=np.float32)
        assert np.all(points[:, 0] >= self.z_begin)
        assert np.all(points[:, 0] < self.z_end)
        return theta_provider(points)


def write_patch(
    path: Path,
    *,
    patch_id="patch-a",
    x_offset=0.0,
    z=10,
    shape=(3, 3),
    mask=None,
) -> None:
    path.mkdir()
    rows, columns = np.mgrid[: shape[0], : shape[1]]
    tifffile.imwrite(
        path / "x.tif",
        (columns + x_offset).astype(np.float32),
        compression=None,
    )
    tifffile.imwrite(path / "y.tif", rows.astype(np.float32), compression=None)
    tifffile.imwrite(path / "z.tif", np.full(shape, z, np.float32), compression=None)
    if mask is not None:
        tifffile.imwrite(path / "mask.tif", np.asarray(mask, dtype=np.uint8))
    (path / "meta.json").write_text(
        json.dumps({"uuid": patch_id, "scale": [1.0, 1.0]}),
        encoding="utf-8",
    )


def write_pcl(path: Path, windings=(0, 1)) -> None:
    path.write_text(
        json.dumps(
            {
                "vc_pointcollections_json_version": "1",
                "collections": {
                    "1": {
                        "name": "test",
                        "points": {
                            "0": {"p": [0, 0, 10], "wind_a": windings[0]},
                            "1": {"p": [1, 0, 10], "wind_a": windings[1]},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_pcl_points(path: Path, points) -> None:
    path.write_text(
        json.dumps(
            {
                "vc_pointcollections_json_version": "1",
                "collections": {
                    "1": {
                        "name": "test",
                        "points": {
                            str(index): {"p": xyz, "wind_a": winding}
                            for index, (xyz, winding) in enumerate(points)
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_tracks(root: Path) -> tuple[Path, Path]:
    tracks = root / "tiny.vctracks"
    tracks.mkdir()
    (tracks / "header.bin").write_bytes(
        struct.pack("<8sIIQQ4Q", b"VCTRK01\0", 1, 64, 2, 4, 0, 0, 0, 0)
    )
    np.asarray(
        [[10, 0, 0], [10, 0, 1], [10, 1, 0], [10, 1, 1]], dtype="<i4"
    ).tofile(tracks / "coordinates.i32")
    np.asarray([0, 2, 4], dtype="<i8").tofile(tracks / "offsets.i64")
    np.asarray([100, 101], dtype="<u8").tofile(tracks / "source_ids.u64")
    np.asarray([0, 1], dtype="i1").tofile(tracks / "family_codes.i8")
    np.asarray([1, 1], dtype="<f8").tofile(tracks / "arclengths.f64")
    np.asarray([1, 1], dtype="<f8").tofile(tracks / "tortuosities.f64")
    np.asarray([[10, 10], [10, 10]], dtype="<i4").tofile(tracks / "z_bounds.i32")
    crossings = root / "tiny.crossings.npz"
    np.savez(
        crossings,
        source_ids=np.asarray([100, 101], dtype="<u8"),
        offsets=np.asarray([0, 1, 2], dtype="<i8"),
        partners=np.asarray([1, 0], dtype="<i4"),
        self_local=np.asarray([0, 0], dtype="<i4"),
        partner_local=np.asarray([0, 0], dtype="<i4"),
        positions=np.asarray([0, 0], dtype="<f8"),
        clearances=np.asarray([0, 0], dtype="<f8"),
    )
    return tracks, crossings


def write_seam_tracks(root: Path) -> tuple[Path, Path]:
    tracks = root / "seam.vctracks"
    tracks.mkdir()
    (tracks / "header.bin").write_bytes(
        struct.pack("<8sIIQQ4Q", b"VCTRK01\0", 1, 64, 2, 3, 0, 0, 0, 0)
    )
    np.asarray([[10, 0, 0], [10, 0, 1], [10, 0, 10]], dtype="<i4").tofile(
        tracks / "coordinates.i32"
    )
    np.asarray([0, 2, 3], dtype="<i8").tofile(tracks / "offsets.i64")
    np.asarray([100, 101], dtype="<u8").tofile(tracks / "source_ids.u64")
    np.asarray([0, 1], dtype="i1").tofile(tracks / "family_codes.i8")
    np.asarray([1, 0], dtype="<f8").tofile(tracks / "arclengths.f64")
    np.asarray([1, 1], dtype="<f8").tofile(tracks / "tortuosities.f64")
    np.asarray([[10, 10], [10, 10]], dtype="<i4").tofile(tracks / "z_bounds.i32")
    crossings = root / "seam.crossings.npz"
    np.savez(
        crossings,
        source_ids=np.asarray([100, 101], dtype="<u8"),
        offsets=np.asarray([0, 1, 2], dtype="<i8"),
        partners=np.asarray([1, 0], dtype="<i4"),
        self_local=np.asarray([1, 0], dtype="<i4"),
        partner_local=np.asarray([0, 1], dtype="<i4"),
        positions=np.asarray([1, 0], dtype="<f8"),
        clearances=np.asarray([0, 0], dtype="<f8"),
    )
    return tracks, crossings


def write_fiber(directory: Path) -> None:
    directory.mkdir()
    (directory / "fiber.json").write_text(
        json.dumps(
            {
                "type": "vc3d_fiber",
                "control_points": [
                    {"position": [0, 0, 40]},
                    {"position": [4, 0, 40]},
                ],
                "line_points": [[0, 0, 40], [4, 0, 40]],
                "hv_classification": {"manual_tag": "", "automatic_tag": "H"},
                "branches": [],
            }
        ),
        encoding="utf-8",
    )


def write_mixed_range_fiber(directory: Path) -> None:
    directory.mkdir()
    (directory / "mixed.json").write_text(
        json.dumps(
            {
                "type": "vc3d_fiber",
                "control_points": [
                    {"position": [0, 0, -4]},
                    {"position": [4, 0, 40]},
                ],
                "line_points": [
                    [0, 0, -4],
                    [1, 0, 0],
                    [2, 0, 40],
                    [3, 0, 80],
                    [4, 0, 84],
                ],
                "hv_classification": {"manual_tag": "", "automatic_tag": "H"},
                "branches": [],
            }
        ),
        encoding="utf-8",
    )


def write_reversed_endpoint_span_fiber(directory: Path) -> None:
    directory.mkdir()
    (directory / "reversed.json").write_text(
        json.dumps(
            {
                "type": "vc3d_fiber",
                # The middle control maps to a line tail and must not expand
                # the interval defined by the endpoint controls.
                "control_points": [[12, 0, 10], [100, 0, 10], [4, 0, 10]],
                "line_points": [
                    [0, 0, 10],
                    [4, 0, 10],
                    [8, 0, 10],
                    [12, 0, 10],
                    [100, 0, 10],
                ],
                "hv_classification": {"manual_tag": "", "automatic_tag": "H"},
                "branches": [],
            }
        ),
        encoding="utf-8",
    )


def test_lifted_holonomy_and_cache_round_trip(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    pcl = tmp_path / "relative.json"
    cache = tmp_path / "cache"
    write_patch(patch)
    write_pcl(pcl)

    graph = spiral_graph.WindingGraph.create(cache, theta_provider)
    added = graph.add_patches([patch])
    assert added.committed and added.nodes_added == 1
    assert graph.has_patch("patch-a")
    assert graph.node_name(graph.patch_node("patch-a")) == "patch-a"
    retained = graph.add_point_collections(
        [pcl], spiral_graph.InputRole.RELATIVE
    )
    assert retained.committed
    assert retained.holonomies_added == 1
    assert retained.conflict is None
    assert graph.stats().constraint_count == 1
    assert graph.stats().holonomy_count == 1
    lifted = graph.lifted_relative_winding("patch-a", "patch-a")
    assert lifted.representative == 0
    assert lifted.period == 1
    cycle = graph.holonomy(0)
    assert cycle.reported_holonomy == -1
    assert cycle.geometric_holonomy == 0
    assert cycle.inconsistency == -1
    audit = graph.holonomy_audits()[0]
    assert audit.reported_holonomy == -1
    assert audit.geometric_holonomy == 0
    assert audit.inconsistency == -1

    graph.save()
    reopened = spiral_graph.WindingGraph.open(cache)
    assert reopened.stats().patch_count == 1
    assert reopened.stats().constraint_count == 1
    assert reopened.stats().holonomy_count == 1
    assert reopened.lifted_relative_winding("patch-a", "patch-a").period == 1


def test_tracks_and_spatial_index(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    write_patch(patch)
    tracks, crossings = write_tracks(tmp_path)
    index = tmp_path / "tiny.winding-index"

    graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert graph.add_patches([patch]).committed
    prepared = graph.prepare_track_index(
        tracks, index, cell_size=4, memory_budget_bytes=64
    )
    assert prepared.points == 4
    assert not prepared.already_present
    result = graph.add_tracks(tracks, crossings, index)
    assert result.committed
    # Both crossing-connected tracks only touch this one patch, so all of
    # their equations reduce to validated same-patch redundancies.
    assert result.constraints_added == 0
    assert graph.stats().constraint_count == 0


def test_checkpoint_range_filters_patches_points_and_tracks(tmp_path):
    inside = tmp_path / "inside.tifxyz"
    outside = tmp_path / "outside.tifxyz"
    degenerate = tmp_path / "degenerate.tifxyz"
    write_patch(inside, patch_id="inside")
    write_patch(outside, patch_id="outside", z=30)
    write_patch(degenerate, patch_id="degenerate", shape=(1, 1))
    provider = BoundedTheta("bounded")
    graph = spiral_graph.WindingGraph(theta_provider=provider)

    added = graph.add_patches([inside, outside, degenerate])
    assert added.committed and added.nodes_added == 1
    assert graph.has_patch("inside")
    assert not graph.has_patch("outside")
    assert not graph.has_patch("degenerate")

    pcl = tmp_path / "mixed.json"
    write_pcl_points(pcl, [([0, 0, 10], 0), ([0, 0, 30], 0)])
    assert graph.add_point_collections(
        [pcl], spiral_graph.InputRole.SAME_WINDING
    ).committed

    tracks, crossings = write_tracks(tmp_path)
    coordinates = np.memmap(
        tracks / "coordinates.i32", dtype="<i4", mode="r+", shape=(4, 3)
    )
    coordinates[2:, 0] = 30
    coordinates.flush()
    np.asarray([[10, 10], [30, 30]], dtype="<i4").tofile(
        tracks / "z_bounds.i32"
    )
    index = tmp_path / "bounded.winding-index"
    graph.prepare_track_index(tracks, index, cell_size=4)
    assert graph.add_tracks(tracks, crossings, index).committed

def test_masked_patch_uses_declared_vertex_validity(tmp_path):
    patch = tmp_path / "masked.tifxyz"
    mask = np.ones((3, 3), dtype=np.uint8)
    mask[0, 0] = 0
    write_patch(patch, patch_id="masked", mask=mask)

    graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    added = graph.add_patches([patch])
    assert added.committed and added.nodes_added == 1
    assert graph.has_patch("masked")


def test_point_collection_roles_and_fiber_source_registration(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    absolute = tmp_path / "absolute.json"
    same = tmp_path / "same.json"
    fibers = tmp_path / "fibers"
    write_patch(patch)
    write_pcl(absolute, windings=(3, 3))
    write_pcl(same, windings=(0, 0))
    write_fiber(fibers)

    absolute_graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert absolute_graph.add_patches([patch]).committed
    anchored = absolute_graph.add_point_collections(
        [absolute], spiral_graph.InputRole.ABSOLUTE
    )
    assert anchored.committed
    assert anchored.anchors_added == 2
    assert absolute_graph.stats().anchored_component_count == 1

    same_graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert same_graph.add_patches([patch]).committed
    related = same_graph.add_point_collections(
        [same], spiral_graph.InputRole.SAME_WINDING
    )
    assert related.committed
    assert related.constraints_added == 1

    fiber_graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert fiber_graph.add_patches([patch]).committed
    fiber_result = fiber_graph.add_fibers(fibers)
    assert fiber_result.committed
    assert fiber_result.constraints_added == 0


def test_fiber_source_has_no_thinning_option_or_manifest_field(tmp_path):
    fibers = tmp_path / "fibers"
    write_reversed_endpoint_span_fiber(fibers)
    cache = tmp_path / "cache"
    graph = spiral_graph.WindingGraph.create(cache, theta_provider)

    with pytest.raises(TypeError, match="min_point_spacing"):
        graph.add_fibers(
            fibers, coordinate_scale=1.0, min_point_spacing=20.0
        )
    result = graph.add_fibers(fibers, coordinate_scale=1.0)
    assert result.committed
    assert result.constraints_added == 0
    graph.save()
    manifest = json.loads((cache / "manifest.json").read_text())
    assert "fiber_min_point_spacing" not in manifest["options"]
    assert "min_spacing" not in manifest["sources"][0]


def test_individual_fiber_can_be_invalidated(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    fibers = tmp_path / "fibers"
    write_patch(patch)
    write_fiber(fibers)
    graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert graph.add_patches([patch]).committed

    result = graph.add_fibers(fibers, invalid_fibers=["fiber.json"])

    assert result.committed
    assert result.constraints_added == 0
    assert graph.constraints() == []


def test_inspect_point_collections_is_read_only(tmp_path):
    patch_a = tmp_path / "a.tifxyz"
    patch_b = tmp_path / "b.tifxyz"
    relative = tmp_path / "holdout.json"
    write_patch(patch_a, patch_id="a")
    write_patch(patch_b, patch_id="b", x_offset=10)
    write_pcl_points(relative, [([0, 0, 10], 2), ([10, 0, 10], 5)])

    graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert graph.add_patches([patch_a, patch_b]).committed
    before = graph.stats()
    constraints = graph.inspect_point_collections(
        [relative], spiral_graph.InputRole.RELATIVE
    )

    assert len(constraints) == 1
    constraint = constraints[0]
    assert graph.node_name(constraint.from_node) == "a"
    assert graph.node_name(constraint.to_node) == "b"
    assert constraint.delta == 3
    after = graph.stats()
    assert after.constraint_count == before.constraint_count
    assert after.component_count == before.component_count
    assert graph.lifted_relative_winding("a", "b") is None


def test_overlapping_attachment_prefers_nearest_surface_over_larger_area(tmp_path):
    larger = tmp_path / "larger.tifxyz"
    nearer = tmp_path / "nearer.tifxyz"
    absolute = tmp_path / "absolute.json"
    write_patch(larger, patch_id="larger", z=11.5, shape=(5, 5))
    write_patch(nearer, patch_id="nearer", z=10.1, shape=(3, 3))
    write_pcl_points(absolute, [([0.5, 0.5, 10.0], 3)])
    options = spiral_graph.GraphOptions()
    options.contact_tolerance = 2.0
    graph = spiral_graph.WindingGraph(options=options, theta_provider=theta_provider)
    assert graph.add_patches([larger, nearer]).committed

    hits = graph.inspect_contacts(
        np.asarray([[10.0, 0.5, 0.5]], dtype=np.float32)
    )[0]
    assert [hit.patch_id for hit in hits] == ["nearer", "larger"]
    assert hits[0].distance < hits[1].distance

    result = graph.add_point_collections(
        [absolute], spiral_graph.InputRole.ABSOLUTE
    )

    assert result.committed
    constraint = graph.constraints()[0]
    assert graph.node_name(constraint.to_node) == "nearer"


def test_cached_patch_can_be_invalidated_before_sources(tmp_path):
    larger = tmp_path / "larger.tifxyz"
    nearer = tmp_path / "nearer.tifxyz"
    absolute = tmp_path / "absolute.json"
    cache = tmp_path / "cache"
    write_patch(larger, patch_id="larger", z=11.5, shape=(5, 5))
    write_patch(nearer, patch_id="nearer", z=10.1, shape=(3, 3))
    write_pcl_points(absolute, [([0.5, 0.5, 10.0], 3)])
    options = spiral_graph.GraphOptions()
    options.contact_tolerance = 2.0
    graph = spiral_graph.WindingGraph.create(cache, theta_provider, options)
    assert graph.add_patches([larger, nearer]).committed

    assert graph.set_patch_valid("nearer", False)
    assert not graph.patch_valid("nearer")
    assert graph.set_patch_valid("nearer", True)
    assert graph.patch_valid("nearer")
    assert graph.set_patch_valid("nearer", False)
    graph.save()

    reopened = spiral_graph.WindingGraph.open(cache, theta_provider)
    assert not reopened.patch_valid("nearer")
    result = reopened.add_point_collections(
        [absolute], spiral_graph.InputRole.ABSOLUTE
    )
    assert result.committed
    constraint = reopened.constraints()[0]
    assert reopened.node_name(constraint.to_node) == "larger"
    with pytest.raises(RuntimeError, match="before dependent"):
        reopened.set_patch_valid("nearer", True)


def test_theta_crossing_signs_match_fit_spiral_potentials(tmp_path):
    patch_a = tmp_path / "a.tifxyz"
    patch_b = tmp_path / "b.tifxyz"
    relative = tmp_path / "relative-seam.json"
    absolute = tmp_path / "absolute-seam.json"
    write_patch(patch_a, patch_id="a")
    write_patch(patch_b, patch_id="b", x_offset=10)
    write_pcl_points(
        relative,
        [([0.5, 0.5, 10], 4), ([10.5, 0.5, 10], 6)],
    )
    write_pcl_points(
        absolute,
        [([1.5, 0.5, 10], 3), ([10.5, 0.5, 10], 3)],
    )

    def seam_theta(zyx):
        xyz = np.asarray(zyx, dtype=np.float32)
        return np.where(xyz[:, 2] < 1, 6.1, 0.1).astype(np.float32)

    seam_theta.geometric_theta = seam_theta

    relative_graph = spiral_graph.WindingGraph(theta_provider=seam_theta)
    assert relative_graph.add_patches([patch_a, patch_b]).committed
    assert relative_graph.add_point_collections(
        [relative], spiral_graph.InputRole.RELATIVE
    ).committed
    # raw annotation delta +2, plus +1 physical winding while theta wraps
    # from 6.1 to 0.1.
    relative = relative_graph.lifted_relative_winding("a", "b")
    assert relative.representative == 3 and relative.period == 0

    absolute_graph = spiral_graph.WindingGraph(theta_provider=seam_theta)
    assert absolute_graph.add_patches([patch_a, patch_b]).committed
    assert absolute_graph.add_point_collections(
        [absolute], spiral_graph.InputRole.ABSOLUTE
    ).committed
    # Patch a's second cell has ThetaCrossingMap potential -1, so an
    # absolute point winding of 3 anchors its root at 2. Patch b anchors at 3.
    absolute = absolute_graph.lifted_relative_winding("a", "b")
    assert absolute.representative == 1 and absolute.period == 0

    tracks, crossings = write_seam_tracks(tmp_path)
    track_graph = spiral_graph.WindingGraph(theta_provider=seam_theta)
    assert track_graph.add_patches([patch_a, patch_b]).committed
    index = tmp_path / "seam.winding-index"
    track_graph.prepare_track_index(tracks, index, cell_size=4)
    assert track_graph.add_tracks(tracks, crossings, index).committed
    track = track_graph.lifted_relative_winding("a", "b")
    assert track.representative == 1 and track.period == 0


def test_late_patches_replay_sources_without_duplicate_constraints(tmp_path):
    patch_a = tmp_path / "a.tifxyz"
    patch_b = tmp_path / "b.tifxyz"
    patch_c = tmp_path / "c.tifxyz"
    relative = tmp_path / "late.json"
    write_patch(patch_a, patch_id="a")
    write_patch(patch_b, patch_id="b", x_offset=10)
    write_patch(patch_c, patch_id="c", x_offset=20)
    write_pcl_points(relative, [([0, 0, 10], 0), ([10, 0, 10], 1)])

    graph = spiral_graph.WindingGraph(theta_provider=theta_provider)
    assert graph.add_patches([patch_a]).committed
    source = graph.add_point_collections(
        [relative], spiral_graph.InputRole.RELATIVE
    )
    assert source.committed and source.constraints_added == 0

    late = graph.add_patches([patch_b])
    assert late.committed and late.constraints_added == 1
    assert graph.stats().constraint_count == 1

    unrelated = graph.add_patches([patch_c])
    assert unrelated.committed and unrelated.constraints_added == 0
    assert graph.stats().constraint_count == 1


def test_cache_rejects_a_different_theta_provider(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    cache = tmp_path / "cache"
    write_patch(patch)
    graph = spiral_graph.WindingGraph.create(cache, KeyedTheta("checkpoint-a"))
    assert graph.add_patches([patch]).committed
    graph.save()

    with pytest.raises(RuntimeError, match="theta provider does not match"):
        spiral_graph.WindingGraph.open(cache, KeyedTheta("checkpoint-b"))


def test_patch_only_v1_cache_is_augmented_with_geometric_theta(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    cache = tmp_path / "cache"
    write_patch(patch)
    graph = spiral_graph.WindingGraph.create(cache, theta_provider)
    assert graph.add_patches([patch]).committed
    graph.save()

    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (cache / "patch_geometric_theta.f32").unlink()
    graph_bytes = bytearray((cache / "graph.bin").read_bytes())
    struct.pack_into("<I", graph_bytes, 8, 1)
    (cache / "graph.bin").write_bytes(graph_bytes)

    migrated = spiral_graph.WindingGraph.open(cache, theta_provider)
    assert migrated.stats().patch_count == 1
    migrated.save()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["version"] == 2
    assert (cache / "patch_geometric_theta.f32").is_file()


def test_geometry_operations_require_independent_provider(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    write_patch(patch)

    def reported_only(zyx):
        return theta_provider(zyx)

    graph = spiral_graph.WindingGraph(theta_provider=reported_only)
    with pytest.raises(RuntimeError, match="must define geometric_theta"):
        graph.add_patches([patch])


def test_fiber_export_registration_recovers_reflected_pose(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    patch.mkdir()
    tifffile.imwrite(patch / "x.tif", np.zeros((40, 40), np.float32))
    source = np.stack(np.meshgrid(np.arange(4) * 4, np.arange(4) * 4), axis=-1).reshape(-1, 2)
    expected_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected_translation = np.array([30.0, -7.0])
    target = source @ expected_matrix.T + expected_translation
    correspondences = [
        (a.astype(np.float64), b.astype(np.float64), 3.2)
        for a, b in zip(source, target, strict=True)
    ]
    options = spiral_graph.LayoutOptions()
    options.min_inliers = 16
    fit, reason = export_component_module._fit_patch(
        "patch",
        correspondences,
        {"path": patch, "scale_row": 1.0, "scale_col": 1.0},
        options,
    )
    assert reason == "accepted"
    assert fit is not None and fit.inliers == 16
    assert fit.pose.reflected
    assert fit.rms < 1e-10
    assert fit.pose.matrix == pytest.approx(expected_matrix)
    assert fit.pose.translation == pytest.approx(expected_translation)


def test_fiber_export_bilinear_sampling_and_boundary_support():
    rows, columns = np.mgrid[:3, :3]
    xyz = np.stack((rows, np.zeros_like(rows), columns), axis=-1).astype(np.float32)
    turn = (rows + columns / 10).astype(np.float64)
    valid = np.ones((3, 3), bool)
    sampled, sampled_turn, supported = export_component_module._sample_patch(
        xyz,
        turn,
        valid,
        np.array([[0.5, 2.0]]),
        np.array([[1.5, 1.0]]),
    )
    assert supported.all()
    assert sampled[0, 0] == pytest.approx([0.5, 0.0, 1.5])
    assert sampled[0, 1] == pytest.approx([2.0, 0.0, 1.0])
    assert sampled_turn[0] == pytest.approx([0.65, 2.1])


def test_patch_winding_requires_one_consistent_inlier_consensus(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    write_patch(patch, shape=(5, 4))
    local = np.stack(
        np.meshgrid(np.arange(4), np.arange(5)), axis=-1
    ).reshape(-1, 2).astype(np.float64)
    local_turn = local[:, 0] * 0.1 / (2 * np.pi)
    fit = export_component_module._Fit(
        "patch-a",
        export_component_module._Pose(np.eye(2), np.zeros(2)),
        0.0,
        len(local),
        12.0,
        0,
        anchors=[
            (point, float(turn + 3))
            for point, turn in zip(local, local_turn, strict=True)
        ],
    )
    spec = {"path": patch, "scale_row": 1.0, "scale_col": 1.0}
    resolved, reason, _ = export_component_module._resolve_patch_winding(
        fit, theta_provider, spec, 16
    )
    assert reason == "accepted"
    assert resolved is fit and fit.winding_offset == 3

    fit.anchors[-1] = (fit.anchors[-1][0], fit.anchors[-1][1] + 1)
    resolved, reason, _ = export_component_module._resolve_patch_winding(
        fit, theta_provider, spec, 16
    )
    assert resolved is None
    assert reason == "overlap_winding_disagreement"


def test_patch_winding_excludes_disconnected_regions_without_consensus(tmp_path):
    patch = tmp_path / "patch.tifxyz"
    mask = np.zeros((5, 9), np.uint8)
    mask[:, :4] = 1
    mask[:, 5:] = 1
    write_patch(patch, shape=mask.shape, mask=mask)
    local = np.stack(
        np.meshgrid(np.arange(4), np.arange(5)), axis=-1
    ).reshape(-1, 2).astype(np.float64)
    local_turn = local[:, 0] * 0.1 / (2 * np.pi)
    fit = export_component_module._Fit(
        "patch-a",
        export_component_module._Pose(np.eye(2), np.zeros(2)),
        0.0,
        len(local),
        32.0,
        0,
        anchors=[
            (point, float(turn + 3))
            for point, turn in zip(local, local_turn, strict=True)
        ],
    )
    spec = {"path": patch, "scale_row": 1.0, "scale_col": 1.0}
    resolved, reason, field = export_component_module._resolve_patch_winding(
        fit, theta_provider, spec, 16
    )
    assert reason == "accepted"
    assert resolved is fit
    assert fit.region_winding_offsets == {0: 3}
    assert field is not None
    assert field[1][:, :4].all()
    assert not field[1][:, 5:].any()


def test_supported_vertex_mask_drops_vertices_without_a_supported_quad():
    occupied = np.zeros((4, 4), bool)
    occupied[:2, :2] = True
    occupied[3, 3] = True
    supported = np.zeros((3, 3), bool)
    supported[0, 0] = True
    output = export_component_module._supported_vertex_mask(
        occupied, supported
    )
    assert output[:2, :2].all()
    assert not output[3, 3]


def test_raster_quarantine_keeps_small_defects_and_rejects_dominant_conflicts():
    records = [
        {
            "patch_id": "small-defect",
            "overlap_samples": 100,
            "agreeing_overlap_samples": 90,
            "conflict_samples": 10,
        },
        {
            "patch_id": "bad-patch",
            "overlap_samples": 100,
            "agreeing_overlap_samples": 30,
            "conflict_samples": 70,
        },
        {
            "patch_id": "insufficient-overlap",
            "overlap_samples": 15,
            "agreeing_overlap_samples": 0,
            "conflict_samples": 15,
        },
    ]
    assert export_component_module._inconsistent_raster_patches(records, 16) == [
        "bad-patch"
    ]


def test_patch_growth_retains_contacts_across_frontier_rounds(monkeypatch):
    identity = export_component_module._Pose(np.eye(2), np.zeros(2))

    def fake_fit(patch_id, correspondences, spec, options):
        required = 16
        if len(correspondences) < required:
            return None, "too_few_contacts"
        return (
            export_component_module._Fit(
                patch_id,
                identity,
                0.0,
                len(correspondences),
                1.0,
                0,
                anchors=[
                    (source.copy(), turn)
                    for source, _, turn in correspondences
                ],
            ),
            "accepted",
        )

    def fake_resolve(fit, provider, spec, min_inliers):
        xyz = np.zeros((1, 1, 3), np.float32)
        return fit, "accepted", (xyz, np.ones((1, 1), bool), np.zeros((1, 1)))

    points = {
        "a": np.array([[0.0, 0.0, 0.0]], np.float32),
        "b": np.array([[2.0, 0.0, 0.0]], np.float32),
        "c": np.array([[1.0, 0.0, 0.0]], np.float32),
    }

    class Graph:
        def patch_layout(self, patch_id):
            return {"vertex_ij": [[0, 0]]}

        def inspect_contacts(self, zyx, tolerance):
            key = int(np.asarray(zyx)[0, 0])
            if key == 0:
                hits = [SimpleNamespace(patch_id="c", row=i, column=0) for i in range(16)]
                hits += [SimpleNamespace(patch_id="b", row=i, column=0) for i in range(8)]
            elif key == 1:
                hits = [SimpleNamespace(patch_id="b", row=i + 8, column=0) for i in range(8)]
            else:
                hits = []
            return [hits]

    monkeypatch.setattr(export_component_module, "_fit_patch", fake_fit)
    monkeypatch.setattr(export_component_module, "_resolve_patch_winding", fake_resolve)
    monkeypatch.setattr(
        export_component_module,
        "_patch_vertices",
        lambda graph, patch_id, spec: (np.zeros((1, 2)), points[patch_id]),
    )
    options = spiral_graph.LayoutOptions()
    options.min_inliers = 16
    options.workers = 1
    specs = {
        patch_id: {
            "valid": True,
            "scale_row": 1.0,
            "scale_col": 1.0,
        }
        for patch_id in points
    }
    seed = [(np.zeros(2), np.zeros(2), 0.0) for _ in range(16)]
    placed, rejected, relative = export_component_module._grow_patches(
        Graph(), specs, {"a": seed}, options, theta_provider
    )
    assert set(placed) == {"a", "b", "c"}
    assert "b" not in rejected
    assert relative


def test_fiber_component_export_cli_has_only_new_layout_controls():
    parser = export_component_module._parser()
    args = parser.parse_args(
        [
            "--cache", "/tmp/cache",
            "--checkpoint", "/tmp/checkpoint",
            "--output", "/tmp/output",
        ]
    )
    assert args.spacing == 20
    assert args.contact_tolerance == 2
    assert args.min_inliers == 16
    assert args.uv_ransac_tolerance == 3
    assert args.max_refit_rms == 2
    assert args.ransac_hypotheses == 512
    assert args.max_raster_samples == 100_000_000
    assert not hasattr(args, "seed")
    assert not hasattr(args, "turn_pixels")
    assert not hasattr(args, "z_spacing")
    assert not hasattr(args, "max_edge_factor")


def test_fiber_first_export_end_to_end(tmp_path):
    patch = tmp_path / "surface.tifxyz"
    patch.mkdir()
    rows, columns = np.mgrid[:31, :31]
    tifffile.imwrite(patch / "x.tif", columns.astype(np.float32), compression=None)
    tifffile.imwrite(patch / "y.tif", np.zeros((31, 31), np.float32), compression=None)
    tifffile.imwrite(patch / "z.tif", rows.astype(np.float32), compression=None)
    (patch / "meta.json").write_text(
        json.dumps({"uuid": "surface", "scale": [1.0, 1.0]})
    )
    fibers = tmp_path / "fibers"
    fibers.mkdir()
    h_line = [[x, 0, 10] for x in range(31)]
    v_line = [[15, 0, z] for z in range(31)]
    (fibers / "h.json").write_text(
        json.dumps(
            {
                "type": "vc3d_fiber",
                "control_points": [[0, 0, 10], [15, 0, 10], [30, 0, 10]],
                "line_points": h_line,
                "hv_classification": {"manual_tag": "H", "automatic_tag": "V"},
                "branches": [
                    {
                        "control_point_index": 1,
                        "branch_file": "v.json",
                        "branch_control_point_index": 1,
                    }
                ],
            }
        )
    )
    (fibers / "v.json").write_text(
        json.dumps(
            {
                "type": "vc3d_fiber",
                "control_points": [[15, 0, 0], [15, 0, 10], [15, 0, 30]],
                "line_points": v_line,
                "hv_classification": {"manual_tag": "", "automatic_tag": "V"},
                "branches": [
                    {
                        "control_point_index": 1,
                        "branch_file": "h.json",
                        "branch_control_point_index": 1,
                    }
                ],
            }
        )
    )
    cache = tmp_path / "cache"
    graph = spiral_graph.WindingGraph.create(cache, theta_provider)
    assert graph.add_patches([patch]).committed
    assert graph.add_fibers(fibers, coordinate_scale=1.0).committed
    graph.save()

    output = tmp_path / "output"
    metadata = export_component_module.export_component(
        cache,
        tmp_path / "unused-checkpoint.ckpt",
        output,
        spacing=5,
        min_inliers=16,
        theta_provider=theta_provider,
    )
    assert metadata["fiber_component"]["fiber_count"] == 2
    assert len(metadata["patches"]) == 1
    assert metadata["raster"]["valid_quads"] > 0
    for name in (
        "overview.png",
        "layout.json",
        "patch_index.tif",
        "winding.tif",
        "fractional_winding.tif",
    ):
        assert (output / name).is_file()
    z = tifffile.imread(output / "surface.tifxyz" / "z.tif")
    mask = tifffile.imread(output / "surface.tifxyz" / "mask.tif") != 0
    occupied_rows = np.nonzero(mask.any(axis=1))[0]
    assert np.mean(z[occupied_rows[0]][mask[occupied_rows[0]]]) > np.mean(
        z[occupied_rows[-1]][mask[occupied_rows[-1]]]
    )
    fractional = tifffile.imread(output / "fractional_winding.tif")
    assert np.all((fractional[mask] >= 0) & (fractional[mask] < 1))
