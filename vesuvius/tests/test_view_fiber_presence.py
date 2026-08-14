import argparse
import json
from dataclasses import replace
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from vesuvius.scripts.view_fiber_presence import (
    AnchorCellGeometry,
    AnchorStageGeometry,
    CropSelection,
    FiberReplayBundle,
    LineObjGeometry,
    ReplayVisualArtifacts,
    anchor_path_representatives,
    apply_replay_geometry_filter,
    build_parser,
    clipping_plane_in_layer_data,
    commit_with_rollback,
    common_shape_edge_width,
    crop_clipping_planes_in_base,
    crop_clipping_planes_in_layer_data,
    distance_visibility_mask,
    fiberlet_colormap_names,
    fiberlet_layer_features,
    fiberlet_quality_colormap_spec,
    filtered_replay_geometry_data,
    load_fiber_replay_bundle,
    make_replay_geometry_filter,
    mask_presence_by_distance,
    open_lazy_crop,
    parse_crop,
    polyline_union_distances_base,
    read_anchor_cell_obj,
    read_anchor_stage_json,
    read_line_obj,
    replace_replay_geometry_filter_sources,
    replay_display_radius_defaults_base,
    replay_distance_transform_base,
    replay_fiberlet_distances_base,
    replay_visual_topology,
    resolve_ome_zarr_level,
    select_base_crop,
    set_common_shape_edge_width,
    validate_anchor_stage_chain,
    validate_replay_reload_compatibility,
)


def test_anchor_stage_layers_are_enabled_by_default_with_explicit_opt_out():
    parser = build_parser()
    assert parser.parse_args(["presence.zarr", "--crop", "0,0,0,1,1,1"]).anchor_stages
    assert parser.parse_args(
        ["presence.zarr", "--crop", "0,0,0,1,1,1", "--anchor-stages"]
    ).anchor_stages
    assert not parser.parse_args(
        ["presence.zarr", "--crop", "0,0,0,1,1,1", "--no-anchor-stages"]
    ).anchor_stages


def test_replay_loader_enables_anchor_stages_by_default():
    assert load_fiber_replay_bundle.__kwdefaults__ == {"include_anchor_stages": True}


def _fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _write_visual_replay(tmp_path):
    generation = tmp_path / "runs" / "abc"
    replay_dir = generation / "replay"
    visual_dir = generation / "visualizations" / "000000"
    (visual_dir / "replay").mkdir(parents=True)
    (visual_dir / "anchors" / "stages").mkdir(parents=True)
    (visual_dir / "paths").mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    reference = b"# vc_fiber_replay_reference version 2\nv 0 0 0\nv 8 0 0\nl 1 2\n"
    greedy = b"# vc_greedy_fiber_replay version 2\ng segment_0\nv 0 0 0\nv 4 0 0\nl 1 2\n"
    fiberlet = b"# vc_fiberlet_graph_replay version 2\ng segment_0\nv 0 0 0\nv 6 0 0\nl 1 2\n"
    failure_obj = b"# vc_fiber_replay_failure version 2\nv 4 0 0\np 1\n"
    local_files = {
        "replay/reference.obj": reference,
        "replay/greedy.obj": greedy,
        "replay/fiberlet.obj": fiberlet,
        "replay/failure.obj": failure_obj,
        "anchors/anchors.json": b"{}\n",
        "anchors/anchors.obj": b"unused\n",
        "anchors/anchors_0.obj": b"unused\n",
        "anchors/anchors_1.obj": b"unused\n",
        "anchors/anchor_cells.obj": b"unused\n",
        "anchors/stages/initialized.json": b"{}\n",
        "anchors/stages/refined.json": b"{}\n",
        "anchors/stages/support.json": b"{}\n",
        "anchors/stages/selection.json": b"{}\n",
        "anchors/stages/nms.json": b"{}\n",
        "paths/fiberlets.json": b"{}\n",
        "paths/fiberlets.obj": b"unused\n",
        "paths/fiberlet_graph.json": b"{}\n",
    }
    for relative, content in local_files.items():
        target = visual_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    sources = {
        "fiber_manifest": "/fiber.json",
        "fiber_manifest_content_hash": "fnv1a64:1",
        "normal_manifest": "/normal.json",
        "normal_manifest_content_hash": "fnv1a64:2",
        "fiber_json": "/reference.json",
        "fiber_json_content_hash": "fnv1a64:3",
    }
    prediction_binding = {
        "mode": "canonical_stored_grid",
        "prediction_to_base_scale": 2.0,
        "prediction_shape_zyx": [4, 4, 4],
    }
    failure = {
        "index": 0,
        "segment_index": 0,
        "reason": "distance_above_threshold",
        "reference_arc_base": 4.0,
        "reference_arc_fraction": 0.5,
        "reference_point_base_xyz": [4, 0, 0],
        "evaluator_point_base_xyz": [4, 0, 0],
        "segment_point_index": 1,
        "candidate_index": None,
        "arc_index": None,
        "candidate_path_point_index": None,
        "error_base_voxels": 21.0,
        "error_ratio": 1.05,
    }
    local = {
        "format": "vc_fiber_replay_visualization",
        "version": 1,
        "identity": {"global_index": 0, "tracer": "greedy", "tracer_failure_index": 0},
        "coordinates": {
            "position_order": "XYZ",
            "position_space": "base_volume",
            "distance_unit": "base_voxels",
        },
        "sources": sources,
        "prediction_binding": prediction_binding,
        "failure": failure,
        "tube": {
            "begin_arc_base": 0.0,
            "end_arc_base": 8.0,
            "radius_base_voxels": 8.0,
            "reference_points_base_xyz": [[0, 0, 0], [8, 0, 0]],
            "cells_zyx": [[0, 0, 0]],
        },
        "volume_crop_base_xyzwhd": [0, 0, 0, 9, 2, 2],
        "reference_points_base_xyz": [[0, 0, 0], [8, 0, 0]],
        "greedy_trace_segments_base_xyz": [[[0, 0, 0], [4, 0, 0]]],
        "fiberlet_trace_segments_base_xyz": [[[0, 0, 0], [6, 0, 0]]],
        "artifacts": {
            key: {"path": key, "content_hash": _fnv1a64(content)}
            for key, content in local_files.items()
        },
    }
    manifest = json.dumps(local).encode()
    (visual_dir / "manifest.json").write_bytes(manifest)
    full_files = {
        "replay/reference.obj": reference,
        "replay/greedy.json": b"{}\n",
        "replay/greedy.obj": greedy,
        "replay/fiberlet.json": b"{}\n",
        "replay/fiberlet.obj": fiberlet,
    }
    for relative, content in full_files.items():
        target = generation / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    bundle = {
        "format": "vc_fiber_replay",
        "version": 2,
        "coordinates": {
            "position_order": "XYZ",
            "position_space": "base_volume",
            "distance_unit": "base_voxels",
        },
        "sources": sources,
        "bindings": {
            "trace": {
                "mode": "trace_options",
                "trace_to_base_scale": 2.0,
                "prediction_to_base_scale": 2.0,
                "prediction_spacing_trace_voxels": 1.0,
            },
            "prediction": prediction_binding,
        },
        "trace_config": {
            "requested": {"beam_width": 8, "beam_lookahead_steps": 2},
            "effective": {"beam_width": 1, "beam_lookahead_steps": 1},
        },
        "fiberlet_config": {},
        "reference_points_base_xyz": [[0, 0, 0], [8, 0, 0]],
        "greedy": {},
        "fiberlet": {},
        "failure_counts": {"greedy": 1, "fiberlet": 0},
        "visualizations": [{
            "global_index": 0,
            "tracer": "greedy",
            "tracer_failure_index": 0,
            "reference_arc_base": 4.0,
            "reference_arc_fraction": 0.5,
            "manifest": {
                "path": "runs/abc/visualizations/000000/manifest.json",
                "content_hash": _fnv1a64(manifest),
            },
        }],
        "artifacts": {
            key: {"path": f"runs/abc/{key}", "content_hash": _fnv1a64(content)}
            for key, content in full_files.items()
        },
    }
    path = tmp_path / "fiber_replay.json"
    path.write_text(json.dumps(bundle))
    return path


def test_loads_indexed_dual_replay_visualization(tmp_path):
    replay = load_fiber_replay_bundle(
        _write_visual_replay(tmp_path), 0, include_anchor_stages=False
    )

    assert replay.status == "distance_above_threshold"
    assert replay.crop_xyzwhd == (0, 0, 0, 9, 2, 2)
    assert replay.prediction_shape_zyx == (4, 4, 4)
    np.testing.assert_array_equal(replay.reference_zyx, [[0, 0, 0], [0, 0, 8]])
    np.testing.assert_array_equal(replay.greedy_segments_zyx[0], [[0, 0, 0], [0, 0, 4]])
    np.testing.assert_array_equal(replay.fiberlet_segments_zyx[0], [[0, 0, 0], [0, 0, 6]])
    np.testing.assert_array_equal(replay.failure_zyx, [[0, 0, 4]])
    assert replay.tube_radius_base_voxels == 8.0
    assert replay.fiber_manifest_content_hash == "fnv1a64:1"


def test_replay_rejects_missing_visualizations(tmp_path):
    path = _write_visual_replay(tmp_path)
    bundle = json.loads(path.read_text())
    bundle["visualizations"] = []
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="no visualizations"):
        load_fiber_replay_bundle(path, 0, include_anchor_stages=False)


def _reload_fixture():
    path = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    line = LineObjGeometry(
        paths_zyx=[path],
        total_groups=1,
        trace_loss_total=[],
        loss_per_prediction_voxel=[],
        relative_quality=[],
    )
    fiberlets = LineObjGeometry(
        paths_zyx=[path],
        total_groups=1,
        trace_loss_total=[2.0],
        loss_per_prediction_voxel=[1.0],
        relative_quality=[0.5],
    )
    stages = tuple(
        AnchorStageGeometry(
            stage=stage,
            paths_zyx=[path],
            features={"candidate_id": [0]},
            record_count=1,
            geometric_record_count=1,
            reasons={},
            binding={},
            records=(),
        )
        for stage in ("initialized", "refined", "support", "selection", "nms")
    )
    replay = FiberReplayBundle(
        path=Path("/tmp/fiber_replay.json"),
        visualization_index=0,
        tracer="greedy",
        tracer_failure_index=0,
        status="distance_above_threshold",
        crop_xyzwhd=(0, 0, 0, 8, 8, 8),
        prediction_shape_zyx=(4, 4, 4),
        prediction_to_base_scale=2.0,
        fiber_manifest_content_hash="fnv1a64:prediction",
        reference_zyx=path,
        greedy_segments_zyx=(path,),
        fiberlet_segments_zyx=(path,),
        failure_zyx=np.asarray([[0.0, 0.0, 2.0]]),
        tube_radius_base_voxels=8.0,
        anchors_obj=Path("anchors.obj"),
        anchor_cells_obj=Path("anchor_cells.obj"),
        anchor_stages=stages,
        paths_obj=Path("fiberlets.obj"),
    )
    artifacts = ReplayVisualArtifacts(
        anchors=line,
        anchor_cells=AnchorCellGeometry(
            centers_zyx=np.asarray([[0.0, 0.0, 1.0]]),
            displacements_zyx=[path],
        ),
        anchor_stages=stages,
        fiberlets=fiberlets,
    )
    return replay, artifacts


def test_replay_reload_compatibility_allows_changed_positive_counts():
    replay, artifacts = _reload_fixture()
    extra_path = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    replacement_artifacts = replace(
        artifacts,
        anchors=replace(
            artifacts.anchors,
            paths_zyx=[*artifacts.anchors.paths_zyx, extra_path],
            total_groups=2,
        ),
    )

    validate_replay_reload_compatibility(
        replay, artifacts, replay, replacement_artifacts
    )
    assert replay_visual_topology(replay, artifacts) == replay_visual_topology(
        replay, replacement_artifacts
    )


def test_replay_reload_rejects_changed_graph_route_layer_topology():
    replay, artifacts = _reload_fixture()
    graph_replay = replace(
        replay,
        fiberlet_segments_zyx=(),
    )

    with pytest.raises(ValueError, match="visual layer topology"):
        validate_replay_reload_compatibility(replay, artifacts, graph_replay, artifacts)


def test_replay_reload_rejects_prediction_source_and_stage_topology_changes():
    replay, artifacts = _reload_fixture()
    with pytest.raises(ValueError, match="different fiber prediction source"):
        validate_replay_reload_compatibility(
            replay,
            artifacts,
            replace(replay, fiber_manifest_content_hash="fnv1a64:other"),
            artifacts,
        )
    validate_replay_reload_compatibility(
        replay,
        artifacts,
        replay,
        replace(
            artifacts,
            fiberlets=replace(artifacts.fiberlets, paths_zyx=[]),
        ),
    )
    with pytest.raises(ValueError, match="topology"):
        validate_replay_reload_compatibility(
            replay,
            artifacts,
            replay,
            replace(
                artifacts,
                anchor_stages=artifacts.anchor_stages[:-1],
            ),
        )


def test_fiberlet_reload_features_follow_changed_path_count():
    _, artifacts = _reload_fixture()
    repeated = replace(
        artifacts.fiberlets,
        paths_zyx=artifacts.fiberlets.paths_zyx * 2,
        trace_loss_total=[2.0, 3.0],
        loss_per_prediction_voxel=[1.0, 1.5],
        relative_quality=[0.5, 0.25],
    )

    features = fiberlet_layer_features(repeated)

    np.testing.assert_array_equal(features["trace_loss_total"], [2.0, 3.0])
    np.testing.assert_array_equal(features["relative_quality"], [0.5, 0.25])


def test_reload_commit_failure_rolls_state_back():
    state = ["old"]

    def commit():
        state[:] = ["partial"]
        raise ValueError("injected setter failure")

    def rollback():
        state[:] = ["old"]

    with pytest.raises(RuntimeError, match="was rolled back"):
        commit_with_rollback(commit, rollback)

    assert state == ["old"]


def test_reads_anchor_cell_centers_and_accepted_offsets(tmp_path):
    path = tmp_path / "anchor_cells.obj"
    path.write_text(
        "# vc_fiberlet_anchor_cells version 1\n"
        "g cell_0_0_0\n"
        "v 1 2 3\n"
        "p 1\n"
        "v 4 5 6\n"
        "l 1 2\n"
        "g cell_0_0_1\n"
        "v 7 8 9\n"
        "p 3\n"
    )

    geometry = read_anchor_cell_obj(path)

    np.testing.assert_array_equal(geometry.centers_zyx, [[3, 2, 1], [9, 8, 7]])
    assert len(geometry.displacements_zyx) == 1
    np.testing.assert_array_equal(geometry.displacements_zyx[0], [[3, 2, 1], [6, 5, 4]])


def _anchor_stage_root(stage, records):
    outcomes = {}
    reasons = {}
    for record in records:
        transition = record["transition"]
        outcomes[transition["outcome"]] = outcomes.get(transition["outcome"], 0) + 1
        if transition["reason"] is not None:
            reasons[transition["reason"]] = reasons.get(transition["reason"], 0) + 1
    return {
        "format": "vc_fiberlet_anchor_stage",
        "version": 1,
        "stage": stage,
        "source": {"manifest": "/fiber.json", "manifest_content_hash": "fnv1a64:1"},
        "coordinates": {
            "position_order": "XYZ",
            "cell_index_order": "ZYX",
            "position_space": "base_volume",
            "prediction_to_base_scale": 2.0,
            "prediction_shape_zyx": [4, 4, 4],
        },
        "selection": {"cells_zyx": [[0, 0, 0]]},
        "parameters": {
            "cell_size_prediction_voxels": 4,
            "gaussian_sigma_prediction_voxels": 2.0,
            "peak_sigma_prediction_voxels": 1.5,
            "peak_axial_sigma_prediction_voxels": 6.0,
            "peak_grid_step_prediction_voxels": 0.5,
            "peak_gradient_weight": 1.0,
            "peak_gradient_reliability_scale": 0.05,
            "gaussian_cutoff_sigmas": 3.0,
            "local_window_radius_prediction_voxels": 4.0,
            "axial_support_half_width_prediction_voxels": 6.0,
            "position_convergence_tolerance_prediction_voxels": 0.0001,
            "nms_maximum_angle_degrees": 10.0,
            "nms_transverse_radius_prediction_voxels": 2.0,
            "nms_longitudinal_radius_prediction_voxels": 1.0,
            "observation_presence_floor": 0.05,
            "minimum_aligned_support": 0.05,
            "merge_maximum_angle_degrees": 10.0,
            "merge_maximum_absolute_objective_loss": 0.01,
            "merge_maximum_relative_objective_loss": 0.05,
            "maximum_seed_count": 8,
            "maximum_iterations": 64,
            "convergence_tolerance": 1e-12,
        },
        "glyph_length_base_voxels": 8.0,
        "summary": {
            "record_count": len(records),
            "geometric_record_count": sum(
                record["geometry"] is not None for record in records
            ),
            "outcomes": outcomes,
            "reasons": reasons,
        },
        "records": records,
    }


def _anchor_record(candidate, outcome, reason=None, geometry=True, parents=None):
    return {
        "cell_zyx": [0, 0, 0],
        "candidate_id": candidate,
        "parent_ids": [] if parents is None else parents,
        "geometry": (
            {"position_base_xyz": [2.0, 4.0, 6.0], "axis_xyz": [1.0, 0.0, 0.0]}
            if geometry
            else None
        ),
        "metrics": {
            "assigned_observations": 4 if geometry else None,
            "objective_contribution": None,
            "aligned_support": 0.8 if geometry else None,
            "directional_coherence": 0.9 if geometry else None,
            "refinement_score": 0.8 if geometry else None,
            "refinement_iterations": 2 if geometry else None,
        },
        "transition": {
            "outcome": outcome,
            "reason": reason,
            "successor_id": None,
            "tested_value": None,
            "threshold": None,
            "suppressor": None,
        },
    }


def test_reads_and_validates_anchor_stage_chain(tmp_path):
    records = {
        "initialized": [
            _anchor_record(0, "continue", parents=[]),
            _anchor_record(1, "rejected", "empty", geometry=False, parents=[]),
        ],
        "refined": [_anchor_record(0, "continue", parents=[0])],
        "support": [_anchor_record(0, "continue", parents=[0])],
        "selection": [_anchor_record(0, "continue", parents=[0])],
        "nms": [_anchor_record(0, "final", parents=[0])],
    }
    stages = []
    for index, (stage, stage_records) in enumerate(records.items()):
        path = tmp_path / f"{stage}.json"
        root = _anchor_stage_root(stage, stage_records)
        if index == 0:
            del root["parameters"]
        elif index == 1:
            del root["parameters"]["peak_gradient_weight"]
            del root["parameters"]["peak_gradient_reliability_scale"]
        elif index == 2:
            root["parameters"]["future_parameter"] = {"opaque": True}
        elif index == 3:
            root["parameters"] = ["opaque", "producer", "metadata"]
        else:
            root["future_metadata"] = {"ignored": True}
        path.write_text(json.dumps(root))
        stages.append(read_anchor_stage_json(path, stage))
    final = LineObjGeometry(
        paths_zyx=[stages[-1].paths_zyx[0]],
        total_groups=1,
        trace_loss_total=[],
        loss_per_prediction_voxel=[],
        relative_quality=[],
    )

    validate_anchor_stage_chain(stages, final)

    assert stages[0].record_count == 2
    assert stages[0].geometric_record_count == 1
    assert stages[0].reasons == {"empty": 1}
    assert stages[-1].features["candidate_id"] == [0]
    np.testing.assert_array_equal(
        stages[-1].paths_zyx[0], [[6.0, 4.0, -2.0], [6.0, 4.0, 6.0]]
    )


def test_anchor_stage_rejects_extra_geometry(tmp_path):
    record = _anchor_record(0, "final", parents=[0])
    record["geometry"]["unknown"] = 1
    path = tmp_path / "nms.json"
    path.write_text(json.dumps(_anchor_stage_root("nms", [record])))

    with pytest.raises(ValueError, match="malformed anchor geometry"):
        read_anchor_stage_json(path, "nms")


def test_anchor_stage_cell_identity_is_not_bounded_by_extractor_parameters(tmp_path):
    record = _anchor_record(0, "final", parents=[0])
    record["cell_zyx"] = [100, 50, 25]
    root = _anchor_stage_root("nms", [record])
    root["selection"]["cells_zyx"] = [[100, 50, 25]]
    root["parameters"]["cell_size_prediction_voxels"] = 1_000_000
    path = tmp_path / "nms.json"
    path.write_text(json.dumps(root))

    stage = read_anchor_stage_json(path, "nms")

    assert stage.record_count == 1
    assert stage.features["cell_zyx"] == ["100_50_25"]


def test_anchor_stage_chain_rejects_mixed_bindings(tmp_path):
    stages = []
    for index, stage in enumerate(
        ("initialized", "refined", "support", "selection", "nms")
    ):
        if stage == "initialized":
            stage_records = [
                _anchor_record(0, "continue", parents=[]),
                _anchor_record(1, "rejected", "empty", geometry=False, parents=[]),
            ]
        else:
            outcome = "final" if stage == "nms" else "continue"
            stage_records = [_anchor_record(0, outcome, parents=[0])]
        root = _anchor_stage_root(stage, stage_records)
        if index == 2:
            root["source"]["manifest_content_hash"] = "fnv1a64:mixed"
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps(root))
        stages.append(read_anchor_stage_json(path, stage))
    final = LineObjGeometry(
        paths_zyx=[stages[-1].paths_zyx[0]],
        total_groups=1,
        trace_loss_total=[],
        loss_per_prediction_voxel=[],
        relative_quality=[],
    )

    with pytest.raises(ValueError, match="mixed bindings"):
        validate_anchor_stage_chain(stages, final)


def test_replay_distance_mask_uses_reference_and_trace_in_base_voxels():
    selection = CropSelection(
        requested_base_xyzwhd=(0, 0, 0, 10, 10, 10),
        slices_zyx=(slice(0, 5), slice(0, 5), slice(0, 5)),
        origin_base_zyx=(0.0, 0.0, 0.0),
        shape_zyx=(5, 5, 5),
    )
    distance = replay_distance_transform_base(
        np.asarray([[0.0, 4.0, 0.0], [0.0, 4.0, 8.0]]),
        np.asarray([[0.0, 8.0, 0.0], [0.0, 8.0, 8.0]]),
        selection,
        (2.0, 2.0, 2.0),
    )

    np.testing.assert_array_equal(distance[0, 2], np.zeros(5))
    np.testing.assert_array_equal(distance[0, 4], np.zeros(5))
    assert distance[0, 3, 2] == pytest.approx(2.0)
    presence = da.from_array(np.ones((5, 5, 5), dtype=np.float32), chunks=(2, 2, 2))
    distance_data = da.from_array(distance, chunks=presence.chunks)
    masked = mask_presence_by_distance(presence, distance_data, 1.0).compute()
    np.testing.assert_array_equal(masked[0, 2], np.ones(5))
    np.testing.assert_array_equal(masked[0, 4], np.ones(5))
    assert masked[0, 3, 2] == 0.0


def test_exact_anchor_distance_uses_polyline_interiors_and_union():
    reference = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    trace = np.asarray([[0.0, 4.0, 0.0], [0.0, 4.0, 10.0]])
    points = np.asarray(
        [
            [0.0, 1.0, 5.0],
            [0.0, 3.0, 5.0],
            [0.0, 0.0, -2.0],
            [3.0, 0.0, 10.0],
        ]
    )

    distances = polyline_union_distances_base(points, (reference, trace))

    np.testing.assert_allclose(distances, [1.0, 1.0, 2.0, 3.0])


def test_exact_anchor_distance_rejects_missing_or_nonfinite_geometry():
    with pytest.raises(ValueError, match="at least one polyline"):
        polyline_union_distances_base(np.zeros((1, 3)), ())
    with pytest.raises(ValueError, match="non-finite"):
        polyline_union_distances_base(
            np.asarray([[0.0, np.nan, 0.0]]),
            (np.zeros((1, 3)),),
        )


def test_exact_fiberlet_distance_uses_segment_interiors_and_degenerate_paths():
    replay, artifacts = _reload_fixture()
    fiberlets = replace(
        artifacts.fiberlets,
        paths_zyx=[
            np.asarray([[0.0, -2.0, 1.0], [0.0, 2.0, 1.0]]),
            np.asarray([[0.0, 3.0, 0.0], [0.0, 3.0, 2.0]]),
            np.asarray([[0.0, 1.0, 1.0]]),
        ],
        trace_loss_total=[1.0, 2.0, 3.0],
        loss_per_prediction_voxel=[0.1, 0.2, 0.3],
        relative_quality=[0.9, 0.8, 0.7],
    )

    distances = replay_fiberlet_distances_base(replay, fiberlets)

    np.testing.assert_allclose(distances, [0.0, 3.0, 1.0])
    np.testing.assert_array_equal(
        distance_visibility_mask(distances, 1.0), [True, False, True]
    )
    assert replay_fiberlet_distances_base(
        replay, replace(fiberlets, paths_zyx=[])
    ).shape == (0,)


def test_replay_display_radius_defaults_are_independent_base_voxel_values():
    defaults = replay_display_radius_defaults_base()
    assert defaults == {"presence": 32.0, "anchors": 32.0, "fiberlets": 16.0}
    defaults["fiberlets"] = 99.0
    assert replay_display_radius_defaults_base()["fiberlets"] == 16.0


def test_anchor_representatives_use_glyph_midpoint_and_offset_target():
    paths = [
        np.asarray([[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]]),
        np.asarray([[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]),
    ]

    np.testing.assert_array_equal(
        anchor_path_representatives(paths),
        [[4.0, 6.0, 8.0], [5.0, 6.0, 7.0]],
    )
    np.testing.assert_array_equal(
        anchor_path_representatives(paths, target_endpoint=True),
        [[6.0, 8.0, 10.0], [9.0, 10.0, 11.0]],
    )


def test_anchor_distance_visibility_mask_is_inclusive_and_supports_infinity():
    np.testing.assert_array_equal(
        distance_visibility_mask(np.asarray([0.0, 1.0, 2.0, np.inf]), 1.0),
        [True, True, False, False],
    )


@pytest.mark.parametrize("radius", [-1.0, np.inf, np.nan])
def test_anchor_distance_filter_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="replay geometry radius"):
        distance_visibility_mask(np.zeros(1), radius)


class _FakeAnchorLayer:
    def __init__(self, *, points=False):
        self.data = np.empty((0, 3)) if points else []
        self.features = {}
        self.selected_data = {1}
        if points:
            self.size = np.asarray([3.5])
            self.face_color = "old"
        else:
            self.edge_width = np.asarray([2.5])
            self.edge_color = "old"


def test_anchor_visual_filter_physically_subsets_shapes_and_features():
    paths = [
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        np.asarray([[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]]),
    ]
    features = {
        "candidate": [10, 11, 12],
        "detail": [{"value": 1}, {"value": 2}, {"value": 3}],
    }
    layer = _FakeAnchorLayer()
    visual_filter = make_replay_geometry_filter(
        key="stage:refined",
        layer=layer,
        source_data=paths,
        source_features=features,
        distances_base_voxels=np.asarray([0.0, 1.0, 2.0]),
        color_attribute="edge_color",
        color_value="magenta",
    )
    paths[0][0, 0] = 99.0
    features["detail"][0]["value"] = 99

    data, selected_features = filtered_replay_geometry_data(visual_filter, 1.0)
    assert len(data) == 2
    assert selected_features == {
        "candidate": [10, 11],
        "detail": [{"value": 1}, {"value": 2}],
    }
    assert data[0][0, 0] == 0.0

    apply_replay_geometry_filter(visual_filter, 1.0)
    assert len(layer.data) == 2
    assert layer.features == selected_features
    assert layer.edge_color == "magenta"
    np.testing.assert_array_equal(layer.edge_width, 2.5)
    assert layer.selected_data == set()

    layer.data[0][0, 0] = 88.0
    layer.edge_width = []
    apply_replay_geometry_filter(visual_filter, 100.0)
    assert len(layer.data) == 3
    assert layer.data[0][0, 0] == 0.0
    assert layer.features["candidate"] == [10, 11, 12]
    assert layer.edge_width == 2.5


def test_anchor_visual_filter_physically_removes_all_point_geometry():
    layer = _FakeAnchorLayer(points=True)
    visual_filter = make_replay_geometry_filter(
        key="cell_centers",
        layer=layer,
        source_data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        distances_base_voxels=np.asarray([1.0, 2.0]),
        color_attribute="face_color",
        color_value="yellow",
    )

    apply_replay_geometry_filter(visual_filter, 0.0)
    assert layer.data.shape == (0, 3)
    assert layer.face_color == "yellow"
    np.testing.assert_array_equal(layer.size, 3.5)

    layer.size = []
    apply_replay_geometry_filter(visual_filter, 1.0)
    np.testing.assert_array_equal(layer.data, [[1.0, 2.0, 3.0]])
    assert layer.size == 3.5


def test_anchor_visual_filter_rejects_misaligned_distances_and_features():
    layer = _FakeAnchorLayer()
    paths = [np.zeros((2, 3)), np.ones((2, 3))]
    with pytest.raises(ValueError, match="distances do not match"):
        make_replay_geometry_filter(
            key="anchors",
            layer=layer,
            source_data=paths,
            distances_base_voxels=np.zeros(1),
            color_attribute="edge_color",
            color_value="cyan",
        )
    with pytest.raises(ValueError, match="feature 'candidate'"):
        make_replay_geometry_filter(
            key="anchors",
            layer=layer,
            source_data=paths,
            source_features={"candidate": [1]},
            distances_base_voxels=np.zeros(2),
            color_attribute="edge_color",
            color_value="cyan",
        )


def test_anchor_visual_filter_reload_handles_zero_counts_and_rollback():
    layer = _FakeAnchorLayer()
    original = make_replay_geometry_filter(
        key="anchors",
        layer=layer,
        source_data=[np.zeros((2, 3)), np.ones((2, 3))],
        distances_base_voxels=np.asarray([0.0, 2.0]),
        color_attribute="edge_color",
        color_value="cyan",
    )
    apply_replay_geometry_filter(original, 0.0)
    assert len(layer.data) == 1

    empty = replace_replay_geometry_filter_sources(
        [original], {"anchors": ([], None)}, {}
    )[0]
    apply_replay_geometry_filter(empty, 100.0)
    assert layer.data == []

    replacement = replace_replay_geometry_filter_sources(
        [original],
        {"anchors": ([np.full((2, 3), 4.0)], None)},
        {"anchors": np.asarray([0.0])},
    )[0]

    def commit():
        apply_replay_geometry_filter(replacement, 0.0)
        raise RuntimeError("injected")

    def rollback():
        apply_replay_geometry_filter(original, 0.0)

    with pytest.raises(RuntimeError, match="rolled back"):
        commit_with_rollback(commit, rollback)
    assert len(layer.data) == 1
    np.testing.assert_array_equal(layer.data[0], np.zeros((2, 3)))

    apply_replay_geometry_filter(replacement, 100.0)
    assert len(layer.data) == 1
    np.testing.assert_array_equal(layer.data[0], np.full((2, 3), 4.0))


def test_fiberlet_filter_keeps_features_and_width_across_empty_replacement():
    layer = _FakeAnchorLayer()
    paths = [np.zeros((2, 3)), np.ones((2, 3))]
    visual_filter = make_replay_geometry_filter(
        key="fiberlets",
        layer=layer,
        source_data=paths,
        source_features={"relative_quality": [0.25, 0.75]},
        distances_base_voxels=np.asarray([1.0, 2.0]),
        color_attribute="edge_color",
        color_value="relative_quality",
        empty_color_value="gray",
    )
    layer._vc_display_edge_width = 4.25
    layer.edge_width = []

    apply_replay_geometry_filter(visual_filter, 1.0)
    assert len(layer.data) == 1
    assert layer.features == {"relative_quality": [0.25]}
    assert layer.edge_color == "relative_quality"
    assert layer.edge_width == 4.25

    apply_replay_geometry_filter(visual_filter, 0.0)
    assert layer.data == []
    assert visual_filter.display_width == 4.25

    empty = replace_replay_geometry_filter_sources(
        [visual_filter],
        {"fiberlets": ([], {"relative_quality": []})},
        {},
    )[0]
    apply_replay_geometry_filter(empty, 16.0)
    assert layer.data == []
    assert layer.edge_color == "gray"
    assert layer.edge_width == 4.25

    restored = replace_replay_geometry_filter_sources(
        [empty],
        {
            "fiberlets": (
                [np.full((2, 3), 3.0)],
                {"relative_quality": [0.9]},
            )
        },
        {"fiberlets": np.asarray([15.0])},
    )[0]
    apply_replay_geometry_filter(restored, 16.0)
    assert len(layer.data) == 1
    assert layer.features == {"relative_quality": [0.9]}
    assert layer.edge_color == "relative_quality"
    assert layer.edge_width == 4.25


def test_replay_bundle_rejects_hash_mismatch(tmp_path):
    path = _write_visual_replay(tmp_path)
    (tmp_path / "runs" / "abc" / "replay" / "greedy.obj").write_text("changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_fiber_replay_bundle(path, 0, include_anchor_stages=False)


def test_replay_bundle_rejects_lexical_escape(tmp_path):
    path = _write_visual_replay(tmp_path)
    bundle = json.loads(path.read_text())
    bundle["artifacts"]["replay/greedy.obj"]["path"] = "../trace.obj"
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="escapes"):
        load_fiber_replay_bundle(path, 0, include_anchor_stages=False)


def test_replay_bundle_rejects_symlink_escape(tmp_path):
    path = _write_visual_replay(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.obj"
    outside.write_text("outside")
    symlink = tmp_path / "runs" / "abc" / "replay" / "escape.obj"
    symlink.symlink_to(outside)
    bundle = json.loads(path.read_text())
    bundle["artifacts"]["replay/greedy.obj"] = {
        "path": "runs/abc/replay/escape.obj",
        "content_hash": _fnv1a64(b"outside"),
    }
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="escapes"):
        load_fiber_replay_bundle(path, 0, include_anchor_stages=False)
    outside.unlink()


def test_fiberlet_quality_colormap_spec_is_red_yellow_green():
    name, colors, controls = fiberlet_quality_colormap_spec()

    assert name == "red-yellow-green"
    np.testing.assert_allclose(
        colors,
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_allclose(controls, [0.0, 0.5, 1.0])


def test_fiberlet_colormap_names_are_custom_first_sorted_and_unique():
    assert fiberlet_colormap_names(
        ["viridis", "magma", "viridis", "red-yellow-green"]
    ) == ("red-yellow-green", "magma", "viridis")


@pytest.mark.parametrize(
    ("edge_width", "expected"),
    [([2.0, 2.0, 2.0], 2.0), (np.asarray([0.25]), 0.25), (1.5, 1.5)],
)
def test_reads_common_width_from_napari_per_shape_values(edge_width, expected):
    class Layer:
        pass

    layer = Layer()
    layer.edge_width = edge_width

    assert common_shape_edge_width(layer) == expected


def test_common_width_uses_default_for_missing_layer_or_empty_shapes():
    class Layer:
        def __init__(self):
            self.edge_width = []

    assert common_shape_edge_width(None) == 2.0
    assert common_shape_edge_width(Layer()) == 2.0


def test_sets_common_width_and_emits_napari_edge_width_event():
    class Event:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1

    class Events:
        def __init__(self):
            self.edge_width = Event()

    class Layer:
        def __init__(self):
            self.edge_width = [2.0, 2.0]
            self.events = Events()

    layer = Layer()
    set_common_shape_edge_width(layer, 0.25)

    assert layer.edge_width == 0.25
    assert layer.events.edge_width.calls == 1


def test_clipping_plane_is_transformed_from_base_to_layer_data():
    class Layer:
        @staticmethod
        def world_to_data(world):
            return (np.asarray(world) - [100.0, 200.0, 300.0]) / [8.0, 4.0, 2.0]

    plane = clipping_plane_in_layer_data(
        Layer(),
        position_base_zyx=(132.0, 220.0, 306.0),
        normal_base_zyx=(0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(plane["position"], [4.0, 5.0, 3.0])
    np.testing.assert_allclose(plane["normal"], [0.0, 1.0, 0.0])
    assert plane["enabled"] is True


def test_base_crop_planes_remain_in_world_coordinates_for_volume_clipping():
    planes = crop_clipping_planes_in_base(
        lower_base_zyx=(10_000.0, 20_000.0, 30_000.0),
        upper_base_zyx=(10_100.0, 20_200.0, 30_300.0),
    )

    assert len(planes) == 6
    np.testing.assert_allclose(planes[0]["position"], [10_000, 20_000, 30_000])
    np.testing.assert_allclose(planes[1]["position"], [10_100, 20_200, 30_300])
    np.testing.assert_allclose(planes[0]["normal"], [1, 0, 0])
    np.testing.assert_allclose(planes[1]["normal"], [-1, 0, 0])


def test_crop_clipping_planes_bound_all_six_sides_in_layer_data():
    class Layer:
        @staticmethod
        def world_to_data(world):
            return (np.asarray(world) - [100.0, 200.0, 300.0]) / [8.0, 4.0, 2.0]

    planes = crop_clipping_planes_in_layer_data(
        Layer(),
        lower_base_zyx=(108.0, 208.0, 304.0),
        upper_base_zyx=(132.0, 220.0, 312.0),
    )

    assert len(planes) == 6
    np.testing.assert_allclose(
        [plane["position"] for plane in planes],
        [
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
        ],
    )
    np.testing.assert_allclose(
        [plane["normal"] for plane in planes],
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )


def test_reads_and_crops_grouped_fiberlet_obj(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
# trace_quality_population successful_scored_fiberlets
# trace_loss_density_unit prediction_voxel
# trace_quality_formula inverse_min_max_low_loss_is_one
# trace_quality_count 2
# trace_loss_density_min 1
# trace_loss_density_max 3
g inside_explicit_edges
# trace_loss_total 4
# trace_loss_per_prediction_voxel 1
# trace_quality_relative 1
v 10 20 30
v 11 21 31
v 12 22 32
l 1 2
l 2 3
g outside_explicit_edges
# trace_loss_total 12
# trace_loss_per_prediction_voxel 3
# trace_quality_relative 0
v 100 200 300
v 101 201 301
v 102 202 302
l 4 5
l 5 6
"""
    )
    geometry = read_line_obj(obj, "paths", (0, 0, 0, 50, 50, 50))

    assert geometry.total_groups == 2
    assert len(geometry.paths_zyx) == 1
    assert geometry.trace_loss_total == [4.0]
    assert geometry.loss_per_prediction_voxel == [1.0]
    assert geometry.relative_quality == [1.0]
    np.testing.assert_array_equal(
        geometry.paths_zyx[0],
        np.asarray([[30, 20, 10], [31, 21, 11], [32, 22, 12]], dtype=np.float32),
    )


def test_rejects_obsolete_fiberlet_material_records(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
mtllib fiberlets.mtl
"""
    )

    with pytest.raises(ValueError, match="unsupported OBJ record 'mtllib'"):
        read_line_obj(obj, "paths", (0, 0, 0, 10, 10, 10))


def test_fiberlet_crop_keeps_geometry_and_metrics_aligned(tmp_path):
    groups = [
        ("outside_first", 100.0, 4.0, 0.0),
        ("inside_first", 10.0, 2.0, 2.0 / 3.0),
        ("outside_between", 200.0, 3.0, 1.0 / 3.0),
        ("partially_inside", -1.0, 1.0, 1.0),
    ]
    obj_lines = [
        "# vc_fiberlets version 1",
        "# trace_quality_population successful_scored_fiberlets",
        "# trace_loss_density_unit prediction_voxel",
        "# trace_quality_formula inverse_min_max_low_loss_is_one",
        "# trace_quality_count 4",
        "# trace_loss_density_min 1",
        "# trace_loss_density_max 4",
    ]
    vertex = 1
    for name, x, density, quality in groups:
        obj_lines.extend(
            [
                f"g {name}",
                f"# trace_loss_total {density * 2}",
                f"# trace_loss_per_prediction_voxel {density}",
                f"# trace_quality_relative {quality!r}",
                f"v {x} 1 1",
                f"v {x + 2} 1 1",
                f"l {vertex} {vertex + 1}",
            ]
        )
        vertex += 2
    (tmp_path / "fiberlets.obj").write_text("\n".join(obj_lines) + "\n")

    geometry = read_line_obj(tmp_path / "fiberlets.obj", "paths", (0, 0, 0, 50, 50, 50))

    assert geometry.total_groups == 4
    assert len(geometry.paths_zyx) == 2
    assert geometry.trace_loss_total == [4.0, 2.0]
    assert geometry.loss_per_prediction_voxel == [2.0, 1.0]
    np.testing.assert_allclose(geometry.relative_quality, [2.0 / 3.0, 1.0])


def test_rejects_disconnected_line_records(tmp_path):
    obj = tmp_path / "anchors.obj"
    obj.write_text(
        """# vc_fiberlet_anchors version 1
g broken
v 0 0 0
v 1 1 1
v 2 2 2
v 3 3 3
l 1 2
l 3 4
"""
    )

    with pytest.raises(ValueError, match="do not form one ordered path"):
        read_line_obj(obj, "anchors", (0, 0, 0, 10, 10, 10))


def test_anchor_geometry_has_no_fiberlet_metrics(tmp_path):
    obj = tmp_path / "anchors.obj"
    obj.write_text(
        """# vc_fiberlet_anchors version 1
g anchor
v 0 0 0
v 1 0 0
l 1 2
"""
    )
    geometry = read_line_obj(obj, "anchors", (0, 0, 0, 10, 10, 10))
    assert geometry.trace_loss_total == []
    assert geometry.loss_per_prediction_voxel == []
    assert geometry.relative_quality == []


def test_empty_fiberlet_geometry_has_no_metrics(tmp_path):
    obj = tmp_path / "fiberlets.obj"
    obj.write_text(
        """# vc_fiberlets version 1
# trace_quality_population successful_scored_fiberlets
# trace_loss_density_unit prediction_voxel
# trace_quality_formula inverse_min_max_low_loss_is_one
# trace_quality_count 0
# trace_loss_density_min none
# trace_loss_density_max none
"""
    )
    geometry = read_line_obj(obj, "paths", (0, 0, 0, 10, 10, 10))

    assert geometry.paths_zyx == []
    assert geometry.trace_loss_total == []
    assert geometry.loss_per_prediction_voxel == []
    assert geometry.relative_quality == []


def make_presence_pyramid(tmp_path):
    root = tmp_path / "presence.ome.zarr"
    root.mkdir()
    (root / ".zgroup").write_text('{"zarr_format": 2}')
    (root / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "3",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [8.0, 8.0, 8.0]}
                                ],
                            },
                            {
                                "path": "4",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [16.0, 16.0, 16.0]}
                                ],
                            },
                        ],
                    }
                ]
            }
        )
    )
    level3 = zarr.open_array(
        root / "3", mode="w", shape=(5, 6, 7), chunks=(2, 3, 4), dtype="u1"
    )
    level3[:] = np.arange(level3.size, dtype=np.uint8).reshape(level3.shape)
    zarr.open_array(root / "4", mode="w", shape=(3, 3, 4), chunks=(2, 2, 2), dtype="u1")
    return root


def test_resolves_finest_level_and_direct_array(tmp_path):
    root = make_presence_pyramid(tmp_path)

    finest = resolve_ome_zarr_level(root)
    direct = resolve_ome_zarr_level(root / "4")

    assert finest.path == "3"
    assert finest.scale_zyx == (8.0, 8.0, 8.0)
    assert direct.path == "4"
    assert direct.scale_zyx == (16.0, 16.0, 16.0)


def test_base_crop_is_clipped_and_keeps_world_origin(tmp_path):
    root = make_presence_pyramid(tmp_path)
    level = resolve_ome_zarr_level(root)

    selection = select_base_crop((5, 6, 7), level, (9, 7, 1, 40, 30, 50))

    assert selection.slices_zyx == (slice(1, 5), slice(1, 5), slice(2, 7))
    assert selection.shape_zyx == (4, 4, 5)
    assert selection.origin_base_zyx == (8.0, 8.0, 16.0)


def test_lazy_crop_does_not_require_napari(tmp_path):
    root = make_presence_pyramid(tmp_path)
    level = resolve_ome_zarr_level(root, "3")

    data, selection = open_lazy_crop(level, (8, 8, 8, 16, 16, 16))

    assert data.shape == (2, 2, 2)
    assert selection.origin_base_zyx == (8.0, 8.0, 8.0)
    np.testing.assert_array_equal(
        data.compute(), zarr.open_array(root / "3", mode="r")[1:3, 1:3, 1:3]
    )


@pytest.mark.parametrize(
    "value",
    ["1,2,3,4,5", "1,2,3,0,5,6", "-1,2,3,4,5,6", "one,2,3,4,5,6"],
)
def test_parse_crop_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_crop(value)
