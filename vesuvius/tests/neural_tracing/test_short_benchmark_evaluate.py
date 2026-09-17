import json
import pytest
from lasagna.inference_provenance import sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import bind_benchmark
from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import scroll_groups


def test_correction_preserves_edited_geometry_and_restores_identity():
    from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import corrected_annotation
    original = dict(evaluation_direction='forward', vc_open_data_coordinate_space='scroll/scan',
                    line_points=[[0, 0, 0], [1, 0, 0]], control_points=[[0, 0, 0], [1, 0, 0]])
    edited = dict(line_points=[[2, 0, 0], [5, 0, 0]],
                  control_points=[dict(position=[2, 0, 0], segment_to_next={'interp_mode': 'trace'})])
    result = corrected_annotation(original, edited, {'corrected_sha256': 'abc'})
    assert result['line_points'] == edited['line_points']
    assert result['control_points'] == edited['control_points']
    assert result['evaluation_direction'] == 'forward'
    assert result['vc_open_data_coordinate_space'] == 'scroll/scan'
    assert result['benchmark_correction']['corrected_sha256'] == 'abc'
    assert 'evaluation_direction' not in edited


def test_pair_profile_projects_to_segments_and_clamps_endpoints():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_pair_benchmark import distance_profile
    result = distance_profile([[5, 3, 0], [12, 0, 0]], [[0, 0, 0], [10, 0, 0]])
    assert result['distance_base'] == pytest.approx([3, 2])
    assert result['nearest_base_xyz'] == [[5, 0, 0], [10, 0, 0]]
    assert result['arc_base'][-1] == pytest.approx(58 ** .5)


def test_scroll_grouping_shares_paris_collections():
    sources = [dict(id='a', coordinate_space='Paris/scan', fiber_manifest='/cache/paris'),
               dict(id='b', coordinate_space='Other/scan', fiber_manifest='/cache/other'),
               dict(id='c', coordinate_space='Paris/scan', fiber_manifest='/cache/paris')]
    groups = scroll_groups({'sources': sources})
    assert [[s['id'] for s in g] for g in groups] == [['a', 'c'], ['b']]


def test_anchor_native_summary_does_not_apply_restart_metrics():
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import finalize_native_summary
    summary = dict(completed=True, backend='anchors', cases=1, anchor_cases=[{'placement': True}])
    assert finalize_native_summary(summary) == summary
    with pytest.raises(ValueError, match='incomplete'):
        finalize_native_summary(dict(summary, cases=2))


def test_weighted_fit_sweep_has_twenty_distinct_candidates_and_legacy():
    from vesuvius.neural_tracing.fiber_trace_3d.anchor_fit_sweep import variants
    rows=list(variants())
    assert len(rows)==21
    assert rows[0]==('legacy',None)
    assert len({name for name, _ in rows})==21
    assert sum(fit is not None and fit['iterations']==0 for _,fit in rows)==2


def test_fiberlet_parameter_sweep_variants_require_legacy_lattice_and_order():
    from vesuvius.neural_tracing.fiber_trace_3d.fiberlet_parameter_sweep import ordered_variants
    entries = [
        dict(name='prewarm', warmup=True, parameters=dict(
            beam_width=16, beam_step_base_voxels=48, lookahead_base_voxels=384)),
        dict(name='baseline', parameters=dict(
            beam_width=16, beam_step_base_voxels=48, lookahead_base_voxels=384)),
        dict(name='fine', parameters=dict(
            beam_width=32, beam_step_base_voxels=24, lookahead_base_voxels=192)),
    ]
    rows = ordered_variants({'fiberlet': {}, 'fiberlet_sweep': {'variants': entries}})
    assert [row['name'] for row in rows] == ['prewarm', 'baseline', 'fine']
    assert rows[0]['warmup'] and not rows[1]['warmup']
    with pytest.raises(ValueError, match='measured fiberlet settings must be unique'):
        ordered_variants({'fiberlet': {}, 'fiberlet_sweep': {'variants': entries + [
            dict(name='baseline-again', parameters=entries[1]['parameters'])]}})
    with pytest.raises(ValueError, match='production anchors'):
        ordered_variants({'fiberlet': {'anchor_fit': {}}, 'fiberlet_sweep': {'variants': entries}})
    bad = json.loads(json.dumps(entries))
    bad[-1]['parameters']['beam_step_base_voxels'] = 25
    with pytest.raises(ValueError, match='8-base lattice'):
        ordered_variants({'fiberlet': {}, 'fiberlet_sweep': {'variants': bad}})


def test_fiberlet_path_sweep_resolves_semantic_defaults_and_rejects_aliases():
    from vesuvius.neural_tracing.fiber_trace_3d.fiberlet_parameter_sweep import ordered_path_variants
    entries = [
        dict(name='prewarm', warmup=True, parameters={}),
        dict(name='baseline', parameters={}),
        dict(name='fine', parameters=dict(
            longitudinal_step_prediction=1.5, transverse_step_prediction=.25)),
    ]
    rows = ordered_path_variants({'fiberlet': {}, 'fiberlet_path_sweep': {'variants': entries}})
    assert rows[0]['resolved_path']['longitudinal_step_prediction'] == 2
    assert rows[2]['resolved_path']['longitudinal_step_prediction'] == 1.5
    assert rows[2]['resolved_path']['transverse_step_prediction'] == .25
    assert set(rows[2]['resolved_path']) == {
        'cell_radius', 'neighborhood_margin_cells', 'longitudinal_step_prediction',
        'transverse_step_prediction', 'maximum_endpoint_angle_degrees',
        'maximum_prediction_deviation_degrees', 'corridor_radius_prediction',
        'invalid_prediction_cost_per_voxel', 'smoothness_weight',
        'smoothness_normal_weight', 'smoothness_tangent_weight',
        'smoothness_free_angle_degrees'}
    with pytest.raises(ValueError, match='measured fiberlet path settings must be unique'):
        ordered_path_variants({'fiberlet': {}, 'fiberlet_path_sweep': {'variants': entries + [
            dict(name='baseline-again', parameters={})]}})
    with pytest.raises(ValueError, match='duplicates'):
        ordered_path_variants({'fiberlet': {}, 'fiberlet_path_sweep': {'variants': [
            dict(name='bad-radius', parameters={'corridor_radius_prediction': 4})]}})
    with pytest.raises(ValueError, match='semantic'):
        ordered_path_variants({'fiberlet': {}, 'fiberlet_path_sweep': {'variants': [
            dict(name='threads', parameters={'parallel_threads': 8})]}})


def test_fiberlet_parameter_sweep_attempt_recovery_and_completed_validation(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiberlet_parameter_sweep import (
        _validate_completed, next_attempt_root,
    )
    candidate = tmp_path / 'candidate'
    (candidate / 'attempt_001').mkdir(parents=True)
    (candidate / 'attempt_003').mkdir()
    assert next_attempt_root(candidate).name == 'attempt_004'
    metrics = candidate / 'attempt_003/metrics.json'
    metrics.write_text('{}')
    variant = dict(name='baseline', warmup=False,
                   parameters=dict(beam_width=16, beam_step_base_voxels=48,
                                   lookahead_base_voxels=384), identity=(16,48.,384.))
    identity = {'sha256': 'matrix'}
    pointer = candidate / 'result.json'
    saved_variant = {key:value for key,value in variant.items() if key != 'identity'}
    pointer.write_text(json.dumps(dict(variant=saved_variant, matrix_identity=identity,
        attempt_root=str(metrics.parent), metrics_sha256=sha256_file(metrics),
        row={'name':'baseline','status':'completed'})))
    assert _validate_completed(pointer, variant, identity)['status'] == 'completed'
    with pytest.raises(ValueError, match='provenance changed'):
        _validate_completed(pointer, variant, {'sha256': 'different-matrix'})
    metrics.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='metrics changed'):
        _validate_completed(pointer, variant, identity)


def test_fiberlet_parameter_sweep_extracts_and_ranks_case_metrics(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiberlet_parameter_sweep import (
        case_diagnostics, ranking_key,
    )
    for index, failure in enumerate((None, 64.0)):
        path = tmp_path / f'scrolls/00/cases/{index}/trace.json'
        path.parent.mkdir(parents=True)
        failures = [] if failure is None else [{'reference_arc_base': failure, 'reason': 'distance_above_threshold'}]
        path.write_text(json.dumps(dict(
            provenance=dict(fiber_path=f'fiber-{index}', coordinate_space='scroll/scan'),
            replay=dict(reference_end_arc_base=128.0, failures=failures))))
    rows = case_diagnostics(tmp_path)
    assert [row['first_failure_or_end_base'] for row in rows] == [128.,64.]
    assert [row['first_failure_reason'] for row in rows] == [None,'distance_above_threshold']
    better = dict(name='better',status='completed',warmup=False,coverage_percent=100,
                  first_failure_total_base=200,global_result={'failures':1})
    worse = dict(name='worse',status='completed',warmup=False,coverage_percent=100,
                 first_failure_total_base=100,global_result={'failures':2})
    assert sorted([worse,better], key=ranking_key) == [better,worse]


def test_directed_binding_preserves_model_and_collections(tmp_path):
    path = tmp_path / 'piece.json'
    path.write_text(json.dumps({'evaluation_direction': 'forward'}))
    case = {'path': str(path), 'sha256': sha256_file(path), 'candidate': {'source_id': 'B'}}
    manifest = {'completed': True, 'case_count': 1, 'cases': [case]}
    config = {'sources': [{'id': 'A', 'fibers': ['old']},
                          {'id': 'B', 'fibers': ['old'], 'fiber_manifest': 'best-model'}],
              'reference_plan': {}, 'live_endpoint': 'old', 'render': True}
    result = bind_benchmark(config, manifest)
    assert result['sources'] == [{'id': 'B', 'fibers': [str(path)], 'fiber_manifest': 'best-model'}]
    assert 'reference_plan' not in result and 'live_endpoint' not in result
    assert not result['render'] and config['render']
    with pytest.raises(ValueError, match='canonical'):
        bind_benchmark(config, manifest, backend='fiberlet')
    config['sources'][1].update(fiberlet_source_identity={'model': 'same'}, fiberlet_cache_root='same-cache')
    graph = bind_benchmark(config, manifest, backend='fiberlet', fiberlet={'storage_chunk_side_base': 128})
    assert graph['backend'] == 'fiberlet'
    assert graph['fiberlet']['storage_chunk_side_base'] == 128
    assert graph['sources'][0]['fiber_manifest'] == 'best-model'
    anchors = bind_benchmark(config, manifest, backend='anchors', fiberlet={'storage_chunk_side_base': 128})
    assert anchors['backend'] == 'anchors'
    config['sources'][1]['input_volume'] = 's3://bucket/scan.zarr'
    rendered = bind_benchmark(config, manifest, render=True)
    assert rendered['render']
    assert rendered['sources'][0]['ct_volume'] == 's3://bucket/scan.zarr'
    assert rendered['sources'][0]['ct_base_volume'] is True
    assert 'ct_volume' not in config['sources'][1]
    config['sources'][1].update(ct_volume='/explicit/ct', ct_base_volume=False)
    assert bind_benchmark(config, manifest, render=True)['sources'][0]['ct_volume'] == '/explicit/ct'
    with pytest.raises(ValueError, match='anchor-only'):
        bind_benchmark(config, manifest, backend='anchors', render=True)
    with pytest.raises(ValueError, match='boolean'):
        bind_benchmark(config, manifest, render=1)
    manifest.update(case_count=2, cases=[case, case])
    with pytest.raises(ValueError, match='duplicate'):
        bind_benchmark(config, manifest)


def test_rejects_changed_annotations(tmp_path):
    path = tmp_path / 'piece.json'
    path.write_text('{}')
    manifest = {'completed': True, 'case_count': 1, 'cases': [
        {'path': str(path), 'sha256': sha256_file(path), 'candidate': {'source_id': 'A'}}]}
    config = {'sources': [{'id': 'A'}]}
    with pytest.raises(ValueError, match='forward'):
        bind_benchmark(config, manifest)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed'):
        bind_benchmark(config, manifest)


def test_render_index_directed_piece_and_missing_page(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import write_render_review_index
    piece = str(tmp_path / '000_piece.json')
    from pathlib import Path
    Path(piece).write_text('{"type":"vc3d_fiber","version":1,"line_points":[[0,0,0],[200,0,0]],"control_points":[[0,0,0],[200,0,0]]}')
    original = tmp_path / 'original.json'
    original.write_text('{"line": [1]}')
    manifest = dict(cases=[dict(path=piece, candidate=dict(fiber=str(original)), historical_restarts=1)])
    case = tmp_path / 'scrolls/00/cases/7'
    (case / 'rendered').mkdir(parents=True)
    (case / 'trace.json').write_text(json.dumps(dict(
        provenance=dict(fiber_path=piece, reverse=False, source_id='scroll', case_index=7, base_voxel_size_um=2),
        replay=dict(reference_begin_arc_base=0, reference_end_arc_base=200,
                    completed_reference_arc_base=200, failures=[dict(reference_arc_base=100)],
                    segments=[dict(trace_points_base_xyz=[[1,2,3],[4,5,6]],
                        start_reference_arc_base=0, end_reference_arc_base=100,
                        termination_reason='distance_above_threshold'),
                        dict(trace_points_base_xyz=[[7,8,9],[10,11,12]],
                        start_reference_arc_base=100, end_reference_arc_base=200,
                        termination_reason='reference_end')]))))
    (case / 'rendered/render.json').write_text(json.dumps(dict(pages=2)))
    result = dict(scroll_artifacts=[str(tmp_path / 'scrolls/00')])
    with pytest.raises(ValueError, match='pages'):
        write_render_review_index(tmp_path, result, manifest)
    for i in range(2):
        (case / f'rendered/strips_{i}.jpg').write_bytes(b'test-placeholder')
    rows = write_render_review_index(tmp_path, result, manifest)
    assert len(rows) == 1 and rows[0]['restarts'] == 1
    assert rows[0]['mean_mm'] == pytest.approx(.2)
    assert rows[0]['failures_per_mm'] == pytest.approx(2.5)
    assert len(rows[0]['images']) == 2
    assert 'image 2' in (tmp_path / 'README.md').read_text()
    assert rows[0]['source_fiber'] == str(original)
    extracted_path = next((case / 'rendered').glob('evalextract_*.json'))
    extracted = json.loads(extracted_path.read_text())
    assert extracted['line_points'] == json.loads(Path(piece).read_text())['line_points']
    assert extracted['review_provenance']['original_fiber'] == str(original)
    assert (case / 'rendered/original.json').read_bytes() == original.read_bytes()
    traces = sorted((case / 'rendered').glob('evaltrace-beam_*.json'))
    first = json.loads(traces[0].read_text())
    second = json.loads(traces[1].read_text())
    assert first['line_points'] == [[1,2,3],[4,5,6]]
    assert second['line_points'] == [[7,8,9],[10,11,12]]
    assert first['control_points'] == first['line_points']
    extracted_path.write_text('{}')
    with pytest.raises(ValueError, match='modified'):
        write_render_review_index(tmp_path, result, manifest)


def test_anchor_report_pools_gaps_and_keeps_empty_pieces():
    from vesuvius.neural_tracing.fiber_trace_3d.anchor_coverage_report import coverage_report
    base = dict(source_id='PHerc1229', cell_side_base=32, base_voxel_size_um=2.4,
                reference_length_base=96, no_anchor_length_base=0, empty_arc_bin_length_base=32,
                arc_bins=3, empty_arc_bins=1, accepted_anchors=[{}, {}], unique_arcs_base=[0, 64],
                interior_gaps_base=[64], leading_gap_base=0, trailing_gap_base=32,
                empty_arc_bin_runs=[1], invalid_normal_tests=0, ellipse_tests=2)
    empty = dict(base, source_id='PHercParis4_fibers', no_anchor_length_base=96,
                 empty_arc_bin_length_base=96, empty_arc_bins=3, accepted_anchors=[], unique_arcs_base=[],
                 interior_gaps_base=[], leading_gap_base=None, trailing_gap_base=None, empty_arc_bin_runs=[3])
    report = coverage_report([base, empty])
    assert set(report['scrolls']) == {'PHerc1299', 'PHercParis4'}
    pooled = report['overall']
    assert pooled['interior_gap_cells']['count'] == 1
    assert pooled['interior_gap_cells']['median'] == 2
    assert pooled['no_anchor_pieces'] == 1
    assert pooled['empty_arc_bins_percent'] == pytest.approx(100 * 4/6)
    assert sum(pooled['empty_bin_runs_cells']['histogram_counts']) == 2
    base['accepted_anchors']=[{'direction_error_degrees': 10, 'reference_arc_base': 0},
                              {'direction_error_degrees': 30, 'reference_arc_base': 64}]
    directions=coverage_report([base])['overall']['direction']
    assert directions['median']==20
    assert directions['within_degrees_percent']['20']==50
    assert coverage_report([base])['overall']['direction_aware_empty_arc_bins_percent']['20']==pytest.approx(200/3)
