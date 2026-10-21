"""Rebuild real GT events from frozen native evidence, without source writes."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d import fiber_gt_constraint_score as score


def test_rebuild_real_gt_events_and_reuse(tmp_path, monkeypatch):
    cache_path = os.environ.get('FIBER_GT_EVENT_CACHE')
    gt_path = os.environ.get('FIBER_GT_ROOT')
    if not cache_path or not gt_path:
        pytest.skip('set FIBER_GT_EVENT_CACHE and FIBER_GT_ROOT to frozen real evidence')
    cache, gt = Path(cache_path), Path(gt_path)
    expected = json.loads((cache / 'cross_events.json').read_text())
    inputs = json.loads((cache / 'inputs.json').read_text())
    output = tmp_path / 'rebuilt'
    output.mkdir()
    # Keep expensive native evidence frozen; exercise the actual Python event
    # construction and its three native dispatches using those recorded outputs.
    (output / 'combined.afv').symlink_to(cache / 'combined.afv')
    calls = []
    def recorded_native(path, settings):
        cfg = json.loads(Path(path).read_text())
        calls.append(cfg)
        assert cfg['afv'] == str(output / 'combined.afv') or cfg.get('normal_curves_output')
        if cfg.get('normal_curves_output'):
            assert cfg['orient_database'] == json.loads((gt / 'uv_whole/config.json').read_text())['database']
            Path(cfg['normal_curves_output']).symlink_to(cache / 'frozen_gt_normals', target_is_directory=True)
        else:
            name = Path(cfg['output']).stem
            Path(cfg['database']).symlink_to(cache / (name + '.sqlite'))
            Path(cfg['output']).symlink_to(cache / (name + '.json'))
            if name == 'orient':
                assert cfg['orient_database'] == str(output / 'extract.sqlite')
    from vesuvius.neural_tracing.fiber_trace_3d import constraint_extraction
    monkeypatch.setattr(constraint_extraction, 'run_native_constraints', recorded_native)
    settings = dict(output_root=str(output), configs=list(inputs['configs']),
                    gt_root=str(gt), extract_template=str(cache / 'extract_config.json'),
                    orient_template=str(cache / 'orient_config.json'), events_only=True, threads=2)
    result = score.run(settings)
    assert len(calls) == 3
    assert result['events'] == expected['events']
    assert result['excluded'] == expected['excluded']
    assert result['combined_sweep_gauge_corrections'] == expected['combined_sweep_gauge_corrections']
    assert score.run(settings) == result
    assert len(calls) == 3
    print('REAL_GT_EVENT_REBUILD', len(result['events']), 'events match frozen output exactly')
    # Changed inputs must not silently reuse a cache with different provenance.
    alternate = tmp_path / 'different_config.json'
    alternate.write_text(Path(settings['configs'][0]).read_text())
    settings['configs'] = [str(alternate)]
    with pytest.raises(ValueError, match='Changed scoring inputs'):
        score.run(settings)


def test_real_enclosure_initializer(tmp_path):
    gt_path = os.environ.get('FIBER_GT_ROOT')
    if not gt_path:
        pytest.skip('set FIBER_GT_ROOT to a frozen real GT solve')
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_width_init import fit_enclosure_widths
    cfg = json.loads((Path(gt_path) / 'uv_whole/config.json').read_text())
    data = score.arrays(cfg['tensor_cache'])
    saved = score.arrays(Path(gt_path) / 'uv_whole/solution.npz')
    _, labels, keep, pick = uv.hard_assignment(saved['state'], len(saved['widths']), cfg, best_nonbroken=True)
    samples = soft.prepare_samples(cfg, data)
    result, anchors, qv, brackets = fit_enclosure_widths(
        data, samples, pick, saved['widths'], 0, np.arange(len(saved['widths'])))
    assert brackets and anchors and qv
    assert np.isfinite(result).all() and (result >= 100).all() and (result <= 50000).all()
    print('REAL_ENCLOSURE_INIT', len(data['source_id']), 'fibers', len(brackets), 'brackets')


def test_build_real_gt_cache_from_scratch_and_score(tmp_path):
    gt_path = os.environ.get('FIBER_GT_ROOT')
    binary = os.environ.get('VC_CONSTRAINT_TEST_BINARY')
    if not gt_path or not binary:
        pytest.skip('set FIBER_GT_ROOT and VC_CONSTRAINT_TEST_BINARY for fresh native extraction')
    gt = Path(gt_path)
    # Copy configuration only. All source AFVs, solves and databases stay read-only.
    local_gt = tmp_path / 'gt'
    (local_gt / 'uv_whole').mkdir(parents=True)
    config = json.loads((gt / 'uv_whole/config.json').read_text())
    (local_gt / 'uv_whole/solution.npz').symlink_to(gt / 'uv_whole/solution.npz')
    for name in ('extract', 'orient'):
        native = json.loads((gt / (name + '_config.json')).read_text())
        native['remote_cache'] = str(tmp_path / 'normal-cache')
        (local_gt / (name + '_config.json')).write_text(json.dumps(native))
    # The native exporter verifies endpoint normals to 1e-12. Prepare matching
    # orientation evidence with this build instead of using normals baked by a
    # historical binary; do not relax that guard or alter the original database.
    from vesuvius.neural_tracing.fiber_trace_3d.constraint_extraction import run_native_constraints
    extraction = json.loads((local_gt / 'extract_config.json').read_text())
    extraction.update(database=str(local_gt / 'raw.sqlite'),
                      output=str(local_gt / 'raw.json'))
    extraction_path = local_gt / 'prepare_extraction.json'
    extraction_path.write_text(json.dumps(extraction))
    run_native_constraints(extraction_path, dict(native_binary=binary))
    orientation = json.loads((local_gt / 'orient_config.json').read_text())
    orientation.update(database=str(local_gt / 'oriented.sqlite'),
                       output=str(local_gt / 'oriented.json'),
                       orient_database=extraction['database'])
    orientation_path = local_gt / 'prepare_orientation.json'
    orientation_path.write_text(json.dumps(orientation))
    run_native_constraints(orientation_path, dict(native_binary=binary))
    config['database'] = orientation['database']
    (local_gt / 'uv_whole/config.json').write_text(json.dumps(config))
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps(dict(fiber_uv_layout=config)))
    root = tmp_path / 'cross'
    settings = dict(output_root=str(root), configs=[str(candidate)], gt_root=str(local_gt),
                    extract_template=str(local_gt / 'extract_config.json'),
                    orient_template=str(local_gt / 'orient_config.json'),
                    native_binary=binary, events_only=True, threads=2)
    result = score.run(settings)
    assert result['events']
    assert (root / 'combined.afv').exists() and (root / 'orient.sqlite').exists()
    assert (root / 'frozen_gt_normals/index.json').exists()
    spec = dict(root=str(root), gt_root=str(local_gt), events='cross_events.json')
    rebuilt = score.FrozenGTScorer(spec, config)
    values = rebuilt.score(gt / 'uv_whole/solution.npz', config)
    assert values['gt_enabled_events'] > 0
    (root / 'events-list.json').write_text(json.dumps(result['events']))
    legacy = score.FrozenGTScorer(dict(spec, events='events-list.json'), config)
    assert values == legacy.score(gt / 'uv_whole/solution.npz', config)
    assert score.run(settings) == result
    # The original direct-score branch must also work without a frozen-cache flag.
    direct = score.run(dict(settings, events_only=False))
    assert direct[0]['constraints'] == len(result['events'])
    print('REAL_GT_FRESH_CACHE', len(result['events']), 'events; envelope/list scoring identical')
