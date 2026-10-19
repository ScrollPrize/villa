"""Three flow splitting cases; all pieces retained, GT labels evaluation-only."""
import argparse
import csv
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vesuvius/src'))
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import (
    arrays, read, write, digest, Paths, graph, large_reference)
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_split import Pieces


def run(settings):
    if settings.get('continuation_weight', 0) != 0:
        raise ValueError('This experiment uses independent pieces, no continuation penalty')
    project, output = Path(settings['project']), Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    reference = read(project/'gt_winding_reference.json')
    configs = dict(free=project/'data/neighbor_round10_pair_v1/free/solve/config.json',
                   joint_intact_gt=project/'data/neighbor_round10_pair_v1/joint/solve/config.json',
                   joint_splittable_gt=project/'data/neighbor_round10_selectable_gt_v1/solve/config.json')
    manifest = dict(settings=settings, accepted_baseline_commit='5122bf36b', cases={}, input_hashes={},
                    notes='All pieces enabled; third arm allows GT splitting, not disabling. Original-start charts.')
    dependencies = {project/'gt_winding_reference.json', project/'data/neighbor_round10_pair_v1/prepared/gt_events.json'}
    gc = project/'data/bbox_gt_solve_v3/uv_whole/config.json'
    dependencies.add(gc)
    dependencies.update(Path(read(gc)[k]) for k in ('afv', 'umbilicus', 'tensor_cache'))
    for name, path in configs.items():
        config = read(path)
        if config.get('tensor_sha256'):
            assert digest(config['tensor_cache']) == config['tensor_sha256']
        dependencies.add(path)
        dependencies.update(Path(config[k]) for k in ('afv', 'umbilicus', 'tensor_cache'))
        manifest['cases'][name] = dict(config_path=str(path), config=config)
    manifest['input_hashes'] = {str(p): digest(p) for p in sorted(dependencies)}
    write(output/'manifest.json', manifest)
    results = {}
    for name, path in configs.items():
        started = time.monotonic()
        print('Preparing '+name, flush=True)
        config = read(path); d = arrays(config['tensor_cache']); paths = Paths(config, d)
        with sqlite3.connect(Path(config['afv']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='gt_source_id_map'").fetchone()
        import json
        gt_sources = list(json.loads(row[0]).values()) if row else []
        protected = gt_sources if name == 'joint_intact_gt' else []
        pieces = Pieces(d, settings['minimum_length_vx'], settings['minimum_perpendicular'], protected)
        g, transport = graph(config, d, paths)
        # Zero-splitting mapping must reproduce all endpoint indices and source identity.
        whole = Pieces(d, 0)
        np.testing.assert_array_equal(whole.parent, np.arange(len(d['source_id'])))
        for key in ('a', 'b'):
            np.testing.assert_array_equal(whole.remap(g, transport)[key], g[key])
        sg = pieces.remap(g, transport)
        result = solve(len(pieces.parent), **sg)
        details = []
        scored = large_reference(project, d, result, paths, reference, details=details, pieces=pieces)
        scored['single_global_gauge']['signs'] = {
            kind: dict(events=sum(e['parallel'] == parallel for e in details),
                       violations=sum(e['global_gauge_sign_violation'] for e in details if e['parallel'] == parallel))
            for kind, parallel in [('perpendicular', False), ('parallel', True)]}
        # Same scorer/lookup equivalence against the accepted whole-fiber solution.
        baseline_name = 'free' if name == 'free' else 'joint'
        baseline = arrays(ROOT/f'global-fiber-min-flow/data/seven_fractional_v1/{baseline_name}/solution.npz')
        old_metrics = read(ROOT/f'global-fiber-min-flow/data/seven_fractional_v1/{baseline_name}/metrics.json')
        assert large_reference(project, d, baseline, paths, reference, pieces=whole) == old_metrics['reference']
        ea, eb = pieces.endpoints('a'), pieces.endpoints('b')
        physical = result['winding'][eb]-result['winding'][ea]+transport['fiber_b']-transport['fiber_a']-transport['connector']
        margin = (~d['parallel']) & (d['sign']*(d['family'][d['j']]-d['family'][d['i']]) < 0)
        wrong = (d['sign'] != 0) & (d['sign']*physical < margin)
        links = np.flatnonzero(np.diff(pieces.parent) == 0)
        jumps = result['winding'][links+1]-result['winding'][links]
        component_sizes = np.bincount(result['component'])
        baseline_components = baseline['component'][pieces.parent]
        child_counts = {str(int(c)): int(len(np.unique(result['component'][baseline_components == c])))
                        for c in np.unique(baseline_components)}
        bad = [e for e in details if e['sign_violation']]
        metrics = dict(original_fibers=len(d['source_id']), pieces=len(pieces.parent),
            split_original_fibers=int(np.sum(np.bincount(pieces.parent) > 1)),
            gt_original_fibers=len(gt_sources), gt_pieces=int(np.isin(pieces.source_id, gt_sources).sum()),
            protected_gt_sources=protected, events=len(d['i']), used_events=int(transport['valid'].sum()),
            excluded_events=int(np.sum(~transport['valid'])), flow_arcs=len(sg['a']),
            retained_length_percent=100., continuation_weight=0,
            boundaries=len(links), nonzero_boundary_jumps=int(np.sum(jumps != 0)),
            boundary_max_jump=int(np.max(abs(jumps), initial=0)),
            boundary_note='Independent component offsets make cross-component jumps unidentifiable',
            identifiable_nonzero_boundary_jumps=int(np.sum((jumps != 0) & (result['component'][links] == result['component'][links+1]))),
            certificate=result['certificate'], component_sizes=component_sizes.tolist(),
            child_components_per_baseline_component=child_counts,
            reference=scored,
            internal_signs={kind:dict(events=int(np.sum(mask & (d['sign'] != 0))), violations=int(np.sum(mask & wrong)))
                            for kind, mask in [('perpendicular', ~d['parallel']), ('parallel', d['parallel'])]})
        folder = output/name; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'solution.npz', source_id=pieces.source_id, parent=pieces.parent,
                            start=pieces.start, end=pieces.end, event_piece_a=ea, event_piece_b=eb,
                            winding=result['winding'], component=result['component'], flow=result['flow'],
                            **sg, **transport)
        write(folder/'gt_violations.json', bad)
        with (folder/'gt_events.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(details)
        metrics['seconds_total'] = time.monotonic()-started
        write(folder/'metrics.json', metrics); results[name] = metrics; write(output/'results.json', results)
        print(name, 'pieces', len(pieces.parent), 'perp L1', scored['perpendicular']['l1'],
              'GT signs', scored['signs'], 'components', result['certificate']['components'], flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('run_split_large.json'))
    run(read(parser.parse_args().config))
