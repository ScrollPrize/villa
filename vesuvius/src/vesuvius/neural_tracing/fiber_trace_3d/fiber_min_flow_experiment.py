"""Run cached winding experiments using an external JSON configuration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import scipy

from .fiber_min_flow import solve
from .fiber_min_flow_benchmark import Paths, graph
from .fiber_min_flow_split import Pieces
from .fiber_min_flow_evaluation import (
    small_reference, large_reference, sign_counts, boundary_diagnostics, compare_solution,
)


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def resolve(base, path):
    return str((base / Path(path).expanduser()).resolve())


def load_config(path):
    """Resolve paths relative to the JSON file declaring them; collect dependencies."""
    path = Path(path).resolve()
    settings = read(path)
    dependencies = {str(path)}
    if set(settings)-{'output_root', 'cases'}:
        raise ValueError('Unknown experiment setting')

    def file(base, value):
        value = resolve(base, value)
        dependencies.add(value)
        return value

    def dataset(value, base):
        if isinstance(value, str):
            source = file(base, value)
            value, base = read(source), Path(source).parent
        else:
            value = dict(value)
        for key in ('fibers', 'umbilicus', 'tensor_cache'):
            value[key] = file(base, value[key])
        return value

    names = set()
    if not settings.get('cases'):
        raise ValueError('At least one case is required')
    for case in settings['cases']:
        if set(case)-{'name', 'dataset', 'target_mode', 'sign_only', 'split', 'evaluation', 'baseline', 'audit'}:
            raise ValueError('Unknown case setting')
        if not isinstance(case.get('sign_only', False), bool):
            raise ValueError('sign_only must be boolean')
        name = case['name']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in names:
            raise ValueError('Case names must be unique, safe directory names')
        names.add(name)
        case['dataset'] = dataset(case['dataset'], path.parent)
        if case.get('target_mode', 'fractional') not in ('fractional', 'rounded'):
            raise ValueError('Unknown target_mode')
        split = case.get('split')
        if split is not None:
            unknown = set(split)-{'minimum_length_vx', 'minimum_perpendicular', 'protected_sources'}
            if unknown:
                raise ValueError(f'Unknown split settings: {sorted(unknown)}')
        evaluation = case.get('evaluation')
        if evaluation:
            if set(evaluation)-{'mode', 'reference', 'events', 'dataset'}:
                raise ValueError('Unknown evaluation setting')
            if evaluation['mode'] not in ('relations', 'cross_events'):
                raise ValueError('Unknown evaluation mode')
            if evaluation['mode'] == 'relations' and split is not None:
                raise ValueError('Split cases require cross_events evaluation')
            evaluation['reference'] = file(path.parent, evaluation['reference'])
            if evaluation['mode'] == 'cross_events':
                evaluation['events'] = file(path.parent, evaluation['events'])
                evaluation['dataset'] = dataset(evaluation['dataset'], path.parent)
        if case.get('baseline'):
            case['baseline'] = file(path.parent, case['baseline'])
        audit = case.get('audit', {})
        if set(audit)-{'details', 'sign_analysis', 'measurements', 'rounded_baseline'}:
            raise ValueError('Unknown audit setting')
        for flag in ('details', 'sign_analysis'):
            if flag in audit and not isinstance(audit[flag], bool):
                raise ValueError(flag+' must be boolean')
        if audit.get('rounded_baseline'):
            if case.get('target_mode', 'fractional') != 'fractional':
                raise ValueError('rounded_baseline requires a fractional trial')
            audit['rounded_baseline'] = file(path.parent, audit['rounded_baseline'])
            metrics = Path(audit['rounded_baseline']).with_name('metrics.json')
            if metrics.is_file():
                dependencies.add(str(metrics))
        if audit.get('measurements'):
            sources = audit['measurements']
            if set(sources)-{'database', 'stage1', 'sign_cosine_min'}:
                raise ValueError('Unknown measurement audit setting')
            for key in ('database', 'stage1'):
                sources[key] = resolve(path.parent, sources[key])
                if Path(sources[key]).is_file():
                    dependencies.add(sources[key])
    settings['output_root'] = resolve(path.parent, settings['output_root'])
    output = Path(settings['output_root'])
    if any(p == output or output in p.parents for p in map(Path, dependencies)):
        raise ValueError('Output directory must not contain input files')
    return settings, dependencies


def run(config_path):
    settings, dependencies = load_config(config_path)
    output = Path(settings['output_root'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; choose a new output_root')
    hashes = {p: digest(p) for p in sorted(dependencies)}
    for case in settings['cases']:
        datasets = [case['dataset']]
        if case.get('evaluation', {}).get('mode') == 'cross_events':
            datasets.append(case['evaluation']['dataset'])
        for config in datasets:
            if config.get('tensor_sha256') and hashes[config['tensor_cache']] != config['tensor_sha256']:
                raise ValueError('Tensor hash mismatch: '+config['tensor_cache'])
    output.mkdir(parents=True, exist_ok=True)
    write(output/'manifest.json', dict(schema_version=1, settings=settings, input_hashes=hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        implementation_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('fiber_min_flow*.py')}))
    results = {}
    for case in settings['cases']:
        started = time.monotonic()
        name, config = case['name'], case['dataset']
        print('Preparing '+name, flush=True)
        d = arrays(config['tensor_cache'])
        paths = Paths(config, d)
        sign_only = case.get('sign_only', False)
        mode = case.get('target_mode', 'fractional')
        g, transport = graph(config, d, paths, sign_only, mode)
        pieces = None
        ea, eb = d['i'], d['j']
        split = case.get('split')
        if split is not None:
            pieces = Pieces(d, split.get('minimum_length_vx', 1000),
                            split.get('minimum_perpendicular', 4), split.get('protected_sources', []))
            whole = Pieces(d, 0)
            # Identity mapping must preserve graph direction, including reversed hinge edges.
            np.testing.assert_array_equal(whole.endpoints('a'), d['i'])
            np.testing.assert_array_equal(whole.endpoints('b'), d['j'])
            g = pieces.remap(g, transport)
            if sign_only:
                reverse = d['sign'][transport['flow_event_index']] < 0
                g['a'][reverse], g['b'][reverse] = g['b'][reverse].copy(), g['a'][reverse].copy()
            ea, eb = pieces.endpoints('a'), pieces.endpoints('b')
        result = solve(len(d['source_id']) if pieces is None else len(pieces.parent), **g)
        solution = dict(source_id=d['source_id'] if pieces is None else pieces.source_id,
                        winding=result['winding'], component=result['component'], flow=result['flow'],
                        **g, **transport)
        if pieces is not None:
            solution.update(parent=pieces.parent, start=pieces.start, end=pieces.end,
                            event_piece_a=ea, event_piece_b=eb)
        else:
            solution['family'] = d['family']
        physical = result['winding'][eb]-result['winding'][ea]+transport['fiber_b']-transport['fiber_a']-transport['connector']
        margin = (~d['parallel']) & (d['sign']*(d['family'][d['j']]-d['family'][d['i']]) < 0)
        metrics = dict(nodes=len(result['winding']), original_fibers=len(d['source_id']),
            events=len(d['i']), used_events=int(transport['valid'].sum()),
            excluded_events=int((~transport['valid']).sum()), flow_arcs=len(g['a']),
            target_mode=mode, sign_only=sign_only, retained_length_percent=100.,
            internal_signs=sign_counts(d['parallel'], d['sign'], physical, margin),
            certificate=result['certificate'], component_sizes=np.bincount(result['component']).tolist(),
            seam=dict(connectors_crossing=int(np.count_nonzero(transport['connector'])),
                      fiber_endpoint_crossings=int(np.sum(abs(transport['fiber_a'])+abs(transport['fiber_b'])))))
        if pieces is not None:
            metrics['split'] = dict(boundary_diagnostics(pieces, result),
                split_original_fibers=int(np.sum(np.bincount(pieces.parent)>1)),
                protected_sources=split.get('protected_sources', []), continuation_weight=0)
        details = []
        evaluation = case.get('evaluation')
        if evaluation:
            reference = read(evaluation['reference'])
            if evaluation['mode'] == 'relations':
                scored = small_reference(d, result, reference, transport)
            else:
                gc = evaluation['dataset']
                gd = arrays(gc['tensor_cache'])
                gp = Paths(gc, gd)
                events = read(evaluation['events'])
                scored = large_reference(d, result, paths, reference, events, gd, gp, details, pieces)
                if pieces is not None:
                    # Score a fresh whole-fiber solve through both lookup paths.
                    wg, _ = graph(config, d, paths, sign_only, mode)
                    wr = solve(len(d['source_id']), **wg)
                    direct = large_reference(d, wr, paths, reference, events, gd, gp)
                    mapped = large_reference(d, wr, paths, reference, events, gd, gp, pieces=whole)
                    if direct != mapped:
                        raise ValueError('Zero-split reference lookup differs')
                    metrics['zero_split_reference_equivalent'] = True
                    original_components = wr['component'][pieces.parent]
                    metrics['split']['child_components_per_whole_component'] = {
                        str(int(c)): int(len(np.unique(result['component'][original_components == c])))
                        for c in np.unique(original_components)}
            metrics['reference'] = scored
        if case.get('baseline'):
            metrics['baseline_comparison'] = compare_solution(solution, arrays(case['baseline']))
        folder = output/name
        folder.mkdir()
        np.savez_compressed(folder/'solution.npz', **solution)
        if evaluation and evaluation['mode'] == 'cross_events':
            write(folder/'gt_violations.json', [e for e in details if e['sign_violation']])
            # Empty coverage is valid and produces an empty event table.
            with (folder/'gt_events.csv').open('w', newline='') as f:
                columns = list(dict.fromkeys(k for event in details for k in event)) if details else ['event_id', 'sign_violation']
                writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
                writer.writeheader()
                writer.writerows(details)
        metrics['seconds_total'] = time.monotonic()-started
        write(folder/'metrics.json', metrics)
        if case.get('audit'):
            from .fiber_min_flow_audit import audit_case
            audit_case(case, d, solution, folder/'audit', case['audit'])
        results[name] = metrics
        write(output/'results.json', results)
        print(name, 'objective', result['certificate']['primal'], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='External experiment JSON')
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
