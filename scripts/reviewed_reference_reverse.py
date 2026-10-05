"""Reference-seeded reverse probes; shared native replay and live inference only."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reviewed_fiber_evaluation import ROOT, REPO
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import slice_line


def read(path):
    return json.loads(path.read_text())


def arc(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def convergence(points, errors, wrong, wrong_errors):
    """Conservative exact segment projection; shared correct path is excluded."""
    p, w = np.asarray(points), np.asarray(wrong)
    if len(p) < 2 or len(w) < 2:
        return dict(wrong_extent_base=0., longest_convergence_base=0., converged=False)
    keep = (np.asarray(wrong_errors[:-1]) > 40) & (np.asarray(wrong_errors[1:]) > 40)
    a, v = w[:-1][keep], np.diff(w, axis=0)[keep]
    length = np.linalg.norm(v, axis=1)
    valid = length > 0
    a, v, length = a[valid], v[valid], length[valid]
    longest = streak = 0.
    if len(a):
        for i, delta in enumerate(np.diff(p, axis=0)):
            size = np.linalg.norm(delta)
            if not size:
                continue
            distances = []
            for q in (p[i], p[i+1]):
                t = np.clip(np.sum((q-a)*v, axis=1)/(length*length), 0, 1)
                distances.append(np.linalg.norm(q-a-t[:, None]*v, axis=1))
            angle = np.abs(v @ delta)/(length*size)
            # The radius10 tube around one segment is convex: both endpoints
            # inside guarantees the entire probe edge stays inside that tube.
            hit = min(errors[i:i+2]) > 40 and np.any((distances[0] <= 10) &
                (distances[1] <= 10) & (angle >= np.cos(np.deg2rad(25))))
            streak = streak + size if hit else 0.
            longest = max(longest, streak)
    return dict(wrong_extent_base=float(length.sum()), longest_convergence_base=float(longest),
                converged=bool(longest >= 80), convergence_testable=bool(length.sum() >= 80))


def errors_for(segment):
    errors = np.full(len(segment['trace_points_base_xyz']), np.inf)
    # Replay records matches only after the seed. Diagnostic seeds lie exactly
    # on their supplied reference. For original filtered/reseeded segments this
    # conservatively excludes their first edge from wrong-branch evidence.
    if len(errors):
        errors[0] = 0.
    # Native matches measure the same anisotropic norm, independent of stop radius.
    for match in segment['matches']:
        errors[match['trace_point_index']] = match['threshold_error_base_voxels']
    indices = [m['trace_point_index'] for m in segment['matches']]
    assert sorted(indices) == list(range(1,len(errors))), 'missing/duplicate native point matches'
    assert np.isfinite(errors).all(), 'nonfinite native threshold error'
    return errors


def run(model):
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import run as evaluate
    old = ROOT/'reverse256_v1'/model
    out = ROOT/'reference_reverse_diagnostic_v1'/model
    summary = read(old/'full/summary.json')
    config = read(old/'full.json')
    workload = read(old/'workload.json')
    paths = sorted((old/'full/cases').glob('*/trace.json'))
    with ThreadPoolExecutor(max_workers=32) as pool:
        artifacts = list(pool.map(read, paths))
    index = {(a['provenance']['source_id'], a['provenance']['fiber_path'],
              a['provenance']['reverse'], a['provenance']['reference_run']['run_index']): (p, a)
             for p, a in zip(paths, artifacts)}
    probes = []
    sources = []
    for source in summary['sources']:
        binding = next(s.copy() for s in config['sources'] if s['id'] == source['source_id'])
        binding['fibers'] = []
        for case in source['cases']:
            if not case['restart_episodes']:
                continue
            fiber = source['backend_config']['fibers'][case['source_index']]
            path, artifact = index[(source['source_id'], fiber, case['direction']=='reverse', case['run_index'])]
            ref = artifact['reference_points_base_xyz']
            replay = artifact['replay']
            for episode in case['restart_episodes']:
                event = case['failures'][episode['first_event_index']]
                segment = replay['segments'][event['segment_index']]
                fail = event['directional_reference_arc_base_voxels'] + replay['reference_begin_arc_base']
                lower = max(segment['start_reference_arc_base'], fail-256)
                wrong = segment['trace_points_base_xyz']
                wrong_length = arc(wrong)[-1]
                definitions = [('control', slice_line(wrong, max(0, wrong_length-256), wrong_length)[::-1], None)]
                for offset in (0,128,256):
                    upper = min(fail+offset, replay['reference_end_arc_base'])
                    definitions.append((f'gt{offset}', slice_line(ref, lower, upper)[::-1], upper-fail))
                for kind, points, effective in definitions:
                    name = f"{path.parent.name}_{event['index']}_{kind}"
                    annotation = out/'annotations'/source['source_id']/f'{name}.json'
                    atomic_write(annotation, dict(type='vc3d_fiber', version=1, line_points=points,
                        control_points=[points[0],points[-1]], evaluation_direction='forward',
                        optimization_mode='native_fiber_trace3d', vc_open_data_coordinate_space=binding['coordinate_space']), durable=False)
                    binding['fibers'].append(str(annotation))
                    probes.append(dict(name=name, kind=kind, scroll=source['source_id'], original_case=int(path.parent.name),
                        event_index=event['index'], source_fiber=fiber, direction=case['direction'],
                        original_artifact=str(path), segment_index=event['segment_index'], annotation=str(annotation),
                        requested_offset_base=None if kind=='control' else int(kind[2:]), effective_offset_base=effective,
                        expected_seed_base_xyz=points[0], requested_length_base=float(arc(points)[-1])))
        if binding['fibers']:
            sources.append(binding)
    config.update(sources=sources, output=str(out/'full'), error_threshold_base_voxels=1e9,
                  tangential_threshold_factor=2, excursion_grace_base_voxels=0, render=False)
    args = config['trace_args']
    i = args.index('--replay-reverse-interval-base')
    del args[i:i+2]
    config['gpu_manager']['label'] = f'reference-reverse-{model}'
    assert config['gpu_manager']['devices'] == '0,1,2,3,4,5'
    assert sha256_file(workload['checkpoint']) == workload['checkpoint_sha256']
    if (out/'evaluation.json').exists() and read(out/'evaluation.json') != config:
        raise ValueError('diagnostic configuration changed; use a new output version')
    if (out/'probes.json').exists() and read(out/'probes.json') != probes:
        raise ValueError('diagnostic geometry/provenance changed; use a new output version')
    atomic_write(out/'probes.json', probes, durable=False)
    atomic_write(out/'evaluation.json', config, durable=False)
    print(f'{model}: {len(probes)} parallel probes', flush=True)
    if not (out/'full/summary.json').exists() or not read(out/'full/summary.json').get('completed'):
        evaluate(out/'evaluation.json', workload['checkpoint'], '0,1,2,3,4,5', workload['native_binary'])
    result = read(out/'full/summary.json')
    assert result['completed'] and result['evaluated_cases'] == len(probes)
    analyze(model)


def analyze(model, allow_partial=False):
    out = ROOT/'reference_reverse_diagnostic_v1'/model
    workload = read(ROOT/'reverse256_v1'/model/'workload.json')
    probes = read(out/'probes.json')
    outputs = {}
    for p in (out/'full/cases').glob('*/trace.json'):
        a = read(p)
        outputs[a['provenance']['fiber_path']] = (p,a)
    rows = []
    for probe in probes:
        if probe['annotation'] not in outputs and allow_partial:
            continue
        p,a = outputs[probe['annotation']]
        segments = a['replay']['segments']
        row = dict(probe, probe_artifact=str(p))
        if not segments or not segments[0]['trace_points_base_xyz']:
            row.update(empty=True, complete=False, tight_correct=False, regular_correct=False)
        else:
            segment = segments[0]
            points = segment['trace_points_base_xyz']
            seed_error = float(np.linalg.norm(np.asarray(points[0])-probe['expected_seed_base_xyz']))
            assert seed_error < 0.1, (probe['name'],seed_error)
            errors = errors_for(segment)
            complete = segment['termination_reason'] == 'reference_end'
            row.update(empty=False, seed_error_base=seed_error, termination=segment['termination_reason'],
                       complete=complete, traced_length_base=float(arc(points)[-1]),
                       max_threshold_error_base=float(max(errors)), tight_correct=bool(complete and max(errors)<=20),
                       regular_correct=bool(complete and max(errors)<=40))
            if probe['kind'] != 'control':
                original = read(Path(probe['original_artifact']))['replay']['segments'][probe['segment_index']]
                row.update(convergence(points, errors, original['trace_points_base_xyz'], errors_for(original)))
        rows.append(row)
    atomic_write(out/('partial_results.json' if len(rows)<len(probes) else 'results.json'),
                 dict(model=model, expected_probes=len(probes), evaluated_probes=len(rows),
                      checkpoint_sha256=workload['checkpoint_sha256'],rows=rows), durable=False)
    for kind in ('control','gt0','gt128','gt256'):
        selected = [r for r in rows if r['kind']==kind]
        print(kind, 'n=',len(selected), 'tight=',sum(r['tight_correct'] for r in selected),
              'regular=',sum(r['regular_correct'] for r in selected),
              'terminations=',{r.get('termination') for r in selected}, flush=True)
    write_report(model)


def write_report(selected_model='gaussian'):
    historical = selected_model in ('gaussian', 'lowres')
    models = ('gaussian','lowres') if historical else ('gaussian','lowres',selected_model)
    filename = 'reviewed_reference_reverse_20260930' if historical else f'reviewed_reference_reverse_{selected_model}'
    destination = REPO/f'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/{filename}.md'
    lines = ['# Reference-seeded reverse diagnostic', '',
        'Source: reviewed_fibers_20260930_v3/reverse256_v1. Only remaining counted ordinary failures after periodic filtering are probed. Historical Gaussian12/lowres15 included for comparison. This diagnostic is not a new full-cohort score.', '',
        'Native beam8/lookahead2; unchanged model-specific step4 prediction voxels (full-resolution models8base, lowres32base). All probes run through shared live_evaluate.run/native parallel evaluation; same inference namespaces and checkpoints, demand-managed GPUs0–5. No production filter changes.', '',
        'Each failure has a reverse control along the final256base voxels of its saved erroneous trace, plus GT-tangent-seeded reverse probes at failure arc and +128/+256base along the original direction. GT probes end at max(segment start,failure−256). Offsets are clipped to the eligible span. These are directed cases, not necessarily distinct fibers.', '',
        'Online lateral stopping uses a finite1e9 radius; no periodic filtering, zero grace. Native endpoint/progress and unchanged step budget still bound each run. Score only the first uninterrupted segment. “Correct” requires reference_end termination AND every saved anisotropic absolute error ≤20 (tight20/40ellipse) or ≤40 (regular40/80ellipse). Merely reaching the projected endpoint is not sufficient. The control uses ordinary reference matching, not precisely the production local reverse checker.', '',
        'Convergence is conservative: ≥80base contiguous travel within10Euclidean base voxels and25° unsigned tangent agreement of saved wrong-branch segments; both paths must be outside regularGT40/80. The seed has no native match: diagnostic seed error is exactly zero by construction; the original segment first edge is conservatively excluded. No detection is not proof of no convergence.', '',
        'Run: set scripts/training_failure_mining_control.json to action=reviewed_reference_reverse and reviewed_model=gaussian or lowres, then `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. analysis_only=true summarizes available artifacts without running probes.', '',
        'For control rows, “within reference” means reproducing the saved **wrong route**, NOT correctness against GT. For GT rows it means following GT.', '',
        '| Model | Probe | Completed artifacts | Within tight supplied reference | Within regular supplied reference |',
        '|---|---|---:|---:|---:|']
    datasets = []
    for model in models:
        root = ROOT/'reference_reverse_diagnostic_v1'/model
        path = root/'results.json'
        if not path.exists():
            path = root/'partial_results.json'
        if not path.exists():
            continue
        data = read(path)
        data.setdefault('expected_probes', len(read(root/'probes.json')))
        data.setdefault('evaluated_probes', len(data['rows']))
        datasets.append((model, path, data))
        for kind in ('control','gt0','gt128','gt256'):
            rows = [r for r in data['rows'] if r['kind']==kind]
            n = len(rows)
            fmt = lambda key: f'{sum(r[key] for r in rows)}/{n} ({100*sum(r[key] for r in rows)/n:.1f}%)' if n else 'pending'
            lines.append(f'| {model} | {kind} | {n}/{data["expected_probes"]//4} | {fmt("tight_correct")} | {fmt("regular_correct")} |')
    for model,path,data in datasets:
        lines += ['', f'## {model}', '', f'Artifacts: `{path.parent}`. Results: `{path.name}`. SHA256 `{data["checkpoint_sha256"]}`.', '',
            f'{data["evaluated_probes"]}/{data["expected_probes"]} artifacts analyzed. Incomplete Gaussian probes are waiting for inference behind a non-pausable training job; do not treat partial denominators as the complete cohort.' if data['evaluated_probes']<data['expected_probes'] else 'All requested probes complete.', '',
            'Cells below are max anisotropic error in base voxels. † means the first segment did not reach the reference endpoint. — means pending. Event indices are zero-based.', '',
            '| Scroll | Original case/event | Control | GT0 | GT+128 | GT+256 |', '|---|---:|---:|---:|---:|---:|']
        groups = {}
        for r in data['rows']:
            groups.setdefault((r['scroll'],r['original_case'],r['event_index']),{})[r['kind']] = r
        for key,g in sorted(groups.items()):
            cells=[]
            for kind in ('control','gt0','gt128','gt256'):
                r=g.get(kind)
                cells.append('—' if r is None else f'{r.get("max_threshold_error_base",float("inf")):.1f}'+('' if r['complete'] else '†'))
            lines.append(f'| {key[0]} | {key[1]}/{key[2]} | '+ ' | '.join(cells)+' |')
        complete = [g for g in groups.values() if len(g)==4]
        both = sum(g['control']['tight_correct'] and any(g[k]['tight_correct'] for k in ('gt0','gt128','gt256')) for g in complete)
        control_failed = sum(not g['control']['tight_correct'] for g in complete)
        residual = len(complete) - both - control_failed
        convergence_count = sum(r.get('converged',False) for r in data['rows'])
        testable = sum(r.get('convergence_testable',False) for r in data['rows'])
        lines += ['', f'Complete four-probe groups: {len(complete)}; wrong-route control tight AND at least one GT probe tight: {both}. Sustained wrong-branch convergence: {convergence_count}/{testable} testable GT probes.', '',
            f'Tight diagnostic partition: {control_failed} fail the saved-wrong-route reverse control; {both} reproduce the wrong route but also have a correct GT-seeded alternative; {residual} reproduce the wrong route with no tight-correct GT probe. The last count is an observed diagnostic remainder, not a proven count of undetectable errors.', '',
            'Interpretation: independently reversible wrong and correct paths can coexist. This supports investigating competing-path evidence, but does not establish a usable ground-truth-free filter. No precision/recall claim is possible without controls on successful traces.']
        clipped = [f'{r["name"]}: requested+{r["requested_offset_base"]}, effective+{r["effective_offset_base"]:.2f}'
                   for r in data['rows'] if r['requested_offset_base'] is not None and
                   abs(r['effective_offset_base']-r['requested_offset_base']) > .01]
        lines += ['', 'Clipped offsets (base voxels; duplicate seeds are not independent evidence): '+('; '.join(clipped) or 'none')+'.']
    if not historical:
        from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import consistency_report
        metrics = {}
        lines += ['', '## Full-cohort periodic-filter comparison', '',
                  'Same 91 fibers / 182 directions. Filtering uses 256-base-voxel windows and a 20/40 ellipse; GT-assisted reseeding uses 10/20 with a local presence peak and direction alignment. Filtered stops are not ordinary restarts. Skipped regions receive no retained-length credit.', '',
                  '| Model / mode | Ordinary restarts | Filtered stops | Retained % | Skipped mm |',
                  '|---|---:|---:|---:|---:|']
        for model in models:
            for mode, folder in [('plain',ROOT/model),('filtered',ROOT/'reverse256_v1'/model)]:
                summary = read(folder/'full/summary.json')
                metrics[f'{model}/{mode}'] = consistency_report(summary)
                s = metrics[f'{model}/{mode}']['combined']
                lines.append(f'| {model}/{mode} | {s["ordinary_restarts"]} | {s["filtered_stops"]} | {s["retained_percent"]:.3f} | {s["filtered_skipped_mm"]:.3f} |')
        atomic_write(destination.with_suffix('.json'),metrics,durable=False)
        lines += ['', '### Current model per scroll', '',
                  '| Scroll | Plain ordinary | Filtered ordinary | Filtered stops | Retained % |',
                  '|---|---:|---:|---:|---:|']
        for scroll,s in metrics[f'{selected_model}/filtered']['scrolls'].items():
            before = metrics[f'{selected_model}/plain']['scrolls'][scroll]
            lines.append(f'| {scroll} | {before["ordinary_restarts"]} | {s["ordinary_restarts"]} | {s["filtered_stops"]} | {s["retained_percent"]:.3f} |')
    destination.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    control = read(REPO/'scripts/training_failure_mining_control.json')
    if control.get('analysis_only'):
        for model in (('gaussian','lowres') if control['reviewed_model']=='all' else (control['reviewed_model'],)):
            analyze(model, allow_partial=True)
    else:
        run(control['reviewed_model'])
