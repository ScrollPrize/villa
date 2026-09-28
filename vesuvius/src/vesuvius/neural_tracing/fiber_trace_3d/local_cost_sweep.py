"""Recorded joint-grid piece-disable / continuation-cut experiments."""
import copy
import csv
import itertools
import json
import re
from pathlib import Path


def matrix():
    return [dict(parallel=p, defect=d, cut=c) for p, d, c in
            itertools.product((True, False), (1, 10, 100), ('hard', .1, 1, 10))]


def trial_config(template, root, point, threads):
    if point.get('options') and not point.get('variant'):
        raise ValueError('option overrides require a distinct variant name')
    case = copy.deepcopy(template['commands'][0])
    name = f"{'parallel' if point['parallel'] else 'orthogonal'}-d{point['defect']}-c{point['cut']}".replace('.', 'p')
    if point.get('fixed_orientation'):
        name += '-fixed'
    if point.get('distance_cost'):
        name += '-distance'
    if point.get('local_events') is False:
        name += '-legacy'
    if point.get('screen_only'):
        name += '-screen'
    if point.get('variant'):
        name += '-' + point['variant']
    args = case['args']
    if point.get('screen_only'):
        for flag, count in (('--reference-prune-offenders', 0),
                            ('--reference-prune-policy', 1),
                            ('--reference-oracle-magnitude-weight', 1),
                            ('--reference-oracle-accept-message-limit', 0),
                            ('--reference-oracle-rounds', 1)):
            if flag in args:
                index = args.index(flag)
                del args[index:index + count + 1]
    if point.get('local_events') is False and '--local-constraint-events' in args:
        args.remove('--local-constraint-events')
    if '--winding-fixed-orientation' in args and not point.get('fixed_orientation'):
        args.remove('--winding-fixed-orientation')
    if point.get('fixed_orientation') and '--winding-fixed-orientation' not in args:
        args.append('--winding-fixed-orientation')
    for key, value in (('--output', str(Path(root) / name)), ('--threads', str(threads))):
        args[args.index(key) + 1] = value
    args += ['--winding-defect-cost', str(point['defect'])]
    if point['cut'] != 'hard':
        args += ['--winding-continuity-step-cost' if point.get('distance_cost') else '--winding-continuity-cut-cost', str(point['cut'])]
    if not point['parallel'] and point.get('local_events') is not False:
        args += ['--local-no-parallel']
    for flag, value in point.get('options', {}).items():
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args += [flag, str(value)]
    case['name'] = name
    return dict(output_root=str(root), reference_dir=template['reference_dir'],
                input_files=template['input_files'], commands=[case])


def audit_events(prefix):
    groups = {}
    with Path(str(prefix) + '_winding_factors.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['local_event'] != '1' or float(row['perpendicular']) <= float(row['parallel']):
                continue
            for side in ('a', 'b'):
                groups.setdefault(row['piece_' + side], []).append(float(row['arc_' + side]))
    counts = []
    for arcs in groups.values():
        unique = []
        for arc in sorted(arcs):
            if not unique or arc - unique[-1] > 1e-6:
                unique.append(arc)
        counts.append(len(unique))
    return dict(pieces_with_orthogonal_events=len(counts), multiple_event_pieces=sum(n > 1 for n in counts),
                maximum_event_groups=max(counts, default=0),
                multiple_orthogonal_factor_pieces=sum(len(arcs) > 1 for arcs in groups.values()),
                maximum_orthogonal_factors=max((len(arcs) for arcs in groups.values()), default=0))


def ladder_shift_metrics(ladder):
    ordered = sorted(ladder, key=lambda row: float(row['reference']))
    if not ordered:
        return dict(ladder_shifts=None, ladder_shift_percent=None)
    offsets = [float(row['winding']) - float(row['reference']) for row in ordered]
    shifts = sum(a != b for a, b in zip(offsets, offsets[1:]))
    return dict(ladder_shifts=shifts, ladder_shift_percent=100 * (1 - shifts / len(ordered)))


def validate_certified_counts(audit):
    if 'certified' not in audit:
        return
    if audit['retained_certified'] + audit.get('parallel_omitted', 0) != audit['certified']:
        raise ValueError('splitting lost certified constraints')
    if audit.get('support_rejected', 0):
        raise ValueError('certified constraints rejected for split support')


def collect(prefix, screen_only=False):
    from .density_penalty_sweep import metrics, independent_metrics
    from .crop_benchmark import bp_metrics
    log = prefix.with_suffix('.log').read_text()
    if screen_only:
        result = independent_metrics(log)
        result.update(constraint_score_percent=None, reference_score_percent=None)
    else:
        try:
            result = metrics(prefix)
        except (ValueError, IndexError, KeyError) as error:
            result = bp_metrics(log)
            result.update(metric_warning=str(error), independent_percent=None)
    states = list(csv.DictReader(Path(str(prefix) + '_consistency.csv').open()))
    from collections import Counter
    result['prepass_classes'] = dict(Counter(r.get('winding_prepass_class', 'unknown') for r in states))
    result['final_classes'] = dict(Counter(r['winding_final_class'] for r in states))
    result['prepass_hv_ties'] = sum(float(r.get('p_h', 0)) == float(r.get('p_v', 0)) > float(r.get('p_mixed', 0)) for r in states)
    result['defect_percent'] = 100 * sum(r['winding_final_class'] == 'defect' for r in states) / len(states)
    cuts = re.findall(r'continuity_cut eligible=(\d+) map_cut=(\d+) energy=([^\s]+)', log)
    result['cut_reports'] = [dict(eligible=int(a), cuts=int(b), energy=float(c),
                                  cut_percent=100 * int(b) / int(a) if int(a) else 0.) for a, b, c in cuts]
    steps = re.findall(r'continuity_step eligible=(\d+) nonzero=(\d+) energy=([^\s]+)', log)
    result['step_reports'] = [dict(eligible=int(a), nonzero=int(b), energy=float(c),
                                   nonzero_percent=100 * int(b) / int(a) if int(a) else 0.) for a, b, c in steps]
    result['decode_repair'] = [{k: float(v) for k, v in re.findall(r'(\w+)=([^\s]+)', line)}
                               for line in log.splitlines() if line.startswith('decode_repair ')]
    result['wall_seconds'] = json.loads(prefix.with_suffix('.run.json').read_text())['wall_seconds']
    result['winding_statuses'] = re.findall(r'fiber winding BP\n[^\n]*\n(\w+)', log)
    result['initial_readout_status'] = re.findall(r'frozen_initial_bp=1[^\n]*bp_status=(\w+)', log)
    result['provenance'] = json.loads(prefix.with_suffix('.run.json').read_text())
    result['event_audit'] = audit_events(prefix)
    result['extraction_audits'] = [{k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                                  for line in log.splitlines() if line.startswith('local constraint events:')]
    for audit in result['extraction_audits']:
        validate_certified_counts(audit)
    result['reference_source_audits'] = [{k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                                        for line in log.splitlines() if line.startswith('reference cross source events:')]
    for audit in result['reference_source_audits']:
        if audit['certified'] != audit['retained'] + audit['missing_owner']:
            raise ValueError('reference remapping lost certified events')
    result.update(ladder_shift_metrics(result.get('ladder', [])))
    if result.get('independent_missing'):
        result.update(ladder_shifts=None, ladder_shift_percent=None,
                      metric_warning='incomplete ladder: shift score not comparable')
    return result


def run_sweep(config):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from lasagna.inference_provenance import atomic_write, sha256_file
    from .crop_benchmark import run
    from .density_penalty_sweep import native_patch, resumable
    cfg = config['local_cost_sweep']
    if cfg.get('report_only'):
        if cfg.get('screen_report'):
            return publish_screen_report(config)
        return publish_report(config)
    template = json.loads(Path(cfg['template']).read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[5]
    binaries = repo / 'volume-cartographer/build/fiber-evaluation/bin'
    binary = binaries / 'vc_fiber_trace_chunk'
    binary_hash = sha256_file(binary)
    libraries = {str(p): sha256_file(p) for p in sorted((binaries.parent / 'core').glob('libvc_fiber_tracer.*')) if p.is_file()}
    patch = native_patch(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'volume-cartographer'], cwd=repo))
    references = {str(p): sha256_file(p) for p in sorted(Path(template['reference_dir']).glob('*.json'))}
    trace = Path(template['commands'][0]['args'][1])
    files = [str(p) for p in sorted(trace.rglob('*')) if p.is_file()]
    if not files:
        raise ValueError('empty source trace artifact')
    template['input_files'] = sorted(set(template['input_files'] + files))
    inputs = {p: sha256_file(p) for p in template['input_files']}

    def trial(point):
        local = trial_config(template, root, point, cfg.get('threads', 48))
        case = local['commands'][0]
        name = case['name']
        prefix = root / name
        atomic_write(root / (name + '-config.json'), local, durable=False)
        command = [str(binary), *case['args']]
        reuse = False
        if prefix.with_suffix('.run.json').exists() and prefix.with_suffix('.source.patch').exists():
            reuse = resumable(json.loads(prefix.with_suffix('.run.json').read_text()), command,
                              inputs, references, binary_hash, libraries, patch,
                              native_patch(prefix.with_suffix('.source.patch').read_bytes()))
        result = dict(name=name, **point, reused=reuse)
        try:
            if not reuse:
                run(local)
            result.update(collect(prefix, point.get('screen_only', False)), status='complete',
                          geometry_sha256=sha256_file(Path(str(prefix) + '_local_event_pieces.csv')) if point.get('local_events') is not False else None)
        except (ValueError, RuntimeError, IndexError, KeyError, OSError, subprocess.CalledProcessError) as error:
            result.update(status='failed', error=str(error))
        print('local cost result: ' + json.dumps(result), flush=True)
        return result

    points = cfg.get('points', matrix())
    names = {trial_config(template, root, p, cfg.get('threads', 48))['commands'][0]['name'] for p in points}
    if len(names) != len(points):
        raise ValueError('duplicate trial names would overwrite benchmark artifacts')
    results = []
    if (root / 'summary.json').exists():
        results = [r for r in json.loads((root / 'summary.json').read_text())['results'] if r['name'] not in names]
    total = len(results) + len(points)
    with ThreadPoolExecutor(max_workers=cfg.get('workers', 4)) as pool:
        futures = [pool.submit(trial, p) for p in points]
        for future in as_completed(futures):
            results.append(future.result())
            hashes = {r['geometry_sha256'] for r in results if r['status'] == 'complete' and r.get('geometry_sha256')}
            if len(hashes) > 1:
                raise RuntimeError('pre-BP geometry changed between cost trials')
            atomic_write(root / 'summary.json', dict(results=sorted(results, key=lambda r: r['name']),
                         complete=len(results) == total), durable=False)


def publish_screen_report(config):
    """Persist screening ladders without implying reference-assisted scores exist."""
    from html import escape
    from lasagna.inference_provenance import atomic_write
    cfg = config['local_cost_sweep']
    roots = [config['output_root'], *cfg.get('supplemental_roots', [])]
    rows = []
    complete = True
    pending_roots = []
    for root in roots:
        summary = Path(root) / 'summary.json'
        if not summary.exists():
            pending_roots.append(root)
            complete = False
            continue
        document = json.loads(summary.read_text())
        complete = complete and document['complete']
        rows.extend(dict(r, artifact_root=root) for r in document['results'])
    for row in rows:
        if row['status'] == 'complete' and cfg.get('reparse_metrics', True):
            row.update(collect(Path(row['artifact_root']) / row['name'], row.get('screen_only', False)))
    repo = Path(__file__).resolve().parents[5]
    destination = repo / cfg['report_destination']
    lines = ['# Winding-step continuation search', '',
             'Same frozen Paris4 2048 traces and26reference windings. Finite continuation energy is lambda*abs(winding displacement), including H/V half steps. No inference or retracing;500messages, message-limit accepted. Screening omits only reference-assisted solves. Null scores mean not measured, never zero.', '',
             'Legacy grouped baseline:38.4615% initial accuracy,73.0769% shift score. Higher is better. Shift score=100*(1-K/26), K adjacent changes in predicted-minus-reference offset; global offset is ignored. All trials and failures retained below. No default promotion.', '',
             '| Trial | Initial accuracy | Shift score | Steps nonzero | Seconds |', '|---|---:|---:|---:|---:|']
    def pct(v):
        return 'n/a' if v is None else f'{v:.4f}%'
    for r in rows:
        if r['status'] != 'complete':
            lines.append(f"| {r['name']} | FAILED:{r.get('error')} | — | — | — |")
            continue
        step = (r.get('step_reports') or [{}])[0]
        lines.append(f"| {r['name']} | {pct(r.get('independent_percent'))} | {pct(r.get('ladder_shift_percent'))} | {pct(step.get('nonzero_percent'))} | {r['wall_seconds']:.1f} |")
    full = [r for r in rows if r['status'] == 'complete' and not r.get('screen_only', False)]
    if full:
        lines += ['', '## Full benchmark (not screening)', '',
                  'Reference-assisted accuracy is not deployment accuracy. Removal is negative problematic/fulfilled percent (closer to zero is better); different graph sizes make legacy a contextual rather than matched removal comparison.', '',
                  '| Trial | Initial accuracy | Shift score | Reference-assisted accuracy | Removal score |',
                  '|---|---:|---:|---:|---:|']
        for r in full:
            lines.append(f"| {r['name']} | {pct(r.get('independent_percent'))} | {pct(r.get('ladder_shift_percent'))} | {pct(r.get('reference_score_percent'))} | {pct(r.get('constraint_score_percent'))} |")
    lines += ['', '## Recorded settings and cohorts', '']
    for r in rows:
        if r['status'] == 'complete':
            lines.append(f"- {r['name']}: options={r.get('options', {})}; prepass={r.get('prepass_classes', {})}; H/V ties={r.get('prepass_hv_ties')}; final={r.get('final_classes', {})}; missing references={r.get('independent_missing')}; retention={r.get('extraction_audits', [])}; decoder={r.get('decode_repair', [])}.")
    lines += ['', '## Provenance', '', 'JSON preserves exact commands, hashes, ladders and certified-retention audits for every row. Four concurrent runs ×48threads unless the recorded command says otherwise; timings are contended, not standalone throughput. Native build: bash vesuvius/scripts/run_fiber3d_native_build. Evaluation entrypoint: /home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config CONFIG.', '', *[f'- `{r}`' for r in roots]]
    if pending_roots:
        lines += ['', 'Pending roots (no completed summary yet):', *[f'- `{r}`' for r in pending_roots]]
    atomic_write(destination.with_suffix('.json'), dict(complete=complete, results=rows, pending_roots=pending_roots), durable=False)
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{60+len(rows)*27}"><g font-family="sans-serif" font-size="12"><text x="10" y="20">Winding step search: initial accuracy (green), shift score (blue), reference-assisted accuracy if measured (purple), all percent</text>']
    for i, r in enumerate(rows):
        y = 45+i*27
        svg.append(f'<text x="10" y="{y+10}">{escape(r["name"])}</text>')
        for j, (key, color) in enumerate((('independent_percent', '#289b65'), ('ladder_shift_percent', '#3984cf'), ('reference_score_percent', '#9759b5'))):
            if r.get(key) is not None:
                x = 650+j*300
                svg.append(f'<rect x="{x}" y="{y}" width="{r[key]*2}" height="14" fill="{color}"/><text x="{x+205}" y="{y+11}">{r[key]:.2f}%</text>')
    destination.with_suffix('.svg').write_text('\n'.join(svg)+ '</g></svg>\n')


def publish_report(config):
    """Regenerate tables from recorded runs, never rerun native evaluation."""
    from html import escape
    from lasagna.inference_provenance import atomic_write, sha256_file
    root = Path(config['output_root'])
    document = json.loads((root / 'summary.json').read_text())
    document['summary_hashes'] = {str(root / 'summary.json'): sha256_file(root / 'summary.json')}
    for row in document['results']:
        row['artifact_root'] = str(root)
    for extra in config['local_cost_sweep'].get('supplemental_roots', []):
        path = Path(extra) / 'summary.json'
        if not path.exists():
            document['complete'] = False
            continue
        supplement = json.loads(path.read_text())
        document['summary_hashes'][str(path)] = sha256_file(path)
        document['complete'] = document['complete'] and supplement['complete']
        document['results'].extend(dict(r, artifact_root=extra) for r in supplement['results'])
    for row in document['results']:
        if row['status'] == 'complete':
            row.update(collect(Path(row['artifact_root']) / row['name']))
    if len({r['geometry_sha256'] for r in document['results'] if r['status'] == 'complete' and r.get('geometry_sha256')}) > 1:
        raise RuntimeError('geometry differs between joint and fixed-orientation trials')
    for key in ('inputs', 'references', 'binary_sha256', 'shared_libraries'):
        if len({json.dumps(r['provenance'][key], sort_keys=True) for r in document['results'] if r['status'] == 'complete'}) > 1:
            raise RuntimeError(f'{key} differs between recorded trials')
    repo = Path(__file__).resolve().parents[5]
    destination = repo / config['local_cost_sweep'].get('report_destination', 'volume-cartographer/docs/fiber_benchmark_runs/2026-09-27-local-costs')
    baselines = json.loads(destination.with_name('2026-09-27-local-events-results.json').read_text())['results']
    document['baselines'] = baselines
    document['historical_control_checks'] = {}
    for mode, old_name in (('parallel', 'local-parallel'), ('orthogonal', 'local-orthogonal')):
        current = next((r for r in document['results'] if r['name'] == f'{mode}-d100-chard-fixed' and r['status'] == 'complete'), None)
        old = next(r for r in baselines if r['name'] == old_name)
        if current:
            document['historical_control_checks'][mode] = {
                key: current.get(key) == old.get(key) for key in
                ('constraint_score_percent', 'independent_percent', 'reference_score_percent', 'defect_percent', 'ladder')}
    document['summary_sha256'] = sha256_file(root / 'summary.json')
    atomic_write(destination.with_suffix('.json'), document, durable=False)
    lines = ['# Local-event piece-disable and continuation-cut cost trials', '',
             f"Recorded trials: {len(document['results'])}; requested batch complete: {document['complete']}.", '',
             'Same frozen Paris4 2048 beam8 / 40×80 / 45° / reverse-consistent traces and 26 reference fibers as the local-event benchmark. No inference or retracing. Native RelWithDebInfo; one run per setting. Primary matrix: four concurrent × 48 threads; supplemental fixed-H/V matrix: two concurrent × 48 threads, overlapping the primary tail. Timings include contention and are not isolated speed benchmarks. Normal alignment and winding limits both 500.', '',
             'The primary trials jointly solve H/V and winding; supplementary fixed-H/V trials isolate the new connection choice under the previous orientation policy. Compare finite cuts against the hard row at the same piece cost, orientation mode and parallel setting.', '',
             'Legacy segmentation has different constraint/continuity counts, so its removal score is context rather than a directly matched cost comparison. JSON records whether rerun fixed-H/V hard controls exactly reproduce the historical local scores and ladders.', '',
             'Piece cost D is the existing Defect unary per incident evidence mass, not an absolute per-piece penalty. Connection cost C is an absolute cost per active-active continuation cut. Keeping a connection requires equal orientation and integer winding. A cut preserves both pieces and ordinary constraints. Defect endpoints incur no extra cut charge. This is a minimized binary/Potts choice, not a marginalized latent switch. Production defaults unchanged.', '',
             'Removal score = −100 × problematic / fulfilled constraints (closer to zero is better). Cuts count as problematic, not free successes. Initial accuracy is independent reference readout from the unconditioned solve. Conditioned accuracy is the established removal/oracle metric and is not a deployment result. Defects and cuts below refer to the initial solve; cut percentage uses active-active continuation links as denominator.', '',
             '| Variant | Piece cost | Cut cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Cut links | Time (s) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if any(r.get('distance_cost') for r in document['results']):
        lines = ['# Distance-weighted continuation experiment', '',
                 'Joint H/V, parallel constraints off, D=10. Same frozen Paris4 2048 traces and 26 references; 500-iteration budgets. Four concurrent runs × 48 threads, one repetition, RelWithDebInfo. No inference or retracing. All six settings, including controls, use the same new binary.', '',
                 'Distance rows use C*abs(q_b-q_a), where q includes calibrated H/V phase. This is an untruncated soft pair potential, not a removable connection or a flat cut. Ordinary evidence remains unchanged. Defect endpoints pay their usual piece unary and no continuation charge. The existing cut diagnostic counts unequal active endpoints; in distance mode these remain coupled during BP. Reference-conditioned inlier selection still treats unequal continuity as problematic.', '',
                 'Initial accuracy measures independent reference readout of the unconditioned solve; conditioned accuracy uses reference-based removal. Removal is negative problematic/fulfilled percentage, nearer zero better, not winding correctness. Production defaults unchanged.', '',
                 '| Variant | Piece cost | Cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Unequal links | Time (s) |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if config['local_cost_sweep'].get('certified_splits'):
        lines = ['# Certified constraints preserved across splitting', '',
                 'Correction: certified measurements, signs, support and source-pair-normalized evidence weights survive splitting unchanged. Only owning piece IDs change, plus continuation links and explicitly requested parallel-factor omission. Previous local/cost experiments used an erroneous post-split support filter; they are historical broken-graph diagnostics, not tests of the intended algorithm.', '',
                 'Same frozen Paris4 2048 beam8/40x80/45-degree/reverse-consistent traces, 26 references and 500-message budget. Message-limit termination is accepted completion. Four simultaneous processes x48 threads, RelWithDebInfo, one run per setting; timings are contended. No inference or retracing.', '',
                 'Compare legacy grouped baseline and fixed-H/V local hard controls at D100; then joint-H/V parallel-off D10 hard, flat-C10 and linear-distance10. Initial = unconditioned reference readout; conditioned = reference-assisted benchmark. Removal = negative problematic/fulfilled %, nearer zero better, not correctness.', '',
                 '| Variant | Piece cost | Cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Unequal links | Time (s) |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def pct(value):
        return 'n/a' if value is None else f'{value:.4f}%'
    for row in baselines:
        lines.append(f"| Previous {row['name']} (fixed H/V) | 100 | hard | {pct(row['constraint_score_percent'])} | {pct(row['independent_percent'])} | {pct(row['reference_score_percent'])} | {pct(row['defect_percent'])} | — | {row['wall_seconds']:.1f} |")
    for row in document['results']:
        if row['status'] != 'complete':
            lines += [f"| {row['name']} | — | — | FAILED: {row.get('error')} | — | — | — | — | — |"]
            continue
        cuts = row['cut_reports'][0]['cut_percent'] if row['cut_reports'] else 0.
        mode = 'fixed H/V' if row.get('fixed_orientation') else 'joint H/V'
        if row.get('local_events') is False:
            mode += ' / legacy grouped'
        if row.get('distance_cost'):
            mode += ' / linear distance'
        lines.append(f"| {'Parallel on' if row['parallel'] else 'Orthogonal only'} / {mode} | {row['defect']} | {row['cut']} | {pct(row['constraint_score_percent'])} | {pct(row.get('independent_percent'))} | {pct(row['reference_score_percent'])} | {pct(row['defect_percent'])} | {pct(cuts)} | {row['wall_seconds']:.1f} |")
    lines += ['', '## Unconditioned ladder shifts', '',
              'Count adjacent changes in predicted-minus-reference offset, ignoring a global offset. Shift score = 100*(1-K/N); one suffix shift in 26 references gives 96.15%. Shifts have arbitrary magnitude, not unit-winding cost.', '',
              '| Trial | Shifts | Shift score |', '|---|---:|---:|']
    for row in document['results']:
        if row['status'] == 'complete':
            lines.append(f"| {row['name']} | {row.get('ladder_shifts')} | {pct(row.get('ladder_shift_percent'))} |")
    if config['local_cost_sweep'].get('certified_splits'):
        lines += ['', '## Certified-event retention audit', '',
                  '| Trial | Certified | Retained | Parallel omitted | Support rejected | Reference certified | Reference retained | Missing owners |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in document['results']:
            if row['status'] != 'complete' or not row.get('extraction_audits'):
                continue
            a = row['extraction_audits'][0]
            b = (row.get('reference_source_audits') or [{}])[0]
            lines.append(f"| {row['name']} | {a.get('certified')} | {a.get('retained_certified')} | {a.get('parallel_omitted')} | {a.get('support_rejected')} | {b.get('certified')} | {b.get('retained')} | {b.get('missing_owner')} |")
        ladder_rows = [r for r in document['results'] if r['status'] == 'complete']
        ladders = [{float(p['reference']): p['winding'] for p in r.get('ladder', [])} for r in ladder_rows]
        references = sorted({w for ladder in ladders for w in ladder})
        lines += ['', '## Unconditioned reference ladders', '',
                  '| Reference | ' + ' | '.join(r['name'] for r in ladder_rows) + ' |',
                  '|---:' + '|---:' * len(ladder_rows) + '|']
        for reference in references:
            lines.append(f'| {reference:g} | ' + ' | '.join(str(ladder.get(reference, 'n/a')) for ladder in ladders) + ' |')
    config_names = (['crop_benchmark_local_step_costs_20260927.json', 'crop_benchmark_local_step_costs_report_20260927.json']
                    if any(r.get('distance_cost') for r in document['results']) else
                    ['crop_benchmark_local_costs_20260927.json', 'crop_benchmark_local_costs_fixed_20260927.json', 'crop_benchmark_local_costs_report_20260927.json'])
    if config['local_cost_sweep'].get('certified_splits'):
        config_names = ['crop_benchmark_certified_splits_20260927.json', 'crop_benchmark_certified_splits_report_20260927.json']
    commands = [f'/home/hendrik/.venv_las/bin/python {repo}/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config {Path(__file__).parent}/configs/{name}' for name in config_names]
    lines += ['', '## Reproduce', '', '```sh', *commands,
              '```', '', f'Artifacts: `{root}`. Every trial has its exact command, input/reference hashes, binary/library hashes, patch, CSVs and log. Resume requires matching provenance. JSON includes ladders when the single-gauge readout is valid, convergence statuses and distinct-event audits; otherwise an explicit metric warning replaces initial accuracy. SVG shows removal score and initial accuracy.', '',
              '## Audit and limitations', '',
              'This is loopy sum-product BP followed by marginal-mode decoding and hard-sign projection, not an exact joint MAP optimizer. Cost changes need not produce monotonic changes in the decoded solution. Flat-cut choices are minimized conditional on endpoint states; distance costs retain a soft coupling. Neither full assignment is globally optimized. See per-run convergence statuses below.', '']
    for row in document['results']:
        if row['status'] == 'complete':
            lines.append(f"- {row['name']}: statuses {row['winding_statuses']}; event audit {row['event_audit']}; certified extraction {row.get('extraction_audits', [])}; reference source audit {row.get('reference_source_audits', [])}." + (f" Metric warning: {row['metric_warning']}." if 'metric_warning' in row else ''))
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    rows = [*baselines, *(r for r in document['results'] if r['status'] == 'complete')]
    score_scale = 300 / max(1., *(abs(r['constraint_score_percent']) for r in rows))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{70 + len(rows)*26}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="12"><text x="10" y="20">Local BP costs — removal score (blue; nearer zero better), independent accuracy (green), conditioned accuracy (orange)</text>']
    for i, row in enumerate(rows):
        y = 48 + i*26
        score = row['constraint_score_percent']
        acc = row.get('independent_percent')
        svg += [f'<text x="10" y="{y}">{escape(row["name"])}</text>',
                f'<rect x="310" y="{y-10}" width="{abs(score)*score_scale}" height="10" fill="#4575b4"/>',
                f'<text x="615" y="{y}">{pct(score)}</text>']
        if acc is not None:
            svg += [f'<rect x="730" y="{y-10}" width="{acc*2}" height="10" fill="#16855b"/>', f'<text x="940" y="{y}">{pct(acc)}</text>']
        conditioned = row['reference_score_percent']
        svg += [f'<rect x="1020" y="{y-10}" width="{conditioned*2}" height="10" fill="#dc8431"/>', f'<text x="1240" y="{y}">{pct(conditioned)}</text>']
    svg += ['</g></svg>']
    destination.with_suffix('.svg').write_text('\n'.join(svg))
