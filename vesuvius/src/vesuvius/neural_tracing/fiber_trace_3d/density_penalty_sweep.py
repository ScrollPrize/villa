"""Resumable direct search using the ordinary recorded crop benchmark runner."""
import copy
import csv
import io
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

from lasagna.inference_provenance import atomic_write, sha256_file


def mapping(density, scale, floor):
    if not all(math.isfinite(x) for x in (density, scale, floor)) or density < 0 or scale <= 0 or not 0 < floor <= 1:
        raise ValueError('invalid density mapping parameters')
    return max(floor, scale / (scale + density))


def coarse_grid():
    return list(itertools.product((.05, .5, 5.), (.01, .1, .5)))


def refine(scale, floor, tested, iteration):
    factor = 10 ** (.5 / (2 ** iteration))
    floors = sorted({f for _, f in tested} | {.001, .8})
    lower = max((f for f in floors if f < floor), default=floor / 10)
    upper = min((f for f in floors if f > floor), default=min(.99, (1 + floor) / 2))
    return [(scale / factor, floor), (scale * factor, floor),
            (scale, math.sqrt(lower * floor)), (scale, math.sqrt(upper * floor))]


def independent_summary_rows(log):
    """One scored gauge per reference; retain missing refs, ignore empty satellites."""
    rows=list(csv.DictReader(io.StringIO(log.split('independent_reference_summary\n')[1]
        .split('independent_reference_candidates')[0])))
    grouped={}
    for row in rows:
        grouped.setdefault(row['reference'],[]).append(row)
    result=[]
    for group in grouped.values():
        if len(group)==1:
            result.extend(group)
            continue
        active=[r for r in group if int(r.get('evaluated','0'))>0]
        empty=[r for r in group if int(r.get('evaluated','0'))==0]
        if len(active)>1 or any(int(r.get('admitted','0'))>0 for r in empty):
            raise ValueError('multiple independent gauges per reference require explicit reporting')
        result.append(active[0] if active else group[0])
    return result


def independent_metrics(log):
    """Read initial winding assignments without requiring a reference-assisted solve."""
    result = {}
    rows = [r for r in csv.DictReader(io.StringIO(log.split('independent_reference_candidates\n')[1].split('independent_reference_end')[0])) if r['rank'] == '1']
    summaries = independent_summary_rows(log)
    expected = {r['reference'] for r in summaries}
    if not expected:
        raise ValueError('empty independent reference summary')
    if not {r['reference'] for r in rows} <= expected:
        raise ValueError('unexpected independent reference candidate')
    if any(not math.isfinite(float(r[k])) for r in rows for k in ('reference', 'winding')):
        raise ValueError('nonfinite independent reference winding')
    if len(expected) != len(summaries) or len({r['reference'] for r in rows}) != len(rows):
        raise ValueError('multiple independent gauges per reference require explicit reporting')
    # A reference with no admitted evidence can have no gauge at all. Keep it
    # missing in the denominator, but never score an uncalibrated candidate.
    assigned = {r['reference'] for r in rows}
    if any(r['calibrated'] != '1' and
           (r.get('evaluated') != '0' or r['reference'] in assigned) for r in summaries):
        raise ValueError('uncalibrated independent reference gauge')
    result['ladder'] = [{k: r[k] for k in ('reference', 'winding', 'hard_sign_errors')} for r in rows]
    result['independent_percent'] = 100 * sum(float(r['reference']) == float(r['winding']) for r in rows) / len(expected)
    result['independent_missing'] = len(expected - {r['reference'] for r in rows})
    return result


def metrics(prefix):
    from .crop_benchmark import bp_metrics
    log = prefix.with_suffix('.log').read_text()
    result = bp_metrics(log)
    if not math.isfinite(result['constraint_score_percent']):
        raise ValueError('nonfinite removal score')
    result.update(independent_metrics(log))
    pieces = list(csv.DictReader(Path(str(prefix) + '_consistency.csv').open()))
    result['defect_percent'] = 100 * sum(r['winding_final_class'] == 'defect' for r in pieces) / len(pieces)
    result['wall_seconds'] = json.loads(prefix.with_suffix('.run.json').read_text())['wall_seconds']
    return result


def best_result(rows):
    # Reproducible ties prefer less penalty reduction, then stable name.
    return sorted(rows, key=lambda r: (-r['constraint_score_percent'], -r['mean_multiplier'], r['name']))[0]


def resumable(record, command, inputs, references, binary_hash, libraries, patch, saved_patch):
    return (record.get('returncode') == 0 and record.get('command') == command
            and record.get('inputs') == inputs and record.get('references') == references
            and record.get('binary_sha256') == binary_hash
            and record.get('shared_libraries') == libraries and patch == saved_patch)


def native_patch(patch):
    """Ignore mutable reports when checking native-source identity."""
    blocks = patch.split(b'diff --git ')
    return b''.join(b'diff --git ' + b for b in blocks[1:]
                    if b.startswith((b'a/volume-cartographer/core/', b'a/volume-cartographer/apps/')))


def run_sweep(config):
    from .crop_benchmark import run
    repo = Path(__file__).resolve().parents[5]
    cfg = config['density_sweep']
    template = json.loads(Path(cfg['template']).read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    binaries = repo / 'volume-cartographer/build/fiber-evaluation/bin'
    binary = binaries / 'vc_fiber_trace_chunk'
    binary_hash = sha256_file(binary)
    libraries = {str(p): sha256_file(p) for p in sorted((binaries.parent / 'core').glob('libvc_fiber_tracer.*')) if p.is_file()}
    patch = native_patch(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'volume-cartographer'], cwd=repo))
    references = {str(p): sha256_file(p) for p in sorted(Path(template['reference_dir']).glob('*.json'))}
    original = {mode: json.loads(Path(template['commands'][i + 1]['args'][-2]).read_text())
                for i, mode in enumerate(('control', 'mean', 'median', 'max'))}
    trace_files = sorted(str(p) for p in Path(original['control']['trace_artifact']).rglob('*') if p.is_file())
    if not trace_files:
        raise ValueError('empty trace artifact')

    def trial(mode, scale, floor):
        name = 'control' if mode == 'control' else f'{mode}-s{scale:.10g}-f{floor:.10g}'
        # Prefixes deliberately have decimal points: append suffixes, never replace them.
        name = name.replace('.', 'p')
        prefix = root / name
        doc = copy.deepcopy(original[mode])
        for row in doc['pieces']:
            row['multiplier'] = 1. if mode == 'control' else mapping(max(0., row['density']), scale, floor)
        doc['mapping'] = dict(scale=scale, floor=floor, formula='max(floor,scale/(scale+d))')
        inp = root / (name + '-input.json')
        atomic_write(inp, doc, durable=False)
        case = copy.deepcopy(template['commands'][1])
        case['name'] = name
        args = case['args']
        for key, value in (('--output', str(prefix)), ('--threads', str(cfg.get('threads', 32))), ('--winding-defect-multipliers', str(inp))):
            args[args.index(key) + 1] = value
        files = [p for p in template['input_files'] if not p.endswith(('control.json', 'mean.json', 'median.json', 'max.json'))] + trace_files + [str(inp)]
        local = dict(output_root=str(root), reference_dir=template['reference_dir'], input_files=files, commands=[case])
        atomic_write(root / (name + '-config.json'), local, durable=False)
        inputs = {p: sha256_file(p) for p in files}
        command = [str(binary), *args]
        record_path = prefix.with_suffix('.run.json')
        reuse = False
        if record_path.exists() and prefix.with_suffix('.source.patch').exists():
            record = json.loads(record_path.read_text())
            reuse = resumable(record, command, inputs, references, binary_hash, libraries,
                              patch, native_patch(prefix.with_suffix('.source.patch').read_bytes()))
        if not reuse:
            run(local)
        result = metrics(prefix)
        result.update(mode=mode, scale=scale, floor=floor, name=name, reused=reuse,
                      mean_multiplier=sum(r['multiplier'] for r in doc['pieces']) / len(doc['pieces']))
        print('density sweep result: ' + json.dumps({k: v for k, v in result.items() if k != 'ladder'}), flush=True)
        return result

    control = trial('control', 1., 1.)
    old = Path(cfg['previous_control'])
    previous = metrics(old)
    if any(control[k] != previous[k] for k in ('constraint_score_percent', 'independent_percent', 'defect_percent', 'ladder')):
        raise RuntimeError('thread-control result differs; retain 128 threads before continuing')
    if Path(str(root / 'control') + '_winding_factors.csv').read_bytes() != Path(str(old) + '_winding_factors.csv').read_bytes():
        raise RuntimeError('control factor mismatch')
    def states(prefix):
        with Path(str(prefix) + '_consistency.csv').open() as stream:
            return [tuple(r[k] for k in ('piece', 'original_trace', 'begin_arc_base', 'end_arc_base', 'winding_prepass_class', 'winding_final_class', 'winding_output', 'winding_valid')) for r in csv.DictReader(stream)]
    if states(root / 'control') != states(old):
        raise RuntimeError('control piece-state mismatch')
    def removal(prefix):
        return sorted(line for p in prefix.parent.glob(prefix.name + '_oracle_fresh_*.obj') for line in p.read_text().splitlines() if line.startswith('o '))
    if removal(root / 'control') != removal(old):
        raise RuntimeError('control removal partition mismatch')
    results = [control]
    tested = {mode: set() for mode in ('mean', 'median', 'max')}

    def persist(stage):
        best = {mode: best_result([control] + [r for r in results if r['mode'] == mode]) for mode in tested}
        atomic_write(root / 'summary.json', dict(stage=stage, results=results, best_including_control=best), durable=False)

    def batch(jobs, stage):
        with ThreadPoolExecutor(max_workers=cfg.get('workers', 4)) as pool:
            futures = {pool.submit(trial, mode, s, f): (mode, s, f) for mode, s, f in jobs}
            for future in as_completed(futures):
                result = future.result()
                mode, s, f = futures[future]
                tested[mode].add((s, f))
                results.append(result)
                persist(stage)

    batch([(mode, s, f) for s, f in coarse_grid() for mode in tested], 'coarse')
    for iteration in range(2):
        jobs = []
        for mode in tested:
            winner = best_result([r for r in results if r['mode'] == mode])
            jobs.extend((mode, s, f) for s, f in refine(winner['scale'], winner['floor'], tested[mode], iteration) if (s, f) not in tested[mode])
        batch(jobs, f'refine-{iteration + 1}')
    persist('complete')
