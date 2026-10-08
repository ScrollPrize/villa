"""Controlled alternative retention experiments through the shared crop runner."""
import copy
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from lasagna.inference_provenance import atomic_write


def measurement_counts(path, count):
    counts = np.zeros(count)
    with Path(path).open() as stream:
        for r in csv.DictReader(stream):
            active = any(float(r[k]) > 0 for k in (
                'effective_parallel_winding_weight', 'effective_perpendicular_winding_weight',
                'effective_parallel_sign_penalty', 'effective_perpendicular_sign_penalty',
                'hard_parallel_sign', 'hard_perpendicular_sign'))
            if active:
                counts[int(r['piece_a'])] += 1
                counts[int(r['piece_b'])] += 1
    return counts


def centered(values, weights, power):
    v = np.asarray(values, dtype=float) ** power
    if not np.all(np.isfinite(v)) or np.any(v <= 0) or weights.sum() <= 0:
        raise ValueError('invalid centered weights')
    return v / np.average(v, weights=weights)


def execute_case(template, root, name, document, extra=()):
    from .crop_benchmark import run
    from .density_penalty_sweep import metrics
    case = copy.deepcopy(template['commands'][1])
    case['name'] = name
    inp = root / (name + '-input.json')
    atomic_write(inp, document, durable=False)
    args = case['args']
    for key, value in (('--output', str(root / name)), ('--threads', '32'), ('--winding-defect-multipliers', str(inp))):
        args[args.index(key) + 1] = value
    args.extend(extra)
    if '--piece-cut-hints' in args:
        index = args.index('--winding-defect-multipliers')
        del args[index:index + 2]
    files = [p for p in template['input_files'] if not p.endswith(('control.json', 'mean.json', 'median.json', 'max.json'))]
    files += [str(p) for p in Path(document['trace_artifact']).rglob('*') if p.is_file()]
    if '--piece-cut-hints' in args:
        files.append(args[args.index('--piece-cut-hints') + 1])
    config = dict(output_root=str(root), reference_dir=template['reference_dir'], input_files=files + [str(inp)], commands=[case])
    atomic_write(root / (name + '-config.json'), config, durable=False)
    run(config)
    result = metrics(root / name)
    log = (root / (name + '.log')).read_text()
    result['measurement_disable'] = [dict(stage=m[0] or 'initial', eligible=int(m[1]), map_off=int(m[2]),
                                         conditional_posterior_sum=float(m[3])) for m in re.findall(
        r'(?:(conditioned|fresh)_)?measurement_disable eligible=(\d+) map_off=(\d+) conditional_posterior_sum=([\d.eE+\-]+)', log)]
    disabled = re.findall(r'explicitly_disabled_retained_constraints=(\d+)', log)
    result['retained_disabled_constraints'] = int(disabled[-1]) if disabled else 0
    result.update(name=name, mapping=document['mapping'])
    return result


def sustained_cuts(segments, threshold, minimum_length):
    density = segments[:, 2] / (segments[:, 1] - segments[:, 0]) * segments[:, 4]
    good = density >= threshold
    starts = np.flatnonzero(good & np.r_[True, ~good[:-1]])
    ends = np.flatnonzero(good & np.r_[~good[1:], True])
    length = float(segments[-1, 1])
    candidates = sorted({float(x) for a, b in zip(starts, ends)
                         if segments[b, 1] - segments[a, 0] >= minimum_length
                         for x in (segments[a, 0], segments[b, 1]) if 32 <= x <= length - 32})
    cuts = []
    for x in candidates:
        if not cuts or x - cuts[-1] >= 32:
            cuts.append(x)
    return cuts


def relocated_cuts(cuts, length, rng):
    lengths = np.diff([0., *cuts, length])
    original = lengths.copy()
    rng.shuffle(lengths)
    if np.array_equal(lengths, original) and len(lengths) > 1:
        lengths = np.roll(lengths, 1)
    return np.cumsum(lengths)[:-1].tolist()


def run_alternatives(config):
    from .crop_benchmark import run
    cfg = config['density_alternatives']
    template = json.loads(Path(cfg['template']).read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    run(dict(output_root=str(root), reference_dir=template['reference_dir'],
             input_files=template['input_files'], commands=[template['commands'][0]]))
    source = Path(cfg['best_results'])
    previous = json.loads(source.read_text())
    oldroot = Path(cfg['sweep_root'])
    control = json.loads((oldroot / 'control-input.json').read_text())
    counts = measurement_counts(oldroot / 'control_winding_factors.csv', len(control['pieces']))
    cases = [('control', control)]
    stage = cfg.get('stage', 'centered')
    for mode in (('mean', 'median', 'max') if stage == 'centered' else ()):
        winner = previous['best_per_statistic'][mode]
        doc = json.loads((oldroot / (winner['name'] + '-input.json')).read_text())
        values = np.array([r['multiplier'] for r in doc['pieces']])
        mean = float(np.average(values, weights=counts))
        # Include original mapping and its genuinely matched uniform control.
        cases.append((mode + '-original', doc))
        uniform = copy.deepcopy(doc)
        uniform['mapping'] = dict(kind='uniform', multiplier=mean)
        for row in uniform['pieces']:
            row['multiplier'] = mean
        cases.append((mode + '-uniform', uniform))
        for power in (.5, 1., 2.):
            candidate = copy.deepcopy(doc)
            candidate['mapping'] = dict(kind='centered', power=power, source=winner['name'], normalization='initial-incident-measurement-weighted')
            for row, value in zip(candidate['pieces'], centered(values, counts, power)):
                row['multiplier'] = float(value)
            cases.append((mode + '-centered-' + str(power).replace('.', 'p'), candidate))
    extras = {}
    if stage == 'length':
        centerroot = Path(cfg['centered_root'])
        prior = json.loads((centerroot / 'summary.json').read_text())
        assert prior['stage'] == 'complete'
        winner = max((r for r in prior['results'] if '-centered-' in r['name']), key=lambda r: r['constraint_score_percent'])
        quality = json.loads((centerroot / (winner['name'] + '-input.json')).read_text())
        cases.append(('centered-control', quality))
        lengths = np.array([r['end_arc_base'] - r['begin_arc_base'] for r in control['pieces']])
        rate = float(counts.sum() / lengths.sum())
        for blend in (.5, 1.):
            measures = (1-blend)*counts + blend*rate*lengths
            for name, source_doc in [('plain', control), ('quality', quality)]:
                doc = copy.deepcopy(source_doc)
                factors = centered([r['multiplier'] for r in doc['pieces']], measures, 1.)
                for row, length, factor in zip(doc['pieces'], lengths, factors):
                    row.update(length_measure=rate*float(length), length_blend=blend, multiplier=float(factor))
                doc['mapping'] = dict(kind='length', blend=blend, rate=rate, quality=winner['name'] if name == 'quality' else None)
                cases.append((f'length-{str(blend).replace(".", "p")}-{name}', doc))
    elif stage == 'cuts':
        import importlib.util
        script = Path(__file__).resolve().parents[5] / 'volume-cartographer/scripts/analyze_trace_cost_removal.py'
        spec = importlib.util.spec_from_file_location('trace_cost_analysis', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        traces = module.load_steps(Path(cfg['steps']))
        all_steps = np.concatenate(list(traces.values()))
        d = all_steps[:, 2] / (all_steps[:, 1]-all_steps[:, 0]) * all_steps[:, 4]
        weights = all_steps[:, 1]-all_steps[:, 0]
        order = np.argsort(d, kind='stable')
        cumulative = np.cumsum(weights[order])
        for quantile, minimum in ((.9, 64.), (.95, 128.)):
            threshold = float(d[order[np.searchsorted(cumulative, quantile*cumulative[-1])]])
            cuts = {str(k): sustained_cuts(v, threshold, minimum) for k, v in traces.items()}
            rng = np.random.default_rng(27092026)
            randomcuts = {str(k): relocated_cuts(cuts[str(k)], float(v[-1, 1]), rng) for k, v in traces.items()}
            for label, values in [('targeted', cuts), ('relocated', randomcuts)]:
                name = f'cuts-q{int(quantile*100)}-{label}'
                hint = root / (name + '-cuts.json')
                atomic_write(hint, dict(version=1, trace_artifact=control['trace_artifact'], cuts=values, threshold=threshold, minimum_length=minimum, steps=cfg['steps']), durable=False)
                doc = copy.deepcopy(control)
                doc['mapping'] = dict(kind=label, quantile=quantile, threshold=threshold, min_run=minimum, cuts=sum(map(len, values.values())))
                cases.append((name, doc))
                extras[name] = ['--piece-cut-hints', str(hint)]
    elif stage == 'measurement':
        for cap in (10., 100., 1000.):
            name = f'measurement-cap-{int(cap)}'
            doc = copy.deepcopy(control)
            doc['mapping'] = dict(kind='measurement-binary-off', cap=cap)
            cases.append((name, doc))
            extras[name] = ['--winding-measurement-disable-cost', str(cap)]
    elif stage != 'centered':
        raise ValueError('unknown alternative stage')
    results = []
    with ThreadPoolExecutor(max_workers=int(cfg.get('workers', 4))) as pool:
        futures = [pool.submit(execute_case, template, root, name, doc, extras.get(name, ())) for name, doc in cases]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            atomic_write(root / 'summary.json', dict(stage='running', results=results), durable=False)
            print('alternative result: ' + json.dumps({k: v for k, v in row.items() if k != 'ladder'}), flush=True)
    atomic_write(root / 'summary.json', dict(stage='complete', results=results), durable=False)
