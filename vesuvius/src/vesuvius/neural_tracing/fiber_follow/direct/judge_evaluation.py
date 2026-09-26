"""Paired event partitions and locked, deterministic joint-policy selection."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ..events import EVENT_VERSION, label_path
from ..geometry import arclength

METRIC_VERSION = 'judge_retained_arc_v1'
FORECAST_THRESHOLDS = (.5, .7, .8, .85, .9, .95, .98)
ACCEPT_THRESHOLDS = (.7, .8, .85, .9, .95, .98)
ALARM_THRESHOLDS = (.1, .25, .4, .5, .6)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paired_row(baseline, retained, events, decision_arc, fiber, judge_induced=True, missing_source=False):
    """Always intersect the COMPLETE baseline's event partition with retained arc."""
    from ..events import truncate_path
    length = float(arclength(retained)[-1])
    expected = truncate_path(baseline, length)
    if expected.shape != retained.shape or not np.allclose(expected, retained, rtol=0, atol=1e-5):
        raise AssertionError('Judge changed forecast geometry before stopping')
    base, judged = events.partition(), events.partition(length)
    y, known = events.labels([decision_arc])
    discarded_good = base['correct']-judged['correct']
    false = bool(judge_induced and known[0] and y[0] and discarded_good > 1e-7)
    unknown = bool(judge_induced and not known[0])
    return dict(fiber=fiber, C0=base['correct'], Cj=judged['correct'], W0=base['wrong'], Wj=judged['wrong'],
                U0=base['unknown'], Uj=judged['unknown'], F=int(false), unknown_stops=int(unknown),
                missing_source=int(missing_source), decision_arc=float(decision_arc), retained_arc=length)


def metrics(rows):
    totals = {k: sum(r[k] for r in rows) for k in ('C0','Cj','W0','Wj','U0','Uj','F','unknown_stops','missing_source')}
    c0, cj, w0, wj, f = (totals[k] for k in ('C0','Cj','W0','Wj','F'))
    totals.update(wrong_reduction=1-wj/w0 if w0 > 0 else None,
                  correct_loss=(c0-cj)/c0 if c0 > 0 else None,
                  false_stops_per_10000=10000*f/c0 if c0 > 0 else None,
                  precision=cj/(cj+wj) if cj+wj > 0 else None,
                  poisson_zero_upper_95=-math.log(.05)*10000/c0 if f == 0 and c0 > 0 else None)
    totals['qualifies'] = bool(w0 > 0 and c0 > 0 and cj+wj > 0 and
                              totals['wrong_reduction'] >= .5 and totals['correct_loss'] <= .02 and
                              totals['false_stops_per_10000'] <= 1 and totals['precision'] >= .95)
    return totals


def bootstrap(rows, replicates=2000):
    groups = sorted(set(r['fiber'] for r in rows), key=str)
    rng = np.random.default_rng(0)
    samples = {k: [] for k in ('wrong_reduction','correct_loss','false_stops_per_10000','precision')}
    for _ in range(replicates):
        selected = rng.choice(len(groups), len(groups))
        report = metrics([r for i in selected for r in rows if r['fiber'] == groups[i]])
        for key in samples:
            samples[key].append(report[key])
    return {k: dict(interval=np.quantile([v for v in values if v is not None], [.025,.975]).tolist()
                       if any(v is not None for v in values) else None,
                    undefined=sum(v is None for v in values)) for k, values in samples.items()}


def freeze_protocol(out, checkpoints, manifest_hash, **settings):
    protocol = dict(checkpoints=sorted((str(Path(p).resolve()), hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in checkpoints),
                    manifest_sha256=manifest_hash, forecast=FORECAST_THRESHOLDS, accept=ACCEPT_THRESHOLDS,
                    alarm=ALARM_THRESHOLDS, delay=8., provisional=32., metric_version=METRIC_VERSION,
                    event_version=EVENT_VERSION, **settings)
    protocol['sha256'] = digest(protocol)
    path = Path(out)/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256'] != protocol['sha256']:
            raise ValueError('Calibration protocol is already frozen')
    else:
        path.write_text(json.dumps(protocol, indent=2))
    return protocol


def select(reports, protocol, out):
    eligible = [r for r in reports if r['metrics']['qualifies']]
    path = Path(out)/'selection.json'
    if path.exists():
        raise FileExistsError('Selection already locked; use a new output directory')
    if not eligible:
        result = dict(status='failed_calibration', default='forecast_only', protocol_sha256=protocol['sha256'])
        (Path(out)/'failed_calibration.json').write_text(json.dumps(result, indent=2))
        return result
    winner = min(eligible, key=lambda r: (-r['metrics']['Cj'], r['metrics']['Wj'], r['metrics']['F'],
                        r['checkpoint_sha256'], r['threshold'], r['judge_policy']['accept'], r['judge_policy']['alarm']))
    ci = bootstrap(winner['rows'])
    supported = all(v['interval'] is not None and v['undefined'] == 0 for v in ci.values())
    if supported:
        supported = (ci['wrong_reduction']['interval'][0] >= .5 and ci['correct_loss']['interval'][1] <= .02 and
                     ci['false_stops_per_10000']['interval'][1] <= 1 and ci['precision']['interval'][0] >= .95 and
                     (winner['metrics']['poisson_zero_upper_95'] is None or winner['metrics']['poisson_zero_upper_95'] <= 1))
    supported = supported and len(set(r['fiber'] for r in winner['rows'])) >= 2
    supported = supported and any(r.get('source', 'real') == 'real' and r['W0'] > 0 for r in winner['rows'])
    selection = {k: v for k, v in winner.items() if k not in ('rows','legacy_baseline','legacy_judge','initialized_follower')}
    selection.update(split='calibration', protocol_sha256=protocol['sha256'], metric_version=METRIC_VERSION,
                     event_version=EVENT_VERSION, bootstrap=ci, opt_in=not supported,
                     manifest_sha256=protocol['manifest_sha256'])
    path.write_text(json.dumps(selection, indent=2, allow_nan=False))
    return selection


def load_selection(path, checkpoint=None):
    selection = json.loads(Path(path).read_text())
    if selection.get('metric_version') != METRIC_VERSION or selection.get('event_version') != EVENT_VERSION:
        raise ValueError('Incompatible calibrated judge policy')
    source = Path(checkpoint or selection['checkpoint'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash changed')
    protocol_path = Path(path).parent/'protocol.json'
    if not protocol_path.exists():
        raise ValueError('Locked selection requires its frozen protocol.json')
    protocol = json.loads(protocol_path.read_text())
    recorded = protocol.pop('sha256')
    if digest(protocol) != recorded or recorded != selection['protocol_sha256']:
        raise ValueError('Calibration protocol hash changed')
    if (selection['threshold'] not in protocol['forecast'] or
        selection['judge_policy']['accept'] not in protocol['accept'] or
        selection['judge_policy']['alarm'] not in protocol['alarm'] or
        selection['judge_policy']['delay'] != protocol['delay'] or
        selection['judge_policy']['provisional'] != protocol['provisional']):
        raise ValueError('Selection settings are outside the frozen protocol')
    return selection


def paired_monitor(judged_tracer, fibers, seeds, baseline_paths):
    """Monitor real departures, misses and tail losses without relabeling prefixes."""
    from ..evaluate import score_trace
    rows, audits, legacy = [], [], []
    for seed, baseline in zip(seeds, baseline_paths):
        f = fibers[seed['fiber']]
        reverse = seed['sign'] < 0
        event = label_path(baseline, f.points[::-1] if reverse else f.points,
                           f.length-seed['t'] if reverse else seed['t'], f.endpoint_stop[0 if reverse else 1])
        paths, reasons = judged_tracer.trace(np.asarray([seed['pos']]), np.asarray([seed['heading']]))
        state = judged_tracer.observed_states[0]
        audit = state['policy'].audit
        row = paired_row(baseline, paths[0], event, audit[-1]['endpoint'], seed['fiber'],
                         reasons[0].startswith('judge_'), any(not all(a['support']) for a in audit))
        alarm = next((a for a in audit if a['reason'] == 'judge_alarm'), None)
        after_onset = [a for a in audit if event.onset is not None and a['endpoint'] >= event.onset]
        detected = event.onset is not None and alarm is not None and alarm['endpoint'] >= event.onset
        row.update(source='real', event_type=event.kind, departure=int(event.onset is not None),
                   detected=int(detected), miss=int(event.onset is not None and not detected),
                   recall_one=int(detected and alarm in after_onset[:1]),
                   recall_two=int(detected and alarm in after_onset[:2]),
                   onset_latency=alarm['endpoint']-event.onset if detected else None,
                   confirmation_latency=alarm['endpoint']-event.confirmation if detected else None,
                   boundary_interval_error=max(event.bracket[0]-state['policy'].accepted, 0.,
                                               state['policy'].accepted-event.bracket[1]) if detected else None,
                   good_discarded=row['C0']-row['Cj'], wrong_retained=row['Wj'], reason=reasons[0])
        rows.append(row)
        if len(audits) < 3:
            state['events'] = event
            audits.append(state)
        legacy.append(score_trace(paths[0], f, seed['t'], seed['sign']))
    report = metrics(rows)
    departed = sum(r['departure'] for r in rows)
    report.update(departures=departed, misses=sum(r['miss'] for r in rows),
                  recall_one=sum(r['recall_one'] for r in rows)/departed if departed else None,
                  recall_two=sum(r['recall_two'] for r in rows)/departed if departed else None,
                  latency=[r['onset_latency'] for r in rows if r['onset_latency'] is not None],
                  by_support={str(m):metrics([r for r in rows if bool(r['missing_source']) == m]) for m in (False,True)})
    return dict(metrics=report, rows=rows, legacy=legacy), audits
