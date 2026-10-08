"""Audit and summarize ordinary-only N2 solves using the established metrics."""
import csv
import io
import json
from pathlib import Path

from .density_penalty_sweep import independent_metrics
from .local_cost_sweep import ladder_shift_metrics


def _csv(rows):
    if not rows:
        return ''
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def per_gauge_metrics(summaries, candidates):
    total = len({r['reference'] for r in summaries})
    gauges = []
    for gauge in sorted({r['gauge'] for r in summaries}, key=int):
        ss = [r for r in summaries if r['gauge'] == gauge]
        cc = [r for r in candidates if r['gauge'] == gauge]
        row = dict(gauge=int(gauge), references_with_observations=len(ss),
                   observations=sum(int(r.get('observations', 0)) for r in ss),
                   tied_best_rows=sum(int(r['best_ties']) > 1 for r in ss),
                   unbounded_best_rows=sum(r['lower_unbounded_best'] == '1' or
                       r['upper_unbounded_best'] == '1' for r in ss))
        log = ('independent_reference_summary\n' + _csv(ss) +
               'independent_reference_candidates\n' + _csv(cc) + 'independent_reference_end\n')
        if not cc or any(r['calibrated'] != '1' for r in ss):
            row.update(independent_percent=None, ladder_shift_percent=None,
                       metric_unavailable='no calibrated candidate ladder')
        else:
            row.update(independent_metrics(log))
            row['correct_references'] = sum(float(r['reference']) == float(r['winding']) for r in row['ladder'])
            row['independent_percent'] = 100 * row['correct_references'] / total
            row['independent_missing'] = total - len(row['ladder'])
            row.update(ladder_shift_metrics(row['ladder']))
            if row['independent_missing']:
                row.update(ladder_shifts=None, ladder_shift_percent=None,
                           metric_unavailable='partial component ladder')
        gauges.append(row)
    return gauges


def summarize(log, cohort, conditioned):
    if cohort['reference_nodes_in_solve'] or cohort['reference_cross_factors_in_solve']:
        raise ValueError('reference evidence leaked into ordinary solve')
    sources = {r['source_trace'] for r in conditioned['traces']}
    pieces = {r['original_piece'] for r in conditioned['pieces']}
    if sources != set(cohort['ordinary_source_trace_ids']) or pieces != set(cohort['ordinary_original_piece_ids']):
        raise ValueError('ordinary cohort differs from the conditioned N2 cohort')
    if len(cohort['ordinary_original_piece_ids']) != len(pieces):
        raise ValueError('duplicate ordinary pieces')
    summaries = list(csv.DictReader(io.StringIO(log.split('independent_reference_summary\n')[1]
        .split('independent_reference_candidates')[0])))
    candidates = list(csv.DictReader(io.StringIO(log.split('independent_reference_candidates\n')[1]
        .split('independent_reference_end')[0])))
    result = dict(cohort_ids_match=True, fibers=len(sources), pieces=len(pieces),
                  readout_summaries=summaries, readout_candidates=candidates,
                  references=len({r['reference'] for r in summaries}),
                  tied_best_rows=sum(int(r['best_ties']) > 1 for r in summaries),
                  unbounded_best_rows=sum(r['lower_unbounded_best'] == '1' or
                      r['upper_unbounded_best'] == '1' for r in summaries))
    result['per_gauge'] = per_gauge_metrics(summaries, candidates)
    result['full_coverage_gauges'] = [r['gauge'] for r in result['per_gauge']
        if not r.get('metric_unavailable') and not r.get('independent_missing')]
    try:
        result.update(independent_metrics(log))
    except ValueError as error:
        # Do not silently select a favorable gauge to manufacture one ladder.
        if str(error) not in ('multiple independent gauges per reference require explicit reporting',
                              'uncalibrated independent reference gauge'):
            raise
        result.update(independent_percent=None, ladder_shifts=None, ladder_shift_percent=None,
                      metric_unavailable=str(error))
        return result
    result['correct_references'] = sum(float(r['reference']) == float(r['winding']) for r in result['ladder'])
    result.update(ladder_shift_metrics(result['ladder']))
    if result['independent_missing']:
        result.update(ladder_shifts=None, ladder_shift_percent=None,
                      metric_unavailable='incomplete ladder: shift score not comparable')
    return result


def report(config):
    from lasagna.inference_provenance import atomic_write, sha256_file
    entries = []
    for item in config['neighborhood_readout_report']['runs']:
        prefix = Path(item['prefix'])
        record = json.loads(prefix.with_suffix('.run.json').read_text())
        if record['returncode'] != 0:
            raise ValueError('ordinary solve did not complete successfully')
        cohort = json.loads(Path(str(prefix) + '_neighborhood_cohort.json').read_text())
        conditioned = json.loads(Path(item['conditioned']).read_text())
        result = summarize(prefix.with_suffix('.log').read_text(), cohort, conditioned)
        result.update(name=item['name'], provenance=record, cohort=cohort,
                      conditioned_artifact=item['conditioned'],
                      conditioned_sha256=sha256_file(item['conditioned']))
        entries.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in (
            'provenance', 'cohort', 'readout_summaries', 'readout_candidates', 'ladder', 'per_gauge')}), flush=True)
        for row in result['per_gauge']:
            if row['gauge'] in result['full_coverage_gauges']:
                print(json.dumps(dict(name=item['name'], full_coverage_component={
                    k: v for k, v in row.items() if k != 'ladder'})), flush=True)
    path = Path(config['neighborhood_readout_report']['output'])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, dict(version=1, runs=entries), durable=False)
    return entries
