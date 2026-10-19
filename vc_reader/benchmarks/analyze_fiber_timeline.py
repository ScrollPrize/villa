"""Summarize producer/consumer timestamps from a real fiber training run."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(path):
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['measured']]
    start, end = rows[0]['trace']['requested'], rows[-1]['trace']['finished']
    all_requests = json.loads((path / 'heading_timeline.json').read_text())
    last_heading = {}
    for request in all_requests:
        if request['context'] is not None:
            last_heading[tuple(request['context'])] = request['finished']
    requests = [r for r in all_requests if start <= r['began'] < end]
    stats = dict(wall=end-start, batches=len(rows), heading_requests=len(requests),
                 heading_queries=sum(r['queries'] for r in requests))
    def summarize(name, values):
        values = sorted(values)
        stats[name] = dict(total=sum(values), mean=statistics.mean(values),
                           p50=statistics.median(values), p95=values[int(.95*(len(values)-1))],
                           max=max(values))
    summarize('heading_service', [r['finished']-r['began'] for r in requests])
    summarize('heading_cpu', [r['cpu_seconds'] for r in requests])
    summarize('heading_queue', [r['began']-r['sent'] for r in requests])
    for name, function in {
        'worker_elapsed': lambda w: w['ready']-w['began'],
        'worker_cpu': lambda w: w['cpu_seconds'],
        'worker_heading': lambda w: w['heading_seconds'],
        'worker_other': lambda w: w['ready']-w['began']-w['heading_seconds'],
    }.items():
        summarize(name, [function(w) for r in rows for w in r['trace']['worker']])
    summarize('loader_wait', [r['loader_seconds'] for r in rows])
    summarize('producer_late', [max(0., max(w['ready'] for w in r['trace']['worker'])-r['trace']['requested']) for r in rows])
    summarize('collection_after_worker', [max(0.,r['trace']['returned']-max(r['trace']['requested'],max(w['ready'] for w in r['trace']['worker']))) for r in rows])
    stats['wait_after_last_heading'] = sum(max(0., w['ready']-max(r['trace']['requested'],
        last_heading.get((w['worker'], w['index']), w['began']))) for r in rows for w in r['trace']['worker'])
    blocked = []
    for index, row in enumerate(rows):
        if row['loader_seconds'] <= .001:
            continue
        ready = sum(all(w['ready'] < row['trace']['requested'] for w in later['trace']['worker'])
                    for later in rows[index+1:])
        if ready:
            blocked.append((row['loader_seconds'], ready))
    stats['ordered_delivery'] = dict(blocked_batches=len(blocked),
        wait_with_later_batches_ready=sum(t for t, n in blocked),
        mean_later_batches_ready=statistics.mean(n for t, n in blocked) if blocked else 0.)
    stats['heading_stages'] = {}
    for request in requests:
        for name, counters in request.get('stages', {}).items():
            total = stats['heading_stages'].setdefault(name, dict(seconds=0., cpu_seconds=0., calls=0))
            for key, value in counters.items():
                total[key] += value
    stages = dict(loader=('requested', 'returned'), images=('returned', 'prepared'),
                  model=('update_started', 'metrics_done'), sync=('metrics_done', 'finished'))
    for name, (a, b) in stages.items():
        intervals = [(r['trace'][a],r['trace'][b]) for r in rows]
        total = sum(y-x for x,y in intervals)
        occupied = sum(max(0.,min(y,q['finished'])-max(x,q['began'])) for x,y in intervals for q in requests)
        finished = [q for q in requests if any(x <= q['finished'] < y for x,y in intervals)]
        stats[name] = dict(seconds=total, heading_active=occupied, heading_finished=len(finished),
                           heading_queries=sum(q['queries'] for q in finished))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps({p.name: analyze(p) for p in args.runs}, indent=2))
