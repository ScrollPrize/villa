import json, math, statistics, hashlib
from pathlib import Path
from collections import defaultdict
root = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full')
raw = (root/'summary.json').read_bytes()
summary = json.loads(raw)
assert summary['completed'] and summary['cases'] == 100
groups = defaultdict(list)
collections = {}
case_lookup = {}
for source in summary['sources']:
    scroll = source['backend_config']['coordinate_space'].split('/')[0]
    scale = source['summary']['base_voxel_size_um']/1000
    rows = []
    for case in source['cases']:
        assert case['evaluation_complete']
        row = dict(failed_mm=[f['traced_span_base_voxels']*scale for f in case['failures']],
                   reference_mm=case['reference_length_base_voxels']*scale,
                   credited_mm=case['traced_length_base_voxels']*scale,
                   seeded=case['seeded_spans'], failure_free=case['failure_free'],
                   physical_failed_mm=[], physical_all_mm=[])
        key = (source['source_id'], case['source_index'], case['direction']=='reverse')
        assert key not in case_lookup
        case_lookup[key] = (case, row)
        rows.append(row)
    collections[source['source_id']] = rows
    groups[scroll].extend(rows)
    expected = source['summary']['mean_failed_span_length_mm']
    assert math.isclose(sum(sum(r['failed_mm']) for r in rows)/sum(len(r['failed_mm']) for r in rows), expected)
seen = set()
for path in sorted((root/'cases').glob('*/trace.json')):
    artifact = json.loads(path.read_text())
    p, replay = artifact['provenance'], artifact['replay']
    source = next(s for s in summary['sources'] if s['source_id']==p['source_id'])
    fiber_index = source['backend_config']['fibers'].index(p['fiber_path'])
    key = (p['source_id'], fiber_index, p['reverse'])
    assert key not in seen
    seen.add(key)
    case, row = case_lookup[key]
    scale = p['base_voxel_size_um']/1000
    segments = replay['segments']
    assert len(replay['failures']) == len(case['failures'])
    assert len(segments) == len(case['failures']) + 1
    positive_spans = []
    for seg in segments:
        begin = seg['matches'][0]['search_begin_arc_base'] if seg['matches'] else seg['end_reference_arc_base']
        positive_spans.append(max(0, seg['end_reference_arc_base']-begin)*scale)
        pts = seg['trace_points_base_xyz']
        row['physical_all_mm'].append(sum(math.dist(a,b) for a,b in zip(pts,pts[1:]))*scale)
    assert math.isclose(sum(positive_spans), row['credited_mm'], rel_tol=1e-8, abs_tol=1e-7)
    for failure, measured in zip(replay['failures'], case['failures']):
        seg = segments[failure['segment_index']]
        begin = seg['matches'][0]['search_begin_arc_base'] if seg['matches'] else failure['reference_arc_base']
        span = max(0,min(case['reference_length_base_voxels'],failure['reference_arc_base']-begin))
        assert math.isclose(span, measured['traced_span_base_voxels'], abs_tol=1e-7)
        row['physical_failed_mm'].append(row['physical_all_mm'][failure['segment_index']])
assert seen == set(case_lookup) and len(seen)==100
def aggregate(rows):
    failed=[x for r in rows for x in r['failed_mm']]
    percentiles=statistics.quantiles(failed, n=100, method='inclusive')
    physical=[x for r in rows for x in r['physical_failed_mm']]
    all_physical=[x for r in rows for x in r['physical_all_mm']]
    n=len(failed); length=sum(r['reference_mm'] for r in rows)
    credit=sum(r['credited_mm'] for r in rows)
    return dict(fibers=len(rows)//2, directions=len(rows), failures=n,
        zero_length_failures=sum(x==0 for x in failed),
        failure_free_directions=sum(r['failure_free'] for r in rows),
        directed_reference_mm=length, credited_mm=credit,
        failed_span_sum_mm=sum(failed), mean_before_restart_mm=statistics.mean(failed),
        median_before_restart_mm=statistics.median(failed),
        p05_before_restart_mm=percentiles[4],
        p95_before_restart_mm=percentiles[94],
        mean_all_attempts_including_terminal_and_zero_mm=credit/(n+len(rows)),
        mean_positive_seeded_span_mm=credit/sum(r['seeded'] for r in rows),
        corpus_reliability_mm=length/(n+1),
        mean_failed_polyline_mm=statistics.mean(physical),
        mean_all_polyline_mm=statistics.mean(all_physical))
global_rows=[r for rows in groups.values() for r in rows]
report=dict(input=str(root), summary_sha256=hashlib.sha256(raw).hexdigest(),
    artifact_cases_validated=len(seen), global_result=aggregate(global_rows),
    scrolls={k:aggregate(v) for k,v in groups.items()},
    collections={k:aggregate(v) for k,v in collections.items()})
assert report['global_result']['failures']==summary['failures']
assert math.isclose(report['global_result']['directed_reference_mm'], summary['directed_reference_length_mm'])
assert math.isclose(report['global_result']['corpus_reliability_mm'],summary['mean_segment_length_mm'])
print(json.dumps(report,indent=2))
