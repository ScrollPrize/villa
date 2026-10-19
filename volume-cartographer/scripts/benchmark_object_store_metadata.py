#!/usr/bin/env python3
"""Consumer-shaped metadata/link diagnostics, using disposable synthetic stores.

Run with the interpreter matching build/python (normally .venv/bin/python).
This complements benchmark_object_store_workflows.py: no consumer is migrated.
SQL diagnostics run in a separate process to isolate SQLite library locks.
SQL projections are lower bounds, not equivalent replacements for store APIs.
"""

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import uuid

from benchmark_object_store import measure


def equal(expected):
    def check(actual):
        assert actual == expected, (actual, expected)
    return check


def identity(index):
    return str(uuid.UUID(int=index + 1))


def sql_diagnostics(path, n):
    # Called only in a fresh process, after the native store has closed.
    results = []
    with sqlite3.connect(path) as sql:
        queries = {
            'rare_edge_ids': ("SELECT id FROM edges WHERE kind='fiber-link' ORDER BY id", 1),
            'approved_crossing_page': (
                "SELECT e.id,a.uuid,b.uuid,e.a_anchor_x,e.b_anchor_x "
                "FROM edges e JOIN objects a ON a.id=e.a_id JOIN objects b ON b.id=e.b_id "
                "WHERE e.kind='crossing' AND e.approval_state='approved' ORDER BY e.id LIMIT 100", 100),
        }
        for indexed in (False, True):
            if indexed:
                sql.execute('CREATE INDEX benchmark_kind_approval ON edges(kind,approval_state,id)')
            for name, (query, count) in queries.items():
                expected = sql.execute(query).fetchall()
                equal(count)(len(expected))
                measure(results, f'sql_{name}_index_{indexed}_lower_bound',
                        lambda q=query: sql.execute(q).fetchall(),
                        check=equal(expected),
                        plan=sql.execute('EXPLAIN QUERY PLAN ' + query).fetchall())
        assert sql.execute("SELECT count(*) FROM edges WHERE kind='crossing'").fetchone()[0] == n
    return results


def workload(Store, root, n, payload_bytes):
    results = []
    path = root / 'metadata.vcobj'
    ids = [identity(i) for i in range(n)]
    with Store.create(str(path), author='benchmark') as db:
        crossings = []
        with db.write() as tx:
            hub = tx.upsert_object(dict(uuid=identity(n), kind='patch'))
            for i, ident in enumerate(ids):
                tx.upsert_object(dict(uuid=ident, kind='track', name=f'track-{i}',
                    tags=['horizontal' if i % 2 == 0 else 'vertical'],
                    track={'dataset_ref': 'fixture', 'external_key': str(i),
                           'family': 'horizontal' if i % 2 == 0 else 'vertical', 'length': i + 1},
                    meta={'family': 'H' if i % 2 == 0 else 'V', 'length': i + 1,
                          'source': {'dataset': 'fixture'}, 'payload': 'x' * payload_bytes}))
            for i, ident in enumerate(ids):
                edge = tx.add_edge(dict(kind='crossing', a=ident, b=ids[(i + 1) % n],
                    a_anchor={'position': [i, 0, 0]}, b_anchor={'position': [i, 1, 0]},
                    meta={'local_indices': {ident: i, ids[(i + 1) % n]: i + 1},
                          'clearance': i + .5, 'detector': 'fixture', 'payload': 'x' * payload_bytes}))
                crossings.append(edge)
                if i % 10 == 0:
                    tx.set_approval(edge, 'approved')
                tx.add_edge(dict(kind='point-patch', a=ident, b=hub,
                    meta={'grid': [i, 0], 'winding_annotation': i % 8}))
            rare = tx.add_edge(dict(kind='fiber-link', a=ids[0], b=ids[2]))
        db.checkpoint()

        def bench(name, fn, check):
            measure(results, name, fn, repeats=3, check=check)

        expected = set(crossings)
        bench('crossings_bulk_records', lambda: db.edges_by_kind('crossing'),
              lambda rows: equal(expected)({e['uuid'] for e in rows}))
        bench('rare_edge_kind', lambda: db.edges_by_kind('fiber-link'),
              lambda rows: equal([rare])([e['uuid'] for e in rows]))

        def reviewed_page():
            return [e for e in db.edges_by_kind('crossing') if e['approval_state'] == 'approved'][:100]

        bench('approved_crossing_page_100', reviewed_page,
              lambda rows: equal(crossings[::10][:100])([e['uuid'] for e in rows]))
        bench('approved_crossing_page_100_sql_projected',
              lambda: db.select_edges(edge_kind='crossing', approval_states=['approved'],
                                      limit=100, fields=['a_anchor', 'b_anchor']),
              lambda rows: equal(sorted(crossings[::10])[:100])([e['uuid'] for e in rows]))
        bench('patch_incident_records', lambda: db.edges_of(hub),
              lambda rows: equal(n)(len(rows)))
        # A track incident to just 3/4 links versus a patch incident to n links.
        bench('track_incident_records', lambda: db.edges_of(ids[1]),
              lambda rows: equal(3)(len(rows)))
        bench('track_crossing_records_scoped',
              lambda: db.select_edges(edge_kind='crossing', endpoint_uuids=[ids[1]], fields=['a_anchor']),
              lambda rows: equal(set(crossings[:2]))({row['uuid'] for row in rows}))
        bench('patch_filtered_neighbours_absent_kind',
              lambda: db.neighbours(hub, edge_kind='crossing'), equal([]))

        def page(fields):
            rows = db.select_objects(object_kind='track', fields=fields)
            return sorted((o for o in rows if o['track']['family'] == 'horizontal'
                           and o['track']['length'] >= n // 2),
                          key=lambda o: (-o['track']['length'], o['uuid']))[:200]

        expected_page = list(reversed(ids[::2]))[:200]
        for fields in (None, ['name', 'track']):
            bench(f'family_length_page_fields_{fields}', lambda f=fields: page(f),
                  lambda rows: equal(expected_page)([o['uuid'] for o in rows]))
        bench('family_length_page_sql_projected',
              lambda: db.select_objects(object_kind='track', family='horizontal', min_length=n//2,
                                        order_by='length_desc', limit=200, fields=['name', 'track']),
              lambda rows: equal(expected_page)([o['uuid'] for o in rows]))

        # Compare repeated graph reads with a resident adjacency map (no SQL in
        # the fitting loop). Database record loading is reported separately;
        # resident map construction is excluded from the lower bound.
        probe_ids = ids[:1000]
        expected_neighbours = [{ids[(i - 1) % n], ids[(i + 1) % n]} for i in range(len(probe_ids))]
        bench('1000_crossing_neighbour_queries',
              lambda: [db.neighbours(i, edge_kind='crossing') for i in probe_ids],
              lambda rows: equal(expected_neighbours)([set(r) for r in rows]))
        adjacency = {ident: [ids[(i - 1) % n], ids[(i + 1) % n]] for i, ident in enumerate(ids)}
        bench('1000_resident_adjacency_copies_lower_bound',
              lambda: [list(adjacency[i]) for i in probe_ids],
              lambda rows: equal(expected_neighbours)([set(r) for r in rows]))
        def read_graph():
            with db.graph_read(object_kind='track', edge_kind='crossing', approval_states=[]) as view:
                return len(view.object_uuids), len(view.edge_uuids), int(view.offsets[-1])
        bench('store_graph_read_view', read_graph, equal((n, n, 2*n)))
        with db.graph_read(object_kind='track', edge_kind='crossing', approval_states=[]) as view:
            bench('1000_store_graph_neighbour_views', lambda: [view.neighbours(i) for i in range(1000)],
                  lambda rows: equal([{(i-1) % n, (i+1) % n} for i in range(1000)])([set(r) for r in rows]))

        for record, read in ((ids[0], db.object), (crossings[1], db.edge)):
            category = 'object' if record == ids[0] else 'edge'
            for batch in (1, 200):
                counter = 0

                def update():
                    nonlocal counter
                    for start in range(0, 200, batch):
                        with db.write() as tx:
                            for _ in range(start, start + batch):
                                counter += 1
                                tx.set_meta(record, 'review_note', counter)
                    return counter

                bench(f'{category}_200_metadata_updates_batch_{batch}', update,
                      lambda value: equal(value)(read(record)['meta']['review_note']))
            db.set_meta(record, 'review_note', 'unchanged')
            db.set_approval(record, 'approved')
            before = read(record)['generation']
            journal_before = db.journal_since()[-1]['seq']
            bench(f'{category}_identical_metadata_write',
                  lambda: db.set_meta(record, 'review_note', 'unchanged'),
                  lambda _: equal('unchanged')(read(record)['meta']['review_note']))
            results.append(dict(name=f'{category}_identical_write_effects',
                generation_delta=read(record)['generation'] - before,
                journal_entries=len(db.journal_since(journal_before)), writes=4,
                approval_before='approved', approval_after=read(record)['approval_state']))
            equal(0)(results[-1]['generation_delta'])
            equal(0)(results[-1]['journal_entries'])
            equal('approved')(results[-1]['approval_after'])

        profile = cProfile.Profile()
        profile.runcall(reviewed_page)
        stats = pstats.Stats(profile)
        results.append(dict(name='review_page_python_profile', top_cumulative=[
            dict(function=str(k), calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
            for k, v in sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:8]]))
        db.checkpoint()
        results.append(dict(name='store_bytes', bytes=path.stat().st_size))
    completed = subprocess.run([sys.executable, __file__, '--sql', str(path), '--n', str(n)],
                               check=True, text=True, stdout=subprocess.PIPE, timeout=120)
    results.extend(json.loads(completed.stdout))
    for row in results:
        if 'samples_ms' in row:
            row['mean_ms'] = statistics.mean(row['samples_ms'])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/vcobj-metadata.json'))
    parser.add_argument('--build-python', type=Path, default=Path(__file__).resolve().parents[1] / 'build/python')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 50000])
    parser.add_argument('--payload-bytes', type=int, default=256)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--sql', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, default=1000, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes + [args.n]) < 1000 or args.payload_bytes < 0:
        parser.error('sizes must be >=1000 and payload bytes nonnegative')
    if args.sql:
        print(json.dumps(sql_diagnostics(args.sql, args.n)))
        return
    if args.worker:
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_editable_skbc_volume_cartographer']
        sys.path.insert(0, str(args.build_python.resolve()))
        import vc.objects
        with tempfile.TemporaryDirectory(prefix='vcobj-metadata-') as directory:
            results = workload(vc.objects.ObjectStore, Path(directory), args.n, args.payload_bytes)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps(dict(n=args.n, edges=2 * args.n + 1, payload_bytes=args.payload_bytes,
            module=vc.objects.__file__, results=results,
            peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024)))
        return
    cache = args.build_python.parent / 'CMakeCache.txt'
    report = dict(platform=platform.platform(), python=sys.version, python_sqlite=sqlite3.sqlite_version,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        build=[s for s in cache.read_text().splitlines() if s.startswith((
            'CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS_RELWITHDEBINFO:', 'VC_MARCH_NATIVE:'))],
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0], scenarios=[])
    for n in args.sizes:
        completed = subprocess.run([sys.executable, __file__, '--worker', '--n', str(n),
            '--payload-bytes', str(args.payload_bytes), '--build-python', str(args.build_python.resolve())],
            check=True, text=True, stdout=subprocess.PIPE, timeout=300)
        report['scenarios'].append(json.loads(completed.stdout))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
