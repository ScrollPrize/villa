#!/usr/bin/env python3
"""Consumer-oriented object-store diagnostics; never touches existing databases.

Build: ninja -C build vc_object_bindings
Run: .venv/bin/python scripts/benchmark_object_store_workflows.py \
       --output /tmp/vcobj-workflows.json
Optional real TIFF fixture: --patch core/test/data/segments/20241113070770

Uses fresh worker processes, temporary stores, default FULL durability, one
warmup and three timed repetitions. Checks run outside the timed region.
SQL/mmap comparisons are explicitly lower bounds, not equivalent store APIs.
Peak RSS includes fixture setup. No cold-cache or end-to-end application claims.
"""

import argparse
import array
import cProfile
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import uuid

from benchmark_object_store import count_is, measure, native_plan, sqlite_rows


def bench(results, name, fn, check=None, **details):
    measure(results, name, fn, repeats=3, check=check, **details)
    results[-1]['mean_ms'] = sum(results[-1]['samples_ms']) / len(results[-1]['samples_ms'])


def catalog(Store, root, n, results):
    with Store.create(str(root / 'catalog.vcobj'), author='benchmark') as db:
        ids = []
        with db.write() as tx:
            for i in range(n):
                ids.append(tx.upsert_object(dict(kind='track', name=f'track-{i}',
                    tags=['ordinary'], meta={'payload': 'x' * 128},
                    track=dict(dataset_ref='fixture', external_key=str(i),
                        bbox=[0, 0, i * 10, 1, 1, i * 10 + 1]))))
            for a, b in zip(ids, ids[1:]):
                tx.add_edge(dict(kind='crossing', a=a, b=b))
        db.checkpoint()
        bench(results, 'catalog_all_records', lambda: db.objects_by_kind('track'), count_is(n))
        bench(results, 'catalog_projected_names',
              lambda: db.select_objects(object_kind='track', fields=['name']), count_is(n))
        bench(results, 'catalog_projected_ids',
              lambda: db.select_objects(object_kind='track', fields=[]), count_is(n))
        for offset in sorted({0, n - 100}):
            bench(results, f'catalog_page_offset_{offset}',
                  lambda o=offset: db.select_objects(object_kind='track', limit=100, offset=o), count_is(100))
        bench(results, 'component_ids', lambda: db.component(ids[0]), count_is(n))
        bench(results, 'component_first_record_limit_1',
              lambda: db.select_objects(connected_to=ids[0], limit=1), count_is(1))
        bench(results, 'graph_adjacency_per_object',
              lambda: [db.neighbours(i) for i in ids],
              lambda v: count_is(2 * (n - 1))([x for row in v for x in row]))
        bench(results, 'graph_projected_export', lambda: db.export_json(fields=['name']),
              lambda v: count_is(n - 1)(json.loads(v)['edges']))

        # Fixed-size narrow Z slab as the total catalog grows.
        def in_slab(records):
            return [o['uuid'] for o in records
                    if o['track']['bbox'][2] <= 501 and o['track']['bbox'][5] >= 500]

        bench(results, 'viewport_hydrate_then_filter', lambda: in_slab(db.objects_by_kind('track')), count_is(1))
        box = [-1, -1, 500, 2, 2, 501]
        expected = [ids[50]]
        for fields in (None, [], ['name', 'generation']):
            bench(results, f'viewport_rtree_fields_{fields}',
                  lambda f=fields: db.select_objects(bbox=box, fields=f),
                  lambda v: assert_equal([row['uuid'] for row in v], expected))
        bench(results, 'viewport_rtree_kind_projected',
              lambda: db.select_objects(bbox=box, object_kind='track', fields=['name']),
              lambda v: assert_equal([row['uuid'] for row in v], expected))
        cached = db.objects_by_kind('track')
        bench(results, 'viewport_cached_python_filter', lambda: in_slab(cached), count_is(1))
        with sqlite3.connect(db.path) as sql:
            query = 'SELECT object_id FROM tracks WHERE bbox_min_z<=501 AND bbox_max_z>=500'
            bench(results, 'sql_bbox_scan_ids_only_lower_bound', lambda: sql.execute(query).fetchall(), count_is(1))
            results.append(dict(name='bbox_query_plan', plan=native_plan(db.path, query)))
            query = """SELECT uuid,kind FROM objects NOT INDEXED WHERE kind='track' AND id IN (
                SELECT b.object_id FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id
                WHERE b.array_name='' AND r.max_x>=-1 AND r.min_x<=2 AND r.max_y>=-1 AND r.min_y<=2 AND r.max_z>=500 AND r.min_z<=501
                AND b.max_x>=-1 AND b.min_x<=2 AND b.max_y>=-1 AND b.min_y<=2 AND b.max_z>=500 AND b.min_z<=501)
                ORDER BY uuid"""
            results.append(dict(name='rtree_query_plan', plan=native_plan(db.path, query)))
        # Profile public API boundary separately from timing samples.
        profiler = cProfile.Profile()
        profiler.runcall(lambda: in_slab(db.objects_by_kind('track')))
        profiler.dump_stats(str(root / 'catalog.prof'))
        import pstats
        stats = pstats.Stats(profiler)
        results.append(dict(name='catalog_profile', top_cumulative=[
            dict(function=str(k), calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
            for k, v in sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:10]]))


def sampling(Store, root, n, results):
    import numpy as np
    # 24 MiB by default: crosses the operation-local eight-chunk cache.
    coords = array.array('d', (float(i) for i in range(n * 3)))
    if sys.byteorder != 'little':
        coords.byteswap()
    payload = coords.tobytes()
    data = dict(kind='fiber', arrays=[
        dict(name='line_points', dtype='<f8', shape=[n, 3], data=payload),
        dict(name='control_points', dtype='<f8', shape=[0, 3], data=b'')])
    with Store.create(str(root / 'sampling.vcobj'), author='benchmark') as db:
        obj = db.upsert_object(dict(kind='fiber'))
        rev = db.put_geometry(obj, data)
        db.checkpoint()
        i = n // 2 + 7
        xyz = [float(i * 3 + axis) for axis in range(3)]
        hits = db.spatial_regions(xyz * 2, object_kind='fiber')
        assert len(hits) == 1
        hit = hits[0]
        start, count = hit['start'][0] * 24, hit['shape'][0] * 24
        expected_block = payload[start:start+count]
        bench(results, 'fiber_block_query', lambda: db.spatial_regions(xyz * 2, object_kind='fiber'),
              lambda v: assert_equal(v, hits))
        def load_block():
            hit = db.spatial_regions(xyz * 2, object_kind='fiber')[0]
            return db.read_geometry_array(hit['revision_uuid'], hit['array_name'], hit['start'][0]*24, hit['shape'][0]*24)
        bench(results, 'fiber_block_query_and_load', load_block, lambda v: assert_equal(v, expected_block))
        results.append(dict(name='fiber_block_working_set', points=n, total_bytes=len(payload), returned_bytes=count))
        chunk = 1048576
        for chunks in (8, 9, 16):
            # Same 512 requested points; change only cache working set/order.
            ranges = [(((i % chunks) * chunk + 23) // 24 * 24, 24) for i in range(512)]
            expected = [payload[o:o+c] for o, c in ranges]
            def verify(value):
                assert value == expected
            bench(results, f'cyclic_{chunks}_chunks_512_points',
                  lambda: db.read_geometry_ranges(rev, 'line_points', ranges), verify)
            def grouped():
                order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
                values = db.read_geometry_ranges(rev, 'line_points', [ranges[i] for i in order])
                output = [None] * len(ranges)
                for i, value in zip(order, values):
                    output[i] = value
                return output
            bench(results, f'grouped_{chunks}_chunks_512_points', grouped, verify)
        rng = random.Random(42)
        ranges = [(rng.randrange(n) * 24, 24) for _ in range(512)]
        expected = [payload[o:o+c] for o, c in ranges]
        bench(results, 'random_512_points', lambda: db.read_geometry_ranges(rev, 'line_points', ranges), verify)
        bench(results, 'random_512_points_grouped', grouped, verify)
        bench(results, 'full_array', lambda: db.read_geometry_array(rev, 'line_points'), lambda v: count_is(len(payload))(v))
        raw = root / 'raw.bin'
        raw.write_bytes(payload)
        indices = np.array([offset // 24 for offset, count in ranges], dtype=np.int64)
        expected_points = np.frombuffer(b''.join(expected), dtype='<f8').reshape(-1, 3)
        with raw.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            bench(results, 'mmap_random_512_points_no_verification_lower_bound',
                  lambda: [mapped[o:o+c] for o, c in ranges], verify)
            mapped_points = np.ndarray((n, 3), dtype='<f8', buffer=mapped)
            bench(results, 'mmap_numpy_random_512_points_lower_bound', lambda: mapped_points[indices],
                  lambda value: np.testing.assert_array_equal(value, expected_points))
            del mapped_points
        with db.graph_read(object_uuids=[obj]) as graph:
            bench(results, 'graph_load_packed_fiber', lambda: graph.read_geometry_arrays('line_points'),
                  lambda value: assert_equal(value.values.tobytes(), payload))
            resident = graph.read_geometry_arrays('line_points').values
        bench(results, 'resident_packed_random_512_points', lambda: resident[indices],
              lambda value: np.testing.assert_array_equal(value, expected_points))
        # One-coordinate edit: cost and retained storage, independently of links.
        changed = dict(kind='fiber', arrays=[dict(a) for a in data['arrays']])
        counter = 0
        def edit():
            nonlocal rev, counter
            counter += 1
            changed['arrays'][0]['data'] = payload[:-8] + struct.pack('<d', -float(counter))
            rev = db.put_geometry(obj, changed, expected_revision=rev)
            return rev
        bench(results, 'single_coordinate_edit', edit)
        assert db.read_geometry_array(rev, 'line_points', len(payload) - 8, 8) == struct.pack('<d', -float(counter))
        db.checkpoint()
        with sqlite3.connect(db.path) as sql:
            count, size = sql.execute('SELECT count(*),sum(length(data)) FROM geometry_chunks').fetchone()
            results.append(dict(name='retained_geometry_storage', chunks=count, bytes=size,
                                original_bytes=len(payload), edits=counter, database_bytes=Path(db.path).stat().st_size))


def tracks(Store, root, n, results):
    # Many small geometries, unlike the single-large-array sampling case.
    with Store.create(str(root / 'tracks.vcobj'), author='benchmark') as db:
        ids, revisions, payloads = [], [], []
        with db.write() as tx:
            for i in range(n):
                payload = struct.pack('<300i', *(i * 300 + j for j in range(300)))
                obj = tx.upsert_object(dict(kind='track'))
                revisions.append(tx.put_geometry(obj, dict(kind='track', arrays=[
                    dict(name='points', dtype='<i4', shape=[100, 3], data=payload)]), expected_revision=None))
                ids.append(obj)
                payloads.append(payload)
        db.checkpoint()
        bench(results, 'load_current_heads_and_arrays', lambda: [
            db.read_geometry_array(db.current_geometry(i)['uuid'], 'points') for i in ids],
            lambda v: assert_equal(v, payloads))
        bench(results, 'load_arrays_known_revisions', lambda: [db.read_geometry_array(r, 'points') for r in revisions],
              lambda v: assert_equal(v, payloads))
        profile = cProfile.Profile()
        profile.runcall(lambda: [db.read_geometry_array(r, 'points') for r in revisions])
        import pstats
        results.append(dict(name='track_read_python_profile', top_cumulative=[
            dict(function=str(k), calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
            for k, v in sorted(pstats.Stats(profile).stats.items(), key=lambda kv: kv[1][3], reverse=True)[:8]]))
        packed_payload = b''.join(payloads)
        def check_packed(value):
            assert value.values.tobytes() == packed_payload
            assert value.offsets.tolist() == [i*100 for i in range(n+1)]
        bench(results, 'load_arrays_packed', lambda: db.read_geometry_arrays(revisions, 'points'), check_packed)
        by_uuid = dict(zip(ids, payloads))
        with db.graph_read(object_kind='track') as graph:
            graph_payload = b''.join(by_uuid[i] for i in graph.object_uuids)
            bench(results, 'graph_load_arrays_packed', lambda: graph.read_geometry_arrays('points'),
                  lambda value: assert_equal(value.values.tobytes(), graph_payload))
        raw = root / 'packed.bin'
        raw.write_bytes(b''.join(payloads))
        with raw.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            bench(results, 'packed_mmap_copy_lower_bound', lambda: [mapped[i*1200:(i+1)*1200] for i in range(n)],
                  lambda v: assert_equal(v, payloads))


def spatial(Store, root, n, results):
    """A full-resolution curved patch; query and load the same tile both ways."""
    rows, cols = (n, n) if isinstance(n, int) else n
    coords = array.array('f', (value for y in range(rows) for x in range(cols)
                              for value in (x, y, 1 + x / 16)))
    if sys.byteorder != 'little':
        coords.byteswap()
    payload = coords.tobytes()
    data = dict(kind='patch', attributes={'scale': [1, 1]}, arrays=[
        dict(name='points', dtype='<f4', shape=[rows, cols, 3], data=payload)])
    with Store.create(str(root / 'spatial.vcobj'), author='benchmark') as db:
        uid = db.upsert_object(dict(kind='patch'))
        start = time.perf_counter_ns()
        revision = db.put_geometry(uid, data)
        results.append(dict(name='initial_geometry_publication', elapsed_ms=(time.perf_counter_ns()-start)/1e6,
                            payload_bytes=len(payload), shape=[rows, cols, 3]))
        db.checkpoint()
        x, y = min(400, cols // 2) + .25, min(400, rows // 2) + .25
        box = [x, y, 1+x/16, x+.5, y+.5, 1+(x+.5)/16]
        hits = db.spatial_regions(box, object_kind='patch')
        assert len(hits) == 1
        tile = hits[0]
        row, col = tile['start']
        height, width = tile['shape']
        ranges = [((y*cols+col)*12, width*12) for y in range(row, row+height)]
        expected = [payload[o:o+c] for o, c in ranges]
        descriptor = json.loads(sqlite_rows(db.path, 'SELECT manifest FROM geometry_revisions')[0][0])['arrays']['points']
        chunk_sizes = dict(sqlite_rows(db.path, 'SELECT r.chunk_index,length(c.data) FROM geometry_revision_chunks r JOIN geometry_chunks c USING(hash)'))
        size = descriptor.get('storage', {}).get('tile_size', 0)
        if size:
            tile_cols = (cols + size - 1) // size
            touched = {r*tile_cols+c for r in range(row//size, (row+height-1)//size+1)
                       for c in range(col//size, (col+width-1)//size+1)}
        else:
            touched = {k for o, c in ranges for k in range(o//1048576, (o+c-1)//1048576+1)}
        results.append(dict(name='viewport_working_set', candidate_regions=len(hits), returned_bytes=sum(map(len, expected)),
                            chunks_touched=len(touched), chunk_bytes_to_verify=sum(chunk_sizes[k] for k in touched)))
        bench(results, 'spatial_regions_query', lambda: db.spatial_regions(box, object_kind='patch'),
              lambda v: assert_equal(v, hits))
        bench(results, 'whole_object_query', lambda: db.select_objects(bbox=box, object_kind='patch', fields=[]), count_is(1))

        def whole_read():
            obj = db.select_objects(bbox=box, object_kind='patch', fields=[])[0]
            rev = db.current_geometry(obj['uuid'])['uuid']
            data = db.read_geometry_array(rev, 'points')
            return [data[o:o+c] for o, c in ranges]

        def region_read():
            hit = db.spatial_regions(box, object_kind='patch')[0]
            return db.read_geometry_ranges(hit['revision_uuid'], hit['array_name'], ranges)

        bench(results, 'whole_object_query_and_load_then_slice', whole_read, lambda v: assert_equal(v, expected))
        bench(results, 'region_query_and_load_slice', region_read, lambda v: assert_equal(v, expected))
        def rectangle_read():
            hit = db.spatial_regions(box, object_kind='patch')[0]
            return db.read_geometry_rectangle(hit['revision_uuid'], 'points', *hit['start'], *hit['shape'])
        bench(results, 'region_query_and_load_rectangle', rectangle_read, lambda v: assert_equal(v, b''.join(expected)))
        empty_box = [x, y, cols/16-1, x+.5, y+.5, cols/16]
        assert len(db.select_objects(bbox=empty_box, fields=[])) == 1
        bench(results, 'region_query_reject_empty_space_in_object_aabb',
              lambda: db.spatial_regions(empty_box), count_is(0))
        results.append(dict(name='index_storage',
            regions=sqlite_rows(db.path, "SELECT count(*) FROM spatial_regions WHERE array_name!=''")[0][0],
            database_bytes=Path(db.path).stat().st_size,
            index_bytes=sqlite_rows(db.path, "SELECT sum(pgsize) FROM dbstat WHERE name LIKE 'spatial_regions%' OR name LIKE 'sqlite_autoindex_spatial_regions%'")[0][0]))
        predicates = ' AND '.join(f'r.max_{a}>={box[i]} AND r.min_{a}<={box[i+3]} AND b.max_{a}>={box[i]} AND b.min_{a}<={box[i+3]}'
                                  for i, a in enumerate('xyz'))
        query = "SELECT o.uuid,h.revision_uuid FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id " + \
            "CROSS JOIN objects o ON o.id=b.object_id CROSS JOIN object_bounds h ON h.object_id=b.object_id " + \
            "WHERE (b.array_name!='' OR h.revision_uuid IS NULL) AND o.kind='patch' AND " + predicates + \
            " ORDER BY o.uuid,b.array_name,b.row_start,b.col_start"
        results.append(dict(name='spatial_region_query_plan', plan=native_plan(db.path, query)))
        counter = 0
        def edit():
            nonlocal revision, counter
            counter += 1
            # The complete geometry API still hashes/scans the provided array;
            # only one tile and one content-addressed chunk actually change.
            offset = (17*cols+17)*12+8
            changed = dict(data)
            changed['arrays'] = [dict(data['arrays'][0], data=payload[:offset]+struct.pack('<f', 10+counter)+payload[offset+4:])]
            revision = db.put_geometry(uid, changed, expected_revision=revision)
            return revision
        bench(results, 'surface_single_coordinate_edit', edit)
        assert db.read_geometry_array(revision, 'points', (17*cols+17)*12+8, 4) == struct.pack('<f', 10+counter)


def spatial_wide(Store, root, n, results):
    spatial(Store, root, (65, n), results)


def assert_equal(actual, expected):
    assert actual == expected


def patch(Store, root, source, results):
    source = Path(source)
    copied = root / 'patch'
    shutil.copytree(source, copied)
    meta = json.loads((copied / 'meta.json').read_text())
    meta['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(source.resolve())))
    (copied / 'meta.json').write_text(json.dumps(meta))
    with Store.create(str(root / 'patch.vcobj'), author='benchmark') as db:
        report = db.ingest_geometry_source('patch', str(copied))
        assert report['ingested'] == 1 and not report['problems'], report
        revision = db.current_geometry(meta['uuid'])
        descriptor = next(a for a in revision['arrays'] if a['name'] == 'points')
        payload = db.read_geometry_array(revision['uuid'], 'points')
        rows, cols, _ = descriptor['shape']
        results.append(dict(name='real_patch_fixture', source=str(source), shape=descriptor['shape'], bytes=len(payload)))
        ranges = [(r * cols * 12 + (cols // 2) * 12, 12) for r in range(rows)]
        expected = [payload[o:o+c] for o, c in ranges]
        bench(results, 'patch_column_separate', lambda: [db.read_geometry_array(revision['uuid'], 'points', o, c) for o, c in ranges],
              lambda v: assert_equal(v, expected))
        bench(results, 'patch_column_batched', lambda: db.read_geometry_ranges(revision['uuid'], 'points', ranges),
              lambda v: assert_equal(v, expected))
        bench(results, 'patch_full_array', lambda: db.read_geometry_array(revision['uuid'], 'points'),
              lambda v: assert_equal(v, payload))
        with db.graph_read(object_uuids=[meta['uuid']]) as graph:
            bench(results, 'real_patch_graph_packed', lambda: graph.read_geometry_arrays('points'),
                  lambda value: assert_equal(value.values.tobytes(), payload))
        xyz = next(p for p in struct.iter_unpack('<fff', payload) if p[2] > 0)
        bench(results, 'real_patch_spatial_projected',
              lambda: db.select_objects(bbox=list(xyz) + list(xyz), fields=['name']),
              lambda v: assert_equal([r['uuid'] for r in v], [meta['uuid']]))
        tiles = db.spatial_regions(list(xyz) * 2, object_kind='patch')
        assert tiles and all(t['revision_uuid'] == revision['uuid'] for t in tiles)
        bench(results, 'real_patch_spatial_regions',
              lambda: db.spatial_regions(list(xyz) * 2, object_kind='patch'),
              lambda v: assert_equal(v, tiles))
        for t in tiles:
            y, x = t['start']
            height, width = t['shape']
            tile_ranges = [((row * cols + x) * 12, width * 12) for row in range(y, y + height)]
            assert db.read_geometry_ranges(t['revision_uuid'], t['array_name'], tile_ranges) == [payload[o:o+c] for o, c in tile_ranges]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/vcobj-workflows.json'))
    parser.add_argument('--build-python', type=Path, default=Path(__file__).resolve().parents[1] / 'build/python')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 50000])
    parser.add_argument('--patch', type=Path)
    parser.add_argument('--worker', choices=['catalog', 'sampling', 'tracks', 'patch', 'spatial', 'spatial_wide'], help=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, default=1000, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes) < 100:
        parser.error('sizes must be >=100')
    if args.worker in ('spatial', 'spatial_wide') and args.n < 65:
        parser.error('spatial side length must be >=65')
    if args.worker:
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_editable_skbc_volume_cartographer']
        sys.path.insert(0, str(args.build_python.resolve()))
        from vc.objects import ObjectStore
        results = []
        with tempfile.TemporaryDirectory(prefix='vcobj-workflow-') as directory:
            globals()[args.worker](ObjectStore, Path(directory), args.patch if args.worker == 'patch' else args.n, results)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps(dict(scenario=args.worker, n=args.n, results=results,
                              module=sys.modules['vc.objects'].__file__,
                              peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024)))
        return
    cases = [('catalog', n) for n in args.sizes] + [('sampling', 1048576)] + [('tracks', n) for n in args.sizes if n <= 10000]
    cases += [('spatial', 1025), ('spatial', 2049), ('spatial_wide', 65537)]
    if args.patch:
        cases.append(('patch', 0))
    report = dict(platform=platform.platform(), python=sys.version,
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python_sqlite=sqlite3.sqlite_version,
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  scenarios=[])
    cache = args.build_python.parent / 'CMakeCache.txt'
    if cache.exists():
        report['build'] = [line for line in cache.read_text().splitlines() if line.startswith((
            'CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS_RELWITHDEBINFO:'))]
    for scenario, n in cases:
        print(f'\n{scenario} n={n}', file=sys.stderr, flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--build-python', str(args.build_python.resolve()),
                   '--worker', scenario, '--n', str(n)]
        if args.patch:
            command.extend(['--patch', str(args.patch.resolve())])
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, timeout=300)
        report['scenarios'].append(json.loads(completed.stdout))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
