#!/usr/bin/env python3
"""Compare physical patch layouts with identical logical bytes and queries.

Build: ninja -C build vc_object_bindings
Run: .venv/bin/python scripts/benchmark_object_store_tiles.py --output /tmp/vcobj-tiles.json
Workers use fresh temporary stores, FULL durability, one warmup and three
measured repetitions. Checks are outside timings. No OS cache dropping.
"""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time

from benchmark_object_store_workflows import bench, assert_equal
from benchmark_object_store import sqlite_rows


def worker(Store, root, rows, cols, tile):
    results = []
    coords = array.array('f', (v for y in range(rows) for x in range(cols) for v in (x, y, 1+x/16)))
    if sys.byteorder != 'little':
        coords.byteswap()
    payload = coords.tobytes()
    data = dict(kind='patch', attributes={'scale': [1, 1]}, arrays=[
        dict(name='points', dtype='<f4', shape=[rows, cols, 3], data=payload)])
    with Store.create(str(root / 'source.vcobj'), author='benchmark') as db:
        uid = db.upsert_object(dict(kind='patch'))
        started = time.perf_counter_ns()
        rev = db.put_geometry(uid, data, patch_tile_size=tile)
        results.append(dict(name='initial_publication', elapsed_ms=(time.perf_counter_ns()-started)/1e6))
        db.checkpoint()
        x, y = min(400, cols//2)+.25, min(400, rows//2)+.25
        box = [x, y, 1+x/16, x+.5, y+.5, 1+(x+.5)/16]
        hits = db.spatial_regions(box, object_kind='patch')
        assert len(hits) == 1
        sy, sx = hits[0]['start']
        h, w = hits[0]['shape']
        ranges = [((r*cols+sx)*12, w*12) for r in range(sy, sy+h)]
        expected_rows = [payload[o:o+c] for o, c in ranges]
        expected = b''.join(expected_rows)
        if tile:
            tile_cols = (cols+tile-1)//tile
            touched = [r*tile_cols+c for r in range(sy//tile, (sy+h-1)//tile+1)
                       for c in range(sx//tile, (sx+w-1)//tile+1)]
        else:
            touched = sorted({k for o, c in ranges for k in range(o//1048576, (o+c-1)//1048576+1)})
        sizes = dict(sqlite_rows(db.path, 'SELECT r.chunk_index,length(c.data) FROM geometry_revision_chunks r JOIN geometry_chunks c USING(hash)'))
        manifest_bytes = sqlite_rows(db.path, 'SELECT length(manifest) FROM geometry_revisions')[0][0]
        results.append(dict(name='working_set', payload_bytes=len(payload), returned_bytes=len(expected),
            verified_bytes=sum(sizes[k] for k in touched), chunks_touched=len(touched),
            total_chunks=len(sizes), manifest_bytes=manifest_bytes, database_bytes=Path(db.path).stat().st_size))
        bench(results, 'full_array', lambda: db.read_geometry_array(rev, 'points'), lambda v: assert_equal(v, payload))
        def rectangle():
            hit = db.spatial_regions(box, object_kind='patch')[0]
            return db.read_geometry_rectangle(hit['revision_uuid'], 'points', *hit['start'], *hit['shape'])
        def byte_ranges():
            hit = db.spatial_regions(box, object_kind='patch')[0]
            return db.read_geometry_ranges(hit['revision_uuid'], 'points', ranges)
        bench(results, 'query_and_rectangle', rectangle, lambda v: assert_equal(v, expected))
        bench(results, 'query_and_byte_ranges', byte_ranges, lambda v: assert_equal(v, expected_rows))
        # Crossing storage boundaries rather than favoring one alignment.
        by, bx = min(127, rows-33), min(127, cols-33)
        boundary = b''.join(payload[(r*cols+bx)*12:(r*cols+bx+33)*12] for r in range(by, by+33))
        bench(results, 'boundary_rectangle', lambda: db.read_geometry_rectangle(rev, 'points', by, bx, 33, 33),
              lambda v: assert_equal(v, boundary))
        counter = 0
        current_payload = payload
        def edit():
            nonlocal counter, rev, current_payload
            counter += 1
            offset = (17*cols+17)*12+8
            current_payload = payload[:offset]+struct.pack('<f', 10+counter)+payload[offset+4:]
            changed = dict(data, arrays=[dict(data['arrays'][0], data=current_payload)])
            rev = db.put_geometry(uid, changed, expected_revision=rev, patch_tile_size=tile)
            return rev
        bench(results, 'single_coordinate_edit', edit)
        assert db.read_geometry_array(rev, 'points') == current_payload
        retained = sqlite_rows(db.path, 'SELECT sum(length(data)) FROM geometry_chunks')[0][0]
        results.append(dict(name='edit_storage', edits=counter, new_payload_bytes=retained-len(payload)))
        clone = 0
        def sync():
            nonlocal clone
            clone += 1
            path = root / f'peer-{clone}.vcobj'
            with Store.create(str(path), author='peer') as peer:
                db.synchronize(peer)
            return path
        def check_sync(path):
            with Store.open(str(path), read_only=True) as peer:
                assert peer.read_geometry_array(rev, 'points') == current_payload
                assert peer.spatial_regions(box) == db.spatial_regions(box)
        bench(results, 'initial_sync_with_history_and_index', sync, check_sync)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/vcobj-tiles.json'))
    parser.add_argument('--build-python', type=Path, default=Path(__file__).resolve().parents[1]/'build/python')
    parser.add_argument('--tile-sizes', type=int, nargs='+', default=[0, 32, 64, 128], choices=[0, 32, 64, 128])
    parser.add_argument('--shapes', nargs='+', default=['1025x1025', '2049x2049', '65x65537'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    shapes = [tuple(map(int, shape.split('x'))) for shape in args.shapes]
    if any(len(shape) != 2 or min(shape) < 65 for shape in shapes):
        parser.error('each shape must be ROWSxCOLS with dimensions >=65')
    if args.worker:
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_editable_skbc_volume_cartographer']
        sys.path.insert(0, str(args.build_python.resolve()))
        from vc.objects import ObjectStore
        with tempfile.TemporaryDirectory(prefix='vcobj-tile-') as folder:
            result = worker(ObjectStore, Path(folder), *shapes[0], args.tile_sizes[0])
        print(json.dumps(dict(shape=shapes[0], tile_size=args.tile_sizes[0], results=result,
            module=sys.modules['vc.objects'].__file__, peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
        return
    report = dict(platform=platform.platform(), python=sys.version, python_sqlite=sqlite3.sqlite_version,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path('core/src/vc_object_store/geometry/Geometry.cpp'), Path('core/src/vc_object_store/geometry/GeometryInternal.hpp')]},
        build=[line for line in (args.build_python.parent/'CMakeCache.txt').read_text().splitlines()
               if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS_RELWITHDEBINFO:'))],
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, scenarios=[])
    for shape in args.shapes:
        for tile in args.tile_sizes:
            print(f'\nshape={shape}, tile={tile}', file=sys.stderr, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--shapes', shape,
                       '--tile-sizes', str(tile), '--build-python', str(args.build_python.resolve())]
            completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, timeout=300)
            report['scenarios'].append(json.loads(completed.stdout))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
