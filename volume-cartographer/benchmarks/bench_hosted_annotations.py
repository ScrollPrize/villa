#!/usr/bin/env python3
"""Read-only nearby-annotation benchmark through an existing HTTP/SSH forward.

The proxy counts requests/response bytes and can delay each hosted response.
VC3D performs the actual spatial query, decoding, scheduling and intersections.
All project and cache writes go beneath --output; the upstream is read-only.
"""
import argparse
import collections
import http.client
import http.server
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--volume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--delay-ms', type=float, default=0)
    parser.add_argument('--stress', action='store_true', help='Also test pending requests, invalidation and controller destruction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = urllib.parse.urlsplit(args.upstream)
    if upstream.scheme != 'http':
        parser.error('Use an HTTP endpoint or a localhost SSH forward')
    lock = threading.Lock()
    counters = collections.Counter()

    class Proxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            if self.path.startswith('/benchmark/'):
                with lock:
                    if self.path.endswith('/reset'):
                        counters.clear()
                    payload = json.dumps(dict(counters)).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.command == 'POST' and not self.path.startswith('/v1/query/'):
                self.send_error(403, 'Benchmark permits only read queries')
                return
            size = int(self.headers.get('Content-Length', '0'))
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=120)
            try:
                started = time.monotonic()
                connection.request(self.command, self.path, self.rfile.read(size), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                payload = response.read()
                upstream_ms = (time.monotonic() - started) * 1000
                route = self.path.split('?')[0]
                if route.startswith('/v1/geometry/'):
                    route = 'geometry/arrays' if '/arrays/' in route else 'geometry/annotations' if '/annotations/' in route else 'geometry/descriptor'
                elif route.startswith('/v1/objects/'):
                    route = 'objects/head'
                elif route.startswith('/v1/artifacts/'):
                    route = 'artifacts'
                with lock:
                    counters['route_ms:' + route] += upstream_ms
                    counters['route_requests:' + route] += 1
                    counters['requests'] += 1
                    counters['freshness_requests' if self.path == '/v1/status' else 'query_requests'] += 1
                    counters['response_bytes'] += len(payload)
                    if '/arrays/' in self.path or self.path.startswith('/v1/artifacts/'):
                        counters['payload_requests'] += 1
                        counters['payload_bytes'] += len(payload)
                if args.delay_ms:
                    time.sleep(args.delay_ms / 1000)
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/octet-stream'))
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                connection.close()

    proxy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{proxy.server_port}'
    results = []
    try:
        for run in range(args.runs):
            for case, mode, cache, repeats in (
                ('sync-cold', 'sync', f'sync-{run}', 1),
                ('async-cold', 'async', f'async-{run}', 2),
                ('async-disk-warm', 'async', f'async-{run}', 1),
            ):
                env = os.environ.copy()
                env.update(QT_QPA_PLATFORM='offscreen', VC3D_CONFIG_DIR=str(args.output / f'config-{case}-{run}'),
                           VC3D_NEARBY_BENCH_ENDPOINT=endpoint, VC3D_NEARBY_BENCH_STATS=endpoint,
                           VC3D_NEARBY_BENCH_VOLUME=str(args.volume), VC3D_NEARBY_BENCH_CACHE=str(args.output / cache),
                           VC3D_NEARBY_BENCH_MODE=mode, VC3D_NEARBY_BENCH_REPETITIONS=str(repeats))
                if args.stress and case == 'async-cold' and run == 0:
                    env['VC3D_NEARBY_BENCH_STRESS'] = '1'
                completed = subprocess.run([str(args.binary), '--test-object-store'], env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
                (args.output / f'{case}-{run}.log').write_text(completed.stdout)
                if completed.returncode:
                    raise RuntimeError(f'{case} failed: {completed.stdout[-5000:]}')
                for line in completed.stdout.splitlines():
                    if not line.startswith('NEARBY_BENCH '):
                        continue
                    result = json.loads(line.removeprefix('NEARBY_BENCH '))
                    result.update(case='async-memory-warm' if result['iteration'] else case, run=run)
                    results.append(result)
                    print(json.dumps(result), flush=True)
        assert len(results) == args.runs * 4, 'Every benchmark case must report results'
        assert all(r['candidates'] and r['markers'] for r in results), 'Use a strip with candidate geometry and intersections'
        assert len({r['geometry_sha256'] for r in results}) == 1, 'Candidate geometry differs'
        assert len({r['markers_sha256'] for r in results}) == 1, 'Intersection markers differ'
        for result in results:
            if result['case'] == 'async-memory-warm':
                assert result['network'].get('query_requests', 0) == 0, 'Unchanged query reached store'
            if result['case'] == 'async-disk-warm':
                assert result['network'].get('payload_requests', 0) == 0, 'Restart downloaded geometry'
        summary = {}
        for case in sorted({r['case'] for r in results}):
            group = [r for r in results if r['case'] == case]
            summary[case] = {}
            for metric in ('elapsed_ms', 'max_event_gap_ms'):
                values = sorted(r[metric] for r in group)
                summary[case][metric] = {'mean': statistics.mean(values), 'p50': statistics.median(values),
                                         'p95': values[min(len(values) - 1, int(.95 * len(values)))]}
        (args.output / 'results.json').write_text(json.dumps({'runs': results, 'summary': summary}, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join()


if __name__ == '__main__':
    main()
