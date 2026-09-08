#!/usr/bin/env python3
"""Repeat real-fiber tracing through local and delayed loopback chunk stores."""
import argparse
import collections
import contextlib
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import urllib.parse

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def positive_finite_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('seconds must be finite and positive') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('seconds must be finite and positive')
    return seconds

def file_sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def resolve_source_path(manifest, relative):
    root = manifest.parent.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('source route escapes its manifest directory')
    return path

def drain_trial_requests(server, start_row, timeout=5):
    with server.rows_condition:
        drained = server.rows_condition.wait_for(
            lambda: server.active_requests == 0, timeout=timeout)
        return drained, list(server.rows[start_row:])

def require_drained_requests(row):
    # Legacy rows have no flag; the analyzer reports that limitation explicitly.
    if 'requests_drained' in row and row['requests_drained'] is not True:
        raise ValueError('trial requests did not drain; subsequent trial labels would be unreliable')

def configured_readers_match(extra_args, metrics):
    if '--model-readers' not in extra_args:
        return True
    index = len(extra_args)-1-extra_args[::-1].index('--model-readers')
    requested = str(int(extra_args[index+1]))
    return metrics.get('configured_max') == requested and metrics.get('adaptive') == '0'

def parse_metric_output(stdout):
    """Shared CLI protocol for loopback and real-remote benchmark runners."""
    profile, metrics, warmup = {}, {}, {}
    for line in stdout.splitlines():
        if not line.startswith('native_trace2cp_'):
            continue
        fields = dict(item.split('=', 1) for item in line.split()[1:] if '=' in item)
        if line.startswith('native_trace2cp_profile'):
            profile = {key:float(value) for key, value in fields.items()}
        elif line.startswith('native_trace2cp_warmup'):
            warmup = {key:float(value) for key, value in fields.items()}
        else:
            metrics.update(fields)
    return metrics, profile, warmup

class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass
    def do_HEAD(self):
        self.serve(False)
    def do_GET(self):
        self.serve(True)
    def serve(self, body):
        started = time.perf_counter()
        with self.server.rows_condition:
            run_tag = self.server.run_tag
            self.server.active_requests += 1
        status, size = 500, 0
        sent = 0
        interrupted = False
        headers_started = False
        error = None
        try:
            # Route parsing, filesystem access, and response writing all belong
            # to this guarded request lifetime, including HEAD requests.
            route = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).strip('/')
            prefix, _, relative = route.partition('/')
            root = self.server.manifests.get(prefix)
            path = resolve_source_path(root, relative) if root else None
            with contextlib.ExitStack() as opened:
                src = opened.enter_context(path.open('rb')) if path and path.is_file() else None
                status = 200 if src is not None else 404
                size = os.fstat(src.fileno()).st_size if src is not None else 0
                time.sleep(self.server.delay)
                headers_started = True
                self.send_response(status)
                self.send_header('Content-Length', str(size))
                self.end_headers()
                if src is not None and body:
                    while chunk := src.read(65536):
                        if self.server.bytes_per_second:
                            now = time.perf_counter()
                            with self.server.bandwidth_lock:
                                finish = max(now, self.server.next_byte_time) + len(chunk)/self.server.bytes_per_second
                                self.server.next_byte_time = finish
                            time.sleep(max(0, finish-time.perf_counter()))
                        self.wfile.write(chunk)
                        sent += len(chunk)
        except (OSError, ValueError) as exc:
            error = type(exc).__name__
            interrupted = headers_started
            self.close_connection = True
            if not headers_started:
                status = 400 if isinstance(exc, ValueError) else (
                    404 if isinstance(exc, FileNotFoundError) else 500)
                size = 0
                try:
                    self.send_response(status)
                    self.send_header('Content-Length', '0')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                except OSError:
                    interrupted = True
        finally:
            row = {'method': self.command, 'path': self.path, 'status': status,
                   'bytes': sent, 'expected_bytes':size if body else 0,
                   'interrupted':interrupted, 'run':run_tag, 'start_s':started,
                   'elapsed_s':time.perf_counter()-started}
            if error:
                row['error'] = error
            with self.server.rows_condition:
                self.server.rows.append(row)
                self.server.active_requests -= 1
                self.server.rows_condition.notify_all()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path, help='New directory; never overwrite prior trials.')
    p.add_argument('--binary', required=True, type=Path)
    p.add_argument('--fiber-json', required=True, type=Path)
    p.add_argument('--fiber-manifest', required=True, type=Path)
    p.add_argument('--normal-manifest', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--timeout-s', type=float, default=180)
    p.add_argument('--drain-timeout-s', type=positive_finite_seconds, default=5.0,
                   help='Time to drain trailing HTTP requests after each process exits; increase for slow links.')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--delay-ms', type=float, default=50)
    p.add_argument('--bandwidth-mib', type=float, default=0,
                   help='Aggregate response body MiB/s across requests; 0 is unlimited.')
    p.add_argument('--modes', nargs='+', choices=['local','cold','warm'], default=['local','cold','warm'])
    p.add_argument('--extra-arg', action='append', default=[],
                   help='Append one argument to the metric command; use --extra-arg=--option for option names.')
    p.add_argument('--save-traces', action='store_true',
                   help='Request a separate exact trace artifact for every trial (updated CLI only).')
    args = p.parse_args()
    if args.trials < 1 or args.threads < 1:
        p.error('trials and threads must be positive')
    if not all(math.isfinite(v) and v >= 0 for v in (args.delay_ms, args.bandwidth_mib)):
        p.error('delay and bandwidth must be finite and nonnegative')
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        p.error('timeout must be finite and positive')
    if len(set(args.modes)) != len(args.modes):
        p.error('modes must be unique')
    if 'warm' in args.modes and ('cold' not in args.modes or args.modes.index('warm') < args.modes.index('cold')):
        p.error('warm must follow cold to reuse its populated disk cache')
    binary = args.binary.resolve(strict=True)
    fiber_input = args.fiber_json.resolve(strict=True)
    manifests = {'fiber':args.fiber_manifest.resolve(strict=True),
                 'normal':args.normal_manifest.resolve(strict=True)}
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads(fiber_input.read_text())
    write_json(run_dir/'workload.json', {
        'source':str(fiber_input), 'source_sha256':file_sha256(fiber_input),
        'line_points':len(original['line_points']), 'control_points':len(original['control_points']),
        'manifests':{k:str(v) for k,v in manifests.items()},
        'manifest_sha256':{k:file_sha256(v) for k,v in manifests.items()},
        'note':'Complete original saved fiber and original manifest bytes; no crop or source changes.'})
    server = Server(('127.0.0.1', 0), Handler)
    server.manifests = manifests
    server.delay = args.delay_ms / 1000
    server.rows = []
    server.rows_lock = threading.Lock()
    server.rows_condition = threading.Condition(server.rows_lock)
    server.active_requests = 0
    server.run_tag = ''
    server.bytes_per_second = args.bandwidth_mib * 1024**2
    server.bandwidth_lock = threading.Lock()
    server.next_byte_time = 0
    write_json(run_dir/'configuration.json', {
        'delay_ms':args.delay_ms, 'bandwidth_mib':args.bandwidth_mib,
        'drain_timeout_s':args.drain_timeout_s,
        'binary':str(binary), 'binary_sha256':file_sha256(binary),
        'threads':args.threads, 'model_prefetch':os.environ.get('VC3D_LINE_MODEL_PREFETCH','default'),
        'trials':args.trials, 'modes':args.modes, 'extra_args':args.extra_arg,
        'save_traces':args.save_traces})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = []
    try:
        for trial in range(1, args.trials+1):
            # A cold run's cache is reused only by its paired warm run.
            cache = run_dir / f'cache-{trial}'
            remote = {}
            for kind, source in manifests.items():
                directory = cache / kind
                directory.mkdir(parents=True)
                remote[kind] = directory / source.name
                shutil.copy2(source, remote[kind])
                write_json(directory / 'lasagna-remote.json', {
                    'artifact_url': f'http://127.0.0.1:{server.server_port}/{kind}',
                    'manifest_file': source.name, 'anonymous': True})
            for mode in args.modes:
                paths = manifests if mode == 'local' else remote
                command = [str(binary), str(paths['fiber']), str(fiber_input),
                           '--normal-manifest', str(paths['normal']), '--threads', str(args.threads), '--quiet']
                command.extend(args.extra_arg)
                if '--gui-reoptimize' in args.extra_arg:
                    command.extend(['--normal-manifest-identity',str(manifests['normal']),
                                    '--fiber-manifest-identity',str(manifests['fiber'])])
                trace_path = run_dir/f'{mode}-{trial}.trace.json'
                if args.save_traces:
                    command.extend(['--trace-output', str(trace_path)])
                if mode != 'local':
                    command.extend(['--remote-cache-dir', str(cache)])
                with server.rows_lock:
                    start_row = len(server.rows)
                    server.run_tag = f'{mode}-{trial}'
                print(f'Starting {run_dir.name} {mode} trial {trial}', flush=True)
                started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True,
                                           env={**os.environ, 'AGENTS_AGENT_MODE':'1',
                                                'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':str(args.threads)}, timeout=args.timeout_s)
                elapsed = time.perf_counter()-started
                stem = run_dir/f'{mode}-{trial}'
                stem.with_suffix('.stdout').write_text(completed.stdout)
                stem.with_suffix('.stderr').write_text(completed.stderr)
                requests_drained, requests = drain_trial_requests(
                    server, start_row, timeout=args.drain_timeout_s)
                write_json(stem.with_suffix('.requests.json'), requests)
                metrics, profile, warmup = parse_metric_output(completed.stdout)
                row = {'mode':mode, 'trial':trial, 'command':command, 'process_wall_s':elapsed,
                       'returncode':completed.returncode, 'request_count':len(requests),
                       'requests_drained':requests_drained,
                       'request_bytes':sum(r['bytes'] for r in requests),
                       'request_status_counts':dict(collections.Counter(r['status'] for r in requests)),
                       'metrics':metrics, 'profile':profile, 'warmup':warmup}
                if args.save_traces and trace_path.exists():
                    row['trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                results.append(row)
                write_json(run_dir/'results.json', results)
                print(json.dumps({k:v for k,v in row.items() if k not in ('command','profile')}), flush=True)
                require_drained_requests(row)
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                if not configured_readers_match(args.extra_arg, metrics):
                    raise RuntimeError('binary did not confirm requested fixed reader configuration')
    finally:
        server.shutdown()
        server.server_close()
    summaries = {}
    for mode in args.modes:
        rows = [r for r in results if r['mode']==mode]
        summaries[mode] = {}
        for field in ('process_wall_s','request_count','request_bytes'):
            values = [r[field] for r in rows]
            summaries[mode][field] = {'mean':statistics.mean(values),'min':min(values),
                                      'median':statistics.median(values),'max':max(values)}
        for field in ('trace_wall_s','trace_cpu_s'):
            values = [float(r['metrics'][field]) for r in rows]
            summaries[mode][field] = {'mean':statistics.mean(values),'min':min(values),
                                      'median':statistics.median(values),'max':max(values)}
    write_json(run_dir/'summary.json', summaries)
    print(json.dumps(summaries, indent=2), flush=True)

if __name__ == '__main__':
    main()
