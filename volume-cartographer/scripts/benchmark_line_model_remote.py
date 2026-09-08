#!/usr/bin/env python3
"""Replay a saved GUI fiber against isolated, real-remote cold/disk-warm caches.

Never clears an existing cache. Each pair creates a new cache with only the
original manifest and remote marker; warm starts after the cold process exits.
Requires the instrumented vc_fiber_trace_metric, Python stdlib, and network access.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from urllib.parse import urlsplit

from benchmark_line_model_prefetch import (
    configured_readers_match, file_sha256, parse_metric_output,
    positive_finite_seconds, resolve_source_path, write_json)
from analyze_line_model_prefetch import canonical_trace, digest, normalized_message


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink in isolated cache: {path}')
        if path.is_file():
            result[str(path.relative_to(root))] = {
                'sha256': file_sha256(path), 'bytes': path.stat().st_size}
    return result


def binary_artifacts(binary):
    """Hash the metric and repository DSOs in the supported CMake build layout.

    Both Ubuntu .so and macOS .dylib names are supported. System libraries are
    external toolchain provenance, not silently represented by the binary hash.
    """
    build = binary.parent.parent
    result = {'bin/' + binary.name: file_sha256(binary)}
    for relative in ('core/libvc_lasagna', 'core/libvc_fiber_tracer', 'core/libvc_core',
                     'utils/libutils', 'utils/libutils_c3d_codec', 'libs/c3d/libc3d'):
        found = False
        for suffix in ('.so', '.dylib'):
            path = build / (relative + suffix)
            if path.is_file():
                result[relative + suffix] = file_sha256(path)
                found = True
        if not found:
            raise ValueError(f'missing repository library in supported build layout: {relative}')
    return result


def merge_payloads(known, current):
    for key, value in current.items():
        if key in known and known[key] != value:
            raise ValueError(f'remote payload changed between trials: {key}')
        known[key] = value


def result_signature(trace, profile):
    return digest(dict(trace=canonical_trace(trace),
                       message=normalized_message(trace['optimization']['message']),
                       candidates=profile['candidates'], generations=profile['generations']))


def verified_run(directory):
    configuration = json.loads((directory / 'configuration.json').read_text())
    if inventory(directory / 'input') != configuration['input_hashes']:
        raise ValueError(f'comparison input changed: {directory}')
    rows = json.loads((directory / 'results.json').read_text())
    expected = {(mode, trial) for mode in ('cold', 'warm')
                for trial in range(1, configuration['trials'] + 1)}
    if len(rows) != len(expected) or {(r['mode'], r['trial']) for r in rows} != expected:
        raise ValueError(f'incomplete comparison: {directory}')
    cached = {trial: inventory(directory / f'cache-{trial}')
              for trial in range(1, configuration['trials'] + 1)}
    for row in rows:
        if row['returncode'] or row['metrics']['pending_requests'] != '0' or row['metrics']['remote_failures'] != '0':
            raise ValueError('comparison contains failed/incomplete reads')
        stem = f"{row['mode']}-{row['trial']}"
        trace = directory / (stem + '.trace.json')
        if file_sha256(trace) != row['trace_sha256'] or result_signature(
                json.loads(trace.read_text()), row['profile']) != row['result_signature']:
            raise ValueError('comparison trace changed')
        if row['inventory'] != stem + '.cache.json':
            raise ValueError('comparison inventory path mismatch')
        for key, value in json.loads((directory / row['inventory']).read_text()).items():
            if cached[row['trial']].get(key) != value:
                raise ValueError(f'comparison cache payload changed: {key}')
    return configuration, rows


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    if not manifest.get('groups'):
        raise ValueError('only grouped relative-path manifests are supported')
    for group in manifest['groups'].values():
        route = group['zarr']
        if urlsplit(route).scheme or Path(route).is_absolute():
            raise ValueError('manifest must use relative model paths')
        resolve_source_path(path, route)
    marker = path.parent / 'lasagna-remote.json'
    remote = json.loads(marker.read_text())
    if remote.get('manifest_file') != path.name:
        raise ValueError('remote marker does not match manifest')
    if urlsplit(remote['artifact_url']).scheme not in ('http', 'https'):
        raise ValueError('remote artifact must be HTTP(S)')
    return marker, remote['artifact_url'].rstrip('/') + '/' + path.name


def summarize(rows):
    summary = {}
    for mode in ('cold', 'warm'):
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {}
        for field in ('process_wall_s', 'trace_wall_s', 'remote_bytes',
                      'span_trace_ms', 'reinit_ms', 'tail_trace_ms', 'report_prefetch_ms'):
            values = [float(r[field] if field in r else r['metrics'][field]) for r in selected]
            summary[mode][field] = dict(mean=statistics.mean(values), min=min(values),
                                       median=statistics.median(values), max=max(values))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('output-dir', 'binary', 'fiber-json', 'fiber-manifest', 'normal-manifest'):
        parser.add_argument('--' + option, required=True, type=Path)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1, help='Native scoring threads; not Ceres threads.')
    parser.add_argument('--model-readers', type=int, default=64)
    parser.add_argument('--warmup-ms', type=int, default=0, help='Saved-corridor warmup before timed solve, 0..10000.')
    parser.add_argument('--dirty-span', type=int, help='Zero-based saved CP span; omitted means full retrace.')
    parser.add_argument('--prefetch', type=int, choices=(0, 1), required=True)
    parser.add_argument('--timeout-s', type=positive_finite_seconds, default=300)
    parser.add_argument('--compare-run', action='append', type=Path, default=[])
    args = parser.parse_args()
    if min(args.trials, args.threads, args.model_readers) < 1:
        parser.error('trials, threads and readers must be positive')
    if not 0 <= args.warmup_ms <= 10000:
        parser.error('warmup must be 0..10000 ms')
    source = args.fiber_json.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    artifacts = binary_artifacts(binary)
    original = json.loads(source.read_text())
    if args.dirty_span is not None and not 0 <= args.dirty_span < len(original['control_points']) - 1:
        parser.error('dirty span outside saved control spans')
    manifests = dict(fiber=args.fiber_manifest.resolve(strict=True),
                     normal=args.normal_manifest.resolve(strict=True))
    validated = {kind: validate_manifest(path) for kind, path in manifests.items()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'input'
    inputs.mkdir()
    shutil.copy2(source, inputs / 'fiber.json')
    frozen = {}
    for kind, path in manifests.items():
        directory = inputs / kind
        directory.mkdir()
        shutil.copy2(path, directory / path.name)
        shutil.copy2(validated[kind][0], directory / 'lasagna-remote.json')
        frozen[kind] = directory / path.name
    provenance = dict(input_hashes=inventory(inputs), dirty_span=args.dirty_span,
                      threads=args.threads, model_readers=args.model_readers, cache_gib=0.5,
                      legacy_read_workers=os.environ.get('VC_LASAGNA_READ_WORKERS', 'default'))
    write_json(output / 'configuration.json', dict(
        **provenance, source=str(source), binary=str(binary),
        binary_sha256=file_sha256(binary), prefetch=args.prefetch, trials=args.trials,
        binary_artifacts=artifacts,
        warmup_ms=args.warmup_ms,
        environment=dict(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1'),
        warm_definition='Fresh process, disk cache from paired cold run; decoded caches start empty.'))
    signatures, payloads = set(), {}
    for other in args.compare_run:
        configuration, prior = verified_run(other)
        if any(configuration[key] != value for key, value in provenance.items()):
            raise ValueError(f'incompatible comparison workload/settings: {other}')
        for row in prior:
            signatures.add(row['result_signature'])
            merge_payloads(payloads, json.loads((other / row['inventory']).read_text()))
    results = []
    for trial in range(1, args.trials + 1):
        cache = output / f'cache-{trial}'
        shutil.copytree(inputs / 'fiber', cache / 'fiber')
        shutil.copytree(inputs / 'normal', cache / 'normal')
        for mode in ('cold', 'warm'):
            stem = f'{mode}-{trial}'
            trace = output / (stem + '.trace.json')
            command = [str(binary), str(cache / 'fiber' / frozen['fiber'].name),
                       str(inputs / 'fiber.json'), '--normal-manifest',
                       str(cache / 'normal' / frozen['normal'].name),
                       '--gui-reoptimize', '--threads', str(args.threads),
                       '--model-readers', str(args.model_readers), '--quiet',
                       '--cache-gib', '0.5', '--prefetch-warmup-ms', str(args.warmup_ms),
                       '--remote-cache-dir', str(cache), '--settle-prefetch-ms', '10000',
                       '--trace-output', str(trace)]
            for kind in manifests:
                command += ['--' + kind + '-manifest-identity', validated[kind][1]]
            if args.dirty_span is not None:
                command += ['--gui-dirty-span', str(args.dirty_span)]
            print(f'Starting {output.name} {stem}', flush=True)
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True,
                timeout=args.timeout_s, env={**os.environ, 'AGENTS_AGENT_MODE': '1',
                    'VC3D_LINE_MODEL_PREFETCH': str(args.prefetch),
                    'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1'})
            elapsed = time.perf_counter() - started
            (output / (stem + '.stdout')).write_text(completed.stdout)
            (output / (stem + '.stderr')).write_text(completed.stderr)
            metrics, profile, warmup = parse_metric_output(completed.stdout)
            row = dict(mode=mode, trial=trial, command=command, returncode=completed.returncode,
                       process_wall_s=elapsed, metrics=metrics, profile=profile, warmup=warmup)
            results.append(row)
            write_json(output / 'results.json', results)
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            if not configured_readers_match(['--model-readers', str(args.model_readers)], metrics):
                raise ValueError('reader configuration not confirmed')
            expected_span = str(args.dirty_span if args.dirty_span is not None else -1)
            if metrics.get('dirty_span') != expected_span:
                raise ValueError('dirty span not confirmed by binary')
            if metrics.get('pending_requests') != '0' or metrics.get('remote_failures') != '0':
                raise ValueError('incomplete remote accounting or failed model reads')
            parsed = json.loads(trace.read_text())
            row['result_signature'] = result_signature(parsed, profile)
            row['trace_sha256'] = file_sha256(trace)
            row['inventory'] = stem + '.cache.json'
            cached = inventory(cache)
            write_json(output / row['inventory'], cached)
            merge_payloads(payloads, cached)
            signatures.add(row['result_signature'])
            write_json(output / 'results.json', results)
            if len(signatures) != 1:
                raise ValueError('exact result/decisions changed between cache states or variants')
            if inventory(inputs) != provenance['input_hashes']:
                raise ValueError('frozen input changed')
            print(json.dumps(dict(run=stem, wall_s=elapsed, metrics=metrics)), flush=True)
    if binary_artifacts(binary) != artifacts:
        raise ValueError('binary or repository libraries changed during measurements')
    summary = summarize(results)
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
