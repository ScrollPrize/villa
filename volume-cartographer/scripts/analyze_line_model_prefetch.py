#!/usr/bin/env python3
"""Validate exact optimizer outputs and source/cache byte identity from benchmark runs."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import urllib.parse
import warnings

from benchmark_line_model_prefetch import (
    configured_readers_match, file_sha256, require_drained_requests, resolve_source_path)


def canonical_trace(trace):
    """Retain every returned field except GUI prose containing timing tables."""
    result = copy.deepcopy(trace)
    if result.get('format') == 'vc_gui_fiber_reoptimization_exact_v1':
        result['optimization'].pop('message', None)
    return result


def normalized_message(message):
    """Mask only known timing fields; retain solver status, reasons, and counts."""
    message = re.sub(r'(?m)^(reinit-reopt\+global\s+)\S+', r'\1<TIME>', message)
    message = re.sub(r'\b(ceres_solve_ms|chunk_prefetch_ms|materialize_ms|total_ms)=\S+',
                     r'\1=<TIME>', message)
    start = message.find('Time (in seconds):')
    stop = message.find('Termination:', start)
    if start >= 0 and stop >= 0:
        timing = message[start:stop]
        timing = re.sub(
            r'(?m)^(\s*(?:Preprocessor|Residual only evaluation|Jacobian & residual evaluation|'
            r'Linear solver|Minimizer|Postprocessor|Total)\s+)\d+\.\d+', r'\1<TIME>', timing)
        message = message[:start] + timing + message[stop:]
    return message


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_provenance(directory, configuration, manifest_overrides):
    workload = json.loads((directory/'workload.json').read_text())
    if not workload.get('source') or not workload.get('source_sha256') or not workload.get('manifests'):
        raise ValueError(f'{directory}: recorded workload provenance is missing')
    source = Path(workload['source']).resolve(strict=True)
    if file_sha256(source) != workload['source_sha256']:
        raise ValueError(f'fiber source provenance changed: {source}')
    manifests = {kind:Path(path) for kind, path in workload['manifests'].items()}
    if set(manifest_overrides) - set(manifests):
        raise ValueError('manifest override has no recorded provenance')
    manifests.update(manifest_overrides)
    manifests = {kind:path.resolve(strict=True) for kind, path in manifests.items()}
    recorded_hashes = workload.get('manifest_sha256', {})
    for kind, path in manifests.items():
        if not recorded_hashes.get(kind):
            raise ValueError(f'missing {kind} manifest provenance hash')
        if file_sha256(path) != recorded_hashes[kind]:
            raise ValueError(f'{kind} manifest provenance changed: {path}')
    return manifests, {
        'source_sha256':workload['source_sha256'],
        'manifest_sha256':{kind:recorded_hashes[kind] for kind in manifests},
        'recorded_source_path':workload['source'],
        'verified_source_path':str(source),
        'recorded_manifest_paths':workload['manifests'],
        'verified_manifest_paths':{kind:str(path) for kind, path in manifests.items()},
        # Historical binaries may intentionally differ between before/after
        # runs, or may no longer be present. Report the runner's recorded hash.
        'binary':configuration['binary'],
        'binary_sha256':configuration.get('binary_sha256'),
        'binary_hash_verification':'recorded by runner; current binary is not rehashed',
        'model_prefetch':configuration.get('model_prefetch', 'unrecorded'),
        'extra_args':configuration['extra_args'],
        'threads':configuration.get('threads'),
        'delay_ms':configuration.get('delay_ms'),
        'bandwidth_mib':configuration.get('bandwidth_mib'),
        'drain_timeout_s':configuration.get('drain_timeout_s'),
    }


def verify_cache_payloads(directory, trials, manifests, seeded_names, manifest_hashes):
    verified = {}
    for trial in sorted(trials):
        for kind, manifest in manifests.items():
            cache = directory/f'cache-{trial}'/kind
            if not cache.is_dir():
                raise ValueError(f'cache directory is missing: {cache}')
            seeded = cache/seeded_names[kind]
            if not seeded.is_file():
                raise ValueError(f'seeded cached manifest is missing: {seeded}')
            if file_sha256(seeded) != manifest_hashes[kind]:
                raise ValueError(f'seeded cached manifest provenance changed: {seeded}')
            for path in sorted(cache.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(cache)
                if relative in (Path(seeded_names[kind]), Path('lasagna-remote.json')):
                    continue
                if not path.resolve().is_relative_to(cache.resolve()):
                    raise ValueError(f'cache payload escapes its cache directory: {path}')
                if relative.parts[0] == '.lasagna-zarr-metadata':
                    relative = Path(*relative.parts[1:])
                original = resolve_source_path(manifest, relative)
                if not original.is_file():
                    raise ValueError(f'cache payload has no source counterpart: {path}')
                if file_sha256(original) != file_sha256(path):
                    raise ValueError(f'cache bytes differ: {path}')
                verified[path] = path.stat().st_size
    return verified


def analyze(directory, manifest_overrides):
    rows = json.loads((directory/'results.json').read_text())
    if not rows or any(row['returncode'] != 0 for row in rows):
        raise ValueError(f'{directory}: empty results or failed trial')
    for row in rows:
        require_drained_requests(row)
    configuration = json.loads((directory/'configuration.json').read_text())
    if any(not configured_readers_match(configuration['extra_args'], row['metrics']) for row in rows):
        raise ValueError(f'{directory}: requested reader configuration was not confirmed by binary')
    expected = {(mode, trial) for mode in configuration['modes']
                for trial in range(1, configuration['trials']+1)}
    observed = [(row['mode'], row['trial']) for row in rows]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError(f'{directory}: incomplete or duplicate trials')
    if configuration.get('save_traces') and any('trace_sha256' not in row for row in rows):
        raise ValueError(f'{directory}: requested exact trace artifact is missing')
    manifests, provenance = verify_provenance(directory, configuration, manifest_overrides)
    seeded_names = {kind:Path(path).name for kind, path in provenance['recorded_manifest_paths'].items()}
    verified_cache = verify_cache_payloads(
        directory, {row['trial'] for row in rows}, manifests, seeded_names,
        provenance['manifest_sha256'])
    compatibility_notes = []
    legacy_rows = [f"{row['mode']}-{row['trial']}" for row in rows if 'requests_drained' not in row]
    if legacy_rows:
        compatibility_notes.append(
            'Legacy rows lack requests_drained: ' + ', '.join(legacy_rows) +
            '; complete cache payloads were verified, but historical request drain cannot be confirmed.')
    if provenance['binary_sha256'] is None:
        compatibility_notes.append('Legacy configuration has no recorded binary SHA-256.')
    if provenance['drain_timeout_s'] is None:
        compatibility_notes.append('Legacy configuration does not record its request-drain timeout.')
    stable_fields = ('segments', 'native_segments', 'lasagna_fallback_segments',
                     'cspline_fallback_segments', 'native_tails', 'lasagna_tails', 'points') \
        if 'native_segments' in rows[0]['metrics'] else (
            'restarts', 'segments', 'lookahead_retries', 'lookahead_retry_recovered', 'err/kvx')
    signatures = {tuple(row['metrics'][key] for key in stable_fields) for row in rows}
    candidates = {row['profile']['candidates'] for row in rows}
    generations = {row['profile']['generations'] for row in rows}
    hashes, message_hashes = set(), set()
    verified_bytes = verified_objects = 0
    details = {}
    for row in rows:
        stem = f"{row['mode']}-{row['trial']}"
        if 'trace_sha256' in row:
            trace_path = directory/f'{stem}.trace.json'
            raw = trace_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['trace_sha256']:
                raise ValueError(f'trace artifact changed: {trace_path}')
            trace = json.loads(raw)
            hashes.add(digest(canonical_trace(trace)))
            if trace.get('format') == 'vc_gui_fiber_reoptimization_exact_v1':
                message_hashes.add(digest(normalized_message(trace['optimization']['message'])))
        request_path = directory/f'{stem}.requests.json'
        requests = json.loads(request_path.read_text()) if request_path.exists() else []
        if not request_path.exists():
            compatibility_notes.append(
                f'{stem}: request log is absent; complete cache payloads were verified independently.')
        details[stem] = dict(collections.Counter(f"{r['method']} {r['status']}" for r in requests))
        for request in requests:
            if request['status'] != 200 or request['method'] != 'GET' or request.get('interrupted'):
                continue
            route = urllib.parse.unquote(urllib.parse.urlsplit(request['path']).path).strip('/')
            kind, relative = route.split('/', 1)
            if kind not in manifests:
                raise ValueError(f'missing {kind} manifest for byte verification')
            original = resolve_source_path(manifests[kind], relative)
            cache = directory/f"cache-{row['trial']}"/kind
            cached = cache/relative
            if original.name in ('.zarray', '.zattrs', '.zgroup', 'zarr.json'):
                cached = cache/'.lasagna-zarr-metadata'/relative
            if cached not in verified_cache:
                raise ValueError(f'logged cache payload is missing or was not verified: {cached}')
            if request['bytes'] != verified_cache[cached]:
                raise ValueError(f'logged byte count differs from complete cache payload: {cached}')
            verified_bytes += request['bytes']
            verified_objects += 1
    if any(len(values) != 1 for values in (signatures, candidates, generations)):
        raise ValueError(f'{directory}: numeric decisions/counts changed between trials')
    if len(hashes) > 1:
        raise ValueError(f'{directory}: exact returned geometry/metadata/decisions changed')
    if len(message_hashes) > 1:
        raise ValueError(f'{directory}: diagnostic text changed beyond recognized timings')
    stages = ('prediction_batch_s', 'prediction_materialize_s', 'normal_batch_s',
              'normal_materialize_s', 'candidate_score_s', 'frontier_s', 'prune_s', 'start_sample_s')
    summary = {
        'provenance':provenance,
        'compatibility_notes':compatibility_notes,
        'identical_result_fields':dict(zip(stable_fields, next(iter(signatures)))),
        'identical_candidate_count':next(iter(candidates)),
        'identical_generation_count':next(iter(generations)),
        'exact_trace_sha256':next(iter(hashes)) if hashes else None,
        'normalized_message_sha256':next(iter(message_hashes)) if message_hashes else None,
        'exact_trace_exclusions':['optimization.message: separately checked after masking known timing fields'],
        'byte_verified_get_objects':verified_objects, 'byte_verified_get_bytes':verified_bytes,
        'byte_verified_cache_objects':len(verified_cache),
        'byte_verified_cache_bytes':sum(verified_cache.values()),
        'byte_verified_seeded_manifests':len({row['trial'] for row in rows}) * len(manifests),
        'request_methods_statuses':details,
        'stage_means_s':{}}
    for mode in ('local', 'cold', 'warm'):
        relevant = [row for row in rows if row['mode'] == mode]
        if relevant:
            summary['stage_means_s'][mode] = {
                stage:statistics.mean(row['profile'][stage] for row in relevant) for stage in stages}
    (directory/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def validate_cross_variants(summaries):
    if any('provenance' not in summary for summary in summaries.values()):
        raise ValueError('workload provenance is missing from a variant')
    workloads = {digest({field:summary['provenance'][field]
                         for field in ('source_sha256', 'manifest_sha256')})
                 for summary in summaries.values()}
    if len(workloads) > 1:
        raise ValueError('fiber source or manifest provenance differs across variants')
    # Binary hashes and runtime prefetch flags intentionally are not equality
    # requirements. Each variant's provenance reports them for before/after use.
    fields = ('identical_result_fields', 'identical_candidate_count', 'identical_generation_count',
              'exact_trace_sha256', 'normalized_message_sha256')
    signatures = {digest({field:summary[field] for field in fields}) for summary in summaries.values()}
    if len(signatures) > 1:
        raise ValueError('result metrics, exact trace, or timing-normalized diagnostic text differs across variants')
    conditions = {}
    for field in ('threads', 'delay_ms', 'bandwidth_mib', 'drain_timeout_s', 'extra_args'):
        values = {label:summary['provenance'].get(field) for label, summary in summaries.items()}
        if len({digest(value) for value in values.values()}) > 1:
            conditions[field] = values
    if conditions:
        warnings.warn(
            'Performance conditions differ; timings are not like-for-like: ' +
            json.dumps(conditions, sort_keys=True), RuntimeWarning, stacklevel=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--fiber-manifest', type=Path)
    parser.add_argument('--normal-manifest', type=Path)
    args = parser.parse_args()
    manifests = {kind:path.resolve(strict=True) for kind, path in (
        ('fiber', args.fiber_manifest), ('normal', args.normal_manifest)) if path is not None}
    summaries = {str(directory):analyze(directory, manifests) for directory in args.run_dirs}
    validate_cross_variants(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
