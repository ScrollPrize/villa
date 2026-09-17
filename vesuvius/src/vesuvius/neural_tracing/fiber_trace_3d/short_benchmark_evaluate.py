"""Evaluate a locked shortened benchmark with one historical model/cache."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from lasagna.inference_provenance import atomic_write, sha256_file


def bind_benchmark(config, manifest, *, backend=None, fiberlet=None, render=False):
    if not isinstance(render, bool):
        raise ValueError('render must be a boolean')
    if render and backend == 'anchors':
        raise ValueError('anchor-only benchmark cannot render traces')
    if not manifest.get('completed') or len(manifest['cases']) != manifest['case_count']:
        raise ValueError('benchmark manifest is incomplete')
    evaluation = deepcopy(config)
    sources = {s['id']: s for s in evaluation['sources']}
    for source in sources.values():
        source['fibers'] = []
    seen = set()
    for case in manifest['cases']:
        path = str(Path(case['path']).resolve())
        if path in seen or sha256_file(path) != case['sha256']:
            raise ValueError('duplicate or changed benchmark annotation')
        seen.add(path)
        if json.loads(Path(path).read_text()).get('evaluation_direction') != 'forward':
            raise ValueError('benchmark must contain directed forward annotations')
        sources[case['candidate']['source_id']]['fibers'].append(path)
    evaluation['sources'] = [s for s in sources.values() if s['fibers']]
    for key in ('live_endpoint', 'reference_plan', 'annotation_sha256'):
        evaluation.pop(key, None)
    evaluation['render'] = render
    if render:
        for source in evaluation['sources']:
            if 'ct_volume' not in source:
                source['ct_volume'] = source['input_volume']
                source['ct_base_volume'] = True
    if backend is not None:
        if backend not in ('fiberlet', 'beam', 'anchors'):
            raise ValueError('unsupported benchmark backend')
        if backend == 'beam':
            evaluation.pop('backend', None)
            evaluation.pop('fiberlet', None)
        else:
            for source in evaluation['sources']:
                if not source.get('fiberlet_source_identity') or not source.get('fiberlet_cache_root'):
                    raise ValueError('fiberlet replay requires canonical source identity/cache root')
            evaluation['backend'] = backend
            evaluation['fiberlet'] = deepcopy(fiberlet or {})
    return evaluation


def scroll_groups(evaluation):
    """Group physical volumes; retain distinct collections sharing one cache."""
    groups = {}
    for source in evaluation['sources']:
        key = (source['coordinate_space'], str(Path(source['fiber_manifest']).resolve()))
        groups.setdefault(key, []).append(source)
    return list(groups.values())


def write_render_review_index(root, result, manifest):
    """Index directed pieces without assuming paired forward/reverse traces."""
    from urllib.parse import quote
    import shutil
    from datetime import datetime, timezone
    from .replay_metrics import reference_segments, segment_statistics
    from vc3d_fiber_format import parse_vc3d_fiber_format
    root = Path(root)
    stamp_file = root / 'review_export_identity.json'
    if stamp_file.exists():
        stamp = json.loads(stamp_file.read_text())['started_at']
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]
        atomic_write(stamp_file, dict(started_at=stamp))
    expected = {str(Path(c['path']).resolve()): c for c in manifest['cases']}
    rows, seen = [], set()
    for directory in result.get('scroll_artifacts', [str(root / 'full')]):
        for artifact in sorted((Path(directory) / 'cases').glob('*/trace.json')):
            document = json.loads(artifact.read_text())
            p, replay = document['provenance'], document['replay']
            path = str(Path(p['fiber_path']).resolve())
            if path not in expected or path in seen or p['reverse']:
                raise ValueError('unexpected or duplicate rendered benchmark piece')
            seen.add(path)
            begin, end = replay['reference_begin_arc_base'], replay['reference_end_arc_base']
            values = reference_segments(dict(reference_length_base_voxels=end-begin,
                evaluated_through_base_voxels=replay['completed_reference_arc_base']-begin,
                evaluation_complete=replay['completed_reference_arc_base'] >= end-1e-7,
                failures=[dict(directional_reference_arc_base_voxels=f['reference_arc_base']-begin)
                          for f in replay['failures']]), p['base_voxel_size_um']/1000)
            stats = segment_statistics(values, 1)
            metadata = artifact.parent / 'rendered/render.json'
            render = json.loads(metadata.read_text())
            pages = [metadata.parent / f'strips_{i}.jpg' for i in range(render['pages'])]
            if not pages or not all(page.is_file() for page in pages):
                raise ValueError('missing rendered benchmark pages')
            original = expected[path]
            copies = {}
            source_path = original['candidate'].get('fiber')
            for source in [source_path]:
                if not source:
                    continue
                source = Path(source)
                name = source.name
                target = metadata.parent / name
                if target.exists() and sha256_file(target) != sha256_file(source):
                    raise ValueError('refusing to overwrite modified review annotation')
                if not target.exists():
                    shutil.copy2(source, target)
                if sha256_file(target) != sha256_file(source):
                    raise ValueError('review annotation copy differs from source')
                copies[name] = str(target.relative_to(root))
            annotation = json.loads(Path(path).read_text())
            if original.get('sha256') and sha256_file(path) != original['sha256']:
                raise ValueError('review source annotation changed')
            sequence = int(Path(path).stem.split('_')[0])
            identity = dict(original_fiber=source_path, benchmark_piece=path,
                benchmark_piece_id=Path(path).stem, case_index=p['case_index'],
                source_id=p['source_id'], coordinate_space=p.get('coordinate_space'),
                evaluation_root=str(root), trace_artifact=str(artifact),
                fiber_manifest=p.get('fiber_manifest'), reference_run=p.get('reference_run'),
                direction='forward')

            def save_fiber(author, sequence_number, fiber):
                parse_vc3d_fiber_format(fiber)
                name = f'{author}_{stamp}_{sequence_number:06d}.json'
                target = metadata.parent / name
                if target.exists() and json.loads(target.read_text()) != fiber:
                    raise ValueError('refusing to overwrite modified review annotation or trace')
                if not target.exists():
                    atomic_write(target, fiber)
                copies[name] = str(target.relative_to(root))

            extracted = dict(annotation, review_provenance=dict(identity, author='evalextract', role='reference'))
            save_fiber('evalextract', sequence, extracted)
            for i, segment in enumerate(replay.get('segments', [])):
                points = segment['trace_points_base_xyz']
                if len(points) < 2:
                    continue  # No line geometry for an unseeded/empty segment.
                fiber = dict(type='vc3d_fiber', version=1,
                    optimization_mode='native_fiber_trace3d', line_points=points,
                    control_points=[points[0], points[-1]],
                    evaluation_trace_provenance=dict(trace=str(artifact), segment_index=i,
                        reference_begin_arc_base=segment['start_reference_arc_base'],
                        reference_end_arc_base=segment['end_reference_arc_base'],
                        termination_reason=segment['termination_reason']))
                author = 'evaltrace-' + ('fiberlet' if replay.get('geometry_backend') == 'fiberlet' else 'beam')
                fiber['review_provenance'] = dict(identity, author=author, role='trace', segment_index=i)
                for key in ('vc_open_data_coordinate_space', 'vc_open_data_source_original_resolution'):
                    if key in annotation:
                        fiber[key] = annotation[key]
                if i >= 1000:
                    raise ValueError('too many trace segments for review naming')
                save_fiber(author, sequence * 1000 + i, fiber)
            # Preserve earlier generic exports, including any user edits, outside
            # the visible fiber list. Never delete or overwrite them.
            old_names = [metadata.parent / n for n in ('reference_fiber.json', 'original_fiber.json')]
            old_names += list(metadata.parent.glob('trace_segment_*.json'))
            for old in old_names:
                if old.exists():
                    archive = metadata.parent / '.legacy-export-names'
                    archive.mkdir(exist_ok=True)
                    target = archive / old.name
                    if target.exists():
                        raise ValueError('legacy export archive already exists')
                    old.rename(target)
            mm = math.fsum(values)
            rows.append(dict(piece=Path(path).stem, collection=p['source_id'],
                source_fiber=original['candidate'].get('fiber'), annotation=path,
                case_index=p['case_index'], length_mm=mm, restarts=stats['failures'],
                historical_restarts=original.get('historical_restarts'),
                failures_per_mm=stats['failures']/mm if mm else None,
                mean_mm=stats['mean_segment_length_mm'],
                trace=str(artifact.relative_to(root)), metadata=str(metadata.relative_to(root)),
                fiber_copies=copies,
                images=[str(page.relative_to(root)) for page in pages]))
    if seen != set(expected):
        raise ValueError('missing rendered benchmark pieces')
    rows.sort(key=lambda r: (-(r['failures_per_mm'] or 0), r['piece']))
    atomic_write(root / 'review_index.json', dict(pieces=rows, selection=manifest.get('selection')))
    lines = ['# Failure review — no exclusions applied', '',
        'Regular evaluation rendering. Sorted by reported restarts/mm; original piece IDs retained.', '',
        '| Piece | Collection | Length mm | Restarts (old → rerun) | Restarts/mm | Mean mm | Images / metadata | Source fiber |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        links = [f'[image {i+1}]({quote(p)})' for i, p in enumerate(r['images'])]
        links += [f'[metadata]({quote(r["metadata"])})', f'[trace]({quote(r["trace"])})']
        links += [f'[{name}]({quote(path)})' for name, path in r['fiber_copies'].items()]
        old = r['historical_restarts'] if r['historical_restarts'] is not None else '—'
        source = f'[{Path(r["source_fiber"]).name}]({quote(r["source_fiber"])})' if r['source_fiber'] else '—'
        rate = f'{r["failures_per_mm"]:.4f}' if r['failures_per_mm'] is not None else '—'
        lines.append(f'| {r["piece"]} | {r["collection"]} | {r["length_mm"]:.4f} | '
                     f'{old} → {r["restarts"]} | {rate} | {r["mean_mm"]:.4f} | {" · ".join(links)} | {source} |')
    (root / 'README.md').write_text('\n'.join(lines)+'\n')
    return rows


def corrected_annotation(original, corrected, provenance):
    """Keep edited geometry/CP semantics; restore only benchmark identity."""
    result = deepcopy(corrected)
    for key in ('evaluation_direction', 'vc_open_data_coordinate_space',
                'vc_open_data_source_original_resolution', 'benchmark_provenance'):
        if key in original:
            result[key] = deepcopy(original[key])
    result['benchmark_correction'] = provenance
    return result


def prepare_corrections(workload):
    """Freeze a new manifest without altering the locked historical benchmark."""
    import subprocess
    manifest = json.loads(Path(workload['benchmark_manifest']).read_text())
    destination = Path(workload['output_root'] + '_inputs')
    destination.mkdir(parents=True, exist_ok=False)
    for correction in workload['benchmark_corrections']:
        case, = [c for c in manifest['cases'] if Path(c['path']).stem == correction['piece']]
        old, source = Path(case['path']), Path(correction['path'])
        if sha256_file(old) != case['sha256'] or sha256_file(source) != correction['sha256']:
            raise ValueError('benchmark correction input changed')
        provenance = dict(original_path=str(old), original_sha256=case['sha256'],
                          corrected_path=str(source), corrected_sha256=correction['sha256'])
        document = corrected_annotation(json.loads(old.read_text()), json.loads(source.read_text()), provenance)
        target = destination / old.name
        atomic_write(target, document)
        case.update(path=str(target), sha256=sha256_file(target), correction=provenance)
    previous = Path(workload['cached_evaluation'])
    config = json.loads((previous / 'summary.json').read_text())['config']
    config = bind_benchmark(config, manifest)
    config_path = destination / 'reference_config.json'
    atomic_write(config_path, config)
    plan = json.loads(subprocess.check_output([workload['native_binary'], 'replay-references',
                                             '--config', str(config_path)], text=True))
    atomic_write(destination / 'reference_plan.json', plan)
    fibers = {str(Path(f['fiber_path']).resolve()): f for s in plan['sources'] for f in s['fibers']}
    for case in manifest['cases']:
        fiber = fibers[str(Path(case['path']).resolve())]
        if len(fiber['runs']) != 1:
            raise ValueError('mini benchmark correction must retain one eligible piece')
        points = fiber['runs'][0]['points_base_xyz']
        length = math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))
        voxel_mm = case['length_mm'] / case['length_base_voxels']
        case.update(length_base_voxels=length, length_mm=length * voxel_mm)
    manifest.update(total_mm=math.fsum(c['length_mm'] for c in manifest['cases']),
                    total_base_voxels=math.fsum(c['length_base_voxels'] for c in manifest['cases']),
                    supersedes=workload['benchmark_manifest'])
    target = destination / 'manifest.json'
    atomic_write(target, manifest)
    return dict(workload, benchmark_manifest=str(target), benchmark_corrections=[])


def run(workload, devices, *, gpu_manager=None, cancel_event=None):
    if workload.get('benchmark_corrections'):
        workload = prepare_corrections(workload)
    if workload.get('anchor_sweep'):
        from .anchor_fit_sweep import run_sweep
        return run_sweep(workload, devices, gpu_manager)
    if workload.get('fiberlet_sweep'):
        from .fiberlet_parameter_sweep import run_sweep
        return run_sweep(workload, devices, gpu_manager)
    if workload.get('fiberlet_path_sweep'):
        from .fiberlet_parameter_sweep import run_path_sweep
        return run_path_sweep(workload, devices, gpu_manager)
    from .cached_trace_profile import extend_cached_replay, CachedReplaySession
    from .replay_metrics import complete_report, correct_summary
    previous = Path(workload['cached_evaluation'])
    summary = json.loads((previous / 'summary.json').read_text())
    if not summary['completed']:
        raise ValueError('historical evaluation incomplete')
    checkpoint = Path(workload['checkpoint'])
    if sha256_file(checkpoint) != workload['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    bindings = json.loads((previous.parent / 'full_bindings.json').read_text())
    if bindings['checkpoint_sha256'] != workload['checkpoint_sha256']:
        raise ValueError('historical checkpoint mismatch')
    manifest_path = Path(workload['benchmark_manifest'])
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if workload.get('backend') in ('fiberlet', 'anchors'):
        from .live_evaluate import bind_fiberlet_identity
        for source in summary['config']['sources']:
            persisted = json.loads((Path(source['fiber_manifest']).parent / 'cache_identity.json').read_text())
            identity = persisted['identity']
            if identity['model']['checkpoint_sha256'] != workload['checkpoint_sha256']:
                raise ValueError('fiberlet cached model mismatch')
            bind_fiberlet_identity(source, identity['source'], identity['model'], source['fiber_manifest'])
    evaluation = bind_benchmark(summary['config'], manifest,
        backend=workload.get('backend'), fiberlet=workload.get('fiberlet'),
        render=workload.get('render', False))
    for source in evaluation['sources']:
        recorded = next(s for s in bindings['sources'] if s['collection'] == source['id'])
        if sha256_file(source['normal_manifest']) != recorded['normal_manifest_sha256']:
            raise ValueError('normal manifest changed')
    root = Path(workload['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    evaluation.update(output=str(root / 'full'), workers=workload.get('benchmark_workers', 128))
    evaluation['log_construction_chunks'] = workload.get('log_construction_chunks', True)
    atomic_write(root / 'evaluation.json', evaluation)
    atomic_write(root / 'provenance.json', dict(workload=workload,
        benchmark_sha256=manifest_hash, native_sha256=sha256_file(workload['native_binary'])))
    start = time.monotonic()
    if workload.get('scroll_sequential', False):
        parts, artifacts = [], []
        with CachedReplaySession(evaluation, workload['native_binary'], checkpoint,
                                devices=devices, gpu_manager=gpu_manager,
                                cancel_event=cancel_event) as session:
            for index, sources in enumerate(scroll_groups(evaluation)):
                destination = root / 'scrolls' / f'{index:02d}'
                current = dict(evaluation, sources=sources, output=str(destination))
                print(f'benchmark scroll-start {index+1}: {[s["id"] for s in sources]}', flush=True)
                session.run(current)
                part = json.loads((destination / 'summary.json').read_text())
                if not part['completed']:
                    raise ValueError('incomplete scroll evaluation')
                parts.append(part)
                artifacts.append(str(destination))
                print(f'benchmark scroll-completed {index+1}: cases={part["cases"]}', flush=True)
        anchor_only = workload.get('backend') == 'anchors'
        result = dict(completed=True, config=evaluation,
            metric_definition=parts[0].get('metric_definition', 'anchor-nearest-projection-flat-caps-v1'),
            reference_selection_policy=parts[0]['reference_selection_policy'],
            cases=sum(p['cases'] for p in parts), failures=sum(p.get('failures', 0) for p in parts),
            directed_reference_length_mm=sum(p['directed_reference_length_mm'] for p in parts),
            sources=[s for p in parts for s in p.get('sources', [])], scroll_artifacts=artifacts)
        if anchor_only:
            result['backend'] = 'anchors'
            result['anchor_cases'] = [c for p in parts for c in p['anchor_cases']]
        else:
            result = correct_summary(result)
        (root / 'full').mkdir()
        atomic_write(root / 'full/summary.json', result)
    else:
        extend_cached_replay(evaluation, workload['native_binary'], checkpoint,
                             devices=devices, gpu_manager=gpu_manager,
                             cancel_event=cancel_event)
    elapsed = time.monotonic() - start
    result = json.loads((root / 'full/summary.json').read_text())
    if result['cases'] != manifest['case_count'] or sha256_file(manifest_path) != manifest_hash:
        raise ValueError('benchmark coverage or identity changed')
    if workload.get('backend') == 'anchors':
        from .anchor_coverage_report import coverage_report, markdown_report
        report = coverage_report(result['anchor_cases'])
        if not math.isclose(report['overall']['length_mm'], manifest['total_mm'], abs_tol=0.01):
            raise ValueError('anchor benchmark length differs from exported length')
        report.update(wall_seconds=elapsed, benchmark_manifest=str(manifest_path),
                      checkpoint_sha256=workload['checkpoint_sha256'])
        atomic_write(root / 'metrics.json', report)
        ranking = []
        for directory in result.get('scroll_artifacts', [str(root / 'full')]):
            for path in sorted((Path(directory) / 'cases').glob('*/anchors.json')):
                c = json.loads(path.read_text())
                gaps = list(c['interior_gaps_base'])
                gaps.extend(c[k] for k in ('leading_gap_base', 'trailing_gap_base') if c[k] is not None)
                worst = max(gaps, default=c['reference_length_base']) / c['cell_side_base']
                ranking.append(dict(source=c['source_id'], fiber_path=c['fiber_path'],
                    case=c['case_index'], anchors_json=str(path),
                    max_gap_including_ends_cells=worst, accepted_anchors=len(c['accepted_anchors']),
                    empty_arc_bins=c['empty_arc_bins'], arc_bins=c['arc_bins']))
        ranking.sort(key=lambda c: (-c['max_gap_including_ends_cells'], c['source'], c['fiber_path']))
        atomic_write(root / 'piece_ranking.json', ranking)
        (root / 'report.md').write_text(markdown_report(report))
        if not workload.get('quiet_report'):
            print(json.dumps(report, indent=2), flush=True)
        return report
    report = complete_report(result)
    report['coverage'] = {s['source_id']: {k: v for k, v in s['summary'].items()
        if k in ('traced_length_mm', 'length_weighted_success_percent', 'failure_free_cases',
                 'directed_cases', 'failure_reasons')} for s in result['sources']}
    if not math.isclose(report['global_result']['directed_reference_mm'], manifest['total_mm'], abs_tol=0.01):
        raise ValueError('benchmark evaluated length differs from exported length')
    report['wall_seconds'] = elapsed
    report['benchmark_manifest'] = str(manifest_path)
    report['checkpoint_sha256'] = workload['checkpoint_sha256']
    atomic_write(root / 'metrics.json', report)
    if evaluation['render']:
        write_render_review_index(root, result, manifest)
    if workload.get('pair_reference'):
        from .fiber_pair_benchmark import write_pair_benchmark
        write_pair_benchmark(root, evaluation, manifest,
            dict(workload['pair_reference'], checkpoint_sha256=workload['checkpoint_sha256']),
            workload['native_binary'])
    if not workload.get('quiet_report'):
        print(json.dumps(report, indent=2), flush=True)
    return report
