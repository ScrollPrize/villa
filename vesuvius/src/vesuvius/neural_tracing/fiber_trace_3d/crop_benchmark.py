"""Recorded CPU crop benchmarks through the approved evaluation runner."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from lasagna.inference_provenance import atomic_write, sha256_file


def restore_historical_metrics(output):
    """Migrate only this runner's old replay summaries, preserving case evidence."""
    for path in sorted(output.glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('metric_definition') != 'complete-reference-partition-v1':
            continue
        summary = data['summary']
        length = summary['directed_reference_length_base_voxels']
        count = summary['total_failures'] + 1 if length > 0 else 0
        summary['reliability_segments'] = count
        summary['mean_segment_length_base_voxels'] = length / count if count else 0
        summary['mean_segment_length_mm'] = summary['mean_segment_length_base_voxels'] * summary['base_voxel_size_um'] / 1000
        summary['mean_segment_length_percent'] = 100 / count if count else 0
        data['version'] = 6
        data['metric_definition'] = 'corpus-failures-plus-one-v1'
        atomic_write(path, data, durable=False)


def bp_metrics(log):
    """Read the existing native benchmark tables, never rounded percentages."""
    constraints = re.search(r'constraint pruning benchmark\n[^\n]*\n([^\n]+)', log)
    references = re.search(r'reference oracle rounds\n[^\n]*\n0\s+([^\n]+)', log)
    if constraints is None or references is None:
        raise ValueError('BP log lacks completed constraint and round-zero tables')
    counts = constraints[1].split()
    # initial, removed-incident, retained-infringed, retained-defect,
    # problematic, retained-fulfilled, then printed percentages.
    problematic, fulfilled = int(counts[4]), int(counts[5])
    fields = references[1].split()
    # pieces, removed, all-exact, all-wrong, all-missing, ...
    exact, wrong, missing = map(int, fields[2:5])
    if min(problematic, exact, wrong, missing) < 0 or fulfilled <= 0 or exact + wrong + missing <= 0:
        raise ValueError('BP table has invalid counts')
    return dict(problematic_unique_constraints=problematic,
                retained_fulfilled_unique_constraints=fulfilled,
                exact_references=exact, wrong_references=wrong, missing_references=missing,
                constraint_score_percent=-100 * problematic / fulfilled,
                reference_score_percent=100 * exact / (exact + wrong + missing))


def run(config):
    if 'fiber_gt_constraint_score' in config:
        from .fiber_gt_constraint_score import run as score_gt
        return score_gt(config['fiber_gt_constraint_score'])
    if 'fiber_json_solve' in config:
        from .fiber_json_solve import run as json_solve
        return json_solve(config['fiber_json_solve'])
    if 'fiber_uv_reference_subset' in config:
        from .fiber_uv_reference_subset import run as reference_subset
        return reference_subset(config['fiber_uv_reference_subset'])
    if 'fiber_reference_comparison' in config:
        from .fiber_reference_comparison import run as reference_comparison
        return reference_comparison(config['fiber_reference_comparison'])
    if 'fiber_reference_surface' in config:
        from .fiber_reference_surface import run as reference_surface
        return reference_surface(config['fiber_reference_surface'])
    if 'fiber_uv_density' in config:
        from .fiber_uv_density import run as density_maps
        return density_maps(config['fiber_uv_density'])
    if 'fiber_uv_hv_rejection' in config:
        from .fiber_uv_hv_rejection import run as hv_rejection
        return hv_rejection(config['fiber_uv_hv_rejection'])
    if 'fiber_uv_kollesis' in config:
        from .fiber_uv_kollesis import run as kollesis_maps
        return kollesis_maps(config['fiber_uv_kollesis'])
    if 'fiber_uv_surface_audit' in config:
        from .fiber_uv_surface import audit as surface_audit
        return surface_audit(config['fiber_uv_surface_audit'])
    if 'fiber_uv_surface' in config:
        from .fiber_uv_surface import run as surface_uv
        return surface_uv(config['fiber_uv_surface'])
    if 'fiber_hv_report' in config:
        from .fiber_hv_stage import report
        return report(config['fiber_hv_report'])
    if 'fiber_axial_features' in config:
        from .fiber_hv_stage import run_features
        return run_features(config['fiber_axial_features'])
    if 'fiber_direction' in config or 'fiber_h_direction' in config:
        from .fiber_direction import run as run_direction
        return run_direction(config.get('fiber_direction',config.get('fiber_h_direction')))
    if 'fiber_h_direction_report' in config:
        from .fiber_direction import report as report_direction
        return report_direction(config['fiber_h_direction_report'])
    if 'fiber_direction_review' in config or 'fiber_h_direction_review' in config:
        from .fiber_direction_review import run as review_direction
        return review_direction(config.get('fiber_direction_review',config.get('fiber_h_direction_review')))
    if 'fiber_uv_layout' in config:
        from .fiber_uv_layout import run as run_uv
        return run_uv(config['fiber_uv_layout'])
    if 'fiber_uv_report' in config:
        from .fiber_uv_layout import report as report_uv
        return report_uv(config['fiber_uv_report'])
    if 'fiber_uv_visualization' in config:
        from .fiber_uv_visualization import run as visualize_uv
        return visualize_uv(config['fiber_uv_visualization'])
    if 'fiber_hv_stage' in config:
        from .fiber_hv_stage import run as run_hv
        return run_hv(config['fiber_hv_stage'])
    if 'fiber_normal_arrow_update' in config:
        from .fiber_normal_sign_visualization import update_arrows
        return update_arrows(config['fiber_normal_arrow_update'])
    if 'fiber_normal_obj_repair' in config:
        from .fiber_normal_sign_visualization import repair_obj_lines
        return repair_obj_lines(config['fiber_normal_obj_repair'])
    if 'fiber_normal_visualization' in config:
        from .fiber_normal_sign_visualization import export
        return export(config['fiber_normal_visualization'])
    if 'fiber_normal_priors' in config:
        from .fiber_normal_sign import run_priors
        return run_priors(config['fiber_normal_priors'])
    if 'fiber_normal_sign' in config:
        from .fiber_normal_sign import run as run_sign
        return run_sign(config['fiber_normal_sign'])
    if 'neighborhood_readout_report' in config:
        from .neighborhood_readout_report import report
        return report(config)
    if 'local_cost_sweep' in config:
        from .local_cost_sweep import run_sweep
        return run_sweep(config)
    if 'local_event_report' in config:
        from .local_constraint_report import report
        return report(config)
    if 'density_alternatives' in config:
        from .density_alternatives import run_alternatives
        return run_alternatives(config)
    if 'density_sweep' in config:
        from .density_penalty_sweep import run_sweep
        return run_sweep(config)
    repo = Path(__file__).resolve().parents[5]
    binaries = repo / 'volume-cartographer/build/fiber-evaluation/bin'
    output = Path(config['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    if config.get('restore_historical_metrics'):
        restore_historical_metrics(output)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    inputs = {str(p): sha256_file(p) for p in config.get('input_files', [])}
    references = {str(p): sha256_file(p) for p in sorted(Path(config['reference_dir']).glob('*.json'))}
    for case in config['commands']:
        binary = binaries / case['binary']
        if case['binary'] not in {'vc_fiber_trace_chunk', 'vc_fiberlets',
                                  'test_fiber_reference_replay_benchmark', 'test_fiberlet_storage',
                                  'test_fiberlet_crop_trace', 'test_fiber_trace3d', 'test_fiber_replay',
                                  'test_fiber_trace_winding_bp'}:
            raise ValueError('unsupported crop executable')
        name = case['name']
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('case name must be a simple filename')
        existing = [output / f'{name}.{suffix}' for suffix in
                    ('log', 'resources.txt', 'run.json', 'source.patch', 'json')]
        if any(path.exists() for path in existing):
            archive = output / 'attempts' / (name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            archive.mkdir(parents=True)
            for path in existing:
                if path.exists():
                    path.rename(archive / path.name)
        command = [str(binary), *case['args']]
        record = dict(revision=revision, build_type='RelWithDebInfo', command=command,
                      binary_sha256=sha256_file(binary), inputs=inputs, references=references,
                      repetitions=1, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        record['shared_libraries'] = {
            str(p): sha256_file(p)
            for p in sorted((binaries.parent / 'core').glob('libvc_fiber_tracer.*'))
            if p.is_file()
        }
        (output / f'{name}.source.patch').write_bytes(subprocess.check_output(
            ['git', 'diff', 'HEAD', '--', 'volume-cartographer'], cwd=repo))
        print(f'crop benchmark start: {name}', flush=True)
        start = time.monotonic()
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(['/usr/bin/time', '-f', 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M',
                                     '-o', str(output / f'{name}.resources.txt'), *command],
                                    stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_seconds=time.monotonic() - start)
        atomic_write(output / f'{name}.run.json', record, durable=False)
        print(json.dumps({'case': name, 'returncode': result.returncode,
                          'wall_seconds': record['wall_seconds'],
                          'record': str(output / f'{name}.run.json')}), flush=True)
        result.check_returncode()
    if config.get('publish_replay_plots') or config.get('publish_bp_plots'):
        plot_path = repo / 'volume-cartographer/docs/fiber_benchmark_plot_data.json'
        document = json.loads(plot_path.read_text())
        points = document['benchmarks']['reference_replay']['points']
        for entry in config.get('publish_replay_plots', []):
            source = output / entry['result']
            result = json.loads(source.read_text())
            summary = result['summary']
            points[:] = [p for p in points if p.get('measurement_id') != entry['id']]
            points.append(dict(
                measurement_id=entry['id'], algorithm=entry['label'],
                method_id=entry['id'], method_label=entry['label'],
                algorithm_date=config.get('measurement_date', '2026-09-23'), algorithm_revision=revision,
                measurement_status='measured', measurement_date=config.get('measurement_date', '2026-09-23'),
                measurement_revision=revision,
                tested_length_mm=summary['directed_reference_length_base_voxels'] * summary['base_voxel_size_um'] / 1000,
                failures=summary['total_failures'],
                run_record=entry.get('run_record', 'docs/fiber_benchmark_runs/2026-09-23-crop-new-predictions.md'),
                input_artifact=str(source), input_sha256=sha256_file(source),
                normal_manifest=result['inputs']['normal_manifest']))
        for entry in config.get('publish_bp_plots', []):
            source = output / entry['log']
            metrics = bp_metrics(source.read_text())
            atomic_write(source.with_suffix('.metrics.json'), metrics, durable=False)
            common = dict(measurement_id=entry['id'], algorithm=entry['label'],
                          method_id=entry['id'], method_label=entry['label'],
                          algorithm_date=config.get('measurement_date', '2026-09-23'), algorithm_revision=revision,
                          measurement_status='measured', measurement_date=config.get('measurement_date', '2026-09-23'),
                          measurement_revision=revision, input_artifact=str(source),
                          input_sha256=sha256_file(source),
                          run_record=entry.get('run_record', 'docs/fiber_benchmark_runs/2026-09-23-dense-beam-bp.md'))
            for metric, keys in (
                ('crop_pruning', ('problematic_unique_constraints', 'retained_fulfilled_unique_constraints')),
                ('pre_pruning_reference', ('exact_references', 'wrong_references', 'missing_references'))):
                points = document['benchmarks'][metric]['points']
                points[:] = [p for p in points if p.get('measurement_id') != entry['id']]
                points.append(dict(common, **{key: metrics[key] for key in keys}))
        atomic_write(plot_path, document, durable=False)
        subprocess.run([sys.executable, str(repo / 'volume-cartographer/scripts/plot_fiber_benchmarks.py')],
                       cwd=repo, check=True)
