"""Package explicit trace choices and curate mutable short benchmarks."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from lasagna.inference_provenance import atomic_write, sha256_file


def _copy_verified(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise ValueError(f'copied artifact differs from source: {source}')
    return {'path': str(destination), 'sha256': source_hash,
            'source_path': str(source.resolve())}


def package_contrast_choice(config: dict) -> Path:
    """Freeze one rendered evaluation case plus an explicit competing trace."""
    root = Path(config['output_root'])
    if root.exists():
        raise FileExistsError(root)
    case_root = Path(config['source_case'])
    source_manifest_path = Path(config['source_benchmark_manifest'])
    source_manifest = json.loads(source_manifest_path.read_text())
    piece = config['piece']
    source_case, = [c for c in source_manifest['cases']
                    if Path(c['path']).stem == piece]
    if sha256_file(source_case['path']) != source_case['sha256']:
        raise ValueError('source benchmark annotation changed')
    rendered = case_root / 'rendered'
    reference_source = Path(source_case['path'])
    alternative_source = rendered / config['alternative_trace']
    reference = json.loads(reference_source.read_text())
    alternative = json.loads(alternative_source.read_text())
    for key in ('vc_open_data_coordinate_space',
                'vc_open_data_source_original_resolution'):
        if reference.get(key) != alternative.get(key):
            raise ValueError('reference/alternative coordinate binding mismatch')
    if reference.get('evaluation_direction') != 'forward' or \
            len(alternative.get('line_points', [])) < 2:
        raise ValueError('contrast fibers are not usable directed lines')

    root.mkdir(parents=True)
    copied = {}
    copied['reference'] = _copy_verified(
        reference_source, root / 'fibers' / f'{piece}.json')
    copied['alternative'] = _copy_verified(
        alternative_source, root / 'choices' / 'competing_beam_trace.json')
    evidence_sources = [case_root / 'trace.json', case_root / 'reference.obj',
                        case_root / 'trace.obj']
    evidence_sources.extend(sorted(rendered.iterdir()))
    evidence = []
    for source in evidence_sources:
        if source.is_file():
            target = root / 'evidence' / ('rendered' if source.parent == rendered else '') / source.name
            evidence.append(_copy_verified(source, target))

    case = deepcopy(source_case)
    case['path'] = copied['reference']['path']
    case['sha256'] = copied['reference']['sha256']
    case['contrast_choices'] = [{
        'path': copied['alternative']['path'],
        'sha256': copied['alternative']['sha256'],
        'role': 'competing-pre-restart-beam-trace',
        'source_segment_index': alternative.get('evaluation_trace_provenance', {}).get('segment_index'),
    }]
    case['contrast_provenance'] = {
        'source_evaluation_case': str(case_root.resolve()),
        'source_trace_sha256': sha256_file(case_root / 'trace.json'),
        'source_render_sha256': sha256_file(rendered / 'render.json'),
        'source_image_sha256': sha256_file(rendered / 'strips_0.jpg'),
    }
    manifest = {
        'version': 1, 'completed': True, 'case_count': 1,
        'total_base_voxels': case['length_base_voxels'],
        'total_mm': case['length_mm'], 'cases': [case],
        'benchmark_kind': 'explicit-contrast-choice',
        'source_manifest': {'path': str(source_manifest_path.resolve()),
                            'sha256': sha256_file(source_manifest_path)},
        'evidence': evidence,
    }
    atomic_write(root / 'manifest.json', manifest)
    unrestricted = deepcopy(manifest)
    unrestricted['benchmark_kind'] = 'explicit-contrast-choice-unrestricted-control'
    unrestricted['cases'][0].pop('contrast_choices')
    atomic_write(root / 'manifest_unrestricted.json', unrestricted)
    readme = [
        '# Explicit contrast-choice benchmark', '',
        f'- Reference: `{Path(case["path"]).relative_to(root)}`',
        '- Alternative: `choices/competing_beam_trace.json`',
        f'- Source evaluation case: `{case_root.resolve()}`',
        '- Current visualization: `evidence/rendered/strips_0.jpg`',
        '- Render metadata: `evidence/rendered/render.json`',
        '', 'The alternative is the saved pre-restart beam segment and is used only',
        'to define an allowed fiberlet traversal corridor. The reference remains',
        'the sole evaluation ground truth.', '',
    ]
    (root / 'README.md').write_text('\n'.join(readme))
    return root / 'manifest.json'


def remove_review_case(config: dict) -> Path:
    """Archive one active case from a mutable ground-truth review draft."""
    root = Path(config['review_root'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('review', {}).get('status') != 'ground-truth-review-draft' or \
            not manifest.get('review', {}).get('mutable'):
        raise ValueError('target is not a mutable ground-truth review draft')
    rendered_case = Path(config['rendered_case'])
    trace = json.loads((rendered_case / 'trace.json').read_text())
    source_path = str(Path(trace['provenance']['fiber_path']).resolve())
    matches = [(index, case) for index, case in enumerate(manifest['cases'])
               if str(Path(case['source_annotation']['path']).resolve()) == source_path]
    if len(matches) != 1:
        raise ValueError('rendered-case provenance does not resolve one active review case')
    row, case = matches[0]
    if trace['provenance']['source_id'] != case['candidate']['source_id'] or \
            bool(trace['provenance']['reverse']):
        raise ValueError('rendered-case source/direction provenance mismatch')
    active = Path(case['path'])
    if sha256_file(active) != case['sha256']:
        raise ValueError('active review annotation changed')
    archive = root / 'removed_ground_truth' / active.name
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    active.rename(archive)
    now = datetime.now(timezone.utc).isoformat()
    revisions = root / 'manifest_revisions'
    revisions.mkdir(exist_ok=True)
    revision = revisions / f'manifest_before_{now.replace(":", "").replace("+", "_")}.json'
    shutil.copy2(manifest_path, revision)
    revision_hash = sha256_file(revision)
    record = {
        'manifest_row_before_removal': row,
        'rendered_case_index': config['rendered_case_index'],
        'candidate_id': case['candidate']['id'],
        'reason': config.get('reason', 'broken-ground-truth'),
        'removed_at': now,
        'active_path_before_removal': str(active),
        'archived_path': str(archive),
        'annotation_sha256': case['sha256'],
        'case': case,
        'rendered_case': str(rendered_case),
        'render_metadata': str(rendered_case / 'rendered/render.json'),
        'render_metadata_sha256': sha256_file(rendered_case / 'rendered/render.json'),
        'rendered_image': str(rendered_case / 'rendered/strips_0.jpg'),
        'rendered_image_sha256': sha256_file(rendered_case / 'rendered/strips_0.jpg'),
        'trace_sha256': sha256_file(rendered_case / 'trace.json'),
        'manifest_revision': str(revision),
        'manifest_revision_sha256': revision_hash,
    }
    manifest['cases'].pop(row)
    manifest.setdefault('removed_ground_truth', []).append(record)
    manifest['case_count'] = len(manifest['cases'])
    manifest['total_base_voxels'] = math.fsum(
        c['length_base_voxels'] for c in manifest['cases'])
    manifest['total_mm'] = math.fsum(c['length_mm'] for c in manifest['cases'])
    manifest['review']['modified_at'] = now
    atomic_write(manifest_path, manifest)
    with (root / 'README.md').open('a') as stream:
        stream.write(f'\n- `{active.name}` — broken ground truth, identified from rendered evaluation '
                     f'case {config["rendered_case_index"]}. The annotation and audit '
                     f'record are retained in `removed_ground_truth/`. The active draft now contains '
                     f'{manifest["case_count"]} cases.\n')
    return manifest_path
