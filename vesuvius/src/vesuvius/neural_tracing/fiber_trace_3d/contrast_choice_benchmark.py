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

def _package_case(root: Path, entry: dict, seen: set) -> tuple:
    """Freeze one reference plus its explicit competing choices."""
    source_manifest_path = Path(entry['source_benchmark_manifest'])
    source_manifest = json.loads(source_manifest_path.read_text())
    piece = entry['piece']
    matches = [c for c in source_manifest['cases'] if Path(c['path']).stem == piece]
    if len(matches) != 1:
        raise ValueError(f'piece does not resolve to one source case: {piece}')
    source_case, = matches
    if sha256_file(source_case['path']) != source_case['sha256']:
        raise ValueError('source benchmark annotation changed')
    reference_source = Path(source_case['path'])
    reference = json.loads(reference_source.read_text())

    # Choices come from a rendered evaluation case, from explicit paths, or both.
    requested = []
    case_root = Path(entry['source_case']) if entry.get('source_case') else None
    if entry.get('alternative_trace'):
        if case_root is None:
            raise ValueError('alternative_trace requires source_case')
        requested.append(dict(path=str(case_root / 'rendered' / entry['alternative_trace']),
                              role='competing-pre-restart-beam-trace'))
    for choice in entry.get('choices', []):
        if not choice.get('role'):
            raise ValueError('every explicit choice needs a role')
        requested.append(dict(path=str(choice['path']), role=choice['role']))
    if not requested:
        raise ValueError('a contrast case needs at least one competing choice')

    copied_reference = _copy_verified(reference_source, root / 'fibers' / f'{piece}.json')
    if copied_reference['sha256'] in seen:
        raise ValueError('duplicate contrast reference')
    seen.add(copied_reference['sha256'])

    choices = []
    for index, choice in enumerate(requested):
        source = Path(choice['path'])
        document = json.loads(source.read_text())
        # A user-supplied companion may have lost its export provenance. A
        # stated binding must still agree; an absent one is recorded so the
        # weaker verification is visible in the manifest.
        unbound = []
        for key in ('vc_open_data_coordinate_space',
                    'vc_open_data_source_original_resolution'):
            if document.get(key) is None:
                unbound.append(key)
            elif reference.get(key) != document.get(key):
                raise ValueError('reference/alternative coordinate binding mismatch')
        if len(document.get('line_points', [])) < 2:
            raise ValueError('contrast choice has no usable line')
        copied = _copy_verified(source, root / 'choices' / piece / f'{index:02d}_{source.name}')
        if copied['sha256'] in seen:
            raise ValueError('duplicate contrast choice')
        seen.add(copied['sha256'])
        choices.append(dict(path=copied['path'], sha256=copied['sha256'],
                            role=choice['role'],
                            source_path=copied['source_path'],
                            unverified_coordinate_bindings=unbound,
                            source_segment_index=document.get(
                                'evaluation_trace_provenance', {}).get('segment_index')))

    if reference.get('evaluation_direction') != 'forward':
        raise ValueError('contrast references must be directed forward lines')

    evidence = []
    provenance = dict(source_manifest=str(source_manifest_path.resolve()),
                      source_manifest_sha256=sha256_file(source_manifest_path))
    if case_root is not None:
        rendered = case_root / 'rendered'
        sources = [case_root / 'trace.json', case_root / 'reference.obj',
                   case_root / 'trace.obj']
        if rendered.is_dir():
            sources.extend(sorted(rendered.iterdir()))
        for source in sources:
            if source.is_file():
                target = root / 'evidence' / piece / (
                    'rendered' if source.parent == rendered else '') / source.name
                evidence.append(_copy_verified(source, target))
        provenance['source_evaluation_case'] = str(case_root.resolve())
        for name, path in (('source_trace_sha256', case_root / 'trace.json'),
                           ('source_render_sha256', rendered / 'render.json'),
                           ('source_image_sha256', rendered / 'strips_0.jpg')):
            if path.is_file():
                provenance[name] = sha256_file(path)

    case = deepcopy(source_case)
    case['path'] = copied_reference['path']
    case['sha256'] = copied_reference['sha256']
    case['contrast_choices'] = choices
    case['contrast_provenance'] = provenance
    return case, evidence


def package_contrast_choice(config: dict) -> Path:
    """Freeze one or more references plus their explicit competing traces."""
    root = Path(config['output_root'])
    if root.exists():
        raise FileExistsError(root)
    entries = config.get('cases')
    if entries is None:
        entries = [{k: config[k] for k in (
            'source_benchmark_manifest', 'source_case', 'piece', 'alternative_trace')
            if k in config}]
    if not entries:
        raise ValueError('no contrast cases requested')

    root.mkdir(parents=True)
    seen = set()
    cases = []
    evidence = []
    for entry in entries:
        case, case_evidence = _package_case(root, entry, seen)
        cases.append(case)
        evidence.extend(case_evidence)

    manifest = {
        'version': 1, 'completed': True, 'case_count': len(cases),
        'total_base_voxels': sum(c['length_base_voxels'] for c in cases),
        'total_mm': sum(c['length_mm'] for c in cases),
        'cases': cases,
        'benchmark_kind': 'explicit-contrast-choice',
        'evidence': evidence,
    }
    atomic_write(root / 'manifest.json', manifest)
    # The control manifest drops every choice, so the scope is the reference only.
    unrestricted = deepcopy(manifest)
    unrestricted['benchmark_kind'] = 'explicit-contrast-choice-unrestricted-control'
    for case in unrestricted['cases']:
        case.pop('contrast_choices', None)
    atomic_write(root / 'manifest_unrestricted.json', unrestricted)

    readme = ['# Explicit contrast-choice benchmark', '']
    for case in cases:
        readme.append(f'- Reference: `{Path(case["path"]).relative_to(root)}`')
        for choice in case['contrast_choices']:
            readme.append(
                f'  - {choice["role"]}: `{Path(choice["path"]).relative_to(root)}`')
    readme += [
        '', 'Every alternative is used only to define the allowed traversal scope',
        '(a fiberlet corridor or a skeleton crop tube). The reference remains the',
        'sole evaluation ground truth. `manifest_unrestricted.json` is the control',
        'without any choice.', '',
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
