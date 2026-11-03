"""Parent identity and full widths in base-volume voxels (never crop units)."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np


def width_metadata(metadata, width):
    result = {**metadata, 'width': float(width)}
    if 'baked_annotations' in result:
        baked = dict(result['baked_annotations'])
        if 'widths_base' in baked:
            baked['widths_base'] = [float(width)] * len(baked['widths_base'])
        result['baked_annotations'] = baked
    return result


@dataclass(frozen=True)
class WidthSource:
    fiber: object
    volume_path: str


def assign_parent_widths(records, sources=()):
    """Resolve manifests once, before workers start; one vote per source piece.

    Prepared reviewed manifests are authoritative, including original parent
    hashes. A standalone file without such provenance is itself a parent.
    """
    manifests, groups, identities = {}, {}, []
    for record in [*records,*sources]:
        fiber = record.fiber
        width = float(fiber.metadata.get('width', 0))
        if not np.isfinite(width) or width < 0:
            raise ValueError(f'invalid annotated width: {fiber.path}')
        if width == 0:
            identities.append(None)
            continue
        path = (Path(fiber.path).resolve() if fiber.path is not None else
                Path('__memory__')/hashlib.sha256(fiber.line_points_xyz.tobytes()).hexdigest())
        provenance = fiber.metadata.get('parent_fiber')
        if provenance is None:
            for folder in (path.parents if fiber.path is not None else ()):
                candidate = folder / 'manifest.json'
                if not candidate.is_file():
                    continue
                if candidate not in manifests:
                    rows = json.loads(candidate.read_text()).get('files', [])
                    lookup = {}
                    for row in rows:
                        if 'parent_sha256' not in row:
                            continue
                        relative = Path(row['path'])
                        if relative.is_absolute() or '..' in relative.parts:
                            raise ValueError(f'invalid piece path in {candidate}')
                        entry = (folder / relative).resolve()
                        if not entry.is_file():
                            entry = (folder / 'test' / relative).resolve()
                        if entry in lookup:
                            raise ValueError(f'duplicate piece provenance in {candidate}')
                        lookup[entry] = dict(sha256=row['parent_sha256'],
                                            name=Path(row['parent']).name)
                    manifests[candidate] = lookup
                if manifests[candidate]:
                    if path not in manifests[candidate]:
                        raise ValueError(f'missing parent provenance for {path}')
                    provenance = manifests[candidate][path]
                    break
        if provenance is None:
            provenance = dict(name=path.name, sha256=hashlib.sha256(str(path).encode()).hexdigest())
        digest = provenance.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'invalid parent fingerprint: {path}')
        identity = dict(provenance,
            coordinate_space=fiber.metadata.get('vc_open_data_coordinate_space', record.volume_path))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        group = groups.setdefault(key, {})
        if path in group and group[path] != width:
            raise ValueError(f'inconsistent width across components of {path}')
        group[path] = width
        identities.append((key, identity))
    result = []
    for record, identity in zip(records, identities[:len(records)], strict=True):
        if identity is not None:
            key, provenance = identity
            baseline = float(np.median(list(groups[key].values())))
            metadata = width_metadata(record.fiber.metadata, baseline)
            metadata.update(parent_fiber=provenance, parent_width_key=key,
                            parent_width_baseline=baseline)
            record = replace(record, fiber=replace(record.fiber, metadata=metadata))
        result.append(record)
    return result


def calibrated_width(current, baseline, samples, config):
    """Damped batch-parent median estimate with immutable annotation tether."""
    if len(samples) == 0:
        return current
    samples = np.asarray(samples)
    if not np.isfinite(samples).all() or np.any(samples <= 0):
        raise ValueError('nonpositive/nonfinite width feedback')
    estimate = float(np.median(samples))
    target = (estimate + config.width_gt_weight * baseline) / (1 + config.width_gt_weight)
    step = config.width_learning_rate * (target - current)
    cap = config.width_max_step_fraction * baseline
    return float(max(np.finfo(float).eps, current + np.clip(step, -cap, cap)))
