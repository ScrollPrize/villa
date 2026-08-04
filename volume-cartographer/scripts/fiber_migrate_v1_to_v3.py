#!/usr/bin/env python3
"""Migrate VC3D fiber annotation JSONs from version 1 to version 3.

Performs exactly the upgrade VC3D itself applies when it re-saves a v1
fiber (fiberSaveSnapshotToJson): version 3, optimization_mode "lasagna",
and a default lasagna segment descriptor (interp_goal "global",
interp_mode "lasagna") on every non-final control point. Geometry is
copied verbatim — control_points and line_points values are never
recomputed or reformatted, preserving the loader's C1 invariant (every
control point an exact ordered subsequence of line_points within 1e-8).

Doing this in bulk, instead of letting VC3D upgrade files one at a time
as they happen to be opened, keeps the sync base consistent: fiber_merge
treats a version mismatch between base/local/remote as a manual
conflict, so a directory should move to v3 in one coordinated step.

The script changes nothing else. In particular it never re-interpolates,
never touches branches or tags, and preserves unknown top-level keys
(e.g. vc_open_data_*) verbatim. hv_classification is recomputed with the
same formula as VC3D (classifyFiberHv) so the migrated file is not
immediately rewritten by the loader's staleness check, and generation is
bumped by one because the file content changed (matching VC3D saves and
vc_lasagna_line_probe's rewrite behavior).

Usage:
    fiber_migrate_v1_to_v3.py <dir> [--dry-run] [--file NAME ...]

<dir> is one project's fiber directory (e.g.
fibers/PHercParis4.volpkg.json/); *.json files are processed
non-recursively, dotfiles and non-vc3d_fiber documents are skipped.
--file restricts migration to the named files (basenames) while still
loading the rest of the directory for the cross-file acceptance check —
useful when repairing a single file after a sync version conflict.
"""
import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fiber_merge
from fiber_loader_checks import loader_issues

# Byte-mirror of makeLasagnaSegmentMetadataJson("global", "", 1.0,
# <default FiberTraceConfig>, null) in core/include/vc/fiber_tracer/
# FiberJson.hpp — what VC3D writes for a v1-upgraded span. The four
# integer config values must stay Python ints: the strict validators
# (FiberJson.hpp, fiber_merge._valid_segment) reject 25.0 for them.
LASAGNA_SEGMENT = {
    'optimizer': 'native_fiber_trace3d',
    'metadata_version': 3,
    'tracer_version': 2,
    'interp_goal': 'global',
    'interp_mode': 'lasagna',
    'metric': None,
    'msg': 'lasagna',
    'normal_manifest': '',
    'fiber_manifest': '',
    'trace_to_base_scale': 1.0,
    'meeting_error_base_voxels': None,
    'meeting_error_ratio': None,
    'meeting_source': '',
    'failure_code': '',
    'failure_detail': '',
    'lasagna_failure_code': '',
    'lasagna_failure_detail': '',
    'config': {
        'step_voxels': 4.0,
        'cone_angle_degrees': 25.0,
        'cone_angle_step_degrees': 5.0,
        'cone_grid_size': 25,
        'beam_width': 8,
        'beam_prune_distance_voxels': 1.0,
        'beam_lookahead_steps': 2,
        'smoothness_weight': 2.0,
        'smoothness_normal_weight': 0.1,
        'smoothness_tangent_weight': 10.0,
        'smoothness_free_angle_degrees': 0.0,
        'cumulative_smoothness_steps': 4,
        'cumulative_smoothness_tangent_weight': 2.0,
        'initial_free_angle_degrees': 0.0,
        'max_step_factor': 3.0,
        'meeting_accept_max_error_ratio': 0.1,
        'endpoint_accept_threshold_base_voxels': 20.0,
    },
}


def _normalized_manual_tag(value):
    """fiberHvTagFromString + ToString, with "unknown" stored as ""."""
    if value in ('H', 'h', 'horizontal'):
        return 'H'
    if value in ('V', 'v', 'vertical'):
        return 'V'
    return ''


def hv_classification(control_points, manual_tag=''):
    """Port of vc::atlas::classifyFiberHv over the control points, in the
    shape fiberSaveSnapshotToJson serializes. Must stay float-identical to
    the C++ (the loader re-runs it and rewrites the file on any 1e-9
    disagreement)."""
    result = {
        'z_distance': 0.0,
        'control_point_length': 0.0,
        'horizontal_score': 0.0,
        'vertical_score': 0.0,
        'automatic_tag': 'unknown',
        'automatic_certainty': 0.0,
        'manual_tag': _normalized_manual_tag(manual_tag),
    }
    if len(control_points) < 2:
        return result
    length = 0.0
    for a, b in zip(control_points[:-1], control_points[1:]):
        step = math.sqrt(sum((y - x) ** 2 for x, y in zip(a, b)))
        if math.isfinite(step):
            length += step
    result['control_point_length'] = length
    if not math.isfinite(length) or length <= 0.0:
        return result
    z_distance = abs(control_points[-1][2] - control_points[0][2])
    vertical = min(max(z_distance / length, 0.0), 1.0)
    result['z_distance'] = z_distance
    result['vertical_score'] = vertical
    result['horizontal_score'] = 1.0 - vertical
    result['automatic_tag'] = 'V' if vertical >= 0.5 else 'H'
    result['automatic_certainty'] = min(max(abs(vertical - 0.5) * 2.0, 0.0),
                                        1.0)
    return result


def migrate_doc(doc):
    """v1 -> v3, in place on a dict from json.load. The dict's key order
    (and any unknown top-level keys) is preserved; only version,
    optimization_mode, control_points, generation, and hv_classification
    change. Geometry values are reused verbatim."""
    positions = doc['control_points']
    doc['version'] = 3
    doc['optimization_mode'] = 'lasagna'
    doc['control_points'] = [
        {'position': p, 'segment_to_next': copy.deepcopy(LASAGNA_SEGMENT)}
        if index + 1 < len(positions) else {'position': p}
        for index, p in enumerate(positions)
    ]
    doc['generation'] = max(1, int(doc.get('generation') or 1)) + 1
    existing_hv = doc.get('hv_classification')
    manual_tag = (existing_hv.get('manual_tag', '')
                  if isinstance(existing_hv, dict) else '')
    doc['hv_classification'] = hv_classification(positions, manual_tag)
    return doc


def _atomic_write(path, doc):
    tmp = path + '.migrate-tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(doc, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def migrate_directory(directory, only_files=None, dry_run=False,
                      log=print):
    """Migrate every v1 fiber in `directory`. Returns a dict of counters;
    'failed' > 0 means at least one file could not be migrated. The
    final docs (migrated or not) are run through loader_issues so
    cross-file branch reciprocity is checked over the whole set."""
    counts = {'migrated': 0, 'already_v3': 0, 'skipped_non_fiber': 0,
              'failed': 0}
    final_docs = {}
    only = set(only_files) if only_files else None
    for name in sorted(os.listdir(directory)):
        if name.startswith('.') or not name.endswith('.json'):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (ValueError, OSError) as ex:
            log(f"  FAILED {name}: unreadable JSON ({ex})")
            counts['failed'] += 1
            continue
        if not fiber_merge.is_fiber_doc(doc):
            if isinstance(doc, dict) and doc.get('type') == 'vc3d_fiber':
                log(f"  FAILED {name}: invalid vc3d_fiber document")
                counts['failed'] += 1
            else:
                counts['skipped_non_fiber'] += 1
            continue
        if doc.get('version', 1) == 3:
            counts['already_v3'] += 1
            final_docs[name] = doc
            continue
        if only is not None and name not in only:
            final_docs[name] = doc
            continue
        old_positions = [list(p) for p in doc['control_points']]
        old_line_points = doc['line_points']
        migrated = migrate_doc(doc)
        new_positions = [cp['position'] for cp in migrated['control_points']]
        if (new_positions != old_positions or
                migrated['line_points'] is not old_line_points or
                not fiber_merge.is_fiber_doc(migrated)):
            log(f"  FAILED {name}: migrated document failed validation")
            counts['failed'] += 1
            continue
        final_docs[name] = migrated
        if not dry_run:
            try:
                _atomic_write(path, migrated)
            except (ValueError, OSError) as ex:
                log(f"  FAILED {name}: could not write ({ex})")
                counts['failed'] += 1
                del final_docs[name]
                continue
        counts['migrated'] += 1
    issues = loader_issues(final_docs)
    return counts, issues


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0])
    parser.add_argument('directory',
                        help="one project's fiber directory")
    parser.add_argument('--dry-run', action='store_true',
                        help='validate and report without writing')
    parser.add_argument('--file', action='append', dest='files',
                        metavar='NAME',
                        help='migrate only this file (repeatable); the '
                             'rest of the directory is still checked')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory}")

    counts, issues = migrate_directory(args.directory,
                                       only_files=args.files,
                                       dry_run=args.dry_run)
    action = 'would migrate' if args.dry_run else 'migrated'
    print(f"{action}: {counts['migrated']}  "
          f"already v3: {counts['already_v3']}  "
          f"skipped (not fibers): {counts['skipped_non_fiber']}  "
          f"failed: {counts['failed']}")
    if issues:
        print(f"loader acceptance found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("loader acceptance: clean")
    return 1 if counts['failed'] or issues else 0


if __name__ == '__main__':
    sys.exit(main())
