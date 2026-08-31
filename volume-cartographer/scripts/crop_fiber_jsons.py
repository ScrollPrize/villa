#!/usr/bin/env python3
"""Crop dense VC3D fiber paths to a base-XYZ bounding box in place."""

import argparse
import json
import math
import os
import sys


_EPSILON = 1.0e-9


def _point(value, context):
    if (not isinstance(value, list) or len(value) != 3 or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(item) for item in value)):
        raise ValueError(f'{context} must be a finite XYZ array')
    return tuple(float(item) for item in value)


def _inside(point, minimum, maximum):
    return all(minimum[axis] <= point[axis] < maximum[axis]
               for axis in range(3))


def _close(first, second):
    return sum((a - b) ** 2 for a, b in zip(first, second)) <= _EPSILON ** 2


def _clip_segment(start, finish, minimum, maximum):
    delta = tuple(finish[axis] - start[axis] for axis in range(3))
    begin = 0.0
    end = 1.0
    for axis in range(3):
        if abs(delta[axis]) <= _EPSILON:
            if not minimum[axis] <= start[axis] < maximum[axis]:
                return None
            continue
        near = (minimum[axis] - start[axis]) / delta[axis]
        far = (maximum[axis] - start[axis]) / delta[axis]
        if near > far:
            near, far = far, near
        begin = max(begin, near)
        end = min(end, far)
        if end <= begin + _EPSILON:
            return None
    return (
        tuple(start[axis] + delta[axis] * begin for axis in range(3)),
        tuple(start[axis] + delta[axis] * end for axis in range(3)),
    )


def clip_polyline(points, minimum, maximum):
    """Return ordered connected runs inside the half-open box."""
    runs = []
    current = []

    def finish_current():
        nonlocal current
        if len(current) >= 2:
            runs.append(current)
        current = []

    for start, finish in zip(points[:-1], points[1:]):
        clipped = _clip_segment(start, finish, minimum, maximum)
        if clipped is None:
            finish_current()
            continue
        clipped_start, clipped_finish = clipped
        if not current or not _close(current[-1], clipped_start):
            finish_current()
            current = [clipped_start]
        if not _close(current[-1], clipped_finish):
            current.append(clipped_finish)
        if not _inside(finish, minimum, maximum):
            finish_current()
    finish_current()
    return runs


def _control_point_position(value, context):
    if isinstance(value, dict):
        value = value.get('position')
    return _point(value, context)


def _squared_segment_distance(point, start, finish):
    delta = tuple(finish[axis] - start[axis] for axis in range(3))
    length_squared = sum(value * value for value in delta)
    if length_squared <= _EPSILON ** 2:
        return sum((point[axis] - start[axis]) ** 2 for axis in range(3))
    fraction = max(0.0, min(1.0, sum(
        (point[axis] - start[axis]) * delta[axis] for axis in range(3)) /
        length_squared))
    return sum((point[axis] - start[axis] - fraction * delta[axis]) ** 2
               for axis in range(3))


def _select_control_point_run(document, runs, minimum, maximum, context):
    controls = document.get('control_points')
    if not isinstance(controls, list):
        raise ValueError(
            f'{context} has disconnected crop runs and no control-point array')
    points = [
        _control_point_position(value, f'{context} control_points[{index}]')
        for index, value in enumerate(controls)
    ]
    points = [point for point in points if _inside(point, minimum, maximum)]
    if not points:
        raise ValueError(
            f'{context} has disconnected crop runs without an in-crop control point')
    tolerance_squared = 1.0e-6
    support = []
    for run in runs:
        support.append(sum(
            min(_squared_segment_distance(point, start, finish)
                for start, finish in zip(run[:-1], run[1:])) <=
            tolerance_squared
            for point in points))
    maximum_support = max(support)
    winners = [index for index, value in enumerate(support)
               if value == maximum_support]
    if maximum_support == 0 or len(winners) != 1:
        raise ValueError(
            f'{context} has ambiguous disconnected crop runs '
            f'(control-point support {support})')
    return runs[winners[0]]


def crop_document(document, minimum, maximum, context='<fiber>'):
    if not isinstance(document, dict) or document.get('type') != 'vc3d_fiber':
        raise ValueError(f'{context} is not a vc3d_fiber document')
    values = document.get('line_points')
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f'{context} line_points must contain at least two points')
    points = [_point(value, f'{context} line_points[{index}]')
              for index, value in enumerate(values)]
    runs = clip_polyline(points, minimum, maximum)
    if not runs:
        raise ValueError(f'{context} does not intersect the crop')
    run = runs[0] if len(runs) == 1 else _select_control_point_run(
        document, runs, minimum, maximum, context)
    result = dict(document)
    result['line_points'] = [list(point) for point in run]
    result['generation'] = max(1, int(document.get('generation') or 1)) + 1
    return result


def _atomic_write(path, document):
    temporary = path + '.crop-tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as output:
            json.dump(document, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def crop_directory(directory, minimum, maximum, tag=None, dry_run=False,
                   log=print):
    counts = {'scanned': 0, 'selected': 0, 'updated': 0, 'before': 0,
              'after': 0, 'skipped': 0, 'failed': 0}
    for name in sorted(os.listdir(directory)):
        if name.startswith('.') or not name.endswith('.json'):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        counts['scanned'] += 1
        try:
            with open(path, encoding='utf-8') as source:
                document = json.load(source)
            if not isinstance(document, dict) or document.get('type') != 'vc3d_fiber':
                counts['skipped'] += 1
                continue
            if tag is not None and tag not in document.get('tags', []):
                counts['skipped'] += 1
                continue
            counts['selected'] += 1
            cropped = crop_document(document, minimum, maximum, path)
            before = len(document['line_points'])
            after = len(cropped['line_points'])
            if not dry_run:
                _atomic_write(path, cropped)
            counts['updated'] += 1
            counts['before'] += before
            counts['after'] += after
            log(f'{name}: {before} -> {after} line points')
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            counts['failed'] += 1
            log(f'{name}: FAILED: {error}')
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument(
        '--bbox', nargs=6, type=float, required=True,
        metavar=('X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'))
    parser.add_argument('--tag', help='only update fibers carrying this tag')
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args(argv)
    if not os.path.isdir(arguments.directory):
        parser.error(f'not a directory: {arguments.directory}')
    minimum = tuple(arguments.bbox[:3])
    maximum = tuple(arguments.bbox[3:])
    if any(not minimum[axis] < maximum[axis] for axis in range(3)):
        parser.error('bbox maximum must be greater than minimum on every axis')
    counts = crop_directory(
        arguments.directory, minimum, maximum, arguments.tag,
        arguments.dry_run)
    print(
        f"scanned={counts['scanned']} selected={counts['selected']} "
        f"updated={counts['updated']} skipped={counts['skipped']} "
        f"failed={counts['failed']} points={counts['before']}->{counts['after']}")
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
