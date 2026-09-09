import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crop_fiber_jsons


def test_crop_document_clips_crossing_path_and_preserves_metadata():
    document = {
        'type': 'vc3d_fiber',
        'version': 3,
        'generation': 4,
        'tags': ['reference'],
        'control_points': [{'position': [5, 5, 5]}],
        'line_points': [[-5, 5, 5], [5, 5, 5], [15, 5, 5]],
    }

    cropped = crop_fiber_jsons.crop_document(
        document, (0, 0, 0), (10, 10, 10))

    assert cropped['line_points'] == [[0.0, 5.0, 5.0],
                                      [5.0, 5.0, 5.0],
                                      [10.0, 5.0, 5.0]]
    assert cropped['control_points'] == document['control_points']
    assert cropped['tags'] == ['reference']
    assert cropped['generation'] == 5
    assert document['generation'] == 4


def test_crop_document_selects_disconnected_run_with_control_point():
    document = {
        'type': 'vc3d_fiber',
        'generation': 1,
        'control_points': [{'position': [5, 8, 8]}],
        'line_points': [
            [-5, 2, 2], [5, 2, 2], [15, 2, 2],
            [15, 8, 8], [5, 8, 8], [-5, 8, 8],
        ],
    }

    cropped = crop_fiber_jsons.crop_document(
        document, (0, 0, 0), (10, 10, 10))

    assert cropped['line_points'] == [[10.0, 8.0, 8.0],
                                      [5.0, 8.0, 8.0],
                                      [0.0, 8.0, 8.0]]


def test_crop_document_rejects_ambiguous_disconnected_reentry():
    document = {
        'type': 'vc3d_fiber',
        'generation': 1,
        'control_points': [],
        'line_points': [
            [-5, 2, 2], [5, 2, 2], [15, 2, 2],
            [15, 8, 8], [5, 8, 8], [-5, 8, 8],
        ],
    }

    try:
        crop_fiber_jsons.crop_document(
            document, (0, 0, 0), (10, 10, 10))
    except ValueError as error:
        assert 'without an in-crop control point' in str(error)
    else:
        raise AssertionError('ambiguous disconnected runs were accepted')


def test_crop_directory_dry_run_does_not_modify_file(tmp_path):
    path = tmp_path / 'fiber.json'
    document = {
        'type': 'vc3d_fiber',
        'generation': 2,
        'tags': ['wanted'],
        'line_points': [[-5, 5, 5], [15, 5, 5]],
    }
    path.write_text(json.dumps(document))

    counts = crop_fiber_jsons.crop_directory(
        tmp_path, (0, 0, 0), (10, 10, 10), tag='wanted', dry_run=True,
        log=lambda _: None)

    assert counts == {
        'scanned': 1, 'selected': 1, 'updated': 1, 'before': 2,
        'after': 2, 'skipped': 0, 'failed': 0,
    }
    assert json.loads(path.read_text()) == document
