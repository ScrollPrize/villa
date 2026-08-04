"""Unit tests for fiber_migrate_v1_to_v3.

Every migrated document must satisfy the same acceptance stack the
migration script itself runs: fiber_merge.is_fiber_doc, per-segment
_valid_segment, and the loader_issues mini-port of VC3D's load-time
validation. Geometry must be value-identical to the input.
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fiber_merge
import fiber_migrate_v1_to_v3 as migrate
from fiber_loader_checks import loader_issues
from test_fiber_merge import cp, make_fiber, make_v3_fiber, make_pair


def write_fiber(directory, doc, name=None):
    name = name or doc['filename']
    path = os.path.join(str(directory), name)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    return path


def read_json(path):
    with open(path) as f:
        return json.load(f)


CPS = [cp(i) for i in range(4)]


def test_migrated_doc_passes_all_validators():
    doc = migrate.migrate_doc(make_fiber(CPS))
    assert fiber_merge.is_fiber_doc(doc)
    assert doc['version'] == 3
    assert doc['optimization_mode'] == 'lasagna'
    segments = [entry['segment_to_next']
                for entry in doc['control_points'][:-1]]
    for segment in segments:
        assert fiber_merge._valid_segment(segment)
        assert segment['interp_goal'] == 'global'
        assert segment['interp_mode'] == 'lasagna'
    assert loader_issues({doc['filename']: doc}) == []


def test_geometry_is_value_identical():
    original = make_fiber(CPS)
    doc = migrate.migrate_doc(copy.deepcopy(original))
    assert doc['line_points'] == original['line_points']
    positions = [entry['position'] for entry in doc['control_points']]
    assert positions == original['control_points']


def test_final_control_point_has_no_segment():
    doc = migrate.migrate_doc(make_fiber(CPS))
    assert 'segment_to_next' not in doc['control_points'][-1]
    for entry in doc['control_points'][:-1]:
        assert 'segment_to_next' in entry


def test_single_and_two_control_point_fibers():
    single = make_fiber([cp(0)])
    single['line_points'] = []
    doc = migrate.migrate_doc(single)
    assert doc['control_points'] == [{'position': cp(0)}]
    assert fiber_merge.is_fiber_doc(doc)

    two = migrate.migrate_doc(make_fiber([cp(0), cp(1)]))
    assert 'segment_to_next' in two['control_points'][0]
    assert 'segment_to_next' not in two['control_points'][1]
    assert fiber_merge.is_fiber_doc(two)


def test_metadata_and_unknown_keys_preserved():
    original = make_fiber(CPS, tags=['approved'],
                          branches=[])
    original['started_at'] = '20260101T000000000'
    original['sequence'] = 7
    original['vc_open_data_coordinate_space'] = 'scroll'
    original['novel_future_key'] = {'nested': [1, 2]}
    doc = migrate.migrate_doc(copy.deepcopy(original))
    for key in ('type', 'filename', 'username', 'started_at', 'sequence',
                'tags', 'branches', 'vc_open_data_coordinate_space',
                'novel_future_key'):
        assert doc[key] == original[key]


def test_generation_bumped_by_one():
    assert migrate.migrate_doc(make_fiber(CPS, generation=5))['generation'] == 6
    missing = make_fiber(CPS)
    del missing['generation']
    assert migrate.migrate_doc(missing)['generation'] == 2
    none = make_fiber(CPS)
    none['generation'] = None
    assert migrate.migrate_doc(none)['generation'] == 2


def test_segment_constant_matches_strict_key_sets():
    assert set(migrate.LASAGNA_SEGMENT) == fiber_merge._SEGMENT_KEYS_V3
    assert set(migrate.LASAGNA_SEGMENT['config']) == fiber_merge._CONFIG_KEYS_V3
    for key in ('cone_grid_size', 'beam_width', 'beam_lookahead_steps',
                'cumulative_smoothness_steps'):
        assert isinstance(migrate.LASAGNA_SEGMENT['config'][key], int)
        assert not isinstance(migrate.LASAGNA_SEGMENT['config'][key], bool)


def test_hv_classification_horizontal_vertical_and_degenerate():
    horizontal = migrate.hv_classification([[0.0, 0.0, 0.0],
                                            [10.0, 0.0, 0.0]])
    assert horizontal['automatic_tag'] == 'H'
    assert horizontal['control_point_length'] == pytest.approx(10.0)
    assert horizontal['vertical_score'] == 0.0
    assert horizontal['horizontal_score'] == 1.0
    assert horizontal['automatic_certainty'] == pytest.approx(1.0)

    vertical = migrate.hv_classification([[0.0, 0.0, 0.0],
                                          [0.0, 0.0, 10.0]])
    assert vertical['automatic_tag'] == 'V'
    assert vertical['vertical_score'] == pytest.approx(1.0)

    degenerate = migrate.hv_classification([[0.0, 0.0, 0.0]])
    assert degenerate['automatic_tag'] == 'unknown'
    assert degenerate['control_point_length'] == 0.0

    zero_length = migrate.hv_classification([[1.0, 1.0, 1.0],
                                             [1.0, 1.0, 1.0]])
    assert zero_length['automatic_tag'] == 'unknown'


def test_hv_manual_tag_preserved_and_normalized():
    doc = make_fiber(CPS)
    doc['hv_classification'] = {'manual_tag': 'h'}
    assert migrate.migrate_doc(doc)['hv_classification']['manual_tag'] == 'H'
    doc = make_fiber(CPS)
    doc['hv_classification'] = {'manual_tag': 'vertical'}
    assert migrate.migrate_doc(doc)['hv_classification']['manual_tag'] == 'V'
    assert migrate.migrate_doc(
        make_fiber(CPS))['hv_classification']['manual_tag'] == ''


def test_directory_migration_and_idempotency(tmp_path):
    write_fiber(tmp_path, make_fiber(CPS, filename='a.json'), 'a.json')
    write_fiber(tmp_path, make_v3_fiber(CPS, filename='b.json'), 'b.json')

    counts, issues = migrate.migrate_directory(str(tmp_path), log=lambda *_: None)
    assert issues == []
    assert counts == {'migrated': 1, 'already_v3': 1,
                      'skipped_non_fiber': 0, 'failed': 0}

    migrated_bytes = open(os.path.join(tmp_path, 'a.json'), 'rb').read()
    v3_bytes = open(os.path.join(tmp_path, 'b.json'), 'rb').read()
    counts, issues = migrate.migrate_directory(str(tmp_path), log=lambda *_: None)
    assert issues == []
    assert counts['migrated'] == 0 and counts['already_v3'] == 2
    assert open(os.path.join(tmp_path, 'a.json'), 'rb').read() == migrated_bytes
    assert open(os.path.join(tmp_path, 'b.json'), 'rb').read() == v3_bytes


def test_linked_pair_stays_reciprocal(tmp_path):
    a, b = make_pair('a.json', 'b.json',
                     [cp(i) for i in range(4)],
                     [cp(i, dz=40.0) for i in range(4)],
                     a_index=3, b_index=0)
    write_fiber(tmp_path, a, 'a.json')
    write_fiber(tmp_path, b, 'b.json')
    counts, issues = migrate.migrate_directory(str(tmp_path), log=lambda *_: None)
    assert counts['migrated'] == 2
    assert issues == []
    migrated = {name: read_json(os.path.join(tmp_path, name))
                for name in ('a.json', 'b.json')}
    assert migrated['a.json']['branches'] == a['branches']
    assert loader_issues(migrated) == []


def test_only_files_restricts_migration(tmp_path):
    write_fiber(tmp_path, make_fiber(CPS, filename='a.json'), 'a.json')
    write_fiber(tmp_path, make_fiber(CPS, filename='b.json'), 'b.json')
    counts, issues = migrate.migrate_directory(
        str(tmp_path), only_files=['a.json'], log=lambda *_: None)
    assert counts['migrated'] == 1
    assert issues == []
    assert read_json(os.path.join(tmp_path, 'a.json'))['version'] == 3
    assert read_json(os.path.join(tmp_path, 'b.json'))['version'] == 1


def test_dry_run_writes_nothing(tmp_path):
    path = write_fiber(tmp_path, make_fiber(CPS, filename='a.json'), 'a.json')
    before = open(path, 'rb').read()
    counts, issues = migrate.migrate_directory(str(tmp_path), dry_run=True,
                                               log=lambda *_: None)
    assert counts['migrated'] == 1
    assert issues == []
    assert open(path, 'rb').read() == before
    assert not [name for name in os.listdir(tmp_path)
                if name.endswith('.migrate-tmp')]


def test_malformed_and_non_fiber_inputs(tmp_path):
    with open(os.path.join(tmp_path, 'broken.json'), 'w') as f:
        f.write('{not json')
    with open(os.path.join(tmp_path, 'other.json'), 'w') as f:
        json.dump({'type': 'something_else'}, f)
    with open(os.path.join(tmp_path, 'invalid.json'), 'w') as f:
        json.dump({'type': 'vc3d_fiber', 'version': 2}, f)
    with open(os.path.join(tmp_path, '.hidden.json'), 'w') as f:
        f.write('{not json either')
    messages = []
    counts, issues = migrate.migrate_directory(str(tmp_path),
                                               log=messages.append)
    assert counts == {'migrated': 0, 'already_v3': 0,
                      'skipped_non_fiber': 1, 'failed': 2}
    assert issues == []
    assert any('broken.json' in message for message in messages)
    assert any('invalid.json' in message for message in messages)


def test_cli_exit_codes(tmp_path):
    write_fiber(tmp_path, make_fiber(CPS, filename='a.json'), 'a.json')
    assert migrate.main([str(tmp_path), '--dry-run']) == 0
    assert migrate.main([str(tmp_path)]) == 0
    with open(os.path.join(tmp_path, 'broken.json'), 'w') as f:
        f.write('{not json')
    assert migrate.main([str(tmp_path)]) == 1
