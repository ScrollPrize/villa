"""auto_link_fibers writes explicit_link tags from both input formats."""

import json
import os
import sys

import pytest
from click.testing import CliRunner

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import auto_link_fibers as alf  # noqa: E402
from spiral_helpers import load_fiber_point_collection  # noqa: E402


def _fiber_doc(tags=None, **extra):
    doc = {
        'type': 'vc3d_fiber', 'version': 1, 'name': 'f',
        'control_points': [[4, 8, 12], [20, 24, 28]],
        'line_points': [[4, 8, 12], [12, 16, 20], [20, 24, 28]],
        'branches': [{'branch_file': 'other.json', 'control_point_index': 0,
                      'branch_control_point_index': 1}],
    }
    if tags is not None:
        doc['tags'] = tags
    doc.update(extra)
    return doc


@pytest.fixture
def fibers(tmp_path):
    folder = tmp_path / 'fibers'
    folder.mkdir()
    docs = {
        'anon_a.json': _fiber_doc(tags=['reviewed', 'explicit_link:stale.tifxyz']),
        'anon_b.json': _fiber_doc(),
        'anon_c.json': _fiber_doc(tags=['explicit_link:keep.tifxyz']),
        'not_a_fiber.json': {'type': 'something_else'},
    }
    for name, doc in docs.items():
        (folder / name).write_text(json.dumps(doc, indent=2 if name != 'anon_b.json' else None))
    return folder


def _links_file(tmp_path):
    path = tmp_path / 'links.json'
    path.write_text(json.dumps({
        'type': 'vc_fiber_patch_links', 'version': 1, 'settings': {},
        'fibers': {
            'anon_a.json': {
                'linked': [{'patch': 'band-1.tifxyz', 'contact': 0.9, 'anchor': True, 'round': 0},
                           {'patch': 'band-2.tifxyz', 'contact': 0.5, 'anchor': False, 'round': 1}],
                'unlinked': [{'patch': 'band-9.tifxyz', 'contact': 0.1}]},
            '/some/where/anon_b.json': {
                'linked': [{'patch': 'patch:band-3.tifxyz', 'contact': 0.7, 'anchor': True, 'round': 0}],
                'unlinked': []},
            'missing.json': {'linked': [{'patch': 'band-4.tifxyz', 'contact': 1.0,
                                         'anchor': True, 'round': 0}], 'unlinked': []},
        },
    }))
    return path


def _review_file(tmp_path):
    path = tmp_path / 'review.json'
    path.write_text(json.dumps({
        'type': 'vc3d_fiber_review', 'version': 2, 'settings': {'x': 1},
        'sources': {'big': {'geometry': list(range(1000))}},
        'fibers': {
            'fiber:1': {
                'path': '/data/fibers/anon_a.json', 'direction': 'H',
                'base': {'nodes': {}}, 'proposals': [{'id': 'patch:band-1.tifxyz'}],
                'pending': None,
                'decisions': {
                    'patch:band-1.tifxyz': {'status': 'accepted', 'order': 0,
                                            'alignment': {'seed': 'fiber:1', 'nodes': {}}},
                    'patch:band-2.tifxyz': {'status': 'rejected', 'order': 1,
                                            'alignment': {}},
                    'patch:band-5.tifxyz': {'status': 'skipped', 'order': 2, 'alignment': {}},
                }},
            'fiber:2': {
                'path': 'anon_c.json', 'direction': 'V', 'base': {}, 'proposals': [],
                'pending': None,
                'decisions': {
                    'patch:band-3.tifxyz': {'status': 'accepted', 'order': 0, 'alignment': {}},
                }},
        },
    }))
    return path


def _tags(folder, name):
    return json.loads((folder / name).read_text()).get('tags')


def test_read_patch_links_strips_prefixes_and_paths(tmp_path):
    links = alf.read_patch_links(str(_links_file(tmp_path)))
    assert links == {'anon_a.json': {'band-1.tifxyz', 'band-2.tifxyz'},
                     'anon_b.json': {'band-3.tifxyz'},
                     'missing.json': {'band-4.tifxyz'}}


def test_read_fiber_review_keeps_accepted_decisions_only(tmp_path):
    links = alf.read_fiber_review(str(_review_file(tmp_path)))
    assert links == {'anon_a.json': {'band-1.tifxyz'}, 'anon_c.json': {'band-3.tifxyz'}}


def test_read_fiber_review_version_1(tmp_path):
    path = tmp_path / 'review1.json'
    path.write_text(json.dumps({
        'type': 'vc3d_fiber_review', 'version': 1,
        'fibers': {'f': {'path': 'anon_a.json', 'direction': 'H', 'decisions': {
            'patch:x': {'accepted': True, 'alignment': {}},
            'patch:y': {'accepted': False, 'alignment': {}}}}}}))
    assert alf.read_fiber_review(str(path)) == {'anon_a.json': {'x'}}


def test_wrong_document_types_are_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'type': 'vc3d_fiber'}))
    with pytest.raises(Exception):
        alf.read_patch_links(str(path))
    with pytest.raises(Exception):
        alf.read_fiber_review(str(path))


def test_dry_run_changes_nothing_and_reports(fibers, tmp_path):
    before = {name: (fibers / name).read_text() for name in os.listdir(fibers)}
    result = CliRunner().invoke(alf.main, ['--links', str(_links_file(tmp_path)),
                                           '--fibers', str(fibers)])
    assert result.exit_code == 0, result.output
    assert 'dry run' in result.output
    assert 'anon_a.json: changed' in result.output
    assert 'missing.json' in result.output and 'not found' in result.output
    assert {name: (fibers / name).read_text() for name in os.listdir(fibers)} == before
    assert not any(p.name.startswith('fibers_backup') for p in tmp_path.iterdir())


def test_apply_merges_links_and_backs_up(fibers, tmp_path):
    backup = tmp_path / 'backup'
    result = CliRunner().invoke(alf.main, ['--links', str(_links_file(tmp_path)),
                                           '--fibers', str(fibers), '--apply',
                                           '--backup-dir', str(backup)])
    assert result.exit_code == 0, result.output
    # Merge keeps the stale tag and the other tags, adds the new ones sorted.
    assert _tags(fibers, 'anon_a.json') == [
        'reviewed', 'explicit_link:stale.tifxyz',
        'explicit_link:band-1.tifxyz', 'explicit_link:band-2.tifxyz']
    assert _tags(fibers, 'anon_b.json') == ['explicit_link:band-3.tifxyz']
    # Untouched fibers are neither rewritten nor backed up.
    assert _tags(fibers, 'anon_c.json') == ['explicit_link:keep.tifxyz']
    assert sorted(os.listdir(backup)) == ['anon_a.json', 'anon_b.json']
    assert json.loads((backup / 'anon_a.json').read_text())['tags'] == [
        'reviewed', 'explicit_link:stale.tifxyz']
    # Other content survives, and the fitter loads the tags.
    doc = json.loads((fibers / 'anon_a.json').read_text())
    assert doc['branches'] == _fiber_doc()['branches']
    assert doc['line_points'] == _fiber_doc()['line_points']
    pcl = load_fiber_point_collection(str(fibers / 'anon_a.json'), 1, min_point_spacing=0)
    assert pcl['metadata']['explicit_patch_links'] == [
        'band-1.tifxyz', 'band-2.tifxyz', 'stale.tifxyz']
    # Indentation style is preserved per file.
    assert (fibers / 'anon_a.json').read_text().startswith('{\n  "')
    assert '\n' not in (fibers / 'anon_b.json').read_text()
    # Idempotent.
    again = CliRunner().invoke(alf.main, ['--links', str(_links_file(tmp_path)),
                                          '--fibers', str(fibers), '--apply',
                                          '--backup-dir', str(backup)])
    assert again.exit_code == 0 and 'nothing to write' in again.output


def test_replace_drops_existing_explicit_links(fibers, tmp_path):
    result = CliRunner().invoke(alf.main, ['--review', str(_review_file(tmp_path)),
                                           '--fibers', str(fibers), '--apply', '--replace',
                                           '--backup-dir', str(tmp_path / 'backup')])
    assert result.exit_code == 0, result.output
    assert _tags(fibers, 'anon_a.json') == ['reviewed', 'explicit_link:band-1.tifxyz']
    assert _tags(fibers, 'anon_c.json') == ['explicit_link:band-3.tifxyz']
    # A fiber absent from the review loses its tags under --replace; one that
    # never had any is left alone.
    assert _tags(fibers, 'anon_b.json') is None
    assert json.loads((fibers / 'not_a_fiber.json').read_text()) == {'type': 'something_else'}


def test_exactly_one_input_is_required(fibers, tmp_path):
    runner = CliRunner()
    assert runner.invoke(alf.main, ['--fibers', str(fibers)]).exit_code != 0
    assert runner.invoke(alf.main, ['--fibers', str(fibers), '--links', str(_links_file(tmp_path)),
                                    '--review', str(_review_file(tmp_path))]).exit_code != 0
