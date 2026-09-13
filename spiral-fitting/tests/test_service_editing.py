"""Integrated immutable transport, catalog, resident and publication lifecycle."""
import copy
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from input_publication import _Publication
from service_editing import EditingWorkspace
from service_http import ApiError

TOKEN = 'test-client'


class Resident:
    def __init__(self):
        self.calls = []
        self.members = {}
        self.fail = False

    def apply_input_changes(self, command, records, influence_config=None):
        self.calls.append((command, copy.deepcopy(records)))
        if self.fail:
            return {'applied': False, 'errors': {'rank0': 'invalid input'}}
        self.members.update({r['id']: r for r in records})
        return {'applied': True}


@pytest.fixture
def workspace(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    path = dataset / 'same_windings.json'
    path.write_text(json.dumps({'collections': {'7': {'name': 'base', 'points': {}}}}))
    resident = Resident()
    workspace = EditingWorkspace(dataset, tmp_path / 'output',
        {'pcl_inputs': [{'path': str(path), 'role': 'same_winding'}]}, lambda: resident)
    workspace.claim(TOKEN, 'claim')
    yield workspace, resident
    workspace.close()


def upload(workspace, name, kind='pcl', input_id=None):
    doc = ({'vc_pointcollections_json_version': '1', 'collections': {'0': {'name': name, 'points': {str(i): {'p': [i, 2, 3], 'creation_time': i} for i in range(2)}}}} if kind == 'pcl' else
           {'type': 'vc3d_fiber', 'version': 1, 'points': []})
    data = json.dumps(doc).encode()
    request = {'upload_id': uuid4().hex, 'id': input_id or str(uuid4()), 'kind': kind,
               'files': [{'name': 'input.json', 'size': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()}]}
    if kind == 'pcl':
        request['role'] = 'same_winding'
    transfer = workspace.uploads.begin(request)['upload_id']
    workspace.uploads.receive(transfer, 'input.json', io.BytesIO(data), len(data))
    workspace.uploads.finalize(transfer)
    return transfer


def change(workspace, input_id, expected, name=None, **kwargs):
    item = {'id': input_id, 'expected_revision': expected, **kwargs}
    if name is not None:
        item['upload_id'] = upload(workspace, name)
    return workspace.change(TOKEN, {'command_id': str(uuid4()), 'changes': [item]})


def commit(workspace, input_id, revision, command=None):
    return workspace.commit(TOKEN, {'command_id': command or str(uuid4()),
        'revisions': [{'id': input_id, 'revision': revision}]})


def test_transport_never_accepts_or_commits_and_revisions_are_exact(workspace):
    ws, resident = workspace
    entry = ws.catalog.entries()[0]
    target = Path(entry.identity.source)
    original = target.read_bytes()
    first = upload(ws, 'first', input_id=entry.identity.id)
    second = upload(ws, 'second', input_id=entry.identity.id)
    assert target.read_bytes() == original
    assert ws.catalog.entry(entry.identity.id).accepted == 1
    for expected, transfer in [(1, first), (2, second)]:
        ws.change(TOKEN, {'command_id': str(uuid4()), 'changes': [{
            'id': entry.identity.id, 'expected_revision': expected, 'upload_id': transfer}]})
    commit(ws, entry.identity.id, 2)
    assert len(resident.calls) == 2  # An older Commit does not roll the fit back.
    assert json.loads(target.read_text())['collections']['7']['name'] == 'first'
    state = ws.catalog.entry(entry.identity.id)
    assert (state.accepted, state.applied, state.persisted) == (3, 3, 2)
    commit(ws, entry.identity.id, 3)
    with pytest.raises(ApiError, match='older revision'):
        commit(ws, entry.identity.id, 2)
    assert json.loads(target.read_text())['collections']['7']['name'] == 'second'


def test_conflict_is_scoped_and_does_not_gate_repairs(workspace):
    ws, _ = workspace
    entry = ws.catalog.entries()[0]
    change(ws, entry.identity.id, 1, 'local')
    target = Path(entry.identity.source)
    external = json.loads(target.read_text())
    external['collections']['7']['name'] = 'external'
    target.write_text(json.dumps(external))
    with pytest.raises(ApiError) as error:
        commit(ws, entry.identity.id, 2, 'conflict')
    assert error.value.payload['conflicts'][0]['id'] == entry.identity.id
    assert not ws.transactions
    assert ws.coordinator.outcome('conflict')['state'] == 'rejected'
    change(ws, entry.identity.id, 2, 'repair')
    assert json.loads(target.read_text()) == external


def test_unrelated_pcl_changes_merge_and_command_path_is_not_interpreted(workspace, tmp_path):
    ws, _ = workspace
    entry = ws.catalog.entries()[0]
    change(ws, entry.identity.id, 1, 'local')
    target = Path(entry.identity.source)
    external = json.loads(target.read_text())
    external['collections']['99'] = {'name': 'unrelated', 'points': {}}
    target.write_text(json.dumps(external))
    result = commit(ws, entry.identity.id, 2, '../../../outside')
    assert result['committed'] == [entry.identity.id]
    actual = json.loads(target.read_text())
    assert actual['collections']['99'] == external['collections']['99']
    assert actual['collections']['7']['name'] == 'local'
    assert not (tmp_path / 'outside').exists()


def test_publication_failure_resumes_without_self_conflict(workspace, monkeypatch):
    ws, resident = workspace
    first = ws.catalog.entries()[0]
    second_id = str(uuid4())
    transfer = upload(ws, '', kind='fiber', input_id=second_id)
    result = ws.change(TOKEN, {'command_id': 'mixed', 'changes': [
        {'id': first.identity.id, 'expected_revision': 1, 'upload_id': upload(ws, 'mixed')},
        {'id': second_id, 'kind': 'fiber', 'expected_revision': 0, 'upload_id': transfer}]})
    request = {'command_id': 'commit-mixed', 'revisions': result['revisions']}
    original = _Publication.publish
    count = 0
    def interrupted(publication):
        nonlocal count
        count += 1
        original(publication)
        if count == 1:
            raise OSError('lost publication acknowledgement')
    monkeypatch.setattr(_Publication, 'publish', interrupted)
    with pytest.raises(ApiError, match='needs recovery'):
        ws.commit(TOKEN, request)
    assert ws.catalog.entry(first.identity.id).persisted == 1
    assert ws.transactions
    with pytest.raises(ApiError, match='needs recovery'):
        change(ws, first.identity.id, 2, 'later')
    result = ws.commit(TOKEN, request)
    assert set(result['committed']) == {first.identity.id, second_id}
    assert ws.commit(TOKEN, request) == result
    assert len(resident.calls) == 1
    assert not ws.transactions


def test_delete_restore_and_monotonic_ids(workspace):
    ws, resident = workspace
    entry = ws.catalog.entries()[0]
    change(ws, entry.identity.id, 1, deleted=True)
    assert Path(entry.identity.source).exists()
    change(ws, entry.identity.id, 2, restore_revision=1)
    change(ws, entry.identity.id, 3, deleted=True)
    commit(ws, entry.identity.id, 4)
    with pytest.raises(ApiError, match='new input'):
        change(ws, entry.identity.id, 4, restore_revision=1)
    new_id = str(uuid4())
    change(ws, new_id, 0, 'new', kind='pcl', role='same_winding')
    assert ws.catalog.entry(new_id).identity.collection_id == 8
    assert resident.members[entry.identity.id]['deleted']


def test_application_failure_stays_editable_and_cannot_commit(workspace):
    ws, resident = workspace
    entry = ws.catalog.entries()[0]
    resident.fail = True
    assert not change(ws, entry.identity.id, 1, 'invalid')['applied']
    with pytest.raises(ApiError, match='did not apply'):
        commit(ws, entry.identity.id, 2)
    assert ws.catalog.entry(entry.identity.id).persisted == 1
    resident.fail = False
    assert change(ws, entry.identity.id, 2, 'fixed')['applied']
    commit(ws, entry.identity.id, 3)


def test_second_service_and_client_cannot_claim_or_mutate(workspace, tmp_path):
    ws, _ = workspace
    with pytest.raises(ApiError, match='does not own'):
        ws.claim('other-client', 'other-claim')
    other = EditingWorkspace(ws.dataset, tmp_path / 'other', {}, lambda: Resident())
    try:
        with pytest.raises(ApiError, match='Another service'):
            other.claim('other-client', 'claim')
    finally:
        other.close()
    assert ws.claim(TOKEN, 'reconnect')['workspace_id'] == ws.id


def test_review_is_scoped_and_never_silently_refreshes_conflicting_base(workspace):
    ws, _ = workspace
    entry = ws.catalog.entries()[0]
    change(ws, entry.identity.id, 1, 'local')
    target = Path(entry.identity.source)
    document = json.loads(target.read_text())
    document['collections']['7']['name'] = 'external'
    target.write_text(json.dumps(document))
    with pytest.raises(ApiError) as error:
        commit(ws, entry.identity.id, 2)
    conflict = error.value.payload['conflicts'][0]
    request = {'command_id': 'review', 'id': entry.identity.id, 'expected_revision': 2,
               'action': 'apply_local_after_review', 'review_token': conflict['review_token']}
    ws.resolve_conflict(TOKEN, request)
    assert json.loads(target.read_text())['collections']['7']['name'] == 'external'
    document['collections']['7']['name'] = 'external-again'
    target.write_text(json.dumps(document))
    with pytest.raises(ApiError) as error:
        commit(ws, entry.identity.id, 2)
    request.update(command_id='use-current', action='use_current',
                   review_token=error.value.payload['conflicts'][0]['review_token'])
    ws.resolve_conflict(TOKEN, request)
    current = ws.catalog.entry(entry.identity.id)
    assert (current.accepted, current.applied, current.persisted) == (3, 3, 3)
    assert ws._document(current.current)['collections']['7']['name'] == 'external-again'


def test_discard_restores_current_dataset_in_one_resident_batch(workspace):
    ws, resident = workspace
    entry = ws.catalog.entries()[0]
    change(ws, entry.identity.id, 1, 'local')
    new_id = str(uuid4())
    change(ws, new_id, 0, 'addition', kind='pcl', role='same_winding')
    target = Path(entry.identity.source)
    before = target.read_bytes()
    result = ws.discard(TOKEN, {'command_id': 'discard', 'revisions': [
        {'id': entry.identity.id, 'revision': 2}, {'id': new_id, 'revision': 1}]})
    assert result['discarded']
    assert target.read_bytes() == before
    assert len(resident.calls[-1][1]) == 2
    assert ws.catalog.entry(new_id).deleted
    assert all(e.accepted == e.persisted for e in ws.catalog.entries())


def test_timeout_reuses_captured_influence_and_accepted_revision(workspace):
    editing, resident = workspace
    settings = {'influence_radius': 12}
    editing.influence = lambda: settings
    seen = []
    original = resident.apply_input_changes
    def interrupted(command, records, influence_config=None):
        seen.append(copy.deepcopy(influence_config))
        if len(seen) == 1:
            raise TimeoutError('lost application response')
        return original(command, records, influence_config)
    resident.apply_input_changes = interrupted
    input_id = str(uuid4())
    request = {'command_id': 'captured-influence', 'changes': [{'id': input_id,
        'kind': 'pcl', 'role': 'same_winding', 'expected_revision': 0,
        'upload_id': upload(editing, 'new')}]}
    with pytest.raises(TimeoutError):
        editing.change(TOKEN, request)
    settings['influence_radius'] = 99
    assert editing.change(TOKEN, request)['applied']
    assert seen == [{'influence_radius': 12}, {'influence_radius': 12}]
    assert editing.catalog.entry(input_id).accepted == 1


def test_rebuild_preserves_desired_inputs_uploads_and_failed_preparation(workspace):
    editing, resident = workspace
    baseline = editing.catalog.entries()[0]
    new_id = str(uuid4())
    transfer = upload(editing, 'retained')
    editing.change(TOKEN, {'command_id': 'new', 'changes': [{'id': new_id,
        'kind': 'pcl', 'role': 'same_winding', 'expected_revision': 0, 'upload_id': transfer}]})
    change(editing, baseline.identity.id, 1, deleted=True)
    desired = editing.catalog.status()
    resident.fail = True
    assert not editing.replay_resident('generation-two')['applied']
    assert editing.resident_generation is None
    resident.fail = False
    assert editing.replay_resident('generation-two')['applied']
    assert editing.uploads.get(transfer).record is not None
    records = resident.calls[-1][1]
    assert {r['id'] for r in records} == {new_id, baseline.identity.id}
    assert next(r for r in records if r['id'] == baseline.identity.id)['deleted']
    calls = len(resident.calls)
    assert editing.replay_resident('generation-two')['applied']
    assert len(resident.calls) == calls
    assert [(e['id'], e['accepted_revision'], e['persisted_revision']) for e in editing.catalog.status()] == [
        (e['id'], e['accepted_revision'], e['persisted_revision']) for e in desired]


def test_use_current_timeout_does_not_accept_twice(workspace):
    editing, resident = workspace
    entry = editing.catalog.entries()[0]
    target = Path(entry.identity.source)
    document = json.loads(target.read_text())
    document['collections']['7']['name'] = 'external'
    target.write_text(json.dumps(document))
    current, _ = editing._external(entry.identity)
    request = {'command_id': 'review-timeout', 'id': entry.identity.id,
        'expected_revision': 1, 'action': 'use_current', 'review_token': editing._review_token(current)}
    original = resident.apply_input_changes
    calls = 0
    def interrupted(command, records, influence_config=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError('lost response')
        return original(command, records, influence_config)
    resident.apply_input_changes = interrupted
    with pytest.raises(TimeoutError):
        editing.resolve_conflict(TOKEN, request)
    assert editing.resolve_conflict(TOKEN, request)['resolved']
    assert editing.catalog.entry(entry.identity.id).accepted == 2
    assert editing.catalog.entry(entry.identity.id).persisted == 2


def test_release_reply_can_be_reconciled_after_lock_is_released(workspace):
    editing, _ = workspace
    assert editing.release(TOKEN, 'release')['released']
    assert editing.release(TOKEN, 'release')['released']
    with pytest.raises(ApiError):
        editing.release('different-owner', 'release')


def test_reconnect_refreshes_clean_targets_and_preserves_dirty_conflicts(workspace):
    editing, resident = workspace
    entry = editing.catalog.entries()[0]
    target = Path(entry.identity.source)
    document = json.loads(target.read_text())
    document['collections']['7']['name'] = 'external'
    document['collections']['100'] = {'name': 'unrelated', 'points': {}}
    target.write_text(json.dumps(document))
    editing.claim(TOKEN, 'reconnect')
    refreshed = editing.catalog.entry(entry.identity.id)
    assert refreshed.accepted == refreshed.applied == refreshed.persisted == 2
    assert editing._collection(refreshed.current, refreshed.identity)['name'] == 'external'
    with pytest.raises(ApiError) as caught:
        change(editing, entry.identity.id, 1, 'older-local')
    conflict, = caught.value.payload['conflicts']
    assert conflict['review_token'] and conflict['expected_revision'] == 2
    assert conflict['current']['name'] == 'external'
    calls = len(resident.calls)
    editing.claim(TOKEN, 'second-reconnect')
    assert len(resident.calls) == calls
    new_id = str(uuid4())
    change(editing, new_id, 0, 'new', kind='pcl', role='same_winding')
    assert editing.catalog.entry(new_id).identity.collection_id == 101
    assert json.loads(target.read_text()) == document
