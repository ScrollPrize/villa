"""Disposable editing storage, teardown barriers and crash reclamation."""

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import pytest
from input_publication import Output, PublicationTransaction, fingerprint
from service_editing import EditingWorkspace
from service_http import ApiError
from spiral_service import ServiceState
from test_service_editing import Resident, upload
from service_fixtures import _attach_fake_session
from workspace_storage import MARKER, reclaim_workspaces


@pytest.fixture
def state(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    state = ServiceState(dataset_root=dataset)
    session = _attach_fake_session(state, output, dataset)
    resident = Resident()
    session.apply_input_changes = resident.apply_input_changes
    state.editing().claim('owner', 'claim')
    yield state
    state.close()


@pytest.mark.parametrize('commit', [False, True])
def test_release_removes_uploads_and_revisions_only(state, commit):
    workspace = state.editing()
    saved = workspace.root.parent.parent / 'saved-fit.pt'
    saved.write_bytes(b'saved fit output')
    input_id = str(uuid4())
    uploaded = upload(workspace, 'draft', kind='fiber', input_id=input_id)
    workspace.change('owner', {'command_id': 'change', 'changes': [
        {'id': input_id, 'kind': 'fiber', 'expected_revision': 0,
         'upload_id': uploaded}]})
    revision, = workspace.catalog.desired()
    target = Path(workspace.catalog.entry(revision.id).identity.source)
    if commit:
        workspace.commit('owner', {'command_id': 'commit', 'revisions': [
            {'id': revision.id, 'revision': revision.number}]})
    expected = target.read_bytes() if commit else None
    artifact = state.input_content_artifact(revision.id, revision.number)['artifact']['id']
    workspace.claim('owner', 'reconnect')
    assert workspace.root.exists()
    assert state.release_editing('owner', 'release') == {'released': True}
    assert not workspace.root.exists()
    assert not workspace.catalog.entries()
    assert saved.read_bytes() == b'saved fit output'
    assert not workspace.uploads.uploads
    assert (target.read_bytes() if target.exists() else None) == expected
    with pytest.raises(ApiError):
        state.artifacts.manifest(artifact)
    fresh = state.editing()
    fresh.claim('owner', 'new-claim')
    assert fresh.id != workspace.id
    assert len(fresh.catalog.entries()) == int(commit)
    assert state.release_editing('owner', 'release') == {'released': True}
    assert fresh.root.exists()
    with pytest.raises(ApiError):
        state.release_editing('wrong-owner', 'release')
    assert fresh.root.exists()


def test_fitter_timeout_retains_files_and_ownership_for_retry(state, monkeypatch):
    workspace = state.editing()
    close = state.session.close
    def timeout():
        raise TimeoutError('still reading inputs')
    monkeypatch.setattr(state.session, 'close', timeout)
    with pytest.raises(TimeoutError):
        state.release_editing('owner', 'release')
    assert workspace.root.exists()
    reclaim_workspaces(workspace.root.parent.parent)
    assert workspace.root.exists()
    with pytest.raises(ApiError):
        state.editing()
    monkeypatch.setattr(state.session, 'close', close)
    state.release_editing('owner', 'release')
    assert not workspace.root.exists()


@pytest.mark.parametrize('user', ['request', 'background', 'mutation'])
def test_close_drains_file_users(state, user):
    workspace = state.editing()
    entered, finish = threading.Event(), threading.Event()
    def work():
        entered.set()
        assert finish.wait(5)
        (workspace.root / 'last-write').write_text('done')
    def request():
        with state.workspace_use():
            work()
    with ThreadPoolExecutor(2) as pool:
        if user == 'background':
            state._start_background(target=work, name='test-construction')
        elif user == 'mutation':
            worker = pool.submit(workspace.coordinator.execute, 'slow', 'test', {}, lambda _: work())
        else:
            worker = pool.submit(request)
        assert entered.wait(5)
        closing = pool.submit(state.release_editing, 'owner', 'release')
        with state._workspace_condition:
            deadline = time.monotonic() + 5
            while not state._workspace_closing:
                remaining = deadline - time.monotonic()
                assert remaining > 0
                state._workspace_condition.wait(remaining)
        assert not closing.done()
        assert workspace.root.exists()
        with pytest.raises(ApiError):
            with state.workspace_use():
                pass
        finish.set()
        closing.result(5)
    assert not workspace.root.exists()


def test_commit_recovery_survives_close(state):
    workspace = state.editing()
    target = workspace.dataset / 'recover.json'
    target.write_bytes(b'original')
    source = workspace.root / 'replacement'
    source.write_bytes(b'replacement')
    transaction = PublicationTransaction.prepare([Output(target, source, fingerprint(target))])
    workspace.transactions['interrupted'] = transaction
    retained = transaction.recovery_paths()
    state.release_editing('owner', 'release')
    assert retained and all(path.exists() for path in retained)
    assert target.read_bytes() == b'original'
    transaction.resume()
    assert target.read_bytes() == b'replacement'
    transaction.release()


def test_subprocess_crash_live_owner_concurrent_reclamation_and_legacy(tmp_path):
    package_root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    output = tmp_path / 'output'
    code = '''
import sys
from service_editing import EditingWorkspace
w = EditingWorkspace(sys.argv[1], sys.argv[2], {}, lambda: None)
w.claim('owner', 'claim')
print(w.root, flush=True)
sys.stdin.read()
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(dataset), str(output)],
                               cwd=package_root, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True)
    try:
        published_root = process.stdout.readline().strip()
        assert published_root, 'workspace subprocess exited before publishing its root'
        root = Path(published_root)
        assert root.is_dir() and root.resolve().is_relative_to(output.resolve())
        legacy = root.parent / 'legacy'
        legacy.mkdir()
        (legacy / 'data').write_text('keep')
        link = root.parent / 'link'
        link.symlink_to(legacy, target_is_directory=True)
        unknown = root.parent / 'unknown'
        unknown.mkdir()
        (unknown / MARKER).write_text('{"version": 999}')
        reclaim_workspaces(output)
        assert root.exists()
        process.kill()
        process.wait(timeout=5)
        command = [sys.executable, '-c',
                   'from workspace_storage import reclaim_workspaces; import sys; reclaim_workspaces(sys.argv[1])',
                   str(output)]
        processes = [subprocess.Popen(command, cwd=package_root) for _ in range(3)]
        assert [p.wait(timeout=10) for p in processes] == [0, 0, 0]
        assert not root.exists()
        assert (legacy / 'data').read_text() == 'keep'
        assert link.is_symlink()
        assert unknown.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_real_scroll_discard_and_reopen(tmp_path):
    source = Path(os.environ.get('SPIRAL_REAL_PCL', '/home/sean/Desktop/spiral_dataset/abs_winding.json'))
    if not source.is_file():
        pytest.skip('Set SPIRAL_REAL_PCL to a real scroll PCL input')
    before = source.read_bytes()
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    target = dataset / 'same_windings.json'
    target.write_bytes(before)
    sources = {'pcl_inputs': [{'path': str(target), 'role': 'same_winding'}]}
    resident = Resident()
    workspace = EditingWorkspace(dataset, tmp_path / 'output', sources, lambda: resident)
    workspace.claim('owner', 'claim')
    baseline = len(workspace.catalog.entries())
    assert baseline
    revision = workspace.catalog.desired()[0]
    workspace.change('owner', {'command_id': 'delete', 'changes': [
        {'id': revision.id, 'expected_revision': revision.number, 'deleted': True}]})
    workspace.close()
    assert not workspace.root.exists()
    fresh = EditingWorkspace(dataset, tmp_path / 'output', sources, lambda: resident)
    try:
        fresh.claim('owner', 'reopen')
        assert len(fresh.catalog.entries()) == baseline
        assert all(r.content is not None for r in fresh.catalog.desired())
        assert target.read_bytes() == before == source.read_bytes()
    finally:
        fresh.close()


def test_shutdown_timeout_does_not_remove_active_mutation(state):
    workspace = state.editing()
    entered, finish = threading.Event(), threading.Event()
    def mutate(_):
        entered.set()
        assert finish.wait(5)
        (workspace.root / 'still-writing').write_bytes(b'draft')
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(workspace.coordinator.execute, 'slow', 'test', {}, mutate)
        assert entered.wait(5)
        try:
            with pytest.raises(TimeoutError):
                workspace.coordinator.shutdown(lambda: pytest.fail('unsafe cleanup'), timeout=0)
            assert workspace.root.exists()
            with pytest.raises(ApiError):
                workspace.coordinator.execute('late', 'test', {}, lambda _: None)
        finally:
            finish.set()
        active.result(5)
    state.release_editing('owner', 'release')
    assert not workspace.root.exists()


def test_close_during_session_construction_drains_installed_fitter(state, monkeypatch):
    import types
    workspace = state.editing()
    entered, finish, closed = threading.Event(), threading.Event(), threading.Event()
    def create(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        (workspace.root / 'construction-write').write_text('done')
        return types.SimpleNamespace(close=closed.set)
    monkeypatch.setitem(sys.modules, 'spiral_runtime', types.SimpleNamespace(create_session=create))
    state._start_background(target=state._build,
        args=(state.session_id, None, None, None, None, None), name='test-build')
    assert entered.wait(5)
    with ThreadPoolExecutor(1) as pool:
        releasing = pool.submit(state.release_editing, 'owner', 'release')
        try:
            with state._workspace_condition:
                assert state._workspace_condition.wait_for(lambda: state._workspace_closing, timeout=5)
            assert workspace.root.exists()
        finally:
            finish.set()
        releasing.result(5)
    assert closed.is_set()
    assert state.session is None
    assert not workspace.root.exists()


def test_partially_published_commit_recovery_survives_close(state, monkeypatch):
    from input_publication import _Publication
    workspace = state.editing()
    targets = [workspace.dataset / name for name in ('first', 'second')]
    source = workspace.root / 'replacement'
    source.write_bytes(b'new')
    for target in targets:
        target.write_bytes(b'old')
    transaction = PublicationTransaction.prepare(
        [Output(target, source, fingerprint(target)) for target in targets])
    publish = _Publication.publish
    def interrupted(publication):
        if publication.output.target == targets[1]:
            raise OSError('interrupted second publication')
        publish(publication)
    monkeypatch.setattr(_Publication, 'publish', interrupted)
    with pytest.raises(OSError):
        transaction.resume()
    workspace.transactions['partial'] = transaction
    assert targets[0].read_bytes() == b'new'
    assert targets[1].read_bytes() == b'old'
    paths = transaction.recovery_paths()
    state.release_editing('owner', 'release')
    assert all(path.exists() for path in paths)
    monkeypatch.setattr(_Publication, 'publish', publish)
    transaction.resume()
    assert all(target.read_bytes() == b'new' for target in targets)
    transaction.release()
