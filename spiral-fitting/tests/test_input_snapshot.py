import errno
from pathlib import Path
import shutil

import pytest

import input_snapshot as snapshot
from input_publication import fingerprint
from service_http import ApiError


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'empty').mkdir(parents=True)
    (root / 'nested').mkdir()
    (root / 'nested' / 'x.tif').write_bytes(bytes(range(256)) * 4096)
    (root / 'meta.json').write_text('{"name":"patch"}')
    return root


@pytest.mark.parametrize('clone', [False, True])
def test_snapshot_fingerprint_and_independent_writes(source, tmp_path, monkeypatch, clone):
    # Exercise both algorithms even on a filesystem without reflinks.
    def clone_file(src, dst):
        if clone:
            shutil.copyfile(src, dst)
        return clone
    monkeypatch.setattr(snapshot, '_clone_file', clone_file)
    expected = fingerprint(source)
    destination = tmp_path / 'snapshot'
    assert snapshot.snapshot_input(source, destination) == expected
    assert fingerprint(destination) == expected
    assert (destination / 'empty').is_dir()
    assert (destination / 'meta.json').stat().st_mtime_ns == (source / 'meta.json').stat().st_mtime_ns
    (source / 'meta.json').write_text('source changed')
    assert (destination / 'meta.json').read_text() == '{"name":"patch"}'
    (destination / 'nested' / 'x.tif').write_bytes(b'destination changed')
    assert (source / 'nested' / 'x.tif').stat().st_size == 256 * 4096


def test_actual_clone_is_independent(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_bytes(b'a' * 8192)
    if not snapshot._clone_file(source, destination):
        assert not destination.exists()
        pytest.skip('This filesystem does not support clones')
    assert source.stat().st_ino != destination.stat().st_ino
    with source.open('r+b') as stream:
        stream.write(b'b')
    assert destination.read_bytes() == b'a' * 8192
    with destination.open('r+b') as stream:
        stream.seek(1)
        stream.write(b'c')
    assert source.read_bytes() == b'b' + b'a' * 8191


@pytest.mark.parametrize('error', [errno.EOPNOTSUPP, errno.EXDEV, errno.ENOTTY])
def test_linux_clone_fallback_removes_empty_destination(tmp_path, monkeypatch, error):
    fcntl = pytest.importorskip('fcntl')
    monkeypatch.setattr(snapshot.sys, 'platform', 'linux')
    def fail(*args):
        raise OSError(error, 'unsupported')
    monkeypatch.setattr(fcntl, 'ioctl', fail)
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_bytes(b'data')
    assert snapshot.snapshot_input(source, destination) == fingerprint(source)
    assert destination.read_bytes() == b'data'


def test_clone_io_error_is_not_hidden(tmp_path, monkeypatch):
    fcntl = pytest.importorskip('fcntl')
    monkeypatch.setattr(snapshot.sys, 'platform', 'linux')
    def fail(*args):
        raise OSError(errno.EIO, 'I/O failure')
    monkeypatch.setattr(fcntl, 'ioctl', fail)
    source = tmp_path / 'source'
    source.write_bytes(b'data')
    with pytest.raises(OSError, match='I/O failure'):
        snapshot.snapshot_input(source, tmp_path / 'destination')


@pytest.mark.parametrize('change', ['bytes', 'added', 'removed', 'empty-directory'])
def test_source_changes_during_snapshot_are_rejected(source, tmp_path, monkeypatch, change):
    monkeypatch.setattr(snapshot, '_clone_file', lambda *_: False)
    original = snapshot._copy_file
    def changing_copy(src, dst):
        digest = original(src, dst)
        if Path(src).name == 'meta.json':
            if change == 'bytes':
                Path(src).write_text('changed')
            elif change == 'added':
                (source / 'added').write_bytes(b'added')
            elif change == 'removed':
                Path(src).unlink()
            else:
                (source / 'new-empty').mkdir()
        return digest
    monkeypatch.setattr(snapshot, '_copy_file', changing_copy)
    with pytest.raises(ApiError, match='changed while taking a snapshot'):
        snapshot.snapshot_input(source, tmp_path / 'snapshot')


def test_corrupt_copy_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'input')
    monkeypatch.setattr(snapshot, '_clone_file', lambda *_: False)
    original = snapshot.sha256_file
    def corrupt_then_hash(path):
        Path(path).write_bytes(b'bad copy')
        return original(path)
    monkeypatch.setattr(snapshot, 'sha256_file', corrupt_then_hash)
    with pytest.raises(ApiError, match='copy failed verification'):
        snapshot.snapshot_input(source, tmp_path / 'destination')


@pytest.mark.parametrize('directory', [False, True])
def test_symlink_inputs_are_rejected(source, tmp_path, directory):
    link = source / 'link'
    link.symlink_to(source / 'nested' if directory else source / 'meta.json',
                    target_is_directory=directory)
    with pytest.raises(ApiError, match='symlink'):
        snapshot.snapshot_input(source, tmp_path / 'destination')


def test_existing_destination_is_not_overwritten(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_bytes(b'new')
    destination.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        snapshot.snapshot_input(source, destination)
    assert destination.read_bytes() == b'keep'


@pytest.mark.parametrize('error', [0, errno.ENOTSUP, errno.EIO])
def test_darwin_clone_call_and_fallback(tmp_path, monkeypatch, error):
    import ctypes
    from types import SimpleNamespace
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_bytes(b'data')
    class Clone:
        def __call__(self, src, dst, flags):
            assert flags == 0
            if error:
                ctypes.set_errno(error)
                return -1
            shutil.copyfile(src, dst)
            return 0
    monkeypatch.setattr(snapshot.sys, 'platform', 'darwin')
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: SimpleNamespace(clonefile=Clone()))
    if error == errno.EIO:
        with pytest.raises(OSError):
            snapshot.snapshot_input(source, destination)
    else:
        assert snapshot.snapshot_input(source, destination) == fingerprint(source)
        assert destination.read_bytes() == b'data'
