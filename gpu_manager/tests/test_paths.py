import os
import socket
import pytest

from villa_gpu_manager import paths


def test_private_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_listening", lambda _: False)
    monkeypatch.setattr(paths, "_writable_location", lambda _: False)
    assert paths.default_socket_path() == str(tmp_path / ".local/run/villa-gpu-manager/manager.sock")


def test_writable_global_without_creating_directory(monkeypatch, tmp_path):
    target = tmp_path / "not-created" / "manager.sock"
    monkeypatch.setattr(paths, "GLOBAL_SOCKET_PATH", target)
    assert paths.default_socket_path() == str(target)
    assert not target.parent.exists()


def test_live_global_even_when_directory_not_writable(monkeypatch, tmp_path):
    target = tmp_path / "manager.sock"
    monkeypatch.setattr(paths, "GLOBAL_SOCKET_PATH", target)
    monkeypatch.setattr(paths, "_writable_location", lambda _: False)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(target))
        server.listen()
        assert paths.default_socket_path() == str(target)
        connection, _ = server.accept()
        with connection:
            assert connection.recv(1) == b""  # No registration/protocol mutation.
    assert not paths._listening(target)  # Stale socket is not a live manager.


def test_non_socket_is_not_a_listener(tmp_path):
    target = tmp_path / "file"
    target.touch()
    assert not paths._listening(target)


def test_location_requires_write_and_search(monkeypatch, tmp_path):
    calls = []
    def access(path, mode):
        calls.append((path, mode))
        return False
    monkeypatch.setattr(paths.os, "access", access)
    assert not paths._writable_location(tmp_path / "missing" / "manager.sock")
    assert calls == [(tmp_path, os.W_OK | os.X_OK)]


def test_inaccessible_ancestor_is_not_writable(monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise PermissionError("inaccessible ancestor")
    monkeypatch.setattr(paths.Path, "exists", denied)
    assert not paths._writable_location(tmp_path / "manager.sock")


def test_explicit_cli_path_and_help_do_not_probe_default(monkeypatch):
    from villa_gpu_manager import cli
    monkeypatch.setattr(cli, "default_socket_path", lambda: pytest.fail("unexpected probe"))
    class Selected(Exception):
        pass
    def client(path):
        assert path == "/explicit.sock"
        raise Selected
    monkeypatch.setattr(cli, "Client", client)
    with pytest.raises(Selected):
        cli.main(["--socket", "/explicit.sock", "status"])
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
