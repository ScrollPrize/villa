"""Shared local service paths for the manager and its clients."""

from pathlib import Path
import os
import socket
import stat


GLOBAL_SOCKET_PATH = Path("/run/villa-gpu-manager/manager.sock")


def _listening(path: Path) -> bool:
    try:
        if not stat.S_ISSOCK(path.lstat().st_mode):
            return False
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.1)
            probe.connect(str(path))
        return True
    except OSError:
        return False


def _writable_location(path: Path) -> bool:
    try:
        parent = path.parent
        # Check without creating directories merely to choose a default.
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        return parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)
    except OSError:
        return False


def default_socket_path() -> str:
    """Prefer the global service/location, otherwise use the private default.

    A connect-only probe creates no registration and claims no resources. Normal
    client handshake still validates the protocol and UID authorization; failures
    after selecting a socket must not trigger a second-manager fallback.
    """
    if _listening(GLOBAL_SOCKET_PATH) or _writable_location(GLOBAL_SOCKET_PATH):
        return str(GLOBAL_SOCKET_PATH)
    return str(Path.home() / ".local/run/villa-gpu-manager/manager.sock")
