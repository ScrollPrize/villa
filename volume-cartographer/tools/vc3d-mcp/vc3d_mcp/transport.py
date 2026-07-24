"""Platform-local byte streams for the VC3D Agent Bridge."""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"


class BridgeReader(Protocol):
    async def readline(self) -> bytes: ...


class BridgeWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def is_closing(self) -> bool: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


def resolve_local_endpoint(configured: str, platform: str | None = None) -> str:
    """Resolve a QLocalServer name to its platform-native endpoint."""
    platform = os.name if platform is None else platform
    if platform == "nt":
        if configured.lower().startswith(WINDOWS_PIPE_PREFIX.lower()):
            return configured
        return WINDOWS_PIPE_PREFIX + configured.lstrip("\\/")

    if os.path.exists(configured):
        return configured

    candidates: list[str] = []
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        candidates.append(os.path.join(tmpdir.rstrip("/"), configured))
    candidates.append(os.path.join("/tmp", configured))
    return next((path for path in candidates if os.path.exists(path)), configured)


async def open_local_connection(
    endpoint: str,
    *,
    limit: int,
    connect_timeout: float,
) -> tuple[BridgeReader, BridgeWriter]:
    """Open the local transport selected by the resolved endpoint."""
    if endpoint.lower().startswith(WINDOWS_PIPE_PREFIX.lower()):
        from .windows_pipe import open_named_pipe_connection

        return await open_named_pipe_connection(
            endpoint,
            limit=limit,
            connect_timeout=connect_timeout,
        )

    return await asyncio.wait_for(
        asyncio.open_unix_connection(endpoint, limit=limit),
        timeout=connect_timeout,
    )
