"""Asyncio stream connection to the named pipe used by QLocalServer on Windows."""

from __future__ import annotations

import asyncio


async def open_named_pipe_connection(
    path: str,
    *,
    limit: int,
    connect_timeout: float,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to a Windows named pipe through asyncio's IOCP proactor."""
    loop = asyncio.get_running_loop()
    create_pipe_connection = getattr(loop, "create_pipe_connection", None)
    if create_pipe_connection is None:
        raise OSError("Windows named pipes require an asyncio proactor event loop")

    reader = asyncio.StreamReader(limit=limit)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await asyncio.wait_for(
        create_pipe_connection(lambda: protocol, path),
        timeout=connect_timeout,
    )
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer
