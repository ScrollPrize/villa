"""Small extensible AF_UNIX JSON-RPC server for domain-tool tests."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class FakeBridgeServer:
    """Record newline-delimited JSON-RPC requests and delegate their replies."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict[str, Any]] = []

    @property
    def received(self) -> list[dict[str, Any]]:
        return self.received_requests

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        writers = list(self._writers)
        for writer in writers:
            writer.close()
        if writers:
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers),
                return_exceptions=True,
            )
        if self._server is not None:
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.append(writer)
        try:
            while line := await reader.readline():
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line.decode("utf-8"))
                self.received_requests.append(message)
                await self.handle_message(message, writer)
        finally:
            if writer in self._writers:
                self._writers.remove(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def handle_message(
        self, message: dict[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        await self.reply(
            writer,
            message.get("id"),
            error={"code": -32601, "message": "Method not found"},
        )

    async def reply(
        self,
        writer: asyncio.StreamWriter,
        request_id: Any,
        *,
        result: Any = None,
        error: Any = None,
    ) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        message["error" if error is not None else "result"] = (
            error if error is not None else result
        )
        await self.send(writer, message)

    async def send(
        self, writer: asyncio.StreamWriter, message: dict[str, Any]
    ) -> None:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()

    def params_for(self, method: str) -> dict[str, Any]:
        for message in reversed(self.received_requests):
            if message.get("method") == method:
                return message.get("params") or {}
        raise AssertionError(f"no request seen for {method}")


class EchoBridgeServer(FakeBridgeServer):
    async def handle_message(
        self, message: dict[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        await self.reply(
            writer,
            message.get("id"),
            result={
                "echoedMethod": message.get("method"),
                "echoedParams": message.get("params") or {},
            },
        )
