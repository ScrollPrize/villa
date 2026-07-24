"""Portable tests for the Windows named-pipe transport."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

from vc3d_mcp import transport
from vc3d_mcp.windows_pipe import open_named_pipe_connection


class EndpointResolutionTest(unittest.TestCase):
    def test_windows_name_becomes_local_named_pipe(self) -> None:
        self.assertEqual(
            transport.resolve_local_endpoint("vc3d-agent-42", platform="nt"),
            r"\\.\pipe\vc3d-agent-42",
        )

    def test_explicit_windows_pipe_is_unchanged(self) -> None:
        endpoint = r"\\.\pipe\vc3d-agent-42"
        self.assertEqual(
            transport.resolve_local_endpoint(endpoint, platform="nt"),
            endpoint,
        )


class NamedPipeConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_connection_uses_proactor_stream_transport(self) -> None:
        transport_mock = mock.Mock()
        loop = mock.Mock()
        loop.create_pipe_connection = mock.AsyncMock(
            return_value=(transport_mock, mock.Mock())
        )
        writer = mock.Mock()

        with (
            mock.patch("asyncio.get_running_loop", return_value=loop),
            mock.patch("asyncio.StreamWriter", return_value=writer) as stream_writer,
        ):
            reader, actual_writer = await open_named_pipe_connection(
                r"\\.\pipe\vc3d-agent-42",
                limit=2048,
                connect_timeout=1.5,
            )

        protocol_factory, path = loop.create_pipe_connection.await_args.args
        protocol = protocol_factory()
        self.assertIsInstance(reader, asyncio.StreamReader)
        self.assertIs(protocol._stream_reader, reader)
        self.assertEqual(path, r"\\.\pipe\vc3d-agent-42")
        self.assertIs(actual_writer, writer)
        stream_writer.assert_called_once_with(transport_mock, protocol, reader, loop)

    async def test_non_proactor_loop_is_rejected(self) -> None:
        class LoopWithoutNamedPipes:
            pass

        with mock.patch(
            "asyncio.get_running_loop",
            return_value=LoopWithoutNamedPipes(),
        ):
            with self.assertRaisesRegex(OSError, "proactor event loop"):
                await open_named_pipe_connection(
                    r"\\.\pipe\vc3d-agent-42",
                    limit=1024,
                    connect_timeout=0.5,
                )

    async def test_transport_routes_pipe_endpoints_to_windows_backend(self) -> None:
        reader = asyncio.StreamReader()
        writer = mock.Mock()
        backend = mock.AsyncMock(return_value=(reader, writer))
        with mock.patch(
            "vc3d_mcp.windows_pipe.open_named_pipe_connection", backend
        ):
            actual = await transport.open_local_connection(
                r"\\.\pipe\vc3d-agent-42",
                limit=2048,
                connect_timeout=1.5,
            )

        self.assertEqual(actual, (reader, writer))
        backend.assert_awaited_once_with(
            r"\\.\pipe\vc3d-agent-42",
            limit=2048,
            connect_timeout=1.5,
        )

    @unittest.skipUnless(os.name == "nt", "requires Windows named pipes")
    async def test_close_completes_while_server_remains_connected(self) -> None:
        loop = asyncio.get_running_loop()
        connected = loop.create_future()
        disconnected = loop.create_future()

        class ServerProtocol(asyncio.Protocol):
            def connection_made(self, pipe_transport) -> None:
                if not connected.done():
                    connected.set_result(pipe_transport)

            def connection_lost(self, exc) -> None:
                if not disconnected.done():
                    disconnected.set_result(exc)

        path = rf"\\.\pipe\vc3d-mcp-test-{os.getpid()}-{id(self)}"
        servers = await loop.start_serving_pipe(ServerProtocol, path)
        try:
            _, writer = await open_named_pipe_connection(
                path,
                limit=1024,
                connect_timeout=2.0,
            )
            await asyncio.wait_for(connected, timeout=2.0)

            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            await asyncio.wait_for(disconnected, timeout=2.0)
        finally:
            for server in servers:
                server.close()
