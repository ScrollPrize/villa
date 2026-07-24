"""Stateful fake VC3D bridge used by the MCP integration tests."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

# A minimal valid 1x1 opaque-red PNG (matches the on-the-wire "base64" the real
# screenshot.capture bridge method returns for an inline, no-filePath capture).
TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
    b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG_BYTES).decode("ascii")


class FakeAgentBridgeServer:
    """Minimal stand-in for AgentBridgeServer: AF_UNIX, newline-delimited
    JSON-RPC 2.0, a couple of canned methods, and stateful jobs whose
    job.status progress history grows and then reaches a terminal state."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict] = []
        # job_id -> kind, so job.status can echo the right kind for each flow.
        self._job_kinds: dict[str, str] = {}
        # job_id -> live record answered by job.status (state grows to terminal).
        self._jobs: dict[str, dict] = {}
        # Segments that have been materialized (fetched). "seg-ph" starts as an
        # unfetched open-data placeholder; fetching it adds it here so a
        # subsequent segments.activate succeeds -- the fetch+retry compose.
        self._fetched: set[str] = set()
        # Per-plane rotation state for viewer.rotate (mirrors the C++ controller's
        # _segXZRotationDeg / _segYZRotationDeg members).
        self._rotation: dict[str, float] = {}
        # Axis-aligned slice mode (viewer.set_axis_aligned_slices / state.get).
        self._axis_enabled: bool = False
        # Same-wrap annotation state: the mode checkbox and whether a
        # chunked viewer holds an uncommitted preview (seeded by shift-click).
        self._wrap_enabled: bool = False
        self._wrap_has_preview: bool = False
        # segmentation.save: when True, model the "nothing to flush" idle response
        # (jobId:null); when False, model a running "autosave" job that succeeds.
        self.save_idle: bool = False
        # Count of accepted client connections (for the connect-race test).
        self.connections: int = 0
        # When False, jobs stay "running" forever (never terminal), so a
        # wait: true call polls until the cap or the peer drops.
        self.finish_jobs: bool = True
        # When False, jobs change state without emitting progress. This models
        # notification loss and terminal jobs that never produced output.
        self.emit_job_progress: bool = True
        # Re-send the latest update immediately before job.status replies to
        # exercise subscription/snapshot ordering and sequence deduplication.
        self.rebroadcast_latest_on_status: bool = False
        # When True, screenshot.capture returns a null/absent base64 even for an
        # inline (no filePath) capture -- models the "bridge unexpectedly gave
        # us no image bytes" edge the tool must degrade gracefully on.
        self.screenshot_drop_base64: bool = False
        # Global viewer render settings (viewer.get_render_settings /
        # viewer.set_render_settings). set merges its params over this and echoes
        # the full merged object back, so a wrapper round-trip is meaningful.
        self._render_settings: dict = {
            "intersectionOpacity": 0.5,
            "intersectionThickness": 1.0,
            "overlayOpacity": 0.25,
            "intersectionMaxSurfaces": 8,
            "planeIntersectionLinesVisible": True,
            "showSurfaceNormals": False,
            "showDirectionHints": False,
            "surfaceOverlayEnabled": True,
            "highlightedSurfaceIds": [],
        }

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()

        writers = list(self._writers)
        for w in writers:
            w.close()
        if writers:
            await asyncio.gather(
                *(w.wait_closed() for w in writers), return_exceptions=True
            )

        if self._server is not None:
            # Bound wait_closed(): on some CPython versions Server.wait_closed()
            # can block indefinitely if a client opened and immediately dropped a
            # connection (e.g. connect()'s abort-when-closed path) so it was
            # never fully accepted. That's a harness/stdlib quirk, not product
            # behavior; don't let it hang teardown.
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                await self._handle_line(line, writer)
        finally:
            if writer in self._writers:
                self._writers.remove(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _handle_line(self, raw: bytes, writer: asyncio.StreamWriter) -> None:
        msg = json.loads(raw.decode("utf-8"))
        self.received_requests.append(msg)
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "ping":
            await self._reply(
                writer,
                req_id,
                result={
                    "pong": True,
                    "pid": 4242,
                    "version": "test-0.0",
                    "protocolVersion": 1,
                },
            )
        elif method == "state.get":
            await self._reply(writer, req_id, result={"vpkg": None, "volume": None})
        elif method == "segmentation.grow":
            job_id = "job-1"
            self._job_kinds[job_id] = "segmentation.grow"
            await self._reply(writer, req_id, result={"jobId": job_id, "kind": "segmentation.grow"})
            self._start_job(job_id)
        elif method == "segmentation.save":
            if self.save_idle:
                await self._reply(
                    writer, req_id,
                    result={"jobId": None, "kind": "segmentation.save",
                            "state": "idle", "pending": False,
                            "saveInProgress": False, "dirtyAfterSave": False},
                )
            else:
                job_id = "job-4"
                self._job_kinds[job_id] = "segmentation.save"
                await self._reply(
                    writer, req_id,
                    result={"jobId": job_id, "source": "autosave",
                            "kind": "segmentation.save", "state": "running",
                            "label": "Save segment"},
                )
                self._start_job(job_id)
        elif method == "tracer.run_trace":
            job_id = "job-5"
            self._job_kinds[job_id] = "tracer.run_trace"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "tracer.run_trace",
                        "source": "tool", "outputDir": "/tmp/traces"},
            )
            self._start_job(job_id)
        elif method == "atlas.search_start":
            job_id = "job-6"
            self._job_kinds[job_id] = "atlas.fiber_search"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "atlas.fiber_search", "source": "atlas"},
            )
            self._start_job(job_id)
        elif method == "never_replies":
            return  # deliberately no response (cancel / disconnect tests)
        elif method == "lasagna.start_optimization":
            job_id = "job-2"
            self._job_kinds[job_id] = "lasagna.optimize"
            await self._reply(
                writer,
                req_id,
                result={"jobId": job_id, "kind": "lasagna.optimize", "source": "lasagna"},
            )
            self._start_job(job_id)
        elif method == "segments.activate":
            seg = params.get("segmentId")
            if seg == "seg-ready" or seg in self._fetched:
                await self._reply(
                    writer, req_id,
                    result={"activated": True,
                            "segment": {"id": seg, "loaded": True, "active": True},
                            "previousSegmentId": None, "alreadyActive": False},
                )
            elif seg == "seg-ph":
                await self._reply(
                    writer, req_id,
                    error={"code": -32005, "message": "Segment could not be activated",
                           "data": {"detail": f"segment {seg} is an open-data "
                                              "placeholder; fetch it first"}},
                )
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "segments.fetch":
            seg = params.get("segmentId")
            if seg == "seg-ready" or seg in self._fetched:
                await self._reply(
                    writer, req_id,
                    result={"fetched": True, "alreadyMaterialized": True,
                            "segment": {"id": seg, "placeholder": False}},
                )
            elif seg == "seg-ph":
                job_id = "job-3"
                self._job_kinds[job_id] = "segments.fetch"
                self._fetched.add(seg)  # materialized once the fetch is requested
                await self._reply(
                    writer, req_id,
                    result={"jobId": job_id, "source": "catalog", "fetched": False,
                            "alreadyMaterialized": False,
                            "segment": {"id": seg, "placeholder": True}},
                )
                self._start_job(job_id)
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "viewer.rotate":
            plane = params.get("plane")
            norm = {"xz": "seg xz", "yz": "seg yz",
                    "seg xz": "seg xz", "seg yz": "seg yz"}.get(plane)
            if norm is None:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "unknown plane",
                           "data": {"param": "plane", "value": plane}},
                )
            else:
                prev = self._rotation.get(norm, 0.0)
                deg = params.get("degrees", 0.0)
                target = prev + deg if params.get("relative", True) else deg
                # match the C++ remainder(., 360) normalization
                target = target - 360.0 * round(target / 360.0)
                self._rotation[norm] = target
                await self._reply(
                    writer, req_id,
                    result={"plane": norm, "degrees": target,
                            "previousDegrees": prev,
                            "relative": params.get("relative", True)},
                )
        elif method == "viewer.set_axis_aligned_slices":
            enabled = params.get("enabled")
            if not isinstance(enabled, bool):
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "enabled (bool) is required",
                           "data": {"param": "enabled"}},
                )
            else:
                self._axis_enabled = enabled  # idempotent set, like setChecked
                await self._reply(writer, req_id, result={"enabled": self._axis_enabled})
        elif method == "wrap_annotation.set_mode":
            self._wrap_enabled = bool(params.get("enabled"))
            await self._reply(writer, req_id, result={"enabled": self._wrap_enabled})
        elif method == "wrap_annotation.commit":
            if not self._wrap_enabled:
                await self._reply(
                    writer, req_id,
                    error={"code": -32002,
                           "message": "same-wrap annotation mode is not enabled"},
                )
            else:
                had_preview = self._wrap_has_preview
                committed = self._wrap_enabled and self._wrap_has_preview
                self._wrap_has_preview = False  # commit consumes the preview
                await self._reply(
                    writer, req_id,
                    result={"committed": committed, "hadPreview": had_preview},
                )
        elif method == "wrap_annotation.undo":
            undone = self._wrap_has_preview
            self._wrap_has_preview = False
            await self._reply(writer, req_id, result={"undone": undone})
        elif method == "screenshot.capture":
            # Mirror the real bridge contract: an inline capture (no filePath)
            # returns the PNG as base64; a to-disk capture (filePath set) writes
            # the file and returns a dict with base64 null plus the path.
            file_path = params.get("filePath")
            if file_path is not None:
                await self._reply(
                    writer, req_id,
                    result={"target": params.get("target"), "filePath": file_path,
                            "base64": None, "width": 1, "height": 1},
                )
            else:
                b64 = None if self.screenshot_drop_base64 else TINY_PNG_B64
                await self._reply(
                    writer, req_id,
                    result={"target": params.get("target"), "filePath": None,
                            "base64": b64, "width": 1, "height": 1},
                )
        elif method == "job.status":
            job_id = params.get("jobId", "job-1")
            rec = self._jobs.get(job_id)
            if rec is None:
                rec = {
                    "jobId": job_id,
                    "kind": self._job_kinds.get(job_id, "segmentation.grow"),
                    "state": "succeeded", "message": "finished",
                    "success": True, "consoleTail": [], "progressHistory": [],
                }
            if self.rebroadcast_latest_on_status and rec["progressHistory"]:
                await self._broadcast(rec["progressHistory"][-1])
            await self._reply(
                writer,
                req_id,
                result={
                    "jobId": rec["jobId"],
                    "kind": rec["kind"],
                    "label": "Grow",
                    "state": rec["state"],
                    "message": rec.get("message"),
                    "success": rec.get("success"),
                    "outputPath": rec.get("outputPath"),
                    "result": rec.get("result"),
                    "consoleTail": list(rec["consoleTail"]),
                    "progressHistory": list(rec["progressHistory"]),
                },
            )
        elif method == "job.cancel":
            job_id = params.get("jobId")
            source = params.get("source")
            if not job_id and not source:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "jobId or source is required",
                           "data": {"param": "jobId"}},
                )
            else:
                resolved = job_id or "job-for-" + str(source)
                await self._reply(
                    writer, req_id,
                    result={"cancelRequested": True, "jobId": resolved,
                            "source": source or "growth",
                            "kind": self._job_kinds.get(resolved, "segmentation.grow")},
                )
        elif method == "project.create":
            path = params.get("path")
            volume = params.get("volume")
            if not isinstance(path, str):
                await self._reply(
                    writer,
                    req_id,
                    error={
                        "code": -32602,
                        "message": "path has the wrong type",
                        "data": {"param": "path"},
                    },
                )
            else:
                if not path.lower().endswith(".volpkg.json"):
                    path += ".volpkg.json"
                name = params.get(
                    "name",
                    Path(path).name[:-len(".volpkg.json")] or "Untitled",
                )
                await self._reply(
                    writer,
                    req_id,
                    result={"path": path, "name": name, "volume": volume},
                )
        elif method == "volume.attach":
            job_id = "job-7"
            self._job_kinds[job_id] = "volume.attach"
            await self._reply(
                writer,
                req_id,
                result={
                    "jobId": job_id,
                    "kind": "volume.attach",
                    "source": "volume",
                    "location": params.get("location"),
                },
            )
            self._start_job(job_id)
            self._jobs[job_id]["result"] = {
                "attached": True,
                "alreadyAttached": False,
                "volumeId": "vol-c",
                "location": params.get("location"),
                "projectPath": "/tmp/project.volpkg.json",
            }
        elif method == "volume.list":
            await self._reply(
                writer, req_id,
                result={"volumeIds": ["vol-a", "vol-b"], "currentVolumeId": "vol-a"},
            )
        elif method == "segments.delete":
            seg = params.get("segmentId")
            if params.get("confirm") is not True:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "destructive; pass confirm=true",
                           "data": {"param": "confirm",
                                    "reason": "destructive; pass confirm=true"}},
                )
            elif seg in ("seg-ready", "seg-ph") or seg in self._fetched:
                self._fetched.discard(seg)
                await self._reply(writer, req_id, result={"deleted": [seg]})
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "segments.rename":
            seg = params.get("segmentId")
            new_name = params.get("newName")
            await self._reply(
                writer, req_id, result={"oldId": seg, "newId": new_name},
            )
        elif method == "viewer.get_render_settings":
            await self._reply(writer, req_id, result=dict(self._render_settings))
        elif method == "viewer.set_render_settings":
            # Merge the provided subset over current state and echo the full set,
            # mirroring the real "reuse get logic" reply contract.
            self._render_settings.update(params)
            await self._reply(writer, req_id, result=dict(self._render_settings))
        elif method == "will_error":
            await self._reply(
                writer,
                req_id,
                error={
                    "code": -32003,
                    "message": "INVALID_COORDINATES",
                    "data": {"point": {"x": 1.0, "y": 2.0, "z": 3.0}},
                },
            )
        else:
            await self._reply(writer, req_id, error={"code": -32601, "message": "Method not found"})

    def _start_job(self, job_id: str) -> None:
        """Create a running job record (synchronously, so job.status can be
        polled immediately) and spawn its simulated progression."""
        kind = self._job_kinds.get(job_id, "segmentation.grow")
        self._jobs[job_id] = {
            "jobId": job_id, "kind": kind, "state": "running",
            "message": "starting", "success": None,
            "outputPath": None, "consoleTail": [], "progressHistory": [],
            "nextSeq": 1,
        }
        asyncio.create_task(self._simulate_job(job_id))

    async def _emit_job_progress(self, rec: dict, **fields) -> None:
        update = {
            "jobId": rec["jobId"],
            "kind": rec["kind"],
            "seq": rec["nextSeq"],
            **fields,
        }
        rec["nextSeq"] += 1
        rec["progressHistory"].append(update)
        rec["progressHistory"] = rec["progressHistory"][-64:]
        await self._broadcast(update)

    async def _simulate_job(self, job_id: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        kind = rec["kind"]
        for text in ("line A", "line B"):
            await asyncio.sleep(0.02)
            rec["consoleTail"].append(text)
            if self.emit_job_progress:
                await self._emit_job_progress(
                    rec, phase="output", message=text
                )
        if not self.finish_jobs:
            return  # stays "running": exercises the wait cap / disconnect
        await asyncio.sleep(0.02)
        rec.update(state="succeeded", success=True, message="finished")
        if self.emit_job_progress:
            await self._emit_job_progress(
                rec, phase="finished", success=True, message="finished",
                outputPath=None, result=rec.get("result"),
            )

    async def _reply(self, writer: asyncio.StreamWriter, req_id, *, result=None, error=None) -> None:
        msg: dict = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        await self._send(writer, msg)

    async def _broadcast(self, params: dict) -> None:
        msg = {"jsonrpc": "2.0", "method": "job.progress", "params": params}
        for w in list(self._writers):
            await self._send(w, msg)

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()
