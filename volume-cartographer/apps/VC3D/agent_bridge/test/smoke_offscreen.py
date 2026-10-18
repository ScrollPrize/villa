#!/usr/bin/env python3
"""Fixture-free offscreen smoke test for the VC3D Agent Bridge.

Exercises the C2/C3/C4 stabilization work against the REAL VC3D binary with a
temporary local volume package and no display (QT_QPA_PLATFORM=offscreen), so
it is safe to run anywhere the build tree exists:

  C4  strict wire-parameter parsing rejects a present-but-malformed value with
      -32602 + data.param, BEFORE touching viewer/controller state, and the
      bridge keeps answering afterwards (no crash, no state corruption).
  C2  a single request line exceeding the 1 MiB per-client bound gets the
      offending client disconnected (best-effort -32600), while other clients
      remain unaffected.
  C3  a second server launched on the same --agent-bridge-name REFUSES (probes
      the live endpoint first and does not unlink it); the first server stays
      reachable.

Prints a single JSON result object on the last stdout line; exits nonzero if any
check fails. Every socket wait carries a timeout so an offscreen hang cannot
block forever.

    python3 smoke_offscreen.py [--vc3d /path/to/VC3D] [--rpc-timeout SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_client import BridgeClient, BridgeError, parse_handshake_line  # noqa: E402
from vc3d_process import VC3DProcess  # noqa: E402
from contract_probe import probe_invalid_inputs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VC3D_BIN = REPO_ROOT / "build" / "ci-release-gcc" / "bin" / "VC3D"
DESCRIPTION_SNAPSHOT = Path(__file__).resolve().parents[1] / "rpc_description.json"

OFFSCREEN_ENV = {"QT_QPA_PLATFORM": "offscreen"}
RPC_TIMEOUT = 10.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Results:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks[name] = {"pass": bool(ok), "detail": detail}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    @property
    def ok(self) -> bool:
        return all(c["pass"] for c in self.checks.values())


def unique_name(tag: str) -> str:
    return f"vc3d-smoke-{tag}-{os.getpid()}-{int(time.time() * 1000) % 100000}"


def launch(
    binary: str,
    name: str,
    volpkg: str | None = None,
    config_home: str | None = None,
) -> VC3DProcess:
    args = ["--agent-bridge", "--agent-bridge-name", name]
    if volpkg is not None:
        args.extend(["--volpkg", volpkg])
    env = dict(OFFSCREEN_ENV)
    if config_home is not None:
        env["XDG_CONFIG_HOME"] = config_home
    return VC3DProcess(
        binary,
        args,
        env_overrides=env,
    )


def require_running(proc: VC3DProcess, results: Results, context: str) -> None:
    if proc.is_running():
        return

    exit_code = proc.exit_code()
    results.record(
        "process_alive",
        False,
        f"VC3D exited during {context} with code {exit_code}",
    )
    captured = proc.tail_log(8000)
    crash_start = next(
        (i for i, line in enumerate(captured) if "VC3D CRASH REPORT" in line),
        None,
    )
    if crash_start is not None:
        start = max(0, crash_start - 20)
        end = next(
            (
                i for i in range(crash_start + 1, len(captured))
                if captured[i] == "--- BYTES AT PC ---"
            ),
            min(len(captured), crash_start + 250),
        )
        diagnostic_lines = captured[start:end]
    else:
        diagnostic_lines = captured[-120:]

    log(f"VC3D diagnostics after unexpected exit during {context}:")
    for line in diagnostic_lines:
        log(f"[VC3D] {line}")
    raise RuntimeError(
        f"VC3D exited during {context} with code {exit_code}"
    )


def create_test_volume(volume: Path, volume_id: str) -> Path:
    level = volume / "0"
    level.mkdir(parents=True)
    (volume / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (volume / ".zattrs").write_text(json.dumps({}))
    (volume / "meta.json").write_text(json.dumps({
        "type": "vol",
        "uuid": volume_id,
        "name": volume_id,
        "format": "zarr",
        "width": 16,
        "height": 16,
        "slices": 16,
        "voxelsize": 1.0,
        "min": 0.0,
        "max": 255.0,
    }))
    (level / ".zarray").write_text(json.dumps({
        "zarr_format": 2,
        "shape": [16, 16, 16],
        "chunks": [16, 16, 16],
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }))
    return volume


def create_test_segment(segment: Path, segment_id: str) -> Path:
    segment.mkdir(parents=True)
    (segment / "meta.json").write_text(json.dumps({
        "type": "seg",
        "uuid": segment_id,
        "name": segment_id,
        "format": "tifxyz",
    }))
    return segment


def create_test_project(root: Path) -> Path:
    create_test_volume(root / "volumes" / "vol1", "vol1")

    volpkg = root / "smoke.volpkg.json"
    volpkg.write_text(json.dumps({
        "name": "smoke",
        "version": 1,
        "volumes": ["volumes"],
    }))
    return volpkg


def wait_for_job(
    client: BridgeClient,
    job_id: str,
    timeout: float = 20.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _ = client.call(
            "job.status",
            {"jobId": job_id},
            timeout=min(RPC_TIMEOUT, timeout),
        )
        if status.get("state") != "running":
            return status
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def expect_param_error(client: BridgeClient, method: str, params: object,
                       expect_param: str) -> tuple[bool, str]:
    """Calls `method`; expects BridgeError(-32602) with data.param == expect_param."""
    try:
        result, _ = client.call(method, params, timeout=RPC_TIMEOUT)
        return False, f"expected -32602, got result keys={list(result)[:6]}"
    except BridgeError as e:
        if e.code != -32602:
            return False, f"expected code -32602, got {e.code} ({e.message})"
        got = e.data.get("param")
        if got != expect_param:
            return False, f"expected data.param={expect_param!r}, got {got!r}"
        return True, f"code=-32602 param={got!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"


def expect_prevalidation_accepts(
    client: BridgeClient,
    method: str,
    params: object,
) -> tuple[bool, str]:
    try:
        client.call(method, params, timeout=RPC_TIMEOUT)
        return True, "request passed descriptor validation"
    except BridgeError as error:
        return (
            error.code != -32602,
            f"request reached handler precondition code={error.code}",
        )
    except Exception as error:  # noqa: BLE001
        return False, f"unexpected {type(error).__name__}: {error}"


def check_rpc_describe(
    client: BridgeClient,
    results: Results,
    update_snapshot: bool = False,
) -> None:
    expected = {
        "viewer.center_on_point",
        "viewer.zoom",
        "viewer.rotate",
        "viewer.set_axis_aligned_slices",
        "viewer.get_render_settings",
        "viewer.set_render_settings",
        "viewer.get_overlay",
        "viewer.set_overlay",
        "viewer.list_overlay_volumes",
        "viewer.set_intersects",
    }
    try:
        description, _ = client.call(
            "rpc.describe",
            {"prefix": "viewer."},
            timeout=RPC_TIMEOUT,
        )
        methods = description.get("methods", {})
        coverage = description.get("coverage", {})
        factor = (
            methods.get("viewer.zoom", {})
            .get("params", {})
            .get("properties", {})
            .get("factor", {})
        )
        surface_ids = (
            methods.get("viewer.set_intersects", {})
            .get("params", {})
            .get("properties", {})
            .get("surfaceIds", {})
        )
        ok = (
            set(methods) == expected
            and description.get("undocumented") == []
            and coverage.get("described") == len(expected)
            and coverage.get("registered") == len(expected)
            and coverage.get("complete") is True
            and factor.get("type") == "number"
            and factor.get("exclusiveMinimum") == 0
            and surface_ids.get("items", {}).get("type") == "string"
        )
        results.record(
            "rpc_describe_viewer",
            ok,
            f"methods={len(methods)} coverage={coverage}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "rpc_describe_viewer",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        description, _ = client.call("rpc.describe", {}, timeout=RPC_TIMEOUT)
        snapshot = {"methods": description.get("methods", {})}
        coverage = description.get("coverage", {})
        complete = (
            description.get("undocumented") == []
            and coverage.get("described") == 123
            and coverage.get("registered") == 123
            and coverage.get("complete") is True
            and len(snapshot["methods"]) == 123
        )
        rendered = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        if update_snapshot:
            DESCRIPTION_SNAPSHOT.write_text(rendered, encoding="utf-8")
        expected = DESCRIPTION_SNAPSHOT.read_text(encoding="utf-8")
        results.record(
            "rpc_description_snapshot",
            complete and rendered == expected,
            f"methods={len(snapshot['methods'])} coverage={coverage}"
            + (" snapshot updated" if update_snapshot else ""),
        )
        report = probe_invalid_inputs(client, snapshot)
        results.record(
            "described_invalid_inputs",
            report.ok,
            report.detail,
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "rpc_description_snapshot",
            False,
            f"{type(error).__name__}: {error}",
        )

    ok, detail = expect_param_error(
        client,
        "rpc.describe",
        {"prefix": 7},
        "prefix",
    )
    results.record("rpc_describe_prefix_type", ok, detail)

    ok, detail = expect_param_error(
        client,
        "fiber.open",
        {"fiberId": "1", "span": [-1, 0]},
        "span",
    )
    results.record("fiber_span_non_negative", ok, detail)

    ok, detail = expect_param_error(
        client,
        "render.tifxyz",
        {
            "segmentId": "probe",
            "outputFormat": "zarr",
            "scale": 1.0e300,
        },
        "scale",
    )
    results.record("render_scale_float_range", ok, detail)

    ok, detail = expect_prevalidation_accepts(
        client,
        "atlas.search_start",
        {"requiredTags": None, "excludedTags": None},
    )
    results.record("atlas_null_tag_filters", ok, detail)


def check_viewer_normalization(client: BridgeClient, results: Results) -> None:
    failures: list[str] = []

    def expect(result: dict, path: str, expected: int | float) -> None:
        actual: object = result
        for part in path.split("."):
            if not isinstance(actual, dict) or part not in actual:
                failures.append(f"{path}: missing from result")
                return
            actual = actual[part]
        if actual != expected:
            failures.append(f"{path}: expected {expected}, got {actual!r}")

    try:
        render, _ = client.call(
            "viewer.set_render_settings",
            {
                "intersectionOpacity": 7.5,
                "intersectionThickness": -7.5,
                "overlayOpacity": -7.5,
                "intersectionMaxSurfaces": -7,
                "volumeWindow": {"low": 10, "high": 5},
                "normalArrowLengthScale": 7.5,
                "normalMaxArrows": 107,
                "samplingStride": -7,
                "zScrollSensitivity": 107,
            },
            timeout=RPC_TIMEOUT,
        )
        for path, expected in {
            "intersectionOpacity": 1,
            "intersectionThickness": 0,
            "overlayOpacity": 0,
            "intersectionMaxSurfaces": 0,
            "volumeWindow.low": 10,
            "volumeWindow.high": 11,
            "normalArrowLengthScale": 2,
            "normalMaxArrows": 100,
            "samplingStride": 1,
            "zScrollSensitivity": 100,
        }.items():
            expect(render, path, expected)

        overlay, _ = client.call(
            "viewer.set_overlay",
            {
                "opacity": 7.5,
                "threshold": 300,
                "maxDisplayedResolution": 12,
                "composite": {"layersFront": -7, "layersBehind": 100},
            },
            timeout=RPC_TIMEOUT,
        )
        for path, expected in {
            "opacity": 1,
            "threshold": 255,
            "maxDisplayedResolution": 5,
            "composite.layersFront": 0,
            "composite.layersBehind": 64,
        }.items():
            expect(overlay, path, expected)

        overlay, _ = client.call(
            "viewer.set_overlay",
            {"window": {"low": 10, "high": 5}},
            timeout=RPC_TIMEOUT,
        )
        expect(overlay, "windowLow", 10)
        expect(overlay, "windowHigh", 11)

        overlay, _ = client.call(
            "viewer.set_overlay",
            {"colormap": ""},
            timeout=RPC_TIMEOUT,
        )
        expect(overlay, "colormap", "")
        overlay, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        expect(overlay, "colormap", "")
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")

    results.record(
        "viewer_normalization",
        not failures,
        "representative clamp/range cases passed"
        if not failures else "; ".join(failures[:4]),
    )


def check_canvas_normalization(client: BridgeClient, results: Results) -> None:
    try:
        result, _ = client.call(
            "canvas.drag",
            {
                "from": {"x": 1, "y": 1, "z": 1},
                "to": {"x": 1, "y": 1, "z": 1},
                "steps": 300,
            },
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "canvas_drag_step_clamp",
            result.get("steps") == 256,
            f"steps={result.get('steps')}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "canvas_drag_step_clamp",
            False,
            f"{type(error).__name__}: {error}",
        )


def check_viewer_registry(client: BridgeClient, results: Results) -> None:
    """Viewers of every viewer manager are addressable by id, and a slot name
    resolves to the main window's own viewer first."""
    try:
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        viewers = state.get("viewers", [])
        ids = [v.get("viewerId") for v in viewers]
        segmentation = [
            v.get("viewerId") for v in viewers if v.get("surfName") == "segmentation"
        ]
        centered, _ = client.call(
            "viewer.center_on_point",
            {"point": {"x": 8, "y": 8, "z": 8}},
            timeout=RPC_TIMEOUT,
        )
        unresolved = []
        for viewer in viewers:
            if viewer.get("kind") != "chunked":
                continue
            # Resolution, not rendering: a zoom by 1.0 must reach every pane.
            try:
                client.call(
                    "viewer.zoom",
                    {"viewer": viewer.get("viewerId"), "factor": 1.0},
                    timeout=RPC_TIMEOUT,
                )
            except BridgeError as error:
                unresolved.append(f"{viewer.get('viewerId')}: {error.code}")
        ok = (
            len(ids) == len(set(ids))
            and bool(segmentation)
            and centered.get("viewerId") == segmentation[0]
            and not unresolved
        )
        results.record(
            "viewer_registry",
            ok,
            f"viewers={len(ids)} segmentation={segmentation} "
            f"default={centered.get('viewerId')} unresolved={unresolved}",
        )
    except Exception as error:  # noqa: BLE001
        results.record("viewer_registry", False, f"{type(error).__name__}: {error}")


def wait_held(client: BridgeClient, key: str, held: bool,
              timeout: float = 5.0) -> tuple[bool, float]:
    """Poll state.get until key is (or is no longer) in heldKeys.

    Returns whether that happened within timeout and the monotonic time it
    was seen, so checks do not depend on how fast the bridge handles a
    request on a slow runner.
    """
    deadline = time.monotonic() + timeout
    while True:
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        keys = [k.get("key") for k in state.get("heldKeys", [])]
        seen = time.monotonic()
        if (key in keys) == held:
            return True, seen
        if seen >= deadline:
            return False, seen
        time.sleep(0.05)


def check_canvas_input(sock_path: str, client: BridgeClient, results: Results) -> None:
    """canvas.key and canvas.wheel reach the viewer through real Qt events."""
    failures: list[str] = []
    try:
        client.call("workspace.switch", {"name": "main"}, timeout=RPC_TIMEOUT)
        tap, _ = client.call("canvas.key", {"key": "Escape"}, timeout=RPC_TIMEOUT)
        if (tap.get("key") != "Esc" or tap.get("heldKeys") != []
                or "accepted" not in tap or tap.get("dismissed") != []):
            failures.append(f"tap={tap}")
        pressed, _ = client.call(
            "canvas.key", {"key": "Shift", "action": "press"}, timeout=RPC_TIMEOUT)
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        held_in_state = [k.get("key") for k in state.get("heldKeys", [])]
        released, _ = client.call(
            "canvas.key", {"key": "Shift", "action": "release"}, timeout=RPC_TIMEOUT)
        if pressed.get("heldKeys") != ["Shift"] or released.get("heldKeys") != []:
            failures.append(f"press={pressed} release={released}")
        if pressed.get("holdMs") != 10000 or held_in_state != ["Shift"]:
            failures.append(f"press limit={pressed.get('holdMs')} state={held_in_state}")
        if released.get("wasHeld") is not True:
            failures.append(f"release wasHeld={released.get('wasHeld')}")
        # A press with a short limit is released without a release call.
        client.call(
            "canvas.key", {"key": "F23", "action": "press", "holdMs": 100},
            timeout=RPC_TIMEOUT)
        released_in_time, _ = wait_held(client, "F23", held=False)
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        if not released_in_time or state.get("heldKeys") != []:
            failures.append(f"press limit kept {state.get('heldKeys')}")
        started = time.monotonic()
        held, _ = client.call(
            "canvas.key", {"key": "F24", "holdMs": 200}, timeout=RPC_TIMEOUT)
        elapsed = time.monotonic() - started
        if held.get("holdMs") != 200 or held.get("heldKeys") != [] or elapsed < 0.19:
            failures.append(f"hold={held} elapsed={elapsed:.3f}")
        # A hold's timer releases only its own press: release the held key
        # early and press it again, and the new press outlives the old timer.
        hold_result: dict = {}

        def hold() -> None:
            try:
                hold_result["value"], _ = client.call(
                    "canvas.key", {"key": "F20", "holdMs": 800}, timeout=RPC_TIMEOUT)
            except Exception as error:  # noqa: BLE001
                hold_result["error"] = f"{type(error).__name__}: {error}"

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        # Release only once the bridge holds the key, however long the
        # request took to arrive; the hold's timer started before this.
        hold_seen, held_at = wait_held(client, "F20", held=True)
        if not hold_seen:
            failures.append("hold of F20 never reached heldKeys")
        client.call("canvas.key", {"key": "F20", "action": "release"}, timeout=RPC_TIMEOUT)
        client.call(
            "canvas.key", {"key": "F20", "action": "press", "holdMs": 5000},
            timeout=RPC_TIMEOUT)
        holder.join(timeout=RPC_TIMEOUT)
        # Look again after the first hold's 800 ms timer has fired.
        time.sleep(max(0.0, held_at + 1.3 - time.monotonic()))
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        still_held = [k.get("key") for k in state.get("heldKeys", [])]
        client.call("canvas.key", {"key": "F20", "action": "release"}, timeout=RPC_TIMEOUT)
        if (hold_result.get("value", {}).get("releasedEarly") is not True
                or still_held != ["F20"]):
            failures.append(f"hold serial: hold={hold_result} held after={still_held}")
        # Tapping a key this connection holds is refused.
        client.call("canvas.key", {"key": "F22", "action": "press"}, timeout=RPC_TIMEOUT)
        try:
            client.call("canvas.key", {"key": "F22"}, timeout=RPC_TIMEOUT)
            failures.append("tap of a held key accepted")
        except BridgeError as error:
            if error.code != -32009:
                failures.append(f"tap of a held key code={error.code}")
        client.call("canvas.key", {"key": "F22", "action": "release"}, timeout=RPC_TIMEOUT)
        # A viewer on a hidden workspace is refused: its key would reach the
        # focused widget's shortcuts instead.
        spiral = [v for v in state.get("viewers", [])
                  if v.get("workspace") == "spiral" and v.get("kind") == "chunked"]
        if spiral:
            try:
                client.call(
                    "canvas.key", {"key": "Escape", "viewer": spiral[0]["viewerId"]},
                    timeout=RPC_TIMEOUT)
                failures.append("key to a hidden viewer accepted")
            except BridgeError as error:
                if error.code != -32009:
                    failures.append(f"hidden viewer code={error.code}")
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")
    for params, param in (
        ({"key": "A,B"}, "key"),
        ({"key": "A", "action": "release", "holdMs": 10}, "holdMs"),
        ({"key": "A", "holdMs": 60001}, "holdMs"),
    ):
        ok, detail = expect_param_error(client, "canvas.key", params, param)
        if not ok:
            failures.append(f"{params}: {detail}")
    results.record(
        "canvas_key",
        not failures,
        "tap, press/release, press limit and hold delivered" if not failures
        else "; ".join(failures[:4]),
    )

    # A key whose shortcut opens a modal dialog (Ctrl+O opens the project
    # browser) gets the dialog answered, while another client's requests
    # wait for the call instead of running inside the dialog's event loop.
    failures = []
    other = None
    try:
        before, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        other = BridgeClient(sock_path, connect_timeout=RPC_TIMEOUT)
        stop = threading.Event()
        pings: list[str] = []

        def ping_loop() -> None:
            while not stop.is_set():
                try:
                    other.call("ping", {}, timeout=RPC_TIMEOUT)
                    pings.append("ok")
                except Exception as error:  # noqa: BLE001
                    pings.append(f"{type(error).__name__}: {error}")
                    return

        pinger = threading.Thread(target=ping_loop, daemon=True)
        pinger.start()
        opened = []
        for _ in range(3):
            result, _ = client.call("canvas.key", {"key": "Ctrl+O"}, timeout=RPC_TIMEOUT)
            opened.extend(result.get("dismissed", []))
            if any(not {"class", "title", "answer"} <= set(entry) for entry in opened):
                failures.append(f"dismissed={opened}")
        stop.set()
        pinger.join(timeout=RPC_TIMEOUT)
        after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        if after.get("vpkg") != before.get("vpkg"):
            failures.append(f"project changed: {after.get('vpkg')}")
        if not pings or any(p != "ok" for p in pings):
            failures.append(f"pings={pings[-3:]}")
        detail = f"dismissed={[(e.get('class'), e.get('answer')) for e in opened]} pings={len(pings)}"
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")
        detail = ""
    finally:
        if other is not None:
            other.close()
    results.record(
        "canvas_key_dialog",
        not failures,
        detail if not failures else "; ".join(failures[:4]),
    )

    try:
        ok, detail = expect_param_error(
            client,
            "canvas.wheel",
            {"viewer": "xy plane", "position": {"x": 5, "y": 5}, "space": "scene",
             "deltaY": 0},
            "deltaY",
        )
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        before = next(
            (v["scale"] for v in state.get("viewers", []) if v.get("surfName") == "xy plane"),
            None,
        )
        wheel, _ = client.call(
            "canvas.wheel",
            {"viewer": "xy plane", "position": {"x": 5, "y": 5}, "space": "scene",
             "deltaY": 120},
            timeout=RPC_TIMEOUT,
        )
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        after = next(
            (v["scale"] for v in state.get("viewers", []) if v.get("surfName") == "xy plane"),
            None,
        )
        ok = ok and wheel.get("notches") == 1.0 and before is not None and after != before
        # A pane on a hidden workspace is refused, so its zoom cannot change
        # where nobody sees it.
        hidden = "no spiral pane"
        spiral = [v for v in state.get("viewers", [])
                  if v.get("workspace") == "spiral" and v.get("kind") == "chunked"]
        if spiral:
            try:
                client.call(
                    "canvas.wheel",
                    {"viewer": spiral[0]["viewerId"], "position": {"x": 5, "y": 5},
                     "space": "scene", "deltaY": 120},
                    timeout=RPC_TIMEOUT,
                )
                ok = False
                hidden = "hidden pane accepted"
            except BridgeError as error:
                ok = ok and error.code == -32009
                hidden = f"hidden pane code={error.code}"
        results.record(
            "canvas_wheel_zoom",
            ok,
            f"{detail}; scale {before} -> {after}; accepted={wheel.get('accepted')}; {hidden}",
        )
    except Exception as error:  # noqa: BLE001
        results.record("canvas_wheel_zoom", False, f"{type(error).__name__}: {error}")


def check_state_changes(sock_path: str, client: BridgeClient, results: Results) -> None:
    """A change made by one client reaches another as state.changed."""
    observer = None
    try:
        observer = BridgeClient(sock_path, connect_timeout=RPC_TIMEOUT)
        before, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        client.call("workspace.switch", {"name": "lasagna"}, timeout=RPC_TIMEOUT)
        note = observer.wait_for_notification(
            "state.changed",
            predicate=lambda p: "workspace" in p.get("changed", []),
            timeout=5.0,
        )
        after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        client.call("workspace.switch", {"name": "main"}, timeout=RPC_TIMEOUT)
        ok = (
            isinstance(before.get("revision"), int)
            and after.get("revision", -1) > before.get("revision", 0)
            and note.get("revision", -1) <= after.get("revision", -1)
        )
        results.record(
            "state_changed_notification",
            ok,
            f"revision {before.get('revision')} -> {after.get('revision')}; note={note}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "state_changed_notification", False, f"{type(error).__name__}: {error}")
    finally:
        if observer is not None:
            observer.close()


def _walk_nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _walk_nodes(child)


def check_ui_inspect(client: BridgeClient, results: Results) -> None:
    """ui.inspect reads panels through Qt's accessibility tree and never
    reports a redacted value."""
    failures: list[str] = []
    try:
        window, _ = client.call("ui.inspect", {"depth": 3}, timeout=RPC_TIMEOUT)
        tree = window.get("tree", {})
        if tree.get("class") != "CWindow" or tree.get("role") != "Window":
            failures.append(f"window root={tree.get('class')}/{tree.get('role')}")
        if not window.get("docks"):
            failures.append("no docks listed")
        visible_docks = [d for d in window.get("docks", []) if d.get("visible")]
        if visible_docks:
            dock_name = visible_docks[0]["objectName"]
            dock, _ = client.call(
                "ui.inspect", {"root": f"dock:{dock_name}"}, timeout=RPC_TIMEOUT)
            if dock.get("tree", {}).get("objectName") != dock_name:
                failures.append(f"dock root={dock.get('tree')}")
        main, _ = client.call(
            "ui.inspect",
            {"root": "workspace:main", "depth": 64, "includeHidden": True},
            timeout=RPC_TIMEOUT,
        )
        nodes = list(_walk_nodes(main.get("tree", {})))
        if any("role" not in n for n in nodes):
            failures.append("a node has no role")
        if not any(n.get("viewerId") for n in nodes):
            failures.append("no viewer pane reported a viewerId")
        if not any(n.get("role") == "SpinBox" and "value" in n for n in nodes):
            failures.append("no spin box value read")
        # Spiral's API key entry (password echo, labeled "API key") is read
        # from a workspace that is not current, and withheld.
        spiral, _ = client.call(
            "ui.inspect",
            {"root": "workspace:spiral", "depth": 64, "includeHidden": True},
            timeout=RPC_TIMEOUT,
        )
        nodes += list(_walk_nodes(spiral.get("tree", {})))
        leaked = [
            n for n in nodes
            if n.get("redacted") and n.get("role") == "EditableText" and "value" in n
        ]
        api_keys = [
            n for n in nodes
            if n.get("role") == "EditableText" and n.get("name") == "API key"
        ]
        if leaked:
            failures.append(f"redaction leaked: {leaked[:1]}")
        if not api_keys or any("value" in n or not n.get("redacted") for n in api_keys):
            failures.append(f"API key entries={api_keys}")
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")
    for params, param in (({"root": "bogus"}, "root"), ({"depth": 65}, "depth")):
        ok, detail = expect_param_error(client, "ui.inspect", params, param)
        if not ok:
            failures.append(f"{params}: {detail}")
    try:
        client.call("ui.inspect", {"root": "dock:noSuchDock"}, timeout=RPC_TIMEOUT)
        failures.append("unknown dock did not fail")
    except BridgeError as error:
        if error.code != -32007:
            failures.append(f"unknown dock code={error.code}")
    results.record(
        "ui_inspect",
        not failures,
        "window, dock and workspace roots read; redaction held" if not failures
        else "; ".join(failures[:4]),
    )


def check_screenshot_targets(client: BridgeClient, results: Results) -> None:
    """Dock and workspace targets, regions, and reveal with restore."""
    failures: list[str] = []

    def capture(params: dict) -> dict:
        result, _ = client.call("screenshot.capture", params, timeout=RPC_TIMEOUT)
        result.pop("base64", None)
        return result

    try:
        client.call("workspace.switch", {"name": "main"}, timeout=RPC_TIMEOUT)
        part = capture({"target": "window",
                        "region": {"x": 0, "y": 0, "width": 64, "height": 32}})
        if (part.get("region") != {"x": 0, "y": 0, "width": 64, "height": 32}
                or part.get("width", 0) < 64):
            failures.append(f"region={part}")
        main = capture({"target": "workspace:main"})
        if main.get("revealed") is not False or not isinstance(main.get("masked"), int):
            failures.append(
                f"current workspace revealed={main.get('revealed')} masked={main.get('masked')}")
        try:
            capture({"target": "workspace:lasagna"})
            failures.append("hidden workspace captured without reveal")
        except BridgeError as error:
            if error.code != -32009:
                failures.append(f"hidden workspace code={error.code}")
        hidden = capture({"target": "workspace:lasagna", "reveal": True})
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        current = (state.get("workspace") or {}).get("current")
        if hidden.get("revealed") is not True or current != "main":
            failures.append(f"reveal={hidden.get('revealed')} restored={current}")
        docks, _ = client.call("ui.inspect", {"depth": 0}, timeout=RPC_TIMEOUT)
        visible = [d for d in docks.get("docks", []) if d.get("visible")]
        if visible:
            capture({"target": f"dock:{visible[0]['objectName']}"})
        try:
            capture({"target": "dock:noSuchDock"})
            failures.append("unknown dock captured")
        except BridgeError as error:
            if error.code != -32007:
                failures.append(f"unknown dock code={error.code}")
        spiral = [v for v in state.get("viewers", []) if v.get("workspace") == "spiral"]
        if spiral:
            capture({"target": spiral[0]["viewerId"], "reveal": True, "maxDim": 256})
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")
    ok, detail = expect_param_error(
        client,
        "screenshot.capture",
        {"region": {"x": 100000, "y": 0, "width": 8, "height": 8}},
        "region",
    )
    if not ok:
        failures.append(f"outside region: {detail}")
    results.record(
        "screenshot_targets",
        not failures,
        "region, dock, workspace and reveal captures behaved" if not failures
        else "; ".join(failures[:4]),
    )


# Every workspace id in tab order (AgentBridgeUi.hpp's agentBridgeWorkspaceIds,
# set as each tab page's "workspaceId" property in CWindow's constructor).
WORKSPACE_IDS = (
    "main",
    "lasagna",
    "atlas",
    "fiber-slice",
    "intersections",
    "spiral",
    "fiber-map",
)


def check_workspaces(client: BridgeClient, results: Results) -> None:
    """workspace.switch reaches every tab, each id is present in this build,
    and the original fiber_slice spelling still works."""
    failures: list[str] = []
    try:
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        available = [
            entry.get("id") for entry in (state.get("workspace") or {}).get("available", [])
        ]
        # Exactly the bridge's ids, in tab order: a tab added to CWindow must
        # be added to agentBridgeWorkspaceIds() too.
        if available != list(WORKSPACE_IDS):
            failures.append(f"available={available}")
        previous = None
        for workspace_id in WORKSPACE_IDS:
            result, _ = client.call(
                "workspace.switch", {"name": workspace_id}, timeout=RPC_TIMEOUT)
            after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
            current = (after.get("workspace") or {}).get("current")
            if result.get("workspace") != workspace_id or current != workspace_id:
                failures.append(f"{workspace_id}: result={result} current={current}")
            if previous is not None and result.get("previous") != previous:
                failures.append(f"{workspace_id}: previous={result.get('previous')}")
            previous = workspace_id
        legacy, _ = client.call(
            "workspace.switch", {"name": "fiber_slice"}, timeout=RPC_TIMEOUT)
        if legacy.get("workspace") != "fiber-slice":
            failures.append(f"fiber_slice alias -> {legacy}")
        client.call("workspace.switch", {"name": "main"}, timeout=RPC_TIMEOUT)
    except Exception as error:  # noqa: BLE001
        failures.append(f"{type(error).__name__}: {error}")
    results.record(
        "workspace_switch_all",
        not failures,
        "every workspace switched in tab order" if not failures else "; ".join(failures[:4]),
    )

    try:
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        viewers = state.get("viewers", [])
        shaped = all("workspace" in v and "visible" in v for v in viewers)
        main_viewers = [v for v in viewers if v.get("workspace") == "main"]
        spiral = [v for v in viewers if v.get("workspace") == "spiral"]
        visible_main = [v for v in main_viewers if v.get("visible")]
        ok = shaped and bool(visible_main) and not any(v.get("visible") for v in spiral)
        results.record(
            "viewers_every_manager",
            ok,
            f"viewers={len(viewers)} main={len(main_viewers)} "
            f"visible main={len(visible_main)} spiral={len(spiral)}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "viewers_every_manager", False, f"{type(error).__name__}: {error}")


def check_project_create(
    client: BridgeClient,
    results: Results,
    root: Path,
) -> None:
    volume = root / "volumes" / "vol1"
    requested = root / "created" / "agent-project"
    created = requested.with_suffix(".volpkg.json")

    try:
        before, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        result, _ = client.call(
            "project.create",
            {
                "path": str(requested),
                "volume": str(volume),
                "tags": ["source:smoke"],
            },
            timeout=RPC_TIMEOUT,
        )
        after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        document = json.loads(created.read_text())
        volume_entry = document.get("volumes", [{}])[0]
        valid = (
            result == {
                "path": str(created),
                "name": "agent-project",
                "volume": str(volume),
            }
            and document.get("name") == "agent-project"
            and document.get("version") == 1
            and volume_entry == {
                "location": str(volume),
                "tags": ["source:smoke"],
            }
            and document.get("segments") == []
            and document.get("normal_grids") == []
            and document.get("lasagna_datasets") == []
            and after.get("vpkg") == before.get("vpkg")
        )
        results.record(
            "project_create",
            valid,
            f"path={result.get('path')} session_unchanged="
            f"{after.get('vpkg') == before.get('vpkg')}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "project_create",
            False,
            f"{type(error).__name__}: {error}",
        )
        return

    try:
        client.call(
            "project.create",
            {"path": str(created), "volume": str(volume)},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "project_create_overwrite_guard",
            False,
            "expected an error, got a result",
        )
    except BridgeError as error:
        results.record(
            "project_create_overwrite_guard",
            error.code == -32005,
            f"returned code={error.code}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "project_create_overwrite_guard",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )

    def expect_volume_error(
        name: str,
        candidate: str,
        expected_detail: str,
    ) -> None:
        output = root / "created" / name
        try:
            client.call(
                "project.create",
                {"path": str(output), "volume": candidate},
                timeout=RPC_TIMEOUT,
            )
            results.record(name, False, "expected an error, got a result")
        except BridgeError as error:
            detail = error.data.get("detail", "")
            results.record(
                name,
                error.code == -32007
                and error.data.get("kind") == "volume"
                and expected_detail in detail
                and not output.with_suffix(".volpkg.json").exists(),
                f"returned code={error.code} detail={detail!r}",
            )
        except Exception as error:  # noqa: BLE001
            results.record(
                name,
                False,
                f"unexpected {type(error).__name__}: {error}",
            )

    expect_volume_error(
        "project_create_invalid_volume",
        str(root / "missing.zarr"),
        "Path does not exist",
    )
    expect_volume_error(
        "project_create_invalid_remote_selector",
        "https://example.test/volume.zarr#unknown=2",
        "unsupported remote volume selector",
    )
    expect_volume_error(
        "project_create_remote_requires_zarr",
        "https://example.test/not-a-zarr",
        "must point directly to a .zarr root",
    )
    expect_volume_error(
        "project_create_rejects_volume_collection",
        str(root / "volumes"),
        "Not a zarr volume",
    )

    ok, detail = expect_param_error(
        client,
        "project.create",
        {"path": "relative/project", "volume": str(volume)},
        "path",
    )
    results.record("project_create_absolute_path", ok, detail)

    try:
        opened, _ = client.call(
            "volume.open",
            {"path": str(created)},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "project_create_then_open",
            opened.get("opened") is True
            and "vol1" in opened.get("volumeIds", []),
            f"opened={opened.get('opened')} volumeIds={opened.get('volumeIds')}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "project_create_then_open",
            False,
            f"{type(error).__name__}: {error}",
        )


def check_volume_attach(
    client: BridgeClient,
    results: Results,
    root: Path,
    volpkg: Path,
) -> None:
    overlay = create_test_volume(root / "overlays" / "vol2", "vol2")
    conflict = create_test_volume(root / "conflicts" / "vol2", "vol2")

    try:
        retry, _ = client.call(
            "volume.attach",
            {"location": str(root / "volumes" / "vol1")},
            timeout=RPC_TIMEOUT,
        )
        terminal = wait_for_job(client, retry["jobId"])
        document = json.loads(volpkg.read_text())
        result = terminal.get("result") or {}
        results.record(
            "volume_attach_existing_relative_entry",
            terminal.get("state") == "succeeded"
            and result.get("attached") is False
            and result.get("alreadyAttached") is True
            and document.get("volumes") == ["volumes"],
            f"state={terminal.get('state')} result={result}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "volume_attach_existing_relative_entry",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        before, _ = client.call("volume.list", {}, timeout=RPC_TIMEOUT)
        started, _ = client.call(
            "volume.attach",
            {
                "location": str(overlay),
                "tags": ["role:overlay", "source:smoke"],
            },
            timeout=RPC_TIMEOUT,
        )
        terminal = wait_for_job(client, started["jobId"])
        volumes, _ = client.call("volume.list", {}, timeout=RPC_TIMEOUT)
        available, _ = client.call(
            "viewer.list_overlay_volumes", {}, timeout=RPC_TIMEOUT)
        overlay_result, _ = client.call(
            "viewer.set_overlay",
            {"volumeId": "vol2", "opacity": 0.4},
            timeout=RPC_TIMEOUT,
        )
        overlay_read, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        selected, _ = client.call(
            "viewer.list_overlay_volumes", {}, timeout=RPC_TIMEOUT)
        render_result, _ = client.call(
            "viewer.set_render_settings",
            {"overlayOpacity": 0.35},
            timeout=RPC_TIMEOUT,
        )
        overlay_after_render, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        results.record(
            "viewer_overlay_controller_sync",
            overlay_read.get("volumeId") == "vol2"
            and selected.get("overlayVolumeId") == "vol2"
            and abs(render_result.get("overlayOpacity", -1) - 0.35) < 1e-6
            and overlay_after_render.get("volumeId") == "vol2"
            and abs(overlay_after_render.get("opacity", -1) - 0.35) < 1e-6,
            f"set={overlay_result} get={overlay_read} "
            f"list_id={selected.get('overlayVolumeId')} "
            f"render_opacity={render_result.get('overlayOpacity')} "
            f"overlay_opacity={overlay_after_render.get('opacity')}",
        )
        document = json.loads(volpkg.read_text())
        entries = document.get("volumes", [])
        attached_entry = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("location") == str(overlay)
            ),
            None,
        )
        listed = {item.get("id"): item for item in available.get("volumes", [])}
        result = terminal.get("result") or {}
        valid = (
            started.get("kind") == "volume.attach"
            and started.get("source") == "volume"
            and terminal.get("state") == "succeeded"
            and result.get("attached") is True
            and result.get("alreadyAttached") is False
            and result.get("volumeId") == "vol2"
            and before.get("currentVolumeId") == "vol1"
            and volumes.get("currentVolumeId") == "vol1"
            and set(volumes.get("volumeIds", [])) == {"vol1", "vol2"}
            and listed.get("vol2", {}).get("current") is False
            and overlay_result.get("volumeId") == "vol2"
            and attached_entry == {
                "location": str(overlay),
                "tags": ["role:overlay", "source:smoke"],
            }
        )
        results.record(
            "volume_attach_overlay_workflow",
            valid,
            f"state={terminal.get('state')} volumeIds={volumes.get('volumeIds')} "
            f"overlay={overlay_result.get('volumeId')}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "volume_attach_overlay_workflow",
            False,
            f"{type(error).__name__}: {error}",
        )
        return

    try:
        expected_overlay, _ = client.call(
            "viewer.set_overlay",
            {
                "colormap": "",
                "opacity": 0.35,
                "window": {"low": 17, "high": 201},
            },
            timeout=RPC_TIMEOUT,
        )
        client.call(
            "volume.open",
            {"path": str(volpkg)},
            timeout=RPC_TIMEOUT,
        )
        restored_overlay, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        results.record(
            "viewer_overlay_reopen",
            restored_overlay == expected_overlay,
            f"expected={expected_overlay} restored={restored_overlay}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "viewer_overlay_reopen",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        retry, _ = client.call(
            "volume.attach",
            {"location": str(overlay), "tags": ["replacement:ignored"]},
            timeout=RPC_TIMEOUT,
        )
        terminal = wait_for_job(client, retry["jobId"])
        document = json.loads(volpkg.read_text())
        matching = [
            entry
            for entry in document.get("volumes", [])
            if isinstance(entry, dict) and entry.get("location") == str(overlay)
        ]
        result = terminal.get("result") or {}
        results.record(
            "volume_attach_idempotent",
            terminal.get("state") == "succeeded"
            and result.get("attached") is False
            and result.get("alreadyAttached") is True
            and matching == [{
                "location": str(overlay),
                "tags": ["role:overlay", "source:smoke"],
            }],
            f"state={terminal.get('state')} result={result}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "volume_attach_idempotent",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        started, _ = client.call(
            "volume.attach",
            {"location": str(conflict)},
            timeout=RPC_TIMEOUT,
        )
        terminal = wait_for_job(client, started["jobId"])
        volumes, _ = client.call("volume.list", {}, timeout=RPC_TIMEOUT)
        document = json.loads(volpkg.read_text())
        persisted = [
            entry
            for entry in document.get("volumes", [])
            if isinstance(entry, dict) and entry.get("location") == str(conflict)
        ]
        results.record(
            "volume_attach_id_conflict",
            terminal.get("state") == "failed"
            and "different volume with id" in terminal.get("message", "").lower()
            and set(volumes.get("volumeIds", [])) == {"vol1", "vol2"}
            and not persisted,
            f"state={terminal.get('state')} message={terminal.get('message')!r}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "volume_attach_id_conflict",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        client.call(
            "volume.attach",
            {"location": str(root / "volumes")},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "volume_attach_rejects_collection",
            False,
            "expected an error, got a result",
        )
    except BridgeError as error:
        results.record(
            "volume_attach_rejects_collection",
            error.code == -32007
            and error.data.get("kind") == "volume"
            and "Not a zarr volume" in error.data.get("detail", ""),
            f"returned code={error.code} detail={error.data.get('detail')!r}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "volume_attach_rejects_collection",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )

    ok, detail = expect_param_error(
        client,
        "volume.attach",
        {"location": "relative/overlay.zarr"},
        "location",
    )
    results.record("volume_attach_absolute_path", ok, detail)


def check_segments_attach(
    client: BridgeClient,
    results: Results,
    root: Path,
    volpkg: Path,
) -> None:
    initial = root / "initial-segments"
    create_test_segment(initial / "initial-segment", "initial-segment")
    source = root / "user-segments"
    segment_id = "20241113080880"
    shutil.copytree(
        REPO_ROOT / "core" / "test" / "data" / "segments" / segment_id,
        source / segment_id,
    )

    try:
        initial_result, _ = client.call(
            "segments.attach",
            {"location": str(initial) + os.sep},
            timeout=RPC_TIMEOUT,
        )
        result, _ = client.call(
            "segments.attach",
            {
                "location": str(source),
                "tags": ["source:smoke", "status:working"],
            },
            timeout=RPC_TIMEOUT,
        )
        listed, _ = client.call("segments.list", {}, timeout=RPC_TIMEOUT)
        document = json.loads(volpkg.read_text())
        initial_locations = {
            entry.get("location") if isinstance(entry, dict) else entry
            for entry in document.get("segments", [])
        }
        matching = [
            entry
            for entry in document.get("segments", [])
            if isinstance(entry, dict) and entry.get("location") == str(source)
        ]
        ids = {segment.get("id") for segment in listed.get("segments", [])}
        valid = (
            initial_result.get("location") == str(initial)
            and str(initial) in initial_locations
            and result == {
                "attached": True,
                "alreadyAttached": False,
                "location": str(source),
                "projectPath": str(volpkg),
                "selected": True,
            }
            and segment_id in ids
            and matching == [{
                "location": str(source),
                "tags": ["source:smoke", "status:working"],
            }]
        )
        results.record(
            "segments_attach_local_source",
            valid,
            f"initial={initial_result} initial_locations={initial_locations} "
            f"result={result} ids={sorted(value for value in ids if value)}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_local_source",
            False,
            f"{type(error).__name__}: {error}",
        )
        return

    try:
        retry, _ = client.call(
            "segments.attach",
            {"location": str(source) + os.sep},
            timeout=RPC_TIMEOUT,
        )
        document = json.loads(volpkg.read_text())
        matching = [
            entry
            for entry in document.get("segments", [])
            if isinstance(entry, dict) and entry.get("location") == str(source)
        ]
        results.record(
            "segments_attach_idempotent",
            retry.get("attached") is False
            and retry.get("alreadyAttached") is True
            and retry.get("selected") is True
            and matching == [{
                "location": str(source),
                "tags": ["source:smoke", "status:working"],
            }],
            f"result={retry}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_idempotent",
            False,
            f"{type(error).__name__}: {error}",
        )

    try:
        client.call(
            "volume.open",
            {"path": str(volpkg)},
            timeout=RPC_TIMEOUT,
        )
        reopened, _ = client.call("segments.list", {}, timeout=RPC_TIMEOUT)
        retry, _ = client.call(
            "segments.attach",
            {"location": str(source)},
            timeout=RPC_TIMEOUT,
        )
        document = json.loads(volpkg.read_text())
        matching = [
            entry
            for entry in document.get("segments", [])
            if isinstance(entry, dict) and entry.get("location") == str(source)
        ]
        ids = {
            segment.get("id")
            for segment in reopened.get("segments", [])
        }
        results.record(
            "segments_attach_reopen",
            segment_id in ids
            and retry.get("alreadyAttached") is True
            and retry.get("selected") is True
            and matching == [{
                "location": str(source),
                "tags": ["source:smoke", "status:working"],
            }],
            f"ids={sorted(value for value in ids if value)} retry={retry}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_reopen",
            False,
            f"{type(error).__name__}: {error}",
        )

    editing_enabled = False
    try:
        client.call(
            "segments.activate",
            {"segmentId": segment_id},
            timeout=RPC_TIMEOUT,
        )
        client.call(
            "segmentation.enable_editing",
            {"enabled": True},
            timeout=RPC_TIMEOUT,
        )
        editing_enabled = True
        blocked = root / "blocked-segments"
        create_test_segment(blocked / "blocked-segment", "blocked-segment")
        before = json.loads(volpkg.read_text())
        try:
            client.call(
                "segments.attach",
                {"location": str(blocked), "select": False},
                timeout=RPC_TIMEOUT,
            )
            results.record(
                "segments_attach_editing_guard",
                False,
                "expected an error, got a result",
            )
        except BridgeError as error:
            after = json.loads(volpkg.read_text())
            results.record(
                "segments_attach_editing_guard",
                error.code == -32004 and after == before,
                f"returned code={error.code} project_unchanged={after == before}",
            )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_editing_guard",
            False,
            f"{type(error).__name__}: {error}",
        )
    finally:
        if editing_enabled:
            try:
                client.call(
                    "segmentation.enable_editing",
                    {"enabled": False},
                    timeout=RPC_TIMEOUT,
                )
            except Exception as error:  # noqa: BLE001
                results.record(
                    "segments_attach_editing_cleanup",
                    False,
                    f"{type(error).__name__}: {error}",
                )

    conflicting = root / "other" / source.name.upper()
    create_test_segment(
        conflicting / "conflicting-segment", "conflicting-segment")
    before = json.loads(volpkg.read_text())
    try:
        client.call(
            "segments.attach",
            {"location": str(conflicting)},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "segments_attach_source_name_conflict",
            False,
            "expected an error, got a result",
        )
    except BridgeError as error:
        after = json.loads(volpkg.read_text())
        results.record(
            "segments_attach_source_name_conflict",
            error.code == -32010
            and error.data.get("sourceName") == source.name.upper()
            and after == before,
            f"returned code={error.code} project_unchanged={after == before}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_source_name_conflict",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )

    ok, detail = expect_param_error(
        client,
        "segments.attach",
        {"location": "relative/segments"},
        "location",
    )
    results.record("segments_attach_absolute_path", ok, detail)

    try:
        client.call(
            "segments.attach",
            {"location": str(root / "missing-segments")},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "segments_attach_invalid_layout",
            False,
            "expected an error, got a result",
        )
    except BridgeError as error:
        results.record(
            "segments_attach_invalid_layout",
            error.code == -32007,
            f"returned code={error.code}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "segments_attach_invalid_layout",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )


def check_open_data_editable_copy(
    client: BridgeClient,
    results: Results,
    root: Path,
    volpkg: Path,
) -> None:
    catalog_root = root / "catalog-segments"
    catalog_folder = catalog_root / "published-L0"
    segment_id = "20241113090990"
    catalog_segment = catalog_folder / segment_id
    shutil.copytree(
        REPO_ROOT / "core" / "test" / "data" / "segments" / segment_id,
        catalog_segment,
    )
    (catalog_segment / "catalog-origin.json").write_text("{}")
    renamed_path = catalog_folder / "must-not-rename"

    try:
        project = json.loads(volpkg.read_text())
        project.setdefault("segments", []).append({
            "location": str(catalog_root),
            "tags": [
                "open-data",
                "immutable",
                "vc-open-data-segment-aggregate",
                "vc-open-data-target-volume-id:vol1",
            ],
        })
        project["output_segments"] = str(catalog_root)
        volpkg.write_text(json.dumps(project))
        client.call(
            "volume.open",
            {"path": str(volpkg)},
            timeout=RPC_TIMEOUT,
        )
        client.call(
            "segments.activate",
            {"segmentId": segment_id},
            timeout=RPC_TIMEOUT,
        )

        try:
            client.call(
                "segments.rename",
                {"segmentId": segment_id, "newName": renamed_path.name},
                timeout=RPC_TIMEOUT,
            )
            results.record(
                "catalog_segment_rename_guard",
                False,
                "expected an immutable-cache error, got a result",
            )
        except BridgeError as error:
            results.record(
                "catalog_segment_rename_guard",
                error.code == -32009
                and error.data.get("action") == "segments.create_editable_copy"
                and catalog_segment.is_dir()
                and not renamed_path.exists(),
                f"returned code={error.code} action={error.data.get('action')}",
            )

        started = time.monotonic()
        try:
            client.call(
                "segmentation.enable_editing",
                {"enabled": True},
                timeout=RPC_TIMEOUT,
            )
            results.record(
                "catalog_segment_editing_fails_fast",
                False,
                "expected an immutable-cache error, got a result",
            )
        except BridgeError as error:
            elapsed = time.monotonic() - started
            results.record(
                "catalog_segment_editing_fails_fast",
                error.code == -32009
                and error.data.get("action") == "segments.create_editable_copy"
                and elapsed < RPC_TIMEOUT,
                f"returned code={error.code} action={error.data.get('action')} "
                f"elapsed={elapsed:.3f}s",
            )

        copied, _ = client.call(
            "segments.create_editable_copy",
            {"segmentId": segment_id},
            timeout=RPC_TIMEOUT,
        )
        editable_segment = Path(copied.get("path", ""))
        editable_root = editable_segment.parent
        project = json.loads(volpkg.read_text())
        editable_entries = [
            entry
            for entry in project.get("segments", [])
            if isinstance(entry, dict) and entry.get("location") == str(editable_root)
        ]
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        active = state.get("activeSurface") or {}
        listed, _ = client.call("segments.list", {}, timeout=RPC_TIMEOUT)
        listed_active = next(
            (segment for segment in listed.get("segments", []) if segment.get("active")),
            {},
        )
        results.record(
            "catalog_segment_create_editable_copy",
            copied.get("created") is True
            and copied.get("alreadyExisted") is False
            and copied.get("activated") is True
            and copied.get("sourcePath") == str(catalog_segment)
            and editable_segment.is_dir()
            and not (editable_segment / "catalog-origin.json").exists()
            and (editable_segment / "x.tif").is_file()
            and (editable_segment / "y.tif").is_file()
            and (editable_segment / "z.tif").is_file()
            and editable_entries == [{
                "location": str(editable_root),
                "tags": [
                    "open-data-editable",
                    "vc-open-data-target-volume-id:vol1",
                ],
            }]
            and project.get("output_segments") == str(editable_root)
            and active.get("id") == segment_id
            and listed_active.get("id") == segment_id
            and listed_active.get("path") == str(editable_segment),
            f"result={copied} active={active} listed_active={listed_active} "
            f"output_segments={project.get('output_segments')} "
            f"entries={editable_entries}",
        )

        retried, _ = client.call(
            "segments.create_editable_copy",
            {"segmentId": segment_id},
            timeout=RPC_TIMEOUT,
        )
        project_after_retry = json.loads(volpkg.read_text())
        editable_entries_after_retry = [
            entry
            for entry in project_after_retry.get("segments", [])
            if isinstance(entry, dict) and entry.get("location") == str(editable_root)
        ]
        listed_after_retry, _ = client.call(
            "segments.list", {}, timeout=RPC_TIMEOUT
        )
        listed_active_after_retry = next(
            (
                segment
                for segment in listed_after_retry.get("segments", [])
                if segment.get("active")
            ),
            {},
        )
        results.record(
            "catalog_segment_create_editable_copy_retry",
            retried.get("created") is False
            and retried.get("alreadyExisted") is True
            and retried.get("activated") is True
            and retried.get("sourcePath") == str(catalog_segment)
            and retried.get("path") == str(editable_segment)
            and editable_entries_after_retry == editable_entries
            and project_after_retry.get("output_segments") == str(editable_root)
            and listed_active_after_retry.get("id") == segment_id
            and listed_active_after_retry.get("path") == str(editable_segment),
            f"result={retried} listed_active={listed_active_after_retry} "
            f"output_segments={project_after_retry.get('output_segments')} "
            f"entries={editable_entries_after_retry}",
        )

        enabled, _ = client.call(
            "segmentation.enable_editing",
            {"enabled": True},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "editable_copy_can_enable_editing",
            enabled.get("enabled") is True,
            f"result={enabled}",
        )
        client.call(
            "segmentation.enable_editing",
            {"enabled": False},
            timeout=RPC_TIMEOUT,
        )

        reopened, _ = client.call(
            "volume.open", {"path": str(volpkg)}, timeout=RPC_TIMEOUT
        )
        listed_after_reopen, _ = client.call(
            "segments.list", {}, timeout=RPC_TIMEOUT
        )
        reopened_segment = next(
            (
                segment
                for segment in listed_after_reopen.get("segments", [])
                if segment.get("id") == segment_id
            ),
            {},
        )
        client.call(
            "segments.activate",
            {"segmentId": segment_id},
            timeout=RPC_TIMEOUT,
        )
        enabled_after_reopen, _ = client.call(
            "segmentation.enable_editing",
            {"enabled": True},
            timeout=RPC_TIMEOUT,
        )
        project_after_reopen = json.loads(volpkg.read_text())
        results.record(
            "editable_copy_survives_reopen",
            reopened.get("opened") is True
            and reopened_segment.get("path") == str(editable_segment)
            and project_after_reopen.get("output_segments") == str(editable_root)
            and enabled_after_reopen.get("enabled") is True,
            f"reopened={reopened} segment={reopened_segment} "
            f"output_segments={project_after_reopen.get('output_segments')} "
            f"editing={enabled_after_reopen}",
        )
        client.call(
            "segmentation.enable_editing",
            {"enabled": False},
            timeout=RPC_TIMEOUT,
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "catalog_segment_editable_copy_workflow",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )


def check_c4(client: BridgeClient, results: Results, volpkg: str,
             broken_fiber_volpkg: str) -> None:
    # Liveness / dispatch sanity.
    try:
        state, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        results.record("liveness_state_get", isinstance(state, dict),
                       f"keys={list(state)[:4] if isinstance(state, dict) else state}")
    except Exception as e:  # noqa: BLE001
        results.record("liveness_state_get", False, f"{type(e).__name__}: {e}")

    # C4: rotate with a string `degrees` — parse rejection precedes the
    # axis-aligned-slice-mode state check.
    ok, detail = expect_param_error(
        client, "viewer.rotate",
        {"plane": "seg xz", "degrees": "abc"}, "degrees")
    results.record("c4_rotate_degrees_string", ok, detail)

    # C4: zoom with a non-numeric `factor`.
    ok, detail = expect_param_error(
        client, "viewer.zoom", {"factor": "abc"}, "factor")
    results.record("c4_zoom_factor_string", ok, detail)

    ok, detail = expect_param_error(
        client,
        "viewer.center_on_point",
        {
            "viewer": "__missing__",
            "point": {"x": "bad", "y": 1, "z": 1},
        },
        "point.x",
    )
    results.record("viewer_prevalidation_precedence", ok, detail)

    ok, detail = expect_param_error(
        client, "volume.open", {"path": 123}, "path")
    results.record("c4_volume_path_number", ok, detail)

    try:
        overlay_before_error, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        client.call(
            "viewer.set_overlay",
            {"volumeId": "__missing__"},
            timeout=RPC_TIMEOUT,
        )
        results.record(
            "viewer_overlay_unknown_volume",
            False,
            "expected -32007, got a result",
        )
    except BridgeError as error:
        overlay_after_error, _ = client.call(
            "viewer.get_overlay", {}, timeout=RPC_TIMEOUT)
        results.record(
            "viewer_overlay_unknown_volume",
            error.code == -32007
            and overlay_after_error == overlay_before_error,
            f"returned code={error.code} "
            f"state_unchanged={overlay_after_error == overlay_before_error}",
        )
    except Exception as error:  # noqa: BLE001
        results.record(
            "viewer_overlay_unknown_volume",
            False,
            f"unexpected {type(error).__name__}: {error}",
        )

    ok, detail = expect_param_error(
        client, "state.get", ["not", "an", "object"], "params")
    results.record("c4_array_params", ok, detail)

    ok, detail = expect_param_error(
        client, "job.status", {"jobId": 123}, "jobId")
    results.record("c4_job_id_number", ok, detail)

    ok, detail = expect_param_error(
        client,
        "segmentation.grow",
        {"direction": "all", "steps": 1.5},
        "steps",
    )
    results.record("c4_int_fractional_rejected", ok, detail)

    try:
        state2, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        results.record("c4_no_corruption_state_get", isinstance(state2, dict),
                       "bridge still answers after malformed requests")
    except Exception as e:  # noqa: BLE001
        results.record("c4_no_corruption_state_get", False, f"{type(e).__name__}: {e}")

    before = None
    try:
        before, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        client.call(
            "volume.open",
            {"path": "/vc3d-smoke/does-not-exist.volpkg.json"},
            timeout=RPC_TIMEOUT,
        )
        results.record("volume_open_failure_is_headless", False,
                       "expected a bridge error, got a result")
    except BridgeError as e:
        results.record("volume_open_failure_is_headless", e.code == -32005,
                       f"returned code={e.code} without blocking")
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_failure_is_headless", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        preserved = before is not None and after.get("vpkg") == before.get("vpkg")
        results.record("volume_open_failure_preserves_project", preserved,
                       f"vpkg unchanged={preserved}")
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_failure_preserves_project", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        client.call(
            "volume.open",
            {"path": volpkg, "volumeId": "missing-volume"},
            timeout=RPC_TIMEOUT,
        )
        results.record("volume_open_unknown_id_preserves_project", False,
                       "expected a bridge error, got a result")
    except BridgeError as e:
        after, _ = client.call("state.get", {}, timeout=RPC_TIMEOUT)
        preserved = before is not None and after.get("vpkg") == before.get("vpkg")
        results.record("volume_open_unknown_id_preserves_project",
                       e.code == -32007 and preserved,
                       f"code={e.code} vpkg unchanged={preserved}")
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_unknown_id_preserves_project", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        opened, _ = client.call(
            "volume.open", {"path": broken_fiber_volpkg}, timeout=RPC_TIMEOUT)
        results.record(
            "volume_open_broken_fibers_is_headless",
            opened.get("opened") is True,
            "broken branch metadata was kept without opening a repair dialog",
        )
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_broken_fibers_is_headless", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        saved, _ = client.call("fiber.save", {}, timeout=RPC_TIMEOUT)
        results.record("fiber_save_completion_response", saved.get("saved") is True,
                       f"saved={saved.get('saved')}")
    except Exception as e:  # noqa: BLE001
        results.record("fiber_save_completion_response", False,
                       f"unexpected {type(e).__name__}: {e}")


UUID_TEXT = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def check_legacy_fiber_uuid(client: BridgeClient, results: Results,
                            fiber_volpkg: str, fiber_file: Path) -> None:
    """Resolve missing legacy identity and shape after volume activation,
    then preserve both across save and reopen."""
    def stored_metadata() -> tuple:
        data = json.loads(fiber_file.read_text())
        return data.get("uuid"), data.get("coordinate_base_shape_zyx")

    try:
        original = json.loads(fiber_file.read_text())
        client.call("volume.open", {"path": fiber_volpkg}, timeout=RPC_TIMEOUT)
        first, first_shape = stored_metadata()
        client.call("fiber.save", {}, timeout=RPC_TIMEOUT)
        after_save, saved_shape = stored_metadata()
        client.call("volume.open", {"path": fiber_volpkg}, timeout=RPC_TIMEOUT)
        after_reopen, reopened_shape = stored_metadata()
        results.record(
            "legacy_fiber_uuid_persisted",
            isinstance(first, str) and UUID_TEXT.fullmatch(first) is not None
            and first == after_save == after_reopen,
            f"on_load={first} after_save={after_save} after_reopen={after_reopen}",
        )
        results.record(
            "legacy_fiber_coordinate_shape_persisted",
            first_shape == saved_shape == reopened_shape == [16, 16, 16],
            f"on_load={first_shape} after_save={saved_shape} "
            f"after_reopen={reopened_shape}",
        )
        reopened = json.loads(fiber_file.read_text())
        results.record(
            "legacy_fiber_geometry_preserved",
            original["control_points"] == [
                point["position"] for point in reopened["control_points"]]
            and original["line_points"] == reopened["line_points"],
            "metadata backfill preserves stored geometry",
        )
    except Exception as e:  # noqa: BLE001
        results.record("legacy_fiber_uuid_persisted", False,
                       f"unexpected {type(e).__name__}: {e}")


def check_c2_oversized(sock_path: str, results: Results) -> None:
    """Send a single unterminated line > 1 MiB; the server must drop that socket
    while a separate fresh connection keeps working."""
    over = 2 * 1024 * 1024  # > kMaxLineBytes (1 MiB), no newline
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(RPC_TIMEOUT)
    closed_by_server = False
    detail = ""
    try:
        raw.connect(sock_path)
        # The server may drop us with EPIPE/ECONNRESET mid-send, or via EOF on read — either proves it.
        chunk = b" " * 65536
        sent = 0
        try:
            while sent < over:
                sent += raw.send(chunk)
            # Fully sent without error: the server must close the read side (EOF).
            deadline = time.monotonic() + RPC_TIMEOUT
            while time.monotonic() < deadline:
                try:
                    data = raw.recv(65536)
                except socket.timeout:
                    break
                if data == b"":
                    closed_by_server = True
                    break
            detail = ("server closed the oversized client (EOF)"
                      if closed_by_server else "server did not close within 10s")
        except (BrokenPipeError, ConnectionResetError) as e:
            # Server closed the socket while we were still sending the payload.
            closed_by_server = True
            detail = f"server dropped mid-send ({type(e).__name__}) after {sent} bytes"
        results.record("c2_oversized_socket_dropped", closed_by_server, detail)
    except Exception as e:  # noqa: BLE001
        results.record("c2_oversized_socket_dropped", False,
                       f"{type(e).__name__}: {e}")
    finally:
        raw.close()

    # A separate fresh connection must be unaffected.
    try:
        fresh = BridgeClient(sock_path, connect_timeout=RPC_TIMEOUT)
        try:
            state, _ = fresh.call("state.get", {}, timeout=RPC_TIMEOUT)
            results.record("c2_other_client_unaffected", isinstance(state, dict),
                           "fresh connection still answers state.get")
        finally:
            fresh.close()
    except Exception as e:  # noqa: BLE001
        results.record("c2_other_client_unaffected", False,
                       f"{type(e).__name__}: {e}")


def check_c2_suffix_bypass(sock_path: str, results: Results) -> None:
    """A VALID line followed by a >1 MiB UNTERMINATED suffix must still get the
    client dropped: the pre-loop bound check is skipped when any newline is
    present, so the residual has to be bounded after the framed lines are
    consumed. Send '{}\\n' + >1 MiB with no further newline."""
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(RPC_TIMEOUT)
    closed_by_server = False
    detail = ""
    try:
        raw.connect(sock_path)
        # A well-formed (if unknown-method) line first, then an oversized tail.
        raw.sendall(b'{"jsonrpc":"2.0","id":1,"method":"state.get","params":{}}\n')
        over = 2 * 1024 * 1024
        chunk = b" " * 65536
        sent = 0
        try:
            while sent < over:
                sent += raw.send(chunk)
            deadline = time.monotonic() + RPC_TIMEOUT
            while time.monotonic() < deadline:
                try:
                    data = raw.recv(65536)
                except socket.timeout:
                    break
                if data == b"":
                    closed_by_server = True
                    break
            detail = ("server dropped the client after the oversized suffix (EOF)"
                      if closed_by_server else "server did not drop within 10s")
        except (BrokenPipeError, ConnectionResetError) as e:
            closed_by_server = True
            detail = f"server dropped mid-send ({type(e).__name__}) after {sent} bytes"
        results.record("c9_suffix_bypass_dropped", closed_by_server, detail)
    except Exception as e:  # noqa: BLE001
        results.record("c9_suffix_bypass_dropped", False, f"{type(e).__name__}: {e}")
    finally:
        raw.close()

    # Bridge still serves other clients after bounding the suffix.
    try:
        fresh = BridgeClient(sock_path, connect_timeout=RPC_TIMEOUT)
        try:
            state, _ = fresh.call("state.get", {}, timeout=RPC_TIMEOUT)
            results.record("c9_other_client_unaffected", isinstance(state, dict),
                           "fresh connection still answers after suffix drop")
        finally:
            fresh.close()
    except Exception as e:  # noqa: BLE001
        results.record("c9_other_client_unaffected", False,
                       f"{type(e).__name__}: {e}")


def _inode(path: str):
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


def check_c3_live_probe(binary: str, name: str, sock_path: str,
                        first_client: BridgeClient, config_home: str,
                        results: Results) -> None:
    """Launch a second server on the same name; it MUST refuse (probe the live
    endpoint and leave it alone). A refusal means: the second exits nonzero /
    logs a collision, never prints its own handshake, and the live socket file is
    NOT replaced (same inode). Detecting takeover requires the inode check — an
    already-established client connection and a bare path-exists check both
    survive an unlink+re-listen and would mask the strand."""
    ino_before = _inode(sock_path)
    second = launch(binary, name, config_home=config_home)
    exit_code = None
    took_over = False
    try:
        # Boot is sub-second; wait for a terminal signal: the second either exits
        # (refused) or prints its own handshake / replaces the socket (took over).
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            exit_code = second.exit_code()
            if exit_code is not None:
                break
            if any(parse_handshake_line(ln) for ln in second.tail_log(120)):
                took_over = True
                break
            if _inode(sock_path) not in (ino_before, None):
                took_over = True
                break
            time.sleep(0.2)
        ino_after = _inode(sock_path)
        inode_changed = (ino_after is not None and ino_after != ino_before)
        logged = any("failed to listen" in ln.lower() for ln in second.tail_log(120))
        refused = (not took_over and not inode_changed
                   and ((exit_code is not None and exit_code != 0) or logged))
        results.record(
            "c3_second_server_refused", refused,
            f"exit_code={exit_code} printed_handshake={took_over} "
            f"logged_collision={logged} inode_changed={inode_changed} "
            f"(ino {ino_before}->{ino_after})")
        # Explicit, separately-scored takeover assertion (the live socket must
        # not be unlinked+recreated).
        results.record(
            "c3_live_socket_not_unlinked", not inode_changed and not took_over,
            "live socket preserved" if not inode_changed
            else "live socket was unlinked + re-listened by the second server")
    finally:
        second.terminate()

    # The already-connected client survives regardless (existing fd), so this is
    # only a weak liveness signal, not proof the endpoint wasn't reclaimed.
    try:
        state, _ = first_client.call("state.get", {}, timeout=RPC_TIMEOUT)
        results.record("c3_first_conn_still_answers", isinstance(state, dict),
                       "pre-existing connection still answers state.get")
    except Exception as e:  # noqa: BLE001
        results.record("c3_first_conn_still_answers", False,
                       f"{type(e).__name__}: {e}")


def main() -> int:
    global RPC_TIMEOUT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vc3d", default=str(DEFAULT_VC3D_BIN),
                    help="path to the VC3D binary")
    ap.add_argument(
        "--rpc-timeout",
        type=float,
        default=RPC_TIMEOUT,
        help="per-call bridge timeout in seconds (default: 10)",
    )
    ap.add_argument(
        "--update-description-snapshot",
        action="store_true",
        help="rewrite the checked-in rpc.describe snapshot from the live binary",
    )
    args = ap.parse_args()
    if args.rpc_timeout <= 0:
        ap.error("--rpc-timeout must be positive")
    RPC_TIMEOUT = args.rpc_timeout

    binary = args.vc3d
    if not os.path.exists(binary):
        log(f"VC3D binary not found: {binary}")
        print(json.dumps({"ok": False, "error": f"binary not found: {binary}"}))
        return 2

    results = Results()
    with tempfile.TemporaryDirectory(prefix="vc3d-smoke-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        volpkg = create_test_project(tmp_path)
        broken_volpkg = tmp_path / "broken.volpkg.json"
        broken_volpkg.write_text(json.dumps({"name": "broken", "version": 1}))
        fiber_dir = Path(tmp_dir) / "fibers" / broken_volpkg.name
        fiber_dir.mkdir(parents=True)
        (fiber_dir / "fiber_smoke.json").write_text(json.dumps({
            "type": "vc3d_fiber",
            "version": 1,
            "username": "smoke",
            "started_at": "2026-01-01T00:00:00Z",
            "sequence": 1,
            "generation": 1,
            "control_points": [[0.0, 0.0, 0.0]],
            "line_points": [],
            "branches": [{}],
        }))
        # Use a separate clean project: broken-link fixtures deliberately
        # defer all load-time repair writes, including UUID backfills.
        legacy_fiber_volpkg = tmp_path / "legacy-fiber.volpkg.json"
        legacy_fiber_volpkg.write_text(json.dumps({
            "name": "legacy-fiber",
            "version": 1,
            "volumes": ["volumes"],
        }))
        legacy_fiber_dir = tmp_path / "fibers" / legacy_fiber_volpkg.name
        legacy_fiber_dir.mkdir(parents=True)
        legacy_fiber = legacy_fiber_dir / "fiber_legacy.json"
        # Both fields are intentionally absent: package fibers load before
        # volume activation, which must complete the deferred metadata save.
        legacy_fiber.write_text(json.dumps({
            "type": "vc3d_fiber",
            "version": 1,
            "username": "smoke",
            "started_at": "2026-01-01T00:00:00Z",
            "sequence": 2,
            "generation": 1,
            "control_points": [[10.0, 10.0, 10.0], [10.0, 10.0, 20.0]],
            "line_points": [[10.0, 10.0, 10.0], [10.0, 10.0, 15.0],
                            [10.0, 10.0, 20.0]],
            "branches": [],
        }))
        name = unique_name("main")
        config_home = tmp_path / "config-main"
        proc = launch(binary, name, str(volpkg), str(config_home))
        client = None
        try:
            sock_path = proc.wait_for_handshake(timeout=60.0)
            log(f"handshake: name={name} path={sock_path}")
            client = BridgeClient(sock_path, connect_timeout=RPC_TIMEOUT)

            check_rpc_describe(
                client,
                results,
                update_snapshot=args.update_description_snapshot,
            )
            check_viewer_normalization(client, results)
            require_running(proc, results, "viewer normalization")
            check_canvas_normalization(client, results)
            check_viewer_registry(client, results)
            check_workspaces(client, results)
            check_canvas_input(sock_path, client, results)
            require_running(proc, results, "canvas input")
            check_state_changes(sock_path, client, results)
            check_ui_inspect(client, results)
            check_screenshot_targets(client, results)

            check_volume_attach(client, results, tmp_path, volpkg)
            check_segments_attach(client, results, tmp_path, volpkg)
            check_open_data_editable_copy(client, results, tmp_path, volpkg)
            check_c4(client, results, str(volpkg), str(broken_volpkg))
            check_legacy_fiber_uuid(client, results, str(legacy_fiber_volpkg),
                                    legacy_fiber)
            check_project_create(client, results, tmp_path)
            check_c2_oversized(sock_path, results)
            check_c2_suffix_bypass(sock_path, results)
            check_c3_live_probe(
                binary,
                name,
                sock_path,
                client,
                str(config_home),
                results,
            )

            results.record("process_alive", proc.is_running(),
                           f"running={proc.is_running()}")
        except Exception as e:  # noqa: BLE001
            results.record("harness", False, f"{type(e).__name__}: {e}")
        finally:
            if client is not None:
                client.close()
            proc.terminate()

    summary = {
        "ok": results.ok,
        "checks": results.checks,
    }
    print(json.dumps(summary))
    return 0 if results.ok else 1


if __name__ == "__main__":
    sys.exit(main())
