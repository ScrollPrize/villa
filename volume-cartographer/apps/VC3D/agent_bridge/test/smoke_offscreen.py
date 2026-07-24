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

    python3 smoke_offscreen.py [--vc3d /path/to/VC3D]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
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


def create_test_project(root: Path) -> Path:
    volume = root / "volumes" / "vol1"
    level = volume / "0"
    level.mkdir(parents=True)
    (volume / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (volume / ".zattrs").write_text(json.dumps({}))
    (volume / "meta.json").write_text(json.dumps({
        "type": "vol",
        "uuid": "vol1",
        "name": "vol1",
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

    volpkg = root / "smoke.volpkg.json"
    volpkg.write_text(json.dumps({
        "name": "smoke",
        "version": 1,
        "volumes": ["volumes"],
    }))
    return volpkg


def expect_param_error(client: BridgeClient, method: str, params: object,
                       expect_param: str) -> tuple[bool, str]:
    """Calls `method`; expects BridgeError(-32602) with data.param == expect_param."""
    try:
        result, _ = client.call(method, params, timeout=10.0)
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
        client.call(method, params, timeout=10.0)
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
            timeout=10.0,
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
        description, _ = client.call("rpc.describe", {}, timeout=10.0)
        snapshot = {"methods": description.get("methods", {})}
        coverage = description.get("coverage", {})
        complete = (
            description.get("undocumented") == []
            and coverage.get("described") == 117
            and coverage.get("registered") == 117
            and coverage.get("complete") is True
            and len(snapshot["methods"]) == 117
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
            timeout=10.0,
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
            timeout=10.0,
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
            timeout=10.0,
        )
        expect(overlay, "windowLow", 10)
        expect(overlay, "windowHigh", 11)
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
            timeout=10.0,
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


def check_project_create(
    client: BridgeClient,
    results: Results,
    root: Path,
) -> None:
    volume = root / "volumes" / "vol1"
    requested = root / "created" / "agent-project"
    created = requested.with_suffix(".volpkg.json")

    try:
        before, _ = client.call("state.get", {}, timeout=10.0)
        result, _ = client.call(
            "project.create",
            {
                "path": str(requested),
                "volume": str(volume),
                "tags": ["source:smoke"],
            },
            timeout=10.0,
        )
        after, _ = client.call("state.get", {}, timeout=10.0)
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
            timeout=10.0,
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
                timeout=10.0,
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
            timeout=10.0,
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


def check_c4(client: BridgeClient, results: Results, volpkg: str,
             broken_fiber_volpkg: str) -> None:
    # Liveness / dispatch sanity.
    try:
        state, _ = client.call("state.get", {}, timeout=10.0)
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
        client.call(
            "viewer.set_overlay",
            {"volumeId": "__missing__"},
            timeout=10.0,
        )
        results.record(
            "viewer_overlay_unknown_volume",
            False,
            "expected -32007, got a result",
        )
    except BridgeError as error:
        results.record(
            "viewer_overlay_unknown_volume",
            error.code == -32007,
            f"returned code={error.code}",
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
        state2, _ = client.call("state.get", {}, timeout=10.0)
        results.record("c4_no_corruption_state_get", isinstance(state2, dict),
                       "bridge still answers after malformed requests")
    except Exception as e:  # noqa: BLE001
        results.record("c4_no_corruption_state_get", False, f"{type(e).__name__}: {e}")

    before = None
    try:
        before, _ = client.call("state.get", {}, timeout=10.0)
        client.call(
            "volume.open",
            {"path": "/vc3d-smoke/does-not-exist.volpkg.json"},
            timeout=10.0,
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
        after, _ = client.call("state.get", {}, timeout=10.0)
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
            timeout=10.0,
        )
        results.record("volume_open_unknown_id_preserves_project", False,
                       "expected a bridge error, got a result")
    except BridgeError as e:
        after, _ = client.call("state.get", {}, timeout=10.0)
        preserved = before is not None and after.get("vpkg") == before.get("vpkg")
        results.record("volume_open_unknown_id_preserves_project",
                       e.code == -32007 and preserved,
                       f"code={e.code} vpkg unchanged={preserved}")
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_unknown_id_preserves_project", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        opened, _ = client.call(
            "volume.open", {"path": broken_fiber_volpkg}, timeout=10.0)
        results.record(
            "volume_open_broken_fibers_is_headless",
            opened.get("opened") is True,
            "broken branch metadata was kept without opening a repair dialog",
        )
    except Exception as e:  # noqa: BLE001
        results.record("volume_open_broken_fibers_is_headless", False,
                       f"unexpected {type(e).__name__}: {e}")

    try:
        saved, _ = client.call("fiber.save", {}, timeout=10.0)
        results.record("fiber_save_completion_response", saved.get("saved") is True,
                       f"saved={saved.get('saved')}")
    except Exception as e:  # noqa: BLE001
        results.record("fiber_save_completion_response", False,
                       f"unexpected {type(e).__name__}: {e}")


def check_c2_oversized(sock_path: str, results: Results) -> None:
    """Send a single unterminated line > 1 MiB; the server must drop that socket
    while a separate fresh connection keeps working."""
    over = 2 * 1024 * 1024  # > kMaxLineBytes (1 MiB), no newline
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(10.0)
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
            deadline = time.monotonic() + 10.0
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
        fresh = BridgeClient(sock_path, connect_timeout=10.0)
        try:
            state, _ = fresh.call("state.get", {}, timeout=10.0)
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
    raw.settimeout(10.0)
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
            deadline = time.monotonic() + 10.0
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
        fresh = BridgeClient(sock_path, connect_timeout=10.0)
        try:
            state, _ = fresh.call("state.get", {}, timeout=10.0)
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
        state, _ = first_client.call("state.get", {}, timeout=10.0)
        results.record("c3_first_conn_still_answers", isinstance(state, dict),
                       "pre-existing connection still answers state.get")
    except Exception as e:  # noqa: BLE001
        results.record("c3_first_conn_still_answers", False,
                       f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vc3d", default=str(DEFAULT_VC3D_BIN),
                    help="path to the VC3D binary")
    ap.add_argument(
        "--update-description-snapshot",
        action="store_true",
        help="rewrite the checked-in rpc.describe snapshot from the live binary",
    )
    args = ap.parse_args()

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
        name = unique_name("main")
        config_home = tmp_path / "config-main"
        proc = launch(binary, name, str(volpkg), str(config_home))
        client = None
        try:
            sock_path = proc.wait_for_handshake(timeout=60.0)
            log(f"handshake: name={name} path={sock_path}")
            client = BridgeClient(sock_path, connect_timeout=10.0)

            check_rpc_describe(
                client,
                results,
                update_snapshot=args.update_description_snapshot,
            )
            check_viewer_normalization(client, results)
            check_canvas_normalization(client, results)

            check_c4(client, results, str(volpkg), str(broken_volpkg))
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
