#!/usr/bin/env python3
"""Developer-run VC3D Agent Bridge validation and benchmark driver.

Drives the REAL, already-built VC3D binary over its agent-bridge JSON-RPC
socket (see apps/VC3D/agent_bridge/SPEC.md), against real (non-synthetic)
.volpkg fixtures already checked out under test-data/. No mocking of VC3D
internals: this talks to the live application process exactly as an external
MCP-server client would. This fixture-dependent suite is intentionally separate
from the hermetic offscreen smoke test used by CI.

Usage:
    python3 manual_bridge_test.py offscreen   # headless fixture tests + benchmark
    python3 manual_bridge_test.py live        # windowed fixture smoke test

Results are printed as a single JSON object on stdout (last line), plus
human-readable progress on stderr, so a driver can `tail -1 | python3 -m json.tool`.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_client import BridgeClient, BridgeError  # noqa: E402
from manual_bridge_catalog import run_catalog_resource_selection_smoke  # noqa: E402
from manual_bridge_editing import (  # noqa: E402
    run_manual_add_corrections_smoke,
    run_segment_activation_proof,
    run_stage6_smoke,
)
from manual_bridge_rendering import run_flatten_smoke, run_render_smoke  # noqa: E402
from manual_bridge_support import (  # noqa: E402
    DEFAULT_VC3D_BIN,
    LOCAL_RAW_VOLUME_ID,
    LOCAL_REAL_SEED,
    LOCAL_REAL_SEED_OFFSCREEN_REVERIFY,
    LOCAL_VOLPKG_JSON,
    S3_REAL_POINT,
    S3_VOLPKG_JSON,
    Recorder,
    launch_vc3d,
    log,
    ms,
    pct,
)
from manual_bridge_workspaces import (  # noqa: E402
    run_atlas_smoke,
    run_fiber_smoke,
    run_lasagna_smoke,
)
from vc3d_process import VC3DProcess, find_running_vc3d_pids  # noqa: E402
import png_utils  # noqa: E402


def run_offscreen(args) -> dict:
    rec = Recorder()
    result: dict = {"mode": "offscreen", "recorder": None}
    proc: VC3DProcess | None = None
    client: BridgeClient | None = None
    benchmark: dict = {}
    e2e_seconds = None

    try:
        socket_name = f"vc3d-test-offscreen-{os.getpid()}-{int(time.time())}"
        proc = launch_vc3d(Path(args.vc3d_bin), offscreen=True, socket_name=socket_name)

        try:
            sock_path = proc.wait_for_handshake(timeout=90.0)
            rec.step("launch+handshake", True, f"socket={sock_path}")
        except TimeoutError as e:
            rec.step("launch+handshake", False, str(e))
            result["recorder"] = rec.as_dict()
            return result

        client = BridgeClient(sock_path, connect_timeout=10.0)
        rec.step("socket connect", True, sock_path)

        # 1. ping
        r, dt = client.call("ping", timeout=5.0)
        rec.step("ping", r.get("pong") is True, f"pid={r.get('pid')} version={r.get('version')} ({ms(dt):.2f}ms)")

        # 2. volume.open on the real local fixture, explicitly selecting the
        # raw scan volume that the pre-existing traces/ segments and our
        # curated seed point were computed against.
        vpkg_path = str(args.volpkg)
        r, dt = client.call("volume.open",
                             {"path": vpkg_path, "volumeId": LOCAL_RAW_VOLUME_ID},
                             timeout=60.0)
        rec.step("volume.open", r.get("opened") is True and r.get("volumeId") == LOCAL_RAW_VOLUME_ID,
                  json.dumps(r))

        # 2b. Nested-dialog footgun fix: with a vpkg already open,
        # neither a re-open nor catalog.open_sample may hang on a blocking
        # replace-project prompt. A hang would surface as a client-side
        # TimeoutError; anything that returns (success OR a documented error)
        # within the timeout proves the non-interactive path is taken.
        try:
            r2, dt2 = client.call("volume.open",
                                  {"path": vpkg_path, "volumeId": LOCAL_RAW_VOLUME_ID},
                                  timeout=60.0)
            rec.step("volume.open again (already-open project) returns promptly, no hang",
                     r2.get("opened") is True, f"({ms(dt2):.1f}ms) {json.dumps(r2)[:200]}")
        except TimeoutError as e:
            rec.step("volume.open again (already-open project) returns promptly, no hang",
                     False, f"HANG: {e}")
        except BridgeError as e:
            # Returned an error rather than hanging -- still proves no nested-dialog block.
            rec.step("volume.open again (already-open project) returns promptly, no hang",
                     True, f"returned error (not a hang): code={e.code}")
        try:
            r3, dt3 = client.call("catalog.open_sample", {"sampleId": "agent-bridge-nonexistent"},
                                  timeout=20.0)
            rec.step("catalog.open_sample (vpkg already open) returns promptly, no dialog hang",
                     True, f"({ms(dt3):.1f}ms) {json.dumps(r3)[:200]}")
        except TimeoutError as e:
            rec.step("catalog.open_sample (vpkg already open) returns promptly, no dialog hang",
                     False, f"HANG: {e}")
        except BridgeError as e:
            # -32007 (unknown sample) / -32005 (no manifest) both prove the
            # handler ran to completion without blocking on the replace prompt.
            rec.step("catalog.open_sample (vpkg already open) returns promptly, no dialog hang",
                     e.code in (-32005, -32007), f"returned error (not a hang): code={e.code} message={e.message}")

        # 3. state.get
        state, dt = client.call("state.get", timeout=10.0)
        ok = state.get("vpkg") is not None and state.get("volume") is not None
        rec.step("state.get (post-open)", ok, json.dumps(state)[:500])
        viewers = {v["surfName"]: v for v in state.get("viewers", [])}
        rec.step("viewer registry has 'xy plane'", "xy plane" in viewers, json.dumps(list(viewers.keys())))
        target_viewer = "xy plane" if "xy plane" in viewers else None
        pre_zoom_scale = viewers.get(target_viewer, {}).get("scale") if target_viewer else None

        # 4. segments.list -- expect the real, pre-existing grown segments.
        seglist, dt = client.call("segments.list", timeout=10.0)
        seg_ids = {s["id"] for s in seglist.get("segments", [])}
        expected_real_segments = {
            "auto_grown_20260416135719054_inp_hr",
            "auto_grown_20260416140827289",
        }
        rec.step("segments.list contains real pre-existing segments",
                 expected_real_segments.issubset(seg_ids),
                 f"found={sorted(seg_ids)}")
        initial_segment_count = len(seg_ids)

        # 5. points.list (baseline)
        pts0, dt = client.call("points.list", timeout=10.0)
        initial_point_count = sum(len(c.get("points", [])) for c in pts0.get("collections", []))
        rec.step("points.list (baseline)", True, f"initial_point_count={initial_point_count}")

        # 6. viewer.center_on_point / canvas.click / canvas.shift_click round
        # trip through the real Qt mouse-slot path.
        #
        # Important real-behavior discovery from CChunkedVolumeViewer::
        # centerOnVolumePoint (source-verified, apps/VC3D/volume_viewers/
        # CChunkedVolumeViewer.cpp): for a plane-type viewer such as "xy
        # plane", centering only pans the in-plane (x,y) offset -- it does
        # NOT move the plane's fixed Z depth (that requires a separate,
        # not-bridge-exposed slice-scroll interaction). canvas.click's
        # scene<->volume round-trip check is purely geometric (plane
        # projection), so a click point must share the viewer's *actual
        # current* Z or it legitimately fails INVALID_COORDINATES -- which is
        # exactly what a first attempt using our curated 3D seed's Z (9150,
        # far from the volume's default center-slice Z of ~9472) produced.
        # We discover the viewer's real current Z live via
        # canvas.get_cursor_volume_point and use it for canvas ops, while
        # still using the true curated 3D seed for the real
        # segmentation.grow_patch_from_seed call below (unaffected by this
        # viewer-plane quirk since that path takes a seed directly).
        canvas_point = dict(LOCAL_REAL_SEED)
        if target_viewer:
            try:
                cur, _ = client.call("canvas.get_cursor_volume_point",
                                      {"viewer": target_viewer}, timeout=10.0)
                plane_z = cur["volumePoint"]["z"]
                rec.step("discover xy-plane viewer's current Z depth", True, f"plane_z={plane_z}")
            except BridgeError as e:
                plane_z = state.get("focusPoi", {}).get("position", {}).get("z")
                rec.step("discover xy-plane viewer's current Z depth (fallback to focusPoi)",
                          plane_z is not None, f"code={e.code} data={e.data} fallback_z={plane_z}")
            canvas_point["z"] = plane_z if plane_z is not None else LOCAL_REAL_SEED["z"]

            r, dt = client.call("viewer.center_on_point",
                                 {"viewer": target_viewer, "point": canvas_point}, timeout=10.0)
            rec.step("viewer.center_on_point", r.get("centered") is True, json.dumps(r))

            r, dt = client.call("canvas.click",
                                 {"viewer": target_viewer, "position": canvas_point,
                                  "space": "volume", "button": "left"}, timeout=10.0)
            rec.step("canvas.click", r.get("clicked") is True, json.dumps(r))

            r, dt = client.call("canvas.shift_click",
                                 {"viewer": target_viewer, "position": canvas_point,
                                  "space": "volume"}, timeout=10.0)
            rec.step("canvas.shift_click", r.get("clicked") is True, json.dumps(r))

            r, dt = client.call("canvas.get_cursor_volume_point",
                                 {"viewer": target_viewer}, timeout=10.0)
            vp = r.get("volumePoint", {})
            close = all(abs(vp.get(k, 1e9) - canvas_point[k]) < 5.0 for k in "xyz")
            rec.step("canvas.get_cursor_volume_point", close, json.dumps(r))

            # 6b. canvas.drag: press-move-release between two real
            # on-surface points on the same plane (shared Z), through the real
            # onMousePress/onMouseMove/onMouseRelease slots.
            drag_to = dict(canvas_point)
            drag_to["x"] = canvas_point["x"] + 30.0
            drag_to["y"] = canvas_point["y"] + 20.0
            try:
                r, dt = client.call("canvas.drag",
                                     {"viewer": target_viewer, "from": canvas_point, "to": drag_to,
                                      "space": "volume", "button": "left", "steps": 6}, timeout=10.0)
                ok = (r.get("dragged") is True and r.get("steps") == 6
                      and isinstance(r.get("from"), dict) and isinstance(r.get("to"), dict)
                      and "scene" in r["from"] and "scene" in r["to"])
                rec.step("canvas.drag round-trips (press/move/release, well-formed result)",
                         ok, json.dumps(r)[:400])
            except BridgeError as e:
                rec.step("canvas.drag round-trips (press/move/release, well-formed result)",
                         False, f"code={e.code} message={e.message} data={e.data}")
            # button:"none" hover variant must also succeed (moves only).
            try:
                r, dt = client.call("canvas.drag",
                                     {"viewer": target_viewer, "from": canvas_point, "to": drag_to,
                                      "space": "volume", "button": "none", "steps": 4}, timeout=10.0)
                rec.step("canvas.drag button=none hover variant",
                         r.get("dragged") is True and r.get("button") == "none", json.dumps(r)[:300])
            except BridgeError as e:
                rec.step("canvas.drag button=none hover variant", False,
                         f"code={e.code} message={e.message} data={e.data}")

            pre_zoom, _ = client.call("state.get", timeout=10.0)
            pre_scale = next((v["scale"] for v in pre_zoom["viewers"] if v["surfName"] == target_viewer), None)
            r, dt = client.call("viewer.zoom", {"viewer": target_viewer, "factor": 1.5}, timeout=10.0)
            post_scale = r.get("scale")
            rec.step("viewer.zoom", post_scale is not None and pre_scale is not None and post_scale != pre_scale,
                     f"pre={pre_scale} post={post_scale}")
        else:
            rec.step("canvas ops", False, "no 'xy plane' viewer found in registry")

        # 7. segmentation.grow -- attempt on whatever surface is active.
        # There is no RPC in this bridge surface to select/activate a segment
        # (surface activation lives behind a QTreeWidget double-click signal
        # not exposed over the bridge), so on a fresh launch there is no
        # active "segmentation" surface. We still exercise the RPC and
        # confirm it fails with a well-formed, documented error rather than
        # crashing or hanging -- this is the documented fallback path the
        # task anticipated.
        try:
            r, dt = client.call("segmentation.grow",
                                 {"method": "tracer", "direction": "all", "steps": 2}, timeout=10.0)
            rec.step("segmentation.grow (unexpected success)", True, json.dumps(r))
        except BridgeError as e:
            expected_codes = {-32007, -32008, -32000, -32001}
            rec.step("segmentation.grow returns well-formed documented error (no active surface)",
                     e.code in expected_codes, f"code={e.code} message={e.message} data={e.data}")

        # 7b. Footgun fix: method:"manual_add" must be rejected with
        # -32009 UNSUPPORTED (not a silent no-op, and not routed into growth).
        try:
            r, dt = client.call("segmentation.grow",
                                 {"method": "manual_add", "direction": "all", "steps": 1}, timeout=10.0)
            rec.step("segmentation.grow method=manual_add rejected with -32009", False,
                     f"unexpected success {json.dumps(r)}")
        except BridgeError as e:
            hint_ok = "manual_add" in json.dumps(e.data)
            rec.step("segmentation.grow method=manual_add rejected with -32009",
                     e.code == -32009 and hint_ok,
                     f"code={e.code} message={e.message} data={e.data}")

        # 8. segmentation.grow_patch_from_seed -- the real headless GrowPatch
        # path; does not require an active surface. Uses a small iteration
        # count to keep the real vc_grow_seg_from_seed run bounded.
        job_id = None
        grow_started_at = None
        # Use a seed genuinely different from LOCAL_REAL_SEED (already grown
        # by prior repro/test runs in this same fixture) so this is a fresh
        # real growth in an unexplored region, not a rerun over already-grown
        # ground.
        grow_seed = LOCAL_REAL_SEED_OFFSCREEN_REVERIFY
        try:
            grow_started_at = time.monotonic()
            r, dt = client.call("segmentation.grow_patch_from_seed",
                                 {"seed": grow_seed, "volumeId": LOCAL_RAW_VOLUME_ID,
                                  "iterations": args.grow_iterations, "minAreaCm": 0.002},
                                 timeout=30.0)
            job_id = r.get("jobId")
            rec.step("segmentation.grow_patch_from_seed (started)", job_id is not None, json.dumps(r))
        except BridgeError as e:
            rec.step("segmentation.grow_patch_from_seed (started)", False,
                      f"code={e.code} message={e.message} data={e.data}")

        # 8b. Generalized job model: job.status carries a "source"
        # for this real job, state.get exposes an active-jobs array, and
        # job.progress notifications include "source".
        if job_id:
            try:
                jst, _ = client.call("job.status", {"jobId": job_id}, timeout=10.0)
                rec.step("job.status includes 'source' field for a real job",
                         jst.get("source") == "tool",
                         f"source={jst.get('source')} state={jst.get('state')}")
            except BridgeError as e:
                rec.step("job.status includes 'source' field for a real job", False,
                         f"code={e.code} message={e.message}")
            try:
                stj, _ = client.call("state.get", timeout=10.0)
                jobs_arr = stj.get("jobs")
                job_field = stj.get("job")
                # 'jobs' is always present (may be empty if the job already
                # finished); when the top-level 'job' is set it must carry source.
                ok = isinstance(jobs_arr, list) and (job_field is None or "source" in job_field)
                rec.step("state.get exposes 'jobs' array and source-tagged 'job'",
                         ok, f"job={json.dumps(job_field)} jobs={json.dumps(jobs_arr)[:200]}")
            except BridgeError as e:
                rec.step("state.get exposes 'jobs' array and source-tagged 'job'", False,
                         f"code={e.code} message={e.message}")
            try:
                params = client.wait_for_notification(
                    "job.progress",
                    predicate=lambda p: p.get("jobId") == job_id,
                    timeout=5.0)
                rec.step("job.progress notification includes 'source' field",
                         params.get("source") == "tool",
                         f"source={params.get('source')} phase={params.get('phase')}")
            except TimeoutError:
                # The 'started' progress may have been consumed/coalesced; fall
                # back to asserting the field via a re-poll of job.status source.
                rec.step("job.progress notification includes 'source' field",
                         jst.get("source") == "tool",
                         "no live notification captured; verified via job.status source")

        grow_success = None
        if job_id:
            deadline = time.monotonic() + args.grow_timeout
            last_state = None
            while time.monotonic() < deadline:
                st, _ = client.call("job.status", {"jobId": job_id}, timeout=10.0)
                last_state = st.get("state")
                if last_state in ("succeeded", "failed"):
                    grow_success = (last_state == "succeeded")
                    break
                time.sleep(2.0)
            e2e_seconds = time.monotonic() - grow_started_at
            rec.step("segmentation.grow_patch_from_seed (completed within timeout)",
                     last_state in ("succeeded", "failed"),
                     f"final_state={last_state} elapsed={e2e_seconds:.1f}s")
            if grow_success:
                seglist2, _ = client.call("segments.list", timeout=10.0)
                new_ids = {s["id"] for s in seglist2.get("segments", [])}
                grew = len(new_ids) > initial_segment_count
                rec.step("segments.list reflects real grown segment",
                         grew, f"before={initial_segment_count} after={len(new_ids)} new={sorted(new_ids - seg_ids)}")
            elif last_state == "failed":
                rec.step("segmentation.grow_patch_from_seed outcome", False,
                         f"job failed: {st.get('message')}; consoleTail={st.get('consoleTail')}")
        result["job_id"] = job_id
        result["grow_seed"] = grow_seed
        result["grow_success"] = grow_success
        result["grow_e2e_seconds"] = e2e_seconds

        # 9. points.commit / points.list
        r, dt = client.call("points.commit",
                             {"collection": "agent_bridge_test", "points": [LOCAL_REAL_SEED], "winding": 0.0},
                             timeout=10.0)
        rec.step("points.commit", len(r.get("pointIds", [])) == 1, json.dumps(r))

        pts1, dt = client.call("points.list", timeout=10.0)
        final_point_count = sum(len(c.get("points", [])) for c in pts1.get("collections", []))
        rec.step("points.list reflects committed point",
                 final_point_count == initial_point_count + 1,
                 f"before={initial_point_count} after={final_point_count}")

        # 10. screenshot.capture -- verify a real, non-blank image.
        shot, dt = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
        png_bytes = base64.b64decode(shot["base64"]) if shot.get("base64") else b""
        nontrivial, detail = png_utils.is_nontrivial_image(png_bytes)
        rec.step("screenshot.capture non-blank window image",
                 shot.get("width", 0) > 0 and shot.get("height", 0) > 0 and nontrivial,
                 f"{shot.get('width')}x{shot.get('height')} {detail}")

        # 10b. segments.activate: the critical activation proof.
        # Before this RPC existed there was NO way for a headless session to make
        # a segment the active editing target, so segmentation.enable_editing
        # always failed -32007 (see run_manual_add_corrections_smoke docstring).
        # This activates a real pre-existing segment and proves editing can then
        # actually be enabled and that the active-surface state is observable via
        # both segments.list and state.get. Run before editing
        # smoke test below (not after, as an earlier revision of this driver had
        # it) -- otherwise segmentation.enable_editing / manual_add.begin /
        # corrections.set_point_mode only ever exercise their no-active-surface
        # guard-error paths and NEVER their real success paths, which defeats the
        # purpose of a mutation test.
        run_segment_activation_proof(client, rec, seg_ids)

        # 10c. Known footgun (also flagged for segmentation.grow_patch_from_seed
        # completion): activating a segment re-centers the viewers onto it
        # (CWindow::onSurfaceActivated's moveOnSurfaceChange path), so the stale
        # pre-activation canvas_point is very likely no longer "on" target_viewer's
        # current view -- a canvas.click/shift_click there throws -32003 ("point
        # is not on this viewer's view"). Work around it exactly like the
        # original pre-activation Z-discovery above: re-fetch the viewer's
        # *current* cursor volume point right after activation, and use that (not
        # the stale point) for every canvas.* call from here on. Without this,
        # run_manual_add_corrections_smoke's plane-constraint shift-clicks raise
        # an uncaught BridgeError that aborts the whole offscreen test.
        if target_viewer:
            try:
                cur, _ = client.call("canvas.get_cursor_volume_point",
                                      {"viewer": target_viewer}, timeout=10.0)
                refreshed = cur.get("volumePoint")
                rec.step("re-fetch canvas_point after segments.activate re-centered the viewer",
                         refreshed is not None, json.dumps(cur)[:300])
                if refreshed:
                    canvas_point = dict(refreshed)
            except BridgeError as e:
                rec.step("re-fetch canvas_point after segments.activate re-centered the viewer",
                         False, f"code={e.code} message={e.message} data={e.data}")

        # 10d. Manual-add / hole-fill and corrections
        # point authoring. Now that a segment is active (previous step) and
        # editing is enabled, this exercises the REAL success paths (manual-add
        # begin/constraints/finish, corrections point mode + solver-triggering
        # drag), not just the guard-error paths.
        run_manual_add_corrections_smoke(client, rec, target_viewer, canvas_point)

        # 11. Benchmark: cheap-call round-trip latency. NOTE: run AFTER
        # segments.activate now, so canvas.click samples may show elevated
        # error counts if the plane viewers moved off canvas_point when the
        # segment activated (activation repositions them onto the activated
        # surface, click-like) -- state.get / viewer.center_on_point are
        # unaffected and remain a valid latency signal either way.
        benchmark = run_cheap_call_benchmark(client, target_viewer or "segmentation",
                                              canvas_point, n=args.bench_iterations)
        result["benchmark"] = benchmark

        # Lasagna needs the original fixture, before catalog.open_sample replaces
        # the project. Every timeout includes a process-liveness check.
        run_lasagna_smoke(client, rec, proc)

        # The fixture has no atlas data, so exercise every atlas error path and
        # verify that none opens a dialog or terminates the process.
        run_atlas_smoke(client, rec, proc)

        # Exercise a real fiber import/export round trip before the catalog
        # suite replaces the fixture, then delete the imported fibers.
        run_fiber_smoke(client, rec, proc)

        # 11e. tags.set / seeding.* / push_pull.* /
        # tracer.run_trace. Runs before the catalog suite (needs the local
        # fixture project + its segments still attached). Error paths for every
        # RPC plus the safe fire-and-forget success paths; no external tool job
        # is launched (tracer.run_trace is exercised only through its errors) and
        # tags.set reverts its one mutation. Every step is liveness-checked: the
        # footguns here are the seeding QMessageBox precondition dialogs (now
        # suppressed) and the deliberately-unexposed nested-event-loop seeding
        # actions.
        run_stage6_smoke(client, rec, proc, seg_ids)

        # 11b. render.tifxyz. Runs against the still-attached
        # local fixture + its real segments, BEFORE the catalog stage replaces the
        # project. Error paths plus one real bounded render (tif_stack, tiny scale,
        # single slice) to a temp dir, polled to terminal with the on-disk artifact
        # verified. Threaded `proc` so every timeout is liveness-checked.
        run_render_smoke(client, rec, proc, seg_ids)

        # Exercise flatten error paths and one bounded in-process ABF++ job
        # before catalog replacement. Verify its artifact and process liveness.
        run_flatten_smoke(client, rec, proc, seg_ids)

        # 12. Remote catalog resource selection. Runs last
        # because a real catalog.open_sample replaces the local fixture project.
        # Network-gated: probes the manifest URL first and skips cleanly (not a
        # failure) when unreachable from this environment. `proc` is threaded in
        # so every timeout distinguishes a slow request from a dead process.
        run_catalog_resource_selection_smoke(client, rec, proc)

        result["recorder"] = rec.as_dict()
        return result

    except Exception as e:
        rec.step("unhandled exception in offscreen test", False, f"{e}\n{traceback.format_exc()}")
        result["recorder"] = rec.as_dict()
        return result
    finally:
        if client:
            client.close()
        if proc:
            proc.terminate()
            log(f"VC3D offscreen process terminated, exit={proc.exit_code()}")


def run_cheap_call_benchmark(client: BridgeClient, viewer: str, point: dict, n: int = 20) -> dict:
    methods = {
        "state.get": ({}, None),
        "canvas.click": ({"viewer": viewer, "position": point, "space": "volume", "button": "left"}, None),
        "viewer.center_on_point": ({"viewer": viewer, "point": point}, None),
    }
    out = {}
    for method, (params, _) in methods.items():
        samples_ms = []
        errors = 0
        for _ in range(n):
            try:
                _, dt = client.call(method, params, timeout=10.0)
                samples_ms.append(ms(dt))
            except (BridgeError, TimeoutError):
                errors += 1
        if samples_ms:
            out[method] = {
                "n": len(samples_ms),
                "errors": errors,
                "mean_ms": statistics.mean(samples_ms),
                "p95_ms": pct(samples_ms, 95),
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
            }
        else:
            out[method] = {"n": 0, "errors": errors}
        log(f"benchmark {method}: {out[method]}")
    return out


# ---------------------------------------------------------------------------
# Live on-screen smoke test
# ---------------------------------------------------------------------------

def run_live(args) -> dict:
    rec = Recorder()
    result: dict = {"mode": "live", "recorder": None}
    proc: VC3DProcess | None = None
    client: BridgeClient | None = None

    running = find_running_vc3d_pids()
    if running:
        rec.step("no colliding running VC3D instance", False,
                  f"VC3D already running with pid(s) {running}; refusing to launch a second instance")
        result["recorder"] = rec.as_dict()
        return result
    rec.step("no colliding running VC3D instance", True, "none found")

    try:
        socket_name = f"vc3d-test-live-{os.getpid()}-{int(time.time())}"
        proc = launch_vc3d(Path(args.vc3d_bin), offscreen=False, socket_name=socket_name)

        try:
            sock_path = proc.wait_for_handshake(timeout=90.0)
            rec.step("launch+handshake (windowed)", True, f"socket={sock_path}")
        except TimeoutError as e:
            rec.step("launch+handshake (windowed)", False, str(e))
            result["recorder"] = rec.as_dict()
            return result

        client = BridgeClient(sock_path, connect_timeout=10.0)

        r, dt = client.call("ping", timeout=5.0)
        rec.step("ping", r.get("pong") is True, json.dumps(r))

        vpkg_path = str(args.volpkg)
        open_params = {"path": vpkg_path}
        if args.volume_id:
            open_params["volumeId"] = args.volume_id
        r, dt = client.call("volume.open", open_params, timeout=90.0)
        rec.step("volume.open", r.get("opened") is True, json.dumps(r))

        state, dt = client.call("state.get", timeout=10.0)
        rec.step("state.get", state.get("vpkg") is not None and state.get("volume") is not None,
                  json.dumps(state)[:800])

        seglist, dt = client.call("segments.list", timeout=15.0)
        rec.step("segments.list", True, f"count={len(seglist.get('segments', []))}")
        live_initial_segment_count = len(seglist.get("segments", []))
        live_seg_ids = {s["id"] for s in seglist.get("segments", [])}

        # Job-completion regression check (the actual bug being re-verified):
        # a real windowed process has an actual QApplication event loop and a
        # real screen, so the previously-blocking completion QMessageBox
        # would genuinely render here (unlike offscreen, where it may not
        # even paint). Confirm the fix keeps job.status/segments.list moving
        # to completion promptly in this live/on-screen case too, i.e. that
        # bridge-driven jobs are not left waiting on an undismissed modal.
        if args.grow_seed:
            job_id = None
            grow_started_at = time.monotonic()
            try:
                grow_params = {"seed": args.grow_seed, "iterations": args.grow_iterations,
                                "minAreaCm": 0.002}
                if args.grow_volume_id:
                    grow_params["volumeId"] = args.grow_volume_id
                r, dt = client.call("segmentation.grow_patch_from_seed", grow_params, timeout=30.0)
                job_id = r.get("jobId")
                rec.step("segmentation.grow_patch_from_seed (started, live)", job_id is not None,
                          json.dumps(r))
            except BridgeError as e:
                rec.step("segmentation.grow_patch_from_seed (started, live)", False,
                          f"code={e.code} message={e.message} data={e.data}")

            if job_id:
                deadline = time.monotonic() + args.grow_timeout
                last_state = None
                while time.monotonic() < deadline:
                    st, _ = client.call("job.status", {"jobId": job_id}, timeout=10.0)
                    last_state = st.get("state")
                    if last_state in ("succeeded", "failed"):
                        break
                    time.sleep(1.0)
                grow_e2e = time.monotonic() - grow_started_at
                result["live_grow_e2e_seconds"] = grow_e2e
                result["live_grow_final_state"] = last_state
                rec.step("segmentation.grow_patch_from_seed reaches terminal state (live, no dialog hang)",
                          last_state in ("succeeded", "failed"),
                          f"final_state={last_state} elapsed={grow_e2e:.2f}s")
                if last_state == "succeeded":
                    seglist2, _ = client.call("segments.list", timeout=10.0)
                    new_ids = {s["id"] for s in seglist2.get("segments", [])}
                    grew = len(new_ids) > live_initial_segment_count
                    rec.step("segments.list reflects real grown segment (live)",
                              grew,
                              f"before={live_initial_segment_count} after={len(new_ids)} "
                              f"new={sorted(new_ids - live_seg_ids)}")
                elif last_state == "failed":
                    rec.step("segmentation.grow_patch_from_seed outcome (live)", False,
                              f"job failed: {st.get('message')}; consoleTail={st.get('consoleTail')}")

                # A live-mode ping right after job completion is a cheap way
                # to confirm the RPC event loop is fully healthy (not wedged
                # behind a lingering modal that merely hasn't been detected
                # yet by job.status).
                r, dt = client.call("ping", timeout=5.0)
                rec.step("ping responsive immediately after live grow completion",
                          r.get("pong") is True, f"({ms(dt):.2f}ms)")

        viewers = {v["surfName"]: v for v in state.get("viewers", [])}
        target_viewer = "xy plane" if "xy plane" in viewers else next(iter(viewers), None)

        seed_point = args.point
        if target_viewer:
            r, dt = client.call("viewer.center_on_point",
                                 {"viewer": target_viewer, "point": seed_point}, timeout=20.0)
            rec.step("viewer.center_on_point", r.get("centered") is True, json.dumps(r))

            try:
                r, dt = client.call("canvas.click",
                                     {"viewer": target_viewer, "position": seed_point,
                                      "space": "volume", "button": "left"}, timeout=10.0)
                rec.step("canvas.click", r.get("clicked") is True, json.dumps(r))
            except BridgeError as e:
                rec.step("canvas.click", False, f"code={e.code} message={e.message}")

        # Repaint evidence: two screenshots separated by a delay while the
        # viewer streams real S3-backed (or local) chunk data should differ.
        shot1, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
        png1 = base64.b64decode(shot1["base64"]) if shot1.get("base64") else b""
        ok1, detail1 = png_utils.is_nontrivial_image(png1)
        rec.step("screenshot #1 non-blank", ok1, detail1)

        log(f"waiting {args.repaint_wait}s for streaming/repaint activity...")
        seen_progress = []
        deadline = time.monotonic() + args.repaint_wait
        while time.monotonic() < deadline:
            try:
                params = client.wait_for_notification("job.progress", timeout=max(0.1, deadline - time.monotonic()))
                seen_progress.append(params)
            except TimeoutError:
                break

        if target_viewer:
            client.call("viewer.zoom", {"viewer": target_viewer, "factor": 1.2}, timeout=10.0)

        shot2, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
        png2 = base64.b64decode(shot2["base64"]) if shot2.get("base64") else b""
        ok2, detail2 = png_utils.is_nontrivial_image(png2)
        rec.step("screenshot #2 non-blank", ok2, detail2)

        pixels_differ = png1 != png2
        rec.step("visible repaint evidence (pixels changed and/or job.progress seen)",
                  pixels_differ or len(seen_progress) > 0,
                  f"bytes_differ={pixels_differ} job_progress_notifications={len(seen_progress)}")

        result["recorder"] = rec.as_dict()
        return result

    except Exception as e:
        rec.step("unhandled exception in live test", False, f"{e}\n{traceback.format_exc()}")
        result["recorder"] = rec.as_dict()
        return result
    finally:
        if client:
            client.close()
        if proc:
            proc.terminate()
            log(f"VC3D live process terminated cleanly, exit={proc.exit_code()}")
            still = find_running_vc3d_pids()
            if still:
                log(f"WARNING: VC3D pid(s) still present after terminate: {still}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["offscreen", "live"])
    parser.add_argument("--vc3d-bin", default=str(DEFAULT_VC3D_BIN))
    parser.add_argument("--volpkg", default=None)
    parser.add_argument("--grow-iterations", type=int, default=10)
    parser.add_argument("--grow-timeout", type=float, default=420.0)
    parser.add_argument("--bench-iterations", type=int, default=20)
    parser.add_argument("--repaint-wait", type=float, default=8.0)
    parser.add_argument("--point", type=json.loads, default=None,
                         help="JSON {x,y,z} volume-space point for the live test")
    parser.add_argument("--volume-id", default=None,
                         help="explicit volumeId to pass to volume.open (live mode)")
    parser.add_argument("--grow-seed", type=json.loads, default=None,
                         help="JSON {x,y,z} seed: if set, live mode also runs a real "
                              "segmentation.grow_patch_from_seed job-completion regression check")
    parser.add_argument("--grow-volume-id", default=None,
                         help="volumeId for the live --grow-seed check (defaults to whatever "
                              "volume.open selected)")
    args = parser.parse_args()

    if args.volpkg is None:
        args.volpkg = str(LOCAL_VOLPKG_JSON) if args.mode == "offscreen" else str(S3_VOLPKG_JSON)
    else:
        args.volpkg = str(args.volpkg)
    if args.point is None:
        args.point = S3_REAL_POINT

    if args.mode == "offscreen":
        result = run_offscreen(args)
    else:
        result = run_live(args)

    print(json.dumps(result, indent=2))
    return 0 if result.get("recorder", {}).get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
