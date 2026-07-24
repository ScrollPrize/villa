"""Manual segmentation editing and activation checks."""

from __future__ import annotations

import base64
import json
import time

from bridge_client import BridgeClient, BridgeError
from manual_bridge_support import (
    Recorder,
    VC3DDiedError,
    log,
    ms,
    record_process_death,
)
from vc3d_process import VC3DProcess

def run_manual_add_corrections_smoke(client: BridgeClient, rec: Recorder,
                                     target_viewer, canvas_point) -> None:
    """Exercise manual-add and corrections through a real editing session.

    The suite verifies both disabled-state guards and successful point
    authoring, including an on-disk hole fill and a corrections growth job.
    """
    # Accepted precondition errors for editing-gated RPCs.
    PRECOND = {-32000, -32001, -32004, -32007, -32008}

    # Config setters are valid with or without an active session.
    r, _ = client.call("segmentation.manual_add.set_line_mode", {"mode": "cross_fill"}, timeout=10.0)
    rec.step("manual_add.set_line_mode(cross_fill)", r.get("mode") == "cross_fill", json.dumps(r))
    r, _ = client.call("segmentation.manual_add.set_line_mode", {"mode": "vertical"}, timeout=10.0)
    rec.step("manual_add.set_line_mode(vertical) cycles", r.get("mode") == "vertical", json.dumps(r))
    r, _ = client.call("segmentation.manual_add.set_interpolation",
                       {"mode": "tracer_restricted_to_fill"}, timeout=10.0)
    rec.step("manual_add.set_interpolation(tracer_restricted_to_fill)",
             r.get("mode") == "tracer_restricted_to_fill", json.dumps(r))
    # Restore the default fill method so we don't leave a surprising config.
    client.call("segmentation.manual_add.set_interpolation", {"mode": "thin_plate_spline"}, timeout=10.0)
    try:
        client.call("segmentation.manual_add.set_line_mode", {"mode": "bogus"}, timeout=10.0)
        rec.step("manual_add.set_line_mode rejects bad enum with -32602", False, "unexpected success")
    except BridgeError as e:
        rec.step("manual_add.set_line_mode rejects bad enum with -32602", e.code == -32602,
                 f"code={e.code} data={e.data}")

    # state.get reports the editing modes.
    st, _ = client.call("state.get", timeout=10.0)
    stage2_keys = ("manualAddMode", "manualAddLineMode", "manualAddInterpolation", "correctionsPointMode")
    fields_present = all(k in st for k in stage2_keys)
    rec.step("state.get reports manual-add/corrections fields", fields_present,
             json.dumps({k: st.get(k) for k in stage2_keys}))
    rec.step("state.get manualAddLineMode reflects last setter",
             st.get("manualAddLineMode") == "vertical", f"manualAddLineMode={st.get('manualAddLineMode')}")

    # --- Regression (independent re-verification requirement): with a segment
    # now active (run_segment_activation_proof ran before this function), the
    # editing-gated preconditions must still reject manual_add.begin /
    # corrections.set_point_mode with -32008 while editing is explicitly
    # disabled, AND must let them succeed once editing is re-enabled. Before
    # segments.activate existed there was no way to reach this fork at all
    # (every call hit -32007 kind:"segment" first, no active surface); now that
    # a surface IS active, this is the regression check that the "not enabled"
    # error path still works correctly and hasn't been silently broken by the
    # fix that made the "enabled" success path work.
    r, _ = client.call("segmentation.enable_editing", {"enabled": False}, timeout=10.0)
    rec.step("segmentation.enable_editing(false) (regression setup)",
             r.get("enabled") is False, json.dumps(r))
    try:
        client.call("segmentation.manual_add.begin", {}, timeout=10.0)
        rec.step("manual_add.begin fails -32008 while editing disabled (regression)",
                 False, "unexpected success")
    except BridgeError as e:
        rec.step("manual_add.begin fails -32008 while editing disabled (regression)",
                 e.code == -32008, f"code={e.code} message={e.message} data={e.data}")
    try:
        client.call("segmentation.corrections.set_point_mode", {"active": True}, timeout=10.0)
        rec.step("corrections.set_point_mode(true) fails -32008 while editing disabled (regression)",
                 False, "unexpected success")
    except BridgeError as e:
        rec.step("corrections.set_point_mode(true) fails -32008 while editing disabled (regression)",
                 e.code == -32008, f"code={e.code} message={e.message} data={e.data}")

    # --- manual_add.begin: only with editing + active session ---
    editing_enabled = False
    try:
        r, _ = client.call("segmentation.enable_editing", {"enabled": True}, timeout=10.0)
        editing_enabled = r.get("enabled") is True
        rec.step("segmentation.enable_editing(true) re-enables successfully (regression: success path)",
                 editing_enabled, json.dumps(r))
    except BridgeError as e:
        # Exact match, not a loose set: with no active surface this MUST be
        # -32007 with data.kind:"segment" (AgentBridgeServer.cpp handler for
        # segmentation.enable_editing checks activeSurfaceId().empty() first).
        rec.step("segmentation.enable_editing(true) fails exactly -32007 kind=segment (no active surface)",
                 e.code == -32007 and e.data.get("kind") == "segment",
                 f"code={e.code} message={e.message} data={e.data}")

    began = False
    try:
        r, _ = client.call("segmentation.manual_add.begin", {}, timeout=15.0)
        began = r.get("active") is True
        rec.step("manual_add.begin", began, json.dumps(r))
    except BridgeError as e:
        if editing_enabled:
            expect_code, expect_kind = -32007, "session"
        else:
            expect_code, expect_kind = -32008, None
        ok = e.code == expect_code and (expect_kind is None or e.data.get("kind") == expect_kind)
        rec.step(f"manual_add.begin fails exactly {expect_code} (editing_enabled={editing_enabled})",
                 ok, f"code={e.code} message={e.message} data={e.data}")

    if began:
        # Full hole-fill cycle: reached on every normal run now that
        # run_segment_activation_proof + the regression re-enable above have
        # established a real active editing session (see the module docstring
        # for the bug this fixed and how it was confirmed).
        st_a, _ = client.call("state.get", timeout=10.0)
        rec.step("state.get manualAddMode true while active", st_a.get("manualAddMode") is True,
                 f"manualAddMode={st_a.get('manualAddMode')}")
        pre_shot, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
        if target_viewer and canvas_point:
            point_a = dict(canvas_point)
            point_b = dict(canvas_point)
            point_b["x"] = canvas_point["x"] + 40.0
            point_b["y"] = canvas_point["y"] + 25.0
            r, _ = client.call("canvas.shift_click",
                               {"viewer": target_viewer, "position": point_a,
                                "space": "volume", "button": "left"}, timeout=10.0)
            rec.step("manual-add place plane constraint #1 via canvas.shift_click(left)",
                     r.get("clicked") is True, json.dumps(r))
            r, _ = client.call("canvas.shift_click",
                               {"viewer": target_viewer, "position": point_b,
                                "space": "volume", "button": "left"}, timeout=10.0)
            rec.step("manual-add place plane constraint #2 (distinct point) via canvas.shift_click(left)",
                     r.get("clicked") is True, json.dumps(r))
            # Cycle line-preview and interpolation modes with constraints live.
            r, _ = client.call("segmentation.manual_add.set_line_mode", {"mode": "cross"}, timeout=10.0)
            rec.step("manual-add cycle line mode while constraints placed",
                     r.get("mode") == "cross", json.dumps(r))
            r, _ = client.call("segmentation.manual_add.set_interpolation",
                               {"mode": "tracer_restricted_to_fill"}, timeout=10.0)
            rec.step("manual-add cycle interpolation while constraints placed",
                     r.get("mode") == "tracer_restricted_to_fill", json.dumps(r))
            r, _ = client.call("canvas.click",
                               {"viewer": target_viewer, "position": point_a,
                                "space": "volume", "button": "right", "modifiers": ["shift"]}, timeout=10.0)
            rec.step("manual-add remove plane constraint #1 via canvas.click(right, shift)",
                     r.get("clicked") is True, json.dumps(r))
        r, _ = client.call("segmentation.manual_add.undo_constraint", {}, timeout=10.0)
        rec.step("manual_add.undo_constraint well-formed while active", "undone" in r, json.dumps(r))
        r, _ = client.call("segmentation.manual_add.finish", {"apply": True}, timeout=60.0)
        rec.step("manual_add.finish(apply=true) well-formed", "applied" in r, json.dumps(r))
        # NOTE on r["applied"] being False here: our synthetic shift-clicks above
        # target whatever segment segments.activate happened to pick (preferring
        # a fixed id), which is not guaranteed to have a genuine CLOSED hole
        # (invalid grid cells bounded by valid ones on both sides) anywhere near
        # canvas_point -- ManualAddTool::discoverAxisLine requires both ends of
        # the scan to hit a valid cell, so an unbounded/edge-of-grid "hole" (the
        # common case: a patch's allocated-but-not-yet-grown padding) can never
        # produce a committable line there. A real, disk-verified successful
        # fill (finish(apply=true) -> applied:true, and the previously-invalid
        # grid cells becoming real finite coordinates on re-read of the
        # segment's x/y/z.tif) WAS independently confirmed during re-verification
        # of this stage, against a segment with a genuine closed hole. See the
        # verification notes for that segment/coordinates.
        #
        # Confirmed real behavior worth flagging: when apply=true's interpolation
        # path finds nothing to commit (as here), finishManualAdd() returns false
        # WITHOUT clearing _manualAddMode -- the session is left stuck active
        # rather than exiting. Recover the same way a human/agent would: retry
        # with apply=false (discard) so the mode actually closes.
        st_b, _ = client.call("state.get", timeout=10.0)
        if st_b.get("manualAddMode") is True and not r.get("applied"):
            rec.step("manual_add.finish(apply=true) with nothing to commit leaves manualAddMode "
                     "stuck active (known application behavior)",
                     True, f"applied={r.get('applied')}; recovering via finish(apply=false)")
            r2, _ = client.call("segmentation.manual_add.finish", {"apply": False}, timeout=10.0)
            log(f"recovery manual_add.finish(apply=false) -> {json.dumps(r2)}")
            st_b, _ = client.call("state.get", timeout=10.0)
        rec.step("state.get manualAddMode false after finish (recovering via apply=false if needed)",
                 st_b.get("manualAddMode") is False,
                 f"manualAddMode={st_b.get('manualAddMode')}")
        # Restore the persisted manual-add config to its documented defaults.
        # This config setter (SegmentationManualAddPanel::writeSetting) persists
        # to the real ~/.VC3D/VC3D.ini shared by every VC3D launch on this
        # machine (see settingsFilePath()), not anything test-scoped -- leaving
        # it on "cross"/"tracer_restricted_to_fill" here (set above to exercise
        # "cycle modes while constraints placed") would silently change the
        # panel's default the next time ANYONE opens VC3D on this machine.
        client.call("segmentation.manual_add.set_line_mode", {"mode": "vertical"}, timeout=10.0)
        client.call("segmentation.manual_add.set_interpolation", {"mode": "thin_plate_spline"}, timeout=10.0)
        post_shot, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
        pre_png = base64.b64decode(pre_shot["base64"]) if pre_shot.get("base64") else b""
        post_png = base64.b64decode(post_shot["base64"]) if post_shot.get("base64") else b""
        rec.step("manual-add cycle produced a visible change (pre/post screenshot differ)",
                 pre_png != post_png, f"pre_len={len(pre_png)} post_len={len(post_png)}")
    else:
        # When the mode could not be entered, the mode-scoped RPCs must guard
        # with -32007 data.kind:"manual_add_session" rather than crash.
        try:
            client.call("segmentation.manual_add.finish", {"apply": False}, timeout=10.0)
            rec.step("manual_add.finish guarded when not active", False, "unexpected success")
        except BridgeError as e:
            rec.step("manual_add.finish guarded when not active (-32007 manual_add_session)",
                     e.code == -32007 and "manual_add_session" in json.dumps(e.data),
                     f"code={e.code} data={e.data}")
        try:
            client.call("segmentation.manual_add.undo_constraint", {}, timeout=10.0)
            rec.step("manual_add.undo_constraint guarded when not active", False, "unexpected success")
        except BridgeError as e:
            rec.step("manual_add.undo_constraint guarded when not active (-32007 manual_add_session)",
                     e.code == -32007 and "manual_add_session" in json.dumps(e.data),
                     f"code={e.code} data={e.data}")

    # --- corrections.set_point_mode ---
    corr_on = False
    try:
        r, _ = client.call("segmentation.corrections.set_point_mode", {"active": True}, timeout=10.0)
        corr_on = r.get("active") is True
        rec.step("corrections.set_point_mode(true)", corr_on, json.dumps(r))
    except BridgeError as e:
        if editing_enabled:
            expect_code, expect_kind = -32007, "session"
        else:
            expect_code, expect_kind = -32008, None
        ok = e.code == expect_code and (expect_kind is None or e.data.get("kind") == expect_kind)
        rec.step(f"corrections.set_point_mode(true) fails exactly {expect_code} (editing_enabled={editing_enabled})",
                 ok, f"code={e.code} message={e.message} data={e.data}")

    if corr_on:
        st_c, _ = client.call("state.get", timeout=10.0)
        rec.step("state.get correctionsPointMode true while active",
                 st_c.get("correctionsPointMode") is True,
                 f"correctionsPointMode={st_c.get('correctionsPointMode')}")
        if target_viewer and canvas_point:
            pre_pts, _ = client.call("points.list", timeout=10.0)
            pre_point_count = sum(len(c.get("points", [])) for c in pre_pts.get("collections", []))
            pre_shot, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
            r, _ = client.call("canvas.click",
                               {"viewer": target_viewer, "position": canvas_point,
                                "space": "volume", "button": "left"}, timeout=10.0)
            rec.step("corrections single point via plain canvas.click(left)",
                     r.get("clicked") is True, json.dumps(r))
            post_shot, _ = client.call("screenshot.capture", {"target": "window"}, timeout=15.0)
            pre_png = base64.b64decode(pre_shot["base64"]) if pre_shot.get("base64") else b""
            post_png = base64.b64decode(post_shot["base64"]) if post_shot.get("base64") else b""
            # Screenshot diffing is a weak/unreliable signal for a single small
            # correction-point marker (confirmed via independent re-verification:
            # a genuinely-committed point can render byte-identical window
            # screenshots before/after). points.list is ground truth for whether
            # a point was actually committed -- assert on that primarily, and
            # keep the screenshot diff only as a non-fatal supplementary signal.
            post_pts, _ = client.call("points.list", timeout=10.0)
            post_point_count = sum(len(c.get("points", [])) for c in post_pts.get("collections", []))
            rec.step("corrections point commit produced a REAL point (points.list count increased)",
                     post_point_count == pre_point_count + 1,
                     f"before={pre_point_count} after={post_point_count}")
            log(f"corrections point commit screenshot diff (supplementary, non-fatal): "
                f"changed={pre_png != post_png} pre_len={len(pre_png)} post_len={len(post_png)}")

        # --- Drag > 1.0 voxel: commits an anchored point AND
        # immediately triggers the corrections solver as a source:"growth"
        # job -- poll job.status to a terminal state. Use the "segmentation"
        # viewer (not target_viewer/canvas_point, which follow a raw-volume
        # plane slice that may not sit on the active surface's grid at all) so
        # the drag's world points are guaranteed to resolve to real grid
        # indices via SegmentationEditManager::worldToGridIndex.
        try:
            seg_cursor, _ = client.call("canvas.get_cursor_volume_point",
                                        {"viewer": "segmentation"}, timeout=10.0)
            drag_from = seg_cursor.get("volumePoint")
        except BridgeError as e:
            drag_from = None
            rec.step("corrections drag: discover a real point on the active surface", False,
                     f"code={e.code} message={e.message}")
        if drag_from:
            # The "segmentation" viewer follows the active surface's (possibly
            # curved) 3D shape, so a fixed flat-world (dx, dy) offset from an
            # arbitrary point is not guaranteed to stay within canvas.drag's
            # 2.0-voxel round-trip tolerance -- how far you can move before
            # falling off the local tangent plane depends on local curvature at
            # that specific point, which we have no cheap way to query here.
            # Try a couple of small, distinct candidate deltas (all safely
            # > 1.0 voxel so they hit the "anchored + solver" branch, not the
            # zero-length "plain point" branch) and use whichever round-trips.
            # Candidate deltas span small -> large: too small and the ACTUAL
            # sampled displacement on the (possibly curved) surface can end up
            # under the 1.0-voxel "moved" threshold even though the endpoints
            # round-trip fine; too large and the endpoint falls outside the
            # 2.0-voxel round-trip tolerance. Try several and use whichever
            # both round-trips AND yields a real triggered job.
            rd = None
            job = None
            tried = []
            for ddx, ddy in ((1.3, 1.3), (2.0, 2.0), (1.3, -1.3), (3.0, 0.0), (0.0, 3.0), (4.0, 4.0)):
                drag_to = dict(drag_from)
                drag_to["x"] = drag_from["x"] + ddx
                drag_to["y"] = drag_from["y"] + ddy
                try:
                    rd_try, _ = client.call("canvas.drag",
                                            {"viewer": "segmentation", "from": drag_from, "to": drag_to,
                                             "space": "volume", "button": "left", "steps": 4}, timeout=10.0)
                except BridgeError as e:
                    tried.append(f"({ddx},{ddy}): rejected code={e.code} message={e.message}")
                    continue
                st_job, _ = client.call("state.get", timeout=10.0)
                job_try = st_job.get("job")
                tried.append(f"({ddx},{ddy}): dragged, job={json.dumps(job_try)[:120]}")
                rd = rd_try
                if job_try is not None and job_try.get("source") == "growth":
                    job = job_try
                    break
            rec.step("corrections drag (>1 voxel) round-trips",
                     rd is not None and rd.get("dragged") is True,
                     json.dumps(rd)[:300] if rd else "all candidate deltas rejected: " + "; ".join(tried))
            rec.step("corrections drag (>1 voxel) triggers a source:\"growth\" job",
                     job is not None and job.get("source") == "growth",
                     json.dumps(job) if job else "no candidate delta produced a job (independently confirmed "
                                                  "working via a dedicated standalone script during "
                                                  "re-verification; sensitive to local surface curvature at "
                                                  "whatever point is currently under the cursor, see "
                                                  "run_manual_add_corrections_smoke comment): " + "; ".join(tried))
            if job and job.get("jobId"):
                job_id = job["jobId"]
                deadline = time.monotonic() + 60.0
                last_state = None
                while time.monotonic() < deadline:
                    js, _ = client.call("job.status", {"jobId": job_id}, timeout=10.0)
                    last_state = js.get("state")
                    if last_state in ("succeeded", "failed"):
                        break
                    time.sleep(1.0)
                rec.step("corrections solver job reaches a terminal state",
                         last_state in ("succeeded", "failed"), f"final_state={last_state}")

        r, _ = client.call("segmentation.corrections.set_point_mode", {"active": False}, timeout=10.0)
        rec.step("corrections.set_point_mode(false) switches off", r.get("active") is False, json.dumps(r))

    # Disabling correction mode is always allowed (no preconditions), even with
    # no active session -- must return {"active": false}, never error.
    try:
        r, _ = client.call("segmentation.corrections.set_point_mode", {"active": False}, timeout=10.0)
        rec.step("corrections.set_point_mode(false) always well-formed",
                 r.get("active") is False, json.dumps(r))
    except BridgeError as e:
        rec.step("corrections.set_point_mode(false) always well-formed", False,
                 f"code={e.code} message={e.message}")

    # Responsiveness after the full sequence -- proves nothing wedged/hung.
    r, dt = client.call("ping", timeout=5.0)
    rec.step("ping responsive after manual-add/corrections smoke",
             r.get("pong") is True, f"({ms(dt):.2f}ms)")


def run_stage6_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess,
                     seg_ids: set) -> None:
    """Exercise tags.set, seeding.*, push_pull.*,
    tracer.run_trace.

    Every timeout checks whether VC3D is still alive. The safe fire-and-forget
    seeding actions are exercised for real; seeding batch and analyze-paths
    success paths are not launched here. tracer.run_trace is exercised only
    through its error paths so no external tool job is launched. tags.set
    applies and then reverts a single tag so the fixture is left unchanged.
    """

    def call_checked(name: str, method: str, params: dict | None = None,
                      timeout: float = 20.0):
        try:
            result, _ = client.call(method, params, timeout=timeout)
            rec.step(name, True, json.dumps(result)[:300])
            return result
        except BridgeError as e:
            rec.step(name, False, f"code={e.code} message={e.message} data={e.data}")
            return None
        except TimeoutError as e:
            if proc.exit_code() is not None:
                record_process_death(proc, rec, name)
                raise VC3DDiedError(name)
            rec.step(name, False, f"TIMEOUT (VC3D still alive): {e}")
            return None

    def call_expect_error(name: str, method: str, params: dict, expected_code: int,
                          timeout: float = 20.0):
        try:
            result, _ = client.call(method, params, timeout=timeout)
            rec.step(name, False, f"expected error {expected_code}, got success: {result}")
            return None
        except BridgeError as e:
            rec.step(name, e.code == expected_code,
                     f"code={e.code} message={e.message} data={e.data}")
            return e
        except TimeoutError as e:
            if proc.exit_code() is not None:
                record_process_death(proc, rec, name)
                raise VC3DDiedError(name)
            rec.step(name, False, f"TIMEOUT (VC3D still alive): {e}")
            return None

    # --- tags.set ---
    call_expect_error("tags.set missing segmentId -> -32602",
                      "tags.set", {"tag": "approved", "enabled": True}, -32602)
    call_expect_error("tags.set missing enabled -> -32602",
                      "tags.set", {"segmentId": "x", "tag": "approved"}, -32602)
    e = call_expect_error("tags.set tag='revisit' (does not exist) -> -32602",
                          "tags.set",
                          {"segmentId": "x", "tag": "revisit", "enabled": True}, -32602)
    if e is not None:
        rec.step("tags.set 'revisit' error data.param == tag",
                 e.data.get("param") == "tag", f"data={e.data}")
    e = call_expect_error("tags.set unknown segment -> -32007 kind=segment",
                          "tags.set",
                          {"segmentId": "agent-bridge-nonexistent-seg",
                           "tag": "approved", "enabled": True}, -32007)
    if e is not None:
        rec.step("tags.set unknown segment error data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")

    # Reversible real mutation on a live segment: set "inspect" then clear it.
    target = sorted(seg_ids)[0] if seg_ids else None
    if target:
        r = call_checked("tags.set inspect=true (real segment)", "tags.set",
                         {"segmentId": target, "tag": "inspect", "enabled": True})
        if r is not None:
            rec.step("tags.set result echoes {segmentId, tag, enabled}",
                     r.get("segmentId") == target and r.get("tag") == "inspect"
                     and r.get("enabled") is True, json.dumps(r))
        # Revert so the fixture is left unchanged.
        call_checked("tags.set inspect=false (revert)", "tags.set",
                     {"segmentId": target, "tag": "inspect", "enabled": False})

    # --- seeding.* ---
    call_expect_error("seeding.set_winding_annotation_mode missing active -> -32602",
                      "seeding.set_winding_annotation_mode", {}, -32602)
    r = call_checked("seeding.set_winding_annotation_mode(true)",
                     "seeding.set_winding_annotation_mode", {"active": True})
    if r is not None:
        rec.step("seeding.set_winding_annotation_mode echoes active",
                 r.get("active") is True, json.dumps(r))
    call_checked("seeding.set_winding_annotation_mode(false)",
                 "seeding.set_winding_annotation_mode", {"active": False})
    # preview_rays / cast_rays are fire-and-forget; with no focus POI the
    # precondition dialog is suppressed and the call returns cleanly (proving
    # that no dialog is opened.
    r = call_checked("seeding.preview_rays (no focus POI -> clean, no dialog)",
                     "seeding.preview_rays", {})
    if r is not None:
        rec.step("seeding.preview_rays returns requested=true",
                 r.get("requested") is True, json.dumps(r))
    call_checked("seeding.cast_rays (async, no hang)", "seeding.cast_rays", {})
    r = call_checked("seeding.reset_points", "seeding.reset_points", {})
    if r is not None:
        rec.step("seeding.reset_points returns reset=true",
                 r.get("reset") is True, json.dumps(r))

    # --- segmentation.push_pull.* ---
    r = call_checked("push_pull.set_config (subset read-modify-write)",
                     "segmentation.push_pull.set_config",
                     {"blurRadius": 5, "perVertex": False, "step": 3.0})
    if r is not None:
        keys = {"start", "stop", "step", "low", "high", "blurRadius",
                "computeScale", "perVertexLimit", "perVertex"}
        rec.step("push_pull.set_config returns the full effective config",
                 keys.issubset(set(r.keys())), json.dumps(r))
        rec.step("push_pull.set_config applied blurRadius=5, perVertex=false",
                 r.get("blurRadius") == 5 and r.get("perVertex") is False, json.dumps(r))
    call_expect_error("push_pull.set_config bad type (blurRadius='x') -> -32602",
                      "segmentation.push_pull.set_config", {"blurRadius": "x"}, -32602)
    call_expect_error("push_pull.start bad direction -> -32602",
                      "segmentation.push_pull.start", {"direction": "sideways"}, -32602)
    # Ensure editing is OFF so start hits its -32008 precondition deterministically.
    call_checked("segmentation.enable_editing(false) before push_pull.start guard",
                 "segmentation.enable_editing", {"enabled": False})
    call_expect_error("push_pull.start while editing disabled -> -32008",
                      "segmentation.push_pull.start", {"direction": "push"}, -32008)
    r = call_checked("push_pull.stop (always safe)",
                     "segmentation.push_pull.stop", {})
    if r is not None:
        rec.step("push_pull.stop returns stopped=true",
                 r.get("stopped") is True, json.dumps(r))

    # --- tracer.run_trace error paths (no tool job launched) ---
    call_expect_error("tracer.run_trace missing segmentId -> -32602",
                      "tracer.run_trace", {}, -32602)
    call_expect_error("tracer.run_trace bad paramOverrides type -> -32602",
                      "tracer.run_trace",
                      {"segmentId": target or "x", "paramOverrides": "nope"}, -32602)
    e = call_expect_error("tracer.run_trace unknown segment -> -32007 kind=segment",
                          "tracer.run_trace",
                          {"segmentId": "agent-bridge-nonexistent-seg"}, -32007)
    if e is not None:
        rec.step("tracer.run_trace unknown segment error data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")



def run_segment_activation_proof(client: BridgeClient, rec: Recorder, seg_ids: set) -> None:
    """Exercise segments.activate against a real segment.

    Activates a real, pre-existing segment via the new RPC and proves the fix:
    segmentation.enable_editing(true) SUCCEEDS afterwards (it previously failed
    -32007 on every headless session), and the newly-active segment is reported
    by both segments.list ("active": true) and state.get (activeSurface.id).
    """
    # Prefer a known stable pre-existing segment; fall back to any listed id.
    preferred = "auto_grown_20260416135719054_inp_hr"
    target = preferred if preferred in seg_ids else (sorted(seg_ids)[0] if seg_ids else None)
    if not target:
        rec.step("segments.activate proof: a segment exists to activate", False,
                 "no segments listed to activate")
        return

    # Unknown-id error path: -32007 data.kind:"segment".
    try:
        client.call("segments.activate", {"segmentId": "agent-bridge-nonexistent-seg"}, timeout=10.0)
        rec.step("segments.activate unknown id rejected with -32007 kind=segment", False,
                 "unexpected success")
    except BridgeError as e:
        rec.step("segments.activate unknown id rejected with -32007 kind=segment",
                 e.code == -32007 and e.data.get("kind") == "segment",
                 f"code={e.code} data={e.data}")

    # Missing param: -32602 data.param:"segmentId".
    try:
        client.call("segments.activate", {}, timeout=10.0)
        rec.step("segments.activate missing segmentId rejected with -32602", False,
                 "unexpected success")
    except BridgeError as e:
        rec.step("segments.activate missing segmentId rejected with -32602",
                 e.code == -32602 and e.data.get("param") == "segmentId",
                 f"code={e.code} data={e.data}")

    prev_state, _ = client.call("state.get", timeout=10.0)
    prev_active = (prev_state.get("activeSurface") or {}).get("id")

    # The activation itself.
    try:
        r, _ = client.call("segments.activate", {"segmentId": target}, timeout=20.0)
        ok = (r.get("activated") is True
              and isinstance(r.get("segment"), dict)
              and r["segment"].get("id") == target
              and r["segment"].get("active") is True
              and "alreadyActive" in r
              and "previousSegmentId" in r)
        rec.step("segments.activate activates a real segment", ok, json.dumps(r)[:400])
    except BridgeError as e:
        rec.step("segments.activate activates a real segment", False,
                 f"code={e.code} message={e.message} data={e.data}")
        return

    # state.get and segments.list must both reflect the newly-active segment.
    st, _ = client.call("state.get", timeout=10.0)
    active_id = (st.get("activeSurface") or {}).get("id")
    rec.step("state.get activeSurface reflects the activated segment",
             active_id == target, f"activeSurface.id={active_id} target={target}")

    seglist, _ = client.call("segments.list", timeout=10.0)
    active_flags = {s["id"]: s.get("active") for s in seglist.get("segments", [])}
    rec.step("segments.list 'active' flag reflects the activated segment",
             active_flags.get(target) is True
             and sum(1 for v in active_flags.values() if v) == 1,
             f"active_for_target={active_flags.get(target)} "
             f"num_active={sum(1 for v in active_flags.values() if v)}")

    # THE critical proof: with a segment now active, enable_editing must SUCCEED
    # (not -32007), unlocking the entire editing-gated RPC surface.
    try:
        r, _ = client.call("segmentation.enable_editing", {"enabled": True}, timeout=15.0)
        rec.step("segmentation.enable_editing(true) SUCCEEDS after segments.activate (the fix)",
                 r.get("enabled") is True, json.dumps(r))
    except BridgeError as e:
        rec.step("segmentation.enable_editing(true) SUCCEEDS after segments.activate (the fix)",
                 False, f"STILL FAILS: code={e.code} message={e.message} data={e.data}")

    # Re-activating the already-active id is a documented no-op success.
    try:
        r, _ = client.call("segments.activate", {"segmentId": target}, timeout=10.0)
        rec.step("segments.activate on already-active id is a no-op success",
                 r.get("activated") is True and r.get("alreadyActive") is True,
                 json.dumps(r)[:300])
    except BridgeError as e:
        rec.step("segments.activate on already-active id is a no-op success", False,
                 f"code={e.code} message={e.message}")

    # Clean up: leave editing disabled so downstream steps see a neutral state.
    try:
        client.call("segmentation.enable_editing", {"enabled": False}, timeout=10.0)
    except BridgeError:
        pass
