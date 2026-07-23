"""Manual Lasagna, atlas, and fiber workspace checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bridge_client import BridgeClient, BridgeError
from manual_bridge_support import (
    LOCAL_REAL_SEED,
    Recorder,
    VC3DDiedError,
    ms,
    record_process_death,
)
from vc3d_process import VC3DProcess

def run_lasagna_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess) -> None:
    """Exercise lasagna.* RPCs and workspace.switch.

    No Lasagna Python service is assumed. Every timeout checks process liveness,
    and the headless calls must never open an interactive dialog.
    """

    def call_checked(name: str, method: str, params: dict | None = None,
                      timeout: float = 15.0):
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
                          timeout: float = 15.0):
        try:
            result, _ = client.call(method, params, timeout=timeout)
            rec.step(name, False, f"expected error {expected_code}, got success: {result}")
        except BridgeError as e:
            rec.step(name, e.code == expected_code,
                     f"code={e.code} message={e.message} data={e.data}")
        except TimeoutError as e:
            if proc.exit_code() is not None:
                record_process_death(proc, rec, name)
                raise VC3DDiedError(name)
            rec.step(name, False, f"TIMEOUT (VC3D still alive): {e}")

    try:
        # 1. service_status always works, no service required.
        call_checked("lasagna.service_status (no service)", "lasagna.service_status")

        # 2. ensure_service (internal mode): report the real outcome -- the
        # fit_service.py script is not expected to be present in this
        # environment, so a clean -32005 (not a hang, not a fabricated pass)
        # is the correct and expected result.
        call_expect_error("lasagna.ensure_service (internal, script not installed)",
                           "lasagna.ensure_service", {}, -32005)

        # 3. list_datasets / jobs -- deferred; with no service
        # running both must fail cleanly and quickly, never hang.
        call_expect_error("lasagna.list_datasets (no service)", "lasagna.list_datasets", {}, -32005)
        call_expect_error("lasagna.jobs (no service)", "lasagna.jobs", {}, -32005)

        # 4. cancel error paths.
        call_expect_error("lasagna.cancel (no active job)", "lasagna.cancel", {}, -32007)
        call_expect_error("lasagna.cancel (unknown job id)", "lasagna.cancel",
                           {"jobId": "job-999999"}, -32007)

        # 5. select_output_segment: unknown name, then a real existing segment.
        call_expect_error("lasagna.select_output_segment (unknown)",
                           "lasagna.select_output_segment",
                           {"name": "agent-bridge-nonexistent-segment"}, -32007)
        segs = call_checked("segments.list (for lasagna select_output_segment)", "segments.list")
        real_seg = (segs or {}).get("segments", [{}])[0].get("id") if segs else None
        if real_seg:
            call_checked("lasagna.select_output_segment (real segment)",
                         "lasagna.select_output_segment", {"name": real_seg})

        # 6. repeat_last with nothing configured -> clean -32005, NOT a hang.
        # This is the exact call that used to hang forever (see docstring).
        call_expect_error("lasagna.repeat_last (nothing configured, must not hang)",
                           "lasagna.repeat_last", {}, -32005)

        # 7. start_optimization with a bad mode string -> -32602.
        call_expect_error("lasagna.start_optimization (bad mode)", "lasagna.start_optimization",
                           {"mode": "not_a_real_mode"}, -32602)

        # 8. workspace.switch: both valid names, then an invalid one.
        call_checked("workspace.switch lasagna", "workspace.switch", {"name": "lasagna"})
        call_checked("workspace.switch fiber_slice", "workspace.switch", {"name": "fiber_slice"})
        call_expect_error("workspace.switch (bad name)", "workspace.switch",
                           {"name": "not_a_real_workspace"}, -32602)

        # 9. Liveness proof: the process must still be fully responsive after
        # everything above, including both service-start paths.
        call_checked("ping after lasagna smoke (proves no hang)", "ping")
    except VC3DDiedError:
        return


def run_atlas_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess) -> None:
    """Exercise atlas.* RPCs.

    The local fixture has no atlas directory, no saved fibers and no Lasagna
    dataset, so success paths are unreachable here -- every step below proves a
    documented error is returned PROMPTLY (no hang: the interactive
    counterparts of these calls reach QMessageBoxes -- 'Load an atlas before
    remapping.', the atlas rebuild prompt, showError in
    showIntersectionInspection -- which would block forever offscreen), and
    that the process stays alive and responsive throughout.
    """

    def call_checked(name: str, method: str, params: dict | None = None,
                      timeout: float = 15.0):
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
                          timeout: float = 15.0):
        try:
            result, _ = client.call(method, params, timeout=timeout)
            rec.step(name, False, f"expected error {expected_code}, got success: {result}")
        except BridgeError as e:
            rec.step(name, e.code == expected_code,
                     f"code={e.code} message={e.message} data={e.data}")
        except TimeoutError as e:
            if proc.exit_code() is not None:
                record_process_death(proc, rec, name)
                raise VC3DDiedError(name)
            rec.step(name, False, f"TIMEOUT (VC3D still alive): {e}")

    try:
        # 1. atlas.status always works; with no atlas open both fields are
        # null and no search is running.
        status = call_checked("atlas.status (no atlas open)", "atlas.status")
        if status is not None:
            ok = (status.get("atlasDir") is None and status.get("atlasName") is None
                  and isinstance(status.get("search"), dict)
                  and status["search"].get("running") is False
                  and status["search"].get("phaseCount") == 5
                  and status["search"].get("resultCount") == 0)
            rec.step("atlas.status shape (dir/name null, search idle, phaseCount 5)",
                     ok, json.dumps(status)[:300])

        # 2. atlas.open error paths: missing params, absolute missing dir,
        # relative missing dir (resolved against the volpkg root). The
        # interactive path would raise dialogs; these must return -32602/-32007.
        call_expect_error("atlas.open (missing atlasDir)", "atlas.open", {}, -32602)
        call_expect_error("atlas.open (absolute dir missing)", "atlas.open",
                           {"atlasDir": "/nonexistent/agent-bridge-atlas"}, -32007)
        call_expect_error("atlas.open (relative dir missing)", "atlas.open",
                           {"atlasDir": "atlases/agent-bridge-nonexistent"}, -32007)

        # 3. atlas.search_start param validation.
        call_expect_error("atlas.search_start (bad mode)", "atlas.search_start",
                           {"mode": "not_a_real_mode"}, -32602)
        call_expect_error("atlas.search_start (negative maxDistance)", "atlas.search_start",
                           {"mode": "non_atlas_only", "maxDistance": -1.0}, -32602)
        call_expect_error("atlas.search_start (requiredTags not an array)",
                           "atlas.search_start",
                           {"mode": "non_atlas_only", "requiredTags": "tag"}, -32602)

        # 4. atlas.search_start preconditions: default mode with no atlas open
        # -> -32007 kind:"atlas"; non_atlas_only with no saved fibers ->
        # -32005 (headless launcher failure). Both used to be QMessageBox
        # paths in the interactive slot -- must NOT hang.
        call_expect_error("atlas.search_start (atlas_to_non_atlas, no atlas open)",
                           "atlas.search_start", {}, -32007)
        call_expect_error("atlas.search_start (non_atlas_only, no saved fibers)",
                           "atlas.search_start", {"mode": "non_atlas_only"}, -32005)

        # 5. atlas.search_cancel with nothing running -> -32007 kind:"job".
        call_expect_error("atlas.search_cancel (idle)", "atlas.search_cancel", {}, -32007)

        # 6. atlas.search_results: empty set is NOT an error (total 0);
        # negative offset / bad limit are -32602.
        res = call_checked("atlas.search_results (empty)", "atlas.search_results")
        if res is not None:
            rec.step("atlas.search_results empty shape (total 0, results [])",
                     res.get("total") == 0 and res.get("offset") == 0
                     and res.get("results") == [],
                     json.dumps(res)[:200])
        call_expect_error("atlas.search_results (negative offset)", "atlas.search_results",
                           {"offset": -1}, -32602)
        call_expect_error("atlas.search_results (limit 0)", "atlas.search_results",
                           {"limit": 0}, -32602)

        # 7. atlas.open_result: no results -> -32007 kind:"result"; bad index
        # type -> -32602. The interactive path shows QMessageBoxes / showError
        # dialogs -- must NOT hang.
        call_expect_error("atlas.open_result (no results)", "atlas.open_result",
                           {"index": 0}, -32007)
        call_expect_error("atlas.open_result (bad index type)", "atlas.open_result",
                           {"index": "zero"}, -32602)

        # 8. atlas.remap / atlas.optimize_snap_candidates with no atlas open
        # -> -32007 kind:"atlas" (interactive: 'Load an atlas before ...'
        # QMessageBoxes -- must NOT hang).
        call_expect_error("atlas.remap (no atlas open)", "atlas.remap", {}, -32007)
        call_expect_error("atlas.optimize_snap_candidates (no atlas open)",
                           "atlas.optimize_snap_candidates", {}, -32007)

        # 9. Liveness proof after everything above.
        call_checked("ping after atlas smoke (proves no hang)", "ping")
    except VC3DDiedError:
        return


def run_fiber_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess) -> None:
    """Exercise fiber.* line-annotation RPCs.

    The local fixture has no saved fibers and (normally) no resolvable Lasagna
    dataset, so workspace-opening success paths are exercised as clean,
    documented errors (never a hang: the interactive counterparts reach
    showError QMessageBoxes and the Lasagna dataset-picker QFileDialog, which
    would block forever offscreen -- LineAnnotationController now runs with
    error dialogs suppressed while the bridge is attached). The
    import/export/tag/delete lifecycle IS exercised for real, end to end,
    through a temp directory: import a crafted vc3d_fiber JSON, verify it in
    fiber.list, tag it, export it to a bundle, re-import the bundle, then
    delete everything imported so the fixture is left as found.
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="vc3d-bridge-fiber-"))
    try:
        # 1. fiber.list baseline: well-formed {"fibers", "knownTags"}.
        fl = call_checked("fiber.list (baseline)", "fiber.list")
        if fl is None:
            return
        rec.step("fiber.list shape (fibers + knownTags arrays)",
                 isinstance(fl.get("fibers"), list) and isinstance(fl.get("knownTags"), list),
                 json.dumps(fl)[:200])
        baseline_ids = {f["fiberId"] for f in fl.get("fibers", [])}

        # 2. fiber.set_follow with no workspace open -> -32007 fiber_workspace.
        # (Run BEFORE any launch attempt below can create a workspace window.)
        e = call_expect_error("fiber.set_follow (no workspace) -> -32007",
                              "fiber.set_follow", {"enabled": True}, -32007)
        if e is not None:
            rec.step("fiber.set_follow error data.kind == fiber_workspace",
                     e.data.get("kind") == "fiber_workspace", f"data={e.data}")
        call_expect_error("fiber.set_follow (non-bool enabled) -> -32602",
                          "fiber.set_follow", {"enabled": "yes"}, -32602)

        # 3. fiber.export error paths.
        call_expect_error("fiber.export (missing path) -> -32602", "fiber.export", {}, -32602)
        call_expect_error("fiber.export (scale 0) -> -32602", "fiber.export",
                          {"path": str(tmp_dir / "x.json"), "scale": 0}, -32602)
        if not baseline_ids:
            call_expect_error("fiber.export (no fibers) -> -32005", "fiber.export",
                              {"path": str(tmp_dir / "empty.json")}, -32005)

        # 4. fiber.import error paths.
        call_expect_error("fiber.import (missing path) -> -32602", "fiber.import", {}, -32602)
        call_expect_error("fiber.import (nonexistent path) -> -32007 kind=path",
                          "fiber.import", {"path": str(tmp_dir / "does-not-exist.json")}, -32007)
        garbage = tmp_dir / "garbage.json"
        garbage.write_text("this is not json {", encoding="utf-8")
        call_expect_error("fiber.import (unparseable JSON) -> -32005", "fiber.import",
                          {"path": str(garbage)}, -32005)
        not_a_fiber = tmp_dir / "not_a_fiber.json"
        not_a_fiber.write_text(json.dumps({"type": "something_else"}), encoding="utf-8")
        call_expect_error("fiber.import (valid JSON, not a fiber) -> -32005", "fiber.import",
                          {"path": str(not_a_fiber)}, -32005)

        # 5. REAL headless import: a crafted, valid vc3d_fiber (control points
        # are an ordered subset of line points, as validateFiberInputControlPoints
        # requires). Coordinates sit near the fixture's curated real seed.
        base = LOCAL_REAL_SEED
        line_points = [[base["x"], base["y"], base["z"] + 10.0 * i] for i in range(5)]
        fiber_json = {
            "type": "vc3d_fiber",
            "version": 1,
            "username": "agentbridge",
            "started_at": "20260720T000000",
            "sequence": 990001,
            "control_points": [line_points[0], line_points[-1]],
            "line_points": line_points,
        }
        import_file = tmp_dir / "agent_bridge_fiber.json"
        import_file.write_text(json.dumps(fiber_json), encoding="utf-8")
        imp = call_checked("fiber.import (real crafted vc3d_fiber)", "fiber.import",
                           {"path": str(import_file)})
        if imp is not None:
            rec.step("fiber.import imported exactly 1, skipped 0",
                     imp.get("imported") == 1 and imp.get("skipped") == 0, json.dumps(imp))

        fl2 = call_checked("fiber.list reflects the import", "fiber.list")
        new_ids = sorted({f["fiberId"] for f in (fl2 or {}).get("fibers", [])} - baseline_ids)
        rec.step("fiber.list gained exactly the imported fiber",
                 fl2 is not None and len(new_ids) == 1,
                 f"new_ids={new_ids}")
        fiber_id = new_ids[0] if new_ids else None
        if fl2 is not None and fiber_id:
            imported_row = next(f for f in fl2["fibers"] if f["fiberId"] == fiber_id)
            rec.step("imported fiber row has real counts (2 control / 5 line points)",
                     imported_row.get("controlPointCount") == 2
                     and imported_row.get("linePointCount") == 5
                     and imported_row.get("lengthVx", 0) > 0,
                     json.dumps(imported_row)[:300])

        # 6. fiber.set_tag lifecycle on the real fiber + error paths.
        call_expect_error("fiber.set_tag (unknown fiber) -> -32007", "fiber.set_tag",
                          {"fiberId": "999999999", "tag": "t", "enabled": True}, -32007)
        call_expect_error("fiber.set_tag (empty tag) -> -32602", "fiber.set_tag",
                          {"fiberId": fiber_id or "1", "tag": "  ", "enabled": True}, -32602)
        if fiber_id:
            r = call_checked("fiber.set_tag enable 'agent_test'", "fiber.set_tag",
                             {"fiberId": fiber_id, "tag": "agent_test", "enabled": True})
            fl3 = call_checked("fiber.list after set_tag", "fiber.list")
            if r is not None and fl3 is not None:
                row = next((f for f in fl3["fibers"] if f["fiberId"] == fiber_id), {})
                rec.step("fiber tag visible on the fiber and in knownTags",
                         "agent_test" in row.get("tags", [])
                         and "agent_test" in fl3.get("knownTags", []),
                         f"tags={row.get('tags')} knownTags={fl3.get('knownTags')}")
            call_checked("fiber.set_tag disable 'agent_test'", "fiber.set_tag",
                         {"fiberId": fiber_id, "tag": "agent_test", "enabled": False})

        # 7. REAL headless export round trip: export -> parse the file ->
        # re-import the bundle.
        export_file = tmp_dir / "agent_bridge_export.json"
        ex = call_checked("fiber.export (real bundle to temp path)", "fiber.export",
                          {"path": str(export_file)})
        reimported_ok = False
        if ex is not None:
            bundle_ok = False
            detail = "export file missing"
            if export_file.exists():
                try:
                    bundle = json.loads(export_file.read_text(encoding="utf-8"))
                    bundle_ok = (bundle.get("type") == "vc3d_fiber_collection"
                                 and len(bundle.get("point_collections", [])) == ex.get("exported"))
                    detail = (f"type={bundle.get('type')} "
                              f"entries={len(bundle.get('point_collections', []))} "
                              f"exported={ex.get('exported')}")
                except Exception as parse_e:  # noqa: BLE001
                    detail = f"unparseable export: {parse_e}"
            rec.step("fiber.export wrote a parseable vc3d_fiber_collection bundle",
                     bundle_ok, detail)
            ri = call_checked("fiber.import (re-import the exported bundle)", "fiber.import",
                              {"path": str(export_file)})
            if ri is not None:
                reimported_ok = ri.get("imported", 0) >= 1
                rec.step("bundle re-import round-trips (imported >= 1)",
                         reimported_ok, json.dumps(ri))

        # 8. fiber.open: error paths, then the real fiber. With no Lasagna
        # dataset resolvable in this fixture the real open fails -32005 with a
        # clean detail -- the interactive path would block on the dataset-picker
        # QFileDialog, so "prompt error, no hang" is the regression being proven.
        call_expect_error("fiber.open (unknown fiber) -> -32007", "fiber.open",
                          {"fiberId": "999999999"}, -32007)
        call_expect_error("fiber.open (conflicting selectors) -> -32602", "fiber.open",
                          {"fiberId": fiber_id or "1", "controlPointIndex": 0,
                           "linePointIndex": 1}, -32602)
        call_expect_error("fiber.open (bad span shape) -> -32602", "fiber.open",
                          {"fiberId": fiber_id or "1", "span": [0]}, -32602)
        if fiber_id:
            try:
                r, dt = client.call("fiber.open", {"fiberId": fiber_id}, timeout=60.0)
                rec.step("fiber.open (real fiber) returned promptly (opened workspace)",
                         r.get("opened") is True, f"({ms(dt):.0f}ms) {json.dumps(r)}")
            except BridgeError as e:
                rec.step("fiber.open (real fiber) fails CLEANLY without dataset-picker hang",
                         e.code == -32005,
                         f"code={e.code} message={e.message} data={e.data}")
            except TimeoutError as e:
                if proc.exit_code() is not None:
                    record_process_death(proc, rec, "fiber.open real fiber")
                    raise VC3DDiedError("fiber.open real fiber")
                rec.step("fiber.open (real fiber) fails CLEANLY without dataset-picker hang",
                         False, f"HANG: {e}")

        # 9. fiber.create_atlas: unknown id, then the real fiber. Same "clean
        # error, no dialog hang" contract (interactive failure path is a
        # showError QMessageBox; success path used to reach the rebuild-prompt-
        # capable interactive atlas display).
        call_expect_error("fiber.create_atlas (unknown fiber) -> -32007", "fiber.create_atlas",
                          {"fiberId": "999999999"}, -32007)
        if fiber_id:
            try:
                r, dt = client.call("fiber.create_atlas", {"fiberId": fiber_id}, timeout=120.0)
                rec.step("fiber.create_atlas (real fiber) unexpected success (dataset resolved)",
                         "atlasDir" in r, f"({ms(dt):.0f}ms) {json.dumps(r)[:300]}")
            except BridgeError as e:
                rec.step("fiber.create_atlas fails CLEANLY (no Lasagna dataset), no dialog hang",
                         e.code == -32005,
                         f"code={e.code} message={e.message} data={e.data}")
            except TimeoutError as e:
                if proc.exit_code() is not None:
                    record_process_death(proc, rec, "fiber.create_atlas real fiber")
                    raise VC3DDiedError("fiber.create_atlas real fiber")
                rec.step("fiber.create_atlas fails CLEANLY (no Lasagna dataset), no dialog hang",
                         False, f"HANG: {e}")

        # 10. fiber.save: the deferred reply confirms persistence without the
        # interactive path's nested event loop.
        sv = call_checked("fiber.save confirms persistence",
                          "fiber.save", {}, timeout=20.0)
        if sv is not None:
            rec.step("fiber.save result shape", sv.get("saved") is True, json.dumps(sv))

        # 11. fiber.launch error paths + the real gesture.
        call_expect_error("fiber.launch (bogus viewer) -> -32002", "fiber.launch",
                          {"viewer": "not-a-viewer",
                           "position": {"x": 0, "y": 0, "z": 0}}, -32002)
        call_expect_error("fiber.launch (point far off the plane) -> -32003", "fiber.launch",
                          {"viewer": "xy plane",
                           "position": {"x": base["x"], "y": base["y"], "z": -50000.0}}, -32003)
        # Real gesture at the xy-plane's live cursor point: launches the
        # workspace, then the seed's optimization needs a Lasagna dataset --
        # success or a clean -32005, never the dataset-picker QFileDialog hang.
        try:
            cur, _ = client.call("canvas.get_cursor_volume_point",
                                 {"viewer": "xy plane"}, timeout=10.0)
            launch_point = cur.get("volumePoint")
        except BridgeError:
            launch_point = None
        if launch_point:
            try:
                r, dt = client.call("fiber.launch",
                                    {"viewer": "xy plane", "position": launch_point},
                                    timeout=60.0)
                rec.step("fiber.launch (real point) returned promptly",
                         r.get("launched") is True, f"({ms(dt):.0f}ms) {json.dumps(r)}")
            except BridgeError as e:
                rec.step("fiber.launch (real point) fails CLEANLY without dataset-picker hang",
                         e.code in (-32005, -32009, -32003),
                         f"code={e.code} message={e.message} data={e.data}")
            except TimeoutError as e:
                if proc.exit_code() is not None:
                    record_process_death(proc, rec, "fiber.launch real point")
                    raise VC3DDiedError("fiber.launch real point")
                rec.step("fiber.launch (real point) fails CLEANLY without dataset-picker hang",
                         False, f"HANG: {e}")

        # 12. fiber.delete error paths, then cleanup of everything we imported
        # (leaves the fixture's fiber set exactly as found).
        call_expect_error("fiber.delete (empty list) -> -32602", "fiber.delete",
                          {"fiberIds": []}, -32602)
        call_expect_error("fiber.delete (unknown id, all-or-nothing) -> -32007", "fiber.delete",
                          {"fiberIds": ["999999999"]}, -32007)
        fl4 = call_checked("fiber.list before cleanup", "fiber.list")
        to_delete = sorted({f["fiberId"] for f in (fl4 or {}).get("fibers", [])} - baseline_ids)
        if to_delete:
            de = call_checked(f"fiber.delete cleanup of {len(to_delete)} imported fiber(s)",
                              "fiber.delete", {"fiberIds": to_delete})
            if de is not None:
                rec.step("fiber.delete removed all imported fibers",
                         sorted(de.get("deleted", [])) == to_delete, json.dumps(de))
            fl5 = call_checked("fiber.list back to baseline after cleanup", "fiber.list")
            if fl5 is not None:
                final_ids = {f["fiberId"] for f in fl5.get("fibers", [])}
                rec.step("fixture fiber set restored to baseline",
                         final_ids == baseline_ids,
                         f"final={sorted(final_ids)} baseline={sorted(baseline_ids)}")

        # Prove none of the dialog-prone calls wedged the event loop.
        call_checked("ping after fiber smoke (proves no hang)", "ping", timeout=10.0)
    except VC3DDiedError:
        return
    finally:
        try:
            for p in tmp_dir.iterdir():
                p.unlink()
            tmp_dir.rmdir()
        except OSError:
            pass
