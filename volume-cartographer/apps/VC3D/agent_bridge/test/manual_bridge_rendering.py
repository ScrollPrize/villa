"""Manual rendering and flattening checks."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from bridge_client import BridgeClient, BridgeError
from manual_bridge_support import Recorder, VC3DDiedError, record_process_death
from vc3d_process import VC3DProcess

def run_render_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess,
                     seg_ids: set, render_timeout: float = 300.0) -> None:
    """Exercise render.tifxyz.

    Exercises every error path (missing/bad params, unknown segment, unknown
    volume) plus a real, bounded render of a pre-existing segment to a temp
    directory, polled to a terminal job state with the output artifact verified
    on disk. Every timeout checks process liveness. No RenderParamsDialog is
    ever reached (the headless
    startRenderSegment launcher builds the runner args directly), so no step can
    hang on a modal dialog.
    """

    def call_checked(name: str, method: str, params: dict | None = None,
                     timeout: float = 30.0):
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
                          timeout: float = 30.0):
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

    target = None
    for candidate in ("auto_grown_20260416135719054_inp_hr",
                      "auto_grown_20260416140827289"):
        if candidate in seg_ids:
            target = candidate
            break
    if target is None and seg_ids:
        target = sorted(seg_ids)[0]

    # --- Error paths (no tool job launched): validated before any job register. ---
    call_expect_error("render.tifxyz missing segmentId -> -32602",
                      "render.tifxyz", {"outputFormat": "zarr"}, -32602)
    e = call_expect_error("render.tifxyz missing outputFormat -> -32602",
                          "render.tifxyz", {"segmentId": target or "x"}, -32602)
    if e is not None:
        rec.step("render.tifxyz missing outputFormat error data.param == outputFormat",
                 e.data.get("param") == "outputFormat", f"data={e.data}")
    e = call_expect_error("render.tifxyz bad outputFormat -> -32602",
                          "render.tifxyz",
                          {"segmentId": target or "x", "outputFormat": "jpeg"}, -32602)
    if e is not None:
        rec.step("render.tifxyz bad outputFormat error data.param == outputFormat",
                 e.data.get("param") == "outputFormat", f"data={e.data}")
    e = call_expect_error("render.tifxyz unknown segment -> -32007 kind=segment",
                          "render.tifxyz",
                          {"segmentId": "agent-bridge-nonexistent-seg",
                           "outputFormat": "tif_stack"}, -32007)
    if e is not None:
        rec.step("render.tifxyz unknown segment error data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")
    e = call_expect_error("render.tifxyz unknown volumeId -> -32007 kind=volume",
                          "render.tifxyz",
                          {"segmentId": target or "x", "outputFormat": "tif_stack",
                           "volumeId": "agent-bridge-nonexistent-volume"}, -32007)
    if e is not None:
        rec.step("render.tifxyz unknown volumeId error data.kind == volume",
                 e.data.get("kind") == "volume", f"data={e.data}")
    call_expect_error("render.tifxyz scale<=0 -> -32602",
                      "render.tifxyz",
                      {"segmentId": target or "x", "outputFormat": "tif_stack",
                       "scale": 0.0}, -32602)
    call_expect_error("render.tifxyz numSlices<1 -> -32602",
                      "render.tifxyz",
                      {"segmentId": target or "x", "outputFormat": "tif_stack",
                       "numSlices": 0}, -32602)

    if target is None:
        rec.step("render.tifxyz real render (no target segment available)", False,
                 "no segment to render")
        return

    # --- Real, bounded render to a temp dir. tif_stack is the format whose disk
    # artifact (a "layers" dir of TIFFs) is trivially verifiable; a tiny scale +
    # single slice keeps the run bounded. Proves the full launch->job->disk path
    # and that outputFormat is threaded through to --tif-output. ---
    out_dir = tempfile.mkdtemp(prefix="vc3d_bridge_render_")
    job_id = None
    started_at = time.monotonic()
    r = call_checked("render.tifxyz (started, tif_stack)", "render.tifxyz",
                     {"segmentId": target, "outputFormat": "tif_stack",
                      "outputDir": out_dir, "scale": 0.1, "numSlices": 1})
    if r is not None:
        job_id = r.get("jobId")
        rec.step("render.tifxyz result shape {jobId,kind,source,outputDir,outputFormat}",
                 job_id is not None and r.get("kind") == "render.tifxyz"
                 and r.get("source") == "tool" and r.get("outputFormat") == "tif_stack"
                 and isinstance(r.get("outputDir"), str),
                 json.dumps(r))

    if job_id:
        # job.status must carry source:"tool" for this real render.
        jst = call_checked("render.tifxyz job.status carries source=tool", "job.status",
                           {"jobId": job_id})
        if jst is not None:
            rec.step("render.tifxyz job.status source == tool",
                     jst.get("source") == "tool", f"source={jst.get('source')}")

        # Concurrency guard: a second render while the first is active -> -32004.
        call_expect_error("render.tifxyz second render while active -> -32004",
                          "render.tifxyz",
                          {"segmentId": target, "outputFormat": "tif_stack",
                           "outputDir": out_dir}, -32004)

        deadline = time.monotonic() + render_timeout
        last_state = None
        st = None
        while time.monotonic() < deadline:
            st = call_checked("render.tifxyz poll job.status", "job.status",
                              {"jobId": job_id}, timeout=15.0)
            if st is None:
                break
            last_state = st.get("state")
            if last_state in ("succeeded", "failed"):
                break
            time.sleep(2.0)
        elapsed = time.monotonic() - started_at
        rec.step("render.tifxyz (completed within timeout)",
                 last_state in ("succeeded", "failed"),
                 f"final_state={last_state} elapsed={elapsed:.1f}s")
        if last_state == "succeeded":
            # The artifact path the RPC reported; verify TIFFs landed there.
            layers_dir = Path(out_dir) / "layers"
            tifs = list(layers_dir.glob("*.tif")) + list(layers_dir.glob("*.tiff")) \
                if layers_dir.is_dir() else []
            rec.step("render.tifxyz produced a real TIFF stack on disk",
                     len(tifs) > 0,
                     f"layers_dir={layers_dir} exists={layers_dir.is_dir()} tif_count={len(tifs)}")
        elif last_state == "failed":
            rec.step("render.tifxyz real render outcome", False,
                     f"job failed: {st.get('message') if st else '?'}; "
                     f"consoleTail={st.get('consoleTail') if st else '?'}")

    # Clean up the temp output so the fixture tree is left untouched.
    try:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def run_flatten_smoke(client: BridgeClient, rec: Recorder, proc: VC3DProcess,
                      seg_ids: set, flatten_timeout: float = 300.0) -> None:
    """Exercise flatten.slim / flatten.abf / flatten.straighten.

    Exercises every error path for all three RPCs (missing/bad params, unknown
    segment, and the source:"flatten" concurrency guard) plus one REAL, bounded,
    in-process ABF++ flatten of a pre-existing segment, polled to a terminal job
    state with the output artifact verified on disk and then cleaned up. Every
    timeout checks process liveness. No SlimFlattenDialog / ABFFlattenDialog /
    StraightenDialog is ever reached (the headless start* launchers go straight
    from validated params to a suppressDialogs=true job), and every terminal /
    mid-pipeline QMessageBox in the three bespoke job classes is gated behind
    suppressDialogs, so no step can hang on a modal dialog.
    """

    def call_checked(name: str, method: str, params: dict | None = None,
                     timeout: float = 30.0):
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
                          timeout: float = 30.0):
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

    target = None
    for candidate in ("auto_grown_20260416135719054_inp_hr",
                      "auto_grown_20260416140827289"):
        if candidate in seg_ids:
            target = candidate
            break
    if target is None and seg_ids:
        target = sorted(seg_ids)[0]
    probe = target or "x"

    # --- flatten.slim error paths (no job launched: validation precedes launch). ---
    e = call_expect_error("flatten.slim missing segmentId -> -32602",
                          "flatten.slim", {}, -32602)
    if e is not None:
        rec.step("flatten.slim missing segmentId data.param == segmentId",
                 e.data.get("param") == "segmentId", f"data={e.data}")
    call_expect_error("flatten.slim iterations<1 -> -32602",
                      "flatten.slim", {"segmentId": probe, "iterations": 0}, -32602)
    call_expect_error("flatten.slim negative tolerance -> -32602",
                      "flatten.slim", {"segmentId": probe, "tolerance": -1.0}, -32602)
    e = call_expect_error("flatten.slim bad energyType -> -32602",
                          "flatten.slim",
                          {"segmentId": probe, "energyType": "bogus"}, -32602)
    if e is not None:
        rec.step("flatten.slim bad energyType data.param == energyType",
                 e.data.get("param") == "energyType", f"data={e.data}")
    call_expect_error("flatten.slim keepPercent<=0 -> -32602",
                      "flatten.slim", {"segmentId": probe, "keepPercent": 0.0}, -32602)
    call_expect_error("flatten.slim keepPercent>100 -> -32602",
                      "flatten.slim", {"segmentId": probe, "keepPercent": 150.0}, -32602)
    call_expect_error("flatten.slim non-bool inpaintHoles -> -32602",
                      "flatten.slim", {"segmentId": probe, "inpaintHoles": "yes"}, -32602)
    e = call_expect_error("flatten.slim unknown segment -> -32007 kind=segment",
                          "flatten.slim",
                          {"segmentId": "agent-bridge-nonexistent-seg"}, -32007)
    if e is not None:
        rec.step("flatten.slim unknown segment data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")

    # --- flatten.abf error paths. ---
    e = call_expect_error("flatten.abf missing segmentId -> -32602",
                          "flatten.abf", {}, -32602)
    if e is not None:
        rec.step("flatten.abf missing segmentId data.param == segmentId",
                 e.data.get("param") == "segmentId", f"data={e.data}")
    call_expect_error("flatten.abf iterations<1 -> -32602",
                      "flatten.abf", {"segmentId": probe, "iterations": 0}, -32602)
    call_expect_error("flatten.abf downsampleFactor<1 -> -32602",
                      "flatten.abf", {"segmentId": probe, "downsampleFactor": 0}, -32602)
    e = call_expect_error("flatten.abf unknown segment -> -32007 kind=segment",
                          "flatten.abf",
                          {"segmentId": "agent-bridge-nonexistent-seg"}, -32007)
    if e is not None:
        rec.step("flatten.abf unknown segment data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")

    # --- flatten.straighten error paths. ---
    e = call_expect_error("flatten.straighten missing segmentId -> -32602",
                          "flatten.straighten", {}, -32602)
    if e is not None:
        rec.step("flatten.straighten missing segmentId data.param == segmentId",
                 e.data.get("param") == "segmentId", f"data={e.data}")
    call_expect_error("flatten.straighten overlapPasses<0 -> -32602",
                      "flatten.straighten",
                      {"segmentId": probe, "overlapPasses": -1}, -32602)
    call_expect_error("flatten.straighten negative trimMaxEdge -> -32602",
                      "flatten.straighten",
                      {"segmentId": probe, "trimMaxEdge": -5.0}, -32602)
    call_expect_error("flatten.straighten non-bool unbend -> -32602",
                      "flatten.straighten", {"segmentId": probe, "unbend": 1}, -32602)
    e = call_expect_error("flatten.straighten unknown segment -> -32007 kind=segment",
                          "flatten.straighten",
                          {"segmentId": "agent-bridge-nonexistent-seg"}, -32007)
    if e is not None:
        rec.step("flatten.straighten unknown segment data.kind == segment",
                 e.data.get("kind") == "segment", f"data={e.data}")

    if target is None:
        rec.step("flatten.abf real flatten (no target segment available)", False,
                 "no segment to flatten")
        return

    # --- Real, bounded, in-process ABF++ flatten of a pre-existing segment. It
    # runs on a QtConcurrent worker (no subprocess), needs no external tool and
    # no current volume, and writes <segment>_abf into the volpkg. A high
    # downsampleFactor + low iterations keeps it bounded. Proves the full
    # launch->job->flattenJobFinished->disk path with dialogs suppressed. ---
    out_dir = None
    job_id = None
    started_at = time.monotonic()
    r = call_checked("flatten.abf (started)", "flatten.abf",
                     {"segmentId": target, "iterations": 3, "downsampleFactor": 4})
    if r is not None:
        job_id = r.get("jobId")
        out_dir = r.get("outputDir")
        rec.step("flatten.abf result shape {jobId,kind,source,outputDir}",
                 job_id is not None and r.get("kind") == "flatten.abf"
                 and r.get("source") == "flatten"
                 and isinstance(r.get("outputDir"), str) and bool(r.get("outputDir")),
                 json.dumps(r))

    if job_id:
        # job.status must carry source:"flatten" for this real flatten.
        jst = call_checked("flatten.abf job.status carries source=flatten", "job.status",
                           {"jobId": job_id})
        if jst is not None:
            rec.step("flatten.abf job.status source == flatten",
                     jst.get("source") == "flatten", f"source={jst.get('source')}")

        # Concurrency guard: a second flatten (any kind) while one is active ->
        # -32004 with data.source:"flatten".
        e = call_expect_error("flatten.slim while flatten active -> -32004",
                              "flatten.slim", {"segmentId": target}, -32004)
        if e is not None:
            rec.step("flatten concurrency guard data.source == flatten",
                     e.data.get("source") == "flatten", f"data={e.data}")

        deadline = time.monotonic() + flatten_timeout
        last_state = None
        st = None
        while time.monotonic() < deadline:
            st = call_checked("flatten.abf poll job.status", "job.status",
                              {"jobId": job_id}, timeout=15.0)
            if st is None:
                break
            last_state = st.get("state")
            if last_state in ("succeeded", "failed"):
                break
            time.sleep(2.0)
        elapsed = time.monotonic() - started_at
        rec.step("flatten.abf (completed within timeout)",
                 last_state in ("succeeded", "failed"),
                 f"final_state={last_state} elapsed={elapsed:.1f}s")
        if last_state == "succeeded" and out_dir:
            out_path = Path(out_dir)
            meta = out_path / "meta.json"
            rec.step("flatten.abf produced a real tifxyz surface on disk",
                     out_path.is_dir() and meta.is_file(),
                     f"outputDir={out_dir} exists={out_path.is_dir()} "
                     f"meta.json={meta.is_file()}")
        elif last_state == "failed":
            rec.step("flatten.abf real flatten outcome", False,
                     f"job failed: {st.get('message') if st else '?'}")

    # Clean up the produced surface so the fixture tree is left untouched. Only
    # remove a path we know we created (endswith _abf, inside the volpkg).
    if out_dir and out_dir.endswith("_abf") and Path(out_dir).is_dir():
        try:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
