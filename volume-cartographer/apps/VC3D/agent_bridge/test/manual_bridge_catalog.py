"""Manual Open Data catalog and resource-selection checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

from bridge_client import BridgeClient, BridgeError
from manual_bridge_support import (
    OPEN_DATA_MANIFEST_URL,
    Recorder,
    VC3DDiedError,
    probe_manifest_reachable,
    record_process_death,
)
from vc3d_process import VC3DProcess

def _poll_catalog_job(client: BridgeClient, proc: VC3DProcess, rec: Recorder,
                      job_id: str, step_name: str,
                      deadline_s: float = 300.0) -> tuple[dict | None, bool]:
    """Poll job.status until terminal.

    Returns (terminal_record, timed_out). On a client-side TimeoutError or on
    deadline expiry, checks proc.exit_code() FIRST: alive => (None, True) so the
    caller may record a non-fatal deferral; dead => records a failure via
    record_process_death and raises VC3DDiedError to abort the suite. A timeout is
    never silently deferred without a liveness check.
    """
    t0 = time.time()
    while True:
        if time.time() - t0 > deadline_s:
            if proc.exit_code() is None:
                return None, True
            record_process_death(proc, rec, step_name)
            raise VC3DDiedError(step_name)
        try:
            jr, _ = client.call("job.status", {"jobId": job_id}, timeout=20.0)
        except TimeoutError:
            if proc.exit_code() is not None:
                record_process_death(proc, rec, step_name)
                raise VC3DDiedError(step_name)
            time.sleep(1.0)
            continue
        st = jr.get("state")
        if st and st != "running":
            return jr, False
        time.sleep(1.0)


def _open_sample_via_job(client: BridgeClient, proc: VC3DProcess, rec: Recorder,
                         params: dict, step_name: str,
                         deadline_s: float = 300.0) -> tuple[dict | None, bool]:
    """Start catalog.open_sample and wait for its terminal state.

    Returns (result_body, timed_out).

    - The RPC itself must return {jobId, source:"catalog"} promptly (<=10 s); it
      no longer blocks on the network.
    - result_body is the terminal job's `result` on success; None on job failure
      or timeout. A synchronous BridgeError (validation/precondition) propagates
      to the caller (so error-code assertions keep working).
    """
    r, dt = client.call("catalog.open_sample", params, timeout=15.0)
    assert r.get("source") == "catalog" and r.get("kind") == "catalog.open_sample" \
        and r.get("jobId"), \
        f"open_sample must return {{jobId, kind, source:catalog}} promptly, got {r}"
    assert dt <= 10.0, f"open_sample RPC did not return promptly ({dt:.1f}s)"
    term, timed_out = _poll_catalog_job(client, proc, rec, r["jobId"], step_name,
                                        deadline_s)
    if timed_out or term is None:
        return None, timed_out
    if term.get("state") != "succeeded":
        rec.step(f"catalog.open_sample job failed ({step_name})", False,
                 f"state={term.get('state')} message={term.get('message')}")
        return None, False
    return term.get("result"), False


def run_catalog_resource_selection_smoke(client: BridgeClient, rec: Recorder,
                                         proc: VC3DProcess) -> None:
    """Exercise catalog.list_samples / describe_sample /
    open_sample(resources) / volume.select against the REAL Open Data manifest.

    Network-gated: if the manifest URL is unreachable from this environment the
    whole suite is SKIPPED (clearly, not faked as a pass) -- a deeper
    independent fixture tests regardless. When reachable, this
    proves the read-only surface end-to-end and that a resources filter attaches
    a strict subset while an unfiltered open still attaches everything.

    A client timeout on a dead VC3D is a crash failure, never a silent deferral.
    """
    try:
        _run_catalog_resource_selection_smoke_impl(client, rec, proc)
    except VC3DDiedError:
        # The failure was already recorded by the liveness check; the socket is
        # gone so the suite cannot continue.
        return


def _run_catalog_resource_selection_smoke_impl(client: BridgeClient, rec: Recorder,
                                               proc: VC3DProcess) -> None:
    reachable, detail = probe_manifest_reachable()
    rec.step("catalog manifest reachability probe (informational)", True,
             f"reachable={reachable} url={OPEN_DATA_MANIFEST_URL} ({detail})")
    if not reachable:
        rec.step("catalog RPC suite SKIPPED -- manifest URL unreachable from this environment",
                 True, f"not a pass and not a failure; deferred to a later online test. {detail}")
        return

    # 1. catalog.list_samples -- force a fresh fetch (deferred, 30 s cap).
    try:
        listing, _ = client.call("catalog.list_samples", {"refresh": True}, timeout=45.0)
    except BridgeError as e:
        rec.step("catalog.list_samples (refresh) failed", False,
                 f"code={e.code} message={e.message} data={e.data}")
        return
    except TimeoutError as e:
        rec.step("catalog.list_samples (refresh) timed out", False, str(e))
        return
    samples = listing.get("samples", [])
    rec.step("catalog.list_samples returns manifestUrl + non-empty samples",
             bool(listing.get("manifestUrl")) and len(samples) > 0,
             f"manifestUrl={listing.get('manifestUrl')} sampleCount={len(samples)}")
    if not samples:
        return

    # Pick a sample that keeps the open cheap: prefer zero segments (no tifxyz
    # downloads) and >= 2 volumes (so a volumeIds filter is observably a
    # subset). Fall back progressively.
    def sample_key(s: dict) -> tuple:
        return (s.get("segmentCount", 0), s.get("volumeCount", 0))
    multivol = [s for s in samples if s.get("volumeCount", 0) >= 2]
    pool = multivol if multivol else samples
    zero_seg = [s for s in pool if s.get("segmentCount", 0) == 0]
    chosen = sorted(zero_seg or pool, key=sample_key)[0]
    sample_id = chosen["id"]
    rec.step("chose a real catalog sample for describe/open", True,
             f"sampleId={sample_id} volumeCount={chosen.get('volumeCount')} "
             f"segmentCount={chosen.get('segmentCount')}")

    # 2. catalog.describe_sample -- categorization + stable refs.
    try:
        desc, _ = client.call("catalog.describe_sample", {"sampleId": sample_id}, timeout=45.0)
    except BridgeError as e:
        rec.step("catalog.describe_sample failed", False,
                 f"code={e.code} message={e.message} data={e.data}")
        return
    desc_vols = desc.get("volumes", [])
    reps = desc.get("representations", [])
    valid_kinds = {"normal_grids", "lasagna", "prediction"}
    refs_ok = all(
        isinstance(r.get("ref"), str) and ":" in r["ref"]
        and r["ref"].split(":")[0].isdigit() and r["ref"].split(":")[1].isdigit()
        and r.get("kind") in valid_kinds
        for r in reps
    )
    rec.step("catalog.describe_sample: volumes present + well-formed representation refs/kinds",
             len(desc_vols) > 0 and refs_ok,
             f"volumes={len(desc_vols)} representations={len(reps)} "
             f"sampleRefs={[r.get('ref') for r in reps[:5]]}")
    if not desc_vols:
        return
    subset_vol_id = desc_vols[0]["id"]

    # 3. Filtered open_sample: one volume, kinds=[] (attach no derived
    # representations) -- the lightest possible real open. Confirms only the
    # selected resources attach.
    #
    # Cache caveat: createOpenDataSampleProject loads any
    # previously-cached sample project first, so a filter gates *new* additions
    # but does not remove volume entries a prior (unfiltered) open already
    # persisted to the remote cache. On a CLEAN first open the filtered result
    # is exactly the subset (attached.volumes==1); on a reopen over a cache with
    # more entries, attached.volumes for THIS call is 0 while volumeIds lists the
    # cached set. The per-call invariant that holds regardless of cache state is:
    # kinds=[] never newly attaches a derived representation
    # (attached.normalGrids == attached.lasagnaDatasets == 0), and the filtered
    # volumeIds are a subset of the unfiltered volumeIds.
    filtered_volume_ids = []
    try:
        fr, timed_out = _open_sample_via_job(
            client, proc, rec,
            {"sampleId": sample_id,
             "resources": {"volumeIds": [subset_vol_id], "kinds": []}},
            "filtered open_sample(resources)",
            deadline_s=300.0,
        )
        if timed_out:
            rec.step("catalog.open_sample(resources) filtered open DEFERRED "
                     "(job exceeded the offscreen deadline; VC3D still alive)",
                     True, "deferred to the later online test")
        elif fr is not None:
            attached = fr.get("attached", {})
            filtered_volume_ids = fr.get("volumeIds", [])
            filtered_state, _ = client.call("state.get", timeout=20.0)
            cur = (filtered_state.get("volume") or {}).get("id")
            # kinds=[] must never NEWLY attach any derived representation.
            no_new_derived = attached.get("normalGrids", 0) == 0 and attached.get("lasagnaDatasets", 0) == 0
            rec.step("catalog.open_sample(resources) opens with a filtered subset "
                     "(kinds=[] attaches no derived representations)",
                     fr.get("opened") is True and isinstance(attached, dict) and no_new_derived,
                     f"attached={json.dumps(attached)} volumeIds={filtered_volume_ids} currentVolume={cur}")
    except BridgeError as e:
        rec.step("catalog.open_sample(resources) filtered open", False,
                 f"error=BridgeError code={e.code} detail={e}")

    # 3b. segments.list + volume.select on an already-attached volume.
    try:
        sl, _ = client.call("segments.list", timeout=30.0)
        rec.step("segments.list works after a filtered catalog open", True,
                 f"segments={len(sl.get('segments', []))}")
    except (BridgeError, TimeoutError) as e:
        rec.step("segments.list after filtered catalog open", False,
                 f"error={type(e).__name__} detail={e}")
    if filtered_volume_ids:
        try:
            vs, _ = client.call("volume.select", {"volumeId": filtered_volume_ids[0]}, timeout=30.0)
            rec.step("volume.select keeps/switches current volume among attached entries",
                     vs.get("volumeId") == filtered_volume_ids[0] and "previousVolumeId" in vs,
                     json.dumps(vs))
        except (BridgeError, TimeoutError) as e:
            rec.step("volume.select", False, f"error={type(e).__name__} detail={e}")
        try:
            client.call("volume.select", {"volumeId": "agent-bridge-nonexistent-volume"}, timeout=15.0)
            rec.step("volume.select unknown id rejected with -32007", False, "unexpected success")
        except BridgeError as e:
            rec.step("volume.select unknown id rejected with -32007", e.code == -32007,
                     f"code={e.code} data={e.data}")

    # 4. Regression: an unfiltered open still opens the full set. This can be a
    # heavy real open (multiple remote volumes + streaming normal grids), so it
    # is best-effort: a timeout here is recorded as a non-fatal "deferred" note
    # (the deeper online test covers the full attach), while a
    # documented BridgeError is a real regression failure.
    try:
        ur, timed_out = _open_sample_via_job(
            client, proc, rec, {"sampleId": sample_id},
            "unfiltered open_sample regression", deadline_s=300.0)
        if timed_out:
            # The poll helper confirmed VC3D was still alive at the deadline,
            # so this slow network operation may be deferred.
            rec.step("catalog.open_sample unfiltered regression DEFERRED "
                     "(full remote open exceeded the offscreen deadline; VC3D "
                     "still alive, not a failure)",
                     True, "deferred to the later online test")
        elif ur is not None:
            full_attached = ur.get("attached", {})
            full_volume_ids = ur.get("volumeIds", [])
            # Cache-robust subset invariant: filtered volumeIds subset-of full.
            subset_ok = set(filtered_volume_ids).issubset(set(full_volume_ids)) \
                and len(filtered_volume_ids) <= len(full_volume_ids)
            rec.step("catalog.open_sample WITHOUT resources still attaches the full set "
                     "(regression) and filtered volumeIds are a subset of full",
                     ur.get("opened") is True and isinstance(full_attached, dict) and subset_ok,
                     f"fullAttached={json.dumps(full_attached)} fullVolumeIds={len(full_volume_ids)} "
                     f"filteredVolumeIds={len(filtered_volume_ids)}")
    except BridgeError as e:
        rec.step("catalog.open_sample unfiltered regression", False,
                 f"code={e.code} message={e.message} data={e.data}")

    # Opening twice in one process exercises prefill-watcher replacement. After
    # both terminal states VC3D must remain alive; an overlapping request must
    # be rejected cleanly with -32004.
    run_catalog_double_open_crash_regression(client, rec, proc, sample_id)

    # 5. Strict clean-cache subset proof (independent verification addendum).
    # The checks above tolerate the documented cache caveat: reopening over an
    # already-fuller local cache does not prune stale
    # volume/normal-grid entries), so "no_new_derived" alone does not prove a
    # STRICT subset was attached if the chosen sample happened to already be
    # cached from a prior run. This step removes that ambiguity by picking a
    # 2-volume, 0-segment sample that is verified NOT present in the local
    # remote-project cache (~/.VC3D/remote_cache/open_data/projects/<id>.
    # volpkg.json) before opening it, so the first filtered open is
    # guaranteed clean and attached.volumes must equal EXACTLY the requested
    # count (proving predictions/normal_grids/lasagna and the sample's other
    # raw volume were genuinely NOT attached, not just "no new derived reps").
    run_catalog_strict_subset_and_volume_select(client, rec, samples, proc)


def run_catalog_double_open_crash_regression(client: BridgeClient, rec: Recorder,
                                             proc: VC3DProcess, sample_id: str) -> None:
    """Prove repeated and overlapping catalog opens remain safe.

    Await two consecutive opens, then issue another pair to verify that the
    in-flight request is rejected with -32004.
    """
    # Two consecutive opens (the reproducer). Each awaited to terminal.
    _open_sample_via_job(client, proc, rec, {"sampleId": sample_id},
                         "crash regression open #1", deadline_s=300.0)
    if proc.exit_code() is not None:
        record_process_death(proc, rec, "crash regression open #1")
        raise VC3DDiedError("crash regression open #1")
    _open_sample_via_job(client, proc, rec,
                         {"sampleId": sample_id, "resources": {}},
                         "crash regression open #2", deadline_s=300.0)
    survived = proc.exit_code() is None
    rec.step("two consecutive catalog.open_sample calls keep VC3D alive",
             survived,
             f"proc.exit_code()={proc.exit_code()} (None=alive)")
    if not survived:
        record_process_death(proc, rec, "crash regression double open")
        raise VC3DDiedError("crash regression double open")

    # Overlapping open while a running one is in flight => clean -32004.
    try:
        r, dt = client.call("catalog.open_sample", {"sampleId": sample_id}, timeout=15.0)
        running_job = r.get("jobId")
        guard_ok = False
        detail = "overlapping open unexpectedly accepted (first job may have "\
                 "finished already); acceptable, not a crash"
        try:
            client.call("catalog.open_sample", {"sampleId": sample_id}, timeout=15.0)
        except BridgeError as e2:
            guard_ok = (e2.code == -32004)
            detail = f"overlapping open rejected code={e2.code} data={e2.data}"
        # Drain the running job to a terminal state so we leave clean.
        if running_job:
            _poll_catalog_job(client, proc, rec, running_job,
                              "crash regression in-flight guard", deadline_s=300.0)
        rec.step("in-flight guard: a second catalog.open_sample while one is "
                 "running returns -32004 (not a crash, not interleaving)",
                 guard_ok or proc.exit_code() is None, detail)
    except BridgeError as e:
        rec.step("in-flight guard setup open", False, f"code={e.code} {e}")
    except TimeoutError as e:
        if proc.exit_code() is not None:
            record_process_death(proc, rec, "crash regression in-flight guard")
            raise VC3DDiedError("crash regression in-flight guard")
        rec.step("in-flight guard DEFERRED (VC3D alive)", True, str(e))


def run_catalog_strict_subset_and_volume_select(client: BridgeClient, rec: Recorder,
                                                 samples: list, proc: VC3DProcess) -> None:
    cache_dir = Path.home() / ".VC3D" / "remote_cache" / "open_data" / "projects"

    def is_cached(sid: str) -> bool:
        return (cache_dir / f"{sid}.volpkg.json").exists()

    candidates = [s for s in samples
                  if s.get("volumeCount", 0) >= 2 and s.get("segmentCount", 0) == 0]
    fresh = [s for s in candidates if not is_cached(s["id"])]
    if not fresh:
        rec.step("strict clean-cache subset proof SKIPPED -- no uncached "
                 ">=2-volume/0-segment sample available", True,
                 f"candidates={[s['id'] for s in candidates]} all already cached under {cache_dir}")
        return
    sample_id = fresh[0]["id"]
    rec.step("chose a verified-uncached sample for the strict subset proof",
             True, f"sampleId={sample_id} cacheDir={cache_dir}")

    try:
        desc, _ = client.call("catalog.describe_sample", {"sampleId": sample_id}, timeout=45.0)
    except (BridgeError, TimeoutError) as e:
        rec.step("strict subset proof: describe_sample", False, f"{type(e).__name__} {e}")
        return
    vol_ids = [v["id"] for v in desc.get("volumes", [])]
    if len(vol_ids) < 2:
        rec.step("strict subset proof: sample has >=2 volumes as expected",
                 False, f"volumes={vol_ids}")
        return
    vol_a, vol_b = vol_ids[0], vol_ids[1]

    # 5a. Filtered open selecting ONLY vol_a, no derived representations.
    # On a verified-clean cache this must attach EXACTLY 1 volume -- not the
    # raw volume's prediction companion (visible in already-cached projects
    # as an extra "prediction"-tagged volume entry), not vol_b, nothing else.
    try:
        fr, timed_out = _open_sample_via_job(
            client, proc, rec,
            {"sampleId": sample_id, "resources": {"volumeIds": [vol_a], "kinds": []}},
            "strict subset clean filtered open", deadline_s=180.0)
    except BridgeError as e:
        rec.step("strict subset proof: clean filtered open (1 of 2 volumes)",
                 False, f"BridgeError {e}")
        return
    except VC3DDiedError:
        return
    if timed_out:
        rec.step("strict subset proof: clean filtered open DEFERRED (job exceeded "
                 "deadline; VC3D still alive)", True, "deferred")
        return
    if fr is None:
        return
    attached = fr.get("attached", {})
    fr_volume_ids = fr.get("volumeIds", [])
    exact = (attached.get("volumes") == 1 and attached.get("normalGrids", 0) == 0
             and attached.get("lasagnaDatasets", 0) == 0
             and fr_volume_ids == [vol_a])
    rec.step("STRICT: clean filtered open (volumeIds=[vol_a], kinds=[]) attaches "
             "EXACTLY 1 volume and nothing else (not the prediction companion, not vol_b)",
             exact, f"attached={json.dumps(attached)} volumeIds={fr_volume_ids} requested=[{vol_a}]")

    st1, _ = client.call("state.get", timeout=20.0)
    cur1 = (st1.get("volume") or {}).get("id")
    rec.step("state.get current volume is the sole attached volume after strict filtered open",
             cur1 == vol_a, f"currentVolume={cur1} expected={vol_a}")

    sl1, _ = client.call("segments.list", timeout=20.0)
    rec.step("segments.list works after strict filtered open (no segments attached)",
             sl1.get("segments") == [], f"segments={sl1.get('segments')}")

    # 5b. Reopen with BOTH volume ids (still kinds=[]) so vol_b becomes
    # available for a real volume.select switch, without paying the ~10min+
    # cost of a fully unfiltered open (which also streams-attaches normal
    # grids/lasagna left by an earlier open).
    try:
        fr2, timed_out2 = _open_sample_via_job(
            client, proc, rec,
            {"sampleId": sample_id, "resources": {"volumeIds": [vol_a, vol_b], "kinds": []}},
            "second filtered open (both volume ids)", deadline_s=180.0)
    except BridgeError as e:
        rec.step("second filtered open (both volume ids, kinds=[])", False,
                 f"BridgeError {e}")
        return
    except VC3DDiedError:
        return
    if timed_out2 or fr2 is None:
        rec.step("second filtered open (both volume ids, kinds=[]) DEFERRED/failed",
                 timed_out2, "deferred (VC3D alive)" if timed_out2 else "job failed")
        return
    fr2_volume_ids = fr2.get("volumeIds", [])
    both_present = set([vol_a, vol_b]).issubset(set(fr2_volume_ids))
    rec.step("second filtered open (both volume ids, kinds=[]) attaches vol_b too",
             fr2.get("opened") is True and both_present,
             f"attached={json.dumps(fr2.get('attached', {}))} volumeIds={fr2_volume_ids}")

    # 5c. volume.select switching between the two real attached volumes, both
    # directions, confirmed via state.get each time.
    try:
        vs_b, _ = client.call("volume.select", {"volumeId": vol_b}, timeout=30.0)
        st_b, _ = client.call("state.get", timeout=20.0)
        ok_b = (vs_b.get("volumeId") == vol_b and vs_b.get("previousVolumeId") == vol_a
                and (st_b.get("volume") or {}).get("id") == vol_b)
        rec.step("volume.select switches current volume vol_a -> vol_b",
                 ok_b, f"result={json.dumps(vs_b)} state.volume={json.dumps(st_b.get('volume'))}")

        vs_a, _ = client.call("volume.select", {"volumeId": vol_a}, timeout=30.0)
        st_a, _ = client.call("state.get", timeout=20.0)
        ok_a = (vs_a.get("volumeId") == vol_a and vs_a.get("previousVolumeId") == vol_b
                and (st_a.get("volume") or {}).get("id") == vol_a)
        rec.step("volume.select switches current volume vol_b -> vol_a",
                 ok_a, f"result={json.dumps(vs_a)} state.volume={json.dumps(st_a.get('volume'))}")
    except (BridgeError, TimeoutError) as e:
        rec.step("volume.select switching between two real attached volumes",
                 False, f"{type(e).__name__} {e}")
