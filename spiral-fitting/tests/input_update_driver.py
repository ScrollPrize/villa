"""Input updates of a resident GPU fit, on real data (stage 2b acceptance).

    python tests/input_update_driver.py STORE DATASET WORK [Z_BEGIN Z_END]

STORE is an ordinary object store (copied as a hub; never modified),
DATASET a dataset directory (spiral-scroll.json, umbilicus.json, lasagna
inputs, outer shell, ...), WORK an empty directory. Needs a GPU. Drives a
hub-bound ServiceState in this process (no HTTP), so the resident model and
optimiser can be inspected around each update, and writes WORK/result.json:

  - initialize at the hub head, train, preview;
  - an editor retires a fitted surface and edits a collection's metadata
    (input counts and the atlas layout change) and pushes;
  - update in place: model parameters, optimiser moments, scheduler and
    iteration are compared bitwise across the installation, and the peak
    device memory of the update is recorded beside the old and new atlas
    sizes;
  - train on, finite losses, preview (old previews keep their position);
  - save a checkpoint and rebuild from it: it resumes at its position;
  - a refused update (a position the hub never reached) leaves a usable
    fit; a failure past the release is recovery-required, and a rebuild
    from the checkpoint recovers it.
"""
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)

import torch  # noqa: E402

import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402

import spiral_service  # noqa: E402
from service_project import with_project  # noqa: E402
from fit_session import resolve_dataset_root  # noqa: E402

store, dataset, work = sys.argv[1:4]
z_begin, z_end = (int(value) for value in sys.argv[4:6]) if len(sys.argv) > 5 else (8900, 9300)
work = Path(work).resolve()
work.mkdir(parents=True, exist_ok=True)
result = {"steps": []}
T0 = time.monotonic()


def step(name, **values):
    entry = {"step": name, "t": round(time.monotonic() - T0, 1), **values}
    result["steps"].append(entry)
    print(json.dumps(entry, default=str)[:600], flush=True)


hub = work / "hub.vcobj"
shutil.copyfile(store, hub)
vos.hub_init(str(hub), "acceptance-curator")
editor = str(work / "editor.vcobj")
vos.hub_clone(str(hub), editor, "editor")
binding = spiral_service.bind_object_store(
    str(work / "service.vcobj"), "spiral-service-acceptance", str(hub))
resolution = spiral_service.bind_service_paths(
    resolve_dataset_root(dataset), work / "output", work / "cache")
REQUEST = {"run": {"z_begin": z_begin, "z_end": z_end, "config": {
    "optimizer_num_training_steps": 2000, "dense_spacing_mode": "grad_mag",
    "loss_weight_dense_spacing_density": 0, "loss_weight_dense_normals": 0,
    "loss_weight_dense_spacing": 0}}}
state = spiral_service.ServiceState(
    dataset_root=str(Path(dataset).resolve()), dataset_resolution=resolution,
    object_store=binding, startup_run=REQUEST["run"])


def settle(what, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = state.status()
        busy = state._building or state._input_update is not None
        if not busy and status["state"] in ("Idle", "Error"):
            return status
        time.sleep(1)
    raise TimeoutError(what)


def train(iterations):
    status = state.status()
    catalog = state.configuration_catalog()
    configuration = {**catalog["defaults"], **(status.get("applied_config") or {})}
    configuration = {key: configuration[key] for key in catalog["defaults"]}
    before = status["current_iteration"]
    state.run({"iterations": iterations, "configuration": configuration,
               "expected_session_revision": state.session_revision,
               "autosave_on_pause": False,
               "dt_loss_schedule": {"enabled": False, "last_fraction": 0.25}})
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        status = state.status()
        if status["state"] == "Idle" and status["current_iteration"] >= before + iterations:
            break
        if status["state"] == "Error":
            raise RuntimeError(status["error"])
        time.sleep(1)
    total = (status.get("latest_metrics") or {}).get("total_loss")
    return status, total


def preview():
    before = state.status().get("preview_source_iteration")
    state.export_preview({})
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        status = state.status()
        if status.get("preview_publish_error"):
            return {"error": status["preview_publish_error"]}
        if (not status.get("preview_exporting") and status.get("preview_artifact")
                and status.get("preview_source_iteration") == status["current_iteration"]
                and status.get("preview_source_iteration") != before):
            return status["preview_artifact"]
        time.sleep(2)
    return {"error": "timed out"}


def snapshot(context, completed):
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(item) for item in value)
        return copy.deepcopy(value)
    return {"model": cpu(context.spiral_and_transform.state_dict()),
            "optimiser": cpu(context.optimiser.state_dict()),
            "scheduler": cpu(context.lr_scheduler.state_dict()),
            "completed": completed,
            "model_id": id(context.spiral_and_transform),
            "optimiser_id": id(context.optimiser)}


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape \
            and bool(torch.equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


try:
    state.lease().claim("acceptance", "claim")
    state.initialize(with_project(REQUEST))
    status = settle("initialize")
    if status["state"] != "Idle":
        raise RuntimeError(status["error"])
    step("initialized", position=status["input_position"], summary=status["input_summary"])
    status, total = train(300)
    first_preview = preview()
    step("trained at the first position", iteration=status["current_iteration"], total_loss=total,
         preview=first_preview)

    # An editor retires a fitted surface (fewer patches: the atlas layout
    # changes) and edits a collection's metadata, then pushes.
    context = state.session._context
    fitted = [uuid for uuid in context.verified_patches if isinstance(uuid, str)]
    retired = fitted[len(fitted) // 2]
    with ObjectStore.open(editor, author="editor") as db:
        db.hub_sync(pull=True, push=False)
        pcl = next(r["uuid"] for r in db.objects_by_kind("point_collection")
                   if not (r.get("meta") or {}).get("role"))
        db.acquire_leases([retired, pcl])
        # Retiring changes its relationships' applicability: they are kept
        # as recorded, with every endpoint pinned.
        endpoints = {retired} | {uuid for edge in db.edges_of(retired) for uuid in (edge["a"], edge["b"])}
        expected = {uuid: db.current_object_revision(uuid)["uuid"] for uuid in endpoints}
        with db.write(expected_revisions=expected) as w:
            w.retire_object(retired)
            w.keep_recorded_relationships_of(retired)
        db.set_meta(pcl, "sampling_group", "acceptance-update")
        reached = db.hub_sync(pull=True, push=True)
    step("edited", retired=retired, collection=pcl, position=reached["published_position"],
         conflicts=reached["conflicts"])

    session = state.session
    before = snapshot(context, session.status()["current_iteration"])
    old_atlas = context.patch_atlas.zyxs_flat.nbytes
    old_patches, old_strips = context.num_verified_patches, len(context.unattached_pcl_strips)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.monotonic()
    state.update_inputs({"inputs_position": reached["published_position"],
                         "expected_input_generation": 0})
    status = settle("update")
    torch.cuda.synchronize()
    peak, after_bytes = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    new_atlas = context.patch_atlas.zyxs_flat.nbytes
    after = snapshot(context, session.status()["current_iteration"])
    step("updated in place", seconds=round(time.monotonic() - t, 1), state=status["state"],
         error=status.get("input_preparation_error"), position=status["input_position"],
         input_generation=status["input_generation"],
         same_session=state.session is session,
         patches=[old_patches, context.num_verified_patches],
         strips=[old_strips, len(context.unattached_pcl_strips)],
         retired_still_fitted=retired in context.verified_patches,
         model_unchanged=identical(before["model"], after["model"]),
         optimiser_unchanged=identical(before["optimiser"], after["optimiser"]),
         scheduler_unchanged=identical(before["scheduler"], after["scheduler"]),
         iteration_unchanged=before["completed"] == after["completed"],
         same_objects=(before["model_id"], before["optimiser_id"]) ==
         (after["model_id"], after["optimiser_id"]),
         memory={"base": base, "peak": peak, "after": after_bytes,
                 "old_atlas": old_atlas, "new_atlas": new_atlas,
                 "peak_over_base": peak - base,
                 "both_atlases_would_need_over_base": new_atlas})
    status, total = train(300)
    second_preview = preview()
    step("trained at the updated position", iteration=status["current_iteration"],
         total_loss=total, finite=total is not None and math.isfinite(total),
         preview=second_preview)
    index = json.loads((work / "output" / ".spiral-published" / "preview-index.json").read_text())
    step("preview index", entries=[{key: entry.get(key) for key in (
        "source_fit_iteration", "input_digest", "input_position")}
        for entry in index["previews"].values()])

    saved = state.save_checkpoint({"name": "after-update.ckpt"})["checkpoint_path"]
    from checkpoint_io import load_checkpoint_cpu
    recorded = load_checkpoint_cpu(saved)["input_manifest"]["object_store"]["position"]
    updated_digest = status["input_digest"]
    step("checkpoint", path=saved, position=recorded)

    # A refused update: the fit keeps training.
    head = status["input_position"]["sequence"]
    state.update_inputs({"inputs_position": head + 10 ** 9})
    status = settle("refused update")
    refused = status["input_preparation_error"]
    status, total = train(20)
    step("refused update leaves the fit", phase=refused and refused["phase"],
         recovery_required=refused and refused["recovery_required"],
         state=status["state"], digest_unchanged=status["input_digest"] == updated_digest,
         total_loss=total)

    # A failure past the release: recovery-required, then a rebuild from
    # the checkpoint resumes at its position.
    import fit_spiral
    materialize = fit_spiral.PatchAtlas.materialize

    def fail(self, device=None):
        raise torch.cuda.OutOfMemoryError("simulated allocation failure")
    fit_spiral.PatchAtlas.materialize = fail
    try:
        state.update_inputs({})
        status = settle("failed update")
    finally:
        fit_spiral.PatchAtlas.materialize = materialize
    failed = status["input_preparation_error"]
    step("failure after release", state=status["state"], phase=failed and failed["phase"],
         recovery_required=failed and failed["recovery_required"], error=status.get("error"))
    state.rebuild(with_project({**REQUEST, "paths": {"checkpoint": saved}}))
    status = settle("recovery")
    step("recovered from the checkpoint", state=status["state"],
         position=status["input_position"], digest_matches=status["input_digest"] == updated_digest,
         iteration=status["current_iteration"])
finally:
    (work / "result.json").write_text(json.dumps(result, indent=1, default=str))
    try:
        state.close()
    except Exception:  # noqa: BLE001
        pass
print("done", work / "result.json")
