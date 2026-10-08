"""Native beam-search parameter sweep over a completed, immutable prediction cache.

Every variant replays the same directed reference set against the same
published predictions with different ``trace_args``; nothing is inferred and a
missing chunk fails the variant. Ranking uses restart episodes (the existing
80-base-voxel grouping), never the raw failure count: several failures within a
short distance mean the prediction is unsolvable there and shrinking such an
episode is not progress. Per-direction episode counts are kept so that a
variant can be diffed against the baseline direction by direction.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from lasagna.inference_provenance import atomic_write, sha256_file

from .replay_metrics import reference_segments

# Tracer flags a sweep may override, keyed without the leading dashes. Values are
# formatted with ``str``; boolean flags take no value.
VALUE_FLAGS = frozenset({
    "step-voxels", "cone-angle-degrees", "cone-angle-step-degrees", "cone-grid-size",
    "beam-width", "beam-prune-distance-voxels", "beam-lookahead-steps",
    "lookahead-parent-cap", "lookahead-retry-parent-cap",
    "smoothness-weight", "smoothness-normal-weight", "smoothness-tangent-weight",
    "smoothness-free-angle-degrees", "cumulative-smoothness-steps",
    "cumulative-smoothness-tangent-weight", "max-step-factor",
    "sample-pool-radius-voxels",
    "width-relative-tolerance",
    "replay-presence-cutoff",
})
# Flags whose value is one of a fixed set of words.
CHOICE_FLAGS = {"sample-pool-mode": ("max", "mean"), "sample-pool-stencil": ("cross", "cube"),
                "width-weighting": ("linear", "cosine"),
                "replay-presence-mode": ("raw", "modulated")}
SWITCH_FLAGS = frozenset({"exhaustive-lookahead"})
INCOMPLETE_MARKERS = ("incomplete.json", "inference_incomplete.json")


def apply_trace_args(args, overrides):
    """Return ``args`` with each override replacing or appending its flag.

    Existing option/value pairs keep their position; new flags append in sorted
    order so the same overrides always produce the same argument list.
    """
    if not isinstance(overrides, dict):
        raise ValueError("beam variant parameters must be a mapping")
    for key, value in overrides.items():
        if key in VALUE_FLAGS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"beam variant flag {key} requires a number")
        elif key in CHOICE_FLAGS:
            if value not in CHOICE_FLAGS[key]:
                raise ValueError(f"beam variant flag {key} must be one of {CHOICE_FLAGS[key]}")
        elif key in SWITCH_FLAGS:
            if value is not True:
                raise ValueError(f"beam variant switch {key} may only be set to true")
        else:
            raise ValueError(f"unsupported beam variant flag: {key}")
    out, index, replaced = [], 0, set()
    while index < len(args):
        name = args[index]
        key = name[2:] if name.startswith("--") else None
        if (key in VALUE_FLAGS or key in CHOICE_FLAGS) and index + 1 < len(args):
            value = overrides.get(key, None)
            out.extend([name, args[index + 1] if value is None else _format(value)])
            replaced.add(key)
            index += 2
        elif key in SWITCH_FLAGS:
            out.append(name)
            replaced.add(key)
            index += 1
        else:
            out.append(name)
            index += 1
    for key in sorted(overrides):
        if key in replaced:
            continue
        out.append("--" + key)
        if key in VALUE_FLAGS or key in CHOICE_FLAGS:
            out.append(_format(overrides[key]))
    return out


def _format(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def ordered_variants(workload):
    settings = workload.get("beam_sweep")
    if not isinstance(settings, dict) or "variants" not in settings:
        raise ValueError("beam_sweep must contain variants")
    unknown = set(settings) - {"variants", "workers", "accept_incomplete_markers",
                               "inference", "gpu_manager", "cache_remap",
                               "reference_tube_base_voxels", "decoded_cache_bytes",
                               "native_environment"}
    if unknown:
        raise ValueError(f"unsupported beam_sweep keys: {sorted(unknown)}")
    variants = settings["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("beam_sweep.variants must be a nonempty list")
    rows, names = [], set()
    for entry in variants:
        if not isinstance(entry, dict) or set(entry) - {"name", "parameters"} or "name" not in entry:
            raise ValueError("beam variant needs a name and optional parameters only")
        name = entry["name"]
        if not isinstance(name, str) or not name or "/" in name or name in names:
            raise ValueError(f"invalid or duplicate beam variant name: {name!r}")
        parameters = entry.get("parameters", {})
        apply_trace_args([], parameters)  # validation only
        names.add(name)
        rows.append({"name": name, "parameters": parameters})
    if rows[0]["parameters"]:
        raise ValueError("the first beam variant must be the unmodified baseline")
    return rows


def remap_cache_path(path, cache_remap):
    """Move a cache path from the recorded namespace to an independent copy.

    ``cache_remap`` is ``{"from": <recorded namespace dir>, "to": <copy dir>}``.
    The copy must hold the same per-source cache directories (identity, model
    manifest and published channels); the sweep then extends the copy, never
    the recorded namespace, so other benchmarks keep exclusive use of it.
    """
    if cache_remap is None:
        return path
    if not isinstance(cache_remap, dict) or set(cache_remap) != {"from", "to"}:
        raise ValueError("cache_remap must map 'from' and 'to' namespace directories")
    source, target = Path(cache_remap["from"]).resolve(), Path(cache_remap["to"]).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(source):
        raise ValueError(f"cache path is outside the remapped namespace: {path}")
    return str(target / resolved.relative_to(source))


def bind_cached_cohort(workload, *, accept_incomplete_markers=False, cache_remap=None):
    """Validate the completed cached evaluation and return its replay binding.

    Mirrors ``cached_trace_profile.run_full``: the checkpoint, cache roots,
    annotations and normal manifests must match what the completed evaluation
    recorded. ``accept_incomplete_markers`` waives stale marker files only; the
    summary must still say completed and every cache root must be recorded by
    the completed inference. Cached-only replay fails loudly on any missing
    chunk, so a stale marker cannot let a partial prediction pass silently.
    With ``cache_remap`` the recorded cache roots are validated first and the
    binding then points at their copies.
    """
    from .cached_trace_profile import require_single_model_profile
    previous = Path(workload["cached_evaluation"])
    summary = json.loads((previous / "summary.json").read_text())
    require_single_model_profile(summary["config"])
    inventory = json.loads((previous / "inference_summary.json").read_text())
    if not summary.get("completed"):
        raise ValueError("cached evaluation requires completed input")
    markers = [name for name in INCOMPLETE_MARKERS if (previous / name).exists()]
    if markers and not accept_incomplete_markers:
        raise ValueError(f"cached evaluation has incomplete markers: {markers}")
    bindings_path = previous.parent / "full_bindings.json"
    bindings = None
    if bindings_path.exists():
        bindings = json.loads(bindings_path.read_text())
        if bindings["checkpoint_sha256"] != workload["checkpoint_sha256"]:
            raise ValueError("completed bindings checkpoint mismatch")
        for source in bindings["sources"]:
            for fiber in source["fibers"]:
                if sha256_file(fiber["path"]) != fiber["sha256"]:
                    raise ValueError("annotation changed since completed evaluation")
    evaluation = deepcopy(summary["config"])
    evaluation.pop("live_endpoint", None)
    evaluation["render"] = False
    hashes = {}
    for source in evaluation["sources"]:
        if bindings is not None:
            recorded = next(b for b in bindings["sources"] if b["collection"] == source["id"])
            if sha256_file(source["normal_manifest"]) != recorded["normal_manifest_sha256"]:
                raise ValueError("normal manifest changed since completed evaluation")
        root = Path(source["fiber_manifest"]).resolve().parent
        if str(root) not in inventory["cache_roots"].values():
            raise ValueError("cache not recorded by completed inference")
        if cache_remap is not None:
            source["fiber_manifest"] = remap_cache_path(source["fiber_manifest"], cache_remap)
            copy = Path(source["fiber_manifest"]).parent
            for name in ("cache_identity.json", "inference.json"):
                if (copy / name).read_bytes() != (root / name).read_bytes():
                    raise ValueError(f"cache copy differs from recorded cache: {copy / name}")
            root = copy
        model = json.loads((root / "inference.json").read_text())["model"]
        if model["checkpoint_sha256"] != workload["checkpoint_sha256"]:
            raise ValueError("cached checkpoint mismatch")
        for path in [*source["fibers"], source["normal_manifest"]]:
            hashes[path] = sha256_file(path)
    return {"evaluation": evaluation, "input_sha256": hashes, "previous": str(previous),
            "previous_failures": summary["failures"], "previous_cases": summary["cases"],
            "incomplete_markers": markers, "cache_remap": cache_remap}


def direction_episodes(summary):
    """Restart episodes per directed case, keyed so variants can be diffed."""
    rows = {}
    for source in summary["sources"]:
        for case in source["cases"]:
            key = f'{source["source_id"]}:{case["source_index"]}:{case["run_index"]}:{case["direction"]}'
            if key in rows:
                raise ValueError(f"duplicate directed case: {key}")
            rows[key] = {"episodes": len(reference_segments(case)) - 1,
                         "raw_failures": len(case["failures"])}
    return rows


def ranking_key(row):
    if row["status"] != "completed":
        return (1, float("inf"), 0.0, row["name"])
    result = row["global_result"]
    return (0, int(result["failures"]), -float(result["mean_segment_length_mm"]), row["name"])


def matrix_markdown(rows, baseline_name):
    out = ["# Beam parameter sweep", "",
           "One frozen prediction cache, checkpoint and reference set; only `trace_args` "
           "differ. Restarts are 80-base-voxel episodes. Improved/worsened count directed "
           "cases whose episode count fell/rose against the baseline.", "",
           "| Rank | Variant | Overrides | Restarts | Raw failures | Improved | Worsened | "
           "Failure-free | Mean mm | Median mm | Wall s | Status |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    ranked = {row["name"]: index + 1 for index, row in enumerate(
        sorted((row for row in rows if row["status"] == "completed"), key=ranking_key))}
    for row in rows:
        overrides = "`" + json.dumps(row["parameters"], sort_keys=True) + "`" if row["parameters"] else "baseline"
        if row["status"] == "completed":
            g = row["global_result"]
            values = (g["failures"], row["raw_failures"], row["improved_cases"], row["worsened_cases"],
                      row["failure_free_cases"], f'{g["mean_segment_length_mm"]:.3f}',
                      f'{g["median_segment_length_mm"]:.3f}', f'{row["wall_seconds"]:.1f}')
        else:
            values = ("—",) * 8
        out.append(f"| {ranked.get(row['name'], '—')} | {row['name']} | {overrides} | "
                   + " | ".join(str(v) for v in values) + f" | {row['status']} |")
    return "\n".join(out) + "\n"


def next_attempt_root(candidate_root):
    candidate_root = Path(candidate_root)
    indices = [int(path.name.removeprefix("attempt_"))
               for path in candidate_root.glob("attempt_[0-9][0-9][0-9]")]
    return candidate_root / f"attempt_{max(indices, default=0) + 1:03d}"


def _validate_completed(pointer, variant, identity_sha, accepted_shas=()):
    saved = json.loads(pointer.read_text())
    if saved["variant"] != variant or saved["matrix_identity_sha256"] not in {identity_sha, *accepted_shas}:
        raise ValueError("completed beam sweep candidate provenance changed")
    if saved["matrix_identity_sha256"] != identity_sha:
        # Re-stamp a pointer written under a superseded identity hash.
        atomic_write(pointer, dict(saved, matrix_identity_sha256=identity_sha))
    if saved["row"]["status"] == "completed":
        metrics = Path(saved["attempt_root"]) / "metrics.json"
        if not metrics.is_file() or sha256_file(metrics) != saved["metrics_sha256"]:
            raise ValueError("completed beam sweep candidate metrics changed")
    return saved["row"]


def run_variant(binding, variant, binary, attempt_root, *, workers=None, run_cached=None):
    """Replay one variant into ``attempt_root`` and return its metric row."""
    import subprocess
    from .cached_trace_profile import run_cached_config, write_metric_comparison
    run_cached = run_cached_config if run_cached is None else run_cached
    evaluation = deepcopy(binding["evaluation"])
    evaluation["trace_args"] = apply_trace_args(evaluation.get("trace_args", []), variant["parameters"])
    evaluation["output"] = str(Path(attempt_root) / "full")
    if workers is not None:
        evaluation["workers"] = int(workers)
    if binding.get("decoded_cache_bytes") is not None:
        evaluation["decoded_cache_bytes"] = int(binding["decoded_cache_bytes"])
    if binding.get("native_environment"):
        evaluation["native_environment"] = dict(binding["native_environment"])
    if binding.get("reference_tube_base_voxels") is not None:
        # Demand sessions prefetch the whole acceptance tube (low priority,
        # drained before the variant finishes). Once it is warm, later variants
        # read only published chunks and never register for a GPU grant.
        evaluation["reference_prefetch_margin_base_voxels"] = float(binding["reference_tube_base_voxels"])
    attempt_root = Path(attempt_root)
    attempt_root.mkdir(parents=True, exist_ok=False)
    config_path = attempt_root / "evaluation.json"
    atomic_write(config_path, evaluation)
    evaluation["reference_plan"] = json.loads(subprocess.check_output(
        [str(binary), "replay-references", "--config", str(config_path)], text=True))
    if any(sha256_file(path) != digest for path, digest in binding["input_sha256"].items()):
        raise ValueError("input changed during reference planning")
    atomic_write(config_path, evaluation)
    started = time.monotonic()
    result = run_cached(evaluation, binary)
    wall = time.monotonic() - started
    expected_cases = 2 * sum(len(fiber["runs"]) for source in evaluation["reference_plan"]["sources"]
                             for fiber in source["fibers"])
    if result["cases"] != expected_cases:
        raise ValueError("cached replay coverage differs from selected reference plan")
    report = write_metric_comparison(result, attempt_root / "metrics.json")["after"]
    episodes = direction_episodes(result)
    atomic_write(attempt_root / "direction_episodes.json", episodes)
    return {"name": variant["name"], "parameters": variant["parameters"], "status": "completed",
            "trace_args": evaluation["trace_args"], "attempt_root": str(attempt_root),
            "wall_seconds": wall, "native_wall_seconds": result.get("wall_seconds"),
            "global_result": report["global_result"], "scrolls": report["scrolls"],
            "collections": report["collections"],
            "raw_failures": sum(row["raw_failures"] for row in episodes.values()),
            "failure_free_cases": sum(row["episodes"] == 0 for row in episodes.values()),
            "direction_episodes": episodes}


def compare_to_baseline(row, baseline):
    if row["status"] != "completed" or baseline is None:
        row["improved_cases"] = row["worsened_cases"] = None
        return row
    mine, base = row["direction_episodes"], baseline["direction_episodes"]
    if mine.keys() != base.keys():
        raise ValueError("variant and baseline evaluated different directed cases")
    row["improved_cases"] = sum(mine[k]["episodes"] < base[k]["episodes"] for k in mine)
    row["worsened_cases"] = sum(mine[k]["episodes"] > base[k]["episodes"] for k in mine)
    row["improved"] = sorted(k for k in mine if mine[k]["episodes"] < base[k]["episodes"])
    row["worsened"] = sorted(k for k in mine if mine[k]["episodes"] > base[k]["episodes"])
    return row


def manager_settings(workload):
    """Demand-inference GPU settings, or ``None`` for cached-only replay.

    ``beam_sweep.inference`` selects the mode: ``"demand"`` extends the bound
    prediction namespace through the GPU manager exactly as the short benchmark
    does, which is required whenever a variant traces where the baseline never
    went; ``"cached"`` replays the published cache only and fails on a miss.
    """
    from .live_evaluate import demand_gpu_manager_settings
    mode = workload["beam_sweep"].get("inference", "cached")
    if mode == "cached":
        return None
    if mode != "demand":
        raise ValueError("beam_sweep.inference must be 'cached' or 'demand'")
    manager = workload["beam_sweep"].get("gpu_manager", workload.get("evaluation", {}).get("gpu_manager"))
    settings = demand_gpu_manager_settings({"gpu_manager": manager}, workload["devices"])
    if settings is None:
        raise ValueError("demand inference requires an enabled gpu_manager")
    return settings


def _session_replay(open_session, *, lock_wait_seconds=1800, sleep=time.sleep, clock=time.monotonic):
    """Replay each variant in its own demand-inference session.

    A session holds the prediction cache writer locks, and other benchmarks on
    this host open the same caches, so the lock is taken per variant rather than
    for the whole sweep. A busy writer is retried for ``lock_wait_seconds``.
    """
    def run_cached(evaluation, binary):
        deadline = clock() + lock_wait_seconds
        while True:
            try:
                session = open_session()
            except RuntimeError as error:
                if "already has a writer" not in str(error) or clock() >= deadline:
                    raise
                print(f"beam-sweep waiting for cache writer: {error}", flush=True)
                sleep(30)
                continue
            try:
                session.run(evaluation)
            finally:
                session.close()
            return json.loads((Path(evaluation["output"]) / "summary.json").read_text())
    return run_cached


def run_sweep(workload):
    import os
    settings = workload["beam_sweep"]
    variants = ordered_variants(workload)
    binding = bind_cached_cohort(workload,
        accept_incomplete_markers=bool(settings.get("accept_incomplete_markers", False)),
        cache_remap=settings.get("cache_remap"))
    tube = settings.get("reference_tube_base_voxels")
    if tube is not None:
        if isinstance(tube, bool) or not isinstance(tube, (int, float)) or tube <= 0:
            raise ValueError("reference_tube_base_voxels must be a positive number")
        if settings.get("inference", "cached") != "demand":
            raise ValueError("reference_tube_base_voxels requires demand inference")
    binding["reference_tube_base_voxels"] = tube
    cache_bytes = settings.get("decoded_cache_bytes")
    if cache_bytes is not None and (isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int)
                                    or cache_bytes < (1 << 30)):
        raise ValueError("decoded_cache_bytes must be an integer of at least 1 GiB")
    binding["decoded_cache_bytes"] = cache_bytes
    native_environment = settings.get("native_environment", {})
    if not isinstance(native_environment, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in native_environment.items()):
        raise ValueError("native_environment must map strings to strings")
    binding["native_environment"] = native_environment
    root = Path(workload["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    binary = Path(workload["native_binary"])
    manager = manager_settings(workload)
    if manager is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        run_cached = None
    else:
        from .cached_trace_profile import CachedReplaySession
        checkpoint = Path(workload["checkpoint"])
        if sha256_file(checkpoint) != workload["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        run_cached = _session_replay(lambda: CachedReplaySession(
            binding["evaluation"], binary, checkpoint, devices=workload["devices"], gpu_manager=manager))
    return _run_variants(workload, variants, binding, binary, root, run_cached=run_cached)


def _fixed_workload(workload):
    """The result-determining part of a sweep workload.

    The variant list is deliberately outside the identity: a step-by-step
    search edits it between runs, and each completed variant carries its own
    parameters in its result pointer. The prefetch margin only changes which
    chunks are warm, and GPU manager settings only change scheduling, so both
    are outside as well.
    """
    settings = workload.get("beam_sweep", {})
    fixed = {key: value for key, value in workload.items() if key not in ("beam_sweep", "evaluation")}
    fixed["evaluation"] = {key: value for key, value in workload.get("evaluation", {}).items()
                           if key != "gpu_manager"}
    # Worker count and the decoded-chunk cache budget change only scheduling
    # and memory use; every directed case is traced single-threaded either way.
    fixed["beam_sweep"] = {key: value for key, value in settings.items()
                           if key not in ("variants", "reference_tube_base_voxels", "gpu_manager",
                                          "workers", "decoded_cache_bytes", "native_environment")}
    return fixed


def _run_variants(workload, variants, binding, binary, root, *, run_cached=None):
    settings = workload["beam_sweep"]
    # The native binary hash is recorded per variant, not in the sweep identity:
    # a search may extend the tracer with new options while it runs, and each
    # result row names the exact binary that produced it.
    identity = {"workload": _fixed_workload(workload), "checkpoint_sha256": workload["checkpoint_sha256"],
                "previous": binding["previous"],
                "previous_failures": binding["previous_failures"],
                "incomplete_markers_waived": binding["incomplete_markers"],
                "input_sha256": binding["input_sha256"]}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    provenance = root / "sweep_provenance.json"
    superseded_shas = set()
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        # Compare under the current rule so that a provenance written when
        # more keys were part of the identity still matches.
        saved_identity = {key: value for key, value in saved.items() if key in identity and key != "workload"}
        saved_identity["workload"] = _fixed_workload(saved["workload"])
        if saved_identity != identity:
            raise ValueError("beam sweep identity changed; use a new output root")
        if saved.get("sha256") != identity_sha:
            superseded_shas.add(saved["sha256"])
            superseded_shas.update(saved.get("superseded_sha256", []))
            atomic_write(provenance, dict(identity, sha256=identity_sha,
                                          superseded_sha256=sorted(superseded_shas)))
        else:
            superseded_shas.update(saved.get("superseded_sha256", []))
    else:
        atomic_write(provenance, dict(identity, sha256=identity_sha))
    workers = settings.get("workers")
    rows, baseline = [], None
    for variant in variants:
        candidate_root = root / variant["name"]
        candidate_root.mkdir(exist_ok=True)
        pointer = candidate_root / "result.json"
        previous_row = (_validate_completed(pointer, variant, identity_sha, superseded_shas)
                        if pointer.exists() else None)
        if previous_row is not None and previous_row["status"] == "completed":
            row = previous_row
            print(f"beam-sweep skip {variant['name']}: completed", flush=True)
        else:
            if previous_row is not None:
                # A failed attempt is retried on rerun; its attempt root is kept.
                print(f"beam-sweep retry {variant['name']}: {previous_row.get('error')}", flush=True)
            attempt_root = next_attempt_root(candidate_root)
            print(f"beam-sweep start {variant['name']} attempt={attempt_root.name} "
                  f"parameters={variant['parameters']}", flush=True)
            try:
                row = run_variant(binding, variant, binary, attempt_root, workers=workers,
                                  run_cached=run_cached)
            except Exception as error:
                row = {"name": variant["name"], "parameters": variant["parameters"], "status": "failed",
                       "attempt_root": str(attempt_root), "error": f"{type(error).__name__}: {error}"}
                atomic_write(candidate_root / "failure.json", row)
                if baseline is None:
                    raise
            row["native_sha256"] = sha256_file(binary)
            saved = {"variant": variant, "matrix_identity_sha256": identity_sha,
                     "native_sha256": row["native_sha256"], "attempt_root": str(attempt_root),
                     "metrics_sha256": (sha256_file(Path(attempt_root) / "metrics.json")
                                        if row["status"] == "completed" else None), "row": row}
            atomic_write(pointer, saved)
        if baseline is None:
            if row["status"] != "completed":
                raise RuntimeError("baseline variant did not complete")
            baseline = row
            if row["global_result"]["failures"] != binding["previous_failures"]:
                print(f"beam-sweep baseline restarts {row['global_result']['failures']} differ from the "
                      f"cached evaluation's {binding['previous_failures']}; compare against this baseline",
                      flush=True)
        compare_to_baseline(row, baseline)
        rows.append(row)
        _write_matrix(root, rows, baseline["name"])
        g = row.get("global_result", {})
        print(f"beam-sweep done {variant['name']}: status={row['status']} "
              f"restarts={g.get('failures')} improved={row.get('improved_cases')} "
              f"worsened={row.get('worsened_cases')}", flush=True)
    _write_matrix(root, rows, baseline["name"])
    atomic_write(root / "ranking.json", [_public(r) for r in sorted(rows, key=ranking_key)])
    atomic_write(root / "completed.json", {"completed": True, "variants": len(rows),
                 "matrix_identity_sha256": identity_sha})
    return rows


def _public(row):
    return {key: value for key, value in row.items() if key != "direction_episodes"}


def _write_matrix(root, rows, baseline_name):
    atomic_write(root / "matrix.json", [_public(r) for r in rows])
    (root / "matrix.md").write_text(matrix_markdown(rows, baseline_name))
