#!/usr/bin/env python3
"""Prepare/audit and run the locked September 2026 classic-checkpoint cohort.

Run with the existing las environment. No training, installs, volume scans or
bulk prefetch. The actual inference/tracer implementations remain shared.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "vesuvius/src"), str(REPO), str(REPO / "gpu_manager/src")]

DEFAULT_CONFIG = REPO / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection.json"


def load_workload(path):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or not config.get("sources"):
        raise ValueError("workload requires version1 and explicit source bindings")
    ids = [s["id"] for s in config["sources"]]
    if len(ids) != len(set(ids)) or config["pilot_source_id"] not in ids:
        raise ValueError("duplicate source IDs or unknown pilot source")
    return config


def validate_locked_membership(split_root, collection):
    """Reject changed test membership/content, including replaced symlinks."""
    from lasagna.inference_provenance import sha256_file
    manifest = json.loads((split_root / "manifest.json").read_text())
    expected = {row["path"]: row["sha256"] for row in manifest["files"]
                if row["source"] == collection and row["split"] == "test"}
    actual = {str(p.relative_to(split_root / "test")): p
              for p in (split_root / "test" / collection).glob("*.json")}
    if actual.keys() != expected.keys():
        raise ValueError(f"locked test membership changed: {collection}")
    for name, path in actual.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"locked annotation hash mismatch: {name}")


def resolve_normal_binding(manager, binding, predictions):
    """Resolve the configured source and exact normal choice, without fallbacks."""
    from lasagna.manager.fiberlets import _completed_dependency, _published_dependency
    from lasagna.manager.catalog import resolve_lasagna_prediction
    sample, scan = binding["sample_id"], binding["volume_id"]
    selection = binding["normal"]
    if selection["kind"] == "catalogue":
        record = resolve_lasagna_prediction(predictions, selection["selector"], sample_id=sample, volume_id=scan)
        return _published_dependency(manager, record)
    if selection["kind"] != "manager-run":
        raise ValueError("unknown normal binding kind")
    normal = _completed_dependency(manager, selection["selector"], expected_kind="lasagna")
    if (normal["source"].get("sample_id"), normal["source"].get("volume_id")) != (sample, scan):
        raise ValueError("approved normal run identifies a different source")
    return {**normal, "base_shape_zyx": normal["manifest"]["base_shape_zyx"],
            "manifest_url": str(normal["manifest_path"])}


def prepare(workload, *, pilot_only=False):
    import numpy as np
    from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
    from lasagna.manager.catalog import CatalogCache, index_volumes, index_lasagna_predictions
    from lasagna.manager.config import load_config
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
    from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import input_root_attributes, validate_input_lattice

    CHECKPOINT, SPLIT, ROOT, BINARY = (Path(workload[k]) for k in ("checkpoint", "split_root", "output_root", "native_binary"))
    CHECKPOINT_SHA = workload["checkpoint_sha256"]
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("classic checkpoint digest differs from prior inference provenance")
    raw = Path(workload["catalog_path"]).read_bytes()
    raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    catalogue_sha = hashlib.sha256(raw).hexdigest()
    cache = CatalogCache(json.loads(raw), {"sha256": catalogue_sha})
    volumes, predictions = index_volumes(cache), index_lasagna_predictions(cache)
    cache_root = subprocess.check_output([str(BINARY.with_name("vc_cache_root"))], text=True).strip()
    manager = replace(load_config(), cache_dir=cache_root)
    audit = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA,
             "catalogue_sha256": catalogue_sha, "split_sha256": sha256_file(SPLIT / "manifest.json"),
             "scope": "pilot" if pilot_only else "full",
             "workload_sha256": json_digest(workload), "sources": []}
    sources, resolved = [], {}
    for binding in workload["sources"]:
        collection, sample, scan, count = (binding[k] for k in ("id", "sample_id", "volume_id", "expected_test_count"))
        if pilot_only and collection != workload["pilot_source_id"]:
            continue
        validate_locked_membership(SPLIT, collection)
        print(f"preflight {collection}: {sample}/{scan}", flush=True)
        matches = [v for v in volumes if (v.sample_id, v.volume_id) == (sample, scan)]
        if len(matches) != 1 or not matches[0].s3_url:
            raise ValueError(f"ambiguous/missing CT: {collection}")
        volume = matches[0]
        if binding["input_volume"].rstrip("/") != volume.s3_url.rstrip("/"):
            raise ValueError("configured CT URL does not match catalogue source identity")
        input_level = int(binding["input_level"])
        coordinate_space = f"{sample}/{scan}@L0"
        resolve_key = (sample, scan, input_level, json.dumps(binding["normal"], sort_keys=True))
        if resolve_key not in resolved:
            normal = resolve_normal_binding(manager, binding, predictions)
            sampler = Vc3dCoordinateSampler(volume.s3_url.rstrip("/"), level=input_level,
                                           cache_root=cache_root, sample_threads=1)
            validate_input_lattice(input_root_attributes(sampler.volume), input_level)
            shape = tuple(int(v) for v in sampler.volume.shape_at(0))
            factor = 1 << input_level
            shape1 = tuple(int(v) for v in sampler.volume.shape_at(input_level))
            if shape1 != tuple((v + factor - 1) // factor for v in shape):
                raise ValueError("CT selected-level extent mismatch")
            if any(abs(a-b) > 1 for a,b in zip(shape, normal["base_shape_zyx"])):
                raise ValueError("normal/base CT extents disagree")
            resolved[resolve_key] = normal, shape
        normal, shape = resolved[resolve_key]
        files = sorted((SPLIT / "test" / collection).glob("*.json"))
        if len(files) != count:
            raise ValueError(f"locked test count changed for {collection}")
        fibers = []
        for path in files:
            document = json.loads(path.read_text())
            if document.get("vc_open_data_coordinate_space", coordinate_space) != coordinate_space:
                raise ValueError(f"coordinate identity mismatch: {path}")
            if document.get("vc_open_data_source_coordinate_level", 0) != 0 or document.get("vc_open_data_source_coordinate_scale_factor", 1) != 1:
                raise ValueError(f"non-base annotation coordinates: {path}")
            if abs(document.get("vc_open_data_source_original_resolution", volume.pixel_size_um) - volume.pixel_size_um) > 1e-6:
                raise ValueError(f"physical spacing mismatch: {path}")
            fiber, = load_fiber_file(path)
            points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > np.array(shape[::-1])-1):
                raise ValueError(f"nonfinite/out-of-bounds reference: {path}")
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length <= 0:
                raise ValueError(f"degenerate reference: {path}")
            fibers.append({"path": str(path), "sha256": sha256_file(path), "length_base_voxels": length,
                           "points": len(points), "min_xyz": points.min(axis=0).tolist(), "max_xyz": points.max(axis=0).tolist(),
                           "embedded_coordinate_space": document.get("vc_open_data_coordinate_space")})
        sources.append({"id": collection, "coordinate_space": coordinate_space,
                        "input_volume": binding["input_volume"].rstrip("/"), "input_level": input_level,
                        "normal_manifest": str(normal["manifest_path"]),
                        "base_voxel_size_um": volume.pixel_size_um, "fibers": [str(f) for f in files]})
        audit["sources"].append({"collection": collection, "sample": sample, "scan": scan,
                                 "base_shape_zyx": shape, "normal_manifest_url": normal["manifest_url"],
                                 "normal_manifest_sha256": normal["manifest_sha256"], "fibers": fibers})
    config = {**workload["evaluation"], "sources": sources, "output": str(ROOT / "full")}
    # Shortest unambiguous WHOLE held-out fiber; no shortened/cropped reference.
    choices = [(f["length_base_voxels"], s["collection"], f["path"]) for s in audit["sources"]
               if s["collection"] == workload["pilot_source_id"] for f in s["fibers"]]
    _, selected, path = min(choices)
    pilot = {**config, "output": str(ROOT / "pilot"),
             "sources": [{**next(s for s in sources if s["id"] == selected), "fibers": [path]}]}
    audit["pilot"] = {"collection": selected, "fiber": path, "directions": 2}
    audit["config_sha256"] = json_digest(pilot if pilot_only else config)
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = [("pilot_bindings.json", audit), ("pilot.json", pilot)] if pilot_only else [
        ("full_bindings.json", audit), ("full.json", config)]
    for name, value in prepared:
        value = json.loads(json.dumps(value))
        target = ROOT / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError(f"refusing to overwrite changed prepared evidence: {target}")
        if not target.exists():
            atomic_write(target, value)
    count = sum(len(s["fibers"]) for s in sources)
    print(f"Prepared {'one whole-fiber pilot/two directions' if pilot_only else f'{count} fibers/{2*count} directions'}. Pilot: {selected}/{Path(path).name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "prepare-pilot", "validate-pilot", "validate-full", "pilot", "full", "profile"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--devices", help="runtime device selection (default from workload)")
    parser.add_argument("--manager-socket")
    parser.add_argument("--manager-priority", type=int)
    parser.add_argument("--unmanaged", action="store_true", help="explicit bypass of configured manager")
    args = parser.parse_args()
    workload = load_workload(args.config)
    devices = args.devices or (workload["pilot_devices"] if args.mode == "pilot" else workload["devices"])
    manager = workload["evaluation"].get("gpu_manager", {})
    if args.mode in ("pilot", "full", "profile") and not args.unmanaged and (
            manager.get("enabled", False) or args.manager_socket is not None):
        from vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation import launch_managed
        return launch_managed([sys.executable, str(Path(__file__).resolve()), args.mode,
                               "--config", str(args.config.resolve())],
                              devices=devices, settings=manager, socket=args.manager_socket,
                              priority=args.manager_priority)
    if args.mode == "profile":
        from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import profile_workload
        profile_workload(workload, devices_override=devices)
        return
    CHECKPOINT, SPLIT, ROOT, BINARY = (Path(workload[k]) for k in ("checkpoint", "split_root", "output_root", "native_binary"))
    validate_only = args.mode.startswith("validate-")
    if validate_only:
        args.mode = args.mode.removeprefix("validate-")
    if args.mode in ("prepare", "prepare-pilot"):
        prepare(workload, pilot_only=args.mode == "prepare-pilot")
        return
    from lasagna.inference_provenance import json_digest, sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import run
    audit = json.loads((ROOT / f"{args.mode}_bindings.json").read_text())
    if audit["workload_sha256"] != json_digest(workload):
        raise ValueError("workload mapping/settings changed after preparation")
    config = json.loads((ROOT / f"{args.mode}.json").read_text())
    if json_digest(config) != audit["config_sha256"]:
        raise ValueError("prepared config changed after audit")
    if sha256_file(CHECKPOINT) != audit["checkpoint_sha256"] or sha256_file(SPLIT / "manifest.json") != audit["split_sha256"]:
        raise ValueError("checkpoint or split changed after preparation")
    for source in audit["sources"]:
        validate_locked_membership(SPLIT, source["collection"])
        for fiber in source["fibers"]:
            if sha256_file(fiber["path"]) != fiber["sha256"]:
                raise ValueError(f"annotation changed: {fiber['path']}")
    expected_normals = {s["collection"]: s["normal_manifest_sha256"] for s in audit["sources"]}
    for source in config["sources"]:
        if sha256_file(source["normal_manifest"]) != expected_normals[source["id"]]:
            raise ValueError(f"normal manifest changed: {source['id']}")
    if args.mode == "full":
        pilot_root = Path(workload.get("pilot_result", ROOT / "pilot"))
        if any((pilot_root / name).exists() for name in ("incomplete.json", "inference_incomplete.json")):
            raise ValueError("pilot is incomplete")
        pilot_summary = json.loads((pilot_root / "summary.json").read_text())
        if not pilot_summary.get("completed") or pilot_summary.get("cases") != 2:
            raise ValueError("successful bidirectional pilot required")
        json.loads((pilot_root / "inference_summary.json").read_text())
    if validate_only:
        print("Checkpoint, locked split membership/content, config and normal manifests verified.", flush=True)
        return
    return run(ROOT / f"{args.mode}.json", CHECKPOINT, devices, BINARY, unmanaged=True)


if __name__ == "__main__":
    raise SystemExit(main())
