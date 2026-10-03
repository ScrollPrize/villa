"""Shared preparation and provenance audit for fiber trace evaluation."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

def validate_locked_membership(split_root, collection, split="test"):
    """Reject changed test membership/content, including replaced symlinks."""
    from lasagna.inference_provenance import sha256_file
    manifest = json.loads((split_root / "manifest.json").read_text())
    expected = {row["path"]: row["sha256"] for row in manifest["files"]
                if row["source"] == collection and row["split"] == split}
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    actual = {str(p.relative_to(split_root / split)): p
              for p in (split_root / split / collection).glob("*.json")}
    if actual.keys() != expected.keys():
        raise ValueError(f"locked test membership changed: {collection}")
    for name, path in actual.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"locked annotation hash mismatch: {name}")


def select_test_files(files, names=None):
    """Select explicit filenames after the complete collection is validated."""
    if names is None:
        return files
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError("test_fibers must be a nonempty list of filenames")
    if len(set(names)) != len(names):
        raise ValueError("duplicate test_fibers selection")
    if not set(names).issubset({p.name for p in files}):
        raise ValueError("test_fibers contains unknown or non-filename entries")
    return [p for p in files if p.name in names]


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
        split = workload.get("split", "test")
        collection, sample, scan = (binding[k] for k in ("id", "sample_id", "volume_id"))
        count = binding[f"expected_{split}_count"]
        if pilot_only and collection != workload["pilot_source_id"]:
            continue
        validate_locked_membership(SPLIT, collection, split)
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
        files = sorted((SPLIT / split / collection).glob("*.json"))
        if len(files) != count:
            raise ValueError(f"locked test count changed for {collection}")
        files = select_test_files(files, binding.get("test_fibers"))
        if "fiber_order_seed" in workload:
            import random
            random.Random(f"{workload['fiber_order_seed']}:{collection}").shuffle(files)
        fibers, excluded = [], []
        for path in files:
            document = json.loads(path.read_text())
            if document.get("vc_open_data_coordinate_space", coordinate_space) != coordinate_space:
                raise ValueError(f"coordinate identity mismatch: {path}")
            if document.get("vc_open_data_source_coordinate_level", 0) != 0 or document.get("vc_open_data_source_coordinate_scale_factor", 1) != 1:
                raise ValueError(f"non-base annotation coordinates: {path}")
            if abs(document.get("vc_open_data_source_original_resolution", volume.pixel_size_um) - volume.pixel_size_um) > 1e-6:
                raise ValueError(f"physical spacing mismatch: {path}")
            fiber, = load_fiber_file(path)
            if len(fiber.control_points_xyz) < 2:
                excluded.append({'path': str(path), 'sha256': sha256_file(path),
                                 'reason': 'no_control_point_bounded_span'})
                continue
            points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
            # Evaluation excludes extrapolated tails outside the CP span. A
            # tail may extend beyond the scan without invalidating its GT span.
            from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index
            cp_indices = [control_point_line_index(fiber, i)
                          for i in range(len(fiber.control_points_xyz))]
            bounded = points[min(cp_indices):max(cp_indices)+1] if cp_indices else points
            if not np.isfinite(points).all() or np.any(bounded < 0) or np.any(bounded > np.array(shape[::-1])-1):
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
                        "base_voxel_size_um": volume.pixel_size_um, "fibers": [f['path'] for f in fibers]})
        audit["sources"].append({"collection": collection, "sample": sample, "scan": scan,
                                 "base_shape_zyx": shape, "normal_manifest_url": normal["manifest_url"],
                                 "normal_manifest_sha256": normal["manifest_sha256"], "fibers": fibers,
                                 "excluded_fibers": excluded})
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
            # These files are reproducible from the locked inputs. Atomic
            # replacement matters; waiting for unrelated multi-terabyte mmap
            # writeback on the evaluation filesystem does not.
            atomic_write(target, value, durable=False)
    count = sum(len(s["fibers"]) for s in sources)
    print(f"Prepared {'one whole-fiber pilot/two directions' if pilot_only else f'{count} fibers/{2*count} directions'}. Pilot: {selected}/{Path(path).name}", flush=True)



def audit_prepared(workload, mode="full"):
    from lasagna.inference_provenance import json_digest, sha256_file
    root = Path(workload["output_root"])
    split = Path(workload["split_root"])
    audit = json.loads((root / f"{mode}_bindings.json").read_text())
    config = json.loads((root / f"{mode}.json").read_text())
    if audit["workload_sha256"] != json_digest(workload):
        raise ValueError("workload mapping/settings changed after preparation")
    if json_digest(config) != audit["config_sha256"]:
        raise ValueError("prepared config changed after audit")
    if (sha256_file(Path(workload["checkpoint"])) != audit["checkpoint_sha256"]
            or sha256_file(split / "manifest.json") != audit["split_sha256"]):
        raise ValueError("checkpoint or split changed after preparation")
    for source in audit["sources"]:
        validate_locked_membership(split, source["collection"], workload.get("split", "test"))
        for fiber in source["fibers"]:
            if sha256_file(Path(fiber["path"])) != fiber["sha256"]:
                raise ValueError(f"annotation changed: {fiber['path']}")
    expected = {s["collection"]: s["normal_manifest_sha256"] for s in audit["sources"]}
    for source in config["sources"]:
        if sha256_file(Path(source["normal_manifest"])) != expected[source["id"]]:
            raise ValueError(f"normal manifest changed: {source['id']}")
    return config, audit


def run_prepared(workload, *, devices, worker_factory=None, cancel_event=None, on_process_start=None):
    from .live_evaluate import run
    audit_prepared(workload)
    return run(Path(workload["output_root"]) / "full.json",
               Path(workload["checkpoint"]), devices, Path(workload["native_binary"]),
               unmanaged=True, worker_factory=worker_factory, cancel_event=cancel_event,
               on_process_start=on_process_start)
