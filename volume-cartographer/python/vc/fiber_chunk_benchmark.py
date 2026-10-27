"""Read-only legacy cache import and isolated chunk conflict-filter benchmarking.

All paths, settings and provenance are explicit inputs. No model/cohort constants,
inference, GPU acquisition, or mutation of existing construction caches.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def import_legacy_cache(manifest, destination):
    """Wrap verified legacy payloads in main's cache contract in a NEW directory.

    The manifest is an explicit provenance attestation, not inferred provenance:
    {version: 1, specification: <native chunk_cache_specification JSON>,
     files: [{path: <original JSON>, sha256: <expected bytes hash>}, ...]}.
    Original files are never changed. Unknown/unverifiable provenance is rejected.
    """
    from . import fiber_trace as ft
    manifest_path = Path(manifest).resolve()
    document = json.loads(manifest_path.read_text())
    spec = document["specification"]
    if document["version"] != 1 or spec["version"] != 1 or not spec["source_identity"]:
        raise ValueError("explicit versioned source/configuration provenance required")
    for key in ("trace", "chunk", "trace_to_base"):
        if key not in spec:
            raise ValueError(f"missing specification field: {key}")
    entries = document["files"]
    if not entries:
        raise ValueError("no legacy files")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("import destination must be new; existing caches are never overwritten")
    # Validate the ENTIRE import before publishing any payload.
    prepared = []
    keys = set()
    for entry in entries:
        path = (manifest_path.parent / entry["path"]).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"legacy payload hash mismatch: {path}")
        payload = json.loads(data)
        ft.load_chunk_paths(str(path))  # native shape/count/finite validation
        if "specification" in payload:
            raise ValueError("input is already a versioned cache, not a legacy payload")
        key = tuple(payload["key"])
        if key in keys:
            raise ValueError(f"duplicate chunk key: {key}")
        keys.add(key)
        core, halo = float(spec["chunk"]["core"]), float(spec["chunk"]["halo"])
        for axis in range(3):
            lo, hi = key[axis]*core-halo, (key[axis]+1)*core+halo
            if abs(payload["lo"][axis]-lo)>1e-6 or abs(payload["hi"][axis]-hi)>1e-6:
                raise ValueError(f"legacy geometry disagrees with attested configuration: {path}")
        payload["specification"] = spec
        payload["legacy_import"] = dict(path=str(path), sha256=digest)
        prepared.append(("_".join(map(str, key))+".json", payload))
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in prepared:
        (destination / name).write_text(json.dumps(payload, allow_nan=False)+"\n")
    (destination / "import-provenance.json").write_text(json.dumps(document, indent=2)+"\n")
    return dict(chunks=len(prepared), destination=str(destination))


def time_filter(paths, normal_manifest, config, *, trace_to_base, workers=(1,), repeats=3):
    """Time only filtering of resident paths after an untimed warmup.

    Normal sampling is warmed but not pinned; timings include any subsequent
    cache misses. Report exact rejection parity, never assume it across workers.
    """
    from . import fiber_trace as ft
    if repeats < 1 or not workers or any(w < 1 for w in workers):
        raise ValueError("positive repeats and workers required")
    if not math.isfinite(trace_to_base) or trace_to_base <= 0:
        raise ValueError("positive trace scale required")
    normals = ft.open_normal_sampler(str(normal_manifest), trace_to_base)
    chunks = [ft.load_chunk_paths(str(path)) for path in paths]
    if not chunks:
        raise ValueError("at least one chunk required")
    cfg = ft.ChunkConfig.from_json(json.dumps(config))
    expected = []
    for chunk in chunks:
        _, raw = ft.filter_chunk_paths(chunk, cfg, normals, trace_to_base)
        expected.append(json.loads(raw)["rejected"])
    rows = []
    for count in workers:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            actual = []
            for chunk in chunks:
                _, raw = ft.filter_chunk_paths(chunk, cfg, normals, trace_to_base, count)
                actual.append(json.loads(raw)["rejected"])
            times.append(time.perf_counter()-start)
            if actual != expected:
                raise RuntimeError("filter rejection mismatch between repeats/workers")
        rows.append(dict(workers=count, chunks=len(chunks), repeats=repeats, seconds=times,
                         min_seconds=min(times), median_seconds=statistics.median(times),
                         max_seconds=max(times), mean_seconds=statistics.mean(times),
                         chunks_per_second=len(chunks)/statistics.mean(times)))
    return dict(timings=rows, rejected=expected, paths=[str(p) for p in paths])


def time_construction(field, normals, trace, config, key, *, workers=(1,4,16,64), repeats=3):
    """Rebuild one chunk without path-cache IO; dataset caches stay warm.

    Includes native scheduling, pinning, seeds and tracing, not inference.
    Dataset handles/configuration are supplied by the caller. GPU settings
    require an existing manager grant; this helper never requests resources.
    """
    from . import fiber_trace as ft
    import numpy as np
    if repeats < 1 or not workers or any(w < 1 for w in workers):
        raise ValueError("positive repeats and workers required")
    expected = ft.build_chunk_paths(field, normals, trace, config, key, 1)
    lines = expected.lines_base
    rows = []
    for count in workers:
        seconds, stages = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            result = ft.build_chunk_paths(field, normals, trace, config, key, count)
            seconds.append(time.perf_counter()-start)
            stages.append(json.loads(result.timings_json))
            actual = result.lines_base
            if (result.seeds != expected.seeds or result.maxima != expected.maxima or
                result.end_reasons != expected.end_reasons or len(actual) != len(lines) or
                any(not np.array_equal(a,b) for a,b in zip(actual, lines))):
                raise RuntimeError("construction changed between worker counts/repeats")
        rows.append(dict(workers=count, repeats=repeats, seconds=seconds, stages=stages,
                         min_seconds=min(seconds), median_seconds=statistics.median(seconds),
                         max_seconds=max(seconds), mean_seconds=statistics.mean(seconds)))
    return dict(key=list(key), maxima=expected.maxima, seeds=expected.seeds,
                paths=len(lines), timings=rows, trace=trace.to_dict(),
                config=json.loads(config.to_json()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("import-legacy")
    migrate.add_argument("manifest", type=Path)
    migrate.add_argument("destination", type=Path)
    timing = commands.add_parser("time-filter")
    timing.add_argument("--normal-manifest", required=True, type=Path)
    timing.add_argument("--config", required=True, type=Path, help="generic ChunkConfig JSON")
    timing.add_argument("--trace-to-base", required=True, type=float)
    timing.add_argument("--workers", nargs="+", type=int, default=[1,4,16,64])
    timing.add_argument("--repeats", type=int, default=3)
    timing.add_argument("--output", required=True, type=Path)
    timing.add_argument("chunks", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "import-legacy":
        print(json.dumps(import_legacy_cache(args.manifest, args.destination)))
    else:
        result = time_filter(args.chunks, args.normal_manifest, json.loads(args.config.read_text()),
                             trace_to_base=args.trace_to_base, workers=args.workers, repeats=args.repeats)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
