#!/usr/bin/env python3
"""Reproducible public-API object-store scaling checks (no production data).

Build: ninja -C build vc_object_bindings
Run: .venv/bin/python scripts/benchmark_object_store.py --output /tmp/vcobj-perf.json

Each scenario runs in a fresh process and temporary directory. Timings include
Python bindings, use the store's default durability settings, and are warm-cache
(one untimed warmup unless explicitly marked). Raw samples are milliseconds;
peak RSS includes fixture creation and all operations within that worker. No
latency thresholds: compare scaling and paired operations on the same machine.
SQL is used only for diagnostics and a clearly labelled lower-bound comparison.
"""

import argparse
import array
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import tempfile
import time

from object_store_sqlite import sqlite_rows


def measure(results, name, operation, repeats=5, warmup=True, check=None, **details):
    if warmup:
        value = operation()
        if check:
            check(value)
        del value
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if check:
            check(value)
        del value
    result = dict(name=name, median_ms=statistics.median(samples),
                  min_ms=min(samples), max_ms=max(samples), samples_ms=samples,
                  warmup=warmup, **details)
    results.append(result)
    print(f"{name}: {result['median_ms']:.3f} ms", file=sys.stderr, flush=True)


def count_is(expected):
    def check(value):
        assert len(value) == expected, (len(value), expected)
    return check


def native_plan(path, query):
    # The Python interpreter can bundle a different SQLite than vc_objects.
    # The system CLI is diagnostic only; its version is recorded in the report.
    if not shutil.which("sqlite3"):
        return "sqlite3 CLI unavailable"
    return subprocess.check_output(["sqlite3", "-readonly", str(path), "EXPLAIN QUERY PLAN " + query], text=True)


def seed(store, n, hub=False):
    ids = []
    with store.write() as tx:
        for i in range(n):
            ids.append(tx.upsert_object(dict(
                kind="note" if i == 0 else "track", name=f"object-{i}",
                tags=["rare"] if i == 0 else ["ordinary"],
                meta={"payload": "x" * 128, "index": i})))
        # A fixed 16-node component plus a growing chain and optional hub.
        for i in range(1, n):
            if i != 16:
                tx.add_edge(dict(kind="chain", a=ids[i - 1], b=ids[i]))
        if hub:
            for i in range(17, n):
                tx.add_edge(dict(kind="hub", a=ids[16], b=ids[i]))
    store.checkpoint()
    return ids


def graph(Store, root, n, results):
    with Store.create(str(root / "graph.vcobj"), author="benchmark") as db:
        start = time.perf_counter()
        ids = seed(db, n, hub=True)
        results.append(dict(name="seed", elapsed_ms=(time.perf_counter() - start) * 1000,
                            objects=n, edges=2 * n - 19))
        measure(results, "lookup_uuid", lambda: db.object(ids[0]), repeats=21,
                check=lambda v: count_is(1)([v] if v["uuid"] == ids[0] else []))
        measure(results, "select_one_uuid", lambda: db.select_objects(object_uuids=[ids[0]]), check=count_is(1))
        measure(results, "indexed_rare_kind", lambda: db.objects_by_kind("note"), check=count_is(1))
        measure(results, "select_rare_kind", lambda: db.select_objects(object_kind="note"), check=count_is(1))
        measure(results, "indexed_rare_tag", lambda: db.objects_by_tag("rare"), check=count_is(1))
        measure(results, "select_rare_tag", lambda: db.select_objects(tag="rare"), check=count_is(1))
        measure(results, "component_16", lambda: db.component(ids[0]), check=count_is(16))
        measure(results, "select_component_16", lambda: db.select_objects(connected_to=ids[0]), check=count_is(16))
        measure(results, "export_one_projected", lambda: db.export_json(object_uuids=[ids[0]], fields=["name"]),
                check=lambda s: count_is(1)(json.loads(s)["objects"]))
        measure(results, "component_large_chain", lambda: db.component(ids[16], edge_kinds=["chain"]), check=count_is(n - 16), repeats=3)
        measure(results, "component_large_unfiltered", lambda: db.component(ids[16]), check=count_is(n - 16), repeats=3)
        measure(results, "component_16_filtered", lambda: db.component(ids[0], edge_kinds=["chain"]), check=count_is(16))
        measure(results, "reachable_chain_depth_2", lambda: db.reachable_within(ids[16], 2, edge_kinds=["chain"]), check=count_is(3))
        measure(results, "hub_neighbours_filtered_chain", lambda: db.neighbours(ids[16], edge_kind="chain"), check=count_is(1))
        measure(results, "hub_neighbours", lambda: db.neighbours(ids[16]), check=count_is(n - 17))
        with sqlite3.connect(db.path) as sql:
            ident = sql.execute("SELECT id FROM objects WHERE uuid=?", (bytes.fromhex(ids[16].replace("-", "")),)).fetchone()[0]
            query = "SELECT id, CASE WHEN a_id=? THEN b_id ELSE a_id END FROM edges WHERE (a_id=? OR b_id=?) AND kind IN (?) ORDER BY id"
            params = (ident, ident, ident, "chain")
            literal_query = query.replace("?", str(ident), 3).replace("?", "'chain'")
            results.append(dict(name="legacy_adjacency_query_plan", python_sqlite_rows=sql.execute("EXPLAIN QUERY PLAN " + query, params).fetchall(),
                                system_sqlite_plan=native_plan(db.path, literal_query)))
            measure(results, "python_sqlite_filtered_adjacency_lower_bound", lambda: sql.execute(query, params).fetchall(), check=count_is(1))
            union = "SELECT id, b_id AS neighbour FROM edges WHERE a_id=? AND kind=? UNION SELECT id, a_id AS neighbour FROM edges WHERE b_id=? AND kind=? ORDER BY id"
            union_params = (ident, "chain", ident, "chain")
            assert sql.execute(query, params).fetchall() == sql.execute(union, union_params).fetchall()
            measure(results, "python_sqlite_union_adjacency_lower_bound", lambda: sql.execute(union, union_params).fetchall(), check=count_is(1))
            results.append(dict(name="union_adjacency_query_plan", system_sqlite_plan=native_plan(db.path,
                union.replace("?", str(ident), 1).replace("?", "'chain'", 1).replace("?", str(ident), 1).replace("?", "'chain'", 1))))
            # Isolate statistics as an explanation, without changing the library.
            sql.execute("ANALYZE")
            results.append(dict(name="analyzed_adjacency_query_plan", system_sqlite_plan=native_plan(db.path, literal_query)))
        results.append(dict(name="database_bytes", bytes=Path(db.path).stat().st_size))
    with Store.open(str(root / "graph.vcobj"), read_only=True) as db:
        measure(results, "analyzed_component_16_filtered", lambda: db.component(ids[0], edge_kinds=["chain"]), check=count_is(16))
        measure(results, "analyzed_reachable_chain_depth_2", lambda: db.reachable_within(ids[16], 2, edge_kinds=["chain"]), check=count_is(3))
        # Verify that ANALYZE resolves the expensive traversal at the measured scales.
        measure(results, "analyzed_component_large_chain", lambda: db.component(ids[16], edge_kinds=["chain"]), check=count_is(n - 16), repeats=3)


def writes(Store, root, n, results):
    for batch in (1, 100, n):
        with Store.create(str(root / f"writes-{batch}.vcobj"), author="benchmark") as db:
            ids = [db.upsert_object(dict(kind="note")) for _ in range(10)]
            counter = 0

            def update():
                nonlocal counter
                for offset in range(0, n, batch):
                    with db.write() as tx:
                        for i in range(offset, min(n, offset + batch)):
                            counter += 1
                            tx.set_meta(ids[i % len(ids)], "value", counter)

            measure(results, f"metadata_{n}_updates_batch_{batch}", update, repeats=3,
                    updates=n, transaction_batch=batch)
            assert db.object(ids[(n - 1) % 10])["meta"]["value"] == counter
            assert len(db.journal_since()) == 10 + 4 * n


def geometry(Store, root, n, results):
    for kind in ("fiber", "point_collection"):
        with Store.create(str(root / f"{kind}.vcobj"), author="benchmark") as db:
            # n points, unique values; avoids an unrealistically deduplicated fixture.
            coords = array.array("d", (float(i) for i in range(n * 3)))
            stable_ids = array.array("q", range(n))
            if sys.byteorder != "little":
                coords.byteswap()
                stable_ids.byteswap()
            payload = coords.tobytes()
            name = "line_points" if kind == "fiber" else "points"
            arrays = [dict(name=name, dtype="<f8", shape=[n, 3], data=payload)]
            arrays.append(dict(name="control_points", dtype="<f8", shape=[0, 3], data=b"") if kind == "fiber"
                          else dict(name="point_ids", dtype="<i8", shape=[n], data=stable_ids.tobytes()))
            data = dict(kind=kind, arrays=arrays)
            obj = db.upsert_object(dict(kind=kind))
            revision = db.put_geometry(obj, data)
            db.checkpoint()
            for count in (24, 4096, 1048576, len(payload)):
                measure(results, f"{kind}_read_{count}_bytes", lambda c=count: db.read_geometry_array(revision, name, 0, c),
                        check=lambda v, c=count: count_is(c)(v), bytes=count)

            def many_points():
                return [db.read_geometry_array(revision, name, i * 24, 24) for i in range(100)]

            measure(results, f"{kind}_100_separate_points", many_points, check=count_is(100))
            measure(results, f"{kind}_100_points_batched_ranges",
                    lambda: db.read_geometry_ranges(revision, name, [(i * 24, 24) for i in range(100)]), check=count_is(100))
            measure(results, f"{kind}_100_points_one_range", lambda: db.read_geometry_array(revision, name, 0, 2400), check=count_is(2400))
            assert b"".join(many_points()) == payload[:2400]
            measure(results, f"{kind}_put_identical", lambda: db.put_geometry(obj, data, expected_revision=revision),
                    check=lambda v: count_is(1)([v] if v == revision else []))
            changed = dict(kind=kind, arrays=[dict(a) for a in arrays])
            changed["arrays"][0]["data"] = payload[:-8] + struct.pack("<d", -1.0)
            degree = 0
            for target in (0, 10, 100):
                with db.write() as tx:
                    for i in range(degree, target):
                        other = tx.upsert_object(dict(kind="note"))
                        tx.add_edge(dict(kind="contact", a=obj, b=other,
                                         a_anchor={"points": [{"array": "point_ids" if kind == "point_collection" else name, "index": [i]}]},
                                         b_anchor={"whole_geometry": True}))
                degree = target
                measure(results, f"{kind}_unrelated_edit_preview_degree_{degree}",
                        lambda: db.affected_geometry_edges(obj, changed), check=count_is(0), degree=degree, points=n)
            new_revision = db.put_geometry(obj, changed, expected_revision=revision)
            assert new_revision != revision
            assert db.read_geometry_array(new_revision, name, len(payload) - 8, 8) == struct.pack("<d", -1.0)


def sync(Store, root, n, results):
    with Store.create(str(root / "local.vcobj"), author="benchmark") as local:
        ids = seed(local, n)
        with Store.create(str(root / "remote.vcobj"), author="benchmark") as remote:
            def synchronize():
                report = local.synchronize(remote)
                assert report["applied"] and not report["conflicts"], report
                return report

            measure(results, "initial_sync", synchronize, repeats=1, warmup=False)
            for history in range(1, 6):
                if history > 1:
                    local.set_meta(ids[0], "edit", history)
                    measure(results, f"one_record_sync_history_{history}", synchronize, repeats=1, warmup=False)
                    assert remote.object(ids[0])["meta"]["edit"] == history
                if history in (1, 5):
                    measure(results, f"noop_sync_history_{history}", synchronize, repeats=3,
                            check=lambda r: count_is(0)(list(range(r["local_changes"] + r["remote_changes"]))))
                    with sqlite3.connect(local.path) as sql:
                        count, size = sql.execute("SELECT count(*), sum(length(state)) FROM sync_baselines").fetchone()
                        versions, record_bytes = sql.execute("SELECT count(*), coalesce(sum(length(state)),0) FROM sync_baseline_records").fetchone()
                        results.append(dict(name=f"baseline_storage_history_{history}", count=count,
                                            json_bytes=size + record_bytes, record_versions=versions))
            local.checkpoint()
            results.append(dict(name="database_bytes", bytes=Path(local.path).stat().st_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-python", type=Path, default=Path(__file__).resolve().parents[1] / "build/python")
    parser.add_argument("--output", type=Path, default=Path("/tmp/vcobj-perf.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--sync-sizes", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--geometry-points", type=int, default=262144)
    parser.add_argument("--worker", choices=["graph", "writes", "geometry", "sync"], help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # Same explicit-build import isolation as python/tests/conftest.py.
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "_editable_skbc_volume_cartographer"]
        sys.path.insert(0, str(args.build_python.resolve()))
        import vc.objects
        results = []
        with tempfile.TemporaryDirectory(prefix="vcobj-perf-") as directory:
            globals()[args.worker](vc.objects.ObjectStore, Path(directory), args.n, results)
        print(json.dumps(dict(scenario=args.worker, n=args.n, results=results,
                              module=vc.objects.__file__, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
        return
    if min(args.sizes + args.sync_sizes) < 20 or args.geometry_points < 43691:
        parser.error("graph/sync sizes must be >=20; geometry points must be >=43691")
    report = dict(platform=platform.platform(), python=sys.version, python_sqlite=sqlite3.sqlite_version,
                  system_sqlite_cli=subprocess.check_output(["sqlite3", "--version"], text=True).strip() if shutil.which("sqlite3") else None,
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), scenarios=[])
    cases = [("graph", n) for n in args.sizes] + [("writes", 1000), ("geometry", args.geometry_points)] + [("sync", n) for n in args.sync_sizes]
    for scenario, n in cases:
        print(f"\n{scenario} n={n}", file=sys.stderr, flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--build-python", str(args.build_python.resolve()),
                                 "--worker", scenario, "--n", str(n)], check=True, text=True, stdout=subprocess.PIPE, timeout=300)
        report["scenarios"].append(json.loads(result.stdout))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
