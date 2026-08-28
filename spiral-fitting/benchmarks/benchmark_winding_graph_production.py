"""Time the native winding graph on real patches/fibers/packed tracks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time

# This benchmark is commonly executed as ``python benchmarks/...py``. In that
# mode Python adds benchmarks/, not the repository root, to sys.path. The
# checkpoint provider reconstructs the training model from root-level modules
# such as checkpoint_io.py and transforms.py, so make that dependency explicit.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from spiral_graph import GraphOptions, InputRole, SpiralThetaProvider, WindingGraph


def emit(stage: str, started: float, graph=None, result=None) -> None:
    payload = {
        "stage": stage,
        "seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if graph is not None:
        stats = graph.stats()
        payload.update(
            patches=stats.patch_count,
            constraints=stats.constraint_count,
            components=stats.component_count,
            anchored_components=stats.anchored_component_count,
            holonomies=stats.holonomy_count,
        )
    if result is not None:
        payload.update(
            committed=result.committed,
            nodes_added=result.nodes_added,
            constraints_added=result.constraints_added,
            anchors_added=result.anchors_added,
            holonomies_added=result.holonomies_added,
        )
        if result.conflict is not None:
            conflict = result.conflict
            cycle = []
            for edge in conflict.cycle:
                item = {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "delta": edge.delta,
                    "geometric_delta": edge.geometric_delta,
                    "closing": edge.closing,
                    "source_type": edge.provenance.source_type,
                    "source": edge.provenance.source,
                    "item": edge.provenance.item,
                    "detail": edge.provenance.detail,
                }
                # Graph/anchor conflicts use graph node ids. Source-theta
                # conflicts instead use source-local ids, which node_name()
                # intentionally does not understand.
                try:
                    item["from_patch"] = graph.node_name(edge.from_node)
                    item["to_patch"] = graph.node_name(edge.to_node)
                except (IndexError, RuntimeError):
                    pass
                cycle.append(item)
            closing = conflict.closing_constraint
            payload["conflict"] = {
                "kind": str(conflict.kind),
                "residual": conflict.residual,
                "cycle_edges": len(cycle),
                "closing_constraint": {
                    "from_node": closing.from_node,
                    "to_node": closing.to_node,
                    "delta": closing.delta,
                    "geometric_delta": closing.geometric_delta,
                    "source_type": closing.provenance.source_type,
                    "source": closing.provenance.source,
                    "item": closing.provenance.item,
                    "detail": closing.provenance.detail,
                },
                "cycle": cycle,
            }
    print(json.dumps(payload, sort_keys=True), flush=True)


def require_committed(result) -> None:
    if not result.committed:
        raise RuntimeError(
            f"stage rejected with {result.conflict.kind}, "
            f"residual {result.conflict.residual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--umbilicus", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument(
        "--invalidate-patch",
        action="append",
        default=[],
        metavar="UUID",
        help=(
            "disable a cached patch as a contact target before adding sources; "
            "repeatable"
        ),
    )
    parser.add_argument(
        "--commit-invalidations",
        action="store_true",
        help="persist requested patch invalidations if subsequent stages commit",
    )
    parser.add_argument(
        "--invalidate-fiber",
        action="append",
        default=[],
        metavar="FILENAME",
        help="omit one exact fiber JSON filename from this source; repeatable",
    )
    parser.add_argument(
        "--absolute",
        type=Path,
        action="append",
        default=[],
        metavar="JSON",
        help="absolute-winding point collection; may be supplied more than once",
    )
    parser.add_argument("--fibers", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--crossings", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=2.5,
        help=(
            "maximum patch-contact distance in surface coordinates for a new "
            "cache (default: 2.5)"
        ),
    )
    args = parser.parse_args()
    if any(value is not None for value in (args.tracks, args.crossings, args.index)):
        if None in (args.tracks, args.crossings, args.index):
            parser.error("--tracks, --crossings, and --index must be supplied together")
    manifest_exists = (
        args.cache is not None and (args.cache / "manifest.json").is_file()
    )

    total_started = time.perf_counter()
    started = time.perf_counter()
    theta = SpiralThetaProvider(
        args.checkpoint,
        umbilicus=args.umbilicus,
        device=args.device,
    )
    emit("checkpoint", started)

    options = GraphOptions()
    options.contact_tolerance = args.contact_tolerance
    if args.cache is None:
        graph = WindingGraph(options=options, theta_provider=theta)
    elif manifest_exists:
        graph = WindingGraph.open(args.cache, theta, options)
    else:
        graph = WindingGraph.create(args.cache, theta, options)

    invalidations_active = False

    def persist() -> None:
        if args.cache is not None and (
            not invalidations_active or args.commit_invalidations
        ):
            graph.save()

    started = time.perf_counter()
    result = graph.add_patches([args.patches])
    emit("patches", started, graph, result)
    require_committed(result)
    persist()

    if args.invalidate_patch:
        invalidated = []
        for patch_id in args.invalidate_patch:
            graph.set_patch_valid(patch_id, False)
            invalidated.append(patch_id)
        invalidations_active = True
        print(json.dumps({
            "stage": "patch_validity",
            "invalid_patches": invalidated,
            "persistent": args.commit_invalidations,
        }, sort_keys=True), flush=True)

    if args.invalidate_fiber:
        invalidations_active = True
        print(json.dumps({
            "stage": "fiber_validity",
            "invalid_fibers": args.invalidate_fiber,
            "persistent": args.commit_invalidations,
        }, sort_keys=True), flush=True)

    if args.absolute:
        started = time.perf_counter()
        result = graph.add_point_collections(args.absolute, InputRole.ABSOLUTE)
        emit("absolute", started, graph, result)
        require_committed(result)
        persist()

    if args.fibers is not None:
        started = time.perf_counter()
        result = graph.add_fibers(
            args.fibers,
            invalid_fibers=args.invalidate_fiber,
        )
        emit("fibers", started, graph, result)
        require_committed(result)
        persist()

    if args.tracks is not None:
        started = time.perf_counter()
        result = graph.add_tracks(args.tracks, args.crossings, args.index)
        emit("tracks", started, graph, result)
        require_committed(result)
        persist()

    audits = graph.holonomy_audits()
    inconsistency_counts = Counter(audit.inconsistency for audit in audits)
    mismatches = [audit for audit in audits if audit.inconsistency != 0]
    print(json.dumps({
        "stage": "geometric_holonomy",
        "cycles": len(audits),
        "consistent_cycles": len(audits) - len(mismatches),
        "inconsistent_cycles": len(mismatches),
        "inconsistency_counts": {
            str(value): count
            for value, count in sorted(inconsistency_counts.items())
        },
        "max_abs_inconsistency": max(
            (abs(audit.inconsistency) for audit in audits), default=0
        ),
        "examples": [
            {
                "constraint_index": audit.constraint_index,
                "reported_holonomy": audit.reported_holonomy,
                "geometric_holonomy": audit.geometric_holonomy,
                "inconsistency": audit.inconsistency,
            }
            for audit in mismatches[:20]
        ],
    }, sort_keys=True), flush=True)

    emit("total", total_started, graph)


if __name__ == "__main__":
    main()
