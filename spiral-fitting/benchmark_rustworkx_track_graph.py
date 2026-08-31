#!/usr/bin/env python3
"""Benchmark a rustworkx graph with tracks as nodes and crossings as edges."""

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import rustworkx as rx


def memory_bytes():
    """Return current RSS and peak RSS on Linux."""
    values = {}
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            if key in {"VmRSS", "VmHWM"}:
                values[key] = int(value.split()[0]) * 1024
    return values["VmRSS"], values["VmHWM"]


def gib(value):
    return value / (1 << 30)


def report(stage, started, baseline):
    rss, peak = memory_bytes()
    print(
        json.dumps(
            {
                "stage": stage,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "rss_gib": round(gib(rss), 3),
                "rss_delta_gib": round(gib(rss - baseline), 3),
                "peak_rss_gib": round(gib(peak), 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crossing_cache", type=Path)
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="Build the induced graph on the first N track rows (default: all).",
    )
    parser.add_argument(
        "--track-chunk-size",
        type=int,
        default=250_000,
        help="Number of CSR track rows converted per rustworkx call.",
    )
    parser.add_argument(
        "--node-chunk-size",
        type=int,
        default=1_000_000,
        help="Number of None-valued nodes added per rustworkx call.",
    )
    parser.add_argument(
        "--benchmark-operations",
        action="store_true",
        help="Time degree reads and removal of every 100th track.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    initial_rss, _ = memory_bytes()
    report("start", started, initial_rss)

    stored = np.load(args.crossing_cache, allow_pickle=False)
    offsets = np.asarray(stored["offsets"], dtype=np.int64)
    partners = np.asarray(stored["partners"], dtype=np.int32)
    total_tracks = len(offsets) - 1
    track_count = (
        total_tracks
        if args.max_tracks is None
        else min(total_tracks, args.max_tracks)
    )
    report("crossing_arrays_loaded", started, initial_rss)

    graph_baseline, _ = memory_bytes()
    graph = rx.PyGraph(multigraph=True)
    for start in range(0, track_count, args.node_chunk_size):
        count = min(args.node_chunk_size, track_count - start)
        # The returned NodeIndices is released after each bounded chunk.
        graph.add_nodes_from((None for _ in range(count)))
    gc.collect()
    report("nodes_built", started, graph_baseline)

    edge_count = 0
    records_scanned = 0
    edge_started = time.perf_counter()
    for row_start in range(0, track_count, args.track_chunk_size):
        row_end = min(track_count, row_start + args.track_chunk_size)
        record_start = int(offsets[row_start])
        record_end = int(offsets[row_end])
        chunk_partners = partners[record_start:record_end]
        counts = np.diff(offsets[row_start : row_end + 1])
        sources = np.repeat(
            np.arange(row_start, row_end, dtype=np.int32), counts
        )
        # The cache is a symmetric directed CSR with one record in each
        # direction. Keeping source < partner yields one undirected edge and
        # also restricts a prefix benchmark to its induced subgraph.
        keep = (sources < chunk_partners) & (chunk_partners < track_count)
        kept_sources = sources[keep]
        kept_partners = chunk_partners[keep]
        added = len(kept_sources)
        if added:
            graph.add_edges_from_no_data(zip(kept_sources, kept_partners))
        edge_count += added
        records_scanned += record_end - record_start
        if (
            row_end == track_count
            or row_end % max(args.track_chunk_size, 1_000_000) == 0
        ):
            rss, _ = memory_bytes()
            print(
                json.dumps(
                    {
                        "stage": "edges_progress",
                        "tracks": row_end,
                        "records_scanned": records_scanned,
                        "edges": edge_count,
                        "edge_seconds": round(
                            time.perf_counter() - edge_started, 3
                        ),
                        "rss_gib": round(gib(rss), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    del sources, kept_sources, kept_partners, chunk_partners, counts, keep
    gc.collect()
    report("graph_built", started, graph_baseline)

    if graph.num_nodes() != track_count or graph.num_edges() != edge_count:
        raise RuntimeError("rustworkx graph counts do not match input counts")
    graph_rss, graph_peak = memory_bytes()
    build_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "stage": "result",
                "rustworkx_version": rx.__version__,
                "pid": os.getpid(),
                "total_cache_tracks": total_tracks,
                "graph_nodes": graph.num_nodes(),
                "directed_records_scanned": records_scanned,
                "graph_edges": graph.num_edges(),
                "build_seconds": round(build_seconds, 3),
                "edge_build_seconds": round(
                    time.perf_counter() - edge_started, 3
                ),
                "graph_rss_delta_gib": round(
                    gib(graph_rss - graph_baseline), 3
                ),
                "peak_rss_gib": round(gib(graph_peak), 3),
                "rss_bytes_per_node_and_edge": round(
                    (graph_rss - graph_baseline)
                    / max(1, graph.num_nodes() + graph.num_edges()),
                    2,
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.benchmark_operations:
        operation_started = time.perf_counter()
        degree_sum = sum(graph.degree(node) for node in range(graph.num_nodes()))
        print(
            json.dumps(
                {
                    "stage": "degree_scan",
                    "nodes": graph.num_nodes(),
                    "degree_sum": degree_sum,
                    "expected_degree_sum": 2 * graph.num_edges(),
                    "seconds": round(time.perf_counter() - operation_started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        remove = range(0, track_count, 100)
        remove_count = (track_count + 99) // 100
        old_edges = graph.num_edges()
        operation_started = time.perf_counter()
        graph.remove_nodes_from(remove)
        print(
            json.dumps(
                {
                    "stage": "remove_one_percent_tracks",
                    "removed_nodes": remove_count,
                    "removed_incident_edges": old_edges - graph.num_edges(),
                    "remaining_nodes": graph.num_nodes(),
                    "remaining_edges": graph.num_edges(),
                    "seconds": round(time.perf_counter() - operation_started, 3),
                    "rss_gib": round(gib(memory_bytes()[0]), 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
