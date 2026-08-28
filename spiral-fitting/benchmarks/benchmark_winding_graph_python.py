"""Pure-Python weighted-union baseline for the native graph benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time


def run(nodes: int, edges: int) -> None:
    parent = list(range(nodes))
    size = [1] * nodes
    delta = [0] * nodes
    constraints = []
    forest = [[] for _ in range(nodes)]

    def find(node):
        potential = 0
        while parent[node] != node:
            potential += delta[node]
            node = parent[node]
        return node, potential

    def add(a, b, required):
        root_a, potential_a = find(a)
        root_b, potential_b = find(b)
        if root_a == root_b:
            assert potential_b - potential_a == required
            constraints.append((a, b, required))
            return
        if size[root_a] >= size[root_b]:
            parent[root_b] = root_a
            delta[root_b] = required + potential_a - potential_b
            size[root_a] += size[root_b]
        else:
            parent[root_a] = root_b
            delta[root_a] = potential_b - potential_a - required
            size[root_b] += size[root_a]
        edge = len(constraints)
        constraints.append((a, b, required))
        forest[a].append(edge)
        forest[b].append(edge)

    for node in range(1, nodes):
        add(0, node, node)
    state = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for _ in range(edges - nodes + 1):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        a = state % nodes
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        b = state % nodes
        add(a, b, b - a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=80_000)
    parser.add_argument("--edges", type=int, default=547_115)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run(args.nodes, args.edges)
        timings.append(time.perf_counter() - start)
    print(
        json.dumps(
            {
                "nodes": args.nodes,
                "edges": args.edges,
                "repeats": args.repeats,
                "mean_seconds": statistics.mean(timings),
                "min_seconds": min(timings),
                "median_seconds": statistics.median(timings),
                "max_seconds": max(timings),
            }
        )
    )


if __name__ == "__main__":
    main()
