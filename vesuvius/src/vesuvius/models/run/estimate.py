"""Report what an inference run will cost before it computes anything.

A whole-scroll pass streams for hours, and the thing that decides how many is not the
forward pass — it is how many times the same chunk gets fetched. villa's sliding window
is 192^3 with half overlap while the store is chunked at 128^3, so the two grids do not
align: one patch touches up to 3x3x3 chunks, and neighbouring patches touch them again.

That re-touching is the whole cost. Counting it is exact — it is a property of the two
grids, not a timing — so it can be reported up front, which is what this module does.

The patch positions are not recomputed here. They are handed in from the dataset that
the real run will use, because deriving a second grid is precisely the bug #1247 fixed:
an ROI-origin grid gave 27 chunks and a flattering 384x where the true full-volume grid
gives 125.
"""

from __future__ import annotations

from collections import OrderedDict

MIB = 1024 ** 2
GIB = 1024 ** 3
TIB = 1024 ** 4


def chunk_ranges(start, patch_size, chunk_size):
    """Per-axis chunk index ranges that a patch starting at `start` overlaps."""
    return [
        range(s // c, (s + p - 1) // c + 1)
        for s, p, c in zip(start, patch_size, chunk_size)
    ]


def morton_key(chunk_index):
    """Interleave the bits of a (z, y, x) chunk index — Z-order.

    Keeps a patch's neighbours in all three axes near it in the traversal, so a cache
    of a given size holds a more useful working set than a raster order does.
    """
    z, y, x = chunk_index
    out = 0
    for bit in range(21):
        out |= ((x >> bit) & 1) << (3 * bit)
        out |= ((y >> bit) & 1) << (3 * bit + 1)
        out |= ((z >> bit) & 1) << (3 * bit + 2)
    return out


ORDERS = ("current", "chunk_blocked", "morton")


def order_positions(positions, chunk_size, order):
    """Reorder patch starts under a traversal policy. `current` is villa's own order."""
    if order == "current":
        return list(positions)
    if order not in ORDERS:
        raise ValueError(f"unknown traversal order {order!r}; expected one of {ORDERS}")

    def chunk_of(pos):
        return tuple(s // c for s, c in zip(pos, chunk_size))

    key = chunk_of if order == "chunk_blocked" else (lambda p: morton_key(chunk_of(p)))
    return sorted(positions, key=key)


def simulate(positions, patch_size, chunk_size, cache_chunks, order="current"):
    """Count chunk fetches for this patch list under an LRU cache of `cache_chunks`.

    `cache_chunks <= 0` means no cache at all, so every touch is a fetch — which is
    what villa does today. Returns (fetches, distinct_chunks).
    """
    lru: OrderedDict = OrderedDict()
    fetches = 0
    distinct = set()
    for pos in order_positions(positions, chunk_size, order):
        rz, ry, rx = chunk_ranges(pos, patch_size, chunk_size)
        for cz in rz:
            for cy in ry:
                for cx in rx:
                    key = (cz, cy, cx)
                    distinct.add(key)
                    if cache_chunks <= 0:
                        fetches += 1
                        continue
                    if key in lru:
                        lru.move_to_end(key)
                        continue
                    fetches += 1
                    lru[key] = True
                    if len(lru) > cache_chunks:
                        lru.popitem(last=False)
    return fetches, len(distinct)


def _fmt_bytes(n):
    if n >= TIB:
        return f"{n / TIB:,.2f} TiB"
    if n >= GIB:
        return f"{n / GIB:,.1f} GiB"
    return f"{n / MIB:,.1f} MiB"


def _fmt_hours(seconds):
    if seconds < 90 * 60:
        return f"{seconds / 60:,.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:,.1f} h"
    return f"{seconds / 86400:,.1f} d"


def build_plan(positions, patch_size, chunk_size, itemsize, cache_sizes_gib=(0, 4, 16)):
    """Cost the run. Returns a dict; `format_plan` renders it.

    `chunk_size` is spatial (z, y, x). `itemsize` is bytes per voxel of the input array.
    """
    chunk_bytes = itemsize
    for c in chunk_size:
        chunk_bytes *= c

    rows = []

    # With no cache every touch is a fetch, so the traversal order cannot change the
    # count. Simulate it once rather than once per order: on a whole scroll that is
    # tens of millions of touches, and running it three times to print the same number
    # three times is both slow and misleading to read.
    uncached_fetches, distinct = simulate(
        positions, patch_size, chunk_size, cache_chunks=0, order="current"
    )
    rows.append({
        "order": None,                       # None == applies to every order
        "cache_gib": 0,
        "cache_chunks": 0,
        "fetches": uncached_fetches,
        "bytes": uncached_fetches * chunk_bytes,
        "amplification": uncached_fetches / distinct if distinct else 0.0,
    })

    for order in ORDERS:
        for gib in cache_sizes_gib:
            if not gib:
                continue                     # the uncached row above covers this
            cache_chunks = int(gib * GIB // chunk_bytes)
            fetches, _ = simulate(
                positions, patch_size, chunk_size, cache_chunks, order
            )
            rows.append({
                "order": order,
                "cache_gib": gib,
                "cache_chunks": cache_chunks,
                "fetches": fetches,
                "bytes": fetches * chunk_bytes,
                "amplification": fetches / distinct if distinct else 0.0,
            })

    return {
        "n_patches": len(positions),
        "patch_size": list(patch_size),
        "chunk_size": list(chunk_size),
        "chunk_bytes": chunk_bytes,
        "distinct_chunks": distinct,
        "floor_bytes": distinct * chunk_bytes,
        "rows": rows,
    }


def format_plan(plan, bandwidth_mbps, volume_desc="", compressor=None, extra_notes=()):
    """Render the plan as the block `--estimate` prints. Returns a list of lines."""
    out = ["", "--estimate: what this run will cost. Nothing is computed and nothing is written.", ""]
    if volume_desc:
        out.append(f"  volume            {volume_desc}")
    out.append(
        f"  chunks            {'x'.join(str(c) for c in plan['chunk_size'])}"
        f"   {_fmt_bytes(plan['chunk_bytes'])} each"
    )
    out.append(
        f"  patch grid        {'x'.join(str(p) for p in plan['patch_size'])}"
        f"   ->  {plan['n_patches']:,} patches"
    )
    out.append("")
    out.append(
        f"  distinct chunks   {plan['distinct_chunks']:,}"
        f"   {_fmt_bytes(plan['floor_bytes'])}"
        f"   <- the floor: every needed chunk fetched exactly once"
    )
    out.append("")
    out.append(f"  {'traversal':<15} {'cache':>7} {'fetches':>13} {'transfer':>13} {'amplification':>14} {'at ' + format(bandwidth_mbps, '.0f') + ' MB/s':>12}")
    out.append(f"  {'-' * 78}")
    labels = {
        None: "any order",
        "current": "current (villa)",
        "chunk_blocked": "chunk-blocked",
        "morton": "morton",
    }
    for row in plan["rows"]:
        cache = "none" if not row["cache_gib"] else f"{row['cache_gib']} GiB"
        secs = row["bytes"] / (bandwidth_mbps * 1e6) if bandwidth_mbps > 0 else 0
        out.append(
            f"  {labels[row['order']]:<15} {cache:>7} {row['fetches']:>13,} {_fmt_bytes(row['bytes']):>13}"
            f" {row['amplification']:>13.2f}x {_fmt_hours(secs):>12}"
        )
    out.append("")
    out.append("  The first row is what this build does today: no chunk cache, so every touch is")
    out.append("  a fetch — and with nothing cached the visit order cannot change that count.")
    out.append("  The rest are what the same patch list would cost under a cache and a different")
    out.append("  visit order — see villa #1177, #1327 and #1331.")

    if compressor is not None:
        out.append("")
        out.append(
            f"  NOTE: this array declares compressor {compressor!r}, so the bytes above are"
        )
        out.append(
            "  uncompressed size. The transfer will be smaller by the compression ratio;"
        )
        out.append("  the fetch counts and the amplification are unaffected.")
    for note in extra_notes:
        out.append(f"  {note}")
    out.append("")
    return out
