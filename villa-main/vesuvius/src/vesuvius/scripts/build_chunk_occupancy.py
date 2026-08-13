#!/usr/bin/env python3
"""Pre-build and upload a zarr chunk-occupancy bitmap to a shared location.

Intended to be invoked as a workflow step before fan-out, so that many
predict partitions (including ephemeral external workers) can hit the same
cache instead of each re-listing S3. The output URL is typically on the
writable working prefix of the inference run; it is then passed to every
predict partition via ``VESUVIUS_CHUNK_OCCUPANCY_URL``.

Example::

    python -m vesuvius.scripts.build_chunk_occupancy \\
        --volume-path s3://scrollprize-volumes/.../volume.zarr/0 \\
        --output s3://scrollprize-reconstruction/.../chunk-occupancy.npz \\
        --input-anon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

from vesuvius.data.zarr_chunk_index import (
    ENV_OVERRIDE_URL,
    build_chunk_occupancy,
    _url_to_fs,
    _zarray_signature,
)


def read_zarray(volume_path: str, *, anon: bool) -> dict:
    zarray_url = volume_path.rstrip("/") + "/.zarray"
    fs, fs_path = _url_to_fs(zarray_url, anon=anon)
    with fs.open(fs_path, "rb") as f:
        return json.loads(f.read())


def extract_chunks_shape(zarray: dict) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    chunks = tuple(int(v) for v in zarray["chunks"])
    shape = tuple(int(v) for v in zarray["shape"])
    return chunks, shape


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume-path", required=True, help="URL of the zarr v2 array (contains .zarray).")
    parser.add_argument("--output", required=True, help="URL where the .npz cache should be written.")
    parser.add_argument("--input-anon", action="store_true", help="Read the input array with unsigned S3 requests.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    volume_path = args.volume_path.rstrip("/")
    output = args.output

    try:
        zarray = read_zarray(volume_path, anon=args.input_anon)
    except Exception as exc:
        print(f"ERROR: failed to read .zarray at {volume_path}: {exc}", file=sys.stderr)
        return 1

    chunks, shape = extract_chunks_shape(zarray)
    print(f"Input array: {volume_path} (shape={shape}, chunks={chunks})")
    print(f"Cache output: {output}")

    # Let build_chunk_occupancy handle everything — including the existing-and-up-to-date
    # short-circuit. Point the override URL at our output so the cache is read from and
    # written to `output` directly.
    os.environ[ENV_OVERRIDE_URL] = output

    occupancy = build_chunk_occupancy(
        volume_path,
        chunks=chunks,
        shape=shape,
        verbose=args.verbose,
        use_cache=True,
        anon=args.input_anon,
    )

    if occupancy is None:
        print("ERROR: build_chunk_occupancy returned None (no cache written).", file=sys.stderr)
        return 1

    # Verify the cache is actually present at `output`. build_chunk_occupancy may have
    # fallen back to sidecar/local if the override write failed, which defeats the point.
    try:
        fs, fs_path = _url_to_fs(output)
        if not fs.exists(fs_path):
            print(
                f"ERROR: expected chunk occupancy cache at {output} after build, "
                f"but it does not exist (write likely failed).",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"ERROR: could not verify cache at {output}: {exc}", file=sys.stderr)
        return 1

    occ = int(occupancy.sum())
    total = int(occupancy.size)
    print(f"Done: {occ}/{total} ({100.0 * occ / total:.1f}%) chunks occupied. Cache written to {output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
