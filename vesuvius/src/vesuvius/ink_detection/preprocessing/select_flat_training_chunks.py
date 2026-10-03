"""Reduce a Hugging Face bucket sync plan to the surface-volume chunks a flat training run reads.

A flat-mode segment ships its whole surface volume (every pyramid level), but training at
``volume_scale`` 0 only ever reads the level-0 chunks under its labeled patches. This runs the
trainer's own patch discovery on the labels and keeps only those chunks, so a first training
run does not need the full download.

1. Sync the labels, the Zarr metadata and ``x.tif`` (segment discovery requires it):

       hf buckets sync hf://buckets/scrollprize/datasets/ink/phercparis4/<segment> ./ink-dataset/phercparis4/<segment> \\
         --include "*_inklabels.zarr/*" --include "*_supervision_mask.zarr/*" \\
         --include "*_validation_mask.zarr/*" --include "*.zarr/.z*" --include "*.zarr/0/.z*" \\
         --include "x.tif" --include "meta.json"

2. Write the full plan without downloading anything:

       hf buckets sync hf://buckets/.../<segment> ./ink-dataset/phercparis4/<segment> --plan full.jsonl

3. Keep what the config needs, then apply (run both from the directory the plan was written in,
   since the plan stores its destination relative to it):

       python -m vesuvius.ink_detection.preprocessing.select_flat_training_chunks \\
         configs/ink_tutorial.json full.jsonl subset.jsonl
       hf buckets sync --apply subset.jsonl
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Sequence

from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config
from vesuvius.ink_detection.volume_io import open_volume
from vesuvius.utils.cli import HyphenUnderscoreParser


# Everything except surface-volume chunks is small and always kept.
SMALL_PATTERNS = (
    "*_inklabels.zarr/*",
    "*_supervision_mask.zarr/*",
    "*_validation_mask.zarr/*",
    "*.zarr/.z*",
    "*.zarr/0/.z*",
    "x.tif",
    "meta.json",
)


def parse_args(argv: Sequence[str] | None = None):
    parser = HyphenUnderscoreParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="flat-mode training config")
    parser.add_argument("plan", type=Path, help="plan from `hf buckets sync --plan`")
    parser.add_argument("output", type=Path, help="reduced plan for `hf buckets sync --apply`")
    parser.add_argument(
        "--plan-root",
        type=Path,
        help="local directory the plan's destination is relative to (default: current directory)",
    )
    return parser.parse_args(argv)


def chunk_keys_for_patches(patches, volume) -> set[str]:
    """Level-0 chunk keys (as stored on disk) touched by the patches' bounding boxes."""
    cz, cy, cx = volume.chunks
    separator = getattr(volume.metadata, "dimension_separator", ".") or "."
    keys = set()
    for patch in patches:
        z0, y0, x0, z1, y1, x1 = patch.bbox
        for iz in range(z0 // cz, (z1 - 1) // cz + 1):
            for iy in range(y0 // cy, (y1 - 1) // cy + 1):
                for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                    keys.add(separator.join(map(str, (iz, iy, ix))))
    return keys


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = training_dataset_config(stage_training_request(args.config).config)
    if config.mode != "flat":
        raise SystemExit(f"only flat mode is supported, config has mode={config.mode!r}")

    lines = [json.loads(line) for line in args.plan.read_text().splitlines() if line.strip()]
    header, operations = lines[0], [op for op in lines[1:] if op.get("type") == "operation"]
    plan_root = ((args.plan_root or Path.cwd()) / Path(header["dest"]).expanduser()).resolve()

    dataset = InkDataset(config, do_augmentations=False)
    if not dataset.segments:
        raise SystemExit(
            "no segment discovered: check segments_path and that x.tif and the label zarrs are synced"
        )

    wanted_chunks: set[str] = set()
    patch_count = 0
    for segment in dataset.segments:
        if segment.scale != 0:
            raise SystemExit(f"{segment.segment_name}: only volume_scale 0 is supported")
        volume_dir = Path(segment.image_volume).resolve()
        try:
            prefix = volume_dir.relative_to(plan_root).as_posix()
        except ValueError:
            continue
        patches = [patch for patch in dataset.patches if patch.segment is segment]
        patch_count += len(patches)
        volume = open_volume(segment.image_volume, 0)
        prefix = "" if prefix == "." else prefix + "/"
        wanted_chunks |= {f"{prefix}0/{key}" for key in chunk_keys_for_patches(patches, volume)}
    if not wanted_chunks:
        raise SystemExit(f"no discovered segment lies under the plan destination {plan_root}")

    def keep(path: str) -> bool:
        return path in wanted_chunks or any(fnmatch.fnmatch(path, p) for p in SMALL_PATTERNS)

    selected = [op for op in operations if keep(op["path"])]
    downloads = [op for op in selected if op["action"] == "download"]
    header = dict(header)
    header["summary"] = {
        "uploads": 0,
        "downloads": len(downloads),
        "deletes": 0,
        "skips": len(selected) - len(downloads),
        "total_size": sum(op.get("size", 0) for op in downloads),
    }
    with args.output.open("w") as stream:
        for record in (header, *selected):
            stream.write(json.dumps(record) + "\n")

    def gb(ops):
        return sum(op.get("size", 0) for op in ops) / 1e9

    print(f"patches: {patch_count}, level-0 chunks: {len(wanted_chunks)}")
    print(f"full plan: {len(operations)} files, {gb(operations):.2f} GB")
    print(f"selected:  {len(selected)} files, {gb(selected):.2f} GB; to download {gb(downloads):.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
