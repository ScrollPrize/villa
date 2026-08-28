#!/usr/bin/env python3
"""Expand the aligned-21 sampling contract into a runnable training config.

``configs/aligned21_hybrid_3d2d.json`` ships one ``datasets`` entry with
``/path/to/...`` placeholders, and the 29 representations it is meant to train
on live separately in ``configs/aligned21_fixed_scroll_prior.json``. Joining the
two by hand means writing 29 entries, each with its own label directory and
surface-volume path, and keeping ``fixed_scroll_prior.target_batch_counts``
consistent with whatever subset you kept.

This script does that join by locating each representation under the roots you
give it. Holding scrolls or segments out turns the same recipe into a
generalisation probe: nothing on the held-out scroll is ever sampled, so its
whole supervision mask stays honest held-out ground truth. Per-scroll batch
quotas are renormalised over the survivors, because
``FixedScrollPriorStratifiedBatchSampler`` rejects a batch whose quotas do not sum
to ``batch_size``, rejects a non-positive quota, and rejects quota keys that do not
exactly match the scrolls the patches came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_RECIPE = CONFIG_DIR / "aligned21_hybrid_3d2d.json"
DEFAULT_CONTRACT = CONFIG_DIR / "aligned21_fixed_scroll_prior.json"


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def walk_dirs(root: Path):
    """Yield directories under root without descending into Zarr stores."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = [entry for entry in current.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            yield entry
            if entry.suffix != ".zarr":
                stack.append(entry)


def find_label_dir(labels_root: Path, segment: str) -> Path | None:
    """A segment's label directory is named after it and holds its inklabels."""
    for candidate in walk_dirs(labels_root):
        if candidate.name == segment and (candidate / f"{segment}_inklabels.zarr").exists():
            return candidate
    return None


def find_volume(volumes_root: Path, segment: str) -> Path | None:
    """Accept either <segment>.zarr or <segment>/<one>.zarr."""
    holder = None
    for candidate in walk_dirs(volumes_root):
        if candidate.suffix == ".zarr" and candidate.stem == segment:
            return candidate
        if candidate.name == segment and candidate.suffix != ".zarr":
            holder = candidate
    if holder is None:
        return None
    stores = sorted(store for store in holder.iterdir() if store.suffix == ".zarr")
    if len(stores) != 1:
        sys.exit(f"error: expected exactly one *.zarr under {holder}, found {len(stores)}")
    return stores[0]


def renormalise(quotas: dict, keep: set, batch_size: int) -> dict:
    """Spread the recipe's quotas over the surviving scrolls, summing to batch_size."""
    live = {scroll: value for scroll, value in quotas.items() if scroll in keep}
    if not live:
        sys.exit("error: every scroll was excluded")
    if batch_size < len(live):
        # Every survivor is floored to one slot below, because the sampler rejects a
        # zero quota outright, so a smaller batch than the scroll count has no answer.
        sys.exit(f"error: batch_size {batch_size} cannot cover {len(live)} scrolls; "
                 "FixedScrollPriorStratifiedBatchSampler needs at least one slot per "
                 "scroll")
    total = sum(live.values())
    exact = {scroll: batch_size * value / total for scroll, value in live.items()}
    out = {scroll: max(1, int(value)) for scroll, value in exact.items()}
    # Largest-remainder top-up so the quotas land exactly on batch_size.
    order = sorted(live, key=lambda s: exact[s] - int(exact[s]), reverse=True)
    index = 0
    while sum(out.values()) < batch_size:
        out[order[index % len(order)]] += 1
        index += 1
    while sum(out.values()) > batch_size:
        victim = max((s for s in order if out[s] > 1), key=lambda s: out[s])
        out[victim] -= 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-root", type=Path, required=True,
                        help="Directory the per-segment label folders live under.")
    parser.add_argument("--volumes-root", type=Path, required=True,
                        help="Directory the ~9 um surface volumes live under.")
    parser.add_argument("--out", type=Path, required=True, help="Config path to write.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training out_dir.")
    parser.add_argument("--exclude-scroll", action="append", default=[], metavar="SCROLL",
                        help="Hold out every representation of this scroll. Repeatable.")
    parser.add_argument("--exclude-segment", action="append", default=[], metavar="SEGMENT",
                        help="Hold out one representation by segment name. Repeatable.")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help=f"Base training config. Default: {DEFAULT_RECIPE.name}")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT,
                        help=f"Sampling contract. Default: {DEFAULT_CONTRACT.name}")
    parser.add_argument("--seed", type=int, default=None, help="Override seed and sampler seed.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Write the config even if some inputs are not prepared yet.")
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    contract = json.loads(args.contract.read_text())

    excluded_scrolls = set(args.exclude_scroll)
    excluded_segments = set(args.exclude_segment)
    known_scrolls = {rep["scroll"] for rep in contract["representations"]}
    unknown = excluded_scrolls - known_scrolls
    if unknown:
        sys.exit(f"error: unknown scroll(s) {sorted(unknown)}; known: {sorted(known_scrolls)}")
    known_segments = {rep["segment"] for rep in contract["representations"]}
    unknown = excluded_segments - known_segments
    if unknown:
        sys.exit(f"error: unknown segment(s) {sorted(unknown)}")

    kept, dropped = [], []
    for rep in contract["representations"]:
        if rep["scroll"] in excluded_scrolls or rep["segment"] in excluded_segments:
            dropped.append(rep)
        else:
            kept.append(rep)
    if not kept:
        sys.exit("error: the exclusions removed every representation")

    groups = OrderedDict()
    missing = []
    for rep in kept:
        segment = rep["segment"]
        labels = find_label_dir(args.labels_root, segment)
        volume = find_volume(args.volumes_root, segment)
        if labels is None:
            missing.append(f"labels for {segment} under {args.labels_root}")
            continue
        if volume is None:
            missing.append(f"surface volume for {segment} under {args.volumes_root}")
            continue
        entry = groups.setdefault((labels.parent, rep["scroll"]), {
            "segments_path": as_posix(labels.parent),
            "segments": [],
            "surface_volume_paths": {},
            "volume_scale": 0,
            "sampling_scroll": rep["scroll"],
            "sampling_physical_segment_keys": {},
            "sampling_representation_keys": {},
        })
        entry["segments"].append(segment)
        entry["surface_volume_paths"][segment] = as_posix(volume)
        entry["sampling_physical_segment_keys"][segment] = rep["physical_segment_key"]
        entry["sampling_representation_keys"][segment] = rep["representation_key"]

    if missing and not args.allow_missing:
        sys.exit("error: could not locate:\n  " + "\n  ".join(sorted(set(missing))))

    batch_size = args.batch_size or int(recipe["batch_size"])
    seed = args.seed if args.seed is not None else int(recipe["seed"])
    surviving = {entry["sampling_scroll"] for entry in groups.values()}
    if not surviving:
        # Only reachable under --allow-missing: without it the locate failure above exits.
        sys.exit(f"error: none of the {len(kept)} representations could be located under "
                 f"{args.labels_root} and {args.volumes_root}")
    quotas = renormalise(contract["target_batch_counts"], surviving, batch_size)
    held_out = sorted({rep["segment"] for rep in dropped})

    config = dict(recipe)
    config["batch_size"] = batch_size
    config["seed"] = seed
    config["fixed_scroll_prior"] = {"seed": seed, "target_batch_counts": quotas}
    config["out_dir"] = as_posix(args.run_dir)
    config["datasets"] = list(groups.values())
    for key, value in (("num_iterations", args.iterations),
                       ("save_every", args.save_every),
                       ("val_every", args.val_every)):
        if value is not None:
            config[key] = value
    arm = f"held out {sorted(excluded_scrolls) or held_out}" if dropped else "no held-out representations"
    kept_count = sum(len(entry["segments"]) for entry in groups.values())
    config["description"] = (f"{recipe['description'].split('.')[0]}. Arm: {arm}; "
                             f"{kept_count} representations, quotas {quotas}.")
    config["held_out_representations"] = held_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  representations : {kept_count} kept, {len(dropped)} held out")
    if dropped:
        print(f"  held out        : {', '.join(held_out)}")
    print(f"  quotas          : {quotas} (batch {batch_size})")
    print(f"  dataset entries : {len(groups)}")
    if missing:
        print(f"  WARNING         : {len(set(missing))} input(s) not located")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
