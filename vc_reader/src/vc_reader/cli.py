"""``vc-reader`` command line: browse the open-data catalog.

    vc-reader samples [QUERY]
    vc-reader volumes SAMPLE [--resolution UM] [--energy KEV]
    vc-reader labels SAMPLE (--volume ID | --resolution UM [--energy KEV]) [--um UM] [--direct-only]
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._fetch import DEFAULT_METADATA_URL
from .catalog import Catalog, Sample
from .errors import VcReaderError


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:g}"


def _cmd_samples(cat: Catalog, args: argparse.Namespace) -> None:
    rows = [["sample", "type", "name", "volumes", "segments", "ink-label sets"]]
    for s in cat.find_samples(args.query):
        name = ", ".join(n for n in (s.display_name, *s.aliases) if n)
        rows.append([s.id, s.type or "-", name or "-", str(len(s.volumes)), str(len(s.segments)), str(len(s.ink_labels()))])
    print(_table(rows))


def _cmd_volumes(cat: Catalog, args: argparse.Namespace) -> None:
    sample = cat.sample(args.sample)
    vols = sample.find_volumes(resolution=args.resolution, energy_kev=args.energy, include_superseded=True)
    rows = [["volume", "um", "keV", "shape zyx", "license", "segments", "labels on", "notes"]]
    for v in vols:
        notes = []
        if v.is_superseded:
            notes.append(f"superseded by {v.superseded_by_id}")
        if v.render_segments is False:
            notes.append("not rendered")
        rows.append(
            [
                v.long_id or v.id,
                _fmt(v.pixel_size_um),
                _fmt(v.energy_kev),
                "x".join(map(str, v.shape_zyx)) if v.shape_zyx else "-",
                v.license.name if v.license else "-",
                str(len(v.segments())),
                str(len(sample.ink_labels(target_volume=v))),
                "; ".join(notes),
            ]
        )
    print(_table(rows))


def _pick_volume(sample: Sample, args: argparse.Namespace):
    if args.volume:
        return sample.volume(args.volume)
    return sample.volume(resolution=args.resolution, energy_kev=args.energy)


def _cmd_labels(cat: Catalog, args: argparse.Namespace) -> None:
    sample = cat.sample(args.sample)
    vol = _pick_volume(sample, args)
    lvl = vol.level(um=args.um) if args.um is not None else vol.level(0)
    print(f"{sample.id} / {vol.long_id}  level {lvl.level} ({_fmt(lvl.voxel_size_um)} um)  {vol.url()}")
    rows = [["segment", "label set", "relation", "labels on", "registration", "coverage", "url"]]
    for m in vol.ink_labels(mappable=not args.direct_only):
        rows.append(
            [
                m.segment.long_id or m.segment.id,
                m.labels.label_set or "-",
                m.relation,
                m.labels.target_volume_id,
                m.labels.registration or "-",
                f"{m.coverage.overlap_ratio:.3f}" if m.coverage else "-",
                m.labels.url() or "-",
            ]
        )
    print(_table(rows))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vc-reader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", default=DEFAULT_METADATA_URL, help="metadata.json URL or local path")
    p.add_argument("--refresh", action="store_true", help="revalidate the cached metadata.json now")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("samples", help="list samples")
    sp.add_argument("query", nargs="?")
    sp.set_defaults(func=_cmd_samples)

    sp = sub.add_parser("volumes", help="list a sample's volumes")
    sp.add_argument("sample")
    sp.add_argument("--resolution", type=float, help="level-0 voxel size in um")
    sp.add_argument("--energy", type=float, help="X-ray energy in keV")
    sp.set_defaults(func=_cmd_volumes)

    sp = sub.add_parser("labels", help="ink labels on, or mappable onto, a volume")
    sp.add_argument("sample")
    sp.add_argument("--volume", help="volume id or long id")
    sp.add_argument("--resolution", type=float, help="level-0 voxel size in um")
    sp.add_argument("--energy", type=float, help="X-ray energy in keV")
    sp.add_argument("--um", type=float, help="working scale in um (selects the pyramid level)")
    sp.add_argument("--direct-only", action="store_true", help="only labels published on this volume")
    sp.set_defaults(func=_cmd_labels)

    args = p.parse_args(argv)
    if args.command == "labels" and not (args.volume or args.resolution):
        p.error("labels: pass --volume or --resolution")
    try:
        cat = Catalog.load(args.metadata, refresh=args.refresh)
        args.func(cat, args)
    except VcReaderError as exc:
        print(f"vc-reader: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
