"""Browse scrolls, direct relationships, and CT resolution candidates."""
import argparse
import sys

from ._fetch import DEFAULT_METADATA_URL
from .catalog import Catalog, DEFAULT_RESOLUTION_REL_TOL
from .browsing import ScrollSource
from .errors import VcReaderError


def _table(rows):
    rows = [[str(c) if c is not None else '-' for c in row] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vc-reader', description=__doc__)
    parser.add_argument('--metadata', default=DEFAULT_METADATA_URL)
    parser.add_argument('--refresh', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    samples = commands.add_parser('samples', help='list scrolls and fragments')
    samples.add_argument('--type', choices=('scroll', 'fragment'))
    scrolls = commands.add_parser('scrolls', help='list scrolls')
    for name in ('volumes', 'labels', 'segments', 'renders', 'representations'):
        p = commands.add_parser(name, help=f'list directly associated {name}')
        p.add_argument('scroll')
        p.add_argument('--volume')
        p.add_argument('--include-superseded', action='store_true')
        if name == 'labels':
            p.add_argument('--label-set')
        if name == 'volumes':
            p.add_argument('--resolution', type=float, help='native or pyramid resolution in micrometers')
            p.add_argument('--rel-tol', type=float, default=DEFAULT_RESOLUTION_REL_TOL)
            p.add_argument('--resolve', action='store_true', help='verify candidates against storage headers')
    args = parser.parse_args(argv)
    try:
        catalog = Catalog.load(args.metadata, refresh=args.refresh)
        if args.command in ('samples', 'scrolls'):
            values = catalog.scrolls if args.command == 'scrolls' else catalog.samples
            if getattr(args, 'type', None):
                values = values.filter(type=args.type)
            rows = [['sample', 'type', 'volumes', 'segments', 'labels']]
            rows.extend([s.id, s.type, len(s.volumes), len(s.segments), len(s.labels)] for s in values)
        else:
            scroll = ScrollSource(args.scroll, args.volume, catalog=catalog)
            values = getattr(scroll, args.command).filter(include_superseded=args.include_superseded)
            if args.command == 'volumes':
                if args.resolution is not None:
                    values = values.at_resolution(args.resolution, rel_tol=args.rel_tol)
                    if args.resolve:
                        values = [v.resolve() for v in values]
                    rows = [['scroll', 'volume', 'level', 'estimated um', 'verified']]
                    rows.extend([v.scroll.id, v.volume.id, v.level, v.estimated_resolution_um, v.verified] for v in values)
                else:
                    rows = [['volume', 'native um', 'labels', 'segments']]
                    rows.extend([v.id, v.pixel_size_um, len(v.labels), len(v.segments)] for v in values)
            elif args.command == 'segments':
                rows = [['segment', 'labels', 'renders']]
                rows.extend([s.id, len(s.labels), len(s.renders)] for s in values)
            else:
                if args.command == 'labels' and args.label_set:
                    values = values.filter(label_set=args.label_set)
                rows = [['id', 'kind', 'native volume', 'segment', 'label set', 'location']]
                rows.extend([a.id, a.artifact_kind, a.volume.id if a.volume else None,
                             a.segment.id if a.segment else None, a.label_set, a.source.storage.location] for a in values)
        print(_table(rows))
    except (VcReaderError, ValueError, OSError) as error:
        print(f'vc-reader: {error}', file=sys.stderr)
        return 1
    return 0
