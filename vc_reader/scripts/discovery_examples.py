"""Discover published segment labels and volume affine mappings without array I/O."""
import argparse
import json
from vc_reader import load, ScrollSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata')
    parser.add_argument('--sample', default='PHerc0009B')
    parser.add_argument('--volume', help='Filter native ownership, not mapped destinations')
    parser.add_argument('--segment', help='Omit to enumerate every segment with labels')
    args = parser.parse_args()
    catalog = load(args.metadata) if args.metadata else load()
    scroll = ScrollSource(args.sample, args.volume, catalog=catalog)
    segments = [scroll.segments[args.segment]] if args.segment else scroll.segments
    print(json.dumps([label.summary() for segment in segments for label in segment.labels], indent=2))


if __name__ == '__main__':
    main()
