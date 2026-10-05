"""Download reviewed annotations and inherit full-parent H/V labels (stdlib only)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

SOURCE = 'https://dl.ash2txt.org/other/annotators/'
COLLECTION = Path('/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection')
OUTPUT_ROOT = Path('/ephemeral/hendrik/fiber3d_train')


class Listing(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag == 'a' and re.fullmatch(r'[A-Za-z0-9_.-]+(?:/)?', href):
            if href not in ('.', '..', './', '../'):
                self.links.add(href)


def fetch(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def links(data):
    parser = Listing()
    parser.feed(data.decode('utf-8'))
    return sorted(parser.links)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inherit(document, filename, originals, split):
    """Resolve a full parent; never infer orientation from piece geometry."""
    coordinate = document.get('vc_open_data_coordinate_space', '')
    scroll = coordinate.split('/')[0]
    if not re.fullmatch(r'PHerc[A-Za-z0-9_]+', scroll):
        raise ValueError(f'{filename}: missing/invalid coordinate-space scroll')
    folder = 'PHerc1229' if scroll == 'PHerc1299' else scroll
    match = re.fullmatch(r'\d+_(.+)_run\d+\.json', filename)
    parent_name = match.group(1) + '.json' if match else filename
    parent = originals / folder / parent_name
    raw_parent = parent.read_bytes()  # Missing parent is fatal, never local fallback.
    parent_doc = json.loads(raw_parent)
    if parent_doc.get('vc_open_data_coordinate_space') != coordinate:
        raise ValueError(f'{filename}: parent coordinate space mismatch: {parent}')
    classification = parent_doc.get('hv_classification') or {}
    label = classification.get('manual_tag') or classification.get('automatic_tag')
    if label not in ('H', 'V'):
        raise ValueError(f'{filename}: parent H/V unresolved: {parent}')
    membership = [part for part in ('train', 'test')
                  if (split / part / folder / parent_name).is_file()]
    # A future reviewed held-out parent must not silently leak into training.
    if membership != ['train']:
        raise ValueError(f'{filename}: parent is not exclusively in training: {membership}')
    result = copy.deepcopy(document)
    result.setdefault('hv_classification', {})['manual_tag'] = label
    return scroll, result, dict(parent=str(parent), parent_sha256=sha(raw_parent),
                               parent_match='filename', parent_split='train',
                               parent_hv_classification=classification,
                               original_hv_classification=document.get('hv_classification'),
                               inherited_hv=label)


def collect(source, originals, split, output, workers=8):
    source = source.rstrip('/') + '/'
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing dataset: {output}')
    root_listing = fetch(source)
    directories = [d for d in links(root_listing) if d.endswith('/')]
    if not directories:
        raise ValueError('No annotation directories found')
    jobs, listings = [], {'index.html': root_listing}
    for directory in directories:
        listing = fetch(source + directory)
        listings[directory + 'index.html'] = listing
        jobs.extend((directory + name) for name in links(listing) if name.endswith('.json'))
    if not jobs:
        raise ValueError('No JSON annotations found')
    print(f'Downloading {len(jobs)} JSONs from {len(directories)} collections', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reviewed-incomplete-', dir=output.parent) as temp:
        staging = Path(temp) / 'dataset'
        staging.mkdir()
        for relative, data in listings.items():
            dest = staging / 'source_listings' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        rows, inventory, counts, seen = [], [], {}, set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (relative, raw) in enumerate(zip(jobs, pool.map(fetch, (source+j for j in jobs))), 1):
                doc = json.loads(raw)
                tags = doc.get('tags', [])
                if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
                    raise ValueError(f'{relative}: invalid tags')
                inventory.append(dict(path=relative, sha256=sha(raw), tags=tags))
                if 'reviewed' in tags:
                    if doc.get('type') != 'vc3d_fiber':
                        raise ValueError(f'{relative}: reviewed document is not a fiber')
                    scroll, result, provenance = inherit(doc, Path(relative).name, originals, split)
                    target = f'{scroll}/{Path(relative).name}'
                    if target in seen:
                        raise ValueError(f'Duplicate output filename: {target}')
                    seen.add(target)
                    raw_dest = staging / 'raw_reviewed' / relative
                    raw_dest.parent.mkdir(parents=True, exist_ok=True)
                    raw_dest.write_bytes(raw)
                    dest = staging / 'train' / target
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    payload = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
                    dest.write_bytes(payload)
                    rows.append(dict(path='train/'+target, source_url=source+relative,
                                     source_sha256=sha(raw), output_sha256=sha(payload),
                                     tags=tags, **provenance))
                    counts[scroll] = counts.get(scroll, 0) + 1
                if index % 100 == 0 or index == len(jobs):
                    print(f'Checked {index}/{len(jobs)}; selected {len(rows)} reviewed', flush=True)
        if not rows:
            raise ValueError('No reviewed fibers found; refusing empty training set')
        manifest = dict(source=source, created_utc=datetime.now(timezone.utc).isoformat(),
                        counts=counts, reviewed=len(rows),
                        reviewed_with_rework=sum('rework' in r['tags'] for r in rows),
                        selection='All reviewed tags, including rework/tricky; train parents only',
                        hv_policy='Parent effective H/V stored as piece manual_tag; no geometric estimation',
                        files=rows, remote_inventory=inventory)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (staging / 'SHA256SUMS').write_text(''.join(f"{r['output_sha256']}  {r['path']}\n" for r in rows))
        # Publish only after all downloads, provenance checks and writes succeed.
        if output.exists():
            raise ValueError(f'Output appeared during collection: {output}')
        staging.rename(output)
    print(json.dumps(dict(output=str(output), counts=counts, reviewed=len(rows),
                          reviewed_with_rework=manifest['reviewed_with_rework']), indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--originals', type=Path, default=COLLECTION/'all')
    parser.add_argument('--split', type=Path, default=COLLECTION/'split_95_5_length_seed20260910')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / datetime.now(timezone.utc).strftime('%Y-%m-%d_reviewed_annotators_%H%M%S'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    collect(args.source, args.originals, args.split, args.output, args.workers)


if __name__ == '__main__':
    main()
