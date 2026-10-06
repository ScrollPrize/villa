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
import subprocess
from urllib.parse import urlparse
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


def inherit(document, filename, originals, split, *, expected_split='train', coordinate=None):
    """Resolve a full parent; never infer orientation from piece geometry."""
    stored_coordinate = document.get('vc_open_data_coordinate_space')
    if coordinate and stored_coordinate and coordinate != stored_coordinate:
        raise ValueError(f'{filename}: piece coordinate space mismatch')
    coordinate = coordinate or stored_coordinate or ''
    scroll = coordinate.split('/')[0]
    if not re.fullmatch(r'PHerc[A-Za-z0-9_]+', scroll):
        raise ValueError(f'{filename}: missing/invalid coordinate-space scroll')
    folders = {'PHerc1299': ('PHerc1229',),
               'PHercParis4': ('PHercParis4_eval_fibers', 'PHercParis4_fibers')}.get(scroll, (scroll,))
    match = re.fullmatch(r'\d+_(.+)_run\d+\.json', filename)
    parent_name = match.group(1) + '.json' if match else filename
    candidates = [originals / folder / parent_name for folder in folders
                  if (originals / folder / parent_name).is_file()]
    if not candidates:
        raise FileNotFoundError(f'{filename}: full parent not found in {folders}')
    if len(candidates) != 1:
        raise ValueError(f'{filename}: ambiguous full parent: {candidates}')
    parent = candidates[0]
    folder = parent.parent.name
    raw_parent = parent.read_bytes()  # Missing parent is fatal, never local fallback.
    parent_doc = json.loads(raw_parent)
    parent_coordinate = parent_doc.get('vc_open_data_coordinate_space')
    if parent_coordinate != coordinate and (parent_coordinate or expected_split != 'test'):
        raise ValueError(f'{filename}: parent coordinate space mismatch: {parent}')
    classification = parent_doc.get('hv_classification') or {}
    label = classification.get('manual_tag') or classification.get('automatic_tag')
    if label not in ('H', 'V'):
        raise ValueError(f'{filename}: parent H/V unresolved: {parent}')
    membership = [part for part in ('train', 'test')
                  if (split / part / folder / parent_name).is_file()]
    # A future reviewed held-out parent must not silently leak into training.
    if expected_split not in ('train', 'test'):
        raise ValueError(f'Invalid expected split: {expected_split}')
    if membership != [expected_split]:
        raise ValueError(f'{filename}: parent is not exclusively in {expected_split}ing: {membership}')
    result = copy.deepcopy(document)
    result['vc_open_data_coordinate_space'] = coordinate
    result.setdefault('hv_classification', {})['manual_tag'] = label
    return scroll, result, dict(parent=str(parent), parent_sha256=sha(raw_parent),
                               parent_match='filename', parent_split=expected_split,
                               parent_hv_classification=classification,
                               original_hv_classification=document.get('hv_classification'),
                               inherited_hv=label)


def bake_test_set(source, originals, split, output, coordinates, baker):
    """Bake an existing fixed manifest selection, never create a new split."""
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing dataset: {output}')
    raw_manifest = (source / 'manifest.json').read_bytes()
    manifest = json.loads(raw_manifest)
    rows = manifest['files']
    if not rows or len(rows) != manifest['expected_count']:
        raise ValueError('Test manifest count mismatch')
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.baked-test-incomplete-', dir=output.parent) as temp:
        staging = Path(temp) / 'dataset'
        staging.mkdir()
        updated = []
        for row in rows:
            relative = Path(row['path'])
            if (row['split'] != 'test' or relative.is_absolute() or '..' in relative.parts
                    or relative.parts[0] != row['source'] or str(relative) in seen):
                raise ValueError(f'Invalid/duplicate test manifest entry: {row}')
            seen.add(str(relative))
            raw = (source / 'test' / relative).read_bytes()
            if sha(raw) != row['sha256']:
                raise ValueError(f'Test source hash mismatch: {relative}')
            doc = json.loads(raw)
            coordinate = coordinates[row['source']]
            if coordinate.split('/')[0] != row['source']:
                raise ValueError(f'Test coordinate mapping mismatch: {relative}')
            _, result, provenance = inherit(doc, relative.name, originals, split,
                expected_split='test', coordinate=coordinate)
            result = baker.bake(result)
            baked = result['baked_annotations']
            if not (len(baked['normals_xyz']) == len(baked['valid']) == len(doc['line_points'])):
                raise ValueError(f'Baked normal count mismatch: {relative}')
            # Only labels and explicit coordinate identity may change.
            allowed = {'baked_annotations', 'hv_classification', 'vc_open_data_coordinate_space'}
            if {k:v for k,v in doc.items() if k not in allowed} != {
                    k:v for k,v in result.items() if k not in allowed}:
                raise ValueError(f'Baking changed test geometry/metadata: {relative}')
            payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
            dest = staging / 'test' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            updated.append(dict(row, sha256=sha(payload), original_piece_sha256=sha(raw),
                original_piece_path=str(source / 'test' / relative),
                coordinate_space=coordinate, baked_valid_points=sum(baked['valid']),
                baked_total_points=len(baked['valid']), **provenance))
            print(f'Baked test {len(updated)}/{len(rows)}: {relative}', flush=True)
        manifest.update(files=updated, source_manifest=str(source / 'manifest.json'),
            source_manifest_sha256=sha(raw_manifest),
            hv_policy='Full held-out parent effective H/V; never piece geometry')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (staging / 'SHA256SUMS').write_text(''.join(
            f"{r['sha256']}  test/{r['path']}\n" for r in updated))
        if output.exists():
            raise ValueError(f'Output appeared during baking: {output}')
        staging.rename(output)
    print(f'Baked test dataset: {output}', flush=True)
    return manifest


class NormalBaker:
    """Native JSON-lines baker; mappings relocate exact manifest basenames.

    Metadata retains all original paths and the resolved manifest bytes/hash.
    No source-field interpolation is implemented here.
    """
    def __init__(self, binary, manifests, cache_root=""):
        self.manifests, self.cache_root = manifests, cache_root
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def bake(self, document):
        if float(document.get('width', 0)) <= 0:
            return document
        sources = sorted({cp.get('segment_to_next', {}).get('normal_manifest', '')
                          for cp in document['control_points']} - {''})
        names = {Path(urlparse(p).path).name for p in sources}
        if len(names) != 1:
            raise ValueError(f'Normal source missing or ambiguous: {sources}')
        name = next(iter(names))
        if name not in self.manifests:
            raise ValueError(f'No explicit normal manifest mapping for {name}')
        manifest = Path(self.manifests[name]).resolve()
        manifest_raw = manifest.read_bytes()
        value = json.loads(manifest_raw)
        # Identity verification includes the sample and source volume ID, not
        # only a filename shared across arbitrary directories.
        coordinate = document['vc_open_data_coordinate_space'].split('/')
        locations = [g['zarr'] for g in value['groups'].values()]
        if not all(coordinate[0] in p and coordinate[1].split('@')[0].split('-')[0] in p for p in locations):
            raise ValueError(f'Normal manifest does not match fiber coordinates: {manifest}')
        request = dict(fiber=document, manifest=str(manifest), cache_root=self.cache_root, threads=1)
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f'Native normal baking failed for {name}; see native error above')
        baked = json.loads(response)
        baked.update(source_manifests=sources, resolved_manifest=value,
                     manifest_sha256=sha(manifest_raw))
        document['baked_annotations'] = baked
        return document


def collect(source, originals, split, output, workers=8, baker=None):
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
                    if baker is not None:
                        result = baker.bake(result)
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
                                     baked_valid_points=sum(result.get('baked_annotations', {}).get('valid', [])),
                                     baked_total_points=len(result.get('baked_annotations', {}).get('normals_xyz', [])),
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
    parser.add_argument('--bake-config', type=Path,
                        default=Path(__file__).with_name('reviewed_normal_baking.json'))
    parser.add_argument('--no-bake', action='store_true', help='Explicitly collect labels without baked normals')
    parser.add_argument('--test-source', type=Path,
                        help='Bake an existing fixed test manifest instead of downloading training data')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.no_bake:
        if args.test_source:
            parser.error('--test-source requires normal baking')
        collect(args.source, args.originals, args.split, args.output, args.workers)
    else:
        settings = json.loads(args.bake_config.read_text())
        root = args.bake_config.resolve().parent
        with NormalBaker(root / settings['binary'],
                         {k: root / v for k, v in settings['manifests'].items()},
                         settings.get('cache_root', '')) as baker:
            if args.test_source:
                bake_test_set(args.test_source, args.originals, args.split, args.output,
                              settings['test_coordinates'], baker)
            else:
                collect(args.source, args.originals, args.split, args.output, args.workers, baker)


if __name__ == '__main__':
    main()
