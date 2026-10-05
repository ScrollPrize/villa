#!/usr/bin/env python3
"""Exercise the Volume-backed remote cache with two real volumes.

The renderer opens --remote-url through Volume::NewFromUrl, so fetched chunks
persist under the shared remote cache root (VC3D settings), keyed by the
URL-derived volume id. This harness isolates that root per run through
VC3D_CONFIG_DIR, so it never touches the user's real cache.

Example (use a crop containing data in both volumes):
  python3 scripts/test_render_volume_remote_cache.py --renderer build/bin/vc_render_tifxyz \
    --source-a URL_A --source-b URL_B --work-dir /path/to/results -- \
    -s /path/to/tifxyz --scale 1 -g 3 -n 1 --crop-x 400 --crop-y 400 \
    --crop-width 128 --crop-height 128 --cache-gb 1

Each render gets a new output directory so the existing-output skip cannot pass
the checks. Work directories, logs and TIFFs are retained for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--source-a', required=True)
    parser.add_argument('--source-b', default=None)
    parser.add_argument('--source-b-group', type=int, default=None,
                        help='group override for source-b renders when its pyramid differs')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('render_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    renderer = args.renderer.resolve()
    extra = args.render_args[1:] if args.render_args[:1] == ['--'] else args.render_args
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='volume-remote-cache-', dir=args.work_dir.resolve()))
    print(f'Results: {work}', flush=True)

    def make_root(name):
        config = work / name / 'config'
        root = work / name / 'cache-root'
        config.mkdir(parents=True)
        (config / 'VC3D.ini').write_text(f'[viewer]\nremote_cache_dir = {root}\n')
        return config, root

    shared_config, shared_root = make_root('shared')
    fresh_config, fresh_root = make_root('fresh-b')

    def render(name, source=None, *, group=None, config=shared_config):
        out = work / name
        command = [str(renderer), '-v', str(out), '--tif-output', name, *extra]
        if group is not None:
            group_option = '-g' if '-g' in command else '--group-idx'
            command[command.index(group_option) + 1] = str(group)
        if source is not None:
            command += ['--remote-url', source]
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=300,
                                env={**os.environ, 'VC3D_CONFIG_DIR': str(config)})
        (work / f'{name}.log').write_text(result.stdout)
        (work / f'{name}.command.json').write_text(json.dumps(command, indent=2))
        tiffs = sorted(out.glob('*.tif'))
        assert result.returncode == 0, f'{name}: see {work / (name + ".log")}'
        assert tiffs, f'{name}: no output TIFF'
        digest = hashlib.sha256(b''.join(path.read_bytes() for path in tiffs)).hexdigest()
        print(f'{name}: {digest}', flush=True)
        return digest

    def payloads(root):
        # Track every cached byte: zarr-mirror objects and decoded level_N
        # copies alike. Warm renders must not change either store.
        return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*')
                if p.is_file() and not p.name.startswith('.') and '.tmp.' not in p.name}

    def volume_dirs(root):
        return sorted(p for p in root.iterdir() if p.is_dir())

    a = render('a-cold', args.source_a)
    first_payloads = payloads(shared_root)
    assert first_payloads, 'cold render persisted no chunks'
    after_a = volume_dirs(shared_root)
    assert len(after_a) == 1, f'expected one cached volume id, got {after_a}'
    cache_a = after_a[0]

    # Same source, different pyramid level: new objects land beside the old
    # ones under the same volume id, and the level-2 objects stay untouched.
    b_group = args.source_b_group if args.source_b_group is not None else 3
    b = render('b-other-level', args.source_a, group=b_group)
    assert b != a, 'choose a group whose pixels differ from the first render'
    combined = payloads(shared_root)
    if not combined.keys() > first_payloads.keys():
        missing = sorted(first_payloads.keys() - combined.keys())
        changed = sorted(k for k in first_payloads.keys() & combined.keys()
                         if combined[k] != first_payloads[k])
        raise AssertionError(
            f'second render disturbed first-render objects; '
            f'{len(missing)} vanished (first: {missing[:3]}), '
            f'{len(changed)} changed (first: {changed[:3]})')
    assert len(volume_dirs(shared_root)) == 1, 'same url must reuse one volume id'
    assert render('a-warm', args.source_a) == a
    assert payloads(shared_root) == combined, 'warm render changed cached payloads'
    assert render('a-after-b', args.source_a) == a
    assert render('b-fresh', args.source_a, group=b_group, config=fresh_config) == b
    assert len(volume_dirs(fresh_root)) == 1, 'fresh root cached more than one volume'

    # A different source URL gets its own volume id and cannot disturb the
    # first source's objects, whatever data it does or does not contain.
    if args.source_b:
        before_b = payloads(shared_root)
        render('b-other-source', args.source_b)
        after_b = payloads(shared_root)
        assert all(after_b.get(k) == v for k, v in before_b.items()),             'second source disturbed first source objects'
        ids = volume_dirs(shared_root)
        assert len(ids) == 2 and cache_a in ids,             f'expected the second source to add one volume id, got {ids}'

    summary = {'source_a_sha256': a, 'other_level_sha256': b,
               'source_a_payloads': len(first_payloads), 'both_payloads': len(combined)}
    (work / 'result.json').write_text(json.dumps(summary, indent=2))
    print('PASS: cold/warm equality, multi-level persistence, per-source isolation '
          'under the shared root, fresh-root equality', flush=True)


if __name__ == '__main__':
    main()
