#!/usr/bin/env python3
"""Exercise remote staging with two real volumes on the same chunk grid.

Example (use a crop containing data in both volumes):
  python3 scripts/test_render_remote_cache.py --renderer build/bin/vc_render_tifxyz \
    --source-a URL_A --source-b URL_B --work-dir /path/to/results -- \
    -s /path/to/tifxyz --scale 1 -g 3 -n 1 --crop-x 400 --crop-y 400 \
    --crop-width 128 --crop-height 128 --cache-gb 1

Each render gets a new output directory so the existing-output skip cannot pass
the checks. Work directories, logs and TIFFs are retained for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--source-a', required=True)
    parser.add_argument('--source-b', required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('render_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    renderer = args.renderer.resolve()
    extra = args.render_args[1:] if args.render_args[:1] == ['--'] else args.render_args
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='remote-cache-', dir=args.work_dir.resolve()))
    print(f'Results: {work}', flush=True)

    def render(name, cache, source=None, *, reject=False, group=None):
        command = [str(renderer), '-v', str(cache), '--tif-output', name,
                   *extra]
        if group is not None:
            group_option = '-g' if '-g' in command else '--group-idx'
            command[command.index(group_option) + 1] = str(group)
        if source is not None:
            command += ['--remote-url', source]
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=300)
        (work / f'{name}.log').write_text(result.stdout)
        (work / f'{name}.command.json').write_text(json.dumps(command, indent=2))
        tiffs = sorted((work / name).glob('*.tif'))
        if reject:
            assert result.returncode != 0, f'{name}: partial cache accepted as local'
            assert '--remote-url' in result.stdout, f'{name}: missing diagnostic'
            assert not tiffs, f'{name}: wrote TIFF despite missing source'
            print(f'PASS {name}: rejected partial cache', flush=True)
            return None
        assert result.returncode == 0, f'{name}: see {work / (name + ".log")}'
        assert tiffs, f'{name}: no output TIFF'
        digest = hashlib.sha256(b''.join(path.read_bytes() for path in tiffs)).hexdigest()
        print(f'{name}: {digest}', flush=True)
        return digest

    def payloads():
        return {p.relative_to(work / 'staged'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (work / 'staged').rglob('*.bin')}

    a = render('a-cold', 'staged', args.source_a)
    first_payloads = payloads()
    assert first_payloads, 'bare relative -v did not persist any chunks'
    assert (work / 'staged' / 'remote_sources').is_dir(), 'cache is not source-scoped'
    assert not (work / 'staged' / '.zgroup').exists(), 'cache root looks like a local Zarr'
    assert render('a-warm', './staged', args.source_a) == a
    assert payloads() == first_payloads, 'warm render changed cached payloads'
    b = render('b-shared', 'staged', args.source_b)
    assert b != a, 'choose a crop where the two real volumes have different pixels'
    combined = payloads()
    assert combined.keys() > first_payloads.keys(), 'second source reused first source paths'
    assert all(combined[p] == digest for p, digest in first_payloads.items())
    assert render('b-fresh', work / 'fresh-b', args.source_b) == b
    assert render('a-after-b', work / 'staged', args.source_a) == a
    render('no-source', 'staged', reject=True)

    # The former layout contains level_N directories and published metadata.
    # Check a populated source subtree too.
    mirror = next(p for p in (work / 'staged').rglob('level_*') if p.is_dir()).parent
    render('mirror-no-source', mirror, reject=True)
    # Recreate the old unscoped layout with real cached chunks. Missing levels
    # must not be treated as zero-filled local data, even outside remote_sources.
    legacy = work / 'legacy'
    shutil.copytree(mirror, legacy)
    render('legacy-no-source', legacy, reject=True)
    render('legacy-unfetched-group', legacy, reject=True, group=2)
    render('legacy-unfetched-level-directory', legacy / '2', reject=True, group=0)
    assert render('legacy-explicit-b', legacy, args.source_b) == b
    summary = {'source_a_sha256': a, 'source_b_sha256': b,
               'source_a_payloads': len(first_payloads), 'both_payloads': len(combined)}
    (work / 'result.json').write_text(json.dumps(summary, indent=2))
    print('PASS: relative/absolute staging, source isolation, rerender equality, '
          'and missing-source rejection', flush=True)


if __name__ == '__main__':
    main()
