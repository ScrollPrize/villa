"""Stage-1 reproducibility (PLAN-S1-final section 5).

    s1_manifest.py freeze OUT_ROOT            copy spiral-fitting (tracked + untracked, no .venv/caches) and the
                                              experiment scripts into OUT_ROOT/<sha>/, record the venv; prints the dir
    s1_manifest.py inputs FROZEN_DIR PATH...  sha256 of every file under the given inputs (symlinks resolved),
                                              streamed with 16 processes -> FROZEN_DIR/inputs.sha256 (+ .json summary)
    s1_manifest.py run RUN_DIR FROZEN_DIR OVERRIDES [--parent CKPT] [--anchors FILE]
                                              write RUN_DIR/manifest.json
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

SPIRAL = '/home/djosey/Documents/villa/spiral-fitting'
WORK = '/media/djosey/nvme4/spiral_0332/work'
PY = f'{SPIRAL}/.venv/bin/python'
WORK_SCRIPTS = ['slice_eval.py', 'slice_eval_frozen.py', 'holdout_score.py', 's1/s1_score.py', 's1/s1_decide.py',
                's1/s1_manifest.py', 's1/chain_s1.sh', '../run_fit.sh']


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def tree_files(root):
    root = os.path.realpath(root)
    if os.path.isfile(root):
        return [root]
    out = []
    for d, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        out += [os.path.join(d, f) for f in sorted(files)]
    return out


def digest_lines(lines):
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def freeze(out_root):
    listed = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=SPIRAL, capture_output=True,
                            text=True, check=True).stdout.split()
    files = sorted(f for f in listed if not f.startswith(('.venv/', 'tests/__pycache__')) and '__pycache__' not in f
                   and os.path.isfile(os.path.join(SPIRAL, f)))
    lines = [f'{sha256_file(os.path.join(SPIRAL, f))}  spiral-fitting/{f}' for f in files]
    # vc3d_fiber_format is imported from the sibling vesuvius/src tree (vc3d_fiber_format_adapter.py)
    villa = os.path.dirname(SPIRAL)
    vfiles = sorted(f for f in subprocess.run(['git', 'ls-files', '-co', '--exclude-standard', 'vesuvius/src/vc3d_fiber_format'],
                                              cwd=villa, capture_output=True, text=True, check=True).stdout.split()
                    if '__pycache__' not in f and os.path.isfile(os.path.join(villa, f)))
    lines += [f'{sha256_file(os.path.join(villa, f))}  {f}' for f in vfiles]
    lines += [f'{sha256_file(os.path.join(WORK, f))}  work/{f}' for f in WORK_SCRIPTS]
    sha = digest_lines(lines)[:16]
    dest = os.path.join(out_root, f's1_{sha}')
    if not os.path.isdir(dest):
        tmp = dest + '.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(tmp, 'spiral-fitting', f)), exist_ok=True)
            shutil.copy2(os.path.join(SPIRAL, f), os.path.join(tmp, 'spiral-fitting', f))
        for f in vfiles:
            os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
            shutil.copy2(os.path.join(villa, f), os.path.join(tmp, f))
        for f in WORK_SCRIPTS:
            target = os.path.join(tmp, 'work', os.path.basename(f) if f.startswith('..') else f)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(WORK, f), target)
        with open(os.path.join(tmp, 'source.sha256'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        freeze_venv(tmp)
        os.rename(tmp, dest)
    print(dest)


def freeze_venv(dest):
    pip = subprocess.run([PY, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
    if not pip.strip():
        pip = subprocess.run(['uv', 'pip', 'freeze', '--python', PY], capture_output=True, text=True).stdout
    venv = os.path.dirname(os.path.dirname(PY))
    sos = sorted(f for f in tree_files(os.path.join(venv, 'lib')) if f.endswith('.so'))
    record = {'python': PY, 'python_sha256': sha256_file(os.path.realpath(PY)), 'pip_freeze': pip.splitlines(),
              'so_sha256': {f: sha256_file(f) for f in sos}}
    with open(os.path.join(dest, 'venv.json'), 'w') as fh:
        json.dump(record, fh, indent=1)


def inputs(frozen, paths):
    started = time.time()
    entries = []
    for p in paths:
        entries += [(p, f) for f in tree_files(p)]
    with ProcessPoolExecutor(16) as pool:
        hashes = list(pool.map(sha256_file, [f for _, f in entries], chunksize=64))
    lines = [f'{h}  {f}' for (_, f), h in zip(entries, hashes)]
    with open(os.path.join(frozen, 'inputs.sha256'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    per_input = {}
    for p in paths:
        per_input[p] = digest_lines([l for (q, _), l in zip(entries, lines) if q == p])
    summary = {'inputs': per_input, 'files': len(lines), 'bytes': sum(os.path.getsize(f) for _, f in entries),
               'all_sha256': digest_lines(lines), 'seconds': round(time.time() - started, 1)}
    with open(os.path.join(frozen, 'inputs.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs'}))


def run_manifest(run_dir, frozen, overrides, parent=None, anchors=None):
    doc = {
        'frozen_root': frozen,
        'source_sha256': sha256_file(os.path.join(frozen, 'source.sha256')),
        'venv_sha256': sha256_file(os.path.join(frozen, 'venv.json')),
        'inputs_sha256': json.load(open(os.path.join(frozen, 'inputs.json')))['all_sha256'],
        'overrides': json.load(open(overrides)), 'overrides_sha256': sha256_file(overrides),
        'parent_checkpoint_sha256': sha256_file(parent) if parent else None,
        'anchors_sha256': sha256_file(anchors) if anchors else None,
        'env': {k: os.environ.get(k) for k in ('SPIRAL_ROOT', 'FIT_DATASET', 'FIT_CACHE', 'WORK_ROOT')},
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as fh:
        json.dump(doc, fh, indent=1)
    print(os.path.join(run_dir, 'manifest.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('freeze'); a.add_argument('out_root')
    b = sub.add_parser('inputs'); b.add_argument('frozen'); b.add_argument('paths', nargs='+')
    c = sub.add_parser('run'); c.add_argument('run_dir'); c.add_argument('frozen'); c.add_argument('overrides')
    c.add_argument('--parent'); c.add_argument('--anchors')
    args = ap.parse_args()
    if args.cmd == 'freeze':
        freeze(args.out_root)
    elif args.cmd == 'inputs':
        inputs(args.frozen, args.paths)
    else:
        run_manifest(args.run_dir, args.frozen, args.overrides, args.parent, args.anchors)


if __name__ == '__main__':
    main()
