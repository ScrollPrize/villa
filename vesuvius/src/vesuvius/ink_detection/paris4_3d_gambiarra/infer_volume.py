"""Full-volume (or boxed) inference of ink_paris4_3d_4.8um_gambiarra: uint8 ink codes on the level-1 (4.8 um) grid.

    P=vesuvius.ink_detection.paris4_3d_gambiarra
    python -m $P.infer_volume --weights model.safetensors --out runs/paris4-ink                         # whole scroll
    python -m $P.infer_volume --weights model.safetensors --out runs/box --box 32381 7712 8064 32509 7840 8192 --unit 256

Writes <out>/code.zarr/0 (zarr v2, uint8, 256^3 chunks, blosc zstd-1 bitshuffle; --compressor none for raw chunks):
    0 = no prediction (absent CT chunk, or outside crop coverage), else clip(floor(10 * logit) + 128, 1, 255)
    (code c in 2..254 <=> logit in [(c - 128) / 10, (c - 127) / 10); see volume.py for the geometry).

The output is split into units (tiles of --unit output voxels, whole chunks). Run one process per GPU with
--shard i/n (each takes every n-th unit, no two processes on the same unit); finished units are recorded in
<out>/state/done. A rerun skips a unit only if its marker matches the unit's plan (tile, CT presence, chunk set) and
the tile on disk holds exactly the recorded chunk files (size and sha256: every recorded chunk is read again). Other
markers, including those of earlier versions without a presence record, are removed; the tile's unplanned chunk files
and temporary files are deleted and the unit is recomputed. The first process to start publishes RUN.json (and, with
'list', presence.npy); a process with other settings is refused.
--box restricts the run to the units overlapping a box (output voxels, z0 y0 x0 z1 y1 x1, end exclusive).

Chunk presence: 'list' lists the store once (S3 ListObjectsV2 or a directory walk; cached in <out>/presence.npy);
'probe' asks per unit window (one HEAD per chunk; for small boxes). Default: probe with --box, otherwise list.

The released volume was computed with --unit 2048 --batch 16 on H100 GPUs. A crop's logits can depend on its batch
slot, GPU and numerical kernels.
"""
import argparse
import hashlib
import io
import itertools
import json
import time
from pathlib import Path

import numpy as np

from vesuvius.ink_detection.paris4_3d_gambiarra import runtime as rt  # sets CUBLAS_WORKSPACE_CONFIG before torch initializes CUDA
import torch

from vesuvius.ink_detection.paris4_3d_gambiarra import volume as V
from vesuvius.ink_detection.paris4_3d_gambiarra.ct import ChunkStore, PARIS4_LEVEL0
from vesuvius.ink_detection.paris4_3d_gambiarra.student import Student, load_weights


def units_of(shape1, unit, box=None):
    grid = [range(-(-s // unit)) for s in shape1]
    out = []
    for idx in itertools.product(*grid):
        lo = np.array(idx) * unit
        hi = np.minimum(lo + unit, shape1)
        if box is None or np.all((lo < box[3:]) & (hi > box[:3])):
            out.append(idx)
    return out


def presence_sha256(plan):
    """sha256 of the CT chunk presence the unit's crops and mask depend on (None outside crop coverage)."""
    if 'presence' not in plan:
        return None
    window = json.dumps([plan['c0'], plan['c1']]).encode() + np.ascontiguousarray(plan['presence']).tobytes()
    return hashlib.sha256(window).hexdigest()


def marker_problem(marker, plan, writer):
    """Why done marker `marker` does not certify its unit (None if it does): it must describe this tile with the
    current CT presence, its state and chunk set must equal the plan's, and the tile on disk must hold exactly the
    recorded chunk files."""
    try:
        record = json.loads(marker.read_text())
    except (OSError, ValueError) as e:
        return f'unreadable marker ({e})'
    if not isinstance(record, dict) or record.get('state') not in ('done', 'empty'):
        return 'malformed marker'
    if record.get('T0') != plan['T0'] or record.get('T1') != plan['T1']:
        return 'the marker describes another unit'
    if 'presence_sha256' not in record:
        return 'the marker records no CT presence (written by an earlier version)'
    if record['presence_sha256'] != presence_sha256(plan):
        return 'the CT presence of the unit changed'
    if (record['state'] == 'empty') != ('empty' in plan):
        return f'marked {record["state"]}, but the plan is {"empty" if "empty" in plan else "not empty"}'
    files = record.get('files', {}) if record['state'] == 'done' else {}
    if not isinstance(files, dict) or not all(isinstance(v, dict) and {'bytes', 'sha256'} <= set(v) for v in files.values()):
        return 'malformed marker'
    if set(files) != {' '.join(map(str, k)) for k in plan['chunks']}:
        return 'the recorded chunks differ from the plan'
    problems = writer.verify(plan['T0'], plan['T1'], files)
    if problems:
        return '; '.join(problems[:3]) + (f'; {len(problems) - 3} more' if len(problems) > 3 else '')
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True, help='model.safetensors of the released model')
    p.add_argument('--out', required=True)
    p.add_argument('--ct', default=PARIS4_LEVEL0, help='level-0 zarr array (URL or local directory)')
    p.add_argument('--unit', type=int, default=2048, help='unit edge in output voxels (a multiple of --chunk)')
    p.add_argument('--chunk', type=int, default=256, help='output chunk edge')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--box', type=int, nargs=6, help='output-voxel box z0 y0 x0 z1 y1 x1')
    p.add_argument('--shard', default='0/1', help='i/n: this process runs units i, i + n, ...')
    p.add_argument('--presence', default='auto', help="'list', 'probe', 'auto' or a saved presence .npy")
    p.add_argument('--compressor', choices=('blosc', 'none'), default='blosc')
    p.add_argument('--fetch-threads', type=int, default=64)
    p.add_argument('--write-threads', type=int, default=8)
    p.add_argument('--no-fsync', action='store_true')
    p.add_argument('--no-check', action='store_true', help='skip the per-unit CPU check of percentiles / normalization')
    p.add_argument('--max-units', type=int, default=0)
    args = p.parse_args(argv)
    if args.unit <= 0 or args.chunk <= 0 or args.unit % args.chunk:
        p.error('--unit must be a positive multiple of --chunk')
    shard, shards = (int(v) for v in args.shard.split('/'))
    if not 0 <= shard < shards:
        p.error('--shard must be i/n with 0 <= i < n')

    store = ChunkStore(args.ct, cache_gib=.5, threads=args.fetch_threads)
    try:
        infer(args, shard, shards, store)
    finally:
        store.close()


def infer(args, shard, shards, store):
    out = Path(args.out)
    (out / 'state' / 'done').mkdir(parents=True, exist_ok=True)
    assert store.chunks == (128, 128, 128), 'expects a level-0 array with 128^3 chunks'
    shape1 = V.output_shape(store.shape)
    meta = V.zarray(shape1, args.chunk, V.CODEC if args.compressor == 'blosc' else None)
    array = out / 'code.zarr' / '0'
    mode = args.presence if args.presence != 'auto' else ('probe' if args.box else 'list')
    run = dict(ct=str(args.ct), ct_shape=list(store.shape), shape=list(shape1), unit=args.unit, batch=args.batch,
               chunk=args.chunk, weights_sha256=rt.sha256_file(args.weights), zarray=meta, presence=mode)
    # Exclusive publication: of processes starting together, one settles the settings; the others must match them.
    rt.create_file(out / 'RUN.json', (json.dumps(run, indent=1, default=str) + '\n').encode())
    old = json.loads((out / 'RUN.json').read_text())
    changed = [k for k in run if old.get(k) != run[k]]
    if changed:
        raise ValueError(f'{out} holds a run with different {changed}; use a new --out')
    # Every process that gets here has these settings: identical, atomic writes (repairs a crash before them).
    rt.write_json(out / 'code.zarr' / '.zgroup', dict(zarr_format=2))
    rt.write_json(array / '.zarray', meta)
    rt.write_json(array / '.zattrs', dict(
        description='ink_paris4_3d_4.8um_gambiarra codes on the PHercParis4 level-1 grid',
        voxel_size_um=4.8, level0_mapping='q0 = 2 * q1 + 0.5 (voxel centres)',
        decode='0: no prediction; c in 2..254: logit in [(c - 128) / 10, (c - 127) / 10); '
               '1: below -12.6; 255: 12.7 and above'))

    presence = None
    if mode == 'list':
        cached = out / 'presence.npy'
        if not cached.exists():
            t = time.time()
            listed = io.BytesIO()
            np.save(listed, store.presence())
            if rt.create_file(cached, listed.getvalue()):
                print(f'presence listed ({time.time() - t:.0f} s)', flush=True)
        presence = np.load(cached)  # every process uses the first published listing
        print(f'presence: {int(presence.sum())} of {presence.size} chunks present', flush=True)
    elif mode != 'probe':
        presence = np.load(mode)
    if presence is not None:
        assert presence.shape == store.grid() and presence.dtype == bool

    def window(c0, c1):
        if presence is None:
            return store.probe(c0, c1)
        return presence[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]

    rt.configure_torch(27)
    torch.use_deterministic_algorithms(False)  # inference only; fixed cuDNN algorithms below
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    student = Student(seed=27)
    load_weights(student.network, args.weights)
    student.eval().requires_grad_(False)
    writer = V.ZarrWriter(array, meta, threads=args.write_threads, fsync=not args.no_fsync)
    try:
        engine = V.Engine(student.low, store, writer, batch=args.batch, check=None if args.no_check else V.cpu_check)
        box = np.array(args.box) if args.box else None
        todo = units_of(shape1, args.unit, box)[shard::shards]
        computed = verified = 0
        for idx in todo:
            name = '_'.join(map(str, idx))
            marker = out / 'state' / 'done' / f'{name}.json'
            if not marker.exists() and args.max_units and computed >= args.max_units:
                break
            t = time.time()
            plan = V.unit_plan(idx, args.unit, store.shape, engine.grid, window, args.chunk)
            if marker.exists():
                problem = marker_problem(marker, plan, writer)
                if problem is None:
                    verified += 1
                    continue
                print(f'unit {name}: done marker rejected ({problem}); recomputing', flush=True)
                marker.unlink()
                if args.max_units and computed >= args.max_units:
                    break
            record = dict(unit=name, T0=plan['T0'], T1=plan['T1'], chunks=[list(k) for k in plan['chunks']],
                          presence_sha256=presence_sha256(plan))
            if 'empty' in plan and plan['chunks']:
                raise RuntimeError(f'unit {name}: present CT in the tile but no crop ({plan["empty"]})')
            writer.clean(plan['T0'], plan['T1'], plan['chunks'])
            if 'empty' in plan:
                record.update(state='empty', reason=plan['empty'])
            else:
                record.update(state='done', crops=plan['crops'], **engine.run_unit(plan))
            record['seconds'] = time.time() - t
            rt.write_json(marker, record)
            computed += 1
            print(f'unit {name}: {record["state"]}, {record.get("crops", 0)} crops, {len(record["chunks"])} chunks, '
                  f'{record["seconds"]:.1f} s', flush=True)
        print(f'done markers verified: {verified}; units computed: {computed}', flush=True)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
