"""Paired v4/v5 cost on captured real streams, without altering source runs.

Capture volume crops once, then run each architecture in a separate process.
Ordinary updates, an additional real endpoint replay, and warm inference are
reported separately. All model weights train (LR zero for stationary timing).
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import build_model
from .candidate_model import initialize_candidate_model
from .data import IdentityObservationBuilder, IdentitySampling, load_contacts, load_hard_spans
from .feature_sequences import FeatureStreamStates
from .neighbor_bank import NeighborBank
from .stratified_replay import take_row
from .train import checkpoint_config, compile_training_model, conv_memory_format, move_batch, optimizer_update
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def capture(args):
    if args.out.exists():
        raise ValueError('Capture requires a fresh output directory')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = replace(checkpoint_config(ck), memory_version=5, recurrent_refinement_steps=2)
    if cfg.feature_memory_revision != 2:
        raise ValueError('Source must be v4 revision 2')
    spec, options = FiberVolumeSpec(**ck['vol_spec']), ck['training_options']
    sample = SampleConfig(**{**ck['sample_cfg'], 'crop': cfg.fine})
    band = ZBand(*(float(z)/spec.grid_scale for z in options['val_z']))
    print('Loading real fiber annotations and replay metadata', flush=True)
    fibers, _ = split_fibers(load_fibers(options['fibers'], grid_scale=spec.grid_scale), band)
    sampling = IdentitySampling(**{**ck.get('identity_sampling', {}),
                                  'rule': ComponentRule(**ck.get('identity_sampling', {}).get('rule', {'lateral_max': 32.}))})
    # Checkpoints store the effective sampling options separately from the original config.
    sampling = replace(sampling, **{name: options[name] for name in sampling.__dataclass_fields__ if name in options})
    bank = NeighborBank(options['negative_bank'], fibers, band, grid_scale=spec.grid_scale,
                        refresh_seconds=1e9, cache_bytes=64 << 20)
    bank.validate_volume(spec)
    replay_paths = json.loads((Path(args.checkpoint).parent/'dagger/replay.json').read_text())
    replay = [OnPolicyStates.load(p) for p in replay_paths]
    builder = IdentityObservationBuilder(cfg, fibers, sampling, negative_bank=bank, augment=True,
        contacts=load_contacts(options['contacts'], fibers, band) if options.get('contacts') else (),
        hard_spans=load_hard_spans(options['hard_spans'], fibers) if options.get('hard_spans') else ())
    dataset = FollowDataset(fibers, spec, sample, band, chunk=2, seed=args.seed, cache_bytes=512 << 20,
                            onpolicy=replay, batch_builder=builder, fresh_fraction=options['fresh_fraction'])
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    files, counts, ended, started = [], {}, set(), time.perf_counter()
    active = set()
    for chunk in dataset:
        path = args.out/f'chunk_{len(files):04d}.pt'
        torch.save(chunk, path)
        files.append(path.name)
        for row in chunk['feature_sequence']:
            for key, reset, end in zip(row['stream_id'].tolist(), row['stream_reset'].tolist(), row['stream_end'].tolist()):
                if reset:
                    active.add(key)
                counts[key] = counts.get(key, 0)+1
                if end:
                    ended.add(key)
                    active.discard(key)
        print(json.dumps(dict(chunks=len(files), crops=sum(counts.values()), ended=len(ended))), flush=True)
        if len(ended) >= args.streams and not active:
            break
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        config=cfg.to_dict(), volume=spec.to_dict(), sampling=asdict(sampling), seed=args.seed,
        replay_paths=replay_paths, stream_lengths=counts, capture_seconds=time.perf_counter()-started,
        files=[dict(path=p, sha256=hashlib.sha256((args.out/p).read_bytes()).hexdigest()) for p in files])
    (args.out/'capture.json').write_text(json.dumps(result, indent=2))


def clone_state(state):
    return {k: v.detach().clone() for k, v in state.items()}


def prepare(model, fixture, max_windows=8):
    """Reconstruct actual causal memory and archives; no invented repeated crops."""
    metadata = json.loads((fixture/'capture.json').read_text())
    states, windows, archives = FeatureStreamStates(), [], []
    for entry in metadata['files']:
        path = fixture/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Captured sample hash changed')
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        sequence = chunk['feature_sequence']
        full = len(sequence) == 2 and all(len(row['hist']) == 2 for row in sequence)
        if full:
            incoming = states.incoming(model, sequence[0], 'cuda')
            windows.append((chunk, clone_state(incoming)))
        for cpu in sequence:
            carried = states.incoming(model, cpu, 'cuda')
            b = move_batch(cpu, 'cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features = model.observation_features(b['x'], b['hist'], b['hmask'])
                state, _ = model.recurrent_memory.observe_tokens(*features, b['x'], carried)
            output = {'memory_'+k: v for k, v in state.items()}
            output.update(zip(('observation_tokens', 'observation_xyz', 'observation_valid'), features))
            states.update(cpu, output, carried)
            states.detach()
            states.replay.record(cpu, output)
            archives.extend(states.replay.pending)
            states.replay.pending.clear()
    if len(windows) < 2 or not archives:
        raise ValueError('Capture needs at least two full two-trace windows and one completed stream')
    # Evenly spaced windows cover startup, established history, and later decisions.
    chosen = np.linspace(0, len(windows)-1, min(len(windows), max_windows)).round().astype(int)
    return [windows[i] for i in chosen], archives, metadata


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    metadata = json.loads((args.fixture/'capture.json').read_text())
    source = Path(metadata['checkpoint'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    cfg = checkpoint_config(ck)
    baseline = build_model(cfg)
    baseline.load_state_dict(ck['model'], strict=True)
    model = baseline if args.version == 4 else initialize_candidate_model(baseline,
        replace(cfg, memory_version=5, recurrent_refinement_steps=2))
    model = model.to('cuda', memory_format=conv_memory_format('cuda'))
    del baseline, ck
    windows, archives, metadata = prepare(model, args.fixture)
    print(json.dumps(dict(prepared_windows=len(windows), replay_lengths=[len(r) for r in archives])), flush=True)
    wrapped = compile_training_model(model) if args.compile else model
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=1e-4)
    states = FeatureStreamStates()
    # Cache inference tensors outside timing, as in the existing model-cost benchmarks.
    inference = []
    for chunk, incoming in windows:
        row = take_row(chunk['feature_sequence'][0], 0)
        inference.append((move_batch(row, 'cuda'), {k: v[:1] for k, v in incoming.items()}))
    modes = args.modes
    result = dict(version=args.version, source_checkpoint=str(source), source_sha256=metadata['checkpoint_sha256'],
        fixture=str(args.fixture.resolve()), fixture_sha256=hashlib.sha256((args.fixture/'capture.json').read_bytes()).hexdigest(),
        config=model.cfg.to_dict(), hardware=torch.cuda.get_device_name(), torch=torch.__version__,
        compiled=args.compile, threads=args.threads, effective_batch=8, microbatch=4, concurrent_traces=2,
        sequence_length=2, precision='BF16 encoder/decoder; FP32 memory',
        protocol='Captured real CT/presence/direction crops; identical causal histories. LR zero; all weights train. Includes losses, backward, clipping, AdamW, EMA and training H2D; excludes loader IO and diagnostics.',
        warmup=args.warmup, repeats=args.repeats, modes={})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def training_step(i, replay):
        chunks = []
        states.states.clear()
        for j in range(2):
            chunk, incoming = windows[(2*i+j) % len(windows)]
            rows = []
            for original in chunk['feature_sequence']:
                row = {k: v for k, v in original.items() if k != 'replay_select'}
                row.update(stream_id=torch.arange(2)+j*2, stream_reset=torch.zeros(2, dtype=torch.bool),
                           stream_end=torch.zeros(2, dtype=torch.bool))
                rows.append(row)
            for r in range(2):
                states.states[j*2+r] = {k: v[r:r+1] for k, v in incoming.items()}
            chunks.append(dict(feature_sequence=rows))
        if replay:
            states.replay.pending.append(archives[i % len(archives)])
        return optimizer_update(wrapped, ema, opt, chunks, i+1, 0., device='cuda', n_commit=16,
                                compute_metrics=False, stream_states=states)

    for mode in modes:
        model.train(mode != 'inference')
        def step(i):
            if mode == 'inference':
                row, incoming = inference[i % len(inference)]
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    out = wrapped(row['x'], row['hist'], row['hmask'], memory=incoming)
                return {'finite': bool(torch.isfinite(out['points']).all())}
            return training_step(i, mode == 'replay')
        warm_started = time.perf_counter()
        for i in range(args.warmup):
            metrics = step(i)
            torch.cuda.synchronize()
            print(json.dumps(dict(mode=mode, warmup=i+1, loss=metrics.get('loss'))), flush=True)
        warm_seconds = time.perf_counter()-warm_started
        torch.cuda.reset_peak_memory_stats()
        timings, replays, recompiles = [], [], []
        from torch._dynamo.utils import counters
        for i in range(args.repeats):
            graphs = counters['stats']['unique_graphs']
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = step(i+args.warmup)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            replays.append(metrics.get('replay_observations', 0))
            recompiles.append(counters['stats']['unique_graphs']-graphs)
        if any(recompiles):
            raise RuntimeError('Measured region compiled new graphs; increase warmup')
        row = dict(compiled=args.compile, mean_ms=float(np.mean(timings)), p50_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)), samples_ms=timings, warmup_seconds=warm_seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            rate_per_second=(1 if mode == 'inference' else 8)*1000/float(np.mean(timings)),
            replay_observations=replays, final_loss=metrics.get('loss'))
        if mode != 'inference':
            row['mean_ms_per_ordinary_crop'] = row['mean_ms']/8
        result['modes'][mode] = row
        (args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(mode=mode, **{k: v for k, v in row.items() if k not in ('samples_ms', 'replay_observations')})), flush=True)
        if args.profile and mode == 'training':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                step(0)
                torch.cuda.synchronize()
            args.out.with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total', row_limit=25))
    print('RESULT '+str(args.out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    cp = sub.add_parser('capture')
    cp.add_argument('--checkpoint', type=Path, required=True)
    cp.add_argument('--out', type=Path, required=True)
    cp.add_argument('--streams', type=int, default=4)
    cp.add_argument('--seed', type=int, default=194)
    rp = sub.add_parser('run')
    rp.add_argument('--fixture', type=Path, required=True)
    rp.add_argument('--out', type=Path, required=True)
    rp.add_argument('--version', type=int, choices=(4, 5), required=True)
    rp.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    rp.add_argument('--warmup', type=int, default=8)
    rp.add_argument('--repeats', type=int, default=20)
    rp.add_argument('--threads', type=int, default=4)
    rp.add_argument('--seed', type=int, default=194)
    rp.add_argument('--memory-fraction', type=float, default=.8)
    rp.add_argument('--modes', nargs='+', choices=('inference', 'training', 'replay'), default=['inference', 'training', 'replay'])
    rp.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    if args.command == 'capture':
        capture(args)
    else:
        if args.out.exists() or min(args.warmup, args.repeats, args.threads) < 1 or not 0 < args.memory_fraction <= 1:
            ap.error('Need fresh output, positive counts, and memory fraction in (0,1]')
        run(args)


if __name__ == '__main__':
    main()
