"""Compare full-size revision-1/2 optimizer updates, optionally with endpoint replay.

Use the same checkpoint ancestry, precision, batch, warmup and repeats for both
revisions. --replay-length measures one additional historical endpoint per update;
it is a replay-event stress measurement, not an amortized training throughput.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import build_model
from .benchmark_memory import synthetic_batch
from .train import checkpoint_config, compile_training_model, conv_memory_format, optimizer_update, move_batch
from .feature_sequences import FeatureStreamStates
from .stratified_replay import stratified_indices, take_row


def replay_fixture(model, batch, length):
    """Repeated synthetic crop evidence at different poses; selected crops re-encode."""
    sample = take_row(batch, 0)
    selected = stratified_indices(length, np.random.default_rng(194))
    device = next(model.parameters()).device
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        b = move_batch(sample, device)
        features = tuple(v.detach() for v in model.observation_features(b['x'], b['hist'], b['hmask']))
    rows = []
    for t in range(length):
        pose = dict(query_position=torch.tensor([[0., 0., float(t)*4]]),
                    query_frame=torch.eye(3)[None], feature_seed_here=torch.tensor([t == 0]))
        observed = dict(sample, x={**sample['x'], **pose}) if t in selected or t == length-1 else None
        rows.append(dict(features=features, pose=pose, batch=observed))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--replay-length', type=int, default=0)
    ap.add_argument('--eager', action='store_true')
    args = ap.parse_args()
    if min(args.warmup, args.repeats) < 1 or args.replay_length < 0 or args.replay_length == 1:
        ap.error('Positive repetitions and zero or at least two replay observations required')
    torch.set_num_threads(4)
    torch.manual_seed(194)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint_config(ck)
    model = build_model(cfg).to('cuda', memory_format=conv_memory_format('cuda'))
    model.load_state_dict(ck['model'])
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=.0003)
    wrapped = model if args.eager else compile_training_model(model)
    chunks = []
    for j in range(2):
        sequence = []
        for t in range(2):
            b = synthetic_batch(cfg, 2, 1)
            b['x']['query_position'][:, 2] = t*4
            b['x']['feature_seed_here'].fill_(t == 0)
            b.update(stream_id=torch.arange(2)+j*2, stream_reset=torch.full((2,), t == 0),
                     stream_end=torch.full((2,), t == 1))
            sequence.append(b)
        chunks.append(dict(feature_sequence=sequence))
    states = FeatureStreamStates()
    replay = replay_fixture(model, chunks[0]['feature_sequence'][0], args.replay_length) if args.replay_length else None

    def step(i):
        if replay:
            states.replay.pending.append(replay)
        return optimizer_update(wrapped, ema, opt, chunks, i, .0003, device='cuda', n_commit=8,
                                compute_metrics=False, stream_states=states)

    print('Starting warmup', flush=True)
    warmup_started = time.perf_counter()
    for i in range(args.warmup):
        metrics = step(i+1)
        torch.cuda.synchronize()
        print('warmup', i+1, metrics['loss'], flush=True)
    warmup_seconds = time.perf_counter()-warmup_started
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for i in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = step(i+args.warmup+1)
        torch.cuda.synchronize()
        timings.append((time.perf_counter()-started)*1000)
    result = dict(checkpoint=args.checkpoint, config=cfg.to_dict(), hardware=torch.cuda.get_device_name(),
        torch=torch.__version__, compiled=not args.eager, precision='BF16 encoder/decoder; FP32 memory',
        effective_batch=8, microbatch=4, sequence_length=2, concurrent_traces=2,
        input='Fixed synthetic full-size CT/presence/directions, two candidate curves per crop; geometry and survival losses; includes AdamW, clipping, EMA; excludes volume IO',
        replay_length=args.replay_length, replay_endpoints_per_update=int(bool(replay)),
        replay_encoder_crops=metrics.get('replay_encoder_crops', 0),
        warmup=args.warmup, warmup_seconds=warmup_seconds, repeats=args.repeats,
        mean_ms=float(np.mean(timings)), p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, samples_ms=timings, final_loss=metrics['loss'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
