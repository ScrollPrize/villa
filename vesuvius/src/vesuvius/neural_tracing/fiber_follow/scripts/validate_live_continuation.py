"""Read-only CPU smoke test of live continuation on a mixed-run checkpoint.

Run with the project Python; writes only the requested JSON report. Uses the
current training weights, real CT and source-local annotations/banks.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.regression.collect import failure_banks, load_dataset
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.live_continuation import LiveContinuation, LiveContinuationSource
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint, move_batch
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, _, _, _, ck = load_checkpoint(args.checkpoint, 'cpu')
    model.load_state_dict(ck['model'])
    model.eval()
    cfg = model.cfg
    sampling = dict(ck['identity_sampling'])
    sampling['rule'] = ComponentRule(**sampling['rule'])
    sampling = IdentitySampling(**sampling)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                         future_step=cfg.future_step, full_observed_history=True)
    records = []
    with torch.no_grad():
        for source in ck['dataset_config']['sources']:
            opts = SimpleNamespace(dataset_name=source['name'], failure_bank=None,
                                   bank_switch_tolerance=.75, bank_own_tolerance=1.5)
            spec, fibers, band = load_dataset(opts, ck)
            detector = failure_banks(opts, ck, fibers, band, spec)
            builder = IdentityObservationBuilder(cfg, fibers, sampling, augment=False,
                                                 negative_bank=detector.banks[0])
            ds = FollowDataset(fibers, spec, sample, band, batch_builder=builder)
            vol = FiberVolume(spec, cache_bytes=256 << 20)
            live = LiveContinuationSource(steps=(4, 4), n_commit=ck['n_commit'],
                                          max_recovery_distance=cfg.max_recovery_distance)
            ds.live_continuation = live
            live.detector = detector
            wrapper = LiveContinuation.__new__(LiveContinuation)
            wrapper.sources = [live]
            rng = np.random.default_rng(723)
            record = dict(dataset=source['name'], attempts=0, continued=0, failures=0, max_depth=0)
            print('Checking '+source['name'], flush=True)
            try:
                for attempt in range(8):
                    fi = int(rng.choice(len(fibers), p=ds.weights))
                    f = fibers[fi]
                    t = float(rng.uniform(.2, .7)*f.length)
                    item = make_sample(f, t, False, sample, rng)
                    item.update(fiber_ref=(fi, t, False), gt_unperturbed=True,
                                source=0, source_step=-1, stratum=-1)
                    item = ds.prepare(item, rng)
                    if not ds.state_allowed(item):
                        continue
                    for depth in range(4):
                        batch = builder([item], vol)
                        batch['_live_states'] = [live.metadata(item)]
                        inputs = move_batch(batch, 'cpu')
                        output = model(inputs['x'], inputs['hist'], inputs['hmask'], n_commit=ck['n_commit'])
                        assert torch.isfinite(output['points']).all()
                        record['attempts'] += 1
                        wrapper.feedback(batch, output, ck['step'])
                        queue = live.chains if depth else live.seeds
                        from queue import Empty
                        try:
                            state = queue.get(timeout=.2)
                        except Empty:
                            break
                        next_item = live.advance(state, ds, vol, rng)
                        if next_item is None:
                            break
                        np.testing.assert_array_equal(next_item['observed_path'][:len(item['observed_path'])], item['observed_path'])
                        assert next_item['fiber_ref'][0] == fi
                        assert next_item['live_depth'] == depth+1
                        assert next_item['source_step'] == ck['step']
                        record['continued'] += 1
                        record['max_depth'] = max(record['max_depth'], depth+1)
                        if next_item['offtrack']:
                            record['failures'] += 1
                            assert live.metadata(next_item) is None
                            # Verify the terminal state's real images and targets too.
                            builder([next_item], vol)
                            break
                        item = next_item
                    if record['max_depth'] == 4:
                        break
                assert record['continued'], f'No accepted live continuation for {source["name"]}'
                records.append(record)
                print(json.dumps(record), flush=True)
            finally:
                live.close()
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck['step'], device='cpu', sources=records)
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
