"""Fixed paired history ablations; geometry and both confidence outcomes stay separate."""
import argparse
import json
from pathlib import Path

import torch

from .train import load_checkpoint, move_batch


def history_inputs(batch, mode):
    """Change historical slabs only; retain current crop, local history and poses."""
    x = dict(batch['x'])
    valid = x['history_valid'].clone()
    if mode == 'seed_only':
        valid[:, 1:] = False
    elif mode == 'none':
        valid[:] = False
    elif mode == 'shuffled_imagery':
        if len(valid) % 2:
            raise ValueError('Imagery swaps require complete matched pairs')
        x['history_slabs'] = x['history_slabs'].clone()
        partner = torch.arange(len(valid), device=valid.device)^1
        x['history_slabs'][:, :, 0] = batch['x']['history_slabs'][partner, :, 0]
    elif mode != 'full':
        raise ValueError(f'Unknown history diagnostic mode: {mode}')
    x['history_valid'] = valid
    return x


@torch.no_grad()
def paired_history_report(model, batch, threshold=.5, *, n_commit=None, on_prediction=None):
    """Batch contains matched identities in adjacent rows and known candidate labels.

    The fixture is frozen by the caller. Shuffle swaps imagery within each pair,
    retaining that row's pose, age, slot order, seed role, heatmap and valid mask.
    """
    if len(batch['hist']) % 2 or 'candidate_points' not in batch:
        raise ValueError('Diagnostic requires complete matched pairs and candidate paths')
    was_training = model.training
    model.eval()
    result = {}
    try:
        for mode in ('full', 'seed_only', 'none', 'shuffled_imagery'):
            x = history_inputs(batch, mode)
            out = model(x, batch['hist'], batch['hmask'], candidates=batch['candidate_points'],
                        confidence_threshold=threshold, n_commit=n_commit)
            window = n_commit or model.cfg.n_future
            known = batch['candidate_mask'][..., :window].bool().all(-1)
            correct = (batch['candidate_labels'][..., :window] > .5).all(-1) & known
            wrong = ((batch['candidate_labels'][..., :window] <= .5) & batch['candidate_mask'][..., :window].bool()).any(-1)
            accepted = out['candidate_confidence'][..., window-1] >= threshold
            distances = (out['points'][:, None]-batch['candidate_points']).square().sum(-1).mean(-1)
            choice = distances.argmin(-1)
            selected_correct = correct.gather(1, choice[:, None]).squeeze(1)
            row = dict(states=len(batch['hist']), geometry_correct=int(selected_correct.sum()),
                       correct_continuations=int(correct.sum()), correct_accepted=int((correct & accepted).sum()),
                       wrong_continuations=int(wrong.sum()), wrong_rejected=int((wrong & ~accepted).sum()))
            row.update(geometry_choice_accuracy=row['geometry_correct']/len(batch['hist']),
                       correct_acceptance=row['correct_accepted']/max(1, row['correct_continuations']),
                       wrong_rejection=row['wrong_rejected']/max(1, row['wrong_continuations']))
            result[mode] = row
            if on_prediction is not None:
                on_prediction(mode, out)
    finally:
        model.train(was_training)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fixture', required=True, help='Frozen trusted tensor batch from paired observations')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh diagnostic output path')
    model, *_ = load_checkpoint(args.checkpoint, args.device)
    batch = move_batch(torch.load(args.fixture, map_location='cpu', weights_only=True), args.device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        report = paired_history_report(model, batch)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
