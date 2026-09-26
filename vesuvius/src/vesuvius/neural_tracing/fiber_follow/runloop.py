"""Method-agnostic pieces of a fiber_follow training run.

Shared by the flow follower, direct follower and beam re-ranker:
run directory creation, the JSON-lines log, the
warmup-cosine schedule, one guarded optimizer step, and checkpoint I/O that
records which architecture and data policy produced the weights.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import DATA_POLICY


def prepare_run_dir(out_root, name, resume=False) -> Path:
    """Create ``out_root/name``; refuse to reuse a directory that holds a run.

    With ``resume`` the directory must already hold a run and is reused.
    """
    out = Path(out_root) / name
    if resume:
        if not (out / 'config.json').exists():
            raise FileNotFoundError(f'{out} holds no run to resume')
        return out
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; use a new --name or --resume')
    out.mkdir(parents=True, exist_ok=True)
    return out


class RunLog:
    """Append JSON lines; optionally format a separate terminal representation."""

    def __init__(self, path, *, formatter=None):
        self._file = Path(path).open('a')
        self._formatter = formatter

    def record(self, values: dict):
        line = json.dumps(values)
        self._file.write(line + '\n')
        self._file.flush()
        print(self._formatter(values) if self._formatter else line, flush=True)

    def close(self):
        self._file.close()


def lr_at(step: int, base_lr: float, warmup: int, total_steps: int) -> float:
    """Linear warmup multiplied by a cosine decay over ``total_steps``."""
    return base_lr * min(1., step / max(1, warmup)) * .5 * (1 + math.cos(math.pi * (step - 1) / total_steps))


def optimizer_step(model, opt, loss, step: int, lr: float):
    """Sets the learning rate, checks the loss, clips gradients, and steps."""
    for group in opt.param_groups:
        group['lr'] = lr
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite training loss at step {step}')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()


def save_checkpoint(path, model, vol_spec, crop, n_history, architecture, extra=None):
    torch.save(dict(architecture=architecture, data_policy=DATA_POLICY, model=model.state_dict(),
                    model_cfg=model.cfg.to_dict(), crop=dataclasses.asdict(crop),
                    n_history=n_history, vol_spec=vol_spec.to_dict(), **(extra or {})), path)


def read_checkpoint(path, architecture, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    accepted = (architecture,) if isinstance(architecture, str) else tuple(architecture)
    if ck['architecture'] not in accepted or ck['data_policy'] != DATA_POLICY:
        raise ValueError(f'Checkpoint {path} is {ck["architecture"]}/{ck["data_policy"]}, '
                         f'not {architecture}/{DATA_POLICY}')
    return ck


@torch.no_grad()
def update_ema(ema, model, step, decay):
    """Update EMA once per optimizer update, ramping the decay in early updates.

    Without the ramp the average still carries 37% of the random initialization
    after 1,000 updates, which is what the first collector and diagnostics use.
    """
    effective_decay = min(decay, (1+step)/(10+step))
    for average, current in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(current.detach(), 1-effective_decay)
    for average, current in zip(ema.buffers(), model.buffers(), strict=True):
        average.copy_(current)


def training_rng_state():
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_training_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def resume_training(ck, model, ema, opt):
    """Restore weights, EMA, optimizer and RNG from a resumable checkpoint.

    Only ``ckpt_*.pt``/``last.pt`` written by training carry optimizer state;
    collector snapshots do not. Loader workers restart their own streams, so
    the sampled data sequence after a resume differs from an uninterrupted run.
    Returns the completed update count and replay samples seen so far.
    """
    if 'optimizer' not in ck:
        raise ValueError('Checkpoint holds no optimizer state; resume from ckpt_*.pt or last.pt of a run')
    model.load_state_dict(ck['model'])
    ema.load_state_dict(ck['ema'])
    opt.load_state_dict(ck['optimizer'])
    restore_training_rng(ck['rng'])
    return int(ck['step']), int(ck.get('replay_seen', 0))
