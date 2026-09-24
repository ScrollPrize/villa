"""Method-agnostic pieces of a fiber_follow training run.

Shared by the spatial follower (``train.py``) and the beam re-ranker
(``beam/train.py``): run directory creation, the JSON-lines log, the
warmup-cosine schedule, one guarded optimizer step, and checkpoint I/O that
records which architecture and data policy produced the weights.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import torch

from vesuvius.neural_tracing.fiber_follow.data import DATA_POLICY


def prepare_run_dir(out_root, name) -> Path:
    """Create ``out_root/name``; refuse to reuse a directory that holds a run."""
    out = Path(out_root) / name
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; use a new --name')
    out.mkdir(parents=True, exist_ok=True)
    return out


class RunLog:
    """Appends one JSON object per line to ``log.jsonl`` and echoes it."""

    def __init__(self, path):
        self._file = Path(path).open('a')

    def record(self, values: dict):
        line = json.dumps(values)
        print(line, flush=True)
        self._file.write(line + '\n')
        self._file.flush()

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
    if ck['architecture'] != architecture or ck['data_policy'] != DATA_POLICY:
        raise ValueError(f'Checkpoint {path} is {ck["architecture"]}/{ck["data_policy"]}, '
                         f'not {architecture}/{DATA_POLICY}')
    return ck
