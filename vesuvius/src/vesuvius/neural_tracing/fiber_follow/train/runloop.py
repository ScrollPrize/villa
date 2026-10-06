"""Method-agnostic pieces of a fiber_follow training run.

Shared by every model type: run directory creation, the JSON-lines log, the
warmup-cosine schedule, one guarded optimizer step, and checkpoint I/O that
records which model type produced the weights.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch



def raise_open_file_limit():
    """Raise the soft descriptor limit before opening mmap caches or workers.

    Workers and collector subprocesses inherit this limit. Keep the macOS
    cap and leave the system hard limit unchanged.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 1 << 20)
        if sys.platform == 'darwin':
            target = min(target, 10240)
        if soft == resource.RLIM_INFINITY or soft >= target:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, ValueError, OSError):
        pass


def prepare_run_dir(out_root, name, resume=False) -> Path:
    """Create ``out_root/name``; refuse to reuse a directory that holds a run.

    With ``resume`` the directory is reused (and created if missing).
    """
    out = Path(out_root) / name
    if resume:
        out.mkdir(parents=True, exist_ok=True)
        return out
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; use a new name or --resume')
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


def lr_at(step: int, base_lr: float, warmup: int, total_steps: int, offset: int = 0) -> float:
    """Linear warmup multiplied by a cosine decay over ``total_steps``.

    With ``offset`` a fresh optimizer continues another run's cosine: the decay is evaluated
    at ``step+offset`` over ``total_steps+offset``; warmup still counts this run's own updates.
    """
    return (base_lr * min(1., step / max(1, warmup))
            * .5 * (1 + math.cos(math.pi * (step + offset - 1) / (total_steps + offset))))


def save_checkpoint(path, model, vol_spec, crop, n_history, model_type, extra=None):
    torch.save(dict(model_type=model_type, model=model.state_dict(),
                    model_cfg=model.cfg.to_dict(), crop=dataclasses.asdict(crop),
                    n_history=n_history, vol_spec=vol_spec.to_dict(), **(extra or {})), path)


def read_checkpoint(path, device='cuda'):
    from vesuvius.neural_tracing.fiber_follow.models.model import MODEL_TYPES
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('model_type') not in MODEL_TYPES:
        raise ValueError(f'Unsupported checkpoint model type {ck.get("model_type")!r} in {path}; checkpoints from '
                         'before the model cleanup need scripts/convert_checkpoint.py')
    return ck


@torch.no_grad()
def update_ema(ema, model, step, decay, *, ramp=True):
    """Update EMA once per optimizer update, ramping the decay in early updates.

    Without the ramp the average still carries 37% of the random initialization
    after 1,000 updates, which is what the first collector and diagnostics use.
    A run initialized from trained model and EMA tensors uses the full decay.
    """
    effective_decay = min(decay, (1+step)/(10+step)) if ramp else decay
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


def resume_training(ck, model, ema, opt, *, reset_optimizer=False):
    """Restore weights, EMA, optimizer and RNG from a resumable checkpoint.

    Only ``ckpt_*.pt``/``last.pt`` written by training carry optimizer state;
    collector snapshots do not. Loader workers restart their own streams, so
    the sampled data sequence after a resume differs from an uninterrupted run.
    With reset_optimizer, retain the supplied fresh optimizer instead of the
    saved moments/groups. Model weights, EMA, RNG and update count still resume.
    Returns the completed update count.
    """
    if 'optimizer' not in ck:
        raise ValueError('Checkpoint holds no optimizer state; resume from ckpt_*.pt or last.pt of a run')
    if reset_optimizer and opt.state:
        raise ValueError('Optimizer reset requires a fresh optimizer with empty state')
    model.load_state_dict(ck['model'])
    ema.load_state_dict(ck['ema'])
    if not reset_optimizer:
        opt.load_state_dict(ck['optimizer'])
    if 'rng' in ck:
        restore_training_rng(ck['rng'])
    return int(ck['step'])
