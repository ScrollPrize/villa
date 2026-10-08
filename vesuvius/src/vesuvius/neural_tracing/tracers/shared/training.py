"""Training-run pieces shared by the tracers: run directory creation, the JSON-lines log, the warmup-cosine schedule,
gradient clipping, the weight EMA and RNG state."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


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


def lr_at(step: int, base_lr: float, warmup: int, total_steps: int, progress: float | None = None) -> float:
    """Linear warmup multiplied by a cosine decay over ``total_steps`` (``step`` counts from 1). ``progress``, a
    fraction of the run done by another measure (e.g. of a time budget), advances the cosine when it is further
    along; the decay then ends at zero."""
    angle = (math.pi * (step - 1) / total_steps if progress is None
             else math.pi * min(1., max((step - 1) / total_steps, progress)))
    return base_lr * min(1., step / max(1, warmup)) * .5 * (1 + math.cos(angle))


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


def clip_training_gradients(model, max_norm):
    """Clip the global gradient norm (``max_norm`` 0 disables clipping); a nonfinite norm raises before any gradient
    is modified. Returns the norm and the clip scale applied."""
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError('The gradient clipping limit must be finite and nonnegative (0 disables clipping)')
    params = list(model.parameters())
    norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None], error_if_nonfinite=True)
    if max_norm > 0:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, norm)
    norm = float(norm)
    return dict(grad_norm=norm, grad_clip_scale=min(1., max_norm/(norm+1e-6)) if max_norm else 1.)
