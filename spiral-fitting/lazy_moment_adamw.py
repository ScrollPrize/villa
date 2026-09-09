"""AdamW with lazy (SparseAdam-style) moments and a shared, robust second
moment for selected parameter groups, plus robust gradient clipping for the
flow lattices.

The clipping median and winsorisation quantile use fixed-stride subsamples
of roughly bounded size. Identical input gradients and state give every DDP
rank the same samples; structured sparse support may be missed. The final
shared mean, clipping, and update diagnostics still scan the full lattice.
See README.md, "Rationale and references", for SparseAdam, Adam-mini and
clipping precedents; the grouping and robust threshold rules here are custom.
"""

import math

import torch


# Elements per chunk of the masked update, bounding its temporaries to a
# fraction of the lattice (the full-scroll high-resolution flow lattice is
# ~400M values per slab).
_CHUNK_ELEMENTS = 1 << 26
# Target size of the fixed-stride subsample the robust statistics read.
STATS_SUBSAMPLE = 1 << 20
# Default winsorisation quantile of the shared second moment (see
# LazyMomentAdamW); None or >= 1 disables the cap.
DEFAULT_CLIP_QUANTILE = 0.99


def _subsample_stride(numel):
    return max(1, int(numel) // STATS_SUBSAMPLE)


def _stage_view(tensor):
    """``tensor`` as [stages, cells]: one plane per leading index of a
    lattice-shaped tensor ([stages, components, *spatial]), or a single plane
    for anything with fewer than three dimensions."""
    if tensor.dim() >= 3:
        return tensor.view(tensor.shape[0], -1)
    return tensor.view(1, -1)


def _num_components(tensor):
    return int(tensor.shape[1]) if tensor.dim() >= 3 else 1


def _nan_to_inf(value):
    # A statistic over an empty subsample is NaN; as a clipping bound that
    # must read "no bound", never propagate into the parameters.
    return torch.where(torch.isnan(value), torch.full_like(value, math.inf), value)


def robust_clip_(grad, multiple):
    """Clip a lattice gradient in place, per stage, at ``multiple`` times the
    median nonzero absolute component value of that stage. This is
    componentwise clipping, not vector-norm clipping, and can change direction.

    The median is read from a fixed-stride subsample of the stage's entries
    (see STATS_SUBSAMPLE), without any host synchronisation. Returns
    ``(threshold, clipped_fraction)``, two float64 tensors of length
    ``stages`` on ``grad.device`` (threshold is +inf when the sample has no
    nonzero entry, even if unsampled entries are active), or ``None`` when
    ``multiple`` is not positive, in
    which case ``grad`` is untouched.
    """
    if multiple is None or float(multiple) <= 0.0:
        return None
    planes = _stage_view(grad)
    stride = _subsample_stride(planes.shape[1])
    thresholds = torch.empty(planes.shape[0], dtype=torch.float64, device=grad.device)
    fractions = torch.zeros_like(thresholds)
    for index in range(planes.shape[0]):
        plane = planes[index]
        sub = plane[::stride]
        magnitude = torch.where(sub != 0, sub.abs(), torch.full_like(sub, math.nan))
        threshold = _nan_to_inf(torch.nanmedian(magnitude).to(torch.float64) * float(multiple))
        bound = threshold.to(plane.dtype)
        clipped = torch.zeros((), dtype=torch.float64, device=grad.device)
        for start in range(0, plane.numel(), _CHUNK_ELEMENTS):
            chunk = plane[start:start + _CHUNK_ELEMENTS]
            clipped += (chunk.abs() > bound).sum(dtype=torch.float64)
            chunk.clamp_(min=-bound, max=bound)
        thresholds[index] = threshold
        fractions[index] = clipped / plane.numel()
    return thresholds, fractions


class LazyMomentAdamW(torch.optim.AdamW):
    """AdamW with optional masked moments and shared second-moment scaling.

    ``lazy_moments=True`` applies an explicit ``grad != 0`` mask to dense
    gradients: only these entries update moments and receive a gradient step.
    This follows torch.optim.SparseAdam's masked-update idea, although that
    implementation uses materialized sparse entries as its mask. The fitter
    passes gradients after clipping, smoothing and influence masks, so active
    entries need not have been directly sampled. Inactive entries retain
    history, including stale momentum. A global per-parameter step counter
    drives bias correction; there are no per-entry touch counters. Lazy
    moments do not resolve first-touch amplification of tiny gradients.

    ``shared_second_moment=True`` uses one denominator per leading slab of
    each parameter (see _stage_view), across all its spatial entries and
    vector components. Coarse/fine lattices and separate flow-stage parameters
    do not share denominators. Per-entry Adam scaling can produce nearly
    sign-sized first updates despite very different gradient magnitudes.
    A shared denominator preserves relative magnitudes and direction of the
    first moment before lazy masking and weight decay, not necessarily of
    the current gradient or the final integrated displacement. Sharing across
    components avoids independently normalizing weak components upward.
    Adam-mini motivates blockwise scaling, but does not validate this grouping
    or its robust aggregation for flow fitting (references in README.md).

    The shared statistic is the mean of positive stored second-moment entries
    after capping them at ``shared_second_moment_clip_quantile`` (default
    0.99), estimated from a fixed-stride sample of positive entries. An empty
    positive sample disables the cap. ``None`` or a value outside (0, 1)
    also disables it. Positivity is a proxy for past activity, not an explicit
    ever-touched mask: decayed values may underflow to zero. With lazy updates
    or changing support, the uncapped mean need not equal a global-time EMA
    of the active gradients' mean square. Full per-entry moments remain
    stored; sharing the denominator does not reduce optimizer-state memory.

    After a custom step, ``conditioning_stats[param]`` holds device tensors:
    ``scale`` (the shared bias-corrected denominator per slab, or None),
    ``update_rms`` (RMS over nonzero gradient updates per slab/component,
    excluding weight decay), and ``update_count`` (nonzero scalar update
    entries). Entries retain the last custom-step statistics if both flags
    are subsequently disabled; they are not fresh diagnostics of fused steps.

    State uses AdamW's ``step``, ``exp_avg`` and ``exp_avg_sq`` format so
    flags can be toggled and checkpoints exchanged with plain AdamW.
    Non-custom groups use AdamW's fused step. Unlike SparseAdam, this class
    uses AdamW's epsilon placement and applies configured decoupled weight
    decay to every entry, including those masked out of the gradient step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditioning_stats = {}

    def step(self, closure=None):
        custom = []
        for group in self.param_groups:
            masked = bool(group.get('lazy_moments'))
            shared = bool(group.get('shared_second_moment'))
            if not (masked or shared):
                continue
            if group.get('amsgrad') or group.get('maximize'):
                raise ValueError(
                    'lazy_moments / shared_second_moment do not support amsgrad or maximize')
            for param in group['params']:
                if param.grad is not None:
                    custom.append((group, param, param.grad, masked, shared))
        # Hide the custom groups' gradients from the fused step, which skips
        # parameters without one; every other group is stepped as usual.
        for _, param, _, _, _ in custom:
            param.grad = None
        try:
            loss = super().step(closure)
        finally:
            for _, param, grad, _, _ in custom:
                param.grad = grad
        with torch.no_grad():
            for group, param, grad, masked, shared in custom:
                self._custom_step(group, param, grad, masked, shared)
        return loss

    def _init_lazy_state(self, group, param):
        # Mirror AdamW's lazy state initialisation so a later fused step (the
        # flag switched off) finds exactly the state it would have created.
        state = self.state[param]
        if group.get('fused') or group.get('capturable'):
            state['step'] = torch.zeros((), dtype=torch.float32, device=param.device)
        else:
            dtype = (torch.float64 if torch.get_default_dtype() == torch.float64
                     else torch.float32)
            state['step'] = torch.tensor(0.0, dtype=dtype)
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        return state

    def _lazy_step(self, group, param, grad):
        # Kept for callers of the original entry point: the masked update
        # with per-cell denominators.
        self._custom_step(group, param, grad, masked=True, shared=False)

    @staticmethod
    def _shared_second_moment(exp_avg_sq, quantile):
        """Winsorised mean over positive second-moment entries, per slab."""
        planes = _stage_view(exp_avg_sq)
        stride = _subsample_stride(planes.shape[1])
        shared_sq = torch.zeros(planes.shape[0], dtype=torch.float64, device=planes.device)
        winsorise = quantile is not None and 0.0 < float(quantile) < 1.0
        for index in range(planes.shape[0]):
            row = planes[index]
            cap = None
            if winsorise:
                sub = row[::stride].to(torch.float64)
                touched_sub = torch.where(sub > 0, sub, torch.full_like(sub, math.nan))
                cap = _nan_to_inf(torch.nanquantile(touched_sub, float(quantile))).to(row.dtype)
            total = torch.zeros((), dtype=torch.float64, device=planes.device)
            count = torch.zeros_like(total)
            for start in range(0, row.numel(), _CHUNK_ELEMENTS):
                chunk = row[start:start + _CHUNK_ELEMENTS]
                # Exclude zero moments from the scale estimate. This is a
                # positivity test, not a stored activity mask; previously
                # active moments can also become zero through underflow.
                count += (chunk > 0).sum(dtype=torch.float64)
                if cap is not None:
                    chunk = torch.minimum(chunk, cap)
                total += chunk.sum(dtype=torch.float64)
            shared_sq[index] = total / count.clamp(min=1.0)
        return shared_sq

    def _custom_step(self, group, param, grad, masked, shared):
        state = self.state[param]
        if len(state) == 0:
            state = self._init_lazy_state(group, param)
        state['step'] += 1
        step = float(state['step'])
        beta1, beta2 = group['betas']
        lr = float(group['lr'])
        eps = float(group['eps'])
        weight_decay = float(group['weight_decay'])
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2_sqrt = (1.0 - beta2 ** step) ** 0.5
        step_size = lr / bias_correction1
        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        flat_param = param.view(-1)
        flat_grad = grad.reshape(-1)
        flat_avg = state['exp_avg'].view(-1)
        flat_sq = state['exp_avg_sq'].view(-1)
        numel = flat_param.numel()

        # Moments first (chunked to bound the temporaries), so a shared
        # denominator sees every cell's updated second moment.
        for start in range(0, numel, _CHUNK_ELEMENTS):
            stop = min(start + _CHUNK_ELEMENTS, numel)
            g = flat_grad[start:stop]
            avg = flat_avg[start:stop]
            sq = flat_sq[start:stop]
            if masked:
                touched = g != 0
                # Moments move only where the gradient is nonzero.
                torch.where(touched, torch.lerp(avg, g, 1.0 - beta1), avg, out=avg)
                torch.where(touched, torch.lerp(sq, g * g, 1.0 - beta2), sq, out=sq)
            else:
                avg.lerp_(g, 1.0 - beta1)
                sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        stages = _stage_view(param).shape[0]
        components = _num_components(param)
        cells = numel // (stages * components)

        shared_denom = None
        if shared:
            shared_sq = self._shared_second_moment(
                state['exp_avg_sq'],
                group.get('shared_second_moment_clip_quantile', DEFAULT_CLIP_QUANTILE))
            shared_denom = (shared_sq.sqrt() / bias_correction2_sqrt + eps).to(param.dtype)

        update_sumsq = torch.zeros(stages, components, dtype=torch.float64, device=param.device)
        update_count = torch.zeros_like(update_sumsq)
        for stage in range(stages):
            denom_scalar = None if shared_denom is None else shared_denom[stage]
            for component in range(components):
                base = (stage * components + component) * cells
                for start in range(base, base + cells, _CHUNK_ELEMENTS):
                    stop = min(start + _CHUNK_ELEMENTS, base + cells)
                    g = flat_grad[start:stop]
                    avg = flat_avg[start:stop]
                    sq = flat_sq[start:stop]
                    if denom_scalar is not None:
                        update = (avg / denom_scalar).mul_(step_size)
                    else:
                        # AdamW's update form (epsilon added after the
                        # bias-corrected root, so a fused step on the same
                        # state continues seamlessly; SparseAdam adds it
                        # before, an epsilon-scale difference).
                        denom = (sq.sqrt() / bias_correction2_sqrt).add_(eps)
                        update = (avg / denom).mul_(step_size)
                    if masked:
                        # Masked to the touched entries.
                        update.masked_fill_(g == 0, 0.0)
                    update_sumsq[stage, component] += torch.linalg.vector_norm(
                        update, dtype=torch.float64) ** 2
                    update_count[stage, component] += (update != 0).sum(dtype=torch.float64)
                    flat_param[start:stop].sub_(update)

        self.conditioning_stats[param] = {
            'scale': None if shared_denom is None else shared_denom.to(torch.float64),
            'update_rms': (update_sumsq / update_count.clamp(min=1.0)).sqrt(),
            'update_count': update_count,
        }
