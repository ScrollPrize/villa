"""AdamW with lazy (SparseAdam-style) moments and a shared, robust second
moment for selected parameter groups, plus robust gradient clipping for the
flow lattices.

Every statistic here (median |g| for clipping, the winsorisation cap of the
shared second moment) is read from a fixed-stride subsample of the plane, so
it is deterministic, identical on every DDP rank, and costs the same at any
lattice size.
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
    median nonzero |g| of that stage.

    The median is read from a fixed-stride subsample of the stage's cells
    (see STATS_SUBSAMPLE), without any host synchronisation. Returns
    ``(threshold, clipped_fraction)``, two float64 tensors of length
    ``stages`` on ``grad.device`` (threshold is +inf for a stage with no
    nonzero gradient), or ``None`` when ``multiple`` is not positive, in
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
    """AdamW whose ``lazy_moments`` groups update only where the gradient is nonzero.

    torch.optim.SparseAdam masks the Adam update to the entries a sparse
    gradient carries: moments and parameters change only there, and untouched
    entries keep their moments instead of decaying the second moment toward
    zero (which otherwise makes an entry's first update after a quiet spell
    disproportionately large). SparseAdam itself requires sparse-layout
    gradients and its own optimizer instance; the flow lattices' gradients
    here are dense accumulators that are mostly zero, so this class applies
    the same masked update to dense gradients, for the groups flagged
    ``lazy_moments=True``, and leaves every other group to AdamW's fused step.

    Groups flagged ``shared_second_moment=True`` (with or without lazy
    moments) divide by one second moment per *stage* of the parameter (the
    leading axis of a lattice, see _stage_view; every vector component of a
    stage shares it) instead of one per cell. Adam's per-cell denominator
    turns a spatially smooth gradient into (nearly) its sign field; the
    shared denominator keeps the gradient's spatial profile in the update, so
    the flow gradient smoothing (flow_grad_smoothing) shapes the step and not
    just its sign. Sharing across components as well keeps the update's
    direction that of the gradient: nearly all supervision pushes radially,
    and a per-component scale would inflate the weakly constrained z and
    tangential components to the same step as the radial one.

    The shared value is a winsorised mean of the stored per-cell second
    moments over the cells ever touched: the plane is capped at the
    ``shared_second_moment_clip_quantile`` quantile (default 0.99, read from a
    fixed-stride subsample) of those cells before averaging, so a handful of
    cells with persistently huge gradients cannot shrink every other cell's
    step. Set the group's quantile to ``None`` (or >= 1) for the plain mean,
    which is exactly the EMA of the mean squared gradient over the touched
    cells. The stored second moment stays per cell.

    After each step ``conditioning_stats[param]`` holds, for every custom
    group's parameter: ``scale`` (the shared bias-corrected denominator per
    stage, or None), ``update_rms`` (root mean square of the nonzero update
    per stage and component, in parameter units) and ``update_count``
    (cells updated per stage and component), all as device tensors so
    reading them is a caller-chosen synchronisation.

    State is kept in AdamW's own format (``step``, ``exp_avg``,
    ``exp_avg_sq``), so checkpoints round-trip with a plain AdamW and both
    flags can be switched on or off between runs. Decoupled weight decay
    still applies to every entry of a lazy group (SparseAdam has none).
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
        """Winsorised mean of the second moment over touched cells, per stage."""
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
                # An untouched cell's second moment is exactly zero (lazy) or
                # has decayed from zero (plain); either way it says nothing
                # about the gradient scale, so only touched cells count.
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
