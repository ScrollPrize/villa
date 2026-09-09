"""AdamW with lazy (SparseAdam-style) moments for selected parameter groups."""

import torch


# Elements per chunk of the masked update, bounding its temporaries to a
# fraction of the lattice (the full-scroll high-resolution flow lattice is
# ~400M values per slab).
_CHUNK_ELEMENTS = 1 << 26


def _plane_view(tensor):
    """``tensor`` as [planes, cells]: one plane per (leading, second) index of a
    lattice-shaped tensor ([stages, components, *spatial]), or a single plane
    for anything with fewer than three dimensions."""
    if tensor.dim() >= 3:
        return tensor.view(tensor.shape[0] * tensor.shape[1], -1)
    return tensor.view(1, -1)


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
    moments) divide by one second moment per plane of the parameter (per
    flow stage and vector component of a lattice, see _plane_view) instead of
    one per cell: the mean of the stored per-cell second moments over the
    cells that have ever been touched. Adam's per-cell denominator turns a
    spatially smooth gradient into (nearly) its sign field; the shared
    denominator keeps the gradient's spatial profile in the update, so the
    flow gradient smoothing (flow_grad_smoothing) shapes the step and not
    just its sign. The stored second moment stays per cell, and its EMA is
    linear, so the shared value is exactly the EMA of the mean squared
    gradient over those cells.

    State is kept in AdamW's own format (``step``, ``exp_avg``,
    ``exp_avg_sq``), so checkpoints round-trip with a plain AdamW and both
    flags can be switched on or off between runs. Decoupled weight decay
    still applies to every entry of a lazy group (SparseAdam has none).
    """

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

        if shared:
            planes = _plane_view(state['exp_avg_sq'])
            # Mean over the cells ever touched: an untouched cell's second
            # moment is exactly zero (lazy) or has decayed from zero (plain),
            # and either way it says nothing about the gradient scale.
            touched_count = torch.zeros(planes.shape[0], dtype=torch.float64, device=planes.device)
            total = torch.zeros_like(touched_count)
            for index in range(planes.shape[0]):
                row = planes[index]
                total[index] = row.sum(dtype=torch.float64)
                touched_count[index] = (row > 0).sum(dtype=torch.float64)
            shared_sq = total / touched_count.clamp(min=1.0)
            shared_denom = (shared_sq.sqrt() / bias_correction2_sqrt + eps).to(param.dtype)
            plane_cells = planes.shape[1]
        else:
            shared_denom = None
            plane_cells = numel

        for plane_start in range(0, numel, plane_cells):
            denom_scalar = None
            if shared_denom is not None:
                denom_scalar = shared_denom[plane_start // plane_cells]
            for start in range(plane_start, plane_start + plane_cells, _CHUNK_ELEMENTS):
                stop = min(start + _CHUNK_ELEMENTS, plane_start + plane_cells)
                g = flat_grad[start:stop]
                avg = flat_avg[start:stop]
                sq = flat_sq[start:stop]
                if denom_scalar is not None:
                    update = (avg / denom_scalar).mul_(step_size)
                else:
                    # AdamW's update form (epsilon added after the bias-corrected
                    # root, so a fused step on the same state continues
                    # seamlessly; SparseAdam adds it before, an epsilon-scale
                    # difference).
                    denom = (sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    update = (avg / denom).mul_(step_size)
                if masked:
                    # Masked to the touched entries.
                    update.masked_fill_(g == 0, 0.0)
                flat_param[start:stop].sub_(update)
