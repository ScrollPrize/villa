"""AdamW with lazy (SparseAdam-style) moments for selected parameter groups."""

import torch


# Elements per chunk of the masked update, bounding its temporaries to a
# fraction of the lattice (the full-scroll high-resolution flow lattice is
# ~400M values per slab).
_CHUNK_ELEMENTS = 1 << 26


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

    State is kept in AdamW's own format (``step``, ``exp_avg``,
    ``exp_avg_sq``), so checkpoints round-trip with a plain AdamW and the flag
    can be switched on or off between runs. Decoupled weight decay still
    applies to every entry of a lazy group (SparseAdam has none).
    """

    def step(self, closure=None):
        lazy = []
        for group in self.param_groups:
            if not group.get('lazy_moments'):
                continue
            if group.get('amsgrad') or group.get('maximize'):
                raise ValueError('lazy_moments does not support amsgrad or maximize')
            for param in group['params']:
                if param.grad is not None:
                    lazy.append((group, param, param.grad))
        # Hide the lazy groups' gradients from the fused step, which skips
        # parameters without one; every other group is stepped as usual.
        for _, param, _ in lazy:
            param.grad = None
        try:
            loss = super().step(closure)
        finally:
            for _, param, grad in lazy:
                param.grad = grad
        with torch.no_grad():
            for group, param, grad in lazy:
                self._lazy_step(group, param, grad)
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
        for start in range(0, flat_param.numel(), _CHUNK_ELEMENTS):
            stop = min(start + _CHUNK_ELEMENTS, flat_param.numel())
            g = flat_grad[start:stop]
            avg = flat_avg[start:stop]
            sq = flat_sq[start:stop]
            touched = g != 0
            # Moments move only where the gradient is nonzero.
            torch.where(touched, torch.lerp(avg, g, 1.0 - beta1), avg, out=avg)
            torch.where(touched, torch.lerp(sq, g * g, 1.0 - beta2), sq, out=sq)
            # AdamW's update form (epsilon added after the bias-corrected root,
            # so a fused step on the same state continues seamlessly; SparseAdam
            # adds it before, an epsilon-scale difference), masked to the
            # touched entries.
            denom = (sq.sqrt() / bias_correction2_sqrt).add_(eps)
            update = (avg / denom).mul_(step_size)
            update.masked_fill_(~touched, 0.0)
            flat_param[start:stop].sub_(update)
