"""Residual capture for a matrix-free Gauss-Newton operator on the flow lattices.

The fitter's losses are sums of penalties of residuals: hinges on radius
deviations, L1 on umbilicus offsets, Huber on shell radii, power means of
point distances. For a loss ``L(r(p))`` the Gauss-Newton operator is

    G v = J^T W J v,     J = dr/dp,

with ``W`` a positive diagonal. Because these penalties are L1-like, their
exact residual-space curvature is zero almost everywhere; the operator used
here is the iteratively-reweighted-least-squares (IRLS) majorizer

    w_i = (dL/dr_i) / r_i     (|r_i| floored),

which is exact Gauss-Newton for squared penalties and the classical
majorization weight for L1, Huber and hinge penalties (Newton's method on
the reweighted quadratic reproduces the IRLS step). It is positive
semidefinite by construction and needs no per-loss formula: the weight is
read off with one small autograd call from the unweighted scalar loss to
its registered residual. For the power-mean reductions (p != 1) the weight
is a positive secant approximation rather than a true majorizer.

Losses opt in with a single call::

    gauss_newton_residuals.register('patch_radius', residual, unweighted_loss)

which is a no-op unless a ``ResidualCapture`` is active. The fitter runs two
passes per product ``G v``:

* ``capture``: forward only, under ``torch.no_grad()``, with the flow
  parameters shifted by ``eps v``; the registered residuals ``r(p + eps v)``
  are stored (they are point-sized, not lattice-sized).
* ``product``: the ordinary forward with gradients on the unshifted
  parameters. In place of each loss family's ``loss.backward()`` the fitter
  calls :meth:`ResidualCapture.family_backward`, which forms
  ``J v ~= (r(p + eps v) - r(p)) / eps`` for the family's registered
  residuals and back-propagates ``W J v`` through them, so the parameter
  gradients accumulate ``J^T W J v``. Loss terms that never register keep
  contributing to the gradient ``g`` only (first order) and are listed as
  omitted.

The two passes replay the same batch (the fitter restores the random state
before each), so registrations line up one to one; any shape mismatch marks
the product as failed rather than mixing residuals.
"""

import contextlib

import torch


_active = None


class ResidualCapture:
    """Registered residuals of one gradient pass. ``mode`` is ``'capture'``
    (store detached residuals) or ``'product'`` (consume ``perturbed``, the
    entries of a capture pass, to back-propagate ``W J v``)."""

    def __init__(self, mode, perturbed=None, epsilon=None, floor=1.0):
        if mode not in ('capture', 'product'):
            raise ValueError(f'unknown capture mode {mode!r}')
        self.mode = mode
        self.perturbed = perturbed
        self.epsilon = None if epsilon is None else float(epsilon)
        self.floor = float(floor)
        self.entries = []      # (name, residual, loss_value)
        self.consumed = 0
        self.used = []         # loss names whose curvature entered the product
        self.omitted = []      # weighted loss names with no registered residual
        self.failed = None     # reason string when the product is unusable

    def register(self, name, residual, loss_value):
        if self.mode == 'capture':
            residual = residual.detach().clone()
        self.entries.append((str(name), residual, loss_value))

    def pending(self):
        """Entries registered since the last family_backward."""
        entries = self.entries[self.consumed:]
        return entries

    def family_backward(self, weighted_losses):
        """Back-propagate ``W J v`` through this family's registered
        residuals (product mode). ``weighted_losses`` maps loss name to the
        weighted scalar the family would otherwise have back-propagated;
        the ratio to the registered unweighted loss recovers the config
        weight. Returns True when a backward ran."""
        if self.mode != 'product':
            raise RuntimeError('family_backward is only valid in product mode')
        entries = self.pending()
        self.consumed = len(self.entries)
        registered = {name for name, _, _ in entries}
        for name, value in weighted_losses.items():
            if name not in registered and torch.is_tensor(value) and value.requires_grad:
                if name not in self.omitted:
                    self.omitted.append(name)
        residuals, upstreams = [], []
        for name, residual, loss_value in entries:
            if name not in weighted_losses:
                # Registered under a name the family does not weight (e.g. a
                # disabled term); nothing to scale it by.
                continue
            if not residual.requires_grad:
                continue
            perturbed = self._perturbed_for(name, residual)
            if perturbed is None:
                return False
            weighted = weighted_losses[name]
            weights = irls_weights(loss_value, residual, self.floor)
            if weights is None:
                continue
            unweighted = loss_value.detach()
            scale = torch.where(unweighted != 0, weighted.detach() / unweighted,
                                torch.zeros_like(unweighted))
            jv = (perturbed - residual.detach()) / self.epsilon
            upstreams.append(weights * jv * scale)
            residuals.append(residual)
            if name not in self.used:
                self.used.append(name)
        if not residuals:
            return False
        torch.autograd.backward(residuals, upstreams)
        return True

    def _perturbed_for(self, name, residual):
        # The capture pass registered the same terms in the same order; take
        # the next perturbed entry of this name.
        position = getattr(self, '_perturbed_position', 0)
        perturbed = self.perturbed or []
        while position < len(perturbed):
            other_name, other_residual, _ = perturbed[position]
            position += 1
            if other_name != name:
                continue
            self._perturbed_position = position
            if other_residual.shape != residual.shape:
                self.failed = (f'residual {name} changed shape between passes: '
                               f'{tuple(other_residual.shape)} vs {tuple(residual.shape)}')
                return None
            return other_residual.to(residual.dtype)
        self._perturbed_position = position
        self.failed = f'residual {name} has no counterpart in the capture pass'
        return None


def register(name, residual, loss_value):
    """Register a loss term's residual tensor and its unweighted scalar loss
    with the active capture; a no-op when none is active."""
    if _active is not None and torch.is_tensor(residual):
        _active.register(name, residual, loss_value)


@contextlib.contextmanager
def active(capture):
    """Make ``capture`` the destination of :func:`register` for the block."""
    global _active
    previous = _active
    _active = capture
    try:
        yield capture
    finally:
        _active = previous


def is_active():
    return _active is not None


def irls_weights(loss_value, residual, floor):
    """``(dL/dr) / r`` per residual entry, with ``|r|`` floored at ``floor``
    and negative values (a penalty locally decreasing in ``|r|``) clamped to
    zero. Returns None when the loss does not depend on the residual."""
    if not (torch.is_tensor(loss_value) and loss_value.requires_grad):
        return None
    (grad,) = torch.autograd.grad(loss_value, residual, retain_graph=True, allow_unused=True)
    if grad is None:
        return None
    r = residual.detach()
    signed_floor = torch.where(r < 0, -r.abs().clamp(min=floor), r.abs().clamp(min=floor))
    return (grad / signed_floor).clamp(min=0.0)
