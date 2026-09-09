"""Sobolev-damped, truncated Hessian-free steps for the flow lattices.

Prototype of SOBOLEV_GAUSS_NEWTON_PLAN.md (the "fast prototype" phase). Each
step solves, approximately and matrix-free,

    (H + lambda A) d = -g

for the flow-lattice parameters, where ``g`` is the current flow gradient,
``A`` is a symmetric positive-definite Sobolev metric on the lattice and
``H`` is a Hessian-vector product. With ``H = 0`` the step is the Sobolev
gradient step ``d = -(1/lambda) A^-1 g``; with curvature the system is solved
by preconditioned conjugate gradients (PCG) with ``A^-1`` as the
preconditioner, stopping at a maximum iteration count, a relative residual
tolerance, non-finite arithmetic or non-positive curvature. The prototype
uses the full Hessian (which may be indefinite, hence the curvature guard),
not the positive-semidefinite generalized Gauss-Newton operator; that is the
production follow-up in the plan.

The fitter's backward runs loss family by loss family through custom
autograd functions with hand-written (Triton) backward kernels and frees
each graph as it goes, so PyTorch second derivatives are not available.
``H v`` is therefore a forward finite difference of the gradient on the same
frozen batch: ``(g(p + eps v) - g(p)) / eps``, with the random state restored
before every re-evaluation (see ``RngSnapshot``). Each PCG iteration costs
one extra forward/backward pass.

Metrics act independently on each leading slab and vector component of a
lattice tensor ``[stages, components, *spatial]``:

* ``CartesianSobolevMetric``: ``A = I + sum_axis l_axis^2 D_axis^T D_axis``,
  first differences along z, y, x with Neumann (dropped-edge) borders.
* ``CylindricalSobolevMetric``: the same energy on a packed cylindrical
  lattice ``[..., nz, total_phi]`` with z edges (Neumann) and cyclic angular
  edges within each ring. There are no radial edges: coupling across rings
  with unequal cell counts needs the overlap weights and basis transport of
  the plan's production section. Ring 0 (the pinned axis) is constrained.

Constrained cells (pinned or influence-masked) are eliminated from every
operator: inputs and outputs are zeroed there, so ``d`` is exactly zero on
them and the free cells see a Dirichlet-zero boundary at the mask.

``GaussianPreconditioner`` lets the existing Gaussian gradient smoother stand
in for ``A^-1``. It is heuristic: its border renormalisation and the unequal
ring radial interpolation have not been shown self-adjoint under the cell
measure, so PCG with it is not a true preconditioned CG. It is offered for an
early cylindrical experiment with across-ring coupling only.

Nothing here has been shown to improve fit quality or wall-clock time; the
module supplies the operators, solver, diagnostics and guards for that
comparison.
"""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

import flow_grad_smoothing


CURVATURE_MODES = ('none', 'finite_difference')
# A CG residual growing by more than this factor in one iteration stops the
# solve (see conjugate_gradient).
RESIDUAL_GROWTH_LIMIT = 2.0
PRECONDITIONERS = ('cg', 'gaussian')


# --------------------------------------------------------------------------
# Block vectors: a "vector" is a list of lattice tensors (one per flow
# parameter), so that the low- and high-resolution lattices of every flow
# stage are solved jointly (one finite-difference gradient evaluation gives
# the Hessian-vector product for all of them at once).

def block_dot(a, b):
    """Sum of elementwise products over all blocks, accumulated in float64."""
    total = None
    for x, y in zip(a, b):
        part = torch.sum(x * y, dtype=torch.float64)
        total = part if total is None else total + part
    return total


def block_norm(a):
    return block_dot(a, a).sqrt()


def _is_finite(scalar):
    return bool(torch.isfinite(scalar))


# --------------------------------------------------------------------------
# Metrics


class LatticeMetric:
    """Symmetric positive-definite operator on one lattice tensor.

    ``constrained`` is a boolean tensor broadcastable to the lattice's
    spatial shape (True = eliminated cell), or None.
    """

    def __init__(self, constrained=None):
        self.constrained = constrained

    def _free(self, v):
        if self.constrained is None:
            return v
        return v.masked_fill(self.constrained, 0.0)

    def differences(self, v):
        """List of first-difference tensors, one per edge family."""
        raise NotImplementedError

    def _laplacian_terms(self, v):
        raise NotImplementedError

    def apply_A(self, v):
        v = self._free(v)
        out = v + self._laplacian_terms(v)
        return self._free(out)

    def quadratic_norm(self, v):
        """``v^T A v`` as a float64 scalar tensor."""
        return torch.sum(v * self.apply_A(v), dtype=torch.float64)

    def roughness(self, v):
        """RMS of first differences over the RMS of the values (float64
        scalar), 0 when ``v`` vanishes. Dimensionless: 0 is a constant
        field, larger is rougher."""
        v = self._free(v)
        sumsq = torch.sum(v * v, dtype=torch.float64)
        edges = None
        count = 0
        for diff in self.differences(v):
            part = torch.sum(diff * diff, dtype=torch.float64)
            edges = part if edges is None else edges + part
            count += diff.numel()
        if edges is None or count == 0:
            return torch.zeros((), dtype=torch.float64, device=v.device)
        rms_values = (sumsq / v.numel()).sqrt()
        rms_edges = (edges / count).sqrt()
        return torch.where(rms_values > 0, rms_edges / rms_values.clamp(min=1e-300),
                           torch.zeros_like(rms_values))

    def solve_A(self, r, *, iterations, tolerance=0.0):
        """Approximate ``A^-1 r`` by ``iterations`` plain CG iterations
        (fewer when the relative residual reaches ``tolerance``).
        Returns ``(x, info)``."""
        r = self._free(r)
        x, info = conjugate_gradient(
            lambda v: [self.apply_A(v[0])], [r],
            max_iterations=iterations, tolerance=tolerance)
        return x[0], info


def _neumann_adjoint_1d(diff, axis):
    # D^T applied to edge values along `axis`: out[i] = diff[i-1] - diff[i]
    # with the missing edges at both ends dropped (Neumann border).
    zeros_shape = list(diff.shape)
    zeros_shape[axis] = 1
    zero = diff.new_zeros(zeros_shape)
    lower = torch.cat([zero, diff], dim=axis)   # diff[i-1] at position i
    upper = torch.cat([diff, zero], dim=axis)   # diff[i] at position i
    return lower - upper


class CartesianSobolevMetric(LatticeMetric):
    """``A = I + lz^2 Dz^T Dz + ly^2 Dy^T Dy + lx^2 Dx^T Dx`` on ``[..., Z, Y, X]``."""

    def __init__(self, lengths_cells, constrained=None):
        super().__init__(constrained)
        lengths = [float(length) for length in lengths_cells]
        if len(lengths) != 3:
            raise ValueError('lengths_cells must hold three per-axis lengths (z, y, x)')
        self.lengths_cells = tuple(lengths)

    def differences(self, v):
        diffs = []
        for axis in (-3, -2, -1):
            if v.shape[axis] < 2:
                continue
            diffs.append(v.narrow(axis, 1, v.shape[axis] - 1)
                         - v.narrow(axis, 0, v.shape[axis] - 1))
        return diffs

    def _laplacian_terms(self, v):
        out = torch.zeros_like(v)
        for axis, length in zip((-3, -2, -1), self.lengths_cells):
            if length <= 0.0 or v.shape[axis] < 2:
                continue
            diff = v.narrow(axis, 1, v.shape[axis] - 1) - v.narrow(axis, 0, v.shape[axis] - 1)
            out.add_(_neumann_adjoint_1d(diff, axis), alpha=length * length)
        return out


def cylindrical_neighbours(ring_num_phi, ring_offsets, device=None):
    """Packed-cell index tables of the next and previous cell around each
    ring (cyclic within the ring; a one-cell ring maps to itself)."""
    num_phi = torch.as_tensor(ring_num_phi, dtype=torch.int64).cpu()
    offsets = torch.as_tensor(ring_offsets, dtype=torch.int64).cpu()
    total = int(num_phi.sum())
    nxt = torch.empty(total, dtype=torch.int64)
    prv = torch.empty(total, dtype=torch.int64)
    for ring in range(num_phi.numel()):
        n = int(num_phi[ring])
        start = int(offsets[ring])
        local = torch.arange(n)
        nxt[start:start + n] = start + (local + 1) % n
        prv[start:start + n] = start + (local - 1) % n
    return nxt.to(device), prv.to(device)


def cylindrical_ring_zero_mask(ring_num_phi, nz, device=None):
    """Boolean ``[nz, total]`` mask that is True on the pinned axis ring 0."""
    num_phi = torch.as_tensor(ring_num_phi, dtype=torch.int64).cpu()
    total = int(num_phi.sum())
    mask = torch.zeros(total, dtype=torch.bool)
    mask[:int(num_phi[0])] = True
    return mask.view(1, total).expand(int(nz), total).contiguous().to(device)


class CylindricalSobolevMetric(LatticeMetric):
    """``A = I + lz^2 Dz^T Dz + lphi^2 Dphi^T Dphi`` on a packed lattice
    ``[..., nz, total_phi]`` (rings end to end along the last axis). Angular
    edges are cyclic within each ring; there are no radial edges. Ring 0 is
    always constrained in addition to ``constrained``."""

    def __init__(self, ring_num_phi, ring_offsets, length_z_cells, length_phi_cells,
                 nz, constrained=None, device=None):
        ring_zero = cylindrical_ring_zero_mask(ring_num_phi, nz, device)
        if constrained is not None:
            constrained = constrained.to(device=ring_zero.device) | ring_zero
        else:
            constrained = ring_zero
        super().__init__(constrained)
        self.length_z_cells = float(length_z_cells)
        self.length_phi_cells = float(length_phi_cells)
        self.next_index, self.prev_index = cylindrical_neighbours(
            ring_num_phi, ring_offsets, device)

    def _phi_diff(self, v):
        # Edge (c -> next(c)) carries v[next(c)] - v[c]; one-cell rings give 0.
        return v.index_select(-1, self.next_index) - v

    def differences(self, v):
        diffs = []
        if v.shape[-2] >= 2:
            diffs.append(v[..., 1:, :] - v[..., :-1, :])
        diffs.append(self._phi_diff(v))
        return diffs

    def _laplacian_terms(self, v):
        out = torch.zeros_like(v)
        if self.length_z_cells > 0.0 and v.shape[-2] >= 2:
            diff = v[..., 1:, :] - v[..., :-1, :]
            out.add_(_neumann_adjoint_1d(diff, -2), alpha=self.length_z_cells ** 2)
        if self.length_phi_cells > 0.0:
            diff = self._phi_diff(v)
            # D^T: cell c receives -diff[c] (it is the tail of its own edge)
            # and +diff[prev(c)] (it is the head of its predecessor's edge).
            adjoint = diff.index_select(-1, self.prev_index) - diff
            out.add_(adjoint, alpha=self.length_phi_cells ** 2)
        return out


class GaussianPreconditioner:
    """HEURISTIC stand-in for ``A^-1``: the existing Gaussian gradient
    smoother applied to the residual. Not shown to be symmetric under the
    lattice cell measure (border renormalisation, unequal-ring radial
    interpolation), so PCG with it is not a true preconditioned CG; the
    Sobolev step ``-(1/lambda) M g`` becomes a smoothed-gradient step.

    ``smoother(tensor)`` smooths a lattice tensor in place.
    """

    def __init__(self, smoother, constrained=None):
        self.smoother = smoother
        self.constrained = constrained

    def __call__(self, r):
        z = r.clone()
        if self.constrained is not None:
            z.masked_fill_(self.constrained, 0.0)
        self.smoother(z)
        if self.constrained is not None:
            z.masked_fill_(self.constrained, 0.0)
        return z


# --------------------------------------------------------------------------
# Solvers


@dataclass
class SolveInfo:
    iterations: int = 0
    residual_initial: float = 0.0
    residual_final: float = 0.0
    reason: str = 'not_run'
    # Sum of the PCG operator evaluations (Hessian-vector products).
    operator_evaluations: int = 0

    def as_dict(self):
        return {
            'iterations': self.iterations,
            'residual_initial': self.residual_initial,
            'residual_final': self.residual_final,
            'reason': self.reason,
            'operator_evaluations': self.operator_evaluations,
            'steihaug': getattr(self, 'steihaug', 0.0),
        }


def conjugate_gradient(operator, rhs, *, max_iterations, tolerance=0.0,
                       preconditioner=None, boundary=None):
    """(Preconditioned) conjugate gradients on block vectors.

    ``operator(v_list)`` returns ``B v`` as a block vector, or None when it
    failed (non-finite arithmetic). ``preconditioner(r_list)`` returns
    ``M^-1 r``. Starts from zero. Stops at the first of: ``max_iterations``;
    ``||r|| <= tolerance * ||r0||``; a non-finite scalar; the residual norm
    growing past ``RESIDUAL_GROWTH_LIMIT`` times the previous iterate's
    (CG's residual norm is not monotone even for a symmetric positive
    operator, but a jump of that size means the operator is inconsistent,
    e.g. a finite-difference secant across a kink) -- the previous iterate
    is kept; non-positive curvature
    ``p^T B p <= 0``. On non-positive curvature the last valid iterate is
    returned, unless ``boundary(x_list, p_list)`` is given: then, as in
    CG-Steihaug, the iterate is moved along ``p`` by the step size that
    callable returns (0 to keep the iterate), the reason stays
    ``nonpositive_curvature`` and ``info.steihaug`` records the move.
    Returns ``(x, SolveInfo)``. The first preconditioned residual
    ``M^-1 rhs`` is kept on the info as ``first_preconditioned`` for the
    caller's fallback step.
    """
    x = [torch.zeros_like(b) for b in rhs]
    r = [b.clone() for b in rhs]
    info = SolveInfo()
    r_norm0 = block_norm(r)
    info.residual_initial = float(r_norm0)
    info.residual_final = info.residual_initial
    info.first_preconditioned = None
    info.steihaug = 0.0
    if not _is_finite(r_norm0):
        info.reason = 'nonfinite_rhs'
        return x, info
    if float(r_norm0) == 0.0 or max_iterations <= 0:
        info.reason = 'zero_rhs' if float(r_norm0) == 0.0 else 'max_iterations'
        return x, info
    z = preconditioner(r) if preconditioner is not None else [t.clone() for t in r]
    info.first_preconditioned = [t.clone() for t in z]
    p = [t.clone() for t in z]
    rz = block_dot(r, z)
    if not _is_finite(rz) or float(rz) <= 0.0:
        info.reason = 'nonfinite' if not _is_finite(rz) else 'indefinite_preconditioner'
        return x, info
    for _ in range(int(max_iterations)):
        Bp = operator(p)
        info.operator_evaluations += 1
        if Bp is None:
            info.reason = 'nonfinite'
            break
        pBp = block_dot(p, Bp)
        if not _is_finite(pBp):
            info.reason = 'nonfinite'
            break
        if float(pBp) <= 0.0:
            info.reason = 'nonpositive_curvature'
            if boundary is not None:
                tau = float(boundary(x, p))
                if math.isfinite(tau) and tau > 0.0:
                    for xi, pi in zip(x, p):
                        xi.add_(pi, alpha=tau)
                    info.steihaug = tau
            break
        alpha = float(rz / pBp)
        for xi, pi in zip(x, p):
            xi.add_(pi, alpha=alpha)
        for ri, Bpi in zip(r, Bp):
            ri.add_(Bpi, alpha=-alpha)
        r_norm = block_norm(r)
        if not _is_finite(r_norm) or float(r_norm) > RESIDUAL_GROWTH_LIMIT * info.residual_final:
            # Roll the iterate back: its residual is not trustworthy (or
            # worse than the previous iterate's).
            for xi, pi in zip(x, p):
                xi.add_(pi, alpha=-alpha)
            info.reason = 'nonfinite' if not _is_finite(r_norm) else 'residual_increased'
            break
        info.iterations += 1
        info.residual_final = float(r_norm)
        if float(r_norm) <= float(tolerance) * float(r_norm0):
            info.reason = 'converged'
            break
        z = preconditioner(r) if preconditioner is not None else [t.clone() for t in r]
        rz_new = block_dot(r, z)
        if not _is_finite(rz_new):
            info.reason = 'nonfinite'
            break
        beta = float(rz_new / rz)
        for pi, zi in zip(p, z):
            pi.mul_(beta).add_(zi)
        rz = rz_new
    else:
        info.reason = 'max_iterations'
    return x, info


def finite_difference_hvp(gradient_fn, base_grads, v, epsilon):
    """Forward-difference Hessian-vector product ``(g(p + eps v) - g(p)) / eps``.

    ``gradient_fn(perturbation_list, scale)`` evaluates the gradient with
    every parameter shifted by ``scale * perturbation`` and returns the
    gradients as a block vector (the caller restores the parameters exactly).
    ``epsilon`` is the absolute perturbation scale. Returns None when the
    result is not finite.
    """
    perturbed = gradient_fn(v, float(epsilon))
    if perturbed is None:
        return None
    out = []
    for gp, g in zip(perturbed, base_grads):
        hv = (gp - g).div_(float(epsilon))
        if not bool(torch.isfinite(hv).all()):
            return None
        out.append(hv)
    return out


# --------------------------------------------------------------------------
# Random state for same-batch re-evaluation


class RngSnapshot:
    """The random state the fitter's samplers draw from (torch CPU and CUDA,
    NumPy's global generator, Python's ``random``), captured to replay one
    batch exactly."""

    def __init__(self):
        self.torch_cpu = torch.get_rng_state()
        self.torch_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.numpy = np.random.get_state()
        self.python = random.getstate()

    def restore(self):
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda)
        np.random.set_state(self.numpy)
        random.setstate(self.python)


# --------------------------------------------------------------------------
# The step


@dataclass
class SobolevSettings:
    enabled: bool = False
    length_voxels: float = 32.0
    damping: float = 1.0
    curvature: str = 'none'
    pcg_iterations: int = 5
    pcg_tolerance: float = 0.1
    inner_cg_iterations: int = 20
    inner_cg_tolerance: float = 1e-3
    step_scale: float = 0.1
    trust_radius: float = 0.0
    max_step_voxels: float = 2.0
    fd_epsilon_voxels: float = 1.0
    diagnostic_interval: int = 0
    adapt_damping: bool = True
    damping_increase: float = 3.0
    damping_decrease: float = 1.5
    damping_max_factor: float = 1e4
    preconditioner: str = 'cg'
    evaluate_step: bool = False

    @classmethod
    def from_config(cls, cfg):
        def read(key, default):
            value = cfg.get(key, default) if hasattr(cfg, 'get') else default
            return default if value is None else value
        settings = cls(
            enabled=bool(read('optimizer_flow_sobolev_gn', False)),
            length_voxels=float(read('optimizer_flow_sobolev_length_voxels', 32.0)),
            damping=float(read('optimizer_flow_sobolev_damping', 1.0)),
            curvature=str(read('optimizer_flow_sobolev_curvature', 'none')),
            pcg_iterations=int(read('optimizer_flow_sobolev_pcg_iterations', 5)),
            pcg_tolerance=float(read('optimizer_flow_sobolev_pcg_tolerance', 0.1)),
            inner_cg_iterations=int(read('optimizer_flow_sobolev_inner_cg_iterations', 20)),
            inner_cg_tolerance=float(read('optimizer_flow_sobolev_inner_cg_tolerance', 1e-3)),
            step_scale=float(read('optimizer_flow_sobolev_step_scale', 0.1)),
            trust_radius=float(read('optimizer_flow_sobolev_trust_radius', 0.0)),
            max_step_voxels=float(read('optimizer_flow_sobolev_max_step_voxels', 2.0)),
            diagnostic_interval=int(read('optimizer_flow_sobolev_diagnostic_interval', 0)),
            adapt_damping=bool(read('optimizer_flow_sobolev_adapt_damping', True)),
            damping_increase=float(read('optimizer_flow_sobolev_damping_increase', 3.0)),
            damping_decrease=float(read('optimizer_flow_sobolev_damping_decrease', 1.5)),
            damping_max_factor=float(read('optimizer_flow_sobolev_damping_max_factor', 1e4)),
            fd_epsilon_voxels=float(read('optimizer_flow_sobolev_fd_epsilon_voxels', 1.0)),
            preconditioner=str(read('optimizer_flow_sobolev_preconditioner', 'cg')),
            evaluate_step=bool(read('optimizer_flow_sobolev_evaluate_step', False)),
        )
        settings.validate()
        return settings

    def validate(self):
        if self.curvature not in CURVATURE_MODES:
            raise ValueError(
                f'optimizer_flow_sobolev_curvature must be one of {CURVATURE_MODES}, '
                f'got {self.curvature!r}')
        if self.preconditioner not in PRECONDITIONERS:
            raise ValueError(
                f'optimizer_flow_sobolev_preconditioner must be one of {PRECONDITIONERS}, '
                f'got {self.preconditioner!r}')
        if not self.damping > 0.0:
            raise ValueError('optimizer_flow_sobolev_damping must be positive')
        if self.curvature == 'finite_difference' and not self.fd_epsilon_voxels > 0.0:
            raise ValueError('optimizer_flow_sobolev_fd_epsilon_voxels must be positive')
        if self.adapt_damping and not (self.damping_increase > 1.0 and self.damping_decrease >= 1.0
                                       and self.damping_max_factor >= 1.0):
            raise ValueError('optimizer_flow_sobolev_damping_increase must exceed 1, '
                             '_decrease and _max_factor must be at least 1')

    @property
    def uses_curvature(self):
        return self.curvature != 'none'


def sobolev_step(grads, metrics, settings, *, hvp=None, preconditioners=None,
                 damping=None):
    """One damped step for the block vector ``grads`` (one lattice gradient
    per flow parameter; ``metrics[i]`` is its ``LatticeMetric``).
    ``damping`` overrides ``settings.damping`` (the fitter's adapted lambda).

    Solves ``(H + lambda A) d = -g`` by PCG when ``hvp`` (a callable on block
    vectors returning ``H v`` or None) is given, otherwise returns the Sobolev
    step ``-(1/lambda) A^-1 g``. ``preconditioners`` optionally overrides the
    per-block ``A^-1`` with callables (see GaussianPreconditioner).

    Falls back to the Sobolev step when PCG stops before completing one
    iteration, and keeps the last valid iterate otherwise. Scales ``d`` down
    to the Sobolev-norm trust radius when one is set. Returns ``(steps,
    stats)``; ``stats`` carries the solver info, timings and per-block step
    RMS/roughness (in lattice units, all as Python floats).
    """
    started = time.perf_counter()
    damping = float(settings.damping if damping is None else damping)
    inner_iterations = 0

    def apply_preconditioner(r):
        nonlocal inner_iterations
        out = []
        for index, (metric, block) in enumerate(zip(metrics, r)):
            if preconditioners is not None and preconditioners[index] is not None:
                out.append(preconditioners[index](block))
                continue
            z, info = metric.solve_A(
                block, iterations=settings.inner_cg_iterations,
                tolerance=settings.inner_cg_tolerance)
            inner_iterations += info.iterations
            out.append(z)
        return out

    rhs = [-g for g in grads]
    stats = {'curvature': 'none' if hvp is None else settings.curvature}
    if hvp is None:
        z = apply_preconditioner(rhs)
        steps = [t.mul_(1.0 / damping) for t in z]
        info = SolveInfo(iterations=0, reason='sobolev_gradient')
        info.residual_initial = float(block_norm(rhs))
        info.residual_final = info.residual_initial
        stats['fallback'] = False
    else:
        def operator(v):
            hv = hvp(v)
            if hv is None:
                return None
            out = []
            for metric, block, hv_block in zip(metrics, v, hv):
                out.append(hv_block + metric.apply_A(block).mul_(damping))
            return out

        def sobolev_sq(v):
            total = None
            for metric, block in zip(metrics, v):
                part = metric.quadratic_norm(block)
                total = part if total is None else total + part
            return float(total) if total is not None else 0.0

        radius_holder = {}

        def boundary(x, p):
            # CG-Steihaug: on a non-positive-curvature direction, walk to the
            # trust boundary. The radius is the configured Sobolev trust
            # radius, else the Sobolev norm of the fallback step
            # -(1/lambda) A^-1 g (the direction is then never trusted further
            # than the Sobolev gradient step would go).
            radius = settings.trust_radius
            if not radius > 0.0:
                radius = radius_holder.get('fallback', 0.0)
            xx = sobolev_sq(x)
            if radius * radius <= xx:
                return 0.0
            # Solve |x + tau p|_A = radius for tau > 0.
            xp = sum(float(torch.sum(xi * metric.apply_A(pi), dtype=torch.float64))
                     for metric, xi, pi in zip(metrics, x, p))
            pp = sobolev_sq(p)
            if not pp > 0.0:
                return 0.0
            disc = xp * xp + pp * (radius * radius - xx)
            return (-xp + math.sqrt(max(disc, 0.0))) / pp

        def preconditioner_recording(r):
            z = apply_preconditioner(r)
            if 'fallback' not in radius_holder:
                # The first call is M^-1 rhs: the fallback step up to 1/lambda.
                radius_holder['fallback'] = math.sqrt(sobolev_sq(z)) / damping
            return z

        steps, info = conjugate_gradient(
            operator, rhs, max_iterations=settings.pcg_iterations,
            tolerance=settings.pcg_tolerance, preconditioner=preconditioner_recording,
            boundary=boundary)
        fallback = info.iterations == 0 and not info.steihaug > 0.0
        if fallback:
            first = getattr(info, 'first_preconditioned', None)
            if first is None:
                first = apply_preconditioner(rhs)
            steps = [t.mul_(1.0 / damping) for t in first]
        stats['fallback'] = fallback

    for step in steps:
        if not bool(torch.isfinite(step).all()):
            # The step itself is unusable; a zero step is the only safe one.
            for t in steps:
                t.zero_()
            info.reason = 'nonfinite_step'
            break

    # Sobolev norm of the step, and the trust-region scaling.
    sobolev_sq = None
    for metric, step in zip(metrics, steps):
        part = metric.quadratic_norm(step)
        sobolev_sq = part if sobolev_sq is None else sobolev_sq + part
    sobolev_norm = float(sobolev_sq.sqrt()) if sobolev_sq is not None else 0.0
    trust_scale = 1.0
    if settings.trust_radius > 0.0 and sobolev_norm > settings.trust_radius:
        trust_scale = settings.trust_radius / sobolev_norm
        for step in steps:
            step.mul_(trust_scale)
    stats.update(info.as_dict())
    stats['inner_cg_iterations'] = inner_iterations
    stats['sobolev_norm'] = sobolev_norm
    stats['trust_scale'] = trust_scale
    stats['damping'] = damping
    stats['blocks'] = []
    for metric, step in zip(metrics, steps):
        rms = float((torch.sum(step * step, dtype=torch.float64) / step.numel()).sqrt())
        stats['blocks'].append({
            'step_rms': rms,
            'roughness': float(metric.roughness(step)),
            'component_rms': [
                float((torch.sum(c * c, dtype=torch.float64) / c.numel()).sqrt())
                for c in step.unbind(1)
            ] if step.dim() >= 3 else [rms],
        })
    stats['solve_seconds'] = time.perf_counter() - started
    return steps, stats


def predicted_reduction(grads, steps, hessian_steps=None):
    """``-(g^T d + 0.5 d^T H d)`` of the quadratic model for the applied step
    (linear model when ``hessian_steps`` is None), as a float."""
    linear = block_dot(grads, steps)
    total = -linear
    if hessian_steps is not None:
        total = total - 0.5 * block_dot(steps, hessian_steps)
    return float(total)


def symmetry_defect(hvp, u, v):
    """Relative asymmetry of a Hessian-vector product on two directions:
    ``|u.Hv - v.Hu| / max(|u.Hv|, |v.Hu|)``; 0 for a symmetric operator, of
    order 1 for an inconsistent one. Returns ``(defect, u_Hv, v_Hu)``, or
    None when a product failed."""
    hv = hvp(v)
    if hv is None:
        return None
    hu = hvp(u)
    if hu is None:
        return None
    u_hv = float(block_dot(u, hv))
    v_hu = float(block_dot(v, hu))
    scale = max(abs(u_hv), abs(v_hu), 1e-300)
    return abs(u_hv - v_hu) / scale, u_hv, v_hu


# Solver outcomes that mean the quadratic model was not trustworthy.
POOR_SOLVE_REASONS = ('nonpositive_curvature', 'residual_increased', 'nonfinite',
                      'nonfinite_step', 'indefinite_preconditioner')


def adapt_damping(damping, reason, settings):
    """Next step's lambda from this step's solver outcome (cross-step
    Levenberg-Marquardt without retries): multiply on a poor solve, divide
    on a clean one, within [settings.damping, settings.damping *
    damping_max_factor]. Returns ``damping`` unchanged when adaptation is off
    or curvature is not used."""
    if not settings.adapt_damping or not settings.uses_curvature:
        return float(damping)
    if reason in POOR_SOLVE_REASONS:
        damping = damping * settings.damping_increase
    elif reason in ('converged', 'max_iterations'):
        damping = damping / settings.damping_decrease
    low = settings.damping
    high = settings.damping * settings.damping_max_factor
    return float(min(max(damping, low), high))


@torch.no_grad()
def apply_steps_(params, steps, step_scale):
    """``param += step_scale * step`` for every block."""
    for param, step in zip(params, steps):
        param.add_(step, alpha=float(step_scale))


# --------------------------------------------------------------------------
# Factories for the fitter's flow fields


def is_cylindrical(flow_field):
    return hasattr(flow_field, '_hr_num_phi')


def lattice_length_cells(length_voxels, cell_voxels, spatial_scale):
    """Physical length in scroll voxels to cells of a lattice whose cells are
    ``cell_voxels * spatial_scale`` wide."""
    return float(length_voxels) / (float(cell_voxels) * float(spatial_scale))


def metric_for_flow_field(flow_field, level, length_cells, constrained=None):
    """The Sobolev metric for lattice ``level`` (0 = low-res, 1 = high-res)
    of a flow field, with isotropic length ``length_cells`` in that
    lattice's cells."""
    param = flow_field.flows[level]
    if is_cylindrical(flow_field):
        tables = ((flow_field._lr_num_phi, flow_field._lr_offsets),
                  (flow_field._hr_num_phi, flow_field._hr_offsets))
        num_phi, offsets = tables[level]
        return CylindricalSobolevMetric(
            num_phi, offsets, length_cells, length_cells, nz=param.shape[-2],
            constrained=constrained, device=param.device)
    return CartesianSobolevMetric((length_cells,) * 3, constrained=constrained)


def gaussian_preconditioner_for_flow_field(flow_field, level, sigma_cells,
                                           across_sigma_cells=0.0, constrained=None):
    """HEURISTIC ``A^-1`` from the existing smoother (see GaussianPreconditioner).
    Widths are in the lattice's own cells."""
    if is_cylindrical(flow_field):
        tables = ((flow_field._lr_num_phi, flow_field._lr_offsets),
                  (flow_field._hr_num_phi, flow_field._hr_offsets))
        num_phi, offsets = tables[level]

        def smoother(tensor):
            flow_grad_smoothing.smooth_cylindrical_(
                tensor, num_phi, offsets, sigma_cells, across_sigma_cells)
    else:
        def smoother(tensor):
            flow_grad_smoothing.smooth_cartesian_(tensor, sigma_cells)
    if constrained is None and is_cylindrical(flow_field):
        param = flow_field.flows[level]
        constrained = cylindrical_ring_zero_mask(
            flow_field._lr_num_phi if level == 0 else flow_field._hr_num_phi,
            param.shape[-2], param.device)
    return GaussianPreconditioner(smoother, constrained)


def describe_settings(settings, cell_voxels, spatial_scale_factor, field_type):
    """One-line startup report of the effective operator per lattice."""
    parts = []
    for name, scale in (('HR', 1), ('LR', spatial_scale_factor)):
        cells = lattice_length_cells(settings.length_voxels, cell_voxels, scale)
        parts.append(f'{name} {cells:.2f} cells')
    report = (f'flow Sobolev step ({field_type}): length {settings.length_voxels:g} voxels = '
              + ', '.join(parts)
              + f'; damping {settings.damping:g}, step scale {settings.step_scale:g}, '
              f'curvature {settings.curvature}')
    if settings.uses_curvature:
        report += (f' (PCG <= {settings.pcg_iterations} iterations, tolerance '
                   f'{settings.pcg_tolerance:g}, finite-difference epsilon '
                   f'{settings.fd_epsilon_voxels:g} voxels)')
    if settings.max_step_voxels > 0.0:
        report += f'; step capped at {settings.max_step_voxels:g} voxels RMS per component'
    if settings.uses_curvature and settings.adapt_damping:
        report += (f'; damping adapts x{settings.damping_increase:g} on poor solves, '
                   f'/{settings.damping_decrease:g} on clean ones')
    if settings.preconditioner == 'gaussian':
        report += ('; preconditioner: Gaussian smoother (HEURISTIC, not shown '
                   'self-adjoint)')
    else:
        report += f'; preconditioner: {settings.inner_cg_iterations}-iteration inner CG on A'
    if field_type == 'cylindrical':
        report += '; no radial edges in A (rings coupled only via a Gaussian preconditioner)'
    return report
