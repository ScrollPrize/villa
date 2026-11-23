"""Metal fused clamped-Adam step for the flatten UV map on MPS.

One dispatch per parameter replaces the dozen elementwise MPS kernels of ``FlattenClampedAdam._torch_step``,
which on MPS cost more in launches than in arithmetic. The arithmetic follows the PyTorch reference in the same
float32 order (division by the bias correction, the parameter write before the delta, the UV-vector cap), so
the step stays within float32 rounding of it. Like the Triton kernel, it needs contiguous float32 tensors with
a leading UV dimension of 2. Compiled once per process with ``torch.mps.compile_shader`` and encoded on
PyTorch's current MPS stream.
"""
from __future__ import annotations

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// s: beta1, 1 - beta1, beta2, 1 - beta2, step_size, bias_correction2_sqrt, eps, max_step; the complements come
// from the host in double, since 1.0f - 0.9f rounds to a different float32 than 1.0 - 0.9
kernel void clamped_adam(device float* param [[buffer(0)]],
                         device const float* grad [[buffer(1)]],
                         device float* exp_avg [[buffer(2)]],
                         device float* exp_avg_sq [[buffer(3)]],
                         constant float* s [[buffer(4)]],
                         constant long& n_vectors [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
    if (long(i) >= n_vectors) return;
    ulong a = i, b = ulong(i) + ulong(n_vectors);
    float w1 = s[1], beta2 = s[2], w2 = s[3], step_size = s[4], bc2 = s[5], eps = s[6], max_step = s[7];
    float g0 = grad[a], g1 = grad[b];
    // lerp as torch: m + w1 * (g - m)
    float m0 = exp_avg[a] + w1 * (g0 - exp_avg[a]);
    float m1 = exp_avg[b] + w1 * (g1 - exp_avg[b]);
    float v0 = exp_avg_sq[a] * beta2 + w2 * (g0 * g0);
    float v1 = exp_avg_sq[b] * beta2 + w2 * (g1 * g1);
    float u0 = precise::divide(m0, precise::divide(precise::sqrt(v0), bc2) + eps) * -step_size;
    float u1 = precise::divide(m1, precise::divide(precise::sqrt(v1), bc2) + eps) * -step_size;
    float p0 = param[a], p1 = param[b];
    // the reference writes p + u in float32 first, so the capped delta is measured after that rounding
    float d0 = (p0 + u0) - p0, d1 = (p1 + u1) - p1;
    float norm = precise::sqrt(d0 * d0 + d1 * d1);
    float scale = min(precise::divide(max_step, max(norm, 1.0e-12f)), 1.0f);
    param[a] = p0 + d0 * scale;
    param[b] = p1 + d1 * scale;
    exp_avg[a] = m0;
    exp_avg[b] = m1;
    exp_avg_sq[a] = v0;
    exp_avg_sq[b] = v1;
}
"""

_lib = None


def _library():
	global _lib
	if _lib is None:
		_lib = torch.mps.compile_shader(_SOURCE)
	return _lib


def clamped_adam_step(
	param: torch.Tensor,
	grad: torch.Tensor,
	exp_avg: torch.Tensor,
	exp_avg_sq: torch.Tensor,
	*,
	beta1: float,
	beta2: float,
	step_size: float,
	bias_correction2_sqrt: float,
	eps: float,
	max_step: float,
) -> None:
	"""In-place clamped Adam step on MPS tensors; same contract as the Triton ``clamped_adam_step``."""
	tensors = (param, grad, exp_avg, exp_avg_sq)
	if param.device.type != "mps":
		raise ValueError("Metal clamped Adam requires MPS tensors")
	if any(t.dtype != torch.float32 for t in tensors):
		raise ValueError(f"Metal clamped Adam requires float32 tensors, got {param.dtype}")
	if not all(t.is_contiguous() for t in tensors):
		raise ValueError("Metal clamped Adam requires contiguous tensors")
	if not (param.shape == grad.shape == exp_avg.shape == exp_avg_sq.shape):
		raise ValueError("Metal clamped Adam tensor shapes must match")
	if param.ndim < 1 or int(param.shape[0]) != 2:
		raise ValueError(f"Metal clamped Adam requires leading UV dimension 2, got {tuple(param.shape)}")

	n_vectors = int(param.numel()) // 2
	if n_vectors <= 0:
		return
	scalars = [
		float(beta1), 1.0 - float(beta1), float(beta2), 1.0 - float(beta2),
		float(step_size), float(bias_correction2_sqrt), float(eps), float(max_step),
	]
	_library().clamped_adam(param, grad, exp_avg, exp_avg_sq, scalars, n_vectors,
							threads=n_vectors, group_size=min(256, n_vectors))
