from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _clamped_adam_kernel(
	param,
	grad,
	exp_avg,
	exp_avg_sq,
	n_vectors,
	beta1,
	one_minus_beta1,
	beta2,
	one_minus_beta2,
	step_size,
	inv_bias_correction2_sqrt,
	eps,
	max_step,
	BLOCK_SIZE: tl.constexpr,
):
	off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = off < n_vectors
	off_other = off + n_vectors

	g0 = tl.load(grad + off, mask=mask)
	g1 = tl.load(grad + off_other, mask=mask)
	m0_old = tl.load(exp_avg + off, mask=mask)
	m1_old = tl.load(exp_avg + off_other, mask=mask)
	v0_old = tl.load(exp_avg_sq + off, mask=mask)
	v1_old = tl.load(exp_avg_sq + off_other, mask=mask)

	# Match Adam's lerp first-moment update and mul/addcmul second moment.
	m0 = m0_old + (g0 - m0_old) * one_minus_beta1
	m1 = m1_old + (g1 - m1_old) * one_minus_beta1
	v0 = beta2 * v0_old + one_minus_beta2 * g0 * g0
	v1 = beta2 * v1_old + one_minus_beta2 * g1 * g1

	denom0 = tl.sqrt(v0) * inv_bias_correction2_sqrt + eps
	denom1 = tl.sqrt(v1) * inv_bias_correction2_sqrt + eps
	update0 = -step_size * m0 / denom0
	update1 = -step_size * m1 / denom1
	p0 = tl.load(param + off, mask=mask)
	p1 = tl.load(param + off_other, mask=mask)
	# Reproduce the reference path's float32 parameter write followed by
	# delta=p_after-p_before before calculating the vector cap.
	candidate0 = p0 + update0
	candidate1 = p1 + update1
	delta0 = candidate0 - p0
	delta1 = candidate1 - p1
	delta_norm = tl.sqrt(delta0 * delta0 + delta1 * delta1)
	update_scale = tl.minimum(1.0, max_step / tl.maximum(delta_norm, 1.0e-12))
	tl.store(param + off, p0 + delta0 * update_scale, mask=mask)
	tl.store(param + off_other, p1 + delta1 * update_scale, mask=mask)
	tl.store(exp_avg + off, m0, mask=mask)
	tl.store(exp_avg + off_other, m1, mask=mask)
	tl.store(exp_avg_sq + off, v0, mask=mask)
	tl.store(exp_avg_sq + off_other, v1, mask=mask)


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
	if param.device.type != "cuda":
		raise ValueError("Triton clamped Adam requires CUDA tensors")
	if param.dtype != torch.float32:
		raise ValueError(f"Triton clamped Adam requires float32 parameters, got {param.dtype}")
	if not all(t.is_contiguous() for t in (param, grad, exp_avg, exp_avg_sq)):
		raise ValueError("Triton clamped Adam requires contiguous tensors")
	if not (param.shape == grad.shape == exp_avg.shape == exp_avg_sq.shape):
		raise ValueError("Triton clamped Adam tensor shapes must match")
	if param.ndim < 1 or int(param.shape[0]) != 2:
		raise ValueError(f"Triton clamped Adam requires leading UV dimension 2, got {tuple(param.shape)}")

	n_vectors = int(param.numel()) // 2
	if n_vectors <= 0:
		return
	block_size = 256
	grid = (triton.cdiv(n_vectors, block_size),)
	_clamped_adam_kernel[grid](
		param,
		grad,
		exp_avg,
		exp_avg_sq,
		n_vectors,
		float(beta1),
		1.0 - float(beta1),
		float(beta2),
		1.0 - float(beta2),
		float(step_size),
		1.0 / float(bias_correction2_sqrt),
		float(eps),
		float(max_step),
		BLOCK_SIZE=block_size,
		num_warps=4,
	)
