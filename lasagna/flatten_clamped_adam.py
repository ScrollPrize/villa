from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class FlattenClampedAdam(torch.optim.Adam):
	"""Adam with the flatten UV-vector update cap applied before the parameter write."""

	def __init__(
		self,
		params: Iterable[torch.Tensor] | Iterable[dict],
		*,
		base_step: float,
		lr: float = 1.0e-3,
		betas: tuple[float, float] = (0.9, 0.999),
		eps: float = 1.0e-8,
	) -> None:
		super().__init__(
			params,
			lr=lr,
			betas=betas,
			eps=eps,
			weight_decay=0.0,
			amsgrad=False,
			foreach=False,
			maximize=False,
			capturable=False,
			differentiable=False,
			fused=False,
		)
		self.base_step = float(base_step)
		self._triton_disabled_reason: str | None = None
		self.last_backend = "none"

	@staticmethod
	def _init_state(state: dict, param: torch.Tensor) -> None:
		if state:
			return
		state["step"] = torch.tensor(0.0, dtype=torch.float32)
		state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
		state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

	@staticmethod
	def _torch_step(
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
		exp_avg.lerp_(grad, 1.0 - beta1)
		exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
		denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
		update = exp_avg.div(denom).mul_(-step_size)
		candidate = param + update
		delta = candidate - param
		if update.ndim >= 1 and int(update.shape[0]) == 2:
			norm = torch.linalg.vector_norm(delta, dim=0, keepdim=True)
			scale = (float(max_step) / norm.clamp_min(1.0e-12)).clamp_max_(1.0)
			param.add_(delta * scale)
		else:
			param.add_(delta.clamp_(min=-float(max_step), max=float(max_step)))

	def _try_triton_step(self, *args, **kwargs) -> bool:
		param = args[0]
		if param.device.type != "cuda" or self._triton_disabled_reason is not None:
			return False
		try:
			from flatten_clamped_adam_triton import clamped_adam_step

			clamped_adam_step(*args, **kwargs)
		except Exception as exc:
			self._triton_disabled_reason = f"{type(exc).__name__}: {exc}"
			print(
				f"[flatten_clamped_adam] Triton disabled after failure: {self._triton_disabled_reason}",
				flush=True,
			)
			return False
		return True

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		used_triton = False
		used_torch = False
		for group in self.param_groups:
			beta1, beta2 = (float(group["betas"][0]), float(group["betas"][1]))
			lr = float(group["lr"])
			eps = float(group["eps"])
			scale_i = int(group.get("_flatten_scale_i", 0))
			max_step = self.base_step * (2.0 ** scale_i)
			if max_step <= 0.0:
				raise ValueError("FlattenClampedAdam base_step must be positive")
			if (
				float(group.get("weight_decay", 0.0)) != 0.0
				or bool(group.get("amsgrad", False))
				or bool(group.get("maximize", False))
				or bool(group.get("capturable", False))
				or bool(group.get("differentiable", False))
			):
				raise ValueError("FlattenClampedAdam supports standard Adam without weight decay or AMSGrad")

			for param in group["params"]:
				grad = param.grad
				if grad is None:
					continue
				if grad.is_sparse:
					raise RuntimeError("FlattenClampedAdam does not support sparse gradients")
				state = self.state[param]
				self._init_state(state, param)
				state["step"].add_(1.0)
				step = float(state["step"].item())
				bias_correction1 = 1.0 - beta1 ** step
				bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
				kwargs = {
					"beta1": beta1,
					"beta2": beta2,
					"step_size": lr / bias_correction1,
					"bias_correction2_sqrt": bias_correction2_sqrt,
					"eps": eps,
					"max_step": max_step,
				}
				args = (param, grad, state["exp_avg"], state["exp_avg_sq"])
				if self._try_triton_step(*args, **kwargs):
					used_triton = True
				else:
					self._torch_step(*args, **kwargs)
					used_torch = True
		self.last_backend = "triton" if used_triton and not used_torch else "torch"
		return loss
