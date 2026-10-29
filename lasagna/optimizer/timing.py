from __future__ import annotations

import os

import torch

from .config import _truthy


def debug_cuda_sync(label: str) -> None:
	if os.environ.get("LASAGNA_SYNC_DEBUG", "0") == "0":
		return
	if torch.cuda.is_available():
		try:
			torch.cuda.synchronize()
		except RuntimeError as exc:
			raise RuntimeError(f"CUDA failure after {label}") from exc


def _cuda_mem_debug_enabled() -> bool:
	return os.environ.get("LASAGNA_CUDA_MEM_DEBUG", "0") != "0"


def _fmt_mib(value: int | float) -> str:
	return f"{float(value) / 1024.0 ** 2:.1f}MiB"


def log_cuda_memory(label: str, *, device: torch.device | int | None = None) -> None:
	if not _cuda_mem_debug_enabled() or not torch.cuda.is_available():
		return
	try:
		dev = device if device is not None else torch.cuda.current_device()
		allocated = torch.cuda.memory_allocated(dev)
		reserved = torch.cuda.memory_reserved(dev)
		max_allocated = torch.cuda.max_memory_allocated(dev)
		max_reserved = torch.cuda.max_memory_reserved(dev)
		free, total = torch.cuda.mem_get_info(dev)
		print(
			f"[cuda_mem] {label}: "
			f"alloc={_fmt_mib(allocated)} reserved={_fmt_mib(reserved)} "
			f"peak_alloc={_fmt_mib(max_allocated)} peak_reserved={_fmt_mib(max_reserved)} "
			f"free={_fmt_mib(free)} total={_fmt_mib(total)}",
			flush=True,
		)
	except RuntimeError as exc:
		print(f"[cuda_mem] {label}: unavailable ({exc})", flush=True)


def format_duration(seconds: float) -> str:
	seconds = max(0.0, float(seconds))
	if seconds < 60.0:
		return f"{seconds:.2f}s"
	minutes, sec = divmod(seconds, 60.0)
	if minutes < 60.0:
		return f"{int(minutes)}m{sec:05.2f}s"
	hours, minutes = divmod(minutes, 60.0)
	return f"{int(hours)}h{int(minutes):02d}m{sec:05.2f}s"


def synchronize_cuda() -> None:
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def flow_timing_enabled(cfg) -> bool:
	if _truthy(os.environ.get("LASAGNA_FLOW_TIMING")):
		return True
	if not isinstance(cfg, dict):
		return False
	return _truthy(cfg.get("profile_cuda_timing", False))


def opt_timing_enabled(stage_args_: dict) -> bool:
	if _truthy(os.environ.get("LASAGNA_OPT_TIMING")):
		return True
	if not isinstance(stage_args_, dict):
		return False
	return _truthy(stage_args_.get("profile_opt", False))


def opt_timing_interval(stage_args_: dict, *, fallback: int) -> int:
	raw = os.environ.get("LASAGNA_OPT_TIMING_INTERVAL", None)
	if raw is None and isinstance(stage_args_, dict):
		raw = stage_args_.get("profile_interval", stage_args_.get("profile_opt_interval", None))
	if raw is None:
		raw = fallback
	return max(1, int(raw))


def opt_timing_sync_cuda(stage_args_: dict) -> bool:
	raw = os.environ.get("LASAGNA_OPT_TIMING_SYNC_CUDA", None)
	if raw is None and isinstance(stage_args_, dict):
		raw = stage_args_.get("profile_cuda_sync", True)
	return _truthy(raw)


class FlowTimingWindow:
	def __init__(self, *, interval: int = 100) -> None:
		self.interval = max(1, int(interval))
		self.count = 0
		self.acc = {
			"total": 0.0,
			"io_prefetch": 0.0,
			"flow_sampling": 0.0,
			"flow_calc": 0.0,
			"opt_step": 0.0,
			"model_forward": 0.0,
			"loss_eval": 0.0,
		}

	def add(self, key: str, seconds: float) -> None:
		self.acc[key] = self.acc.get(key, 0.0) + max(0.0, float(seconds))

	def finish_iter(self, *, label: str, step1: int, max_steps: int) -> None:
		self.count += 1
		if (step1 % self.interval) != 0 and step1 != max_steps:
			return
		if self.count <= 0:
			return
		total = max(1.0e-12, self.acc.get("total", 0.0))
		io_prefetch = self.acc.get("io_prefetch", 0.0)
		flow_sampling = self.acc.get("flow_sampling", 0.0)
		flow_calc = self.acc.get("flow_calc", 0.0)
		opt_step = self.acc.get("opt_step", 0.0)
		measured = io_prefetch + flow_sampling + flow_calc + opt_step
		other = max(0.0, total - measured)
		rows = [
			("io/prefetch", io_prefetch),
			("flow sampling", flow_sampling),
			("flow calc", flow_calc),
			("opt step", opt_step),
			("other", other),
		]
		print(f"[flow_timing] {label} {step1}/{max_steps} over {self.count} iters", flush=True)
		print(f"{'part':<16s} {'runtime_%':>9s} {'ms/it':>10s}", flush=True)
		for name, seconds in rows:
			pct = 100.0 * seconds / total
			ms_it = 1000.0 * seconds / float(self.count)
			print(f"{name:<16s} {pct:9.2f} {ms_it:10.2f}", flush=True)
		self.count = 0
		for key in list(self.acc.keys()):
			self.acc[key] = 0.0


class OptTimingWindow:
	_ORDER = [
		"cache_sync",
		"model_point_prefetch",
		"model_forward",
		"loss_prefetch",
		"loss_eval",
		"chunk_stats",
		"zero_grad",
		"backward",
		"optimizer_step",
		"model_updates",
		"next_prefetch",
		"progress",
		"status",
		"ensure_data",
		"snapshot",
	]

	def __init__(self, *, interval: int = 100, sync_cuda: bool = True) -> None:
		self.interval = max(1, int(interval))
		self.sync_cuda = bool(sync_cuda)
		self.count = 0
		self.acc: dict[str, float] = {"total": 0.0}

	def sync(self) -> None:
		if self.sync_cuda:
			synchronize_cuda()

	def add(self, key: str, seconds: float) -> None:
		self.acc[key] = self.acc.get(key, 0.0) + max(0.0, float(seconds))

	def finish_iter(self, *, label: str, step1: int, max_steps: int) -> None:
		self.count += 1
		if (step1 % self.interval) != 0 and step1 != max_steps:
			return
		if self.count <= 0:
			return
		total = max(1.0e-12, self.acc.get("total", 0.0))
		primary_keys = [k for k in self._ORDER if self.acc.get(k, 0.0) > 0.0]
		extra_primary = sorted(
			k for k, v in self.acc.items()
			if k not in {"total", *primary_keys} and not k.startswith("loss:") and v > 0.0
		)
		loss_keys = sorted(k for k, v in self.acc.items() if k.startswith("loss:") and v > 0.0)
		measured = sum(self.acc.get(k, 0.0) for k in primary_keys + extra_primary)
		other = max(0.0, total - measured)
		print(
			f"[opt_timing] {label} {step1}/{max_steps} over {self.count} iters "
			f"sync_cuda={int(self.sync_cuda)}",
			flush=True,
		)
		print(f"{'part':<24s} {'runtime_%':>9s} {'ms/it':>10s} {'total_s':>10s}", flush=True)
		for key in primary_keys + extra_primary:
			seconds = self.acc.get(key, 0.0)
			print(
				f"{key:<24s} {100.0 * seconds / total:9.2f} "
				f"{1000.0 * seconds / float(self.count):10.2f} {seconds:10.3f}",
				flush=True,
			)
		if other > 0.0:
			print(
				f"{'other':<24s} {100.0 * other / total:9.2f} "
				f"{1000.0 * other / float(self.count):10.2f} {other:10.3f}",
				flush=True,
			)
		if loss_keys:
			print("[opt_timing] loss term breakdown is inside loss_eval", flush=True)
			for key in loss_keys:
				seconds = self.acc.get(key, 0.0)
				print(
					f"{key:<24s} {100.0 * seconds / total:9.2f} "
					f"{1000.0 * seconds / float(self.count):10.2f} {seconds:10.3f}",
					flush=True,
				)
		self.count = 0
		for key in list(self.acc.keys()):
			self.acc[key] = 0.0
