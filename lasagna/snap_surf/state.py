from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_station
import opt_loss_winding_density

from .config import SnapSurfConfig

class _DirectionState:
	def __init__(self, *, source_rank: int, target_rank: int) -> None:
		self.source_rank = int(source_rank)
		self.target_rank = int(target_rank)
		self.source_shape: tuple[int, ...] | None = None
		self.target_shape: tuple[int, ...] | None = None
		self.map: torch.Tensor | None = None
		self.valid: torch.Tensor | None = None
		self.orientation_sign: int = 1
		self.seed_base_idx: tuple[int, ...] | None = None

	def ensure(
		self,
		*,
		source_shape: tuple[int, ...],
		target_shape: tuple[int, ...],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		if (
			self.map is not None
			and self.valid is not None
			and self.source_shape == source_shape
			and self.target_shape == target_shape
			and self.map.device == device
			and self.map.dtype == dtype
		):
			return
		self.source_shape = tuple(int(v) for v in source_shape)
		self.target_shape = tuple(int(v) for v in target_shape)
		self.map = torch.full((*self.source_shape, self.target_rank), float("nan"), device=device, dtype=dtype)
		self.valid = torch.zeros(self.source_shape, device=device, dtype=torch.bool)
		self.orientation_sign = 1
		self.seed_base_idx = None

	def count(self) -> int:
		if self.valid is None:
			return 0
		return int(self.valid.sum().detach().cpu())

class _MapInitState:
	def __init__(self) -> None:
		self.done: bool = False
		self.active_quad: torch.Tensor | None = None
		self.blocked_quad: torch.Tensor | None = None
		self.scale_active_quads: list[torch.Tensor | None] = []
		self.scale_blocked_quads: list[torch.Tensor | None] = []
		self.uv: torch.Tensor | None = None
		self.uv_guess: torch.Tensor | None = None
		self.ext_pos: torch.Tensor | None = None
		self.ext_normals: torch.Tensor | None = None
		self.ext_valid: torch.Tensor | None = None
		self.ext_quad_valid: torch.Tensor | None = None
		self.ext_coords: torch.Tensor | None = None
		self.uv_pyramid: list[torch.Tensor] | None = None
		self.scale_level: int = 0
		self.target_scale_level: int = 0
		self.scale_strides: list[int] = [1]
		self.model_depth: int | None = None
		self.seed_ext_sample_hw: tuple[int, int] | None = None
		self.seed_model_quad: tuple[int, int, int] | None = None
		self.seed_model_distance: float = float("inf")
		self.seed_ext_distance: float = float("inf")
		self.seed_init_count: int = 0
		self.orientation_sign: int = 1
		self.normal_sign: int = 1
		self.total_iters: int = 0
		self.grow_steps: int = 0
		self.opt_blocks: int = 0
		self.global_opt_blocks: int = 0
		self.rim_only_blocks: int = 0
		self.rim_problem_blocks: int = 0
		self.rim_blocks_since_global_opt: int = 0
		self.repair_blocks: int = 0
		self.added_total: int = 0
		self.last_terms: dict[str, torch.Tensor] = {}
		self.last_growth_terms: dict[str, torch.Tensor] = {}
		self.surface_initial_done: bool = False
		self.surface_first_global_done: bool = False
		self.surface_last_global_done: bool = False
		self.surface_last_update_step: int | None = None
		self.sparse_pruned_total: int = 0
		self.add_sample_loss_sum: float = 0.0
		self.add_sample_weight: float = 0.0
		self.add_bad_samples: float = 0.0
		self.add_total_samples: float = 0.0
		self.add_success_quads: float = 0.0
		self.add_total_quads: float = 0.0
		self.fringe_sample_loss_sum: float = 0.0
		self.fringe_sample_weight: float = 0.0
		self.fringe_bad_samples: float = 0.0
		self.fringe_total_samples: float = 0.0
		self.fringe_success_quads: float = 0.0
		self.fringe_total_quads: float = 0.0
		self.interval_add_sample_loss_sum: float = 0.0
		self.interval_add_sample_weight: float = 0.0
		self.interval_add_bad_samples: float = 0.0
		self.interval_add_total_samples: float = 0.0
		self.interval_add_success_quads: float = 0.0
		self.interval_add_total_quads: float = 0.0
		self.interval_fringe_sample_loss_sum: float = 0.0
		self.interval_fringe_sample_weight: float = 0.0
		self.interval_fringe_bad_samples: float = 0.0
		self.interval_fringe_total_samples: float = 0.0
		self.interval_fringe_success_quads: float = 0.0
		self.interval_fringe_total_quads: float = 0.0
		self.fringe_debug_rows: int = 0
		self.scale_levels_used: int = 1
		self.progress_rows: int = 0
		self.progress_last_time: float | None = None
		self.progress_last_iter: int = 0
		self.blocked_last_revisit_iter: int = 0
		self.stats: dict[str, float] = {}

	def reset(self) -> None:
		self.__init__()

	def active_count(self) -> int:
		if self.active_quad is None:
			return 0
		return int(self.active_quad.sum().detach().cpu())

	def current_stride(self) -> int:
		if 0 <= int(self.scale_level) < len(self.scale_strides):
			return int(self.scale_strides[int(self.scale_level)])
		return 1

class _SurfaceState:
	def __init__(self) -> None:
		self.model_to_ext = _DirectionState(source_rank=3, target_rank=2)
		self.ext_to_model = _DirectionState(source_rank=2, target_rank=3)
		self.map_init = _MapInitState()
		self.ext_seed_hw: tuple[int, int] | None = None
		self.seed_ext_distance: float | None = None
		self.seed_ext_key: tuple[float, float, float] | None = None
		self.seed_ext_point_xyz: tuple[float, float, float] | None = None
		self.snap_eval_count: int = 0

	def reset_map_init(self) -> None:
		self.map_init.reset()

	def ensure(
		self,
		*,
		model_shape: tuple[int, int, int],
		ext_shape: tuple[int, int],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		old_model_shape = self.model_to_ext.source_shape
		old_ext_shape = self.model_to_ext.target_shape
		self.model_to_ext.ensure(
			source_shape=model_shape,
			target_shape=ext_shape,
			device=device,
			dtype=dtype,
		)
		self.ext_to_model.ensure(
			source_shape=ext_shape,
			target_shape=model_shape,
			device=device,
			dtype=dtype,
		)
		if old_model_shape != model_shape or old_ext_shape != ext_shape:
			self.ext_seed_hw = None
			self.seed_ext_distance = None
			self.seed_ext_key = None
			self.seed_ext_point_xyz = None
			self.snap_eval_count = 0
			self.reset_map_init()

__all__ = [name for name in globals() if not name.startswith('__')]
