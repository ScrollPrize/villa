from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

import fit_data


@dataclass(frozen=True)
class ModelParams3D:
	mesh_step: int          # height step in fullres voxels
	winding_step: int       # radial step per winding in fullres voxels
	subsample_mesh: int     # HR height subsample factor
	subsample_winding: int  # HR width subsample factor
	scaledown: float        # data xy downscale factor
	z_step_eff: int         # effective z spacing in fullres voxels
	volume_extent: tuple[float, float, float, float, float, float] | None  # (x0,y0,z0,x1,y1,z1) fullres bbox
	pyramid_d: bool         # whether depth axis participates in pyramid
	model_w: float | None = None
	model_h: float | None = None


@dataclass(frozen=True)
class FitResult3D:
	xyz_lr: torch.Tensor        # (D, Hm, Wm, 3) LR mesh in fullres voxels
	xyz_hr: torch.Tensor        # (D, He, We, 3) HR mesh
	data: fit_data.FitData3D    # full volume data
	data_s: fit_data.FitData3D  # sampled at HR positions
	target_plain: torch.Tensor  # (D, 1, He, We)
	target_mod: torch.Tensor    # (D, 1, He, We)
	amp_lr: torch.Tensor        # (D, 1, Hm, Wm)
	bias_lr: torch.Tensor       # (D, 1, Hm, Wm)
	mask_hr: torch.Tensor       # (D, 1, He, We)
	mask_lr: torch.Tensor       # (D, 1, Hm, Wm)
	normals: torch.Tensor       # (D, Hm, Wm, 3) detached unit normals
	xy_conn: torch.Tensor       # (D, Hm, Wm, 3, 3) — [prev, self, next], each 3D fullres
	mask_conn: torch.Tensor     # (D, 1, Hm, Wm, 3) — validity per connection point
	sign_conn: torch.Tensor     # (D, 1, Hm, Wm, 2) — ray param sign [prev, next]
	params: ModelParams3D
	gt_normal_lr: torch.Tensor | None = None  # (D, Hm, Wm, 3) GT unit normals at LR mesh positions
	ext_conn: list | None = None
	cyl_xyz: torch.Tensor | None = None      # (N*D, Hm, Wm, 3) analytic cylinder samples
	cyl_normals: torch.Tensor | None = None  # (N*D, Hm, Wm, 3) analytic cylinder normals
	cyl_centers: torch.Tensor | None = None  # (N, 3) cylinder axis anchor [cx, cy, zc]
	cyl_axes: torch.Tensor | None = None     # (N, 3) unit cylinder axis direction
	cyl_params: torch.Tensor | None = None   # analytic: (N, 6); shell: (H, W, 2) free xy delta
	cyl_count: int = 0
	cyl_shell_mode: bool = False
	cyl_shell_step: float = 500.0
	cyl_shell_base_xyz: torch.Tensor | None = None
	cyl_shell_dirs: torch.Tensor | None = None
	cyl_shell_w_offsets: torch.Tensor | None = None
	cyl_shell_delta_xy: torch.Tensor | None = None
	cyl_shell_index: int = 0
	# Per ext surface: (mask, offset, ext_P, ext_N, full_h, full_w)
	# ext_P/ext_N = ext corner pos/normal (detached), full_h/full_w = model grid position (row+u, col+v)
	# Shapes: (D, H_ext, W_ext, ...). Model quad corners are re-gathered from xyz_lr in the loss.


class Model3D(nn.Module):
	def __init__(
		self,
		*,
		device: torch.device,
		depth: int,
		mesh_h: int,
		mesh_w: int,
		mesh_step: int,
		winding_step: int,
		subsample_mesh: int = 4,
		subsample_winding: int = 4,
		scaledown: float = 1.0,
		z_step_eff: int = 1,
		z_center: float = 0.0,
		arc_cx: float = 0.0,
		arc_cy: float = 0.0,
		arc_radius: float = 1000.0,
		arc_angle0: float = -0.5,
		arc_angle1: float = 0.5,
		straight_cx: float = 0.0,
		straight_cy: float = 0.0,
		straight_angle: float = 0.0,
		straight_half_w: float = 100.0,
		init_mode: str = "cylinder_seed",
		volume_extent: tuple[float, float, float, float, float, float] | None = None,
		pyramid_d: bool = True,
	) -> None:
		super().__init__()
		self.depth = max(1, int(depth))
		self.mesh_h = max(2, int(mesh_h))
		self.mesh_w = max(2, int(mesh_w))
		self.z_center = float(z_center)
		self.init_mode = str(init_mode)
		self.cylinder_enabled = self.init_mode == "cylinder_seed"
		self.cyl_best_idx = 0
		self.arc_enabled = False
		self.straight_enabled = False
		self.pyramid_d = bool(pyramid_d) and self.depth > 1

		self.params = ModelParams3D(
			mesh_step=int(mesh_step),
			winding_step=int(winding_step),
			subsample_mesh=int(subsample_mesh),
			subsample_winding=int(subsample_winding),
			scaledown=float(scaledown),
			z_step_eff=max(1, int(z_step_eff)),
			volume_extent=volume_extent,
			pyramid_d=self.pyramid_d,
		)

		# Arc parameters (fullres coordinates)
		self.arc_cx = nn.Parameter(torch.tensor(float(arc_cx), device=device, dtype=torch.float32))
		self.arc_cy = nn.Parameter(torch.tensor(float(arc_cy), device=device, dtype=torch.float32))
		self.arc_radius = nn.Parameter(torch.tensor(float(arc_radius), device=device, dtype=torch.float32))
		self.arc_angle0 = nn.Parameter(torch.tensor(float(arc_angle0), device=device, dtype=torch.float32))
		self.arc_angle1 = nn.Parameter(torch.tensor(float(arc_angle1), device=device, dtype=torch.float32))

		# Straight parameters (fullres coordinates)
		self.straight_cx = nn.Parameter(torch.tensor(float(straight_cx), device=device, dtype=torch.float32))
		self.straight_cy = nn.Parameter(torch.tensor(float(straight_cy), device=device, dtype=torch.float32))
		self.straight_angle = nn.Parameter(torch.tensor(float(straight_angle), device=device, dtype=torch.float32))
		self.straight_half_w = nn.Parameter(torch.tensor(float(straight_half_w), device=device, dtype=torch.float32))

		# Multi-start analytic cylinder seed params:
		# [radius, ellipse_k, seed_phase, tilt_x, tilt_y, roll]. The center is
		# derived so q=0 stays on the seed point, and height is centered
		# at the seed along the optimized cylinder axis.
		self.cyl_params = nn.Parameter(self._default_cylinder_params(device=device))
		self.cyl_shell_w_offsets = nn.Parameter(torch.zeros(self.mesh_h, self.mesh_w, device=device, dtype=torch.float32))
		self.cyl_seed_xyz = torch.zeros(3, device=device, dtype=torch.float32)
		self.cyl_shell_mode = False
		self.cyl_shell_target_count = 6
		self.cyl_shell_initial_radius = 1000.0
		self.cyl_shell_step = 500.0
		self.cyl_shell_initial_step = 10.0
		self.cyl_shell_z_step = 200.0
		self.cyl_shell_width_target_step = 200.0
		self.cyl_shell_min_width = 20
		self.cyl_shell_seed_z = float(z_center)
		self.cyl_shell_model_h: float | None = None
		self.cyl_shell_z: torch.Tensor | None = None
		self.cyl_shell_base: torch.Tensor | None = None
		self.cyl_shell_dirs: torch.Tensor | None = None
		self.cyl_shell_completed: list[torch.Tensor] = []
		self.cyl_shell_current_index = 0
		self.cyl_shell_active = False

		# Residual mesh pyramid: (3, D, H, W) per scale, 5 levels
		n_scales = 5
		self.mesh_ms = self._build_zero_pyramid(
			n_scales=n_scales, channels=3, d=self.depth, h=self.mesh_h, w=self.mesh_w, device=device,
			pyramid_d=self.pyramid_d,
		)
		# Connection offsets buffer: (4, D, Hm, Wm) — [prev_h, prev_w, next_h, next_w]
		# Not gradient-optimized; updated by update_conn_offsets() after each step.
		self.register_buffer("conn_offsets", torch.zeros(4, self.depth, self.mesh_h, self.mesh_w, device=device, dtype=torch.float32))

		# External reference surfaces: frozen meshes for offset optimization
		self._ext_surfaces: list[torch.Tensor] = []          # each (H_ext, W_ext, 3)
		self._ext_valid: list[torch.Tensor] = []             # each (H_ext, W_ext) bool
		self._ext_conn_offsets: list[torch.Tensor] = []      # each (2, D, H_ext, W_ext) — [h_off, w_off]
		self._ext_conn_params: list[dict] = []                # cached intersection params per ext surface
		self._ext_normals: list[torch.Tensor] = []            # each (H_ext, W_ext, 3) precomputed unit normals
		self._ext_offsets: list[float] = []                   # target integral offset per ext surface

		# Amplitude and bias for data matching (deferred but needed for FitResult3D)
		amp_init = torch.full((self.depth, 1, self.mesh_h, self.mesh_w), 1.0, device=device, dtype=torch.float32)
		bias_init = torch.full((self.depth, 1, self.mesh_h, self.mesh_w), 0.5, device=device, dtype=torch.float32)
		self.amp = nn.Parameter(amp_init)
		self.bias = nn.Parameter(bias_init)

	@staticmethod
	def _build_zero_pyramid(*, n_scales: int, channels: int, d: int, h: int, w: int, device: torch.device, pyramid_d: bool) -> nn.ParameterList:
		shapes: list[tuple[int, int, int]] = [(d, h, w)]
		for _ in range(1, max(1, n_scales)):
			dp, hp, wp = shapes[-1]
			if pyramid_d:
				shapes.append((max(2, (dp + 1) // 2), max(2, (hp + 1) // 2), max(2, (wp + 1) // 2)))
			else:
				shapes.append((dp, max(2, (hp + 1) // 2), max(2, (wp + 1) // 2)))
		print(f"[model] pyramid levels (C={channels}, pyramid_d={pyramid_d}): {' -> '.join(f'{d}x{h}x{w}' for d,h,w in shapes)}")
		return nn.ParameterList([
			nn.Parameter(torch.zeros(channels, di, hi, wi, device=device, dtype=torch.float32))
			for di, hi, wi in shapes
		])

	@staticmethod
	def _default_cylinder_params(*, device: torch.device) -> torch.Tensor:
		return torch.zeros(1, 6, device=device, dtype=torch.float32)

	# --- 3D pyramid operations ---

	@staticmethod
	def _upsample_crop(src: torch.Tensor, d_t: int, h_t: int, w_t: int, *, pyramid_d: bool) -> torch.Tensor:
		if pyramid_d:
			return F.interpolate(src.unsqueeze(0), size=(d_t, h_t, w_t),
								mode='trilinear', align_corners=True).squeeze(0)
		else:
			return F.interpolate(src, size=(h_t, w_t),
								mode='bilinear', align_corners=True)

	@staticmethod
	def _integrate_pyramid_3d(src: nn.ParameterList, *, pyramid_d: bool) -> torch.Tensor:
		v = src[-1]
		for d in reversed(list(src[:-1])):
			v = Model3D._upsample_crop(v, d.shape[1], d.shape[2], d.shape[3], pyramid_d=pyramid_d) + d
		return v

	@staticmethod
	def _construct_pyramid_from_flat_3d(flat: torch.Tensor, n_scales: int, *, pyramid_d: bool) -> nn.ParameterList:
		shapes: list[tuple[int, int, int]] = [(int(flat.shape[1]), int(flat.shape[2]), int(flat.shape[3]))]
		for _ in range(1, n_scales):
			d, h, w = shapes[-1]
			if pyramid_d:
				shapes.append((max(2, (d + 1) // 2), max(2, (h + 1) // 2), max(2, (w + 1) // 2)))
			else:
				shapes.append((d, max(2, (h + 1) // 2), max(2, (w + 1) // 2)))
		targets: list[torch.Tensor] = [flat]
		for (d, h, w) in shapes[1:]:
			if pyramid_d:
				t = F.interpolate(targets[-1].unsqueeze(0), size=(d, h, w),
								  mode='trilinear', align_corners=True).squeeze(0)
			else:
				t = F.interpolate(targets[-1], size=(h, w),
								  mode='bilinear', align_corners=True)
			targets.append(t)
		residuals: list[torch.Tensor | None] = [None] * len(targets)
		recon = targets[-1]
		residuals[-1] = targets[-1]
		for i in range(len(targets) - 2, -1, -1):
			up = Model3D._upsample_crop(recon, *targets[i].shape[1:], pyramid_d=pyramid_d)
			residuals[i] = targets[i] - up
			recon = up + residuals[i]
		return nn.ParameterList([nn.Parameter(r) for r in residuals])

	@staticmethod
	def _construct_pyramid_from_flat_3d_masked(
		flat: torch.Tensor, valid: torch.Tensor, n_scales: int, *, pyramid_d: bool,
	) -> nn.ParameterList:
		"""Build residual pyramid with validity-aware downsampling.

		flat: (C, D, H, W), valid: (H, W) bool.
		Invalid regions get residual=0 so integration naturally inpaints them
		from coarser (valid) structure.
		"""
		C = flat.shape[0]
		D = int(flat.shape[1])
		# Compute enough scales to reach 1×1
		shapes: list[tuple[int, int, int]] = [(D, int(flat.shape[2]), int(flat.shape[3]))]
		for _ in range(1, n_scales):
			d, h, w = shapes[-1]
			if pyramid_d:
				shapes.append((max(1, (d + 1) // 2), max(1, (h + 1) // 2), max(1, (w + 1) // 2)))
			else:
				shapes.append((d, max(1, (h + 1) // 2), max(1, (w + 1) // 2)))

		# Build validity mask pyramid: (1, D, H, W) float
		valid_4d = valid.float().unsqueeze(0).unsqueeze(0).expand(1, D, -1, -1)  # (1, D, H, W)

		# Masked downsampling: weighted average excluding invalid
		targets: list[torch.Tensor] = [flat]
		valids: list[torch.Tensor] = [valid_4d]
		for (d_t, h_t, w_t) in shapes[1:]:
			prev_data = targets[-1] * valids[-1]  # zero out invalid before pooling
			prev_valid = valids[-1]
			if pyramid_d:
				data_down = F.interpolate(
					(prev_data).unsqueeze(0), size=(d_t, h_t, w_t),
					mode='trilinear', align_corners=True).squeeze(0)
				valid_down = F.interpolate(
					prev_valid.unsqueeze(0), size=(d_t, h_t, w_t),
					mode='trilinear', align_corners=True).squeeze(0)
			else:
				# (C, D, H, W) → treat D as batch: (C*D, 1, H, W) for 2D interpolate
				CD = C * D
				data_down = F.interpolate(
					prev_data.reshape(CD, 1, prev_data.shape[2], prev_data.shape[3]),
					size=(h_t, w_t), mode='bilinear', align_corners=True
				).reshape(C, D, h_t, w_t)
				valid_down = F.interpolate(
					prev_valid.reshape(D, 1, prev_valid.shape[2], prev_valid.shape[3]),
					size=(h_t, w_t), mode='bilinear', align_corners=True
				).reshape(1, D, h_t, w_t)
			# Normalize by valid weight
			target = data_down / valid_down.clamp(min=1e-6)
			valid_mask = (valid_down > 0.01).float()
			targets.append(target)
			valids.append(valid_mask)

		# Build residuals: masked so invalid regions contribute zero
		residuals: list[torch.Tensor | None] = [None] * len(targets)
		recon = targets[-1]
		residuals[-1] = targets[-1]
		for i in range(len(targets) - 2, -1, -1):
			up = Model3D._upsample_crop(recon, *targets[i].shape[1:], pyramid_d=pyramid_d)
			residuals[i] = (targets[i] - up) * valids[i]
			recon = up + residuals[i]
		return nn.ParameterList([nn.Parameter(r) for r in residuals])

	# --- Analytic cylinder seed ---

	def init_cylinder_seed(
		self,
		*,
		seed: tuple[float, float, float],
		model_w: float,
		model_h: float,
		volume_extent_fullres: tuple[int, int, int] | None,
	) -> None:
		"""Initialize the experimental umbilicus tube grower from seed z.

		The current experiment keeps the public cylinder_seed trigger, but the
		actual geometry is built lazily once FitData3D has supplied the
		umbilicus lookup. Seed x/y and model_w are intentionally ignored.
		"""
		device = self.cyl_params.device
		with torch.no_grad():
			self.cyl_seed_xyz = torch.tensor([float(seed[0]), float(seed[1]), float(seed[2])],
											 device=device, dtype=torch.float32)
			self.cyl_params = nn.Parameter(torch.zeros(self.mesh_h, self.mesh_w, 2, device=device, dtype=torch.float32))
			self.cyl_shell_w_offsets = nn.Parameter(torch.zeros(self.mesh_h, self.mesh_w, device=device, dtype=torch.float32))
		self.params = replace(self.params, model_w=None, model_h=float(model_h))
		self.cyl_shell_mode = True
		self.cyl_shell_seed_z = float(seed[2])
		self.cyl_shell_model_h = float(model_h)
		self.cyl_shell_z = None
		self.cyl_shell_base = None
		self.cyl_shell_dirs = None
		self.cyl_shell_completed = []
		self.cyl_shell_current_index = 0
		self.cyl_shell_active = False
		self.cylinder_enabled = True
		self.cyl_best_idx = 0
		self.arc_enabled = False
		self.straight_enabled = False
		self.init_mode = "cylinder_seed"

	def _set_shell_grid_shape(self, *, h: int, w: int) -> None:
		h = max(2, int(h))
		w = max(3, int(w))
		device = self.cyl_params.device
		self.depth = 1
		self.mesh_h = h
		self.mesh_w = w
		self.conn_offsets = torch.zeros(4, 1, h, w, device=device, dtype=torch.float32)
		self.amp = nn.Parameter(torch.ones(1, 1, h, w, device=device, dtype=torch.float32))
		self.bias = nn.Parameter(torch.full((1, 1, h, w), 0.5, device=device, dtype=torch.float32))

	def _set_fused_grid_shape(self, *, d: int, h: int, w: int) -> None:
		d = max(1, int(d))
		h = max(2, int(h))
		w = max(3, int(w))
		device = self.cyl_params.device
		self.depth = d
		self.mesh_h = h
		self.mesh_w = w
		self.conn_offsets = torch.zeros(4, d, h, w, device=device, dtype=torch.float32)
		self.amp = nn.Parameter(torch.ones(d, 1, h, w, device=device, dtype=torch.float32))
		self.bias = nn.Parameter(torch.full((d, 1, h, w), 0.5, device=device, dtype=torch.float32))

	def prepare_umbilicus_tube_init(self, data: fit_data.FitData3D) -> None:
		"""Build the first pending shell from the umbilicus lookup."""
		if not self.cyl_shell_mode or self.cyl_shell_base is not None:
			return
		if self.cyl_shell_model_h is None:
			raise ValueError("cylinder shell init missing model_h")
		device = self.cyl_params.device
		dtype = self.cyl_params.dtype
		z_step = max(1.0, float(self.cyl_shell_z_step))
		model_h = max(float(self.cyl_shell_model_h), z_step)
		H = max(2, int(math.ceil(model_h / z_step)) + 1)
		actual_z_step = model_h / float(max(1, H - 1))
		radius0 = self._first_shell_radius()
		W = self._shell_width_for_radius(radius0)
		self._set_shell_grid_shape(h=H, w=W)
		z = self._shell_z_values(device=device, dtype=dtype, h=H)
		base, dirs = self._umbilicus_base_shell(data=data, z=z, w=W)
		self.cyl_shell_z = z.detach()
		self.cyl_shell_base = base.detach()
		self.cyl_shell_dirs = dirs.detach()
		self.cyl_params = nn.Parameter(
			self._initial_shell_delta_xy(dirs, target_step=self._shell_target_offset_for_index(0)).to(device=device, dtype=dtype)
		)
		self.cyl_shell_w_offsets = nn.Parameter(torch.zeros(H, W, device=device, dtype=dtype))
		print(f"[model] umbilicus tube init: shells={self.cyl_shell_target_count} "
			  f"H={H} W={W} z_center={self.cyl_shell_seed_z:.1f} "
			  f"model_h={model_h:.1f} z_step={actual_z_step:.1f} "
			  f"z_step_target={z_step:.1f} initial_radius={radius0:.1f} "
			  f"shell_step={self.cyl_shell_step:.1f}",
			  flush=True)

	def _shell_z_values(self, *, device: torch.device, dtype: torch.dtype, h: int) -> torch.Tensor:
		model_h = (
			float(self.cyl_shell_model_h)
			if self.cyl_shell_model_h is not None
			else float(self.params.mesh_step) * float(max(1, h - 1))
		)
		if h <= 1:
			return torch.full((1,), float(self.cyl_shell_seed_z), device=device, dtype=dtype)
		t = torch.linspace(-0.5, 0.5, h, device=device, dtype=dtype)
		return float(self.cyl_shell_seed_z) + t * float(model_h)

	def _shell_width_for_radius(self, radius: float) -> int:
		step = max(1.0, float(self.cyl_shell_width_target_step))
		circ = max(0.0, 2.0 * math.pi * float(radius))
		return max(int(self.cyl_shell_min_width), int(math.ceil(circ / step)))

	def _first_shell_radius(self) -> float:
		return max(1.0, float(getattr(self, "cyl_shell_initial_radius", self.cyl_shell_step)))

	def _shell_target_offset_for_index(self, idx: int) -> float:
		idx = int(idx)
		if idx == 0:
			return self._first_shell_radius()
		return max(1.0, float(self.cyl_shell_step))

	def _current_shell_target_offset(self) -> float:
		return self._shell_target_offset_for_index(int(getattr(self, "cyl_shell_current_index", 0)))

	def _umbilicus_base_shell(
		self,
		*,
		data: fit_data.FitData3D,
		z: torch.Tensor,
		w: int,
	) -> tuple[torch.Tensor, torch.Tensor]:
		device = z.device
		dtype = z.dtype
		angles = torch.arange(w, device=device, dtype=dtype) * (2.0 * math.pi / float(w))
		dir_xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
		umb_xy = data.umbilicus_xy_at_z(z).to(device=device, dtype=dtype)
		base_xy = umb_xy[:, None, :].expand(int(z.shape[0]), w, 2)
		base_z = z[:, None, None].expand(int(z.shape[0]), w, 1)
		base = torch.cat([base_xy, base_z], dim=-1)
		dirs = torch.cat([
			dir_xy[None, :, :].expand(int(z.shape[0]), w, 2),
			torch.zeros(int(z.shape[0]), w, 1, device=device, dtype=dtype),
		], dim=-1)
		return base, dirs

	@staticmethod
	def _unit_vertex_normals_for_shell(xyz: torch.Tensor) -> torch.Tensor:
		if xyz.ndim == 3:
			xyz_b = xyz.unsqueeze(0)
			squeeze = True
		else:
			xyz_b = xyz
			squeeze = False
		dh = torch.zeros_like(xyz_b)
		dh[:, 1:-1] = xyz_b[:, 2:] - xyz_b[:, :-2]
		dh[:, 0] = xyz_b[:, 1] - xyz_b[:, 0]
		dh[:, -1] = xyz_b[:, -1] - xyz_b[:, -2]
		dw = torch.roll(xyz_b, shifts=-1, dims=2) - torch.roll(xyz_b, shifts=1, dims=2)
		n = torch.cross(dh, dw, dim=-1)
		n = F.normalize(n, dim=-1, eps=1.0e-8)
		return n.squeeze(0) if squeeze else n

	def _outward_xy_dirs_for_shell(self, shell: torch.Tensor, data: fit_data.FitData3D) -> torch.Tensor:
		with torch.no_grad():
			n = self._unit_vertex_normals_for_shell(shell)
			z = shell[..., 2]
			umb_xy = data.umbilicus_xy_at_z(z)
			radial_xy = shell[..., :2] - umb_xy
			radial_xy = radial_xy / radial_xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
			n_xy = n[..., :2]
			n_xy_len = n_xy.norm(dim=-1, keepdim=True)
			n_xy = torch.where(n_xy_len > 1.0e-8, n_xy / n_xy_len.clamp(min=1.0e-8), radial_xy)
			n_xy = torch.where(((n_xy * radial_xy).sum(dim=-1, keepdim=True) < 0.0), -n_xy, n_xy)
			dirs = torch.cat([n_xy, torch.zeros_like(n_xy[..., :1])], dim=-1)
			return dirs.detach()

	def _gt_xy_dirs_for_reference_shell(self, shell: torch.Tensor, data: fit_data.FitData3D) -> torch.Tensor:
		with torch.no_grad():
			surf_n = self._unit_vertex_normals_for_shell(shell)
			z = shell[..., 2]
			umb_xy = data.umbilicus_xy_at_z(z)
			radial_xy = shell[..., :2] - umb_xy
			radial_xy = radial_xy / radial_xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
			surf_xy = surf_n[..., :2]
			surf_xy_len = surf_xy.norm(dim=-1, keepdim=True)
			surf_xy = torch.where(surf_xy_len > 1.0e-8, surf_xy / surf_xy_len.clamp(min=1.0e-8), radial_xy)
			surf_xy = torch.where(((surf_xy * radial_xy).sum(dim=-1, keepdim=True) < 0.0), -surf_xy, surf_xy)
			surf_out = torch.cat([surf_xy, torch.zeros_like(surf_xy[..., :1])], dim=-1)
			sampled = data.grid_sample_fullres(
				shell.detach().unsqueeze(0),
				diff=False,
				channels={"grad_mag", "nx", "ny"},
			)
			gt_n = sampled.normal_3d
			if gt_n is None:
				return self._outward_xy_dirs_for_shell(shell, data)
			gt_n = gt_n.squeeze(0)
			gt_n = gt_n.to(device=shell.device, dtype=shell.dtype)
			gt_n = F.normalize(gt_n, dim=-1, eps=1.0e-8)
			gt_n = torch.where(((gt_n * surf_out).sum(dim=-1, keepdim=True) < 0.0), -gt_n, gt_n)

			gt_xy = gt_n[..., :2]
			gt_xy_len = gt_xy.norm(dim=-1, keepdim=True)
			gt_xy = torch.where(gt_xy_len > 1.0e-8, gt_xy / gt_xy_len.clamp(min=1.0e-8), surf_xy)
			gt_xy = torch.where(((gt_xy * surf_xy).sum(dim=-1, keepdim=True) < 0.0), -gt_xy, gt_xy)
			gt_xy = torch.where(((gt_xy * surf_xy).sum(dim=-1, keepdim=True) > 0.5), gt_xy, surf_xy)
			return torch.cat([gt_xy, torch.zeros_like(gt_xy[..., :1])], dim=-1).detach()

	def _initial_shell_delta_xy(self, dirs: torch.Tensor, *, target_step: float) -> torch.Tensor:
		step = max(0.0, float(target_step))
		return dirs[..., :2] * step

	def _shell_delta_xy_params(self) -> torch.Tensor:
		if self.cyl_params.ndim == 3 and int(self.cyl_params.shape[-1]) == 2:
			return self.cyl_params
		raise ValueError(f"invalid cylinder shell params shape: {tuple(self.cyl_params.shape)}")

	def _shell_w_offset_values(self) -> torch.Tensor:
		delta_xy = self._shell_delta_xy_params()
		if int(getattr(self, "cyl_shell_current_index", 0)) <= 0:
			return torch.zeros_like(delta_xy[..., 0])
		if self.cyl_shell_w_offsets.shape != delta_xy.shape[:2]:
			return torch.zeros_like(delta_xy[..., 0])
		return self.cyl_shell_w_offsets

	@staticmethod
	def _interp_width_at_offsets(field: torch.Tensor, offsets: torch.Tensor, *, offset_scale: float = 1.0) -> torch.Tensor:
		if field.ndim == 2:
			field_in = field.unsqueeze(-1)
			squeeze = True
		else:
			field_in = field
			squeeze = False
		H = int(field_in.shape[0])
		W = int(field_in.shape[1])
		C = int(field_in.shape[2])
		if W <= 1:
			out = field_in.expand(H, max(1, W), C)
			return out.squeeze(-1) if squeeze else out
		device = field_in.device
		dtype = field_in.dtype
		base_w = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
		phase = base_w + offsets.to(device=device, dtype=dtype) * float(offset_scale)
		i0_floor = torch.floor(phase)
		frac = (phase - i0_floor).unsqueeze(-1)
		i0 = torch.remainder(i0_floor.to(dtype=torch.long), W)
		i1 = torch.remainder(i0 + 1, W)
		i0e = i0.unsqueeze(-1).expand(H, W, C)
		i1e = i1.unsqueeze(-1).expand(H, W, C)
		p0 = torch.gather(field_in, 1, i0e)
		p1 = torch.gather(field_in, 1, i1e)
		out = p0 + frac * (p1 - p0)
		return out.squeeze(-1) if squeeze else out

	def _current_base_conn_and_dirs(self) -> tuple[torch.Tensor, torch.Tensor]:
		if self.cyl_shell_base is None or self.cyl_shell_dirs is None:
			raise ValueError("umbilicus tube shell has not been prepared")
		base = self.cyl_shell_base
		dirs = self.cyl_shell_dirs
		offsets = self._shell_w_offset_values()
		if int(getattr(self, "cyl_shell_current_index", 0)) <= 0:
			return base, F.normalize(dirs, dim=-1, eps=1.0e-8)
		base_conn = self._interp_width_at_offsets(base, offsets)
		dirs_conn = self._interp_width_at_offsets(dirs, offsets)
		return base_conn, F.normalize(dirs_conn, dim=-1, eps=1.0e-8)

	def _shell_delta_xyz(self) -> torch.Tensor:
		delta_xy = self._shell_delta_xy_params()
		return torch.cat([delta_xy, torch.zeros_like(delta_xy[..., :1])], dim=-1)

	def _shell_offset_stats(self) -> tuple[float, float, float]:
		with torch.no_grad():
			dist = self._shell_delta_xy_params().norm(dim=-1)
			return (
				float(dist.mean().detach().cpu()),
				float(dist.amin().detach().cpu()),
				float(dist.amax().detach().cpu()),
			)

	def _shell_width_step_stats(self) -> tuple[float, float, float]:
		with torch.no_grad():
			xyz = self.current_cylinder_shell_xyz().detach()
			w_len = (torch.roll(xyz, shifts=-1, dims=1) - xyz).norm(dim=-1)
			return (
				float(w_len.mean().detach().cpu()),
				float(w_len.amin().detach().cpu()),
				float(w_len.amax().detach().cpu()),
			)

	@staticmethod
	def _fmt_xyz(p: torch.Tensor) -> str:
		p = p.detach().cpu()
		return f"({float(p[0]):.1f},{float(p[1]):.1f},{float(p[2]):.1f})"

	@staticmethod
	def _shell_width_edge_extrema_str(shell: torch.Tensor) -> tuple[str, str]:
		w_len = (torch.roll(shell, shifts=-1, dims=1) - shell).norm(dim=-1)
		H = int(w_len.shape[0])
		W = int(w_len.shape[1])

		def _edge_str(name: str, flat_idx_t: torch.Tensor) -> str:
			flat_idx = int(flat_idx_t.detach().cpu())
			h = flat_idx // W
			w0 = flat_idx % W
			w1 = (w0 + 1) % W
			length = float(w_len[h, w0].detach().cpu())
			p0 = Model3D._fmt_xyz(shell[h, w0])
			p1 = Model3D._fmt_xyz(shell[h, w1])
			return f"{name}: h={h}/{H - 1} w={w0}->{w1} len={length:.1f} p0={p0} p1={p1}"

		return _edge_str("min_edge", torch.argmin(w_len)), _edge_str("max_edge", torch.argmax(w_len))

	def current_cylinder_shell_xyz(self) -> torch.Tensor:
		if self.cyl_shell_base is None or self.cyl_shell_dirs is None:
			raise ValueError("umbilicus tube shell has not been prepared")
		delta = self._shell_delta_xyz()
		base, _normal_dirs = self._current_base_conn_and_dirs()
		base = base.to(device=delta.device, dtype=delta.dtype)
		return base + delta

	def current_cylinder_shell_normals(self) -> torch.Tensor:
		return self._unit_vertex_normals_for_shell(self.current_cylinder_shell_xyz())

	def begin_cylinder_shell(self, idx: int, data: fit_data.FitData3D) -> None:
		self.prepare_umbilicus_tube_init(data)
		idx = int(idx)
		if idx < 0 or idx >= int(self.cyl_shell_target_count):
			raise ValueError(f"invalid shell index {idx}")
		device = self.cyl_params.device
		dtype = self.cyl_params.dtype
		if idx == 0:
			if self.cyl_shell_z is None:
				raise ValueError("missing initial shell z values")
			W = self._shell_width_for_radius(self._first_shell_radius())
			base, dirs = self._umbilicus_base_shell(data=data, z=self.cyl_shell_z.to(device=device, dtype=dtype), w=W)
		else:
			if len(self.cyl_shell_completed) < idx:
				raise ValueError(f"cannot start shell {idx}: previous shell is missing")
			prev = self.cyl_shell_completed[idx - 1].to(device=device, dtype=dtype)
			base = prev
			dirs = self._gt_xy_dirs_for_reference_shell(prev, data).to(device=device, dtype=dtype)
			W = int(prev.shape[1])
		H = int(base.shape[0])
		self._set_shell_grid_shape(h=H, w=W)
		self.cyl_shell_base = base.detach()
		self.cyl_shell_dirs = dirs.detach()
		target_offset = self._shell_target_offset_for_index(idx)
		self.cyl_params = nn.Parameter(self._initial_shell_delta_xy(dirs, target_step=target_offset).to(device=device, dtype=dtype))
		self.cyl_shell_w_offsets = nn.Parameter(torch.zeros(H, W, device=device, dtype=dtype))
		self.cyl_shell_current_index = idx
		self.cyl_shell_active = True
		self.cylinder_enabled = True
		self.cyl_shell_mode = True
		step_avg, step_min, step_max = self._shell_offset_stats()
		wstep_avg, wstep_min, wstep_max = self._shell_width_step_stats()
		print(f"[model] shell {idx + 1}/{self.cyl_shell_target_count}: H={H} W={W} "
			  f"base={'umbilicus' if idx == 0 else 'previous'} "
			  f"target_step={target_offset:.1f} "
			  f"offset_avg={step_avg:.1f} min={step_min:.1f} max={step_max:.1f} "
			  f"wstep_avg={wstep_avg:.1f} min={wstep_min:.1f} max={wstep_max:.1f}", flush=True)

	def _resample_shell_width(self, shell: torch.Tensor, target_w: int) -> torch.Tensor:
		target_w = max(3, int(target_w))
		src_w = int(shell.shape[1])
		if src_w == target_w:
			return shell.detach().clone()
		device = shell.device
		dtype = shell.dtype
		phase = torch.arange(target_w, device=device, dtype=dtype) * (float(src_w) / float(target_w))
		i0 = torch.floor(phase).to(dtype=torch.long)
		frac = (phase - i0.to(dtype=dtype)).view(1, target_w, 1)
		i0 = torch.remainder(i0, src_w)
		i1 = torch.remainder(i0 + 1, src_w)
		p0 = shell.index_select(1, i0)
		p1 = shell.index_select(1, i1)
		return (p0 + frac * (p1 - p0)).contiguous().detach()

	def _target_width_for_shell(self, shell: torch.Tensor, data: fit_data.FitData3D) -> int:
		with torch.no_grad():
			umb_xy = data.umbilicus_xy_at_z(shell[..., 2])
			radius = (shell[..., :2] - umb_xy).norm(dim=-1).mean()
			return self._shell_width_for_radius(float(radius.detach().cpu()))

	def complete_current_cylinder_shell(self, data: fit_data.FitData3D) -> None:
		with torch.no_grad():
			step_avg, step_min, step_max = self._shell_offset_stats()
			shell_opt = self.current_cylinder_shell_xyz().detach()
			min_edge_str, max_edge_str = self._shell_width_edge_extrema_str(shell_opt)
			target_w = self._target_width_for_shell(shell_opt, data)
			shell = self._resample_shell_width(shell_opt, target_w)
			idx = int(self.cyl_shell_current_index)
			if len(self.cyl_shell_completed) > idx:
				self.cyl_shell_completed[idx] = shell
			elif len(self.cyl_shell_completed) == idx:
				self.cyl_shell_completed.append(shell)
			else:
				raise ValueError(f"cannot store shell {idx}: shell list has gap")
			self.cyl_shell_active = False
			w_len = (torch.roll(shell, shifts=-1, dims=1) - shell).norm(dim=-1)
			wstep_avg = float(w_len.mean().detach().cpu())
			wstep_min = float(w_len.amin().detach().cpu())
			wstep_max = float(w_len.amax().detach().cpu())
			print(f"[model] completed shell {idx + 1}/{self.cyl_shell_target_count}: "
				  f"H={int(shell.shape[0])} W={int(shell.shape[1])} "
				  f"offset_avg={step_avg:.1f} min={step_min:.1f} max={step_max:.1f} "
				  f"wstep_avg={wstep_avg:.1f} min={wstep_min:.1f} max={wstep_max:.1f}", flush=True)
			print(f"[model] shell {idx + 1} optimized edge extrema: {min_edge_str}; {max_edge_str}",
				  flush=True)

	def cylinder_shells_done(self) -> bool:
		return self.cyl_shell_mode and len(self.cyl_shell_completed) >= int(self.cyl_shell_target_count)

	def fused_cylinder_shell_mesh_flat(self) -> torch.Tensor:
		shells = [s.detach() for s in self.cyl_shell_completed]
		if not shells and self.cyl_shell_base is not None and self.cyl_shell_active:
			shells = [self.current_cylinder_shell_xyz().detach()]
		elif self.cyl_shell_base is not None and self.cyl_shell_active:
			shells = shells + [self.current_cylinder_shell_xyz().detach()]
		if not shells:
			raise ValueError("no umbilicus tube shells available to fuse")
		max_w = max(int(s.shape[1]) for s in shells)
		shells = [self._resample_shell_width(s, max_w) if int(s.shape[1]) != max_w else s.detach().clone()
				  for s in shells]
		shells = [torch.cat([s, s[:, :1]], dim=1).contiguous() for s in shells]
		stack = torch.stack(shells, dim=0)
		return stack.permute(3, 0, 1, 2).contiguous()

	@staticmethod
	def _validate_cyl_params(params: torch.Tensor) -> None:
		if params.ndim != 2 or params.shape[1] != 6:
			raise ValueError(f"cyl_params must have shape (N, 6), got {tuple(params.shape)}")

	@staticmethod
	def _cylinder_frame(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return axis and two perpendicular cross-section basis vectors."""
		Model3D._validate_cyl_params(params)
		tilt_x = params[:, 3]
		tilt_y = params[:, 4]
		roll = params[:, 5]
		axis = torch.stack([tilt_x, tilt_y, torch.ones_like(tilt_x)], dim=-1)
		axis = F.normalize(axis, dim=-1, eps=1.0e-8)

		ax = axis[:, 0]
		ay = axis[:, 1]
		az = axis[:, 2]
		inv = 1.0 / (1.0 + az).clamp(min=1.0e-6)
		b = -ax * ay * inv
		u = torch.stack([1.0 - ax * ax * inv, b, -ax], dim=-1)
		u = F.normalize(u, dim=-1, eps=1.0e-8)
		v = torch.cross(axis, u, dim=-1)
		v = F.normalize(v, dim=-1, eps=1.0e-8)
		cos_r = torch.cos(roll).view(-1, 1)
		sin_r = torch.sin(roll).view(-1, 1)
		u0 = u
		v0 = v
		u = cos_r * u0 + sin_r * v0
		v = -sin_r * u0 + cos_r * v0
		return axis, u, v

	def _cylinder_h_values(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		H = self.mesh_h
		if H <= 1:
			return torch.zeros(1, device=device, dtype=dtype)
		h_extent = (
			float(self.params.model_h)
			if self.params.model_h is not None
			else float(self.params.mesh_step) * float(max(0, H - 1))
		)
		idx = torch.arange(H, device=device, dtype=dtype) - float(H // 2)
		step = h_extent / float(max(1, H - 1))
		return idx * step

	def _cylinder_width_offsets(self, *, params: torch.Tensor) -> torch.Tensor:
		device = params.device
		dtype = params.dtype
		W = self.mesh_w
		idx = torch.arange(W, device=device, dtype=dtype) - float(W // 2)
		width = (
			float(self.params.model_w)
			if self.params.model_w is not None
			else float(self.params.mesh_step) * float(max(0, W - 1))
		)
		step = width / float(max(1, W - 1))
		return idx.view(1, W) * step

	@staticmethod
	def _ellipse_theta_from_arc_offsets(
		*,
		seed_theta: torch.Tensor,
		offsets: torch.Tensor,
		a: torch.Tensor,
		b: torch.Tensor,
		samples: int = 2049,
	) -> torch.Tensor:
		device = seed_theta.device
		dtype = seed_theta.dtype
		N = int(seed_theta.shape[0])
		S = max(17, int(samples))
		if S % 2 == 0:
			S += 1
		theta_grid = torch.linspace(0.0, 2.0 * math.pi, S, device=device, dtype=dtype)
		sin_t = torch.sin(theta_grid).view(1, S)
		cos_t = torch.cos(theta_grid).view(1, S)
		speed = torch.sqrt(
			(a.view(N, 1) * sin_t) ** 2 +
			(b.view(N, 1) * cos_t) ** 2
		).clamp(min=1.0e-8)
		dtheta = (2.0 * math.pi) / float(S - 1)
		seg = 0.5 * (speed[:, 1:] + speed[:, :-1]) * dtheta
		cum = torch.cat([torch.zeros(N, 1, device=device, dtype=dtype), seg.cumsum(dim=1)], dim=1)
		circ = cum[:, -1].clamp(min=1.0e-8)

		def _interp_arc(theta: torch.Tensor) -> torch.Tensor:
			t = torch.remainder(theta, 2.0 * math.pi)
			pos = t * (float(S - 1) / (2.0 * math.pi))
			i0 = torch.floor(pos).to(dtype=torch.long).clamp(min=0, max=S - 2)
			frac = (pos - i0.to(dtype=dtype)).clamp(min=0.0, max=1.0)
			c0 = cum.gather(1, i0.view(N, 1)).squeeze(1)
			c1 = cum.gather(1, (i0 + 1).view(N, 1)).squeeze(1)
			return c0 + frac * (c1 - c0)

		seed_arc = _interp_arc(seed_theta)
		target_arc = seed_arc.view(N, 1) + offsets.to(device=device, dtype=dtype)
		turns = torch.floor(target_arc / circ.view(N, 1))
		target_mod = torch.remainder(target_arc, circ.view(N, 1))
		idx_hi = torch.searchsorted(cum.contiguous(), target_mod.contiguous(), right=False)
		idx_hi = idx_hi.clamp(min=1, max=S - 1)
		idx_lo = idx_hi - 1
		c0 = cum.gather(1, idx_lo)
		c1 = cum.gather(1, idx_hi)
		t0 = theta_grid.gather(0, idx_lo.reshape(-1)).reshape_as(target_mod)
		t1 = theta_grid.gather(0, idx_hi.reshape(-1)).reshape_as(target_mod)
		frac = ((target_mod - c0) / (c1 - c0).clamp(min=1.0e-8)).clamp(min=0.0, max=1.0)
		return t0 + frac * (t1 - t0) + turns * (2.0 * math.pi)

	def _cylinder_samples_for_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return analytic cylinder samples/normals for params shaped (N, 6)."""
		self._validate_cyl_params(params)
		device = params.device
		dtype = params.dtype
		N = int(params.shape[0])
		D = self.depth
		H = self.mesh_h
		W = self.mesh_w

		r = params[:, 0].clamp(min=1.0)
		k = params[:, 1].clamp(min=-0.80, max=0.80)
		seed_theta = params[:, 2]
		axis, u, v = self._cylinder_frame(params)
		a = (r * (1.0 + k)).clamp(min=1.0)
		b = (r * (1.0 - k)).clamp(min=1.0)

		offsets = self._cylinder_width_offsets(params=params).expand(N, W)
		theta = self._ellipse_theta_from_arc_offsets(
			seed_theta=seed_theta,
			offsets=offsets,
			a=a,
			b=b,
		)

		cos_p = torch.cos(theta)
		sin_p = torch.sin(theta)
		local_x = a.view(N, 1) * cos_p
		local_y = b.view(N, 1) * sin_p

		nx_local = cos_p / a.view(N, 1)
		ny_local = sin_p / b.view(N, 1)
		n_len = torch.sqrt(nx_local * nx_local + ny_local * ny_local).clamp(min=1.0e-8)
		nx_local = nx_local / n_len
		ny_local = ny_local / n_len

		seed = self.cyl_seed_xyz.to(device=device, dtype=dtype)
		seed_cos = torch.cos(seed_theta)
		seed_sin = torch.sin(seed_theta)
		seed_radial = a.view(N, 1) * seed_cos.view(N, 1) * u
		seed_radial = seed_radial + b.view(N, 1) * seed_sin.view(N, 1) * v
		center = seed.view(1, 3) - seed_radial

		radial = local_x.view(N, W, 1) * u.view(N, 1, 3)
		radial = radial + local_y.view(N, W, 1) * v.view(N, 1, 3)
		normal = nx_local.view(N, W, 1) * u.view(N, 1, 3)
		normal = normal + ny_local.view(N, W, 1) * v.view(N, 1, 3)
		normal = F.normalize(normal, dim=-1, eps=1.0e-8)

		h_line = self._cylinder_h_values(device=device, dtype=dtype)
		d_offsets = (
			torch.arange(D, device=device, dtype=dtype) - float(D // 2)
		) * float(self.params.winding_step)

		xyz = center.view(N, 1, 1, 1, 3)
		xyz = xyz + h_line.view(1, 1, H, 1, 1) * axis.view(N, 1, 1, 1, 3)
		xyz = xyz + radial.view(N, 1, 1, W, 3)
		xyz = xyz + d_offsets.view(1, D, 1, 1, 1) * normal.view(N, 1, 1, W, 3)
		normal = normal.view(N, 1, 1, W, 3).expand(N, D, H, W, 3)
		return xyz.reshape(N * D, H, W, 3), normal.reshape(N * D, H, W, 3)

	def cylinder_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
		if self.cyl_shell_mode:
			xyz = self.current_cylinder_shell_xyz()
			normal = self.current_cylinder_shell_normals()
			return xyz.unsqueeze(0), normal.unsqueeze(0)
		return self._cylinder_samples_for_params(self.cyl_params)

	def cylinder_centers(self) -> torch.Tensor:
		if self.cyl_shell_mode:
			return torch.empty(0, 3, device=self.cyl_params.device, dtype=self.cyl_params.dtype)
		params = self.cyl_params
		self._validate_cyl_params(params)
		device = params.device
		dtype = params.dtype
		r = params[:, 0].clamp(min=1.0)
		k = params[:, 1].clamp(min=-0.80, max=0.80)
		seed_theta = params[:, 2]
		_axis, u, v = self._cylinder_frame(params)
		a = (r * (1.0 + k)).clamp(min=1.0)
		b = (r * (1.0 - k)).clamp(min=1.0)
		seed = self.cyl_seed_xyz.to(device=device, dtype=dtype)
		seed_radial = a.view(-1, 1) * torch.cos(seed_theta).view(-1, 1) * u
		seed_radial = seed_radial + b.view(-1, 1) * torch.sin(seed_theta).view(-1, 1) * v
		return seed.view(1, 3) - seed_radial

	def cylinder_axes(self) -> torch.Tensor:
		if self.cyl_shell_mode:
			return torch.empty(0, 3, device=self.cyl_params.device, dtype=self.cyl_params.dtype)
		axis, _u, _v = self._cylinder_frame(self.cyl_params)
		return axis

	def _prefetch_cylinder_samples(self, data: fit_data.FitData3D, xyz: torch.Tensor) -> None:
		if not data.sparse_caches:
			return
		pts = xyz.detach()
		for cache in data.sparse_caches.values():
			if not ({"grad_mag", "nx", "ny"} & set(cache.channels)):
				continue
			spacing = data._spacing_for(cache.channels[0])
			cache.prefetch(pts, data.origin_fullres, spacing)
		for cache in data.sparse_caches.values():
			if {"grad_mag", "nx", "ny"} & set(cache.channels):
				cache.sync()

	def cylinder_candidate_errors(self, data: fit_data.FitData3D) -> torch.Tensor:
		"""Detached per-candidate cyl_normal errors used for status and baking."""
		if self.cyl_shell_mode:
			return torch.zeros(1, device=self.cyl_params.device, dtype=self.cyl_params.dtype)
		with torch.no_grad():
			xyz, normals = self.cylinder_samples()
			self._prefetch_cylinder_samples(data, xyz)
			sampled = data.grid_sample_fullres(xyz.detach(), channels={"grad_mag", "nx", "ny"})
			target = sampled.normal_3d
			N = int(self.cyl_params.shape[0])
			if target is None or sampled.grad_mag is None:
				return torch.full((N,), float("inf"), device=xyz.device, dtype=xyz.dtype)
			cyl_xy = normals[..., :2]
			cyl_xy = cyl_xy / cyl_xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
			target_xy_raw = target[..., :2]
			target_xy_len = target_xy_raw.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
			target_xy = target_xy_raw / target_xy_len
			umb_xy = data.umbilicus_xy_at_z(xyz[..., 2])
			radial_xy = xyz[..., :2] - umb_xy
			radial_len = radial_xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
			radial_xy = radial_xy / radial_len
			target_dot_radial = (target_xy * radial_xy).sum(dim=-1)
			target_xy = torch.where(target_dot_radial.unsqueeze(-1) < 0.0, -target_xy, target_xy)
			dot = (cyl_xy * target_xy).sum(dim=-1).clamp(min=-1.0, max=1.0)
			lm = 1.0 - dot
			radial_weight = (target_xy * radial_xy).sum(dim=-1).clamp(min=0.0, max=1.0)
			in_plane_weight = target_xy_raw.norm(dim=-1).clamp(min=0.0, max=1.0)
			valid_normal = ((target_xy_len.squeeze(-1) > 1.0e-7) & (radial_len.squeeze(-1) > 1.0e-7)).to(dtype=lm.dtype)
			mask = (sampled.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype)
			mask = mask * radial_weight * in_plane_weight * valid_normal
			lm_c = lm.reshape(N, self.depth, self.mesh_h, self.mesh_w)
			mask_c = mask.reshape(N, self.depth, self.mesh_h, self.mesh_w)
			wsum = mask_c.sum(dim=(1, 2, 3))
			err = (lm_c * mask_c).sum(dim=(1, 2, 3)) / wsum.clamp(min=1.0)
			return torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))

	def best_cylinder_index(self, data: fit_data.FitData3D | None = None) -> int:
		if self.cyl_shell_mode:
			return 0
		if getattr(self, "cyl_best_idx", None) is not None:
			return max(0, min(int(self.cyl_best_idx), int(self.cyl_params.shape[0]) - 1))
		if data is None:
			return 0
		err = self.cylinder_candidate_errors(data)
		if not torch.isfinite(err).any().detach().cpu().item():
			return 0
		return int(torch.argmin(err).detach().cpu())

	def set_best_cylinder_index(self, idx: int) -> None:
		self.cyl_best_idx = max(0, min(int(idx), int(self.cyl_params.shape[0]) - 1))

	def keep_cylinder_candidates(self, indices: list[int]) -> int:
		if not indices:
			return int(self.cyl_params.shape[0])
		with torch.no_grad():
			idx = torch.tensor(indices, device=self.cyl_params.device, dtype=torch.long)
			idx = idx.clamp(min=0, max=int(self.cyl_params.shape[0]) - 1)
			kept = self.cyl_params.detach().index_select(0, idx).clone()
			self.cyl_params = nn.Parameter(kept)
		self.cyl_best_idx = 0
		return int(self.cyl_params.shape[0])

	def cylinder_mesh_flat(self, *, candidate_idx: int) -> torch.Tensor:
		if self.cyl_shell_mode:
			return self.fused_cylinder_shell_mesh_flat()
		idx = max(0, min(int(candidate_idx), int(self.cyl_params.shape[0]) - 1))
		xyz, _normal = self._cylinder_samples_for_params(self.cyl_params[idx:idx + 1])
		return xyz.reshape(self.depth, self.mesh_h, self.mesh_w, 3).permute(3, 0, 1, 2).contiguous()

	def bake_cylinder_into_mesh(self, data: fit_data.FitData3D | None = None) -> None:
		"""Absorb the lowest-error analytic cylinder candidate into mesh_ms."""
		if not self.cylinder_enabled:
			return
		if self.cyl_shell_mode:
			with torch.no_grad():
				final = self.fused_cylinder_shell_mesh_flat()
				self._set_fused_grid_shape(d=int(final.shape[1]), h=int(final.shape[2]), w=int(final.shape[3]))
				self.mesh_ms = self._construct_pyramid_from_flat_3d(final, len(self.mesh_ms), pyramid_d=self.pyramid_d)
				self.conn_offsets.zero_()
				for ext_off in self._ext_conn_offsets:
					ext_off.zero_()
			self.cylinder_enabled = False
			self.cyl_shell_mode = False
			return
		idx = self.best_cylinder_index(data)
		with torch.no_grad():
			final = self.cylinder_mesh_flat(candidate_idx=idx)
			self.mesh_ms = self._construct_pyramid_from_flat_3d(final, len(self.mesh_ms), pyramid_d=self.pyramid_d)
			self.conn_offsets.zero_()
			for ext_off in self._ext_conn_offsets:
				ext_off.zero_()
		self.cylinder_enabled = False

	def mesh_flat_for_save(self, *, data: fit_data.FitData3D | None = None) -> torch.Tensor:
		if self.cylinder_enabled:
			if self.cyl_shell_mode:
				return self.fused_cylinder_shell_mesh_flat().detach().clone()
			idx = self.best_cylinder_index(data)
			return self.cylinder_mesh_flat(candidate_idx=idx).detach().clone()
		return self.mesh_coarse().detach().clone()

	# --- Arc base positions ---

	def _arc_base_positions(self) -> torch.Tensor:
		"""Compute (3, D, H, W) base positions from arc params. All fullres coords."""
		device = self.arc_cx.device
		D = self.depth
		H = self.mesh_h
		W = self.mesh_w

		# Theta: angular position along arc (width axis)
		theta = torch.linspace(float(self.arc_angle0.detach()), float(self.arc_angle1.detach()),
							   W, device=device, dtype=torch.float32)

		# Radius: each depth layer is a different winding
		# Center D layers around arc_radius
		d_offsets = torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0
		r = self.arc_radius + d_offsets * float(self.params.winding_step)  # (D,)

		# Height: mesh along z axis
		h_extent = float(self.params.mesh_step) * (H - 1)
		z = self.z_center + torch.linspace(-h_extent / 2.0, h_extent / 2.0, H, device=device, dtype=torch.float32)

		# Broadcast to (D, H, W) then stack as (3, D, H, W)
		# x = cx + r * cos(theta)
		# y = cy + r * sin(theta)
		# z = z (height)
		cos_t = torch.cos(theta)  # (W,)
		sin_t = torch.sin(theta)  # (W,)

		x = self.arc_cx + r.view(D, 1, 1) * cos_t.view(1, 1, W)  # (D, 1, W)
		y = self.arc_cy + r.view(D, 1, 1) * sin_t.view(1, 1, W)  # (D, 1, W)
		x = x.expand(D, H, W)
		y = y.expand(D, H, W)
		z_grid = z.view(1, H, 1).expand(D, H, W)

		return torch.stack([x, y, z_grid], dim=0)  # (3, D, H, W)

	# --- Straight base positions ---

	def _straight_base_positions(self) -> torch.Tensor:
		"""Compute (3, D, H, W) base positions from straight params. All fullres coords.

		Width axis maps to positions along a line in XY defined by
		(straight_cx, straight_cy) + t * (cos(angle), sin(angle)).
		Depth axis maps to perpendicular offsets (windings).
		"""
		device = self.straight_cx.device
		D = self.depth
		H = self.mesh_h
		W = self.mesh_w

		# t: position along line direction (width axis)
		t = torch.linspace(
			-float(self.straight_half_w.detach()),
			float(self.straight_half_w.detach()),
			W, device=device, dtype=torch.float32,
		)

		# Line direction and perpendicular
		cos_a = torch.cos(self.straight_angle)
		sin_a = torch.sin(self.straight_angle)

		# Depth offsets perpendicular to line (winding layers)
		d_offsets = torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0
		perp = d_offsets * float(self.params.winding_step)  # (D,)

		# Height: mesh along z axis
		h_extent = float(self.params.mesh_step) * (H - 1)
		z = self.z_center + torch.linspace(-h_extent / 2.0, h_extent / 2.0, H, device=device, dtype=torch.float32)

		# x = cx + t * cos(a) + perp * (-sin(a))
		# y = cy + t * sin(a) + perp * cos(a)
		x = self.straight_cx + t.view(1, 1, W) * cos_a + perp.view(D, 1, 1) * (-sin_a)
		y = self.straight_cy + t.view(1, 1, W) * sin_a + perp.view(D, 1, 1) * cos_a
		x = x.expand(D, H, W)
		y = y.expand(D, H, W)
		z_grid = z.view(1, H, 1).expand(D, H, W)

		return torch.stack([x, y, z_grid], dim=0)  # (3, D, H, W)

	# --- Mesh access ---

	def mesh_coarse(self) -> torch.Tensor:
		"""Integrate residual pyramid -> (3, D, H, W)."""
		return self._integrate_pyramid_3d(self.mesh_ms, pyramid_d=self.pyramid_d)

	def _grid_xyz(self) -> torch.Tensor:
		"""(D, Hm, Wm, 3) mesh positions in fullres voxel coords."""
		if self.cylinder_enabled:
			if self.cyl_shell_mode:
				return self.current_cylinder_shell_xyz().unsqueeze(0)
			idx = self.best_cylinder_index(None)
			xyz, _normal = self._cylinder_samples_for_params(self.cyl_params[idx:idx + 1])
			return xyz.reshape(self.depth, self.mesh_h, self.mesh_w, 3)
		residuals = self.mesh_coarse()  # (3, D, H, W)
		if self.arc_enabled:
			base = self._arc_base_positions()
			xyz = base + residuals
		elif self.straight_enabled:
			base = self._straight_base_positions()
			xyz = base + residuals
		else:
			xyz = residuals
		return xyz.permute(1, 2, 3, 0)  # (D, H, W, 3)

	def _grid_xyz_hr(self, xyz_lr: torch.Tensor) -> torch.Tensor:
		"""Bilinear upsample H,W only. D stays at LR.

		xyz_lr: (D, Hm, Wm, 3) -> (D, He, We, 3)
		"""
		He = max(2, (self.mesh_h - 1) * int(self.params.subsample_mesh) + 1)
		We = max(2, (self.mesh_w - 1) * int(self.params.subsample_winding) + 1)
		t = xyz_lr.permute(0, 3, 1, 2)  # (D, 3, Hm, Wm)
		t = F.interpolate(t, size=(He, We), mode='bilinear', align_corners=True)
		return t.permute(0, 2, 3, 1)  # (D, He, We, 3)

	# --- Connection vectors ---

	@staticmethod
	def _vertex_normals(xyz_lr: torch.Tensor) -> torch.Tensor:
		"""Compute per-vertex normals via central differences.

		xyz_lr: (D, Hm, Wm, 3) -> normals: (D, Hm, Wm, 3), unit length.
		"""
		# Edge vectors along H (central diff, forward/backward at boundaries)
		edge_h = torch.zeros_like(xyz_lr)
		edge_h[:, 1:-1] = xyz_lr[:, 2:] - xyz_lr[:, :-2]
		edge_h[:, 0] = xyz_lr[:, 1] - xyz_lr[:, 0]
		edge_h[:, -1] = xyz_lr[:, -1] - xyz_lr[:, -2]
		# Edge vectors along W
		edge_w = torch.zeros_like(xyz_lr)
		edge_w[:, :, 1:-1] = xyz_lr[:, :, 2:] - xyz_lr[:, :, :-2]
		edge_w[:, :, 0] = xyz_lr[:, :, 1] - xyz_lr[:, :, 0]
		edge_w[:, :, -1] = xyz_lr[:, :, -1] - xyz_lr[:, :, -2]
		# Normal = cross(edge_h, edge_w) — unnormalized is fine because the
		# ray-bilinear-patch quadratic coefficients scale as |n|², so u,v are
		# scale-invariant.  Normalizing introduces sqrt(0) grad issues.
		n = torch.cross(edge_h, edge_w, dim=-1)
		return n

	def _xyz_conn(self, xyz_lr: torch.Tensor, data: fit_data.FitData3D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Compute connection points to neighbor depth slices.

		Returns:
			xy_conn: (D, Hm, Wm, 3, 3) — [prev_conn, self, next_conn], each 3D fullres.
			mask_conn: (D, 1, Hm, Wm, 3) — validity per connection point.
		"""
		D, Hm, Wm, _ = xyz_lr.shape
		device = xyz_lr.device
		normals = self._vertex_normals(xyz_lr).detach()  # (D, Hm, Wm, 3) — constant for grad

		# conn_offsets: (4, D, Hm, Wm) — [prev_h, prev_w, next_h, next_w]
		prev_h_off = self.conn_offsets[0]
		prev_w_off = self.conn_offsets[1]
		next_h_off = self.conn_offsets[2]
		next_w_off = self.conn_offsets[3]

		def _intersect_direction(src_xyz: torch.Tensor, src_n: torch.Tensor, nb_xyz: torch.Tensor, h_off: torch.Tensor, w_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
			"""Ray-bilinear-patch intersection for one direction.

			src_xyz: (B, Hm, Wm, 3) — source vertex positions (ray origins).
			src_n: (B, Hm, Wm, 3) — source vertex normals (ray directions).
			nb_xyz: (B, Hm, Wm, 3) — neighbor slice positions.
			h_off, w_off: (B, Hm, Wm) — offsets.

			Returns: (conn_pt, u, v, row, col, valid) where conn_pt is (B, Hm, Wm, 3)
			         and valid is (B, Hm, Wm) combining bounds and UV checks.
			"""
			B = src_xyz.shape[0]
			h_idx_b = torch.arange(Hm, device=device, dtype=torch.float32).view(1, Hm, 1).expand(B, Hm, Wm)
			w_idx_b = torch.arange(Wm, device=device, dtype=torch.float32).view(1, 1, Wm).expand(B, Hm, Wm)

			target_h = h_idx_b + h_off
			target_w = w_idx_b + w_off
			row = target_h.floor().clamp(0, Hm - 2).long()
			col = target_w.floor().clamp(0, Wm - 2).long()
			frac_h = target_h - row.float()
			frac_w = target_w - col.float()

			# Gather quad corners from neighbor slice: P00, P10, P01, P11
			d_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, Hm, Wm)
			P00 = nb_xyz[d_idx, row, col]              # (B, Hm, Wm, 3)
			P10 = nb_xyz[d_idx, row + 1, col]
			P01 = nb_xyz[d_idx, row, col + 1]
			P11 = nb_xyz[d_idx, row + 1, col + 1]

			# Ray: O + s*n, Patch: Q(u,v) = (1-u)(1-v)*P00 + u(1-v)*P10 + (1-u)*v*P01 + u*v*P11
			# Rearrange: Q(u,v) = P00 + u*a + v*b + u*v*c  where:
			O = src_xyz
			n = src_n
			a = P10 - P00
			b = P01 - P00
			c = P11 - P10 - P01 + P00
			g = P00 - O

			# 2D cross product for axis pair (i, j): vec_i * n_j - vec_j * n_i
			def cross2(vec: torch.Tensor, i: int, j: int) -> torch.Tensor:
				return vec[..., i] * n[..., j] - vec[..., j] * n[..., i]

			# All three axis pair projections: (X,Y), (X,Z), (Y,Z)
			# Each gives: G_k + u*A_k + v*(B_k + u*C_k) = 0
			Ap = [cross2(a, 0, 1), cross2(a, 0, 2), cross2(a, 1, 2)]
			Bp = [cross2(b, 0, 1), cross2(b, 0, 2), cross2(b, 1, 2)]
			Cp = [cross2(c, 0, 1), cross2(c, 0, 2), cross2(c, 1, 2)]
			Gp = [cross2(g, 0, 1), cross2(g, 0, 2), cross2(g, 1, 2)]

			# Quadratic in u from eliminating v between two projections.
			# Three possible pairs; pick the best-conditioned (largest |alpha|).
			qpairs = [(0, 1), (0, 2), (1, 2)]
			alphas = []
			betas_q = []
			gammas = []
			for p, q in qpairs:
				alphas.append(Ap[p] * Cp[q] - Ap[q] * Cp[p])
				betas_q.append(Ap[p] * Bp[q] - Ap[q] * Bp[p] + Gp[p] * Cp[q] - Gp[q] * Cp[p])
				gammas.append(Gp[p] * Bp[q] - Gp[q] * Bp[p])

			abs_a = [aa.abs() for aa in alphas]
			sel_q0 = (abs_a[0] >= abs_a[1]) & (abs_a[0] >= abs_a[2])
			sel_q1 = (~sel_q0) & (abs_a[1] >= abs_a[2])

			alpha = torch.where(sel_q0, alphas[0], torch.where(sel_q1, alphas[1], alphas[2]))
			beta = torch.where(sel_q0, betas_q[0], torch.where(sel_q1, betas_q[1], betas_q[2]))
			gamma = torch.where(sel_q0, gammas[0], torch.where(sel_q1, gammas[1], gammas[2]))

			eps = 1e-12
			# Discriminant
			disc = beta * beta - 4.0 * alpha * gamma
			disc_safe = disc.clamp(min=0.0)
			sqrt_disc = torch.sqrt(disc_safe + 1e-12)

			# Two solutions
			alpha_abs = alpha.abs()
			is_linear = alpha_abs < eps

			# Quadratic solutions
			u1 = (-beta + sqrt_disc) / (2.0 * alpha + eps * is_linear.float())
			u2 = (-beta - sqrt_disc) / (2.0 * alpha + eps * is_linear.float())
			# Linear fallback: u = -gamma / beta
			u_lin = -gamma / (beta + eps * (beta.abs() < eps).float())

			u1 = torch.where(is_linear, u_lin, u1)
			u2 = torch.where(is_linear, u_lin, u2)

			# Pick u closest to frac_h (stored offset hint)
			u = torch.where((u1 - frac_h).abs() <= (u2 - frac_h).abs(), u1, u2)

			# Recover v from the best-conditioned projection (largest |denom_v|)
			denom_v = [Bp[k] + u * Cp[k] for k in range(3)]
			numer_v = [-(Gp[k] + u * Ap[k]) for k in range(3)]
			abs_dv = [d.abs() for d in denom_v]

			sel_v0 = (abs_dv[0] >= abs_dv[1]) & (abs_dv[0] >= abs_dv[2])
			sel_v1 = (~sel_v0) & (abs_dv[1] >= abs_dv[2])

			dv = torch.where(sel_v0, denom_v[0], torch.where(sel_v1, denom_v[1], denom_v[2]))
			nv = torch.where(sel_v0, numer_v[0], torch.where(sel_v1, numer_v[1], numer_v[2]))
			v = nv / (dv + eps * (dv.abs() < eps).float())

			# Connection point: Q(u, v) = P00 + u*a + v*b + u*v*c
			conn_pt = P00 + u.unsqueeze(-1) * a + v.unsqueeze(-1) * b + (u * v).unsqueeze(-1) * c

			# Sign of ray parameter s: positive = forward along normal, negative = backward
			s_sign = ((conn_pt - O) * n).sum(dim=-1).sign()  # (B, Hm, Wm)
			s_sign = torch.where(s_sign == 0, torch.ones_like(s_sign), s_sign)

			# Combined validity: target must be in mesh bounds AND u,v in [0,1]
			in_bounds = (target_h >= 0) & (target_h <= Hm - 1) & (target_w >= 0) & (target_w <= Wm - 1)
			uv_ok = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
			valid = (in_bounds & uv_ok).to(dtype=src_xyz.dtype)

			return conn_pt, u, v, row, col, valid, s_sign

		# --- Compute connections for prev (d-1) and next (d+1) ---
		# Built via cat/stack for clean autograd (no in-place version tracking).
		self._conn_params: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

		if D >= 2:
			# Prev connections (d -> d-1): source is slices [1:], neighbor is [:-1]
			prev_conn, prev_u, prev_v, prev_row, prev_col, prev_valid, prev_s_sign = _intersect_direction(
				xyz_lr[1:], normals[1:], xyz_lr[:-1], prev_h_off[1:], prev_w_off[1:]
			)
			self._conn_params["prev"] = (prev_u, prev_v, prev_row, prev_col)

			# Next connections (d -> d+1): source is slices [:-1], neighbor is [1:]
			next_conn, next_u, next_v, next_row, next_col, next_valid, next_s_sign = _intersect_direction(
				xyz_lr[:-1], normals[:-1], xyz_lr[1:], next_h_off[:-1], next_w_off[:-1]
			)
			self._conn_params["next"] = (next_u, next_v, next_row, next_col)

			# Boundary fallback: mirror the valid direction (detached)
			boundary_prev = (2.0 * xyz_lr[0] - next_conn[0]).detach()
			boundary_next = (2.0 * xyz_lr[-1] - prev_conn[-1]).detach()

			prev_full = torch.cat([boundary_prev.unsqueeze(0), prev_conn], dim=0)   # (D, Hm, Wm, 3)
			next_full = torch.cat([next_conn, boundary_next.unsqueeze(0)], dim=0)   # (D, Hm, Wm, 3)

			xy_conn = torch.stack([prev_full, xyz_lr, next_full], dim=-1)  # (D, Hm, Wm, 3, 3)

			# Sign of ray parameter at intersection: +1 forward, -1 backward along normal
			ones_boundary = torch.ones(1, Hm, Wm, device=device, dtype=xyz_lr.dtype)
			prev_sign_full = torch.cat([ones_boundary, prev_s_sign], dim=0)   # (D, Hm, Wm)
			next_sign_full = torch.cat([next_s_sign, ones_boundary], dim=0)   # (D, Hm, Wm)
			sign_conn = torch.stack([prev_sign_full, next_sign_full], dim=-1).unsqueeze(1)  # (D, 1, Hm, Wm, 2)

			# Intersection validity: bounds + UV combined (boundary slices get zeros)
			zeros = torch.zeros(1, Hm, Wm, device=device, dtype=xyz_lr.dtype)
			prev_uv_ok = torch.cat([zeros, prev_valid], dim=0)  # (D, Hm, Wm)
			next_uv_ok = torch.cat([next_valid, zeros], dim=0)

			# Connection masks: sample validity AND patch intersection validity
			def _valid_mask(gm: torch.Tensor) -> torch.Tensor:
				return (gm.squeeze(0).squeeze(0) > 0.0).to(dtype=xyz_lr.dtype).unsqueeze(1)

			mask_prev = _valid_mask(data.grid_sample_fullres(prev_full.detach(), channels={"grad_mag"}).grad_mag)
			mask_center = _valid_mask(data.grid_sample_fullres(xyz_lr.detach(), channels={"grad_mag"}).grad_mag)
			mask_next = _valid_mask(data.grid_sample_fullres(next_full.detach(), channels={"grad_mag"}).grad_mag)

			# Apply uv validity (also zeros boundary edges: d=0 prev, d=D-1 next)
			mask_prev = mask_prev * prev_uv_ok.unsqueeze(1)
			mask_next = mask_next * next_uv_ok.unsqueeze(1)

			mask_conn = torch.stack([mask_prev, mask_center, mask_next], dim=-1)  # (D, 1, Hm, Wm, 3)
		else:
			# D=1: no connections possible
			zeros = torch.zeros_like(xyz_lr)
			xy_conn = torch.stack([zeros, xyz_lr, zeros], dim=-1)
			mask_conn = torch.zeros(D, 1, Hm, Wm, 3, device=device, dtype=xyz_lr.dtype)
			sign_conn = torch.ones(D, 1, Hm, Wm, 2, device=device, dtype=xyz_lr.dtype)

		return xy_conn, mask_conn, sign_conn, normals

	def update_conn_offsets(self) -> None:
		"""Update conn_offsets buffer from last intersection parameters. Call after opt.step()."""
		params = getattr(self, "_conn_params", None)
		if params is None or self.depth < 2:
			return
		D = self.depth
		Hm = self.mesh_h
		Wm = self.mesh_w
		device = self.conn_offsets.device

		h_idx = torch.arange(Hm, device=device, dtype=torch.float32).view(1, Hm, 1).expand(D, Hm, Wm)
		w_idx = torch.arange(Wm, device=device, dtype=torch.float32).view(1, 1, Wm).expand(D, Hm, Wm)

		with torch.no_grad():
			if "prev" in params:
				u, v, row, col = params["prev"]
				# These cover slices [1:D]
				self.conn_offsets[0, 1:] = row.float() + u - h_idx[1:]
				self.conn_offsets[1, 1:] = col.float() + v - w_idx[1:]
			if "next" in params:
				u, v, row, col = params["next"]
				# These cover slices [0:D-1]
				self.conn_offsets[2, :-1] = row.float() + u - h_idx[:-1]
				self.conn_offsets[3, :-1] = col.float() + v - w_idx[:-1]
			# Degenerate quadratic solves can produce NaN; sanitize to avoid
			# garbage indices from NaN.long() in the next forward pass.
			self.conn_offsets.nan_to_num_(0.0)

	# --- External surface support ---

	@staticmethod
	def _compute_ext_normals(xyz: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
		"""Compute unit normals for an external surface using central differences.

		Uses one-sided differences at boundaries and next to invalid vertices.
		xyz: (H, W, 3), valid: (H, W) bool → (H, W, 3) unit normals (zero at invalid).
		"""
		H, W, _ = xyz.shape
		# h-tangent: fwd + bwd where both neighbors valid → central diff
		fwd_h = torch.zeros_like(xyz)
		bwd_h = torch.zeros_like(xyz)
		if H > 1:
			diff_h = xyz[1:] - xyz[:-1]
			pair_h = (valid[1:] & valid[:-1]).unsqueeze(-1)
			diff_h = torch.where(pair_h, diff_h, torch.zeros_like(diff_h))
			fwd_h[:-1] = diff_h
			bwd_h[1:] = diff_h
		dh = fwd_h + bwd_h
		# w-tangent
		fwd_w = torch.zeros_like(xyz)
		bwd_w = torch.zeros_like(xyz)
		if W > 1:
			diff_w = xyz[:, 1:] - xyz[:, :-1]
			pair_w = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
			diff_w = torch.where(pair_w, diff_w, torch.zeros_like(diff_w))
			fwd_w[:, :-1] = diff_w
			bwd_w[:, 1:] = diff_w
		dw = fwd_w + bwd_w
		n = torch.cross(dh, dw, dim=-1)
		n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
		n[~valid] = 0.0
		return n

	@staticmethod
	def _ray_bilinear_intersect(O: torch.Tensor, n: torch.Tensor,
								M00: torch.Tensor, M10: torch.Tensor,
								M01: torch.Tensor, M11: torch.Tensor,
								frac_h: torch.Tensor, frac_w: torch.Tensor,
								) -> tuple[torch.Tensor, torch.Tensor]:
		"""Ray-bilinear-patch intersection.

		Shoots ray from O along direction n, intersects with bilinear patch
		defined by corners M00, M10, M01, M11.  frac_h/frac_w are used to
		disambiguate the two quadratic roots (pick closer to expected).

		Returns (u, v) — unclamped intersection parameters.
		All inputs: (..., 3) for points/directions, (...) for frac_h/frac_w.
		"""
		eps = 1e-12
		a = M10 - M00
		b = M01 - M00
		c = M11 - M10 - M01 + M00
		g = M00 - O

		def cross2(vec: torch.Tensor, i: int, j: int) -> torch.Tensor:
			return vec[..., i] * n[..., j] - vec[..., j] * n[..., i]

		Ap = [cross2(a, 0, 1), cross2(a, 0, 2), cross2(a, 1, 2)]
		Bp = [cross2(b, 0, 1), cross2(b, 0, 2), cross2(b, 1, 2)]
		Cp = [cross2(c, 0, 1), cross2(c, 0, 2), cross2(c, 1, 2)]
		Gp = [cross2(g, 0, 1), cross2(g, 0, 2), cross2(g, 1, 2)]

		qpairs = [(0, 1), (0, 2), (1, 2)]
		alphas, betas_q, gammas = [], [], []
		for p, q in qpairs:
			alphas.append(Ap[p] * Cp[q] - Ap[q] * Cp[p])
			betas_q.append(Ap[p] * Bp[q] - Ap[q] * Bp[p] + Gp[p] * Cp[q] - Gp[q] * Cp[p])
			gammas.append(Gp[p] * Bp[q] - Gp[q] * Bp[p])

		abs_a = [aa.abs() for aa in alphas]
		sel_q0 = (abs_a[0] >= abs_a[1]) & (abs_a[0] >= abs_a[2])
		sel_q1 = (~sel_q0) & (abs_a[1] >= abs_a[2])
		alpha = torch.where(sel_q0, alphas[0], torch.where(sel_q1, alphas[1], alphas[2]))
		beta = torch.where(sel_q0, betas_q[0], torch.where(sel_q1, betas_q[1], betas_q[2]))
		gamma = torch.where(sel_q0, gammas[0], torch.where(sel_q1, gammas[1], gammas[2]))

		disc = (beta * beta - 4.0 * alpha * gamma).clamp(min=0.0)
		sqrt_disc = torch.sqrt(disc + eps)
		is_linear = alpha.abs() < eps
		u1 = (-beta + sqrt_disc) / (2.0 * alpha + eps * is_linear.float())
		u2 = (-beta - sqrt_disc) / (2.0 * alpha + eps * is_linear.float())
		u_lin = -gamma / (beta + eps * (beta.abs() < eps).float())
		u1 = torch.where(is_linear, u_lin, u1)
		u2 = torch.where(is_linear, u_lin, u2)
		u = torch.where((u1 - frac_h).abs() <= (u2 - frac_h).abs(), u1, u2)

		denom_v = [Bp[k] + u * Cp[k] for k in range(3)]
		numer_v = [-(Gp[k] + u * Ap[k]) for k in range(3)]
		abs_dv = [d.abs() for d in denom_v]
		sel_v0 = (abs_dv[0] >= abs_dv[1]) & (abs_dv[0] >= abs_dv[2])
		sel_v1 = (~sel_v0) & (abs_dv[1] >= abs_dv[2])
		dv = torch.where(sel_v0, denom_v[0], torch.where(sel_v1, denom_v[1], denom_v[2]))
		nv = torch.where(sel_v0, numer_v[0], torch.where(sel_v1, numer_v[1], numer_v[2]))
		v = nv / (dv + eps * (dv.abs() < eps).float())

		return u, v

	def add_external_surface(self, xyz: torch.Tensor, valid: torch.Tensor | None = None,
						   offset: float = 1.0) -> int:
		"""Register a frozen external reference surface.

		xyz: (H_ext, W_ext, 3) float32 mesh positions in fullres coords.
		valid: (H_ext, W_ext) bool — validity mask. None = all valid.
		offset: target grad_mag integral from model surface to this external surface.
		Returns the index of the added surface.
		"""
		dev = self.conn_offsets.device
		idx = len(self._ext_surfaces)
		if valid is None:
			valid = torch.ones(xyz.shape[0], xyz.shape[1], dtype=torch.bool, device=dev)
		valid_dev = valid.to(device=dev)
		xyz_dev = xyz.detach().to(device=dev)
		finite_xyz = torch.isfinite(xyz_dev).all(dim=-1)
		valid_dev = valid_dev & finite_xyz
		xyz_dev = torch.where(
			valid_dev.unsqueeze(-1),
			xyz_dev,
			torch.full_like(xyz_dev, float("nan")),
		)
		self._ext_surfaces.append(xyz_dev)
		self._ext_valid.append(valid_dev)
		self._ext_normals.append(Model3D._compute_ext_normals(xyz_dev, valid_dev))
		H_ext, W_ext = int(xyz.shape[0]), int(xyz.shape[1])
		self._ext_conn_offsets.append(
			torch.zeros(2, self.depth, H_ext, W_ext,
						device=dev, dtype=torch.float32))
		self._ext_conn_params.append({})
		self._ext_offsets.append(float(offset))
		return idx

	def update_ext_conn_offsets(self) -> None:
		"""Two-pass intersection update using current (post-step) model params.

		Pass 1: intersect with current quad → raw (u, v), unclamped.
		Pass 2: shift quad idx based on pass-1 result (clamped), re-intersect
		        with new quad → final (u, v), unclamped.
		Store offset from updated quad. The forward pass masks by uv ∈ [0,1].
		"""
		if not self._ext_surfaces:
			return
		Hm = self.mesh_h
		Wm = self.mesh_w
		D = self.depth
		xyz_lr = self._grid_xyz().detach()  # current post-step positions
		device = xyz_lr.device

		with torch.no_grad():
			for i, ext_xyz in enumerate(self._ext_surfaces):
				H_ext, W_ext, _ = ext_xyz.shape
				ext_off = self._ext_conn_offsets[i]
				h_off, w_off = ext_off[0], ext_off[1]
				ext_norms = self._ext_normals[i]
				ext_corner_valid_2d = (
					self._ext_valid[i] &
					torch.isfinite(ext_xyz).all(dim=-1) &
					torch.isfinite(ext_norms).all(dim=-1) &
					(ext_norms.norm(dim=-1) > 1e-8)
				)
				ext_corner_valid = ext_corner_valid_2d.unsqueeze(0).expand(D, -1, -1)

				r_idx = torch.arange(H_ext, device=device, dtype=torch.float32).view(1, H_ext, 1).expand(D, H_ext, W_ext)
				c_idx = torch.arange(W_ext, device=device, dtype=torch.float32).view(1, 1, W_ext).expand(D, H_ext, W_ext)
				d_idx = torch.arange(D, device=device).view(D, 1, 1).expand(D, H_ext, W_ext)
				ext_P = ext_xyz.unsqueeze(0).expand(D, -1, -1, -1)
				ext_N = ext_norms.unsqueeze(0).expand(D, -1, -1, -1)

				# Current model grid position from stored offset
				model_h = r_idx + h_off
				model_w = c_idx + w_off
				row = model_h.floor().clamp(0, Hm - 2).long()
				col = model_w.floor().clamp(0, Wm - 2).long()
				frac_h = (model_h - row.float()).clamp(0, 1)
				frac_w = (model_w - col.float()).clamp(0, 1)

				# Gather model quad at current idx
				M00 = xyz_lr[d_idx, row, col]
				M10 = xyz_lr[d_idx, (row + 1).clamp(max=Hm - 1), col]
				M01 = xyz_lr[d_idx, row, (col + 1).clamp(max=Wm - 1)]
				M11 = xyz_lr[d_idx, (row + 1).clamp(max=Hm - 1), (col + 1).clamp(max=Wm - 1)]

				# PASS 1: intersect → raw (u, v), unclamped
				u1, v1 = Model3D._ray_bilinear_intersect(
					ext_P, ext_N, M00, M10, M01, M11, frac_h, frac_w)

				# Update idx: shift quad based on pass-1 result, clamp to valid range
				pass1_valid = ext_corner_valid & torch.isfinite(u1) & torch.isfinite(v1)
				new_model_h_raw = row.float() + u1
				new_model_w_raw = col.float() + v1
				new_model_h = torch.where(pass1_valid, new_model_h_raw, torch.zeros_like(new_model_h_raw)).clamp(0, Hm - 2)
				new_model_w = torch.where(pass1_valid, new_model_w_raw, torch.zeros_like(new_model_w_raw)).clamp(0, Wm - 2)
				new_row = new_model_h.floor().clamp(0, Hm - 2).long()
				new_col = new_model_w.floor().clamp(0, Wm - 2).long()
				new_frac_h = new_model_h - new_row.float()
				new_frac_w = new_model_w - new_col.float()

				# Gather model quad at updated idx
				M00 = xyz_lr[d_idx, new_row, new_col]
				M10 = xyz_lr[d_idx, (new_row + 1).clamp(max=Hm - 1), new_col]
				M01 = xyz_lr[d_idx, new_row, (new_col + 1).clamp(max=Wm - 1)]
				M11 = xyz_lr[d_idx, (new_row + 1).clamp(max=Hm - 1), (new_col + 1).clamp(max=Wm - 1)]

				# PASS 2: re-intersect → final (u, v), unclamped
				u2, v2 = Model3D._ray_bilinear_intersect(
					ext_P, ext_N, M00, M10, M01, M11, new_frac_h, new_frac_w)

				# Store offset from updated position
				new_h_off = new_row.float() + u2 - r_idx
				new_w_off = new_col.float() + v2 - c_idx
				# Clamp so model_h stays in valid quad range [0, Hm-2]
				new_h_off = (r_idx + new_h_off).clamp(0, Hm - 2) - r_idx
				new_w_off = (c_idx + new_w_off).clamp(0, Wm - 2) - c_idx
				update_valid = pass1_valid & torch.isfinite(new_h_off) & torch.isfinite(new_w_off)
				zeros = torch.zeros_like(new_h_off)
				self._ext_conn_offsets[i][0] = torch.where(update_valid, new_h_off, zeros)
				self._ext_conn_offsets[i][1] = torch.where(update_valid, new_w_off, zeros)

	@staticmethod
	def from_tifxyz_crop(xyz: torch.Tensor, valid: torch.Tensor, *,
						 device: torch.device, mesh_step: int = 100,
						 winding_step: int = 25, subsample_mesh: int = 4,
						 subsample_winding: int = 4) -> "Model3D":
		"""Create a depth=1 model from pre-cropped tifxyz tensors.

		xyz: (H, W, 3) float32 — invalid vertices should already be zeroed.
		valid: (H, W) bool — True for valid vertices.

		Invalid vertices are inpainted via masked scale-space pyramid reconstruction.
		"""
		H, W, _ = xyz.shape
		import math
		mdl = Model3D(
			device=device, depth=1, mesh_h=H, mesh_w=W,
			mesh_step=mesh_step, winding_step=winding_step,
			subsample_mesh=subsample_mesh, subsample_winding=subsample_winding,
			arc_cx=0.0, arc_cy=0.0, arc_radius=1000.0,
			arc_angle0=-0.5, arc_angle1=0.5,
			init_mode="arc", pyramid_d=False,
		)
		flat_mesh = xyz.to(device=device).permute(2, 0, 1).unsqueeze(1)  # (3, 1, H, W)
		valid_dev = valid.to(device=device)
		mdl.arc_enabled = False
		mdl.straight_enabled = False
		# Inpaint invalid vertices via masked pyramid, then rebuild with
		# the standard (non-masked) constructor so the pyramid has the same
		# number of scales and residual distribution as checkpoint-loaded models.
		n_inpaint = max(5, int(math.ceil(math.log2(max(H, W)))) + 1)
		inpaint_ms = Model3D._construct_pyramid_from_flat_3d_masked(
			flat_mesh, valid_dev, n_scales=n_inpaint, pyramid_d=mdl.pyramid_d)
		with torch.no_grad():
			flat_inpainted = Model3D._integrate_pyramid_3d(inpaint_ms, pyramid_d=mdl.pyramid_d)
		n_scales = len(mdl.mesh_ms)  # standard scale count from constructor
		mdl.mesh_ms = Model3D._construct_pyramid_from_flat_3d(
			flat_inpainted, n_scales, pyramid_d=mdl.pyramid_d)
		return mdl

	@staticmethod
	def from_tifxyz(path: str, *, device: torch.device, mesh_step: int = 100,
					winding_step: int = 25, subsample_mesh: int = 4,
					subsample_winding: int = 4) -> "Model3D":
		"""Create a depth=1 model initialized from a tifxyz directory.

		Invalid vertices (VC3D sentinel -1,-1,-1) are inpainted via
		masked scale-space pyramid reconstruction.
		"""
		from tifxyz_io import load_tifxyz
		xyz, valid, meta = load_tifxyz(path, device=device)
		# Derive mesh_step from meta scale if available
		scale = meta.get("scale")
		if scale is not None and isinstance(scale, list) and len(scale) >= 1 and float(scale[0]) > 0:
			mesh_step = max(1, int(round(1.0 / float(scale[0]))))
		return Model3D.from_tifxyz_crop(
			xyz, valid, device=device, mesh_step=mesh_step,
			winding_step=winding_step, subsample_mesh=subsample_mesh,
			subsample_winding=subsample_winding)

	def _intersect_ext_surfaces(self, xyz_lr: torch.Tensor, data: fit_data.FitData3D
							   ) -> list | None:
		"""Intersect ext surface corners with model quads.

		For each ext surface corner, use stored per-corner offset to find the
		corresponding model quad, then ray-bilinear-patch intersect to get precise
		(u, v) within that model quad.

		Returns list of (mask, offset, ext_P, ext_N, full_h, full_w) per surface,
		with shapes (D, H_ext, W_ext, ...).
		full_h/full_w = row + u, col + v — continuous model grid position.
		Model quad corners are re-gathered from xyz_lr in the loss function.
		"""
		if not self._ext_surfaces:
			return None
		D, Hm, Wm, _ = xyz_lr.shape
		device = xyz_lr.device
		results = []

		for ei, ext_xyz in enumerate(self._ext_surfaces):
			H_ext, W_ext, _ = ext_xyz.shape
			offset = self._ext_offsets[ei]
			ext_off = self._ext_conn_offsets[ei]  # (2, D, H_ext, W_ext)
			h_off = ext_off[0]  # (D, H_ext, W_ext)
			w_off = ext_off[1]
			ext_norms = self._ext_normals[ei]  # (H_ext, W_ext, 3)
			ext_corner_valid_2d = (
				self._ext_valid[ei] &
				torch.isfinite(ext_xyz).all(dim=-1) &
				torch.isfinite(ext_norms).all(dim=-1) &
				(ext_norms.norm(dim=-1) > 1e-8)
			)
			if H_ext > 1 and W_ext > 1:
				ext_quad_valid_2d = (
					ext_corner_valid_2d[:-1, :-1] &
					ext_corner_valid_2d[1:, :-1] &
					ext_corner_valid_2d[:-1, 1:] &
					ext_corner_valid_2d[1:, 1:]
				)
				ext_corner_used_2d = torch.zeros_like(ext_corner_valid_2d)
				ext_corner_used_2d[:-1, :-1] |= ext_quad_valid_2d
				ext_corner_used_2d[1:, :-1] |= ext_quad_valid_2d
				ext_corner_used_2d[:-1, 1:] |= ext_quad_valid_2d
				ext_corner_used_2d[1:, 1:] |= ext_quad_valid_2d
			else:
				ext_quad_valid_2d = torch.zeros(
					max(0, H_ext - 1), max(0, W_ext - 1),
					device=device, dtype=torch.bool)
				ext_corner_used_2d = torch.zeros_like(ext_corner_valid_2d)

			# Ext corner grid indices
			r_idx = torch.arange(H_ext, device=device, dtype=torch.float32).view(1, H_ext, 1).expand(D, H_ext, W_ext)
			c_idx = torch.arange(W_ext, device=device, dtype=torch.float32).view(1, 1, W_ext).expand(D, H_ext, W_ext)

			# Map ext corners to model grid positions
			model_h = r_idx + h_off  # (D, H_ext, W_ext) — unclamped
			model_w = c_idx + w_off

			# In-bounds mask (before clamping) — strict: need valid quad at (row, row+1)
			in_bounds = (model_h >= 0) & (model_h < Hm - 1) & (model_w >= 0) & (model_w < Wm - 1)

			# Clamp for safe indexing (clamped entries will be masked out)
			model_h_c = model_h.clamp(0, Hm - 1)
			model_w_c = model_w.clamp(0, Wm - 1)
			row = model_h_c.floor().clamp(0, Hm - 2).long()
			col = model_w_c.floor().clamp(0, Wm - 2).long()
			frac_h = model_h_c - row.float()
			frac_w = model_w_c - col.float()

			# Gather model quad corners (detached for intersection only)
			d_idx = torch.arange(D, device=device).view(D, 1, 1).expand(D, H_ext, W_ext)
			M00 = xyz_lr[d_idx, row, col].detach()
			M10 = xyz_lr[d_idx, row + 1, col].detach()
			M01 = xyz_lr[d_idx, row, col + 1].detach()
			M11 = xyz_lr[d_idx, row + 1, col + 1].detach()

			# Ext corner position and normal (detached, frozen)
			ext_P = ext_xyz.unsqueeze(0).expand(D, -1, -1, -1)  # (D, H_ext, W_ext, 3)
			ext_N = ext_norms.unsqueeze(0).expand(D, -1, -1, -1)
			ext_corner_used = ext_corner_used_2d.unsqueeze(0).expand(D, -1, -1)
			nan_p = torch.full_like(ext_P, float("nan"))
			ext_P = torch.where(ext_corner_used.unsqueeze(-1), ext_P, nan_p)
			ext_N = torch.where(ext_corner_used.unsqueeze(-1), ext_N, nan_p)

			# Ray-bilinear-patch intersection → (u, v) unclamped
			u, v = Model3D._ray_bilinear_intersect(
				ext_P, ext_N, M00, M10, M01, M11, frac_h, frac_w)

			# Full model grid position: row + u, col + v
			full_h = row.float() + u  # (D, H_ext, W_ext)
			full_w = col.float() + v

			# Validity: ext corner valid, model quad in bounds, intersection in [0,1]²
			uv_ok = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
			valid = (in_bounds & uv_ok & ext_corner_used).to(dtype=xyz_lr.dtype).unsqueeze(1)
			full_h = torch.where(valid.squeeze(1).bool(), full_h, torch.full_like(full_h, float("nan")))
			full_w = torch.where(valid.squeeze(1).bool(), full_w, torch.full_like(full_w, float("nan")))

			# Cache intersection params for conn_offset update
			self._ext_conn_params[ei] = {"u": u, "v": v, "row": row, "col": col}

			results.append((
				valid, offset,
				ext_P.detach(), ext_N.detach(),
				full_h.detach(), full_w.detach(),
				ext_quad_valid_2d.unsqueeze(0).unsqueeze(1).expand(D, -1, -1, -1).to(dtype=xyz_lr.dtype),
			))

		return results

	def forward(self, data: fit_data.FitData3D) -> FitResult3D:
		xyz_lr = self._grid_xyz()  # (D, Hm, Wm, 3)
		xyz_hr = self._grid_xyz_hr(xyz_lr)  # (D, He, We, 3)
		hr_channels = {"grad_mag"}
		if data.has_channel("pred_dt"):
			hr_channels.add("pred_dt")
		data_s = data.grid_sample_fullres(xyz_hr, channels=hr_channels)
		xy_conn, mask_conn, sign_conn, normals = self._xyz_conn(xyz_lr, data)

		D = self.depth
		He = int(xyz_hr.shape[1])
		We = int(xyz_hr.shape[2])
		Hm = self.mesh_h
		Wm = self.mesh_w

		# Target: cosine pattern along width (only meaningful when cos channel is loaded)
		if data.cos is not None:
			periods = max(1, Wm - 1)
			xs = torch.linspace(0.0, float(periods), We, device=xyz_lr.device, dtype=torch.float32)
			phase = (2.0 * torch.pi) * xs.view(1, 1, 1, We)
			target_plain = 0.5 + 0.5 * torch.cos(phase).expand(D, 1, He, We)

			amp_lr = self.amp.clamp(0.1, 1.0)
			bias_lr = self.bias.clamp(0.0, 0.45)
			amp_hr = F.interpolate(amp_lr, size=(He, We), mode="bilinear", align_corners=True)
			bias_hr = F.interpolate(bias_lr, size=(He, We), mode="bilinear", align_corners=True)
			target_mod = (bias_hr + amp_hr * (target_plain - 0.5)).clamp(0.0, 1.0)
		else:
			target_plain = torch.zeros(D, 1, He, We, device=xyz_lr.device)
			target_mod = torch.zeros(D, 1, He, We, device=xyz_lr.device)
			amp_lr = self.amp.clamp(0.1, 1.0)
			bias_lr = self.bias.clamp(0.0, 0.45)

		# Masking via grad_mag > 0 + GT normals at LR positions
		data_lr = data.grid_sample_fullres(xyz_lr.detach(), channels={"grad_mag", "nx", "ny"})
		mask_hr = (data_s.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=torch.float32).unsqueeze(1)
		mask_lr = (data_lr.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=torch.float32).unsqueeze(1)
		gt_normal_lr = data_lr.normal_3d  # (D, Hm, Wm, 3) or None

		# External surface intersections
		ext_conn = self._intersect_ext_surfaces(xyz_lr, data)
		cyl_xyz = None
		cyl_normals = None
		cyl_axes = None
		cyl_count = 0
		if self.cylinder_enabled:
			if self.cyl_shell_mode:
				cyl_xyz = xyz_lr
				cyl_normals = self.current_cylinder_shell_normals().unsqueeze(0)
				cyl_centers = None
				cyl_axes = None
				cyl_count = 1
				cyl_shell_base_xyz = self.cyl_shell_base
				cyl_shell_dirs = self.cyl_shell_dirs
				cyl_shell_w_offsets = self._shell_w_offset_values()
				cyl_shell_delta_xy = self._shell_delta_xy_params()
				cyl_shell_index = int(self.cyl_shell_current_index)
			else:
				cyl_xyz, cyl_normals = self.cylinder_samples()
				cyl_centers = self.cylinder_centers()
				cyl_axes = self.cylinder_axes()
				cyl_count = int(self.cyl_params.shape[0])
				cyl_shell_base_xyz = None
				cyl_shell_dirs = None
				cyl_shell_w_offsets = None
				cyl_shell_delta_xy = None
				cyl_shell_index = 0
		else:
			cyl_centers = None
			cyl_shell_base_xyz = None
			cyl_shell_dirs = None
			cyl_shell_w_offsets = None
			cyl_shell_delta_xy = None
			cyl_shell_index = 0

		return FitResult3D(
			xyz_lr=xyz_lr,
			xyz_hr=xyz_hr,
			data=data,
			data_s=data_s,
			target_plain=target_plain,
			target_mod=target_mod,
			amp_lr=amp_lr,
			bias_lr=bias_lr,
			mask_hr=mask_hr,
			mask_lr=mask_lr,
			normals=normals,
			xy_conn=xy_conn,
			mask_conn=mask_conn,
			sign_conn=sign_conn,
			params=self.params,
			gt_normal_lr=gt_normal_lr,
			ext_conn=ext_conn,
			cyl_xyz=cyl_xyz,
			cyl_normals=cyl_normals,
			cyl_centers=cyl_centers,
			cyl_axes=cyl_axes,
			cyl_params=self.cyl_params if self.cylinder_enabled else None,
			cyl_count=cyl_count,
			cyl_shell_mode=bool(self.cyl_shell_mode and self.cylinder_enabled),
			cyl_shell_step=float(self._current_shell_target_offset()),
			cyl_shell_base_xyz=cyl_shell_base_xyz,
			cyl_shell_dirs=cyl_shell_dirs,
			cyl_shell_w_offsets=cyl_shell_w_offsets,
			cyl_shell_delta_xy=cyl_shell_delta_xy,
			cyl_shell_index=cyl_shell_index,
		)

	def opt_params(self) -> dict[str, list[nn.Parameter]]:
		out: dict[str, list[nn.Parameter]] = {
			"mesh_ms": list(self.mesh_ms),
			"amp": [self.amp],
			"bias": [self.bias],
		}
		if self.arc_enabled:
			out["arc_cx"] = [self.arc_cx]
			out["arc_cy"] = [self.arc_cy]
			out["arc_radius"] = [self.arc_radius]
			out["arc_angle0"] = [self.arc_angle0]
			out["arc_angle1"] = [self.arc_angle1]
		if self.straight_enabled:
			out["straight_cx"] = [self.straight_cx]
			out["straight_cy"] = [self.straight_cy]
			out["straight_angle"] = [self.straight_angle]
			out["straight_half_w"] = [self.straight_half_w]
		if self.cylinder_enabled:
			out["cyl_params"] = [self.cyl_params]
			if self.cyl_shell_mode and int(getattr(self, "cyl_shell_current_index", 0)) > 0:
				out["cyl_params"].append(self.cyl_shell_w_offsets)
		return out

	def crop_depth(self, d_lo: int, d_hi: int) -> None:
		"""Crop the model to only keep depth layers [d_lo, d_hi).

		Integrates the mesh pyramid to flat, slices along depth, rebuilds
		the pyramid, and slices conn_offsets, amp, bias accordingly.
		"""
		d_lo = max(0, int(d_lo))
		d_hi = min(self.depth, int(d_hi))
		if d_lo == 0 and d_hi == self.depth:
			return
		new_depth = d_hi - d_lo
		if new_depth < 1:
			raise ValueError(f"crop_depth: empty range [{d_lo}, {d_hi})")
		print(f"[model] crop_depth: [{d_lo}, {d_hi}) — {self.depth} -> {new_depth} layers")

		with torch.no_grad():
			# Integrate pyramid to flat, slice, rebuild
			flat = self._integrate_pyramid_3d(self.mesh_ms, pyramid_d=self.pyramid_d)  # (3, D, H, W)
			flat = flat[:, d_lo:d_hi]
			n_scales = len(self.mesh_ms)
			self.mesh_ms = self._construct_pyramid_from_flat_3d(flat, n_scales, pyramid_d=self.pyramid_d)

			# Slice conn_offsets
			self.conn_offsets = self.conn_offsets[:, d_lo:d_hi].contiguous()

			# Slice amp and bias
			self.amp = nn.Parameter(self.amp.data[d_lo:d_hi].contiguous())
			self.bias = nn.Parameter(self.bias.data[d_lo:d_hi].contiguous())

		self.depth = new_depth

	def bake_arc_into_mesh(self) -> None:
		"""Absorb arc transform into mesh_ms, disable arc."""
		with torch.no_grad():
			final = self._arc_base_positions() + self.mesh_coarse()
			self.mesh_ms = self._construct_pyramid_from_flat_3d(final, len(self.mesh_ms), pyramid_d=self.pyramid_d)
			self.arc_enabled = False

	def bake_straight_into_mesh(self) -> None:
		"""Absorb straight transform into mesh_ms, disable straight."""
		with torch.no_grad():
			final = self._straight_base_positions() + self.mesh_coarse()
			self.mesh_ms = self._construct_pyramid_from_flat_3d(final, len(self.mesh_ms), pyramid_d=self.pyramid_d)
			self.straight_enabled = False

	def load_state_dict_compat(self, state_dict: dict, *, strict: bool = False) -> tuple[list[str], list[str]]:
		st = dict(state_dict)
		st.pop("_model_params_", None)
		st.pop("_fit_config_", None)
		st.pop("_corr_points_results_", None)
		# Drop legacy conn_offset_ms pyramid keys
		for k in list(st.keys()):
			if k.startswith("conn_offset_ms."):
				st.pop(k)
		st.pop("cyl_params", None)
		st.pop("cyl_shell_w_offsets", None)
		incompat = super().load_state_dict(st, strict=bool(strict))
		return list(incompat.missing_keys), list(incompat.unexpected_keys)

	@classmethod
	def from_checkpoint(cls, state_dict: dict, *, device: torch.device) -> 'Model3D':
		"""Construct a Model3D from a saved checkpoint state_dict."""
		mp = state_dict["_model_params_"]
		# Get flat mesh — either directly stored or integrated from old pyramid
		if "mesh_flat" in state_dict:
			flat = state_dict["mesh_flat"].to(device=device, dtype=torch.float32)
		else:
			# Legacy: integrate pyramid levels
			saved_pyramid_d = bool(mp.get("pyramid_d", False))
			n_levels = sum(1 for k in state_dict if k.startswith("mesh_ms.") and k[len("mesh_ms."):].isdigit())
			old_levels = nn.ParameterList([
				nn.Parameter(state_dict[f"mesh_ms.{i}"].to(device=device, dtype=torch.float32))
				for i in range(n_levels)
			])
			flat = cls._integrate_pyramid_3d(old_levels, pyramid_d=saved_pyramid_d)
			print(f"[model] integrated legacy pyramid ({n_levels} levels, pyramid_d={saved_pyramid_d}) to flat mesh")
		_c, D, H, W = (int(v) for v in flat.shape)
		mdl = cls(
			device=device,
			depth=D,
			mesh_h=H,
			mesh_w=W,
			mesh_step=int(mp["mesh_step"]),
			winding_step=int(mp["winding_step"]),
			subsample_mesh=int(mp["subsample_mesh"]),
			subsample_winding=int(mp["subsample_winding"]),
			scaledown=float(mp["scaledown"]),
			z_step_eff=int(mp["z_step_eff"]),
			volume_extent=mp.get("volume_extent"),
			pyramid_d=bool(mp.get("pyramid_d", True)),
		)
		# Reconstruct pyramid from flat
		n_scales = len(mdl.mesh_ms)
		mdl.mesh_ms = cls._construct_pyramid_from_flat_3d(flat, n_scales, pyramid_d=mdl.pyramid_d)
		# Load remaining state (skip mesh keys)
		st_rest = {k: v for k, v in state_dict.items()
				   if not k.startswith("mesh_ms.") and k != "mesh_flat"}
		mdl.load_state_dict_compat(st_rest, strict=False)
		mdl.arc_enabled = False
		mdl.straight_enabled = False
		mdl.cylinder_enabled = False
		return mdl
