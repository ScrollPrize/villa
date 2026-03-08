from __future__ import annotations

from dataclasses import dataclass
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
		volume_extent: tuple[float, float, float, float, float, float] | None = None,
		pyramid_d: bool = True,
	) -> None:
		super().__init__()
		self.depth = max(1, int(depth))
		self.mesh_h = max(2, int(mesh_h))
		self.mesh_w = max(2, int(mesh_w))
		self.z_center = float(z_center)
		self.arc_enabled = True
		self.pyramid_d = bool(pyramid_d)

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

		# Residual mesh pyramid: (3, D, H, W) per scale, 5 levels
		n_scales = 5
		self.mesh_ms = self._build_zero_pyramid(
			n_scales=n_scales, channels=3, d=self.depth, h=self.mesh_h, w=self.mesh_w, device=device,
			pyramid_d=self.pyramid_d,
		)
		# Connection offsets buffer: (4, D, Hm, Wm) — [prev_h, prev_w, next_h, next_w]
		# Not gradient-optimized; updated by update_conn_offsets() after each step.
		self.register_buffer("conn_offsets", torch.zeros(4, self.depth, self.mesh_h, self.mesh_w, device=device, dtype=torch.float32))

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
		return nn.ParameterList([
			nn.Parameter(torch.zeros(channels, di, hi, wi, device=device, dtype=torch.float32))
			for di, hi, wi in shapes
		])

	# --- 3D pyramid operations ---

	@staticmethod
	def _upsample_crop(src: torch.Tensor, d_t: int, h_t: int, w_t: int, *, pyramid_d: bool) -> torch.Tensor:
		if pyramid_d:
			up = F.interpolate(src.unsqueeze(0), scale_factor=2.0, mode='trilinear', align_corners=True).squeeze(0)
			return up[:, :d_t, :h_t, :w_t]
		else:
			up = F.interpolate(src, scale_factor=2.0, mode='bilinear', align_corners=True)
			return up[:, :, :h_t, :w_t]

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

	# --- Mesh access ---

	def mesh_coarse(self) -> torch.Tensor:
		"""Integrate residual pyramid -> (3, D, H, W)."""
		return self._integrate_pyramid_3d(self.mesh_ms, pyramid_d=self.pyramid_d)

	def _grid_xyz(self) -> torch.Tensor:
		"""(D, Hm, Wm, 3) mesh positions in fullres voxel coords."""
		residuals = self.mesh_coarse()  # (3, D, H, W)
		if self.arc_enabled:
			base = self._arc_base_positions()
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

			mask_prev = _valid_mask(data.grid_sample_fullres(prev_full.detach()).grad_mag)
			mask_center = _valid_mask(data.grid_sample_fullres(xyz_lr.detach()).grad_mag)
			mask_next = _valid_mask(data.grid_sample_fullres(next_full.detach()).grad_mag)

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

	def forward(self, data: fit_data.FitData3D) -> FitResult3D:
		xyz_lr = self._grid_xyz()  # (D, Hm, Wm, 3)
		xyz_hr = self._grid_xyz_hr(xyz_lr)  # (D, He, We, 3)
		data_s = data.grid_sample_fullres(xyz_hr)
		xy_conn, mask_conn, sign_conn, normals = self._xyz_conn(xyz_lr, data)

		D = self.depth
		He = int(xyz_hr.shape[1])
		We = int(xyz_hr.shape[2])
		Hm = self.mesh_h
		Wm = self.mesh_w

		# Target: cosine pattern along width
		periods = max(1, Wm - 1)
		xs = torch.linspace(0.0, float(periods), We, device=xyz_lr.device, dtype=torch.float32)
		phase = (2.0 * torch.pi) * xs.view(1, 1, 1, We)
		target_plain = 0.5 + 0.5 * torch.cos(phase).expand(D, 1, He, We)

		amp_lr = self.amp.clamp(0.1, 1.0)
		bias_lr = self.bias.clamp(0.0, 0.45)
		amp_hr = F.interpolate(amp_lr, size=(He, We), mode="bilinear", align_corners=True)
		bias_hr = F.interpolate(bias_lr, size=(He, We), mode="bilinear", align_corners=True)
		target_mod = (bias_hr + amp_hr * (target_plain - 0.5)).clamp(0.0, 1.0)

		# Masking via grad_mag > 0
		mask_hr = (data.grid_sample_fullres(xyz_hr.detach()).grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=torch.float32).unsqueeze(1)
		mask_lr = (data.grid_sample_fullres(xyz_lr.detach()).grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=torch.float32).unsqueeze(1)

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
		return out

	def bake_arc_into_mesh(self) -> None:
		"""Absorb arc transform into mesh_ms, disable arc."""
		with torch.no_grad():
			final = self._arc_base_positions() + self.mesh_coarse()
			self.mesh_ms = self._construct_pyramid_from_flat_3d(final, len(self.mesh_ms), pyramid_d=self.pyramid_d)
			self.arc_enabled = False

	def load_state_dict_compat(self, state_dict: dict, *, strict: bool = False) -> tuple[list[str], list[str]]:
		st = dict(state_dict)
		st.pop("_model_params_", None)
		st.pop("_fit_config_", None)
		st.pop("_corr_points_results_", None)
		# Drop legacy conn_offset_ms pyramid keys
		for k in list(st.keys()):
			if k.startswith("conn_offset_ms."):
				st.pop(k)
		incompat = super().load_state_dict(st, strict=bool(strict))
		return list(incompat.missing_keys), list(incompat.unexpected_keys)

	@classmethod
	def from_checkpoint(cls, state_dict: dict, *, device: torch.device) -> 'Model3D':
		"""Construct a Model3D from a saved checkpoint state_dict."""
		mp = state_dict["_model_params_"]
		mesh0 = state_dict["mesh_ms.0"]
		_c, D, H, W = (int(v) for v in mesh0.shape)
		saved_pyramid_d = bool(mp.get("pyramid_d", False))
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
			pyramid_d=True,
		)
		if not saved_pyramid_d:
			# Checkpoint was saved with pyramid_d=False — D is constant across levels.
			# Integrate old pyramid, then rebuild with pyramid_d=True.
			n_levels = sum(1 for k in state_dict if k.startswith("mesh_ms.") and k[len("mesh_ms."):].isdigit())
			old_levels = nn.ParameterList([
				nn.Parameter(state_dict[f"mesh_ms.{i}"].to(device=device, dtype=torch.float32))
				for i in range(n_levels)
			])
			flat = cls._integrate_pyramid_3d(old_levels, pyramid_d=False)
			mdl.mesh_ms = cls._construct_pyramid_from_flat_3d(flat, n_levels, pyramid_d=True)
			# Load remaining state (skip mesh_ms keys handled above)
			st_rest = {k: v for k, v in state_dict.items() if not k.startswith("mesh_ms.")}
			mdl.load_state_dict_compat(st_rest, strict=False)
			print(f"[model] rebuilt mesh pyramid with pyramid_d=True from old pyramid_d=False checkpoint")
		else:
			mdl.load_state_dict_compat(state_dict, strict=False)
		mdl.arc_enabled = False
		return mdl
