from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import tifffile


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import flatten_slim
import model as fit_model


def _flat_grid(h: int, w: int, *, sx: float = 1.0, sy: float = 1.0) -> torch.Tensor:
	yy = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	xx = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	zz = torch.zeros(h, w, dtype=torch.float32)
	return torch.stack([xx * sx, yy * sy, zz], dim=-1)


class FlattenSlimTest(unittest.TestCase):
	def test_flat_regular_grid_keeps_identity_uv_and_exports_valid_points(self) -> None:
		xyz = _flat_grid(5, 5)
		valid = torch.ones(5, 5, dtype=torch.bool)
		cfg = flatten_slim.SlimConfig(iters=2, pcg_iters=20, mg_levels=1)

		result = flatten_slim.solve_flatten(xyz, valid, config=cfg)

		expected = torch.stack(torch.meshgrid(
			torch.arange(5, dtype=torch.float32),
			torch.arange(5, dtype=torch.float32),
			indexing="ij",
		), dim=-1)
		self.assertLess(float((result.uv - expected).abs().amax()), 1.0e-4)
		self.assertGreater(int(result.output_mask.sum()), 0)
		self.assertLessEqual(result.energy_history[-1], result.energy_history[0] + 1.0e-6)

	def test_stagnant_slim_iterations_stop_early(self) -> None:
		xyz = _flat_grid(5, 5)
		valid = torch.ones(5, 5, dtype=torch.bool)
		cfg = flatten_slim.SlimConfig(iters=40, pcg_iters=20, mg_levels=1)

		result = flatten_slim.solve_flatten(xyz, valid, config=cfg)

		self.assertLessEqual(len(result.energy_history), 2)
		self.assertEqual(result.stats.get("stop_reason"), "energy_stagnated")

	def test_source_hole_remains_invalid_after_uv_inversion(self) -> None:
		xyz = _flat_grid(4, 4)
		valid = torch.ones(4, 4, dtype=torch.bool)
		valid[1, 1] = False
		system = flatten_slim.build_slim_system(xyz, valid)

		_out_xyz, out_mask, _out_map = flatten_slim.invert_uv_to_tifxyz(
			system,
			system.initial_uv,
			output_margin=0.0,
			min_shape=(4, 4),
		)

		self.assertFalse(bool(out_mask[1, 1]))

	def test_trust_region_clamps_without_amplifying_small_updates(self) -> None:
		small = torch.full((2, 2, 2), 0.1)
		clamped_small, raw_small = flatten_slim._trust_region_update(small, 1.0)
		self.assertAlmostEqual(raw_small, float(torch.linalg.vector_norm(small, dim=-1).amax()))
		self.assertLess(float((clamped_small - small).abs().amax()), 1.0e-7)

		large = torch.full((2, 2, 2), 10.0)
		clamped_large, raw_large = flatten_slim._trust_region_update(large, 1.0)
		self.assertGreater(raw_large, 1.0)
		self.assertLessEqual(float(torch.linalg.vector_norm(clamped_large, dim=-1).amax()), 1.0 + 1.0e-6)

	def test_inverse_map_solver_reduces_sampled_canvas_energy(self) -> None:
		xyz = _flat_grid(10, 10)
		yy = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 10)
		xyz[..., 0] = xyz[..., 0] + 0.08 * (yy - 4.5).square()
		valid = torch.ones(10, 10, dtype=torch.bool)
		cfg = flatten_slim.SlimConfig(
			iters=8,
			pcg_iters=30,
			mg_levels=1,
			max_update=0.2,
			stop_rel_threshold=0.0,
		)

		result = flatten_slim.solve_inverse_flatten(xyz, valid, config=cfg)

		self.assertLess(result.energy_history[-1], result.energy_history[0])
		self.assertGreater(int(result.output_mask.sum()), 0)
		self.assertEqual(tuple(result.output_xyz.shape[:2]), tuple(result.output_map_yx.shape[:2]))

	def test_matrix_free_operator_is_symmetric(self) -> None:
		system = flatten_slim.build_slim_system(_flat_grid(5, 6), torch.ones(5, 6, dtype=torch.bool))
		g = torch.Generator().manual_seed(123)
		x = torch.randn(5, 6, 2, generator=g) * system.vertex_mask.unsqueeze(-1)
		y = torch.randn(5, 6, 2, generator=g) * system.vertex_mask.unsqueeze(-1)

		ax = flatten_slim.apply_global_operator(system, x)
		ay = flatten_slim.apply_global_operator(system, y)
		lhs = (x * ay).sum()
		rhs = (ax * y).sum()

		self.assertAlmostEqual(float(lhs), float(rhs), places=5)

	def test_pcg_residual_decreases(self) -> None:
		system = flatten_slim.build_slim_system(_flat_grid(8, 8), torch.ones(8, 8, dtype=torch.bool))
		g = torch.Generator().manual_seed(321)
		b = torch.randn(8, 8, 2, generator=g) * system.vertex_mask.unsqueeze(-1)

		_x, stats = flatten_slim.pcg_solve(
			lambda u: flatten_slim.apply_global_operator(system, u),
			b,
			mask=system.vertex_mask,
			max_iter=12,
			tol=1.0e-8,
		)

		self.assertLess(stats.final_residual, stats.initial_residual)

	def test_multigrid_preconditioner_improves_short_pcg(self) -> None:
		xyz = _flat_grid(16, 16)
		yy = torch.arange(16, dtype=torch.float32).view(16, 1)
		xx = torch.arange(16, dtype=torch.float32).view(1, 16)
		xyz[..., 2] = 0.15 * torch.sin(yy * 0.4) * torch.cos(xx * 0.3)
		system = flatten_slim.build_slim_system(xyz, torch.ones(16, 16, dtype=torch.bool))
		g = torch.Generator().manual_seed(456)
		b = torch.randn(16, 16, 2, generator=g) * system.vertex_mask.unsqueeze(-1)

		_x0, s0 = flatten_slim.pcg_solve(
			lambda u: flatten_slim.apply_global_operator(system, u),
			b,
			mask=system.vertex_mask,
			max_iter=6,
			tol=1.0e-8,
		)
		mg = flatten_slim.MultigridPreconditioner(system, levels="auto", pre_smooth=2, post_smooth=2)
		_x1, s1 = flatten_slim.pcg_solve(
			lambda u: flatten_slim.apply_global_operator(system, u),
			b,
			mask=system.vertex_mask,
			preconditioner=mg,
			max_iter=6,
			tol=1.0e-8,
		)

		self.assertLessEqual(s1.final_residual, s0.final_residual * 1.05)

	def test_bilinear_quad_inversion_accepts_inside_and_rejects_outside(self) -> None:
		points = np.asarray([[0.25, 0.75], [1.5, 0.5]], dtype=np.float64)
		quad = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=np.float64)
		quads = np.repeat(quad[None, :, :, :], 2, axis=0)

		s, t, residual2 = flatten_slim.bilinear_inverse_points(points, quads)

		self.assertAlmostEqual(float(s[0, 0]), 0.25, places=6)
		self.assertAlmostEqual(float(t[0, 0]), 0.75, places=6)
		self.assertTrue(np.isfinite(residual2[0, 0]))
		self.assertFalse(np.isfinite(residual2[1, 0]))

	def test_slim_mode_bypasses_optimizer_stages_and_writes_outputs(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			tifxyz = root / "input.tifxyz"
			tifxyz.mkdir()
			xyz = _flat_grid(4, 4).numpy().astype(np.float32)
			tifffile.imwrite(str(tifxyz / "x.tif"), xyz[..., 0])
			tifffile.imwrite(str(tifxyz / "y.tif"), xyz[..., 1])
			tifffile.imwrite(str(tifxyz / "z.tif"), xyz[..., 2])
			(tifxyz / "meta.json").write_text(json.dumps({"scale": [1.0, 1.0]}), encoding="utf-8")
			cfg_path = root / "flatten_slim.json"
			cfg_path.write_text(json.dumps({
				"args": {
					"model-init": "flatten",
					"flatten_solver": "slim",
					"flatten_slim_iters": 1,
					"flatten_slim_pcg_iters": 8,
					"flatten_slim_mg_levels": 1,
					"device": "cpu",
				},
				"external_surfaces": [{"path": str(tifxyz)}],
			}), encoding="utf-8")
			out = root / "out"

			rc = fit.main([str(cfg_path), "--out-dir", str(out)])

			self.assertEqual(rc, 0)
			self.assertTrue((out / "model_final.pt").exists())
			self.assertTrue((out / "tifxyz" / "flatten.tifxyz" / "x.tif").exists())
			st = torch.load(out / "model_final.pt", map_location="cpu", weights_only=False)
			self.assertIn("flatten_map_flat", st)
			self.assertIn("flatten_slim_uv", st)


if __name__ == "__main__":
	unittest.main()
