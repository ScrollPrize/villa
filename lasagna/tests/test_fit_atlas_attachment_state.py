from __future__ import annotations

import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import fit_data
import model as fit_model
import optimizer
import opt_loss_atlas_line


def _lines() -> fit_data.AtlasLines3D:
	return fit_data.AtlasLines3D(
		target_xyz=torch.tensor(
			[[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
			dtype=torch.float32,
		),
		normal_xyz=torch.tensor(
			[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
			dtype=torch.float32,
		),
		model_h=torch.tensor([1.0, 2.0], dtype=torch.float32),
		model_w=torch.tensor([3.0, 4.0], dtype=torch.float32),
		object_ids=("fiber-a", "fiber-a"),
		source_indices=(7, 8),
		is_control_point=torch.tensor([True, False]),
		is_snap_point=torch.tensor([False, True]),
		atlas_winding_model_ranges=((0, 0.0, 10.0),),
	)


def _payload(*, sample_h: torch.Tensor | None = None, sample_w: torch.Tensor | None = None) -> opt_loss_atlas_line.AtlasLineDebugPayload:
	if sample_h is None:
		sample_h = torch.tensor([[5.5, 6.5]], dtype=torch.float32)
	if sample_w is None:
		sample_w = torch.tensor([[7.5, 8.5]], dtype=torch.float32)
	return opt_loss_atlas_line.AtlasLineDebugPayload(
		model_xyz=torch.zeros((1, 3, 4, 3), dtype=torch.float32),
		valid=torch.ones((1, 2), dtype=torch.bool),
		target_xyz=torch.zeros((1, 2, 3), dtype=torch.float32),
		hit_xyz=torch.zeros((1, 2, 3), dtype=torch.float32),
		model_normal=torch.zeros((1, 2, 3), dtype=torch.float32),
		signed_delta=torch.zeros((1, 2), dtype=torch.float32),
		is_control=torch.tensor([[True, False]], dtype=torch.bool),
		is_snap=torch.tensor([[False, True]], dtype=torch.bool),
		sample_model_h=sample_h,
		sample_model_w=sample_w,
		normal_proxy_target_xyz=torch.zeros((1, 3, 4, 3), dtype=torch.float32),
		normal_proxy_valid=torch.ones((1, 3, 4), dtype=torch.bool),
		object_ids=("fiber-a", "fiber-a"),
		atlas_winding_model_ranges=((0, 0.0, 10.0),),
	)


def _active_atlas_stages() -> list[optimizer.Stage]:
	return optimizer.load_stages_cfg({
		"base": {"normal": 0.0, "atlas_line_control": 1.0},
		"stages": [{
			"name": "atlas_reopt",
			"opt": {
				"steps": 1,
				"lr": 0.0,
				"params": ["mesh_ms"],
			},
		}],
	})


def _fit_data(lines: fit_data.AtlasLines3D) -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=None,
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		atlas_lines=lines,
		_vol_size=(1, 1, 1),
	)


class TinyAtlasModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		H, W = 3, 4
		y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		z = torch.zeros_like(x)
		self.xyz = torch.stack([x, y, z], dim=-1).unsqueeze(0).contiguous()

	def forward(self, data: fit_data.FitData3D, needs=None) -> fit_model.FitResult3D:
		D, H, W, _ = self.xyz.shape
		normals = torch.zeros_like(self.xyz)
		normals[..., 2] = 1.0
		return fit_model.FitResult3D(
			xyz_lr=self.xyz,
			xyz_hr=None,
			data=data,
			data_s=None,
			data_lr=None,
			target_plain=None,
			target_mod=None,
			amp_lr=torch.ones(D, 1, H, W, dtype=self.xyz.dtype),
			bias_lr=torch.zeros(D, 1, H, W, dtype=self.xyz.dtype),
			mask_hr=None,
			mask_lr=None,
			normals=normals,
			xy_conn=None,
			mask_conn=None,
			sign_conn=None,
			params=fit_model.ModelParams3D(
				mesh_step=1,
				winding_step=1,
				subsample_mesh=1,
				subsample_winding=1,
				scaledown=1.0,
				z_step_eff=1,
				volume_extent=None,
				pyramid_d=False,
			),
		)


class FitAtlasAttachmentStateTest(unittest.TestCase):
	def test_attachment_state_uses_debug_payload_model_coordinates(self) -> None:
		state = fit._atlas_line_attachment_state_from_debug_payload(
			lines=_lines(),
			payload=_payload(),
		)

		self.assertEqual(state["format"], "lasagna_atlas_line_attachment_state")
		self.assertTrue(torch.equal(state["model_h"], torch.tensor([5.5, 6.5], dtype=torch.float32)))
		self.assertTrue(torch.equal(state["model_w"], torch.tensor([7.5, 8.5], dtype=torch.float32)))
		self.assertEqual(state["object_ids"], ["fiber-a", "fiber-a"])
		self.assertEqual(state["source_indices"], [7, 8])
		self.assertEqual(state["atlas_winding_model_ranges"], [[0, 0.0, 10.0]])

	def test_model_reopt_restores_atlas_lines_from_checkpoint_state(self) -> None:
		state = fit._atlas_line_attachment_state_from_debug_payload(
			lines=_lines(),
			payload=_payload(),
		)
		restored = fit._checkpoint_atlas_lines_for_reopt(
			checkpoint_state={fit._ATLAS_LINE_ATTACHMENT_STATE_KEY: state},
			stages=_active_atlas_stages(),
			device=torch.device("cpu"),
		)

		self.assertIsNotNone(restored)
		assert restored is not None
		self.assertTrue(torch.equal(restored.model_h, torch.tensor([5.5, 6.5], dtype=torch.float32)))
		self.assertTrue(torch.equal(restored.model_w, torch.tensor([7.5, 8.5], dtype=torch.float32)))
		self.assertEqual(restored.object_ids, ("fiber-a", "fiber-a"))
		self.assertEqual(restored.source_indices, (7, 8))
		self.assertEqual(restored.is_control_point.tolist(), [True, False])
		self.assertEqual(restored.is_snap_point.tolist(), [False, True])

	def test_checkpoint_population_writes_attachment_state(self) -> None:
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
			model_h=torch.tensor([0.0], dtype=torch.float32),
			model_w=torch.tensor([0.0], dtype=torch.float32),
			object_ids=("fiber-b",),
			source_indices=(9,),
			is_control_point=torch.tensor([True]),
			is_snap_point=torch.tensor([False]),
		)
		st: dict = {}

		results = fit._populate_atlas_checkpoint_state(
			st,
			mdl=TinyAtlasModel(),
			data=_fit_data(lines),
		)

		self.assertIsNotNone(results)
		self.assertIn(fit._ATLAS_LINE_ATTACHMENT_STATE_KEY, st)
		self.assertIn("_atlas_control_points_results_", st)
		state = st[fit._ATLAS_LINE_ATTACHMENT_STATE_KEY]
		self.assertTrue(torch.allclose(state["model_h"], torch.tensor([1.0], dtype=torch.float32)))
		self.assertTrue(torch.allclose(state["model_w"], torch.tensor([1.0], dtype=torch.float32)))
		record = st["_atlas_control_points_results_"]["records"][0]
		self.assertAlmostEqual(record["model_h"], 1.0, delta=1.0e-6)
		self.assertAlmostEqual(record["model_w"], 1.0, delta=1.0e-6)

	def test_active_atlas_reopt_without_attachment_state_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "_atlas_line_attachment_state_"):
			fit._checkpoint_atlas_lines_for_reopt(
				checkpoint_state={},
				stages=_active_atlas_stages(),
				device=torch.device("cpu"),
			)

	def test_inactive_atlas_reopt_does_not_require_attachment_state(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {"normal": 1.0, "atlas_line_control": 0.0},
			"stages": [{
				"name": "normal_only",
				"opt": {
					"steps": 1,
					"lr": 0.0,
					"params": ["mesh_ms"],
				},
			}],
		})

		self.assertIsNone(
			fit._checkpoint_atlas_lines_for_reopt(
				checkpoint_state={},
				stages=stages,
				device=torch.device("cpu"),
			)
		)

	def test_attachment_state_rejects_multi_depth_debug_payload(self) -> None:
		with self.assertRaisesRegex(ValueError, "depth-1"):
			fit._atlas_line_attachment_state_from_debug_payload(
				lines=_lines(),
				payload=_payload(
					sample_h=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
					sample_w=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
				),
			)


if __name__ == "__main__":
	unittest.main()
