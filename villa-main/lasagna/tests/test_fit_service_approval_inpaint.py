from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_service


def _b64(data: bytes) -> str:
	return fit_service.base64.b64encode(data).decode("ascii")


class FitServiceApprovalInpaintTest(unittest.TestCase):
	def test_request_volume_shape_accepts_top_level_and_job_spec(self) -> None:
		body = {
			"volume_shape_zyx": [10, 20, 30],
			"job_spec": {"volume_shape_zyx": [10, 20, 30]},
		}

		self.assertEqual(fit_service._request_volume_shape_zyx(body), (10, 20, 30))

	def test_request_volume_shape_rejects_mismatch(self) -> None:
		with self.assertRaisesRegex(ValueError, "must match"):
			fit_service._request_volume_shape_zyx({
				"volume_shape_zyx": [10, 20, 30],
				"job_spec": {"volume_shape_zyx": [10, 20, 31]},
			})

	def test_seed_mode_ignores_external_surface_without_consumer(self) -> None:
		cfg = {"args": {"model-init": "seed"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz"}]}
		args = cfg["args"]

		tifxyz_dir = fit_service._wire_external_surface_for_request(
			cfg=cfg,
			args_section=args,
			model_init="seed",
			ext_offset_enabled=False,
		)

		self.assertIsNone(tifxyz_dir)
		self.assertNotIn("approval-inpaint-tifxyz", args)
		self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_wires_external_surface(self) -> None:
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}, "external_surfaces": [{"path": "/tmp/ref.tifxyz"}]}
		args = cfg["args"]

		tifxyz_dir = fit_service._wire_external_surface_for_request(
			cfg=cfg,
			args_section=args,
			model_init="seed",
			ext_offset_enabled=False,
		)

		self.assertEqual(tifxyz_dir, "/tmp/ref.tifxyz")
		self.assertEqual(args["approval-inpaint-tifxyz"], "/tmp/ref.tifxyz")
		self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_requires_seed_model_init(self) -> None:
		cfg = {"args": {"model-init": "ext", "approval-inpaint": True}}

		with self.assertRaisesRegex(ValueError, "only valid with args.model-init=seed"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="ext",
				ext_offset_enabled=False,
			)

	def test_approval_inpaint_requires_one_external_surface(self) -> None:
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}}

		with self.assertRaisesRegex(ValueError, "approval-inpaint requires exactly one external_surfaces entry, got 0"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="seed",
				ext_offset_enabled=False,
			)

		cfg["external_surfaces"] = [{"path": "/tmp/a.tifxyz"}, {"path": "/tmp/b.tifxyz"}]
		with self.assertRaisesRegex(ValueError, "approval-inpaint requires exactly one external_surfaces entry, got 2"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="seed",
				ext_offset_enabled=False,
			)

	def test_ext_init_wires_external_surface(self) -> None:
		cfg = {"args": {"model-init": "ext"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz"}]}

		tifxyz_dir = fit_service._wire_external_surface_for_request(
			cfg=cfg,
			args_section=cfg["args"],
			model_init="ext",
			ext_offset_enabled=False,
		)

		self.assertEqual(tifxyz_dir, "/tmp/ref.tifxyz")
		self.assertEqual(cfg["args"]["tifxyz-init"], "/tmp/ref.tifxyz")

	def test_ext_init_requires_one_external_surface(self) -> None:
		cfg = {"args": {"model-init": "ext"}}

		with self.assertRaisesRegex(ValueError, "model-init=ext requires exactly one external_surfaces entry, got 0"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="ext",
				ext_offset_enabled=False,
			)

	def test_flatten_validates_external_surface_without_wiring_tifxyz_init(self) -> None:
		cfg = {"args": {"model-init": "flatten"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz"}]}

		tifxyz_dir = fit_service._wire_external_surface_for_request(
			cfg=cfg,
			args_section=cfg["args"],
			model_init="flatten",
			ext_offset_enabled=False,
		)

		self.assertIsNone(tifxyz_dir)
		self.assertEqual(cfg["external_surfaces"], [{"path": "/tmp/ref.tifxyz"}])
		self.assertNotIn("tifxyz-init", cfg["args"])

	def test_flatten_requires_external_surface(self) -> None:
		cfg = {"args": {"model-init": "flatten"}}

		with self.assertRaisesRegex(ValueError, "model-init=flatten requires exactly one external_surfaces entry, got 0"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="flatten",
				ext_offset_enabled=False,
			)

	def test_ext_offset_requires_config_external_surfaces(self) -> None:
		cfg = {"args": {"model-init": "seed"}}

		with self.assertRaisesRegex(ValueError, "ext_offset requires exactly one external_surfaces entry, got 0"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="seed",
				ext_offset_enabled=True,
			)

	def test_global_map_validates_external_surface_without_wiring_tifxyz_init(self) -> None:
		cfg = {"args": {"model-init": "seed"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz", "offset": -1.0}]}
		args = cfg["args"]

		tifxyz_dir = fit_service._wire_external_surface_for_request(
			cfg=cfg,
			args_section=args,
			model_init="seed",
			ext_offset_enabled=False,
			global_map_enabled=True,
		)

		self.assertIsNone(tifxyz_dir)
		self.assertEqual(cfg["external_surfaces"], [{"path": "/tmp/ref.tifxyz", "offset": -1.0}])
		self.assertNotIn("tifxyz-init", args)

	def test_global_map_requires_external_surfaces(self) -> None:
		cfg = {"args": {"model-init": "seed"}}

		with self.assertRaisesRegex(ValueError, "snap_surf_map/global_map requires exactly one external_surfaces entry, got 0"):
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="seed",
				ext_offset_enabled=False,
				global_map_enabled=True,
			)

	def test_global_map_detection_from_map_stage_params(self) -> None:
		cfg = {
			"base": {"snap_surf_map": 0.0},
			"stages": [
				{"name": "map_init", "steps": 100, "params": ["map_surf_affine"]},
			],
		}

		self.assertTrue(fit_service._config_global_map_enabled(cfg))

	def test_global_map_detection_from_nested_map_opt(self) -> None:
		cfg = {
			"base": {"snap_surf_map": 0.0},
			"stages": [
				{
					"name": "snap",
					"steps": 100,
					"params": ["mesh_ms"],
					"args": {"snap_surf_map": {"map_opt": {"params": ["map_surf_ms"]}}},
				},
			],
		}

		self.assertTrue(fit_service._config_global_map_enabled(cfg))

	def test_self_map_snap_surf_map_does_not_require_external_surface(self) -> None:
		cfg = {
			"args": {
				"model-init": "seed",
				"self-map-init": "multi_wrap_d",
				"init-grow": {
					"enabled": True,
					"initial_depth": 1,
					"order": ["up"],
					"step": 1,
				},
			},
			"base": {"snap_surf_map": 0.1},
			"stages": [
				{
					"name": "expand-z",
					"stages": [
						{
							"name": "expand_up",
							"steps": 1,
							"params": ["mesh_ms"],
							"args": {
								"snap_surf_map": {
									"map_opt": {
										"steps": 1,
										"params": ["map_surf_ms"],
									}
								}
							},
						}
					],
				}
			],
		}

		self.assertFalse(fit_service._config_global_map_enabled(cfg))
		self.assertIsNone(
			fit_service._wire_external_surface_for_request(
				cfg=cfg,
				args_section=cfg["args"],
				model_init="seed",
				ext_offset_enabled=False,
				global_map_enabled=fit_service._config_global_map_enabled(cfg),
			)
		)

	def test_snap_surf_map_debug_obj_dir_rewrites_existing_outputs(self) -> None:
		cfg = {
			"stages": [
				{
					"name": "map_uv_fine",
					"params": ["map_surf_ms"],
					"args": {"debug_obj_dir": "snap_surf_objs"},
				},
				{
					"name": "snap",
					"params": ["mesh_ms"],
					"args": {
						"snap_surf_map": {
							"map_opt": {
								"args": {"debug_obj_dir": "snap_surf_objs"},
							}
						}
					},
				},
			],
		}

		fit_service._set_snap_surf_map_debug_obj_dir(cfg, "/tmp/snap_surf_objs")

		self.assertEqual(cfg["stages"][0]["args"]["debug_obj_dir"], "/tmp/snap_surf_objs")
		self.assertEqual(
			cfg["stages"][1]["args"]["snap_surf_map"]["map_opt"]["args"]["debug_obj_dir"],
			"/tmp/snap_surf_objs",
		)
		self.assertNotIn("snap_surf", cfg["stages"][1]["args"])

	def test_seed_mode_ignores_surplus_model_transport(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			model_input = fit_service._decode_model_for_request(
				body={"model_data": _b64(b"checkpoint")},
				tmp_dir=td,
				model_init="seed",
			)

			self.assertIsNone(model_input)
			self.assertFalse((Path(td) / "model_input.pt").exists())

	def test_model_mode_decodes_model_transport(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			model_input = fit_service._decode_model_for_request(
				body={"model_data": _b64(b"checkpoint")},
				tmp_dir=td,
				model_init="model",
			)

			self.assertIsNotNone(model_input)
			self.assertEqual(Path(model_input).read_bytes(), b"checkpoint")

if __name__ == "__main__":
	unittest.main()
