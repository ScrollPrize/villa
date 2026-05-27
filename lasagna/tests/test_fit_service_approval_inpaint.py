from __future__ import annotations

import base64
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
	return base64.b64encode(data).decode("ascii")


class FitServiceApprovalInpaintTest(unittest.TestCase):
	def test_seed_mode_accepts_generic_tifxyz_and_ignores_without_consumer(self) -> None:
		body = {
			"tifxyz": {
				"x.tif": _b64(b"x"),
				"y.tif": _b64(b"y"),
				"z.tif": _b64(b"z"),
				"meta.json": _b64(b"{}"),
				"approval.tif": _b64(b"approval"),
				"d.tif": _b64(b"d"),
			}
		}
		cfg = {"args": {"model-init": "seed"}}
		args = cfg["args"]

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=args,
				tmp_dir=td,
				model_init="seed",
				ext_offset_enabled=False,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual((Path(tifxyz_dir) / "approval.tif").read_bytes(), b"approval")
			self.assertNotIn("approval-inpaint-tifxyz", args)
			self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_wires_generic_tifxyz(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x"), "approval.tif": _b64(b"approval")}}
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}}
		args = cfg["args"]

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=args,
				tmp_dir=td,
				model_init="seed",
				ext_offset_enabled=False,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual(args["approval-inpaint-tifxyz"], tifxyz_dir)
			self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_requires_seed_model_init(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x")}}
		cfg = {"args": {"model-init": "ext", "approval-inpaint": True}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "only valid with args.model-init=seed"):
				fit_service._decode_tifxyz_for_request(
					body=body,
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="ext",
					ext_offset_enabled=False,
				)

	def test_approval_inpaint_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "approval-inpaint requires request tifxyz"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="seed",
					ext_offset_enabled=False,
				)

	def test_ext_init_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "ext"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "model-init=ext requires request tifxyz"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="ext",
					ext_offset_enabled=False,
				)

	def test_flatten_does_not_synthesize_generic_tifxyz_as_external_surface(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x"), "y.tif": _b64(b"y"), "z.tif": _b64(b"z")}}
		cfg = {"args": {"model-init": "flatten"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz"}]}

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=cfg["args"],
				tmp_dir=td,
				model_init="flatten",
				ext_offset_enabled=False,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual(cfg["external_surfaces"], [{"path": "/tmp/ref.tifxyz"}])
			self.assertNotIn("tifxyz-init", cfg["args"])

	def test_flatten_requires_generic_tifxyz_or_external_surface(self) -> None:
		cfg = {"args": {"model-init": "flatten"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "model-init=flatten requires config external_surfaces"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="flatten",
					ext_offset_enabled=False,
				)

	def test_ext_offset_requires_config_external_surfaces(self) -> None:
		cfg = {"args": {"model-init": "seed"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "ext_offset is enabled but config has no external_surfaces"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="seed",
					ext_offset_enabled=True,
				)

	def test_global_map_does_not_synthesize_generic_tifxyz_as_external_surface(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x"), "y.tif": _b64(b"y"), "z.tif": _b64(b"z")}}
		cfg = {"args": {"model-init": "seed"}, "external_surfaces": [{"path": "/tmp/ref.tifxyz", "offset": -1.0}]}
		args = cfg["args"]

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=args,
				tmp_dir=td,
				model_init="seed",
				ext_offset_enabled=False,
				global_map_enabled=True,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual(cfg["external_surfaces"], [{"path": "/tmp/ref.tifxyz", "offset": -1.0}])
			self.assertNotIn("tifxyz-init", args)

	def test_global_map_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "seed"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "snap_surf_map/global_map is enabled but config has no external_surfaces"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
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
