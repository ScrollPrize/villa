from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import cyl_init_tool
import model as fit_model
import tifxyz_io


class _UmbilicusData:
	sparse_caches: dict = {}

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


def _model() -> fit_model.Model3D:
	return fit_model.Model3D(
		device=torch.device("cpu"),
		depth=1,
		mesh_h=3,
		mesh_w=5,
		mesh_step=10,
		winding_step=10,
		subsample_mesh=1,
		subsample_winding=1,
		init_mode="cylinder_seed",
		pyramid_d=False,
	)


def _shell(h: int = 2, w: int = 3) -> torch.Tensor:
	z = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	x = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	y = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	return torch.stack([x, y, z], dim=-1)


class CylinderInitToolConfigTest(unittest.TestCase):
	def test_parser_keeps_stage_config_and_input_lasagna_json_separate(self) -> None:
		args = cyl_init_tool._build_parser().parse_args([
			"init_cyl.json",
			"--input", "volume.lasagna.json",
			"--out-dir", "out",
			"--shell-count", "4",
		])

		self.assertEqual(args.config, "init_cyl.json")
		self.assertEqual(args.input, "volume.lasagna.json")
		self.assertEqual(args.out_dir, "out")
		self.assertEqual(args.shell_count, 4)

	def test_parser_does_not_expose_fit_mesh_options(self) -> None:
		help_text = cyl_init_tool._build_parser().format_help()

		for opt in (
			"--mesh-h",
			"--mesh-w",
			"--mesh-step",
			"--model-h",
			"--model-w",
			"--seed",
			"--pyramid-d",
			"--model-output",
		):
			self.assertNotIn(opt, help_text)

	def test_stage_filter_keeps_only_progression_stages(self) -> None:
		cfg = {
			"args": {"init-mode": "cylinder_seed"},
			"base": {"cyl_normal": 1.0},
			"stages": [
				{"name": "cyl_init", "steps": 1, "args": {"model-step": 20}},
				{"name": "mesh_refine", "steps": 1},
				{"name": "cyl_grow", "steps": 2, "lr": [1.0, 0.1]},
				{"name": "cyl_grow_refine", "steps": 3, "w_fac": {"cyl_normal": 0.5}},
				{"name": "cyl_polish", "steps": 4},
			],
		}

		got = cyl_init_tool._filter_cylinder_config(cfg)
		got_start = cyl_init_tool._filter_cylinder_config(cfg, start_shell=True)

		self.assertEqual([s["name"] for s in got["stages"]], ["cyl_init", "cyl_grow", "cyl_grow_refine"])
		self.assertEqual([s["name"] for s in got_start["stages"]], ["cyl_grow", "cyl_grow_refine"])
		self.assertEqual(got["base"], {"cyl_normal": 1.0})
		self.assertEqual(got["stages"][0]["args"], {"model-step": 20})
		self.assertEqual(got["stages"][2]["w_fac"], {"cyl_normal": 0.5})

	def test_shell_count_override_replaces_stage_values(self) -> None:
		cfg = {
			"stages": [
				{"name": "cyl_init", "args": {"cyl_max_shells": 6}},
				{"name": "cyl_grow", "args": {"cyl_shell_search_max_shells": 7}},
				{"name": "cyl_grow_refine", "args": {}},
			],
		}
		got = cyl_init_tool._filter_cylinder_config(cfg)

		cyl_init_tool._apply_shell_count_override(got, 3)

		for stage in got["stages"]:
			self.assertEqual(stage["args"]["cyl_max_shells"], 3)

	def test_shell_count_override_applies_without_cyl_init(self) -> None:
		cfg = {
			"stages": [
				{"name": "cyl_init", "args": {"cyl_max_shells": 6}},
				{"name": "cyl_grow", "steps": 1},
			],
		}
		got = cyl_init_tool._filter_cylinder_config(cfg, start_shell=True)

		cyl_init_tool._apply_shell_count_override(got, 2)

		self.assertEqual([stage["name"] for stage in got["stages"]], ["cyl_grow"])
		self.assertEqual(got["stages"][0]["args"]["cyl_max_shells"], 2)

	def test_derived_z_range_uses_umbilicus_padding_without_clamp(self) -> None:
		data = SimpleNamespace(
			umbilicus_points=torch.tensor([
				[0.0, 0.0, 10.0],
				[0.0, 0.0, 60.0],
				[0.0, 0.0, 110.0],
			])
		)

		z0, z1 = cyl_init_tool._derive_z_range_from_umbilicus(data)

		self.assertAlmostEqual(z0, 0.0, delta=1.0e-6)
		self.assertAlmostEqual(z1, 120.0, delta=1.0e-6)


class CylinderInitToolModelTest(unittest.TestCase):
	def test_exact_z_range_sets_center_and_shell_height_without_volume_clamp(self) -> None:
		mdl = _model()
		mdl.cyl_shell_z_step = 5.0

		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 20.0),
			model_w=100.0,
			model_h=20.0,
			volume_extent_fullres=(100, 100, 15),
			exact_z_range=(10.0, 30.0),
		)
		mdl.prepare_umbilicus_tube_init(_UmbilicusData())
		shell = mdl.current_cylinder_shell_xyz().detach()

		self.assertAlmostEqual(float(mdl.cyl_shell_z_center_target), 20.0, delta=1.0e-6)
		self.assertAlmostEqual(float(mdl.cyl_shell_model_h), 20.0, delta=1.0e-6)
		self.assertAlmostEqual(float(mdl.params.model_h), 20.0, delta=1.0e-6)
		self.assertAlmostEqual(float(shell[0, :, 2].mean()), 10.0, delta=1.0e-5)
		self.assertAlmostEqual(float(shell[-1, :, 2].mean()), 30.0, delta=1.0e-5)


class CylinderInitToolTifxyzTest(unittest.TestCase):
	def test_export_appends_duplicate_wrap_column(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			out = Path(td) / "shell_0001.tifxyz"
			cyl_init_tool._export_shell_tifxyz(out_dir=out, shell_xyz=_shell(), scale=0.1)

			xyz, valid, _meta = tifxyz_io.load_tifxyz(out, device="cpu")

		self.assertEqual(tuple(xyz.shape), (2, 4, 3))
		self.assertTrue(bool(valid.all()))
		self.assertTrue(torch.allclose(xyz[:, 0], xyz[:, -1]))

	def test_start_shell_validation_rejects_unwrapped_and_strips_duplicate_column(self) -> None:
		wrapped = torch.cat([_shell(), _shell()[:, :1]], dim=1)
		valid = torch.ones(wrapped.shape[:2], dtype=torch.bool)

		got = cyl_init_tool._strip_wrapped_start_shell(wrapped, valid)

		self.assertEqual(tuple(got.shape), (2, 3, 3))
		self.assertTrue(torch.allclose(got, _shell()))

		bad = wrapped.clone()
		bad[:, -1, 0] += 1.0
		with self.assertRaisesRegex(ValueError, "first and last columns"):
			cyl_init_tool._strip_wrapped_start_shell(bad, valid)

		invalid = valid.clone()
		invalid[0, 0] = False
		with self.assertRaisesRegex(ValueError, "invalid"):
			cyl_init_tool._strip_wrapped_start_shell(wrapped, invalid)

	def test_exporter_callback_writes_incrementing_shell_directories(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			exporter = cyl_init_tool._ShellExporter(out_dir=Path(td), scale=0.1)
			exporter(shell_index=0, shell_xyz=_shell(), stage_label="stage0.cyl_init", data=_UmbilicusData())
			exporter(shell_index=1, shell_xyz=_shell(), stage_label="stage1.cyl_grow_shell2", data=_UmbilicusData())

			first, _, _ = tifxyz_io.load_tifxyz(Path(td) / "shell_0001.tifxyz", device="cpu")
			second, _, _ = tifxyz_io.load_tifxyz(Path(td) / "interm_shell_0002.tifxyz", device="cpu")

		self.assertEqual(tuple(first.shape), (2, 4, 3))
		self.assertEqual(tuple(second.shape), (2, 4, 3))


if __name__ == "__main__":
	unittest.main()
