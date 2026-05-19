from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_data
import model as fit_model
import optimizer


def _stage_cfg(stages: list[dict]) -> dict:
	return {
		"args": {"init-mode": "cylinder_seed"},
		"base": {"cyl_normal": 1.0},
		"stages": stages,
	}


def _role_stage(
	name: str,
	*,
	params: list[str] | None = None,
	args: dict | None = None,
	steps: int = 0,
) -> dict:
	stage = {
		"name": name,
		"steps": int(steps),
		"lr": 0.1,
		"params": ["cyl_params"] if params is None else params,
	}
	if args is not None:
		stage["args"] = dict(args)
	return stage


class CylinderSeedRoleValidationTest(unittest.TestCase):
	def test_valid_roles_pass(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init"),
			_role_stage("cyl_grow"),
			_role_stage("cyl_grow_refine"),
			_role_stage("cyl_refine"),
			_role_stage("cyl_polish"),
		]))

		self.assertEqual(
			[s.name for s in stages],
			["cyl_init", "cyl_grow", "cyl_grow_refine", "cyl_refine", "cyl_polish"],
		)

	def test_cyl_refine_is_not_required(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init"),
			_role_stage("cyl_grow"),
		]))

		self.assertEqual([s.name for s in stages], ["cyl_init", "cyl_grow"])

	def test_cyl_grow_is_not_required(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init"),
		]))

		self.assertEqual([s.name for s in stages], ["cyl_init"])

	def test_missing_cyl_init_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "missing required stage role"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_grow"),
			]))

	def test_duplicate_role_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "duplicated"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_init"),
				_role_stage("cyl_grow"),
				_role_stage("cyl_grow"),
			]))

	def test_extra_cyl_params_stage_after_grow_passes(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init"),
			_role_stage("cyl_grow"),
			_role_stage("custom_cyl_stage"),
		]))

		self.assertEqual([s.name for s in stages], ["cyl_init", "cyl_grow", "custom_cyl_stage"])

	def test_extra_cyl_params_stage_before_grow_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "contiguous before later"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_init"),
				_role_stage("cylinder_seed"),
				_role_stage("cyl_grow"),
			]))

	def test_cyl_grow_refine_before_grow_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "must follow cyl_grow"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_init"),
				_role_stage("cyl_grow_refine"),
				_role_stage("cyl_grow"),
			]))

	def test_role_without_cyl_params_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "must have params"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_init"),
				_role_stage("cyl_grow", params=["mesh_ms"]),
			]))

	def test_old_cyl_step_arg_fails(self) -> None:
		with self.assertRaisesRegex(ValueError, "no longer supported"):
			optimizer.load_stages_cfg(_stage_cfg([
				_role_stage("cyl_init"),
				_role_stage("cyl_grow"),
				_role_stage("cyl_refine", args={"cyl_shell_width_step": 50.0}),
			]))

	def test_model_step_arg_passes(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init"),
			_role_stage("cyl_grow"),
			_role_stage("cyl_refine", args={"model-step": 50.0}),
		]))

		self.assertEqual(stages[-1].global_opt.args["model-step"], 50.0)

	def test_load_stages_reports_json_syntax_error(self) -> None:
		with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
			f.write('{"stages": [}')
			path = f.name
		try:
			with self.assertRaisesRegex(ValueError, r"invalid JSON.*line .*column"):
				optimizer.load_stages(path)
		finally:
			os.unlink(path)

	def test_missing_stages_key_reports_missing_key(self) -> None:
		with self.assertRaisesRegex(ValueError, "missing required key 'stages'"):
			optimizer.load_stages_cfg({"base": {}})

	def test_wrong_stages_type_reports_type(self) -> None:
		with self.assertRaisesRegex(ValueError, "got dict"):
			optimizer.load_stages_cfg({"stages": {}})


class _FakeData:
	sparse_caches: dict = {}
	corr_points = None
	winding_volume = None


class _FakeShellModel(torch.nn.Module):
	def __init__(
		self,
		classifications: list[object],
		*,
		width_step_avg: float = 100.0,
		reset_width_on_resample: bool = True,
		bracket_fail_on_call: int | None = None,
	) -> None:
		super().__init__()
		self.param = torch.nn.Parameter(torch.tensor(0.0))
		self.cylinder_enabled = True
		self.cyl_shell_mode = True
		self.cyl_shell_completed: list[torch.Tensor] = []
		self.cyl_shell_search_max_shells = 5
		self.cyl_shell_target_count = 1
		self.cyl_shell_width_target_step = 100.0
		self.cyl_shell_growth_factor = 2.0
		self.cyl_shell_current_width_step = 100.0
		self.cyl_shell_search_direction = 1
		self.cyl_shell_search_crossed = False
		self.cyl_shell_search_done = False
		self.cyl_shell_search_last_class = None
		self.classifications = list(classifications)
		self.begin_calls: list[tuple[int, int]] = []
		self.refine_calls: list[int] = []
		self.resample_calls: list[int] = []
		self.step_resample_calls: list[float] = []
		self.radius_checks: list[tuple[int, int, float]] = []
		self.forward_width_steps: list[float] = []
		self.measure_calls = 0
		self.bracket_calls = 0
		self.bracket_fail_on_call = bracket_fail_on_call
		self._width_step_avg = float(width_step_avg)
		self._reset_width_on_resample = bool(reset_width_on_resample)
		self._active_idx = 0
		self._active_refine = False

	def opt_params(self) -> dict[str, list[torch.nn.Parameter]]:
		return {"cyl_params": [self.param]}

	def prepare_umbilicus_tube_init(self, data: _FakeData) -> None:
		return None

	def begin_cylinder_shell(self, idx: int, data: _FakeData, *, direction: int = 1) -> None:
		self.begin_calls.append((int(idx), int(direction)))
		self.cyl_shell_search_direction = int(direction)
		self._active_idx = int(idx)
		self._active_refine = False

	def begin_cylinder_shell_refine(self, data: _FakeData) -> None:
		idx = len(self.cyl_shell_completed) - 1
		self.refine_calls.append(idx)
		self._active_idx = idx
		self._active_refine = True

	def resample_current_cylinder_shell_width_for_growth(self, data: _FakeData, *, direction: int = 1) -> None:
		self.resample_calls.append(int(direction))

	def resample_current_cylinder_shell_width_to_step(self, data: _FakeData, target_step: float) -> None:
		self.step_resample_calls.append(float(target_step))
		self.cyl_shell_width_target_step = float(target_step)
		self.cyl_shell_current_width_step = float(target_step)
		if self._reset_width_on_resample:
			self._width_step_avg = float(target_step)

	def complete_current_cylinder_shell(self, data: _FakeData) -> None:
		shell = torch.zeros(2, 3, 3)
		idx = self._active_idx
		if len(self.cyl_shell_completed) > idx:
			self.cyl_shell_completed[idx] = shell
		elif len(self.cyl_shell_completed) == idx:
			self.cyl_shell_completed.append(shell)
		else:
			raise AssertionError("shell list gap in fake model")

	def assert_cylinder_shell_brackets_seed(self, shell=None, *, label: str = "cylinder shell") -> None:
		self.bracket_calls += 1
		if self.bracket_fail_on_call is not None and self.bracket_calls >= int(self.bracket_fail_on_call):
			raise ValueError(f"{label}: fake bracket failure")

	def measure_seed_vs_current_cylinder_shell(self, seed=None):
		self.measure_calls += 1
		if not self.classifications:
			cls = "edge"
			signed = 0.0
		else:
			raw = self.classifications.pop(0)
			if isinstance(raw, tuple):
				cls = str(raw[0])
				signed = float(raw[1])
			else:
				cls = str(raw)
				signed = 0.0 if cls == "edge" else (100.0 if cls == "outside" else -100.0)
		return fit_model.SeedShellMetrics(
			classification=cls,
			signed_distance=signed,
			abs_distance=abs(signed),
			tolerance=1.0e-3,
			row_index=0,
		)

	def cylinder_shell_search_target_radius(self, idx: int, *, direction: int) -> float:
		radius = 1000.0 + float(direction) * float(idx) * 100.0
		self.radius_checks.append((int(idx), int(direction), radius))
		return radius

	def _shell_width_step_stats(self) -> tuple[float, float, float]:
		avg = float(self._width_step_avg)
		return (avg, 0.9 * avg, 1.1 * avg)

	def current_cylinder_shell_xyz(self) -> torch.Tensor:
		return torch.zeros(2, 3, 3)

	def forward(self, data: _FakeData, *, needs=None) -> fit_model.FitResult3D:
		self.forward_width_steps.append(float(self.cyl_shell_current_width_step))
		xyz = self.param.view(1, 1, 1, 1).expand(1, 2, 3, 3)
		return fit_model.FitResult3D(
			xyz_lr=xyz,
			xyz_hr=None,
			data=data,
			data_s=None,
			data_lr=None,
			target_plain=None,
			target_mod=None,
			amp_lr=torch.zeros(1, 1, 2, 3),
			bias_lr=torch.zeros(1, 1, 2, 3),
			mask_hr=None,
			mask_lr=None,
			normals=None,
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
			gt_normal_lr=None,
			ext_conn=None,
			cyl_xyz=xyz,
			cyl_normals=torch.ones_like(xyz),
			cyl_centers=None,
			cyl_axes=None,
			cyl_params=None,
			cyl_count=1,
			cyl_shell_mode=True,
			cyl_shell_width_step=float(self.cyl_shell_current_width_step),
			cyl_shell_delta_xyz=torch.zeros(2, 3, 3),
			cyl_shell_index=self._active_idx,
		)


def _zero_cyl_loss(*, res: fit_model.FitResult3D):
	zero = res.xyz_lr.sum() * 0.0
	return zero, (zero.view(1, 1, 1, 1),), (zero.view(1, 1, 1, 1),)


class CylinderSeedDynamicSearchSmokeTest(unittest.TestCase):
	def _run_search(
		self,
		classifications: list[object],
		*,
		max_shells: int | None = None,
		grow_steps: int = 5,
		grow_refine_steps: int | None = None,
		grow_refine_args: dict | None = None,
		post_steps: int = 1,
		post_model_step: float = 100.0,
		width_step_avg: float = 100.0,
		init_args: dict | None = None,
		grow_args: dict | None = None,
		reset_width_on_resample: bool = True,
		bracket_fail_on_call: int | None = None,
	) -> tuple[_FakeShellModel, str]:
		stage_defs = [
			_role_stage("cyl_init", args=init_args),
			_role_stage("cyl_grow", steps=grow_steps, args=grow_args),
		]
		if grow_refine_steps is not None:
			stage_defs.append(_role_stage("cyl_grow_refine", steps=grow_refine_steps, args=grow_refine_args))
		stage_defs.extend([
			_role_stage("cyl_refine", args={"model-step": post_model_step}, steps=post_steps),
			_role_stage("cyl_polish", steps=post_steps),
		])
		stages = optimizer.load_stages_cfg(_stage_cfg(stage_defs))
		mdl = _FakeShellModel(
			classifications,
			width_step_avg=width_step_avg,
			reset_width_on_resample=reset_width_on_resample,
			bracket_fail_on_call=bracket_fail_on_call,
		)
		if max_shells is not None:
			mdl.cyl_shell_search_max_shells = int(max_shells)
		stdout = io.StringIO()
		with mock.patch.object(optimizer.opt_loss_cyl, "cyl_normal_loss", _zero_cyl_loss):
			with contextlib.redirect_stdout(stdout):
				optimizer.optimize(
					model=mdl,
					data=_FakeData(),
					stages=stages,
					snapshot_interval=0,
					snapshot_fn=lambda **_: None,
					seed_xyz=(0.0, 0.0, 0.0),
				)
		return mdl, stdout.getvalue()

	def test_init_only_without_grow_outputs_first_shell_and_skips_later_stages(self) -> None:
		stages = optimizer.load_stages_cfg(_stage_cfg([
			_role_stage("cyl_init", steps=1),
			_role_stage("cyl_refine", steps=1),
			_role_stage("cyl_polish", steps=1),
		]))
		mdl = _FakeShellModel(["outside"])
		stdout = io.StringIO()
		with mock.patch.object(optimizer.opt_loss_cyl, "cyl_normal_loss", _zero_cyl_loss):
			with contextlib.redirect_stdout(stdout):
				optimizer.optimize(
					model=mdl,
					data=_FakeData(),
					stages=stages,
					snapshot_interval=0,
					snapshot_fn=lambda **_: None,
					seed_xyz=(0.0, 0.0, 0.0),
				)

		self.assertEqual(mdl.begin_calls, [(0, 1)])
		self.assertEqual(mdl.refine_calls, [])
		self.assertEqual(len(mdl.cyl_shell_completed), 1)
		self.assertEqual(mdl.measure_calls, 0)
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertRegex(stdout.getvalue(), r"(?m)^\s*1\s+0\s+")
		self.assertIn("no cyl_grow stage", stdout.getvalue())

	def test_default_grow_runs_to_shell_cap_and_collapses_to_last(self) -> None:
		mdl, stdout = self._run_search(["inside"])

		self.assertIn("shell cap 5", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
		self.assertEqual(mdl.measure_calls, 0)
		self.assertEqual(mdl.bracket_calls, 0)
		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertFalse(mdl.cyl_shell_search_crossed)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_shrink_grow_direction_runs_fixed_reverse_growth(self) -> None:
		mdl, _stdout = self._run_search(
			["outside"],
			grow_args={"cyl_grow_direction": "shrink"},
		)

		self.assertEqual(mdl.begin_calls, [(0, 1), (1, -1), (2, -1), (3, -1), (4, -1)])
		self.assertEqual(
			mdl.radius_checks,
			[(1, -1, 900.0), (2, -1, 800.0), (3, -1, 700.0), (4, -1, 600.0)],
		)
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_invalid_grow_direction_reports_clear_error(self) -> None:
		for bad in ("sideways", "auto", 0):
			with self.subTest(bad=bad):
				with self.assertRaisesRegex(ValueError, "cyl_grow_direction"):
					self._run_search(
						["outside", "inside"],
						grow_args={"cyl_grow_direction": bad},
					)

	def test_step_push_loss_is_only_active_during_grow_role(self) -> None:
		cfg = {
			"args": {"init-mode": "cylinder_seed"},
			"base": {"cyl_normal": 1.0, "cyl_step_push": 1.0},
			"stages": [
				_role_stage("cyl_init", steps=1),
				_role_stage("cyl_grow", steps=1),
				_role_stage("cyl_grow_refine", steps=1),
				_role_stage("cyl_refine", steps=1),
			],
		}
		stages = optimizer.load_stages_cfg(cfg)
		mdl = _FakeShellModel(["outside", "inside"])
		calls: list[int] = []

		def _record_step_push(*, res: fit_model.FitResult3D):
			calls.append(int(getattr(res, "cyl_shell_index", -1)))
			return _zero_cyl_loss(res=res)

		with mock.patch.object(optimizer.opt_loss_cyl, "cyl_normal_loss", _zero_cyl_loss):
			with mock.patch.object(optimizer.opt_loss_cyl, "cyl_step_push_loss", _record_step_push):
				with contextlib.redirect_stdout(io.StringIO()):
					optimizer.optimize(
						model=mdl,
						data=_FakeData(),
						stages=stages,
						snapshot_interval=0,
						snapshot_fn=lambda **_: None,
						seed_xyz=(0.0, 0.0, 0.0),
					)

		self.assertGreater(len(calls), 0)
		self.assertEqual(set(calls), {1, 2, 3, 4})

	def test_output_all_shells_flag_preserves_all_progression_shells(self) -> None:
		mdl, _stdout = self._run_search(
			["outside"],
			init_args={"cyl_output_all_shells": True},
		)

		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 5)

	def test_shell_cap_is_normal_completion_and_collapses_completed_shells(self) -> None:
		mdl, stdout = self._run_search(
			["outside"],
			grow_steps=2,
			max_shells=2,
			post_steps=0,
			width_step_avg=150.0,
			reset_width_on_resample=False,
		)

		self.assertNotIn("ERROR", stdout)
		self.assertIn("shell cap 2", stdout)
		self.assertIn("outputting completed shells", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1)])
		self.assertEqual(mdl.refine_calls, [0, 0])
		self.assertEqual(mdl.resample_calls, [])
		self.assertEqual(mdl.step_resample_calls, [100.0])
		self.assertFalse(mdl.cyl_shell_search_crossed)
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_search_cap_arg_overrides_model_default(self) -> None:
		mdl, stdout = self._run_search(
			["outside"],
			grow_steps=2,
			init_args={"cyl_output_all_shells": True},
			grow_args={"cyl_max_shells": 2},
			post_steps=0,
			width_step_avg=150.0,
			reset_width_on_resample=False,
		)

		self.assertIn("shell cap 2", stdout)
		self.assertIn("adding cylinder shell 2/2", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1)])
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 2)

	def test_grow_does_not_measure_or_bracket_seed(self) -> None:
		mdl, stdout = self._run_search(
			["outside", ("outside", 1000.0)],
			grow_steps=3,
			post_steps=1,
			bracket_fail_on_call=2,
		)

		self.assertNotIn("fake bracket failure", stdout)
		self.assertEqual(mdl.measure_calls, 0)
		self.assertEqual(mdl.bracket_calls, 0)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_grow_steps_zero_still_builds_shells_to_cap_without_iterations(self) -> None:
		mdl, stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=0,
			post_steps=0,
		)

		self.assertIn("adding cylinder shell 2/5", stdout)
		self.assertNotIn("ERROR", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_grow_status_reports_shell_column_every_status_interval(self) -> None:
		_mdl, stdout = self._run_search(
			["outside"] + [("outside", 1000.0)] * 100 + [("inside", -1.0)],
			grow_steps=100,
			max_shells=2,
			post_steps=0,
		)

		self.assertIn("adding cylinder shell 2/2", stdout)
		self.assertRegex(stdout, r"(?m)^\s*shell\s+step\s+loss")
		self.assertRegex(stdout, r"(?m)^\s*2\s+0\s+")
		self.assertRegex(stdout, r"(?m)^\s*2\s+100\s+")
		self.assertNotRegex(stdout, r"(?m)^\s*2\s+1\s+")
		self.assertIn("cavg", stdout)
		self.assertIn("ctgt", stdout)
		self.assertIn("iavg", stdout)
		self.assertIn("ifrac", stdout)
		self.assertNotIn("wmin", stdout)
		self.assertNotIn("wmax", stdout)

	def test_grow_refine_runs_after_each_grow_shell_with_fixed_start_avg_step(self) -> None:
		mdl, stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=1,
			grow_refine_steps=1,
			post_steps=0,
			width_step_avg=123.0,
		)

		self.assertNotIn("ERROR", stdout)
		self.assertIn(1, mdl.refine_calls)
		self.assertIn(4, mdl.refine_calls)
		self.assertIn(150.0, mdl.forward_width_steps)
		self.assertIn(100.0, mdl.forward_width_steps)

	def test_grow_refine_runs_without_seed_detection(self) -> None:
		mdl, stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=1,
			grow_refine_steps=1,
			post_steps=0,
		)

		self.assertNotIn("ERROR", stdout)
		self.assertEqual(mdl.refine_calls[0], 1)
		self.assertEqual(mdl.measure_calls, 0)
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_grow_refine_stops_search_when_ifrac_exceeds_default(self) -> None:
		with mock.patch.object(
			optimizer.opt_loss_cyl,
			"cyl_shell_width_edge_stats",
			return_value={"valid_avg_vx": 100.0, "invalid_avg_vx": 100.0, "invalid_frac": 0.75},
		):
			mdl, stdout = self._run_search(
				["outside"],
				grow_steps=1,
				grow_refine_steps=1,
				post_steps=1,
			)

		self.assertIn("cyl_refine_max_ifrac", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1)])
		self.assertEqual(mdl.refine_calls, [1])
		self.assertTrue(bool(getattr(mdl, "cyl_shell_abort", False)))

	def test_grow_refine_ifrac_stop_can_be_disabled(self) -> None:
		with mock.patch.object(
			optimizer.opt_loss_cyl,
			"cyl_shell_width_edge_stats",
			return_value={"valid_avg_vx": 100.0, "invalid_avg_vx": 100.0, "invalid_frac": 0.75},
		):
			mdl, stdout = self._run_search(
				["outside"],
				grow_steps=1,
				grow_refine_steps=1,
				grow_refine_args={"cyl_refine_max_ifrac": -1},
				post_steps=0,
			)

		self.assertNotIn("cyl_refine_max_ifrac", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
		self.assertEqual(mdl.refine_calls[:4], [1, 2, 3, 4])
		self.assertFalse(bool(getattr(mdl, "cyl_shell_abort", False)))

	def test_seed_distance_does_not_stop_fixed_count_growth(self) -> None:
		mdl, stdout = self._run_search(
			[
				"outside",
				("outside", 0.5),
				("outside", 0.5),
			],
			grow_steps=5,
			grow_refine_steps=1,
			post_steps=0,
		)

		self.assertNotIn("ERROR", stdout)
		self.assertEqual(mdl.begin_calls, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
		self.assertEqual(mdl.refine_calls[0], 1)
		self.assertEqual(mdl.measure_calls, 0)
		self.assertTrue(mdl.cyl_shell_search_done)
		self.assertEqual(len(mdl.cyl_shell_completed), 1)

	def test_linear_grow_resamples_back_to_model_step_after_pass(self) -> None:
		mdl, _stdout = self._run_search(
			[
				("outside", 1000.0),
				("outside", 1000.0),
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=3,
		)

		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_linear_grow_uses_model_step_without_compounding_from_actual_avg(self) -> None:
		mdl, _stdout = self._run_search(
			["outside"] + [("outside", 1000.0)] * 10 + [("inside", -1.0)],
			grow_steps=10,
			post_steps=0,
			width_step_avg=50.0,
		)

		self.assertIn(150.0, mdl.forward_width_steps)
		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_grow_target_linearly_expands_to_default_factor(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=2,
			post_steps=0,
		)

		self.assertIn(125.0, mdl.forward_width_steps)
		self.assertIn(150.0, mdl.forward_width_steps)
		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_grow_target_linearly_shrinks_to_inverse_default_factor(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"inside",
				("inside", -1000.0),
				("inside", -1000.0),
				("outside", 1.0),
			],
			grow_steps=2,
			post_steps=0,
			grow_args={"cyl_grow_direction": "inward"},
		)

		self.assertIn(100.0 + 0.5 * ((100.0 / 1.5) - 100.0), mdl.forward_width_steps)
		self.assertIn(100.0 / 1.5, mdl.forward_width_steps)
		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_grow_factor_arg_changes_linear_target_end(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=2,
			post_steps=0,
			grow_args={"cyl_grow_factor": 1.2},
		)

		self.assertIn(110.0, mdl.forward_width_steps)
		self.assertIn(120.0, mdl.forward_width_steps)
		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_growth_resamples_each_shell_back_to_model_step(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"outside",
				("outside", 1000.0),
				("inside", -1.0),
			],
			grow_steps=2,
			post_steps=0,
			width_step_avg=150.0,
		)

		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_post_cylinder_stages_track_current_avg_wtarget(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"edge",
				("outside", 1000.0),
				("outside", 1000.0),
				("outside", 1000.0),
				("outside", 1000.0),
			],
			post_steps=1,
			width_step_avg=150.0,
		)

		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0])
		self.assertEqual(len(mdl.cyl_shell_completed), 1)
		self.assertTrue(mdl.cyl_shell_search_done)

	def test_refine_initial_eval_uses_current_avg_wtarget(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"edge",
				("outside", 1000.0),
				("outside", 1000.0),
				("outside", 1000.0),
				("outside", 1000.0),
			],
			post_steps=0,
			width_step_avg=150.0,
		)

		self.assertIn(100.0, mdl.forward_width_steps)

	def test_changed_refine_model_step_resamples_before_initial_eval(self) -> None:
		mdl, _stdout = self._run_search(
			[
				"edge",
				("outside", 100.0),
				("outside", 100.0),
				("outside", 100.0),
				("outside", 100.0),
			],
			post_model_step=50.0,
			post_steps=0,
			width_step_avg=150.0,
		)

		self.assertEqual(mdl.step_resample_calls, [100.0, 100.0, 100.0, 100.0, 50.0])
		self.assertIn(50.0, mdl.forward_width_steps)

	def test_cyl_outside_field_builds_for_grow_reuses_for_refine_and_replaces_next_shell(self) -> None:
		cfg = {
			"args": {"init-mode": "cylinder_seed"},
			"base": {"cyl_normal": 1.0, "cyl_outside": 1.0},
			"stages": [
				_role_stage("cyl_init"),
				_role_stage("cyl_grow", steps=1, args={
					"cyl_max_shells": 3,
					"cyl_outside_grid_step": 8.0,
					"cyl_outside_chunk_size": 4,
					"cyl_outside_deep_interp_chunks": 1.5,
					"cyl_outside_deep_blend_chunks": 0.5,
				}),
				_role_stage("cyl_grow_refine", steps=1, args={"cyl_outside_sample_factor": 3}),
			],
		}
		stages = optimizer.load_stages_cfg(cfg)
		mdl = _FakeShellModel(["outside"])
		build_calls: list[torch.Tensor] = []
		chunk_sizes: list[int] = []
		deep_interp_values: list[float] = []
		deep_blend_values: list[float] = []

		def _fake_build(
			shell,
			*,
			grid_step,
			device,
			bbox=None,
			chunk_size=8,
			deep_interp_chunks=10.0,
			deep_blend_chunks=2.0,
			**_kwargs,
		):
			build_calls.append(shell.detach().clone())
			chunk_sizes.append(int(chunk_size))
			deep_interp_values.append(float(deep_interp_chunks))
			deep_blend_values.append(float(deep_blend_chunks))
			return optimizer.cyl_sdf_volume.CylOutsideVolume(
				volume=torch.zeros(1, 1, 1, 1, dtype=torch.uint8, device=device),
				origin=(0.0, 0.0, 0.0),
				spacing=(float(grid_step), float(grid_step), float(grid_step)),
				shape=(1, 1, 1),
				depth_max=float(len(build_calls)),
			)

		with mock.patch.object(optimizer.opt_loss_cyl, "cyl_normal_loss", _zero_cyl_loss):
			with mock.patch.object(optimizer.opt_loss_cyl, "cyl_outside_loss", _zero_cyl_loss):
				with mock.patch.object(optimizer.cyl_sdf_volume, "build_previous_shell_inside_depth_volume", side_effect=_fake_build):
					with contextlib.redirect_stdout(io.StringIO()):
						optimizer.optimize(
							model=mdl,
							data=_FakeData(),
							stages=stages,
							snapshot_interval=0,
							snapshot_fn=lambda **_: None,
							seed_xyz=(0.0, 0.0, 0.0),
						)

		self.assertEqual(len(build_calls), 2)
		self.assertEqual(chunk_sizes, [4, 4])
		self.assertEqual(deep_interp_values, [1.5, 1.5])
		self.assertEqual(deep_blend_values, [0.5, 0.5])
		self.assertEqual(float(mdl.cyl_outside_depth_max), 2.0)
		self.assertEqual(int(mdl.cyl_outside_sample_factor), 3)


if __name__ == "__main__":
	unittest.main()
