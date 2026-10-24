from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from optimizer.reporting import StageReporter
from optimizer.timing import FlowTimingWindow, OptTimingWindow


def capture_status(reporter: StageReporter, **kwargs) -> list[str]:
	args = dict(step_label="flatten 1/30", loss_val=1.25, tv={}, pv={}, its=25.0)
	args.update(kwargs)
	with redirect_stdout(io.StringIO()) as output:
		reporter.print_status(**args)
	return output.getvalue().splitlines()


class StageReporterTest(unittest.TestCase):
	def test_columns_keep_labels_values_and_parameter_order(self) -> None:
		reporter = StageReporter(stage_args={}, step_width=16)
		lines = capture_status(
			reporter,
			tv={"flatten_sdir_contrib": 3.0, "flatten_sdir": 1.5},
			pv={"bias": 0.25, "amp": 2.0},
		)
		self.assertEqual(
			lines[-2].split(),
			["step", "loss", "it/s", "f_sdir", "f_sdirW", "p:amp", "p:bias"],
		)
		self.assertEqual(
			lines[-1].split(),
			["flatten", "1/30", "1.2500", "25.0", "1.5000", "3.0000", "2.0000", "0.2500"],
		)

	def test_header_repeats_without_repeating_legend(self) -> None:
		reporter = StageReporter(stage_args={}, step_width=16)
		first = capture_status(reporter)
		for _ in range(19):
			self.assertEqual(len(capture_status(reporter)), 1)
		self.assertEqual(capture_status(reporter), first[-2:])

	def test_column_changes_refresh_legend_and_stages_have_separate_state(self) -> None:
		reporter = StageReporter(stage_args={}, step_width=16)
		first = capture_status(reporter)
		changed = capture_status(reporter, tv={"flatten_sdir": 2.0})
		self.assertEqual(changed[0], "[optimizer] progress columns")
		self.assertEqual(changed[-2].split(), ["step", "loss", "it/s", "f_sdir"])
		other_stage = StageReporter(stage_args={}, step_width=16)
		self.assertEqual(capture_status(other_stage), first)

	def test_shell_passes_keep_header_state_and_widen_step_column(self) -> None:
		reporter = StageReporter(stage_args={}, step_width=16)
		first = capture_status(reporter, shell_no=1)
		self.assertEqual(first[-2].split(), ["shell", "step", "loss", "it/s"])
		for _ in range(20):
			self.assertEqual(len(capture_status(reporter, shell_no=2)), 1)
		reporter.ensure_step_width(32)
		reporter.ensure_step_width(16)
		forced = capture_status(reporter, shell_no=2, force_header=True)
		self.assertEqual(len(forced), 2)
		self.assertEqual(forced[0].index("loss"), first[-2].index("loss") + 16)
		self.assertEqual(forced[1].split()[0], "2")

	def test_verbose_flags_reveal_diagnostic_columns(self) -> None:
		for flag, visible, diagnostic in (
			("map_status", "snaps_map_snap", "snaps_map_active"),
			("pred_dt_flow_status", "pred_dt_gate_gt01", "pred_dt_gate_gt0"),
			("atlas_line_status", "atlas_line", "atlas_line_samples"),
		):
			with self.subTest(flag=flag):
				terms = {visible: 1.0, diagnostic: 2.0}
				quiet = capture_status(StageReporter(stage_args={}, step_width=16), tv=terms)
				self.assertEqual(quiet[-1].split()[4:], ["1.0000"])
				for prefix in ("verbose_", "debug_"):
					reporter = StageReporter(stage_args={prefix + flag: True}, step_width=16)
					verbose = capture_status(reporter, tv=terms)
					self.assertCountEqual(verbose[-1].split()[4:], ["1.0000", "2.0000"])


class TimingWindowTest(unittest.TestCase):
	def test_windows_flush_at_interval_and_final_step_then_reset(self) -> None:
		for window_type in (FlowTimingWindow, OptTimingWindow):
			with self.subTest(window=window_type.__name__):
				window = window_type(interval=2)
				with redirect_stdout(io.StringIO()) as output:
					window.add("total", 0.2)
					window.finish_iter(label="flatten", step1=1, max_steps=3)
					self.assertEqual(output.getvalue(), "")
					window.add("total", 0.4)
					window.finish_iter(label="flatten", step1=2, max_steps=3)
					window.add("total", 0.1)
					window.finish_iter(label="flatten", step1=3, max_steps=3)
				lines = output.getvalue().splitlines()
				self.assertIn("flatten 2/3 over 2 iters", lines[0])
				self.assertTrue(any("flatten 3/3 over 1 iters" in line for line in lines))
				other_rows = [line.split() for line in lines if line.startswith("other ")]
				self.assertEqual([row[2] for row in other_rows], ["300.00", "100.00"])


if __name__ == "__main__":
	unittest.main()
