from __future__ import annotations

import argparse
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import cli_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser()
	p.add_argument("--input")
	p.add_argument("--seed", type=float, nargs=3)
	p.add_argument("--progress", action="store_true")
	p.add_argument("--extra", nargs="*")
	return p


class SplitCfgArgvTest(unittest.TestCase):
	def test_json_value_of_option_is_not_a_config(self):
		paths, rest = cli_json.split_cfg_argv(
			["base.json", "--input", "vol.lasagna.json", "--seed", "1", "2", "3"], _parser())
		self.assertEqual(paths, ["base.json"])
		self.assertEqual(rest, ["--input", "vol.lasagna.json", "--seed", "1", "2", "3"])

	def test_json_after_flag_is_a_config(self):
		paths, rest = cli_json.split_cfg_argv(["--progress", "stage.json", "--input", "v.json"], _parser())
		self.assertEqual(paths, ["stage.json"])
		self.assertEqual(rest, ["--progress", "--input", "v.json"])

	def test_equals_form_and_variadic_values(self):
		paths, rest = cli_json.split_cfg_argv(
			["--input=v.json", "a.json", "--extra", "x.json", "y.json"], _parser())
		self.assertEqual(paths, ["a.json"])
		self.assertEqual(rest, ["--input=v.json", "--extra", "x.json", "y.json"])

	def test_without_parser_keeps_old_behaviour(self):
		paths, rest = cli_json.split_cfg_argv(["--input", "v.json", "c.json"])
		self.assertEqual(paths, ["v.json", "c.json"])
		self.assertEqual(rest, ["--input"])

	def test_parse_args_reads_input_json_value(self):
		args = cli_json.parse_args(_parser(), ["--input", "vol.lasagna.json"])
		self.assertEqual(args.input, "vol.lasagna.json")


if __name__ == "__main__":
	unittest.main()
