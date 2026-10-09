from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _option_value_count(action: argparse.Action) -> int:
	"""How many following argv tokens an option consumes as its value(s)."""
	nargs = action.nargs
	if nargs is None or nargs == "?":
		return 1
	if isinstance(nargs, int):
		return nargs  # 0 for flags such as store_true
	if nargs in ("*", "+", argparse.REMAINDER):
		return sys.maxsize
	return 0


def split_cfg_argv(
	argv: list[str] | None,
	parser: argparse.ArgumentParser | None = None,
) -> tuple[list[str], list[str] | None]:
	"""Separate config ``*.json`` paths from the remaining CLI tokens.

	With ``parser``, a ``.json`` token that is the value of a preceding option
	(for example ``--input vol.lasagna.json``) stays with that option instead of
	being taken as a config file.
	"""
	if argv is None:
		return [], None
	paths: list[str] = []
	rest: list[str] = []
	pending_values = 0
	for a in argv:
		if pending_values > 0 and not a.startswith("-"):
			rest.append(a)
			pending_values -= 1
			continue
		pending_values = 0
		if a.startswith("-"):
			rest.append(a)
			if parser is not None and "=" not in a:
				action = parser._option_string_actions.get(a)
				if action is not None:
					pending_values = _option_value_count(action)
			continue
		if str(a).endswith(".json"):
			paths.append(a)
			continue
		rest.append(a)
	return paths, rest


def _coerce_action_default(*, a: argparse.Action, v: object) -> object:
	if v is None:
		return None
	if a.nargs in {"+", "*"}:
		if not isinstance(v, list):
			return v
		if a.type is None:
			return [str(x) for x in v]
		return [a.type(x) for x in v]
	if isinstance(a.nargs, int):
		if not isinstance(v, list):
			return v
		if a.type is None:
			return [str(x) for x in v]
		return [a.type(x) for x in v]
	if a.type is None:
		return v
	return a.type(v)


def merge_cfgs(paths: list[str]) -> dict:
	def _merge(a: object, b: object) -> object:
		if isinstance(a, dict) and isinstance(b, dict):
			out = dict(a)
			for k, v in b.items():
				if k in out:
					out[k] = _merge(out[k], v)
				else:
					out[k] = v
			return out
		return b

	merged: dict = {}
	for pj in paths:
		try:
			cfg = json.loads(Path(pj).read_text(encoding="utf-8"))
		except json.JSONDecodeError as exc:
			raise ValueError(
				f"invalid JSON in config {pj}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
			) from exc
		if not isinstance(cfg, dict):
			raise ValueError(f"json must contain an object at top-level: {pj}")
		merged = _merge(merged, cfg)
	return merged


def add_args(p: argparse.ArgumentParser) -> None:
	p.add_argument(
		"cfg",
		nargs="*",
		help="Optional json config file(s) merged in order; any non-flag *.json is treated as config.",
	)


def apply_defaults_from_cfg_args(p: argparse.ArgumentParser, cfg: dict) -> None:
	args_cfg = cfg.get("args", {})
	if args_cfg is None:
		args_cfg = {}
	if not isinstance(args_cfg, dict):
		raise ValueError("json key 'args' must be an object")

	actions = {a.dest: a for a in p._actions if getattr(a, "dest", None)}
	opts: dict[str, argparse.Action] = {}
	for a in p._actions:
		for s in getattr(a, "option_strings", []) or []:
			o = str(s).lstrip("-")
			if o:
				opts[o] = a
	defaults: dict[str, object] = {}
	for k, v in args_cfg.items():
		key = str(k).lstrip("-")
		a = opts.get(key)
		if a is None:
			a = actions.get(key.replace("-", "_"))
		if a is None:
			continue
		dest = str(a.dest)
		defaults[dest] = _coerce_action_default(a=a, v=v)

	p.set_defaults(**defaults)


def parse_args(p: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
	cfg_paths, argv_rest = split_cfg_argv(argv, p)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = merge_cfgs(cfg_paths) if cfg_paths else {}
	apply_defaults_from_cfg_args(p, cfg)
	return p.parse_args(argv_rest)
