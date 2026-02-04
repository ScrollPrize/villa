from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("config")
	g.add_argument(
		"--json",
		action="append",
		default=[],
		help="Optional JSON config(s). Later files override earlier. CLI flags override JSON.",
	)


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


def parse_args(p: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
	pre = argparse.ArgumentParser(add_help=False)
	pre.add_argument("--json", action="append", default=[])
	pre_ns, _rest = pre.parse_known_args(argv)
	paths = [str(x) for x in (pre_ns.json or [])]

	merged: dict[str, object] = {}
	for pj in paths:
		cfg = json.loads(Path(pj).read_text(encoding="utf-8"))
		if not isinstance(cfg, dict):
			raise ValueError(f"--json must contain an object at top-level: {pj}")
		args_cfg = cfg.get("args", {})
		if args_cfg is None:
			args_cfg = {}
		if not isinstance(args_cfg, dict):
			raise ValueError(f"--json must contain an object in key 'args': {pj}")
		merged.update(args_cfg)

	actions = {a.dest: a for a in p._actions if getattr(a, "dest", None)}
	defaults: dict[str, object] = {}
	for k, v in merged.items():
		dest = str(k)
		a = actions.get(dest)
		if a is None:
			continue
		defaults[dest] = _coerce_action_default(a=a, v=v)

	p.set_defaults(**defaults)
	return p.parse_args(argv)

