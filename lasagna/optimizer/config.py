from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from snap_surf import map_global as snap_surf_map_global


def _require_consumed_dict(*, where: str, cfg: dict) -> None:
	if cfg:
		bad = sorted(cfg.keys())
		print(f"WARNING stages_json: {where}: unknown key(s): {bad}")


@dataclass(frozen=True)
class OptSettings:
	steps: int
	lr: float | list[float]
	params: list[str]
	min_scaledown: int
	default_mul: float | None
	w_fac: dict | float | None
	eff: dict[str, float]
	base_eff: dict[str, float]
	steps_auto: bool = False
	args: dict | None = None
	kind: str = "model"


@dataclass(frozen=True)
class Stage:
	name: str
	global_opt: OptSettings | None
	children: tuple["Stage", ...] = ()
	grow: dict | None = None

	@property
	def is_expand(self) -> bool:
		return self.name.startswith("expand-")


CYLINDER_SEED_INIT_STAGE_ROLES = ("cyl_init", "cyl_grow", "cyl_grow_refine")
CYLINDER_STAGE_STEP_ARG = "model-step"
OLD_CYLINDER_STAGE_STEP_ARGS = ("cyl_shell_width_step", "cyl_width_step", "cyl_step_size", "wstep_target")
CYLINDER_OUTPUT_ALL_SHELLS_ARGS = ("cyl_output_all_shells", "cyl_shell_output_all")
CYLINDER_MAX_SEARCH_SHELLS_ARGS = ("cyl_max_shells", "cyl_shell_max_shells", "cyl_shell_search_max_shells")
CYLINDER_GROW_DIRECTION_ARG = "cyl_grow_direction"
CYLINDER_OUTSIDE_GRID_STEP_ARG = "cyl_outside_grid_step"
CYLINDER_OUTSIDE_SAMPLE_FACTOR_ARG = "cyl_outside_sample_factor"
CYLINDER_OUTSIDE_THREADS_ARG = "cyl_outside_threads"
CYLINDER_OUTSIDE_CHUNK_SIZE_ARG = "cyl_outside_chunk_size"
CYLINDER_OUTSIDE_DEEP_INTERP_CHUNKS_ARG = "cyl_outside_deep_interp_chunks"
CYLINDER_OUTSIDE_DEEP_BLEND_CHUNKS_ARG = "cyl_outside_deep_blend_chunks"
CYLINDER_REFINE_MAX_IFRAC_ARGS = ("cyl_refine_max_ifrac", "cyl_grow_refine_max_ifrac")
DEFAULT_CYLINDER_REFINE_MAX_IFRAC = 0.5
CYLINDER_LOSS_NAMES = (
	"cyl_normal", "cyl_center", "cyl_smooth", "cyl_z_smooth", "cyl_step",
	"cyl_z_center", "cyl_step_push", "cyl_radial_mean", "cyl_bend", "cyl_conn_mesh", "cyl_conn_gt",
	"cyl_base_mesh", "cyl_base_gt", "cyl_outside",
)


MODEL_OPT_PARAMS = {"mesh_ms", "amp", "bias", "cyl_params", "map_flatten_ms"}
MAP_OPT_PARAMS = {"map_surf_affine", "map_surf_ms"}
PARAM_REPLACEMENTS = {
	"flatten_map_ms": "map_flatten_ms",
	"map_affine": "map_surf_affine",
	"affine": "map_surf_affine",
	"map_uv_ms": "map_surf_ms",
}
MODEL_INTERNAL_PARAM = {
	"map_flatten_ms": "flatten_map_ms",
}
MAP_LOSS_NAMES = (
	"map_dist",
	"map_vec_normal",
	"map_surface_normal",
	"map_turn",
	"map_smooth",
	"map_bend",
	"map_jac",
	"map_metric_smooth",
	"map_area_smooth",
	"map_dense_prior",
	"map_station_t",
)
MAP_STAGE_LOSS_TO_GLOBAL = {
	"dist": "map_dist",
	"vec": "map_vec_normal",
	"norm": "map_surface_normal",
	"turn": "map_turn",
	"smooth": "map_smooth",
	"bend": "map_bend",
	"jac": "map_jac",
	"metric_smooth": "map_metric_smooth",
	"area_smooth": "map_area_smooth",
	"prior": "map_dense_prior",
}


def normalize_cylinder_grow_direction(raw: object = "outward") -> int:
	if raw is None:
		raw = "outward"
	if isinstance(raw, str):
		value = raw.strip().lower()
		if value in {"outward", "outwards", "grow", "expand", "+", "+1", "1"}:
			return 1
		if value in {"inward", "inwards", "shrink", "contract", "-", "-1"}:
			return -1
	else:
		try:
			value_f = float(raw)
		except (TypeError, ValueError):
			value_f = None
		if value_f == 1.0:
			return 1
		if value_f == -1.0:
			return -1
	raise ValueError(
		f"cylinder stage arg '{CYLINDER_GROW_DIRECTION_ARG}' must be "
		f"'outward' or 'inward', got {raw!r}"
	)


def _stage_to_modifiers(
	base: dict[str, float],
	default_mul: float | None,
	w_fac: dict | float | None,
	scalar_terms: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
	eff = {k: float(v) for k, v in base.items()}
	if default_mul is not None:
		for name in base.keys():
			eff[name] = float(base[name]) * float(default_mul)
	if isinstance(w_fac, (int, float)):
		terms = scalar_terms if scalar_terms is not None else set(base.keys())
		for name in terms:
			if name in base:
				eff[name] = float(base[name]) * float(w_fac)
	elif isinstance(w_fac, dict):
		for k, v in w_fac.items():
			if v is None:
				continue
			eff[str(k)] = float(base.get(str(k), 0.0)) * float(v)

	mods: dict[str, float] = {}
	for name, val in eff.items():
		b = float(base.get(name, 0.0))
		mods[name] = (float(val) / b) if b != 0.0 else 0.0
	return eff, mods


def _parse_opt_settings(
	*,
	stage_name: str,
	opt_cfg: dict,
	base: dict[str, float],
) -> OptSettings:
	opt_cfg = dict(opt_cfg)
	steps_raw = opt_cfg.get("steps", 0)
	lr_raw = opt_cfg.get("lr", 1e-3)
	if isinstance(lr_raw, list):
		if not lr_raw:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.lr: must be a number or a non-empty list")
		lr: float | list[float] = [float(v) for v in lr_raw]
	else:
		lr = float(lr_raw)
	params = opt_cfg.get("params", [])
	if not isinstance(params, list):
		params = []
	params = [str(p) for p in params]
	for old, new in PARAM_REPLACEMENTS.items():
		if old in params:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.params: use '{new}' instead of '{old}'")
	model_params = set(params) & MODEL_OPT_PARAMS
	map_params = set(params) & MAP_OPT_PARAMS
	bad_params = sorted(set(params) - MODEL_OPT_PARAMS - MAP_OPT_PARAMS)
	if bad_params:
		raise ValueError(f"stages_json: stage '{stage_name}' opt.params: unknown name(s): {bad_params}")
	if model_params and map_params:
		raise ValueError(
			f"stages_json: stage '{stage_name}' opt.params: cannot mix model params {sorted(model_params)} "
			f"with map params {sorted(map_params)}; put concurrent map optimization under args.snap_surf_map.map_opt"
		)
	kind = "map" if map_params else "model"
	min_scaledown = max(0, int(opt_cfg.get("min_scaledown", 0)))
	default_mul = opt_cfg.get("default_mul", None)
	w_fac = opt_cfg.get("w_fac", None)
	args_raw = opt_cfg.get("args", None)
	# Back-compat: translate old "auto_offset": true → args dict
	if args_raw is None and opt_cfg.get("auto_offset", False):
		args_raw = {"winding_offset_autocrop": True}
	if args_raw is not None and not isinstance(args_raw, dict):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'args' must be an object or null")
	args = dict(args_raw) if args_raw else {}
	steps_auto = isinstance(steps_raw, str) and steps_raw.strip().lower() == "auto"
	if steps_auto:
		steps = max(1, int(args.get("auto_steps_max", 10000)))
	elif isinstance(steps_raw, str):
		try:
			steps = max(0, int(steps_raw))
		except ValueError as exc:
			raise ValueError(
				f"stages_json: stage '{stage_name}' opt.steps: expected an integer or 'auto'"
		) from exc
	else:
		steps = max(0, int(steps_raw))
	opt_cfg.pop("steps", None)
	opt_cfg.pop("lr", None)
	opt_cfg.pop("params", None)
	opt_cfg.pop("min_scaledown", None)
	opt_cfg.pop("default_mul", None)
	opt_cfg.pop("w_fac", None)
	opt_cfg.pop("auto_offset", None)
	opt_cfg.pop("args", None)
	_require_consumed_dict(where=f"stage '{stage_name}' opt", cfg=opt_cfg)
	if default_mul is not None:
		default_mul = float(default_mul)
	if kind == "map" and w_fac is not None and not isinstance(w_fac, (dict, int, float)):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'w_fac' must be an object, number, or null for map stages")
	if kind == "model" and w_fac is not None and not isinstance(w_fac, (dict, int, float)):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'w_fac' must be an object, number, or null")
	if isinstance(w_fac, dict):
		bad_terms = sorted(set(str(k) for k in w_fac.keys()) - set(base.keys()))
		if bad_terms:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.w_fac: unknown term(s): {bad_terms}")
		if kind == "map":
			bad_map_terms = sorted(set(str(k) for k in w_fac.keys()) - set(MAP_LOSS_NAMES))
			if bad_map_terms:
				raise ValueError(
					f"stages_json: stage '{stage_name}' opt.w_fac: map stages may only override map loss term(s); "
					f"got {bad_map_terms}"
				)
	scalar_terms = set(MAP_LOSS_NAMES) if kind == "map" else (set(base.keys()) - set(MAP_LOSS_NAMES))
	eff, _mods = _stage_to_modifiers(base, default_mul, w_fac, scalar_terms=scalar_terms)
	if float(eff.get("atlas_line_snap", 0.0)) > 0.0:
		conflicting = [
			name for name in ("atlas_line", "atlas_line_control", "atlas_line_other")
			if float(eff.get(name, 0.0)) > 0.0
		]
		if conflicting:
			raise ValueError(
				f"stages_json: stage '{stage_name}' atlas_line_snap cannot be combined with "
				+ ", ".join(conflicting)
			)
	if steps_auto and "cyl_params" in params:
		raise ValueError(f"stages_json: stage '{stage_name}' opt.steps='auto' is not supported for cyl_params stages")
	if "cyl_params" in params:
		if params != ["cyl_params"]:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.params: cyl_params must be optimized alone")
		for old_key in OLD_CYLINDER_STAGE_STEP_ARGS:
			if old_key in args:
				raise ValueError(
					f"stages_json: stage '{stage_name}' opt.args: '{old_key}' is no longer supported; "
					f"use '{CYLINDER_STAGE_STEP_ARG}'"
				)
		if CYLINDER_STAGE_STEP_ARG in args and float(args[CYLINDER_STAGE_STEP_ARG]) <= 0.0:
			raise ValueError(
				f"stages_json: stage '{stage_name}' opt.args.{CYLINDER_STAGE_STEP_ARG}: must be > 0"
		)
		if not any(float(eff.get(name, 0.0)) != 0.0 for name in CYLINDER_LOSS_NAMES):
			raise ValueError(f"stages_json: stage '{stage_name}' with cyl_params requires a nonzero cylinder loss")
	if "map_flatten_ms" in params and not any(
		float(eff.get(name, 0.0)) != 0.0
		for name in ("flatten_sdir", "flatten_map_step", "flatten_edge_step", "flatten_avg_offset", "flatten_orient")
	):
		raise ValueError(f"stages_json: stage '{stage_name}' with map_flatten_ms requires a nonzero flatten loss")
	return OptSettings(
		steps=steps,
		lr=lr,
		params=params,
		min_scaledown=min_scaledown,
		default_mul=default_mul,
		w_fac=w_fac,
		eff=eff,
		base_eff={k: float(v) for k, v in base.items()},
		steps_auto=steps_auto,
		args=args,
		kind=kind,
	)


lambda_global: dict[str, float] = {
	"normal": 1.0,
	"step": 0.0,
	"smooth_step": 0.0,
	"avg_step": 0.0,
	"smooth": 0.0,
	"winding_density": 0.0,
	"data": 0.0,
	"data_plain": 0.0,
	"pred_dt": 0.0,
	"corr": 0.0,
	"winding_vol": 0.0,
	"station_n": 0.0,
	"station_t": 0.0,
	"bend": 0.0,
	"ext_offset": 0.0,
	"snap_surf_map": 0.0,
	"atlas_line": 0.0,
	"atlas_line_control": 0.0,
	"atlas_line_other": 0.0,
	"atlas_line_snap": 0.0,
	"map_dist": 1.0,
	"map_vec_normal": 1.0,
	"map_surface_normal": 1.0,
	"map_turn": 10.0,
	"map_smooth": 0.05,
	"map_bend": 0.01,
	"map_jac": 1.0,
	"map_metric_smooth": 0.05,
	"map_area_smooth": 0.02,
	"map_dense_prior": 0.001,
	"map_station_t": 0.0,
	"cyl_normal": 0.0,
	"cyl_center": 0.0,
	"cyl_smooth": 0.0,
	"cyl_z_smooth": 0.0,
	"cyl_z_center": 0.0,
	"cyl_step_push": 0.0,
	"cyl_step": 0.0,
	"cyl_radial_mean": 0.0,
	"cyl_bend": 0.0,
	"cyl_conn_mesh": 0.0,
	"cyl_conn_gt": 0.0,
	"cyl_base_mesh": 0.0,
	"cyl_base_gt": 0.0,
	"cyl_outside": 0.0,
	"flatten_sdir": 0.0,
	"flatten_map_step": 0.0,
	"flatten_edge_step": 0.0,
	"flatten_avg_offset": 0.0,
	"flatten_orient": 0.0,
}


def _init_mode_from_args(args_cfg: object) -> str | None:
	if not isinstance(args_cfg, dict):
		return None
	model_init = str(args_cfg.get("model-init", args_cfg.get("model_init", "seed"))).strip().lower()
	if model_init and model_init != "seed":
		return None
	init_mode = args_cfg.get("init-mode", args_cfg.get("init_mode", None))
	return None if init_mode is None else str(init_mode).strip().lower()


def _model_init_from_args(args_cfg: object) -> str | None:
	if not isinstance(args_cfg, dict):
		return None
	return str(args_cfg.get("model-init", args_cfg.get("model_init", "seed"))).strip().lower()


def _validate_cylinder_seed_stage_roles(stages: list[Stage]) -> None:
	role_positions: dict[str, int] = {}
	seen_grow = False
	seen_non_role = False
	for i, stage in enumerate(stages):
		is_role = stage.name in CYLINDER_SEED_INIT_STAGE_ROLES
		if is_role:
			if seen_non_role:
				raise ValueError(
					"stages_json: cylinder_seed stages must be contiguous before later cylinder stages"
				)
			if stage.name in role_positions:
				raise ValueError(f"stages_json: cylinder_seed stage '{stage.name}' is duplicated")
			role_positions.setdefault(stage.name, i)
			if stage.global_opt.params != ["cyl_params"]:
				raise ValueError(
					f"stages_json: cylinder_seed stage '{stage.name}' must have params ['cyl_params']"
				)
			if stage.name == "cyl_grow":
				seen_grow = True
			if stage.name == "cyl_grow_refine" and not seen_grow:
				raise ValueError("stages_json: cylinder_seed stage 'cyl_grow_refine' must follow cyl_grow")
		else:
			if role_positions:
				seen_non_role = True
	missing = [name for name in ("cyl_init",) if name not in role_positions]
	if missing:
		raise ValueError(f"stages_json: cylinder_seed missing required stage role(s): {missing}")
	if role_positions["cyl_init"] != 0:
		raise ValueError(
			"stages_json: cylinder_seed stages before cyl_init "
			"must be only cyl_init; later stages are skipped when cyl_grow is absent"
		)
	if role_positions["cyl_init"] != min(role_positions.values()):
		raise ValueError(
			"stages_json: cylinder_seed stages must appear in order "
			"cyl_init, cyl_grow, cyl_grow_refine"
		)
	if "cyl_grow" not in role_positions:
		init_pos = role_positions["cyl_init"]
		for stage in stages[:init_pos]:
			raise ValueError(
				"stages_json: cylinder_seed stages before cyl_init "
				"must be only cyl_init; later stages are skipped when cyl_grow is absent"
		)
		return
	first_non_role = next((i for i, stage in enumerate(stages) if i > role_positions["cyl_init"] and stage.name not in CYLINDER_SEED_INIT_STAGE_ROLES), len(stages))
	for stage in stages[:first_non_role]:
		if stage.name not in CYLINDER_SEED_INIT_STAGE_ROLES:
			raise ValueError(
				"stages_json: cylinder_seed stages before later stages "
				"must be only cyl_init/cyl_grow/cyl_grow_refine; later stages run normally"
		)


def load_stages_cfg(cfg: dict, *, init_mode: str | None = None) -> list[Stage]:
	cfg = dict(cfg)
	args_cfg = cfg.pop("args", None)
	top_args = dict(args_cfg) if isinstance(args_cfg, dict) else {}
	model_init = _model_init_from_args(args_cfg)
	if init_mode is None:
		init_mode = _init_mode_from_args(args_cfg)
	else:
		init_mode = str(init_mode).strip().lower()
	base = dict(lambda_global)
	if model_init == "flatten":
		base = {k: 0.0 for k in base.keys()}
	base_cfg = cfg.pop("base", None)
	if isinstance(base_cfg, dict):
		bad_base = sorted(set(str(k) for k in base_cfg.keys()) - set(base.keys()))
		if bad_base:
			raise ValueError(f"stages_json: base: unknown term(s): {bad_base}")
		for k, v in base_cfg.items():
			base[str(k)] = float(v)

	stages_cfg = cfg.pop("stages", None)
	if stages_cfg is None:
		raise ValueError("stages_json: missing required key 'stages'")
	if not isinstance(stages_cfg, list):
		raise ValueError(
			f"stages_json: expected key 'stages' to be a non-empty list, got {type(stages_cfg).__name__}"
		)
	if not stages_cfg:
		raise ValueError("stages_json: expected key 'stages' to be a non-empty list, got an empty list")
	_require_consumed_dict(where="top-level", cfg=cfg)

	def _load_stage_list(raw_stages: list, *, where: str) -> list[Stage]:
		out: list[Stage] = []
		for s in raw_stages:
			if not isinstance(s, dict):
				raise ValueError(f"stages_json: each stage in {where} must be an object")
			s = dict(s)
			name = str(s.pop("name", ""))
			children_cfg = s.pop("stages", None)
			grow_cfg = s.pop("grow", None)
			if children_cfg is not None or name.startswith("expand-"):
				if not name.startswith("expand-"):
					raise ValueError(f"stages_json: wrapper stage '{name}' must be named expand-*")
				if name not in {"expand-z", "expand-xyz"}:
					raise ValueError(f"stages_json: unsupported expand stage '{name}' (expected expand-z or expand-xyz)")
				if not isinstance(children_cfg, list) or not children_cfg:
					raise ValueError(f"stages_json: stage '{name}' requires non-empty nested 'stages' list")
				if grow_cfg is not None and not isinstance(grow_cfg, dict):
					raise ValueError(f"stages_json: stage '{name}' field 'grow' must be an object or null")
				_require_consumed_dict(where=f"stage '{name}'", cfg=s)
				out.append(Stage(
					name=name,
					global_opt=None,
					children=tuple(_load_stage_list(children_cfg, where=f"stage '{name}'.stages")),
					grow=None if grow_cfg is None else dict(grow_cfg),
				))
				continue
			global_opt_cfg = s.pop("global_opt", None)
			if global_opt_cfg is None:
				global_opt_cfg = s.pop("opt", None)
			if global_opt_cfg is None:
				global_opt_cfg = dict(s)
				s.clear()
			_require_consumed_dict(where=f"stage '{name}'", cfg=s)
			if not isinstance(global_opt_cfg, dict):
				raise ValueError(f"stages_json: stage '{name}' field 'global_opt' must be an object")
			if "flatten_edge_step_global_scale" in top_args:
				global_opt_cfg = dict(global_opt_cfg)
				stage_args = dict(global_opt_cfg.get("args") or {})
				stage_args.setdefault(
					"flatten_edge_step_global_scale",
					top_args["flatten_edge_step_global_scale"],
				)
				global_opt_cfg["args"] = stage_args
			global_opt = _parse_opt_settings(stage_name=name, opt_cfg=global_opt_cfg, base=base)
			out.append(Stage(name=name, global_opt=global_opt))
		return out

	out = _load_stage_list(stages_cfg, where="top-level stages")
	if init_mode == "cylinder_seed":
		_validate_cylinder_seed_stage_roles([s for s in out if s.global_opt is not None])
	return out


def load_stages(path: str) -> list[Stage]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			cfg = json.load(f)
	except json.JSONDecodeError as exc:
		raise ValueError(
			f"stages_json: invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
		) from exc
	if not isinstance(cfg, dict):
		raise ValueError("stages_json: expected an object")
	return load_stages_cfg(cfg)


def total_steps_for_stages(stages: list[Stage]) -> int:
	total = 0
	for stage in stages:
		if stage.children:
			total += total_steps_for_stages(list(stage.children))
			continue
		if stage.global_opt is None:
			continue
		total += max(0, stage.global_opt.steps)
	return total


def _lr_last(lr: float | list[float]) -> float:
	if isinstance(lr, list):
		return float(lr[-1])
	return float(lr)


def _global_map_w_fac_from_eff(*, base: dict[str, float], eff: dict[str, float]) -> dict[str, float]:
	out: dict[str, float] = {}
	for stage_name, global_name in MAP_STAGE_LOSS_TO_GLOBAL.items():
		base_val = float(base.get(global_name, 0.0))
		eff_val = float(eff.get(global_name, 0.0))
		out[stage_name] = (eff_val / base_val) if base_val != 0.0 else 0.0
	return out


def _global_map_args_from_eff(args: dict, *, base: dict[str, float], eff: dict[str, float]) -> dict:
	out = dict(args)
	map_init = dict(out.get("map_init", {})) if isinstance(out.get("map_init", {}), dict) else {}
	map_init.update({
		"w_dist": float(base.get("map_dist", 0.0)),
		"w_vec_normal": float(base.get("map_vec_normal", 0.0)),
		"w_surface_normal": float(base.get("map_surface_normal", 0.0)),
		"map_turn": float(base.get("map_turn", 0.0)),
		"w_smooth": float(base.get("map_smooth", 0.0)),
		"w_bend": float(base.get("map_bend", 0.0)),
		"w_jac": float(base.get("map_jac", 0.0)),
		"w_metric_smooth": float(base.get("map_metric_smooth", 0.0)),
		"w_area_smooth": float(base.get("map_area_smooth", 0.0)),
		"w_dense_prior": float(base.get("map_dense_prior", 0.0)),
	})
	out["map_init"] = map_init
	out["map_station_t"] = float(eff.get("map_station_t", 0.0))
	return out


def _global_map_stage_from_opt_settings(*, name: str, opt_cfg: OptSettings, args: dict) -> snap_surf_map_global.GlobalMapStageConfig:
	if opt_cfg.kind != "map":
		raise ValueError(f"stage '{name}' is not a map optimization stage")
	return snap_surf_map_global.GlobalMapStageConfig(
		name=name,
		steps=int(opt_cfg.steps),
		lr=float(_lr_last(opt_cfg.lr)),
		params=tuple({"map_surf_affine": "affine", "map_surf_ms": "map_uv_ms"}.get(p, p) for p in opt_cfg.params),
		min_scaledown=int(opt_cfg.min_scaledown),
		w_fac=_global_map_w_fac_from_eff(base=opt_cfg.base_eff, eff=opt_cfg.eff),
		args=_global_map_args_from_eff(args, base=opt_cfg.base_eff, eff=opt_cfg.eff),
	)


def global_map_config_from_stages_cfg(cfg: dict) -> snap_surf_map_global.GlobalMapConfig:
	stages = load_stages_cfg(cfg)
	map_stages: list[snap_surf_map_global.GlobalMapStageConfig] = []
	base_eff: dict[str, float] | None = None

	def _collect(raw_stages: list[Stage]) -> None:
		nonlocal base_eff
		for stage in raw_stages:
			if stage.children:
				raise ValueError("global map fixture configs do not support nested/expand stages")
			if stage.global_opt is None:
				continue
			if stage.global_opt.kind != "map":
				raise ValueError(
					f"global map fixture config stage '{stage.name}' is not a map stage; "
					"fixture configs must contain only map optimization stages"
				)
			if base_eff is None:
				base_eff = dict(stage.global_opt.base_eff)
			map_stages.append(_global_map_stage_from_opt_settings(
				name=stage.name,
				opt_cfg=stage.global_opt,
				args=stage.global_opt.args or {},
			))

	_collect(stages)
	if not map_stages:
		raise ValueError("global map fixture config contains no map optimization stages")
	return snap_surf_map_global.GlobalMapConfig(base=dict(base_eff or {}), stages=tuple(map_stages))


def load_global_map_config(path: str | Path) -> snap_surf_map_global.GlobalMapConfig:
	try:
		with open(path, "r", encoding="utf-8") as f:
			cfg = json.load(f)
	except json.JSONDecodeError as exc:
		raise ValueError(
			f"stages_json: invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
		) from exc
	if not isinstance(cfg, dict):
		raise ValueError("stages_json: expected an object")
	return global_map_config_from_stages_cfg(cfg)


def _resolve_snap_surf_map_fixture_export_args(args: dict, *, out_dir: str | None) -> dict:
	if not isinstance(args, dict):
		return args
	if out_dir is None:
		return dict(args)
	out = dict(args)
	base = Path(out_dir)

	def _resolve_path(value):
		if value in (None, "", False):
			return value
		path = Path(str(value))
		return str(path if path.is_absolute() else base / path)

	for key in ("fixture_export_dir", "map_fixture_export_dir"):
		if key in out:
			out[key] = _resolve_path(out[key])
	for key in ("export_fixture", "fixture_export"):
		raw = out.get(key)
		if isinstance(raw, str):
			out[key] = _resolve_path(raw)
		elif isinstance(raw, dict):
			nested = dict(raw)
			for path_key in ("dir", "out_dir", "path"):
				if path_key in nested:
					nested[path_key] = _resolve_path(nested[path_key])
			out[key] = nested
	return out


def _truthy(value) -> bool:
	if isinstance(value, bool):
		return value
	if value is None:
		return False
	if isinstance(value, (int, float)):
		return value != 0
	return str(value).strip().lower() not in {"", "0", "false", "no", "off"}
