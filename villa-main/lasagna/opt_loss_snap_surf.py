from __future__ import annotations

from types import ModuleType

from snap_surf.config import SnapSurfConfig, SnapSurfMapInitConfig
from snap_surf.debug_obj import set_debug_step as _set_debug_obj_step
import snap_surf.config as _config
import snap_surf.state as _state
import snap_surf.tensor as _tensor
import snap_surf.legacy as _legacy
import snap_surf.map_pyramid as _map_pyramid
import snap_surf.map_objective as _map_objective
import snap_surf.debug_obj as _debug_obj
import snap_surf.map_fixture_io as _map_fixture_io

_MODULES: tuple[ModuleType, ...] = (
	_config,
	_state,
	_tensor,
	_legacy,
	_map_pyramid,
	_map_objective,
	_debug_obj,
	_map_fixture_io,
)

_last_stats: dict[str, float] = {}

for _module in _MODULES:
	for _name, _value in vars(_module).items():
		if _name.startswith("__"):
			continue
		globals().setdefault(_name, _value)


def reset_state() -> None:
	global _last_stats
	_last_stats = {}
	_set_debug_obj_step(None)


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def update_last_stats(values: dict[str, float]) -> None:
	_last_stats.update({str(k): float(v) for k, v in values.items()})


def set_debug_step(step: int | None, *, label: str | None = None) -> None:
	_set_debug_obj_step(None if step is None else int(step), label=label)


def __getattr__(name: str):
	for module in _MODULES:
		if hasattr(module, name):
			return getattr(module, name)
	raise AttributeError(name)


__all__ = [
	"SnapSurfConfig",
	"SnapSurfMapInitConfig",
	"last_stats",
	"reset_state",
	"set_debug_step",
	"update_last_stats",
]
