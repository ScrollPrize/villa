from __future__ import annotations

from types import ModuleType

from snap_surf.api import configure_snap_surf, last_stats, reset_state, set_debug_step, snap_surf_loss, update_last_stats
import snap_surf.api as _api
import snap_surf.config as _config
import snap_surf.state as _state
import snap_surf.tensor as _tensor
import snap_surf.legacy as _legacy
import snap_surf.map_pyramid as _map_pyramid
import snap_surf.map_objective as _map_objective
import snap_surf.map_growth as _map_growth
import snap_surf.debug_obj as _debug_obj
import snap_surf.map_fixture_io as _map_fixture_io

_MODULES: tuple[ModuleType, ...] = (
	_config,
	_state,
	_tensor,
	_legacy,
	_map_pyramid,
	_map_objective,
	_map_growth,
	_debug_obj,
	_map_fixture_io,
)
_RUNTIME_NAMES = {
	"_cfg", "_active", "_seed_xyz", "_states", "_last_stats", "_offset_debug_printed",
	"_debug_step", "_debug_label", "_stage_label",
}

for _module in _MODULES:
	for _name, _value in vars(_module).items():
		if _name.startswith("__") or _name in _RUNTIME_NAMES:
			continue
		globals().setdefault(_name, _value)


def __getattr__(name: str):
	if name in _RUNTIME_NAMES:
		return getattr(_api, name)
	for module in (_api, *_MODULES):
		if hasattr(module, name):
			return getattr(module, name)
	raise AttributeError(name)


__all__ = [
	"SnapSurfConfig",
	"SnapSurfMapInitConfig",
	"configure_snap_surf",
	"last_stats",
	"reset_state",
	"set_debug_step",
	"snap_surf_loss",
	"update_last_stats",
]
