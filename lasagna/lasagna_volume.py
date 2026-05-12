"""Lasagna volume JSON config (.lasagna.json).

A lasagna volume is a collection of channel groups, each stored as a separate
zarr array at its own resolution. The JSON manifest describes the groups,
their channels, scaledowns, and coordinate system metadata.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


LASAGNA_VOLUME_VERSION = 2


@dataclass
class ChannelGroup:
	"""One zarr array containing one or more channels at a common resolution."""
	zarr_path: str          # relative to the .lasagna.json file
	scaledown: int          # OME-Zarr pyramid level; actual factor = 2**scaledown
	channels: list[str]     # ordered; index = position in CZYX zarr

	@property
	def sd_fac(self) -> int:
		"""Actual scale factor = 2**scaledown."""
		return 1 << self.scaledown

	def to_dict(self) -> dict:
		return {
			"zarr": self.zarr_path,
			"scaledown": self.scaledown,
			"channels": self.channels,
		}

	@staticmethod
	def from_dict(d: dict) -> ChannelGroup:
		return ChannelGroup(
			zarr_path=str(d["zarr"]),
			scaledown=int(d["scaledown"]),
			channels=[str(c) for c in d["channels"]],
		)


@dataclass
class LasagnaVolume:
	"""In-memory representation of a .lasagna.json manifest."""
	path: Path
	version: int = LASAGNA_VOLUME_VERSION
	source_to_base: float = 1.0
	crops: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
	base_shape_zyx: tuple[int, int, int] | None = None
	grad_mag_encode_scale: float = 1000.0
	grad_mag_factor: float = 1.0
	umbilicus_json: str = ""
	groups: dict[str, ChannelGroup] = field(default_factory=dict)

	# --- queries ---

	def channel_group(self, channel_name: str) -> tuple[ChannelGroup, int]:
		"""Find which group a channel belongs to and its index within it."""
		for g in self.groups.values():
			if channel_name in g.channels:
				return g, g.channels.index(channel_name)
		raise KeyError(f"channel {channel_name!r} not found in any group; "
					   f"available: {self.all_channels()}")

	def all_channels(self) -> list[str]:
		"""All channel names across all groups, in group-insertion order."""
		out: list[str] = []
		for g in self.groups.values():
			out.extend(g.channels)
		return out

	def zarr_abs_path(self, group_name: str) -> Path:
		"""Absolute path to a group's zarr."""
		g = self.groups[group_name]
		return self.path.parent / g.zarr_path

	def umbilicus_abs_path(self) -> Path:
		"""Absolute path to the required umbilicus control-point JSON."""
		if not self.umbilicus_json:
			raise ValueError(f"lasagna volume {self.path} missing required 'umbilicus_json'")
		return self.path.parent / self.umbilicus_json

	# --- persistence ---

	def save(self) -> None:
		"""Write JSON to self.path."""
		self.version = LASAGNA_VOLUME_VERSION
		d: dict = {
			"version": LASAGNA_VOLUME_VERSION,
			"source_to_base": self.source_to_base,
			"grad_mag_encode_scale": self.grad_mag_encode_scale,
			"grad_mag_factor": self.grad_mag_factor,
			"umbilicus_json": self.umbilicus_json,
			"groups": {name: g.to_dict() for name, g in self.groups.items()},
		}
		if self.crops:
			d["crops"] = [list(c) for c in self.crops]
		if self.base_shape_zyx is not None:
			d["base_shape_zyx"] = list(self.base_shape_zyx)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self.path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

	@staticmethod
	def load(path: str | Path) -> LasagnaVolume:
		"""Load a .lasagna.json file. Raises on any problem."""
		p = Path(path)
		if not p.name.endswith(".lasagna.json"):
			raise ValueError(
				f"expected .lasagna.json file, got: {p.name}\n"
				"Lasagna volumes must be described by a .lasagna.json manifest."
			)
		d = json.loads(p.read_text(encoding="utf-8"))
		version = int(d.get("version", 1))
		umbilicus_json = str(d.get("umbilicus_json", "")).strip()
		if not umbilicus_json:
			raise ValueError(f"lasagna volume {p} missing required 'umbilicus_json'")
		# Load crops list (new format) or migrate from single crop_xyzwhd (old)
		crops_raw = d.get("crops")
		crops: list[tuple[int, int, int, int, int, int]] = []
		if isinstance(crops_raw, list):
			for c in crops_raw:
				t = tuple(int(v) for v in c)
				if len(t) != 6:
					raise ValueError(f"each crop must have 6 elements, got {len(t)}")
				crops.append(t)
		else:
			old_crop = d.get("crop_xyzwhd")
			if old_crop is not None:
				t = tuple(int(v) for v in old_crop)
				if len(t) != 6:
					raise ValueError(f"crop_xyzwhd must have 6 elements, got {len(t)}")
				crops.append(t)
		bshape = d.get("base_shape_zyx")
		if bshape is not None:
			bshape = tuple(int(v) for v in bshape)
			if len(bshape) != 3:
				raise ValueError(f"base_shape_zyx must have 3 elements, got {len(bshape)}")
		groups: dict[str, ChannelGroup] = {}
		for name, gd in d.get("groups", {}).items():
			groups[str(name)] = ChannelGroup.from_dict(gd)
		return LasagnaVolume(
			path=p.resolve(),
			version=version,
			source_to_base=float(d.get("source_to_base", 1.0)),
			crops=crops,
			base_shape_zyx=bshape,
			grad_mag_encode_scale=float(d.get("grad_mag_encode_scale", 1000.0)),
			grad_mag_factor=float(d.get("grad_mag_factor", 1.0)),
			umbilicus_json=umbilicus_json,
			groups=groups,
		)

	def add_crop(self, crop: tuple[int, int, int, int, int, int]) -> None:
		"""Append a crop region if not already present."""
		if crop not in self.crops:
			self.crops.append(crop)

	def update_group(self, name: str, group: ChannelGroup) -> None:
		"""Add or replace a group, then save."""
		self.groups[name] = group
		self.save()

	@staticmethod
	def is_lasagna_json(path: str) -> bool:
		"""Check if path ends with .lasagna.json."""
		return str(path).rstrip("/").endswith(".lasagna.json")
