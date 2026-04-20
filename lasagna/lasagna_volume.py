"""Lasagna volume JSON config (.lasagna.json).

A lasagna volume is a collection of channel groups, each stored as a separate
zarr array at its own resolution. The JSON manifest describes the groups,
their channels, scaledowns, and coordinate system metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChannelGroup:
	"""One zarr array containing one or more channels at a common resolution."""
	zarr_path: str          # relative to the .lasagna.json file
	scaledown: int          # downsample factor relative to source volume
	channels: list[str]     # ordered; index = position in CZYX zarr

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
	version: int = 1
	source_to_base: float = 1.0
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None
	grad_mag_encode_scale: float = 1000.0
	grad_mag_factor: float = 1.0
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

	# --- persistence ---

	def save(self) -> None:
		"""Write JSON to self.path."""
		d: dict = {
			"version": self.version,
			"source_to_base": self.source_to_base,
			"grad_mag_encode_scale": self.grad_mag_encode_scale,
			"grad_mag_factor": self.grad_mag_factor,
			"groups": {name: g.to_dict() for name, g in self.groups.items()},
		}
		if self.crop_xyzwhd is not None:
			d["crop_xyzwhd"] = list(self.crop_xyzwhd)
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
		if version != 1:
			raise ValueError(f"unsupported lasagna volume version: {version}")
		crop = d.get("crop_xyzwhd")
		if crop is not None:
			crop = tuple(int(v) for v in crop)
			if len(crop) != 6:
				raise ValueError(f"crop_xyzwhd must have 6 elements, got {len(crop)}")
		groups: dict[str, ChannelGroup] = {}
		for name, gd in d.get("groups", {}).items():
			groups[str(name)] = ChannelGroup.from_dict(gd)
		return LasagnaVolume(
			path=p.resolve(),
			version=version,
			source_to_base=float(d.get("source_to_base", 1.0)),
			crop_xyzwhd=crop,
			grad_mag_encode_scale=float(d.get("grad_mag_encode_scale", 1000.0)),
			grad_mag_factor=float(d.get("grad_mag_factor", 1.0)),
			groups=groups,
		)

	def update_group(self, name: str, group: ChannelGroup) -> None:
		"""Add or replace a group, then save."""
		self.groups[name] = group
		self.save()

	@staticmethod
	def is_lasagna_json(path: str) -> bool:
		"""Check if path ends with .lasagna.json."""
		return str(path).rstrip("/").endswith(".lasagna.json")
