"""Lasagna volume JSON config (.lasagna.json).

A lasagna volume is a collection of channel groups, each stored as a separate
zarr array at its own resolution. The JSON manifest describes the groups,
their channels, scaledowns, and coordinate system metadata.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


LASAGNA_VOLUME_VERSION = 2


@dataclass
class ChannelGroup:
	"""One channel array, or an older flat CZYX preprocessing/fit group."""
	zarr_path: str          # relative to the .lasagna.json file
	scaledown: int          # OME-Zarr pyramid level; actual factor = 2**scaledown
	channels: list[str]     # current: one 3D channel; older CZYX: channel order

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
	init_shell_dir: str = ""
	groups: dict[str, ChannelGroup] = field(default_factory=dict)
	provenance_json: str = ""
	extra_fields: dict = field(default_factory=dict)
	validity_channel: str | None = None

	@property
	def base_directory(self) -> Path:
		return self.path if self.version == 3 else self.path.parent

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
		return self.base_directory / g.zarr_path

	def umbilicus_abs_path(self) -> Path:
		"""Absolute path to the optional umbilicus control-point JSON."""
		if not self.umbilicus_json:
			raise ValueError(f"lasagna volume {self.path} missing required 'umbilicus_json'")
		return self.base_directory / self.umbilicus_json

	def init_shell_dir_abs_path(self) -> Path:
		"""Absolute path to the optional shell-dir-crop initialization directory."""
		if not self.init_shell_dir:
			raise ValueError(f"lasagna volume {self.path} missing required 'init_shell_dir'")
		return self.base_directory / self.init_shell_dir

	# --- persistence ---

	def _backup_path(self, backup_suffix: str | None) -> Path:
		suffix = backup_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
		stem = self.path.name
		if stem.endswith(".lasagna.json"):
			stem = stem[:-len(".lasagna.json")]
		else:
			stem = self.path.stem
		candidate = self.path.with_name(f"{stem}_old.{suffix}.lasagna.json")
		i = 1
		while candidate.exists():
			candidate = self.path.with_name(f"{stem}_old.{suffix}.{i}.lasagna.json")
			i += 1
		return candidate

	def save(self, *, backup_existing: bool = False, backup_suffix: str | None = None) -> None:
		"""Write JSON to self.path atomically."""
		if self.version == 3:
			self._save_compact()
			return
		self.version = LASAGNA_VOLUME_VERSION
		d: dict = {
			"version": LASAGNA_VOLUME_VERSION,
			"source_to_base": self.source_to_base,
			"grad_mag_encode_scale": self.grad_mag_encode_scale,
			"grad_mag_factor": self.grad_mag_factor,
			"groups": {name: g.to_dict() for name, g in self.groups.items()},
		}
		d.update(self.extra_fields)
		if self.provenance_json:
			d["provenance"] = self.provenance_json
		if self.umbilicus_json:
			d["umbilicus_json"] = self.umbilicus_json
		if self.init_shell_dir:
			d["init_shell_dir"] = self.init_shell_dir
		if self.crops:
			d["crops"] = [list(c) for c in self.crops]
		if self.base_shape_zyx is not None:
			d["base_shape_zyx"] = list(self.base_shape_zyx)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		payload = json.dumps(d, indent=2) + "\n"
		tmp_path: Path | None = None
		try:
			for _ in range(100):
				tmp_path = self.path.with_name(
					f".{self.path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
				)
				try:
					fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
					break
				except FileExistsError:
					continue
			else:
				raise FileExistsError(f"could not create temporary manifest next to {self.path}")
			with os.fdopen(fd, "w", encoding="utf-8") as f:
				f.write(payload)
				f.flush()
				os.fsync(f.fileno())
			if self.path.exists():
				try:
					shutil.copymode(self.path, tmp_path)
				except OSError:
					pass
			if backup_existing and self.path.exists():
				shutil.copy2(self.path, self._backup_path(backup_suffix))
			tmp_path.replace(self.path)
		finally:
			if tmp_path is not None:
				try:
					tmp_path.unlink()
				except FileNotFoundError:
					pass

	@staticmethod
	def load(path: str | Path, *, require_umbilicus: bool = False) -> LasagnaVolume:
		"""Load a .lasagna.json file. Raises on any problem."""
		p = Path(path)
		compact = p.is_dir()
		if not compact and not p.name.endswith(".lasagna.json"):
			raise ValueError(
				f"expected .lasagna.json file, got: {p.name}\n"
				"Lasagna volumes must be described by a .lasagna.json manifest."
			)
		d = (json.loads((p / ".zattrs").read_text(encoding="utf-8"))["lasagna"]
			 if compact else json.loads(p.read_text(encoding="utf-8")))
		if compact:
			LasagnaVolume.validate_compact(d)
		known_fields = {
			"version", "source_to_base", "grad_mag_encode_scale", "grad_mag_factor",
			"groups", "umbilicus_json", "init_shell_dir", "crops", "base_shape_zyx",
			"crop_xyzwhd", "provenance", "validity_channel",
		}
		version = int(d.get("version", 1))
		umbilicus_raw = d.get("umbilicus_json", "")
		umbilicus_json = "" if umbilicus_raw is None else str(umbilicus_raw).strip()
		if require_umbilicus and not umbilicus_json:
			raise ValueError(f"lasagna volume {p} missing required 'umbilicus_json'")
		init_shell_dir = str(d.get("init_shell_dir", "")).strip()
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
			init_shell_dir=init_shell_dir,
			groups=groups,
			provenance_json=str(d.get("provenance", "")).strip(),
			extra_fields={key: value for key, value in d.items() if key not in known_fields},
			validity_channel=d.get("validity_channel"),
		)

	@staticmethod
	def validate_compact(d: dict) -> None:
		if d.get("version") != 3:
			raise ValueError("embedded Lasagna manifest must have version 3")
		shape = d.get("base_shape_zyx")
		if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
			raise ValueError("v3 requires positive base_shape_zyx")
		scale = d.get("source_to_base")
		if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
			raise ValueError("v3 requires positive source_to_base")
		groups = d.get("groups")
		if not isinstance(groups, dict) or not groups:
			raise ValueError("v3 requires groups")
		names = []
		for name, group in groups.items():
			n = group.get("scaledown")
			if type(n) is not int or not 0 <= n <= 30 or name != f"sd{n}" or group.get("zarr") != f"sd{n}/{n}":
				raise ValueError("invalid v3 resolution group")
			channels = group.get("channels")
			if not isinstance(channels, list) or not channels or any(not isinstance(c, str) or not c for c in channels):
				raise ValueError("invalid v3 channels")
			names.extend(channels)
		if len(set(names)) != len(names) or d.get("validity_channel") not in names:
			raise ValueError("v3 requires unique channels and a declared validity_channel")
		if "blank_invalid" in d and type(d["blank_invalid"]) is not bool:
			raise ValueError("blank_invalid must be boolean")
		if "width_encoding" in d and d["width_encoding"] not in ("uint8-base-voxels", "float32-input-voxels"):
			raise ValueError("unsupported width_encoding")
		if "grad_mag" in names:
			for field_name in ("grad_mag_encode_scale", "grad_mag_factor"):
				value = d.get(field_name)
				if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
					raise ValueError(f"invalid {field_name}")

	def _save_compact(self) -> None:
		from lasagna.inference_provenance import atomic_write
		d = dict(self.extra_fields)
		d.update(version=3, source_to_base=self.source_to_base,
			base_shape_zyx=list(self.base_shape_zyx or ()),
			groups={name: group.to_dict() for name, group in self.groups.items()},
			validity_channel=self.validity_channel)
		if "grad_mag" in self.all_channels():
			d.update(grad_mag_encode_scale=self.grad_mag_encode_scale, grad_mag_factor=self.grad_mag_factor)
		for key, value in (("provenance", self.provenance_json), ("crops", self.crops),
			("umbilicus_json", self.umbilicus_json), ("init_shell_dir", self.init_shell_dir)):
			if value:
				d[key] = value
		self.validate_compact(d)
		self.path.mkdir(parents=True, exist_ok=True)
		attrs_path = self.path / ".zattrs"
		attrs = json.loads(attrs_path.read_text()) if attrs_path.exists() else {}
		attrs["lasagna"] = d
		atomic_write(attrs_path, attrs)

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
