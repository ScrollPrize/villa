"""The flatten checkpoint state is computed once and handed off in memory.

The interactive Spiral preview flatten used to write the checkpoint to disk
and then torch.load it twice (UV sidecar export and tifxyz export) before
deleting it. These tests pin the in-memory handoff that replaced that: the
sink state must match the saved file exactly, and a state-driven
fit2tifxyz export must reproduce the file-driven one bit for bit.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import tifffile


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import fit2tifxyz
import optimizer


def _write_source_tifxyz(root: Path, h: int = 6, w: int = 9) -> Path:
	yy = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	xx = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	zz = torch.zeros(h, w, dtype=torch.float32)
	xyz = torch.stack([xx, yy, zz], dim=-1).numpy()
	tifxyz = root / "input.tifxyz"
	tifxyz.mkdir()
	tifffile.imwrite(str(tifxyz / "x.tif"), xyz[..., 0])
	tifffile.imwrite(str(tifxyz / "y.tif"), xyz[..., 1])
	tifffile.imwrite(str(tifxyz / "z.tif"), xyz[..., 2])
	(tifxyz / "meta.json").write_text(
		json.dumps({"scale": [1.0, 1.0]}), encoding="utf-8")
	return tifxyz


def _write_cfg(root: Path, tifxyz: Path) -> Path:
	cfg_path = root / "flatten.json"
	cfg_path.write_text(json.dumps({
		"args": {
			"model-init": "flatten",
			"flatten_solver": "forward",
			"flatten_initial_inversion": False,
			"flatten_output_margin": 0.0,
			"device": "cpu",
		},
		"external_surfaces": [{"path": str(tifxyz)}],
		"base": {"flatten_sdir": 1.0},
		"stages": [{
			"name": "flatten",
			"steps": 1,
			"lr": 0.001,
			"params": ["map_flatten_ms"],
			"min_scaledown": 0,
		}],
	}), encoding="utf-8")
	return cfg_path


def _run_fit(argv: list[str], *, state_sink: dict | None = None) -> int:
	# The post-optimizer path is under test; the optimizer itself is not.
	with mock.patch.object(optimizer, "optimize"):
		return fit.main(argv, state_sink=state_sink)


class FlattenStateHandoffTest(unittest.TestCase):
	def test_sink_state_matches_saved_checkpoint(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			cfg_path = _write_cfg(root, _write_source_tifxyz(root))
			model_path = root / "flatten-model.pt"
			sink: dict = {}

			rc = _run_fit(
				[str(cfg_path), "--model-output", str(model_path)],
				state_sink=sink)

			self.assertEqual(rc, 0)
			self.assertTrue(model_path.is_file())
			state = sink["flatten_state"]
			for key in ("flatten_forward_uv", "mesh_flat", "flatten_map_flat",
						"flatten_point_mask", "_model_params_", "_fit_config_"):
				self.assertIn(key, state)
			for value in state.values():
				if torch.is_tensor(value):
					self.assertEqual(value.device.type, "cpu")
			loaded = torch.load(
				model_path, map_location="cpu", weights_only=False)
			self.assertEqual(sorted(loaded.keys()), sorted(state.keys()))
			for key, value in state.items():
				if torch.is_tensor(value):
					self.assertTrue(
						torch.equal(loaded[key], value),
						f"sink state differs from checkpoint at {key!r}")

	def test_sink_is_populated_without_a_model_output_file(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			cfg_path = _write_cfg(root, _write_source_tifxyz(root))
			sink: dict = {}

			rc = _run_fit([str(cfg_path)], state_sink=sink)

			self.assertEqual(rc, 0)
			self.assertIn("flatten_state", sink)
			self.assertEqual(
				[p.name for p in root.iterdir() if p.suffix == ".pt"], [])

	def test_state_driven_export_matches_file_driven_export(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			cfg_path = _write_cfg(root, _write_source_tifxyz(root))
			model_path = root / "flatten-model.pt"
			sink: dict = {}
			rc = _run_fit(
				[str(cfg_path), "--model-output", str(model_path)],
				state_sink=sink)
			self.assertEqual(rc, 0)

			def export(out: Path, input_path: str, state: dict | None) -> Path:
				fit2tifxyz.main([
					"--input", input_path,
					"--output", str(out),
					"--single-segment",
					"--output-name", "flat.tifxyz",
					"--omit-model",
					"--flatten-map-output", str(out / "map.npy"),
				], state=state)
				return out / "flat.tifxyz"

			from_file = export(root / "out_file", str(model_path), None)
			from_state = export(
				root / "out_state", str(root / "does-not-exist.pt"),
				sink["flatten_state"])

			for name in ("x.tif", "y.tif", "z.tif", "d.tif"):
				np.testing.assert_array_equal(
					tifffile.imread(str(from_file / name)),
					tifffile.imread(str(from_state / name)),
					err_msg=f"state-driven export differs at {name}")
			np.testing.assert_array_equal(
				np.load(str(root / "out_file" / "map.npy")),
				np.load(str(root / "out_state" / "map.npy")))

	def test_out_dir_export_reuses_state_and_matches_mesh(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			cfg_path = _write_cfg(root, _write_source_tifxyz(root))
			out = root / "out"
			sink: dict = {}

			rc = _run_fit(
				[str(cfg_path), "--out-dir", str(out)], state_sink=sink)

			self.assertEqual(rc, 0)
			self.assertTrue((out / "model_final.pt").is_file())
			mesh = sink["flatten_state"]["mesh_flat"].numpy()
			exported = tifffile.imread(
				str(out / "tifxyz" / "flatten.tifxyz" / "x.tif"))
			np.testing.assert_array_equal(exported, mesh[0, 0])


if __name__ == "__main__":
	unittest.main()
