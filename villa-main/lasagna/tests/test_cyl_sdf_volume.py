from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cyl_sdf_volume


def _cylinder_shell(h: int = 4, w: int = 8, *, radius: float = 5.0, height: float = 4.0) -> torch.Tensor:
	z = torch.linspace(-0.5 * height, 0.5 * height, h).view(h, 1).expand(h, w)
	theta = torch.arange(w, dtype=torch.float32).view(1, w) * (2.0 * torch.pi / float(w))
	theta = theta.expand(h, w)
	return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1).contiguous()


class CylSdfVolumeTests(unittest.TestCase):
	def test_capped_shell_mesh_is_watertight_and_outward_oriented(self) -> None:
		shell = _cylinder_shell()
		verts, faces = cyl_sdf_volume.capped_shell_mesh(shell)

		edges: Counter[tuple[int, int]] = Counter()
		for face in faces.tolist():
			for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
				edges[tuple(sorted((int(a), int(b))))] += 1
		self.assertTrue(edges)
		self.assertEqual(set(edges.values()), {2})

		v = verts.to(dtype=torch.float64)
		f = faces.to(dtype=torch.long)
		p0 = v[f[:, 0]]
		p1 = v[f[:, 1]]
		p2 = v[f[:, 2]]
		n = torch.cross(p1 - p0, p2 - p0, dim=-1)
		center = v[:-2].mean(dim=0)
		face_center = (p0 + p1 + p2) / 3.0
		outward = ((face_center - center) * n).sum(dim=-1)
		self.assertGreater(float(outward.min()), 0.0)

	def test_sqrt_depth_encoding_is_monotonic_and_decodes_with_depth_max(self) -> None:
		depth = torch.tensor([0.0, 1.0, 4.0, 9.0])
		encoded = cyl_sdf_volume.encode_inside_depth(depth, 9.0)
		self.assertTrue(bool((encoded[1:] >= encoded[:-1]).all()))
		self.assertEqual(int(encoded[-1]), 255)
		decoded = cyl_sdf_volume.decode_inside_depth(encoded, 9.0)
		self.assertAlmostEqual(float(decoded[-1]), 9.0, delta=1.0e-6)
		self.assertLess(abs(float(decoded[2]) - 4.0), 0.1)

	def test_builder_passes_full_capped_mesh_and_stores_depth_max(self) -> None:
		shell = _cylinder_shell(h=3, w=5)
		calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]] = []

		class _Ext:
			def build_inside_depth_volume(
				self,
				verts,
				faces,
				origin,
				spacing,
				shape,
				progress_label="",
				requested_threads=0,
				chunk_size=8,
				deep_interp_chunks=10.0,
				deep_blend_chunks=2.0,
			):
				calls.append((
					verts.clone(),
					faces.clone(),
					shape.clone(),
					int(chunk_size),
					float(deep_interp_chunks),
					float(deep_blend_chunks),
				))
				Z, Y, X = (int(v) for v in shape.tolist())
				return torch.zeros(1, Z, Y, X, dtype=torch.uint8), 7.5

		with mock.patch.object(cyl_sdf_volume, "_get_ext_module", return_value=_Ext()):
			field = cyl_sdf_volume.build_previous_shell_inside_depth_volume(shell, grid_step=2.0, chunk_size=0)

		self.assertEqual(len(calls), 1)
		verts, faces, _shape, chunk_size, deep_interp_chunks, deep_blend_chunks = calls[0]
		self.assertEqual(int(verts.shape[0]), int(shell.shape[0] * shell.shape[1]) + 2)
		self.assertGreater(int(faces.shape[0]), int(shell.shape[0] * shell.shape[1]))
		self.assertEqual(chunk_size, 0)
		self.assertEqual(deep_interp_chunks, cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_INTERP_CHUNKS)
		self.assertEqual(deep_blend_chunks, cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_BLEND_CHUNKS)
		self.assertEqual(field.depth_max, 7.5)
		self.assertEqual(tuple(field.volume.shape), (1, *field.shape))

	def test_builder_uses_chunked_extension_default(self) -> None:
		shell = _cylinder_shell(h=3, w=5)
		calls: list[tuple[int, float, float]] = []

		class _Ext:
			def build_inside_depth_volume(
				self,
				verts,
				faces,
				origin,
				spacing,
				shape,
				progress_label="",
				requested_threads=0,
				chunk_size=8,
				deep_interp_chunks=10.0,
				deep_blend_chunks=2.0,
			):
				calls.append((int(chunk_size), float(deep_interp_chunks), float(deep_blend_chunks)))
				Z, Y, X = (int(v) for v in shape.tolist())
				return torch.zeros(1, Z, Y, X, dtype=torch.uint8), 0.0

		with mock.patch.object(cyl_sdf_volume, "_get_ext_module", return_value=_Ext()):
			cyl_sdf_volume.build_previous_shell_inside_depth_volume(shell, grid_step=2.0)

		self.assertEqual(calls, [(
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_CHUNK_SIZE,
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_INTERP_CHUNKS,
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_BLEND_CHUNKS,
		)])


@unittest.skipUnless(cyl_sdf_volume.libigl_headers_available(), "libigl/Eigen headers are required")
class CylSdfVolumeExtensionTests(unittest.TestCase):
	def test_chunked_and_full_scan_match_small_cylinder(self) -> None:
		shell = _cylinder_shell(h=5, w=16, radius=4.0)
		bbox = (-5.5, -5.5, -2.5, 5.5, 5.5, 2.5)

		full = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=0,
		)
		chunked = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=4,
			deep_interp_chunks=0.0,
		)

		self.assertEqual(full.shape, chunked.shape)
		self.assertAlmostEqual(full.depth_max, chunked.depth_max, delta=1.0e-6)
		self.assertTrue(torch.equal(full.volume, chunked.volume))

	def test_chunked_scan_handles_non_multiple_edge_chunks(self) -> None:
		shell = _cylinder_shell(h=5, w=16, radius=3.5)
		bbox = (-4.8, -4.9, -2.2, 4.7, 5.2, 2.6)

		full = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=0,
		)
		chunked = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=8,
			deep_interp_chunks=0.0,
		)

		self.assertEqual(full.shape, chunked.shape)
		self.assertAlmostEqual(full.depth_max, chunked.depth_max, delta=1.0e-6)
		self.assertTrue(torch.equal(full.volume, chunked.volume))

	def test_surface_seed_finds_shifted_tiny_shell_when_chunk_center_is_outside(self) -> None:
		shell = _cylinder_shell(h=3, w=12, radius=0.75) + torch.tensor([2.0, 2.0, 2.0])
		bbox = (0.0, 0.0, 0.0, 15.0, 15.0, 15.0)

		full = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=0,
		)
		chunked = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=8,
			deep_interp_chunks=0.0,
		)

		self.assertGreater(int((full.volume > 0).sum().item()), 0)
		self.assertEqual(full.shape, chunked.shape)
		self.assertAlmostEqual(full.depth_max, chunked.depth_max, delta=1.0e-6)
		self.assertTrue(torch.equal(full.volume, chunked.volume))

	def test_non_positive_chunk_size_uses_full_scan_path(self) -> None:
		shell = _cylinder_shell(h=4, w=12, radius=3.0)
		bbox = (-4.0, -4.0, -2.5, 4.0, 4.0, 2.5)

		full_zero = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=0,
		)
		full_negative = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=-1,
		)

		self.assertAlmostEqual(full_zero.depth_max, full_negative.depth_max, delta=1.0e-6)
		self.assertTrue(torch.equal(full_zero.volume, full_negative.volume))

	def test_deep_interp_uses_coarse_chunks_without_hard_chunk_plane_jump(self) -> None:
		shell = _cylinder_shell(h=16, w=32, radius=12.0, height=30.0)
		bbox = (-13.5, -13.5, -16.0, 13.5, 13.5, 16.0)

		field = cyl_sdf_volume.build_previous_shell_inside_depth_volume(
			shell,
			grid_step=1.0,
			bbox=bbox,
			threads=1,
			chunk_size=4,
			deep_interp_chunks=0.25,
			deep_blend_chunks=0.25,
		)

		self.assertGreater(field.depth_max, 0.0)
		decoded = cyl_sdf_volume.decode_inside_depth(field.volume, field.depth_max)[0]
		for axis, size in enumerate(field.shape):
			for plane in range(4, size, 4):
				left = decoded.select(axis, plane - 1)
				right = decoded.select(axis, plane)
				active = (left > 0.0) & (right > 0.0)
				if bool(active.any()):
					self.assertLess(float((left[active] - right[active]).abs().max()), 8.0)


if __name__ == "__main__":
	unittest.main()
