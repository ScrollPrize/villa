import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import zarr

import fit_data
import sparse_tensorstore_cache
from lasagna_volume import ChannelGroup, LasagnaVolume


def _write_manifest(
	root: Path,
	*,
	groups: dict,
	base_shape=(64, 48, 32),
	umbilicus_json: str | None = "umbilicus.json",
) -> Path:
	path = root / "vol.lasagna.json"
	d = {
		"version": 2,
		"source_to_base": 1.0,
		"base_shape_zyx": list(base_shape),
		"grad_mag_encode_scale": 1000.0,
		"grad_mag_factor": 1.0,
		"groups": groups,
	}
	if umbilicus_json is not None:
		d["umbilicus_json"] = umbilicus_json
	path.write_text(
		json.dumps(d) + "\n",
		encoding="utf-8",
	)
	return path


def _surface_group(zarr_name: str = "surface.zarr") -> dict:
	return {
		"surface": {
			"zarr": zarr_name,
			"scaledown": 0,
			"channels": ["grad_mag", "nx", "ny"],
		},
	}


def _write_u8_zarr(path: Path, shape: tuple[int, ...]) -> None:
	arr = zarr.open(str(path), mode="w", shape=shape, chunks=shape, dtype="uint8")
	arr[:] = np.zeros(shape, dtype=np.uint8)


class _FakeSparseChunkGroupCache:
	def __init__(
		self,
		*,
		channels: list[str],
		zarr_path: str,
		vol_shape_zyx: tuple[int, int, int],
		channel_indices: dict[str, int],
		is_3d_zarr: bool,
		device: torch.device,
	) -> None:
		self.channels = list(channels)
		self.zarr_path = zarr_path
		self.vol_shape_zyx = vol_shape_zyx
		self.channel_indices = dict(channel_indices)
		self.is_3d_zarr = bool(is_3d_zarr)
		self.chunk_table = torch.zeros(1, dtype=torch.int64, device=device)


class _FakeTensorStoreSparseChunkGroupCache(_FakeSparseChunkGroupCache):
	created: list[tuple[str, bool]] = []

	def __init__(self, *, zarr_path: str, remote: bool = False, **kwargs) -> None:
		super().__init__(zarr_path=zarr_path, **kwargs)
		self.remote = bool(remote)
		type(self).created.append((zarr_path, self.remote))


class FitDataManifestLazyTests(unittest.TestCase):
	def test_lasagna_volume_preserves_provenance_and_unknown_fields(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			manifest = _write_manifest(Path(td), groups={})
			raw = json.loads(manifest.read_text(encoding="utf-8"))
			raw.update(provenance="inference.json", future_field={"value": 3})
			manifest.write_text(json.dumps(raw) + "\n", encoding="utf-8")

			volume = LasagnaVolume.load(manifest)
			volume.save()

			saved = json.loads(manifest.read_text(encoding="utf-8"))
			self.assertEqual(saved["provenance"], "inference.json")
			self.assertEqual(saved["future_field"], {"value": 3})

	def test_lasagna_volume_load_allows_missing_umbilicus_by_default(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups={}, umbilicus_json=None)

			vol = LasagnaVolume.load(manifest)

			self.assertEqual(vol.umbilicus_json, "")
			with self.assertRaisesRegex(ValueError, "umbilicus_json"):
				vol.umbilicus_abs_path()

	def test_lasagna_volume_load_requires_umbilicus_when_requested(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups={}, umbilicus_json=None)

			with self.assertRaisesRegex(ValueError, "umbilicus_json"):
				LasagnaVolume.load(manifest, require_umbilicus=True)

	def test_lasagna_volume_save_omits_empty_umbilicus(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "saved.lasagna.json"

			LasagnaVolume(path=manifest).save()

			raw = json.loads(manifest.read_text(encoding="utf-8"))
			self.assertNotIn("umbilicus_json", raw)

	def test_lasagna_volume_save_uses_atomic_replace(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "saved.lasagna.json"
			replace_calls: list[tuple[Path, Path]] = []
			real_replace = Path.replace

			def _replace(src: Path, dst: Path) -> Path:
				replace_calls.append((src, Path(dst)))
				self.assertTrue(src.exists())
				return real_replace(src, dst)

			with mock.patch.object(Path, "replace", autospec=True, side_effect=_replace):
				LasagnaVolume(path=manifest).save()

			self.assertEqual(len(replace_calls), 1)
			tmp_path, final_path = replace_calls[0]
			self.assertEqual(final_path, manifest)
			self.assertEqual(tmp_path.parent, manifest.parent)
			self.assertTrue(tmp_path.name.startswith(f".{manifest.name}.tmp."))
			self.assertFalse(tmp_path.exists())
			self.assertTrue(manifest.exists())

	def test_lasagna_volume_save_backs_up_existing_manifest(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "saved.lasagna.json"
			manifest.write_text(
				json.dumps({"version": 1, "groups": {"old": {"zarr": "old.zarr", "scaledown": 0, "channels": ["old"]}}}) + "\n",
				encoding="utf-8",
			)
			vol = LasagnaVolume(
				path=manifest,
				groups={
					"cos": ChannelGroup(zarr_path="cos.ome.zarr/0", scaledown=0, channels=["cos"]),
				},
			)

			vol.save(backup_existing=True, backup_suffix="20260727_120000")

			backups = sorted(root.glob("saved_old.20260727_120000*.lasagna.json"))
			self.assertEqual([p.name for p in backups], ["saved_old.20260727_120000.lasagna.json"])
			old_raw = json.loads(backups[0].read_text(encoding="utf-8"))
			self.assertIn("old", old_raw["groups"])
			new_raw = json.loads(manifest.read_text(encoding="utf-8"))
			self.assertEqual(set(new_raw["groups"]), {"cos"})

	def test_lasagna_volume_save_backup_uses_collision_suffix(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "saved.lasagna.json"
			manifest.write_text(json.dumps({"version": 1, "groups": {}}) + "\n", encoding="utf-8")
			(root / "saved_old.fixed.lasagna.json").write_text("occupied\n", encoding="utf-8")

			LasagnaVolume(path=manifest).save(backup_existing=True, backup_suffix="fixed")

			self.assertTrue((root / "saved_old.fixed.1.lasagna.json").exists())
			self.assertEqual((root / "saved_old.fixed.lasagna.json").read_text(encoding="utf-8"), "occupied\n")

	def test_lasagna_volume_save_skips_backup_for_new_manifest(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "new.lasagna.json"

			LasagnaVolume(path=manifest).save(backup_existing=True, backup_suffix="20260727_120000")

			self.assertEqual(list(root.glob("new_old.*.lasagna.json")), [])

	def test_preprocessed_params_uses_manifest_shape_without_opening_zarrs(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups={
					"cos": {"zarr": "missing_cos.ome.zarr/0", "scaledown": 0, "channels": ["cos"]},
					"surface": {
						"zarr": "missing_surface.ome.zarr/2",
						"scaledown": 2,
						"channels": ["grad_mag", "nx", "ny"],
					},
				},
			)

			params = fit_data.get_preprocessed_params(str(manifest))

			self.assertEqual(params["scaledown"], 1.0)
			self.assertEqual(params["volume_extent_fullres"], (32, 48, 64))

	def test_preprocessed_params_allows_channel_only_manifest(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups={
					"surface": {
						"zarr": "missing_surface.ome.zarr/2",
						"scaledown": 2,
						"channels": ["grad_mag", "nx", "ny"],
					},
				},
				umbilicus_json=None,
			)

			params = fit_data.get_preprocessed_params(str(manifest))

			self.assertEqual(params["scaledown"], 4.0)
			self.assertEqual(params["volume_extent_fullres"], (32, 48, 64))

	def test_opened_zarr_shape_is_checked_against_manifest(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				base_shape=(16, 16, 16),
				groups={
					"grad_mag": {
						"zarr": "grad_mag.ome.zarr/1",
						"scaledown": 1,
						"channels": ["grad_mag"],
					},
				},
			)
			vol = LasagnaVolume.load(manifest)
			group = vol.groups["grad_mag"]

			with self.assertRaisesRegex(ValueError, "zarr shape mismatch"):
				fit_data._validate_group_zarr_shape(
					vol=vol,
					group_name="grad_mag",
					group=group,
					zarr_path=str(root / group.zarr_path),
					shape=(7, 8, 8),
				)

	def test_optional_load_3d_streaming_does_not_require_umbilicus(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_u8_zarr(root / "surface.zarr", (3, 4, 5, 6))
			manifest = _write_manifest(
				root,
				groups=_surface_group(),
				base_shape=(4, 5, 6),
				umbilicus_json=None,
			)

			with mock.patch("sparse_cache.SparseChunkGroupCache", _FakeSparseChunkGroupCache):
				data = fit_data.load_3d_streaming(
					path=str(manifest),
					device=torch.device("cpu"),
					sparse_prefetch_backend="python-zarr",
				)

			self.assertEqual(data.size, (4, 5, 6))
			self.assertIsNone(data.umbilicus_points)
			self.assertIsNone(data.umbilicus_xy_lookup)
			self.assertEqual(set(data.channel_spacing or {}), {"grad_mag", "nx", "ny"})

	def test_streaming_source_keeps_complete_local_zarr(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_u8_zarr(root / "surface.zarr", (3, 4, 5, 6))
			manifest = _write_manifest(root, groups=_surface_group())
			vol = LasagnaVolume.load(manifest)

			source = fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

			self.assertFalse(source.remote)
			self.assertEqual(Path(source.zarr_path), root / "surface.zarr")

	def test_streaming_source_uses_validated_remote_marker_for_vc3d_split_cache(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups=_surface_group("surface.ome.zarr/4"),
				base_shape=(4, 5, 6),
				umbilicus_json=None,
			)
			mirror = root / ".lasagna-zarr-metadata" / "surface.ome.zarr" / "4"
			mirror.mkdir(parents=True)
			(mirror / ".zarray").write_text('{"shape":[3,4,5,6]}', encoding="utf-8")
			(root / "lasagna-remote.json").write_text(
				json.dumps({
					"artifact_url": "https://example.test/artifacts/lasagna/",
					"manifest_file": manifest.name,
				}) + "\n",
				encoding="utf-8",
			)
			vol = LasagnaVolume.load(manifest)

			source = fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

			self.assertTrue(source.remote)
			self.assertEqual(source.zarr_path, "https://example.test/artifacts/lasagna/surface.ome.zarr/4")

	def test_catalog_marker_stays_remote_when_metadata_was_copied_into_partial_cache(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups=_surface_group("surface.ome.zarr/4"))
			partial = root / "surface.ome.zarr" / "4"
			partial.mkdir(parents=True)
			(partial / ".zarray").write_text('{"shape":[3,4,5,6]}', encoding="utf-8")
			(partial / "0.0.0").write_bytes(b"one cached chunk")
			(root / "lasagna-remote.json").write_text(
				json.dumps({
					"artifact_url": "https://example.test/artifacts/lasagna",
					"manifest_file": manifest.name,
				}) + "\n",
				encoding="utf-8",
			)
			vol = LasagnaVolume.load(manifest)

			source = fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

			self.assertTrue(source.remote)
			self.assertEqual(source.zarr_path, "https://example.test/artifacts/lasagna/surface.ome.zarr/4")

	def test_remote_artifact_join_preserves_query_and_fragment(self) -> None:
		self.assertEqual(
			fit_data._join_remote_artifact_url(
				"https://example.test/artifacts/lasagna/?token=a%2Fb#anchor",
				("surface.ome.zarr", "4"),
			),
			"https://example.test/artifacts/lasagna/surface.ome.zarr/4?token=a%2Fb#anchor",
		)

	def test_load_streaming_split_cache_uses_remote_tensorstore_not_partial_local_store(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups=_surface_group("surface.ome.zarr/4"),
				base_shape=(4, 5, 6),
				umbilicus_json=None,
			)
			mirror = root / ".lasagna-zarr-metadata" / "surface.ome.zarr" / "4"
			mirror.mkdir(parents=True)
			(mirror / ".zarray").write_text('{"shape":[3,4,5,6]}', encoding="utf-8")
			(root / "surface.ome.zarr" / "4").mkdir(parents=True)
			(root / "lasagna-remote.json").write_text(
				json.dumps({
					"artifact_url": "https://example.test/artifacts/lasagna",
					"manifest_file": manifest.name,
				}) + "\n",
				encoding="utf-8",
			)
			_FakeTensorStoreSparseChunkGroupCache.created.clear()
			with (
				mock.patch.object(
					fit_data, "_probe_tensorstore_zarr_shape", return_value=(3, 4, 5, 6)
				),
				mock.patch(
					"sparse_tensorstore_cache.TensorStoreSparseChunkGroupCache",
					_FakeTensorStoreSparseChunkGroupCache,
				),
				mock.patch.object(fit_data.zarr, "open", side_effect=AssertionError("partial local store opened")),
			):
				data = fit_data.load_3d_streaming(
					path=str(manifest), device=torch.device("cpu"), sparse_prefetch_backend="tensorstore"
				)

			self.assertEqual(data.size, (4, 5, 6))
			self.assertEqual(
				_FakeTensorStoreSparseChunkGroupCache.created,
				[("https://example.test/artifacts/lasagna/surface.ome.zarr/4", True)],
			)

	def test_streaming_source_rejects_split_cache_without_valid_marker(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups=_surface_group("surface.ome.zarr/4"))
			mirror = root / ".lasagna-zarr-metadata" / "surface.ome.zarr" / "4"
			mirror.mkdir(parents=True)
			(mirror / ".zarray").write_text("{}", encoding="utf-8")
			(root / "lasagna-remote.json").write_text(
				json.dumps({"artifact_url": "https://example.test/a", "manifest_file": "other.lasagna.json"}),
				encoding="utf-8",
			)
			vol = LasagnaVolume.load(manifest)

			with self.assertRaisesRegex(ValueError, "does not identify the opened manifest"):
				fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

	def test_streaming_source_rejects_split_cache_without_marker(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups=_surface_group("surface.ome.zarr/4"))
			mirror = root / ".lasagna-zarr-metadata" / "surface.ome.zarr" / "4"
			mirror.mkdir(parents=True)
			(mirror / ".zarray").write_text("{}", encoding="utf-8")
			vol = LasagnaVolume.load(manifest)

			with self.assertRaisesRegex(ValueError, "no adjacent lasagna-remote.json"):
				fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

	def test_streaming_source_rejects_unsafe_remote_group_path(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(root, groups=_surface_group("../outside.zarr"))
			(root / "lasagna-remote.json").write_text(
				json.dumps({"artifact_url": "https://example.test/a", "manifest_file": manifest.name}),
				encoding="utf-8",
			)
			vol = LasagnaVolume.load(manifest)

			with self.assertRaisesRegex(ValueError, "safe relative path"):
				fit_data.resolve_streaming_zarr_source(vol, vol.groups["surface"])

	def test_tensorstore_kvstore_specs_cover_local_http_https_and_s3(self) -> None:
		self.assertEqual(
			sparse_tensorstore_cache.zarr_kvstore_spec("C:/data/a.zarr", remote=False),
			{"driver": "file", "path": "C:/data/a.zarr"},
		)
		self.assertEqual(
			sparse_tensorstore_cache.zarr_kvstore_spec(
				"http://localhost/a.zarr/4", remote=True
			),
			{"driver": "http", "base_url": "http://localhost/a.zarr/4/"},
		)
		self.assertEqual(
			sparse_tensorstore_cache.zarr_kvstore_spec(
				"https://example.test/a.zarr/4", remote=True
			),
			{"driver": "http", "base_url": "https://example.test/a.zarr/4/"},
		)
		self.assertEqual(
			sparse_tensorstore_cache.zarr_kvstore_spec(
				"https://example.test/a.zarr/4?token=a%2Fb#ignored", remote=True
			),
			{
				"driver": "http",
				"base_url": "https://example.test/a.zarr/4/?token=a%2Fb",
			},
		)
		self.assertEqual(
			sparse_tensorstore_cache.zarr_kvstore_spec(
				"s3://bucket/prefix/a.zarr/4", remote=True
			),
			{"driver": "s3", "bucket": "bucket", "path": "prefix/a.zarr/4"},
		)

	def test_load_3d_streaming_requires_umbilicus_when_requested(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups=_surface_group("missing.zarr"),
				base_shape=(4, 5, 6),
				umbilicus_json=None,
			)

			with self.assertRaisesRegex(ValueError, "umbilicus_json"):
				fit_data.load_3d_streaming(
					path=str(manifest),
					device=torch.device("cpu"),
					sparse_prefetch_backend="python-zarr",
					require_umbilicus=True,
				)

	def test_present_umbilicus_json_is_validated_even_when_optional(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups=_surface_group("missing.zarr"),
				base_shape=(4, 5, 6),
				umbilicus_json="missing_umbilicus.json",
			)

			with self.assertRaisesRegex(ValueError, "umbilicus_json not found"):
				fit_data.load_3d_streaming(
					path=str(manifest),
					device=torch.device("cpu"),
					sparse_prefetch_backend="python-zarr",
				)

	def test_umbilicus_xy_at_z_without_lookup_fails_clearly(self) -> None:
		data = fit_data.FitData3D(
			cos=None,
			grad_mag=None,
			nx=None,
			ny=None,
			pred_dt=None,
			corr_points=None,
			winding_volume=None,
			origin_fullres=(0.0, 0.0, 0.0),
			spacing=(1.0, 1.0, 1.0),
		)

		with self.assertRaisesRegex(ValueError, "missing required umbilicus lookup"):
			data.umbilicus_xy_at_z(torch.tensor([0.0]))


if __name__ == "__main__":
	unittest.main()
